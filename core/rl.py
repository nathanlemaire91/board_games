"""Double DQN self-play training of the NNUE evaluation network, for any game.

    uv run python -m core.rl [--game awale] [--resume]

The NNUE gives V(x): the value of position x for the side to move, in [-1, 1].
Moves are scored with afterstates, negamax style:

    Q(s, a) = sign(s, a) * V(T(s, a))

with sign -1 when the opponent is to move after a, and +1 when the same player
moves again (the jumps of a multiple jump in checkers).

Each stored transition is an afterstate x reached by a move. Its target follows
Double DQN: the online network picks the best move a* of the side to move in x,
and the target network evaluates it:

    a*        = argmax_a' sign(x, a') * V_online(T(x, a'))
    target(x) = gamma * sign(x, a*) * V_target(T(x, a*))      (exact result if x is terminal)

Training runs until interrupted. Every iteration plays epsilon-greedy self-play
games with the online network, trains on a replay buffer and syncs the target
network. The full training state is checkpointed every --save-minutes and on Ctrl+C.

Gradient steps run on --device (CUDA when available). Self-play evaluates one
position at a time, which is faster on the CPU, so games are played by a CPU copy
of the online network that is refreshed after every iteration.
"""

import argparse
import copy
import random
import time
from collections import deque
from pathlib import Path

import torch
from torch import nn

from core.game import Game, GameState
from core.nnue import NNUE, state_features
from games import DEFAULT_GAME, GAMES, get_game

GAMMA = 0.99
WEIGHTS_NAME = "nnue.pt"  # In the game's models directory
CHECKPOINT_NAME = "rl_checkpoint.pt"


class Successors:
    """The afterstates of a position, one per legal move, and how their values count for the mover.

    features: (moves, 2, num_active_features), in int16 to keep the replay buffer small
    sign: -1 where the opponent is to move next, +1 where the mover moves again
    terminal, terminal_value: games over, and their exact value for the side to move
    """

    def __init__(self, game: Game, state: GameState):
        self.moves = state.get_possible_moves()
        self.states = []
        features, sign, terminal, terminal_value = [], [], [], []
        for move in self.moves:
            next_state = state.copy()
            next_state.make_move(move)
            self.states.append(next_state)
            features.append(state_features(game, next_state))
            sign.append(1.0 if next_state.current_player == state.current_player else -1.0)
            over = next_state.is_over()
            terminal.append(over)
            terminal_value.append(float(next_state.result(next_state.current_player)) if over else 0.0)
        self.features = torch.stack(features).to(torch.int16)
        self.sign = torch.tensor(sign)
        self.terminal = torch.tensor(terminal)
        self.terminal_value = torch.tensor(terminal_value)


def afterstate_values(net: NNUE, features, terminal, terminal_value) -> torch.Tensor:
    """V for the side to move in each afterstate, exact for terminals.

    features: (..., 2, num_active_features)
    """
    shape = features.shape[:-2]
    values = torch.tanh(net(features.reshape(-1, *features.shape[-2:]).long())).reshape(shape)
    return torch.where(terminal, terminal_value, values)


def q_values(values: torch.Tensor, sign: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """Q(s, a) = sign * V(T(s, a)), the afterstates' values for the mover, with illegal moves at -inf."""
    q = sign * values
    return q if mask is None else q.masked_fill(~mask, float("-inf"))


@torch.no_grad()
def choose_slot(net: NNUE, successors: Successors, epsilon: float = 0.0) -> int:
    """Epsilon-greedy: the index in successors.moves of a random move, or of the best one."""
    if random.random() < epsilon:
        return random.randrange(len(successors.moves))
    values = afterstate_values(net, successors.features, successors.terminal, successors.terminal_value)
    return q_values(values, successors.sign).argmax().item()


def choose_move(game: Game, net: NNUE, state: GameState, epsilon: float = 0.0) -> tuple[int, Successors]:
    successors = Successors(game, state)
    return choose_slot(net, successors, epsilon), successors


class ReplayBuffer:
    """Afterstates, each with what its target needs: its exact value if it ends the game,
    else the features, signs and exact values of its successors (unpadded)."""

    def __init__(self, game: Game, capacity: int):
        self.game = game
        self.transitions = deque(maxlen=capacity)

    def __len__(self):
        return len(self.transitions)

    def add(self, features: torch.Tensor, successors: Successors | None, value: float = 0.0):
        """An afterstate's features, and its successors, or None if it ends the game with `value`."""
        replies = None if successors is None else (
            successors.features, successors.sign, successors.terminal, successors.terminal_value
        )
        self.transitions.append((features, replies, value))

    def sample(self, batch_size: int):
        """A batch, successors padded to the most legal moves among them (padding masked out)."""
        batch = random.sample(self.transitions, batch_size)
        features = torch.stack([t[0] for t in batch])
        terminal = torch.tensor([t[1] is None for t in batch])
        value = torch.tensor([t[2] for t in batch])
        width = max((len(t[1][1]) for t in batch if t[1] is not None), default=1)
        succ_features = torch.zeros(batch_size, width, *features.shape[1:], dtype=features.dtype)
        succ_mask = torch.zeros(batch_size, width, dtype=torch.bool)
        succ_sign = torch.zeros(batch_size, width)
        succ_terminal = torch.zeros(batch_size, width, dtype=torch.bool)
        succ_value = torch.zeros(batch_size, width)
        # Terminal rows get one fake legal move so argmax stays well defined; their target is overridden
        succ_mask[:, 0] = True
        for row, (_, replies, _) in enumerate(batch):
            if replies is None:
                continue
            row_features, row_sign, row_terminal, row_value = replies
            count = len(row_sign)
            succ_features[row, :count] = row_features
            succ_mask[row, :count] = True
            succ_sign[row, :count] = row_sign
            succ_terminal[row, :count] = row_terminal
            succ_value[row, :count] = row_value
        return features, terminal, value, succ_features, succ_mask, succ_sign, succ_terminal, succ_value


def play_self_play_game(game: Game, net: NNUE, buffer: ReplayBuffer, epsilon: float):
    """Each position's successors are computed once: to choose its move, then as the targets of
    the afterstate that led to it."""
    successors = Successors(game, game.new_game())
    while True:
        slot = choose_slot(net, successors, epsilon)
        features = successors.features[slot].clone()  # Not a view: it would keep all the successors alive
        if successors.terminal[slot]:
            buffer.add(features, None, successors.terminal_value[slot].item())
            return
        successors = Successors(game, successors.states[slot])
        buffer.add(features, successors)


def train_step(online: NNUE, target: NNUE, buffer: ReplayBuffer, optimizer, batch_size: int, max_grad_norm: float, device) -> float:
    batch = [tensor.to(device, non_blocking=True) for tensor in buffer.sample(batch_size)]
    features, terminal, value, succ_features, succ_mask, succ_sign, succ_terminal, succ_value = batch
    with torch.no_grad():
        # Double DQN: online network selects the move, target network evaluates it
        online_values = afterstate_values(online, succ_features, succ_terminal, succ_value)
        best_move = q_values(online_values, succ_sign, succ_mask).argmax(dim=1, keepdim=True)
        target_values = afterstate_values(target, succ_features, succ_terminal, succ_value)
        td_target = GAMMA * q_values(target_values, succ_sign).gather(1, best_move).squeeze(1)
        td_target = torch.where(terminal, value, td_target)

    prediction = torch.tanh(online(features.long()))
    loss = nn.functional.smooth_l1_loss(prediction, td_target)
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(online.parameters(), max_grad_norm)
    optimizer.step()
    return loss.item()


def learning_rate_factor(step: int, warmup_steps: int, half_life: int, min_factor: float) -> float:
    """Linear warmup, then exponential decay towards a floor (never reaches zero)."""
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    return max(min_factor, 0.5 ** ((step - warmup_steps) / half_life))


def save_checkpoint(path: Path, weights_path: Path, snapshot_dir: Path, online, target, optimizer, scheduler, iteration, step):
    checkpoint = {
        "online": online.state_dict(),
        "target": target.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "iteration": iteration,
        "step": step,
    }
    # Write to a temp file then rename, so an interrupted save never corrupts the previous one
    # Each save also keeps a snapshot of the weights, so elo.py can rate checkpoints against each other
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = snapshot_dir / f"nnue_iter{iteration:06d}.pt"
    # Weights are stored on the CPU so they load on machines without a GPU
    weights = {name: tensor.cpu() for name, tensor in online.state_dict().items()}
    for data, destination in ((checkpoint, path), (weights, weights_path), (weights, snapshot_path)):
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        torch.save(data, temporary)
        temporary.replace(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--game", choices=GAMES, default=DEFAULT_GAME)
    parser.add_argument("--games-per-iteration", type=int, default=50)
    parser.add_argument("--train-steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--buffer-size", type=int, default=100_000)
    parser.add_argument("--lr", type=float, default=1e-3, help="Peak learning rate")
    parser.add_argument("--lr-warmup-steps", type=int, default=1_000)
    parser.add_argument("--lr-half-life", type=int, default=200_000, help="Gradient steps for the learning rate to halve")
    parser.add_argument("--lr-min-factor", type=float, default=0.05, help="Learning rate floor, as a fraction of --lr")
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--epsilon-decay-iterations", type=int, default=50)
    parser.add_argument("--target-sync", type=int, default=1, help="Sync target network every N iterations")
    parser.add_argument("--save-minutes", type=float, default=10.0)
    parser.add_argument("--weights", type=Path, help=f"NNUE weights, used by the players (default: <snapshot-dir>/{WEIGHTS_NAME})")
    parser.add_argument("--checkpoint", type=Path, help=f"Full training state (default: <snapshot-dir>/{CHECKPOINT_NAME})")
    parser.add_argument("--snapshot-dir", type=Path, help="Per-save weight snapshots, rated by elo.py (default: models/<game>)")
    parser.add_argument("--resume", action="store_true", help="Resume from --checkpoint, or from --weights if absent")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device for gradient steps")
    args = parser.parse_args()
    game = get_game(args.game)
    args.snapshot_dir = args.snapshot_dir or game.models_dir
    args.weights = args.weights or args.snapshot_dir / WEIGHTS_NAME
    args.checkpoint = args.checkpoint or args.snapshot_dir / CHECKPOINT_NAME
    device = torch.device(args.device)
    print(f"training on {device}" + (f" ({torch.cuda.get_device_name(device)})" if device.type == "cuda" else ""))

    online = NNUE(game.num_features).to(device)
    optimizer = torch.optim.Adam(online.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: learning_rate_factor(step, args.lr_warmup_steps, args.lr_half_life, args.lr_min_factor),
    )
    target = copy.deepcopy(online)
    target.requires_grad_(False)
    iteration, step = 0, 0

    if args.resume and args.checkpoint.exists():
        checkpoint = torch.load(args.checkpoint, map_location=device)
        online.load_state_dict(checkpoint["online"])
        target.load_state_dict(checkpoint["target"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        iteration, step = checkpoint["iteration"], checkpoint["step"]
        print(f"resumed from {args.checkpoint} at iteration {iteration}, step {step}")
    elif args.resume and args.weights.exists():
        online.load_state_dict(torch.load(args.weights, map_location=device))
        target.load_state_dict(online.state_dict())
        print(f"resumed weights from {args.weights} (fresh optimizer)")

    actor = copy.deepcopy(online).cpu()  # Plays the self-play games
    actor.requires_grad_(False)
    buffer = ReplayBuffer(game, args.buffer_size)
    last_save = time.monotonic()
    print("training until interrupted (Ctrl+C saves and exits)", flush=True)

    try:
        while True:
            iteration += 1
            progress = min((iteration - 1) / max(args.epsilon_decay_iterations, 1), 1.0)
            epsilon = args.epsilon_start + (args.epsilon_end - args.epsilon_start) * progress

            for _ in range(args.games_per_iteration):
                play_self_play_game(game, actor, buffer, epsilon)

            losses = []
            if len(buffer) >= args.batch_size:
                for _ in range(args.train_steps):
                    losses.append(train_step(online, target, buffer, optimizer, args.batch_size, args.max_grad_norm, device))
                    scheduler.step()
                    step += 1
            if iteration % args.target_sync == 0:
                target.load_state_dict(online.state_dict())
            actor.load_state_dict(online.state_dict())

            mean_loss = sum(losses) / len(losses) if losses else float("nan")
            line = (
                f"iter {iteration:5d}  step {step:8d}  eps={epsilon:.2f}  "
                f"lr={scheduler.get_last_lr()[0]:.2e}  buffer={len(buffer):6d}  loss={mean_loss:.4f}"
            )
            print(line, flush=True)

            if time.monotonic() - last_save >= args.save_minutes * 60:
                save_checkpoint(args.checkpoint, args.weights, args.snapshot_dir, online, target, optimizer, scheduler, iteration, step)
                last_save = time.monotonic()
                print(f"saved {args.checkpoint}, {args.weights} and a snapshot in {args.snapshot_dir}", flush=True)
    except KeyboardInterrupt:
        save_checkpoint(args.checkpoint, args.weights, args.snapshot_dir, online, target, optimizer, scheduler, iteration, step)
        print(f"\ninterrupted: saved {args.checkpoint}, {args.weights} and a snapshot in {args.snapshot_dir}")


if __name__ == "__main__":
    main()
