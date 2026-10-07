"""Double DQN self-play training of the NNUE evaluation network.

The NNUE gives V(x): the value of position x for the side to move, in [-1, 1].
Moves are scored with afterstates, negamax style:

    Q(s, a) = -V(T(s, a))        (the opponent is to move after a)

Each stored transition is an afterstate x reached by a move. Its target follows
Double DQN: the online network picks the opponent's best reply a* in x, and the
target network evaluates it:

    a*        = argmax_a' -V_online(T(x, a'))
    target(x) = -gamma * V_target(T(x, a*))      (exact result if x is terminal)

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

from awale import HALF_BOARD_SIZE, BoardState, new_game
from nnue import NNUE, NUM_ACTIVE_FEATURES, state_features

GAMMA = 0.99
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
WEIGHTS_NAME = "nnue.pt"
CHECKPOINT_NAME = "rl_checkpoint.pt"
EMPTY_FEATURES = torch.zeros(2, NUM_ACTIVE_FEATURES, dtype=torch.long)


class Successors:
    """All afterstates of a position, padded to HALF_BOARD_SIZE for batching."""

    def __init__(self, state: BoardState):
        self.moves = state.get_possible_moves()
        self.states = []
        features = [EMPTY_FEATURES] * HALF_BOARD_SIZE
        self.mask = torch.zeros(HALF_BOARD_SIZE, dtype=torch.bool)
        self.terminal = torch.zeros(HALF_BOARD_SIZE, dtype=torch.bool)
        self.terminal_value = torch.zeros(HALF_BOARD_SIZE)
        for slot, move in enumerate(self.moves):
            next_state = state.copy()
            next_state.make_move(move)
            self.states.append(next_state)
            features[slot] = state_features(next_state)
            self.mask[slot] = True
            if next_state.is_over():
                self.terminal[slot] = True
                self.terminal_value[slot] = next_state.result(next_state.current_player)
        self.features = torch.stack(features)


def afterstate_values(net: NNUE, features, mask, terminal, terminal_value) -> torch.Tensor:
    """V for the side to move in each afterstate; exact for terminals, -inf-safe for padding.

    features: (..., HALF_BOARD_SIZE, 2, NUM_ACTIVE_FEATURES)
    """
    shape = features.shape[:-2]
    values = torch.tanh(net(features.reshape(-1, 2, NUM_ACTIVE_FEATURES))).reshape(shape)
    values = torch.where(terminal, terminal_value, values)
    return values.masked_fill(~mask, 0.0)


def q_values(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Q(s, a) = -V(T(s, a)), with illegal moves at -inf."""
    return (-values).masked_fill(~mask, float("-inf"))


@torch.no_grad()
def choose_move(net: NNUE, state: BoardState, epsilon: float = 0.0) -> tuple[int, Successors]:
    successors = Successors(state)
    if random.random() < epsilon:
        slot = random.randrange(len(successors.moves))
    else:
        values = afterstate_values(
            net, successors.features, successors.mask, successors.terminal, successors.terminal_value
        )
        slot = q_values(values, successors.mask).argmax().item()
    return slot, successors


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.transitions = deque(maxlen=capacity)

    def __len__(self):
        return len(self.transitions)

    def add(self, state: BoardState, features: torch.Tensor):
        """Store afterstate `state` with everything needed to compute its target later."""
        if state.is_over():
            successors = None
            terminal, value = True, float(state.result(state.current_player))
        else:
            successors = Successors(state)
            terminal, value = False, 0.0
        self.transitions.append((features, successors, terminal, value))

    def sample(self, batch_size: int):
        batch = random.sample(self.transitions, batch_size)
        features = torch.stack([t[0] for t in batch])
        terminal = torch.tensor([t[2] for t in batch])
        value = torch.tensor([t[3] for t in batch])
        empty = torch.zeros(HALF_BOARD_SIZE, dtype=torch.bool)
        succ_features = torch.stack(
            [t[1].features if t[1] else torch.zeros(HALF_BOARD_SIZE, 2, NUM_ACTIVE_FEATURES, dtype=torch.long) for t in batch]
        )
        # Terminal rows get one fake legal move so argmax stays well defined; their target is overridden
        succ_mask = torch.stack([t[1].mask if t[1] else empty.clone().index_fill_(0, torch.tensor([0]), True) for t in batch])
        succ_terminal = torch.stack([t[1].terminal if t[1] else empty for t in batch])
        succ_value = torch.stack([t[1].terminal_value if t[1] else torch.zeros(HALF_BOARD_SIZE) for t in batch])
        return features, terminal, value, succ_features, succ_mask, succ_terminal, succ_value


def play_self_play_game(net: NNUE, buffer: ReplayBuffer, epsilon: float):
    state = new_game()
    while not state.is_over():
        slot, successors = choose_move(net, state, epsilon)
        state = successors.states[slot]
        buffer.add(state, successors.features[slot])


def train_step(online: NNUE, target: NNUE, buffer: ReplayBuffer, optimizer, batch_size: int, max_grad_norm: float, device) -> float:
    batch = [tensor.to(device, non_blocking=True) for tensor in buffer.sample(batch_size)]
    features, terminal, value, succ_features, succ_mask, succ_terminal, succ_value = batch
    with torch.no_grad():
        # Double DQN: online network selects the reply, target network evaluates it
        online_values = afterstate_values(online, succ_features, succ_mask, succ_terminal, succ_value)
        best_reply = q_values(online_values, succ_mask).argmax(dim=1, keepdim=True)
        target_values = afterstate_values(target, succ_features, succ_mask, succ_terminal, succ_value)
        td_target = GAMMA * target_values.gather(1, best_reply).squeeze(1) * -1
        td_target = torch.where(terminal, value, td_target)

    prediction = torch.tanh(online(features))
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
    parser.add_argument("--weights", type=Path, help=f"NNUE weights, loadable by nnue.py/mcts (default: <snapshot-dir>/{WEIGHTS_NAME})")
    parser.add_argument("--checkpoint", type=Path, help=f"Full training state (default: <snapshot-dir>/{CHECKPOINT_NAME})")
    parser.add_argument("--snapshot-dir", type=Path, default=MODELS_DIR, help="Per-save weight snapshots, rated by elo.py")
    parser.add_argument("--resume", action="store_true", help="Resume from --checkpoint, or from --weights if absent")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device for gradient steps")
    args = parser.parse_args()
    args.weights = args.weights or args.snapshot_dir / WEIGHTS_NAME
    args.checkpoint = args.checkpoint or args.snapshot_dir / CHECKPOINT_NAME
    device = torch.device(args.device)
    print(f"training on {device}" + (f" ({torch.cuda.get_device_name(device)})" if device.type == "cuda" else ""))

    online = NNUE().to(device)
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
    buffer = ReplayBuffer(args.buffer_size)
    last_save = time.monotonic()
    print("training until interrupted (Ctrl+C saves and exits)", flush=True)

    try:
        while True:
            iteration += 1
            progress = min((iteration - 1) / max(args.epsilon_decay_iterations, 1), 1.0)
            epsilon = args.epsilon_start + (args.epsilon_end - args.epsilon_start) * progress

            for _ in range(args.games_per_iteration):
                play_self_play_game(actor, buffer, epsilon)

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
