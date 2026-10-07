"""Players that pick a move in any position of a game, so different strategies can face each other.

    RandomPlayer()                      uniformly random legal move
    BlunderingPlayer(player, rate)      `player`, but a random move with probability `rate`
    MCTSPlayer(game, iterations)        Monte Carlo tree search with random rollouts, reusing its tree between moves
    NNUEPlayer.load(game, path)         greedy one-ply search with an NNUE evaluation
    AlphaBetaPlayer.load(game, path, depth)   alpha-beta search with an NNUE evaluation and the game's
                                        exact results (Awale: the endgame table)

MCTS and alpha-beta run in C: build the game's library with `make -C games/<name>/c`.

parse_players() builds them from command line specs: "random", "mcts:<iterations>",
"mcts:<iterations>:fresh" (new tree every move), a .pt file / a directory of
.pt files for NNUE players, or "ab:<depth>:<.pt file or directory>" for alpha-beta.

Search players (mcts, ab) take a time per move with an "@<seconds>" suffix, such as
"mcts@0.5" or "ab:0:nnue.pt@1". They then search until the time runs out, the
iterations or depth becoming a cap: an omitted iteration count or a depth of 0
means no cap.
"""

import random
from abc import ABC, abstractmethod
from pathlib import Path

import torch

from core.c_mcts import CMCTS
from core.c_search import CAlphaBeta
from core.game import Game, GameState
from core.nnue import NNUE
from core.rl import CHECKPOINT_NAME, WEIGHTS_NAME, choose_move


class Player(ABC):
    name: str

    @abstractmethod
    def choose_move(self, state: GameState) -> int:
        """A legal move for the side to move in a position that is not over."""


class RandomPlayer(Player):
    name = "random"

    def choose_move(self, state: GameState) -> int:
        return random.choice(state.get_possible_moves())


class BlunderingPlayer(Player):
    """Plays a uniformly random legal move with probability `blunder_rate`, else `player`'s move:
    a weaker, more human opponent."""

    def __init__(self, player: Player, blunder_rate: float, name: str | None = None):
        self.player = player
        self.blunder_rate = blunder_rate
        self.name = name or f"{player.name}~{blunder_rate:g}"

    def choose_move(self, state: GameState) -> int:
        if random.random() < self.blunder_rate:
            return random.choice(state.get_possible_moves())
        return self.player.choose_move(state)


class MCTSPlayer(Player):
    def __init__(self, game: Game, iterations: int | None = 1000, reuse: bool = True, move_time: float | None = None):
        self.name = "mcts" + ("" if iterations is None else f":{iterations}") + ("" if reuse else ":fresh") + time_suffix(move_time)
        self.search = CMCTS(game, iterations=iterations, reuse=reuse, move_time=move_time)

    def choose_move(self, state: GameState) -> int:
        return self.search.best_move(state)


class NNUEPlayer(Player):
    def __init__(self, game: Game, net: NNUE, name: str = "nnue"):
        self.name = name
        self.game = game
        self.net = net.eval()

    @classmethod
    def load(cls, game: Game, path: Path) -> "NNUEPlayer":
        return cls(game, load_network(game, path), name=str(path))

    def choose_move(self, state: GameState) -> int:
        slot, successors = choose_move(self.game, self.net, state)
        return successors.moves[slot]


class AlphaBetaPlayer(Player):
    def __init__(self, game: Game, net: NNUE, depth: int, name: str = "ab", exact: bool = True,
                 move_time: float | None = None):
        self.name = name
        self.search = CAlphaBeta(game, net, depth, move_time, exact=exact)

    @classmethod
    def load(cls, game: Game, path: Path, depth: int, move_time: float | None = None) -> "AlphaBetaPlayer":
        name = f"ab:{depth}:{path}{time_suffix(move_time)}"
        return cls(game, load_network(game, path), depth, name=name, move_time=move_time)

    def choose_move(self, state: GameState) -> int:
        return self.search.best_move(state)


def time_suffix(move_time: float | None) -> str:
    return "" if move_time is None else f"@{move_time:g}"


def split_move_time(spec: str) -> tuple[str, float | None]:
    """("mcts", 0.5) for "mcts@0.5"; a spec without a time suffix is returned as is."""
    base, separator, suffix = spec.rpartition("@")
    if not separator:
        return spec, None
    try:
        move_time = float(suffix)
    except ValueError:
        return spec, None  # An "@" belonging to a file name
    if not move_time > 0:
        raise ValueError(f"invalid player {spec!r}: the time per move must be positive")
    return base, move_time


def load_network(game: Game, path: Path) -> NNUE:
    """Load plain weights (nnue.pt, snapshots) or a full rl.py checkpoint."""
    data = torch.load(path, map_location="cpu")
    net = NNUE(game.num_features)
    net.load_state_dict(data["online"] if "online" in data else data)
    return net.eval()


def weight_files(path: Path) -> list[Path]:
    """`path` itself, or the .pt files in it if it is a directory.

    In a directory, rl.py's latest weights and checkpoint are skipped: they duplicate the newest snapshot.
    """
    if not path.is_dir():
        return [path]
    return sorted(file for file in path.glob("*.pt") if file.name not in (WEIGHTS_NAME, CHECKPOINT_NAME))


def parse_players(game: Game, specs: list[str], move_time: float | None = None) -> list[Player]:
    """Players for `specs`; search players without a time suffix get `move_time` seconds per move."""
    players = []
    for full_spec in specs:
        spec, spec_time = split_move_time(full_spec)
        timed = spec_time or move_time
        path = Path(spec)
        if spec_time and (spec == "random" or (path.exists() and not spec.startswith("ab:"))):
            raise ValueError(f"invalid player {full_spec!r}: only search players (mcts, ab) take a time per move")
        if spec == "random":
            players.append(RandomPlayer())
        elif spec.split(":")[0] == "mcts":
            parts = spec.split(":")
            iterations = parts[1] if len(parts) > 1 else ""
            options = parts[2:]
            if not (iterations.isdigit() or iterations == "") or options not in ([], ["fresh"]):
                raise ValueError(f"unknown player {full_spec!r}: expected mcts[:<iterations>[:fresh]][@<seconds>]")
            if iterations == "":
                iterations = None if timed else 1000
            players.append(MCTSPlayer(game, iterations and int(iterations), reuse=not options, move_time=timed))
        elif spec.startswith("ab:"):
            _, depth, weights = (spec.split(":", 2) + [""])[:3]
            if not depth.isdigit() or not Path(weights).exists():
                raise ValueError(f"unknown player {full_spec!r}: expected ab:<depth>:<.pt file or directory>[@<seconds>]")
            if int(depth) < 1 and not timed:
                raise ValueError(f"invalid player {full_spec!r}: depth 0 (no limit) needs a time per move")
            players.extend(AlphaBetaPlayer.load(game, file, int(depth), timed) for file in weight_files(Path(weights)))
        elif path.exists():
            players.extend(NNUEPlayer.load(game, file) for file in weight_files(path))
        else:
            raise ValueError(
                f"unknown player {full_spec!r}: expected random, mcts[:<iterations>[:fresh]][@<seconds>], "
                "ab:<depth>:<weights>[@<seconds>], a .pt file or a directory"
            )
    return players
