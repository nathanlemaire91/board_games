"""Players that pick a move for any Awale position, so different strategies can face each other.

    RandomPlayer()              uniformly random legal move
    MCTSPlayer(iterations)      Monte Carlo tree search with random rollouts, reusing its tree between moves
                                (the C implementation: build it with `make -C src_c`)
    NNUEPlayer.load(path)       greedy one-ply search with an NNUE evaluation
    AlphaBetaPlayer.load(path, depth)   alpha-beta search with an NNUE evaluation and the endgame table

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

from awale import BoardState
from endgame import endgame_table
from c_mcts import CMCTS
from nnue import NNUE
from rl import CHECKPOINT_NAME, WEIGHTS_NAME, choose_move
from search import AlphaBeta


class Player(ABC):
    name: str

    @abstractmethod
    def choose_move(self, state: BoardState) -> int:
        """A legal move for the side to move in a position that is not over."""


class RandomPlayer(Player):
    name = "random"

    def choose_move(self, state: BoardState) -> int:
        return random.choice(state.get_possible_moves())


class MCTSPlayer(Player):
    def __init__(self, iterations: int | None = 1000, reuse: bool = True, move_time: float | None = None):
        self.name = "mcts" + ("" if iterations is None else f":{iterations}") + ("" if reuse else ":fresh") + time_suffix(move_time)
        self.search = CMCTS(iterations=iterations, reuse=reuse, move_time=move_time)

    def choose_move(self, state: BoardState) -> int:
        return self.search.best_move(state)


class NNUEPlayer(Player):
    def __init__(self, net: NNUE, name: str = "nnue"):
        self.name = name
        self.net = net.eval()

    @classmethod
    def load(cls, path: Path) -> "NNUEPlayer":
        return cls(load_network(path), name=str(path))

    def choose_move(self, state: BoardState) -> int:
        slot, successors = choose_move(self.net, state)
        return successors.moves[slot]


class AlphaBetaPlayer(Player):
    def __init__(self, net: NNUE, depth: int, name: str = "ab", use_endgame_table: bool = True, move_time: float | None = None):
        self.name = name
        self.search = AlphaBeta(net, depth, endgame_table() if use_endgame_table else None, move_time)

    @classmethod
    def load(cls, path: Path, depth: int, move_time: float | None = None) -> "AlphaBetaPlayer":
        return cls(load_network(path), depth, name=f"ab:{depth}:{path}{time_suffix(move_time)}", move_time=move_time)

    def choose_move(self, state: BoardState) -> int:
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


def load_network(path: Path) -> NNUE:
    """Load plain weights (nnue.pt, snapshots) or a full rl.py checkpoint."""
    data = torch.load(path, map_location="cpu")
    net = NNUE()
    net.load_state_dict(data["online"] if "online" in data else data)
    return net.eval()


def weight_files(path: Path) -> list[Path]:
    """`path` itself, or the .pt files in it if it is a directory.

    In a directory, rl.py's latest weights and checkpoint are skipped: they duplicate the newest snapshot.
    """
    if not path.is_dir():
        return [path]
    return sorted(file for file in path.glob("*.pt") if file.name not in (WEIGHTS_NAME, CHECKPOINT_NAME))


def parse_players(specs: list[str], move_time: float | None = None) -> list[Player]:
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
            players.append(MCTSPlayer(iterations and int(iterations), reuse=not options, move_time=timed))
        elif spec.startswith("ab:"):
            _, depth, weights = (spec.split(":", 2) + [""])[:3]
            if not depth.isdigit() or not Path(weights).exists():
                raise ValueError(f"unknown player {full_spec!r}: expected ab:<depth>:<.pt file or directory>[@<seconds>]")
            if int(depth) < 1 and not timed:
                raise ValueError(f"invalid player {full_spec!r}: depth 0 (no limit) needs a time per move")
            players.extend(AlphaBetaPlayer.load(file, int(depth), timed) for file in weight_files(Path(weights)))
        elif path.exists():
            players.extend(NNUEPlayer.load(file) for file in weight_files(path))
        else:
            raise ValueError(
                f"unknown player {full_spec!r}: expected random, mcts[:<iterations>[:fresh]][@<seconds>], "
                "ab:<depth>:<weights>[@<seconds>], a .pt file or a directory"
            )
    return players
