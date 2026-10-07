"""Endgame table: the exact result of every checkers position with up to max_pieces pieces.

Built by retrograde analysis in C (c/endgame.c, which describes the method and the layout),
cached in models/checkers, and read in place by the C alpha-beta as its exact results.
EndgameTable.result gives the same results in Python, for the tests.

Positions are stored as their side to move sees them, at the start of a turn, and their
results follow the draw rules of rules.py:
- values: +t if the side to move wins whenever at least t quiet moves (king steps) are
  left before the QUIET_LIMIT draw, -t if they lose whenever at least t are, 0 for a draw,
- lengths: a bound on the moves in which the winner forces the result, which is only
  claimed with that many moves left before MAX_PLIES (LENGTH_UNKNOWN: too long to tell).

Built on first use (seconds for 4 pieces, minutes for 5); to build one ahead of time:

    uv run python -m games.checkers.endgame [max_pieces]
"""

import ctypes
import time
from functools import lru_cache
from math import comb
from pathlib import Path

import numpy as np

from core.game import ROOT
from games.checkers.rules import EMPTY, NUM_SQUARES, QUIET_LIMIT, BoardState, is_king, owner

DEFAULT_MAX_PIECES = 4
MAX_TABLE_PIECES = 5
LENGTH_UNKNOWN = 255
TABLES_DIR = ROOT / "models" / "checkers"

MEN_SQUARES = 28  # Own men stand on squares 0-27, the opponent's on 4-31
OWN_MEN_AREA = (1 << MEN_SQUARES) - 1
OPPONENT_MEN_AREA = OWN_MEN_AREA << (NUM_SQUARES - MEN_SQUARES)
ALL_SQUARES = (1 << NUM_SQUARES) - 1
OWN_MEN, OWN_KINGS, OPPONENT_MEN, OPPONENT_KINGS = range(4)


def materials(max_pieces: int) -> list[tuple[int, int, int, int]]:
    """(own men, own kings, opponent's men, opponent's kings) of the table's materials, in its order."""
    counts = range(max_pieces + 1)
    return [(om, ok, pm, pk) for om in counts for ok in counts for pm in counts for pk in counts
            if om + ok >= 1 and pm + pk >= 1 and om + ok + pm + pk <= max_pieces]


def relative_sets(state: BoardState) -> list[int]:
    """The pieces as the side to move sees them (the board turned around for player 1), as
    sets of squares (bit masks): own men, own kings, opponent's men, opponent's kings."""
    sets = [0, 0, 0, 0]
    for square, piece in enumerate(state.board):
        if piece != EMPTY:
            seen = square if state.current_player == 0 else NUM_SQUARES - 1 - square
            sets[(OWN_MEN if owner(piece) == state.current_player else OPPONENT_MEN) + is_king(piece)] |= 1 << seen
    return sets


def rank_set(squares: int, area: int) -> int:
    """The rank of a set of squares of `area` among the sets of its size: a combination
    (colex) of their places in `area`."""
    rank, size = 0, 1
    while squares:
        lowest = squares & -squares
        rank += comb((area & (lowest - 1)).bit_count(), size)
        squares ^= lowest
        size += 1
    return rank


class EndgameTable:
    def __init__(self, max_pieces: int, values: np.ndarray, lengths: np.ndarray):
        self.max_pieces = max_pieces
        self.values = values  # int8
        self.lengths = lengths  # uint8
        self.blocks = {}  # Material: (its first entry, entries per placement of its men)
        size = 0
        for material in materials(max_pieces):
            own_men, own_kings, opponent_men, opponent_kings = material
            free = NUM_SQUARES - own_men - opponent_men
            kings_size = comb(free, own_kings) * comb(free - own_kings, opponent_kings)
            self.blocks[material] = size, kings_size
            size += comb(MEN_SQUARES, own_men) * comb(MEN_SQUARES, opponent_men) * kings_size
        if not len(values) == len(lengths) == size:
            raise ValueError(f"the table of up to {max_pieces} pieces has {size} entries, not {len(values)}")

    def index(self, state: BoardState) -> int | None:
        """The entry of a position at the start of a turn, None if the table does not hold its material."""
        if state.jumping is not None:
            return None
        sets = relative_sets(state)
        material = tuple(squares.bit_count() for squares in sets)
        if material not in self.blocks:
            return None
        first, kings_size = self.blocks[material]
        own_men, own_kings, opponent_men, opponent_kings = material
        men_rank = (rank_set(sets[OWN_MEN], OWN_MEN_AREA) * comb(MEN_SQUARES, opponent_men)
                    + rank_set(sets[OPPONENT_MEN], OPPONENT_MEN_AREA))
        free = ALL_SQUARES & ~(sets[OWN_MEN] | sets[OPPONENT_MEN])
        kings_rank = (rank_set(sets[OWN_KINGS], free) * comb(NUM_SQUARES - own_men - opponent_men - own_kings, opponent_kings)
                      + rank_set(sets[OPPONENT_KINGS], free & ~sets[OWN_KINGS]))
        return first + men_rank * kings_size + kings_rank

    def result(self, state: BoardState) -> int | None:
        """+1 / 0 / -1 for the side to move with perfect play, as the C alpha-beta reads it; None when
        the table does not hold the position, or for a win the game may stop at MAX_PLIES before."""
        index = self.index(state)
        if index is None:
            return None
        value, length = int(self.values[index]), int(self.lengths[index])
        if value == 0 or abs(value) > QUIET_LIMIT - state.quiet_moves:
            return 0
        if length == LENGTH_UNKNOWN or length > state.moves_left:
            return None
        return 1 if value > 0 else -1

    @classmethod
    def build(cls, max_pieces: int) -> "EndgameTable":
        from core.c_library import library
        from games.checkers import GAME

        lib = library(GAME)
        lib.checkers_endgame_size.argtypes = [ctypes.c_int]
        lib.checkers_endgame_size.restype = ctypes.c_int64
        lib.checkers_endgame_build.argtypes = [
            ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int8), ctypes.POINTER(ctypes.c_uint8),
        ]
        lib.checkers_endgame_build.restype = ctypes.c_int
        size = lib.checkers_endgame_size(max_pieces)
        if size < 0:
            raise ValueError(f"max_pieces must be between 2 and {MAX_TABLE_PIECES}")
        values, lengths = np.zeros(size, dtype=np.int8), np.zeros(size, dtype=np.uint8)
        for pieces in range(2, max_pieces + 1):  # Each from the ones with fewer pieces
            start = time.monotonic()
            pointers = values.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)), lengths.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
            if not lib.checkers_endgame_build(max_pieces, pieces, *pointers):
                raise RuntimeError(f"building the {pieces}-piece positions failed (out of memory, or see the C error above)")
            print(f"  {pieces} pieces solved in {time.monotonic() - start:.1f}s", flush=True)
        return cls(max_pieces, values, lengths)

    def save(self, path: Path):
        np.savez_compressed(path, max_pieces=self.max_pieces, values=self.values, lengths=self.lengths)

    @classmethod
    def load(cls, path: Path) -> "EndgameTable":
        data = np.load(path)
        return cls(int(data["max_pieces"]), data["values"], data["lengths"])


@lru_cache
def endgame_table(max_pieces: int = DEFAULT_MAX_PIECES) -> EndgameTable:
    """The table for `max_pieces`, built on first use and cached in models/."""
    path = TABLES_DIR / f"endgame_{max_pieces}.npz"
    if path.exists():
        return EndgameTable.load(path)
    print(f"building the endgame table for up to {max_pieces} pieces...", flush=True)
    start = time.monotonic()
    table = EndgameTable.build(max_pieces)
    path.parent.mkdir(parents=True, exist_ok=True)
    table.save(path)
    print(f"built {len(table.values)} entries in {time.monotonic() - start:.1f}s, saved to {path}", flush=True)
    return table


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build the endgame table ahead of time.")
    parser.add_argument("max_pieces", type=int, nargs="?", default=DEFAULT_MAX_PIECES)
    endgame_table(parser.parse_args().max_pieces)
