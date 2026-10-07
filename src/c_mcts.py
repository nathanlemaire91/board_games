"""The C MCTS of src_c/, loaded with ctypes. Same search as mcts.py, many times faster.

Build the library first with `make -C src_c` (from the repository root).
"""

import ctypes
import random
from pathlib import Path

from awale import BOARD_SIZE, BoardState

SOURCE_DIR = Path(__file__).resolve().parent.parent / "src_c"
LIBRARY_PATH = SOURCE_DIR / "libawale.so"
LIBRARY_SOURCES = ("awale.c", "awale.h", "mcts.c", "mcts.h", "python_api.c")
NO_LIMIT = -1  # MCTS_NO_LIMIT in mcts.h

_library = None


def library() -> ctypes.CDLL:
    """libawale.so, loaded on first use so that importing this module never needs it."""
    global _library
    if _library is None:
        if not LIBRARY_PATH.exists():
            raise RuntimeError(f"{LIBRARY_PATH} is missing: build it with `make -C {SOURCE_DIR}`")
        newest_source = max((SOURCE_DIR / name).stat().st_mtime for name in LIBRARY_SOURCES)
        if newest_source > LIBRARY_PATH.stat().st_mtime:
            raise RuntimeError(f"{LIBRARY_PATH} is older than its sources: rebuild it with `make -C {SOURCE_DIR}`")
        lib = ctypes.CDLL(str(LIBRARY_PATH))
        lib.awale_mcts_new.argtypes = [ctypes.c_long, ctypes.c_double, ctypes.c_int, ctypes.c_uint64]
        lib.awale_mcts_new.restype = ctypes.c_void_p
        lib.awale_mcts_delete.argtypes = [ctypes.c_void_p]
        lib.awale_mcts_delete.restype = None
        lib.awale_mcts_best_move.argtypes = [ctypes.c_void_p, ctypes.c_uint8 * BOARD_SIZE] + [ctypes.c_int] * 4
        lib.awale_mcts_best_move.restype = ctypes.c_int
        _library = lib
    return _library


class CMCTS:
    """Like mcts.MCTS: `iterations` caps each search (None: no cap, which needs `move_time` in seconds).

    The C random generator is seeded from Python's `random`, so seeding `random` makes searches repeatable."""

    def __init__(self, iterations: int | None = 1000, reuse: bool = True, move_time: float | None = None):
        if iterations is None and move_time is None:
            raise ValueError("unlimited iterations need a time limit")
        self.lib = library()
        seed = random.getrandbits(64) or 1
        self.handle = self.lib.awale_mcts_new(
            NO_LIMIT if iterations is None else iterations, move_time or 0.0, reuse, seed
        )
        if not self.handle:
            raise MemoryError("could not allocate the C MCTS")

    def best_move(self, state: BoardState) -> int | None:
        board = (ctypes.c_uint8 * BOARD_SIZE)(*state.board)
        move = self.lib.awale_mcts_best_move(
            self.handle, board, *state.players_seeds, int(state.current_player), state.moves_played
        )
        return None if move < 0 else move

    def __del__(self):
        if getattr(self, "handle", None):
            self.lib.awale_mcts_delete(self.handle)
            self.handle = None
