"""The C MCTS of core/c/mcts.c for any game, loaded with ctypes.

UCT with random rollouts, reusing the subtree of the position reached since the last
search. Build the game's library first with `make -C games/<name>/c`.
"""

import random

from core.c_library import check_move, library, position_args
from core.game import Game, GameState

NO_LIMIT = -1  # MCTS_NO_LIMIT in mcts.h


class CMCTS:
    """`iterations` caps each search (None: no cap, which needs `move_time` in seconds).

    The C random generator is seeded from Python's `random`, so seeding `random` makes searches repeatable."""

    def __init__(self, game: Game, iterations: int | None = 1000, reuse: bool = True, move_time: float | None = None):
        if iterations is None and move_time is None:
            raise ValueError("unlimited iterations need a time limit")
        self.game = game
        self.lib = library(game)
        seed = random.getrandbits(64) or 1
        self.handle = self.lib.py_mcts_new(NO_LIMIT if iterations is None else iterations, move_time or 0.0, reuse, seed)
        if not self.handle:
            raise MemoryError("could not allocate the C MCTS")

    def best_move(self, state: GameState) -> int | None:
        move = check_move(self.lib.py_mcts_best_move(self.handle, *position_args(self.game, state)), state)
        return None if move < 0 else move

    def __del__(self):
        if getattr(self, "handle", None):
            self.lib.py_mcts_delete(self.handle)
            self.handle = None
