"""The C alpha-beta of core/c/search.c for any game, loaded with ctypes.

Negamax alpha-beta with iterative deepening, a transposition table, the game's exact
results (Awale's endgame table) and an optional time limit. Leaves are scored by the
NNUE network in C (core/c/nnue.c), whose accumulators are updated incrementally move
by move as in Stockfish. Build the game's library first with `make -C games/<name>/c`.
"""

import ctypes

import numpy as np
import torch

from core.c_library import check_move, library, position_args
from core.game import Game, GameState
from core.nnue import NNUE

TABLE_BITS = 20  # 2^20 transposition entries of 16 bytes: 16 MB per searcher
NO_DEPTH_LIMIT = 1000  # Beyond any game's length: the C code caps it at the plies left


def network_weights(net: NNUE) -> np.ndarray:
    """The network's parameters as one float32 array, in the field order of Network in nnue.h.
    The linear layers are transposed to input-major, the layout the C forward pass reads."""
    linears = [layer for layer in net.layers if isinstance(layer, torch.nn.Linear)]
    parts = [net.feature_transformer.weight, net.feature_bias]
    for layer in linears:
        parts += [layer.weight.T, layer.bias]
    return np.concatenate([part.detach().cpu().float().numpy().ravel() for part in parts])


class CAlphaBeta:
    """Searches to `depth`, or as deep as `move_time` seconds allow (depth 0: no depth limit).

    With `exact`, the game's setup_search attaches its exact results (such as an endgame table)."""

    def __init__(self, game: Game, net: NNUE, depth: int, move_time: float | None = None, exact: bool = True,
                 table_bits: int = TABLE_BITS):
        if depth < 1 and move_time is None:
            raise ValueError("an unlimited depth needs a time limit")
        self.game = game
        self.lib = library(game)
        self.keep_alive = []  # Data the C code reads in place, such as an endgame table
        weights = network_weights(net)
        expected = self.lib.py_network_floats()
        if len(weights) != expected:
            raise ValueError(f"the network has {len(weights)} parameters, {game.name}'s C code expects {expected}: "
                             "the network and the game's C feature count or nnue.h disagree")
        self.handle = self.lib.py_ab_new(
            weights.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), len(weights),
            min(depth, NO_DEPTH_LIMIT), move_time or 0.0, table_bits,
        )
        if not self.handle:
            raise MemoryError("could not allocate the C alpha-beta")
        if exact and game.setup_search is not None:
            game.setup_search(self)

    def best_move(self, state: GameState) -> int:
        return check_move(self.lib.py_ab_best_move(self.handle, *position_args(self.game, state)), state)

    def stats(self) -> tuple[int, int, float]:
        """Of the last search: nodes searched, depth completed, score for the side to move."""
        nodes, depth, score = ctypes.c_uint64(), ctypes.c_int(), ctypes.c_float()
        self.lib.py_ab_stats(self.handle, ctypes.byref(nodes), ctypes.byref(depth), ctypes.byref(score))
        return nodes.value, depth.value, score.value

    def evaluate(self, state: GameState) -> float:
        """The C network's evaluation, for the side to move, in [-1, 1]."""
        return self.lib.py_ab_evaluate(self.handle, *position_args(self.game, state))

    def __del__(self):
        if getattr(self, "handle", None):
            self.lib.py_ab_delete(self.handle)
            self.handle = None
