"""Awale's NNUE inputs: one-hot "hole h holds n seeds" plus "store holds n seeds", seen
from each player's point of view (own holes first). Every position has exactly
NUM_ACTIVE_FEATURES active features, one per slot, and a move only changes the slots
it touched, so the accumulators are updated incrementally like in Stockfish.

Must match game_features in c/game.c.
"""

from games.awale.rules import BOARD_SIZE, HALF_BOARD_SIZE, BoardState

MAX_HOLE_SEEDS = 24  # Seed counts above this share the last bucket
MAX_STORE_SEEDS = 48
HOLE_BUCKETS = MAX_HOLE_SEEDS + 1
STORE_BUCKETS = MAX_STORE_SEEDS + 1

NUM_FEATURES = BOARD_SIZE * HOLE_BUCKETS + 2 * STORE_BUCKETS
NUM_ACTIVE_FEATURES = BOARD_SIZE + 2


def feature_indices(state: BoardState, perspective: int) -> list[int]:
    """Active feature indices of a position seen from `perspective`'s side."""
    features = []
    for slot in range(BOARD_SIZE):
        hole = (slot + perspective * HALF_BOARD_SIZE) % BOARD_SIZE
        seeds = min(state.board[hole], MAX_HOLE_SEEDS)
        features.append(slot * HOLE_BUCKETS + seeds)
    store_offset = BOARD_SIZE * HOLE_BUCKETS
    own_store = min(state.players_seeds[perspective], MAX_STORE_SEEDS)
    opponent_store = min(state.players_seeds[1 - perspective], MAX_STORE_SEEDS)
    features.append(store_offset + own_store)
    features.append(store_offset + STORE_BUCKETS + opponent_store)
    return features
