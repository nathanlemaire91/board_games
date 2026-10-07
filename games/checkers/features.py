"""Checkers' NNUE inputs, seen from each player's side with the board turned so that their
pieces move up it: one slot per square for what it holds (empty, own man, own king,
opponent's man, opponent's king), and one slot for the square of the piece in the middle
of a multiple jump, or none. Every position has exactly NUM_ACTIVE_FEATURES active
features, one per slot, and a move only changes the slots it touched (two squares for a
step, three for a jump, and the jump slot), so the accumulators are updated incrementally
like in Stockfish.

Must match game_features in c/game.c.
"""

from games.checkers.rules import EMPTY, NUM_SQUARES, BoardState, is_king, owner

SQUARE_VALUES = 5  # Empty, own man, own king, opponent's man, opponent's king
JUMPING_VALUES = NUM_SQUARES + 1  # No multiple jump going on, or the jumping piece's square

NUM_FEATURES = NUM_SQUARES * SQUARE_VALUES + JUMPING_VALUES
NUM_ACTIVE_FEATURES = NUM_SQUARES + 1


def seen_from(square: int, perspective: int) -> int:
    """The square as `perspective` sees the board: turned around for player 1."""
    return square if perspective == 0 else NUM_SQUARES - 1 - square


def feature_indices(state: BoardState, perspective: int) -> list[int]:
    """Active feature indices of a position seen from `perspective`'s side."""
    features = []
    for slot in range(NUM_SQUARES):
        piece = state.board[seen_from(slot, perspective)]
        value = 0 if piece == EMPTY else 1 + is_king(piece) + 2 * (owner(piece) != perspective)
        features.append(slot * SQUARE_VALUES + value)
    jumping = 0 if state.jumping is None else 1 + seen_from(state.jumping, perspective)
    features.append(NUM_SQUARES * SQUARE_VALUES + jumping)
    return features
