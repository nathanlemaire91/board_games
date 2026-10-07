"""Awale (Oware abapa): rules, NNUE features, endgame table, desktop and web interfaces.

    rules.py      the rules (BoardState), used by training, the endgame table and the interfaces
    features.py   the NNUE inputs
    endgame.py    exact results for positions with few seeds left, used by the alpha-beta
    play.py       desktop window (renderer.py): uv run python -m games.awale.play human mcts@1
    c/            the rules in C and their side of core/c/game_api.h: make -C games/awale/c
    web/          the browser page served by core/server.py
    tests/        parity checks between the C and Python code (make -C games/awale/c check)
"""

import ctypes

from core.game import Game
from games.awale import features
from games.awale.rules import HALF_BOARD_SIZE, MAX_MOVES, BoardState, new_game


def encode(state: BoardState) -> list[int]:
    """The ints game_decode reads in c/game.c."""
    return [*state.board, *state.players_seeds, int(state.current_player), state.moves_played]


def to_json(state: BoardState) -> dict:
    return {"board": state.board, "seeds": state.players_seeds}


def attach_endgame_table(searcher):
    """Gives a C alpha-beta searcher the endgame table, which it reads in place."""
    from games.awale.endgame import endgame_table  # Loads or builds the table: only when searching

    table = endgame_table()
    gains = table.gains
    if gains.dtype != "int8" or not gains.flags.c_contiguous or gains.shape[0] != MAX_MOVES + 1:
        raise ValueError("the endgame table's gains must be a contiguous int8 array of MAX_MOVES + 1 rows")
    set_endgame = searcher.lib.awale_set_endgame
    set_endgame.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int8), ctypes.c_int64, ctypes.c_int]
    set_endgame.restype = ctypes.c_int
    pointer = gains.ctypes.data_as(ctypes.POINTER(ctypes.c_int8))
    if not set_endgame(searcher.handle, pointer, gains.shape[1], table.max_seeds):
        raise ValueError(f"the endgame table does not hold every board of up to {table.max_seeds} seeds")
    searcher.keep_alive.append(table)


GAME = Game(
    name="awale",
    new_game=new_game,
    max_legal_moves=HALF_BOARD_SIZE,
    num_features=features.NUM_FEATURES,
    num_active_features=features.NUM_ACTIVE_FEATURES,
    features=features.feature_indices,
    encode=encode,
    to_json=to_json,
    setup_search=attach_endgame_table,
)
