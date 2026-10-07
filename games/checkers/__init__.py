"""Checkers (English draughts): rules, NNUE features, web interface.

    rules.py      the rules (BoardState), used by training and the interfaces
    features.py   the NNUE inputs
    endgame.py    endgame table, exact results for the alpha-beta (built in c/endgame.c)
    c/            the rules in C and their side of core/c/game_api.h: make -C games/checkers/c
    web/          the browser page served by core/server.py --game checkers
    tests/        parity checks between the C and Python code (make -C games/checkers/c check)

A multiple jump is played as one move per jump, the same player moving again: the
engines of core/ handle a player moving several times in a row.
"""

import ctypes

from core.game import Game
from games.checkers import features
from games.checkers.rules import MAX_LEGAL_MOVES, MAX_PLIES, QUIET_LIMIT, BoardState, move_squares, new_game


def encode(state: BoardState) -> list[int]:
    """The ints game_decode reads in c/game.c."""
    jumping = -1 if state.jumping is None else state.jumping
    return [*state.board, jumping, state.current_player, state.quiet_moves, state.moves_played]


def to_json(state: BoardState) -> dict:
    """The board, and each legal move's squares so the page can play them by clicks."""
    moves = [] if state.is_over() else state.get_possible_moves()
    return {
        "board": state.board,
        "jumping": state.jumping,
        "quiet_moves": state.quiet_moves,
        "quiet_limit": QUIET_LIMIT,
        "max_plies": MAX_PLIES,
        "moves": [dict(zip(("move", "from", "to", "captured"), (move, *move_squares(state, move)))) for move in moves],
    }


def attach_endgame_table(searcher):
    """Gives a C alpha-beta searcher the endgame table, which it reads in place."""
    from games.checkers.endgame import endgame_table  # Loads or builds the table: only when searching

    table = endgame_table()
    set_endgame = searcher.lib.checkers_set_endgame
    set_endgame.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int8), ctypes.POINTER(ctypes.c_uint8),
                            ctypes.c_int64, ctypes.c_int]
    set_endgame.restype = ctypes.c_int
    values = table.values.ctypes.data_as(ctypes.POINTER(ctypes.c_int8))
    lengths = table.lengths.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
    if not set_endgame(searcher.handle, values, lengths, len(table.values), table.max_pieces):
        raise ValueError(f"the endgame table does not hold every position of up to {table.max_pieces} pieces")
    searcher.keep_alive.append(table)


GAME = Game(
    name="checkers",
    new_game=new_game,
    max_legal_moves=MAX_LEGAL_MOVES,
    num_features=features.NUM_FEATURES,
    num_active_features=features.NUM_ACTIVE_FEATURES,
    features=features.feature_indices,
    encode=encode,
    to_json=to_json,
    setup_search=attach_endgame_table,
)
