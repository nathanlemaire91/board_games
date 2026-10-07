"""Checks the endgame table that c/endgame.c builds against rules.py.

    uv run python -m games.checkers.tests.check_endgame [--max-pieces 4] [--samples 200]

(after `make -C games/checkers/c`; builds the table if missing). Checks:
- indices: the C and Python entries of random positions are the same,
- recurrence: for positions sampled from every material, the stored result and length are
  what the turns rules.py allows and the stored entries of the positions they lead to give
  (c/endgame.c's definitions: a win needs a winning turn, a loss every turn losing),
- two pieces, exhaustively: the results agree with a plain minimax over the game itself,
  quiet moves counted, for every number of quiet moves left; and a win or a loss stays one
  with only `length` moves left before MAX_PLIES, as the table claims,
- short wins of every material: with `length` moves left and the fewest quiet moves the
  table says it takes, the C alpha-beta without the table, searching to the end of the
  game, finds the same result; with one quiet move less, a draw,
- known endgames: two kings beat one, three kings beat one, one king does not beat one.
"""

import argparse
import ctypes
import random
import sys
from functools import lru_cache
from pathlib import Path

from core.c_library import library, position_args
from core.c_search import CAlphaBeta
from core.nnue import NNUE
from games.checkers import GAME
from games.checkers.endgame import LENGTH_UNKNOWN, EndgameTable, endgame_table, materials
from games.checkers.rules import EMPTY, MAX_PLIES, NUM_SQUARES, QUIET_LIMIT, BoardState, make_piece, owner

sys.setrecursionlimit(100_000)  # The minimax follows whole quiet sequences


def check(condition: bool, message: str):
    if not condition:
        sys.exit(f"endgame table: {message}")


def random_position(material: tuple[int, int, int, int], player: int, rng: random.Random) -> BoardState:
    """A position of that material, `player` to move: own men and kings first, as `player` sees the board."""
    own_men, own_kings, opponent_men, opponent_kings = material
    men = rng.sample(range(28), own_men)
    opponent = rng.sample([square for square in range(4, NUM_SQUARES) if square not in men], opponent_men)
    free = [square for square in range(NUM_SQUARES) if square not in men + opponent]
    kings = rng.sample(free, own_kings + opponent_kings)
    board = [EMPTY] * NUM_SQUARES
    for squares, piece in ((men, make_piece(player, False)), (kings[:own_kings], make_piece(player, True)),
                           (opponent, make_piece(1 - player, False)), (kings[own_kings:], make_piece(1 - player, True))):
        for square in squares:
            board[square if player == 0 else NUM_SQUARES - 1 - square] = piece
    return BoardState(board, player)


def turns(state: BoardState) -> list[tuple[BoardState, int, bool]]:
    """(position after the turn, its moves, whether it is a king step) for each way to play the turn."""
    result = []
    for move in state.get_possible_moves():
        child = state.copy()
        child.make_move(move)
        if child.current_player == state.current_player:  # A multiple jump goes on
            result += [(after, moves + 1, False) for after, moves, _ in turns(child)]
        else:
            result.append((child, 1, child.quiet_moves > state.quiet_moves))
    return result


def stored(table: EndgameTable, state: BoardState) -> tuple[int, int]:
    """The entry of a position at the start of a turn; a side without pieces has lost."""
    index = table.index(state)
    if index is None:
        check(not any(piece != EMPTY and owner(piece) == state.current_player for piece in state.board),
              f"no entry for {state}")
        return -1, 0
    return int(table.values[index]), int(table.lengths[index])


def value_at(state: BoardState, value: int, quiet_left: int) -> int:
    """A position's result with that many quiet moves left: lost without a move, even with none left."""
    if not state.get_possible_moves():
        return -1
    return 0 if value == 0 or abs(value) > quiet_left else (1 if value > 0 else -1)


def add_length(moves: int, length: int) -> int:
    return min(moves + length, LENGTH_UNKNOWN)


def expected_entry(table: EndgameTable, state: BoardState) -> tuple[int, int]:
    """The entry the turns from `state` give, from the entries of where they lead."""
    options = [(child, moves, quiet, *stored(table, child)) for child, moves, quiet in turns(state)]
    if not options:
        return -1, 0
    for quiet_left in range(1, QUIET_LIMIT + 1):
        # A king step leaves one quiet move less, any other move resets the count
        results = [-value_at(child, value, quiet_left - 1 if quiet else QUIET_LIMIT)
                   for child, moves, quiet, value, length in options]
        best = max(results)
        if best > 0:
            return quiet_left, min(add_length(option[1], option[4]) for option, result in zip(options, results) if result > 0)
        if best < 0:
            return -quiet_left, max(add_length(option[1], option[4]) for option in options)
    return 0, 0


def two_piece_positions() -> list[BoardState]:
    """Every position of two pieces, both sides to move, game not over."""
    positions = []
    for material in materials(2):
        for player in (0, 1):
            own_men, own_kings, opponent_men, opponent_kings = material
            for first in range(NUM_SQUARES):
                for second in range(NUM_SQUARES):
                    own_piece, own_area = (make_piece(player, False), range(28)) if own_men else (make_piece(player, True), range(32))
                    other, other_area = (make_piece(1 - player, False), range(4, 32)) if opponent_men else (make_piece(1 - player, True), range(32))
                    if first == second or first not in own_area or second not in other_area:
                        continue
                    board = [EMPTY] * NUM_SQUARES
                    board[first if player == 0 else NUM_SQUARES - 1 - first] = own_piece
                    board[second if player == 0 else NUM_SQUARES - 1 - second] = other
                    state = BoardState(board, player)
                    if state.get_possible_moves():
                        positions.append(state)
    return positions


@lru_cache(maxsize=None)
def quiet_minimax(board: tuple, player: int, jumping: int | None, quiet_moves: int) -> int:
    """The game's result for the side to move under the quiet move limit alone (the move count kept at 0)."""
    state = BoardState(list(board), player, jumping, quiet_moves)
    if state.is_over():
        return state.result(player)
    best = -1
    for move in state.get_possible_moves():
        child = state.copy()
        child.make_move(move)
        value = quiet_minimax(tuple(child.board), child.current_player, child.jumping, child.quiet_moves)
        best = max(best, value if child.current_player == player else -value)
    return best


@lru_cache(maxsize=None)
def minimax(key: tuple) -> int:
    """The game's result for the side to move, every rule included (key: BoardState.key())."""
    *board, jumping, player, quiet_moves, moves_played = key
    state = BoardState(board, player, jumping, quiet_moves, moves_played)
    if state.is_over():
        return state.result(player)
    best = -1
    for move in state.get_possible_moves():
        child = state.copy()
        child.make_move(move)
        value = minimax(child.key())
        best = max(best, value if child.current_player == player else -value)
    return best


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--max-pieces", type=int, default=4)
    parser.add_argument("--table", type=Path, help="A table file to check (default: the cached one, built if missing)")
    parser.add_argument("--samples", type=int, default=200, help="Positions per material for the recurrence check")
    parser.add_argument("--max-length", type=int, default=20, help="Longest wins searched to the end")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    table = EndgameTable.load(args.table) if args.table else endgame_table(args.max_pieces)
    all_materials = materials(table.max_pieces)

    index = library(GAME).checkers_endgame_index
    index.argtypes = [ctypes.POINTER(ctypes.c_int32), ctypes.c_int, ctypes.c_int]
    index.restype = ctypes.c_int64
    positions = [random_position(rng.choice(all_materials), rng.randrange(2), rng) for _ in range(5000)]
    bad = sum(index(*position_args(GAME, state), table.max_pieces) != table.index(state) for state in positions)
    check(bad == 0, f"{bad} of {len(positions)} C indices differ from Python's")
    print(f"indices: C and Python agree on {len(positions)} positions")

    checked = 0
    for material in all_materials:
        for _ in range(args.samples):
            state = random_position(material, rng.randrange(2), rng)
            entry = stored(table, state)
            check(entry == expected_entry(table, state), f"{state}: stored {entry}, its turns give {expected_entry(table, state)}")
            checked += 1
    print(f"recurrence: {checked} positions of {len(all_materials)} materials follow from the positions after their turns")

    positions = two_piece_positions()
    for state in positions:
        value, length = stored(table, state)
        for quiet_left in range(1, QUIET_LIMIT + 1):
            actual = quiet_minimax(tuple(state.board), state.current_player, None, QUIET_LIMIT - quiet_left)
            check(actual == value_at(state, value, quiet_left), f"{state} with {quiet_left} quiet moves left: "
                  f"the game gives {actual}, the table {value}")
        if value:  # With the fewest quiet moves and only `length` moves left, still a win or a loss
            late = BoardState(state.board[:], state.current_player, None, QUIET_LIMIT - abs(value), MAX_PLIES - length)
            check(minimax(late.key()) == (1 if value > 0 else -1) == table.result(late), f"{late}: not decided in {length} moves")
    print(f"two pieces: {len(positions)} positions match the game's minimax for each number of quiet moves left, "
          "and their wins and losses come within their lengths")

    # Every line ends within `length` moves, at MAX_PLIES: a search that deep, without the table, is exact
    net = NNUE(GAME.num_features)
    searched = 0
    for material in all_materials:
        for _ in range(args.samples):
            state = random_position(material, rng.randrange(2), rng)
            value, length = stored(table, state)
            if not value or length > args.max_length or not state.get_possible_moves():
                continue
            for quiet_left, expected in ((abs(value), 1 if value > 0 else -1), (abs(value) - 1, 0)):
                if quiet_left == 0:
                    continue
                late = BoardState(state.board[:], state.current_player, None, QUIET_LIMIT - quiet_left, MAX_PLIES - length)
                search = CAlphaBeta(GAME, net, length, exact=False)
                search.best_move(late)
                _, _, score = search.stats()
                check(score == expected, f"{late}: the search to the end gives {score}, the table {expected}")
            searched += 1
    print(f"short wins: {searched} positions of length up to {args.max_length} are won or lost in that many moves "
          "with the quiet moves the table says, drawn with one less, searched to the end")

    def share(material: tuple[int, int, int, int], result: int) -> float:
        """The share of random positions of that material with that result, all quiet moves left."""
        values = [stored(table, random_position(material, 0, rng))[0] for _ in range(2000)]
        return sum((value > 0) - (value < 0) == result for value in values) / len(values)

    if table.max_pieces >= 4:
        two_kings, three_kings, one_king = share((0, 2, 0, 1), 1), share((0, 3, 0, 1), 1), share((0, 1, 0, 1), 0)
        check(two_kings > 0.99 and three_kings == 1.0 and one_king > 0.5, f"two kings beat one in {two_kings:.1%} "
              f"of positions, three in {three_kings:.1%}, one king draws one in {one_king:.1%}")
        print(f"known endgames: two kings beat one in {two_kings:.1%} of positions, three kings in {three_kings:.1%}, "
              f"one king draws against one in {one_king:.1%}")


if __name__ == "__main__":
    main()
