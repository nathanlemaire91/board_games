"""Checks rules.py against known move counts and hand-built positions, then that the C
rules match rules.py, position by position, over random games.

    uv run python -m games.checkers.tests.check_rules [games]      (after `make -C games/checkers/c`)

- Perft: the number of move sequences from the opening, counted in turns (a multiple
  jump being one), must be the published counts of English draughts.
- Positions: multiple jumps, crowning, kings and men, the end of the game.
- C vs Python: plays random games with the Python rules, replays their moves through
  checkers_trace and compares every position, legal move list, end of game and winner.
"""

import random
import subprocess
import sys
from pathlib import Path

from games.checkers.rules import (
    DOWN_LEFT, EMPTY, MAX_PLIES, NUM_DIRECTIONS, NUM_SQUARES, QUIET_LIMIT, UP_LEFT, UP_RIGHT, BoardState, make_piece,
    new_game,
)

TRACE = Path(__file__).resolve().parent.parent / "c" / "checkers_trace"
PERFT = [7, 49, 302, 1469, 7361, 36768, 179740, 845931]  # Depths 1 to 8
MAN_0, KING_0, MAN_1, KING_1 = make_piece(0, False), make_piece(0, True), make_piece(1, False), make_piece(1, True)


def perft(state: BoardState, depth: int) -> int:
    if depth == 0:
        return 1
    total = 0
    for move in state.get_possible_moves():
        child = state.copy()
        child.make_move(move)
        total += perft(child, depth if child.current_player == state.current_player else depth - 1)
    return total


def position(pieces: dict[int, int], player: int = 0, quiet_moves: int = 0, moves_played: int = 0) -> BoardState:
    board = [EMPTY] * NUM_SQUARES
    for square, piece in pieces.items():
        board[square] = piece
    return BoardState(board, player, quiet_moves=quiet_moves, moves_played=moves_played)


def move(square: int, direction: int) -> int:
    return square * NUM_DIRECTIONS + direction


def check(condition: bool, message: str):
    if not condition:
        sys.exit(f"rules.py: {message}")


def check_positions():
    # A double jump, 9 over 13 to 18 then over 21 to 25: the same player moves again, with that piece only
    state = position({9: MAN_0, 2: MAN_0, 13: MAN_1, 21: MAN_1, 5: MAN_1, 30: MAN_1})
    check(state.get_possible_moves() == [move(9, UP_RIGHT)], "capturing is mandatory")
    state.make_move(move(9, UP_RIGHT))
    check(state.current_player == 0 and state.jumping == 18, "after a jump that can go on, the same player moves")
    check(state.board[13] == EMPTY and state.board[9] == EMPTY and state.board[18] == MAN_0, "a jump captures")
    # The man on 2 could now jump 5 too, but only the jumping piece moves
    check(state.get_possible_moves() == [move(18, UP_LEFT)], "only the jumping piece moves")
    state.make_move(move(18, UP_LEFT))
    check(state.current_player == 1 and state.jumping is None and state.board[25] == MAN_0, "the jump ends")
    check(state.quiet_moves == 0 and state.moves_played == 2, "each jump is a move")

    # Crowning ends the move: the new king on 29 cannot go on over 25
    state = position({22: MAN_0, 26: MAN_1, 25: MAN_1})
    state.make_move(move(22, UP_LEFT))
    check(state.board[29] == KING_0 and state.current_player == 1 and state.jumping is None, "crowning ends the move")

    # Men capture forward only, kings both ways
    check(position({18: MAN_0, 13: MAN_1}).get_possible_moves() == [move(18, UP_LEFT), move(18, UP_RIGHT)],
          "a man does not capture backward")
    check(position({18: KING_0, 13: MAN_1}).get_possible_moves() == [move(18, DOWN_LEFT)], "a king captures backward")
    state = position({18: KING_0, 31: MAN_1}, quiet_moves=5)
    state.make_move(move(18, DOWN_LEFT))
    check(state.quiet_moves == 6 and state.board[13] == KING_0, "a king step is a quiet move")

    # The end: a player without moves loses, even on reaching a draw limit with the same move
    check(position({31: MAN_1}).winner() == 1, "a player without pieces loses")
    check(position({24: MAN_0, 28: MAN_1}).winner() == 1, "a blocked player loses")
    state = position({0: KING_0, 9: MAN_0, 5: KING_0, 4: MAN_1}, quiet_moves=QUIET_LIMIT - 1)
    state.make_move(move(5, DOWN_LEFT))
    check(state.is_over() and state.winner() == 0, "blocking the last piece wins, draw limit or not")
    state = position({0: KING_0, 31: KING_1}, quiet_moves=QUIET_LIMIT - 1)
    state.make_move(move(0, UP_RIGHT))
    check(state.is_over() and state.winner() is None, "the quiet move limit draws")
    state = position({0: KING_0, 31: KING_1}, moves_played=MAX_PLIES - 1)
    state.make_move(move(0, UP_RIGHT))
    check(state.is_over() and state.winner() is None, "the move limit draws")


def describe(state) -> str:
    winner = state.winner()
    jumping = -1 if state.jumping is None else state.jumping
    fields = [*state.board, jumping, state.current_player, state.quiet_moves, state.moves_played, int(state.is_over())]
    fields.append(-1 if winner is None else winner)
    return " ".join(map(str, fields)) + " |" + "".join(f" {move}" for move in state.get_possible_moves())


def main():
    games = int(sys.argv[1]) if len(sys.argv) > 1 else 10_000
    for depth, expected in enumerate(PERFT, 1):
        check(perft(new_game(), depth) == expected, f"perft({depth}) should be {expected}")
    check_positions()
    print(f"ok: rules.py gives the published perft counts to depth {len(PERFT)} and passes the position checks")

    random.seed(0)
    expected, move_lines = [], []
    multiple_jumps = 0
    for _ in range(games):
        state = new_game()
        lines, moves = [describe(state)], []
        while not state.is_over():
            move = random.choice(state.get_possible_moves())
            player = state.current_player
            state.make_move(move)
            multiple_jumps += state.current_player == player
            moves.append(move)
            lines.append(describe(state))
        expected.append(lines)
        move_lines.append(" ".join(map(str, moves)))

    output = subprocess.run(
        [TRACE], input="\n".join(move_lines) + "\n", capture_output=True, text=True, check=True
    ).stdout
    actual = [game.splitlines() for game in output.split("\n\n")[:-1]]
    if len(actual) != games:
        sys.exit(f"checkers_trace replayed {len(actual)} games, expected {games}")

    positions = 0
    for game, (python_lines, c_lines) in enumerate(zip(expected, actual)):
        for ply, (python_line, c_line) in enumerate(zip(python_lines, c_lines)):
            if python_line != c_line:
                sys.exit(f"game {game}, ply {ply}:\n  python: {python_line}\n  c:      {c_line}\n  moves: {move_lines[game]}")
        if len(python_lines) != len(c_lines):
            sys.exit(f"game {game}: {len(python_lines)} positions in python, {len(c_lines)} in c")
        positions += len(python_lines)
    print(f"ok: {games} games, {positions} positions identical, {multiple_jumps} moves by a player moving again")


if __name__ == "__main__":
    main()
