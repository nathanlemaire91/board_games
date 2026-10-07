"""Checks that the C rules match rules.py, position by position, over random games.

    uv run python -m games.awale.tests.check_rules [games]      (after `make -C games/awale/c`)

Plays random games with the Python rules, replays their moves through awale_trace
and compares every position, legal move list, end of game and winner.
"""

import random
import subprocess
import sys
from pathlib import Path

from games.awale.rules import new_game

TRACE = Path(__file__).resolve().parent.parent / "c" / "awale_trace"


def describe(state) -> str:
    winner = state.winner()
    fields = [*state.board, *state.players_seeds, state.current_player, state.moves_played, int(state.is_over())]
    fields.append(-1 if winner is None else winner)
    return " ".join(map(str, fields)) + " |" + "".join(f" {move}" for move in state.get_possible_moves())


def main():
    games = int(sys.argv[1]) if len(sys.argv) > 1 else 10_000
    random.seed(0)
    expected, move_lines = [], []
    for _ in range(games):
        state = new_game()
        lines, moves = [describe(state)], []
        while not state.is_over():
            move = random.choice(state.get_possible_moves())
            state.make_move(move)
            moves.append(move)
            lines.append(describe(state))
        expected.append(lines)
        move_lines.append(" ".join(map(str, moves)))

    output = subprocess.run(
        [TRACE], input="\n".join(move_lines) + "\n", capture_output=True, text=True, check=True
    ).stdout
    actual = [game.splitlines() for game in output.split("\n\n")[:-1]]
    if len(actual) != games:
        sys.exit(f"awale_trace replayed {len(actual)} games, expected {games}")

    positions = 0
    for game, (python_lines, c_lines) in enumerate(zip(expected, actual)):
        for ply, (python_line, c_line) in enumerate(zip(python_lines, c_lines)):
            if python_line != c_line:
                sys.exit(f"game {game}, ply {ply}:\n  python: {python_line}\n  c:      {c_line}\n  moves: {move_lines[game]}")
        if len(python_lines) != len(c_lines):
            sys.exit(f"game {game}: {len(python_lines)} positions in python, {len(c_lines)} in c")
        positions += len(python_lines)
    print(f"ok: {games} games, {positions} positions identical")


if __name__ == "__main__":
    main()
