"""Compares the C alpha-beta (core/c_search.py) with a plain Python negamax, on Awale.

    uv run python -m games.awale.tests.check_search [--weights models/awale/nnue.pt] [--positions 100] [--depth 4]

(after `make -C games/awale/c`). Checks, on positions from random games:
- endgame table: the C board ranks are the Python table's indices,
- evaluation: the C network (fresh accumulators) against the PyTorch one,
- search: fixed-depth searches from fresh C searchers, whose root scores must be the
  exact depth-d negamax values (same leaves: game results, endgame table, clamped
  network scores), and whose moves should match, barring ties.
"""

import argparse
import ctypes
import random
import sys
from pathlib import Path

from core.c_library import library
from core.c_search import CAlphaBeta
from core.nnue import NNUE
from core.players import load_network
from core.rl import WEIGHTS_NAME
from games.awale import GAME
from games.awale.endgame import CODE_BASE, EndgameTable, endgame_table
from games.awale.rules import BOARD_SIZE, BoardState

MAX_EVALUATION = 0.999  # As in core/c/search.c


def random_positions(count: int, rng: random.Random) -> list[BoardState]:
    """Positions spread over random games, none of them over."""
    positions = []
    while len(positions) < count:
        state = GAME.new_game()
        stop = rng.randrange(0, 80)
        while not state.is_over() and state.moves_played < stop:
            state.make_move(rng.choice(state.get_possible_moves()))
        if not state.is_over():
            positions.append(state)
    return positions


def negamax(net: NNUE, table: EndgameTable, state: BoardState, depth: int) -> float:
    """The exact depth-`depth` value for the side to move, without pruning."""
    if state.is_over():
        return state.result(state.current_player)
    if table.covers(state):
        return table.result(state)
    if depth == 0:
        return max(-MAX_EVALUATION, min(MAX_EVALUATION, net.evaluate(GAME, state)))
    best = -float("inf")
    for move in state.get_possible_moves():
        child = state.copy()
        child.make_move(move)
        best = max(best, -negamax(net, table, child, depth - 1))
    return best


def root_scores(net: NNUE, table: EndgameTable, state: BoardState, depth: int) -> dict[int, float]:
    scores = {}
    for move in state.get_possible_moves():
        child = state.copy()
        child.make_move(move)
        scores[move] = -negamax(net, table, child, depth - 1)
    return scores


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", type=Path, default=GAME.models_dir / WEIGHTS_NAME)
    parser.add_argument("--positions", type=int, default=100)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    net = load_network(GAME, args.weights)
    table = endgame_table()
    failures = 0

    Board = ctypes.c_uint8 * BOARD_SIZE
    rank = library(GAME).awale_endgame_rank
    rank.argtypes = [Board, ctypes.c_int]
    rank.restype = ctypes.c_int64
    wanted = set(rng.sample(range(len(table.index)), 5000))
    boards = {i: [(code // CODE_BASE**hole) % CODE_BASE for hole in range(BOARD_SIZE)]
              for code, i in table.index.items() if i in wanted}
    bad_ranks = sum(rank(Board(*board), table.max_seeds) != i for i, board in boards.items())
    print(f"endgame ranks: {len(boards) - bad_ranks}/{len(boards)} match")
    failures += bad_ranks > 0

    positions = random_positions(args.positions, rng)
    evaluator = CAlphaBeta(GAME, net, 1)
    differences = [abs(evaluator.evaluate(state) - net.evaluate(GAME, state)) for state in positions]
    print(f"evaluation: max difference {max(differences):.2e} over {len(positions)} positions")
    failures += max(differences) > 1e-4

    score_mismatches = move_mismatches = 0
    worst = 0.0
    for state in positions:
        scores = root_scores(net, table, state, args.depth)
        best = max(scores.values())
        search = CAlphaBeta(GAME, net, args.depth)
        move = search.best_move(state)
        _, _, score = search.stats()
        worst = max(worst, abs(score - best))
        score_mismatches += abs(score - best) > 1e-4
        move_mismatches += abs(scores[move] - best) > 1e-4  # Any move scoring the best is right
    print(f"depth-{args.depth} search: {len(positions) - score_mismatches}/{len(positions)} scores match "
          f"(max difference {worst:.2e}), {len(positions) - move_mismatches}/{len(positions)} moves are best")
    failures += score_mismatches > 0 or move_mismatches > 0
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
