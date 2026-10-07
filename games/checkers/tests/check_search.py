"""Compares the C alpha-beta (core/c_search.py) with a plain Python negamax, on checkers.

    uv run python -m games.checkers.tests.check_search [--weights models/checkers/nnue.pt] [--positions 100] [--depth 3]

(after `make -C games/checkers/c`). Without trained weights, a random network with
large enough weights to tell positions apart. Checks, on positions from random games,
some of them in the middle of a multiple jump:
- the endgame table, used by both searches (its own checks: check_endgame.py),
- evaluation: the C network (fresh accumulators) against the PyTorch one,
- search: fixed-depth searches from fresh C searchers, whose root scores must be the
  exact depth-d negamax values, depth counting turns (a multiple jump is searched to its
  end at the same depth, without changing sides), and whose moves should match,
  barring ties.
"""

import argparse
import random
import sys
from pathlib import Path

import torch

from core.c_search import CAlphaBeta
from core.nnue import NNUE
from core.players import load_network
from core.rl import WEIGHTS_NAME
from games.checkers import GAME
from games.checkers.endgame import endgame_table
from games.checkers.rules import BoardState

MAX_EVALUATION = 0.999  # As in core/c/search.c


def random_network(seed: int) -> NNUE:
    torch.manual_seed(seed)
    net = NNUE(GAME.num_features)
    torch.nn.init.normal_(net.feature_transformer.weight, std=0.2)
    return net.eval()


def random_positions(count: int, rng: random.Random) -> list[BoardState]:
    """Positions spread over random games, none of them over, one in three in the middle of a multiple jump."""
    positions = []
    while len(positions) < count:
        state = GAME.new_game()
        stop = rng.randrange(0, 250)  # Late positions reach the endgame table
        jumping = len(positions) % 3 == 0
        while not state.is_over() and (state.moves_played < stop or jumping and state.jumping is None):
            state.make_move(rng.choice(state.get_possible_moves()))
        if not state.is_over():
            positions.append(state)
    return positions


def negamax(net: NNUE, state: BoardState, depth: int) -> float:
    """The exact depth-`depth` value for the side to move, without pruning."""
    if state.is_over():
        return state.result(state.current_player)
    exact = endgame_table().result(state)
    if exact is not None:
        return exact
    if depth == 0:
        return max(-MAX_EVALUATION, min(MAX_EVALUATION, net.evaluate(GAME, state)))
    return max(root_scores(net, state, depth).values())


def root_scores(net: NNUE, state: BoardState, depth: int) -> dict[int, float]:
    """Each move's score for the side to move: a move after which they move again keeps the turn."""
    scores = {}
    for move in state.get_possible_moves():
        child = state.copy()
        child.make_move(move)
        if child.current_player == state.current_player:
            scores[move] = negamax(net, child, depth)
        else:
            scores[move] = -negamax(net, child, depth - 1)
    return scores


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", type=Path, default=GAME.models_dir / WEIGHTS_NAME)
    parser.add_argument("--positions", type=int, default=100)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    if args.weights.exists():
        net = load_network(GAME, args.weights)
    else:
        print(f"{args.weights} not found: random network")
        net = random_network(args.seed)
    failures = 0

    positions = random_positions(args.positions, rng)
    jumping = sum(state.jumping is not None for state in positions)
    evaluator = CAlphaBeta(GAME, net, 1)
    differences = [abs(evaluator.evaluate(state) - net.evaluate(GAME, state)) for state in positions]
    print(f"evaluation: max difference {max(differences):.2e} over {len(positions)} positions "
          f"({jumping} in the middle of a multiple jump)")
    failures += max(differences) > 1e-4

    score_mismatches = move_mismatches = 0
    worst = 0.0
    for state in positions:
        scores = root_scores(net, state, args.depth)
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
