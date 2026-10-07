"""Alpha-beta search with an NNUE evaluation at the leaves, Stockfish style.

Negamax alpha-beta: every score is from the side to move's point of view, in
[-1, 1] like the network's output, with finished games scored exactly (+1, 0, -1).

- Iterative deepening: depth 1, 2, ... up to the target depth, each iteration
  ordering moves with the best moves found by the previous ones.
- Transposition table: positions already searched (often reached through another
  move order) reuse their score and best move.
- Endgame table: positions with few seeds left are scored exactly, without search.
- Time limit (optional): deepening stops when the time per move runs out, and the
  move of the last completed iteration is played.
"""

import math
import time
from dataclasses import dataclass

from awale import MAX_MOVES, BoardState
from endgame import EndgameTable
from nnue import NNUE, NumpyEvaluator

EXACT, LOWER_BOUND, UPPER_BOUND = range(3)
MAX_TRANSPOSITIONS = 2_000_000
MAX_EVALUATION = 0.999  # Keeps network scores strictly below proven results, which are exactly +-1
CLOCK_CHECK_NODES = 1024  # Nodes searched between two looks at the clock


class OutOfTime(Exception):
    pass


@dataclass
class Transposition:
    depth: int
    score: float
    bound: int
    best_move: int | None


class AlphaBeta:
    """Searches to `depth`, or as deep as `move_time` seconds allow (depth 0: no depth limit)."""

    def __init__(self, net: NNUE, depth: int, endgame: EndgameTable | None = None, move_time: float | None = None):
        if depth < 1 and move_time is None:
            raise ValueError("an unlimited depth needs a time limit")
        self.evaluate = NumpyEvaluator(net)
        # No search goes deeper than the moves left in a game
        self.depth = depth if depth >= 1 else MAX_MOVES
        self.move_time = move_time
        self.endgame = endgame
        self.transpositions: dict[tuple, Transposition] = {}
        self.nodes = 0
        self.deadline = math.inf

    def best_move(self, state: BoardState) -> int:
        if len(self.transpositions) > MAX_TRANSPOSITIONS:
            self.transpositions.clear()
        self.nodes = 0
        # Depth 1 always completes, so there is a move to play however short the time
        self.deadline = math.inf
        score, move = self.search_root(state, 1)
        if self.move_time is not None:
            self.deadline = time.perf_counter() + self.move_time
        for depth in range(2, min(self.depth, state.moves_left) + 1):
            if abs(score) == 1:
                break  # Proven win or loss: searching deeper cannot change it
            try:
                score, move = self.search_root(state, depth)
            except OutOfTime:
                break  # Nodes finished before the deadline stay in the table; the partial root is dropped
        return move

    def search_root(self, state: BoardState, depth: int) -> tuple[float, int]:
        alpha, best_move = -math.inf, None
        for move in self.ordered_moves(state):
            child = state.copy()
            child.make_move(move)
            score = -self.negamax(child, depth - 1, -math.inf, -alpha)
            if score > alpha:
                alpha, best_move = score, move
        self.transpositions[state.key()] = Transposition(depth, alpha, EXACT, best_move)
        return alpha, best_move

    def negamax(self, state: BoardState, depth: int, alpha: float, beta: float) -> float:
        self.nodes += 1
        if self.nodes % CLOCK_CHECK_NODES == 0 and time.perf_counter() > self.deadline:
            raise OutOfTime
        if state.is_over():
            return state.result(state.current_player)
        if self.endgame and self.endgame.covers(state):
            return self.endgame.result(state)
        if depth == 0:
            return max(-MAX_EVALUATION, min(MAX_EVALUATION, self.evaluate(state)))

        key = state.key()
        entry = self.transpositions.get(key)
        if entry and entry.depth >= depth:
            if entry.bound == EXACT:
                return entry.score
            if entry.bound == LOWER_BOUND:
                alpha = max(alpha, entry.score)
            else:
                beta = min(beta, entry.score)
            if alpha >= beta:
                return entry.score

        original_alpha = alpha
        best_score, best_move = -math.inf, None
        for move in self.ordered_moves(state, entry):
            child = state.copy()
            child.make_move(move)
            score = -self.negamax(child, depth - 1, -beta, -alpha)
            if score > best_score:
                best_score, best_move = score, move
            alpha = max(alpha, score)
            if alpha >= beta:
                break

        if best_score <= original_alpha:
            bound = UPPER_BOUND
        elif best_score >= beta:
            bound = LOWER_BOUND
        else:
            bound = EXACT
        self.transpositions[key] = Transposition(depth, best_score, bound, best_move)
        return best_score

    def ordered_moves(self, state: BoardState, entry: Transposition | None = None) -> list[int]:
        """Legal moves, the best one from a previous search first."""
        moves = state.get_possible_moves()
        if entry is None:
            entry = self.transpositions.get(state.key())
        if entry and entry.best_move in moves:
            moves.remove(entry.best_move)
            moves.insert(0, entry.best_move)
        return moves
