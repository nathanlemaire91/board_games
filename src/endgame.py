"""Endgame table: exact results for every position with few seeds left on the board.

For each board holding at most `max_seeds` seeds, seen from the side to move,
the table stores the best capture difference (own captures minus opponent's)
that the side to move can force from now on, for every number of moves left
before the MAX_MOVES cut-off. It is built backwards, one move left at a time:

    gain(0, board) = 0
    gain(r, board) = max over moves of  captured - gain(r - 1, next board)
                     (no further gain if the move ends the game)

The result of a position is then the sign of (captures difference so far + gain):
the side to move can force at least that final difference, and the opponent can
hold it to at most that. The "more than 24 seeds" stop is safely ignored: once a
player has a majority, no continuation can change the winner.

Successor boards come from BoardState itself, so the rules stay in awale.py.
"""

import time
from functools import lru_cache
from pathlib import Path

import numpy as np

from awale import BOARD_SIZE, HALF_BOARD_SIZE, MAX_MOVES, TOTAL_SEEDS, BoardState

DEFAULT_MAX_SEEDS = 11
TABLES_DIR = Path(__file__).resolve().parent.parent / "models"
CODE_BASE = TOTAL_SEEDS // 2 + 1  # Hole counts in the table never exceed max_seeds <= 24


def encode(board: list[int]) -> int:
    code = 0
    for seeds in reversed(board):
        code = code * CODE_BASE + seeds
    return code


def all_boards(max_seeds: int) -> list[list[int]]:
    """Every distribution of at most `max_seeds` seeds over the holes."""
    boards = []

    def fill(prefix: list[int], remaining: int):
        if len(prefix) == BOARD_SIZE:
            boards.append(prefix)
            return
        for seeds in range(remaining + 1):
            fill(prefix + [seeds], remaining - seeds)

    fill([], max_seeds)
    return boards


class EndgameTable:
    def __init__(self, max_seeds: int, codes: np.ndarray, gains: np.ndarray):
        self.max_seeds = max_seeds
        self.index = dict(zip(codes.tolist(), range(len(codes))))
        self.gains = gains  # (MAX_MOVES + 1, boards), int8

    def covers(self, state: BoardState) -> bool:
        return state.seeds_on_board() <= self.max_seeds

    def result(self, state: BoardState) -> int:
        """+1 / 0 / -1 for the side to move, with perfect play, in a covered position that is not over."""
        board = state.board_from(state.current_player)
        gain = int(self.gains[state.moves_left, self.index[encode(board)]])
        captured = state.players_seeds[state.current_player] - state.players_seeds[state.opponent]
        return (captured + gain > 0) - (captured + gain < 0)

    @classmethod
    def build(cls, max_seeds: int) -> "EndgameTable":
        if not 0 <= max_seeds <= TOTAL_SEEDS // 2:
            raise ValueError(f"max_seeds must be between 0 and {TOTAL_SEEDS // 2}")
        boards = all_boards(max_seeds)
        codes = np.array([encode(board) for board in boards], dtype=np.int64)
        index = dict(zip(codes.tolist(), range(len(codes))))

        # One move per own hole; illegal moves are masked out
        successor = np.zeros((len(boards), HALF_BOARD_SIZE), dtype=np.int64)
        captured = np.zeros((len(boards), HALF_BOARD_SIZE), dtype=np.int16)
        ends_game = np.zeros((len(boards), HALF_BOARD_SIZE), dtype=bool)
        legal = np.zeros((len(boards), HALF_BOARD_SIZE), dtype=bool)
        for i, board in enumerate(boards):
            state = BoardState(board, current_player=0)
            for move in state.get_possible_moves():
                next_state = state.copy()
                next_state.make_move(move)
                legal[i, move] = True
                captured[i, move] = next_state.players_seeds[0]
                ends_game[i, move] = next_state.is_over()
                successor[i, move] = index[encode(next_state.board_from(1))]

        gains = np.zeros((MAX_MOVES + 1, len(boards)), dtype=np.int8)
        for moves_left in range(1, MAX_MOVES + 1):
            future = np.where(ends_game, 0, gains[moves_left - 1][successor].astype(np.int16))
            candidates = np.where(legal, captured - future, np.iinfo(np.int16).min)
            # Boards where the side to move has no move are game over and never looked up
            gains[moves_left] = np.where(legal.any(axis=1), candidates.max(axis=1), 0)
        return cls(max_seeds, codes, gains)

    def save(self, path: Path):
        codes = np.fromiter(self.index, dtype=np.int64, count=len(self.index))
        np.savez_compressed(path, max_seeds=self.max_seeds, codes=codes, gains=self.gains)

    @classmethod
    def load(cls, path: Path) -> "EndgameTable":
        data = np.load(path)
        return cls(int(data["max_seeds"]), data["codes"], data["gains"])


@lru_cache
def endgame_table(max_seeds: int = DEFAULT_MAX_SEEDS) -> EndgameTable:
    """The table for `max_seeds`, built on first use and cached in models/."""
    path = TABLES_DIR / f"endgame_{max_seeds}.npz"
    if path.exists():
        return EndgameTable.load(path)
    print(f"building the endgame table for up to {max_seeds} seeds...", flush=True)
    start = time.monotonic()
    table = EndgameTable.build(max_seeds)
    path.parent.mkdir(parents=True, exist_ok=True)
    table.save(path)
    print(f"built {len(table.index)} boards in {time.monotonic() - start:.1f}s, saved to {path}", flush=True)
    return table


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build the endgame table ahead of time.")
    parser.add_argument("max_seeds", type=int, nargs="?", default=DEFAULT_MAX_SEEDS)
    endgame_table(parser.parse_args().max_seeds)
