"""The interface between a game and the reusable code of core/.

A game lives in games/<name>/ and describes itself with a Game (its __init__.py's
GAME), registered in games/__init__.py. Its positions are GameStates; the engines
(players, training, ratings, web server) only use what is declared here.

Its C side (games/<name>/c) implements core/c/game_api.h, and its web page lives in
games/<name>/web.
"""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Self

ROOT = Path(__file__).resolve().parent.parent


class GameState(Protocol):
    """A position. Players are 0 and 1; moves are ints from 0 to 127.

    Players usually take turns, but a player may move several times in a row (the jumps of a
    multiple jump in checkers): the engines read current_player after each move rather than
    assuming it alternates."""

    current_player: int
    moves_played: int

    def copy(self) -> Self: ...

    def get_possible_moves(self) -> list[int]:
        """The legal moves, in increasing order."""

    def make_move(self, move: int) -> None: ...

    def is_over(self) -> bool: ...

    def winner(self) -> int | None:
        """The winner of a finished game, None for a draw (for an unfinished game: the player ahead)."""

    def result(self, player: int) -> int:
        """+1 if `player` wins, -1 if they lose, 0 for a draw."""


@dataclass(frozen=True)
class Game:
    name: str
    new_game: Callable[[], GameState]
    max_legal_moves: int  # The most legal moves a position can have

    # NNUE inputs: the active features of a position seen from a player's side, one per slot,
    # out of num_features. Must match game_features in the game's C code.
    num_features: int
    num_active_features: int
    features: Callable[[GameState, int], list[int]]

    # A position as the ints the C code's game_decode reads
    encode: Callable[[GameState], list[int]]

    # Web page: the game-specific fields of a position's JSON (board, scores...)
    to_json: Callable[[GameState], dict[str, Any]]

    # Called with each new C alpha-beta searcher (core/c_search.py), to attach exact
    # results such as an endgame table; None if the game has none
    setup_search: Callable[[Any], None] | None = None

    @property
    def directory(self) -> Path:
        return ROOT / "games" / self.name

    @property
    def library_path(self) -> Path:
        """Built by `make -C games/<name>/c`."""
        return self.directory / "c" / f"lib{self.name}.so"

    @property
    def web_dir(self) -> Path:
        return self.directory / "web"

    @property
    def models_dir(self) -> Path:
        """Trained weights, snapshots, rating results and other generated data."""
        return ROOT / "models" / self.name
