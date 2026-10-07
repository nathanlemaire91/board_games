"""The games, each in its own directory (see core/game.py for what a game provides)."""

from core.game import Game


def get_game(name: str) -> Game:
    """The Game named `name`, imported on demand so a game's dependencies load only when used."""
    if name not in GAMES:
        raise ValueError(f"unknown game {name!r}: expected one of {', '.join(GAMES)}")
    module = __import__(f"games.{name}", fromlist=["GAME"])
    return module.GAME


GAMES = ["awale", "checkers"]
DEFAULT_GAME = "awale"
