"""Web server to play a game in a browser: human vs AI, AI vs AI or human vs human.

The AI is alpha-beta search with the latest NNUE weights in models/<game>, at a level
from 1 to 10 (LEVELS): the lower levels search a few moves ahead and sometimes play a
random move, the higher ones search for longer and longer.

    uv run python -m core.server [--game awale] [--host 127.0.0.1] [--port 8000] [--new-token]

from the repository root, then open the URL it prints, http://127.0.0.1:8000/<token>/.
Everything, page and API, is served under that secret token and every other path is
a 404, so only people given the link can play. The token is kept in .server_token at
the repository root, so links survive restarts; --new-token replaces it, which
revokes the old links. The page itself lives in games/<game>/web.

Matches are kept in memory (the oldest are dropped past MAX_MATCHES), and a finished
match frees its AI players, whose search tables can be large. AI moves are searched
when the page asks for them, one request per move, so the page can show the thinking
time and pace AI vs AI games. Requests carry the ply they expect to play, so a
repeated or stale request never plays a move twice.
"""

import argparse
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
import uvicorn
from fastapi import APIRouter, FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from core.game import ROOT, Game
from core.players import AlphaBetaPlayer, BlunderingPlayer, Player, weight_files
from core.rl import WEIGHTS_NAME
from games import DEFAULT_GAME, GAMES, get_game

TOKEN_FILE = ROOT / ".server_token"
MAX_MATCHES = 20  # Unfinished matches hold their AI players' search tables


@dataclass(frozen=True)
class Level:
    label: str
    depth: int = 0  # Turns searched ahead (a multiple jump is one), 0 for as deep as move_time allows
    move_time: float | None = None  # Seconds per move
    blunder_rate: float = 0.0  # Chance of a random move instead


LEVELS = [
    Level("beginner", depth=1, blunder_rate=0.5),
    Level("easy", depth=1, blunder_rate=0.25),
    Level("easy", depth=1),
    Level("medium", depth=3),
    Level("medium", depth=4),
    Level("hard", move_time=0.1),
    Level("hard", move_time=0.3),
    Level("expert", move_time=1.0),
    Level("expert", move_time=3.0),
    Level("maximum", move_time=10.0),
]
MIN_DIFFICULTY, MAX_DIFFICULTY = 1, len(LEVELS)


def latest_weights(game: Game) -> Path | None:
    """rl.py's latest weights, else the newest snapshot."""
    if (game.models_dir / WEIGHTS_NAME).exists():
        return game.models_dir / WEIGHTS_NAME
    if not game.models_dir.is_dir():
        return None
    snapshots = [path for path in weight_files(game.models_dir) if path.suffix == ".pt"]
    return snapshots[-1] if snapshots else None


class PlayerConfig(BaseModel):
    kind: Literal["human", "ai"]
    difficulty: int = Field(5, ge=MIN_DIFFICULTY, le=MAX_DIFFICULTY, description="ai: the level, see LEVELS")


def build_player(game: Game, config: PlayerConfig) -> Player | None:
    """None for a human: the AI is alpha-beta with the latest NNUE weights. Raises ValueError without weights."""
    if config.kind == "human":
        return None
    path = latest_weights(game)
    if path is None:
        raise ValueError(f"no NNUE weights in {game.models_dir}")
    level = LEVELS[config.difficulty - MIN_DIFFICULTY]
    player = AlphaBetaPlayer.load(game, path, level.depth, level.move_time)
    name = f"AI level {config.difficulty}"
    if level.blunder_rate:
        return BlunderingPlayer(player, level.blunder_rate, name)
    player.name = name
    return player


class Match:
    def __init__(self, game: Game, players: tuple[Player | None, Player | None]):
        self.id = uuid.uuid4().hex
        self.game = game
        self.players = players
        self.names = ["human" if player is None else player.name for player in players]
        self.humans = [player is None for player in players]
        self.state = game.new_game()
        self.last_move: int | None = None
        self.thinking_time = [0.0, 0.0]
        self.moves_made = [0, 0]
        self.lock = threading.Lock()  # One move at a time, AI searches included

    def play(self, move: int):
        self.state.make_move(move)
        self.last_move = move
        if self.state.is_over():
            self.players = (None, None)  # Frees search tables and trees; check_turn refuses moves from now on

    def to_json(self) -> dict:
        state = self.state
        over = state.is_over()
        return self.game.to_json(state) | {
            "id": self.id,
            "current_player": int(state.current_player),
            "ply": state.moves_played,
            "over": over,
            "winner": state.winner() if over else None,
            "legal_moves": [] if over else state.get_possible_moves(),
            "last_move": self.last_move,
            "players": [
                {
                    "name": self.names[side],
                    "human": self.humans[side],
                    "moves": self.moves_made[side],
                    "thinking_time": self.thinking_time[side],
                }
                for side in range(2)
            ],
        }


matches: OrderedDict[str, Match] = OrderedDict()
matches_lock = threading.Lock()


def get_match(match_id: str) -> Match:
    with matches_lock:
        match = matches.get(match_id)
        if match is None:
            raise HTTPException(404, "unknown game: it may have expired, start a new one")
        matches.move_to_end(match_id)
        return match


def check_turn(match: Match, ply: int, human: bool):
    state = match.state
    if state.moves_played != ply:
        raise HTTPException(409, f"the game is at ply {state.moves_played}, not {ply}")
    if state.is_over():
        raise HTTPException(409, "the game is over")
    if match.humans[state.current_player] != human:
        raise HTTPException(409, f"it is {'an AI' if human else 'a human'}'s turn")


class NewMatch(BaseModel):
    players: tuple[PlayerConfig, PlayerConfig]


class HumanMove(BaseModel):
    ply: int
    move: int


class AIMove(BaseModel):
    ply: int


def make_api(game: Game) -> APIRouter:
    api = APIRouter(prefix="/api")

    @api.get("/options")
    def options():
        path = latest_weights(game)
        return {
            "weights": None if path is None else path.name,
            "min_difficulty": MIN_DIFFICULTY,
            "max_difficulty": MAX_DIFFICULTY,
            "labels": [level.label for level in LEVELS],
        }

    @api.post("/games")
    def create_match(request: NewMatch):
        try:
            players = tuple(build_player(game, config) for config in request.players)
        except (ValueError, RuntimeError) as error:  # RuntimeError: the C library is not built
            raise HTTPException(400, str(error))
        match = Match(game, players)
        with matches_lock:
            matches[match.id] = match
            while len(matches) > MAX_MATCHES:
                matches.popitem(last=False)
        return match.to_json()

    @api.get("/games/{match_id}")
    def get_state(match_id: str):
        return get_match(match_id).to_json()

    @api.post("/games/{match_id}/move")
    def human_move(match_id: str, request: HumanMove):
        match = get_match(match_id)
        with match.lock:
            check_turn(match, request.ply, human=True)
            if request.move not in match.state.get_possible_moves():
                raise HTTPException(400, f"illegal move {request.move}")
            match.play(request.move)
            return match.to_json()

    @api.post("/games/{match_id}/ai-move")
    def ai_move(match_id: str, request: AIMove):
        """Searches and plays the AI's move. A plain def: FastAPI runs it in a worker thread."""
        match = get_match(match_id)
        if not match.lock.acquire(blocking=False):
            raise HTTPException(409, "already playing a move in this game")
        try:
            check_turn(match, request.ply, human=False)
            side = int(match.state.current_player)
            start = time.perf_counter()
            move = match.players[side].choose_move(match.state.copy())
            seconds = time.perf_counter() - start
            match.play(move)
            match.thinking_time[side] += seconds
            match.moves_made[side] += 1
            return match.to_json() | {"move_seconds": seconds}
        finally:
            match.lock.release()

    return api


def load_token(renew: bool) -> str:
    """The secret URL prefix: a random UUID, created on first use and kept in TOKEN_FILE."""
    if not renew and TOKEN_FILE.exists():
        token = TOKEN_FILE.read_text().strip()
        if token:
            return token
    token = uuid.uuid4().hex
    TOKEN_FILE.write_text(token + "\n")
    TOKEN_FILE.chmod(0o600)
    return token


def create_app(game: Game, token: str) -> FastAPI:
    """Serves the page and the API under /<token>/ only. The docs are off: they would list the token."""
    app = FastAPI(title=game.name, docs_url=None, redoc_url=None, openapi_url=None)
    app.include_router(make_api(game), prefix=f"/{token}")

    @app.middleware("http")
    async def revalidate(request, call_next):
        # The page changes with the code: browsers and Cloudflare check their copy (ETag) before reusing it,
        # instead of keeping it for hours
        response = await call_next(request)
        response.headers.setdefault("Cache-Control", "no-cache")
        return response

    @app.get(f"/{token}", include_in_schema=False)
    def add_slash():
        return RedirectResponse(f"/{token}/")  # The page loads its files and the API by relative URLs

    # Last, so the API routes take precedence over files
    app.mount(f"/{token}", StaticFiles(directory=game.web_dir, html=True), name="web")
    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--game", choices=GAMES, default=DEFAULT_GAME)
    parser.add_argument("--host", default="127.0.0.1", help="0.0.0.0 to accept other machines")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--new-token", action="store_true", help="Replace the secret URL token, revoking old links")
    args = parser.parse_args()
    game = get_game(args.game)
    token = load_token(args.new_token)
    host = "127.0.0.1" if args.host == "0.0.0.0" else args.host
    print(f"{game.name} at http://{host}:{args.port}/{token}/", flush=True)
    torch.set_num_threads(1)  # The network is tiny: threading overhead outweighs the gain
    uvicorn.run(create_app(game, token), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
