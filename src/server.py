"""Web server to play Awale in a browser: human vs AI, AI vs AI or human vs human.

    uv run python server.py [--host 127.0.0.1] [--port 8000]

then open http://127.0.0.1:8000. The page itself lives in web/ at the repository root.

Games are kept in memory (the oldest are dropped past MAX_GAMES), and a finished
game frees its AI players, whose search tables can be large. AI moves are
searched when the page asks for them, one request per move, so the page can show
the thinking time and pace AI vs AI games. Requests carry the ply they expect to
play, so a repeated or stale request never plays a move twice.
"""

import argparse
import threading
import time
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import Literal

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from awale import BoardState, new_game
from players import AlphaBetaPlayer, MCTSPlayer, NNUEPlayer, Player, RandomPlayer, weight_files
from rl import MODELS_DIR, WEIGHTS_NAME

WEB_DIR = Path(__file__).resolve().parent.parent / "web"
MAX_GAMES = 20  # Unfinished games hold their AI players' search tables
MAX_MOVE_TIME = 60.0  # Seconds: keeps one request from tying up a worker for long


def available_weights() -> list[str]:
    """Weight files a browser may pick, by name: the latest weights first, then the snapshots."""
    names = [WEIGHTS_NAME] if (MODELS_DIR / WEIGHTS_NAME).exists() else []
    return names + [path.name for path in weight_files(MODELS_DIR) if path.suffix == ".pt"]


class PlayerConfig(BaseModel):
    kind: Literal["human", "random", "mcts", "ab", "nnue"]
    weights: str | None = Field(None, description="ab, nnue: a file name from /api/options")
    depth: int = Field(0, ge=0, le=100, description="ab: depth cap, 0 for none (needs move_time)")
    iterations: int | None = Field(None, ge=1, description="mcts: iteration cap, none needs move_time")
    reuse: bool = Field(True, description="mcts: keep the search tree between moves")
    move_time: float | None = Field(None, gt=0, le=MAX_MOVE_TIME, description="mcts, ab: seconds per move")


def build_player(config: PlayerConfig) -> Player | None:
    """None for a human. Raises ValueError for settings that cannot work."""
    if config.kind == "human":
        return None
    if config.kind == "random":
        return RandomPlayer()
    if config.kind == "mcts":
        if config.iterations is None and config.move_time is None:
            raise ValueError("MCTS needs an iteration count or a time per move")
        return MCTSPlayer(config.iterations, reuse=config.reuse, move_time=config.move_time)

    # Only files listed by the server: a browser never names an arbitrary path
    if config.weights not in available_weights():
        raise ValueError(f"unknown weights {config.weights!r}")
    path = MODELS_DIR / config.weights
    if config.kind == "nnue":
        player = NNUEPlayer.load(path)
        player.name = config.weights
        return player
    if config.depth == 0 and config.move_time is None:
        raise ValueError("alpha-beta without a depth limit needs a time per move")
    player = AlphaBetaPlayer.load(path, config.depth, config.move_time)
    player.name = f"ab:{config.depth}:{config.weights}" + ("" if config.move_time is None else f"@{config.move_time:g}")
    return player


class Game:
    def __init__(self, players: tuple[Player | None, Player | None]):
        self.id = uuid.uuid4().hex
        self.players = players
        self.names = ["human" if player is None else player.name for player in players]
        self.humans = [player is None for player in players]
        self.state = new_game()
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
        return {
            "id": self.id,
            "board": state.board,
            "seeds": state.players_seeds,
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


games: OrderedDict[str, Game] = OrderedDict()
games_lock = threading.Lock()


def get_game(game_id: str) -> Game:
    with games_lock:
        game = games.get(game_id)
        if game is None:
            raise HTTPException(404, "unknown game: it may have expired, start a new one")
        games.move_to_end(game_id)
        return game


def check_turn(game: Game, ply: int, human: bool):
    state: BoardState = game.state
    if state.moves_played != ply:
        raise HTTPException(409, f"the game is at ply {state.moves_played}, not {ply}")
    if state.is_over():
        raise HTTPException(409, "the game is over")
    if game.humans[state.current_player] != human:
        raise HTTPException(409, f"it is {'an AI' if human else 'a human'}'s turn")


app = FastAPI(title="Awale")


@app.get("/api/options")
def options():
    return {"weights": available_weights(), "max_move_time": MAX_MOVE_TIME}


class NewGame(BaseModel):
    players: tuple[PlayerConfig, PlayerConfig]


@app.post("/api/games")
def create_game(request: NewGame):
    try:
        players = tuple(build_player(config) for config in request.players)
    except (ValueError, RuntimeError) as error:  # RuntimeError: the C library is not built
        raise HTTPException(400, str(error))
    game = Game(players)
    with games_lock:
        games[game.id] = game
        while len(games) > MAX_GAMES:
            games.popitem(last=False)
    return game.to_json()


@app.get("/api/games/{game_id}")
def get_state(game_id: str):
    return get_game(game_id).to_json()


class HumanMove(BaseModel):
    ply: int
    move: int


@app.post("/api/games/{game_id}/move")
def human_move(game_id: str, request: HumanMove):
    game = get_game(game_id)
    with game.lock:
        check_turn(game, request.ply, human=True)
        if request.move not in game.state.get_possible_moves():
            raise HTTPException(400, f"illegal move {request.move}")
        game.play(request.move)
        return game.to_json()


class AIMove(BaseModel):
    ply: int


@app.post("/api/games/{game_id}/ai-move")
def ai_move(game_id: str, request: AIMove):
    """Searches and plays the AI's move. A plain def: FastAPI runs it in a worker thread."""
    game = get_game(game_id)
    if not game.lock.acquire(blocking=False):
        raise HTTPException(409, "already playing a move in this game")
    try:
        check_turn(game, request.ply, human=False)
        side = int(game.state.current_player)
        start = time.perf_counter()
        move = game.players[side].choose_move(game.state.copy())
        seconds = time.perf_counter() - start
        game.play(move)
        game.thinking_time[side] += seconds
        game.moves_made[side] += 1
        return game.to_json() | {"move_seconds": seconds}
    finally:
        game.lock.release()


# Last, so the API routes take precedence over files
app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="127.0.0.1", help="0.0.0.0 to accept other machines")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    torch.set_num_threads(1)  # The network is tiny: threading overhead outweighs the gain
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
