"""Elo ratings for a game's players, from round-robin games between them.

Players are given as specs (see players.py), for example, from the repository root:

    uv run python -m core.elo models/awale random mcts:200 ab:4:models/awale/nnue_iter000204.pt

Search players can be given a time per move, either each its own with an
"@<seconds>" suffix (mcts@0.5, ab:0:models/awale/nnue.pt@1) or all at once with
--move-time. The table shows the time each player actually spent per move.

Every pair of players plays --games games. Greedy NNUE players are deterministic,
so each pair of games starts from the same random opening of --opening-plies
moves, played once with each colour.

Results are kept in --results, so a later run with new players only plays the
new pairings. Ratings are fitted on all results at once (Bradley-Terry, the model
behind Elo), with a couple of virtual draws per pairing so a 100% score stays
finite. The random player is anchored at 0 Elo, or the first player without it.
"""

import argparse
import json
import math
import random
import time
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import torch

from core.game import Game, GameState
from core.players import Player, RandomPlayer, parse_players
from games import DEFAULT_GAME, GAMES, get_game

VIRTUAL_DRAWS = 2


def random_opening(game: Game, plies: int) -> GameState:
    state = game.new_game()
    for _ in range(plies):
        if state.is_over():
            break
        state.make_move(random.choice(state.get_possible_moves()))
    return state


class ThinkingTime:
    """Total seconds spent choosing moves, and moves played, per player name."""

    def __init__(self):
        self.seconds: dict[str, float] = defaultdict(float)
        self.moves: dict[str, int] = defaultdict(int)

    def choose_move(self, player: Player, state: GameState) -> int:
        start = time.perf_counter()
        move = player.choose_move(state)
        self.seconds[player.name] += time.perf_counter() - start
        self.moves[player.name] += 1
        return move

    def per_move(self, name: str) -> float | None:
        return self.seconds[name] / self.moves[name] if self.moves[name] else None


def play_game(first: Player, second: Player, opening: GameState, clock: ThinkingTime) -> float:
    """Score of `first`, who plays player 0's side: 1, 0.5 or 0."""
    state = opening.copy()
    while not state.is_over():
        player = first if state.current_player == 0 else second
        state.make_move(clock.choose_move(player, state))
    return (state.result(0) + 1) / 2


def play_match(game: Game, a: Player, b: Player, games: int, opening_plies: int, clock: ThinkingTime) -> float:
    """Score of `a` over `games` games (an even number), swapping colours on each opening."""
    score = 0.0
    for _ in range(games // 2):
        opening = random_opening(game, opening_plies)
        score += play_game(a, b, opening, clock)
        score += 1 - play_game(b, a, opening, clock)
    return score


class Results:
    """Head-to-head results, stored as {"a\\tb": [score of a, games]} with a < b."""

    def __init__(self, path: Path):
        self.path = path
        self.pairs: dict[str, list] = json.loads(path.read_text()) if path.exists() else {}

    @staticmethod
    def key(a: str, b: str) -> str:
        return "\t".join(sorted((a, b)))

    def games(self, a: str, b: str) -> int:
        return self.pairs.get(self.key(a, b), [0, 0])[1]

    def score(self, a: str, b: str) -> float:
        """Points scored by `a` against `b`."""
        first_score, total = self.pairs.get(self.key(a, b), [0, 0])
        return first_score if a < b else total - first_score

    def add(self, a: str, b: str, score_a: float, games: int):
        record = self.pairs.setdefault(self.key(a, b), [0.0, 0])
        record[0] += score_a if a < b else games - score_a
        record[1] += games

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.pairs, indent=1))


def fit_ratings(names: list[str], results: Results, anchor: str) -> dict[str, float]:
    """Bradley-Terry maximum likelihood via minorization-maximization, on the Elo scale."""
    # Draws count as half a win; virtual draws keep ratings finite
    score = {a: sum(results.score(a, b) + VIRTUAL_DRAWS / 2 for b in names if b != a) for a in names}
    games = {(a, b): results.games(a, b) + VIRTUAL_DRAWS for a in names for b in names if a != b}

    strength = {name: 1.0 for name in names}
    for _ in range(10_000):
        new_strength = {
            a: score[a] / sum(games[a, b] / (strength[a] + strength[b]) for b in names if b != a) for a in names
        }
        change = max(abs(new_strength[n] - strength[n]) / strength[n] for n in names)
        strength = new_strength
        if change < 1e-9:
            break
    return {name: 400 * math.log10(strength[name] / strength[anchor]) for name in names}


def print_table(names: list[str], results: Results, ratings: dict[str, float], clock: ThinkingTime):
    """Thinking times only cover this run's games: "-" for players whose results all come from earlier runs."""
    print(f"\n{'rank':>4}  {'elo':>7}  {'score':>6}  {'games':>5}  {'ms/move':>8}  player")
    for rank, a in enumerate(sorted(names, key=ratings.get, reverse=True), 1):
        score = sum(results.score(a, b) for b in names if b != a)
        games = sum(results.games(a, b) for b in names if b != a)
        per_move = clock.per_move(a)
        thinking = "-" if per_move is None else f"{per_move * 1000:.1f}"
        print(f"{rank:>4}  {ratings[a]:>7.0f}  {score / max(games, 1):>6.1%}  {games:>5}  {thinking:>8}  {a}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--game", choices=GAMES, default=DEFAULT_GAME)
    parser.add_argument(
        "players", nargs="*",
        help="random, mcts[:<iterations>[:fresh]][@<seconds>], ab:<depth>:<weights>[@<seconds>], "
        "NNUE .pt files or directories of them (default: models/<game> and random)",
    )
    parser.add_argument(
        "--move-time", type=float,
        help="Seconds per move for search players (mcts, ab) without their own @<seconds>",
    )
    parser.add_argument("--games", type=int, default=20, help="Games per pairing (rounded up to an even number)")
    parser.add_argument("--opening-plies", type=int, default=4, help="Random moves played before the players take over")
    parser.add_argument("--results", type=Path, help="Default: models/<game>/elo_results.json")
    args = parser.parse_args()
    game = get_game(args.game)
    if not args.players:
        args.players = ([str(game.models_dir)] if game.models_dir.is_dir() else []) + ["random"]
    args.results = args.results or game.models_dir / "elo_results.json"
    if args.move_time is not None and args.move_time <= 0:
        parser.error("--move-time must be positive")
    torch.set_num_threads(1)  # The network is tiny: threading overhead outweighs the gain

    try:
        players = {player.name: player for player in parse_players(game, args.players, args.move_time)}
    except (ValueError, RuntimeError) as error:  # RuntimeError: the C library is not built
        parser.error(str(error))
    if len(players) < 2:
        parser.error("need at least two players")
    names = list(players)

    results = Results(args.results)
    games_per_pair = args.games + args.games % 2
    clock = ThinkingTime()
    pairings = [(a, b) for a, b in combinations(names, 2) if results.games(a, b) < games_per_pair]
    print(f"{len(names)} players, {len(pairings)} pairings to play")

    for index, (a, b) in enumerate(pairings, 1):
        games = games_per_pair - results.games(a, b)
        games += games % 2
        results.add(a, b, play_match(game, players[a], players[b], games, args.opening_plies, clock), games)
        results.save()
        print(f"[{index}/{len(pairings)}] {a} vs {b}: {results.score(a, b):g}/{results.games(a, b)}", flush=True)

    anchor = RandomPlayer.name if RandomPlayer.name in players else names[0]
    print_table(names, results, fit_ratings(names, results, anchor), clock)


if __name__ == "__main__":
    main()
