"""Play Awale in a window: human vs AI, AI vs AI or human vs human.

    uv run python play.py human mcts@1                      you (player 0, bottom) against 1 s MCTS
    uv run python play.py ab:0:../models/nnue.pt@0.5 human  you play second
    uv run python play.py mcts@0.5 ab:6:../models/nnue.pt   watch two AIs
    uv run python play.py human human

The first player moves first and plays the bottom row. Players are "human" or
any player spec from players.py; search players (mcts, ab) think for their
"@<seconds>" or --move-time per move. AIs think in the background, so the window
stays responsive and shows their thinking time as it runs. --min-delay keeps fast
AIs from moving faster than you can follow: a move is shown no earlier than that
many seconds after the previous one, thinking time included.

Press N or the button for a new game (same players), Space to pause or resume AI moves.
"""

import argparse
import threading
import time
import tkinter as tk

import torch

from awale import BoardState, new_game
from players import Player, parse_players
from renderer import BoardRenderer

HUMAN = "human"
REFRESH_MS = 100  # How often the thinking clock is redrawn


def parse_player(spec: str, move_time: float | None) -> Player | None:
    """None for a human, else the single AI player `spec` describes."""
    if spec == HUMAN:
        return None
    players = parse_players([spec], move_time)
    if len(players) != 1:
        raise ValueError(f"{spec!r} describes {len(players)} players, expected one: give a single .pt file")
    return players[0]


def player_label(player: Player | None) -> str:
    return HUMAN if player is None else player.name


class Game:
    """Alternates turns between humans (clicks) and AIs (searched in a worker thread)."""

    def __init__(self, players: tuple[Player | None, Player | None], min_delay: float):
        self.players = players
        self.min_delay = min_delay
        self.renderer = BoardRenderer(new_game(), on_move=self.play)
        controls = self.renderer.controls
        tk.Button(controls, text="New game (N)", command=self.new_game).pack(side="left", padx=4)
        self.pause_button = tk.Button(controls, text="Pause (Space)", command=self.toggle_pause)
        self.pause_button.pack(side="left", padx=4)
        self.renderer.root.bind("n", lambda _: self.new_game())
        self.renderer.root.bind("<space>", lambda _: self.toggle_pause())
        self.paused = False
        self.new_game()

    @property
    def state(self) -> BoardState:
        return self.renderer.state

    def new_game(self):
        # Bumping the generation makes a search still running for the previous game drop its move
        self.generation = getattr(self, "generation", 0) + 1
        self.searching = False  # An AI move being searched, or waiting for --min-delay
        self.renderer.state = new_game()
        self.renderer.last_move = None
        self.thinking_time = [0.0, 0.0]
        self.moves_made = [0, 0]
        self.last_move_shown = time.perf_counter()
        self.last_report = ""
        self.next_turn()

    def toggle_pause(self):
        self.paused = not self.paused
        self.pause_button.config(text="Resume (Space)" if self.paused else "Pause (Space)")
        if self.searching:
            return  # The running search goes on; its move is shown, then the pause applies
        self.next_turn()

    def labels(self) -> tuple[str, str]:
        labels = []
        for side, player in enumerate(self.players):
            label = f"P{side}: {player_label(player)}"
            if player is not None and self.moves_made[side]:
                average = self.thinking_time[side] / self.moves_made[side]
                label += f"   ({average:.2f} s/move, {self.thinking_time[side]:.1f} s total)"
            labels.append(label)
        return tuple(labels)

    def redraw(self, status: str | None = None):
        self.renderer.labels = self.labels()
        self.renderer.draw(status)

    def result_text(self) -> str:
        winner = self.state.winner()
        seeds = self.state.players_seeds
        if winner is None:
            return f"Draw, {seeds[0]}-{seeds[1]}. N for a new game"
        return f"P{winner} ({player_label(self.players[winner])}) wins {seeds[winner]}-{seeds[1 - winner]}. N for a new game"

    def next_turn(self):
        side = int(self.state.current_player)
        player = self.players[side]
        self.renderer.accept_clicks = player is None and not self.state.is_over()
        if self.state.is_over():
            self.redraw(f"{self.last_report}Game over: {self.result_text()}")
        elif player is None:
            self.redraw(f"{self.last_report}P{side} (human) to play: click a highlighted hole")
        elif self.paused:
            self.redraw(f"{self.last_report}Paused: Space to resume")
        else:
            self.start_search(side, player)

    def start_search(self, side: int, player: Player):
        self.searching = True
        result = {}
        state = self.state.copy()  # The search never sees the board the window draws

        def search():
            start = time.perf_counter()
            try:
                result["move"] = player.choose_move(state)
            except Exception as error:  # Shown in the window rather than lost in the thread
                result["error"] = error
            result["seconds"] = time.perf_counter() - start

        started = time.perf_counter()
        thread = threading.Thread(target=search, daemon=True)
        thread.start()
        self.poll_search(thread, result, side, player, started, self.generation)

    def poll_search(self, thread, result, side, player, started, generation):
        if generation != self.generation:
            return  # A new game started meanwhile
        if thread.is_alive():
            elapsed = time.perf_counter() - started
            self.redraw(f"{self.last_report}P{side} ({player.name}) thinking... {elapsed:.1f} s")
            self.renderer.root.after(REFRESH_MS, self.poll_search, thread, result, side, player, started, generation)
            return
        if "error" in result:
            self.searching = False
            self.redraw(f"P{side} ({player.name}) failed: {result['error']}")
            return
        self.thinking_time[side] += result["seconds"]
        self.moves_made[side] += 1
        # Fast moves wait so that each one stays on screen at least min_delay
        wait = self.min_delay - (time.perf_counter() - self.last_move_shown)
        self.renderer.root.after(
            max(0, int(wait * 1000)), self.apply_ai_move, result["move"], result["seconds"], side, player, generation
        )

    def apply_ai_move(self, move: int, seconds: float, side: int, player: Player, generation: int):
        if generation != self.generation:
            return
        self.searching = False
        self.make_move(move, f"P{side} ({player.name}) played {move} in {seconds:.2f} s. ")
        self.next_turn()  # When paused, shows the move then waits for Space

    def play(self, move: int):
        """A human's click on a playable hole."""
        side = int(self.state.current_player)
        self.make_move(move, f"P{side} (human) played {move}. ")
        self.next_turn()

    def make_move(self, move: int, report: str):
        self.state.make_move(move)
        self.renderer.last_move = move
        self.last_move_shown = time.perf_counter()
        self.last_report = report

    def run(self):
        self.renderer.root.mainloop()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("first", help="Player 0, who moves first: human or a player spec (see players.py)")
    parser.add_argument("second", help="Player 1: human or a player spec")
    parser.add_argument("--move-time", type=float, help="Seconds per move for search players without their own @<seconds>")
    parser.add_argument("--min-delay", type=float, default=0.5, help="Minimum seconds between two AI moves on screen")
    args = parser.parse_args()
    if args.move_time is not None and args.move_time <= 0:
        parser.error("--move-time must be positive")
    if args.min_delay < 0:
        parser.error("--min-delay cannot be negative")
    torch.set_num_threads(1)  # The network is tiny: threading overhead outweighs the gain

    try:
        players = (parse_player(args.first, args.move_time), parse_player(args.second, args.move_time))
    except (ValueError, RuntimeError) as error:  # RuntimeError: the C library is not built
        parser.error(str(error))
    Game(players, args.min_delay).run()


if __name__ == "__main__":
    main()
