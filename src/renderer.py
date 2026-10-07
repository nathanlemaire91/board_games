import tkinter as tk
from typing import Callable

from awale import BOARD_SIZE, HALF_BOARD_SIZE, BoardState, new_game

# Layout in board units: the board is drawn at WIDTH x HEIGHT, then scaled to fit the window
HOLE_RADIUS = 40
HOLE_SPACING = 100
MARGIN = 60
WIDTH = MARGIN * 2 + HOLE_SPACING * HALF_BOARD_SIZE
HEIGHT = 340
MIN_WINDOW_SIZE = (360, 260)

BOARD_COLOR = "#8b5a2b"
HOLE_COLOR = "#5c3a1a"
PLAYABLE_COLOR = "#e0a030"
LAST_MOVE_COLOR = "#f5e6c8"
TEXT_COLOR = "#f5e6c8"

# Numbers are drawn as seven-segment digits with canvas lines, so they scale with the board.
# Tk text cannot be relied on for that: some Tk builds (such as the one bundled with uv's
# Python) have no scalable fonts and draw every size with the same bitmap font.
#   segment: ((x0, y0), (x1, y1)) in a digit box 1 wide and 2 high, y pointing down
SEGMENTS = {
    "a": ((0, 0), (1, 0)), "b": ((1, 0), (1, 1)), "c": ((1, 1), (1, 2)), "d": ((0, 2), (1, 2)),
    "e": ((0, 1), (0, 2)), "f": ((0, 0), (0, 1)), "g": ((0, 1), (1, 1)),
}
DIGIT_SEGMENTS = {
    "0": "abcdef", "1": "bc", "2": "abged", "3": "abgcd", "4": "fgbc",
    "5": "afgcd", "6": "afgedc", "7": "abc", "8": "abcdefg", "9": "abcdfg",
}
SEED_DIGIT_HEIGHT = 26
INDEX_DIGIT_HEIGHT = 10
STORE_DIGIT_HEIGHT = 18


class BoardRenderer:
    """Renders a BoardState in a Tk window. Clicking a playable hole plays it.

    With `on_move`, a click on a playable hole calls on_move(hole) instead of playing
    it, and clicks are only taken while `accept_clicks` is set: the caller then plays
    the moves and redraws. `labels` name the players next to their stores, and
    `last_move` (a hole) is outlined.

    The board scales with the window, keeping its proportions. Callers can add
    widgets under it in `controls`.
    """

    def __init__(self, state: BoardState, labels: tuple[str, str] = ("P0", "P1"), on_move: Callable[[int], None] | None = None):
        self.state = state
        self.labels = labels
        self.on_move = on_move
        self.accept_clicks = True
        self.last_move: int | None = None
        self.root = tk.Tk()
        self.root.title("Awale")
        self.root.minsize(*MIN_WINDOW_SIZE)
        # Packed bottom-up before the canvas, so a small window shrinks the board rather than hiding them
        self.controls = tk.Frame(self.root)
        self.controls.pack(side="bottom", pady=(0, 6))
        self.status = tk.Label(self.root, font=("Helvetica", 14))
        self.status.pack(side="bottom", fill="x", pady=6)
        self.canvas = tk.Canvas(self.root, width=WIDTH, height=HEIGHT, bg=BOARD_COLOR, highlightthickness=0)
        self.canvas.pack(side="top", fill="both", expand=True)
        self.scale, self.offset = 1.0, (0.0, 0.0)
        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<Configure>", self.on_resize)
        self.draw()

    def on_resize(self, event):
        # Largest scale at which the whole board fits, centred in the canvas
        self.scale = min(event.width / WIDTH, event.height / HEIGHT)
        self.offset = ((event.width - WIDTH * self.scale) / 2, (event.height - HEIGHT * self.scale) / 2)
        self.status.config(wraplength=max(1, event.width - 20))  # Long status lines wrap instead of being cut
        self.draw_board()

    def to_screen(self, x: float, y: float) -> tuple[float, float]:
        return self.offset[0] + x * self.scale, self.offset[1] + y * self.scale

    def font(self, size: int, *style: str) -> tuple:
        return ("Helvetica", max(1, round(size * self.scale)), *style)

    def hole_center(self, index: int):
        # Sowing goes counter-clockwise: player 0's holes (0-5) along the bottom
        # left to right, player 1's holes (6-11) along the top right to left.
        if index < HALF_BOARD_SIZE:
            column, y = index, HEIGHT - 110
        else:
            column, y = BOARD_SIZE - 1 - index, 110
        x = MARGIN + HOLE_SPACING * column + HOLE_SPACING // 2
        return x, y

    def hole_at(self, x: int, y: int):
        """The hole under canvas pixel (x, y), or None."""
        x, y = (x - self.offset[0]) / self.scale, (y - self.offset[1]) / self.scale
        for index in range(BOARD_SIZE):
            cx, cy = self.hole_center(index)
            if (x - cx) ** 2 + (y - cy) ** 2 <= HOLE_RADIUS ** 2:
                return index
        return None

    def draw(self, status: str | None = None):
        """Redraws the board; `status` replaces the default "Player n to play" / result line."""
        self.draw_board()
        over = self.state.is_over()
        if status is not None:
            self.status.config(text=status)
        elif not over:
            self.status.config(text=f"Player {int(self.state.current_player)} to play")
        elif self.state.winner() is None:
            self.status.config(text="Game over: draw")
        else:
            self.status.config(text=f"Game over: player {self.state.winner()} wins")

    def draw_board(self):
        self.canvas.delete("all")
        over = self.state.is_over()
        playable = set() if over or not self.accept_clicks else set(self.state.get_possible_moves())
        radius = HOLE_RADIUS * self.scale

        for index in range(BOARD_SIZE):
            board_x, board_y = self.hole_center(index)
            x, y = self.to_screen(board_x, board_y)
            is_playable = index in playable
            if is_playable:
                outline, width = PLAYABLE_COLOR, 5
            elif index == self.last_move:
                outline, width = LAST_MOVE_COLOR, 3
            else:
                outline, width = HOLE_COLOR, 1
            self.canvas.create_oval(
                x - radius, y - radius, x + radius, y + radius,
                fill=HOLE_COLOR, outline=outline, width=max(1, width * self.scale),
            )
            self.draw_number(board_x, board_y, SEED_DIGIT_HEIGHT, self.state.board[index])
            label_y = board_y + HOLE_RADIUS + 14 if index < HALF_BOARD_SIZE else board_y - HOLE_RADIUS - 14
            self.draw_number(board_x, label_y, INDEX_DIGIT_HEIGHT, index)

        for player, y in ((1, 110), (0, HEIGHT - 110)):
            self.canvas.create_text(*self.to_screen(MARGIN // 2, y - 16), text=f"P{player}", fill=TEXT_COLOR, font=self.font(12, "bold"))
            self.draw_number(MARGIN // 2, y + 8, STORE_DIGIT_HEIGHT, self.state.players_seeds[player])
            label_y = 30 if player else HEIGHT - 30
            self.canvas.create_text(*self.to_screen(WIDTH // 2, label_y), text=self.labels[player], fill=TEXT_COLOR, font=self.font(11))

    def draw_number(self, x: float, y: float, height: float, number: int):
        """Draws `number` centred on board point (x, y), its digits `height` board units tall."""
        digits = str(number)
        digit_width, gap = height * 0.5, height * 0.3
        thickness = max(1.0, height * 0.14 * self.scale)
        left = x - (len(digits) * digit_width + (len(digits) - 1) * gap) / 2
        top = y - height / 2
        unit = height / 2  # Board units per digit box unit vertically; the box is 1 wide by 2 high
        for position, digit in enumerate(digits):
            digit_left = left + position * (digit_width + gap)
            for segment in DIGIT_SEGMENTS[digit]:
                (x0, y0), (x1, y1) = SEGMENTS[segment]
                self.canvas.create_line(
                    *self.to_screen(digit_left + x0 * digit_width, top + y0 * unit),
                    *self.to_screen(digit_left + x1 * digit_width, top + y1 * unit),
                    fill=TEXT_COLOR, width=thickness, capstyle="round",
                )

    def on_click(self, event):
        index = self.hole_at(event.x, event.y)
        if index is None or self.state.is_over() or not self.accept_clicks:
            return
        if index not in self.state.get_possible_moves():
            return
        if self.on_move:
            self.on_move(index)
            return
        self.state.make_move(index)
        self.last_move = index
        self.draw()

    def run(self):
        self.root.mainloop()


def render(state: BoardState):
    BoardRenderer(state).run()


if __name__ == "__main__":
    render(new_game())
