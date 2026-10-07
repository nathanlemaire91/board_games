"""Checkers (English draughts) rules: a BoardState is a position, implementing core/game.py's GameState.

The 32 dark squares of the 8x8 board are numbered row by row from player 0's side, four
per row: square s is on row s // 4, row 0 being player 0's back row. Player 0 (dark
pieces, moves first) starts on squares 0-11 and moves up the board, player 1 starts on
squares 20-31 and moves down. Turning the board around maps square s to 31 - s.

A move is one step or one jump of a piece: square * 4 + direction, the directions being
up-left, up-right, down-left and down-right as player 0 sees the board. Men move forward,
kings both ways. Capturing is mandatory, though any capture may be chosen. A multiple
jump is a series of moves by the same player: after a jump, if the jumping piece can jump
again, its player moves again and only that piece's jumps are legal. A man reaching the
far row is crowned king, which ends its move.

A player who cannot move (no pieces left, or all blocked) loses. The game is drawn after
QUIET_LIMIT moves in a row without a capture or a man move, or after MAX_PLIES moves.

The C port (c/checkers.c) is checked against this file by tests/check_rules.py.
"""

NUM_SQUARES = 32
ROWS = 8
SQUARES_PER_ROW = 4
NUM_DIRECTIONS = 4
UP_LEFT, UP_RIGHT, DOWN_LEFT, DOWN_RIGHT = range(NUM_DIRECTIONS)
MAX_PIECES = 12  # Per player
MAX_LEGAL_MOVES = MAX_PIECES * NUM_DIRECTIONS
QUIET_LIMIT = 80  # Moves in a row without a capture or a man move: 40 per player
MAX_PLIES = 300

# A square holds EMPTY or a piece, 1 + 2 * player + (1 for a king)
EMPTY = 0
CROWNING_ROW = (ROWS - 1, 0)  # Per player


def make_piece(player: int, king: bool) -> int:
    return 1 + 2 * player + king


def owner(piece: int) -> int:
    return (piece - 1) // 2


def is_king(piece: int) -> bool:
    return piece != EMPTY and piece % 2 == 0


def row(square: int) -> int:
    return square // SQUARES_PER_ROW


def column(square: int) -> int:
    """0-7 from player 0's left: dark squares are where row + column is even."""
    return 2 * (square % SQUARES_PER_ROW) + row(square) % 2


def step(square: int, direction: int) -> int | None:
    """The square next to `square` in `direction`, None off the board."""
    to_row = row(square) + (1 if direction in (UP_LEFT, UP_RIGHT) else -1)
    to_column = column(square) + (1 if direction in (UP_RIGHT, DOWN_RIGHT) else -1)
    if 0 <= to_row < ROWS and 0 <= to_column < ROWS:
        return to_row * SQUARES_PER_ROW + to_column // 2
    return None


STEPS = [[step(square, direction) for direction in range(NUM_DIRECTIONS)] for square in range(NUM_SQUARES)]
# The directions each piece moves in: men forward, kings both ways
DIRECTIONS = {
    make_piece(0, False): (UP_LEFT, UP_RIGHT),
    make_piece(1, False): (DOWN_LEFT, DOWN_RIGHT),
    make_piece(0, True): tuple(range(NUM_DIRECTIONS)),
    make_piece(1, True): tuple(range(NUM_DIRECTIONS)),
}


class BoardState:
    def __init__(self, board: list[int], current_player: int = 0, jumping: int | None = None,
                 quiet_moves: int = 0, moves_played: int = 0):
        self.board = board
        self.current_player = current_player
        self.jumping = jumping  # The square of the piece in the middle of a multiple jump, None if none
        self.quiet_moves = quiet_moves  # Moves in a row without a capture or a man move
        self.moves_played = moves_played

    def copy(self) -> "BoardState":
        return BoardState(self.board[:], self.current_player, self.jumping, self.quiet_moves, self.moves_played)

    def jumps(self, square: int) -> list[int]:
        """The jumps of the piece on `square`."""
        piece = self.board[square]
        moves = []
        for direction in DIRECTIONS[piece]:
            over = STEPS[square][direction]
            if over is None or self.board[over] == EMPTY or owner(self.board[over]) == owner(piece):
                continue
            to = STEPS[over][direction]
            if to is not None and self.board[to] == EMPTY:
                moves.append(square * NUM_DIRECTIONS + direction)
        return moves

    def get_possible_moves(self) -> list[int]:
        """The moves the rules allow, in increasing order, ignoring the draw limits (see is_over)."""
        if self.jumping is not None:
            return self.jumps(self.jumping)
        own = [square for square, piece in enumerate(self.board)
               if piece != EMPTY and owner(piece) == self.current_player]
        jumps = [move for square in own for move in self.jumps(square)]
        if jumps:
            return jumps
        return [square * NUM_DIRECTIONS + direction for square in own for direction in DIRECTIONS[self.board[square]]
                if STEPS[square][direction] is not None and self.board[STEPS[square][direction]] == EMPTY]

    def make_move(self, move: int):
        origin, direction = divmod(move, NUM_DIRECTIONS)
        piece = self.board[origin]
        to = STEPS[origin][direction]
        captured = self.board[to] != EMPTY  # A legal move onto an occupied square is a jump over it
        if captured:
            self.board[to] = EMPTY
            to = STEPS[to][direction]
        self.board[origin] = EMPTY
        crowned = not is_king(piece) and row(to) == CROWNING_ROW[owner(piece)]
        self.board[to] = make_piece(owner(piece), True) if crowned else piece
        self.quiet_moves = 0 if captured or not is_king(piece) else self.quiet_moves + 1
        self.moves_played += 1
        if captured and not crowned and self.jumps(to):
            self.jumping = to  # The same player moves again
        else:
            self.jumping = None
            self.current_player = self.opponent

    @property
    def opponent(self) -> int:
        return 1 - self.current_player

    @property
    def moves_left(self) -> int:
        return MAX_PLIES - self.moves_played

    def draw_limit_reached(self) -> bool:
        return self.quiet_moves >= QUIET_LIMIT or self.moves_played >= MAX_PLIES

    def is_over(self) -> bool:
        return self.draw_limit_reached() or not self.get_possible_moves()

    def material(self, player: int) -> int:
        """Men count 2, kings 3."""
        return sum(3 if is_king(piece) else 2 for piece in self.board if piece != EMPTY and owner(piece) == player)

    def winner(self) -> int | None:
        """The winner of a finished game, None for a draw. A player who cannot move loses, even
        when a draw limit is reached at the same time.

        For an unfinished game: the player with more material, None if level."""
        if not self.get_possible_moves():
            return self.opponent
        if self.draw_limit_reached():
            return None
        lead = self.material(0) - self.material(1)
        return None if lead == 0 else (0 if lead > 0 else 1)

    def result(self, player: int) -> int:
        """+1 if `player` wins, -1 if they lose, 0 for a draw."""
        winner = self.winner()
        return 0 if winner is None else (1 if winner == player else -1)

    def key(self) -> tuple:
        """Hashable identity of the position."""
        return (*self.board, self.jumping, self.current_player, self.quiet_moves, self.moves_played)

    def __eq__(self, other) -> bool:
        return isinstance(other, BoardState) and self.key() == other.key()

    __hash__ = None

    def __repr__(self) -> str:
        return f"BoardState({self.board}, {self.current_player}, {self.jumping}, {self.quiet_moves}, {self.moves_played})"


def move_squares(state: BoardState, move: int) -> tuple[int, int, int | None]:
    """(from, to, captured square or None) of a legal move in `state`."""
    origin, direction = divmod(move, NUM_DIRECTIONS)
    to = STEPS[origin][direction]
    if state.board[to] == EMPTY:
        return origin, to, None
    return origin, STEPS[to][direction], to


def new_game() -> BoardState:
    board = [EMPTY] * NUM_SQUARES
    for square in range(MAX_PIECES):
        board[square] = make_piece(0, False)
        board[NUM_SQUARES - 1 - square] = make_piece(1, False)
    return BoardState(board)
