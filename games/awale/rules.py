"""Awale rules: a BoardState is a position, implementing core/game.py's GameState.

Holes 0-5 belong to player 0, holes 6-11 to player 1. The C port (c/awale.c) is
checked against this file by tests/check_rules.py.
"""

BOARD_SIZE = 12
HALF_BOARD_SIZE = BOARD_SIZE // 2
TOTAL_SEEDS = 48
MAX_MOVES = 100  # The game stops after this many moves and is scored on captured seeds


class BoardState:
    def __init__(self, board: list, current_player: bool, player_0_seeds: int = 0, player_1_seeds: int = 0, moves_played: int = 0):
        self.board = board
        self.players_seeds = [player_0_seeds, player_1_seeds]
        self.current_player = current_player
        self.moves_played = moves_played

    def copy(self) -> "BoardState":
        return BoardState(self.board[:], self.current_player, *self.players_seeds, self.moves_played)

    def get_possible_moves(self):
        if self.current_player == 0:
            return [i for i in range(HALF_BOARD_SIZE) if self.board[i] > 0]
        else:
            return [i for i in range(HALF_BOARD_SIZE, BOARD_SIZE) if self.board[i] > 0]

    def make_move(self, move: int):
        seeds_to_sow = self.board[move]
        self.board[move] = 0
        index = move
        while seeds_to_sow > 0:
            index = (index + 1) % BOARD_SIZE
            if index != move:  # Skip the origin pit on the second lap
                self.board[index] += 1
                seeds_to_sow -= 1

        right_side = lambda index: index >= HALF_BOARD_SIZE if self.current_player == 0 else index < HALF_BOARD_SIZE
        if right_side(index) and self.board[index] in (2, 3):
            captured_seeds = 0
            while right_side(index) and self.board[index] in (2, 3):
                captured_seeds += self.board[index]
                self.board[index] = 0
                index = index - 1 if index else 11
            self.players_seeds[self.current_player] += captured_seeds
        
        # Switch player after the move
        self.current_player = self.opponent
        self.moves_played += 1

    @property
    def opponent(self) -> int:
        return 1 - self.current_player

    @property
    def moves_left(self) -> int:
        return MAX_MOVES - self.moves_played

    def seeds_on_board(self) -> int:
        return sum(self.board)

    def board_from(self, player: int) -> list[int]:
        """The holes seen from `player`'s side: their own six first, then the opponent's."""
        offset = player * HALF_BOARD_SIZE
        return self.board[offset:] + self.board[:offset]

    def key(self) -> tuple:
        """Hashable identity of the position."""
        return (*self.board, *self.players_seeds, self.current_player, self.moves_played)

    def is_over(self) -> bool:
        return (
            max(self.players_seeds) > TOTAL_SEEDS // 2
            or all(seeds == 0 for seeds in self.board[:HALF_BOARD_SIZE])
            or all(seeds == 0 for seeds in self.board[HALF_BOARD_SIZE:])
            or self.moves_played >= MAX_MOVES
        )

    def winner(self) -> int | None:
        """The player with more captured seeds (0 or 1), None if level.

        For a finished game this is the winner. It is also meaningful for an
        unfinished one, as the player currently ahead."""
        if self.players_seeds[0] == self.players_seeds[1]:
            return None
        return 0 if self.players_seeds[0] > self.players_seeds[1] else 1

    def result(self, player: int) -> int:
        """+1 if `player` wins, -1 if they lose, 0 for a draw."""
        winner = self.winner()
        return 0 if winner is None else (1 if winner == player else -1)

    def __eq__(self, other) -> bool:
        return isinstance(other, BoardState) and self.key() == other.key()

    __hash__ = None


def new_game() -> BoardState:
    return BoardState([4] * BOARD_SIZE, current_player=0)
