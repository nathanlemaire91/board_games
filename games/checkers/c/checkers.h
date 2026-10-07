/* Checkers (English draughts) rules, ported from games/checkers/rules.py.
 *
 * The 32 dark squares of the 8x8 board are numbered row by row from player 0's side, four per
 * row: square s is on row s / 4, row 0 being player 0's back row. Player 0 (dark pieces, moves
 * first) starts on squares 0-11 and moves up the board, player 1 on squares 20-31 and moves down.
 *
 * A move is one step or one jump of a piece: square * 4 + direction, the directions being
 * up-left, up-right, down-left and down-right as player 0 sees the board. Men move forward,
 * kings both ways. Capturing is mandatory, though any capture may be chosen. A multiple jump is
 * a series of moves by the same player: after a jump, if the jumping piece can jump again, its
 * player moves again and only that piece's jumps are legal. A man reaching the far row is
 * crowned king, which ends its move.
 *
 * A player who cannot move loses. The game is drawn after QUIET_LIMIT moves in a row without a
 * capture or a man move, or after MAX_PLIES moves.
 */
#ifndef CHECKERS_H
#define CHECKERS_H

#include <stdbool.h>
#include <stdint.h>

#define NUM_SQUARES 32
#define ROWS 8
#define SQUARES_PER_ROW 4
#define NUM_DIRECTIONS 4
#define MAX_PIECES 12 /* Per player */
#define MAX_LEGAL_MOVES (MAX_PIECES * NUM_DIRECTIONS)
#define QUIET_LIMIT 80 /* Moves in a row without a capture or a man move: 40 per player */
#define MAX_PLIES 300
#define NO_SQUARE (-1)
#define NO_WINNER (-1)

/* A square holds EMPTY or a piece, 1 + 2 * player + (1 for a king) */
#define EMPTY 0

static inline int make_piece(int player, bool king) { return 1 + 2 * player + king; }
static inline int owner(int piece) { return (piece - 1) / 2; }
static inline bool is_king(int piece) { return piece != EMPTY && piece % 2 == 0; }

typedef struct {
    uint8_t board[NUM_SQUARES];
    int8_t jumping;        /* The square of the piece in the middle of a multiple jump, NO_SQUARE if none */
    uint8_t current_player;
    uint8_t quiet_moves;   /* Moves in a row without a capture or a man move */
    uint16_t moves_played;
} BoardState;

BoardState new_game(void);

/* Writes the moves the rules allow (in increasing order, ignoring the draw limits) to `moves`, returns how many */
int get_possible_moves(const BoardState *state, int moves[MAX_LEGAL_MOVES]);

/* Whether the piece on `square` can jump */
bool can_jump(const BoardState *state, int square);

void make_move(BoardState *state, int move);

static inline int opponent(const BoardState *state) { return 1 - state->current_player; }

static inline int moves_left(const BoardState *state) { return MAX_PLIES - state->moves_played; }

bool draw_limit_reached(const BoardState *state);

bool is_over(const BoardState *state);

/* The winner of a finished game, NO_WINNER for a draw; a player who cannot move loses, even when
 * a draw limit is reached at the same time. For an unfinished game, the player with more material
 * (men 2, kings 3), NO_WINNER if level. */
int winner(const BoardState *state);

/* +1 if `player` wins, -1 if they lose, 0 for a draw */
int result(const BoardState *state, int player);

bool states_equal(const BoardState *a, const BoardState *b);

#endif
