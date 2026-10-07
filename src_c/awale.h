/* Awale rules, ported from src/awale.py.
 *
 * Holes 0-5 belong to player 0, holes 6-11 to player 1. A move sows the seeds of
 * one of your holes counterclockwise, skipping the origin hole, and captures the
 * opponent's holes left with 2 or 3 seeds, walking back from the last one sown.
 * The game ends when a player has captured more than half the seeds, when a side
 * is empty, or after MAX_MOVES moves; it is then scored on captured seeds.
 */
#ifndef AWALE_H
#define AWALE_H

#include <stdbool.h>
#include <stdint.h>

#define BOARD_SIZE 12
#define HALF_BOARD_SIZE (BOARD_SIZE / 2)
#define TOTAL_SEEDS 48
#define MAX_MOVES 100 /* The game stops after this many moves and is scored on captured seeds */
#define NO_WINNER (-1)

typedef struct {
    uint8_t board[BOARD_SIZE];
    uint8_t players_seeds[2];
    uint8_t current_player;
    uint8_t moves_played;
} BoardState;

BoardState new_game(void);

/* Writes the legal moves (holes of the side to move that hold seeds) to `moves`, returns how many */
int get_possible_moves(const BoardState *state, int moves[HALF_BOARD_SIZE]);

void make_move(BoardState *state, int move);

static inline int opponent(const BoardState *state) { return 1 - state->current_player; }

static inline int moves_left(const BoardState *state) { return MAX_MOVES - state->moves_played; }

bool is_over(const BoardState *state);

/* The player with more captured seeds (0 or 1), NO_WINNER if level.
 * For a finished game this is the winner; for an unfinished one, the player ahead. */
int winner(const BoardState *state);

/* +1 if `player` wins, -1 if they lose, 0 for a draw */
int result(const BoardState *state, int player);

bool states_equal(const BoardState *a, const BoardState *b);

#endif
