#include "awale.h"

#include <string.h>

BoardState new_game(void) {
    BoardState state = {0};
    memset(state.board, 4, sizeof state.board);
    return state;
}

int get_possible_moves(const BoardState *state, int moves[HALF_BOARD_SIZE]) {
    int first = state->current_player * HALF_BOARD_SIZE;
    int count = 0;
    for (int hole = first; hole < first + HALF_BOARD_SIZE; hole++) {
        if (state->board[hole] > 0) moves[count++] = hole;
    }
    return count;
}

/* Whether `hole` is on the opponent's side, where the side to move captures */
static bool right_side(const BoardState *state, int hole) {
    return state->current_player == 0 ? hole >= HALF_BOARD_SIZE : hole < HALF_BOARD_SIZE;
}

void make_move(BoardState *state, int move) {
    int seeds_to_sow = state->board[move];
    state->board[move] = 0;
    int index = move;
    while (seeds_to_sow > 0) {
        index = (index + 1) % BOARD_SIZE;
        if (index != move) { /* Skip the origin hole on the second lap */
            state->board[index]++;
            seeds_to_sow--;
        }
    }

    int captured_seeds = 0;
    while (right_side(state, index) && (state->board[index] == 2 || state->board[index] == 3)) {
        captured_seeds += state->board[index];
        state->board[index] = 0;
        index = index ? index - 1 : BOARD_SIZE - 1;
    }
    state->players_seeds[state->current_player] += captured_seeds;

    /* Switch player after the move */
    state->current_player = opponent(state);
    state->moves_played++;
}

static bool side_empty(const BoardState *state, int player) {
    for (int hole = player * HALF_BOARD_SIZE; hole < (player + 1) * HALF_BOARD_SIZE; hole++) {
        if (state->board[hole]) return false;
    }
    return true;
}

bool is_over(const BoardState *state) {
    return state->players_seeds[0] > TOTAL_SEEDS / 2 || state->players_seeds[1] > TOTAL_SEEDS / 2 ||
           side_empty(state, 0) || side_empty(state, 1) || state->moves_played >= MAX_MOVES;
}

int winner(const BoardState *state) {
    if (state->players_seeds[0] == state->players_seeds[1]) return NO_WINNER;
    return state->players_seeds[0] > state->players_seeds[1] ? 0 : 1;
}

int result(const BoardState *state, int player) {
    int w = winner(state);
    return w == NO_WINNER ? 0 : (w == player ? 1 : -1);
}

bool states_equal(const BoardState *a, const BoardState *b) {
    return memcmp(a->board, b->board, sizeof a->board) == 0 && a->players_seeds[0] == b->players_seeds[0] &&
           a->players_seeds[1] == b->players_seeds[1] && a->current_player == b->current_player &&
           a->moves_played == b->moves_played;
}
