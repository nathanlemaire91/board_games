/* Awale's endgame table (games/awale/endgame.py) as the alpha-beta's exact results.
 *
 * Python builds the table and passes its gains array, read in place: gains[moves_left][board]
 * is the best capture difference the side to move can force, boards (seen from the side to
 * move, own holes first) in lexicographic order of their hole counts.
 */
#include <stdlib.h>

#include "game_api.h"
#include "search.h"

typedef struct {
    const int8_t *gains; /* Not owned: Python keeps the array alive */
    int64_t num_boards;
    int max_seeds;
} EndgameTable;

/* binomial(n, k), for the board ranks */
static uint64_t binomial(int n, int k) {
    if (k < 0 || k > n) return 0;
    uint64_t result = 1;
    for (int i = 1; i <= k; i++) result = result * (uint64_t)(n - k + i) / (uint64_t)i;
    return result;
}

/* Number of boards with at most `max_seeds` seeds ranked before `board`, the table's order */
static int64_t endgame_rank(const uint8_t board[BOARD_SIZE], int max_seeds) {
    /* Before `board` come, for each hole i, the boards that agree on holes 0..i-1 and hold fewer
     * seeds in hole i: with `remaining` seeds left for holes i..11 and k = 11 - i holes after
     * hole i, there are C(r + k, k) ways to put at most r seeds in k holes, and summing over hole
     * i's values v < board[i] (r = remaining - v) gives
     * C(remaining + k + 1, k + 1) - C(remaining - board[i] + k + 1, k + 1). */
    int64_t rank = 0;
    int remaining = max_seeds;
    for (int hole = 0; hole < BOARD_SIZE; hole++) {
        int k = BOARD_SIZE - 1 - hole, seeds = board[hole];
        rank += (int64_t)(binomial(remaining + k + 1, k + 1) - binomial(remaining - seeds + k + 1, k + 1));
        remaining -= seeds;
    }
    return rank;
}

bool game_exact_result(const void *oracle, const GameState *state, int *result) {
    const EndgameTable *table = oracle;
    int seeds = 0;
    for (int hole = 0; hole < BOARD_SIZE; hole++) seeds += state->board[hole];
    if (seeds > table->max_seeds) return false;
    uint8_t board[BOARD_SIZE]; /* Seen from the side to move */
    int offset = state->current_player * HALF_BOARD_SIZE;
    for (int slot = 0; slot < BOARD_SIZE; slot++) board[slot] = state->board[(slot + offset) % BOARD_SIZE];
    int gain = table->gains[(int64_t)moves_left(state) * table->num_boards + endgame_rank(board, table->max_seeds)];
    int total = state->players_seeds[state->current_player] - state->players_seeds[opponent(state)] + gain;
    *result = (total > 0) - (total < 0);
    return true;
}

/* ---------- Entry points for Python ---------- */

/* Gives `ab` the gains array (GAME_MAX_PLIES + 1 rows of `num_boards`), which must outlive it.
 * Returns 0 if `num_boards` is not the number of boards with at most `max_seeds` seeds, or out of memory. */
int awale_set_endgame(AlphaBeta *ab, const int8_t *gains, int64_t num_boards, int max_seeds) {
    uint8_t last[BOARD_SIZE] = {0};
    last[0] = (uint8_t)max_seeds; /* The last board in lexicographic order */
    if (max_seeds < 0 || max_seeds > TOTAL_SEEDS / 2 || endgame_rank(last, max_seeds) != num_boards - 1) return 0;
    EndgameTable *table = malloc(sizeof *table);
    if (!table) return 0;
    *table = (EndgameTable){gains, num_boards, max_seeds};
    alpha_beta_set_oracle(ab, table);
    return 1;
}

int64_t awale_endgame_rank(const uint8_t board[BOARD_SIZE], int max_seeds) { return endgame_rank(board, max_seeds); }
