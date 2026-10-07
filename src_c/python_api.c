/* Entry points for Python (src/c_mcts.py loads libawale.so with ctypes).
 *
 * Positions cross the boundary as plain values rather than a BoardState, so the
 * Python side does not depend on the struct's layout.
 */
#include <stdlib.h>

#include "awale.h"
#include "mcts.h"

/* A searcher with mcts.py's defaults, apart from the given settings.
 * iterations: MCTS_NO_LIMIT for no cap; move_time: seconds per search, <= 0 for none. */
MCTS *awale_mcts_new(long iterations, double move_time, int reuse, uint64_t seed) {
    MCTS *mcts = malloc(sizeof *mcts);
    if (!mcts) return NULL;
    mcts_init(mcts, seed);
    mcts->iterations = iterations;
    mcts->move_time = move_time;
    mcts->reuse = reuse;
    return mcts;
}

void awale_mcts_delete(MCTS *mcts) {
    if (!mcts) return;
    mcts_free(mcts);
    free(mcts);
}

/* The move MCTS picks in this position, -1 if the game is over */
int awale_mcts_best_move(MCTS *mcts, const uint8_t board[BOARD_SIZE], int player_0_seeds, int player_1_seeds,
                         int current_player, int moves_played) {
    BoardState state;
    for (int hole = 0; hole < BOARD_SIZE; hole++) state.board[hole] = board[hole];
    state.players_seeds[0] = (uint8_t)player_0_seeds;
    state.players_seeds[1] = (uint8_t)player_1_seeds;
    state.current_player = (uint8_t)current_player;
    state.moves_played = (uint8_t)moves_played;
    return mcts_best_move(mcts, &state);
}
