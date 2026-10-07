/* Entry points for Python (core/c_library.py loads lib<game>.so with ctypes).
 *
 * Positions cross the boundary as the ints of the game's Python `encode`, decoded by
 * game_decode, so the Python side never depends on the GameState layout. Entry points
 * taking a position return BAD_POSITION when it does not decode.
 */
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "game_api.h"
#include "mcts.h"
#include "search.h"

#define BAD_POSITION (-2)

/* ---------- MCTS (core/c_mcts.py) ---------- */

/* A searcher with the default settings, apart from the given ones.
 * iterations: MCTS_NO_LIMIT for no cap; move_time: seconds per search, <= 0 for none. */
MCTS *py_mcts_new(long iterations, double move_time, int reuse, uint64_t seed) {
    MCTS *mcts = malloc(sizeof *mcts);
    if (!mcts) return NULL;
    mcts_init(mcts, seed);
    mcts->iterations = iterations;
    mcts->move_time = move_time;
    mcts->reuse = reuse;
    return mcts;
}

void py_mcts_delete(MCTS *mcts) {
    if (!mcts) return;
    mcts_free(mcts);
    free(mcts);
}

/* The move MCTS picks in this position, -1 if the game is over */
int py_mcts_best_move(MCTS *mcts, const int32_t *position, int count) {
    GameState state;
    if (!game_decode(position, count, &state)) return BAD_POSITION;
    return mcts_best_move(mcts, &state);
}

/* ---------- Alpha-beta with NNUE (core/c_search.py) ---------- */

/* Floats in a Network, which Python checks its weights against */
long py_network_floats(void) { return (long)(sizeof(Network) / sizeof(float)); }

/* A searcher owning a copy of `weights`, laid out as Network's fields (see c_search.py).
 * NULL if the weights do not fit a Network or memory runs out. */
AlphaBeta *py_ab_new(const float *weights, long num_floats, int depth, double move_time, int table_bits) {
    if (num_floats != py_network_floats()) return NULL;
    Network *net = malloc(sizeof *net);
    if (!net) return NULL;
    memcpy(net, weights, sizeof *net);
    AlphaBeta *ab = alpha_beta_new(net, depth, move_time, table_bits);
    if (!ab) free(net);
    return ab;
}

void py_ab_delete(AlphaBeta *ab) { alpha_beta_delete(ab); }

/* The move to play in a position that is not over */
int py_ab_best_move(AlphaBeta *ab, const int32_t *position, int count) {
    GameState state;
    if (!game_decode(position, count, &state)) return BAD_POSITION;
    return alpha_beta_best_move(ab, &state);
}

/* Statistics of the last search: nodes searched, depth completed, score of the move for the side to move */
void py_ab_stats(const AlphaBeta *ab, uint64_t *nodes, int *depth, float *score) {
    *nodes = ab->nodes;
    *depth = ab->completed_depth;
    *score = ab->score;
}

/* The network's evaluation of a position for the side to move, from fresh accumulators; NAN if it does not decode */
float py_ab_evaluate(const AlphaBeta *ab, const int32_t *position, int count) {
    GameState state;
    if (!game_decode(position, count, &state)) return NAN;
    Accumulator acc;
    accumulator_refresh(ab->net, &acc, &state);
    return nnue_evaluate(ab->net, &acc, game_side_to_move(&state));
}
