/* Alpha-beta search with an NNUE evaluation at the leaves, for any game of game_api.h.
 *
 * Negamax alpha-beta: every score is from the side to move's point of view, in
 * [-1, 1] like the network's output, with finished games scored exactly (+1, 0, -1).
 *
 * - Turns: the depth counts turns, not moves. A move after which the same player moves
 *   again (a jump of a multiple jump in checkers) keeps the depth and the score's sign,
 *   so the network never evaluates a position in the middle of a turn.
 * - Iterative deepening: depth 1, 2, ... up to the target depth, each iteration
 *   ordering moves with the best moves found by the previous ones.
 * - Transposition table: positions already searched reuse their score and best move.
 *   A fixed-size table of the game's position keys, two entries per bucket: one kept
 *   for the deepest search (until a newer search replaces it), one always replaced.
 * - Exact results (optional): positions the game's oracle covers, such as Awale's
 *   endgame table, are scored exactly, without search.
 * - Accumulators: the search keeps one per ply and computes them lazily, only when a
 *   leaf is evaluated, by updating from the nearest ancestor that has one.
 * - Time limit (optional): deepening stops when the time per move runs out, and the
 *   move of the last completed iteration is played. A forced move (the only legal one)
 *   is played at once.
 */
#ifndef SEARCH_H
#define SEARCH_H

#include <stdbool.h>
#include <stdint.h>

#include "game_api.h"
#include "nnue.h"

#define EXACT 0
#define LOWER_BOUND 1
#define UPPER_BOUND 2
#define NO_MOVE (-1)

/* Deepest search: no game lasts longer, and Transposition.depth is a byte */
#define MAX_DEPTH (GAME_MAX_PLIES < INT8_MAX ? GAME_MAX_PLIES : INT8_MAX)

typedef struct {
    uint64_t key;
    float score;
    int8_t depth;
    uint8_t bound;
    int8_t best_move;
    uint8_t generation; /* The search that stored it, for replacement */
} Transposition;

typedef struct {
    GameState state;
    Accumulator acc;
    bool acc_computed;
} Ply;

typedef struct {
    Network *net;
    int depth;        /* Iteration cap, in turns; MAX_DEPTH when only the time limits the search */
    double move_time; /* Seconds per move, <= 0 for none */
    void *oracle;     /* For game_exact_result, NULL for none; owned (freed with free) */

    Transposition *table;
    uint64_t table_mask; /* Number of entries - 1, a power of two minus one */
    uint8_t generation;

    Ply plies[GAME_MAX_PLIES + 2]; /* plies[0] is the root */
    double deadline;
    bool stopped; /* Out of time: the running iteration unwinds and is dropped */

    /* Statistics of the last search */
    uint64_t nodes;
    int completed_depth;
    float score;
} AlphaBeta;

/* NULL when out of memory. `net` is owned by the searcher from now on; `table_bits`: log2 of the
 * transposition table's entries. `depth` < 1 means no depth limit, which needs a move time. */
AlphaBeta *alpha_beta_new(Network *net, int depth, double move_time, int table_bits);
void alpha_beta_delete(AlphaBeta *ab);

/* Hands `oracle` (malloc'd) to the searcher, for game_exact_result; replaces any previous one */
void alpha_beta_set_oracle(AlphaBeta *ab, void *oracle);

/* The move to play in `state`, which is not over */
int alpha_beta_best_move(AlphaBeta *ab, const GameState *state);

#endif
