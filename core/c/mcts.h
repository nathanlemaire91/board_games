/* UCT Monte Carlo tree search with random rollouts, for any game of game_api.h.
 *
 * With reuse, the subtree of the position reached since the last search (our move,
 * then usually the opponent's reply, each possibly several moves in a row) is kept,
 * along with its statistics.
 * With a move time, each search runs for that many seconds, the iteration count
 * being an optional cap (MCTS_NO_LIMIT: no cap).
 */
#ifndef MCTS_H
#define MCTS_H

#include <stdbool.h>
#include <stdint.h>

#include "game_api.h"

#define MCTS_NO_LIMIT (-1)
#define NO_NODE (-1)

typedef struct {
    GameState state;
    int32_t parent;
    int32_t children[GAME_MAX_MOVES];
    uint8_t num_children;
    uint8_t untried_moves[GAME_MAX_MOVES];
    uint8_t num_untried;
    int8_t move;              /* Move leading here from the parent, -1 for a root */
    uint8_t player_just_moved; /* Wins are counted from this player's point of view: the parent's side to move */
    uint32_t visits;
    double wins;
} Node;

typedef struct {
    long iterations;   /* MCTS_NO_LIMIT needs a move time */
    double move_time;  /* Seconds per search, <= 0 for no time limit */
    double exploration;
    int max_rollout_moves; /* Awale games can loop without captures, so rollouts are cut off */
    bool reuse;
    uint64_t rng;

    Node *nodes; /* Pool of the current tree, nodes[root] being its root */
    int32_t num_nodes, capacity;
    int32_t root; /* NO_NODE before the first search */
} MCTS;

/* Default settings: 1000 iterations, sqrt(2) exploration, 200-move rollouts, reuse */
void mcts_init(MCTS *mcts, uint64_t seed);
void mcts_free(MCTS *mcts);

/* Searches `state` and returns the index of its root node in mcts->nodes */
int32_t mcts_search(MCTS *mcts, const GameState *state);

/* The most visited move after a search, -1 if the game is already over */
int mcts_best_move(MCTS *mcts, const GameState *state);

#endif
