/* Awale's types and sizes for the engines of core/c (see core/c/game_api.h) */
#ifndef GAME_H
#define GAME_H

#include "awale.h"

typedef BoardState GameState;

#define GAME_MAX_MOVES HALF_BOARD_SIZE /* A move is one of your own holes */
#define GAME_MAX_PLIES MAX_MOVES

/* NNUE features, as in games/awale/features.py: "hole holds n seeds" and "store holds n seeds",
 * seen from a player's side (own holes first) */
#define MAX_HOLE_SEEDS 24 /* Seed counts above this share the last bucket */
#define MAX_STORE_SEEDS 48
#define HOLE_BUCKETS (MAX_HOLE_SEEDS + 1)
#define STORE_BUCKETS (MAX_STORE_SEEDS + 1)
#define GAME_NUM_FEATURES (BOARD_SIZE * HOLE_BUCKETS + 2 * STORE_BUCKETS)
#define GAME_NUM_ACTIVE_FEATURES (BOARD_SIZE + 2)

#endif
