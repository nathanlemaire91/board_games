/* NNUE evaluation, the network of core/nnue.py, with the inputs prepared Stockfish style
 * (https://official-stockfish.github.io/docs/nnue-pytorch-wiki/docs/nnue.html#preparing-the-inputs):
 *
 * - Sparse inputs: a position is the list of its active features, from each player's
 *   perspective (game_features). Every slot holds exactly one feature, so there are
 *   always GAME_NUM_ACTIVE_FEATURES of them.
 * - Accumulator: the feature transformer's output for both perspectives, the sum of
 *   the active features' weight rows plus the bias. A move changes only the slots it
 *   touched, so the accumulator is updated from the previous one by subtracting the
 *   removed features' rows and adding the added ones, or refreshed from scratch when
 *   that is cheaper (a long sowing changes most slots).
 * - Forward pass: [stm accumulator, opponent accumulator] -> ClippedReLU -> Linear ->
 *   ClippedReLU -> Linear -> ClippedReLU -> Linear -> tanh.
 *
 * Weights stay in float32, so scores match the PyTorch network up to rounding.
 */
#ifndef NNUE_H
#define NNUE_H

#include <stdint.h>

#include "game_api.h"

/* Must match core/nnue.py */
#define ACCUMULATOR_SIZE 256
#define HIDDEN_SIZE 32

typedef struct {
    float feature_weights[GAME_NUM_FEATURES][ACCUMULATOR_SIZE]; /* One row per feature, as nn.Embedding */
    float feature_bias[ACCUMULATOR_SIZE];
    /* Linear layers, input-major (nn.Linear's weight transposed): weight[input][output], so that
     * each input adds a contiguous row to the outputs, and inputs at zero are skipped */
    float l1_weight[2 * ACCUMULATOR_SIZE][HIDDEN_SIZE], l1_bias[HIDDEN_SIZE];
    float l2_weight[HIDDEN_SIZE][HIDDEN_SIZE], l2_bias[HIDDEN_SIZE];
    float l3_weight[HIDDEN_SIZE], l3_bias;
} Network;

typedef struct {
    float values[2][ACCUMULATOR_SIZE];             /* Indexed by perspective (player 0, player 1) */
    int16_t features[2][GAME_NUM_ACTIVE_FEATURES]; /* The active features they sum */
} Accumulator;

/* Recomputes both perspectives from the active features */
void accumulator_refresh(const Network *net, Accumulator *acc, const GameState *state);

/* `acc` for `state`, from `previous` (the accumulator of the position before the move) */
void accumulator_update(const Network *net, Accumulator *acc, const Accumulator *previous, const GameState *state);

/* Evaluation in [-1, 1] for `side_to_move` (1 = winning) */
float nnue_evaluate(const Network *net, const Accumulator *acc, int side_to_move);

#endif
