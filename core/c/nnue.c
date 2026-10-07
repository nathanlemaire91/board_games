#include "nnue.h"

#include <math.h>
#include <string.h>

static void add_row(float *restrict values, const float *restrict row) {
    for (int i = 0; i < ACCUMULATOR_SIZE; i++) values[i] += row[i];
}

static void subtract_row(float *restrict values, const float *restrict row) {
    for (int i = 0; i < ACCUMULATOR_SIZE; i++) values[i] -= row[i];
}

static void refresh_perspective(const Network *net, Accumulator *acc, int perspective) {
    float *values = acc->values[perspective];
    memcpy(values, net->feature_bias, sizeof net->feature_bias);
    for (int i = 0; i < GAME_NUM_ACTIVE_FEATURES; i++) add_row(values, net->feature_weights[acc->features[perspective][i]]);
}

void accumulator_refresh(const Network *net, Accumulator *acc, const GameState *state) {
    for (int perspective = 0; perspective < 2; perspective++) {
        game_features(state, perspective, acc->features[perspective]);
        refresh_perspective(net, acc, perspective);
    }
}

void accumulator_update(const Network *net, Accumulator *acc, const Accumulator *previous, const GameState *state) {
    for (int perspective = 0; perspective < 2; perspective++) {
        int16_t *features = acc->features[perspective];
        game_features(state, perspective, features);
        int16_t removed[GAME_NUM_ACTIVE_FEATURES], added[GAME_NUM_ACTIVE_FEATURES];
        int changed = 0;
        for (int slot = 0; slot < GAME_NUM_ACTIVE_FEATURES; slot++) {
            if (features[slot] != previous->features[perspective][slot]) {
                removed[changed] = previous->features[perspective][slot];
                added[changed++] = features[slot];
            }
        }
        /* An update touches 2 rows per changed slot, a refresh one per active feature */
        if (2 * changed >= GAME_NUM_ACTIVE_FEATURES) {
            refresh_perspective(net, acc, perspective);
            continue;
        }
        float *values = acc->values[perspective];
        memcpy(values, previous->values[perspective], sizeof acc->values[perspective]);
        for (int i = 0; i < changed; i++) {
            subtract_row(values, net->feature_weights[removed[i]]);
            add_row(values, net->feature_weights[added[i]]);
        }
    }
}

static float clipped_relu(float x) { return x < 0.0f ? 0.0f : (x > 1.0f ? 1.0f : x); }

/* output = weight^T @ ClippedReLU(input) + bias, weight input-major: [input][output] */
static void linear(int inputs, int outputs, const float *weight, const float *bias, const float *input, float *output) {
    memcpy(output, bias, (size_t)outputs * sizeof *output);
    for (int i = 0; i < inputs; i++) {
        float x = clipped_relu(input[i]);
        if (x == 0.0f) continue; /* Sparse after ClippedReLU: many inputs are zero */
        const float *row = weight + (size_t)i * outputs;
        for (int j = 0; j < outputs; j++) output[j] += x * row[j];
    }
}

float nnue_evaluate(const Network *net, const Accumulator *acc, int side_to_move) {
    /* The side to move's accumulator first, so the network knows whose turn it is */
    float input[2 * ACCUMULATOR_SIZE];
    memcpy(input, acc->values[side_to_move], sizeof acc->values[0]);
    memcpy(input + ACCUMULATOR_SIZE, acc->values[1 - side_to_move], sizeof acc->values[0]);
    float hidden1[HIDDEN_SIZE], hidden2[HIDDEN_SIZE], output;
    linear(2 * ACCUMULATOR_SIZE, HIDDEN_SIZE, &net->l1_weight[0][0], net->l1_bias, input, hidden1);
    linear(HIDDEN_SIZE, HIDDEN_SIZE, &net->l2_weight[0][0], net->l2_bias, hidden1, hidden2);
    linear(HIDDEN_SIZE, 1, net->l3_weight, &net->l3_bias, hidden2, &output);
    return tanhf(output);
}
