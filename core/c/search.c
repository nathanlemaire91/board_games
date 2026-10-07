#define _POSIX_C_SOURCE 199309L

#include "search.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MAX_EVALUATION 0.999f  /* Keeps network scores strictly below proven results, which are exactly +-1 */
#define CLOCK_CHECK_NODES 1024 /* Nodes searched between two looks at the clock, a power of two */

static double now(void) {
    struct timespec time;
    clock_gettime(CLOCK_MONOTONIC, &time);
    return time.tv_sec + time.tv_nsec * 1e-9;
}

/* ---------- Transposition table ---------- */

static Transposition *probe(AlphaBeta *ab, uint64_t key) {
    Transposition *bucket = &ab->table[key & ab->table_mask & ~1ULL];
    if (bucket[0].key == key) return &bucket[0];
    if (bucket[1].key == key) return &bucket[1];
    return NULL;
}

static void store(AlphaBeta *ab, uint64_t key, int depth, float score, int bound, int best_move) {
    Transposition *bucket = &ab->table[key & ab->table_mask & ~1ULL];
    Transposition *slot;
    if (bucket[1].key == key) {
        slot = &bucket[1];
    } else if (bucket[0].key == key || depth >= bucket[0].depth || bucket[0].generation != ab->generation) {
        slot = &bucket[0]; /* The deepest entry of the current search */
    } else {
        slot = &bucket[1]; /* Always replaced */
    }
    *slot = (Transposition){key, score, (int8_t)depth, (uint8_t)bound, (int8_t)best_move, ab->generation};
}

/* ---------- Search ---------- */

AlphaBeta *alpha_beta_new(Network *net, int depth, double move_time, int table_bits) {
    if (table_bits < 1 || table_bits > 30) return NULL;
    AlphaBeta *ab = calloc(1, sizeof *ab);
    if (!ab) return NULL;
    ab->table = calloc((size_t)1 << table_bits, sizeof *ab->table);
    if (!ab->table) {
        free(ab);
        return NULL;
    }
    ab->table_mask = ((uint64_t)1 << table_bits) - 1;
    ab->net = net;
    ab->depth = depth >= 1 && depth < MAX_DEPTH ? depth : MAX_DEPTH;
    ab->move_time = move_time;
    return ab;
}

void alpha_beta_delete(AlphaBeta *ab) {
    if (!ab) return;
    free(ab->table);
    free(ab->net);
    free(ab->oracle);
    free(ab);
}

void alpha_beta_set_oracle(AlphaBeta *ab, void *oracle) {
    free(ab->oracle);
    ab->oracle = oracle;
}

/* The NNUE score of plies[ply], computing the missing accumulators from the nearest ancestor that has one */
static float evaluate(AlphaBeta *ab, int ply) {
    int computed = ply;
    while (!ab->plies[computed].acc_computed) computed--; /* The root always has one */
    for (int next = computed + 1; next <= ply; next++) {
        Ply *p = &ab->plies[next];
        accumulator_update(ab->net, &p->acc, &ab->plies[next - 1].acc, &p->state);
        p->acc_computed = true;
    }
    float score = nnue_evaluate(ab->net, &ab->plies[ply].acc, game_side_to_move(&ab->plies[ply].state));
    return fmaxf(-MAX_EVALUATION, fminf(MAX_EVALUATION, score));
}

/* Legal moves of plies[ply], `first` (a previous search's best move) leading if legal */
static int ordered_moves(const GameState *state, int first, int moves[GAME_MAX_MOVES]) {
    int count = game_legal_moves(state, moves);
    for (int i = 1; i < count; i++) {
        if (moves[i] == first) {
            memmove(&moves[1], &moves[0], (size_t)i * sizeof *moves);
            moves[0] = first;
            break;
        }
    }
    return count;
}

/* plies[ply + 1] becomes the position after `move` */
static void play_child(AlphaBeta *ab, int ply, int move) {
    Ply *child = &ab->plies[ply + 1];
    child->state = ab->plies[ply].state;
    game_play(&child->state, move);
    child->acc_computed = false;
}

static float negamax(AlphaBeta *ab, int ply, int depth, float alpha, float beta);

/* The score of plies[ply + 1], reached by a move, for the side to move in plies[ply] (window
 * alpha, beta for them too). When they move again, the turn goes on: same depth, same sign. */
static float child_score(AlphaBeta *ab, int ply, int depth, float alpha, float beta) {
    if (game_side_to_move(&ab->plies[ply + 1].state) == game_side_to_move(&ab->plies[ply].state)) {
        return negamax(ab, ply + 1, depth, alpha, beta);
    }
    return -negamax(ab, ply + 1, depth - 1, -beta, -alpha);
}

static float negamax(AlphaBeta *ab, int ply, int depth, float alpha, float beta) {
    ab->nodes++;
    if ((ab->nodes & (CLOCK_CHECK_NODES - 1)) == 0 && now() > ab->deadline) ab->stopped = true;
    if (ab->stopped) return 0.0f; /* Unwinding: the score is dropped */

    const GameState *state = &ab->plies[ply].state;
    if (game_is_over(state)) return (float)game_result(state, game_side_to_move(state));
    int exact;
    if (ab->oracle && game_exact_result(ab->oracle, state, &exact)) return (float)exact;
    if (depth == 0) return evaluate(ab, ply);

    uint64_t key = game_key(state);
    Transposition *entry = probe(ab, key);
    int first = NO_MOVE;
    if (entry) {
        first = entry->best_move;
        if (entry->depth >= depth) {
            if (entry->bound == EXACT) return entry->score;
            if (entry->bound == LOWER_BOUND) alpha = fmaxf(alpha, entry->score);
            else beta = fminf(beta, entry->score);
            if (alpha >= beta) return entry->score;
        }
    }

    float original_alpha = alpha, best_score = -INFINITY;
    int best_move = NO_MOVE, moves[GAME_MAX_MOVES];
    int count = ordered_moves(state, first, moves);
    for (int i = 0; i < count; i++) {
        play_child(ab, ply, moves[i]);
        float score = child_score(ab, ply, depth, alpha, beta);
        if (ab->stopped) return 0.0f;
        if (score > best_score) {
            best_score = score;
            best_move = moves[i];
        }
        alpha = fmaxf(alpha, score);
        if (alpha >= beta) break;
    }

    int bound = best_score <= original_alpha ? UPPER_BOUND : (best_score >= beta ? LOWER_BOUND : EXACT);
    store(ab, key, depth, best_score, bound, best_move);
    return best_score;
}

/* Searches the root to `depth`; *move stays unchanged if the time runs out meanwhile */
static void search_root(AlphaBeta *ab, int depth, float *score, int *move) {
    const GameState *state = &ab->plies[0].state;
    uint64_t key = game_key(state);
    Transposition *entry = probe(ab, key);
    int moves[GAME_MAX_MOVES];
    int count = ordered_moves(state, entry ? entry->best_move : NO_MOVE, moves);
    float alpha = -INFINITY;
    int best_move = NO_MOVE;
    for (int i = 0; i < count; i++) {
        play_child(ab, 0, moves[i]);
        float move_score = child_score(ab, 0, depth, alpha, INFINITY);
        if (ab->stopped) return; /* The partial iteration is dropped */
        if (move_score > alpha) {
            alpha = move_score;
            best_move = moves[i];
        }
    }
    store(ab, key, depth, alpha, EXACT, best_move);
    *score = alpha;
    *move = best_move;
}

int alpha_beta_best_move(AlphaBeta *ab, const GameState *state) {
    ab->generation++;
    ab->nodes = 0;
    ab->stopped = false;
    Ply *root = &ab->plies[0];
    root->state = *state;
    accumulator_refresh(ab->net, &root->acc, state);
    root->acc_computed = true;

    /* Depth 1 always completes, so there is a move to play however short the time */
    ab->deadline = INFINITY;
    float score = 0.0f;
    int move = NO_MOVE;
    search_root(ab, 1, &score, &move);
    ab->completed_depth = 1;
    if (ab->move_time > 0) ab->deadline = now() + ab->move_time;
    /* No search goes deeper than the moves left in a game */
    int max_depth = ab->depth < game_plies_left(state) ? ab->depth : game_plies_left(state);
    int moves[GAME_MAX_MOVES];
    /* With a time limit, a forced move is played at once rather than searched for the whole time */
    if (ab->move_time > 0 && game_legal_moves(state, moves) == 1) max_depth = 1;
    for (int depth = 2; depth <= max_depth; depth++) {
        if (fabsf(score) == 1.0f) break; /* Proven win or loss: searching deeper cannot change it */
        search_root(ab, depth, &score, &move);
        if (ab->stopped) break; /* Nodes finished before the deadline stay in the table */
        ab->completed_depth = depth;
    }
    ab->score = score;
    return move;
}
