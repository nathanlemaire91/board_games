/* Command line front end for the C MCTS.
 *
 *     ./mcts [-i iterations] [-t seconds] [-s seed] [--fresh] [--vs-random games]
 *
 * By default, searches the opening position and prints each move's visits and win
 * rate, like `python mcts.py`. With --vs-random, plays that many games against a
 * uniformly random player, alternating colours, and prints MCTS's score.
 * -i 0 with -t means no iteration cap.
 */
#define _POSIX_C_SOURCE 199309L

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "awale.h"
#include "mcts.h"

static double now(void) {
    struct timespec time;
    clock_gettime(CLOCK_MONOTONIC, &time);
    return time.tv_sec + time.tv_nsec * 1e-9;
}

static void usage(const char *program) {
    fprintf(stderr, "usage: %s [-i iterations] [-t seconds] [-s seed] [--fresh] [--vs-random games]\n", program);
    exit(EXIT_FAILURE);
}

static void print_root_stats(MCTS *mcts) {
    BoardState state = new_game();
    double start = now();
    int32_t root_index = mcts_search(mcts, &state); /* Before reading mcts->nodes: the search may move the pool */
    const Node *root = &mcts->nodes[root_index];
    double elapsed = now() - start;
    for (int move = 0; move < BOARD_SIZE; move++) {
        for (int i = 0; i < root->num_children; i++) {
            const Node *child = &mcts->nodes[root->children[i]];
            if (child->move == move) {
                printf("move %d: visits=%u win rate=%.2f\n", move, child->visits, child->wins / child->visits);
            }
        }
    }
    printf("%u iterations in %.3f s (%.0f/s)\n", root->visits, elapsed, root->visits / elapsed);
}

/* MCTS's score (1, 0.5 or 0) in one game against a random player, MCTS playing `side` */
static double play_vs_random(MCTS *mcts, int side, uint64_t *rng, double *thinking, long *moves) {
    BoardState state = new_game();
    int legal[HALF_BOARD_SIZE];
    while (!is_over(&state)) {
        int move;
        if (state.current_player == side) {
            double start = now();
            move = mcts_best_move(mcts, &state);
            *thinking += now() - start;
            ++*moves;
        } else {
            int count = get_possible_moves(&state, legal);
            *rng = *rng * 6364136223846793005ULL + 1442695040888963407ULL;
            move = legal[(int)(((*rng >> 33) * (uint64_t)count) >> 31)];
        }
        make_move(&state, move);
    }
    return (result(&state, side) + 1) / 2.0;
}

int main(int argc, char **argv) {
    long iterations = 1000, games = 0;
    double move_time = 0;
    uint64_t seed = (uint64_t)time(NULL);
    int fresh = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--fresh")) {
            fresh = 1;
        } else if (i + 1 < argc && !strcmp(argv[i], "-i")) {
            iterations = strtol(argv[++i], NULL, 10);
        } else if (i + 1 < argc && !strcmp(argv[i], "-t")) {
            move_time = strtod(argv[++i], NULL);
        } else if (i + 1 < argc && !strcmp(argv[i], "-s")) {
            seed = strtoull(argv[++i], NULL, 10);
        } else if (i + 1 < argc && !strcmp(argv[i], "--vs-random")) {
            games = strtol(argv[++i], NULL, 10);
        } else {
            usage(argv[0]);
        }
    }
    if (iterations < 0 || move_time < 0 || games < 0) usage(argv[0]);
    if (iterations == 0 && move_time <= 0) {
        fprintf(stderr, "-i 0 (no iteration cap) needs a time limit (-t)\n");
        return EXIT_FAILURE;
    }

    MCTS mcts;
    mcts_init(&mcts, seed);
    mcts.iterations = iterations ? iterations : MCTS_NO_LIMIT;
    mcts.move_time = move_time;
    mcts.reuse = !fresh;

    if (games == 0) {
        print_root_stats(&mcts);
    } else {
        uint64_t rng = seed ^ 0xD1B54A32D192ED03ULL;
        double score = 0, thinking = 0;
        long moves = 0;
        for (long game = 0; game < games; game++) {
            score += play_vs_random(&mcts, (int)(game % 2), &rng, &thinking, &moves);
        }
        printf("mcts vs random: %g/%ld (%.1f%%), %.2f ms/move\n", score, games, 100 * score / games,
               1000 * thinking / moves);
    }
    mcts_free(&mcts);
    return EXIT_SUCCESS;
}
