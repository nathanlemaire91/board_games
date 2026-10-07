#define _POSIX_C_SOURCE 199309L

#include "mcts.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define INITIAL_CAPACITY 4096

static double now(void) {
    struct timespec time;
    clock_gettime(CLOCK_MONOTONIC, &time);
    return time.tv_sec + time.tv_nsec * 1e-9;
}

/* xorshift64*: fast, and plenty for picking among a few dozen moves */
static uint64_t next_random(uint64_t *rng) {
    *rng ^= *rng >> 12;
    *rng ^= *rng << 25;
    *rng ^= *rng >> 27;
    return *rng * 0x2545F4914F6CDD1DULL;
}

static int random_below(uint64_t *rng, int n) {
    return (int)(((next_random(rng) >> 32) * (uint64_t)n) >> 32);
}

void mcts_init(MCTS *mcts, uint64_t seed) {
    *mcts = (MCTS){
        .iterations = 1000,
        .move_time = 0,
        .exploration = sqrt(2.0),
        .max_rollout_moves = 200,
        .reuse = true,
        .rng = seed ? seed : 0x9E3779B97F4A7C15ULL, /* xorshift must not start at zero */
        .root = NO_NODE,
    };
}

void mcts_free(MCTS *mcts) {
    free(mcts->nodes);
    mcts->nodes = NULL;
    mcts->num_nodes = mcts->capacity = 0;
    mcts->root = NO_NODE;
}

static void reserve(Node **nodes, int32_t *capacity, int32_t needed) {
    if (needed <= *capacity) return;
    int32_t new_capacity = *capacity ? *capacity : INITIAL_CAPACITY;
    while (new_capacity < needed) new_capacity *= 2;
    Node *grown = realloc(*nodes, (size_t)new_capacity * sizeof(Node));
    if (!grown) {
        fprintf(stderr, "mcts: out of memory for %d nodes\n", new_capacity);
        exit(EXIT_FAILURE);
    }
    *nodes = grown;
    *capacity = new_capacity;
}

/* Appends a node for `state` and returns its index. Invalidates Node pointers into the pool. */
static int32_t new_node(MCTS *mcts, const GameState *state, int32_t parent, int move) {
    /* The parent's side to move made the move, whoever moves next: players may move several
     * times in a row. A root's wins are never read. */
    int mover = parent == NO_NODE ? 1 - game_side_to_move(state) : game_side_to_move(&mcts->nodes[parent].state);
    reserve(&mcts->nodes, &mcts->capacity, mcts->num_nodes + 1);
    Node *node = &mcts->nodes[mcts->num_nodes];
    node->state = *state;
    node->parent = parent;
    node->num_children = 0;
    node->move = (int8_t)move;
    node->player_just_moved = (uint8_t)mover;
    node->visits = 0;
    node->wins = 0;
    node->num_untried = 0;
    if (!game_is_over(state)) {
        int moves[GAME_MAX_MOVES];
        node->num_untried = (uint8_t)game_legal_moves(state, moves);
        for (int i = 0; i < node->num_untried; i++) node->untried_moves[i] = (uint8_t)moves[i];
    }
    return mcts->num_nodes++;
}

static bool is_terminal(const Node *node) { return node->num_untried == 0 && node->num_children == 0; }

static int32_t best_child(const MCTS *mcts, const Node *node) {
    double log_visits = log(node->visits);
    int32_t best = node->children[0];
    double best_score = -INFINITY;
    for (int i = 0; i < node->num_children; i++) {
        const Node *child = &mcts->nodes[node->children[i]];
        double score = child->wins / child->visits + mcts->exploration * sqrt(log_visits / child->visits);
        if (score > best_score) { /* Strict: ties go to the first child, like Python's max */
            best_score = score;
            best = node->children[i];
        }
    }
    return best;
}

static int32_t expand(MCTS *mcts, int32_t index) {
    Node *node = &mcts->nodes[index];
    int slot = random_below(&mcts->rng, node->num_untried);
    int move = node->untried_moves[slot];
    node->untried_moves[slot] = node->untried_moves[--node->num_untried];
    GameState state = node->state;
    game_play(&state, move);
    int32_t child = new_node(mcts, &state, index, move);
    node = &mcts->nodes[index]; /* new_node may have moved the pool */
    node->children[node->num_children++] = child;
    return child;
}

static int32_t select_node(const MCTS *mcts, int32_t index) {
    while (mcts->nodes[index].num_untried == 0 && mcts->nodes[index].num_children > 0) {
        index = best_child(mcts, &mcts->nodes[index]);
    }
    return index;
}

static int rollout(MCTS *mcts, GameState state) {
    int moves[GAME_MAX_MOVES];
    for (int i = 0; i < mcts->max_rollout_moves; i++) {
        int count = game_legal_moves(&state, moves);
        if (!count) break;
        game_play(&state, moves[random_below(&mcts->rng, count)]);
    }
    /* A rollout cut short goes to the player ahead (game_winner of an unfinished game) */
    return game_winner(&state);
}

static void backpropagate(MCTS *mcts, int32_t index, int winner) {
    while (index != NO_NODE) {
        Node *node = &mcts->nodes[index];
        node->visits++;
        if (winner == GAME_NO_WINNER) {
            node->wins += 0.5;
        } else if (winner == node->player_just_moved) {
            node->wins += 1;
        }
        index = node->parent;
    }
}

/* The node for `state` among the previous root and the positions that follow it until the root's
 * side to move is to move again (our move, then the opponent's: with alternating turns, the
 * children and grandchildren), or NO_NODE */
static int32_t reusable_root(const MCTS *mcts, const GameState *state) {
    if (mcts->root == NO_NODE) return NO_NODE;
    int32_t *queue = malloc((size_t)mcts->num_nodes * sizeof *queue); /* Breadth first, each node once */
    if (!queue) return NO_NODE;
    int side = game_side_to_move(&mcts->nodes[mcts->root].state);
    int32_t found = NO_NODE, head = 0, tail = 0;
    queue[tail++] = mcts->root;
    while (head < tail && found == NO_NODE) {
        int32_t index = queue[head++];
        const Node *node = &mcts->nodes[index];
        if (game_states_equal(&node->state, state)) {
            found = index;
        } else if (index == mcts->root || game_side_to_move(&node->state) != side) {
            for (int c = 0; c < node->num_children; c++) queue[tail++] = node->children[c];
        }
    }
    free(queue);
    return found;
}

/* Copies the subtree under `index` into a new pool, so the rest of the old tree is freed */
static void keep_subtree(MCTS *mcts, int32_t index) {
    Node *kept = NULL;
    int32_t num_kept = 0, capacity = 0;
    reserve(&kept, &capacity, 1);
    kept[num_kept] = mcts->nodes[index];
    kept[num_kept++].parent = NO_NODE; /* Detach, so backpropagation stops here */
    /* Breadth first: kept[i] is copied, its children still point into the old pool until visited */
    for (int32_t i = 0; i < num_kept; i++) {
        reserve(&kept, &capacity, num_kept + kept[i].num_children);
        for (int c = 0; c < kept[i].num_children; c++) {
            kept[num_kept] = mcts->nodes[kept[i].children[c]];
            kept[num_kept].parent = i;
            kept[i].children[c] = num_kept++;
        }
    }
    free(mcts->nodes);
    mcts->nodes = kept;
    mcts->num_nodes = num_kept;
    mcts->capacity = capacity;
    mcts->root = 0;
}

int32_t mcts_search(MCTS *mcts, const GameState *state) {
    if (mcts->iterations == MCTS_NO_LIMIT && mcts->move_time <= 0) {
        fprintf(stderr, "mcts: unlimited iterations need a time limit\n");
        exit(EXIT_FAILURE);
    }
    int32_t reused = mcts->reuse ? reusable_root(mcts, state) : NO_NODE;
    if (reused != NO_NODE) {
        keep_subtree(mcts, reused);
    } else {
        mcts->num_nodes = 0;
        mcts->root = new_node(mcts, state, NO_NODE, -1);
    }

    double deadline = mcts->move_time > 0 ? now() + mcts->move_time : INFINITY;
    for (long iteration = 0;; iteration++) {
        /* At least one iteration, so the root has a child to play */
        if (iteration == mcts->iterations || (iteration && now() > deadline)) break;
        int32_t node = select_node(mcts, mcts->root);
        if (!is_terminal(&mcts->nodes[node])) node = expand(mcts, node);
        int winner = rollout(mcts, mcts->nodes[node].state);
        backpropagate(mcts, node, winner);
    }

    int32_t root = mcts->root;
    if (!mcts->reuse) mcts->root = NO_NODE;
    return root;
}

int mcts_best_move(MCTS *mcts, const GameState *state) {
    int32_t root_index = mcts_search(mcts, state); /* Before reading mcts->nodes: the search may move the pool */
    const Node *root = &mcts->nodes[root_index];
    if (root->num_children == 0) return -1; /* Game is already over */
    const Node *best = &mcts->nodes[root->children[0]];
    for (int i = 1; i < root->num_children; i++) {
        const Node *child = &mcts->nodes[root->children[i]];
        if (child->visits > best->visits) best = child;
    }
    return best->move;
}
