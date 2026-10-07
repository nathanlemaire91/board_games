/* Awale behind core/c/game_api.h: the rules of awale.c, plus position keys, NNUE features and
 * the position format Python sends (games/awale/__init__.py's encode). */
#include <string.h>

#include "game_api.h"

GameState game_new(void) { return new_game(); }

int game_legal_moves(const GameState *state, int moves[GAME_MAX_MOVES]) {
    return is_over(state) ? 0 : get_possible_moves(state, moves);
}

void game_play(GameState *state, int move) { make_move(state, move); }

int game_side_to_move(const GameState *state) { return state->current_player; }

bool game_is_over(const GameState *state) { return is_over(state); }

int game_winner(const GameState *state) { return winner(state); }

int game_plies_left(const GameState *state) { return moves_left(state); }

bool game_states_equal(const GameState *a, const GameState *b) { return states_equal(a, b); }

/* splitmix64: a fixed pseudo-random key for each (thing, value), with no table to set up */
static uint64_t zobrist(uint64_t index) {
    uint64_t z = (index + 1) * 0x9E3779B97F4A7C15ULL;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

uint64_t game_key(const GameState *state) {
    enum { SEED_VALUES = TOTAL_SEEDS + 1 };
    uint64_t key = 0;
    for (int hole = 0; hole < BOARD_SIZE; hole++) key ^= zobrist(hole * SEED_VALUES + state->board[hole]);
    uint64_t offset = BOARD_SIZE * SEED_VALUES;
    key ^= zobrist(offset + state->players_seeds[0]);
    key ^= zobrist(offset + SEED_VALUES + state->players_seeds[1]);
    offset += 2 * SEED_VALUES;
    key ^= zobrist(offset + state->current_player);
    key ^= zobrist(offset + 2 + state->moves_played);
    return key;
}

void game_features(const GameState *state, int perspective, int16_t features[GAME_NUM_ACTIVE_FEATURES]) {
    for (int slot = 0; slot < BOARD_SIZE; slot++) {
        int hole = (slot + perspective * HALF_BOARD_SIZE) % BOARD_SIZE;
        int seeds = state->board[hole] < MAX_HOLE_SEEDS ? state->board[hole] : MAX_HOLE_SEEDS;
        features[slot] = (int16_t)(slot * HOLE_BUCKETS + seeds);
    }
    int store_offset = BOARD_SIZE * HOLE_BUCKETS;
    int own = state->players_seeds[perspective], other = state->players_seeds[1 - perspective];
    features[BOARD_SIZE] = (int16_t)(store_offset + (own < MAX_STORE_SEEDS ? own : MAX_STORE_SEEDS));
    features[BOARD_SIZE + 1] =
        (int16_t)(store_offset + STORE_BUCKETS + (other < MAX_STORE_SEEDS ? other : MAX_STORE_SEEDS));
}

/* values: the 12 holes, player 0's and player 1's captured seeds, the side to move, moves played */
bool game_decode(const int32_t *values, int count, GameState *state) {
    if (count != BOARD_SIZE + 4) return false;
    int seeds = 0;
    for (int i = 0; i < BOARD_SIZE + 2; i++) {
        if (values[i] < 0 || values[i] > TOTAL_SEEDS) return false;
        seeds += values[i];
    }
    int player = values[BOARD_SIZE + 2], moves_played = values[BOARD_SIZE + 3];
    if (seeds > TOTAL_SEEDS || (player != 0 && player != 1) || moves_played < 0 || moves_played > MAX_MOVES) {
        return false;
    }
    memset(state, 0, sizeof *state);
    for (int hole = 0; hole < BOARD_SIZE; hole++) state->board[hole] = (uint8_t)values[hole];
    state->players_seeds[0] = (uint8_t)values[BOARD_SIZE];
    state->players_seeds[1] = (uint8_t)values[BOARD_SIZE + 1];
    state->current_player = (uint8_t)player;
    state->moves_played = (uint8_t)moves_played;
    return true;
}
