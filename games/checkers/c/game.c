/* Checkers behind core/c/game_api.h: the rules of checkers.c, plus position keys, NNUE features and
 * the position format Python sends (games/checkers/__init__.py's encode). Exact results: endgame.c. */
#include <string.h>

#include "game_api.h"

GameState game_new(void) { return new_game(); }

int game_legal_moves(const GameState *state, int moves[GAME_MAX_MOVES]) {
    /* Without a move, the game is over too */
    return draw_limit_reached(state) ? 0 : get_possible_moves(state, moves);
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
    enum { PIECE_VALUES = 5 }; /* EMPTY and the four pieces */
    uint64_t key = 0;
    for (int square = 0; square < NUM_SQUARES; square++) {
        if (state->board[square] != EMPTY) key ^= zobrist(square * PIECE_VALUES + state->board[square]);
    }
    uint64_t offset = NUM_SQUARES * PIECE_VALUES;
    key ^= zobrist(offset + 1 + state->jumping); /* NO_SQUARE is -1 */
    offset += NUM_SQUARES + 1;
    key ^= zobrist(offset + state->current_player);
    offset += 2;
    key ^= zobrist(offset + state->quiet_moves);
    offset += QUIET_LIMIT + 1;
    key ^= zobrist(offset + state->moves_played);
    return key;
}

/* The square as `perspective` sees the board: turned around for player 1 */
static int seen_from(int square, int perspective) { return perspective == 0 ? square : NUM_SQUARES - 1 - square; }

void game_features(const GameState *state, int perspective, int16_t features[GAME_NUM_ACTIVE_FEATURES]) {
    for (int slot = 0; slot < NUM_SQUARES; slot++) {
        int piece = state->board[seen_from(slot, perspective)];
        int value = piece == EMPTY ? 0 : 1 + is_king(piece) + 2 * (owner(piece) != perspective);
        features[slot] = (int16_t)(slot * SQUARE_VALUES + value);
    }
    int jumping = state->jumping == NO_SQUARE ? 0 : 1 + seen_from(state->jumping, perspective);
    features[NUM_SQUARES] = (int16_t)(NUM_SQUARES * SQUARE_VALUES + jumping);
}

/* values: the 32 squares, the jumping square (-1 for none), the side to move, quiet moves, moves played.
 * Rejects positions the rules cannot reach in ways the engines rely on: more than MAX_PIECES pieces
 * a side (GAME_MAX_MOVES), a man on its crowning row, a jumping piece that is not the mover's or cannot jump. */
bool game_decode(const int32_t *values, int count, GameState *state) {
    if (count != NUM_SQUARES + 4) return false;
    int jumping = values[NUM_SQUARES], player = values[NUM_SQUARES + 1];
    int quiet_moves = values[NUM_SQUARES + 2], moves_played = values[NUM_SQUARES + 3];
    if (jumping < NO_SQUARE || jumping >= NUM_SQUARES || (player != 0 && player != 1) || quiet_moves < 0 ||
        quiet_moves > QUIET_LIMIT || moves_played < 0 || moves_played > MAX_PLIES) {
        return false;
    }
    memset(state, 0, sizeof *state);
    int pieces[2] = {0, 0};
    for (int square = 0; square < NUM_SQUARES; square++) {
        int piece = values[square];
        if (piece < EMPTY || piece > make_piece(1, true)) return false;
        if (piece != EMPTY) {
            int crowning_row = owner(piece) == 0 ? ROWS - 1 : 0;
            if (!is_king(piece) && square / SQUARES_PER_ROW == crowning_row) return false;
            if (++pieces[owner(piece)] > MAX_PIECES) return false;
        }
        state->board[square] = (uint8_t)piece;
    }
    state->jumping = (int8_t)jumping;
    state->current_player = (uint8_t)player;
    state->quiet_moves = (uint8_t)quiet_moves;
    state->moves_played = (uint16_t)moves_played;
    if (jumping != NO_SQUARE && (state->board[jumping] == EMPTY || owner(state->board[jumping]) != player ||
                                 !can_jump(state, jumping))) {
        return false;
    }
    return true;
}
