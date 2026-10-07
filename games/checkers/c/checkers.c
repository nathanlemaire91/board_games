#include "checkers.h"

#include <string.h>

/* STEPS[square][direction]: the square next to `square` in `direction`, NO_SQUARE off the board.
 * Directions: up-left, up-right, down-left, down-right, as player 0 sees the board. */
#define ROW(s) ((s) / SQUARES_PER_ROW)
#define COLUMN(s) (2 * ((s) % SQUARES_PER_ROW) + ROW(s) % 2) /* 0-7 from player 0's left */
#define STEP(s, rows, columns)                                                                         \
    (ROW(s) + (rows) >= 0 && ROW(s) + (rows) < ROWS && COLUMN(s) + (columns) >= 0 &&                    \
             COLUMN(s) + (columns) < ROWS                                                               \
         ? (ROW(s) + (rows)) * SQUARES_PER_ROW + (COLUMN(s) + (columns)) / 2                            \
         : NO_SQUARE)
#define STEPS_FROM(s) {STEP(s, 1, -1), STEP(s, 1, 1), STEP(s, -1, -1), STEP(s, -1, 1)}
static const int8_t STEPS[NUM_SQUARES][NUM_DIRECTIONS] = {
    STEPS_FROM(0),  STEPS_FROM(1),  STEPS_FROM(2),  STEPS_FROM(3),  STEPS_FROM(4),  STEPS_FROM(5),
    STEPS_FROM(6),  STEPS_FROM(7),  STEPS_FROM(8),  STEPS_FROM(9),  STEPS_FROM(10), STEPS_FROM(11),
    STEPS_FROM(12), STEPS_FROM(13), STEPS_FROM(14), STEPS_FROM(15), STEPS_FROM(16), STEPS_FROM(17),
    STEPS_FROM(18), STEPS_FROM(19), STEPS_FROM(20), STEPS_FROM(21), STEPS_FROM(22), STEPS_FROM(23),
    STEPS_FROM(24), STEPS_FROM(25), STEPS_FROM(26), STEPS_FROM(27), STEPS_FROM(28), STEPS_FROM(29),
    STEPS_FROM(30), STEPS_FROM(31),
};

/* A piece moves in directions first_direction to end_direction - 1: men forward, kings both ways */
static int first_direction(int piece) { return is_king(piece) || owner(piece) == 0 ? 0 : 2; }
static int end_direction(int piece) { return is_king(piece) || owner(piece) == 1 ? 4 : 2; }

BoardState new_game(void) {
    BoardState state = {.jumping = NO_SQUARE};
    for (int square = 0; square < MAX_PIECES; square++) {
        state.board[square] = (uint8_t)make_piece(0, false);
        state.board[NUM_SQUARES - 1 - square] = (uint8_t)make_piece(1, false);
    }
    return state;
}

/* Writes the jumps of the piece on `square` to moves[count...], returns the new count */
static int add_jumps(const BoardState *state, int square, int moves[], int count) {
    int piece = state->board[square];
    for (int direction = first_direction(piece); direction < end_direction(piece); direction++) {
        int over = STEPS[square][direction];
        if (over == NO_SQUARE || state->board[over] == EMPTY || owner(state->board[over]) == owner(piece)) continue;
        int to = STEPS[over][direction];
        if (to != NO_SQUARE && state->board[to] == EMPTY) moves[count++] = square * NUM_DIRECTIONS + direction;
    }
    return count;
}

bool can_jump(const BoardState *state, int square) {
    int moves[NUM_DIRECTIONS];
    return add_jumps(state, square, moves, 0) > 0;
}

int get_possible_moves(const BoardState *state, int moves[MAX_LEGAL_MOVES]) {
    if (state->jumping != NO_SQUARE) return add_jumps(state, state->jumping, moves, 0);
    int count = 0;
    for (int square = 0; square < NUM_SQUARES; square++) {
        int piece = state->board[square];
        if (piece != EMPTY && owner(piece) == state->current_player) count = add_jumps(state, square, moves, count);
    }
    if (count) return count; /* Capturing is mandatory */
    for (int square = 0; square < NUM_SQUARES; square++) {
        int piece = state->board[square];
        if (piece == EMPTY || owner(piece) != state->current_player) continue;
        for (int direction = first_direction(piece); direction < end_direction(piece); direction++) {
            int to = STEPS[square][direction];
            if (to != NO_SQUARE && state->board[to] == EMPTY) moves[count++] = square * NUM_DIRECTIONS + direction;
        }
    }
    return count;
}

void make_move(BoardState *state, int move) {
    int origin = move / NUM_DIRECTIONS, direction = move % NUM_DIRECTIONS;
    int piece = state->board[origin];
    int to = STEPS[origin][direction];
    bool captured = state->board[to] != EMPTY; /* A legal move onto an occupied square is a jump over it */
    if (captured) {
        state->board[to] = EMPTY;
        to = STEPS[to][direction];
    }
    state->board[origin] = EMPTY;
    bool crowned = !is_king(piece) && ROW(to) == (owner(piece) == 0 ? ROWS - 1 : 0);
    state->board[to] = (uint8_t)(crowned ? make_piece(owner(piece), true) : piece);
    state->quiet_moves = captured || !is_king(piece) ? 0 : state->quiet_moves + 1;
    state->moves_played++;
    if (captured && !crowned && can_jump(state, to)) {
        state->jumping = (int8_t)to; /* The same player moves again */
    } else {
        state->jumping = NO_SQUARE;
        state->current_player = (uint8_t)opponent(state);
    }
}

bool draw_limit_reached(const BoardState *state) {
    return state->quiet_moves >= QUIET_LIMIT || state->moves_played >= MAX_PLIES;
}

static bool can_move(const BoardState *state) {
    int moves[MAX_LEGAL_MOVES];
    return get_possible_moves(state, moves) > 0;
}

bool is_over(const BoardState *state) { return draw_limit_reached(state) || !can_move(state); }

static int material(const BoardState *state, int player) {
    int total = 0;
    for (int square = 0; square < NUM_SQUARES; square++) {
        int piece = state->board[square];
        if (piece != EMPTY && owner(piece) == player) total += is_king(piece) ? 3 : 2;
    }
    return total;
}

int winner(const BoardState *state) {
    if (!can_move(state)) return opponent(state);
    if (draw_limit_reached(state)) return NO_WINNER;
    int lead = material(state, 0) - material(state, 1);
    return lead == 0 ? NO_WINNER : (lead > 0 ? 0 : 1);
}

int result(const BoardState *state, int player) {
    int w = winner(state);
    return w == NO_WINNER ? 0 : (w == player ? 1 : -1);
}

bool states_equal(const BoardState *a, const BoardState *b) {
    return memcmp(a->board, b->board, sizeof a->board) == 0 && a->jumping == b->jumping &&
           a->current_player == b->current_player && a->quiet_moves == b->quiet_moves &&
           a->moves_played == b->moves_played;
}
