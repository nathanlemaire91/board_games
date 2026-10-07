/* Checkers' endgame table (games/checkers/endgame.py): the exact result of every position with up to max_pieces
 * pieces, built here by retrograde analysis and read by the alpha-beta as its exact results.
 *
 * Positions are seen from the side to move, whose ("own") pieces move up the board, turned around when player 1 is to
 * move. Only positions at the start of a turn are stored: in the middle of a multiple jump, the search plays it out.
 * Results follow the draw rules, with two bytes per position:
 *
 * - values: the result depends on r, the quiet moves left before the QUIET_LIMIT draw (quiet moves are king steps,
 *   every other move resets the count). With more of them, a win or a loss stays one, so each position is decided from
 *   some r on: +t if the side to move wins whenever r >= t, -t if they lose whenever r >= t, 0 if it is a draw for
 *   any r.
 * - lengths: for a win or a loss, a bound on the moves (each jump counting one) in which the winner forces it,
 *   whatever the loser does and whatever r >= t. The game also stops at MAX_PLIES, which turns into a draw any win
 *   coming later: the table claims a win or a loss only with that many moves left. A draw stays one either way.
 *   LENGTH_UNKNOWN: too long to count.
 *
 * Layout: entries are grouped by material (own men, own kings, opponent's men, opponent's kings), in lexicographic
 * order of their counts (make_layout). Within a material, by the sets of squares of the own men, the opponent's men,
 * the own kings and the opponent's kings, each ranked as a combination (rank_set): the men among the 28 squares they
 * can stand on, the kings among the squares left. Entries whose own and opponent's men overlap are unused.
 *
 * Retrograde analysis: positions are solved by quiet class, the positions with the same men, both sides to move (two
 * blocks of entries, that king steps turn into each other). A class only depends on positions with fewer pieces
 * (after a capture) or fewer rows left for the men to cross (after a man move), solved before it. Within a class, the
 * results for r = 1, 2, ... follow from those for r - 1, until they stop changing.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "game_api.h"
#include "search.h"

#define MAX_TABLE_PIECES 5
#define MAX_MATERIALS 128         /* 85 materials of up to 5 pieces */
#define LENGTH_UNKNOWN 255
#define UNSOLVED INT8_MIN         /* A value not computed yet, while building */
#define NO_EXIT (-2)              /* exit_value of a position without capture or man move */
#define MEN_SQUARES 28
#define OWN_MEN_AREA 0x0FFFFFFFu  /* Squares 0-27: own men never stand on the far row */
#define OPPONENT_MEN_AREA 0xFFFFFFF0u
#define MAX_SUCCESSORS 1024

enum { OWN_MEN, OWN_KINGS, OPPONENT_MEN, OPPONENT_KINGS };

typedef struct {
    int pieces[4];      /* Own men, own kings, opponent's men, opponent's kings */
    int64_t offset;     /* Its first entry */
    int64_t kings_size; /* Entries per placement of the men: the placements of the kings */
    int64_t size;
} Material;

typedef struct {
    int max_pieces, count;
    int64_t size;
    Material materials[MAX_MATERIALS];
    int8_t ids[MAX_TABLE_PIECES + 1][MAX_TABLE_PIECES + 1][MAX_TABLE_PIECES + 1][MAX_TABLE_PIECES + 1]; /* -1: none */
} Layout;

typedef struct {
    Layout layout;
    const int8_t *values; /* Not owned: Python keeps the arrays alive */
    const uint8_t *lengths;
} EndgameTable;

/* ---------- Positions and their entries ---------- */

static int64_t CHOOSE[NUM_SQUARES + 1][MAX_TABLE_PIECES + 1];

static void init_choose(void) {
    if (CHOOSE[0][0]) return;
    for (int n = 0; n <= NUM_SQUARES; n++) {
        CHOOSE[n][0] = 1;
        for (int k = 1; k <= MAX_TABLE_PIECES; k++) CHOOSE[n][k] = n ? CHOOSE[n - 1][k - 1] + CHOOSE[n - 1][k] : 0;
    }
}

static int total_pieces(const int pieces[4]) { return pieces[0] + pieces[1] + pieces[2] + pieces[3]; }

/* The materials of up to max_pieces pieces, each side having some */
static bool make_layout(int max_pieces, Layout *layout) {
    if (max_pieces < 2 || max_pieces > MAX_TABLE_PIECES) return false;
    init_choose();
    memset(layout, 0, sizeof *layout);
    memset(layout->ids, -1, sizeof layout->ids);
    layout->max_pieces = max_pieces;
    int c[4];
    for (c[0] = 0; c[0] <= max_pieces; c[0]++) {
        for (c[1] = 0; c[1] <= max_pieces; c[1]++) {
            for (c[2] = 0; c[2] <= max_pieces; c[2]++) {
                for (c[3] = 0; c[3] <= max_pieces; c[3]++) {
                    if (c[0] + c[1] == 0 || c[2] + c[3] == 0 || total_pieces(c) > max_pieces) continue;
                    Material *material = &layout->materials[layout->count];
                    memcpy(material->pieces, c, sizeof c);
                    int free = NUM_SQUARES - c[0] - c[2];
                    material->kings_size = CHOOSE[free][c[1]] * CHOOSE[free - c[1]][c[3]];
                    material->size = CHOOSE[MEN_SQUARES][c[0]] * CHOOSE[MEN_SQUARES][c[2]] * material->kings_size;
                    material->offset = layout->size;
                    layout->size += material->size;
                    layout->ids[c[0]][c[1]][c[2]][c[3]] = (int8_t)layout->count++;
                }
            }
        }
    }
    return true;
}

/* The rank of `set` among the sets of as many squares of `area`: a combination (colex) of their places in `area` */
static int64_t rank_set(uint32_t set, uint32_t area) {
    int64_t rank = 0;
    for (int i = 1; set; i++, set &= set - 1) {
        uint32_t below = (set & -set) - 1; /* The squares below its lowest one */
        rank += CHOOSE[__builtin_popcount(area & below)][i];
    }
    return rank;
}

static uint32_t nth_square(uint32_t area, int n) {
    while (n--) area &= area - 1;
    return area & -area;
}

/* The set of `count` squares of `area` of that rank (see rank_set) */
static uint32_t unrank_set(int64_t rank, int count, uint32_t area) {
    uint32_t set = 0;
    int limit = __builtin_popcount(area);
    for (int i = count; i >= 1; i--) {
        int place = i - 1;
        while (place + 1 < limit && CHOOSE[place + 1][i] <= rank) place++;
        rank -= CHOOSE[place][i];
        set |= nth_square(area, place);
        limit = place;
    }
    return set;
}

/* Square s as seen from the other side, 31 - s, for each square of the set */
static uint32_t turned(uint32_t set) {
    uint32_t result = 0;
    for (; set; set &= set - 1) result |= 1u << (NUM_SQUARES - 1 - __builtin_ctz(set));
    return result;
}

/* The pieces of `state` seen from its side to move, as sets of squares: own men, own kings, opponent's men and kings */
static void relative_sets(const BoardState *state, uint32_t sets[4]) {
    memset(sets, 0, 4 * sizeof *sets);
    for (int square = 0; square < NUM_SQUARES; square++) {
        int piece = state->board[square];
        if (piece == EMPTY) continue;
        int seen = state->current_player == 0 ? square : NUM_SQUARES - 1 - square;
        sets[(owner(piece) == state->current_player ? OWN_MEN : OPPONENT_MEN) + is_king(piece)] |= 1u << seen;
    }
}

/* The position of those pieces, player 0 (own) to move */
static BoardState state_from_sets(const uint32_t sets[4]) {
    BoardState state = {.jumping = NO_SQUARE};
    for (int kind = OWN_MEN; kind <= OPPONENT_KINGS; kind++) {
        for (uint32_t set = sets[kind]; set; set &= set - 1) {
            state.board[__builtin_ctz(set)] = (uint8_t)make_piece(kind >= OPPONENT_MEN, kind % 2);
        }
    }
    return state;
}

static int64_t men_rank(const Material *material, uint32_t own_men, uint32_t opponent_men) {
    return rank_set(own_men, OWN_MEN_AREA) * CHOOSE[MEN_SQUARES][material->pieces[OPPONENT_MEN]] +
           rank_set(opponent_men, OPPONENT_MEN_AREA);
}

/* The entry of a position (its sets of pieces), -1 if the table does not hold its material */
static int64_t table_index(const Layout *layout, const uint32_t sets[4]) {
    int c[4];
    for (int kind = 0; kind < 4; kind++) c[kind] = __builtin_popcount(sets[kind]);
    if (total_pieces(c) > layout->max_pieces) return -1;
    int id = layout->ids[c[0]][c[1]][c[2]][c[3]];
    if (id < 0) return -1; /* A side without pieces */
    const Material *material = &layout->materials[id];
    uint32_t men = sets[OWN_MEN] | sets[OPPONENT_MEN];
    int64_t kings_rank = rank_set(sets[OWN_KINGS], ~men) * CHOOSE[NUM_SQUARES - c[0] - c[2] - c[1]][c[3]] +
                         rank_set(sets[OPPONENT_KINGS], ~(men | sets[OWN_KINGS]));
    return material->offset + men_rank(material, sets[OWN_MEN], sets[OPPONENT_MEN]) * material->kings_size +
           kings_rank;
}

/* ---------- Results ---------- */

bool game_exact_result(const void *oracle, const GameState *state, int *result) {
    const EndgameTable *table = oracle;
    if (state->jumping != NO_SQUARE) return false; /* In the middle of a multiple jump: the search plays it out */
    uint32_t sets[4];
    relative_sets(state, sets);
    int64_t index = table_index(&table->layout, sets);
    if (index < 0) return false;
    int value = table->values[index], length = table->lengths[index];
    if (value == 0 || abs(value) > QUIET_LIMIT - state->quiet_moves) {
        *result = 0; /* A draw, which the game stopping at MAX_PLIES leaves a draw */
        return true;
    }
    if (length == LENGTH_UNKNOWN || length > moves_left(state)) return false; /* The game may stop before */
    *result = value > 0 ? 1 : -1;
    return true;
}

/* ---------- Retrograde analysis ---------- */

/* A position at the end of a turn, seen from the opponent who moves next */
typedef struct {
    uint32_t sets[4];
    int moves; /* The moves of the turn: a multiple jump takes several */
    bool quiet; /* A king step */
} Successor;

/* Appends the positions at the end of each turn the side to move can play (each way of playing a multiple jump) */
static int add_successors(const BoardState *state, int moves_before, Successor successors[MAX_SUCCESSORS], int count) {
    int moves[MAX_LEGAL_MOVES];
    int legal = get_possible_moves(state, moves);
    for (int i = 0; i < legal; i++) {
        BoardState child = *state;
        make_move(&child, moves[i]);
        if (child.current_player == state->current_player) { /* The multiple jump goes on */
            count = add_successors(&child, moves_before + 1, successors, count);
        } else if (count < MAX_SUCCESSORS) {
            relative_sets(&child, successors[count].sets);
            successors[count].moves = moves_before + 1;
            successors[count].quiet = child.quiet_moves > state->quiet_moves; /* Neither a capture nor a man move */
            count++;
        } else {
            fprintf(stderr, "endgame: more than %d successors\n", MAX_SUCCESSORS);
            exit(EXIT_FAILURE);
        }
    }
    return count;
}

static int add_length(int moves, int length) {
    return length >= LENGTH_UNKNOWN - moves ? LENGTH_UNKNOWN : moves + length;
}

/* The entries of a material with a placement of its men: its kings placed every way */
typedef struct {
    const Material *material;
    uint32_t own_men, opponent_men;
    int64_t base; /* Its first entry */
} Block;

static Block make_block(const Layout *layout, int material_id, uint32_t own_men, uint32_t opponent_men) {
    const Material *material = &layout->materials[material_id];
    int64_t base = material->offset + men_rank(material, own_men, opponent_men) * material->kings_size;
    return (Block){material, own_men, opponent_men, base};
}

/* The position of the block's entry base + local */
static void block_position(const Block *block, int64_t local, uint32_t sets[4]) {
    const int *c = block->material->pieces;
    uint32_t men = block->own_men | block->opponent_men;
    int64_t opponent_kings = CHOOSE[NUM_SQUARES - c[0] - c[2] - c[1]][c[3]];
    sets[OWN_MEN] = block->own_men;
    sets[OPPONENT_MEN] = block->opponent_men;
    sets[OWN_KINGS] = unrank_set(local / opponent_kings, c[OWN_KINGS], ~men);
    sets[OPPONENT_KINGS] = unrank_set(local % opponent_kings, c[OPPONENT_KINGS], ~(men | sets[OWN_KINGS]));
}

typedef struct {
    Layout layout;
    int8_t *values;
    uint8_t *lengths;
    bool failed;
} Builder;

/* The value (at r = QUIET_LIMIT, as after a capture or a man move) and length of a solved position, seen from its
 * side to move */
static void solved_entry(Builder *builder, const uint32_t sets[4], int *value, int *length) {
    if (!(sets[OWN_MEN] | sets[OWN_KINGS])) { /* No pieces left: lost */
        *value = -1;
        *length = 0;
        return;
    }
    int64_t index = table_index(&builder->layout, sets);
    *value = builder->values[index];
    *length = builder->lengths[index];
    if (*value == UNSOLVED) builder->failed = true; /* Classes solved out of order: a bug */
}

/* Solves a quiet class: blocks[0] and its partner blocks[1], the same men seen from the other side (one block when
 * they are the same). Returns false if out of memory. */
static bool solve_class(Builder *builder, const Block blocks[2], int num_blocks) {
    int64_t block_size = blocks[0].material->kings_size, n = block_size * num_blocks;
    int kings = 0; /* Each king steps at most 4 ways */
    for (int side = 0; side < num_blocks; side++) kings += blocks[side].material->pieces[OWN_KINGS];
    int64_t *child_start = malloc((size_t)(n + 1) * sizeof *child_start);
    int32_t *children = malloc((size_t)(block_size * NUM_DIRECTIONS * kings + 1) * sizeof *children);
    int8_t *exit_value = malloc((size_t)n), *previous = malloc((size_t)n), *current = malloc((size_t)n);
    int8_t *threshold = calloc((size_t)n, 1);
    uint8_t *exit_win = malloc((size_t)n), *exit_loss = malloc((size_t)n), *length = calloc((size_t)n, 1);
    Successor *successors = malloc(MAX_SUCCESSORS * sizeof *successors);
    bool ok = child_start && children && exit_value && previous && current && threshold && exit_win && exit_loss &&
              length && successors;

    /* The turns from each position: king steps stay in the class, captures and man moves leave it */
    child_start[0] = 0;
    for (int64_t p = 0; ok && p < n; p++) {
        int side = (int)(p / block_size);
        const Block *partner = &blocks[num_blocks - 1 - side];
        uint32_t sets[4];
        block_position(&blocks[side], p % block_size, sets);
        BoardState state = state_from_sets(sets);
        int count = add_successors(&state, 0, successors, 0);
        int64_t next = child_start[p];
        exit_value[p] = NO_EXIT;
        exit_win[p] = LENGTH_UNKNOWN;
        exit_loss[p] = 0;
        for (int i = 0; i < count; i++) {
            if (successors[i].quiet) {
                int64_t local = table_index(&builder->layout, successors[i].sets) - partner->base;
                if (local < 0 || local >= block_size) { /* A king step leaving the class: a bug */
                    builder->failed = true;
                    continue;
                }
                children[next++] = (int32_t)(local + (num_blocks - 1 - side) * block_size);
                continue;
            }
            int value, child_length;
            solved_entry(builder, successors[i].sets, &value, &child_length);
            int ours = value > 0 ? -1 : (value < 0 ? 1 : 0); /* After a capture or a man move, r is QUIET_LIMIT */
            int total = add_length(successors[i].moves, child_length);
            if (ours > exit_value[p]) exit_value[p] = (int8_t)ours;
            if (ours == 1 && total < exit_win[p]) exit_win[p] = (uint8_t)total;
            if (ours == -1 && total > exit_loss[p]) exit_loss[p] = (uint8_t)total;
        }
        child_start[p + 1] = next;
        /* At r = 0 the game is drawn, or lost without a move, which is so for any r */
        previous[p] = count ? 0 : -1;
        threshold[p] = count ? 0 : -1;
    }

    /* Results for r = 1, 2, ...: a position gets decided once a move wins (an exit, or a king step to a position lost
     * at r - 1), or once every move loses */
    for (int r = 1; ok && r <= QUIET_LIMIT; r++) {
        bool changed = false;
        for (int64_t p = 0; p < n; p++) {
            current[p] = previous[p];
            if (threshold[p]) continue; /* Decided, for good */
            int best = exit_value[p];
            for (int64_t c = child_start[p]; c < child_start[p + 1]; c++) {
                if (-previous[children[c]] > best) best = -previous[children[c]];
            }
            if (best == 0) continue;
            changed = true;
            current[p] = (int8_t)best;
            threshold[p] = (int8_t)(best * r);
            int bound = best > 0 ? (exit_value[p] == 1 ? exit_win[p] : LENGTH_UNKNOWN) : exit_loss[p];
            for (int64_t c = child_start[p]; c < child_start[p + 1]; c++) {
                int child = children[c], total = add_length(1, length[child]);
                if (best > 0 && previous[child] == -1 && total < bound) bound = total; /* The fastest win */
                if (best < 0 && total > bound) bound = total;                           /* The longest loss */
            }
            length[p] = (uint8_t)bound;
        }
        int8_t *swap = previous;
        previous = current;
        current = swap;
        if (!changed) break;
    }

    for (int64_t p = 0; ok && p < n; p++) {
        int64_t entry = blocks[p / block_size].base + p % block_size;
        builder->values[entry] = threshold[p];
        builder->lengths[entry] = threshold[p] ? length[p] : 0;
    }
    free(child_start);
    free(children);
    free(exit_value);
    free(previous);
    free(current);
    free(threshold);
    free(exit_win);
    free(exit_loss);
    free(length);
    free(successors);
    return ok;
}

/* Rows the men have left to cross: every man move lowers it */
static int rows_left(uint32_t own_men, uint32_t opponent_men) {
    int rows = 0;
    for (; own_men; own_men &= own_men - 1) rows += ROWS - 1 - __builtin_ctz(own_men) / SQUARES_PER_ROW;
    for (; opponent_men; opponent_men &= opponent_men - 1) rows += __builtin_ctz(opponent_men) / SQUARES_PER_ROW;
    return rows;
}

/* Solves every position of exactly `pieces` pieces, those with fewer being solved */
static bool build_pieces(Builder *builder, int pieces) {
    Layout *layout = &builder->layout;
    uint8_t *done[MAX_MATERIALS] = {0};
    bool ok = true;
    for (int id = 0; id < layout->count; id++) {
        const Material *material = &layout->materials[id];
        if (total_pieces(material->pieces) != pieces) continue;
        memset(builder->values + material->offset, UNSOLVED, (size_t)material->size);
        memset(builder->lengths + material->offset, 0, (size_t)material->size);
        done[id] = calloc((size_t)(material->size / material->kings_size), 1);
        ok = ok && done[id];
    }
    /* Men moves lower the rows left, so classes are solved in that order */
    for (int rows = 0; ok && rows <= (ROWS - 1) * pieces; rows++) {
        for (int id = 0; ok && id < layout->count; id++) {
            const Material *material = &layout->materials[id];
            if (!done[id]) continue;
            const int *c = material->pieces;
            int64_t opponent_placements = CHOOSE[MEN_SQUARES][c[OPPONENT_MEN]];
            for (int64_t placement = 0; ok && placement < material->size / material->kings_size; placement++) {
                if (done[id][placement]) continue;
                uint32_t own_men = unrank_set(placement / opponent_placements, c[OWN_MEN], OWN_MEN_AREA);
                uint32_t opponent_men = unrank_set(placement % opponent_placements, c[OPPONENT_MEN], OPPONENT_MEN_AREA);
                if (own_men & opponent_men) { /* Unused entries */
                    int64_t base = material->offset + placement * material->kings_size;
                    memset(builder->values + base, 0, (size_t)material->kings_size);
                    done[id][placement] = 1;
                    continue;
                }
                if (rows_left(own_men, opponent_men) != rows) continue;
                int partner_id = layout->ids[c[OPPONENT_MEN]][c[OPPONENT_KINGS]][c[OWN_MEN]][c[OWN_KINGS]];
                Block blocks[2] = {make_block(layout, id, own_men, opponent_men),
                                   make_block(layout, partner_id, turned(opponent_men), turned(own_men))};
                int num_blocks = blocks[1].base == blocks[0].base ? 1 : 2;
                done[id][placement] = 1;
                done[partner_id][(blocks[1].base - blocks[1].material->offset) / blocks[1].material->kings_size] = 1;
                ok = solve_class(builder, blocks, num_blocks) && !builder->failed;
            }
        }
    }
    for (int id = 0; id < layout->count; id++) free(done[id]);
    return ok;
}

/* ---------- Entry points for Python ---------- */

/* Entries of the table of up to max_pieces pieces, -1 if max_pieces is out of range */
int64_t checkers_endgame_size(int max_pieces) {
    Layout layout;
    return make_layout(max_pieces, &layout) ? layout.size : -1;
}

/* Solves the positions of exactly `pieces` pieces into the table of up to max_pieces (values and lengths of
 * checkers_endgame_size entries), those with fewer pieces being solved. Returns 0 if out of memory. */
int checkers_endgame_build(int max_pieces, int pieces, int8_t *values, uint8_t *lengths) {
    Builder *builder = malloc(sizeof *builder);
    if (!builder || !make_layout(max_pieces, &builder->layout) || pieces < 2 || pieces > max_pieces) {
        free(builder);
        return 0;
    }
    builder->values = values;
    builder->lengths = lengths;
    builder->failed = false;
    bool ok = build_pieces(builder, pieces);
    if (builder->failed) fprintf(stderr, "endgame: a class read an unsolved position\n");
    free(builder);
    return ok;
}

/* Gives `ab` the table (values and lengths of `size` entries, which must outlive it). Returns 0 if `size` is not the
 * size of the table of up to max_pieces pieces, or out of memory. */
int checkers_set_endgame(AlphaBeta *ab, const int8_t *values, const uint8_t *lengths, int64_t size, int max_pieces) {
    EndgameTable *table = malloc(sizeof *table);
    if (!table || !make_layout(max_pieces, &table->layout) || table->layout.size != size) {
        free(table);
        return 0;
    }
    table->values = values;
    table->lengths = lengths;
    alpha_beta_set_oracle(ab, table);
    return 1;
}

/* The entry of a position (the ints of game_decode) in the table of up to max_pieces pieces: -1 if it does not hold
 * its material, -2 if the position does not decode */
int64_t checkers_endgame_index(const int32_t *position, int count, int max_pieces) {
    GameState state;
    Layout layout;
    if (!game_decode(position, count, &state) || !make_layout(max_pieces, &layout)) return -2;
    uint32_t sets[4];
    relative_sets(&state, sets);
    return table_index(&layout, sets);
}
