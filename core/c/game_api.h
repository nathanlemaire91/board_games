/* The interface between a game and the reusable engines of core/c (MCTS, alpha-beta, NNUE).
 *
 * A game lives in games/<name>/c and provides:
 *
 * - game.h, included below, which defines:
 *     GameState                 a plain struct, copied by value
 *     GAME_MAX_MOVES            the most legal moves a position can have
 *     GAME_MAX_PLIES            the longest game, in plies (bounds the search stack)
 *     GAME_NUM_FEATURES         NNUE input features in all
 *     GAME_NUM_ACTIVE_FEATURES  NNUE features active in every position, one per slot
 *
 * - the functions declared below, compiled with the engines into lib<name>.so (see
 *   games/awale/c/Makefile). Moves are ints from 0 to 127 (stored in a byte), players 0 and 1.
 *
 * Players usually take turns, but a player may move several times in a row, such as the jumps of
 * a multiple jump in checkers: the engines compare game_side_to_move before and after each move
 * and never assume it alternates. Each run of moves by one player has to be finite.
 */
#ifndef GAME_API_H
#define GAME_API_H

#include <stdbool.h>
#include <stdint.h>

#include "game.h"

#define GAME_NO_WINNER (-1)

/* ---------- Rules ---------- */

GameState game_new(void);

/* Writes the legal moves to `moves`, returns how many (0 once the game is over) */
int game_legal_moves(const GameState *state, int moves[GAME_MAX_MOVES]);

void game_play(GameState *state, int move);

int game_side_to_move(const GameState *state);

bool game_is_over(const GameState *state);

/* The winner of a finished game, GAME_NO_WINNER for a draw. For an unfinished game, the player
 * ahead, which scores MCTS rollouts cut short */
int game_winner(const GameState *state);

/* At most how many plies are left before the game ends: no search goes deeper */
int game_plies_left(const GameState *state);

bool game_states_equal(const GameState *a, const GameState *b);

/* A hash of the whole position, for the transposition table */
uint64_t game_key(const GameState *state);

/* +1 if `player` wins, -1 if they lose, 0 for a draw */
static inline int game_result(const GameState *state, int player) {
    int winner = game_winner(state);
    return winner == GAME_NO_WINNER ? 0 : (winner == player ? 1 : -1);
}

/* ---------- NNUE inputs ---------- */

/* The active features of `state` seen from `perspective`'s side. Slot i always holds one feature,
 * so the accumulators are updated by comparing the slots of two positions */
void game_features(const GameState *state, int perspective, int16_t features[GAME_NUM_ACTIVE_FEATURES]);

/* ---------- Search extras ---------- */

/* An exact result (+1 / 0 / -1 for the side to move, perfect play) for a position that is not over,
 * from `oracle`, game data such as an endgame table attached to the searcher by the game's own
 * entry points. False when the oracle does not cover the position. */
bool game_exact_result(const void *oracle, const GameState *state, int *result);

/* ---------- Python ---------- */

/* A position sent by Python as `count` ints (the game's Python `encode`); false if malformed */
bool game_decode(const int32_t *values, int count, GameState *state);

#endif
