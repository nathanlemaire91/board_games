/* Checkers' types and sizes for the engines of core/c (see core/c/game_api.h) */
#ifndef GAME_H
#define GAME_H

#include "checkers.h"

typedef BoardState GameState;

#define GAME_MAX_MOVES MAX_LEGAL_MOVES /* Each of 12 pieces in each of 4 directions */
#define GAME_MAX_PLIES MAX_PLIES

/* NNUE features, as in games/checkers/features.py, seen from a player's side with the board turned
 * so that their pieces move up: what each square holds (empty, own man, own king, opponent's man,
 * opponent's king), and the square of the piece in the middle of a multiple jump, or none */
#define SQUARE_VALUES 5
#define JUMPING_VALUES (NUM_SQUARES + 1)
#define GAME_NUM_FEATURES (NUM_SQUARES * SQUARE_VALUES + JUMPING_VALUES)
#define GAME_NUM_ACTIVE_FEATURES (NUM_SQUARES + 1)

#endif
