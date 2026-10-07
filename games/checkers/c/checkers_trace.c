/* Replays games for tests/check_rules.py: each stdin line is one game, as space separated moves.
 * Prints one line per position, starting with the opening one:
 *
 *     <32 squares> <jumping> <player> <quiet moves> <moves played> <is over> <winner> | <legal moves>
 *
 * with jumping -1 when no multiple jump is going on, and winner -1 for none. A blank line ends each game.
 */
#include <stdio.h>
#include <stdlib.h>

#include "checkers.h"

static void print_state(const BoardState *state) {
    for (int square = 0; square < NUM_SQUARES; square++) printf("%d ", state->board[square]);
    printf("%d %d %d %d %d %d |", state->jumping, state->current_player, state->quiet_moves, state->moves_played,
           is_over(state), winner(state));
    int moves[MAX_LEGAL_MOVES];
    int count = get_possible_moves(state, moves);
    for (int i = 0; i < count; i++) printf(" %d", moves[i]);
    printf("\n");
}

int main(void) {
    static char line[1 << 16];
    while (fgets(line, sizeof line, stdin)) {
        BoardState state = new_game();
        print_state(&state);
        char *cursor = line, *end;
        for (long move = strtol(cursor, &end, 10); end != cursor; move = strtol(cursor, &end, 10)) {
            cursor = end;
            make_move(&state, (int)move);
            print_state(&state);
        }
        printf("\n");
    }
    return EXIT_SUCCESS;
}
