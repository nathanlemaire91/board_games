/* Replays games for tests/check_rules.py: each stdin line is one game, as space separated moves.
 * Prints one line per position, starting with the opening one:
 *
 *     <12 holes> <seeds 0> <seeds 1> <player> <moves played> <is over> <winner> | <legal moves>
 *
 * with winner -1 for a level score. A blank line ends each game.
 */
#include <stdio.h>
#include <stdlib.h>

#include "awale.h"

static void print_state(const BoardState *state) {
    for (int hole = 0; hole < BOARD_SIZE; hole++) printf("%d ", state->board[hole]);
    printf("%d %d %d %d %d %d |", state->players_seeds[0], state->players_seeds[1], state->current_player,
           state->moves_played, is_over(state), winner(state));
    int moves[HALF_BOARD_SIZE];
    int count = get_possible_moves(state, moves);
    for (int i = 0; i < count; i++) printf(" %d", moves[i]);
    printf("\n");
}

int main(void) {
    char line[4096];
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
