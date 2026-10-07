GAME = checkers
GAME_SOURCES = checkers.c game.c endgame.c
GAME_HEADERS = checkers.h game.h
include ../../../core/c/engine.mk

.DEFAULT_GOAL := all
all: libcheckers.so mcts checkers_trace

# Replays games for tests/check_rules.py
checkers_trace: checkers_trace.c checkers.c checkers.h
	$(CC) $(CFLAGS) -o $@ checkers_trace.c checkers.c $(LDLIBS)

# Compares the C rules with rules.py over random games, the endgame table with the rules, and the C alpha-beta
# with a Python negamax
check: checkers_trace libcheckers.so
	cd $(ROOT) && uv run python -m games.checkers.tests.check_rules
	cd $(ROOT) && uv run python -m games.checkers.tests.check_endgame
	cd $(ROOT) && uv run python -m games.checkers.tests.check_search

clean:
	rm -f mcts checkers_trace libcheckers.so

.PHONY: all check clean
