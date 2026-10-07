GAME = awale
GAME_SOURCES = awale.c game.c endgame.c
GAME_HEADERS = awale.h game.h
include ../../../core/c/engine.mk

.DEFAULT_GOAL := all
all: libawale.so mcts awale_trace

# Replays games for tests/check_rules.py
awale_trace: awale_trace.c awale.c awale.h
	$(CC) $(CFLAGS) -o $@ awale_trace.c awale.c $(LDLIBS)

# Compares the C rules with rules.py over random games, and the C alpha-beta with a Python negamax
check: awale_trace libawale.so
	cd $(ROOT) && uv run python -m games.awale.tests.check_rules
	cd $(ROOT) && uv run python -m games.awale.tests.check_search

clean:
	rm -f mcts awale_trace libawale.so

.PHONY: all check clean
