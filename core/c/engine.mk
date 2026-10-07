# Builds a game's C library and MCTS command line from the engines of core/c.
#
# A game's Makefile (games/<name>/c/Makefile) sets GAME, GAME_SOURCES and GAME_HEADERS
# (its rules and its side of game_api.h), then includes this file. It gets:
#
#     lib$(GAME).so   loaded by core/c_library.py
#     mcts            the MCTS command line (core/c/mcts_main.c)

ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST)))/../..)
CORE := $(ROOT)/core/c

CFLAGS ?= -O3 -march=native -Wall -Wextra -std=c11
CPPFLAGS += -I. -I$(CORE)
LDLIBS += -lm

ENGINE_SOURCES := $(CORE)/mcts.c $(CORE)/search.c $(CORE)/nnue.c
ENGINE_HEADERS := $(wildcard $(CORE)/*.h)

lib$(GAME).so: $(ENGINE_SOURCES) $(CORE)/python_api.c $(GAME_SOURCES) $(ENGINE_HEADERS) $(GAME_HEADERS)
	$(CC) $(CPPFLAGS) $(CFLAGS) -fPIC -shared -o $@ $(filter %.c,$^) $(LDLIBS)

mcts: $(CORE)/mcts_main.c $(ENGINE_SOURCES) $(GAME_SOURCES) $(ENGINE_HEADERS) $(GAME_HEADERS)
	$(CC) $(CPPFLAGS) $(CFLAGS) -o $@ $(filter %.c,$^) $(LDLIBS)
