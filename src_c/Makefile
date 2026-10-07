CC ?= cc
CFLAGS ?= -O2 -march=native -Wall -Wextra -std=c11
LDLIBS = -lm

all: mcts awale_trace libawale.so

# Loaded by src/c_mcts.py (keep its LIBRARY_SOURCES in sync)
libawale.so: awale.c mcts.c python_api.c awale.h mcts.h
	$(CC) $(CFLAGS) -fPIC -shared -o $@ awale.c mcts.c python_api.c $(LDLIBS)

mcts: main.o mcts.o awale.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

awale_trace: awale_trace.o awale.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

main.o: main.c awale.h mcts.h
mcts.o: mcts.c mcts.h awale.h
awale.o: awale.c awale.h
awale_trace.o: awale_trace.c awale.h

# Compares the C rules with src/awale.py over random games
check: awale_trace
	cd .. && uv run python src_c/check_parity.py

clean:
	rm -f *.o mcts awale_trace libawale.so

.PHONY: all check clean
