# Board games

Game engines (MCTS, alpha-beta with an NNUE evaluation, self-play training, Elo
ratings, a web server) shared by any two-player game, and the games that use them:
Awale and checkers.

Players usually take turns, but a game may let a player move several times in a row:
checkers plays a multiple jump as one move per jump. The engines read the side to move
after each move instead of assuming it alternates: MCTS credits each move to the player
who made it, the alpha-beta keeps the score's sign and depth (which counts turns) while
the same player moves on, and training scores such moves with their afterstate's value
as is rather than negated.

## Layout

```
core/                   reusable by every game
  game.py               the interface a game implements in Python (Game, GameState)
  nnue.py               NNUE network (PyTorch)
  rl.py                 self-play training of the network
  players.py            random, MCTS, NNUE and alpha-beta players, built from command line specs
  elo.py                ratings from round-robin games between players
  server.py             web server: play in a browser against the alpha-beta
  c_library.py, c_mcts.py, c_search.py    ctypes bindings to the C engines
  c/                    the C engines
    game_api.h          the interface a game implements in C
    mcts.c search.c nnue.c python_api.c   engines and their Python entry points
    engine.mk           build rules a game's Makefile includes

games/
  __init__.py           the list of games
  awale/
    __init__.py         Awale's Game descriptor
    rules.py            the rules
    features.py         NNUE inputs
    endgame.py          endgame table, exact results for the alpha-beta
    play.py renderer.py desktop window
    c/                  the rules in C, their side of game_api.h, the endgame table lookup, Makefile
    web/                the browser page
    tests/              C vs Python parity checks
  checkers/             English draughts, laid out like awale/ (no desktop window)
    rules.py features.py   the rules, multiple jumps as one move per jump, and the NNUE inputs
    endgame.py          endgame table of up to 4 pieces, built by retrograde analysis in c/endgame.c
    c/ web/ tests/      the rules in C, the browser page, perft, endgame and C vs Python checks

models/<game>/          trained weights, snapshots, endgame tables, Elo results
scripts/                serving the web page (start_web.sh) and port forwarding
```

## Commands

From the repository root:

```
make -C games/awale/c                 # build libawale.so (needed by MCTS and alpha-beta players)
make -C games/awale/c check           # C vs Python parity checks
uv run python -m core.rl --resume     # train the network (Ctrl+C saves)
uv run python -m core.elo models/awale random mcts@0.1 ab:0:models/awale/nnue.pt@0.1
uv run python -m games.awale.play human ab:0:models/awale/nnue.pt@1
./scripts/start_web.sh                # web server and Cloudflare tunnel
```

Scripts take `--game` (Awale by default). For checkers:

```
make -C games/checkers/c              # build libcheckers.so
make -C games/checkers/c check        # perft, rule positions, endgame table, C vs Python parity checks
uv run python -m games.checkers.endgame 5   # a 5-piece table (minutes, 300 MB); the default is 4
uv run python -m core.rl --game checkers --resume
uv run python -m core.elo --game checkers random mcts@0.1 ab:0:models/checkers/nnue.pt@0.1
./scripts/start_web.sh --game checkers
```

## Adding a game

1. **Rules in Python**: `games/<name>/rules.py`, a state class with the methods of
   `GameState` in `core/game.py` (`copy`, `get_possible_moves`, `make_move`,
   `is_over`, `winner`, `result`, `current_player`, `moves_played`). A move is an
   int from 0 to 127; `current_player` may stay the same after a move.
2. **NNUE features**: a function giving a position's active features from one
   player's point of view, a fixed number of them, one per slot.
3. **Descriptor**: `games/<name>/__init__.py` defines `GAME = Game(...)`, and the name
   goes in `GAMES` in `games/__init__.py`.
4. **C side** in `games/<name>/c`: `game.h` (the `GameState` struct and the
   `GAME_*` sizes) and the functions of `core/c/game_api.h`, mirroring the Python rules
   and features. A `Makefile` sets `GAME`, `GAME_SOURCES` and `GAME_HEADERS` and
   includes `core/c/engine.mk`. A game without exact results makes
   `game_exact_result` return false.
5. **Web page** (optional): `games/<name>/web`, using the JSON of `Game.to_json`.

Then training, players, ratings and the web server work for it with `--game <name>`.
