"""A game's C library (engines of core/c + the game's c/ directory), loaded with ctypes
for c_mcts.py and c_search.py.

Build it first with `make -C games/<name>/c` (from the repository root).
"""

import ctypes
from pathlib import Path

from core.game import ROOT, Game, GameState

CORE_SOURCE_DIR = ROOT / "core" / "c"
BAD_POSITION = -2  # From python_api.c: the position did not decode

_libraries: dict[str, ctypes.CDLL] = {}


def sources(game: Game) -> list[Path]:
    """Every file the library is built from."""
    return [path for directory in (CORE_SOURCE_DIR, game.library_path.parent)
            for path in directory.iterdir() if path.suffix in (".c", ".h", ".mk") or path.name == "Makefile"]


def library(game: Game) -> ctypes.CDLL:
    """lib<game>.so, loaded on first use so that importing this module never needs it."""
    if game.name not in _libraries:
        path = game.library_path
        build = f"`make -C {path.parent.relative_to(ROOT)}`"
        if not path.exists():
            raise RuntimeError(f"{path} is missing: build it with {build}")
        if max(source.stat().st_mtime for source in sources(game)) > path.stat().st_mtime:
            raise RuntimeError(f"{path} is older than its sources: rebuild it with {build}")
        lib = ctypes.CDLL(str(path))
        position = [ctypes.POINTER(ctypes.c_int32), ctypes.c_int]
        signatures = {
            "py_mcts_new": ([ctypes.c_long, ctypes.c_double, ctypes.c_int, ctypes.c_uint64], ctypes.c_void_p),
            "py_mcts_delete": ([ctypes.c_void_p], None),
            "py_mcts_best_move": ([ctypes.c_void_p] + position, ctypes.c_int),
            "py_network_floats": ([], ctypes.c_long),
            "py_ab_new": (
                [ctypes.POINTER(ctypes.c_float), ctypes.c_long, ctypes.c_int, ctypes.c_double, ctypes.c_int],
                ctypes.c_void_p,
            ),
            "py_ab_delete": ([ctypes.c_void_p], None),
            "py_ab_best_move": ([ctypes.c_void_p] + position, ctypes.c_int),
            "py_ab_stats": (
                [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_int),
                 ctypes.POINTER(ctypes.c_float)],
                None,
            ),
            "py_ab_evaluate": ([ctypes.c_void_p] + position, ctypes.c_float),
        }
        for name, (argtypes, restype) in signatures.items():
            function = getattr(lib, name)
            function.argtypes = argtypes
            function.restype = restype
        _libraries[game.name] = lib
    return _libraries[game.name]


def position_args(game: Game, state: GameState) -> tuple:
    """A position as the C entry points take it: the game's encoding and its length."""
    values = game.encode(state)
    return (ctypes.c_int32 * len(values))(*values), len(values)


def check_move(move: int, state: GameState) -> int:
    if move == BAD_POSITION:
        raise ValueError(f"the C library could not decode the position {state!r}")
    return move
