# -*- coding: utf-8 -*-
"""Stockfish-eval arbiter for the self-play harness.

An independent per-move centipawn signal: while the two engines play, Stockfish scores each resulting
position so we can compare the engine's own eval against a strong oracle. It runs strictly BETWEEN
moves (both engine processes are idle, blocked on stdin) and is time-gated, so it never competes for
CPU with either engine's search.

Reuses find_stockfish() from diagnostics/stockfish_oracle.py (the same path logic the diagnostics
tools use). Scores are reported White-POV in centipawns, to line up with the driver's eval_white_pov.
"""

import os
import shutil

import chess
import chess.engine


def find_stockfish():
    """Locate a Stockfish binary: env vars, then PATH, then known local paths. None if not found.

    Prefers diagnostics/stockfish_oracle.find_stockfish (single source of truth); falls back to an
    inline copy if that import is unavailable (ENGINE_DIR is on sys.path via the driver)."""
    try:
        from diagnostics.stockfish_oracle import find_stockfish as _oracle_find
        return _oracle_find()
    except Exception:
        pass
    for var in ("STOCKFISH_PATH", "STOCKFISH"):
        p = os.environ.get(var)
        if p and os.path.exists(p):
            return p
    w = shutil.which("stockfish")
    if w:
        return w
    for c in (
        "/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/stockfish/stockfish-windows-x86-64-avx2.exe",
        "/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/stockfish/stockfish.exe",
        "../stockfish/stockfish-windows-x86-64-avx2.exe",
    ):
        if os.path.exists(c):
            return c
    return None


class Arbiter:
    """A persistent Stockfish process that scores positions White-POV, time-gated by default.

    Runs at Threads=1 and only between engine moves, so it cannot steal CPU from either engine's
    search. evaluate() returns (cp_white_pov, best_uci, reached_depth); cp/best are None if Stockfish
    returns no score/pv (e.g. a terminal position)."""

    def __init__(self, path, movetime=0.3, depth=None):
        self.movetime = movetime
        self.depth = depth
        self.engine = chess.engine.SimpleEngine.popen_uci(path)
        try:
            self.engine.configure({"Threads": 1})
        except Exception:
            pass

    def evaluate(self, board):
        # depth set => fixed-depth (reproducible); else time-gated (bounded wall-clock, known depth).
        limit = (chess.engine.Limit(depth=self.depth) if self.depth
                 else chess.engine.Limit(time=self.movetime))
        info = self.engine.analyse(board, limit)
        score = info.get("score")
        cp = score.pov(chess.WHITE).score(mate_score=100000) if score is not None else None
        pv = info.get("pv")
        best = pv[0].uci() if pv else None
        return cp, best, info.get("depth")

    def close(self):
        try:
            self.engine.quit()
        except Exception:
            pass
