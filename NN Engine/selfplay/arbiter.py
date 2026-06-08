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
import re
import shutil
import subprocess

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
        self.path = path
        self.movetime = movetime
        self.depth = depth
        self.engine = chess.engine.SimpleEngine.popen_uci(path)
        self._raw = None  # lazily-opened raw pipe for the `eval` (static/NNUE) command
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

    def evaluate_static(self, board):
        """Stockfish's STATIC (NNUE) eval of `board` via the UCI `eval` command — White-POV centipawns.

        This is the leaf eval with NO search, the apples-to-apples yardstick for our handcrafted static eval
        (comparing our static eval to SF's SEARCH eval would conflate static miscalibration with the normal
        static-vs-search gap). `eval` is a Stockfish extension SimpleEngine doesn't model, so it runs over a
        dedicated raw pipe. Returns None for in-check / unparseable positions (a static eval is only meaningful
        in a quiet position anyway)."""
        if board.is_check():
            return None
        proc = self._ensure_raw()
        if proc is None:
            return None
        try:
            proc.stdin.write("position fen %s\neval\nisready\n" % board.fen())
            proc.stdin.flush()
        except Exception:
            return None
        lines = []
        while True:
            line = proc.stdout.readline()
            if not line:
                break
            s = line.strip()
            if s == "readyok":
                break
            lines.append(s)
        return self._parse_static_cp(lines)

    def _ensure_raw(self):
        if self._raw is None:
            try:
                self._raw = subprocess.Popen(
                    [self.path], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL, text=True, bufsize=1)
            except Exception:
                self._raw = None
                return None
            self._raw.stdin.write("uci\n")
            self._raw.stdin.flush()
            while True:
                line = self._raw.stdout.readline()
                if not line or line.strip() == "uciok":
                    break
        return self._raw

    @staticmethod
    def _parse_static_cp(lines):
        """Pull White-POV centipawns from `eval` output. Prefer 'Final evaluation' (NNUE + SF scaling), fall
        back to the raw 'NNUE evaluation' line. SF reports these '(white side)', so no perspective flip."""
        final_val = nnue_val = None
        for s in lines:
            low = s.lower()
            if "final evaluation" in low:
                if "none" in low:  # "Final evaluation: none (in check)"
                    return None
                m = re.search(r"[-+]?\d+\.\d+", s)
                if m:
                    final_val = float(m.group(0))
            elif "nnue evaluation" in low:
                m = re.search(r"[-+]?\d+\.\d+", s)
                if m:
                    nnue_val = float(m.group(0))
        val = final_val if final_val is not None else nnue_val
        return None if val is None else int(round(val * 100))

    def close(self):
        try:
            self.engine.quit()
        except Exception:
            pass
        if self._raw is not None:
            try:
                self._raw.stdin.write("quit\n")
                self._raw.stdin.flush()
                self._raw.wait(timeout=2)
            except Exception:
                try:
                    self._raw.kill()
                except Exception:
                    pass
