"""Equal-time solve rates on bench-format suites (FEN | best moves), one UCI engine process per call, pinned to one core.

Usage: python diagnostics/_theme_movetime.py BINARY CORE MOVETIME_S KNOBS FILE [FILE ...]
  KNOBS: the engine's Knobs option ("NAME=V,NAME=V"), or "-" for none.
Prints one line per file: name, solved, positions. Each position is a new game (ucinewgame), own book off. Companion of the
fixed-depth `bench` runs; built for the lazy eval critical-situation test (LAZY-EVAL-RESEARCH-2026-10-30.md §4 item 5), where
speed may buy back what a lazy value costs.
"""
import os
import sys

import chess
import chess.engine


def main(binary, core, movetime, knobs, files):
    eng = chess.engine.SimpleEngine.popen_uci(["taskset", "-c", str(core), binary])
    opts = {"OwnBook": False}
    if knobs != "-":
        opts["Knobs"] = knobs
    eng.configure(opts)
    for f in files:
        n = solved = 0
        for line in open(f):
            line = line.strip()
            if not line or line.startswith("#") or "|" not in line:
                continue
            fen, best = (x.strip() for x in line.split("|", 1))
            r = eng.play(chess.Board(fen), chess.engine.Limit(time=movetime), game=object())
            n += 1
            solved += r.move is not None and r.move.uci() in best.split()
        print(f"{os.path.splitext(os.path.basename(f))[0]:18s} {solved:4d} {n:4d}", flush=True)
    eng.quit()


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0], int(a[1]), float(a[2]), a[3], a[4:])
