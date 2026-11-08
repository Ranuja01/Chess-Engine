"""Equal-time (or fixed-depth) solve rates on bench-format suites (FEN | best moves), one UCI engine process per call, pinned to one core.

Usage: python diagnostics/_theme_movetime.py BINARY CORE LIMIT KNOBS FILE [FILE ...]
  LIMIT: seconds per move (e.g. 0.25), or `dN` for a fixed depth N (e.g. d10).
  KNOBS: the engine's Knobs option ("NAME=V,NAME=V"), or "-" for none.
  EACH=DIR (env, optional): also write DIR/<suite>.txt, one line per position: index, 1/0 solved, the move played.
Prints one line per file: name, solved, positions. Each position is a new game (ucinewgame: search tables and caches cleared),
own book off. Companion of the fixed-depth `bench` runs; built for the lazy eval critical-situation test
(LAZY-EVAL-RESEARCH-2026-10-30.md §4 item 5), where speed may buy back what a lazy value costs. The per-position output pairs two
arms position by position (11-08, the lazy eval passer guard test).
"""
import os
import sys

import chess
import chess.engine


def main(binary, core, limit, knobs, files):
    lim = chess.engine.Limit(depth=int(limit[1:])) if limit.startswith("d") else chess.engine.Limit(time=float(limit))
    each = os.environ.get("EACH")
    if each:
        os.makedirs(each, exist_ok=True)
    eng = chess.engine.SimpleEngine.popen_uci(["taskset", "-c", str(core), binary])
    opts = {"OwnBook": False}
    if knobs != "-":
        opts["Knobs"] = knobs
    eng.configure(opts)
    for f in files:
        name = os.path.splitext(os.path.basename(f))[0]
        out = open(os.path.join(each, name + ".txt"), "w") if each else None
        n = solved = 0
        for line in open(f):
            line = line.strip()
            if not line or line.startswith("#") or "|" not in line:
                continue
            fen, best = (x.strip() for x in line.split("|", 1))
            r = eng.play(chess.Board(fen), lim, game=object())
            ok = r.move is not None and r.move.uci() in best.split()
            if out:
                out.write(f"{n} {int(ok)} {r.move.uci() if r.move else '-'}\n")
            n += 1
            solved += ok
        if out:
            out.close()
        print(f"{name:18s} {solved:4d} {n:4d}", flush=True)
    eng.quit()


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0], int(a[1]), a[2], a[3], a[4:])
