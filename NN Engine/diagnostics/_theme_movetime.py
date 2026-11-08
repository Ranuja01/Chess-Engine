"""Solve rates on bench-format suites (FEN | best moves) at equal time, fixed depth or fixed nodes, one UCI engine process per
call, pinned to one core. Works for our binary and for reference engines (Stockfish).

Usage: python diagnostics/_theme_movetime.py BINARY CORE LIMIT KNOBS FILE [FILE ...]
  LIMIT: seconds per move (e.g. 0.25), `dN` for a fixed depth N (e.g. d10), or `nN` for N nodes per move (e.g. n250000).
  KNOBS: our engine's Knobs option ("NAME=V,NAME=V"), or "-" for none (a reference engine).
  UCI_OPTS=Name=Value|Name=Value (env, optional): other UCI options, e.g. "Use NNUE=false|Hash=128" (true/false and integers
    are converted). OwnBook is turned off when the engine has it.
  EACH=DIR (env, optional): also write DIR/<suite>.txt, one line per position: index, 1/0 solved, the move played.
Prints one line per file: name, solved, positions. Each position is a new game (ucinewgame: our search tables and caches are
cleared). Built for the lazy eval critical-situation test (LAZY-EVAL.md §6); the per-position output pairs two arms position by
position; the node limit compares engines at equal work, the regime of REFERENCE-BENCH-LADDER.md.
"""
import os
import sys

import chess
import chess.engine


def parse_value(v):
    if v.lower() in ("true", "false"):
        return v.lower() == "true"
    return int(v) if v.lstrip("-").isdigit() else v


def main(binary, core, limit, knobs, files):
    if limit.startswith("d"):
        lim = chess.engine.Limit(depth=int(limit[1:]))
    elif limit.startswith("n"):
        lim = chess.engine.Limit(nodes=int(limit[1:]))
    else:
        lim = chess.engine.Limit(time=float(limit))
    each = os.environ.get("EACH")
    if each:
        os.makedirs(each, exist_ok=True)
    eng = chess.engine.SimpleEngine.popen_uci(["taskset", "-c", str(core), binary])
    opts = {}
    if "OwnBook" in eng.options:
        opts["OwnBook"] = False
    if knobs != "-":
        opts["Knobs"] = knobs
    for kv in filter(None, os.environ.get("UCI_OPTS", "").split("|")):
        k, v = kv.split("=", 1)
        opts[k.strip()] = parse_value(v.strip())
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
