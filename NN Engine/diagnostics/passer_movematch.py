# -*- coding: utf-8 -*-
"""Move-match guard for the passer suite (diagnostics/suites/passers.csv).

Win%/MSE fitting is BLIND to move choice; this checks whether the engine still PICKS the suite's
Stockfish-best move after a shallow search under a given knob set. Run baseline (V3 off) vs a candidate
(V3 + Stage-1 knobs) and compare match%. Knobs are passed as KEY=VAL argv and pushed into the environment
BEFORE the engine module is imported (the engine reads them in initialize_engine).

Run (from NN Engine/, single-core):
    python diagnostics/passer_movematch.py                       # baseline (V3 off)
    python diagnostics/passer_movematch.py ENABLE_PASSER_V3=1 PASSER_MAG_SCALE=150 PASSER_CONTEST_STOP=140 \
        PASSER_CONTEST_PATH=70 PASSER_REAR_ENEMY=180
"""
import os
import sys
import csv

# Push KEY=VAL argv into the environment before importing the engine (same gotcha as passer_verify.py).
for arg in sys.argv[1:]:
    if "=" in arg:
        k, v = arg.split("=", 1)
        os.environ[k] = v

os.environ.setdefault("USE_OPENING_BOOK", "0")
os.environ.setdefault("PRESET", "LIGHTNING")
os.environ.setdefault("MAX_DEPTH", "10")

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SUITE = os.path.join(THIS_DIR, "suites", "passers.csv")

from tactical_test import run_one  # noqa: E402  (imports + initializes the engine)


def main():
    rows = []
    with open(SUITE, newline="") as f:
        for r in csv.DictReader(f):
            fen = r.get("fen_start") or r.get("fen")
            best = r.get("sf_best")
            if fen and best:
                rows.append((fen.strip(), best.strip(), (r.get("cat") or "").strip()))

    solved = 0
    by_cat = {}
    for fen, best, cat in rows:
        res = run_one(fen, {best})
        ok = bool(res.get("solved"))
        solved += ok
        c = by_cat.setdefault(cat or "?", [0, 0])
        c[0] += ok
        c[1] += 1

    n = len(rows)
    print(f"passers.csv move-match: {solved}/{n} = {100.0*solved/max(n,1):.1f}%")
    for cat in sorted(by_cat):
        s, t = by_cat[cat]
        print(f"  {cat:12s} {s:3d}/{t:3d} = {100.0*s/max(t,1):.1f}%")


if __name__ == "__main__":
    main()
