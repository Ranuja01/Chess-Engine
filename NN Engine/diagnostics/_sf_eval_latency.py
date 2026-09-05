# -*- coding: utf-8 -*-
"""How fast can we ask a persistent Stockfish for a STATIC eval? Decides the SF-oracle experiment's scope.

The oracle experiment substitutes SF's evaluation for ours inside get_board_evaluation, so every eval-cache
MISS costs one UCI round trip. Feasibility is therefore set entirely by that round trip:
    ~0.3 ms  -> ~1M unique positions affordable -> oracle viable at fixed depth 8 with a margin sweep on top
    ~3   ms  -> ~100k affordable               -> depth 6, ~30 positions, move-choice comparison only
This measures it before any engine wiring is written.

Also verifies the OUTPUT FORMAT and the SIGN, which is the highest-risk part of the port: ours is
millipawns ABSOLUTE Black-positive, SF's is pawns White-POV, so the conversion is
    ours = -(sf_pawns * 1000)
and getting it backwards yields a plausible but inverted experiment.

Run (from NN Engine/):
    pyrun diagnostics/_sf_eval_latency.py --engine sf11 --n 400
"""

import os
import sys
import time
import argparse
import subprocess

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)

import chess


def resolve_engine(name):
    if name == 'sf18':
        p = os.environ.get('STOCKFISH_PATH')
        if not p:
            raise SystemExit("STOCKFISH_PATH is not set (SF18).")
        return p
    if name == 'sf11':
        from eval_vs_sf11 import SF11
        return SF11
    raise SystemExit("unknown engine %r" % name)


# ⚠️ DO NOT hand-roll a `eval` parser here. The first version of this script did, and it was WRONG in a way
# that looked like a great result: it read until the first line containing "evaluation", which matches SF11's
# per-TERM table rows, and it never synchronised on `isready`. Unread output stayed in the pipe, so each call
# read the PREVIOUS call's tail -- reporting 0.112 ms/call (impossibly fast) with 27/400 "parse failures".
# `eval_vs_sf11.SF11Eval` is the canonical, validated reader (probe_fens.py depends on it): it drains to
# `readyok` and regexes specifically for "Total evaluation:". Reuse it.
from eval_vs_sf11 import SF11Eval


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--engine', default='sf11')
    ap.add_argument('--n', type=int, default=400)
    ap.add_argument('--classical', action='store_true', help='SF15/18: force NNUE off')
    args = ap.parse_args()

    path = resolve_engine(args.engine)
    if not os.path.exists(path):
        raise SystemExit("engine binary not found: %s" % path)

    # Walk a real game so the FENs are varied and legal, mirroring what a search would ask about.
    board = chess.Board()
    fens = []
    import random
    rng = random.Random(1234)
    while len(fens) < args.n:
        if board.is_game_over():
            board = chess.Board()
        fens.append(board.fen())
        board.push(rng.choice(list(board.legal_moves)))

    eng = SF11Eval(path)
    try:
        # Warm up (first call pays process/table init).
        eng.eval(fens[0])

        t0 = time.time()
        vals = [eng.eval(f)[0] for f in fens]
        dt = time.time() - t0

        ok = [v for v in vals if v is not None]
        per = (dt / len(fens)) * 1000.0
        print("[sf_eval_latency] engine=%s n=%d  parsed=%d/%d" % (args.engine, len(fens), len(ok), len(fens)))
        print("  total %.2fs   per-call %.3f ms   -> %.0f evals/sec" % (dt, per, len(fens) / dt if dt else 0))
        print()
        print("  FEASIBILITY (unique positions affordable in a 10-minute budget): %.0f" % (600.0 / (per / 1000.0)))
        print("    depth 6, ~50 positions needs roughly 100k-300k unique -> %s"
              % ("OK" if 600.0 / (per / 1000.0) > 300000 else "TIGHT"))
        print("    depth 8, ~50 positions needs roughly 1M+ unique       -> %s"
              % ("OK" if 600.0 / (per / 1000.0) > 1000000 else "TOO SLOW"))
        print()
        print("  SIGN CHECK (SF is pawns White-POV; ours = -(sf*1000) millipawns Black-positive)")
        for f, v in list(zip(fens, vals))[:3]:
            print("    sf=%-8s ours_would_be=%-9s %s" % (v, None if v is None else -int(v * 1000), f))
    finally:
        eng.close()


if __name__ == "__main__":
    main()
