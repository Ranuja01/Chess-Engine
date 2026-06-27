# -*- coding: utf-8 -*-
"""Phase-0a probe: eval-cache hit-rate + qsearch share over the WAC suite.

The engine prints EVAL CACHE VISITS/HITS and Q SEARCH VISITS to stdout (search_engine.cpp ~1239-1249),
but tactical_test._parse_output discards them. This reuses tactical_test's machinery (model load, EPD
loader, fd-level stdout capture, the ChessAI driver) and sums those counters instead, so we can bound the
lazy-eval upside: addressable win ~= (1 - hit_rate) x (full_cost - light_cost).

Run from NN Engine/ under the WSL python that built ChessAI:
    python diagnostics/_cachehit_probe.py [suite=wac.epd] [limit=all]
"""
import os, re, sys
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
# Pin all thread pools BEFORE importing the engine so the search tree is byte-deterministic and matches
# the single-thread baseline (engine move-choice is OMP-thread-count-dependent).
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("USE_OPENING_BOOK", "0")
# Match the wac d10 baseline (the probe must mirror the suite it's bounding).
os.environ.setdefault("PRESET", "LONG_FORMAT")
os.environ.setdefault("MAX_DEPTH", "10")
os.environ.setdefault("LIGHT_GAP_PROBE", "1")   # accumulates the light-eval gap histogram in the engine

THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS)

import chess
from tactical_test import load_epd, _captured_stdout, blackModel, whiteModel, SUITES_DIR
from ChessAI import ChessAI

_V = re.compile(r'EVAL CACHE VISITS:\s*(\d+)')
_H = re.compile(r'EVAL CACHE HITS:\s*(\d+)')
_Q = re.compile(r'Q SEARCH VISITS:\s*(\d+)')
_LG = re.compile(r'LIGHT_GAP n=(\d+) h=([\d,]+) cap=(\d+) pas=(\d+) lat=(\d+) adv=(\d+)')
_BUCKETS = ["<100", "100-250", "250-500", "500-1000", "1000-2000", "2000-4000", ">=4000"]


def main():
    suite = sys.argv[1] if len(sys.argv) > 1 else "wac.epd"
    limit = sys.argv[2] if len(sys.argv) > 2 else "all"
    path = suite if os.path.exists(suite) else os.path.join(SUITES_DIR, suite)
    positions = load_epd(path)
    if limit != "all":
        positions = positions[:int(limit)]

    tot_v = tot_h = tot_q = 0
    lg = None   # last (cumulative) LIGHT_GAP tuple: (n, [7 buckets], cap, pas, lat, adv)
    for idx, (fen, best, label, raw_epd) in enumerate(positions):
        board = chess.Board(fen)
        with _captured_stdout() as buf:
            ai = ChessAI(blackModel, whiteModel, board, board.turn)
            ai.alphaBetaWrapper()
            sys.stdout.flush()
            buf.seek(0)
            out = buf.read()
        v = int(_V.findall(out)[-1]) if _V.search(out) else 0
        h = int(_H.findall(out)[-1]) if _H.search(out) else 0
        q = int(_Q.findall(out)[-1]) if _Q.search(out) else 0
        tot_v += v; tot_h += h; tot_q += q
        m = _LG.findall(out)
        if m:
            n, hist, cap, pas, lat, adv = m[-1]
            lg = (int(n), [int(x) for x in hist.split(",")], int(cap), int(pas), int(lat), int(adv))
        if (idx + 1) % 50 == 0:
            print(f"  ...{idx+1}/{len(positions)}", flush=True)

    print(f"\nPositions: {len(positions)}")
    print(f"EVAL CACHE VISITS: {tot_v}")
    print(f"EVAL CACHE HITS:   {tot_h}")
    print(f"Q SEARCH VISITS:   {tot_q}")
    if tot_v:
        hr = tot_h / tot_v
        print(f"\neval cache hit-rate = {hr*100:.1f}%")
        print(f"miss fraction       = {(1-hr)*100:.1f}%   (= lazy-eval addressable share)")

    if lg:
        n, hist, cap, pas, lat, adv = lg
        print(f"\n=== LIGHT-EVAL GAP (|capture+passed+latent+advanced| per eval, n={n}) ===")
        print(f"{'bucket(milli-pawn)':>18}  {'count':>12}  {'pct':>6}  {'cumPct':>6}")
        cum = 0
        for label_b, c in zip(_BUCKETS, hist):
            cum += c
            pct = 100.0 * c / n if n else 0
            print(f"{label_b:>18}  {c:>12}  {pct:>5.1f}%  {100.0*cum/n if n else 0:>5.1f}%")
        print(f"\nper-term abs contribution (sum over evals): "
              f"capture={cap}  passed={pas}  latent={lat}  advanced={adv}")


if __name__ == "__main__":
    main()
