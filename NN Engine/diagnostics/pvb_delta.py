# -*- coding: utf-8 -*-
"""Played-vs-best per-term eval DELTA on STS failures — which term makes our (wrong) move out-score the best.

For each failure in a theme (from sts_results_<tag>.csv), make OUR move and the BEST move (EPD c9), take
ev_breakdown of BOTH resulting positions, and compute the per-term delta from the MOVER's POV:
    delta_T = mover_pov(after_OUR)[T] - mover_pov(after_BEST)[T]
A positive mean delta_T means term T favored our (wrong) move over the best move (over-credits our
choice); a strongly negative delta means it favored the best move (and, if it's a pawn term we could
turn up, boosting it would help pick the best move). We picked our move => total delta > 0. Ranking the
terms shows the culprit to dampen (positive) and the under-valued lever to boost (negative), across the
whole PAWN CLUSTER (pt_pawns placement, passed_pawn_support, pawn_majority, pawn_struct) + space/pieces.

Env knobs set the eval config (read at ChessAI init) — run with the pawn/mobility terms ENABLED to test
whether they would favor the push. Run (WSL, from NN Engine/):
    SCALE_CAPTURE_GAINS=70 PAWN_MAJORITY_MAG_MG=20 ... python diagnostics/pvb_delta.py <tag> [theme] [dump_n]
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
import csv
import contextlib
from collections import defaultdict

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ENGINE_DIR = os.path.dirname(THIS_DIR)
sys.path.insert(0, ENGINE_DIR)
sys.path.insert(0, THIS_DIR)

import chess  # noqa: E402
from sts_test import load_sts_epd  # noqa: E402

STS_EPD = os.path.join(THIS_DIR, "suites", "STS1-STS15_LAN_v3.epd")
RESULTS_DIR = os.path.join(THIS_DIR, "results")

# Pawn cluster first, then the interacting terms. (total = the whole eval; the rest should ~sum to it.)
DELTA_TERMS = ["total", "pieces", "pt_pawns", "pt_knights", "pt_bishops", "pt_rooks", "pt_queens",
               "passed_pawn_support", "pawn_majority", "pawn_struct", "central", "mobility", "outpost", "space", "rook_cond",
               "material", "capture_gains", "king_safety", "latent_threat", "pair_bonus", "piece_value_boost"]


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "ship"
    theme_filter = sys.argv[2] if len(sys.argv) > 2 else None
    dump_n = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    csv_path = os.path.join(RESULTS_DIR, "sts_results_%s.csv" % tag)
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    fails = []
    for r in rows:
        if r.get("score", "") == "":
            continue
        try:
            if int(r["score"]) < int(r["max"]):
                fails.append(r)
        except ValueError:
            continue
    if theme_filter:
        fails = [r for r in fails if theme_filter.lower() in r["theme"].lower()]
    best_of = {}
    for fen, score_map, mx, theme, epd_id in load_sts_epd(STS_EPD):
        if score_map:
            best_of[fen] = max(score_map, key=score_map.get)

    from ChessAI import ChessAI  # noqa
    seed = chess.Board()
    with open(os.devnull, "w") as _dn, contextlib.redirect_stdout(_dn):
        ai = ChessAI(None, None, seed, seed.turn)

    def mv_pov(bd, term, white):
        v = bd.get(term, 0) / 1000.0
        return -v if white else v   # mover-POV pawns (White wants Black-positive eval negative)

    agg = defaultdict(float)
    detail = []
    nn = 0
    n_cap = 0   # our-move-is-capture misses excluded (static eval can't see the refutation -> material confound)
    for r in fails:
        fen = r["fen"]
        our = r["engine"]
        best = best_of.get(fen)
        if not best or our == best:
            continue
        try:
            b = chess.Board(fen)
            white = b.turn
            om = chess.Move.from_uci(our)
            if b.is_capture(om):     # QUIET-only: skip capture misses (static +material confound)
                n_cap += 1
                continue
            bo = chess.Board(fen); bo.push_uci(our)
            bb = chess.Board(fen); bb.push_uci(best)
        except Exception:
            continue
        eo = ai.ev_breakdown(bo)
        eb = ai.ev_breakdown(bb)
        if eo.get("checkmate") or eb.get("checkmate"):
            continue
        nn += 1
        d = {}
        for t in DELTA_TERMS:
            d[t] = mv_pov(eo, t, white) - mv_pov(eb, t, white)
            agg[t] += d[t]
        detail.append((d["total"], fen, our, best, d))
    if nn == 0:
        print("no scorable played-vs-best pairs")
        return 1

    print("PLAYED-vs-BEST term delta  theme~%s  n_quiet=%d  (excluded %d capture-misses: static +material confound)"
          % (theme_filter, nn, n_cap))
    print("(mover-POV: +delta => term favored OUR quiet move over best [over-credit]; -delta => favored best)")
    print("  %-22s %8s" % ("term", "mean d"))
    for t in DELTA_TERMS:
        print("  %-22s %+8.3f" % (t, agg[t] / nn))
    print("\n  --- worst %d (largest total delta = most wrongly preferred our move) ---" % dump_n)
    for tot, fen, our, best, d in sorted(detail, key=lambda x: -x[0])[:dump_n]:
        drivers = sorted(((t, d[t]) for t in DELTA_TERMS if t != "total"), key=lambda kv: -kv[1])
        pos = "  ".join("%s=%+.2f" % (t, v) for t, v in drivers[:5] if abs(v) > 0.03)
        print("  our=%-6s best=%-6s totΔ=%+.2f | %s" % (our, best, tot, pos))
        print("       %s" % fen)
    return 0


if __name__ == "__main__":
    sys.exit(main())
