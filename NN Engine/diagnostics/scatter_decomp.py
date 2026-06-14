# -*- coding: utf-8 -*-
"""
Localize the near-equal endgame eval scatter by term.

The accuracy proxy showed our static eval is ~unbiased near equality but scatters ~1.5p mean / ~2.2p
stdev vs SF_static (|SF|<1) -- the steep-sigmoid Elo lever. In near-equal positions the "true" total
is ~0, so each additive term's contribution IS roughly its error contribution. The term most correlated
with the total error (and with the largest stdev) is the scatter driver.

Uses ev_breakdown's existing fields: pt_pawns..pt_kings (per-piece-type, populated in BOTH mid and
endgame paths) + capture_gains / passed_pawn_support / latent_threat / central / piece_value_boost.
Joins SF_static from the base accuracy run. Positions where advanced_endgame_eval REPLACED total are
excluded (their additive terms are pre-replace, so the decomposition wouldn't sum to total).

    python diagnostics/scatter_decomp.py [band]      # band default 1.0 (|SF_static| < band)
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys
import json
import math
import statistics as st

import chess

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ENGINE_DIR = os.path.dirname(THIS_DIR)
sys.path.insert(0, ENGINE_DIR)
sys.path.insert(0, os.path.join(ENGINE_DIR, "selfplay"))
RESULTS = os.path.join(THIS_DIR, "results")

# Full additive decomposition of the final eval, valid in BOTH the midgame path and the endgame
# REPLACE path: when advanced_endgame_eval fires, pt_* are its additive INPUT and ae_matedrive/ae_passer
# are its two internal deltas, so pt_* + the rest still sum to the final total.
TERMS = ["pt_pawns", "pt_knights", "pt_bishops", "pt_rooks", "pt_queens", "pt_kings",
         "capture_gains", "passed_pawn_support", "latent_threat", "central",
         "ae_matedrive", "ae_passer", "pair_bonus", "piece_value_boost"]


def wp(v):
    return -v / 1000.0


def corr(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / n
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs) / n)
    sy = math.sqrt(sum((y - my) ** 2 for y in ys) / n)
    return cov / (sx * sy) if sx * sy > 0 else 0.0


def main():
    band = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
    sf = {}
    for l in open(os.path.join(RESULTS, "symup_acc_base.jsonl")):
        r = json.loads(l)
        sf[r["fen"]] = r.get("sf_static")
    fens = [l.strip() for l in open(os.path.join(RESULTS, "symup_corpus.fen")) if l.strip()]

    from ChessAI import ChessAI
    ai = ChessAI(None, None, chess.Board(), True)

    rows, fired, recon_max = [], 0, 0.0
    for fen in fens:
        s = sf.get(fen)
        if s is None or abs(s) >= band:
            continue
        bd = ai.ev_breakdown(chess.Board(fen))
        if bd.get("checkmate"):
            continue
        if bd.get("advanced_endgame_fired"):
            fired += 1
        rec = {"fen": fen, "sf": s, "our": wp(bd["total"]), "err": wp(bd["total"]) - s}
        for t in TERMS:
            rec[t] = wp(bd[t])
        recon_max = max(recon_max, abs(sum(rec[t] for t in TERMS) - rec["our"]))
        rows.append(rec)

    if not rows:
        print("no near-equal positions in band")
        return
    errs = [r["err"] for r in rows]
    print("near-equal |SF_static|<%.1f endgame positions: n=%d  (%d had advanced_endgame fire)  recon_max_gap=%.3f"
          % (band, len(rows), fired, recon_max))
    print("  total error: mean=%+.2f  scatter(mean|err|)=%.2f  stdev=%.2f" %
          (st.mean(errs), st.mean(abs(e) for e in errs), st.pstdev(errs)))
    print("  per-term (White-POV pawns; in near-equal the true total ~ SF, so a term's mean=bias, "
          "stdev=scatter it injects, corr=how much it drives the error):")
    print("    %-22s %8s %8s %8s %9s" % ("term", "mean", "stdev", "mean|x|", "corr_err"))
    stats = []
    for t in TERMS:
        xs = [r[t] for r in rows]
        stats.append((t, st.mean(xs), st.pstdev(xs), st.mean(abs(x) for x in xs), corr(xs, errs)))
    for t, m, sd, ma, c in sorted(stats, key=lambda z: abs(z[4]), reverse=True):
        print("    %-22s %+8.2f %8.2f %8.2f %+9.2f" % (t, m, sd, ma, c))


if __name__ == "__main__":
    main()
