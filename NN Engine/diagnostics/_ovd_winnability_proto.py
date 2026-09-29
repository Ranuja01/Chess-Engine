# -*- coding: utf-8 -*-
"""OvD PROTOTYPE, endgame half -- WINNABILITY: do SF's complexity inputs predict whether an EDGE converts, beyond the eval?

Design: dev_notes/TEXEL-C3-DETECTORS-DESIGN-2026-09-27.md §8 (OvD's eg leg = winnability; universal shape SF11 `initiative`
/ SF12+ `winnable` / Ethereal `evaluateComplexity`: it can grow or shrink an advantage but never flips the sign). v2 has
no such term: its endgame can only say "draw" (the classifier) or full value.

TEST. Rows in the endgame-ish range (phase256 < PHASE) where one side holds an edge (EDGE_LO <= |eval| <= EDGE_HI mp),
scored from the STRONGER side's point of view: residual = its game score − the logistic of its eval. Each SF11 input is
correlated with that residual (a positive r = more of this input ⇒ the edge converts more often than the eval says):
  pawns (total) · both_flanks (pawns on a-d AND e-h) · passed (both sides' passers) · outflanking (king file distance −
  rank distance) · pawn_ending (no non-pawn material) · and SF11's COMPLEXITY composite
  9·passed + 12·pawns + 9·outflanking + 21·both_flanks + 51·pawn_ending − 43·almost_unwinnable − 110
  (infiltration omitted: it needs the king's side-of-board, a detail for the build, not the pilot).
Pure Python, no engine; feasibility only.

  pyrun diagnostics/_ovd_winnability_proto.py [N=600000] [PHASE=96] [EDGE_LO=1000] [EDGE_HI=4000]
"""
import os, sys, math
import numpy as np
import pandas as pd

KV = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
N = int(KV.get("N", 600000))
PHASE = float(KV.get("PHASE", 96))
EDGE_LO, EDGE_HI = float(KV.get("EDGE_LO", 1000)), float(KV.get("EDGE_HI", 4000))
DATA = KV.get("DATA", "/mnt/e/chess_data/texel")
THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS)
from _pawn_term_overlap import terms

QS, KS_ = 0x0F0F0F0F0F0F0F0F, 0xF0F0F0F0F0F0F0F0


def parse(fen):
    wp = bp = 0
    npm = 0
    wk = bk = None
    r, f = 7, 0
    for ch in fen.split(" ", 1)[0]:
        if ch == "/":
            r -= 1; f = 0
        elif ch <= "8":
            f += ord(ch) - 48
        else:
            sq = r * 8 + f
            if ch == "P": wp |= 1 << sq
            elif ch == "p": bp |= 1 << sq
            elif ch == "K": wk = sq
            elif ch == "k": bk = sq
            else: npm += 1
            f += 1
    return wp, bp, npm, wk, bk


def main():
    st = pd.read_csv(os.path.join(DATA, "fitC_stage1.csv.gz"), usecols=["fen", "result_white"])
    fz = np.load(os.path.join(DATA, "fitC_features.npz"))
    full, ph = fz["total"].astype(np.float64), fz["phase"].astype(np.float64)
    E = -full                                              # White-positive
    sel = np.where((ph >= 0) & (ph < PHASE) & (np.abs(E) >= EDGE_LO) & (np.abs(E) <= EDGE_HI))[0]
    rng = np.random.default_rng(1)
    if len(sel) > N:
        sel = np.sort(rng.choice(sel, size=N, replace=False))
    y = st["result_white"].values[sel].astype(np.float64)
    e = E[sel]
    strong_white = e > 0
    ys = np.where(strong_white, y, 1.0 - y)               # the stronger side's score
    es = np.abs(e)
    # K on these rows, then the stronger side's residual
    def mse(k):
        return float(np.mean((1.0 / (1.0 + np.exp(-k * es)) - ys) ** 2))
    lo, hi = 1e-5, 1e-2
    for _ in range(60):
        a = math.exp(math.log(lo) + 0.382 * (math.log(hi) - math.log(lo)))
        b = math.exp(math.log(lo) + 0.618 * (math.log(hi) - math.log(lo)))
        lo, hi = (lo, b) if mse(a) < mse(b) else (a, hi)
    K = (lo + hi) / 2
    res = ys - 1.0 / (1.0 + np.exp(-K * es))

    feats = {k: np.zeros(len(sel)) for k in ("pawns", "both_flanks", "passed", "outflanking", "pawn_ending", "complexity")}
    for i, fen in enumerate(st["fen"].values[sel]):
        wp, bp, npm, wk, bk = parse(fen)
        pawns = bin(wp | bp).count("1")
        both = int(bool((wp | bp) & QS) and bool((wp | bp) & KS_))
        passed = bin(terms(wp, bp, True)["passed"]).count("1") + bin(terms(bp, wp, False)["passed"]).count("1")
        outfl = (abs((wk & 7) - (bk & 7)) - abs((wk >> 3) - (bk >> 3))) if wk is not None and bk is not None else 0
        pe = int(npm == 0)
        almost = int(passed == 0 and outfl < 0 and not both)
        cx = 9 * passed + 12 * pawns + 9 * outfl + 21 * both + 51 * pe - 43 * almost - 110
        for k, v in (("pawns", pawns), ("both_flanks", both), ("passed", passed), ("outflanking", outfl),
                     ("pawn_ending", pe), ("complexity", cx)):
            feats[k][i] = v
    print("WINNABILITY PILOT  rows %d (phase256 < %.0f, stronger side's edge %.0f-%.0f mp)  K %.6f"
          % (len(sel), PHASE, EDGE_LO, EDGE_HI, K))
    print("  stronger side scores %.3f; the eval predicts %.3f" % (ys.mean(), (ys - res).mean()))
    se = 1.0 / math.sqrt(max(len(sel) - 3, 1))
    for k, f in feats.items():
        r = np.corrcoef(f, res)[0, 1] if f.std() > 0 else float("nan")
        print("  %-12s mean %7.2f   r(feature, stronger-side residual) %+.4f  (±%.4f, %.1fσ)" % (k, f.mean(), r, se, r / se))
    q = np.quantile(feats["complexity"], [0.2, 0.4, 0.6, 0.8])
    edges = [-np.inf] + list(q) + [np.inf]
    print("  complexity quintiles -> stronger side's mean residual:")
    for a, b in zip(edges[:-1], edges[1:]):
        m = (feats["complexity"] > a) & (feats["complexity"] <= b)
        if m.sum():
            print("    (%6.0f, %6.0f]  n %6d  residual %+.4f   score %.3f" % (a, b, m.sum(), res[m].mean(), ys[m].mean()))
    print("⚠️ Feasibility pilot only; the fit + games decide.")


if __name__ == "__main__":
    main()
