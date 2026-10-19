# -*- coding: utf-8 -*-
"""Is our eval MIS-SCALED BY MATERIAL CLASS? (2026-10-19; memory sf-cp-labels-are-material-scaled-use-wdl)
  V2_PRESET=candidate pyrun diagnostics/_wdl_calibration.py [IN=/mnt/e/chess_data/wdlcal/cal_wdl.csv] [META=…/cal_in.csv]
The retune objective read SF18 cp through ONE fixed cp → win% formula; SF ≥ 15.1 scales cp by material, so the retune may have
fitted per-class scales to SF's convention. For each class: the best single multiplier s on OUR candidate static eval
(pred = 100·σ(k·s·ours)) against (a) the old label σ(k·sf_cp) and (b) SF18's WDL expected score (w + d/2). Reads:
  · s per class under each label — a class whose s moves far from the global s under WDL is mis-scaled vs the true win model
  · MSE gain from class-specific s over one global s, under each label (how much a per-class rescale would buy)
6,000 held-out quiet own-play rows (val split of corpus_v2), 1,500 per stage.
"""
import os, sys, csv
import numpy as np
from scipy.optimize import minimize_scalar
KV = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
IN = KV.get("IN", "/mnt/e/chess_data/wdlcal/cal_wdl.csv")
META = KV.get("META", "/mnt/e/chess_data/wdlcal/cal_in.csv")
THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(THIS))
import chess, ChessAI

K = 0.00368208
sig = lambda x: 100.0 / (1.0 + np.exp(-np.clip(x, -60, 60)))
NPM = {chess.KNIGHT: 1, chess.BISHOP: 1, chess.ROOK: 2, chess.QUEEN: 4}


def mclass(b):
    pm = {t: len(b.pieces(t, chess.WHITE)) + len(b.pieces(t, chess.BLACK)) for t in NPM}
    npm = sum(NPM[t] * n for t, n in pm.items())
    if npm == 0: return "pawn only"
    if pm[chess.QUEEN]: return "queens on" if npm > 12 else "queen ending"
    if pm[chess.ROOK]: return "rooks, no queens" if npm > 6 else "rook ending"
    return "minors only"


def main():
    meta = {r["fen"]: r for r in csv.DictReader(open(META, newline=""))}
    rows = [r for r in csv.DictReader(open(IN, newline="")) if r["fen"] in meta and abs(float(r["sf_cp_white"])) < 20000]
    AI = ChessAI.ChessAI(None, None, chess.Board(), True)
    ours, cls, stg, ycp, yw = [], [], [], [], []
    for r in rows:
        b = chess.Board(r["fen"])
        ours.append(np.clip(-int(AI.ev_breakdown(b)["total"]) / 10.0, -3000, 3000))      # White cp (engine is Black-positive)
        cls.append(mclass(b)); stg.append(meta[r["fen"]]["stage"])
        ycp.append(sig(K * float(r["sf_cp_white"]))); yw.append((float(r["w"]) + 0.5 * float(r["d"])) / 10.0)
    ours, cls, stg, ycp, yw = map(np.array, (ours, cls, stg, ycp, yw))
    print("rows %d · mean |old label − WDL| %.1f win%% pts" % (len(rows), np.mean(np.abs(ycp - yw))))

    def best_s(m, y):
        r = minimize_scalar(lambda s: np.mean((sig(K * s * ours[m]) - y[m]) ** 2), bounds=(0.05, 10), method="bounded")
        return r.x, r.fun
    allm = np.ones(len(ours), bool)
    for lname, y in (("OLD label (SF cp → fixed formula)", ycp), ("SF18 WDL expected score", yw)):
        sg, mg = best_s(allm, y)
        print("\n%s — global best s %.2f (MSE %.1f)" % (lname, sg, mg))
        print("  %-18s %5s  %6s  %8s  %8s" % ("class", "n", "best s", "MSE@glob", "MSE@own"))
        tot_own = 0.0
        for groups in (("stage", stg), ("material", cls)):
            for c in sorted(set(groups[1])):
                m = groups[1] == c
                s, mo = best_s(m, y)
                mgl = np.mean((sig(K * sg * ours[m]) - y[m]) ** 2)
                if groups[0] == "material": tot_own += mo * m.sum()
                print("  %-18s %5d  %6.2f  %8.1f  %8.1f" % (c, m.sum(), s, mgl, mo))
        print("  per-material-class rescale vs one global scale: %.1f → %.1f (%+.1f%%)"
              % (mg, tot_own / len(ours), 100 * (tot_own / len(ours) / mg - 1)))


if __name__ == "__main__":
    main()
