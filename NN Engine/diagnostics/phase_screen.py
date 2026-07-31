# -*- coding: utf-8 -*-
"""Phase-stratified incremental validity: does SF's king-safety / threats carry OUTCOME signal in the ENDGAME, or
only midgame? Answers the per-function phase-gating question from data (KS should gate OFF in the late endgame
where an active king is GOOD; immediate threats are real all-phase). Δ = held-out result-loss drop from adding the
feature to sigmoid(a·our_eval), by-game, computed separately on midgame (is_endgame=0) and endgame (=1) subsets.

Usage: python diagnostics/phase_screen.py <sf11-labelled corpus.csv>
"""
import csv, os, sys
import numpy as np
from scipy.optimize import minimize

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORPUS = sys.argv[1] if len(sys.argv) > 1 else os.path.join(BASE, "selfplay", "tune_data", "threats_corpus_v2.csv")
FEATURES = ["sf11_kingsafety", "sf11_threats", "sf11_mobility", "our_threats"]


def col(rows, n):
    return np.array([float(r[n]) if r.get(n, "") not in ("", "----", None) else 0.0 for r in rows])


def sig(z):
    return 1.0 / (1.0 + np.exp(-z))


def heldout(X, r, tr, te):
    res = minimize(lambda w: float(np.mean((sig(X[tr] @ w) - r[tr]) ** 2)),
                   np.array([0.4] + [0.0] * (X.shape[1] - 1)), method="Nelder-Mead",
                   options={"maxiter": 4000, "xatol": 1e-5, "fatol": 1e-9})
    return float(np.mean((sig(X[te] @ res.x) - r[te]) ** 2))


def screen(rows, label):
    n = len(rows)
    if n < 500:
        print("\n[%s] n=%d (too small)" % (label, n)); return
    threats_col = col(rows, "threats") if "threats" in rows[0] else np.zeros(n)
    our = -(col(rows, "our_total") - threats_col) / 1000.0
    r = col(rows, "result_white")
    ones = np.ones(n)
    games = np.array([row.get("game", "") for row in rows])
    uniq = sorted(set(g for g in games if g))
    fvals = {}
    for f in FEATURES:
        if f == "our_threats":
            fvals[f] = -threats_col / 1000.0
        elif f in rows[0]:
            fvals[f] = col(rows, f)
    deltas = {f: [] for f in fvals}
    for fold in range(4):
        rng = np.random.default_rng(7 + fold)
        ug = np.array(uniq); rng.shuffle(ug)
        ctrl = set(ug[: max(1, int(len(ug) * 0.3))].tolist())
        te = np.array([g in ctrl for g in games])
        te, tr = np.where(te)[0], np.where(~te)[0]
        base = heldout(np.column_stack([our, ones]), r, tr, te)
        for f, v in fvals.items():
            if np.std(v) < 1e-9:
                deltas[f].append(0.0); continue
            deltas[f].append(base - heldout(np.column_stack([our, v, ones]), r, tr, te))
    print("\n[%s] n=%d  (Δ held-out result-loss from adding each feature; + = outcome-predictive)" % (label, n))
    for f in FEATURES:
        if f in deltas:
            print("  %-16s %+.6f ± %.6f" % (f.replace("sf11_", ""), np.mean(deltas[f]), np.std(deltas[f])))


def main():
    rows = list(csv.DictReader(open(CORPUS)))
    eg = lambda r: r.get("is_endgame") in ("1", 1)
    print("corpus %s  n=%d" % (os.path.basename(CORPUS), len(rows)))
    screen([r for r in rows if not eg(r)], "MIDGAME")
    screen([r for r in rows if eg(r)], "ENDGAME")


if __name__ == "__main__":
    main()
