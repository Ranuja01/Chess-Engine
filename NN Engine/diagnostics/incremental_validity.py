# -*- coding: utf-8 -*-
"""Incremental validity of each SF11 feature over OUR eval, for predicting GAME OUTCOMES (Fable's compass).

The flat retune proved our feature BASIS is at its outcome-optimum (reweighting can't help). The open question:
is a MISSING feature (e.g. SF's king-safety) outcome-predictive that our basis can't synthesize? Test it directly
without building anything: for each SF11 feature f, fit sigmoid(a*our_eval + b*f) vs the game result and measure
the held-out result-loss (Brier/MSE) DROP vs the baseline sigmoid(a*our_eval). A drop well above the retune
flatness floor (~0.00003) = f carries outcome signal our eval lacks = a real Elo lever (SF proposes, OUTCOMES
screen, games ratify). This is the "re-point the apparatus onto outcome-predictiveness" tool.

Usage:  python diagnostics/incremental_validity.py [corpus.csv]   (needs SF11-labelled corpus)
"""
import csv
import os
import sys

import numpy as np
from scipy.optimize import minimize

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORPUS = sys.argv[1] if len(sys.argv) > 1 else os.path.join(BASE, "selfplay", "tune_data", "cond_corpus_v2.csv")

SF_FEATURES = ["sf11_kingsafety", "sf11_threats", "sf11_mobility", "sf11_space", "sf11_passed",
               "sf11_material", "sf11_imbalance", "sf11_pawns", "sf11_rooks"]


def col(rows, name):
    return np.array([float(r[name]) if r.get(name, "") not in ("", "----", None) else 0.0 for r in rows])


def sig(z):
    return 1.0 / (1.0 + np.exp(-z))


def fit_loss(X, r, tr, te):
    """Fit sigmoid(X @ w) to result r by MSE on train `tr`, return held-out MSE on `te`."""
    def mse(w, idx):
        return float(np.mean((sig(X[idx] @ w) - r[idx]) ** 2))
    w0 = np.zeros(X.shape[1]); w0[0] = 0.4       # ~1/K start on the our_eval column
    res = minimize(lambda w: mse(w, tr), w0, method="Nelder-Mead",
                   options={"maxiter": 4000, "xatol": 1e-5, "fatol": 1e-9})
    return mse(res.x, te), res.x


def main():
    rows = list(csv.DictReader(open(CORPUS)))
    have = set(rows[0].keys()) if rows else set()
    n = len(rows)
    # If the corpus was generated with our candidate term ON, our_total INCLUDES it -> subtract so the baseline is
    # term-free and we can measure OUR term's incremental validity (vs the SF11 column = the ceiling). The term to
    # isolate is argv[2] (default "threats"): e.g. "king_safety" for the KS re-adjudication screen.
    OUR_TERM = sys.argv[2] if len(sys.argv) > 2 else "threats"
    our_col = col(rows, OUR_TERM) if OUR_TERM in have else np.zeros(n)
    our = -(col(rows, "our_total") - our_col) / 1000.0           # White-POV pawns, candidate-free baseline
    r = col(rows, "result_white")
    ones = np.ones(n)
    games = np.array([row.get("game", "") for row in rows])
    uniq = sorted(set(g for g in games if g))
    by_game = len(uniq) > 5
    FOLDS = 4 if by_game else 1
    print("corpus %s  n=%d  split=%s%s" % (os.path.basename(CORPUS), n,
          "by-game (%d games)" % len(uniq) if by_game else "by-position",
          ", %d folds (sign-stability)" % FOLDS if by_game else ""))
    print("(retune flatness floor from the full joint fit was ~0.00003)\n")

    feats = [f for f in SF_FEATURES if f in have and np.std(col(rows, f)) > 1e-9]
    fvals = {f: col(rows, f) for f in feats}
    if OUR_TERM in have and np.std(our_col) > 1e-9:           # OUR impl vs the sf11 ceiling for the same feature
        label = "our_" + OUR_TERM
        feats = [label] + feats
        fvals[label] = -our_col / 1000.0                      # Black-positive milli-pawns -> White-POV pawns
    deltas = {f: [] for f in feats}   # per-fold held-out Δ
    signs = {f: [] for f in feats}    # per-fold coefficient sign on f
    base_losses = []
    for fold in range(FOLDS):
        rng = np.random.default_rng(7 + fold)
        if by_game:
            ug = np.array(uniq); rng.shuffle(ug)
            ctrl_games = set(ug[: max(1, int(len(ug) * 0.3))].tolist())
            is_te = np.array([g in ctrl_games for g in games])
            te, tr = np.where(is_te)[0], np.where(~is_te)[0]
        else:
            perm = rng.permutation(n); te, tr = perm[: n // 3], perm[n // 3:]
        base_te, _ = fit_loss(np.column_stack([our, ones]), r, tr, te)
        base_losses.append(base_te)
        for f in feats:
            aug_te, w = fit_loss(np.column_stack([our, fvals[f], ones]), r, tr, te)
            deltas[f].append(base_te - aug_te)
            signs[f].append(np.sign(w[1]))
    print("baseline held-out result-loss (our eval only) = %.6f\n" % np.mean(base_losses))
    print("%-14s %11s %9s %8s   %s" % ("+ SF11 feat", "mean-delta", "std", "sign-ok", "verdict"))
    ranked = sorted(feats, key=lambda f: -np.mean(deltas[f]))
    for f in ranked:
        md, sd = float(np.mean(deltas[f])), float(np.std(deltas[f]))
        sign_ok = len(set(signs[f])) == 1                     # coefficient sign stable across folds
        verdict = ("REAL LEVER" if md > 0.0005 and sign_ok
                   else "unstable" if md > 0.0005 and not sign_ok
                   else "marginal" if md > 0.0001 else "no signal")
        print("%-14s %+11.6f %9.6f %8s   %s" % (f.replace("sf11_", ""), md, sd, "yes" if sign_ok else "NO", verdict))


if __name__ == "__main__":
    main()
