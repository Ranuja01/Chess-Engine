# -*- coding: utf-8 -*-
"""Disciplined OUTCOME-Texel fit — the joint eval recalibration that was designed but never run
(memory: outcome-texel-campaign). Fits per-term eval SCALE factors s_t to predict GAME OUTCOMES with a
FITTED K (kills the scale-invariant "shrink everything" confound that killed the SF-total fits).

Model (our eval is Black-positive milli-pawns; corpus columns from tune_corpus):
    total'(s) = our_total + Σ_t (s_t - 1) * term_t          # re-weight the tunable terms
    E_white   = -total'(s) / 1000                            # White-POV pawns
    P(white)  = 1 / (1 + exp(-E_white / K))                  # logistic, K in pawns
Loss = mean logloss(P, result_white) + LAMBDA * Σ (s_t-1)^2  # Δw L2 toward current hand-values (strong prior).

Discipline (from the campaign note): fit K ALONE (s=1) -> freeze -> fit s (±25% trust region) -> refit K once.
PIN material/pieces (the unit; don't let material re-express global scale). KEEP-OUT tactical terms
(capture_gains/latent_threat/king_safety/threats -> they stay on the gated-knob + lightning path). By-GAME
holdout (positions from one game share the outcome label -> a by-position split leaks). The HELD-OUT logloss
delta is the honest signal: negative = the re-weighting GENERALIZES.

Run (WSL dispatcher):
    pyrun selfplay/outcome_texel.py --corpus selfplay/tune_data/outcome_corpus.csv
"""
import sys
import csv
import argparse

import numpy as np
from scipy.optimize import minimize

CORPUS_TERMS = ["material", "pieces", "capture_gains", "passed_pawn_support", "latent_threat", "threats",
                "king_safety", "central", "imbalance_white", "imbalance_black", "pair_bonus",
                "piece_value_boost", "pawn_majority", "pawn_struct", "outpost", "mobility",
                "pt_pawns", "pt_knights", "pt_bishops", "pt_rooks", "pt_queens", "pt_kings"]
# 'pieces' (=material+PST bulk) PINNED so scaling pt_* adds only the placement DELTA on top (no double-count).
PIN = {"material", "pieces"}                                   # the unit — do not fit
KEEPOUT = {"capture_gains", "latent_threat", "king_safety", "threats"}   # tactical -> gated-knob path


def sigmoid(e, k):
    return 1.0 / (1.0 + np.exp(-e / k))


def logloss(p, y, w=None):
    p = np.clip(p, 1e-9, 1.0 - 1e-9)
    ll = y * np.log(p) + (1.0 - y) * np.log(1.0 - p)
    if w is None:
        return float(-np.mean(ll))
    return float(-np.sum(w * ll) / np.sum(w))


def fit_k(e_base, y, w=None, grid=None):
    grid = np.arange(0.5, 24.01, 0.1) if grid is None else grid
    best_k, best = 2.0, 1e18
    for k in grid:
        l = logloss(sigmoid(e_base, k), y, w)
        if l < best:
            best, best_k = l, k
    return best_k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--lam", type=float, default=0.02, help="Δw L2 strength (toward s=1)")
    ap.add_argument("--tr", type=float, default=0.25, help="trust region: |s-1| <= tr")
    ap.add_argument("--holdout", type=float, default=0.2, help="fraction of GAMES held out")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.corpus)))
    if not rows:
        print("empty corpus"); return
    present = [t for t in CORPUS_TERMS if t in rows[0]]
    fit_terms = [t for t in present if t not in PIN and t not in KEEPOUT]

    def col(name):
        return np.array([float(r[name]) for r in rows], dtype=float)

    y = col("result_white")
    our_total = col("our_total")
    w_all = col("weight") if "weight" in rows[0] else np.ones(len(rows))
    Tmat = np.stack([col(t) for t in fit_terms], axis=1)   # (N,F) engine units
    games = np.array([r["game"] for r in rows])

    # by-GAME holdout
    rng = np.random.RandomState(args.seed)
    uniq = np.array(sorted(set(games.tolist())))
    rng.shuffle(uniq)
    n_hold = max(1, int(len(uniq) * args.holdout))
    hold = set(uniq[:n_hold].tolist())
    is_hold = np.array([g in hold for g in games])
    tr, ho = ~is_hold, is_hold

    e_white = -our_total / 1000.0
    K = fit_k(e_white[tr], y[tr], w_all[tr])
    base_tr = logloss(sigmoid(e_white[tr], K), y[tr], w_all[tr])
    base_ho = logloss(sigmoid(e_white[ho], K), y[ho], w_all[ho])

    def total_wpov(s):
        return -(our_total + Tmat.dot(s - 1.0)) / 1000.0

    def loss(s):
        e = total_wpov(s)[tr]
        return logloss(sigmoid(e, K), y[tr], w_all[tr]) + args.lam * float(np.sum((s - 1.0) ** 2))

    s0 = np.ones(len(fit_terms))
    res = minimize(loss, s0, method="L-BFGS-B",
                   bounds=[(1.0 - args.tr, 1.0 + args.tr)] * len(fit_terms))
    s = res.x
    K2 = fit_k(total_wpov(s)[tr], y[tr], w_all[tr])          # refit K once with fitted weights
    fit_tr = logloss(sigmoid(total_wpov(s)[tr], K2), y[tr], w_all[tr])
    fit_ho = logloss(sigmoid(total_wpov(s)[ho], K2), y[ho], w_all[ho])

    print("N=%d games=%d (holdout %d) fit_terms=%d  lam=%.3g tr=%.2f"
          % (len(rows), len(uniq), n_hold, len(fit_terms), args.lam, args.tr))
    print("K: baseline=%.2f refit=%.2f" % (K, K2))
    print("logloss TRAIN base=%.5f fit=%.5f  d=%+.5f" % (base_tr, fit_tr, fit_tr - base_tr))
    print("logloss HOLD  base=%.5f fit=%.5f  d=%+.5f   <-- negative = generalizes (real signal)"
          % (base_ho, fit_ho, fit_ho - base_ho))
    print("\nfitted scales (s>1 = UNDER-weighted term wants MORE; s<1 = OVER-weighted wants LESS):")
    for t, sv in sorted(zip(fit_terms, s), key=lambda kv: -abs(kv[1] - 1.0)):
        print("  %-22s %.3f  (%+d%%)" % (t, sv, round(100 * (sv - 1.0))))


if __name__ == "__main__":
    main()
