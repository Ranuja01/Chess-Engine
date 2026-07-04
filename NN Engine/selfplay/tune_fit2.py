#!/usr/bin/env python3
"""Disciplined outcome-Texel fit (Tier 1a: per-term SCALE multipliers).

Adds the guardrails the naive tune_fit.py lacks, which the 37k-corpus run showed are necessary
(fixed K=2.0 blunt-shrinks 7/11 terms to the bounds as a scale proxy):
  - FITTED K  -> absorbs the global eval-hotness so terms are not shrunk to express scale.
  - lambda-blend label = lam*game_result + (1-lam)*SF_WDL(sf_static_cp).
  - RIDGE toward the current hand values (scale=1) + per-round TRUST region.
Fits per-term scale multipliers on the PRECOMPUTED corpus (no engine call, no rebuild). Raw-constant
fitting (Tier 1b, the bigger Fable prize) needs a regenerated per-parameter corpus + a rebuild.

Split is by POSITION (cond_corpus_v2 has no game_id) -> a leakage caveat; regenerate the corpus with
a game_id column for a leakage-free game/time split before trusting any shipped scales.
"""
import csv, argparse
import numpy as np


def load(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    data = {}
    for c in rows[0].keys():
        try:
            data[c] = np.array([float(x[c]) for x in rows])
        except ValueError:
            pass  # non-numeric column (fen, status) -> skip
    data["_n"] = len(rows)
    data["_status"] = np.array([x.get("status", "") for x in rows])
    data["_phase"] = data.get("phase_score", np.zeros(len(rows)))
    return data


def white_pawns(total_abs):
    return -total_abs / 1000.0


def adjusted_total(data, idx, terms, scales):
    total = data["our_total"][idx].copy()
    for t, s in zip(terms, scales):
        total += (s - 1.0) * data[t][idx]
    return total


def sigmoid(x, K):
    return 1.0 / (1.0 + np.exp(-x / K))


def make_buckets(data, terms, mode):
    """Split each term into conditioning buckets (phase or status) using EXACT existing columns.
    A term T is replaced by T@a,T@b where T@a = T*mask_a etc. (T@a+T@b == T -> baseline unchanged at
    all-scales-1). Tests whether a term wants a different weight per phase/status = Fable's tier-2
    conditioning, done at term granularity with no featurization risk."""
    if mode == "none":
        return terms
    out = []
    if mode == "phase":
        mid = (data["phase_score"] <= 64).astype(float)
        masks = {"mid": mid, "end": 1.0 - mid}
    elif mode == "status":
        ne = (data["_status"] == "near_equal").astype(float)
        masks = {"eq": ne, "dec": 1.0 - ne}
    else:
        raise SystemExit("unknown --split-terms %s" % mode)
    for t in terms:
        for name, m in masks.items():
            col = "%s@%s" % (t, name)
            data[col] = data[t] * m
            out.append(col)
    return out


def label_blend(data, idx, lam, sf_wdl_k):
    res = data["result_white"][idx]
    sf_wdl = sigmoid(data["sf_static_cp"][idx] / 100.0, sf_wdl_k)
    return lam * res + (1.0 - lam) * sf_wdl


def data_loss(data, idx, terms, scales, K, label):
    ours = white_pawns(adjusted_total(data, idx, terms, scales))
    return float(np.mean((sigmoid(ours, K) - label) ** 2))


def total_loss(data, idx, terms, scales, K, label, ridge):
    return data_loss(data, idx, terms, scales, K, label) + ridge * float(np.sum((np.asarray(scales) - 1.0) ** 2))


def fit_K(data, idx, terms, scales, label, lo=0.5, hi=6.0):
    """1-D search for the global logistic scale with weights frozen (the anti-hijack step)."""
    best_K, best = lo, 1e18
    for K in np.arange(lo, hi + 1e-9, 0.05):
        l = data_loss(data, idx, terms, scales, K, label)
        if l < best:
            best, best_K = l, float(K)
    return best_K, best


def fit_scales(data, idx, terms, K, label, ridge, lo, hi, trust, start):
    scales = np.array(start, dtype=float)
    round_start = scales.copy()
    base = total_loss(data, idx, terms, scales, K, label, ridge)
    for _ in range(40):
        improved = False
        for j in range(len(terms)):
            for step in (0.25, 0.1, 0.03):
                for d in (1, -1):
                    cand = scales.copy()
                    v = cand[j] + d * step
                    v = min(round_start[j] + trust, max(round_start[j] - trust, v))  # trust region
                    v = min(hi, max(lo, v))                                          # hard bounds
                    cand[j] = v
                    lc = total_loss(data, idx, terms, cand, K, label, ridge)
                    if lc < base - 1e-9:
                        scales, base = cand, lc
                        improved = True
        if not improved:
            break
    return scales, base


def logloss(data, idx, terms, scl, K, lbl):
    p = np.clip(sigmoid(white_pawns(adjusted_total(data, idx, terms, scl)), K), 1e-9, 1 - 1e-9)
    return float(np.mean(-(lbl * np.log(p) + (1 - lbl) * np.log(1 - p))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--terms", required=True)
    ap.add_argument("--lam", type=float, default=1.0, help="1=pure game-result, 0=pure SF-WDL")
    ap.add_argument("--sf-wdl-k", type=float, default=2.0)
    ap.add_argument("--ridge", type=float, default=0.02)
    ap.add_argument("--trust", type=float, default=0.35)
    ap.add_argument("--lo", type=float, default=0.2)
    ap.add_argument("--hi", type=float, default=3.0)
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--control-frac", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--split-terms", default="none", choices=["none", "phase", "status"],
                    help="condition each term into buckets (phase: mid/end, status: eq/dec) and fit separately")
    args = ap.parse_args()

    terms = [t.strip() for t in args.terms.split(",") if t.strip()]
    data = load(args.corpus)
    terms = make_buckets(data, terms, args.split_terms)
    n = data["_n"]
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n)
    n_ctrl = int(n * args.control_frac)
    ctrl_idx, train_idx = perm[:n_ctrl], perm[n_ctrl:]

    lbl_tr = label_blend(data, train_idx, args.lam, args.sf_wdl_k)
    lbl_ct = label_blend(data, ctrl_idx, args.lam, args.sf_wdl_k)

    print("corpus n=%d train=%d control=%d lam=%.2f ridge=%.3f trust=%.2f rounds=%d" %
          (n, len(train_idx), len(ctrl_idx), args.lam, args.ridge, args.trust, args.rounds))
    print("terms=%s" % terms)
    print("split=POSITION (leakage caveat -- regen corpus w/ game_id for a leakage-free split)")

    scales, K = np.ones(len(terms)), 2.0
    for rd in range(args.rounds):
        K, _ = fit_K(data, train_idx, terms, scales, lbl_tr)                                  # freeze weights, fit K
        scales, sl = fit_scales(data, train_idx, terms, K, lbl_tr, args.ridge, args.lo, args.hi, args.trust, scales)  # freeze K, fit weights
        print("  round %d: K=%.2f  train_loss(+ridge)=%.5f" % (rd + 1, K, sl))

    print("\nfitted scales (percent for the engine SCALE_* knob):")
    for t, s in zip(terms, scales):
        flag = "  <-- BOUND" if (s <= args.lo + 1e-6 or s >= args.hi - 1e-6) else ""
        print("  %-22s %.2f -> %d%s" % (t, s, round(100 * s), flag))
    print("\nfitted K = %.2f   (naive baseline used fixed K=2.0)" % K)

    print("\nheld-out (control) MSE:      base(scale1,K2) %.5f -> fit %.5f" %
          (data_loss(data, ctrl_idx, terms, np.ones(len(terms)), 2.0, lbl_ct),
           data_loss(data, ctrl_idx, terms, scales, K, lbl_ct)))
    print("held-out (control) logloss:  base %.4f -> fit %.4f" %
          (logloss(data, ctrl_idx, terms, np.ones(len(terms)), 2.0, lbl_ct),
           logloss(data, ctrl_idx, terms, scales, K, lbl_ct)))


if __name__ == "__main__":
    main()
