# -*- coding: utf-8 -*-
"""Texel-style fit of per-term eval SCALE factors against Stockfish's static eval.

Model: total'(s) = our_total + sum_t (s_t - 1) * term_t, with `pieces` (material+PST) held fixed. We fit the
scales s_t to match SF's NNUE static eval through a logistic, which weights near-equal positions most (where
our eval's scatter actually costs games). Reports train vs held-out error and a per-stratum gap breakdown so
an extreme/overfit scale is visible before anything is wired into the engine.

The scales map directly to the engine's SCALE_* knobs (percent): ship s_t as round(100 * s_t).

Run from NN Engine/:
    python selfplay/tune_fit.py --corpus selfplay/tune_data/corpus.csv
    python selfplay/tune_fit.py --corpus selfplay/tune_data/corpus.csv --terms passed_pawn_support,latent_threat,central,capture_gains
"""

import sys
import csv
import argparse

import numpy as np

# Logistic scale in pawns: maps the eval to a win-probability-like value. Fixed (not fitted) to avoid the
# degenerate "flatten everything to 0.5" minimum; the before/after comparison is robust to the exact value.
K_PAWNS = 2.0
DEFAULT_TERMS = ["passed_pawn_support", "latent_threat", "central", "capture_gains"]
ALL_TERMS = ["pieces", "capture_gains", "passed_pawn_support", "latent_threat", "central",
             "imbalance_white", "imbalance_black", "pair_bonus", "piece_value_boost"]


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x / K_PAWNS))


def load(path):
    rows = list(csv.DictReader(open(path)))
    data = {
        "our_total": np.array([float(r["our_total"]) for r in rows]),
        "sf_cp": np.array([float(r["sf_static_cp"]) for r in rows]),
        "phase": np.array([int(r["phase_score"]) for r in rows]),
        "endgame": np.array([int(r["is_endgame"]) for r in rows]),
        "status": np.array([r["status"] for r in rows]),
        "result": np.array([float(r["result_white"]) for r in rows]),
    }
    for t in ALL_TERMS:
        data[t] = np.array([float(r[t]) for r in rows])
    return data


def white_pawns(total_abs):
    return -total_abs / 1000.0


def adjusted_total(data, idx, terms, scales):
    total = data["our_total"][idx].copy()
    for t, s in zip(terms, scales):
        total += (s - 1.0) * data[t][idx]
    return total


def target_sig(data, idx, mode):
    """The value our sigmoid(eval) should match: SF static (distillation) or the game result (Texel)."""
    if mode == "result":
        return data["result"][idx]
    return sigmoid(data["sf_cp"][idx] / 100.0)


def loss(data, idx, terms, scales, mode):
    ours = white_pawns(adjusted_total(data, idx, terms, scales))
    return float(np.mean((sigmoid(ours) - target_sig(data, idx, mode)) ** 2))


def fit(data, idx, terms, mode, lo=0.2, hi=3.0):
    """Coordinate descent on the scales (bounded). Simple, dependency-light, and convex enough here."""
    scales = np.ones(len(terms))
    best = loss(data, idx, terms, scales, mode)
    for _ in range(40):
        improved = False
        for j in range(len(terms)):
            for step in (0.25, 0.1, 0.03):
                for direction in (1, -1):
                    cand = scales.copy()
                    cand[j] = min(hi, max(lo, cand[j] + direction * step))
                    lc = loss(data, idx, terms, cand, mode)
                    if lc < best - 1e-9:
                        scales, best = cand, lc
                        improved = True
        if not improved:
            break
    return scales, best


def mae_pawns(data, idx, terms, scales, clip=6.0):
    ours = np.clip(white_pawns(adjusted_total(data, idx, terms, scales)), -clip, clip)
    target = np.clip(data["sf_cp"][idx] / 100.0, -clip, clip)
    return float(np.mean(np.abs(ours - target)))


def stratum_report(data, idx, terms, scales):
    out = []
    for eg, egname in ((0, "midgame"), (1, "endgame")):
        for st in ("near_equal", "white_winning", "black_winning"):
            sel = idx[(data["endgame"][idx] == eg) & (data["status"][idx] == st)]
            if len(sel) == 0:
                continue
            before = mae_pawns(data, sel, terms, np.ones(len(terms)))
            after = mae_pawns(data, sel, terms, scales)
            out.append((egname, st, len(sel), before, after))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="selfplay/tune_data/corpus.csv")
    ap.add_argument("--terms", default=",".join(DEFAULT_TERMS))
    ap.add_argument("--target", choices=["sf", "result"], default="sf")
    ap.add_argument("--lo", type=float, default=0.2)
    ap.add_argument("--hi", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--control-frac", type=float, default=0.3)
    args = ap.parse_args()

    terms = [t.strip() for t in args.terms.split(",") if t.strip()]
    data = load(args.corpus)
    n = len(data["our_total"])
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n)
    n_ctrl = int(n * args.control_frac)
    ctrl_idx, train_idx = perm[:n_ctrl], perm[n_ctrl:]
    mode = args.target

    print("corpus n=%d  train=%d  control=%d  terms=%s  target=%s  K=%.1f" % (n, len(train_idx), len(ctrl_idx), terms, mode, K_PAWNS))

    base = np.ones(len(terms))
    scales, train_loss = fit(data, train_idx, terms, mode, args.lo, args.hi)

    print("\nfitted scales (percent for the engine knob):")
    for t, s in zip(terms, scales):
        flag = "  <-- EXTREME" if (s > 2.0 or s < 0.5) else ""
        print("  %-22s %.2f  -> %d%s" % (t, s, round(100 * s), flag))

    print("\n%s-loss (lower=better):" % mode)
    print("  train    base %.5f -> fit %.5f" % (loss(data, train_idx, terms, base, mode), loss(data, train_idx, terms, scales, mode)))
    print("  control  base %.5f -> fit %.5f" % (loss(data, ctrl_idx, terms, base, mode), loss(data, ctrl_idx, terms, scales, mode)))

    print("\nMAE pawns (clip +-6, lower=better):")
    print("  train    base %.3f -> fit %.3f" % (mae_pawns(data, train_idx, terms, base), mae_pawns(data, train_idx, terms, scales)))
    print("  control  base %.3f -> fit %.3f" % (mae_pawns(data, ctrl_idx, terms, base), mae_pawns(data, ctrl_idx, terms, scales)))

    print("\nper-stratum MAE pawns (control split): phase  status  n  base -> fit")
    for egname, st, nsel, before, after in stratum_report(data, ctrl_idx, terms, scales):
        print("  %-8s %-14s n=%-5d  %.3f -> %.3f" % (egname, st, nsel, before, after))

    # Secondary cross-check: do the SF-fitted scales also align our eval better with actual game results?
    res = data["result"][ctrl_idx]
    for name, sc in (("base", base), ("fit", scales)):
        pred = sigmoid(white_pawns(adjusted_total(data, ctrl_idx, terms, sc)))
        print("  result-logloss (%s): %.4f" % (name, float(np.mean(-(res * np.log(pred + 1e-9) + (1 - res) * np.log(1 - pred + 1e-9))))))


if __name__ == "__main__":
    main()
