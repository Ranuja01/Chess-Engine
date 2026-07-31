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
             "imbalance_white", "imbalance_black", "pair_bonus", "piece_value_boost",
             "king_safety", "material", "pawn_majority", "pawn_struct", "outpost", "mobility"]
SF11_COLS = ["sf11_material", "sf11_imbalance", "sf11_mobility", "sf11_kingsafety", "sf11_threats",
             "sf11_passed", "sf11_space", "sf11_pawns", "sf11_knights", "sf11_bishops",
             "sf11_rooks", "sf11_queens", "sf11_total"]


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x / K_PAWNS))


def fit_k(result, ours_pawns, grid=None):
    """Standard Texel K-fit: the logistic slope (in pawns) that best maps our baseline eval -> win-prob,
    minimizing MSE vs the game RESULT. Well-posed for the OUTCOME target (unlike the scale-invariant SF-total
    target, where a strength-neutral global scale makes K degenerate -> that is why K_PAWNS was pinned). Fit on
    TRAIN only so the held-out split stays clean."""
    grid = np.arange(0.5, 6.01, 0.1) if grid is None else grid
    best_k, best = 2.0, 1e18
    for k in grid:
        pred = 1.0 / (1.0 + np.exp(-ours_pawns / k))
        l = float(np.mean((pred - result) ** 2))
        if l < best:
            best, best_k = l, k
    return best_k


def load(path):
    rows = list(csv.DictReader(open(path)))
    data = {
        "our_total": np.array([float(r["our_total"]) for r in rows]),
        "sf_cp": np.array([float(r["sf_static_cp"]) for r in rows]),
        "phase": np.array([int(r["phase_score"]) for r in rows]),
        "endgame": np.array([int(r["is_endgame"]) for r in rows]),
        "status": np.array([r["status"] for r in rows]),
        "result": np.array([float(r["result_white"]) for r in rows]),
        # game id for the by-GAME held-out split (older corpora lack it -> fall back to a by-position split).
        "game": np.array([r.get("game", "") for r in rows]),
    }
    for t in ALL_TERMS:
        if rows and t in rows[0]:                 # older corpora lack king_safety/material columns
            data[t] = np.array([float(r[t]) for r in rows])
    for c in SF11_COLS:                           # SF11 per-term breakdown (blank/"----" -> 0.0)
        if rows and c in rows[0]:
            data[c] = np.array([float(r[c]) if r[c] not in ("", "----") else 0.0 for r in rows])
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


# A global eval multiplier never changes move choice (argmax is scale-invariant). When --scale-inv is set,
# each candidate gets its OWN best global scale, so the per-term fit measures only RELATIVE re-balancing
# (strength-relevant) instead of partly absorbing a strength-neutral global magnitude mismatch vs SF11.
SCALE_GRID = np.arange(0.40, 1.205, 0.02)


def loss(data, idx, terms, scales, mode, scale_inv=False):
    ours = white_pawns(adjusted_total(data, idx, terms, scales))
    tgt = target_sig(data, idx, mode)
    if not scale_inv:
        return float(np.mean((sigmoid(ours) - tgt) ** 2))
    best = 1e18
    for s in SCALE_GRID:
        l = float(np.mean((sigmoid(s * ours) - tgt) ** 2))
        if l < best:
            best = l
    return best


def best_scale(data, idx, terms, scales, mode):
    ours = white_pawns(adjusted_total(data, idx, terms, scales))
    tgt = target_sig(data, idx, mode)
    bs, bl = 1.0, 1e18
    for s in SCALE_GRID:
        l = float(np.mean((sigmoid(s * ours) - tgt) ** 2))
        if l < bl:
            bl, bs = l, s
    return bs


def fit(data, idx, terms, mode, lo=0.2, hi=3.0, scale_inv=False):
    """Coordinate descent on the scales (bounded). Simple, dependency-light, and convex enough here."""
    scales = np.ones(len(terms))
    best = loss(data, idx, terms, scales, mode, scale_inv)
    for _ in range(40):
        improved = False
        for j in range(len(terms)):
            for step in (0.25, 0.1, 0.03):
                for direction in (1, -1):
                    cand = scales.copy()
                    cand[j] = min(hi, max(lo, cand[j] + direction * step))
                    lc = loss(data, idx, terms, cand, mode, scale_inv)
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
    ap.add_argument("--fit-k", action="store_true",
                    help="fit the logistic K to the game RESULT on train (outcome-Texel) instead of the pinned 2.0")
    ap.add_argument("--scale-inv", action="store_true",
                    help="scale-invariant: give each candidate its own best global scale so the fit measures "
                         "RELATIVE re-balancing (strength-relevant), not strength-neutral magnitude matching")
    ap.add_argument("--attrib", action="store_true",
                    help="per-term hotness: fit each term ALONE (scale-invariant) and rank by SF11-shape gain")
    ap.add_argument("--pin", default="",
                    help="comma list term=scale to BAKE into the baseline before fitting (e.g. capture_gains=0.7), "
                         "so the remaining terms are attributed against the CALIBRATED eval, not the raw one")
    ap.add_argument("--corr", action="store_true",
                    help="correspondence: correlate our terms + the residual gap against SF11's per-term "
                         "breakdown (STATIC placement vs DYNAMIC mobility/threats) to see what a term really tracks")
    args = ap.parse_args()

    terms = [t.strip() for t in args.terms.split(",") if t.strip()]
    data = load(args.corpus)
    if args.pin:
        for kv in args.pin.split(","):
            t, v = kv.split("="); t, v = t.strip(), float(v)
            data["our_total"] = data["our_total"] + (v - 1.0) * data[t]   # bake the pin into the baseline eval
            if t in terms:
                terms.remove(t)
        print("pinned baseline: %s" % args.pin)
    n = len(data["our_total"])
    rng = np.random.default_rng(args.seed)
    mode = args.target
    # Held-out control split BY GAME when game ids are present (positions from one game share the outcome label
    # and are near-duplicates -> a by-position split leaks the holdout; by-game does not). Fall back to by-
    # position only for older corpora without a game column.
    uniq = sorted(g for g in set(data["game"]) if g)
    if len(uniq) > 5:
        uarr = np.array(uniq)
        rng.shuffle(uarr)
        n_ctrl_g = max(1, int(len(uarr) * args.control_frac))
        ctrl_games = set(uarr[:n_ctrl_g].tolist())
        is_ctrl = np.array([g in ctrl_games for g in data["game"]])
        ctrl_idx = np.where(is_ctrl)[0]
        train_idx = np.where(~is_ctrl)[0]
        split_kind = "by-game (%d games, %d control)" % (len(uniq), len(ctrl_games))
    else:
        perm = rng.permutation(n)
        n_ctrl = int(n * args.control_frac)
        ctrl_idx, train_idx = perm[:n_ctrl], perm[n_ctrl:]
        split_kind = "by-position (no game ids in corpus)"

    if args.fit_k:
        if mode != "result":
            print("WARNING: --fit-k only meaningful with --target result; ignoring")
        else:
            global K_PAWNS
            K_PAWNS = fit_k(data["result"][train_idx], white_pawns(data["our_total"][train_idx]))
            print("fitted K = %.2f pawns (on train, vs game result)" % K_PAWNS)

    print("corpus n=%d  train=%d  control=%d  split=%s  terms=%s  target=%s  K=%.2f"
          % (n, len(train_idx), len(ctrl_idx), split_kind, terms, mode, K_PAWNS))

    base = np.ones(len(terms))
    si = args.scale_inv

    if args.corr:
        # Correspondence: is our `pieces` under-read really PLACEMENT, or OvD/mobility in disguise? Our terms
        # are Black-positive milli-pawns -> White-POV pawns = -v/1000; SF11 terms are already White-POV pawns.
        wp = lambda a: -a / 1000.0
        def corr(a, b):
            if np.std(a) < 1e-9 or np.std(b) < 1e-9:
                return 0.0
            return float(np.corrcoef(a, b)[0, 1])
        sfcols = [c for c in SF11_COLS if c in data and c != "sf11_total"]
        # SF11 bundles: STATIC placement (per-piece + pawns + space) vs DYNAMIC (mobility + threats).
        sf_static = sum(data[c] for c in ["sf11_pawns", "sf11_knights", "sf11_bishops", "sf11_rooks",
                                          "sf11_queens", "sf11_space"] if c in data)
        sf_dyn = sum(data[c] for c in ["sf11_mobility", "sf11_threats"] if c in data)
        our_terms = {t: wp(data[t]) for t in ["pieces", "central", "latent_threat", "piece_value_boost",
                                              "capture_gains"] if t in data}
        our_terms["imbalance_net"] = wp(data["imbalance_black"] - data["imbalance_white"])
        residual = data["sf11_total"] - wp(data["our_total"])   # + => SF reads higher (we UNDER-read here)

        print("\ncorr(our term, SF11 term)  [%s]:" % ("capg pinned" if args.pin else "raw"))
        print("  %-16s %8s %8s | " % ("", "STATIC", "DYNAMIC")
              + " ".join("%6s" % c.replace("sf11_", "")[:6] for c in sfcols))
        for name, a in our_terms.items():
            print("  %-16s %8.2f %8.2f | " % (name, corr(a, sf_static), corr(a, sf_dyn))
                  + " ".join("%6.2f" % corr(a, data[c]) for c in sfcols))
        print("\ncorr(residual gap [SF - ours, + = we under-read], SF11 term):")
        print("  %-16s %8.2f %8.2f | " % ("residual", corr(residual, sf_static), corr(residual, sf_dyn))
              + " ".join("%6.2f" % corr(residual, data[c]) for c in sfcols))
        print("\n  -> our `pieces` aligning with STATIC (not DYNAMIC) = genuine placement; with DYNAMIC = OvD/mobility mask")
        return

    if args.attrib:
        # Per-term hotness: fit each term ALONE (scale-invariant), so each term's over/under-read is
        # isolated from cross-term collinearity. scale<1 => HOT (we over-read vs SF11); >1 => COLD.
        print("\nper-term hotness (solo scale-invariant fit vs SF11):  term  scale  control-loss base->fit")
        rows = []
        for t in terms:
            sc, _ = fit(data, train_idx, [t], mode, args.lo, args.hi, True)
            lb = loss(data, ctrl_idx, [t], np.ones(1), mode, True)
            lf = loss(data, ctrl_idx, [t], sc, mode, True)
            rows.append((t, float(sc[0]), lb, lf, lb - lf))
        for t, s, lb, lf, d in sorted(rows, key=lambda r: r[4], reverse=True):
            tag = "HOT (over-read)" if s < 0.9 else ("COLD (under-read)" if s > 1.1 else "~ok")
            print("  %-22s %.2f   %.5f -> %.5f  (Δ%.5f)  %s" % (t, s, lb, lf, d, tag))
        return

    scales, train_loss = fit(data, train_idx, terms, mode, args.lo, args.hi, si)

    if si:
        print("[scale-invariant: best global scale base s=%.2f -> fit s=%.2f (strength-neutral, discarded)]"
              % (best_scale(data, ctrl_idx, terms, base, mode), best_scale(data, ctrl_idx, terms, scales, mode)))
    print("\nfitted scales (percent for the engine knob):")
    for t, s in zip(terms, scales):
        flag = "  <-- EXTREME" if (s > 2.0 or s < 0.5) else ""
        print("  %-22s %.2f  -> %d%s" % (t, s, round(100 * s), flag))

    print("\n%s-loss%s (lower=better):" % (mode, " scale-inv" if si else ""))
    print("  train    base %.5f -> fit %.5f" % (loss(data, train_idx, terms, base, mode, si), loss(data, train_idx, terms, scales, mode, si)))
    print("  control  base %.5f -> fit %.5f" % (loss(data, ctrl_idx, terms, base, mode, si), loss(data, ctrl_idx, terms, scales, mode, si)))

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
