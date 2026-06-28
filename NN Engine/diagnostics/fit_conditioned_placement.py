# -*- coding: utf-8 -*-
"""Rich per-piece-type detector-conditioned placement fit — the fair test of the conditioned-placement idea.

Each piece-type's placement contribution pt_j gets its own gain g_j(detectors) = 1 + sum_k w_jk d_k. The
new eval is our + sum_j (g_j - 1) * pt_j. We fit all w_jk by (ridge) least squares to minimize the our-vs-SF
gap on the curated important-midgame corpus, train/test split, and compare held-out gap MSE for:

  (0) baseline           : g_j = 1                     (static placement = today)
  (1) per-piece FLAT      : g_j = const_j               (each piece-type its own scale; no detectors)
  (2) per-piece CONDITIONED: g_j = 1 + w_j . detectors  (the full idea)

The decisive number is (2) BEYOND (1): if conditioning each piece-type on detectors meaningfully beats just
re-scaling each piece-type, the placement variance is detector-explainable -> worth the C++ build. If not,
it's irreducible scatter (NNUE territory) and a global/per-piece scale is all that's available.

Run (Windows or WSL python, no engine/SF):  python diagnostics/fit_conditioned_placement.py
"""
import os
import csv
import numpy as np

CORPUS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "selfplay", "tune_data", "midgame_corpus.csv")
PT = ["pt_pawns", "pt_knights", "pt_bishops", "pt_rooks", "pt_queens", "pt_kings"]
RIDGE = 1.0


def main():
    rows = list(csv.DictReader(open(CORPUS)))
    feat = [c for c in rows[0].keys() if c not in (["fen", "phase_score", "sf_static_cp", "our_total", "pieces"] + PT)]
    E, sf, PTcp, D = [], [], [], []
    for r in rows:
        E.append(-float(r["our_total"]) / 10.0)              # our eval White-POV cp
        sf.append(float(r["sf_static_cp"]))                  # SF static White-POV cp
        PTcp.append([-float(r[j]) / 10.0 for j in PT])       # each piece-type placement, White-POV cp
        D.append([float(r[k]) for k in feat])
    E, sf, PTcp, D = map(np.asarray, (E, sf, PTcp, D))
    gap = E - sf
    n = len(gap)
    print(f"[fit] {n} important-midgame positions, {len(feat)} detectors, {len(PT)} piece-types")
    print(f"[fit] baseline gap: mean {gap.mean():+.1f} std {gap.std():.1f} MSE {np.mean(gap**2):.0f}")
    print(f"[fit] mean |pt| (cp): " + "  ".join(f"{PT[j][3:]}={np.abs(PTcp[:,j]).mean():.0f}" for j in range(len(PT))))

    rng = np.random.default_rng(0)
    idx = rng.permutation(n); cut = int(0.7 * n); tr, te = idx[:cut], idx[cut:]

    def fit(cols):
        # cols: (n, m) design; minimize ||gap + cols w||^2 + RIDGE||w||^2 ; new_gap = gap + cols w
        A = cols[tr]
        G = A.T @ A + RIDGE * np.eye(A.shape[1])
        w = np.linalg.solve(G, -A.T @ gap[tr])
        return np.mean((gap[te] + cols[te] @ w) ** 2), w

    base = np.mean(gap[te] ** 2)
    # (1) per-piece flat: one column per piece-type = pt_j (its own scale)
    flat_cols = PTcp
    flat_mse, _ = fit(flat_cols)
    # (2) per-piece conditioned: columns = pt_j * d_k for every (j,k), plus pt_j (the const term)
    cond_list = [PTcp]                                       # per-piece const (== flat)
    for k in range(len(feat)):
        cond_list.append(PTcp * D[:, [k]])                  # pt_j * detector_k interactions
    cond_cols = np.column_stack(cond_list)
    cond_mse, w = fit(cond_cols)

    print(f"\n[fit] held-out gap MSE (lower = better):")
    print(f"   (0) baseline static placement   : {base:8.0f}")
    print(f"   (1) per-piece FLAT scale         : {flat_mse:8.0f}  ({100*(1-flat_mse/base):+.1f}% vs baseline)")
    print(f"   (2) per-piece CONDITIONED        : {cond_mse:8.0f}  ({100*(1-cond_mse/base):+.1f}% vs baseline)")
    print(f"   >> CONDITIONING beyond per-piece flat: {100*(1-cond_mse/flat_mse):+.1f}%  (the value of the idea)")

    # Which (piece-type x detector) interactions carry weight? (skip the first len(PT) const columns)
    inter = w[len(PT):].reshape(len(feat), len(PT))
    print(f"\n[fit] strongest piece x detector interactions (|coef| on pt_j*d_k):")
    flat = [(abs(inter[k, j]), feat[k], PT[j][3:], inter[k, j]) for k in range(len(feat)) for j in range(len(PT))]
    for mag, fk, pj, v in sorted(flat, reverse=True)[:12]:
        print(f"     {pj:8} x {fk:14} {v:+.3f}")


if __name__ == "__main__":
    main()
