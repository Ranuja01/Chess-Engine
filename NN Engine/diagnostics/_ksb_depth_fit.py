# -*- coding: utf-8 -*-
"""KS-B (shelter + storm, C3-a) cells fitted on the DEPTH target — the first step of the full KS tune (owner, 10-02).

Why: the triangulation aggregate (111 quiet equal-material cases, dev_notes/TRIANGULATION-2026-10-02.md) found
STRUCTURAL king safety the consistent statically-expressible miss (SF11's KS helps 48 vs 17, +24 cp mean; ours reads 0
where a king is exposed by STRUCTURE). KS-B is built (eval_v2.cpp ksb_side, cells 106-161 of v2_features) but its d6
OUTCOME-fitted values lost at depth (−22, C3 doc §14). Here the target is depth-independent and search-aware:
  ours' = ours_d10 (our d10 SEARCH, White cp) + Δ_KSB,   Δ_KSB = −Σ_k diff_k · θ_k · phase/256 / 10
  (diff = Black − White cell counts from fitC_features.npz; θ in mp, mg leg only — SF puts shelter/storm in mg)
Loss: mean (win%(ours') − win%(SF18 d14))², L2 on θ, val by game-hash 15%. STM nuisance fitted, not shipped.
Output: the engine's KSB_V2_FILE format (`k leg start fitted`, leg 0 = mg), loaded by `KSB_V2=1 KSB_V2_FILE=...`.

  pyrun diagnostics/_ksb_depth_fit.py [LAMBDA=1e-3,1e-2,1e-1] [OUT=ksb_depth.txt]
"""
import os, sys, csv, glob, hashlib
import numpy as np
from scipy.optimize import minimize

KV = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
THIS = os.path.dirname(os.path.abspath(__file__))
DATA = "/mnt/e/chess_data/texel"
K = 0.00368208
KSB0, KSB1 = 106, 162                     # shelter 106-129, storm 130-161 (eval_v2.h layout)


def wp(cp):
    return 100.0 / (1.0 + np.exp(-K * np.clip(cp, -1500, 1500)))


def main():
    ours_d = {}
    for p in glob.glob(os.path.join(THIS, "ks_sets", "fitC_mg_ours_d10_s*of4.csv")):
        for r in csv.DictReader(open(p, newline="")):
            ours_d[r["fen"]] = float(r["ours_cp_white"])
    sample = {r["fen"]: (int(r["row"]), r["game_id"]) for r in
              csv.DictReader(open(os.path.join(THIS, "ks_sets/fitC_mg_sample.csv"))) if r["src"] == "std"}
    z = np.load(os.path.join(DATA, "fitC_features.npz"))
    DIFF, PH = z["diff"], z["phase"]
    rows, sf, base, val, stm = [], [], [], [], []
    for r in csv.DictReader(open(os.path.join(THIS, "ks_sets/fitC_mg_sf18.csv"), newline="")):
        f = r["fen"]
        if not r.get("best_cp") or f not in sample or f not in ours_d or abs(float(r["best_cp"])) >= 50000:
            continue
        row, gid = sample[f]
        rows.append(row); sf.append(float(r["best_cp"])); base.append(ours_d[f])
        val.append(int(hashlib.md5(gid.encode()).hexdigest()[:8], 16) % 100 < 15)
        stm.append(1.0 if f.split()[1] == "w" else -1.0)
    rows, sf, base, val, stm = map(np.array, (rows, sf, base, val, stm))
    X = DIFF[rows, KSB0:KSB1].astype(np.float64) * (PH[rows].astype(np.float64) / 256.0)[:, None]
    live = X.std(0) > 0
    tgt = wp(sf)
    tr = ~val
    print("KS-B DEPTH FIT  rows %d (train %d, val %d) · cells %d (%d with variation) · rows touched %.1f%%"
          % (len(sf), tr.sum(), val.sum(), X.shape[1], live.sum(), 100 * (np.abs(X).sum(1) > 0).mean()))

    def cp(p):
        return base - (X @ p[:-1]) / 10.0 + p[-1] * stm      # Black-positive mp cells → White cp

    def loss(p, m):
        return float(np.mean((wp(cp(p)[m]) - tgt[m]) ** 2))

    nk = X.shape[1]
    s0 = minimize(lambda q: loss(np.r_[np.zeros(nk), q], tr), [0.0]).x[0]
    pb = np.r_[np.zeros(nk), s0]
    best = None
    for lam in [float(x) for x in KV.get("LAMBDA", "1e-3,1e-2,1e-1").split(",")]:
        def fg(p):
            c = cp(p)[tr]
            q = wp(c)
            r = q - tgt[tr]
            g = 2.0 * r * q * (1 - q / 100.0) * K * (np.abs(c) < 1500)
            gw = -(X[tr] * g[:, None]).mean(0) / 10.0
            gs = float((g * stm[tr]).mean())
            pen = lam * float(p[:-1] @ p[:-1]) / 1e4
            return float(np.mean(r * r)) + pen, np.r_[gw + 2 * lam * p[:-1] / 1e4, gs]
        res = minimize(fg, pb.copy(), jac=True, method="L-BFGS-B", options={"maxiter": 3000})
        p = res.x
        vb, vf = loss(pb, val), loss(p, val)
        touched = val & (np.abs(X).sum(1) > 0)
        print("  λ=%-6g val %+.2f%% · val rows the cells touch (%d) %+.2f%% · |θ| max %.0f mp, mean %.0f"
              % (lam, 100 * (vf / vb - 1), touched.sum(), 100 * (loss(p, touched) / loss(pb, touched) - 1),
                 np.abs(p[:-1]).max(), np.abs(p[:-1]).mean()))
        if best is None or vf < best[1]:
            best = (lam, vf, p)
    lam, _, p = best
    out = os.path.join(DATA, KV.get("OUT", "ksb_depth.txt"))
    with open(out, "w") as f:
        f.write("# KS-B cells fitted on the DEPTH target (_ksb_depth_fit.py, λ=%g); k leg start fitted (mp, mg leg)\n" % lam)
        for k in range(nk):
            if live[k] and round(p[k]) != 0:
                f.write("%d 0 0 %.0f\n" % (KSB0 + k, p[k]))
    print("  best λ=%g → wrote %s" % (lam, out))


if __name__ == "__main__":
    main()
