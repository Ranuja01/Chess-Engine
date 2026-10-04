# -*- coding: utf-8 -*-
"""PAWN STRUCTURE + PASSERS, Texel-fitted on the DEPTH target (owner, 10-03: "consolidate pawns, then Texel tune").

Record check (10-03): a pawn/passer fit on a depth target was NEVER tried — Fit C1 (09-27) fitted these columns on d6
OUTCOMES jointly with everything and was judged at 50k self-play; C1 also WANTED mg legs that are hard-zero today
(passer rank-7 mg < 0, weak-unopposed mg < 0). This fits the C1 pawn columns of v2_features (eval_v2.h layout):
  66 doubled · 67-74 isolated by file · 75 backward · 76 weak-unopposed       (STRUCT)
  77-84 passed by relative rank · 85-92 candidate by rank · 93-96 passer king terms   (PASSER)
BOTH legs free (mg AND eg — the mg legs of passers / isolated / backward / weak-unopposed are 0 today).
Model (White cp): ours' = ours_d10 − Σ_k diff_k · (δmg_k·ph + δeg_k·(256−ph))/256 / 10 + STM nuisance (not shipped),
diff = Black − White counts (fitC_features.npz), δ = fitted − start (start = theta_mg/eg). Target SF18 d14, val by game
15%. ARMS: STRUCT · PASSER · JOINT — gated per part (the KS lesson: parts can behave differently).
⚠️ BASE: ours_d10 comes from the depth pass of the engine shipped at the time it ran (`fitC_mg_ours_d10_s*of4.csv`);
re-run that pass after a ship so the fit nests on top of the CURRENT engine.
Output: C1_V2_FILE format (`k leg start fitted`) — loaded by `C1_V2_FIT=1 C1_V2_FILE=...` (eval_v2.cpp v2_c1_init).

  pyrun diagnostics/_pawn_depth_fit.py [LAMBDA=1e-3,1e-2,1e-1] [OURS=fitC_mg_ours_d10] [TAG=pawn_depth]
"""
import os, sys, csv, glob, hashlib
import numpy as np
from scipy.optimize import minimize

KV = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
THIS = os.path.dirname(os.path.abspath(__file__))
DATA = "/mnt/e/chess_data/texel"
K = 0.00368208
STRUCT = list(range(66, 77))
PASSER = list(range(77, 97))


def wp(cp):
    return 100.0 / (1.0 + np.exp(-K * np.clip(cp, -1500, 1500)))


def main():
    ours_d = {}
    for p in glob.glob(os.path.join(THIS, "ks_sets", KV.get("OURS", "fitC_mg_ours_d10") + "_s*of4.csv")):
        for r in csv.DictReader(open(p, newline="")):
            ours_d[r["fen"]] = float(r["ours_cp_white"])
    sample = {r["fen"]: (int(r["row"]), r["game_id"]) for r in
              csv.DictReader(open(os.path.join(THIS, "ks_sets/fitC_mg_sample.csv"))) if r["src"] == "std"}
    z = np.load(os.path.join(DATA, "fitC_features.npz"))
    DIFF, PH, TMG, TEG = z["diff"], z["phase"], z["theta_mg"], z["theta_eg"]
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
    ph = PH[rows].astype(np.float64)
    cols = STRUCT + PASSER
    D = DIFF[rows][:, cols].astype(np.float64)
    # ☠️ FILE-MIRROR TIE (2026-10-04): fitting isolated-by-file independently broke the file-mirror gate (2,435 / 3,170
    # violations, queue #14). Tie a=h, b=g, c=f, d=e: pool the mirrored file's counts into the a-d column, zero the e-h
    # column, and copy the fitted δ back after the fit (the shipped start table is already mirror-symmetric).
    MIRROR = [(cols.index(67 + f), cols.index(67 + 7 - f)) for f in range(4)]
    for lo, hi in MIRROR:
        D[:, lo] += D[:, hi]
        D[:, hi] = 0.0
    Xmg = D * (ph / 256.0)[:, None]
    Xeg = D * ((256.0 - ph) / 256.0)[:, None]
    X = np.concatenate([Xmg, Xeg], 1)                       # params: [δmg (len cols), δeg (len cols)]
    nc = len(cols)
    tgt = wp(sf)
    tr = ~val
    print("PAWN DEPTH FIT  rows %d (val %d) · columns %d × 2 legs · rows touched %.1f%%"
          % (len(sf), val.sum(), nc, 100 * (np.abs(D).sum(1) > 0).mean()))

    def cp(p):
        return base - (X @ p[:-1]) / 10.0 + p[-1] * stm

    def loss(p, m):
        return float(np.mean((wp(cp(p)[m]) - tgt[m]) ** 2))

    s0 = minimize(lambda q: loss(np.r_[np.zeros(2 * nc), q], tr), [0.0]).x[0]
    pb = np.r_[np.zeros(2 * nc), s0]
    vb = loss(pb, val)

    def fit(mask, lam):
        m2 = np.r_[mask, mask]
        def fg(p):
            pp = p.copy(); pp[:-1][~m2] = 0.0
            c = cp(pp)[tr]; q = wp(c); r = q - tgt[tr]
            g = 2.0 * r * q * (1 - q / 100.0) * K * (np.abs(c) < 1500)
            gw = -(X[tr] * g[:, None]).mean(0) / 10.0
            gw[~m2] = 0.0
            return float(np.mean(r * r)) + lam * float(pp[:-1] @ pp[:-1]) / 1e4, \
                np.r_[gw + 2 * lam * pp[:-1] / 1e4, float((g * stm[tr]).mean())]
        res = minimize(fg, pb.copy(), jac=True, method="L-BFGS-B", options={"maxiter": 3000})
        p = res.x.copy(); p[:-1][~m2] = 0.0
        for lo, hi in MIRROR:                       # the mirrored file gets the tied value (both legs)
            p[hi] = p[lo]; p[nc + hi] = p[nc + lo]
        return p

    arms = {"STRUCT": np.array([c in STRUCT for c in cols]), "PASSER": np.array([c in PASSER for c in cols]),
            "JOINT": np.ones(nc, bool)}
    out = {}
    for lam in [float(x) for x in KV.get("LAMBDA", "1e-3,1e-2,1e-1").split(",")]:
        line = []
        for name, mask in arms.items():
            p = fit(mask, lam)
            out[(name, lam)] = p
            line.append("%s %+.2f%%" % (name, 100 * (loss(p, val) / vb - 1)))
        print("  λ=%-6g " % lam + " · ".join(line))
    lam = float(KV.get("PICK", "1e-2"))
    p = out[("JOINT", lam)]
    names = ["doubled"] + ["iso_%s" % "abcdefgh"[i] for i in range(8)] + ["backward", "weak_unopp"] + \
        ["passed_r%d" % i for i in range(8)] + ["cand_r%d" % i for i in range(8)] + ["pk_them", "pk_us", "ck_them", "ck_us"]
    print("\n  JOINT λ=%g, notable moves (fitted − start, mp; |δ| ≥ 40):" % lam)
    for i, c in enumerate(cols):
        dm, de = p[i], p[nc + i]
        if abs(dm) >= 40 or abs(de) >= 40:
            print("    %-11s mg %+5.0f → %+5.0f   eg %+5.0f → %+5.0f" % (names[i], TMG[c], TMG[c] + dm, TEG[c], TEG[c] + de))
    tag = KV.get("TAG", "pawn_depth")
    for name in arms:
        pp = out[(name, lam)]
        with open(os.path.join(DATA, "%s_%s.txt" % (tag, name.lower())), "w") as f:
            f.write("# pawn/passer depth fit (_pawn_depth_fit.py) arm %s λ=%g — k leg start fitted (mp)\n" % (name, lam))
            for i, c in enumerate(cols):
                for leg, (st, d) in enumerate(((TMG[c], pp[i]), (TEG[c], pp[nc + i]))):
                    if round(d) != 0:
                        f.write("%d %d %.0f %.0f\n" % (c, leg, st, st + d))
    print("  wrote %s_{struct,passer,joint}.txt (λ=%g)" % (tag, lam))


if __name__ == "__main__":
    main()
