# -*- coding: utf-8 -*-
"""PASSER SYSTEM fit on the DEPTH target (dev_notes/PASSER-SYSTEM-DESIGN-2026-10-03.md; owner: passers FIRST, pawn
structure held fixed).

Columns (v2_features index; `_px_export.py` npz): existing passer columns 77-96 (passed / candidate by rank, the 4 king
terms) + the 51 PX cells 184-234 — BOTH legs free. Pawn structure (66-76) stays at its shipped values.
Base / target: ours' = ours_d10 (our d10 SEARCH of the CURRENT ship, White cp) − Σ diff·δ_blend/10 + STM nuisance, vs
SF18 d14 labels; mg AND eg labelled sets (passers matter most in the endgame — needs the eg depth pass). Val by FEN hash.
ARMS = per-block (gated per part afterwards): RANK (77-96 only) · STOP (PX 0-23: blocked by type + free + path) ·
SUPPORT (PX 24-35, 47-48: stop defended, pawn-defended, phalanx, pieces behind) · ESCORT (PX 36-46: king distances) ·
MISC (PX 49-50: file, square rule) · ALL. Each block also read on top of RANK (blocks must pay incrementally).
Output: C1_V2_FILE lines for 77-96 and PX_V2_FILE lines for 184-234 (`k leg start fitted`).

  pyrun diagnostics/_px_depth_fit.py [OURS_MG=fitC_mg_ours1003_d10] [OURS_EG=fitC_eg_ours1003_d10] [LAMBDA=1e-2] [TAG=px_depth]
"""
import os, sys, csv, glob, hashlib
import numpy as np
from scipy.optimize import minimize

KV = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
THIS = os.path.dirname(os.path.abspath(__file__))
DATA = "/mnt/e/chess_data/texel"
K = 0.00368208
RANK = list(range(77, 97))
PX0 = 184
BLOCKS = {"STOP": [PX0 + i for i in range(0, 24)],
          "SUPPORT": [PX0 + i for i in list(range(24, 36)) + [47, 48]],
          "ESCORT": [PX0 + i for i in range(36, 47)],
          "MISC": [PX0 + 49, PX0 + 50]}


def wp(cp):
    return 100.0 / (1.0 + np.exp(-K * np.clip(cp, -1500, 1500)))


def load_ours(prefix):
    d = {}
    for p in glob.glob(os.path.join(THIS, "ks_sets", prefix + "_s*of4.csv")):
        for r in csv.DictReader(open(p, newline="")):
            d[r["fen"]] = float(r["ours_cp_white"])
    return d


def main():
    ours = load_ours(KV.get("OURS_MG", "fitC_mg_ours1003_d10"))
    ours.update(load_ours(KV.get("OURS_EG", "fitC_eg_ours1003_d10")))
    z = np.load(os.path.join(DATA, KV.get("FEAT", "px_labelled.npz")))
    idx = {f: i for i, f in enumerate(z["fen"])}
    rows, sf, base, val, stm = [], [], [], [], []
    for lab in ("fitC_mg_sf18.csv", "fitC_eg_sf18.csv"):
        for r in csv.DictReader(open(os.path.join(THIS, "ks_sets", lab), newline="")):
            f = r["fen"]
            if not r.get("best_cp") or f not in ours or f not in idx or abs(float(r["best_cp"])) >= 50000:
                continue
            if z["flags"][idx[f]] & 3:          # draw-classifier / tier-2b rows: the linear model does not hold
                continue
            rows.append(idx[f]); sf.append(float(r["best_cp"])); base.append(ours[f])
            val.append(int(hashlib.md5(f.encode()).hexdigest()[:8], 16) % 100 < 15)
            stm.append(1.0 if f.split()[1] == "w" else -1.0)
    rows, sf, base, val, stm = map(np.array, (rows, sf, base, val, stm))
    ph = z["phase"][rows].astype(np.float64)
    cols = RANK + [c for b in BLOCKS.values() for c in b]
    D = z["diff"][rows][:, cols].astype(np.float64)
    X = np.concatenate([D * (ph / 256.0)[:, None], D * ((256.0 - ph) / 256.0)[:, None]], 1)
    nc = len(cols)
    tgt, tr = wp(sf), ~val
    print("PX DEPTH FIT  rows %d (val %d) · columns %d × 2 legs · rows with a PX cell %.1f%%"
          % (len(sf), val.sum(), nc, 100 * (np.abs(D[:, len(RANK):]).sum(1) > 0).mean()))

    def cp(p):
        return base - (X @ p[:-1]) / 10.0 + p[-1] * stm

    def loss(p, m):
        return float(np.mean((wp(cp(p)[m]) - tgt[m]) ** 2))

    s0 = minimize(lambda q: loss(np.r_[np.zeros(2 * nc), q], tr), [0.0]).x[0]
    pb = np.r_[np.zeros(2 * nc), s0]
    vb = loss(pb, val)
    lam = float(KV.get("LAMBDA", 1e-2))

    def fit(colset):
        mask = np.array([c in colset for c in cols])
        m2 = np.r_[mask, mask]
        def fg(p):
            pp = p.copy(); pp[:-1][~m2] = 0.0
            c = cp(pp)[tr]; q = wp(c); r = q - tgt[tr]
            g = 2.0 * r * q * (1 - q / 100.0) * K * (np.abs(c) < 1500)
            gw = -(X[tr] * g[:, None]).mean(0) / 10.0
            gw[~m2] = 0.0
            return float(np.mean(r * r)) + lam * float(pp[:-1] @ pp[:-1]) / 1e4, \
                np.r_[gw + 2 * lam * pp[:-1] / 1e4, float((g * stm[tr]).mean())]
        res = minimize(fg, pb.copy(), jac=True, method="L-BFGS-B", options={"maxiter": 4000})
        p = res.x.copy(); p[:-1][~m2] = 0.0
        return p

    arms = {"RANK": set(RANK)}
    for b, cs in BLOCKS.items():
        arms[b] = set(cs)
        arms["RANK+" + b] = set(RANK) | set(cs)
    arms["ALL"] = set(cols)
    out = {}
    for name, cs in arms.items():
        p = fit(cs)
        out[name] = p
        print("  %-14s val %+.2f%%" % (name, 100 * (loss(p, val) / vb - 1)))
    p = out["ALL"]
    tag = KV.get("TAG", "px_depth")
    with open(os.path.join(DATA, tag + "_c1.txt"), "w") as f1, open(os.path.join(DATA, tag + "_px.txt"), "w") as f2:
        f1.write("# passer rank columns, ALL arm (_px_depth_fit.py λ=%g) — k leg start fitted (mp)\n" % lam)
        f2.write("# PX passer cells, ALL arm (_px_depth_fit.py λ=%g) — k leg start fitted (mp)\n" % lam)
        for i, c in enumerate(cols):
            for leg in (0, 1):
                d = p[i + leg * nc]
                if round(d) == 0:
                    continue
                st = float((z["theta_mg"] if leg == 0 else z["theta_eg"])[c])
                (f2 if c >= PX0 else f1).write("%d %d %.0f %.0f\n" % (c, leg, st, st + d))
    print("  wrote %s_c1.txt (C1_V2_FILE, 77-96) and %s_px.txt (PX_V2_FILE)" % (tag, tag))


if __name__ == "__main__":
    main()
