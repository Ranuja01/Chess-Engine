# -*- coding: utf-8 -*-
"""TEXEL FIT W: the OvD eg WINNABILITY weights, fitted on top of the SHIPPED eval (nested: everything else frozen).

Model (Black-positive mp): E = T + s·max(C·g, −|T|), s = sign(T), g = (256 − phase)/256,
C = Σ w_j·in_j + BASE over the 7 inputs of eval_v2.cpp win_inputs (data: _texel_win_pass.py with WIN_V2=0, so T is the
shipped total). Texel MSE vs game results, K fitted on T and frozen, games weighted equally, decided rows (|T| > DECIDED)
down-weighted; holdouts val_hash (10% of games) and val_block (games >= BLOCK_FROM); L2 on the weights (per-unit scale).
Two starts: ZERO and an SF-like PRIOR (SF12's complexity weights, in our mp via the pawn-eg conversion ×4.69).
Reports held-out change overall and on ENDGAME rows (phase256 < 96) where the term acts; writes the knob line.

  pyrun diagnostics/_texel_win_fit.py [LAMBDA=1e-8] [TAG=fitW]
"""
import os, sys, time, json, math
import numpy as np
import pandas as pd
from scipy.optimize import minimize

THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS)
import _texel_pst_fit as A

KV = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
DATA = KV.get("DATA", "/mnt/e/chess_data/texel")
LAMBDA = float(KV.get("LAMBDA", 1e-8))
BLOCK_FROM = int(KV.get("BLOCK_FROM", 27000))
DECIDED, DECIDED_W = float(KV.get("DECIDED", 3000)), float(KV.get("DECIDED_W", 0.25))
TAG = KV.get("TAG", "fitW")
NAMES = ["PASSED", "PAWNS", "OUTFLANK", "INFILT", "FLANKS", "PAWN_END", "UNWIN", "BASE"]
# SF12 `winnable` complexity weights (9 passed, 12 pawns, 9 outflanking, 24 infiltration, 21 both flanks, 51 pawn
# ending, −43 almost unwinnable, −110) in SF internal eg units; ×4.69 converts to our mp (pawn eg 213 → 1000).
PRIOR = np.array([9, 12, 9, 24, 21, 51, -43, -110], dtype=np.float64) * 4.69
T0 = time.time()


def log(*a):
    print("[fitW %5.0fs]" % (time.time() - T0), *a, flush=True)


def main():
    st = pd.read_csv(os.path.join(DATA, "fitC_stage1.csv.gz"), usecols=["game_id", "split", "result_white"])
    z = np.load(os.path.join(DATA, "fitC_win.npz"))
    T, ph, X = z["total"].astype(np.float64), z["phase"].astype(np.float64), z["inputs"].astype(np.float64)
    keep = (ph >= 0) & (np.abs(T) < 30000)
    st, T, ph, X = st[keep].reset_index(drop=True), T[keep], ph[keep], X[keep]
    X = np.concatenate([X, np.ones((len(X), 1))], axis=1)       # BASE column
    y = st["result_white"].values.astype(np.float64)
    g = (256.0 - ph) / 256.0
    s = np.sign(T)
    gnum = st["game_id"].str.extract(r"game_(\d+)")[0].astype(int).values
    vblock = gnum >= BLOCK_FROM
    vhash = (st["split"] == "val").values & ~vblock
    train = ~vblock & ~vhash
    eg = ph < 96
    w = 1.0 / st.groupby("game_id")["game_id"].transform("size").values.astype(np.float64)
    w = np.where(np.abs(T) > DECIDED, w * DECIDED_W, w)
    K = A.fit_K(T[train], y[train], w[train])
    log("rows %d (train %d, val_hash %d, val_block %d), K %.6f" % (len(T), train.sum(), vhash.sum(), vblock.sum(), K))

    def model(wt, m):
        C = X[m] @ wt * g[m]
        return T[m] + s[m] * np.maximum(C, -np.abs(T[m]))

    def mse(E, m):
        p = 1.0 / (1.0 + np.exp(np.clip(K * E, -60, 60)))
        return float(np.sum(w[m] * (p - y[m]) ** 2) / w[m].sum())

    base = {k: mse(T[m], m) for k, m in (("val_hash", vhash), ("val_block", vblock),
                                         ("eg_hash", vhash & eg), ("eg_block", vblock & eg))}
    log("START (shipped)  " + "  ".join("%s %.6f" % kv for kv in base.items()))
    Xt, Tt, st_, gt, yt, wt_ = X[train], T[train], s[train], g[train], y[train], w[train]
    wsum = wt_.sum()
    scale = np.array([50, 50, 50, 100, 100, 200, 200, 500], dtype=np.float64)   # plausible move per weight (mp)

    def fg(d):
        wv = d * scale
        C = Xt @ wv * gt
        act = C > -np.abs(Tt)
        E = Tt + st_ * np.maximum(C, -np.abs(Tt))
        p = 1.0 / (1.0 + np.exp(np.clip(K * E, -60, 60)))
        res = p - yt
        loss = float(np.sum(wt_ * res * res) / wsum) + LAMBDA * float(d @ d) * 1e4
        dE = (wt_ * 2.0 * res * (-K) * p * (1.0 - p)) / wsum
        gw = (Xt * (st_ * gt * act)[:, None]).T @ dE
        return loss, gw * scale + 2.0 * LAMBDA * 1e4 * d

    out = {}
    for name, start in (("zero", np.zeros(8)), ("sf_prior", PRIOR / scale)):
        r = minimize(fg, start, jac=True, method="L-BFGS-B", options={"maxiter": 500, "gtol": 1e-12, "ftol": 1e-14})
        wv = r.x * scale
        res = {k: mse(model(wv, m), m) for k, m in (("val_hash", vhash), ("val_block", vblock),
                                                     ("eg_hash", vhash & eg), ("eg_block", vblock & eg))}
        out[name] = {"w": dict(zip(NAMES, wv.tolist())), "res": res}
        log("FIT from %-8s  " % name + "  ".join("%s %+.3f%%" % (k, 100 * (res[k] / base[k] - 1)) for k in res)
            + "   iters %d" % r.nit)
        log("   weights (mp): " + " ".join("%s=%.0f" % (k, v) for k, v in zip(NAMES, wv)))
    # prior as-is (no fit) for reference
    pr = {k: mse(model(PRIOR, m), m) for k, m in (("val_hash", vhash), ("val_block", vblock),
                                                  ("eg_hash", vhash & eg), ("eg_block", vblock & eg))}
    log("SF PRIOR unfitted  " + "  ".join("%s %+.3f%%" % (k, 100 * (pr[k] / base[k] - 1)) for k in pr))
    best = min(out, key=lambda k: out[k]["res"]["val_block"])
    wv = [out[best]["w"][k] for k in NAMES]
    line = "WIN_V2=1 " + " ".join("WIN_V2_%s=%d" % (k, int(round(v))) for k, v in zip(NAMES, wv))
    open(os.path.join(DATA, "win_%s.txt" % TAG), "w").write("# Fit W (%s start), val_block %+.3f%%\n%s\n" % (
        best, 100 * (out[best]["res"]["val_block"] / base["val_block"] - 1), line))
    json.dump({"K": K, "base": base, "fits": out, "prior": pr}, open(os.path.join(DATA, "%s_report.json" % TAG), "w"),
              indent=1)
    log("chosen: %s  ->  %s" % (best, line))


if __name__ == "__main__":
    main()
