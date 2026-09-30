# -*- coding: utf-8 -*-
"""POT winnability, fitted to SF18 SEARCH labels (depth-independent) instead of our d6 game outcomes.

Why (C3 doc §12-14): Fit W, fitted to d6 GAME OUTCOMES, improved held-out prediction but hurt play at depth (fresh
seeds −25 Elo); its labels priced endgame convertibility for shallow play and its swings reached ~1.7 pawns. Here the
target is SF18's d14 search score of the same position (independent of our search depth) and the adjustment is CAPPED.

Model (White-POV cp for the loss; engine totals are Black-positive mp): our cp = −(T + adj)/10 with
adj = s·clip(max(C·g, −|T|), −CAP, CAP), s = sign(T), g = (256 − phase)/256, C = Σ w·in + BASE (win_inputs order).
Loss = mean (win%(our cp) − win%(SF cp))², win%(cp) = 100/(1+exp(−0.00368208·cp)), cp clamped ±1500 (toolkit convention).
Split by game hash (15% validation). Reports MSE before (shipped) / after, and the fitted knob line.

  pyrun diagnostics/_texel_win_sf_fit.py [LABELS=ks_sets/fitC_eg_sf18.csv] [SAMPLE=ks_sets/fitC_eg_sample.csv]
        [CAP=500] [LAMBDA=1e-4] [TAG=fitWsf]
"""
import os, sys, csv, hashlib, math
import numpy as np
from scipy.optimize import minimize

KV = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
THIS = os.path.dirname(os.path.abspath(__file__))
DATA = KV.get("DATA", "/mnt/e/chess_data/texel")
LABELS = os.path.join(THIS, KV.get("LABELS", "ks_sets/fitC_eg_sf18.csv"))
SAMPLE = os.path.join(THIS, KV.get("SAMPLE", "ks_sets/fitC_eg_sample.csv"))
CAP = float(KV.get("CAP", 500))
LAMBDA = float(KV.get("LAMBDA", 1e-4))
TAG = KV.get("TAG", "fitWsf")
K_WIN = 0.00368208
NAMES = ["PASSED", "PAWNS", "OUTFLANK", "INFILT", "FLANKS", "PAWN_END", "UNWIN", "BASE"]


def winpct(cp):
    return 100.0 / (1.0 + np.exp(-K_WIN * np.clip(cp, -1500, 1500)))


def main():
    sample = {r["fen"]: (int(r["row"]), r["game_id"]) for r in csv.DictReader(open(SAMPLE, newline=""))}
    lab = [(r["fen"], float(r["best_cp"])) for r in csv.DictReader(open(LABELS, newline="")) if r.get("best_cp")]
    z = np.load(os.path.join(DATA, "fitC_win.npz"))
    rows, sf, val = [], [], []
    for fen, cp in lab:
        if fen not in sample:
            continue
        row, gid = sample[fen]
        rows.append(row)
        sf.append(cp)
        val.append(int(hashlib.md5(gid.encode()).hexdigest()[:8], 16) % 100 < 15)
    rows, sf, val = np.array(rows), np.array(sf), np.array(val)
    T = z["total"][rows].astype(np.float64)
    ph = z["phase"][rows].astype(np.float64)
    X = np.concatenate([z["inputs"][rows].astype(np.float64), np.ones((len(rows), 1))], axis=1)
    ok = (ph >= 0) & (np.abs(T) < 30000) & (np.abs(sf) < 50000)
    T, ph, X, sf, val = T[ok], ph[ok], X[ok], sf[ok], val[ok]
    g = (256.0 - ph) / 256.0
    s = np.sign(T)
    tgt = winpct(sf)
    tr = ~val
    print("rows %d (train %d, val %d)  CAP %.0f mp" % (len(T), tr.sum(), val.sum(), CAP))

    def adj(wv, m):
        C = X[m] @ wv * g[m]
        a = np.clip(np.maximum(C, -np.abs(T[m])), -CAP, CAP)
        return s[m] * a, (C > -np.abs(T[m])) & (np.abs(C) < CAP)

    def mse(wv, m):
        a, _ = adj(wv, m)
        return float(np.mean((winpct(-(T[m] + a) / 10.0) - tgt[m]) ** 2))

    base = {"train": mse(np.zeros(8), tr), "val": mse(np.zeros(8), val)}
    scale = np.array([50, 50, 50, 100, 100, 200, 200, 500], dtype=np.float64)

    def fg(d):
        wv = d * scale
        a, act = adj(wv, tr)
        cp = -(T[tr] + a) / 10.0
        p = winpct(cp)
        r = p - tgt[tr]
        loss = float(np.mean(r * r)) + LAMBDA * float(d @ d)
        dp = p * (1 - p / 100.0) * K_WIN * (np.abs(cp) < 1500)        # d win%/d cp
        dcp = -(s[tr] * g[tr] * act) / 10.0                            # d cp / d C-weight row factor
        grad = (X[tr] * (2.0 * r * dp * dcp)[:, None]).mean(axis=0)
        return loss, grad * scale + 2.0 * LAMBDA * d

    r = minimize(fg, np.zeros(8), jac=True, method="L-BFGS-B", options={"maxiter": 500, "gtol": 1e-12, "ftol": 1e-14})
    wv = r.x * scale
    fit = {"train": mse(wv, tr), "val": mse(wv, val)}
    for k in ("train", "val"):
        print("  %-5s shipped %.3f  fitted %.3f  (%+.2f%%)" % (k, base[k], fit[k], 100 * (fit[k] / base[k] - 1)))
    print("  weights (mp): " + " ".join("%s=%.0f" % (k, v) for k, v in zip(NAMES, wv)) + "   iters %d" % r.nit)
    line = "WIN_V2=1 " + " ".join("WIN_V2_%s=%d" % (k, int(round(v))) for k, v in zip(NAMES, wv))
    open(os.path.join(DATA, "win_%s.txt" % TAG), "w").write(
        "# Fit W-SF (SF18 labels, CAP %.0f): val %+.2f%%\n%s\n" % (CAP, 100 * (fit["val"] / base["val"] - 1), line))
    print("  ->", line)
    print("  ⚠️ the engine applies NO cap yet: if this ships, the cap must be added to win_adjust (knob WIN_V2_CAP).")


if __name__ == "__main__":
    main()
