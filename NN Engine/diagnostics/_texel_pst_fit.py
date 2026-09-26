# -*- coding: utf-8 -*-
"""TEXEL FIT A: v2's tapered piece-square tables against our own game results.

The recipe that was never run on v1 or v2 (EVAL-V2-INVENTORY-2026-09-25.md §3): the logistic of the static
eval against GAME RESULTS, on quiet positions from our own games, fitting the PST cells jointly. The rest of
the eval is held FIXED -- it is read once per position by _texel_engine_pass.py (MODE=zero) -- so the model
is exactly linear in the PST and the fit never calls the engine.

MODEL (Black-positive millipawns, like the engine):
    E_i  = fixed_i + sum_c x_ic * theta_c
    x_ic = (+1 black / -1 white) * (phase_i/256 for an mg cell, (256-phase_i)/256 for an eg cell)
    P(White scores) = sigmoid(-K * E_i),   loss = weighted mean (P - result_white)^2   (classic Texel MSE)

IDENTIFIABILITY AND REPLICATION (agreed 2026-09-26; each guards a failure the record documents):
  - half-board cells: files a/h, b/g, c/f, d/e are ONE parameter -> 6 pieces x 2 legs x 32 = 384, and the file
    mirror holds by construction;
  - each (piece, leg) table's MEAN is PINNED at its start: a PST mean is indistinguishable from material, and
    material stays out of this fit (the -85.6 Elo precedent). For kings it also removes a pure gauge freedom;
  - L2 toward the start (LAMBDA2) plus a smoothness penalty on the CHANGE between neighbouring cells
    (LAMBDA_S), so rarely-occupied cells stay near their prior or borrow from their neighbours;
  - every game weighs the same (1 / its row count), and positions already decided (|E| > DECIDED mp) weigh
    DECIDED_W;
  - K is fitted on the START tables, then frozen (a free K lets the fit rescale instead of learn);
  - two holdouts: val_hash (10% of games by hash, from stage 1) and val_run (every game of HOLD_RUN);
  - BOOT bootstrap refits over resampled games: the shipped change is their mean, and a cell whose mean change
    is under STAB x its bootstrap sd is returned to its start value.
Held-out loss only decides which candidates reach games; a NODE_LIMIT SPRT decides the ship.

  python diagnostics/_texel_pst_fit.py [DATA=E:/chess_data/texel] [LAMBDAS=1e-8,1e-7,1e-6] [LAMBDA_S_MULT=4]
        [BOOT=5] [BOOT_ROWS=1200000] [HOLD_RUN=spsaks2] [DECIDED=3000] [DECIDED_W=0.25] [STAB=2.0]
"""
import os, sys, time, math, json
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.optimize import minimize

KV = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
DATA = KV.get("DATA", "E:/chess_data/texel")
LAMBDAS = [float(x) for x in KV.get("LAMBDAS", "1e-8,1e-7,1e-6").split(",")]
LAMBDA_S_MULT = float(KV.get("LAMBDA_S_MULT", 4))
BOOT = int(KV.get("BOOT", 5))
BOOT_ROWS = int(KV.get("BOOT_ROWS", 1200000))
HOLD_RUN = KV.get("HOLD_RUN", "spsaks2")
DECIDED = float(KV.get("DECIDED", 3000))
DECIDED_W = float(KV.get("DECIDED_W", 0.25))
STAB = float(KV.get("STAB", 2.0))
SEED = int(KV.get("SEED", 1))
PIECES = "pnbrqk"
NPAR = 6 * 2 * 32


def log(*a):
    print("[fit %6.0fs]" % (time.time() - T0), *a, flush=True)


def pidx(t, leg, sq):
    """Tied parameter index of piece t, leg (0 mg / 1 eg), White-POV square sq (a1 = 0)."""
    r, f = sq >> 3, sq & 7
    return ((t * 2 + leg) * 8 + r) * 4 + min(f, 7 - f)


def load_start_table(path):
    vals = [int(t) for line in open(path) if not line.startswith("#") for t in line.split()]
    assert len(vals) == 768, "start table must hold 768 values"
    full = np.array(vals, dtype=np.float64).reshape(6, 2, 64)
    theta0 = np.zeros(NPAR)
    cnt = np.zeros(NPAR)
    for t in range(6):
        for leg in range(2):
            for sq in range(64):
                p = pidx(t, leg, sq)
                theta0[p] += full[t, leg, sq]
                cnt[p] += 1
    return theta0 / cnt, full


def build_features(fens, phase, chunk=100000):
    """Sparse X (n x 384) from FEN board fields; values are signed phase weights (Black-positive).
    Built in chunks converted to numpy as they go: Python lists for all ~185M entries would need ~15 GB."""
    # (piece char, square) -> (signed mg column, eg column) lookup, so the inner loop is one dict read.
    lut = {}
    for ch in "PNBRQKpnbrqk":
        t = PIECES.index(ch.lower())
        for sq in range(64):
            s, sgn = (sq, -1.0) if ch.isupper() else (sq ^ 56, 1.0)
            lut[(ch, sq)] = (pidx(t, 0, s), pidx(t, 1, s), sgn)
    wmg_all = phase / 256.0
    parts = []
    for c0 in range(0, len(fens), chunk):
        r_l, c_l, s_l, m_l = [], [], [], []
        for i in range(c0, min(c0 + chunk, len(fens))):
            board = fens[i].split(" ", 1)[0]
            r, f = 7, 0
            for ch in board:
                if ch == "/":
                    r -= 1
                    f = 0
                elif ch <= "8":
                    f += ord(ch) - 48
                else:
                    cm, ce, sgn = lut[(ch, r * 8 + f)]
                    r_l.append(i); c_l.append(cm); s_l.append(sgn); m_l.append(1)
                    r_l.append(i); c_l.append(ce); s_l.append(sgn); m_l.append(0)
                    f += 1
        rr = np.array(r_l, dtype=np.int32)
        wm = wmg_all[rr]
        vv = np.array(s_l, dtype=np.float32) * np.where(np.array(m_l, dtype=bool), wm, 1.0 - wm).astype(np.float32)
        parts.append((rr, np.array(c_l, dtype=np.int32), vv))
    rows = np.concatenate([p[0] for p in parts])
    cols = np.concatenate([p[1] for p in parts])
    vals = np.concatenate([p[2] for p in parts])
    X = sp.csr_matrix((vals, (rows, cols)), shape=(len(fens), NPAR))
    X.sum_duplicates()
    return X


def neighbour_pairs():
    pairs = []
    for t in range(6):
        for leg in range(2):
            base = (t * 2 + leg) * 32
            for r in range(8):
                for fp in range(4):
                    p = base + r * 4 + fp
                    if r < 7:
                        pairs.append((p, p + 4))
                    if fp < 3:
                        pairs.append((p, p + 1))
    return np.array(pairs)


def make_projector(active):
    """Zero inactive cells and remove each (piece, leg) group's mean over its ACTIVE cells."""
    groups = [np.where(active[g * 32:(g + 1) * 32])[0] + g * 32 for g in range(12)]

    def proj(v):
        out = np.where(active, v, 0.0)
        for idx in groups:
            if len(idx):
                out[idx] -= out[idx].mean()
        return out
    return proj


def fit(X, fixed, y, w, theta0, K, lam2, lams, pairs, proj, x0=None):
    wsum = w.sum()
    pa, pb = pairs[:, 0], pairs[:, 1]

    def f_and_g(d):
        d = proj(d)
        E = fixed + X @ (theta0 + d)
        p = 1.0 / (1.0 + np.exp(np.clip(K * E, -60, 60)))       # sigmoid(-K E)
        res = p - y
        loss = float(np.sum(w * res * res) / wsum)
        dE = (w * 2.0 * res * (-K) * p * (1.0 - p)) / wsum
        g = X.T @ dE
        diff = d[pa] - d[pb]
        loss += lam2 * float(d @ d) + lams * float(diff @ diff)
        g = g + 2.0 * lam2 * d
        gs = np.zeros_like(d)
        np.add.at(gs, pa, 2.0 * lams * diff)
        np.add.at(gs, pb, -2.0 * lams * diff)
        g = proj(g + gs)
        return loss, g

    r = minimize(f_and_g, np.zeros(NPAR) if x0 is None else x0, jac=True, method="L-BFGS-B",
                 options={"maxiter": 400, "gtol": 1e-12, "ftol": 1e-13})
    return proj(r.x), r


def mse(X, fixed, y, w, theta, K):
    E = fixed + X @ theta
    p = 1.0 / (1.0 + np.exp(np.clip(K * E, -60, 60)))
    return float(np.sum(w * (p - y) ** 2) / w.sum())


def fit_K(E, y, w):
    lo, hi = 1e-5, 1e-2
    f = lambda k: float(np.sum(w * (1.0 / (1.0 + np.exp(np.clip(k * E, -60, 60))) - y) ** 2) / w.sum())
    for _ in range(60):
        a = math.exp(math.log(lo) + (math.log(hi) - math.log(lo)) * 0.382)
        b = math.exp(math.log(lo) + (math.log(hi) - math.log(lo)) * 0.618)
        if f(a) < f(b):
            hi = b
        else:
            lo = a
    return (lo + hi) / 2


def expand(theta):
    full = np.zeros((6, 2, 64))
    for t in range(6):
        for leg in range(2):
            for sq in range(64):
                full[t, leg, sq] = theta[pidx(t, leg, sq)]
    return full


def write_table(path, full, header):
    with open(path, "w") as fo:
        fo.write("# %s\n" % header)
        for t in range(6):
            for leg in range(2):
                fo.write("# %s %s\n" % ("pawn knight bishop rook queen king".split()[t], "eg" if leg else "mg"))
                for r in range(8):
                    fo.write(" ".join(str(int(round(full[t, leg, r * 8 + f]))) for f in range(8)) + "\n")


def main():
    log("loading")
    st = pd.read_csv(os.path.join(DATA, "v2_stage1.csv.gz"),
                     usecols=["game_id", "split", "fen", "result_white"])
    z = pd.read_csv(os.path.join(DATA, "pass", "zero_0_of_1.csv"))
    fu = pd.read_csv(os.path.join(DATA, "pass", "full_0_of_1.csv"))
    assert len(z) == len(st) == len(fu), "pass files do not cover stage 1 (%d / %d / %d)" % (len(z), len(fu), len(st))
    st["fixed"] = z["total"].values
    st["phase"] = z["phase256"].values
    st["full"] = fu["total"].values
    if int(KV.get("ROWS", 0)):
        # smoke-test subset: a random sample (the file is ordered by run, so a prefix would be one run only)
        st = st.sample(n=int(KV["ROWS"]), random_state=SEED).sort_index()
    theta0, start_full = load_start_table(os.path.join(DATA, "pst_default.txt"))

    keep = (st["phase"] >= 0) & (st["fixed"].abs() < 30000) & (st["full"].abs() < 30000)
    st = st[keep].reset_index(drop=True)
    log("rows after phase/mate filter:", len(st))
    X = build_features(st["fen"].values, st["phase"].values.astype(np.float64))
    fixed = st["fixed"].values.astype(np.float64)
    y = st["result_white"].values.astype(np.float64)
    # PST-inert rows (draw classifier, exact KPK): the engine's full eval ignores the PST, so the linear model
    # cannot describe them. Identified by the residual of the start tables against the engine's own value.
    resid = st["full"].values - (fixed + X @ theta0)
    inert = np.abs(resid) > 25
    log("start-table residual |full - (fixed + X theta0)|: median %.2f mp, p99 %.1f, inert rows %d (%.2f%%)"
        % (np.median(np.abs(resid)), np.percentile(np.abs(resid), 99), inert.sum(), 100 * inert.mean()))

    is_run = st["game_id"].str.startswith(HOLD_RUN + "_").values
    is_hash = (st["split"] == "val").values
    train = ~is_run & ~is_hash & ~inert
    vhash = is_hash & ~is_run & ~inert
    vrun = is_run & ~inert
    gsize = st.groupby("game_id")["fen"].transform("size").values.astype(np.float64)
    w = 1.0 / gsize
    E0 = fixed + X @ theta0
    w = np.where(np.abs(E0) > DECIDED, w * DECIDED_W, w)
    log("train %d  val_hash %d  val_run(%s) %d" % (train.sum(), vhash.sum(), HOLD_RUN, vrun.sum()))

    K = fit_K(E0[train], y[train], w[train])
    log("K (start tables, frozen) = %.6f / mp" % K)
    active = np.asarray((abs(X[train]).sum(axis=0) > 0)).ravel()
    proj = make_projector(active)
    pairs = neighbour_pairs()
    log("active cells %d of %d" % (active.sum(), NPAR))

    Xt, Xh, Xr = X[train], X[vhash], X[vrun]
    base = {"train": mse(Xt, fixed[train], y[train], w[train], theta0, K),
            "val_hash": mse(Xh, fixed[vhash], y[vhash], w[vhash], theta0, K),
            "val_run": mse(Xr, fixed[vrun], y[vrun], w[vrun], theta0, K)}
    log("START  train %.6f  val_hash %.6f  val_run %.6f" % (base["train"], base["val_hash"], base["val_run"]))

    rng = np.random.default_rng(SEED)
    tr_idx = np.where(train)[0]
    sub = np.sort(rng.choice(tr_idx, size=min(BOOT_ROWS, len(tr_idx)), replace=False))
    results = []
    for lam in LAMBDAS:
        d, r = fit(X[sub], fixed[sub], y[sub], w[sub], theta0, K, lam, lam * LAMBDA_S_MULT, pairs, proj)
        th = theta0 + d
        res = {"lambda2": lam, "val_hash": mse(Xh, fixed[vhash], y[vhash], w[vhash], th, K),
               "val_run": mse(Xr, fixed[vrun], y[vrun], w[vrun], th, K), "iters": int(r.nit),
               "max_abs_change": float(np.abs(d).max())}
        results.append(res)
        log("lambda2 %.0e  val_hash %.6f (%+.3f%%)  val_run %.6f (%+.3f%%)  max|change| %.0f mp  iters %d"
            % (lam, res["val_hash"], 100 * (res["val_hash"] / base["val_hash"] - 1), res["val_run"],
               100 * (res["val_run"] / base["val_run"] - 1), res["max_abs_change"], r.nit))
    best = min(results, key=lambda r: r["val_run"])
    lam = best["lambda2"]
    log("chosen lambda2 = %.0e (best val_run)" % lam)

    # Bootstrap over GAMES: resample train games with replacement, fit, collect the change vectors.
    games = st["game_id"].values
    tr_games = np.unique(games[train])
    rows_by_game = pd.Series(np.arange(len(st)))[train].groupby(games[train]).apply(np.array)
    deltas = []
    for b in range(BOOT):
        pick = rng.choice(tr_games, size=len(tr_games), replace=True)
        idx = np.concatenate([rows_by_game[g] for g in pick])
        if len(idx) > BOOT_ROWS:
            idx = rng.choice(idx, size=BOOT_ROWS, replace=False)
        idx = np.sort(idx)
        d, r = fit(X[idx], fixed[idx], y[idx], w[idx], theta0, K, lam, lam * LAMBDA_S_MULT, pairs, proj)
        deltas.append(d)
        log("bootstrap %d/%d  val_run %.6f  iters %d" % (b + 1, BOOT, mse(Xr, fixed[vrun], y[vrun], w[vrun], theta0 + d, K), r.nit))
    D = np.array(deltas)
    dmean, dsd = D.mean(axis=0), D.std(axis=0, ddof=1) if BOOT > 1 else np.zeros(NPAR)
    stable = np.abs(dmean) >= STAB * np.maximum(dsd, 1e-9)
    dfinal = proj(np.where(stable, dmean, 0.0))
    th = theta0 + dfinal
    final = {"train": mse(Xt, fixed[train], y[train], w[train], th, K),
             "val_hash": mse(Xh, fixed[vhash], y[vhash], w[vhash], th, K),
             "val_run": mse(Xr, fixed[vrun], y[vrun], w[vrun], th, K)}
    log("FINAL  stable cells %d of %d active" % ((stable & active).sum(), active.sum()))
    for k in ("train", "val_hash", "val_run"):
        log("  %-8s start %.6f  fitted %.6f  (%+.3f%%)" % (k, base[k], final[k], 100 * (final[k] / base[k] - 1)))

    out_full = expand(th)
    tag = KV.get("TAG", "fitA")
    write_table(os.path.join(DATA, "pst_%s.txt" % tag), out_full,
                "Texel fit A lambda2=%g lambdaS=%g K=%.6f boot=%d stable=%d; val_run %+.3f%%"
                % (lam, lam * LAMBDA_S_MULT, K, BOOT, int((stable & active).sum()),
                   100 * (final["val_run"] / base["val_run"] - 1)))
    names = "pawn knight bishop rook queen king".split()
    log("per-table change (mp): mean |change| / max |change| / new spread vs start spread")
    for t in range(6):
        for leg in range(2):
            new = out_full[t, leg]
            old = start_full[t, leg]
            live = np.array([active[pidx(t, leg, s)] for s in range(64)])
            if not live.any():
                continue
            ch = np.abs(new - old)[live]
            log("  %-6s %s  %6.1f / %6.1f   spread %5.0f vs %5.0f"
                % (names[t], "eg" if leg else "mg", ch.mean(), ch.max(), np.ptp(new[live]), np.ptp(old[live])))
    json.dump({"K": K, "lambda2": lam, "base": base, "final": final, "grid": results,
               "stable": int((stable & active).sum()), "active": int(active.sum())},
              open(os.path.join(DATA, "pst_%s_report.json" % tag), "w"), indent=1)
    log("wrote", os.path.join(DATA, "pst_%s.txt" % tag))


if __name__ == "__main__":
    T0 = time.time()
    main()
