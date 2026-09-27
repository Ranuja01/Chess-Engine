# -*- coding: utf-8 -*-
"""TEXEL FIT C1: v2's PST plus every linear C1 table (mobility, pawn structure, passers, placement), jointly.

Extends _texel_pst_fit.py (fit A, shipped) from one block to five. The engine side is _texel_feature_pass.py,
whose per-block gate already proved sum(count x theta) reproduces each engine block score within rounding.

MODEL (Black-positive mp):  E = fixed + X_pst theta_pst + X_c1 theta_c1,   fixed = total - X theta_0
  C1 columns: (Black - White count of feature k) x (phase/256 for the mg leg, (256-phase)/256 for eg).
  Ties: isolated files a/h, b/g, c/f, d/e share a parameter; passer king terms are eg-only (as the scorer).
IDENTIFIABILITY: per-(piece, leg) PST mean pinned (fit A); per-(type, leg) MOBILITY mean pinned -- a constant
  added to every mobility cell of a type is a re-pricing of that piece's material, which stays out of the fit.
REGULARISATION: L2 toward the start + smoothness on the change along PST squares, mobility move counts and passer
  ranks. Same data rules as fit A: equal game weights, decided rows x DECIDED_W, by-game val_hash + whole-run val_run.
VARIANTS: pst (C1 frozen; a control: fit A was fitted on this data) · c1 (PST frozen at fit A) · joint.

  python diagnostics/_texel_c1_fit.py [VARIANTS=pst,c1,joint] [LAMBDAS=1e-11,3e-12] [BOOT=3] [SUB=1500000]
        [PASS=E:/chess_data/texel/c1_std.npz] [START_PST=E:/chess_data/texel/pst_fitA.txt] [TAG=c1]
"""
import os, sys, time, json, math
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.optimize import minimize

THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS)
import _texel_pst_fit as A   # PST helpers: pidx, load_start_table, build_features, neighbour_pairs, fit_K, write_table

KV = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
DATA = KV.get("DATA", "E:/chess_data/texel")
PASS = KV.get("PASS", os.path.join(DATA, "c1_std.npz"))
START_PST = KV.get("START_PST", os.path.join(DATA, "pst_fitA.txt"))
VARIANTS = KV.get("VARIANTS", "pst,c1,joint").split(",")
LAMBDAS = [float(x) for x in KV.get("LAMBDAS", "1e-11,3e-12").split(",")]
LS_MULT = float(KV.get("LAMBDA_S_MULT", 4))
BOOT = int(KV.get("BOOT", 3))
SUB = int(KV.get("SUB", 1500000))
HOLD_RUN = KV.get("HOLD_RUN", "spsaks2")
DECIDED, DECIDED_W, STAB = float(KV.get("DECIDED", 3000)), float(KV.get("DECIDED_W", 0.25)), float(KV.get("STAB", 2.0))
TAG = KV.get("TAG", "c1")
MAXITER = int(KV.get("MAXITER", 1500))
# Minimum-support freeze: a parameter whose feature is non-zero in fewer than MIN_GAMES training games stays at its
# start value -- a handful of games cannot pin a value, and the bootstrap filter alone is too noisy at small BOOT.
MIN_GAMES = int(KV.get("MIN_GAMES", 2000))
T0 = time.time()
A.T0 = T0


def log(*a):
    print("[c1 %6.0fs]" % (time.time() - T0), *a, flush=True)


# ---- C1 parameter map: (feature k, leg) -> parameter index after the 384 PST cells ----------------------------
NPST = 384
MOB = [(0, 9), (9, 23), (23, 38), (38, 66)]      # N, B, R, Q move-count cell ranges
BLOCK_OF = {}
for k in range(106):
    BLOCK_OF[k] = ("mob" if k < 66 else "pawn" if k < 77 else "passer" if k < 97 else "place")


def build_c1_map():
    pmap, names, tie_src = {}, [], {}
    nxt = NPST
    for k in range(106):
        for leg in (0, 1):
            if 93 <= k <= 96 and leg == 0:
                continue                           # passer king terms: eg only, as the scorer
            if 67 <= k <= 74:                      # isolated by file: tie the mirrored files
                f = k - 67
                cls = min(f, 7 - f)
                key = ("iso", cls, leg)
                if key in tie_src:
                    pmap[(k, leg)] = tie_src[key]
                    continue
                tie_src[key] = nxt
            pmap[(k, leg)] = nxt
            names.append("f%d_%s" % (k, "eg" if leg else "mg"))
            nxt += 1
    return pmap, names, nxt


PMAP, C1_NAMES, NP = build_c1_map()


def c1_groups_and_pairs():
    groups, pairs = [], []
    for (a, b) in MOB:                               # mobility: mean pin + smoothness along move count, per leg
        for leg in (0, 1):
            ids = [PMAP[(k, leg)] for k in range(a, b)]
            groups.append(ids)
            pairs += list(zip(ids[:-1], ids[1:]))
    for base in (77, 85):                            # passed / candidate rank tables, ranks 1..6
        for leg in (0, 1):
            ids = [PMAP[(base + r, leg)] for r in range(1, 7)]
            pairs += list(zip(ids[:-1], ids[1:]))
    return groups, pairs


def main():
    log("loading pass", PASS)
    z = np.load(PASS)
    st = pd.read_csv(os.path.join(DATA, "v2_stage1.csv.gz"), usecols=["game_id", "split", "fen", "result_white"])
    rows = z["row"]
    st = st.iloc[rows].reset_index(drop=True)
    D = z["diff"].astype(np.float32)
    ph = z["phase"].astype(np.float64)
    total = z["total"].astype(np.float64)
    fl = z["flags"]
    tmg, teg = z["theta_mg"], z["theta_eg"]
    keep = (ph >= 0) & ((fl & 3) == 0) & (np.abs(total) < 30000)
    st, D, ph, total = st[keep].reset_index(drop=True), D[keep], ph[keep], total[keep]
    log("rows", len(st))

    theta0 = np.zeros(NP)
    pst0, start_full = A.load_start_table(START_PST)
    theta0[:NPST] = pst0
    for (k, leg), p in PMAP.items():
        theta0[p] = (tmg if leg == 0 else teg)[k]

    log("building PST features")
    Xp = A.build_features(st["fen"].values, ph)
    log("building C1 features")
    wmg = (ph / 256.0).astype(np.float32)
    ii, kk = np.nonzero(D)
    vals = D[ii, kk]
    r_l, c_l, v_l = [], [], []
    for leg, wl in ((0, wmg), (1, 1.0 - wmg)):
        cols = np.array([PMAP.get((k, leg), -1) for k in range(106)])[kk]
        m = cols >= 0
        r_l.append(ii[m]); c_l.append(cols[m]); v_l.append(vals[m] * wl[ii[m]])
    Xc = sp.csr_matrix((np.concatenate(v_l), (np.concatenate(r_l), np.concatenate(c_l))), shape=(len(st), NP))
    Xc.sum_duplicates()
    X = (sp.hstack([Xp, sp.csr_matrix((len(st), NP - NPST))]) + Xc).tocsr()
    fixed = total - X @ theta0
    y = st["result_white"].values.astype(np.float64)
    log("X nnz %d  (%.1f per row)" % (X.nnz, X.nnz / len(st)))

    is_run = st["game_id"].str.startswith(HOLD_RUN + "_").values
    is_hash = (st["split"] == "val").values
    train, vhash, vrun = ~is_run & ~is_hash, is_hash & ~is_run, is_run
    gsize = st.groupby("game_id")["fen"].transform("size").values.astype(np.float64)
    w = np.where(np.abs(total) > DECIDED, DECIDED_W, 1.0) / gsize
    K = A.fit_K(total[train], y[train], w[train])
    log("K = %.6f / mp; train %d val_hash %d val_run %d" % (K, train.sum(), vhash.sum(), vrun.sum()))

    # Support per parameter = number of distinct TRAINING GAMES in which its column is non-zero.
    Xt = X[train].tocsc()
    gid = pd.factorize(st["game_id"].values[train])[0]
    support = np.array([len(np.unique(gid[Xt.indices[Xt.indptr[j]:Xt.indptr[j + 1]]])) for j in range(NP)])
    active = support >= MIN_GAMES
    log("support freeze: %d of %d parameters have >= %d training games (%d frozen with some support)"
        % (active.sum(), NP, MIN_GAMES, ((support > 0) & ~active).sum()))
    del Xt
    pst_groups = [list(range(g * 32, (g + 1) * 32)) for g in range(12)]
    c1_groups, c1_pairs = c1_groups_and_pairs()
    pairs = np.array(list(map(tuple, A.neighbour_pairs())) + c1_pairs)

    def make_proj(free):
        gs = [np.array([i for i in g if free[i] and active[i]]) for g in pst_groups + c1_groups]
        gs = [g for g in gs if len(g)]

        def proj(v):
            out = np.where(free & active, v, 0.0)
            for g in gs:
                out[g] -= out[g].mean()
            return out
        return proj

    def fit(idx, lam, proj):
        Xi, fi, yi, wi = X[idx], fixed[idx], y[idx], w[idx]
        ws = wi.sum()
        pa, pb = pairs[:, 0], pairs[:, 1]
        lams = lam * LS_MULT

        def fg(d):
            d = proj(d)
            E = fi + Xi @ (theta0 + d)
            p = 1.0 / (1.0 + np.exp(np.clip(K * E, -60, 60)))
            res = p - yi
            loss = float(np.sum(wi * res * res) / ws)
            dE = (wi * 2.0 * res * (-K) * p * (1.0 - p)) / ws
            g = Xi.T @ dE
            diff = d[pa] - d[pb]
            loss += lam * float(d @ d) + lams * float(diff @ diff)
            gs_ = np.zeros_like(d)
            np.add.at(gs_, pa, 2.0 * lams * diff)
            np.add.at(gs_, pb, -2.0 * lams * diff)
            return loss, proj(g + 2.0 * lam * d + gs_)
        r = minimize(fg, np.zeros(NP), jac=True, method="L-BFGS-B",
                     options={"maxiter": MAXITER, "gtol": 1e-12, "ftol": 1e-13})
        return proj(r.x), r.nit

    def mse(mask, theta):
        E = fixed[mask] + X[mask] @ theta
        p = 1.0 / (1.0 + np.exp(np.clip(K * E, -60, 60)))
        return float(np.sum(w[mask] * (p - y[mask]) ** 2) / w[mask].sum())

    base = {"val_hash": mse(vhash, theta0), "val_run": mse(vrun, theta0)}
    log("START (shipped: fit A PST + current C1)  val_hash %.6f  val_run %.6f" % (base["val_hash"], base["val_run"]))
    rng = np.random.default_rng(1)
    tr_idx = np.where(train)[0]
    sub = np.sort(rng.choice(tr_idx, size=min(SUB, len(tr_idx)), replace=False))
    free_sets = {"pst": np.arange(NP) < NPST, "c1": np.arange(NP) >= NPST, "joint": np.ones(NP, bool)}
    report = {"K": K, "base": base, "variants": {}}
    best = None
    for vname in VARIANTS:
        proj = make_proj(free_sets[vname])
        for lam in LAMBDAS:
            d, it = fit(sub, lam, proj)
            th = theta0 + d
            vh, vr = mse(vhash, th), mse(vrun, th)
            log("%-6s lambda2 %.0e  val_hash %.6f (%+.3f%%)  val_run %.6f (%+.3f%%)  max|d| pst %.0f c1 %.0f  iters %d"
                % (vname, lam, vh, 100 * (vh / base["val_hash"] - 1), vr, 100 * (vr / base["val_run"] - 1),
                   np.abs(d[:NPST]).max(), np.abs(d[NPST:]).max(), it))
            report["variants"]["%s_%g" % (vname, lam)] = {"val_hash": vh, "val_run": vr}
            if vname == "joint" and (best is None or vr < best[0]):
                best = (vr, lam)
    if best is None:
        json.dump(report, open(os.path.join(DATA, "%s_report.json" % TAG), "w"), indent=1)
        return
    lam = best[1]
    proj = make_proj(free_sets["joint"])
    games = st["game_id"].values
    tr_games = np.unique(games[train])
    rows_by_game = pd.Series(np.arange(len(st)))[train].groupby(games[train]).apply(np.array)
    Ds = []
    for b in range(BOOT):
        pick = rng.choice(tr_games, size=len(tr_games), replace=True)
        idx = np.concatenate([rows_by_game[g] for g in pick])
        if len(idx) > SUB:
            idx = rng.choice(idx, size=SUB, replace=False)
        d, it = fit(np.sort(idx), lam, proj)
        Ds.append(d)
        log("bootstrap %d/%d  val_run %.6f" % (b + 1, BOOT, mse(vrun, theta0 + d)))
    Dm = np.array(Ds)
    dmean, dsd = Dm.mean(0), (Dm.std(0, ddof=1) if BOOT > 1 else np.zeros(NP))
    stable = np.abs(dmean) >= STAB * np.maximum(dsd, 1e-9)
    th = theta0 + proj(np.where(stable, dmean, 0.0))
    fin = {"val_hash": mse(vhash, th), "val_run": mse(vrun, th)}
    log("FINAL joint lambda2 %.0e  stable %d/%d active  val_hash %+.3f%%  val_run %+.3f%%"
        % (lam, (stable & active).sum(), active.sum(), 100 * (fin["val_hash"] / base["val_hash"] - 1),
           100 * (fin["val_run"] / base["val_run"] - 1)))
    A.write_table(os.path.join(DATA, "pst_%s.txt" % TAG), A.expand(th[:NPST]), "Texel %s joint, lambda2=%g" % (TAG, lam))
    with open(os.path.join(DATA, "c1params_%s.txt" % TAG), "w") as fo:
        fo.write("# feature_k leg start fitted (mp)\n")
        for (k, leg), p in sorted(PMAP.items()):
            fo.write("%d %d %.2f %.2f\n" % (k, leg, theta0[p], th[p]))
    for blk in ("mob", "pawn", "passer", "place"):
        ids = sorted({PMAP[(k, leg)] for (k, leg) in PMAP if BLOCK_OF[k] == blk})
        ch = np.abs(th[ids] - theta0[ids])
        log("  %-6s params %3d  mean|change| %6.1f mp  max %6.1f" % (blk, len(ids), ch.mean(), ch.max()))
    report["final"] = fin
    report["lambda2"] = lam
    json.dump(report, open(os.path.join(DATA, "%s_report.json" % TAG), "w"), indent=1)


if __name__ == "__main__":
    main()
