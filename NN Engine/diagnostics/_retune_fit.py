# -*- coding: utf-8 -*-
"""STRUCTURAL RETUNE — static component on the assembled corpus (2026-10-10, step 1).

Data: $D/corpus_v1.csv (_corpus_assemble.py) + $D/feat_ship_cNN.npz (_px_export.py IN=…, same row order; ship static
`total` in the same pass). Model (the _revival_screen fitter): ours' = static·(1+s) − X·δ/10 + t·stm, static = the
ship's static eval (White cp), δ = change in mp from the SHIP (L2 toward the ship), s / t = SCALE / STM nuisances (never
shipped). X = [337 v2_features columns × mg/eg legs | PST file-tied 192 × legs | Kaufman census cells + N/B/R/Q
corrections (untapered)], all Black − White. Rows with flags & 3 (draw classifier / tier-2b) are dropped.

  pyrun diagnostics/_retune_fit.py [EXPERIMENT=weights] [TRAIN_N=250000] [VAL_N=120000] [LAMBDA=1e-2] [SEED=1]

EXPERIMENT=weights (owner 10-10: "quick experimental understanding of what's optimal"): the same fit under row-weighting
arms — band scheme {flat, default, strong} × train rows {quiet, all}, then stress-slice weight {0.25, 0.5} — every arm
scored on the SAME held-out own-play games with FLAT weights (quiet rows; per band / per phase stage), plus the KP / VAR
held-out stress rows, plus how far each arm's eval moves from the default arm on held-out positions (rms cp). ☠️ A flat
held-out score favours the flat arm by construction: read the per-band TRADE-OFF and the rms-cp distance, not the total.
"""
import os, sys, csv, glob, time
import numpy as np
import chess
import scipy.sparse as sp
from scipy.optimize import minimize

KV = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS)
D = KV.get("D", "/mnt/e/chess_data/retune")
K = 0.00368208
wp = lambda cp: 100.0 / (1.0 + np.exp(-K * np.clip(cp, -1500, 1500)))
SCHEMES = {   # band → weight
    "flat":    {"level": 1, "0.5-1.5": 1, "1.5-4": 1, "4-15": 1, ">=15": 1, "mate": 1},
    "default": {"level": 1, "0.5-1.5": 1, "1.5-4": 1, "4-15": 0.5, ">=15": 0.2, "mate": 0.1},
    "strong":  {"level": 1, "0.5-1.5": 1, "1.5-4": 1, "4-15": 0.25, ">=15": 0, "mate": 0},
}
BANDS = ["level", "0.5-1.5", "1.5-4", "4-15", "mate"]
STAGES = ["mg", "mgl", "egl", "eg"]


def pst_kauf_row(fen):
    import _texel_kauf_fit as KF
    b = chess.Board(fen)
    occ = np.zeros(192, np.float32)
    for sq, pc in b.piece_map().items():
        rel = sq if pc.color == chess.WHITE else sq ^ 56
        f = chess.square_file(rel)
        occ[(pc.piece_type - 1) * 32 + chess.square_rank(rel) * 4 + min(f, 7 - f)] += 1 if pc.color == chess.BLACK else -1
    return occ, -np.array(KF.feats(b), np.float32)


def load():
    t0 = time.time()
    cols = {c: [] for c in ("fen", "slice", "split", "quiet", "band", "stage", "sf_cp")}
    with open(os.path.join(D, "corpus_v1.csv"), newline="") as fh:
        rd = csv.reader(fh); hdr = next(rd); ix = {c: hdr.index(c) for c in cols}
        for r in rd:
            for c, i in ix.items():
                cols[c].append(r[i])
    chunks = sorted(glob.glob(os.path.join(D, "feat_ship_c*.npz")))
    flags = np.concatenate([np.load(p)["flags"] for p in chunks])
    assert len(flags) == len(cols["fen"]), "export / corpus row counts differ"
    rng = np.random.default_rng(int(KV.get("SEED", 1)))
    sl = np.array(cols["slice"]); split = np.array(cols["split"])
    ok = (flags & 3) == 0
    own = np.isin(sl, ["A", "A2", "V1"])
    pick = lambda m, n: np.sort(rng.permutation(np.flatnonzero(m))[:n])
    tr_own = pick(own & (split == "train") & ok, int(KV.get("TRAIN_N", 250000)))
    va_own = pick(own & (split == "val") & ok, int(KV.get("VAL_N", 120000)))
    tr_st = np.flatnonzero(~own & (split == "train") & ok)
    ho_st = np.flatnonzero(~own & (split == "stress_hold") & ok)
    idx = np.concatenate([tr_own, va_own, tr_st, ho_st])
    role = np.array(["tr_own"] * len(tr_own) + ["va_own"] * len(va_own) + ["tr_st"] * len(tr_st) + ["ho_st"] * len(ho_st))
    print("  rows: train own %d · val own %d · stress train %d · stress hold %d (flags&3 dropped %d)"
          % (len(tr_own), len(va_own), len(tr_st), len(ho_st), (~ok).sum()), flush=True)
    # take the selected rows chunk by chunk (the full diff matrix would be ~650 MB of int16)
    order = np.argsort(idx)
    sidx = idx[order]                         # sidx[k] is row idx[order[k]]
    V = np.zeros((len(idx), 0), np.float32); ph = np.zeros(len(idx), np.float32); total = np.zeros(len(idx))
    c0 = 0
    for p in chunks:
        z = np.load(p); n = len(z["flags"])
        sel = sidx[(sidx >= c0) & (sidx < c0 + n)]
        if V.shape[1] == 0:
            V = np.zeros((len(idx), z["diff"].shape[1]), np.float32)
        dst = order[np.searchsorted(sidx, sel)]
        V[dst] = z["diff"][sel - c0]; ph[dst] = z["phase"][sel - c0]; total[dst] = z["total"][sel - c0]
        assert all(z["fen"][s - c0] == cols["fen"][s] for s in sel[::499]), "export / corpus row order differ in %s" % p
        c0 += n
    occ = np.zeros((len(idx), 192), np.float32); kf = None
    for j, i in enumerate(idx):
        o, k = pst_kauf_row(cols["fen"][i])
        occ[j] = o
        if kf is None:
            kf = np.zeros((len(idx), len(k)), np.float32)
        kf[j] = k
    a, e = (ph / 256.0)[:, None], ((256.0 - ph) / 256.0)[:, None]
    X = sp.hstack([sp.csr_matrix(V * a), sp.csr_matrix(V * e), sp.csr_matrix(occ * a), sp.csr_matrix(occ * e),
                   sp.csr_matrix(kf)], format="csr")
    names = (["v2:%d:mg" % k for k in range(V.shape[1])] + ["v2:%d:eg" % k for k in range(V.shape[1])]
             + ["pst:%d:mg" % k for k in range(192)] + ["pst:%d:eg" % k for k in range(192)]
             + ["kauf:%d" % k for k in range(kf.shape[1])])
    live = np.flatnonzero(np.asarray((X[role == "tr_own"] != 0).sum(0)).ravel() > 0)
    X = X[:, live].tocsr(); names = [names[k] for k in live]
    meta = {
        "role": role, "slice": sl[idx], "quiet": np.array([cols["quiet"][i] == "1" for i in idx]),
        "band": np.array([cols["band"][i] for i in idx]), "stage": np.array([cols["stage"][i] for i in idx]),
        "sf": np.array([float(cols["sf_cp"][i]) for i in idx]), "static": -total / 10.0,
        "stm": np.array([1.0 if cols["fen"][i].split()[1] == "w" else -1.0 for i in idx]),
    }
    print("  design: %d rows × %d live columns, nnz %d · load %.0fs" % (X.shape[0], X.shape[1], X.nnz, time.time() - t0), flush=True)
    return X, names, meta


def make_fit(X, meta, lam):
    base, stm, tgt = meta["static"], meta["stm"], wp(meta["sf"])
    nk = X.shape[1]

    def cp(p, m):
        return base[m] * (1.0 + p[-1]) - (X[m] @ p[:nk]) / 10.0 + p[-2] * stm[m]

    def fit(w, m, p0=None, nuis_only=False):
        # Optimised in STANDARDISED coordinates u = δ·σ (σ = column rms on these rows; the columns span counts of 1 to
        # Kaufman products of ~64, which left L-BFGS at its iteration cap), the L2 prior still on δ in mp; the nuisances
        # (STM in cp, SCALE as a fraction) are scaled to O(1) the same way.
        Xm, bm, sm, tm = X[m], base[m], stm[m], tgt[m]
        sig = np.sqrt(np.asarray(Xm.multiply(Xm).mean(0)).ravel()) + 1e-9
        Xs = (Xm @ sp.diags(1.0 / sig)).tocsr(); XsT = Xs.T.tocsr()
        ns = np.array([10.0, 0.1])               # u_t = t/10, u_s = s/0.1
        wn = w / w.sum()

        def fg(q):
            if nuis_only:
                u, nu = np.zeros(nk), q * ns
            else:
                u, nu = q[:nk], q[nk:] * ns
            c = bm * (1.0 + nu[1]) - (Xs @ u) / 10.0 + nu[0] * sm
            pr = wp(c); r = pr - tm
            g = 2.0 * r * pr * (1 - pr / 100.0) * K * (np.abs(c) < 1500) * wn
            gn = np.array([float(g @ sm), float(g @ bm)]) * ns
            loss = float(wn @ (r * r))
            if nuis_only:
                return loss, gn
            d = u / sig
            return loss + lam * float(d @ d) / 1e4, np.r_[-(XsT @ g) / 10.0 + 2 * lam * d / sig / 1e4, gn]
        if nuis_only:
            q0 = np.zeros(2)
        elif p0 is None:
            q0 = np.zeros(nk + 2)
        else:
            q0 = np.r_[p0[:nk] * sig, p0[nk:] / ns]
        res = minimize(fg, q0, jac=True, method="L-BFGS-B",
                       options={"maxiter": int(KV.get("MAXITER", 2000)), "maxcor": 30,
                                "ftol": float(KV.get("FTOL", 1e-10)), "gtol": 1e-9})
        if nuis_only:
            p = np.r_[np.zeros(nk), res.x * ns]
        else:
            p = np.r_[res.x[:nk] / sig, res.x[nk:] * ns]
        return p, "%d it%s" % (res.nit, "" if res.success else " (NOT CONVERGED: %s)" % str(res.message)[:40])

    def loss(p, m):
        return float(np.mean((wp(cp(p, m)) - tgt[m]) ** 2)) if m.any() else float("nan")
    return fit, loss, cp


def experiment_weights(X, names, meta):
    lam = float(KV.get("LAMBDA", 1e-2))
    fit, loss, cp = make_fit(X, meta, lam)
    role, quiet = meta["role"], meta["quiet"]
    va = (role == "va_own") & quiet
    scopes = [("val quiet", va), ("val all", role == "va_own")]
    scopes += [("band " + b, va & (meta["band"] == b)) for b in BANDS]
    scopes += [("stage " + s, va & (meta["stage"] == s)) for s in STAGES]
    scopes += [("KP hold", (role == "ho_st") & (meta["slice"] == "KP")), ("VAR hold", (role == "ho_st") & (meta["slice"] == "VAR"))]
    # one baseline for every arm: the ship (δ = 0) with nuisances fitted on the quiet own-play train rows, flat
    m0 = (role == "tr_own") & quiet
    p_base, _ = fit(np.ones(m0.sum()), m0, nuis_only=True)
    base = {n: loss(p_base, m) for n, m in scopes}
    arms = [(s, q, 0.0) for q in ("quiet", "all") for s in ("flat", "default", "strong")] + [("default", "quiet", 0.25), ("default", "quiet", 0.5)]
    if KV.get("ARMS"):
        keep = KV["ARMS"].split(",")
        arms = [a for a in arms if "%s/%s/%g" % a in keep]
    res = {}
    print("\nWEIGHTS EXPERIMENT · λ=%g · baseline = ship, val quiet win%% MSE %.2f (n=%d)" % (lam, base["val quiet"], va.sum()), flush=True)
    for scheme, rows_, w_st in arms:
        m = ((role == "tr_own") | ((role == "tr_st") & (w_st > 0)))
        if rows_ == "quiet":
            m &= quiet
        w = np.array([SCHEMES[scheme][b] for b in meta["band"][m]], float)
        w[role[m] == "tr_st"] *= w_st
        keepw = w > 0
        mm = np.flatnonzero(m)[keepw]; mask = np.zeros(len(role), bool); mask[mm] = True
        t0 = time.time()
        p, nit = fit(w[keepw], mask)
        res[(scheme, rows_, w_st)] = p
        print("  arm %-7s %-5s stress %.2f · %d train rows · %s · %.0fs · max|δ| %.0f mp · rms δ %.1f mp"
              % (scheme, rows_, w_st, mask.sum(), nit, time.time() - t0, np.abs(p[:-2]).max(), np.sqrt(np.mean(p[:-2] ** 2))), flush=True)
    labels = ["%s/%s/%g" % a for a in res]
    print("\n  val win%% MSE change vs the ship (flat-scored; negative = better)")
    print("  %-14s %8s " % ("scope", "n") + " ".join("%17s" % l for l in labels))
    for n, m in scopes:
        print("  %-14s %8d " % (n, m.sum()) + " ".join("%+16.2f%%" % (100 * (loss(p, m) / base[n] - 1)) for p in res.values()))
    ref = res.get(("default", "quiet", 0.0))
    if ref is not None:
        mv = role == "va_own"
        print("\n  eval distance from default/quiet/0 on held-out own-play positions (rms cp, cp moved = X·Δδ/10)")
        for l, p in zip(labels, res.values()):
            d = (X[mv] @ (p[:-2] - ref[:-2])) / 10.0
            print("  %-22s rms %5.1f cp · p95 |Δ| %5.1f cp" % (l, np.sqrt(np.mean(d ** 2)), np.percentile(np.abs(d), 95)))
        dship = (X[mv] @ ref[:-2]) / 10.0
        print("  (default/quiet/0 itself moves the ship by rms %.1f cp, p95 %.1f cp)" % (np.sqrt(np.mean(dship ** 2)), np.percentile(np.abs(dship), 95)))
    np.savez_compressed(os.path.join(D, "fit_weights_exp%s.npz" % KV.get("TAG", "")), names=np.array(names), labels=np.array(labels),
                        deltas=np.array([p for p in res.values()]))


if __name__ == "__main__":
    X, names, meta = load()
    {"weights": experiment_weights}[KV.get("EXPERIMENT", "weights")](X, names, meta)
