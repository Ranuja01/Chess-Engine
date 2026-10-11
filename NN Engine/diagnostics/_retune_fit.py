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


def d10_override(d10, fens):
    """D10_GLOB=<csv glob> (10-11): our d10 depth rows of the CURRENT ship (e.g. A_ours1011_d10_s*of4.csv) replace the corpus's
    ours_d10 column (the 10-04 ship's pass); rows the new pass lacks become NaN (excluded from the depth target)."""
    if not KV.get("D10_GLOB"):
        return d10
    new = {}
    for p in glob.glob(os.path.join(D, KV["D10_GLOB"])):
        for r in csv.DictReader(open(p, newline="")):
            if r.get("ours_cp_white") not in (None, ""):
                new[" ".join(r["fen"].split()[:4])] = float(r["ours_cp_white"])
    out = np.array([new.get(" ".join(f.split()[:4]), np.nan) for f in fens])
    print("  D10_GLOB %s: %d rows → %d matched" % (KV["D10_GLOB"], len(new), int(np.isfinite(out).sum())), flush=True)
    return out


def load():
    t0 = time.time()
    cols = {c: [] for c in ("fen", "slice", "split", "quiet", "band", "stage", "sf_cp", "ours_d10")}
    with open(os.path.join(D, "corpus_v1.csv"), newline="") as fh:
        rd = csv.reader(fh); hdr = next(rd); ix = {c: hdr.index(c) for c in cols}
        for r in rd:
            for c, i in ix.items():
                cols[c].append(r[i])
    # FEAT=<prefix> (10-11): the feature/static export of the CURRENT ship (feat_ship = the 10-04 ship, feat_gen2 = ship gen 2)
    chunks = sorted(glob.glob(os.path.join(D, KV.get("FEAT", "feat_ship") + "_c*.npz")))
    flags = np.concatenate([np.load(p)["flags"] for p in chunks])
    assert len(flags) == len(cols["fen"]), "export / corpus row counts differ"
    rng = np.random.default_rng(int(KV.get("SEED", 1)))
    sl = np.array(cols["slice"]); split = np.array(cols["split"])
    ok = (flags & 3) == 0
    own = np.isin(sl, ["A", "A2", "V1"])
    pick = lambda m, n: np.sort(rng.permutation(np.flatnonzero(m))[:n])
    if KV.get("EXPERIMENT") == "hybrid":
        # every slice-A row (the only rows with our d10 depth score) is kept; the rest of the quota is sampled from A2/V1
        isA = sl == "A"
        def pick(m, n):
            a = np.flatnonzero(m & isA)
            return np.sort(np.r_[a, rng.permutation(np.flatnonzero(m & ~isA))[:max(0, n - len(a))]])
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
    blocks_ = [sp.csr_matrix(V * a), sp.csr_matrix(V * e), sp.csr_matrix(occ * a), sp.csr_matrix(occ * e), sp.csr_matrix(kf)]
    names = (["v2:%d:mg" % k for k in range(V.shape[1])] + ["v2:%d:eg" % k for k in range(V.shape[1])]
             + ["pst:%d:mg" % k for k in range(192)] + ["pst:%d:eg" % k for k in range(192)]
             + ["kauf:%d" % k for k in range(kf.shape[1])])
    if KV.get("VAL") == "1":
        # PIECE VALUES (10-11, V2_VAL_* knobs): Black − White piece counts × leg; δ = mp per piece. The pawn gets only an mg leg
        # (EVAL_V2_PAWN_MG) — the endgame pawn stays the unit of account.
        cnt = np.zeros((len(idx), 5), np.float32)
        for j, i in enumerate(idx):
            pl = cols["fen"][i].split()[0]
            for c_ in pl:
                if c_.isalpha() and c_.lower() in "pnbrq":
                    cnt[j, "pnbrq".index(c_.lower())] += 1 if c_.islower() else -1
        blocks_.append(sp.csr_matrix(np.c_[cnt[:, :1] * a, cnt[:, 1:] * a, cnt[:, 1:] * e]))
        names += ["val:P:mg"] + ["val:%s:mg" % p for p in "NBRQ"] + ["val:%s:eg" % p for p in "NBRQ"]
    X = sp.hstack(blocks_, format="csr")
    live = np.flatnonzero(np.asarray((X[role == "tr_own"] != 0).sum(0)).ravel() > 0)
    X = X[:, live].tocsr(); names = [names[k] for k in live]
    meta = {
        "role": role, "slice": sl[idx], "quiet": np.array([cols["quiet"][i] == "1" for i in idx]),
        "band": np.array([cols["band"][i] for i in idx]), "stage": np.array([cols["stage"][i] for i in idx]),
        "sf": np.array([float(cols["sf_cp"][i]) for i in idx]), "static": -total / 10.0,
        "stm": np.array([1.0 if cols["fen"][i].split()[1] == "w" else -1.0 for i in idx]),
        "d10": d10_override(np.array([float(cols["ours_d10"][i]) if cols["ours_d10"][i] else np.nan for i in idx]),
                            [cols["fen"][i] for i in idx]),
        "fen": [cols["fen"][i] for i in idx],
    }
    print("  design: %d rows × %d live columns, nnz %d · load %.0fs" % (X.shape[0], X.shape[1], X.nnz, time.time() - t0), flush=True)
    return X, names, meta


def make_fit(X, meta, lam):
    base, stm, tgt = meta["static"], meta["stm"], wp(meta["sf"])
    nk = X.shape[1]

    def cp(p, m):
        return base[m] * (1.0 + p[-1]) - (X[m] @ p[:nk]) / 10.0 + p[-2] * stm[m]

    def fit(w, m, p0=None, nuis_only=False, lam_=None):
        lm = lam if lam_ is None else lam_
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
            return loss + lm * float(d @ d) / 1e4, np.r_[-(XsT @ g) / 10.0 + 2 * lm * d / sig / 1e4, gn]
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
    for sl_ in ("KP", "VAR"):
        hm = (role == "ho_st") & (meta["slice"] == sl_)
        scopes += [(sl_ + " hold", hm), (sl_ + " hold quiet", hm & quiet)]
    # one baseline for every arm: the ship (δ = 0) with nuisances fitted on the quiet own-play train rows, flat
    m0 = (role == "tr_own") & quiet
    p_base, _ = fit(np.ones(m0.sum()), m0, nuis_only=True)
    base = {n: loss(p_base, m) for n, m in scopes}
    if KV.get("EXPERIMENT", "weights") == "lambda":
        # EXPERIMENT=lambda ARM=flat/quiet/0.5 LAMBDAS=…: how much of the static gain survives a tighter pull to the ship
        s_, q_, w_ = KV.get("ARM", "flat/quiet/0.5").split("/")
        arms = [(s_, q_, float(w_), float(l)) for l in KV.get("LAMBDAS", "0.01,0.1,1,10").split(",")]
    else:
        arms = [(s, q, 0.0, lam) for q in ("quiet", "all") for s in ("flat", "default", "strong")] + \
               [("default", "quiet", 0.25, lam), ("default", "quiet", 0.5, lam)]
        if KV.get("ARMS"):
            keep = KV["ARMS"].split(",")
            arms = [a for a in arms if "%s/%s/%g" % a[:3] in keep]
    res = {}
    print("\n%s EXPERIMENT · baseline = ship, val quiet win%% MSE %.2f (n=%d)" % (KV.get("EXPERIMENT", "weights").upper(), base["val quiet"], va.sum()), flush=True)
    for scheme, rows_, w_st, lam_a in arms:
        m = ((role == "tr_own") | ((role == "tr_st") & (w_st > 0)))
        if rows_ == "quiet":
            m &= quiet
        w = np.array([SCHEMES[scheme][b] for b in meta["band"][m]], float)
        w[role[m] == "tr_st"] *= w_st
        keepw = w > 0
        mm = np.flatnonzero(m)[keepw]; mask = np.zeros(len(role), bool); mask[mm] = True
        t0 = time.time()
        p, nit = fit(w[keepw], mask, lam_=lam_a)
        res[(scheme, rows_, w_st, lam_a)] = p
        dship = (X[role == "va_own"] @ p[:-2]) / 10.0
        print("  arm %-7s %-5s stress %.2f λ %-6g · %d train rows · %s · %.0fs · max|δ| %.0f mp · rms δ %.1f mp · "
              "moves the ship rms %.1f cp (p95 %.1f)"
              % (scheme, rows_, w_st, lam_a, mask.sum(), nit, time.time() - t0, np.abs(p[:-2]).max(),
                 np.sqrt(np.mean(p[:-2] ** 2)), np.sqrt(np.mean(dship ** 2)), np.percentile(np.abs(dship), 95)), flush=True)
    labels = ["%s/%s/%g/λ%g" % a for a in res]
    print("\n  val win%% MSE change vs the ship (flat-scored; negative = better)")
    print("  %-14s %8s " % ("scope", "n") + " ".join("%17s" % l for l in labels))
    for n, m in scopes:
        print("  %-14s %8d " % (n, m.sum()) + " ".join("%+16.2f%%" % (100 * (loss(p, m) / base[n] - 1)) for p in res.values()))
    ref = res.get(("default", "quiet", 0.0, lam))
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


def fit_hybrid(X, meta, ms, ws_rows, md, Wd, lam, wstat, wdep, p0=None, pin_groups=(), pin_w=0.0):
    """The agreed HYBRID objective (10-10): wstat·L_static/L_s0 + wdep·L_depth/L_d0 + λ·|δ|²/1e4, δ SHARED; each target
    has its own (STM, SCALE) nuisances. Static rows ms (row weights ws_rows), base = ship static; depth rows md, base =
    our d10 search of the ship, design Wd = X with the non-structural columns zeroed (mobility: the dynamic lane).
    Returns p = [δ …, t_s, s_s, t_d, s_d] (δ in mp)."""
    tgt = wp(meta["sf"]); stm = meta["stm"]
    nk = X.shape[1]
    Xs_, Wd_ = X[ms], Wd[md]
    # column scale over BOTH targets' rows: a column silent on the quiet static rows but live on the depth rows got
    # σ ≈ 1e-9 here at first, a ~1e9 depth gradient, and an optimiser that stopped after one iteration
    Xu = X[ms | md]
    sig = np.sqrt(np.asarray(Xu.multiply(Xu).mean(0)).ravel())
    sig[sig < 1e-6] = 1.0
    Ds = sp.diags(1.0 / sig)
    A_s = (Xs_ @ Ds).tocsr(); A_sT = A_s.T.tocsr()
    A_d = (Wd_ @ Ds).tocsr(); A_dT = A_d.T.tocsr()
    bs, bd = meta["static"][ms], meta["d10"][md]
    ss, sd = stm[ms], stm[md]
    ts, td = tgt[ms], tgt[md]
    wn_s = ws_rows / ws_rows.sum(); wn_d = np.full(md.sum(), 1.0 / md.sum())
    ns = np.array([10.0, 0.1])

    def part(A, b, s, t, wn, u, nu):
        c = b * (1.0 + nu[1]) - (A @ u) / 10.0 + nu[0] * s
        pr = wp(c); r = pr - t
        g = 2.0 * r * pr * (1 - pr / 100.0) * K * (np.abs(c) < 1500) * wn
        return float(wn @ (r * r)), g, np.array([float(g @ s), float(g @ b)])

    def base_loss(A, b, s, t, wn):            # nuisance-only baseline on these rows
        f = lambda q: part(A, b, s, t, wn, np.zeros(nk), q * ns)[0]
        r = minimize(lambda q: (f(q), part(A, b, s, t, wn, np.zeros(nk), q * ns)[2] * ns), np.zeros(2), jac=True, method="L-BFGS-B")
        return r.fun
    Ls0 = base_loss(A_s, bs, ss, ts, wn_s) if wstat else 1.0
    Ld0 = base_loss(A_d, bd, sd, td, wn_d) if wdep else 1.0

    def fg(q):
        u = q[:nk]; o = 0.0; gu = np.zeros(nk); gn = np.zeros(4)
        if wstat:
            l, g, n_ = part(A_s, bs, ss, ts, wn_s, u, q[nk:nk + 2] * ns)
            o += wstat * l / Ls0; gu += -wstat * (A_sT @ g) / 10.0 / Ls0; gn[:2] = wstat * n_ * ns / Ls0
        if wdep:
            l, g, n_ = part(A_d, bd, sd, td, wn_d, u, q[nk + 2:] * ns)
            o += wdep * l / Ld0; gu += -wdep * (A_dT @ g) / 10.0 / Ld0; gn[2:] = wdep * n_ * ns / Ld0
        d = u / sig
        # λ is defined against the static loss scale (~125 win%²) so it means the same as in the weights / lambda runs
        lw = lam / 125.0
        pen, gpin = 0.0, np.zeros(nk)
        for g in pin_groups:                  # PIN (10-11): each PST group's MEAN δ held at 0 — material lives in val:*
            mg_ = float(d[g].mean())
            pen += pin_w * mg_ * mg_ / 1e4
            gpin[g] += 2.0 * pin_w * mg_ / len(g) / 1e4
        return o + lw * float(d @ d) / 1e4 + pen, np.r_[gu + 2 * lw * d / sig / 1e4 + gpin / sig, gn]
    q0 = np.zeros(nk + 4) if p0 is None else np.r_[p0[:nk] * sig, p0[nk:nk + 2] / ns, p0[nk + 2:] / ns]
    res = minimize(fg, q0, jac=True, method="L-BFGS-B",
                   options={"maxiter": int(KV.get("MAXITER", 2000)), "maxcor": 30, "ftol": float(KV.get("FTOL", 1e-10)), "gtol": 1e-9})
    p = np.r_[res.x[:nk] / sig, res.x[nk:nk + 2] * ns, res.x[nk + 2:] * ns]
    return p, "%d it%s" % (res.nit, "" if res.success else " (cap)")


def experiment_hybrid(X, names, meta):
    """EXPERIMENT=hybrid LAMBDAS=0.1,1 — per λ: STATIC-only vs DEPTH-only vs HYBRID (1:1), flat / quiet / stress 0.5,
    piece-value cells (kauf 36-39) FROZEN (no engine loader; values[] is never tuned). Scored on held-out own-play games:
    static (quiet rows) and depth (slice-A rows with a d10 row), each with the arm's own nuisances, per phase stage."""
    role, quiet = meta["role"], meta["quiet"]
    # frozen: piece-value cells (no engine home) and the king-distance passer cells' mg leg (v2 93-96: the engine reads
    # only their eg leg, c1_kd in v2_c1_init — an mg δ there could never be exported)
    keep = np.array([not ((n.startswith("kauf:") and int(n.split(":")[1]) >= 36)
                          or (n.startswith("v2:") and 93 <= int(n.split(":")[1]) <= 96 and n.endswith(":mg"))) for n in names])
    X = X[:, np.flatnonzero(keep)].tocsr(); names = [n for n, k in zip(names, keep) if k]
    # FILE-MIRROR TIES (10-10: the untied fit broke eval(flip_h) on 77% of positions, all from the per-file isolated-pawn
    # cells): v2 67+f and 67+(7−f) share one parameter per leg; the saved δ carries it on both names.
    col = {n: j for j, n in enumerate(names)}
    ties = []                                 # (kept index, merged-away name)
    for leg in ("mg", "eg"):
        for f in range(4):
            a_, b_ = "v2:%d:%s" % (67 + f, leg), "v2:%d:%s" % (74 - f, leg)
            if a_ in col and b_ in col:
                ties.append((col[a_], b_))
            elif b_ in col and a_ not in col:   # only the mirror is live: keep it under its own name
                pass
    if ties:
        M = sp.lil_matrix((X.shape[1], X.shape[1]))
        for j in range(X.shape[1]):
            M[j, j] = 1.0
        drop = set()
        for ja, nb in ties:
            jb = col[nb]; M[jb, ja] = 1.0; M[jb, jb] = 0.0; drop.add(jb)
        keepc = np.array([j for j in range(X.shape[1]) if j not in drop])
        X = (X @ M.tocsr())[:, keepc].tocsr()
        tied_names = names
        names = [names[j] for j in keepc]
        print("  file-mirror ties: %d pairs merged (isolated-pawn cells)" % len(ties), flush=True)
    struct = np.array([not (n.startswith("v2:") and int(n.split(":")[1]) < 66) for n in names], float)
    # PIN=<weight> (10-11, default 1000 with VAL=1, else off): PST mean per (piece, leg) held at the ship's
    pin_w = float(KV.get("PIN", "1000" if KV.get("VAL") == "1" else "0"))
    pin_groups = []
    if pin_w > 0:
        grp = {}
        for j, n in enumerate(names):
            if n.startswith("pst:"):
                _, k, leg = n.split(":")
                grp.setdefault((int(k) // 32, leg), []).append(j)
        pin_groups = [np.array(v) for v in grp.values()]
        print("  PST mean pin: %d groups, weight %g" % (len(pin_groups), pin_w), flush=True)
    Wd = (X @ sp.diags(struct)).tocsr()
    okd = (meta["slice"] == "A") & ~np.isnan(meta["d10"]) & (np.abs(meta["sf"]) < 50000) & (np.abs(np.nan_to_num(meta["d10"])) < 4999)
    w_st = float(KV.get("STRESS_W", 0.5))
    ms = ((role == "tr_own") | (role == "tr_st")) & quiet
    ws_rows = np.where(role[ms] == "tr_st", w_st, 1.0)
    md = (role == "tr_own") & okd
    vs, vd = (role == "va_own") & quiet, (role == "va_own") & okd
    print("\nHYBRID EXPERIMENT · static train %d (quiet, stress w %.2f) · depth train %d · val static %d · val depth %d · cols %d (mobility out of the depth design)"
          % (ms.sum(), w_st, md.sum(), vs.sum(), vd.sum(), X.shape[1]), flush=True)
    tgt, stm = wp(meta["sf"]), meta["stm"]

    moves = {}

    def move_of(p):                           # X·δ/10 on every row, computed once per arm
        k = id(p)
        if k not in moves:
            moves[k] = (X @ p[:X.shape[1]]) / 10.0
        return moves[k]

    def score(p, base, m, nu):
        c = base[m] * (1.0 + nu[1]) - move_of(p)[m] + nu[0] * stm[m]
        return float(np.mean((wp(c) - tgt[m]) ** 2))

    def refit_nuis(p, base, m):               # val scoring: the arm's δ, nuisances re-fitted on the TRAIN rows of that target
        mt = (role == "tr_own") & m
        f = lambda q: score(p, base, mt, q)
        return minimize(f, np.zeros(2), method="Nelder-Mead", options=dict(xatol=1e-5, fatol=1e-8, maxiter=3000)).x
    zero = np.zeros(X.shape[1])
    scopes = [("static all", vs, "s")] + [("static " + s, vs & (meta["stage"] == s), "s") for s in STAGES] + \
             [("depth all", vd, "d")] + [("depth " + s, vd & (meta["stage"] == s), "d") for s in STAGES]
    nus0 = {"s": refit_nuis(zero, meta["static"], quiet), "d": refit_nuis(zero, np.nan_to_num(meta["d10"]), okd)}
    base = {n: score(zero, meta["static"] if t == "s" else np.nan_to_num(meta["d10"]), m, nus0[t]) for n, m, t in scopes}
    res = {}
    modes = [m for m in (("static", 1, 0), ("depth", 0, 1), ("hybrid", 1, 1)) if m[0] in KV.get("MODES", "static,depth,hybrid").split(",")]
    for lam in [float(x) for x in KV.get("LAMBDAS", "0.1,1").split(",")]:
        for label, a, b in modes:
            t0 = time.time()
            p, info = fit_hybrid(X, meta, ms, ws_rows, md, Wd, lam, a, b, pin_groups=pin_groups, pin_w=pin_w)
            mv = (X[role == "va_own"] @ p[:X.shape[1]]) / 10.0
            res["%s/λ%g" % (label, lam)] = p
            print("  %-7s λ %-5g · %s · %.0fs · rms δ %.1f mp · moves the ship rms %.1f cp" % (label, lam, info, time.time() - t0,
                  np.sqrt(np.mean(p[:X.shape[1]] ** 2)), np.sqrt(np.mean(mv ** 2))), flush=True)
    nus = {k: {"s": refit_nuis(p, meta["static"], quiet), "d": refit_nuis(p, np.nan_to_num(meta["d10"]), okd)} for k, p in res.items()}
    print("\n  val win%% MSE change vs the ship (negative = better)")
    print("  %-14s %7s " % ("scope", "n") + " ".join("%14s" % k for k in res))
    for n, m, t in scopes:
        bb = meta["static"] if t == "s" else np.nan_to_num(meta["d10"])
        print("  %-14s %7d " % (n, m.sum()) + " ".join("%+13.2f%%" % (100 * (score(p, bb, m, nus[k][t]) / base[n] - 1)) for k, p in res.items()))
    blocks = {}
    for j, n in enumerate(names):
        blocks.setdefault(_block_of(n), []).append(j)
    Xv = X[role == "va_own"]
    print("\n  share of the eval change by block (covariance with the total move, held-out own-play)")
    for k, p in res.items():
        d = p[:X.shape[1]]; tot = Xv @ d / 10.0
        sh = sorted(((float(np.mean((Xv[:, js] @ d[js] / 10.0) * tot) / max(1e-9, np.mean(tot ** 2))), b) for b, js in blocks.items()), reverse=True)
        print("  %-12s " % k + " · ".join("%s %+.0f%%" % (b, 100 * s) for s, b in sh[:6]))
    # expand the tied parameters back onto both mirrored names (the export reads δ by name)
    out_names, out_d = list(names), [list(p[:X.shape[1]]) for p in res.values()]
    for ja, nb in ties:
        na = tied_names[ja]; j = names.index(na)
        out_names.append(nb)
        for row in out_d:
            row.append(row[j])
    deltas = np.array([np.r_[np.array(row), p[X.shape[1]:]] for row, p in zip(out_d, res.values())])
    np.savez_compressed(os.path.join(D, "fit_hybrid%s.npz" % KV.get("TAG", "")), names=np.array(out_names),
                        labels=np.array(list(res)), deltas=deltas)


def _block_of(n):
    if n.startswith("pst"):
        return "PST" + n.split(":")[2]
    if n.startswith("kauf"):
        return "KAUF"
    if n.startswith("val"):
        return "VAL" + n.split(":")[2]
    k = int(n.split(":")[1]); leg = n.split(":")[2]
    for lo, hi, nm in ((0, 66, "MOB"), (66, 77, "STRUCT"), (77, 97, "PASS"), (97, 106, "PLACE"), (106, 162, "KSB"),
                       (162, 172, "KFL"), (172, 184, "KPROT"), (184, 235, "PX"), (235, 337, "CONN")):
        if lo <= k < hi:
            return nm + leg
    return "?"


if __name__ == "__main__":
    X, names, meta = load()
    if KV.get("EXPERIMENT") == "hybrid":
        experiment_hybrid(X, names, meta)
    else:
        experiment_weights(X, names, meta)    # EXPERIMENT=weights | lambda (the arm list differs; scoring is shared)
