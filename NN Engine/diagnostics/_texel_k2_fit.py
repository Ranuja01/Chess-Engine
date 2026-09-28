# -*- coding: utf-8 -*-
"""TEXEL FIT K2: king-safety STRUCTURE arms, each gated against Fit K1p on held-out loss.

Design: dev_notes/TEXEL-C3-DETECTORS-DESIGN-2026-09-27.md §10. K1p re-priced every CONTINUOUS KS number; what is left
is structure: the attacker weight per type (N/B/R/Q, compile-time until 2026-09-28), the coordination multiplier, and
the MODES a fit cannot choose between (x-ray attackers, the defence-aware attacker weight, the two-attacker gate).
Each mode is its own ARM; every arm starts from K1p's values, so the start loss IS K1p and an arm's held-out change is
its increment over K1p directly.

ATTRIBUTION: the PSTs and C3 cells are FROZEN at K1p (their contribution is precomputed once), only KS is fitted. A
"base" arm with exactly K1p's freedom is the control -- it must read ~0 over the start, or the harness is not at K1p.

KS model per king (real arithmetic, as _texel_k_fit.py, with the attacker term opened up):
    att = (Σ_t W_t · A_t) · (1 + (n - 1) · COORD / 256)   if n > 0     (at COORD 256 this is w_att · n)
    u   = gate · max(0, att + F · θ_lin)                               F / θ_lin as in _texel_k_fit (onset pinned)
    D   = MAX u² / (u² + HALF²);   v = D (phase + EG (256 - phase)) / 256;   KS = v(White king) - v(Black king)
Mode sources for (A, n): plain = (att_t, n_att) · xray = (att_x_t, n_att_x) · defaware = (share_t / 256, n_att).
gate (arm "gate") = n_att >= 2 or (n_att >= 1 and the enemy queen is on).

  pyrun diagnostics/_texel_k2_fit.py [ARMS=base,att,xray,defaware,gate] [START=fitK1p] [KS_LAMBDA=1e-5] [TAG=fitK2]
        [KS=fitC_ks2.npz]  (the channel pass run AFTER the 2026-09-28 per-type export build)
"""
import os, sys, time, json
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.optimize import minimize

THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS)
import _texel_pst_fit as A

KV = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
DATA = KV.get("DATA", "/mnt/e/chess_data/texel")
STAGE = KV.get("STAGE", "fitC_stage1.csv.gz")
ZERO = KV.get("ZERO", "fitC_pass/zero_0_of_1.csv")
FEAT = KV.get("FEAT", "fitC_features.npz")
KSF = KV.get("KS", "fitC_ks2.npz")
START = KV.get("START", "fitK1p")
REF_PST = KV.get("REF_PST", "pst_fitA.txt")          # the SHIPPED table: used only to find engine-inert rows
ARMS = KV.get("ARMS", "base,att,xray,defaware,gate").split(",")
KS_LAMBDA = float(KV.get("KS_LAMBDA", 1e-5))
BLOCK_FROM = int(KV.get("BLOCK_FROM", 27000))
DECIDED = float(KV.get("DECIDED", 3000))
DECIDED_W = float(KV.get("DECIDED_W", 0.25))
FIT_ROWS = int(KV.get("FIT_ROWS", 1000000))
MAXITER = int(KV.get("MAXITER", 300))
SEED = int(KV.get("SEED", 1))
TAG = KV.get("TAG", "fitK2")
T0 = time.time()

C3_K0, C3_N = 106, 78
C3_FILES = [("ksb_v2", 106, 56), ("kfl_v2", 162, 10), ("kprot_v2", 172, 12)]
TYPES = "nbrq"

KS_ATT = ["W_N", "W_B", "W_R", "W_Q", "COORD"]
KS_LIN = ["WEAK", "ADJ", "CHK_R", "CHK_Q", "CHK_B", "CHK_N", "NO_QUEEN", "ONSET", "ADJ_INST", "UNSAFE", "BLOCKERS",
          "FLANK_ATT", "FLANK_ATT2", "FLANK_DEF", "KNIGHT_DEF", "CONTEST_EXCESS", "CONTEST_SQ", "CONTEST_SQ_Q"]
KS_NL = ["MAX", "HALF", "EG_PCT"]
KS_NAMES = KS_ATT + KS_LIN + KS_NL
NA, NL_ = len(KS_ATT), len(KS_LIN)
SHIP = dict(W_N=31, W_B=31, W_R=47, W_Q=78, COORD=256, WEAK=57, ADJ=61, CHK_R=122, CHK_Q=126, CHK_B=80, CHK_N=152,
            NO_QUEEN=321, ONSET=450, ADJ_INST=0, UNSAFE=0, BLOCKERS=0, FLANK_ATT=0, FLANK_ATT2=0, FLANK_DEF=0,
            KNIGHT_DEF=0, CONTEST_EXCESS=0, CONTEST_SQ=0, CONTEST_SQ_Q=0, MAX=4000, HALF=600, EG_PCT=100)
SCALE = dict(W_N=10, W_B=10, W_R=15, W_Q=20, COORD=64, WEAK=30, ADJ=30, CHK_R=60, CHK_Q=60, CHK_B=60, CHK_N=60,
             NO_QUEEN=150, ONSET=150, ADJ_INST=20, UNSAFE=40, BLOCKERS=40, FLANK_ATT=10, FLANK_ATT2=4, FLANK_DEF=10,
             KNIGHT_DEF=60, CONTEST_EXCESS=20, CONTEST_SQ=40, CONTEST_SQ_Q=40, MAX=1000, HALF=150, EG_PCT=25)
BOUNDS = dict(W_N=(0, None), W_B=(0, None), W_R=(0, None), W_Q=(0, None), COORD=(0, 512), FLANK_DEF=(0, None),
              KNIGHT_DEF=(0, None), MAX=(0, None), HALF=(100, None), EG_PCT=(0, 150))
PINNED_ALWAYS = {"ONSET"}                            # owner rule: the threshold is never loosened by a fit here


def log(*a):
    print("[fitK2 %6.0fs]" % (time.time() - T0), *a, flush=True)


def read_ks_knobs(path):
    d = {}
    for line in open(path):
        if line.startswith("#"):
            continue
        for kv in line.split():
            k, v = kv.split("=", 1)
            d[k.replace("KS_V2_", "")] = float(v)
    return d


def side_data(ch, names, mode):
    """(A n x 4, n, gate, F n x NL_) for ONE king under an attacker MODE."""
    ix = {nm: i for i, nm in enumerate(names)}
    c = lambda nm: ch[:, ix[nm]].astype(np.float64)
    q = (c("enemy_queen") > 0).astype(np.float64)
    if mode == "xray":
        Am = np.stack([c("att_x_" + t) for t in TYPES], 1)
        nn = c("n_att_x")
    elif mode == "defaware":
        Am = np.stack([c("share_" + t) / 256.0 for t in TYPES], 1)
        nn = c("n_att")
    else:
        Am = np.stack([c("att_" + t) for t in TYPES], 1)
        nn = c("n_att")
    gate = ((c("n_att") >= 2) | ((c("n_att") >= 1) & (q > 0))).astype(np.float64)
    fa = c("flank_att")
    F = np.stack([c("weak"), c("adj"), (c("chk_r") > 0) * 1.0, (c("chk_q") > 0) * 1.0, (c("chk_b") > 0) * 1.0,
                  (c("chk_n") > 0) * 1.0, -(1.0 - q), -np.ones(len(ch)), c("adj_inst"), c("unsafe"), c("blockers"),
                  fa, fa * fa / 8.0, -c("flank_def"), -c("knight_def"), c("contest_excess"), c("contest_sq"),
                  c("contest_sq") * q], 1)
    return Am.astype(np.float32), nn, gate, F.astype(np.float32)


def ks_eval(th, sides, ph, use_gate, grad=False):
    W, C = th[:4], th[4]
    lin = th[NA:NA + NL_]
    MAX, H, EG = th[NA + NL_], th[NA + NL_ + 1], th[NA + NL_ + 2] / 100.0
    g = (ph + EG * (256.0 - ph)) / 256.0
    out = 0.0
    J = np.zeros((len(ph), len(th))) if grad else None
    for (Am, nn, gate, F), sgn in zip(sides, (1.0, -1.0)):
        has = (nn > 0).astype(np.float64)
        m = (1.0 + (nn - 1.0) * C / 256.0) * has
        aw = Am @ W
        ur = aw * m + F @ lin
        gt = gate if use_gate else 1.0
        u = np.maximum(ur, 0.0) * gt
        uu, hh = u * u, H * H
        den = uu + hh
        D = MAX * uu / den
        out = out + sgn * D * g
        if grad:
            act = ((ur > 0) * gt).astype(np.float64)
            k = sgn * MAX * 2.0 * u * hh / (den * den) * g * act
            J[:, :4] += k[:, None] * Am * m[:, None]
            J[:, 4] += k * aw * (nn - 1.0) / 256.0 * has
            J[:, NA:NA + NL_] += k[:, None] * F
            J[:, NA + NL_] += sgn * (uu / den) * g
            J[:, NA + NL_ + 1] += sgn * (-MAX * uu * 2.0 * H / (den * den)) * g
            J[:, NA + NL_ + 2] += sgn * D * (256.0 - ph) / 256.0 / 100.0
    return (out, J) if grad else out


def mse_of(E, y, w, K):
    p = 1.0 / (1.0 + np.exp(np.clip(K * E, -60, 60)))
    return float(np.sum(w * (p - y) ** 2) / w.sum())


def main():
    log("loading")
    st = pd.read_csv(os.path.join(DATA, STAGE), usecols=["game_id", "split", "fen", "result_white"])
    z = pd.read_csv(os.path.join(DATA, ZERO))
    fz = np.load(os.path.join(DATA, FEAT))
    kz = np.load(os.path.join(DATA, KSF))
    names = [str(x) for x in kz["names"]]
    assert "att_n" in names, "%s predates the per-type export -- rerun _texel_ks_pass.py after the 09-28 build" % KSF
    n = len(st)
    assert len(z) == n and len(fz["row"]) == n and len(kz["row"]) == n
    phase = z["phase256"].values.astype(np.float64)
    zero_total = z["total"].values.astype(np.float64)
    full = fz["total"].astype(np.float64)
    ks_eng = kz["ks_engine"].astype(np.float64)
    keep = (phase >= 0) & ((fz["flags"] & 3) == 0) & (np.abs(full) < 30000) & (np.abs(zero_total) < 30000)
    idx = np.where(keep)[0]
    st = st.iloc[idx].reset_index(drop=True)
    phase, zero_total, full, ks_eng = phase[idx], zero_total[idx], full[idx], ks_eng[idx]
    D3 = fz["diff"][idx, C3_K0:C3_K0 + C3_N].astype(np.float32)
    CH = kz["ch"][idx]
    log("rows", len(st))

    # Frozen blocks at K1p: PST table + C3 cells, precomputed once.
    th_pst, _ = A.load_start_table(os.path.join(DATA, "pst_%s.txt" % START))
    th_ref, _ = A.load_start_table(os.path.join(DATA, REF_PST))
    Xp = A.build_features(st["fen"].values, phase)
    c3 = np.zeros(2 * C3_N)
    for fname, k0, ncell in C3_FILES:
        for line in open(os.path.join(DATA, "%s_%s.txt" % (fname, START))):
            if line.startswith("#"):
                continue
            k, leg, _s, v = line.split()
            c3[int(leg) * C3_N + int(k) - C3_K0] = float(v)
    wm = (phase / 256.0).astype(np.float32)
    D3s = sp.csr_matrix(D3)
    Xc = sp.hstack([sp.diags(wm) @ D3s, sp.diags(1.0 - wm) @ D3s]).tocsr()
    fixed = zero_total - ks_eng
    frozen = fixed + Xp @ th_pst + Xc @ c3
    y = st["result_white"].values.astype(np.float64)

    # Engine-inert rows (draw classifier etc.) are found against the SHIPPED model, where the engine's own total is known.
    k1p = read_ks_knobs(os.path.join(DATA, "ks_%s.txt" % START))
    th_ship = np.array([SHIP[k] for k in KS_NAMES])
    th_start = np.array([k1p.get(k, SHIP[k]) for k in KS_NAMES])
    plain = [side_data(CH[:, s], names, "plain") for s in (0, 1)]
    E_ref = fixed + Xp @ th_ref + ks_eval(th_ship, plain, phase, False)
    inert = np.abs(full - E_ref) > 25
    log("shipped reference vs engine: median %.2f mp, inert rows %d" % (np.median(np.abs(full - E_ref)), inert.sum()))

    gnum = st["game_id"].str.extract(r"game_(\d+)")[0].astype(int).values
    vblock = (gnum >= BLOCK_FROM) & ~inert
    vhash = (st["split"] == "val").values & ~vblock & ~inert
    train = ~vblock & ~vhash & ~inert
    gsize = st.groupby("game_id")["fen"].transform("size").values.astype(np.float64)
    w = 1.0 / gsize
    E0 = frozen + ks_eval(th_start, plain, phase, False)
    w = np.where(np.abs(E0) > DECIDED, w * DECIDED_W, w)
    K = A.fit_K(E0[train], y[train], w[train])
    base = {k: mse_of(E0[m], y[m], w[m], K) for k, m in (("val_hash", vhash), ("val_block", vblock))}
    log("K %.6f   START (= %s)  val_hash %.6f  val_block %.6f" % (K, START, base["val_hash"], base["val_block"]))

    rng = np.random.default_rng(SEED)
    tr = np.where(train)[0]
    sub = np.sort(rng.choice(tr, size=min(FIT_ROWS, len(tr)), replace=False))
    sc = np.array([SCALE[k] for k in KS_NAMES])
    results = {}
    for arm in ARMS:
        mode = {"xray": "xray", "defaware": "defaware"}.get(arm, "plain")
        use_gate = arm == "gate"
        free_att = arm != "base"
        sides = [side_data(CH[:, s], names, mode) for s in (0, 1)]
        sides_sub = [tuple(a[sub] if hasattr(a, "__len__") else a for a in sd) for sd in sides]
        bnds = []
        for k, s0, s in zip(KS_NAMES, th_start, sc):
            if k in PINNED_ALWAYS or (k in KS_ATT and not free_att):
                bnds.append((0.0, 0.0))
                continue
            lo, hi = BOUNDS.get(k, (None, None))
            bnds.append(((lo - s0) / s if lo is not None else None, (hi - s0) / s if hi is not None else None))
        fr, yr, wr, phr = frozen[sub], y[sub], w[sub], phase[sub]
        wsum = wr.sum()

        def fg(d):
            ksv, J = ks_eval(th_start + sc * d, sides_sub, phr, use_gate, grad=True)
            E = fr + ksv
            p = 1.0 / (1.0 + np.exp(np.clip(K * E, -60, 60)))
            res = p - yr
            loss = float(np.sum(wr * res * res) / wsum) + KS_LAMBDA * float(d @ d)
            dE = (wr * 2.0 * res * (-K) * p * (1.0 - p)) / wsum
            return loss, (J.T @ dE) * sc + 2.0 * KS_LAMBDA * d

        r = minimize(fg, np.zeros(len(KS_NAMES)), jac=True, method="L-BFGS-B", bounds=bnds,
                     options={"maxiter": MAXITER, "gtol": 1e-12, "ftol": 1e-13})
        th = th_start + sc * r.x
        E = frozen + ks_eval(th, sides, phase, use_gate)
        res = {k: mse_of(E[m], y[m], w[m], K) for k, m in (("val_hash", vhash), ("val_block", vblock))}
        results[arm] = {"val_hash": res["val_hash"], "val_block": res["val_block"], "ks": dict(zip(KS_NAMES, th.tolist()))}
        log("ARM %-9s vs K1p: val_hash %+.3f%%  val_block %+.3f%%  iters %d  | W %s COORD %.0f" % (
            arm, 100 * (res["val_hash"] / base["val_hash"] - 1), 100 * (res["val_block"] / base["val_block"] - 1), r.nit,
            "/".join("%.0f" % v for v in th[:4]), th[4]))
        knobs = ["KS_V2_%s=%d" % (k, int(round(v))) for k, v in zip(KS_NAMES, th)]
        if mode == "xray":
            knobs.append("KS_V2_ATT_XRAY=1")
        if mode == "defaware":
            knobs.append("KS_V2_DEFAWARE=1")
        if use_gate:
            knobs.append("KS_V2_GATE=1")
        with open(os.path.join(DATA, "ks_%s_%s.txt" % (TAG, arm)), "w") as fo:
            fo.write("# Fit K2 arm %s (start %s): val_hash %+.3f%% val_block %+.3f%% vs start\n" % (
                arm, START, 100 * (res["val_hash"] / base["val_hash"] - 1), 100 * (res["val_block"] / base["val_block"] - 1)))
            fo.write(" ".join(knobs) + "\n")
    json.dump({"K": K, "start": START, "base": base, "arms": results}, open(os.path.join(DATA, "%s_report.json" % TAG), "w"),
              indent=1)
    log("wrote ks_%s_<arm>.txt and %s_report.json" % (TAG, TAG))


if __name__ == "__main__":
    main()
