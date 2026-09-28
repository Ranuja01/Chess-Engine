# -*- coding: utf-8 -*-
"""TEXEL FIT K: the king-safety block -- tapered PSTs + KS (re-priced, non-linear) + the C3 king detectors -- fitted
JOINTLY on our own game results, as NESTED arms from one data load.

Design: dev_notes/TEXEL-C3-DETECTORS-DESIGN-2026-09-27.md §7 (the joint KS fit) and §8b (owner call 2026-09-27: OvD
is new and the KS items are known, so they are fitted NESTED -- Fit K here, Fit K+O later gated against Fit K).

MODEL (Black-positive millipawns, like the engine):
    E_i = fixed_i + X_pst,i · θ_pst + X_c3,i · θ_c3 + KS_i(θ_ks)
    fixed_i = engine total with the PST zeroed (_texel_engine_pass MODE=zero) minus the engine's own KS score
    KS_i    = v(White king) - v(Black king), a dangerous White king favouring Black, where per king
        u = max(0, F · θ_lin)       F = [w_att·n_att, weak, adj, [chk_r], [chk_q], [chk_b], [chk_n], -[no enemy queen],
                                         -1 (onset), adj_inst, unsafe, blockers, flank_att, flank_att²/8, -flank_def,
                                         -knight_def, contest_excess, contest_sq, contest_sq·[enemy queen]]
        D = MAX · u² / (u² + HALF²)
        v = D · (phase + EG·(256 - phase)) / 256        (EG = KS_V2_EG_PCT / 100)
    which is eval_v2.cpp ks_units / ks_danger_mp / the KS blend at COORD = 256 (then w_att·(256+(n-1)·256)>>8 is
    exactly w_att·n), in real arithmetic. θ_lin[0] is a scale on the attacker term (1 = shipped).
    P(White scores) = sigmoid(-K E), Texel MSE, K fitted on the start and frozen.

NESTED ARMS (one data load; each arm frees the named blocks, the rest stay at their start):
    P    PST only                  -- the re-fit baseline on this data (Fit A's recipe; C1 showed re-fits saturate)
    PK   PST + KS                  -- does re-pricing king safety pay beyond the PST?
    PKC  PST + KS + C3 detectors   -- Fit K: do shelter/storm, flank/king-pawn distance and KingProtector add?
Each arm is read against the START and against the arm below it: a block must pay INCREMENTALLY.

REGULARISATION (Fit A's recipe for the linear cells, plus a relative L2 for KS):
  - PST: half-board tied cells, per-(piece, leg) mean pinned, L2 toward the start + neighbour smoothness;
  - C3 cells: L2 toward 0 (their start), minimum-support freeze (MIN_GAMES);
  - KS: θ_ks = start + SCALE ⊙ δ with an L2 on δ (dimensionless), and bounds: defender weights >= 0 (they are
    SUBTRACTED -- the channel law), HALF >= 100, MAX >= 0, EG in [0, 150];
  - holdouts: val_hash (10% of games by hash) and val_block (every game numbered >= BLOCK_FROM -- a contiguous block
    of the opening book, i.e. different openings); bootstrap over games for the final arm with a stability filter.

  pyrun diagnostics/_texel_k_fit.py [DATA=/mnt/e/chess_data/texel] [LAMBDAS=1e-8,1e-7,1e-6] [KS_LAMBDA=1e-4]
        [ARMS=P,PK,PKC] [BOOT=3] [BOOT_ROWS=1000000] [BLOCK_FROM=27000] [ROWS=0] [TAG=fitK]
"""
import os, sys, time, json, math
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
KSF = KV.get("KS", "fitC_ks.npz")
START_PST = KV.get("START_PST", "pst_fitA.txt")
LAMBDAS = [float(x) for x in KV.get("LAMBDAS", "1e-8,1e-7,1e-6").split(",")]
LAMBDA_S_MULT = float(KV.get("LAMBDA_S_MULT", 4))
KS_LAMBDA = float(KV.get("KS_LAMBDA", 1e-4))
ARMS = KV.get("ARMS", "P,PK,PKC").split(",")
BOOT = int(KV.get("BOOT", 3))
BOOT_ROWS = int(KV.get("BOOT_ROWS", 1000000))
BLOCK_FROM = int(KV.get("BLOCK_FROM", 27000))
DECIDED = float(KV.get("DECIDED", 3000))
DECIDED_W = float(KV.get("DECIDED_W", 0.25))
STAB = float(KV.get("STAB", 2.0))
MIN_GAMES = int(KV.get("MIN_GAMES", 300))
ROWS = int(KV.get("ROWS", 0))
SEED = int(KV.get("SEED", 1))
TAG = KV.get("TAG", "fitK")
MAXITER = int(KV.get("MAXITER", 300))

C3_K0, C3_N = 106, 78                  # v2_features columns of the C3 cells (shelter/storm 56, flank 10, protector 12)
C3_BLOCKS = [("KSB_V2", 106, 56), ("KFL_V2", 162, 10), ("KPROT_V2", 172, 12)]

# ---- KS parameterisation -----------------------------------------------------------------------------------------
KS_LIN = ["ATT_SCALE", "WEAK", "ADJ", "CHK_R", "CHK_Q", "CHK_B", "CHK_N", "NO_QUEEN", "ONSET", "ADJ_INST", "UNSAFE",
          "BLOCKERS", "FLANK_ATT", "FLANK_ATT2", "FLANK_DEF", "KNIGHT_DEF", "CONTEST_EXCESS", "CONTEST_SQ",
          "CONTEST_SQ_Q"]
KS_NL = ["MAX", "HALF", "EG_PCT"]
KS_NAMES = KS_LIN + KS_NL
KS_START = dict(ATT_SCALE=1.0, WEAK=57, ADJ=61, CHK_R=122, CHK_Q=126, CHK_B=80, CHK_N=152, NO_QUEEN=321, ONSET=450,
                ADJ_INST=0, UNSAFE=0, BLOCKERS=0, FLANK_ATT=0, FLANK_ATT2=0, FLANK_DEF=0, KNIGHT_DEF=0,
                CONTEST_EXCESS=0, CONTEST_SQ=0, CONTEST_SQ_Q=0, MAX=4000, HALF=600, EG_PCT=100)
# One unit of δ is "a plausible move" for each parameter, so a single relative L2 treats them alike.
KS_SCALE = dict(ATT_SCALE=0.25, WEAK=30, ADJ=30, CHK_R=60, CHK_Q=60, CHK_B=60, CHK_N=60, NO_QUEEN=150, ONSET=150,
                ADJ_INST=20, UNSAFE=40, BLOCKERS=40, FLANK_ATT=10, FLANK_ATT2=4, FLANK_DEF=10, KNIGHT_DEF=60,
                CONTEST_EXCESS=20, CONTEST_SQ=40, CONTEST_SQ_Q=40, MAX=1000, HALF=150, EG_PCT=25)
# ATT_SCALE is PINNED at 1: the engine has no knob for it (the per-type attacker weights are constexpr), so a fitted
# value could not ship. Free it only after adding the knob.
KS_BOUNDS = dict(ATT_SCALE=(1, 1), FLANK_DEF=(0, None), KNIGHT_DEF=(0, None), MAX=(0, None), HALF=(100, None),
                 EG_PCT=(0, 150), ONSET=(0, None))
# PIN=ONSET[,...]: hold these KS parameters at their start. Owner rule 2026-09-27: KS fires only on REAL danger and a
# threshold is never loosened to raise the fire rate -- Fit K1 took the onset 450 -> 242 and firing 2-4x, so the
# onset-pinned fit is the arm that respects the rule by construction.
for _k in [p for p in KV.get("PIN", "").split(",") if p]:
    KS_BOUNDS[_k] = (KS_START[_k], KS_START[_k])
NKS = len(KS_NAMES)
S0 = np.array([KS_START[k] for k in KS_NAMES], dtype=np.float64)
SC = np.array([KS_SCALE[k] for k in KS_NAMES], dtype=np.float64)
NLIN = len(KS_LIN)


def log(*a):
    print("[fitK %6.0fs]" % (time.time() - T0), *a, flush=True)


def ks_design(ch, names):
    """Per-king unit-linear design F (n x NLIN) from the KS pass channels of ONE king."""
    ix = {n: i for i, n in enumerate(names)}
    c = lambda n: ch[:, ix[n]].astype(np.float64)
    q = (c("enemy_queen") > 0).astype(np.float64)
    fa = c("flank_att")
    cols = [c("w_att") * c("n_att"), c("weak"), c("adj"), (c("chk_r") > 0) * 1.0, (c("chk_q") > 0) * 1.0,
            (c("chk_b") > 0) * 1.0, (c("chk_n") > 0) * 1.0, -(1.0 - q), -np.ones(len(ch)), c("adj_inst"),
            c("unsafe"), c("blockers"), fa, fa * fa / 8.0, -c("flank_def"), -c("knight_def"), c("contest_excess"),
            c("contest_sq"), c("contest_sq") * q]
    return np.stack(cols, axis=1).astype(np.float32)


def ks_eval(th, Fw, Fb, ph, grad=False):
    """KS (Black-positive mp) for θ_ks = th; with grad, also dKS/dθ (n x NKS)."""
    lin, MAX, H, EG = th[:NLIN], th[NLIN], th[NLIN + 1], th[NLIN + 2] / 100.0
    g = (ph + EG * (256.0 - ph)) / 256.0
    out, J = 0.0, (np.zeros((len(ph), NKS)) if grad else None)
    for F, sgn in ((Fw, 1.0), (Fb, -1.0)):
        ur = F @ lin
        u = np.maximum(ur, 0.0)
        uu, hh = u * u, H * H
        den = uu + hh
        D = MAX * uu / den
        out = out + sgn * D * g
        if grad:
            dDdu = MAX * 2.0 * u * hh / (den * den)
            act = (ur > 0).astype(np.float64)
            J[:, :NLIN] += sgn * (dDdu * g * act)[:, None] * F
            J[:, NLIN] += sgn * (uu / den) * g
            J[:, NLIN + 1] += sgn * (-MAX * uu * 2.0 * H / (den * den)) * g
            J[:, NLIN + 2] += sgn * D * (256.0 - ph) / 256.0 / 100.0
    return (out, J) if grad else out


def mse_of(E, y, w, K):
    p = 1.0 / (1.0 + np.exp(np.clip(K * E, -60, 60)))
    return float(np.sum(w * (p - y) ** 2) / w.sum())


def main():
    log("loading", STAGE)
    st = pd.read_csv(os.path.join(DATA, STAGE), usecols=["game_id", "split", "fen", "result_white"])
    z = pd.read_csv(os.path.join(DATA, ZERO))
    fz = np.load(os.path.join(DATA, FEAT))
    kz = np.load(os.path.join(DATA, KSF))
    n = len(st)
    assert len(z) == n and len(fz["row"]) == n and len(kz["row"]) == n, \
        "passes do not cover the stage (%d / %d / %d / %d)" % (n, len(z), len(fz["row"]), len(kz["row"]))
    assert (fz["row"] == np.arange(n)).all() and (kz["row"] == np.arange(n)).all() and (z["row"].values == np.arange(n)).all()
    phase = z["phase256"].values.astype(np.float64)
    zero_total = z["total"].values.astype(np.float64)
    full = fz["total"].astype(np.float64)
    ks_eng = kz["ks_engine"].astype(np.float64)
    flags = fz["flags"]
    names = [str(x) for x in kz["names"]]
    D3 = fz["diff"][:, C3_K0:C3_K0 + C3_N].astype(np.float32)
    CH = kz["ch"]

    keep = (phase >= 0) & ((flags & 3) == 0) & (np.abs(full) < 30000) & (np.abs(zero_total) < 30000)
    idx = np.where(keep)[0]
    if ROWS:
        rng0 = np.random.default_rng(SEED)
        idx = np.sort(rng0.choice(idx, size=min(ROWS, len(idx)), replace=False))
    st = st.iloc[idx].reset_index(drop=True)
    phase, zero_total, full, ks_eng, D3, CH = phase[idx], zero_total[idx], full[idx], ks_eng[idx], D3[idx], CH[idx]
    log("rows after filter:", len(st))

    # ---- design matrices ------------------------------------------------------------------------------------------
    theta_pst0, start_full = A.load_start_table(os.path.join(DATA, START_PST))
    Xp = A.build_features(st["fen"].values, phase)
    wm = (phase / 256.0).astype(np.float32)
    D3s = sp.csr_matrix(D3)                                  # built sparse: a dense n x 156 float would be ~1.2 GB
    Xc = sp.hstack([sp.diags(wm) @ D3s, sp.diags(1.0 - wm) @ D3s]).tocsr()   # [mg cells | eg cells]
    NC = 2 * C3_N
    Fw, Fb = ks_design(CH[:, 0], names), ks_design(CH[:, 1], names)
    fixed = zero_total - ks_eng
    y = st["result_white"].values.astype(np.float64)

    ks0 = ks_eval(S0, Fw, Fb, phase)
    E0 = fixed + Xp @ theta_pst0 + ks0
    resid = full - E0
    inert = np.abs(resid) > 25
    log("start model vs engine |full - E0|: median %.2f mp  p99 %.1f  inert rows %d (%.2f%%)  [KS real-vs-int: max %.1f mp]"
        % (np.median(np.abs(resid)), np.percentile(np.abs(resid), 99), inert.sum(), 100 * inert.mean(),
           np.abs(ks0 - ks_eng).max()))

    gnum = st["game_id"].str.extract(r"game_(\d+)")[0].astype(int).values
    vblock = (gnum >= BLOCK_FROM) & ~inert
    vhash = (st["split"] == "val").values & ~vblock & ~inert
    train = ~vblock & ~vhash & ~inert
    gsize = st.groupby("game_id")["fen"].transform("size").values.astype(np.float64)
    w = 1.0 / gsize
    w = np.where(np.abs(E0) > DECIDED, w * DECIDED_W, w)
    log("train %d  val_hash %d  val_block(game >= %d) %d" % (train.sum(), vhash.sum(), BLOCK_FROM, vblock.sum()))

    K = A.fit_K(E0[train], y[train], w[train])
    log("K (start, frozen) = %.6f / mp" % K)

    # Support: games in which a C3 cell is non-zero (train only); thin cells stay at 0.
    games = st["game_id"].values
    gcode = pd.factorize(games)[0]                          # integer game codes: np.unique on strings is slow
    c3_support = np.zeros(C3_N, dtype=np.int64)
    tr_codes = gcode[train]
    D3t = D3[train]
    for j in range(C3_N):
        c3_support[j] = len(np.unique(tr_codes[D3t[:, j] != 0]))
    c3_active = c3_support >= MIN_GAMES
    log("C3 cells with >= %d games of support: %d of %d" % (MIN_GAMES, c3_active.sum(), C3_N))
    pst_active = np.asarray((abs(Xp[train]).sum(axis=0) > 0)).ravel()
    proj_pst = A.make_projector(pst_active)
    pairs = A.neighbour_pairs()
    pa, pb = pairs[:, 0], pairs[:, 1]

    NP_ = A.NPAR
    bounds_ks = []
    for k in KS_NAMES:
        lo, hi = KS_BOUNDS.get(k, (None, None))
        s0, sc = KS_START[k], KS_SCALE[k]
        bounds_ks.append(((lo - s0) / sc if lo is not None else None, (hi - s0) / sc if hi is not None else None))

    def unpack(x):
        return x[:NP_], x[NP_:NP_ + NC], x[NP_ + NC:]

    def model(x, rows):
        dp, dc, dk = unpack(x)
        th_ks = S0 + SC * dk
        return fixed[rows] + Xp[rows] @ (theta_pst0 + proj_pst(dp)) + Xc[rows] @ dc + ks_eval(th_ks, Fw[rows], Fb[rows], phase[rows])

    def fit_arm(arm, rows, lam, x0=None):
        free_p, free_k, free_c = "P" in arm, "K" in arm, "C" in arm
        Xpr, Xcr, Fwr, Fbr = Xp[rows], Xc[rows], Fw[rows], Fb[rows]
        fr, yr, wr, phr = fixed[rows], y[rows], w[rows], phase[rows]
        wsum = wr.sum()
        c3mask = np.concatenate([c3_active, c3_active]).astype(np.float64)

        def fg(x):
            dp, dc, dk = unpack(x)
            dp = proj_pst(dp) if free_p else np.zeros(NP_)
            dc = dc * c3mask if free_c else np.zeros(NC)
            dk = dk if free_k else np.zeros(NKS)
            if free_k:
                ksv, J = ks_eval(S0 + SC * dk, Fwr, Fbr, phr, grad=True)
            else:
                ksv, J = ks_eval(S0, Fwr, Fbr, phr), None
            E = fr + Xpr @ (theta_pst0 + dp) + Xcr @ dc + ksv
            p = 1.0 / (1.0 + np.exp(np.clip(K * E, -60, 60)))
            res = p - yr
            loss = float(np.sum(wr * res * res) / wsum)
            dE = (wr * 2.0 * res * (-K) * p * (1.0 - p)) / wsum
            g = np.zeros_like(x)
            if free_p:
                diff = dp[pa] - dp[pb]
                loss += lam * float(dp @ dp) + lam * LAMBDA_S_MULT * float(diff @ diff)
                gp = Xpr.T @ dE + 2.0 * lam * dp
                gs = np.zeros(NP_)
                np.add.at(gs, pa, 2.0 * lam * LAMBDA_S_MULT * diff)
                np.add.at(gs, pb, -2.0 * lam * LAMBDA_S_MULT * diff)
                g[:NP_] = proj_pst(gp + gs)
            if free_c:
                loss += lam * float(dc @ dc)
                g[NP_:NP_ + NC] = (Xcr.T @ dE + 2.0 * lam * dc) * c3mask
            if free_k:
                loss += KS_LAMBDA * float(dk @ dk)
                g[NP_ + NC:] = (J.T @ dE) * SC + 2.0 * KS_LAMBDA * dk
            return loss, g

        bnds = [(None, None)] * (NP_ + NC) + (bounds_ks if free_k else [(0, 0)] * NKS)
        r = minimize(fg, np.zeros(NP_ + NC + NKS) if x0 is None else x0, jac=True, method="L-BFGS-B", bounds=bnds,
                     options={"maxiter": MAXITER, "gtol": 1e-12, "ftol": 1e-13})
        x = r.x.copy()
        x[:NP_] = proj_pst(x[:NP_]) if free_p else 0.0
        x[NP_:NP_ + NC] = x[NP_:NP_ + NC] * c3mask if free_c else 0.0
        if not free_k:
            x[NP_ + NC:] = 0.0
        return x, r

    tr_idx = np.where(train)[0]
    rng = np.random.default_rng(SEED)
    sub = np.sort(rng.choice(tr_idx, size=min(BOOT_ROWS, len(tr_idx)), replace=False))
    x_start = np.zeros(NP_ + NC + NKS)
    base = {k: mse_of(model(x_start, m), y[m], w[m], K) for k, m in
            (("train", train), ("val_hash", vhash), ("val_block", vblock))}
    log("START  train %.6f  val_hash %.6f  val_block %.6f" % (base["train"], base["val_hash"], base["val_block"]))

    # λ chosen on the FULL arm (PKC); the nested arms reuse it so they differ only in which blocks are free.
    results = {}
    lam_best, best_vb = LAMBDAS[0], None
    for lam in LAMBDAS:
        x, r = fit_arm("PKC", sub, lam)
        vb = mse_of(model(x, vblock), y[vblock], w[vblock], K)
        vh = mse_of(model(x, vhash), y[vhash], w[vhash], K)
        log("lambda %.0e (PKC)  val_hash %+.3f%%  val_block %+.3f%%  iters %d"
            % (lam, 100 * (vh / base["val_hash"] - 1), 100 * (vb / base["val_block"] - 1), r.nit))
        if best_vb is None or vb < best_vb:
            best_vb, lam_best = vb, lam
    log("chosen lambda = %.0e (best val_block)" % lam_best)

    arm_x = {}
    for arm in ARMS:
        x, r = fit_arm(arm, sub, lam_best)
        arm_x[arm] = x
        res = {k: mse_of(model(x, m), y[m], w[m], K) for k, m in (("val_hash", vhash), ("val_block", vblock))}
        results[arm] = res
        th_ks = S0 + SC * x[NP_ + NC:]
        log("ARM %-4s val_hash %+.3f%%  val_block %+.3f%%  iters %d%s" % (
            arm, 100 * (res["val_hash"] / base["val_hash"] - 1), 100 * (res["val_block"] / base["val_block"] - 1),
            r.nit, ("   KS: " + " ".join("%s=%.3g" % (k, v) for k, v in zip(KS_NAMES, th_ks))) if "K" in arm else ""))
    for a, b in (("PK", "P"), ("PKC", "PK")):
        if a in results and b in results:
            log("INCREMENT %s over %s: val_hash %+.3f%%  val_block %+.3f%%" % (
                a, b, 100 * (results[a]["val_hash"] / results[b]["val_hash"] - 1),
                100 * (results[a]["val_block"] / results[b]["val_block"] - 1)))

    # ---- bootstrap the final arm over games; stability filter; write engine-loadable outputs ---------------------
    final_arm = ARMS[-1]
    rows_by_game = pd.Series(np.arange(len(st)))[train].groupby(games[train]).apply(np.array)
    ugames = np.array(rows_by_game.index)
    deltas = []
    for b in range(BOOT):
        pick = rng.choice(ugames, size=len(ugames), replace=True)
        bi = np.concatenate([rows_by_game[g] for g in pick])
        if len(bi) > BOOT_ROWS:
            bi = rng.choice(bi, size=BOOT_ROWS, replace=False)
        x, r = fit_arm(final_arm, np.sort(bi), lam_best, x0=arm_x[final_arm])
        deltas.append(x)
        log("bootstrap %d/%d  val_block %+.3f%%  iters %d" % (
            b + 1, BOOT, 100 * (mse_of(model(x, vblock), y[vblock], w[vblock], K) / base["val_block"] - 1), r.nit))
    Dx = np.array(deltas)
    dmean = Dx.mean(axis=0)
    dsd = Dx.std(axis=0, ddof=1) if BOOT > 1 else np.zeros_like(dmean)
    stable = np.abs(dmean) >= STAB * np.maximum(dsd, 1e-9)
    xf = np.where(stable, dmean, 0.0)
    xf[:NP_] = proj_pst(xf[:NP_])
    fin = {k: mse_of(model(xf, m), y[m], w[m], K) for k, m in (("train", train), ("val_hash", vhash), ("val_block", vblock))}
    for k in ("train", "val_hash", "val_block"):
        log("FINAL %-9s start %.6f  fitted %.6f  (%+.3f%%)" % (k, base[k], fin[k], 100 * (fin[k] / base[k] - 1)))

    dp, dc, dk = unpack(xf)
    A.write_table(os.path.join(DATA, "pst_%s.txt" % TAG), A.expand(theta_pst0 + dp),
                  "Texel fit K (%s) lambda=%g K=%.6f boot=%d; val_block %+.3f%%"
                  % (final_arm, lam_best, K, BOOT, 100 * (fin["val_block"] / base["val_block"] - 1)))
    for name, k0, ncell in C3_BLOCKS:
        with open(os.path.join(DATA, "%s_%s.txt" % (name.lower(), TAG)), "w") as fo:
            fo.write("# Texel fit K %s cells: feature_k leg start fitted (load with %s=1 %s_FILE=<this>)\n" % (name, name, name))
            for j in range(ncell):
                c = k0 - C3_K0 + j
                for leg in (0, 1):
                    v = dc[leg * C3_N + c]
                    fo.write("%d %d 0 %d\n" % (k0 + j, leg, int(round(v))))
    th_ks = S0 + SC * dk
    knobs = []
    for k, v in zip(KS_NAMES, th_ks):
        if k == "ATT_SCALE":
            continue
        knobs.append("KS_V2_%s=%d" % (k, int(round(v))))
    with open(os.path.join(DATA, "ks_%s.txt" % TAG), "w") as fo:
        fo.write("# Texel fit K KS knobs (ATT_SCALE=%.3f is NOT an engine knob yet: 1.0 = shipped)\n" % th_ks[0])
        fo.write(" ".join(knobs) + "\n")
    log("KS fitted: " + " ".join("%s=%.3g(sd %.2g)" % (k, v, s * sc) for k, v, s, sc in
                                zip(KS_NAMES, th_ks, dsd[NP_ + NC:], SC)))
    log("C3 cells kept after stability: %d of %d active" % (int((stable[NP_:NP_ + NC] & (dc != 0)).sum()),
                                                         2 * int(c3_active.sum())))
    json.dump({"K": K, "lambda": lam_best, "base": base, "arms": results, "final": fin, "final_arm": final_arm,
               "ks": dict(zip(KS_NAMES, th_ks.tolist()))}, open(os.path.join(DATA, "%s_report.json" % TAG), "w"), indent=1)
    log("wrote pst_%s.txt, ksb/kfl/kprot_v2_%s.txt, ks_%s.txt, %s_report.json" % (TAG, TAG, TAG, TAG))


if __name__ == "__main__":
    T0 = time.time()
    A.T0 = T0
    main()
