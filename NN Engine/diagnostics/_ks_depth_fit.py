# -*- coding: utf-8 -*-
"""FULL KS TUNE, step 2 (owner, 10-02): the KS attack model (Fit K's shipped knobs) + KS-B shelter/storm cells, fitted
JOINTLY on the DEPTH target, as nested arms. Step 1 (`_ksb_depth_fit.py`) priced KS-B alone (val −3.08%).

Target / base: SF18 d14 label vs ours' = ours_d10 (our d10 SEARCH, White cp) + Δ, Δ = −[KS(θ) − KS(θ_ship)]/10
− Σ diff·θ_ksb·phase/256/10 (Black-positive mp → White cp). KS(θ) = `_texel_k2_fit.ks_eval` (eval_v2.cpp's KS in real
arithmetic, plain attacker mode, no gate — the shipped modes) on the per-type channels of fitC_ks2.npz. A CLOSURE check
first: KS(θ_ship) must match the engine's published KS (ks_engine) on these rows.
ARMS: KS (θ_ks free, ONSET pinned — owner rule) · KSB (cells only) · KS+KSB (joint) — each read vs the base and vs the
arm below (a block must pay incrementally). θ_ks = ship + SCALE ⊙ δ with L2 on δ (K2's scales/bounds); cells L2.

  pyrun diagnostics/_ks_depth_fit.py [KS_LAMBDA=1e-3] [KSB_LAMBDA=1e-2] [OUT=ks_depth]
"""
import os, sys, csv, glob, hashlib
import numpy as np
from scipy.optimize import minimize

KV = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS)
import _texel_k2_fit as K2

DATA = "/mnt/e/chess_data/texel"
KW = 0.00368208
KSB0, KSB1 = 106, 162
SHIP = dict(K2.SHIP)
SHIP.update(W_N=31, W_B=31, W_R=47, W_Q=78, COORD=256, WEAK=64, ADJ=50, CHK_R=193, CHK_Q=260, CHK_B=141, CHK_N=189,
            NO_QUEEN=402, ONSET=450, ADJ_INST=-12, UNSAFE=19, BLOCKERS=0, FLANK_ATT=11, FLANK_ATT2=-1, FLANK_DEF=0,
            KNIGHT_DEF=15, CONTEST_EXCESS=14, CONTEST_SQ=30, CONTEST_SQ_Q=19, MAX=4000, HALF=646, EG_PCT=100)


def wp(cp):
    return 100.0 / (1.0 + np.exp(-KW * np.clip(cp, -1500, 1500)))


def main():
    ours_d = {}
    for p in glob.glob(os.path.join(THIS, "ks_sets", "fitC_mg_ours_d10_s*of4.csv")):
        for r in csv.DictReader(open(p, newline="")):
            ours_d[r["fen"]] = float(r["ours_cp_white"])
    sample = {r["fen"]: (int(r["row"]), r["game_id"]) for r in
              csv.DictReader(open(os.path.join(THIS, "ks_sets/fitC_mg_sample.csv"))) if r["src"] == "std"}
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
    kz = np.load(os.path.join(DATA, "fitC_ks2.npz"))
    names = [str(x) for x in kz["names"]]
    CH, PHK, KSE = kz["ch"][rows], kz["phase"][rows].astype(np.float64), kz["ks_engine"][rows].astype(np.float64)
    sides = [K2.side_data(CH[:, s], names, "plain") for s in (0, 1)]
    fz = np.load(os.path.join(DATA, "fitC_features.npz"))
    XB = fz["diff"][rows, KSB0:KSB1].astype(np.float64) * (fz["phase"][rows].astype(np.float64) / 256.0)[:, None]
    th_ship = np.array([SHIP[n] for n in K2.KS_NAMES], dtype=np.float64)
    ks_ship = K2.ks_eval(th_ship, sides, PHK, False)
    ok = PHK >= 0
    print("KS DEPTH FIT  rows %d (val %d)" % (len(sf), val.sum()))
    print("  CLOSURE: python KS(ship) vs engine ks_engine — max |diff| %.1f mp · mean |diff| %.2f · corr %.5f"
          % (np.abs(ks_ship - KSE)[ok].max(), np.abs(ks_ship - KSE)[ok].mean(), np.corrcoef(ks_ship[ok], KSE[ok])[0, 1]))
    tgt = wp(sf)
    tr = ~val
    free = np.array([n not in K2.PINNED_ALWAYS for n in K2.KS_NAMES])
    scale = np.array([K2.SCALE[n] for n in K2.KS_NAMES], dtype=np.float64)
    bounds_ks = [((K2.BOUNDS[n][0] - SHIP[n]) / K2.SCALE[n] if n in K2.BOUNDS and K2.BOUNDS[n][0] is not None else None,
                  (K2.BOUNDS[n][1] - SHIP[n]) / K2.SCALE[n] if n in K2.BOUNDS and K2.BOUNDS[n][1] is not None else None)
                 if f_ else (0.0, 0.0) for n, f_ in zip(K2.KS_NAMES, free)]
    nks, nb = len(th_ship), XB.shape[1]
    lam_ks, lam_b = float(KV.get("KS_LAMBDA", 1e-3)), float(KV.get("KSB_LAMBDA", 1e-2))

    def model(d, wb, s):
        th = th_ship + scale * d
        ks = K2.ks_eval(th, sides, PHK, False)
        return base - (ks - ks_ship) / 10.0 - (XB @ wb) / 10.0 + s * stm

    def loss(d, wb, s, m):
        return float(np.mean((wp(model(d, wb, s)[m]) - tgt[m]) ** 2))

    def fit(use_ks, use_b):
        def f(p):
            d = p[:nks] if use_ks else np.zeros(nks)
            wb = p[nks:nks + nb] if use_b else np.zeros(nb)
            s = p[-1]
            th = th_ship + scale * d
            ks, J = K2.ks_eval(th, sides, PHK, False, grad=True)
            c = base - (ks - ks_ship) / 10.0 - (XB @ wb) / 10.0 + s * stm
            q = wp(c[tr]); r = q - tgt[tr]
            g = 2.0 * r * q * (1 - q / 100.0) * KW * (np.abs(c[tr]) < 1500)
            gd = -(J[tr] * g[:, None]).mean(0) / 10.0 * scale if use_ks else np.zeros(nks)
            gb = -(XB[tr] * g[:, None]).mean(0) / 10.0 if use_b else np.zeros(nb)
            loss_ = float(np.mean(r * r)) + lam_ks * float(d @ d) + lam_b * float(wb @ wb) / 1e4
            return loss_, np.r_[gd + 2 * lam_ks * d, gb + 2 * lam_b * wb / 1e4, float((g * stm[tr]).mean())]
        bnds = (bounds_ks if use_ks else [(0.0, 0.0)] * nks) + ([(None, None)] * nb if use_b else [(0.0, 0.0)] * nb) \
            + [(None, None)]
        res = minimize(f, np.zeros(nks + nb + 1), jac=True, method="L-BFGS-B", bounds=bnds, options={"maxiter": 3000})
        p = res.x
        return p[:nks], p[nks:nks + nb], p[-1]

    d0, b0, s0 = fit(False, False)
    vb = loss(d0, b0, s0, val)
    res = {}
    for arm, uk, ub in (("KS", True, False), ("KSB", False, True), ("KS+KSB", True, True)):
        d, wb, s = fit(uk, ub)
        res[arm] = (d, wb, s, loss(d, wb, s, val))
        print("  %-7s val %+.2f%% vs base" % (arm, 100 * (res[arm][3] / vb - 1)))
    j = res["KS+KSB"]
    print("  joint vs KSB alone: %+.2f%% (does re-pricing the attack model add on top of shelter/storm?)"
          % (100 * (j[3] / res["KSB"][3] - 1)))
    th = th_ship + scale * j[0]
    print("  joint KS knobs (ship → fit): " + " ".join("%s %g→%.0f" % (n, SHIP[n], v) for n, v in zip(K2.KS_NAMES, th)
                                                      if abs(v - SHIP[n]) >= 0.5))
    tag = KV.get("OUT", "ks_depth")
    with open(os.path.join(DATA, tag + "_ksb.txt"), "w") as f:
        f.write("# KS-B cells, joint KS+KSB depth fit (_ks_depth_fit.py)\n")
        for k in range(nb):
            if round(j[1][k]) != 0:
                f.write("%d 0 0 %.0f\n" % (KSB0 + k, j[1][k]))
    with open(os.path.join(DATA, tag + "_ks.txt"), "w") as f:
        f.write(" ".join("KS_V2_%s=%d" % (n, round(v)) for n, v in zip(K2.KS_NAMES, th)) + "\n")
    print("  wrote %s_ksb.txt and %s_ks.txt" % (tag, tag))


if __name__ == "__main__" and KV.get("MODE") not in ("dump", "closure"):
    main()


def dump():
    """MODE=dump OUT=<csv> [LIMIT=3000]: the engine's published king_safety (KS-A total, Black-positive mp) per labelled
    std row under the CURRENT env (knobs latch at init ⇒ one process per knob set)."""
    import chess
    os.environ.setdefault("PRESET", "LONG_FORMAT"); os.environ.setdefault("USE_OPENING_BOOK", "0")
    sys.path.insert(0, os.path.dirname(THIS))
    import ChessAI
    ai = ChessAI.ChessAI(None, None, chess.Board(), True)
    sample = {r["fen"]: int(r["row"]) for r in csv.DictReader(open(os.path.join(THIS, "ks_sets/fitC_mg_sample.csv")))
              if r["src"] == "std"}
    lim, n = int(KV.get("LIMIT", 3000)), 0
    with open(KV["OUT"], "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["row", "ks"])
        for fen, row in sample.items():
            if n >= lim:
                break
            w.writerow([row, int(ai.ev_breakdown(chess.Board(fen)).get("king_safety", 0))]); n += 1


def closure():
    """MODE=closure SHIP=<csv> NEW=<csv> KS=<knob file>: engine Δking_safety (new − ship) vs the Python model's Δ on the
    same rows. Real vs integer arithmetic ⇒ a few mp of rounding is expected; exit 1 if max |diff| > 10 mp or nothing moves."""
    a = {int(r["row"]): int(r["ks"]) for r in csv.DictReader(open(KV["SHIP"]))}
    b = {int(r["row"]): int(r["ks"]) for r in csv.DictReader(open(KV["NEW"]))}
    rows = np.array(sorted(set(a) & set(b)))
    kz = np.load(os.path.join(DATA, "fitC_ks2.npz"))
    names = [str(x) for x in kz["names"]]
    sides = [K2.side_data(kz["ch"][rows][:, s], names, "plain") for s in (0, 1)]
    ph = kz["phase"][rows].astype(np.float64)
    new = dict(SHIP)
    for kv in open(KV["KS"]).read().split():
        k, v = kv.split("="); new[k.replace("KS_V2_", "")] = float(v)
    th0 = np.array([SHIP[n] for n in K2.KS_NAMES], float)
    th1 = np.array([new[n] for n in K2.KS_NAMES], float)
    dm = K2.ks_eval(th1, sides, ph, False) - K2.ks_eval(th0, sides, ph, False)
    de = np.array([b[r] - a[r] for r in rows], float)
    diff = np.abs(dm - de)
    print("KS CLOSURE  rows %d · engine Δ non-zero %d · max |model−engine| %.1f mp · mean %.2f · corr %.5f  VERDICT %s"
          % (len(rows), int((de != 0).sum()), diff.max(), diff.mean(), np.corrcoef(dm, de)[0, 1],
             "OK" if diff.max() <= 10 and (de != 0).sum() > 0 else "☠️ DIVERGES"))
    sys.exit(0 if diff.max() <= 10 and (de != 0).sum() > 0 else 1)


if KV.get("MODE") == "dump":
    dump()
elif KV.get("MODE") == "closure":
    closure()
