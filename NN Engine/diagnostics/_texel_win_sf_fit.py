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

MODE=scale (2026-09-30): the REFERENCE form instead — a multiplicative endgame SCALE FACTOR (SF11/15, Ethereal, Weiss
all use one; the additive form above is discontinuous at T=0, C3 doc §12 end). T' = T·(1 + g·(f − 1)),
f = clip(64 + BASE + Σ w·x, LO, HI)/64, x LEADER-relative (the leader = sign(T); continuous at 0 because T·f → 0):
  SP strong-side pawns (4/4) · ONEFLANK all pawns on one flank (4/4) · OCB_PURE / OCB_MIX opposite bishops, alone /
  with other pieces (4/4) · PAWN_END (3/4) · PASSED strong-side passers · OUTFLANK · INFILT (SF only).
Features come from the FEN (python-chess); the engine side is ported only if this passes.
  pyrun diagnostics/_texel_win_sf_fit.py MODE=scale [HI=64,72,80] [LO=0] [LAMBDA=1e-6]
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


SC_NAMES = ["BASE", "SP", "ONEFLANK", "OCB_PURE", "OCB_MIX", "PAWN_END", "PASSED", "OUTFLANK", "INFILT"]
QS, KS = 0x0F0F0F0F0F0F0F0F, 0xF0F0F0F0F0F0F0F0


def scale_features(fen, strong_black):
    import chess
    b = chess.Board(fen)
    s = chess.BLACK if strong_black else chess.WHITE
    pw = int(b.pawns)
    sp = len(b.pieces(chess.PAWN, s))
    oneflank = 0 if (pw & QS and pw & KS) else 1
    wb, bb = b.pieces(chess.BISHOP, chess.WHITE), b.pieces(chess.BISHOP, chess.BLACK)
    ocb = len(wb) == 1 and len(bb) == 1 and \
        (chess.square_file(list(wb)[0]) + chess.square_rank(list(wb)[0])) % 2 != \
        (chess.square_file(list(bb)[0]) + chess.square_rank(list(bb)[0])) % 2
    others = int(b.knights | b.rooks | b.queens) != 0
    npm = int(b.knights | b.bishops | b.rooks | b.queens) != 0
    # strong-side passers: no enemy pawn ahead on the same or adjacent files
    passed = 0
    for sq in b.pieces(chess.PAWN, s):
        f, r = chess.square_file(sq), chess.square_rank(sq)
        blocked = False
        for e in b.pieces(chess.PAWN, not s):
            ef, er = chess.square_file(e), chess.square_rank(e)
            if abs(ef - f) <= 1 and ((er > r) if s == chess.WHITE else (er < r)):
                blocked = True
                break
        passed += not blocked
    wk, bk = b.king(chess.WHITE), b.king(chess.BLACK)
    outflank = abs(chess.square_file(wk) - chess.square_file(bk)) - abs(chess.square_rank(wk) - chess.square_rank(bk))
    infilt = 1 if (chess.square_rank(wk) > 3 or chess.square_rank(bk) < 4) else 0
    return [1, sp, oneflank, int(ocb and not others), int(ocb and others), int(not npm), passed, outflank, infilt]


def main_scale():
    his = [float(h) for h in KV.get("HI", "64,72,80").split(",")]
    LO = float(KV.get("LO", 0))
    lam = float(KV.get("LAMBDA", 1e-6))
    sample = {r["fen"]: (int(r["row"]), r["game_id"]) for r in csv.DictReader(open(SAMPLE, newline=""))}
    lab = [(r["fen"], float(r["best_cp"])) for r in csv.DictReader(open(LABELS, newline="")) if r.get("best_cp")]
    z = np.load(os.path.join(DATA, "fitC_win.npz"))
    fens, rows, sf, val = [], [], [], []
    for fen, cp in lab:
        if fen in sample:
            row, gid = sample[fen]
            fens.append(fen); rows.append(row); sf.append(cp)
            val.append(int(hashlib.md5(gid.encode()).hexdigest()[:8], 16) % 100 < 15)
    rows, sf, val = np.array(rows), np.array(sf), np.array(val)
    T = z["total"][rows].astype(np.float64)
    ph = z["phase"][rows].astype(np.float64)
    ok = (ph >= 0) & (np.abs(T) < 30000) & (np.abs(sf) < 50000)
    X = np.array([scale_features(f, t > 0) for f, t, o in zip(fens, T, ok) if o], dtype=np.float64)
    if "KEEP" in KV:                                    # zero out the features not kept (BASE always kept)
        keep = set(KV["KEEP"].split(",")) | {"BASE"}
        X[:, [i for i, k in enumerate(SC_NAMES) if k not in keep]] = 0.0
    T, ph, sf, val = T[ok], ph[ok], sf[ok], val[ok]
    g = (256.0 - ph) / 256.0
    tgt = winpct(sf)
    tr = ~val
    print("MODE=scale rows %d (train %d, val %d)" % (len(T), tr.sum(), val.sum()))
    print("  feature means: " + " ".join("%s=%.2f" % (k, v) for k, v in zip(SC_NAMES, X.mean(axis=0))))

    def tprime(wv, m, hi):
        f = np.clip(64.0 + X[m] @ wv, LO, hi) / 64.0
        return T[m] * (1.0 + g[m] * (f - 1.0)), f

    def mse(wv, m, hi):
        tp, _ = tprime(wv, m, hi)
        return float(np.mean((winpct(-tp / 10.0) - tgt[m]) ** 2))

    zero = np.zeros(len(SC_NAMES))
    base = {"train": mse(zero, tr, 64), "val": mse(zero, val, 64)}
    # SF15.1 'otherwise' branch as an unfitted prior: min(64, 36 + 7·sp) − 8·oneflank, OCB pure 18+4·passed
    prior = np.array([-28, 7, -8, -18, -20, 0, 0, 0, 0], dtype=np.float64)
    print("  shipped (f≡1)       train %.3f  val %.3f" % (base["train"], base["val"]))
    print("  SF15-shaped prior   train %+.2f%%  val %+.2f%%" % tuple(
        100 * (mse(prior, m, 64) / base[k] - 1) for k, m in (("train", tr), ("val", val))))
    near = np.abs(T) < 200
    out = []
    for hi in his:
        def loss(d):
            return mse(d, tr, hi) + lam * float(d @ d)
        best = None
        for x0 in (zero, prior):
            r = minimize(loss, x0, method="Powell", options={"maxiter": 20000, "xtol": 1e-2, "ftol": 1e-9})
            if best is None or r.fun < best.fun:
                best = r
        wv = best.x
        tpa, fa = tprime(wv, np.ones(len(T), bool), hi)
        ratio = np.abs(tpa) / np.maximum(np.abs(T), 1)
        vt, vv = mse(wv, tr, hi), mse(wv, val, hi)
        print("  HI=%-3.0f fitted  train %+.2f%%  val %+.2f%% | f mean %.3f  shrunk %.1f%%  grown %.1f%%  max stretch %.3f"
              "  | near-level rows (%d): max |T'-T| %.0f mp"
              % (hi, 100 * (vt / base["train"] - 1), 100 * (vv / base["val"] - 1), fa.mean(),
                 100 * (fa < 1).mean(), 100 * (fa > 1).mean(), ratio.max(), near.sum(),
                 np.abs(tpa - T)[near].max() if near.any() else 0))
        print("         weights: " + " ".join("%s=%.1f" % (k, v) for k, v in zip(SC_NAMES, wv)))
        out.append("# HI=%.0f LO=%.0f val %+.2f%%\n" % (hi, LO, 100 * (vv / base["val"] - 1))
                   + " ".join("%s=%.1f" % (k, v) for k, v in zip(SC_NAMES, wv)))
    open(os.path.join(DATA, "win_%s.txt" % KV.get("TAG", "fitWscale")), "w").write("\n".join(out) + "\n")


if __name__ == "__main__":
    main_scale() if KV.get("MODE") == "scale" else main()
