# -*- coding: utf-8 -*-
"""KAUFMAN (census material imbalance), TEXEL-FITTED on SF18 SEARCH labels — never tried before (record check 10-01).

Why: the queen lead (C3 doc §18b; memory `v2-overvalues-queen-vs-minor-compensation`): with one side queenless, SF18
rates the queen side −5.3pp vs our static eval (−6.1 vs minors). v2's `kaufman_mp` (SF11's quadratic census, built at
KAUF_V2_MAG=0) was closed 09-18 only as SF's fixed tables × one global scalar on §I corpus MSE — the cells were never
fitted. The owner (10-01): Kaufman was never Texel-tuned, so that may be where it shines. The 09-18 header's "fitting
the cells is NOT the plan" was a rule about d6-OUTCOME corpus fits; this fits SF18 labels (depth-independent).

Model (White-POV cp): ours = (−T + Σ w·f)/10 + s·STM, T = shipped static total (fitC_win.npz, Black-positive mp);
f = the census features in kaufman_mp's own order (SF slots 0=pair 1=P 2=N 3=B 4=R 5=Q, cells pt2 ≤ pt1):
  OURS  cells: cw1·cw2 − cb1·cb2        THEIRS cells (pt2 < pt1): cw1·cb2 − cb1·cw2
w in mp per unit (White-POV); STM = a side-to-move nuisance term (absorbs the +2.4pp static-vs-search bias; NOT shipped).
Loss: mean (win%(ours) − win%(SF18))², L2 on w. Rows: the SF18-labelled std middlegame + endgame samples joined to
fitC_win.npz; val by game hash (15%).
ARMS: FULL (all cells incl. the bishop pair — the register says Kaufman should OWN the pair) · QUEEN (only cells with a
queen index) · SF prior (KAUF_OURS/THEIRS × 1000/2048, unfitted, for reference).

  pyrun diagnostics/_texel_kauf_fit.py [LAMBDA=1e-3,1e-2,1e-1] [TAG=fitKauf]
"""
import os, sys, csv, hashlib, math
import numpy as np
import chess
from scipy.optimize import minimize

KV = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
THIS = os.path.dirname(os.path.abspath(__file__))
DATA = KV.get("DATA", "/mnt/e/chess_data/texel")
K_WIN = 0.00368208
SF_OURS = [[1438, 0, 0, 0, 0, 0], [40, 38, 0, 0, 0, 0], [32, 255, -62, 0, 0, 0], [0, 104, 4, 0, 0, 0],
           [-26, -2, 47, 105, -208, 0], [-189, 24, 117, 133, -134, -6]]
SF_THEIRS = [[0] * 6, [36, 0, 0, 0, 0, 0], [9, 63, 0, 0, 0, 0], [59, 65, 42, 0, 0, 0], [46, 39, 24, -24, 0, 0],
             [97, 100, -42, 137, 268, 0]]
SLOT = ["pair", "P", "N", "B", "R", "Q"]
CELLS = [("O", a, b) for a in range(6) for b in range(a + 1)] + [("T", a, b) for a in range(6) for b in range(a)]
# LINEAR per-piece value corrections (N, B, R, Q; the pawn stays the unit): without them the census cells absorb any
# piece-VALUE error through pawn-count products (fit 1: N x P +80, R x P +73) — value vs imbalance must be separable.
CELLS += [("L", a, 0) for a in (2, 3, 4, 5)]


def winpct(cp):
    return 100.0 / (1.0 + np.exp(-K_WIN * np.clip(cp, -1500, 1500)))


def counts(b):
    cw, cb = [0] * 6, [0] * 6
    cw[0] = int(len(b.pieces(chess.BISHOP, chess.WHITE)) >= 2)
    cb[0] = int(len(b.pieces(chess.BISHOP, chess.BLACK)) >= 2)
    for i, p in enumerate((chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN)):
        cw[i + 1], cb[i + 1] = len(b.pieces(p, chess.WHITE)), len(b.pieces(p, chess.BLACK))
    return cw, cb


def feats(b):
    cw, cb = counts(b)
    return [cw[a] * cw[c] - cb[a] * cb[c] if k == "O" else (cw[a] * cb[c] - cb[a] * cw[c] if k == "T" else cw[a] - cb[a])
            for k, a, c in CELLS]


def load():
    X, T, sf, val, stm, qi, eg = [], [], [], [], [], [], []
    z = np.load(os.path.join(DATA, "fitC_win.npz"))["total"]
    for lab, smp in (("fitC_mg_sf18.csv", "fitC_mg_sample.csv"), ("fitC_eg_sf18.csv", "fitC_eg_sample.csv")):
        sample = {}
        for r in csv.DictReader(open(os.path.join(THIS, "ks_sets", smp), newline="")):
            if r.get("src", "std") == "std":
                sample[r["fen"]] = (int(r["row"]), r["game_id"])
        for r in csv.DictReader(open(os.path.join(THIS, "ks_sets", lab), newline="")):
            if not r.get("best_cp") or r["fen"] not in sample:
                continue
            row, gid = sample[r["fen"]]
            t, s = float(z[row]), float(r["best_cp"])
            if abs(t) >= 30000 or abs(s) >= 50000:
                continue
            b = chess.Board(r["fen"])
            X.append(feats(b)); T.append(t); sf.append(s)
            val.append(int(hashlib.md5(gid.encode()).hexdigest()[:8], 16) % 100 < 15)
            stm.append(1.0 if b.turn == chess.WHITE else -1.0)
            wq, bq = len(b.pieces(chess.QUEEN, chess.WHITE)), len(b.pieces(chess.QUEEN, chess.BLACK))
            qi.append((wq > 0) != (bq > 0))
            eg.append(lab.startswith("fitC_eg"))
    return (np.array(X, float), np.array(T, float), np.array(sf, float), np.array(val), np.array(stm, float),
            np.array(qi), np.array(eg))


def main():
    X, T, sf, val, stm, qi, eg = load()
    tgt = winpct(sf)
    tr = ~val
    print("KAUFMAN TEXEL FIT  rows %d (train %d, val %d) · queen-imbalance rows %d" % (len(T), tr.sum(), val.sum(), qi.sum()))

    # SCALE=1: a free GLOBAL SCALE nuisance α (T → T·(1+α)), fitted in the baseline AND every arm and never shipped —
    # SF18 search scores run larger than static evals, so without it an arm can "win" by stretching evals (fit 2:
    # the cell arms stretched ×1.06, the rejected WSF HI>64 pattern). With it, only re-shaping can win.
    SCALE = KV.get("SCALE", "0") == "1"

    def cp(w, s):
        return (-T * (1.0 + s[1]) + X @ w) / 10.0 + s[0] * stm

    def mse(w, s, m):
        return float(np.mean((winpct(cp(w, s)[m]) - tgt[m]) ** 2))

    def fit(mask, lam):
        """Fit the cells in `mask` (+ STM, + α if SCALE); scaled so one unit ≈ 10 mp."""
        idx = np.where(mask)[0]
        nk = len(idx)
        def fg(p):
            w = np.zeros(X.shape[1]); w[idx] = p[:nk] * 10.0
            a = p[nk + 1] if SCALE else 0.0
            c = cp(w, (p[nk], a))[tr]
            q = winpct(c)
            r = q - tgt[tr]
            dq = q * (1 - q / 100.0) * K_WIN * (np.abs(c) < 1500)
            g = 2.0 * r * dq
            gw = (X[tr][:, idx] * g[:, None]).mean(0)
            gs = float((g * stm[tr]).mean())
            loss = float(np.mean(r * r)) + lam * float(p[:nk] @ p[:nk])
            grad = [gw + 2 * lam * p[:nk], [gs]]
            if SCALE:
                grad.append([float((g * -T[tr] / 10.0).mean())])
            return loss, np.concatenate(grad)
        res = minimize(fg, np.zeros(nk + 1 + int(SCALE)), jac=True, method="L-BFGS-B", options={"maxiter": 3000})
        w = np.zeros(X.shape[1]); w[idx] = res.x[:nk] * 10.0
        return w, (res.x[nk], res.x[nk + 1] if SCALE else 0.0)

    # baseline: shipped (no Kaufman) with only the STM nuisance fitted — the fair comparison for every arm
    w0, s0 = fit(np.zeros(X.shape[1], bool), 0.0)
    SUBS = (("train", tr), ("val", val), ("val_mg", val & ~eg), ("val_eg", val & eg), ("val_qi", val & qi))
    base = {k: mse(w0, s0, m) for k, m in SUBS}
    print("  shipped + nuisances (STM %.1f cp, scale α %+.3f)   train %.3f  val %.3f  val queen-imbalance %.3f"
          % (s0[0], s0[1], base["train"], base["val"], base["val_qi"]))
    sfw = np.array([0.0 if k == "L" else (SF_OURS if k == "O" else SF_THEIRS)[a][c] * 1000.0 / 2048.0 for k, a, c in CELLS])
    cp0 = np.abs(cp(w0, s0))
    rep = lambda w, s: " ".join("%s %+6.2f%%" % (k, 100 * (mse(w, s, m) / base[k] - 1)) for k, m in SUBS)         + "  stretch %.3f" % (np.median(np.abs(cp(w, s))[cp0 > 50] / cp0[cp0 > 50]))
    print("  SF tables, unfitted (MAG 1000):        " + rep(sfw, s0))
    lin = np.array([k == "L" for k, _, _ in CELLS])
    queen = np.array([(a == 5 or c == 5) and k != "L" for k, a, c in CELLS])
    cells = ~lin
    out = []
    for lam in [float(x) for x in KV.get("LAMBDA", "1e-3,1e-2,1e-1").split(",")]:
        for name, mask in (("CELLS", cells), ("QUEEN", queen), ("LIN", lin), ("LIN+Q", lin | queen),
                           ("LIN+CELLS", np.ones(len(CELLS), bool))):
            w, s = fit(mask, lam)
            print("  %-9s λ=%-6g %s%s" % (name, lam, rep(w, s), ("   values N/B/R/Q %+.0f %+.0f %+.0f %+.0f mp" % tuple(w[lin])) if lin[mask].any() else ""))
            out.append((name, lam, mse(w, s, val), w))
    best = min(out, key=lambda o: o[2])
    print("\n  best on val: %s λ=%g — cells (mp per unit, White-POV; |w| ≥ 5 shown):" % (best[0], best[1]))
    for (k, a, c), v in zip(CELLS, best[3]):
        if abs(v) >= 5:
            if k == "L":
                print("    VALUE  %-4s          %+7.1f mp per piece" % (SLOT[a], v))
            else:
                print("    %s %-4s x %-4s %+7.1f" % ("OURS  " if k == "O" else "THEIRS", SLOT[a], SLOT[c], v))
    np.savez(os.path.join(DATA, "kauf_%s.npz" % KV.get("TAG", "fitKauf")),
             cells=np.array(["%s_%d_%d" % c for c in CELLS]), **{"%s_%g" % (n, l): w for n, l, _, w in out})


if __name__ == "__main__" and KV.get("MODE") != "export":
    main()


def export():
    """MODE=export NPZ=kauf_fitKaufS.npz ARM=CELLS_0.01 OUT=kauf_full.txt — write one arm's cells in the engine's
    KAUF_V2_FILE format (`O a b mp` / `T a b mp`, White-POV mp per unit; KAUF_V2_FORM=3, MAG 1000 = as fitted)."""
    z = np.load(os.path.join(DATA, KV["NPZ"]))
    w = z[KV["ARM"]]
    out = os.path.join(DATA, KV["OUT"])
    with open(out, "w") as f:
        f.write("# Kaufman cells, Texel-fitted on SF18 labels (_texel_kauf_fit.py %s, arm %s)\n" % (KV["NPZ"], KV["ARM"]))
        for c, v in zip(z["cells"], w):
            k, a, b = c.split("_")
            if k in ("O", "T") and round(v) != 0:
                f.write("%s %s %s %.0f\n" % (k, a, b, v))
    print("wrote", out)


if KV.get("MODE") == "export":
    export()
