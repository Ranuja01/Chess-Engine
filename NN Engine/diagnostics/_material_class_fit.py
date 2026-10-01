# -*- coding: utf-8 -*-
"""NARROW MATERIAL-CLASS terms, fitted on the DEPTH target (C3 doc §18f; owner 10-01 "build and find out").

Kaufman's census cells lost at 250k (−30 / −20) although the misjudgements persist at depth; they fired on 52-85% of
positions. Lesson: a term must be the SAME SHAPE as the error. Here each term fires ONLY inside its material class:
  QUEEN   A has a queen, the other side none:  adj_A = q0 + qR·dR + qM·dM + qP·dP
          (d = the OTHER side's surplus of rooks / minors / pawns over A — the compensation for the queen)
  R2M     A has +1 rook, −2 minors (queens equal):            adj_A = r2m
  MINOR   A has +1 minor, ≥ 2 fewer pawns (Q, R equal):        adj_A = mp0 + mpP·(pawn deficit − 2)
  PAIR    A has the bishop pair, the other side not (minors, Q, R equal):  adj_A = pair
Target: SF18 d14 label; base: OUR d10 SEARCH score (the depth residual — the part search cannot fix). Model (White cp):
  ours' = ours_d10 + mgw · Σ adj (White-oriented) + STM nuisance (not shipped), mgw = phase/256 (fades into the endgame,
  where the winnability scale factor owns the score). Rows: std + VARIANT labelled mg rows with a d10 score (the depth
  target needs no static total, so the 4,944 variant rows join). Val by FEN hash (15%).

  pyrun diagnostics/_material_class_fit.py [LAMBDA=1e-4]
"""
import os, sys, csv, glob, math, hashlib
import numpy as np
import chess
from scipy.optimize import minimize

KV = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
THIS = os.path.dirname(os.path.abspath(__file__))
K = 0.00368208
wp = lambda cp: 100.0 / (1.0 + np.exp(-K * np.clip(cp, -1500, 1500)))
NAMES = ["q0", "qR", "qM", "qP", "r2m", "mp0", "mpP", "pair"]
CLASS_OF = {"q0": "QUEEN", "qR": "QUEEN", "qM": "QUEEN", "qP": "QUEEN", "r2m": "R2M", "mp0": "MINOR", "mpP": "MINOR",
            "pair": "PAIR"}


def phase256(b):
    """v2's phase (256 = full middlegame): non-pawn material share, SF-style N/B 1, R 2, Q 4 of 24."""
    w = sum(len(b.pieces(p, c)) * v for p, v in ((chess.KNIGHT, 1), (chess.BISHOP, 1), (chess.ROOK, 2), (chess.QUEEN, 4))
            for c in (chess.WHITE, chess.BLACK))
    return min(256, w * 256 // 24)


def class_feats(b):
    """White-oriented feature vector (len 8) and the set of classes present."""
    x = np.zeros(len(NAMES))
    present = set()
    cnt = {c: {p: len(b.pieces(p, c)) for p in (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN)}
           for c in (chess.WHITE, chess.BLACK)}
    for A in (chess.WHITE, chess.BLACK):
        s = 1.0 if A == chess.WHITE else -1.0
        a, o = cnt[A], cnt[not A]
        dR, dM, dP = o[chess.ROOK] - a[chess.ROOK], (o[chess.KNIGHT] + o[chess.BISHOP]) - (a[chess.KNIGHT] + a[chess.BISHOP]), \
            o[chess.PAWN] - a[chess.PAWN]
        if a[chess.QUEEN] > 0 and o[chess.QUEEN] == 0:
            x[0] += s; x[1] += s * dR; x[2] += s * dM; x[3] += s * dP
            present.add("QUEEN")
        mn = (a[chess.KNIGHT] + a[chess.BISHOP]) - (o[chess.KNIGHT] + o[chess.BISHOP])
        if a[chess.QUEEN] == o[chess.QUEEN] and a[chess.ROOK] - o[chess.ROOK] == 1 and mn == -2:
            x[4] += s; present.add("R2M")
        if a[chess.QUEEN] == o[chess.QUEEN] and a[chess.ROOK] == o[chess.ROOK] and mn == 1 and dP >= 2:
            x[5] += s; x[6] += s * (dP - 2); present.add("MINOR")
        if mn == 0 and a[chess.QUEEN] == o[chess.QUEEN] and a[chess.ROOK] == o[chess.ROOK] \
                and a[chess.BISHOP] >= 2 and o[chess.BISHOP] < 2:
            x[7] += s; present.add("PAIR")
    return x, present


def main():
    ours_d = {}
    for p in glob.glob(os.path.join(THIS, "ks_sets", "fitC_mg_ours_d10_s*of4.csv")):
        for r in csv.DictReader(open(p, newline="")):
            ours_d[r["fen"]] = float(r["ours_cp_white"])
    X, base, sf, mgw, stm, val, cls = [], [], [], [], [], [], []
    for r in csv.DictReader(open(os.path.join(THIS, "ks_sets/fitC_mg_sf18.csv"), newline="")):
        f = r["fen"]
        if not r.get("best_cp") or f not in ours_d or abs(float(r["best_cp"])) >= 50000:
            continue
        b = chess.Board(f)
        x, present = class_feats(b)
        X.append(x); base.append(ours_d[f]); sf.append(float(r["best_cp"])); mgw.append(phase256(b) / 256.0)
        stm.append(1.0 if b.turn == chess.WHITE else -1.0)
        val.append(int(hashlib.md5(f.encode()).hexdigest()[:8], 16) % 100 < 15)
        cls.append(present)
    X, base, sf, mgw, stm, val = map(np.array, (X, base, sf, mgw, stm, val))
    tgt = wp(sf)
    tr = ~val
    lam = float(KV.get("LAMBDA", 1e-4))
    print("MATERIAL-CLASS FIT on the DEPTH target  rows %d (train %d, val %d)" % (len(sf), tr.sum(), val.sum()))
    for c in ("QUEEN", "R2M", "MINOR", "PAIR"):
        print("  class %-6s rows %5d" % (c, sum(c in s for s in cls)))

    def cp(p):
        return base + mgw * (X @ p[:-1]) + p[-1] * stm

    def loss(p, m):
        return float(np.mean((wp(cp(p)[m]) - tgt[m]) ** 2))

    p0 = np.zeros(len(NAMES) + 1)
    s0 = minimize(lambda q: loss(np.r_[np.zeros(len(NAMES)), q], tr), [0.0]).x[0]       # STM-only baseline
    pb = np.r_[np.zeros(len(NAMES)), s0]
    res = minimize(lambda p: loss(p, tr) + lam * float(p[:-1] @ p[:-1]), np.r_[np.zeros(len(NAMES)), s0],
                   method="Powell", options={"maxiter": 40000, "xtol": 1e-2, "ftol": 1e-10})
    p = res.x
    print("\n  val MSE (win%%², vs our d10 search + STM): base %.3f → fitted %.3f (%+.2f%%)"
          % (loss(pb, val), loss(p, val), 100 * (loss(p, val) / loss(pb, val) - 1)))
    for c in ("QUEEN", "R2M", "MINOR", "PAIR"):
        m = val & np.array([c in s for s in cls])
        if m.sum() >= 20:
            print("    class %-6s val rows %4d: %+.2f%%" % (c, m.sum(), 100 * (loss(p, m) / loss(pb, m) - 1)))
    print("  fitted (cp, A-oriented; × mg weight): " + " ".join("%s=%+.0f" % (n, v) for n, v in zip(NAMES, p[:-1]))
          + "   (STM nuisance %.1f cp)" % p[-1])
    out = os.path.join("/mnt/e/chess_data/texel", "mcl_fit.txt")
    with open(out, "w") as f:
        f.write("# material-class terms (cp, A-oriented, x phase/256) fitted on the depth target, _material_class_fit.py\n")
        for n, v in zip(NAMES, p[:-1]):
            f.write("%s %.1f\n" % (n, v))
    print("  wrote", out)


if __name__ == "__main__" and KV.get("MODE") not in ("dump", "closure"):
    main()


def dump():
    """MODE=dump OUT=<csv> [LIMIT=6000]: per labelled FEN, the engine's published total and v2_phase256 under the CURRENT
    env (knobs latch at init ⇒ one process per setting: run once with MCL_V2=1 + values, once with MCL_V2=0)."""
    os.environ.setdefault("PRESET", "LONG_FORMAT"); os.environ.setdefault("USE_OPENING_BOOK", "0")
    sys.path.insert(0, os.path.dirname(THIS))
    import ChessAI
    ai = ChessAI.ChessAI(None, None, chess.Board(), True)
    lim = int(KV.get("LIMIT", 6000))
    with open(KV["OUT"], "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["fen", "total", "phase"])
        for i, r in enumerate(csv.DictReader(open(os.path.join(THIS, "ks_sets/fitC_mg_sf18.csv"), newline=""))):
            if i >= lim:
                break
            bd = ai.ev_breakdown(chess.Board(r["fen"]))
            w.writerow([r["fen"], int(bd["total"]), int(bd.get("v2_phase256", -1))])


def closure():
    """MODE=closure ON=<csv> OFF=<csv> + the MCL_V2_* values in env: on − off must equal −trunc(S·phase/256) exactly,
    S = the White-oriented class sum (integer mp). Exit 1 on any mismatch or if nothing fires."""
    vals = {k: int(os.environ.get("MCL_V2_" + k, "0")) for k in ("Q0", "QR", "QM", "QP", "R2M", "MP0", "MPP", "PAIR")}
    wv = np.array([vals[k] for k in ("Q0", "QR", "QM", "QP", "R2M", "MP0", "MPP", "PAIR")], float)
    off = {r["fen"]: int(r["total"]) for r in csv.DictReader(open(KV["OFF"], newline=""))}
    bad = live = n = 0
    for r in csv.DictReader(open(KV["ON"], newline="")):
        x, _ = class_feats(chess.Board(r["fen"]))
        S = int(round(float(x @ wv)))
        ph = int(r["phase"])
        q = abs(S * ph) // 256
        model = -(q if S * ph >= 0 else -q)
        d = int(r["total"]) - off[r["fen"]]
        n += 1; live += d != 0; bad += d != model
    print("MCL CLOSURE  rows %d · live %d · mismatches %d  VERDICT %s" % (n, live, bad, "EXACT" if bad == 0 else "☠️ DIVERGES"))
    sys.exit(0 if bad == 0 and live > 0 else 1)


if KV.get("MODE") == "dump":
    dump()
elif KV.get("MODE") == "closure":
    closure()
