# -*- coding: utf-8 -*-
"""IMBALANCE screen by MATERIAL SIGNATURE, at DEPTH (C3 doc §18d-e; owner 10-01: "build and find out").

Kaufman lesson: the broad census lost at 250k (−30 / −20) although the queen misjudgement PERSISTS at depth (−5.7pp) —
a term must be the same SHAPE as the error. So first: which material classes does our SEARCH still misjudge?
  residual toward side A = win%(SF18 d14) − win%(ours), static (shipped eval) and depth (our d10 search), A-oriented.
Classes (A = the side holding the first-named material; everything not named is equal):
  Q vs no-Q (by A's rook diff) · R vs 2 minors · exchange (R vs minor) · minor vs ≥2 pawns · bishop pair vs none
  (equal minor counts) · B vs N (A has the extra bishop, the other side the extra knight).
B vs N OPENNESS (owner: openness overlaps mobility — never double-pay): depth residual toward the bishop side vs
openness (pawnless files, −rams) and lever count, PARTIAL on the bishop + knight mobility counts (C1 cols 0-22, both
sides) and side to move. Only signal that survives mobility could justify an openness-conditioned minor value.

  pyrun diagnostics/_imbalance_depth_screen.py
"""
import os, csv, glob, math
import numpy as np
import chess

THIS = os.path.dirname(os.path.abspath(__file__))
DATA = "/mnt/e/chess_data/texel"
K = 0.00368208
wp = lambda cp: 100.0 / (1.0 + np.exp(-K * np.clip(cp, -1500, 1500)))

ours_d = {}
for p in glob.glob(os.path.join(THIS, "ks_sets", "fitC_mg_ours_d10_s*of4.csv")):
    for r in csv.DictReader(open(p, newline="")):
        ours_d[r["fen"]] = float(r["ours_cp_white"])
sample = {r["fen"]: int(r["row"]) for r in csv.DictReader(open(os.path.join(THIS, "ks_sets/fitC_mg_sample.csv")))
          if r["src"] == "std"}
T = np.load(os.path.join(DATA, "fitC_win.npz"))["total"]
DIFF = np.load(os.path.join(DATA, "fitC_features.npz"))["diff"]          # Black − White counts


def cnt(b, c):
    return {p: len(b.pieces(p, c)) for p in (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN)}


def classes(b):
    """[(class name, A colour)] for the signatures present."""
    w, k = cnt(b, chess.WHITE), cnt(b, chess.BLACK)
    out = []
    for A, a, o in ((chess.WHITE, w, k), (chess.BLACK, k, w)):
        d = {p: a[p] - o[p] for p in a}
        minors = d[chess.KNIGHT] + d[chess.BISHOP]
        if a[chess.QUEEN] > 0 and o[chess.QUEEN] == 0:
            out.append(("Q vs no-Q, A rooks %+d" % max(-2, min(1, d[chess.ROOK])), A))
            out.append(("Q vs no-Q (all)", A))
        if d[chess.QUEEN] == 0 and d[chess.ROOK] == 1 and minors == -2:
            out.append(("R vs 2 minors", A))
        if d[chess.QUEEN] == 0 and d[chess.ROOK] == 1 and minors == -1:
            out.append(("exchange (R vs minor)", A))
        if d[chess.QUEEN] == 0 and d[chess.ROOK] == 0 and minors == 1 and d[chess.PAWN] <= -2:
            out.append(("minor vs >=2 pawns", A))
        if minors == 0 and d[chess.QUEEN] == 0 and d[chess.ROOK] == 0 and a[chess.BISHOP] >= 2 and o[chess.BISHOP] < 2:
            out.append(("bishop pair vs none", A))
        if d[chess.QUEEN] == 0 and d[chess.ROOK] == 0 and d[chess.BISHOP] == 1 and d[chess.KNIGHT] == -1:
            out.append(("B vs N", A))
    return out


def openness(b):
    pw = int(b.pawns)
    open_files = sum(1 for f in range(8) if not pw & chess.BB_FILES[f])
    wp_, bp_ = int(b.pieces(chess.PAWN, chess.WHITE)), int(b.pieces(chess.PAWN, chess.BLACK))
    rams = chess.popcount((wp_ << 8) & bp_ & chess.BB_ALL)
    levers = chess.popcount(((wp_ & ~chess.BB_FILE_A) << 7 | (wp_ & ~chess.BB_FILE_H) << 9) & bp_ & chess.BB_ALL)
    return open_files - 0.5 * rams, levers


rows = []
for r in csv.DictReader(open(os.path.join(THIS, "ks_sets/fitC_mg_sf18.csv"), newline="")):
    f = r["fen"]
    if not r.get("best_cp") or f not in sample or f not in ours_d:
        continue
    b = chess.Board(f)
    sf, st, dp = float(r["best_cp"]), -float(T[sample[f]]) / 10.0, ours_d[f]
    for name, A in classes(b):
        s = 1.0 if A == chess.WHITE else -1.0
        rows.append((name, s * (wp(sf) - wp(st)), s * (wp(sf) - wp(dp)), f, s, sample[f], b.turn == A))

print("IMBALANCE DEPTH SCREEN — residual toward A (SF18 d14 − ours), pp; std mg rows with a d10 search score")
print("  %-28s %5s %16s %16s" % ("class", "n", "static", "DEPTH"))
for name in sorted({x[0] for x in rows}):
    sel = [x for x in rows if x[0] == name]
    a, d = np.array([x[1] for x in sel]), np.array([x[2] for x in sel])
    se = lambda v: v.std() / math.sqrt(len(v)) if len(v) > 1 else float("nan")
    print("  %-28s %5d %+7.2f (se %.2f) %+7.2f (se %.2f)%s" % (name, len(sel), a.mean(), se(a), d.mean(), se(d),
          "   ★ persists" if len(sel) >= 30 and abs(d.mean()) > 2.5 * se(d) else ""))

# B vs N openness, partial on mobility
bn = [x for x in rows if x[0] == "B vs N"]
if len(bn) >= 50:
    y = np.array([x[2] for x in bn])
    op = np.array([openness(chess.Board(x[3]))[0] for x in bn])
    lv = np.array([openness(chess.Board(x[3]))[1] for x in bn])
    # mobility counts, A-oriented (diff is Black − White ⇒ multiply by −s for White-A)
    mob = np.array([-x[4] * DIFF[x[5], :23].astype(float) for x in bn])
    stm = np.array([float(x[6]) for x in bn])
    Z = np.c_[mob, stm, np.ones(len(bn))]
    res = lambda v: v - Z @ np.linalg.lstsq(Z, v, rcond=None)[0]
    ry = res(y)
    print("\n  B vs N (n %d), depth residual toward the BISHOP side:" % len(bn))
    for name, v in (("openness (pawnless files − rams/2)", op), ("levers (potential to open)", lv)):
        r0 = np.corrcoef(v, y)[0, 1]
        r1 = np.corrcoef(res(v), ry)[0, 1]
        print("    %-36s raw r %+.3f (%.1fσ) · after mobility r %+.3f (%.1fσ)"
              % (name, r0, r0 * math.sqrt(len(bn) - 3), r1, r1 * math.sqrt(len(bn) - 3)))
