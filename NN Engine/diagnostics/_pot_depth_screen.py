# -*- coding: utf-8 -*-
"""POT DEPTH SCREEN — do POT's type gates explain what our SEARCH still misjudges?

POT = "OvD reworked" (the owner's v1 invention); types: dev_notes/POT-TYPE-DEFINITIONS-2026-09-30.md. Every earlier POT
screen used the STATIC residual, which is mostly short-horizon noise our search fixes (side-to-move +2.4 → +0.8 at depth,
C3 doc §18d). POT claims what search cannot see, so this is its first fair test:
  y = win%(SF18 d14) − win%(our d10 SEARCH), White-POV (`_depth_residual_pass.py`)
  x_k = gate_k(White) − gate_k(Black) for T1/T3/T4/T5 (`_pot_coverage.gates`: precursors present ∧ result absent),
        plus each type's KEY precursor intensity (lever reach count / majority size), net White − Black.
Raw corr, and BEYOND controls (ridge out-of-fold on the engine KS total + 68 KS channels + the 184 C1/C3 feature diffs
+ side to move) — a POT signal must survive what the eval already has (no-overlap rule).

  pyrun diagnostics/_pot_depth_screen.py
"""
import os, csv, glob, math
import numpy as np
import chess
import _pot_coverage as P

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
_k = np.load(os.path.join(DATA, "fitC_ks2.npz"))
KS, CH = _k["ks_engine"], _k["ch"]
DIFF = np.load(os.path.join(DATA, "fitC_features.npz"))["diff"]

NAMES = ["T1 gate", "T3 gate", "T4 gate", "T5 gate", "central lever reach", "majority size", "minority lever reach"]


def side_feats(b, A):
    g = P.gates(b, A)
    D = not A
    dp = P.pawns(b, D)
    maj = 0
    for fm in P.FLANKS.values():
        maj += max(0, chess.popcount(P.pawns(b, A) & fm) - chess.popcount(dp & fm))
    minr = sum(P.lever_reach(b, A, 2, dp & fm) for fm in P.FLANKS.values()
               if 1 <= chess.popcount(P.pawns(b, A) & fm) < chess.popcount(dp & fm))
    return [float("T1" in g), float("T3" in g), float("T4" in g), float("T5" in g),
            float(P.lever_reach(b, A, 2, dp & P.CENTRE)), float(maj), float(minr)]


X, y, C = [], [], []
for r in csv.DictReader(open(os.path.join(THIS, "ks_sets/fitC_mg_sf18.csv"), newline="")):
    f = r["fen"]
    if not r.get("best_cp") or f not in sample or f not in ours_d:
        continue
    b = chess.Board(f)
    if chess.popcount(int(b.occupied)) < 20:
        continue
    row = sample[f]
    X.append(np.array(side_feats(b, chess.WHITE)) - np.array(side_feats(b, chess.BLACK)))
    y.append(wp(float(r["best_cp"])) - wp(ours_d[f]))
    C.append(np.r_[-float(KS[row]), CH[row].reshape(-1).astype(float), -DIFF[row].astype(float),
                   1.0 if b.turn == chess.WHITE else -1.0])
X, y, C = np.array(X), np.array(y), np.array(C)
C = C[:, C.std(0) > 0]
C = (C - C.mean(0)) / C.std(0)
n = len(y)
fold = np.arange(n) % 5


def oof(t, lam=50.0):
    out = np.zeros(n)
    for k in range(5):
        tr, te = fold != k, fold == k
        A = C[tr]
        w = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ (t[tr] - t[tr].mean()))
        out[te] = t[te] - (t[tr].mean() + C[te] @ w)
    return out


ey = oof(y)
print("POT DEPTH SCREEN  std middlegame rows %d · controls %d cols explain %.1f%% of the depth residual"
      % (n, C.shape[1], 100 * (1 - ey.var() / y.var())))
print("  %-22s %8s %7s %10s %9s %7s" % ("feature (White−Black)", "raw r", "σ", "BEYOND r", "σ", "fires"))
for k, name in enumerate(NAMES):
    v = X[:, k]
    if v.std() == 0:
        continue
    r0 = np.corrcoef(v, y)[0, 1]
    r1 = np.corrcoef(oof(v), ey)[0, 1]
    print("  %-22s %+8.4f %7.1f %+10.4f %9.1f %6.1f%%" % (name, r0, r0 * math.sqrt(n - 3), r1, r1 * math.sqrt(n - 3),
                                                          100 * (v != 0).mean()))
print("  read: |BEYOND σ| ≥ 3 ⇒ a POT signal the eval + KS do not carry, persisting through our search.")
