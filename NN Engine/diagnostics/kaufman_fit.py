# -*- coding: utf-8 -*-
"""Fit OUR OWN Kaufman quadratic material-imbalance coefficients from data (public Kaufman MODEL, our values).
The imbalance scalar is LINEAR in its coefficients, so this is a ridge least-squares fit:
  target  = SF11_static_eval - our_eval(pairs OFF)      (White-POV, milli-pawns; Kaufman will own the pairs)
  feature = piece-count PRODUCTS (White-minus-Black difference form)
  beta    = the material-census-correlated part of our systematic eval gap
Train/held-out split (fixed seed) proves generalization. Emits C++ constexpr tables (milli-pawn units) +
held-out mean|gap| before/after. No SF constants used.

Run:  pyrun diagnostics/kaufman_fit.py [N] [lambda]     (N default 800, lambda default 50)
"""
import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ.setdefault('KS_SAFE_CHECK_DEF', '5')
# Fit target = our eval with the flat pair bonuses OFF (Kaufman will replace them). Set BEFORE ChessAI import.
os.environ['BISHOP_PAIR_BONUS'] = '0'
os.environ['KNIGHT_PAIR_BONUS'] = '0'
for _a in sys.argv[1:]:
    if '=' in _a: _k, _v = _a.split('=', 1); os.environ[_k] = _v
THIS = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(THIS)); sys.path.insert(0, THIS)
import numpy as np, chess
from sts_test import load_sts_epd
from eval_vs_sf11 import SF11Eval, SF11

nums = [int(a) for a in sys.argv[1:] if a.isdigit()]
N = nums[0] if nums else 800
LAM = float(nums[1]) if len(nums) > 1 else 50.0

# Extended piece-type index: 0=BISHOP_PAIR, 1=P, 2=N, 3=B, 4=R, 5=Q (SF convention). Lower-triangular pt2<=pt1.
PT = {0: None, 1: chess.PAWN, 2: chess.KNIGHT, 3: chess.BISHOP, 4: chess.ROOK, 5: chess.QUEEN}
def counts(board, color):
    c = [0] * 6
    c[0] = 1 if len(board.pieces(chess.BISHOP, color)) >= 2 else 0
    for i in range(1, 6):
        c[i] = len(board.pieces(PT[i], color))
    return c

# Feature layout: for each (pt1, pt2) with pt2<=pt1 -> an "ours" coeff; and a "theirs" coeff EXCEPT the
# theirs-diagonal (identically zero: c_w*c_b - c_b*c_w = 0). Build index maps.
FEATS = []  # (kind, pt1, pt2)
for pt1 in range(6):
    for pt2 in range(pt1 + 1):
        FEATS.append(('O', pt1, pt2))
for pt1 in range(6):
    for pt2 in range(pt1 + 1):
        if pt1 == pt2: continue
        FEATS.append(('T', pt1, pt2))

def feat_row(board):
    cw, cb = counts(board, chess.WHITE), counts(board, chess.BLACK)
    row = []
    for kind, pt1, pt2 in FEATS:
        if kind == 'O':
            row.append(cw[pt1] * cw[pt2] - cb[pt1] * cb[pt2])
        else:
            row.append(cw[pt1] * cb[pt2] - cb[pt1] * cw[pt2])
    return row

allpos = load_sts_epd(STS := os.path.join(THIS, "suites", "STS1-STS15_LAN_v3.epd"))
import random
random.Random(1234).shuffle(allpos)
allpos = allpos[:N]

from ChessAI import ChessAI
ai = ChessAI(None, None, chess.Board(), True)
sf = SF11Eval(SF11)

X, y = [], []
kept = 0
for item in allpos:
    fen = item[0] if isinstance(item, (tuple, list)) else item
    b = chess.Board(fen)
    if b.is_check(): continue
    try:
        bd = ai.ev_breakdown(b)
        our = -bd.get("total", 0.0) / 1000.0          # White-POV pawns, pairs OFF
        sf_total, _ = sf.eval(fen)                      # White-POV pawns
    except Exception:
        continue
    if sf_total is None: continue
    X.append(feat_row(b))
    y.append((sf_total - our) * 1000.0)                 # residual in MILLI-pawns (White-POV)
    kept += 1
sf.close()

X = np.array(X, dtype=float); y = np.array(y, dtype=float)
# Drop all-zero feature columns (unidentifiable in this corpus).
nz = np.where(np.abs(X).sum(axis=0) > 0)[0]
Xn = X[:, nz]
# Train / held-out split.
rng = np.random.RandomState(7); idx = rng.permutation(len(y)); half = len(y) // 2
tr, ho = idx[:half], idx[half:]
Xtr, ytr, Xho, yho = Xn[tr], y[tr], Xn[ho], y[ho]
# Ridge closed form.
A = Xtr.T @ Xtr + LAM * np.eye(Xtr.shape[1])
beta_nz = np.linalg.solve(A, Xtr.T @ ytr)
beta = np.zeros(len(FEATS)); beta[nz] = beta_nz

def mean_abs_gap(Xs, ys, use_beta):
    pred = Xs @ (beta_nz if use_beta else np.zeros_like(beta_nz))
    return np.mean(np.abs(ys - pred)) / 1000.0     # back to pawns

print("kept=%d  features=%d (nz=%d)  lambda=%.0f" % (kept, len(FEATS), len(nz), LAM))
print("HELD-OUT mean|gap| (pairs-off baseline): %.3f  ->  with Kaufman fit: %.3f  (train: %.3f -> %.3f)" % (
    mean_abs_gap(Xho, yho, False), mean_abs_gap(Xho, yho, True),
    mean_abs_gap(Xtr, ytr, False), mean_abs_gap(Xtr, ytr, True)))
# Reduction ON THE IMBALANCE SUBSET (held-out positions where Kaufman actually fires, |pred| > 0.15 pawns):
pred_ho = Xho @ beta_nz
mask = np.abs(pred_ho) > 150.0
if mask.sum() > 0:
    base_sub = np.mean(np.abs(yho[mask])) / 1000.0
    fit_sub = np.mean(np.abs(yho[mask] - pred_ho[mask])) / 1000.0
    print("IMBALANCE SUBSET (held-out, |Kaufman pred|>0.15p): n=%d/%d  mean|gap| %.3f -> %.3f  (%.0f%% of held-out fire)" % (
        mask.sum(), len(yho), base_sub, fit_sub, 100.0 * mask.sum() / len(yho)))

# Emit the two 6x6 integer tables (milli-pawn units), lower-triangular.
O = np.zeros((6, 6), dtype=int); T = np.zeros((6, 6), dtype=int)
for w, (kind, pt1, pt2) in zip(beta, FEATS):
    (O if kind == 'O' else T)[pt1][pt2] = int(round(w))
names = ["PAIR", "P", "N", "B", "R", "Q"]
print("\n// KAUFMAN_OURS[pt1][pt2] (milli-pawn), idx 0=pair,1=P,2=N,3=B,4=R,5=Q")
for r in range(6): print("{ " + ", ".join("%5d" % O[r][c] for c in range(6)) + " },  // " + names[r])
print("// KAUFMAN_THEIRS[pt1][pt2]")
for r in range(6): print("{ " + ", ".join("%5d" % T[r][c] for c in range(6)) + " },  // " + names[r])
print("\nSANITY (expect: pair[0][0]>0 bishop-pair; N-P ours[2][1]>0 knight-loves-pawns; R-R ours[4][4]<0 rook redundancy):")
print("  pair[0][0]=%d  knight_x_pawn ours[2][1]=%d  rook_x_rook ours[4][4]=%d  knight_x_knight ours[2][2]=%d" % (
    O[0][0], O[2][1], O[4][4], O[2][2]))
