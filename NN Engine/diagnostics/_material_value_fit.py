# -*- coding: utf-8 -*-
"""Clean material isolation (material-ONLY, no KS). Compute PURE material from piece counts x our values (exactly
linear), and:
  (1) how far the ev_breakdown 'material' term is from pure -> quantifies what PST/AST/mobility is bundled in it.
  (2) fit (pure_material - SF11 Material) against piece-count imbalance -> per-piece coefficient = our-minus-SF
      value in pawns; residual = how much of SF's Material term is non-piece-count (phase/positional).
DIAGNOSTIC ONLY (no value changes). ~1 core (our engine + SF11).

Run: bash <runner> pyrun diagnostics/_material_value_fit.py
"""
import os, sys, signal
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
import csv, chess
import numpy as np
THIS_DIR = os.path.dirname(os.path.abspath(__file__)); ENGINE_DIR = os.path.dirname(THIS_DIR)
sys.path.insert(0, ENGINE_DIR); sys.path.insert(0, THIS_DIR)
from eval_vs_sf11 import SF11Eval, SF11
class _TO(Exception): pass
signal.signal(signal.SIGALRM, lambda s, f: (_ for _ in ()).throw(_TO()))

SETS = ['selfplay/games/vssf_2400/dp_fens.csv', 'diagnostics/_sprt_hurt.csv', 'diagnostics/_sprt_helped.csv']
fens = []
for path in SETS:
    p = os.path.join(ENGINE_DIR, path)
    if not os.path.exists(p): continue
    for r in csv.DictReader(open(p)):
        fen = (r.get('fen_start') or '').strip()
        if fen and not chess.Board(fen).is_check():
            fens.append(fen)
fens = list(dict.fromkeys(fens))

from ChessAI import ChessAI
seed = chess.Board(); ai = ChessAI(None, None, seed, seed.turn)
sf11 = SF11Eval(SF11)
PT = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]
NAMES = ['P', 'N', 'B', 'R', 'Q']
OURVAL = {'P': 1.0, 'N': 3.25, 'B': 3.45, 'R': 5.0, 'Q': 10.0}

X, y, impurity = [], [], []
for fen in fens:
    b = chess.Board(fen)
    signal.alarm(8)
    try:
        _, terms = sf11.eval(fen); signal.alarm(0)
    except _TO:
        continue
    finally:
        signal.alarm(0)
    if terms is None: continue
    counts = [(len(b.pieces(pt, chess.WHITE)) - len(b.pieces(pt, chess.BLACK))) for pt in PT]
    pure_mat = sum(counts[i] * OURVAL[NAMES[i]] for i in range(5))       # White-POV pawns, exactly linear
    bd_mat = -ai.ev_breakdown(b).get('material', 0) / 1000.0             # the (possibly impure) breakdown term
    sf_mat = terms.get('Material', 0.0)
    impurity.append(bd_mat - pure_mat)
    X.append(counts + [1.0]); y.append(pure_mat - sf_mat)

X = np.array(X); y = np.array(y); imp = np.array(impurity)
coef, *_ = np.linalg.lstsq(X, y, rcond=None)
print("MATERIAL isolation  (n=%d positions; material-only)" % len(y))
print("(1) breakdown 'material' term vs PURE piece-value material:")
print("    mean(bd - pure) = %+.3f   mean|bd - pure| = %.3f pawns   (this is the PST/AST/mobility bundled in 'material')" % (imp.mean(), np.mean(np.abs(imp))))
print("(2) PURE material vs SF11 Material, per-piece fit (coef = OUR-minus-SF value, pawns):")
print("    residual mean|.| = %.3f pawns   (how non-linear/positional SF's Material term is)" % np.mean(np.abs(X @ coef - y)))
for i, nm in enumerate(NAMES):
    print("    %s: ours=%.2f  we value it %+.2f vs SF  => SF-implied ~%.2f" % (nm, OURVAL[nm], coef[i], OURVAL[nm]-coef[i]))
print("    intercept = %+.3f" % coef[-1])
