# -*- coding: utf-8 -*-
"""Collective per-term triangulation of the collapse set: for each collapse FEN, decompose the gap vs SF11
(the hand-fixable classical ceiling) into KS / Material / capture_gains, and tally which term DOMINATES.
Answers 'what drives our collapses' at the distribution level, not one FEN at a time.

Skips in-check FENs (SF11's classical eval returns None on them and its reader has no timeout -> hang), and
guards every SF11 call with a SIGALRM timeout as a belt-and-braces against an unattended stall.

Run (via the allowlisted runner, which sets STOCKFISH_PATH):
  bash <runner> pyrun diagnostics/_collapse_term_gaps.py [fens.csv]
"""
import os, sys, signal
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
import csv
import chess

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ENGINE_DIR = os.path.dirname(THIS_DIR)
sys.path.insert(0, ENGINE_DIR)
sys.path.insert(0, THIS_DIR)
from eval_vs_sf11 import SF11Eval, SF11

FENS = sys.argv[1] if len(sys.argv) > 1 else 'selfplay/games/vssf_2400/dp_fens.csv'

class _Timeout(Exception):
    pass

def _alarm(signum, frame):
    raise _Timeout()

signal.signal(signal.SIGALRM, _alarm)

# load fens
rows = []
with open(os.path.join(ENGINE_DIR, FENS)) as f:
    rd = csv.DictReader(f)
    col = 'fen_start' if 'fen_start' in (rd.fieldnames or []) else (rd.fieldnames or ['fen'])[0]
    for r in rd:
        fen = (r.get(col) or '').strip()
        if fen:
            rows.append(fen)

from ChessAI import ChessAI
seed = chess.Board()
ai = ChessAI(None, None, seed, seed.turn)
sf11 = SF11Eval(SF11)

def sf11_terms(fen):
    signal.alarm(8)
    try:
        tot, terms = sf11.eval(fen)
        signal.alarm(0)
        return tot, terms
    except _Timeout:
        return None, None
    finally:
        signal.alarm(0)

n_check = n_sf_fail = 0
recs = []   # (fen, ks_gap, mat_gap, cap_mag, driver)
for fen in rows:
    b = chess.Board(fen)
    if b.is_check():
        n_check += 1
        continue
    bd = ai.ev_breakdown(b)
    ks_o = -bd.get('king_safety', 0) / 1000.0        # White-POV pawns
    # MATERIAL FAMILY: our material valuation is spread across these terms; SF11 bundles it as Material+Imbalance.
    # Compare families, not the raw 'material' term to 'Material' (that mismatch inflates the material gap).
    mat_o = -(bd.get('material', 0) + bd.get('imbalance_white', 0) + bd.get('imbalance_black', 0)
              + bd.get('pair_bonus', 0) + bd.get('piece_value_boost', 0)) / 1000.0
    cap_o = -bd.get('capture_gains', 0) / 1000.0
    tot11, terms11 = sf11_terms(fen)
    if tot11 is None or terms11 is None:
        n_sf_fail += 1
        continue
    ks11 = terms11.get('King safety', 0.0)
    mat11 = terms11.get('Material', 0.0) + terms11.get('Imbalance', 0.0)
    ks_gap = ks_o - ks11          # how much LESS king-danger we read than SF (signed, White-POV)
    mat_gap = mat_o - mat11        # our material-FAMILY over/under vs SF (Material+Imbalance)
    cap_mag = cap_o                # SF has no capture_gains term; our value is the artifact magnitude
    # HANDLE split: base piece-values (compile-time, heavy) vs imbalance-family (runtime knobs).
    base_o = -bd.get('material', 0) / 1000.0
    imb_o = -(bd.get('imbalance_white', 0) + bd.get('imbalance_black', 0)
              + bd.get('pair_bonus', 0) + bd.get('piece_value_boost', 0)) / 1000.0
    base_gap = base_o - terms11.get('Material', 0.0)
    imb_gap = imb_o - terms11.get('Imbalance', 0.0)
    driver = max((('KS', abs(ks_gap)), ('MAT', abs(mat_gap)), ('CAPG', abs(cap_mag))), key=lambda t: t[1])[0]
    recs.append((fen, ks_gap, mat_gap, cap_mag, driver, base_gap, imb_gap))

sf11.close()

import statistics as st
def col(i):
    return [abs(r[i]) for r in recs]

print("collapse-set term triangulation vs SF11  (%d fens; %d in-check skipped; %d sf-fail)"
      % (len(recs), n_check, n_sf_fail))
if recs:
    print("mean|gap|  KS=%.3f  MAT=%.3f  CAPG=%.3f" % (st.mean(col(1)), st.mean(col(2)), st.mean(col(3))))
    print("  MAT split  BASE-values=%.3f  IMBALANCE-family=%.3f  (handle: base=compile-time, imb=runtime knobs)"
          % (st.mean([abs(r[5]) for r in recs]), st.mean([abs(r[6]) for r in recs])))
    from collections import Counter
    c = Counter(r[4] for r in recs)
    print("dominant-driver count:  " + "  ".join("%s=%d" % (k, c.get(k, 0)) for k in ('KS', 'MAT', 'CAPG')))
    # positions where our KS is a real UNDER-read (we read less danger than SF by > 1 pawn)
    ks_under = [r for r in recs if r[1] > 1.0]
    cap_big = [r for r in recs if abs(r[3]) > 1.0]
    print("KS under-read >1.0: %d/%d    capgains |mag|>1.0: %d/%d" % (len(ks_under), len(recs), len(cap_big), len(recs)))
    print("\nper-fen (sorted by dominant magnitude):")
    for r in sorted(recs, key=lambda r: -max(abs(r[1]), abs(r[2]), abs(r[3])))[:25]:
        print("  %-4s ks_gap=%+.2f mat_gap=%+.2f capg=%+.2f  %s" % (r[4], r[1], r[2], r[3], r[0]))
