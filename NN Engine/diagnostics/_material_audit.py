# -*- coding: utf-8 -*-
"""Audit whether ev_breakdown's `material` term equals raw material under our OWN piece-value table.

Motivation (2026-07-30): on a real collapse position det_w_pieceval/det_b_pieceval reported 51.25/54.00
where hand-summing cpp_bitboard.h's `values` table gives 35.70/35.45 -- a 3.00-pawn differential error
exactly equal to the pawn-count difference. This checks whether that reproduces at scale, and whether the
residual tracks pawn count (double-count signature) or something else.

ALL OUTPUT IS WHITE-POV (positive = good for White); our eval is Black-positive internally and is negated.

Run: pyrun diagnostics/_material_audit.py [N=400] [SRC=diagnostics/ks_sets/position_bank.csv]
"""
import os, sys, csv
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
THIS = os.path.dirname(os.path.abspath(__file__))
ENGINE_DIR = os.path.dirname(THIS)
sys.path.insert(0, ENGINE_DIR)
os.chdir(ENGINE_DIR)

import chess
from ChessAI import ChessAI

# cpp_bitboard.h:141  constexpr std::array<int,7> values = {0,1000,3250,3450,5000,10000,12000}
VAL = {chess.PAWN: 1000, chess.KNIGHT: 3250, chess.BISHOP: 3450,
       chess.ROOK: 5000, chess.QUEEN: 10000, chess.KING: 12000}

N = 400
SRC = os.path.join(THIS, "ks_sets", "position_bank.csv")
for a in sys.argv[1:]:
    if a.isdigit():
        N = int(a)
    elif a.endswith(".csv") or a.endswith(".epd"):
        SRC = a

fens = []
if SRC.endswith(".csv"):
    for r in csv.DictReader(open(SRC)):
        f = r.get("fen") or r.get("decision_fen")
        if f:
            fens.append(f)
else:
    for ln in open(SRC):
        ln = ln.strip()
        if ln:
            fens.append(ln.split(";")[0])
fens = fens[:N]

def raw_material(board):
    """White-minus-black piece-value sum under OUR table, in millipawns."""
    tot = 0
    for sq, pc in board.piece_map().items():
        v = VAL[pc.piece_type]
        tot += v if pc.color == chess.WHITE else -v
    return tot

seed = chess.Board(fens[0])
ai = ChessAI(None, None, seed, seed.turn)

rows = []
for fen in fens:
    try:
        b = chess.Board(fen)
    except Exception:
        continue
    bd = ai.ev_breakdown(b)
    if bd.get("checkmate"):
        continue
    mat_reported = -bd.get("material", 0)          # -> White-POV
    dw, db = bd.get("det_w_pieceval", 0), bd.get("det_b_pieceval", 0)
    if dw == 0 and db == 0:
        continue                                    # insufficient-material short-circuit
    mat_expected = raw_material(b)
    resid = mat_reported - mat_expected
    pawn_diff = (len(b.pieces(chess.PAWN, chess.WHITE))
                 - len(b.pieces(chess.PAWN, chess.BLACK)))
    # per-side check too: does each accumulator match its own raw sum?
    rw = sum(VAL[p.piece_type] for p in b.piece_map().values() if p.color == chess.WHITE)
    rb = sum(VAL[p.piece_type] for p in b.piece_map().values() if p.color == chess.BLACK)
    rows.append((fen, mat_reported, mat_expected, resid, pawn_diff, (dw - rw, db - rb)))

exact = sum(1 for r in rows if r[3] == 0)
print(f"positions audited: {len(rows)}   material EXACT: {exact}   mismatched: {len(rows)-exact}")

if rows:
    res = [r[3] for r in rows]
    print(f"residual (reported - expected, millipawns, White-POV):")
    print(f"   mean={sum(res)/len(res):+.0f}  min={min(res):+d}  max={max(res):+d}")
    # does the residual track the pawn-count difference?
    byp = {}
    for _, _, _, r, pd, _ in rows:
        byp.setdefault(pd, []).append(r)
    print("\n   residual by (white_pawns - black_pawns):")
    for pd in sorted(byp):
        v = byp[pd]
        print(f"      pawn_diff {pd:+d}:  n={len(v):>4}  mean_resid={sum(v)/len(v):+8.0f}")
    sw = [r[5][0] for r in rows]; sb = [r[5][1] for r in rows]
    print(f"\n   PER-SIDE accumulator vs raw sum (det_w - raw_w, det_b - raw_b):")
    print(f"      white: mean={sum(sw)/len(sw):+.0f}  min={min(sw):+d}  max={max(sw):+d}")
    print(f"      black: mean={sum(sb)/len(sb):+.0f}  min={min(sb):+d}  max={max(sb):+d}")
    print("\n   worst 5 by |residual|:")
    for fen, rep, exp, r, pd, sd in sorted(rows, key=lambda x: -abs(x[3]))[:5]:
        print(f"      resid={r:+7d} reported={rep:+7d} expected={exp:+7d} pawn_diff={pd:+d} "
              f"side_excess=(w{sd[0]:+d},b{sd[1]:+d})  {fen[:44]}")
