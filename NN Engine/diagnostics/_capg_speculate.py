# -*- coding: utf-8 -*-
"""Re-dump full eval breakdowns (pin ON) for the capgains-relevant collapse fens + SF11 total, our POV,
so we can speculate on what actually drives the residual over-read (tempo barely moved it)."""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
os.environ['ENABLE_CAPG_PIN'] = '1'
import sys
THIS = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(THIS)); sys.path.insert(0, THIS)
import chess
from eval_vs_sf11 import SF11Eval, SF11
from ChessAI import ChessAI

FENS = [
    ("PIN (fixed)",       "rn2k2r/4bppp/2p5/1pQn4/6P1/P4N2/P2BR2P/1K6 b kq - 0 25"),
    ("tempo-motivator",   "1rbq1rk1/ppp2pb1/7p/2n1pnpP/4Q3/2NP1NP1/PPPB1PB1/2K1R2R w - - 2 15"),
    ("deep-poison Rxa1",  "r3r1k1/3nbppp/1p1pb3/3p2P1/1P1N1P1P/1Q2B3/4BP2/R5K1 b - - 0 27"),
    ("multi-square x4",   "1rBq4/2p2pk1/3p2p1/3Pp2n/2p1P2r/1RN1N3/P1P2P1P/4R1K1 w - - 0 31"),
    ("imbalance-driven",  "1r6/2r1P3/2bR4/2P2kN1/7P/2n2PK1/6P1/1q2R3 w - - 1 49"),
]
KEYS = ["material", "capture_gains", "piece_value_boost", "pieces", "pt_pawns",
        "imbalance_white", "imbalance_black", "king_safety", "total"]
ai = ChessAI(None, None, chess.Board(), True)
sf = SF11Eval(SF11)
VAL = {chess.PAWN: 1.0, chess.KNIGHT: 3.25, chess.BISHOP: 3.45, chess.ROOK: 5.0, chess.QUEEN: 10.0}
try:
    print("%-18s %3s %7s | %s | %7s %7s" % ("case", "stm", "raw", " ".join("%9s" % k[:9] for k in KEYS), "SF11tot", "gap"))
    for lab, fen in FENS:
        b = chess.Board(fen); us = b.turn; them = not us; pov = 1.0 if us else -1.0
        bd = ai.ev_breakdown(b)
        vals = [(-bd.get(k, 0.0) / 1000.0) * pov for k in KEYS]
        raw = sum(VAL[pt] * (len(b.pieces(pt, us)) - len(b.pieces(pt, them))) for pt in VAL)
        sf_tot = sf.eval(fen)[1].get("Total", sf.eval(fen)[0]) * pov
        our_tot = vals[-1]
        print("%-18s %3s %+7.2f | %s | %+7.2f %+7.2f" % (
            lab, "W" if us else "B", raw, " ".join("%+9.2f" % v for v in vals), sf_tot, our_tot - sf_tot))
finally:
    sf.close()
