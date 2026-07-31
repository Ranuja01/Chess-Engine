# -*- coding: utf-8 -*-
"""Dump the capgains pending-capture stacks + resolution totals for the high-fold NO-PIN collapse fens,
to see the multi-square over-resolution. Requires the engine built with the CAPG_DEBUG_DUMP diagnostic;
set CAPG_DEBUG_DUMP=1 so the gated stderr dump fires inside ev_breakdown (search path stays byte-identical)."""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
os.environ['CAPG_DEBUG_DUMP'] = '1'
import sys
THIS = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(THIS)); sys.path.insert(0, THIS)
import chess
from ChessAI import ChessAI

# High-fold NO-PIN positions from ks_pin_phantom.py (fold, our_mat, raw); plus the pin case for contrast.
FENS = [
    ("PIN  fold+17.05", "rn2k2r/4bppp/2p5/1pQn4/6P1/P4N2/P2BR2P/1K6 b kq - 0 25"),
    ("NOPIN fold+10.80", "r3r1k1/3nbppp/1p1pb3/3p2P1/1P1N1P1P/1Q2B3/4BP2/R5K1 b - - 0 27"),
    ("NOPIN fold+10.00", "1r6/2r1P3/2bR4/2P2kN1/7P/2n2PK1/6P1/1q2R3 w - - 1 49"),
    ("NOPIN fold +6.07", "1rBq4/2p2pk1/3p2p1/3Pp2n/2p1P2r/1RN1N3/P1P2P1P/4R1K1 w - - 0 31"),
]
VAL = {1: "P", 2: "N", 3: "B", 4: "R", 5: "Q", 6: "K"}
ai = ChessAI(None, None, chess.Board(), True)
for lab, fen in FENS:
    b = chess.Board(fen)
    sys.stderr.flush()
    print("\n================= %s | %s (%s to move) =================" % (lab, fen, "W" if b.turn else "B"))
    sys.stdout.flush()
    bd = ai.ev_breakdown(b)
    sys.stderr.flush()
    us = b.turn
    pov = 1.0 if us else -1.0
    print("  our material term (our POV) = %+.2f   capture_gains term = %+.2f" % (
        (-bd.get("material", 0.0)/1000.0)*pov, (-bd.get("capture_gains", 0.0)/1000.0)*pov))
