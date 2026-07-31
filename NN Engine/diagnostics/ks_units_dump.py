# -*- coding: utf-8 -*-
"""Dump RAW king-safety UNITS (g_ks_units_white/black, pre-floor) for FENs, next to KS_FLOOR, so we can see
whether an attack SF11 scores 2-5 pawns lands BELOW our floor (units < KS_FLOOR -> zeroed). Run:
pyrun diagnostics/ks_units_dump.py "<fen>" [...]"""
import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ.setdefault('KS_SAFE_CHECK_DEF', '5')
THIS = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(THIS)); sys.path.insert(0, THIS)
import chess
from ChessAI import ChessAI
ai = ChessAI(None, None, chess.Board(), True)
FLOOR = int(os.environ.get("KS_FLOOR", "13"))
print("KS_FLOOR = %d   (units below this -> KS zeroed)" % FLOOR)
print("%-6s %10s %10s %8s  fen" % ("stm", "unitsW", "unitsB", "KSpawns"))
for fen in [a for a in sys.argv[1:] if "/" in a]:
    b = chess.Board(fen)
    bd = ai.ev_breakdown(b)
    uw = bd.get("det_ks_units_w", 0.0); ub = bd.get("det_ks_units_b", 0.0)
    ks = -bd.get("king_safety", 0.0) / 1000.0
    print("%-6s %10.3f %10.3f %+8.2f  %s" % ("b" if b.turn else "w", uw, ub, ks, fen))
