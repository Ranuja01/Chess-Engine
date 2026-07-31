# -*- coding: utf-8 -*-
"""Dump the full static eval breakdown for FEN(s) given on argv, sorted by |contribution| (our-POV pawns).
Isolates WHICH eval term drives an over-read. Run: pyrun diagnostics/dump_eval.py "<fen>" ["<fen2>" ...]"""
import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ.setdefault('KS_SAFE_CHECK_DEF', '5')
for _a in sys.argv[1:]:                       # KEY=VAL args -> env (before ChessAI import), FENs contain '/'
    if '=' in _a and '/' not in _a:
        _k, _v = _a.split('=', 1); os.environ[_k] = _v
THIS = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(THIS)); sys.path.insert(0, THIS)
import chess
from ChessAI import ChessAI
ai = ChessAI(None, None, chess.Board(), True)
fens = [a for a in sys.argv[1:] if "/" in a]
for fen in fens:
    b = chess.Board(fen)
    bd = ai.ev_breakdown(b)
    stm_black = b.turn == chess.BLACK
    print("\nFEN:", fen, " (side to move: %s)" % ("black" if stm_black else "white"))
    items = [(k, v) for k, v in bd.items() if k != "total"]
    # report in ABSOLUTE black-positive millipawns -> pawns; positive = good for Black
    items.sort(key=lambda kv: -abs(kv[1]))
    for k, v in items:
        if abs(v) >= 1.0:
            print("   %-26s %+8.2f  (black-positive pawns)" % (k, v / 1000.0))
    print("   %-26s %+8.2f" % ("TOTAL", bd.get("total", 0.0) / 1000.0))
