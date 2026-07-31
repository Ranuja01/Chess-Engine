# -*- coding: utf-8 -*-
"""Trigger the existing KS_DEBUG_DUMP (cpp_bitboard.cpp:5151) to print per-king KS sub-components
(attacked-squares / weak / safe-checks / attacker-pieces / defender-pieces / open-files / units / danger) for
FENs on argv, so we can see WHICH signal is smallest on the broken attacks before fitting. Sets KS_DEBUG_DUMP in
the process env (os.environ -> putenv -> C++ getenv) so no shell env-prefix is needed.
Run: pyrun diagnostics/ks_subcomponent_dump.py "<fen>" [...]"""
import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ.setdefault('KS_SAFE_CHECK_DEF', '5')
os.environ['KS_DEBUG_DUMP'] = '1'          # must be set BEFORE any eval call (C++ getenv reads it live)
os.environ['KS_TRACE'] = '1'               # per-zone-square trace (attackers/defenders/weak verdict)
os.environ['KS_FLOOR'] = '0'               # disable the early-return floor so floored positions still dump
THIS = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(THIS)); sys.path.insert(0, THIS)
import chess
from ChessAI import ChessAI
ai = ChessAI(None, None, chess.Board(), True)
for fen in [a for a in sys.argv[1:] if "/" in a]:
    sys.stderr.write("\n=== %s ===\n" % fen); sys.stderr.flush()
    ai.ev_breakdown(chess.Board(fen))      # KSD lines are emitted to stderr by the C++ dump
    sys.stdout.flush(); sys.stderr.flush()
