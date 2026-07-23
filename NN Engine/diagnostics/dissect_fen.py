# -*- coding: utf-8 -*-
"""Dump our INTERNAL eval breakdown (incl. raw KS attack-units) for one or more FENs, to run a differential
diagnosis (why does KS/attack over-fire on one position but read correctly on another). Knobs via KEY=VAL argv.
  pyrun diagnostics/dissect_fen.py "<fen1>" "<fen2>" ENABLE_PASSER_V3=1
"""
import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'; os.environ.setdefault('KS_SAFE_CHECK_DEF', '5')
for a in sys.argv[1:]:
    if '=' in a and '/' not in a:
        k, v = a.split('=', 1); os.environ[k] = v
THIS = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(THIS)); sys.path.insert(0, THIS)
import chess
from ChessAI import ChessAI
ai = ChessAI(None, None, chess.Board(), True)

fens = [a for a in sys.argv[1:] if '/' in a]
# millipawn terms (absolute Black-positive; shown white-POV = negated)
MP = ["material", "br_kaufman", "king_safety", "latent_threat", "central", "capture_gains",
      "passed_pawn_support", "imbalance_white", "imbalance_black",
      "pt_knights", "pt_bishops", "pt_rooks", "pt_queens"]
RAW = ["det_ks_units_w", "det_ks_units_b"]  # raw attack-units against each king (not millipawns)
for fen in fens:
    bd = ai.ev_breakdown(chess.Board(fen))
    print("FEN:", fen)
    print("  our total %+.2f (white-POV)" % (-bd.get("total", 0) / 1000.0))
    print("  KS attack-units:  vs-White-K %d   vs-Black-K %d" % (bd.get("det_ks_units_w", 0), bd.get("det_ks_units_b", 0)))
    parts = []
    for k in MP:
        v = -bd.get(k, 0) / 1000.0
        if abs(v) >= 0.20:
            parts.append("%s %+.2f" % (k, v))
    print("  terms (white-POV, |>=0.2|): " + "  ".join(parts))
    print()
