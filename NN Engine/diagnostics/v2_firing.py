# -*- coding: utf-8 -*-
"""Passer v2 firing probe: for the key passer FENs, print our static breakdown under whatever ENABLE_PASSER_V2
the runner set (env is applied before ChessAI import by the pyrun wrapper). Run twice (V2=0 then V2=1) and
diff by eye. Over-reads (fen3, P2) should DEFLATE toward SF18; the keep-out P3 (SF18 +4.84) must HOLD.
Reports our-POV (Black-positive internal flipped to side-to-move-POV)."""
import os, sys
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
os.environ.setdefault('KS_SAFE_CHECK_DEF', '5')   # shipped default
# Apply any KEY=VAL passed on argv to the environment BEFORE importing ChessAI (Config is read at import).
for a in sys.argv[1:]:
    if '=' in a:
        k, v = a.split('=', 1); os.environ[k] = v
THIS = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(THIS)); sys.path.insert(0, THIS)
import chess
from ChessAI import ChessAI

FENS = [
    ("fen3 (OVER-READ, deflate)", "rn6/5p2/pBp1pk2/P4p2/1p5b/5B1P/1P2K3/3R4 b - - 1 39"),
    ("P2   (OVER-READ, deflate)", "3r4/pp6/3k1p1p/3rp1b1/P1Rp2p1/3B4/2K3PP/4BR2 b - - 2 31"),
    ("P3   (KEEP-OUT, must hold, SF18 +4.84)", "1k2r3/p7/8/4PQ2/3PK3/P3P3/2q2P1P/5B2 w - - 1 40"),
]
TERMS = ["pt_pawns", "passed_pawn_support", "ae_passer", "material", "capture_gains", "total"]
ai = ChessAI(None, None, chess.Board(), True)
print("ENABLE_PASSER_V2=%s KS_SAFE_CHECK_DEF=%s" % (os.environ.get("ENABLE_PASSER_V2", "0"), os.environ.get("KS_SAFE_CHECK_DEF")))
for lab, fen in FENS:
    b = chess.Board(fen); pov = 1.0 if b.turn else -1.0
    bd = ai.ev_breakdown(b)
    cells = "  ".join("%s=%+.2f" % (t, (-bd.get(t, 0.0) / 1000.0) * pov) for t in TERMS)
    print("  %-40s %s" % (lab, cells))
