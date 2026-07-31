# -*- coding: utf-8 -*-
"""KS shortfall breakdown: our KS internals (KS_DEBUG_DUMP: attacked-zone / weak / safe-checks / attacker-
pieces / defender-pieces / open-files / units / danger, per king) vs SF11's King-safety term, on the KS-class
collapse positions. Goal: find where WE under-count the counter-attack (esp. danger to OUR OWN king).
P2b = we defend (should be bad); P3 = we attack (should be good) -> tests the offense/defense asymmetry."""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
os.environ['KS_DEBUG_DUMP'] = '1'
os.environ['KS_FLOOR'] = '0'   # disable deadzone so the dump prints even for modest-unit kings
import sys
THIS = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(THIS)); sys.path.insert(0, THIS)
import chess
from eval_vs_sf11 import SF11Eval, SF11
from ChessAI import ChessAI

CASES = [
    ("P2b DEFEND (Black attacks Wk)", "1rBq4/2p2pk1/3p2p1/3Pp2n/2p1P2r/1RN1N3/P1P2P1P/4R1K1 w - - 0 31"),
    ("P3  ATTACK (White mates Bk)",   "1r6/2r1P3/2bR4/2P2kN1/7P/2n2PK1/6P1/1q2R3 w - - 1 49"),
]
ai = ChessAI(None, None, chess.Board(), True)
sf = SF11Eval(SF11)
try:
    for lab, fen in CASES:
        b = chess.Board(fen); us = b.turn; pov = 1.0 if us else -1.0
        print("\n===== %s =====" % lab)
        print("  FEN:", fen)
        sys.stderr.flush(); sys.stdout.flush()
        bd = ai.ev_breakdown(b); sys.stderr.flush()
        # our king_safety term is Black-positive; report BOTH raw and White-POV
        ks_bp = bd.get("king_safety", 0.0) / 1000.0
        print("  OUR king_safety term: %+.2f (Black-pos)  = %+.2f (White-POV)" % (ks_bp, ks_bp * -1))
        print("  SF11 King safety term: %+.2f (White-POV)" % sf.eval(fen)[1].get("King safety", 0.0))
finally:
    sf.close()
