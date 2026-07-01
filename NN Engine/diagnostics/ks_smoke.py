# -*- coding: utf-8 -*-
"""Smoke-test the KS_INTERACT 'coffin' term: it should ADD king-danger on an open/attacked king and leave a
quiet/closed/defended king ~unchanged. Bakes the m4000ctl anchor env internally (prompt-free via `pyrun`);
KS_INTERACT is argv[1]. Run twice and diff the king_safety term:
    pyrun diagnostics/ks_smoke.py 0
    pyrun diagnostics/ks_smoke.py 96
"""
import os
os.environ['ENABLE_KS_REPLACE_LT'] = '1'
os.environ['KING_SAFETY_MAG'] = '4000'
os.environ['MOD_KS_CONTROL'] = '256'
import sys
os.environ['KS_INTERACT'] = sys.argv[1] if len(sys.argv) > 1 else '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ENGINE_DIR = os.path.dirname(THIS_DIR)
sys.path.insert(0, ENGINE_DIR)

import chess
from ChessAI import ChessAI

# Midgame positions (many pieces ⇒ KS not phase-tapered). The interaction needs an OPEN file at the king +
# attackers + few defenders. sheltered = intact f7g7h7 pawns (open_files 0 ⇒ interaction must stay 0, a
# control). gfile_open / exposed = the black g-pawn is gone so the g-file is open with White pieces bearing in.
TESTS = [
    ("sheltered_ctrl", "r2qr1k1/ppp2ppp/2n5/3p4/3Pn3/2PB1N2/PP3PPP/R1BQR1K1 w - - 0 1"),
    ("gfile_open",     "r2qr1k1/ppp2p1p/2n4B/3p4/3Pn3/2PB1N2/PP3PPP/R2QR1K1 w - - 0 1"),
    ("exposed_heavy",  "r2q1r1k/ppp2p1p/2n4B/3p1Q2/3Pn3/2PB1N2/PP3PPP/R3R1K1 w - - 0 1"),
]

seed = chess.Board()
ai = ChessAI(None, None, seed, seed.turn)
print("KS_INTERACT=%s" % os.environ['KS_INTERACT'])
for label, fen in TESTS:
    bd = ai.ev_breakdown(chess.Board(fen))
    print("  %-15s phase=%3d  ks_units_b=%3d  king_safety=%6d  total=%7d"
          % (label, bd["phase_score"], bd["det_ks_units_b"], bd["king_safety"], bd["total"]))
