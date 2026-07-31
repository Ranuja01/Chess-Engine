# -*- coding: utf-8 -*-
"""Confirm defensive-only KS_SAFE_CHECK_DEF=5: (1) P2b (White defends) fires ~SF11's -1.40; (2) color-symmetry
holds despite the turn-dependent defensive weight -- eval(fen) must equal -eval(mirror) where mirror swaps
colors AND side-to-move. Tests total-eval symmetry over the danger corpus."""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
os.environ['KS_SAFE_CHECK_DEF'] = '5'
import sys
THIS = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(THIS)); sys.path.insert(0, THIS)
import chess
from ChessAI import ChessAI
ai = ChessAI(None, None, chess.Board(), True)

def total_bp(fen):
    return ai.ev_breakdown(chess.Board(fen)).get("total", 0.0) / 1000.0   # Black-positive absolute

P2B = "1rBq4/2p2pk1/3p2p1/3Pp2n/2p1P2r/1RN1N3/P1P2P1P/4R1K1 w - - 0 31"
b = chess.Board(P2B)
ks = ai.ev_breakdown(b).get("king_safety", 0.0) / 1000.0 * -1
print("P2b (DEF=5) king_safety = %+.2f (White-POV)   [target ~ SF11 -1.40]" % ks)

danger = [ln.rstrip("\n").split("\t",1)[-1].strip() for ln in open(os.path.join(THIS,"ks_sets","danger.txt")) if ln.strip()]
worst = 0.0
for fen in danger:
    b = chess.Board(fen)
    m = b.mirror()   # swaps colors AND side-to-move
    d = total_bp(fen) + total_bp(m.fen())   # should be ~0 (Black-pos: total(mirror) = -total(orig))
    worst = max(worst, abs(d))
print("worst |total(fen) + total(mirror)| over danger corpus = %.4f pawns (0 = perfectly symmetric)" % worst)
