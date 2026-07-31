# -*- coding: utf-8 -*-
import os, sys
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
# KS on, safe-check dominant, phase taper WIDENED to isolate the units question
os.environ.update(dict(KING_SAFETY_MAG="4000"))   # realistic operating MAG (ks_tournament used 4000); default weights
THIS = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(THIS))
import chess
from ChessAI import ChessAI
ai = ChessAI(None, None, chess.Board(), True)
fens = [l.split("\t", 1)[1].strip() for l in open(os.path.join(THIS, "ks_sets", "danger.txt"))]
print("fen                                                            phase  is_eg   ourKS(taper~off)")
hi = 0
for f in fens[:14]:
    b = chess.Board(f)
    bd = ai.ev_breakdown(b)
    ps = bd.get("phase_score"); eg = bd.get("is_endgame")
    povsign = 1.0 if b.turn == chess.WHITE else -1.0
    ks = (-bd.get("king_safety", 0.0) / 1000.0) * povsign
    print("%-58s  %4s   %5s   %+.2f" % (f[:58], ps, eg, ks))
import statistics
allps = []
for f in fens:
    ps = ai.ev_breakdown(chess.Board(f)).get("phase_score")
    if ps is not None: allps.append(ps)
print("\nphase_score over danger set: mean=%.0f min=%d max=%d  #>=104(taper=0 at DEFAULT)=%d/%d"
      % (statistics.mean(allps), min(allps), max(allps), sum(1 for p in allps if p >= 104), len(allps)))
