# -*- coding: utf-8 -*-
"""Residual KS under-read: with the SHIP bundle active, where does our king_safety still fall far short of
SF11's King safety on the danger set? Dump the worst under-reads (SF11 sees danger we still miss) so we can
reason about which SF sub-signal (king-flank, mobility-in-danger, pattern) we lack. Our POV pawns.

Run: pyrun diagnostics/ks_residual.py
"""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
os.environ.update(dict(ENABLE_KS_REPLACE_LT="1", KING_SAFETY_MAG="3000", KS_DEFENDER="0",
                       ENABLE_KS_SF_WEAK="1", ENABLE_KS_SF_SAFECHECK="1", KS_FLOOR="13", KS_NO_QUEEN="6"))
import sys
THIS = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(THIS)); sys.path.insert(0, THIS)
import chess
from eval_vs_sf11 import SF11Eval, SF11
from ChessAI import ChessAI
ai = ChessAI(None, None, chess.Board(), True)
sf = SF11Eval(SF11)

rows = []
try:
    for ln in open(os.path.join(THIS, "ks_sets", "danger.txt")):
        fen = ln.rstrip("\n").split("\t", 1)[1]
        b = chess.Board(fen)
        povsign = 1.0 if b.turn == chess.WHITE else -1.0
        bd = ai.ev_breakdown(b)
        our_ks = (-bd.get("king_safety", 0.0) / 1000.0) * povsign     # our-POV pawns
        _, terms = sf.eval(fen)
        sf_ks = terms.get("King safety", 0.0) * povsign
        rows.append((our_ks - sf_ks, our_ks, sf_ks, fen))   # gap>0 = we UNDER-read danger (ours less negative)
finally:
    sf.close()

import statistics
rows.sort(reverse=True)   # biggest under-read first
print("danger set: mean ourKS=%.2f  mean SF11KS=%.2f  (our-POV; more negative = more danger)"
      % (statistics.mean(r[1] for r in rows), statistics.mean(r[2] for r in rows)))
print("\nWORST UNDER-READS (SF11 sees danger our KS still misses):")
print("%-58s %8s %8s %8s" % ("fen", "ourKS", "SF11KS", "gap"))
for gap, ours, sfk, fen in rows[:10]:
    print("%-58s %+8.2f %+8.2f %+8.2f" % (fen[:58], ours, sfk, gap))
