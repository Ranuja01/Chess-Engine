# -*- coding: utf-8 -*-
"""Isolate whether the turn-dependent KS_SAFE_CHECK_DEF breaks COLOR SYMMETRY of the king_safety TERM
specifically (my change), vs the pre-existing total-eval asymmetry (capgains is also turn-dependent).
Runs the KS-term mirror test at DEF=3 (baseline) and DEF=5 in separate child processes."""
import os, sys
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
THIS = os.path.dirname(os.path.abspath(__file__))
if len(sys.argv) > 1:
    os.environ['KS_SAFE_CHECK_DEF'] = sys.argv[1]
    sys.path.insert(0, os.path.dirname(THIS)); sys.path.insert(0, THIS)
    import chess
    from ChessAI import ChessAI
    ai = ChessAI(None, None, chess.Board(), True)
    def ks(fen): return ai.ev_breakdown(chess.Board(fen)).get("king_safety", 0.0) / 1000.0
    def tot(fen): return ai.ev_breakdown(chess.Board(fen)).get("total", 0.0) / 1000.0
    danger = [ln.rstrip("\n").split("\t",1)[-1].strip() for ln in open(os.path.join(THIS,"ks_sets","danger.txt")) if ln.strip()]
    wks = wtot = 0.0
    for fen in danger:
        m = chess.Board(fen).mirror().fen()
        wks  = max(wks,  abs(ks(fen)  + ks(m)))
        wtot = max(wtot, abs(tot(fen) + tot(m)))
    print("DEF=%s : worst |KS_term(f)+KS_term(mirror)|=%.4f   worst |total(f)+total(mirror)|=%.4f" % (sys.argv[1], wks, wtot))
    sys.exit(0)
import subprocess
for v in ["3", "5"]:
    subprocess.run([sys.executable, os.path.abspath(__file__), v])
