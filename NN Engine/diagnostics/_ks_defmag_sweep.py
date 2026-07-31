# -*- coding: utf-8 -*-
"""Preliminary sweep of KS_DEF_MAG (defensive danger multiplier) ON TOP of KS_SAFE_CHECK_DEF=5. Per value:
P2b White-king KS (deepens?), calm-control leaks (does the bigger magnitude leak more false-fires?). Fresh
child per value. (STS is run separately via the runner's sts sub since it needs the engine harness.)"""
import os, sys
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
THIS = os.path.dirname(os.path.abspath(__file__))
if len(sys.argv) > 1:
    os.environ["KS_SAFE_CHECK_DEF"] = "5"
    os.environ["KS_DEF_MAG"] = sys.argv[1]
    sys.path.insert(0, os.path.dirname(THIS)); sys.path.insert(0, THIS)
    import chess
    from ChessAI import ChessAI
    ai = ChessAI(None, None, chess.Board(), True)
    def ks(fen):
        b = chess.Board(fen); pov = 1.0 if b.turn else -1.0
        return (ai.ev_breakdown(b).get("king_safety", 0.0) / 1000.0) * -1
    P2B = "1rBq4/2p2pk1/3p2p1/3Pp2n/2p1P2r/1RN1N3/P1P2P1P/4R1K1 w - - 0 31"
    calm = [ln.rstrip("\n").split("\t",1)[-1].strip() for ln in open(os.path.join(THIS,"ks_sets","control_calm.txt")) if ln.strip()]
    cl = sum(1 for f in calm if abs(ks(f)) > 0.02)
    print("MAG=%-4s  P2b_KS=%+.2f  calm_leaks=%d/%d" % (sys.argv[1], ks(P2B), cl, len(calm)))
    sys.exit(0)
import subprocess
for v in ["100", "150", "200", "300"]:
    subprocess.run([sys.executable, os.path.abspath(__file__), v])
