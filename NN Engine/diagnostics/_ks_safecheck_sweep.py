# -*- coding: utf-8 -*-
"""Sweep KS_SAFE_CHECK at the real KS_FLOOR=13, one FRESH process per value (C++ Config is read once at init).
Parent forks a child per value; child sets KS_SAFE_CHECK in env BEFORE importing ChessAI. Report: P2b White-king
KS (should clear the floor toward SF11's -1.40), DANGER-corpus fires (real attacks lifted), CALM-control leaks."""
import os, sys
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
THIS = os.path.dirname(os.path.abspath(__file__))

if len(sys.argv) > 1:   # ---- CHILD: one KS_SAFE_CHECK value ----
    os.environ["KS_SAFE_CHECK"] = sys.argv[1]
    sys.path.insert(0, os.path.dirname(THIS)); sys.path.insert(0, THIS)
    import chess
    from ChessAI import ChessAI
    ai = ChessAI(None, None, chess.Board(), True)
    def load(fp):
        return [ln.rstrip("\n").split("\t", 1)[-1].strip() for ln in open(os.path.join(THIS, "ks_sets", fp)) if ln.strip()]
    def ks(fen):
        b = chess.Board(fen); pov = 1.0 if b.turn else -1.0
        return (ai.ev_breakdown(b).get("king_safety", 0.0) / 1000.0) * -1  # White-POV
    P2B = "1rBq4/2p2pk1/3p2p1/3Pp2n/2p1P2r/1RN1N3/P1P2P1P/4R1K1 w - - 0 31"
    EPS = 0.02
    danger, calm = load("danger.txt"), load("control_calm.txt")
    df = sum(1 for f in danger if abs(ks(f)) > EPS)
    cl = sum(1 for f in calm if abs(ks(f)) > EPS)
    print("%3s | %+13.2f | %5d / %-6d | %5d / %-6d" % (sys.argv[1], ks(P2B), df, len(danger), cl, len(calm)))
    sys.exit(0)

# ---- PARENT: fork a child per value ----
import subprocess
print("%3s | %-14s | %-14s | %-14s" % ("SC", "P2b KS(W-POV)", "danger fires", "CALM leaks"))
sys.stdout.flush()
for v in [3, 4, 5, 6, 7]:
    subprocess.run([sys.executable, os.path.abspath(__file__), str(v)])
