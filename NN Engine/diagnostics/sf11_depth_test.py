# -*- coding: utf-8 -*-
"""Hypothesis test: is the pawn over-read a DEPTH problem or an EVAL problem? For each position, compare OUR
static eval to SF11 (classical HCE, apples-to-apples) searched at OUR leaf depth (~12) vs deep (24) vs SF18.
If SF11@12 already sees the truth (~ SF11@24 / SF18) while we read +5, our EVAL is the bug (not depth). If
SF11@12 ALSO over-values, the position needs deeper search than we reach."""
import os, sys
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
os.environ.setdefault('KS_SAFE_CHECK_DEF', '5')
for a in sys.argv[1:]:
    if '=' in a: k, v = a.split('=', 1); os.environ[k] = v
THIS = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(THIS)); sys.path.insert(0, THIS)
import chess, chess.engine
from ChessAI import ChessAI
sys.path.insert(0, os.path.join(os.path.dirname(THIS), "selfplay"))
from arbiter import find_stockfish

SF11 = "/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/stockfish_11/stockfish-11-win/Windows/stockfish_20011801_x64_bmi2.exe"

FENS = [
    ("fen3 (eg pawn over-read)", "rn6/5p2/pBp1pk2/P4p2/1p5b/5B1P/1P2K3/3R4 b - - 1 39"),
    ("P2   (eg pawn over-read)", "3r4/pp6/3k1p1p/3rp1b1/P1Rp2p1/3B4/2K3PP/4BR2 b - - 2 31"),
    ("P3   (genuine, SF18 +4.84)", "1k2r3/p7/8/4PQ2/3PK3/P3P3/2q2P1P/5B2 w - - 1 40"),
]

ai = ChessAI(None, None, chess.Board(), True)
def our_static(fen):
    b = chess.Board(fen); pov = 1.0 if b.turn else -1.0
    bd = ai.ev_breakdown(b); return (-bd.get("total", 0.0) / 1000.0) * pov

def sf_score(eng, fen, depth):
    b = chess.Board(fen); info = eng.analyse(b, chess.engine.Limit(depth=depth)); s = info["score"].white()
    pov = 1.0 if b.turn else -1.0
    v = ("M%d" % s.mate()) if s.is_mate() else "%+.2f" % (s.score() / 100.0 * pov)
    mv = b.san(info["pv"][0]) if info.get("pv") else "?"
    return "%s(%s)" % (v, mv)

sf11 = chess.engine.SimpleEngine.popen_uci(SF11)
sf18 = chess.engine.SimpleEngine.popen_uci(find_stockfish())
try:
    print("%-28s %8s | %-14s %-14s | %-14s" % ("position (our-POV)", "our_stat", "SF11@12", "SF11@24", "SF18@22"))
    for lab, fen in FENS:
        print("%-28s %+8.2f | %-14s %-14s | %-14s" % (
            lab, our_static(fen), sf_score(sf11, fen, 12), sf_score(sf11, fen, 24), sf_score(sf18, fen, 22)))
finally:
    sf11.quit(); sf18.quit()
