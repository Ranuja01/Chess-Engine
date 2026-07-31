# -*- coding: utf-8 -*-
"""How do SF11 static + SF18 search differentiate the P2 sub-positions (where the hanging rook SHOULD matter)
from the original (where White wins anyway)? And is P2b a KS/attack case? Show SF11 attack-relevant terms
(King safety, Threats, Mobility, Space, Passed) + Total, and SF18 search. White-POV, pawns."""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
import sys
THIS = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(THIS)); sys.path.insert(0, THIS)
import chess, chess.engine
from eval_vs_sf11 import SF11Eval, SF11
sys.path.insert(0, os.path.join(os.path.dirname(THIS), "selfplay"))
from arbiter import find_stockfish

CASES = [
    ("P2 orig (a1 hang)", "r3r1k1/3nbppp/1p1pb3/3p2P1/1P1N1P1P/1Q2B3/4BP2/R5K1 b - - 0 27"),
    ("P2 sub-a",          "r3r1k1/1bpnbppp/1p1p4/6P1/1P1N1P1P/1Q2B3/4BP2/R5K1 b - - 0 27"),
    ("P2 sub-b",          "r3r1k1/1bpnbppp/1p1p4/8/1P1N1P2/1Q2B1P1/4BP1P/R5K1 b - - 0 27"),
    ("P2b (Qg5+ attack)", "1rBq4/2p2pk1/3p2p1/3Pp2n/2p1P2r/1RN1N3/P1P2P1P/4R1K1 w - - 0 31"),
]
TERMS = ["King safety", "Threats", "Mobility", "Space", "Passed", "Material", "Total"]
sf = SF11Eval(SF11)
sf18 = chess.engine.SimpleEngine.popen_uci(find_stockfish())
def sf18s(fen):
    b=chess.Board(fen); info=sf18.analyse(b, chess.engine.Limit(depth=28)); sc=info["score"].white()
    mv=b.san(info["pv"][0]) if info.get("pv") else "?"
    return "%-6s %s" % (mv, ("mate %d"%sc.mate()) if sc.is_mate() else "%+.2f"%(sc.score()/100.0))
try:
    print("%-20s %s | %s" % ("case", " ".join("%10s"%t[:10] for t in TERMS), "SF18"))
    for lab, fen in CASES:
        _, terms = sf.eval(fen)
        row = " ".join("%+10.2f" % terms.get(t, 0.0) for t in TERMS)
        print("%-20s %s | %s" % (lab, row, sf18s(fen)))
finally:
    sf.close(); sf18.quit()
