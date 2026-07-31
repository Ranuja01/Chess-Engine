# -*- coding: utf-8 -*-
"""Classic full diagnostic for the pt_pawns class: our full eval-term breakdown vs SF11 full term table vs
SF18-search, on the deferred 'fen 3' (passed-pawn / immobile-piece position) + the 3 worst pt_pawns collapses.
All our-POV, pawns."""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
os.environ['KS_SAFE_CHECK_DEF'] = '5'
import sys
THIS = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(THIS)); sys.path.insert(0, THIS)
import chess, chess.engine
from ChessAI import ChessAI
from eval_vs_sf11 import SF11Eval, SF11
sys.path.insert(0, os.path.join(os.path.dirname(THIS), "selfplay"))
from arbiter import find_stockfish

FENS = [
    ("DEFERRED fen3", "rn6/5p2/pBp1pk2/P4p2/1p5b/5B1P/1P2K3/3R4 b - - 1 39"),
    ("ptp WORST",     "r4rk1/2R4p/2pN1p2/1p4p1/p2p3P/5bP1/8/4R1K1 b - - 1 30"),
    ("ptp 2nd",       "1k2r3/p7/8/4PQ2/3PK3/P3P3/2q2P1P/5B2 w - - 1 40"),
    ("ptp 3rd",       "3r4/pp6/3k1p1p/3rp1b1/P1Rp2p1/3B4/2K3PP/4BR2 b - - 2 31"),
]
OUR = ["material", "capture_gains", "pieces", "pt_pawns", "pt_knights", "pt_bishops", "pt_rooks",
       "passed_pawn_support", "piece_value_boost", "imbalance_white", "king_safety", "mobility", "total"]
ai = ChessAI(None, None, chess.Board(), True)
sf = SF11Eval(SF11); sf18 = chess.engine.SimpleEngine.popen_uci(find_stockfish())
def sf18s(fen):
    b = chess.Board(fen); info = sf18.analyse(b, chess.engine.Limit(depth=26)); s = info["score"].white()
    pov = 1.0 if b.turn else -1.0; mv = b.san(info["pv"][0]) if info.get("pv") else "?"
    return "%s %s" % (mv, ("M%d"%s.mate()) if s.is_mate() else "%+.2f"%(s.score()/100.0*pov))
try:
    for lab, fen in FENS:
        b = chess.Board(fen); pov = 1.0 if b.turn else -1.0
        bd = ai.ev_breakdown(b)
        print("\n===== %s  (%s to move) =====" % (lab, "White" if b.turn else "Black"))
        print("  FEN:", fen)
        print("  OURS:", "  ".join("%s=%+.2f" % (t, (-bd.get(t,0.0)/1000.0)*pov) for t in OUR if abs(bd.get(t,0.0))>1))
        _, terms = sf.eval(fen)
        print("  SF11:", "  ".join("%s=%+.2f" % (k, v*pov) for k,v in sorted(terms.items(), key=lambda x:-abs(x[1])) if abs(v)>=0.05 and k!="Total"))
        print("  SF11 Total=%+.2f | SF18 search=%s" % (terms.get("Total",0.0)*pov, sf18s(fen)))
finally:
    sf.close(); sf18.quit()
