# -*- coding: utf-8 -*-
"""P1 original (Qe4, attacked by Nc5) vs modified (Qe2, safe): does our capgains distinguish them, or credit
the e5 pawn in both? Dumps our material/capg/total (pin ON) + the capgains capture list + SF11 static + SF18.
All our-eval numbers White-POV (side to move is White both)."""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
os.environ['ENABLE_CAPG_PIN'] = '1'
os.environ['CAPG_DEBUG_DUMP'] = '1'
import sys
THIS = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(THIS)); sys.path.insert(0, THIS)
import chess, chess.engine
from eval_vs_sf11 import SF11Eval, SF11
sys.path.insert(0, os.path.join(os.path.dirname(THIS), "selfplay"))
from arbiter import find_stockfish

ORIG = "1rbq1rk1/ppp2pb1/7p/2n1pnpP/4Q3/2NP1NP1/PPPB1PB1/2K1R2R w - - 2 15"   # Qe4 (attacked by Nc5)
MOD  = "1rbq1rk1/ppp2pb1/7p/2n1pnpP/8/2NP1NP1/PPPBQPB1/2K1R2R w - - 2 15"      # Qe2 (safe)

ai_mod = None
from ChessAI import ChessAI
ai = ChessAI(None, None, chess.Board(), True)
sf = SF11Eval(SF11)
sf18 = chess.engine.SimpleEngine.popen_uci(find_stockfish())

def sf18s(fen):
    b=chess.Board(fen); info=sf18.analyse(b, chess.engine.Limit(depth=26)); sc=info["score"].white()
    mv=b.san(info["pv"][0]) if info.get("pv") else "?"
    return "%s %s" % (mv, ("mate %d"%sc.mate()) if sc.is_mate() else "%+.2f"%(sc.score()/100.0))

try:
    for lab, fen in [("ORIG Qe4(attacked)", ORIG), ("MOD  Qe2(safe)", MOD)]:
        print("\n===== %s =====" % lab)
        sys.stderr.flush(); sys.stdout.flush()
        bd = ai.ev_breakdown(b := chess.Board(fen)); sys.stderr.flush()
        print("  OUR (White-POV): material=%+.2f capg=%+.2f total=%+.2f" % (
            -bd.get('material',0)/1000.0, -bd.get('capture_gains',0)/1000.0, -bd.get('total',0)/1000.0))
        print("  SF11 static Total=%+.2f | SF18 search=%s" % (sf.eval(fen)[1].get('Total',0.0), sf18s(fen)))
finally:
    sf.close(); sf18.quit()
