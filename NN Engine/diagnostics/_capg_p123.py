# -*- coding: utf-8 -*-
"""P1: SF11 full static term story. P2: SF11 static + SF18 search for hanging-rook (a1) vs safe-rook (c1),
to test whether SF accounts for the en-prise rook. P3: SF18 search to confirm the mating attack (SF11 static
can't see mate). All SF scores in WHITE POV (pawns), matching lichess convention."""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
import sys
THIS = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(THIS)); sys.path.insert(0, THIS)
import chess, chess.engine
from eval_vs_sf11 import SF11Eval, SF11
sys.path.insert(0, os.path.join(os.path.dirname(THIS), "selfplay"))
from arbiter import find_stockfish

P1  = "1rbq1rk1/ppp2pb1/7p/2n1pnpP/4Q3/2NP1NP1/PPPB1PB1/2K1R2R w - - 2 15"
P2o = "r3r1k1/3nbppp/1p1pb3/3p2P1/1P1N1P1P/1Q2B3/4BP2/R5K1 b - - 0 27"       # rook a1 (hanging)
P2m = "r3r1k1/3nbppp/1p1pb3/3p2P1/1P1N1P1P/1Q2B3/4BP2/2R3K1 b - - 0 27"      # rook c1 (safe)
P3  = "1r6/2r1P3/2bR4/2P2kN1/7P/2n2PK1/6P1/1q2R3 w - - 1 49"

sf = SF11Eval(SF11)
sf18 = chess.engine.SimpleEngine.popen_uci(find_stockfish())

def sf11_terms(fen):
    tot, terms = sf.eval(fen)  # White-POV
    return tot, terms

def sf18_search(fen, depth=26):
    b = chess.Board(fen)
    info = sf18.analyse(b, chess.engine.Limit(depth=depth))
    sc = info["score"].white()
    mv = b.san(info["pv"][0]) if info.get("pv") else "?"
    if sc.is_mate():
        return "%s (mate in %d)" % (mv, sc.mate())
    return "%s (%+.2f)" % (mv, sc.score()/100.0)

try:
    print("===== P1 tempo-motivator: SF11 static term story (White-POV) =====")
    tot, terms = sf11_terms(P1)
    for k, v in sorted(terms.items(), key=lambda kv: -abs(kv[1])):
        if abs(v) >= 0.01 or k == "Total":
            print("  %-14s %+6.2f" % (k, v))
    print("  SF18 search:", sf18_search(P1))

    print("\n===== P2 hanging-rook A/B (White-POV) =====")
    for lab, fen in [("a1 HANGING", P2o), ("c1 SAFE", P2m)]:
        tot, terms = sf11_terms(fen)
        print("  [%s] SF11 static Total=%+.2f  Material=%+.2f  Threats=%+.2f  |  SF18=%s" % (
            lab, terms.get("Total", tot), terms.get("Material", 0.0), terms.get("Threats", 0.0), sf18_search(fen)))

    print("\n===== P3 imbalance-driven: is it really a White attack/mate? =====")
    tot, terms = sf11_terms(P3)
    print("  SF11 static Total=%+.2f  King safety=%+.2f  Threats=%+.2f  Material=%+.2f" % (
        terms.get("Total", tot), terms.get("King safety", 0.0), terms.get("Threats", 0.0), terms.get("Material", 0.0)))
    print("  SF18 search:", sf18_search(P3, depth=30))
finally:
    sf.close(); sf18.quit()
