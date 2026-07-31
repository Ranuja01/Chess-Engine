# -*- coding: utf-8 -*-
"""Side-by-side SF11-static vs OUR static term breakdown (WHITE-POV pawns) for FENs on argv, to find WHICH
feature SF11 credits that we miss/mistune on attack positions. SF11 terms are MG-Total column. Mapping
(per eval_vs_sf11 header): our king_safety~SF 'King safety', latent_threat~'Threats', passed_pawn_support~
'Passed', central~'Space', imbalance~'Imbalance', pieces~Knights/Bishops/Rooks/Queens placement.
Run: pyrun diagnostics/compare_terms.py "<fen>" [...]"""
import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ.setdefault('KS_SAFE_CHECK_DEF', '5')
THIS = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(THIS)); sys.path.insert(0, THIS)
import chess
from ChessAI import ChessAI
from eval_vs_sf11 import SF11Eval, SF11
ai = ChessAI(None, None, chess.Board(), True)
sf11 = SF11Eval(SF11)

def our_wpov(bd, key):
    return -bd.get(key, 0.0) / 1000.0   # abs black-positive -> white POV

fens = [a for a in sys.argv[1:] if "/" in a]
for fen in fens:
    bd = ai.ev_breakdown(chess.Board(fen))
    sf_tot, sf = sf11.eval(fen)
    our_tot = our_wpov(bd, "total")
    print("\n=== %s ===" % fen)
    print("  TOTAL      ours %+7.2f   SF11 %+7.2f   (SF11 - ours = %+.2f)" % (our_tot, sf_tot, sf_tot - our_tot))
    # aligned term pairs (our_key, sf_label)
    pairs = [("king_safety", "King safety"), ("latent_threat", "Threats"), ("passed_pawn_support", "Passed"),
             ("central", "Space"), ("imbalance_white", "Imbalance"), ("material", "Material")]
    print("  %-16s %8s %8s %8s" % ("term", "ours", "SF11", "SF-ours"))
    for ok, sl in pairs:
        o = our_wpov(bd, ok); s = sf.get(sl, 0.0)
        print("  %-16s %+8.2f %+8.2f %+8.2f" % (sl, o, s, s - o))
    # SF11's own biggest terms (what drives its eval)
    st = sorted(((t, v) for t, v in sf.items() if t != "Total" and abs(v) > 0.15), key=lambda kv: -abs(kv[1]))
    print("  SF11 top terms: " + "  ".join("%s=%+.2f" % (t, v) for t, v in st[:8]))
sf11.close()
