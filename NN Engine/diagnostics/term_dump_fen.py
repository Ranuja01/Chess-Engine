# -*- coding: utf-8 -*-
"""Per-term OURS vs SF11-static for a SINGLE fen (white-POV pawns) to localize an eval over/under-read.
  pyrun diagnostics/term_dump_fen.py "<fen>"
"""
import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'; os.environ.setdefault('KS_SAFE_CHECK_DEF', '5')
THIS = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(THIS)); sys.path.insert(0, THIS)
import chess
from ChessAI import ChessAI
from eval_vs_sf11 import SF11Eval, SF11
ai = ChessAI(None, None, chess.Board(), True); sf = SF11Eval(SF11)
fen = [a for a in sys.argv[1:] if "/" in a][0]
bd = ai.ev_breakdown(chess.Board(fen))
def owp(*ks): return -sum(bd.get(k, 0.0) for k in ks) / 1000.0
sf_total, sf_terms = sf.eval(fen)
print("fen:", fen)
print("our total %+.2f   SF11 total %+.2f   (white-POV pawns)\n" % (-bd.get("total", 0.0)/1000.0, sf_total))
PAIRS = [("Material+Kauf", ["material", "br_kaufman"], "Material"), ("KingSafety", ["king_safety"], "King safety"),
         ("Threats/latent", ["latent_threat"], "Threats"), ("Passed", ["passed_pawn_support"], "Passed"),
         ("Space/central", ["central", "det_central"], "Space"), ("Imbalance", ["imbalance_white", "imbalance_black"], "Imbalance"),
         ("CaptureGains", ["capture_gains"], None), ("pt_rooks", ["pt_rooks"], "Rooks"),
         ("pt_queens", ["pt_queens"], "Queens"), ("pt_knights", ["pt_knights"], "Knights"), ("pt_bishops", ["pt_bishops"], "Bishops")]
print("%-16s %8s %8s %8s" % ("term", "ours", "sf11", "diff"))
for label, ks, sflabel in PAIRS:
    o = owp(*ks); s = sf_terms.get(sflabel, 0.0) if sflabel else float('nan')
    print("%-16s %+8.2f %8s %8s" % (label, o, ("%+.2f" % s) if sflabel else "  n/a", ("%+.2f" % (o - s)) if sflabel else ""))
