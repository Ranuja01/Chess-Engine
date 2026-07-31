# -*- coding: utf-8 -*-
"""Side-by-side over-read dossier for human eyeballing: per FEN, our total + term breakdown vs SF11-static total +
term breakdown vs SF18-search. Aligned terms (our -> SF11 label). All WHITE-POV pawns (+ = White better).
NOTE: our 'pieces' breakdown key is a CUMULATIVE snapshot (includes material) -> NOT shown as a term; use the
clean per-term deltas below. Run: pyrun diagnostics/dossier_overread.py "<fen>" [...]"""
import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'; os.environ.setdefault('KS_SAFE_CHECK_DEF', '5')
THIS = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(THIS)); sys.path.insert(0, THIS)
import chess, chess.engine
from ChessAI import ChessAI
from eval_vs_sf11 import SF11Eval, SF11
sys.path.insert(0, os.path.join(os.path.dirname(THIS), "selfplay"))
from arbiter import find_stockfish
ai = ChessAI(None, None, chess.Board(), True)
sf11 = SF11Eval(SF11)
sf18 = chess.engine.SimpleEngine.popen_uci(find_stockfish())

def owp(bd, *keys): return -sum(bd.get(k, 0.0) for k in keys) / 1000.0    # our black-pos milli -> white-POV pawns
def s18(fen):
    b = chess.Board(fen); i = sf18.analyse(b, chess.engine.Limit(depth=20)); s = i["score"].white()
    return 99.0 if (s.is_mate() and s.mate() > 0) else (-99.0 if s.is_mate() else s.score() / 100.0)

# aligned: label, our-keys, SF11-label
PAIRS = [("Material",   ["material", "br_kaufman"], "Material"),
         ("Knights",    ["pt_knights"],  "Knights"),
         ("Bishops",    ["pt_bishops"],  "Bishops"),
         ("Rooks",      ["pt_rooks"],    "Rooks"),
         ("Queens",     ["pt_queens"],   "Queens"),
         ("KingSafety", ["king_safety"], "King safety"),
         ("Threats",    ["latent_threat"], "Threats"),
         ("Passed",     ["passed_pawn_support"], "Passed"),
         ("Space",      ["central", "det_central"], "Space"),
         ("Imbalance",  ["imbalance_white", "imbalance_black"], "Imbalance")]

for fen in [a for a in sys.argv[1:] if "/" in a]:
    b = chess.Board(fen); bd = ai.ev_breakdown(b); sf_tot, st = sf11.eval(fen); srch = s18(fen)
    print("\n" + "=" * 78)
    print("FEN: %s  (%s to move)" % (fen, "White" if b.turn else "Black"))
    print("  OUR total = %+.2f    SF11-static = %+.2f    SF18-search = %+.2f   (WHITE-POV pawns)" % (
        owp(bd, "total"), sf_tot, srch))
    print("  %-12s %10s %10s %10s" % ("term", "OURS", "SF11", "OURS-SF11"))
    for label, okeys, sflabel in PAIRS:
        o = owp(bd, *okeys); s = st.get(sflabel, 0.0)
        print("  %-12s %+10.2f %+10.2f %+10.2f" % (label, o, s, o - s))
    # SF11 terms we have no clean analogue for
    print("  SF11-only  : " + "  ".join("%s=%+.2f" % (t, st[t]) for t in ("Mobility", "Space") if t in st))
sf11.close(); sf18.quit()
