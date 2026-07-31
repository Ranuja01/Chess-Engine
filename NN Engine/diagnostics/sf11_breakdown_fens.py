# -*- coding: utf-8 -*-
"""For a hand-picked FEN set: dump OUR per-term breakdown next to SF11-STATIC's FULL per-term breakdown
(so we can see how SF11's HCE reads it), plus the SF18-search engine's UCI IDENTITY + a depth-20 eval
(to verify what our 'SF18 truth' actually is vs e.g. lichess cloud). White-POV pawns throughout.
  pyrun diagnostics/sf11_breakdown_fens.py
"""
import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'; os.environ.setdefault('KS_SAFE_CHECK_DEF', '5')
THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(THIS)); sys.path.insert(0, THIS)
sys.path.insert(0, os.path.join(os.path.dirname(THIS), "selfplay"))
import chess, chess.engine
from ChessAI import ChessAI
from eval_vs_sf11 import SF11Eval, SF11
from arbiter import find_stockfish
ai = ChessAI(None, None, chess.Board(), True)
sf11 = SF11Eval(SF11)

FENS = [
    ("1 drawn-RP", "8/2r4k/8/6PK/6P1/8/1R6/8 w - - 7 59"),
    ("2 tricky-2hang", "r1bq1rk1/4npb1/6pp/1pppp3/QP1nP1P1/P1NPB3/3N1PBP/R3K2R w KQ - 0 16"),
    ("3 central-wall", "r3k3/pp2pp2/2p3p1/1q1p2b1/3P2n1/BP1N4/P1P1Kp2/Q2R1N2 b q - 1 21"),
    ("4 black-passers", "rr4k1/5pbp/3N1np1/3P4/5N2/pp3PPB/1n1B3P/4RK1R w - - 0 26"),
    ("5 up4-passers", "4k3/3qb1p1/2np3P/p3p3/Q3P3/p3B3/5P1K/5B2 b - - 0 32"),
    ("6 e4-promotes", "4B3/8/P3k3/2p5/2P1pp1P/1P2P3/3r4/1K6 w - - 0 60"),
]

OUR = [("mat", ["material"]), ("kauf", ["br_kaufman"]), ("place", ["pt_pawns", "pt_knights", "pt_bishops", "pt_rooks", "pt_queens"]),
       ("capg", ["capture_gains"]), ("pvb", ["piece_value_boost"]), ("space", ["central"]),
       ("threat", ["latent_threat"]), ("ks", ["king_safety"]), ("passed", ["passed_pawn_support"]),
       ("imb", ["imbalance_white", "imbalance_black"]), ("pair", ["pair_bonus"])]

sf18 = chess.engine.SimpleEngine.popen_uci(find_stockfish())
print("SF18 engine id: %s\n" % sf18.id.get("name", "?"))


def s18(fen, d):
    s = sf18.analyse(chess.Board(fen), chess.engine.Limit(depth=d))["score"].white()
    return "mate%d" % s.mate() if s.is_mate() else "%+.2f" % (s.score() / 100.0)


for label, fen in FENS:
    bd = ai.ev_breakdown(chess.Board(fen))
    our_tot = -bd.get("total", 0.0) / 1000.0
    sf_tot, sf_terms = sf11.eval(fen)
    print("=== %s ===" % label)
    print("  %s" % fen)
    print("  TOTALS (white-POV):  ours %+.2f   SF11-static %+.2f   SF18(d20) %s   SF18(d24) %s"
          % (our_tot, sf_tot, s18(fen, 20), s18(fen, 24)))
    ours = "  ".join("%s%+.2f" % (lab, -sum(bd.get(k, 0.0) for k in ks) / 1000.0) for lab, ks in OUR
                     if abs(sum(bd.get(k, 0.0) for k in ks)) >= 100)
    print("  OURS  : %s" % ours)
    sfl = "  ".join("%s%+.2f" % (k, v) for k, v in sf_terms.items() if abs(v) >= 0.05)
    print("  SF11  : %s\n" % sfl)
sf18.quit()
