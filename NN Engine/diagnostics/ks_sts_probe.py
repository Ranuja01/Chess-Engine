# -*- coding: utf-8 -*-
"""Probe the king-safety term on STS positions that REGRESSED when the KS recalibration
(KS_FLOOR=6 KS_SAFE_CHECK=8 KS_ATTACK_COUNT=2) was enabled. The recalibration improves KS accuracy
on the genuine-attack corpus but drops STS -88 on central/pawn maneuvering themes. Hypothesis: the
lowered floor + heavier weights wake the KS term on ORDINARY middlegame king pressure (units in the
old 6-13 deadzone) that the fit's validation tiers (calm controls + sharp-attack targets) never saw,
distorting move choice on quiet positional positions.

Sets the KS config from argv BEFORE importing the engine (Config is read once at import), so run twice:
  pyrun diagnostics/ks_sts_probe.py off
  pyrun diagnostics/ks_sts_probe.py on
Each prints, per FEN: raw pre-floor units (white/black king) and the resulting KS term (white-POV pawns).
"""
import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

mode = sys.argv[1] if len(sys.argv) > 1 else 'off'
if mode == 'on':
    os.environ['KS_FLOOR'] = '6'
    os.environ['KS_SAFE_CHECK'] = '8'
    os.environ['KS_ATTACK_COUNT'] = '2'
# off = engine defaults (KS_FLOOR=13, KS_SAFE_CHECK=3, KS_ATTACK_COUNT=1)

THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(THIS)); sys.path.insert(0, THIS)
import chess
from ChessAI import ChessAI
ai = ChessAI(None, None, chess.Board(), True)

# (idx, theme, fen) — the largest-swing STS regressors + a few gainers for contrast.
REGRESSORS = [
    (0,   "PawnCtr/Undermine", "1kr5/3n4/q3p2p/p2n2p1/PppB1P2/5BP1/1P2Q2P/3R2K1 w - - 0 1"),
    (52,  "KnightOutpost",     "8/1q6/3pn1k1/2p1p1p1/2P1P1Pp/1rBP1p1P/2Q2P2/R5K1 b - - 0 1"),
    (249, "PawnPlayCenter",    "3r2k1/1pp1q2p/p5pb/2n1p2r/5PQ1/P1PP3P/1P5K/1BBR1R2 w - - 0 1"),
    (225, "CenterControl",     "2r3k1/p2npp2/p2p1bp1/3P4/1qN1PQ1P/1P2B3/r5P1/1R3R1K b - - 0 1"),
    (0,   "Undermine.b",       "1kr5/3n4/q3p2p/p2n2p1/PppB1P2/5BP1/1P2Q2P/3R2K1 w - - 0 1"),
    (61,  "SquareVacancy",     "5r1k/5rp1/p1n1p1qp/2P1p3/P7/4QN1P/5PP1/2R1R2K w - - 0 1"),
    (244, "PawnPlayCenter",    "1rb2rk1/2q2pb1/3p1np1/1pnP2Bp/4P3/1pN2PN1/3QB1PP/1RR3K1 w - - 0 1"),
    (248, "PawnPlayCenter",    "2rrb1k1/1p3pp1/p1p4p/P5q1/2BP4/1P1RPQP1/5P2/3R1K2 w - - 0 1"),
    (122, "OfferSimpl",        "1r3rk1/1nqb1p1p/p3p1p1/1ppPb3/2P1N3/1P1Q2PP/P2B2BK/1RR5 w - - 0 1"),
    (138, "OfferSimpl",        "r3k1r1/2q1bp2/2p1p1np/p1Pp2P1/Pp1Pn2P/1P2P3/1B2NNP1/R3QRK1 b q - 0 1"),
    (221, "CenterControl",     "1n1rr1k1/1pq2pp1/3b2p1/2p3N1/P1P5/P3B2P/2Q2PP1/R2R2K1 w - - 0 1"),
    (256, "PawnPlayCenter",    "r2r2k1/p1q3pp/1p1npp2/2p5/P1PP4/B3PP2/6PP/2RQ1RK1 w - - 0 1"),
]
GAINERS = [
    (144, "AKPC.gain",         "r4rk1/1p2qppp/1np5/p2pNb2/P2Pn3/2NBP3/1PQ2PPP/2R2RK1 b - - 0 1"),
    (209, "KingActivity.gain", "3b4/2k2p2/2p1p1p1/pP1pP2p/P2P1P2/2P3P1/6K1/8 w - - 0 1"),
    (46,  "KnightOut.gain",    "r1bq1rk1/pp2ppbp/2np1np1/8/2PNP3/2N1B3/PP2BPPP/R2Q1RK1 w - - 0 1"),
]

print("=== KS probe mode=%s (FLOOR=%s SAFE_CHECK=%s ATTACK_COUNT=%s) ==="
      % (mode, os.environ.get('KS_FLOOR', '13'), os.environ.get('KS_SAFE_CHECK', '3'),
         os.environ.get('KS_ATTACK_COUNT', '1')))
print("%-5s %-18s %8s %8s %9s  (unitsW/B = raw pre-floor; KS = white-POV pawns)" %
      ("idx", "theme", "unitsW", "unitsB", "KS"))


def dump(label, rows):
    print("--- %s ---" % label)
    for idx, theme, fen in rows:
        b = chess.Board(fen)
        bd = ai.ev_breakdown(b)
        uw = bd.get("det_ks_units_w", 0.0); ub = bd.get("det_ks_units_b", 0.0)
        ks = -bd.get("king_safety", 0.0) / 1000.0
        print("%-5d %-18s %8.1f %8.1f %+9.2f  %s" % (idx, theme, uw, ub, ks, fen))


dump("REGRESSORS", REGRESSORS)
dump("GAINERS", GAINERS)
