#!/usr/bin/env python3
"""Full per-term static breakdown dump for one or more FENs (our eval), incl. the attack-layer
offense/defense/mobility/pieceval detector split for White vs Black. Compare against SF11's own
per-term table (printf 'position fen ...\\neval\\nquit' | stockfish_11.exe) to localize over/under-credit.
Set eval env knobs (KING_SAFETY_MAG, ROOK_*, SCALE_*) BEFORE first ChessAI() to test a config.
Usage: [env knobs] python eval_dump.py '<fen>' ['<fen2>' ...]
"""
import os, sys, chess
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(BASE); sys.path.insert(0, BASE)
from ChessAI import ChessAI
b0 = chess.Board()
ai = ChessAI(None, None, b0, b0.turn)

MAIN = ["material", "pieces", "pt_pawns", "pt_knights", "pt_bishops", "pt_rooks", "pt_queens", "pt_kings",
        "central", "capture_gains", "latent_threat", "king_safety", "imbalance_white", "imbalance_black",
        "pair_bonus", "piece_value_boost", "mobility", "outpost", "pawn_struct", "pawn_majority",
        "passed_pawn_support", "space", "rook_cond"]
DET = ["det_w_offense", "det_b_offense", "det_w_defense", "det_b_defense",
       "det_w_mobility", "det_b_mobility", "det_w_pieceval", "det_b_pieceval",
       "det_central", "det_ks_units_w", "det_ks_units_b", "det_pawn_count"]

for fen in sys.argv[1:]:
    bd = ai.ev_breakdown(chess.Board(fen))
    print("\n==== %s" % fen)
    print("  TOTAL white_pawns = %+.3f   phase=%s" % (-bd["total"] / 1000.0, bd.get("phase_score")))
    print("  -- main terms (White-POV pawns, + favours White) --")
    for t in MAIN:
        if t in bd and bd[t] != 0:
            print("     %-20s %+.3f" % (t, -bd[t] / 1000.0))
    print("  -- detector raw (attack-layer offense/defense; higher = that side credited more) --")
    for t in DET:
        if t in bd:
            print("     %-20s %s" % (t, bd[t]))
