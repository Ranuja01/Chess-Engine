# -*- coding: utf-8 -*-
"""Is a flagged endgame passer below the deferral threshold paid TWICE?

`evaluate_pawns_endgame` defers (and discards) the inline rank bonus only when `ppIncrement >= 300`, but
`getPPIncrement` flags a pawn as passed well below that -- base 200, minus PP_BLOCKADE_PEN 100 for a minor
on the stop square = 100. Such a pawn takes the `else` branch and adds `3 * default_midgame_pawn_rank_bonus`
inline, while `evaluate_passers` independently pays every pawn in the passed bitboard.

If that reading is right, a blockaded endgame passer shows BOTH a non-zero `passed_pawn_support` (the
evaluate_passers payment) and an inflated `pt_pawns` (the inline payment) relative to the same position with
the pawn absent -- and the inline part should vanish under ENABLE_PASSER_V3=0, where nothing is deferred.

  pyrun diagnostics/_endgame_passer_doublepay.py
"""
import os, sys

for _a in sys.argv[1:]:
    if '=' in _a:
        _k, _v = _a.split('=', 1)
        os.environ[_k] = _v

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(THIS)); sys.path.insert(0, THIS)
import chess
from ChessAI import ChessAI

# White pawn d5 (rank 5), black KNIGHT on d6 = a secure minor blockade on the stop square.
# No black pawns anywhere => the pawn IS flagged passed, yet ppIncrement = 200 - PP_BLOCKADE_PEN = 100,
# far below the endgame threshold of 300. Deep endgame so the endgame evaluator runs.
CASES = [
    ("blockaded passer d5 (minor on d6)", "7k/8/3n4/3P4/8/8/8/7K w - - 0 1", chess.D5),
    ("free passer d5 (nothing ahead)",    "7k/8/8/3P4/8/8/8/7K w - - 0 1",   chess.D5),
    ("free passer d6 (more advanced)",    "7k/8/3P4/8/8/8/8/7K w - - 0 1",   chess.D6),
]


def main():
    ai = ChessAI(None, None, chess.Board(), True)
    print("Endgame passer payment attribution (White-POV cp; our eval is Black-positive so values negated).")
    print("`pawn_delta` = pt_pawns WITH the pawn minus WITHOUT it -> the INLINE payment.")
    print("`passer_delta` = passed_pawn_support delta -> the evaluate_passers payment.\n")
    print("  %-38s %10s %12s %12s %10s" % ("case", "phase", "pawn_delta", "passer_delta", "total"))
    for label, fen, sq in CASES:
        b = chess.Board(fen)
        wo = b.copy(); wo.remove_piece_at(sq)
        a, c = ai.ev_breakdown(wo), ai.ev_breakdown(b)
        pawn_d = -(c["pt_pawns"] - a["pt_pawns"]) / 10.0
        pass_d = -(c["passed_pawn_support"] - a["passed_pawn_support"]) / 10.0
        tot_d = -(c["total"] - a["total"]) / 10.0
        print("  %-38s %10d %12.1f %12.1f %10.1f"
              % (label, c["phase_score"], pawn_d, pass_d, tot_d))
    print("\n  If BOTH pawn_delta and passer_delta are non-zero for the blockaded case, the inline bonus and")
    print("  evaluate_passers are both paying for the same pawn.")


if __name__ == "__main__":
    main()
