# -*- coding: utf-8 -*-
"""King-safety detection TRACE — attribute the under-fire gap to specific components.

For each FEN where SF11 sees a king attack we read ~0 (`eval_vs_sf11.py` under-fire list), this prints
side by side:
  - what the ENGINE currently registers: `det_ks_units_w/b` (live C++ king_safety_danger raw units,
    captured when the KS block runs) + the resulting `king_safety` term, with the m4000ctl anchor ON;
  - what the FULL wired model WOULD register: the per-component preview from `ks_explain.components_for_king`
    (king_ring2 zone + the currently-DEAD `KS_WEAK`/`KS_STORM` terms), broken out by component.

The DELTA between the two is the dead-knob / small-zone contribution the engine fails to see — i.e. it
names which component (weak squares, pawn storm, wider zone) accounts for the missed danger. Reuses the
existing trace surfaces; it does not reimplement king-danger detection.

Run in WSL from NN Engine/ (no SF11 needed — FENs are passed in or defaulted):
    /home/ranuja/anaconda3/bin/python diagnostics/ks_trace.py            # the 8 default under-fire FENs
    /home/ranuja/anaconda3/bin/python diagnostics/ks_trace.py "<fen>" ...
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# Anchor ON so the C++ KS block actually runs and det_ks_units_w/b populate (else gated off => 0).
os.environ["ENABLE_KS_REPLACE_LT"] = "1"
os.environ["KING_SAFETY_MAG"] = "4000"
os.environ["MOD_KS_CONTROL"] = "256"
import sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ENGINE_DIR = os.path.dirname(THIS_DIR)
sys.path.insert(0, ENGINE_DIR)
sys.path.insert(0, THIS_DIR)

import chess  # noqa: E402
from ks_explain import components_for_king  # noqa: E402

# The 8 worst KS-under-fire positions (eval_vs_sf11.py 400 anchor), with SF11's White-POV King-safety read.
DEFAULT = [
    ("r1k5/4qp2/P1p1b1p1/6Pp/n3P2P/6Q1/2N5/1K1R2R1 b - - 0 1", -7.62),
    ("2r5/p4p1k/1p5p/2qPnN2/P2R4/1P4Pp/7P/3Q2K1 b - - 0 1", -4.52),
    ("r1b1rnk1/pp3pq1/2p3p1/6P1/2B2P1R/2P5/PP1Q2P1/2K4R w - - 0 1", 4.78),
    ("r1br1k2/1pq2pb1/1np1p1pp/2N1N3/p2P1P1P/P3P1R1/1PQ3P1/1BR3K1 w - - 0 1", 4.70),
    ("2r4k/pp3q1b/5PpQ/3p4/3Bp3/1P6/P5RP/6K1 w - - 0 1", 4.41),
    ("r2q2kr/p3n3/1p1Bp1bp/3pP1pN/5PPn/3B3Q/2P2K2/1R5R w - - 0 1", 4.12),
    ("1rb2rk1/2p3pp/p1p1p3/2N5/8/1PQ2PPq/P3P3/R2R2K1 w - - 0 1", -4.57),
    ("2b1k2r/5p2/pq1pNp1b/1p6/2r1PPBp/3Q4/PPP3PP/1K1RR3 w k - 0 1", 3.38),
]


def comp_line(c):
    a = c["att_by_type"]
    return ("N%d B%d R%d Q%d | atk %d weak %d safe %d storm %d open %d shield %d | UNITS %d"
            % (a[chess.KNIGHT], a[chess.BISHOP], a[chess.ROOK], a[chess.QUEEN],
               c["attacked_zone_squares"], c["weak_squares"], c["safe_checks"], c["storm"],
               c["open_files"], c["shield"], c["units"]))


def main():
    args = sys.argv[1:]
    cases = [(f, None) for f in args] if args else DEFAULT
    from ChessAI import ChessAI
    seed = chess.Board()
    ai = ChessAI(None, None, seed, seed.turn)
    for fen, sf_ks in cases:
        board = chess.Board(fen)
        bd = ai.ev_breakdown(board)
        eu_w, eu_b = bd.get("det_ks_units_w", 0), bd.get("det_ks_units_b", 0)
        ks_term = -bd.get("king_safety", 0) / 1000.0  # White-POV pawns
        cw = components_for_king(board, chess.WHITE)
        cb = components_for_king(board, chess.BLACK)
        # the under-fire king = the one SF11 flags (negative SF_KS => White king dangerous, favours Black)
        print("=" * 100)
        hdr = "SF11_KS=%+.2f  " % sf_ks if sf_ks is not None else ""
        print("%s%s" % (hdr, fen))
        print("  ENGINE (anchor ON): det_units W=%d B=%d   king_safety term=%+.2f (White-POV)" % (eu_w, eu_b, ks_term))
        print("  MODEL White-king : %s" % comp_line(cw))
        print("  MODEL Black-king : %s" % comp_line(cb))
    return 0


if __name__ == "__main__":
    sys.exit(main())
