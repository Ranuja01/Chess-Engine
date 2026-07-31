# -*- coding: utf-8 -*-
"""Answer the LOGIC question: on the SF-quiet over-fire positions (SF11-static stays silent, we fire),
is our KS driven by PROXIMITY (attacked_zone_squares + attacker presence) with NO realizable safe-check,
whereas genuine attacks carry safe-checks? Uses KS_DEBUG_DUMP (per-king KSD line: attsq/weak/safe/attpc).
Counts are config-independent (raw geometry), so defaults are fine.

  pyrun diagnostics/ks_logic_probe.py
Emits, per FEN, a '### label' marker then the C++ 'KSD W/B ...' lines (all on stderr, ordered).
"""
import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ.setdefault('KS_SAFE_CHECK_DEF', '5')
os.environ['KS_DEBUG_DUMP'] = '1'
os.environ['KS_FLOOR'] = '0'
THIS = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(THIS)); sys.path.insert(0, THIS)
import chess
from ChessAI import ChessAI
ai = ChessAI(None, None, chess.Board(), True)

OVERFIRE = [   # SF11-static QUIET, we fire, STS regressed (from ks_overfire_vs_sf11.py)
    "1kr5/3n4/q3p2p/p2n2p1/PppB1P2/5BP1/1P2Q2P/3R2K1 w - - 0 1",
    "5r1k/5rp1/p1n1p1qp/2P1p3/P7/4QN1P/5PP1/2R1R2K w - - 0 1",
    "2n2rk1/p2b2bp/3p2p1/q1pPp2n/P1P1P3/1QN5/3BBNPP/1R4K1 w - - 0 1",
    "2r2r1k/1q1nbpp1/p3p2p/2P1P3/1p1N1B1P/6Q1/PP3PP1/R2R2K1 b - - 0 1",
    "r3k1r1/2q1bp2/2p1p1np/p1Pp2P1/Pp1Pn2P/1P2P3/1B2NNP1/R3QRK1 b q - 0 1",
    "4r1k1/p5b1/P2p1pp1/q1pP3p/2Pn1BbP/2NP2P1/3Q2BK/1R6 w - - 0 1",
    "1r1r2k1/4pp2/1PR3p1/p2P1n1p/1p1q1Q2/5P1P/2B1RPPK/8 w - - 0 1",
    "2b5/2p1r2k/1pP2q1p/p2Pp3/4R3/1PN1Q2P/P2KP3/8 w - - 0 1",
    "6r1/r2Nbnk1/2R3pp/p4p2/PpB2PP1/1P6/6K1/3R4 w - - 0 1",
    "r3r3/2P4k/3Bbbqp/ppQ2pp1/4pPP1/1P6/P1R2N1P/3R2K1 w - - 0 1",
    "r4rk1/ppp3b1/3p1q1p/3Ppn2/P1P3n1/2NQ1N2/1P1B1PP1/R3R1K1 w - - 0 1",
    "2r1r1k1/5npp/3q4/1QpP1p2/1p6/4PP2/1B2R1PP/2R3K1 w - - 0 1",
]
ATTACK = [   # SF11-static FIRES (tier=attack) — genuine danger
    "3r1bkr/2q3pp/1p1Npp2/pPn1P3/5B2/1P6/2P2PPP/R2QR1K1 w - - 0 1",
    "1b1r4/3rkp2/p3p2p/4q3/P5P1/2RBP3/P1Q4P/1R3K2 b - - 0 1",
    "2r1q2k/7p/p1np1P1P/8/1pP2R2/8/PP1Q4/R1KN2r1 b - - 0 1",
    "3r4/3pkpp1/4p3/2p3q1/6r1/1P6/P5B1/2R1RQK1 b - - 0 1",
]


def run(label, fens):
    for fen in fens:
        sys.stderr.write("### %s  %s\n" % (label, fen)); sys.stderr.flush()
        ai.ev_breakdown(chess.Board(fen))
        sys.stderr.flush()


run("OVERFIRE", OVERFIRE)
run("ATTACK", ATTACK)
