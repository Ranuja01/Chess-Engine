# -*- coding: utf-8 -*-
"""KS-term triangulation: OUR king_safety term vs SF11's king-safety term, per FEN, so we can see whether OUR
magnitude balloons (relative to the classical reference) in specific position types (e.g. queenless middlegames).
Uses the existing triangulation pieces: ChessAI.ev_breakdown (ours) + SF11Eval (SF11 classical, the last
classical-KS Stockfish's same term). Also fires the live C++ KSD per-king dump (KS_DEBUG_DUMP, cpp_bitboard.cpp:5710)
so we see our raw units/danger, not the ks_explain PREVIEW / retired latent_threat field. Static eval, no search =>
single-core, no contamination. Usage: pyrun diagnostics/_ks_live.py [KSD=1] '<fen>' ['<fen>' ...]"""
import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
THIS = os.path.dirname(os.path.abspath(__file__)); ENGINE = os.path.dirname(THIS)
sys.path.insert(0, ENGINE); sys.path.insert(0, THIS)

args = [a for a in sys.argv[1:]]
if any(a == "KSD=1" for a in args):
    os.environ['KS_DEBUG_DUMP'] = '1'; args = [a for a in args if a != "KSD=1"]

import chess
from ChessAI import ChessAI
from eval_vs_sf11 import SF11Eval, SF11

sf11 = SF11Eval(SF11)


def our_ks(board):
    ai = ChessAI(None, None, board, board.turn)
    b = ai.ev_breakdown(board)   # dict, values in millipawns (int)
    v = b.get("king_safety") if isinstance(b, dict) else None
    return (v / 1000.0) if v is not None else None   # -> pawns, to compare with SF11's per-term (pawns)


def sf11_ks(fen):
    try:
        tot, terms = sf11.eval(fen)
    except Exception as e:
        return None, "err:%s" % e
    if terms is None:
        return None, "None(in-check?)"
    hit = [(k, v) for k, v in terms.items() if "king" in k.lower()]
    return (hit[0][1] if hit else None), (hit[0][0] if hit else "no-king-term")


print("%-70s %8s %8s %8s" % ("fen", "OUR_KS", "SF11_KS", "ratio"))
for fen in args:
    board = chess.Board(fen)
    o = our_ks(board)
    s, tag = sf11_ks(fen)
    r = ("%.2f" % (o / s)) if (o and s) else "-"
    print("%-70s %8s %8s %8s   (%s)" % (fen, ("%.2f" % o) if o is not None else "?",
                                        ("%.2f" % s) if s is not None else "?", r, tag))
    sys.stdout.flush()
sf11.close()
