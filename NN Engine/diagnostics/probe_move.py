"""
Ask the engine for its move in one position, under an arbitrary knob set.

Usage: pyrun diagnostics/probe_move.py FEN="<fen>" [KEY=VAL ...]

KEY=VAL arguments are applied to os.environ BEFORE the extension is imported -- Config latches at
extension init, and the runner's pyrun sub forwards argv rather than env, so this is the only way to
set knobs through it. One process per knob setting.
"""
import os
import sys
import chess
from timeit import default_timer as timer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

fen = None
for arg in sys.argv[1:]:
    if "=" not in arg:
        continue
    k, v = arg.split("=", 1)
    if k == "FEN":
        fen = v
    else:
        os.environ[k] = v

if fen is None:
    print("need FEN=<fen>")
    raise SystemExit(1)

board = chess.Board(fen)
print("FEN  :", fen)
print("turn :", "black" if board.turn == chess.BLACK else "white", " legal:", len(list(board.legal_moves)))

from ChessAI import ChessAI

ai = ChessAI(None, None, board, board.turn)
t0 = timer()
move = ai.alphaBetaWrapper()
dt = timer() - t0

print("RESULT chosen=%s san=%s time=%.2fs" % (
    move.uci() if move else "none",
    board.san(move) if move else "-",
    dt,
))
