"""
Search probe for the 2026-07-29 SF18 loss (see dev_notes/observed-loss-2026-07-29-sf18-pin.md).

Position after 41. Ra6: Ra6/Be6/Kf6 share the sixth rank, so the bishop is ABSOLUTELY PINNED and falls
to 42. Rbb6. The engine played 41...Rf4 (neither unpins nor defends) and lost the piece by force.

Question this answers: does the engine ever pick an unpinning king move, and does root razoring change
the answer? Run once per knob setting -- Config latches at extension init, so one process per arm.
"""
import os
import sys
import chess
from timeit import default_timer as timer

# The compiled extension lives in the NN Engine root, not in diagnostics/.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FEN = "8/8/R3bk1p/4p3/p6r/8/2P3P1/1R4K1 b - - 3 41"
SAVING = {"f6g7", "f6f7"}   # the two moves that break the pin
PLAYED = "h4f4"             # 41...Rf4, what it chose in the game

board = chess.Board(FEN)
print("FEN:", FEN)
print("bishop e6 absolutely pinned:", board.is_pinned(chess.BLACK, chess.E6))
print("legal moves:", len(list(board.legal_moves)))

# Accept KEY=VAL arguments and apply them BEFORE importing the extension: Config latches at extension
# init, so anything set afterwards is ignored. The runner's pyrun sub forwards argv rather than env,
# which is why this is done here instead of on the command line.
for arg in sys.argv[1:]:
    if "=" in arg:
        k, v = arg.split("=", 1)
        os.environ[k] = v
        print("knob:", k, "=", v)

from ChessAI import ChessAI

ai = ChessAI(None, None, board, board.turn)

t0 = timer()
move = ai.alphaBetaWrapper()
elapsed = timer() - t0

uci = move.uci() if move is not None else "(none)"
print("\n===== RESULT =====")
print("chosen      :", uci, "  san:", board.san(move) if move is not None else "-")
print("elapsed     : %.2fs" % elapsed)
print("unpins?     :", "YES" if uci in SAVING else "NO")
print("same as game:", "YES (reproduced 41...Rf4)" if uci == PLAYED else "no")
