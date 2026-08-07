# -*- coding: utf-8 -*-
"""Print the CAPG_TRACE capture sequence for a position AND its mirror, side by side.

The capture-gains rank bonus applies in one mirror orientation and contributes nothing in the other
(base `capture_gains` tracks CAPG_PAWN_RANK_CLAMP exactly, mirror sits at the bare pawn value at every
clamp setting). Three separate readings of the source produced three wrong explanations, so this stops
reading and prints the actual per-capture inputs: which capture was chosen, the piece types the branch
tests, whether it fired, and the resulting prb.

Squares are printed as names so the two orientations can be compared by eye -- under a correct mirror
every line should pair up with its rank-flipped twin and identical `fired`/|prb|.

  CAPG_TRACE=1 pyrun diagnostics/_capg_trace_pair.py FEN=<board-only> [TURN=w|b]
"""
import os, sys

for _a in sys.argv[1:]:
    if '=' in _a:
        _k, _v = _a.split('=', 1)
        os.environ[_k] = _v

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
os.environ['CAPG_TRACE'] = '1'
THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(THIS)); sys.path.insert(0, THIS)
import chess
from ChessAI import ChessAI

FEN = os.environ.get("FEN", "8/6k1/1Rp5/8/8/4p3/5P2/4K3")
PT = {0: ".", 1: "P", 2: "N", 3: "B", 4: "R", 5: "Q", 6: "K"}


def main():
    ai = ChessAI(None, None, chess.Board(), True)
    b = chess.Board(FEN)
    if os.environ.get("TURN"):
        b.turn = (os.environ["TURN"].strip().lower() == "w")
    m = b.mirror()

    for lbl, board in (("BASE  ", b), ("MIRROR", board_m := m)):
        d = ai.ev_breakdown(board)
        sys.stderr.flush()
        print("\n=== %s  %s" % (lbl, board.fen()))
        print("    capture_gains=%s  material=%s" % (d.get("capture_gains"), d.get("material")))
    print("\nThe CAPG lines above are on stderr, interleaved in call order.")
    print("Read them as: side from->to  capturedType/moverType  fired  prb")
    print("Under a correct mirror each BASE line must have a rank-flipped MIRROR twin with the same")
    print("`fired` and the same |prb|. A line that fires on one side only IS the defect.")


if __name__ == "__main__":
    main()
