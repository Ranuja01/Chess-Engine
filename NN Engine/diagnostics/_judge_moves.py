# -*- coding: utf-8 -*-
"""Judge specific candidate moves with SF18: is the move OUR engine picked actually worse than the alternative?

`_pick_probe.py` says WHICH move we choose; it cannot say whether choosing it was wrong. This scores each
named candidate by playing it and searching the resulting position with SF18, then reports the loss against
SF18's own best move. All scores are side-to-move POV in pawns, so positive = good for the side to move.

  pyrun diagnostics/_judge_moves.py <fens-file> <label>:<uci>,<uci>... [<label>:...] [DEPTH=20]
"""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
import sys
import chess
import chess.engine

DEPTH = int(os.environ.get('DEPTH', '20'))
args = []
for a in sys.argv[1:]:
    if a.startswith('DEPTH='):
        DEPTH = int(a.split('=', 1)[1])
    else:
        args.append(a)

path, cands = args[0], {}
for spec in args[1:]:
    if ':' in spec:
        lbl, moves = spec.split(':', 1)
        cands[lbl] = moves.split(',')

fens = {}
with open(path) as fh:
    for line in fh:
        if '\t' in line:
            lbl, fen = line.rstrip('\n').split('\t', 1)
            fens[lbl] = fen

eng = chess.engine.SimpleEngine.popen_uci(os.environ["STOCKFISH_PATH"])


def score_stm(info, board):
    """SF score in pawns from the point of view of the side to move in `board`."""
    return info["score"].pov(board.turn).score(mate_score=100000) / 100.0


try:
    for lbl, fen in fens.items():
        if lbl not in cands:
            continue
        board = chess.Board(fen)
        best = eng.analyse(board, chess.engine.Limit(depth=DEPTH))
        best_mv = best["pv"][0]
        best_sc = score_stm(best, board)
        print("\n%s   [%s]" % (lbl, fen))
        print("  SF18 best  %-6s  %+.2f" % (best_mv.uci(), best_sc))
        for uci in cands[lbl]:
            try:
                mv = chess.Move.from_uci(uci)
                if mv not in board.legal_moves:
                    print("  %-10s ILLEGAL" % uci)
                    continue
            except Exception:
                print("  %-10s UNPARSEABLE" % uci)
                continue
            board.push(mv)
            info = eng.analyse(board, chess.engine.Limit(depth=DEPTH))
            # after pushing, side to move flipped: negate to get the mover's POV
            sc = -score_stm(info, board)
            board.pop()
            print("  %-10s %+.2f   (loss %+.2f vs best)" % (uci, sc, sc - best_sc))
finally:
    eng.quit()
