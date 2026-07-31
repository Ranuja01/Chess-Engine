"""
Ask Stockfish for its verdict on the two critical positions of the 2026-07-30 SF11 loss.

Establishes whether 29...a4 is actually an error before it is banked as a failure case -- our engine
chooses it cold at both baseline and the gravity candidate, so "we played it" is not evidence it is wrong.
STOCKFISH_PATH is exported by the runner's pyrun sub.
"""
import os
import chess
import chess.engine

SF = os.environ.get("STOCKFISH_PATH")
print("stockfish:", SF)

POSITIONS = [
    ("move 29 (we play a4)",   "1r2k1r1/4bR2/4p3/p3N2p/3Pb1p1/1P4P1/P3K2P/2R5 b - - 1 29", "a5a4"),
    ("move 32 (we play Rxa4)", "r3k1r1/2R2R2/4p3/4N1bp/P2Pb1p1/6P1/P3K2P/8 b - - 4 32",     "a8a4"),
]

with chess.engine.SimpleEngine.popen_uci(SF) as eng:
    for label, fen, ours in POSITIONS:
        board = chess.Board(fen)
        ours_mv = chess.Move.from_uci(ours)

        # Top 3 choices at a depth deep enough to be authoritative for this material count.
        infos = eng.analyse(board, chess.engine.Limit(depth=22), multipv=3)
        print("\n===", label)
        print("FEN:", fen)
        best_cp = None
        for i, info in enumerate(infos, 1):
            pv = info.get("pv", [])
            san = board.san(pv[0]) if pv else "?"
            score = info["score"].pov(board.turn)
            if i == 1:
                best_cp = score
            line = " ".join(board.variation_san(pv[:6]).split()) if pv else ""
            print(f"  {i}. {san:8s} {str(score):>10s}   {line}")

        # Score our move specifically, so we can price the difference.
        info_ours = eng.analyse(board, chess.engine.Limit(depth=22), root_moves=[ours_mv])
        ours_score = info_ours["score"].pov(board.turn)
        print(f"  OURS: {board.san(ours_mv):8s} {str(ours_score):>10s}")
        try:
            loss = best_cp.score(mate_score=100000) - ours_score.score(mate_score=100000)
            print(f"  => centipawn loss vs SF best: {loss}")
        except Exception:
            print("  => (mate score involved; compare by eye)")
