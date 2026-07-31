# -*- coding: utf-8 -*-
"""Ground-truth side of the material-overread middle-layer artifact: for the 3 representative fens, report
SF11 (classical HCE) and SF18 (NNUE) SEARCH bestmove + score, plus SF18 static NNUE eval. All scores OUR
point of view (positive = good for the side to move's opponent-flipped... no: OUR POV = good for us), pawns."""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
import sys
THIS = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(THIS)); sys.path.insert(0, THIS)
import chess, chess.engine
from eval_vs_sf11 import SF11
sys.path.insert(0, os.path.join(os.path.dirname(THIS), "selfplay"))
from arbiter import find_stockfish

FENS = [
    ("WORST |fold|",    "rn2k2r/4bppp/2p5/1pQn4/6P1/P4N2/P2BR2P/1K6 b kq - 0 25"),
    ("MEDIAN |fold|",   "1rbq1rk1/ppp2pb1/7p/2n1pnpP/4Q3/2NP1NP1/PPPB1PB1/2K1R2R w - - 2 15"),
    ("SMALLEST |fold|", "rn6/5p2/pBp1pk2/P4p2/1p5b/5B1P/1P2K3/3R4 b - - 1 39"),
]
DEPTH = 22
sf18_path = find_stockfish()
print("SF11:", SF11.split("/")[-1])
print("SF18:", (sf18_path or "NOT FOUND").split("/")[-1])

def pov_cp(score, us):
    # score is white-relative PovScore.white(); convert to our POV pawns
    v = score.white().score(mate_score=100000)
    return (v if us == chess.WHITE else -v) / 100.0

def search(engine, board, us):
    info = engine.analyse(board, chess.engine.Limit(depth=DEPTH))
    mv = info["pv"][0] if info.get("pv") else None
    return (board.san(mv) if mv else "?"), pov_cp(info["score"], us)

e11 = chess.engine.SimpleEngine.popen_uci(SF11)
e18 = chess.engine.SimpleEngine.popen_uci(sf18_path)
try:
    hdr = "%-16s %3s | %-8s %7s | %-8s %7s | %8s" % (
        "pick", "stm", "SF11 mv", "SF11cp", "SF18 mv", "SF18cp", "SF18stat")
    print("\n" + hdr); print("-" * len(hdr))
    for lab, fen in FENS:
        b = chess.Board(fen); us = b.turn
        m11, c11 = search(e11, b, us)
        m18, c18 = search(e18, b, us)
        st = e18.analyse(b, chess.engine.Limit(depth=1))
        s18stat = pov_cp(st["score"], us)
        print("%-16s %3s | %-8s %+7.2f | %-8s %+7.2f | %+8.2f" % (
            lab, "W" if us else "B", m11, c11, m18, c18, s18stat))
finally:
    e11.quit(); e18.quit()
