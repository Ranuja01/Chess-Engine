# -*- coding: utf-8 -*-
"""Color-mirror symmetry test for king safety. KS is a single symmetric computation (danger_white - danger_black),
netted and absolute Black-positive. So for ANY position, KS(pos) must equal -KS(mirror(pos)) exactly (mirror =
vertical flip + color swap). Any residual = a color-asymmetry BUG in the detector — which would show up as the
'wrong-sign' cases and be catastrophic under iterative deepening (side-to-move flips every ply).

Sets KS on (MAG=4000, defender=0) at process start so the detector is active, then reports the residual
r = KS_abs(pos) + KS_abs(mirror) (should be ~0) over the danger + calm sets.

Run: pyrun diagnostics/ks_symmetry.py
"""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
os.environ.update(dict(KING_SAFETY_MAG="4000", KS_DEFENDER="0", ENABLE_KS_SF_WEAK="1",
                       ENABLE_KS_SF_SAFECHECK="1", KS_FLOOR="13", KS_NO_QUEEN="6"))   # F13 ship bundle
import sys
THIS = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(THIS))
import chess
from ChessAI import ChessAI
ai = ChessAI(None, None, chess.Board(), True)


def ks_abs(board):
    bd = ai.ev_breakdown(board)
    if bd.get("checkmate"):
        return None
    return bd.get("king_safety", 0.0) / 1000.0     # absolute Black-positive pawns


def run_set(name):
    resids = []
    worst = []
    for ln in open(os.path.join(THIS, "ks_sets", name + ".txt")):
        fen = ln.rstrip("\n").split("\t", 1)[1]
        try:
            b = chess.Board(fen)
        except ValueError:
            continue
        a = ks_abs(b)
        m = ks_abs(b.mirror())
        if a is None or m is None:
            continue
        r = a + m                      # should be ~0 if color-symmetric
        resids.append(abs(r))
        worst.append((abs(r), a, m, fen))
    import statistics
    resids.sort()
    worst.sort(reverse=True)
    n = len(resids)
    print("[%s] n=%d  mean|resid|=%.3f  median=%.3f  max=%.3f  #|resid|>0.20=%d"
          % (name, n, statistics.mean(resids) if n else 0, resids[n // 2] if n else 0,
             max(resids) if n else 0, sum(1 for r in resids if r > 0.20)))
    for r, a, m, fen in worst[:4]:
        if r > 0.20:
            print("    resid %+.2f  KS=%+.2f  KS(mirror)=%+.2f  %s" % (r, a, m, fen))


for s in ("danger", "control_calm", "control_eg"):
    run_set(s)
