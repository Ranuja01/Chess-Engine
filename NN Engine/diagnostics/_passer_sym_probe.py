# -*- coding: utf-8 -*-
"""Per-term colour-symmetry diff for a single FEN vs its mirror.

Localises which eval term breaks eval(mirror(b)) == -eval(b) on one position.
  pyrun diagnostics/_passer_sym_probe.py [FEN="..."] ENABLE_PASSER_DETECT_SF=1
"""
import os, sys

for _a in sys.argv[1:]:
    if '=' in _a:
        _k, _v = _a.split('=', 1)
        os.environ[_k] = _v

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(THIS)); sys.path.insert(0, THIS)
import chess
from ChessAI import ChessAI

FEN = os.environ.get("FEN", "2q1r1k1/6p1/2p1r3/p4pp1/P1QRn2P/1P6/KB6/6R1 w - - 0 54")

ai = ChessAI(None, None, chess.Board(), True)
b = chess.Board(FEN)
m = b.mirror()
db = ai.ev_breakdown(b)
dm = ai.ev_breakdown(m)

print("FEN:", FEN)
print("total b=%s  mirror=%s  sum=%s (should be 0)\n" % (db.get("total"), dm.get("total"), db.get("total", 0) + dm.get("total", 0)))
print("%-28s %12s %12s %12s" % ("term", "b", "mirror", "b+mirror"))
for k in sorted(set(db) | set(dm)):
    x, y = db.get(k, 0), dm.get(k, 0)
    if not isinstance(x, int) or not isinstance(y, int):
        continue
    s = x + y
    mark = "  <== ASYM" if s != 0 else ""
    if s != 0:
        print("%-28s %12s %12s %12s%s" % (k, x, y, s, mark))
