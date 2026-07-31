# -*- coding: utf-8 -*-
"""Are the rewound collapse start-FENs actually recoverable (SF18 ~equal), our POV? If most are still lost the
replay is uninformative -> rewind further. Minimal SF18 depth check."""
import os, sys
import chess, chess.engine
THIS = os.path.dirname(os.path.abspath(__file__))
sf = chess.engine.SimpleEngine.popen_uci(os.environ["STOCKFISH_PATH"])
depth = int(sys.argv[1]) if len(sys.argv) > 1 else 14
path = sys.argv[2] if len(sys.argv) > 2 else os.path.join(THIS, "ks_sets", "rewound.txt")
vals = []
try:
    for ln in open(path):
        if not ln.strip():
            continue
        fen = ln.rstrip("\n").split("\t", 1)[-1].strip()
        b = chess.Board(fen)
        info = sf.analyse(b, chess.engine.Limit(depth=depth))
        v = info["score"].pov(b.turn).score(mate_score=100000) / 100.0   # our POV (side to move = us)
        vals.append(v)
finally:
    sf.quit()
import statistics
vals.sort()
n = len(vals)
rec = sum(1 for v in vals if v >= -1.0)
print("SF18 d%d over %d rewound FENs (our POV): median=%.2f min=%.2f max=%.2f" % (depth, n, vals[n//2], vals[0], vals[-1]))
print("recoverable (>= -1.0): %d/%d   equalish (-1..+1.5): %d   already-lost (< -1.5): %d"
      % (rec, n, sum(1 for v in vals if -1.0 <= v <= 1.5), sum(1 for v in vals if v < -1.5)))
