# -*- coding: utf-8 -*-
"""Cold single-position eval probe: run the engine on each FEN given as an argument
(fresh ChessAI per FEN = empty caches/history) and print eval + chosen move + depth + nodes.

Honors the same PRESET / MAX_DEPTH / feature env knobs the engine reads, so you can A/B a
position across configs (e.g. base vs LMP) at the same time control. Reuses tactical_test.run_one
(which captures the engine's stdout, now that the Evaluation/Positions flush is fixed).

Run (WSL, from NN Engine/):
    PRESET=STANDARD MAX_DEPTH=64 USE_OPENING_BOOK=0 \
      /home/ranuja/anaconda3/bin/python diagnostics/eval_fen.py "<fen1>" "<fen2>" ...

Eval sign = side-to-move relative (positive = good for the side to move).
"""

import sys
from tactical_test import run_one

for fen in sys.argv[1:]:
    r = run_one(fen, set())
    ev = r['eval']
    print(f"eval={ev if ev is not None else '?':>8}  move={str(r['uci']):<6}  depth={r['depth']}  nodes={r['nodes']}")
    print(f"    {fen}")
