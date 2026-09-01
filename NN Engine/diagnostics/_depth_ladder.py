"""Run our search on one FEN at a ladder of fixed depths and print the chosen move.

Answers "search or eval?" for a suspected tactical miss: if the engine switches OFF the losing
move as depth rises, it is a DEPTH/search problem (we simply did not see far enough). If it keeps
choosing the losing move at every depth, the leaf evaluation of the refutation line is wrong and it
is an EVAL problem. Knobs latch at init, so MAX_DEPTH cannot vary within one process -- the caller
passes one depth per invocation and a driver loops.

Run: pyrun diagnostics/_depth_ladder.py FEN=<fen> DEPTH=<n> [KEY=VAL knobs]
"""
import os, sys
OPTS = {}
for a in sys.argv[1:]:
    if '=' in a:
        k, v = a.split('=', 1)
        if k in ('FEN', 'DEPTH'):
            OPTS[k] = v
        else:
            os.environ[k] = v
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
os.environ['USE_OPENING_BOOK'] = '0'
os.environ['MAX_DEPTH'] = OPTS.get('DEPTH', '10')
os.environ.setdefault('PRESET', 'LONG_FORMAT')
THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(THIS)); sys.path.insert(0, THIS)
from tactical_test import run_one
fen = OPTS['FEN']
r = run_one(fen, [])
print(f"DEPTH={os.environ['MAX_DEPTH']}  move={r.get('uci')}  eval={r.get('eval')}  "
      f"reached_depth={r.get('depth')}  nodes={r.get('nodes')}")
