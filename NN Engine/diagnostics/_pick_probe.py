# -*- coding: utf-8 -*-
"""What move does OUR engine actually pick from a position, and how does that change under knobs?

Bench suites tell us how often we match a reference move; they do not tell us WHY a specific real-game move
was chosen. This runs the real search on an explicit FEN list and prints the chosen move + eval + depth, so a
game mistake can be re-run under different knobs (e.g. ENABLE_ROOT_RAZOR=0) to test whether the move was
never searched at all rather than searched and rejected.

Reads 'label<TAB>fen' lines, same format as probe_fens.py.
  pyrun diagnostics/_pick_probe.py <file> [EXPECT=label:uci,...] [KEY=VAL ...]
"""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
import sys

# Knobs must reach the environment BEFORE the extension is imported -- they latch at init.
ARGS = []
for a in sys.argv[1:]:
    if '=' in a and not a.endswith('.fens') and not a.startswith('EXPECT='):
        k, v = a.split('=', 1)
        os.environ[k] = v
    else:
        ARGS.append(a)

THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(THIS))
sys.path.insert(0, THIS)
from tactical_test import run_one

path = None
expect = {}
for a in ARGS:
    if a.startswith('EXPECT='):
        for pair in a.split('=', 1)[1].split(','):
            if ':' in pair:
                lbl, uci = pair.split(':', 1)
                expect[lbl] = uci
    else:
        path = a

if not path:
    print(__doc__)
    raise SystemExit

rows = []
with open(path) as fh:
    for line in fh:
        line = line.rstrip('\n')
        if not line or '\t' not in line:
            continue
        label, fen = line.split('\t', 1)
        rows.append((label, fen))

print("%-24s %-7s %-8s %5s %8s   %s" % ("label", "picked", "expect", "depth", "eval", "match"))
for label, fen in rows:
    r = run_one(fen, set())
    want = expect.get(label)
    mark = '' if want is None else ('OK' if r['uci'] == want else '<<< DIFFERS')
    print("%-24s %-7s %-8s %5s %8s   %s"
          % (label, r['uci'], want or '-', r['depth'], r['eval'], mark))
