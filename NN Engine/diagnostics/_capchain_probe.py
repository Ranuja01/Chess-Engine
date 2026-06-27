# -*- coding: utf-8 -*-
"""Single-FEN engine probe: print chosen move / eval / depth / nodes for the env-configured engine.
Untracked dev probe (capchain v2 node-cost test). Usage:
    PRESET=LIGHTNING ENABLE_LMR_CAPCHAIN=1 CAPCHAIN_REDUCE_LESS=2 python diagnostics/_capchain_probe.py "<fen>"
"""
import os, sys
THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS)
from tactical_test import run_one

fen = sys.argv[1]
r = run_one(fen, set())
q = r.get('qnodes')
nn = r['nodes'] or 0
qq = q or 0
share = (100.0 * qq / (nn + qq)) if (nn + qq) else 0.0
print(f"move={r['uci']:<6} eval={str(r['eval']):>8} depth={r['depth']} nodes={nn} qnodes={qq} qshare={share:.0f}%")
