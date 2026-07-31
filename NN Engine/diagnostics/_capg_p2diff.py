# -*- coding: utf-8 -*-
"""P2 term-by-term SF11 static DIFF: hanging-rook (a1) vs safe-rook (c1), White-POV, to see WHERE SF11's
small +0.28 safe-rook preference lives (candidate compensator/detector signal). Also prints our own POV
handling sanity for P3."""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
import sys
THIS = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(THIS)); sys.path.insert(0, THIS)
from eval_vs_sf11 import SF11Eval, SF11
P2o = "r3r1k1/3nbppp/1p1pb3/3p2P1/1P1N1P1P/1Q2B3/4BP2/R5K1 b - - 0 27"
P2m = "r3r1k1/3nbppp/1p1pb3/3p2P1/1P1N1P1P/1Q2B3/4BP2/2R3K1 b - - 0 27"
sf = SF11Eval(SF11)
try:
    _, th = sf.eval(P2o); _, ts = sf.eval(P2m)
    keys = sorted(set(th) | set(ts), key=lambda k: -abs(ts.get(k,0)-th.get(k,0)))
    print("%-14s %8s %8s %8s" % ("term(WhitePOV)", "HANGING", "SAFE", "diff(S-H)"))
    for k in keys:
        h, s = th.get(k,0.0), ts.get(k,0.0)
        if abs(s-h) >= 0.005 or k=="Total":
            print("  %-12s %+8.2f %+8.2f %+8.2f" % (k, h, s, s-h))
finally:
    sf.close()
