# -*- coding: utf-8 -*-
"""SPEED of a knob setting on a fixed position set (2026-10-16): fixed-depth search per row, summed nodes and wall time ⇒ NPS.

  PRESET=LONG_FORMAT MAX_DEPTH=10 USE_OPENING_BOOK=0 V2_PRESET=shipped [knobs] pyrun diagnostics/_pot_speed.py IN=<csv> [N=600] TAG=x

One process per setting (knobs latch at init) and run SOLO — concurrent load makes wall time meaningless. NPS isolates per-node
cost (the eval's cycles); node count differences (pruning) are reported separately. ⚠️ Node-limited games cannot see eval cost,
so this is the only place a slower eval term shows up before a timed run.
"""
import os, sys, csv, time
KV = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS); sys.path.insert(0, os.path.dirname(THIS))
from tactical_test import run_one

rows = [r["fen"] for r in csv.DictReader(open(KV["IN"], newline=""))][: int(KV.get("N", "600"))]
nodes, secs, n = 0, 0.0, 0
for f in rows:
    o = run_one(f, [])
    if o["nodes"] is None:
        continue
    nodes += o["nodes"]; secs += o["time"]; n += 1
print("SPEED %s: %d rows · nodes %d · time %.1f s · NPS %.0f" % (KV.get("TAG", "?"), n, nodes, secs, nodes / max(secs, 1e-9)))
