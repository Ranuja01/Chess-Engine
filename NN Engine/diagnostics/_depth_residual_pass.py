# -*- coding: utf-8 -*-
"""DEPTH-RESIDUAL PASS: our own SEARCH score on SF18-labelled positions, so the target becomes
    residual = win%(SF18 d14 search) − win%(OUR d-N search)
= what our search STILL misjudges at depth — the eval knowledge search cannot supply (memory
`eval-headroom-is-failures-that-persist-as-depth-rises`). The static residual (SF18 − our static eval) mixes that with
short-horizon errors our search fixes anyway (tactics, the move being available: +2.4pp toward the mover, C3 doc §18a).
Serves the POT middlegame study and the queen/Kaufman lead (C3 doc §18b).

Engine side: tactical_test.run_one (cold per FEN, search tables cleared). Its eval is SIDE-TO-MOVE POV in millipawns
(verified 2026-10-01 on 12 labelled rows: ours/10 ≈ SF cp); mates (|ev| ≥ 9,000,000) are written as ±MATE_CP.
Output (append, resumable by FEN): fen, ours_cp_white, depth, nodes.

  PRESET=LONG_FORMAT MAX_DEPTH=10 USE_OPENING_BOOK=0 V2_PRESET=shipped \
  pyrun diagnostics/_depth_residual_pass.py IN=ks_sets/fitC_mg_sf18.csv OUT=ks_sets/fitC_mg_ours_d10.csv [SHARD=0/4] [LIMIT=0]
Run one process per shard (≤ 4 concurrent — WSL rule), then concatenate.
"""
import os, sys, csv, time
import chess

KV = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS)
from tactical_test import run_one

IN = os.path.join(THIS, KV.get("IN", "ks_sets/fitC_mg_sf18.csv"))
OUT = os.path.join(THIS, KV.get("OUT", "ks_sets/fitC_mg_ours_d10.csv"))
k, n = (int(x) for x in KV.get("SHARD", "0/1").split("/"))
if n > 1:
    OUT = OUT.replace(".csv", "_s%dof%d.csv" % (k, n))
MATE_CP = 5000

rows = [r for r in csv.DictReader(open(IN, newline="")) if r.get("best_cp")]
rows = [r for i, r in enumerate(rows) if i % n == k]
if int(KV.get("LIMIT", 0)):
    rows = rows[:int(KV["LIMIT"])]
done = set()
if os.path.exists(OUT):
    done = {r["fen"] for r in csv.DictReader(open(OUT, newline=""))}
new = not os.path.exists(OUT)
f = open(OUT, "a", newline="")
w = csv.writer(f)
if new:
    w.writerow(["fen", "ours_cp_white", "depth", "nodes"])
t0, cnt = time.time(), 0
for r in rows:
    if r["fen"] in done:
        continue
    b = chess.Board(r["fen"])
    o = run_one(r["fen"], [])
    ev = o["eval"]
    if ev is None:
        continue
    cp = max(-MATE_CP, min(MATE_CP, ev / 10.0)) if abs(ev) < 9_000_000 else (MATE_CP if ev > 0 else -MATE_CP)
    w.writerow([r["fen"], "%.1f" % (cp if b.turn == chess.WHITE else -cp), o["depth"], o["nodes"]])
    cnt += 1
    if cnt % 100 == 0:
        f.flush()
        sys.stderr.write("[depth pass %d/%d] %d rows, %.1f/s\n" % (k, n, cnt, cnt / (time.time() - t0)))
f.close()
print("DEPTH PASS shard %d/%d: %d new rows -> %s (%.0fs)" % (k, n, cnt, OUT, time.time() - t0))
