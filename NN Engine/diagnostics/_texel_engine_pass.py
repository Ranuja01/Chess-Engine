# -*- coding: utf-8 -*-
"""TEXEL DATASET, STAGE 2: the engine's side of each stage-1 position.

For the PST fit the eval splits into a FIXED part (everything except the tapered PST) and the PST itself,
which is linear in piece occupancy. This pass asks the engine for the fixed part and the phase, once, so the
fitter never has to call the engine again (EVAL-V2-TAPERED-PST-DESIGN-2026-09-25.md; PST_V2_* knobs in
search_engine.h).

  MODE=zero  PST_V2_TAPERED=1 PST_V2_ZERO=1 -> per row: total (Black-positive mp, no PST) and v2_phase256
  MODE=full  PST_V2_TAPERED=1               -> per row: total with the default tables. Rows where full ==
             zero although pieces stand on non-zero cells are PST-INERT (draw classifier, exact KPK) and the
             fitter drops them: the PST cannot move their value, so they carry no gradient.

Knobs latch once per process, so each MODE is its own process; SHARD/NSHARD split the rows between
processes (row i belongs to shard i % NSHARD). Output: one CSV per (mode, shard): row,total[,phase256].

  pyrun diagnostics/_texel_engine_pass.py MODE=zero SHARD=0 NSHARD=2 V2_PRESET=shipped PST_V2_TAPERED=1 PST_V2_ZERO=1
        [IN=/mnt/e/chess_data/texel/v2_stage1.csv.gz] [OUT_DIR=/mnt/e/chess_data/texel/pass] [LIMIT=0]
"""
import os, sys, csv, gzip, time

for _a in sys.argv[1:]:
    if '=' in _a:
        _k, _v = _a.split('=', 1)
        os.environ[_k] = _v

THIS = os.path.dirname(os.path.abspath(__file__))
ENGINE = os.path.dirname(THIS)
sys.path.insert(0, ENGINE)
os.environ.setdefault("PRESET", "LONG_FORMAT")
os.environ.setdefault("USE_OPENING_BOOK", "0")

MODE = os.environ.get("MODE", "zero")
SHARD = int(os.environ.get("SHARD", "0"))
NSHARD = int(os.environ.get("NSHARD", "1"))
IN = os.environ.get("IN", "/mnt/e/chess_data/texel/v2_stage1.csv.gz")
OUT_DIR = os.environ.get("OUT_DIR", "/mnt/e/chess_data/texel/pass")
LIMIT = int(os.environ.get("LIMIT", "0"))

if MODE not in ("zero", "full"):
    sys.exit("MODE must be zero or full")
want_zero = os.environ.get("PST_V2_ZERO", "0") == "1"
if os.environ.get("PST_V2_TAPERED", "0") != "1" or want_zero != (MODE == "zero"):
    sys.exit("☠️ MODE=%s needs PST_V2_TAPERED=1 and PST_V2_ZERO=%d -- refusing to write a mislabelled pass"
             % (MODE, 1 if MODE == "zero" else 0))

import chess
import ChessAI

ai = ChessAI.ChessAI(None, None, chess.Board(), True)
os.makedirs(OUT_DIR, exist_ok=True)
out_path = os.path.join(OUT_DIR, "%s_%d_of_%d.csv" % (MODE, SHARD, NSHARD))
t0 = time.time()
n = 0
with gzip.open(IN, "rt") as fi, open(out_path, "w", newline="") as fo:
    w = csv.writer(fo)
    w.writerow(["row", "total", "phase256"] if MODE == "zero" else ["row", "total"])
    for i, r in enumerate(csv.DictReader(fi)):
        if LIMIT and i >= LIMIT:
            break
        if i % NSHARD != SHARD:
            continue
        b = chess.Board(r["fen"])
        if MODE == "zero":
            bd = ai.ev_breakdown(b)
            w.writerow([i, int(bd["total"]), int(bd.get("v2_phase256", -1))])
        else:
            w.writerow([i, int(ai.ev(b))])
        n += 1
        if n % 200000 == 0:
            sys.stderr.write("[pass %s %d/%d] %d rows, %.0f/s\n" % (MODE, SHARD, NSHARD, n, n / (time.time() - t0)))
            sys.stderr.flush()
sys.stderr.write("[pass %s %d/%d] DONE %d rows in %.0fs -> %s\n" % (MODE, SHARD, NSHARD, n, time.time() - t0, out_path))
