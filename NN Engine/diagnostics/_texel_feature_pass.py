# -*- coding: utf-8 -*-
"""TEXEL FIT C1, engine side: per-position feature counts for the linear C1 blocks, plus the engine's own block scores.

For each stage-1 row: ChessAI.v2_feature_counts (Black - White counts per parameter, eval_v2.h v2_features),
ev_breakdown's total, phase256 and the four C1 block scores (mobility, pawn_struct, v2_passers, v2_placement).
The fitter models every C1 block as sum(count x theta) on top of a FIXED remainder, so the pass also CHECKS that
model against the engine: per block, predicted = sum over params of diff x (theta_mg * p/256 + theta_eg *
(256-p)/256) must match the published block score within its truncation budget. A residual beyond that is an
extraction bug, never noise -- the same rule the PST pass met at median 0.00 mp.

  pyrun diagnostics/_texel_feature_pass.py [IN=/mnt/e/chess_data/texel/v2_stage1.csv.gz] [OUT=/mnt/e/chess_data/texel/c1_std.npz]
        [LIMIT=0] V2_PRESET=shipped
Output .npz: row (int32), diff (int16, n x 106), phase (int16), total (int32), blocks (int32, n x 4), flags (int8),
theta_mg / theta_eg (float64, 106).
"""
import os, sys, csv, gzip, time
import numpy as np

for _a in sys.argv[1:]:
    if '=' in _a:
        _k, _v = _a.split('=', 1)
        os.environ[_k] = _v
THIS = os.path.dirname(os.path.abspath(__file__))
ENGINE = os.path.dirname(THIS)
sys.path.insert(0, ENGINE)
os.environ.setdefault("PRESET", "LONG_FORMAT")
os.environ.setdefault("USE_OPENING_BOOK", "0")
IN = os.environ.get("IN", "/mnt/e/chess_data/texel/v2_stage1.csv.gz")
OUT = os.environ.get("OUT", "/mnt/e/chess_data/texel/c1_std.npz")
LIMIT = int(os.environ.get("LIMIT", "0"))

import chess
import ChessAI

ai = ChessAI.ChessAI(None, None, chess.Board(), True)
tmg, teg = (np.array(x) for x in ChessAI.v2_feature_theta())
P = ChessAI.V2F_PER_SIDE
BLOCKS = [("mobility", slice(0, 66)), ("pawn_struct", slice(66, 77)), ("v2_passers", slice(77, 97)),
          ("v2_placement", slice(97, 106)),
          # C3-a king shelter + pawn storm (2026-09-27). ⚠️ With KSB_V2 off its block score is ABSENT (read as 0 here)
          # and theta is 0, so the gate passes VACUOUSLY; prove closure under KSB_V2=1 with a non-zero test table.
          ("v2_shelter", slice(106, 162)),
          ("v2_kflank", slice(162, 172)),       # C3-b pawnless flank + king-pawn distance (same vacuity caveat)
          ("v2_kprot", slice(172, 184))]        # C3-c KingProtector (same vacuity caveat)

rows, diffs, phases, totals, blocks, flags = [], [], [], [], [], []
t0 = time.time()
with gzip.open(IN, "rt") as f:
    for i, r in enumerate(csv.DictReader(f)):
        if LIMIT and i >= LIMIT:
            break
        b = chess.Board(r["fen"])
        wc, bc, fl = ChessAI.v2_feature_counts(b)
        bd = ai.ev_breakdown(b)
        rows.append(i)
        diffs.append([x - y for x, y in zip(bc, wc)])
        phases.append(int(bd.get("v2_phase256", -1)))
        totals.append(int(bd["total"]))
        blocks.append([int(bd.get(k, 0)) for k, _ in BLOCKS])
        flags.append(fl)
        if (i + 1) % 200000 == 0:
            sys.stderr.write("[c1 pass] %d rows, %.0f/s\n" % (i + 1, (i + 1) / (time.time() - t0)))
            sys.stderr.flush()

D = np.array(diffs, dtype=np.int32)
ph = np.array(phases, dtype=np.float64)
B = np.array(blocks, dtype=np.int64)
F = np.array(flags, dtype=np.int8)
if (F & 4).any():
    sys.exit("☠️ a live knob outside the extractor's model (flag 4) -- the fixed part would absorb it; refusing")
np.savez_compressed(OUT, row=np.array(rows, dtype=np.int32), diff=D.astype(np.int16), phase=ph.astype(np.int16),
                    total=np.array(totals, dtype=np.int32), blocks=B.astype(np.int32), flags=F, theta_mg=tmg, theta_eg=teg)

# ---- the gate: does sum(count x theta) reproduce each published block score? -------------------------------
ok = (ph >= 0) & ((F & 3) == 0)
w_mg = ph / 256.0
print("C1 FEATURE PASS  %d rows (%d scored; %d draw/tier-2b/terminal excluded)  %.0fs"
      % (len(rows), ok.sum(), (~ok).sum(), time.time() - t0))
print("  block          rows_live   median|res|   p99|res|   max|res|   (res = engine - sum(count x theta), mp)")
for bi, (name, sl) in enumerate(BLOCKS):
    pred = (D[:, sl] * (tmg[sl] * w_mg[:, None] + teg[sl] * (1.0 - w_mg[:, None]))).sum(axis=1)
    res = np.abs(B[:, bi] - pred)[ok]
    live = (B[:, bi] != 0)[ok].sum()
    print("  %-14s %9d   %11.2f   %8.2f   %8.1f" % (name, live, np.median(res), np.percentile(res, 99), res.max()))
print("wrote", OUT)
