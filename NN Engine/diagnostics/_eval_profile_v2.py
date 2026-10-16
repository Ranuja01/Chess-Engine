# -*- coding: utf-8 -*-
"""v2 EVAL PROFILE by phase (2026-10-16) — per-block cycles/call and share of the whole v2 eval (V2_EVAL).

  (profile build: bash selfplay/overnight_runner.sh build_profile — ALWAYS `build` afterwards to restore production)
  V2_PRESET=shipped [knobs] pyrun diagnostics/_eval_profile_v2.py [TAG=npm1_sp250] [REPS=500] [CAP=150]

Reuses eval_profile.py's position collection (self-play game.jsonl, phase bins). eval_profile.py's own %-table is v1-based
(its exclusive base = v1's terms, ~0 under EVAL_ARM=1), so it is meaningless for v2; this reads the V2_* scopes instead.
Blocks are NESTED in V2_EVAL; the unscoped remainder (draw class, tempo, mobility scoring outside ATTACK, …) is printed.
⚠️ __rdtsc cycles on WSL include timer/frequency noise — compare shares and orders of magnitude, not single cycles.
"""
import os, sys
for _a in sys.argv[1:]:
    if "=" in _a:
        _k, _v = _a.split("=", 1)
        os.environ[_k] = _v
THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS); sys.path.insert(0, os.path.dirname(THIS))
os.environ.setdefault("PRESET", "LONG_FORMAT"); os.environ.setdefault("USE_OPENING_BOOK", "0")
sys.argv = sys.argv[:1]                                   # eval_profile parses argv only in its main(); keep ours out
import chess
import eval_profile as EP
from ChessAI import ChessAI

TAG = os.environ.get("TAG", "npm1_sp250")
REPS = int(os.environ.get("REPS", "500"))
CAP = int(os.environ.get("CAP", "150"))
V2 = ["V2_CONTEXT", "V2_RUNG0", "V2_KAUF", "V2_ATTACK", "V2_KS", "V2_PAWNENTRY", "V2_STRUCT", "V2_PLACEMENT", "V2_C3",
      "V2_PX", "V2_WIN"]

ai = ChessAI(None, None, chess.Board(), True)
if ai.get_profile() == []:
    sys.exit("production build — run `bash selfplay/overnight_runner.sh build_profile` first")
binned, files = EP.collect_positions(TAG, CAP)
if binned is None:
    sys.exit("no games for tag %s" % TAG)
print("v2 EVAL PROFILE — tag %s (%d games), %d reps/position, cap %d/bin" % (TAG, len(files), REPS, CAP))
rows = {}
for b in EP.PHASE_BINS:
    if not binned[b]:
        continue
    snap = {r["term"]: r for r in EP.profile_bin(ai, binned[b], REPS)}
    tot = snap["V2_EVAL"]["cycles"] or 1
    evals = snap["V2_EVAL"]["calls"] or 1
    rows[b] = (snap, tot, evals)
bins = list(rows)
print("%-13s" % "block" + "".join("%15s" % b.split("_", 1)[1][:13] for b in bins))
print("%-13s" % "V2_EVAL cyc/ev" + "".join("%15.0f" % (rows[b][1] / rows[b][2]) for b in bins))
for t in V2:
    line = "%-13s" % t
    for b in bins:
        snap, tot, evals = rows[b]
        r = snap.get(t, {"cycles": 0, "calls": 0})
        line += "%9.1f%% %4.0f" % (100.0 * r["cycles"] / tot, r["cycles"] / evals)
    print(line)
line = "%-13s" % "(unscoped)"
for b in bins:
    snap, tot, evals = rows[b]
    scoped = sum(snap.get(t, {"cycles": 0})["cycles"] for t in V2)        # all V2 blocks are siblings inside V2_EVAL
    line += "%9.1f%%     " % (100.0 * (tot - scoped) / tot)
print(line)
print("columns: % of V2_EVAL cycles · cycles per eval. All V2_* blocks are siblings inside V2_EVAL (none nested in another).")
