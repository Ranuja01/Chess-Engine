# -*- coding: utf-8 -*-
"""Per-CLASS guard check: does a candidate improve some position classes by breaking others?

STS is one corpus with its own composition, so a king-safety DAMPER can score well there simply because
STS skews positional/quiet. This recomputes our eval over position_bank.csv (geo_class + phase labelled,
SF18-anchored) under the CURRENT knob config and reports win%-space error PER CLASS and PER PHASE BUCKET,
so a gain in one family that costs another is visible instead of averaged away.

Knobs latch at init => one process per setting. Run twice and diff the printed tables:
    pyrun diagnostics/_class_guard.py ENABLE_MATERIAL_COUNT_FIX=1 ENABLE_PASSER_V3=1
    pyrun diagnostics/_class_guard.py ENABLE_MATERIAL_COUNT_FIX=1 ENABLE_PASSER_V3=1 MOD_KS_REALIZ=128
"""
import os, sys, csv, math
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
for _a in sys.argv[1:]:
    if '=' in _a:
        _k, _v = _a.split('=', 1)
        os.environ[_k] = _v
THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(THIS))
os.chdir(os.path.dirname(THIS))

import chess
from ChessAI import ChessAI

WIN_K = 0.00368208
def winpct(cp): return 50.0 + 50.0 * (2.0 / (1.0 + math.exp(-WIN_K * cp)) - 1.0)

rows = []
for r in csv.DictReader(open(os.path.join(THIS, "ks_sets", "position_bank.csv"))):
    if not r.get("sf18"):
        continue
    try:
        sf18 = float(r["sf18"])
    except Exception:
        continue
    if abs(sf18) > 20.0:
        continue
    rows.append(r)

seed = chess.Board(rows[0]["fen"])
ai = ChessAI(None, None, seed, seed.turn)

byclass, byphase, allerr = {}, {}, []
for r in rows:
    try:
        b = chess.Board(r["fen"])
    except Exception:
        continue
    bd = ai.ev_breakdown(b)
    if bd.get("checkmate"):
        continue
    ours = -bd["total"] / 1000.0
    e = winpct(ours * 100.0) - winpct(float(r["sf18"]) * 100.0)
    allerr.append(e)
    byclass.setdefault(r.get("geo_class") or "?", []).append(e)
    try:
        ph = int(float(r.get("phase_score") or 0))
    except Exception:
        ph = 0
    bucket = "open(0-42)" if ph <= 42 else ("mid(43-85)" if ph <= 85 else "end(86-128)")
    byphase.setdefault(bucket, []).append(e)

def mse(v): return sum(x * x for x in v) / len(v)
def mae(v): return sum(abs(x) for x in v) / len(v)

cfg = " ".join(f"{k}={os.environ[k]}" for k in
               ("ENABLE_MATERIAL_COUNT_FIX", "ENABLE_PASSER_V3", "MOD_KS_REALIZ", "MOD_KS_BACKING")
               if k in os.environ)
print(f"CONFIG: {cfg or '(defaults)'}")
print(f"ALL                     n={len(allerr):>5}  MAE={mae(allerr):>7.3f}  MSE={mse(allerr):>8.1f}")
print("\nby geo_class:")
for k in sorted(byclass, key=lambda k: -len(byclass[k])):
    v = byclass[k]
    print(f"  {k:<22} n={len(v):>5}  MAE={mae(v):>7.3f}  MSE={mse(v):>8.1f}")
print("\nby phase:")
for k in ("open(0-42)", "mid(43-85)", "end(86-128)"):
    if k in byphase:
        v = byphase[k]
        print(f"  {k:<22} n={len(v):>5}  MAE={mae(v):>7.3f}  MSE={mse(v):>8.1f}")
