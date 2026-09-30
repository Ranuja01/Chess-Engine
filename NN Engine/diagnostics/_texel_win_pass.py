# -*- coding: utf-8 -*-
"""TEXEL FIT W (OvD eg WINNABILITY), engine side: per stage-1 row, the winnability INPUTS, the engine total, the
winnability adjustment and phase -- and the CLOSURE gate: with any WIN_V2_* weights set, the Python model
    adj = sign(T) · max(trunc(C·(256−phase)/256), −|T|),   C = Σ w·in + BASE,   T = total − adj
must equal the engine's published v2_winnab on every row (integer arithmetic, so any residual is a model bug).
Run once with test weights (closure, LIMIT=20000) and once with WIN_V2=0 on all rows (the fit's data: T is then the
shipped total).

  pyrun diagnostics/_texel_win_pass.py V2_PRESET=shipped [WIN_V2=1 WIN_V2_PAWNS=.. ...] [IN=..] [OUT=..] [LIMIT=0]
"""
import os, sys, csv, gzip, time
import numpy as np

for _a in sys.argv[1:]:
    if '=' in _a:
        _k, _v = _a.split('=', 1)
        os.environ[_k] = _v
THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(THIS))
os.environ.setdefault("PRESET", "LONG_FORMAT")
os.environ.setdefault("USE_OPENING_BOOK", "0")
IN = os.environ.get("IN", "/mnt/e/chess_data/texel/fitC_stage1.csv.gz")
OUT = os.environ.get("OUT", "/mnt/e/chess_data/texel/fitC_win.npz")
LIMIT = int(os.environ.get("LIMIT", "0"))
KEYS = ["passed", "pawns", "outflanking", "infiltration", "both_flanks", "pawn_ending", "almost_unwinnable"]
WN = ["WIN_V2_PASSED", "WIN_V2_PAWNS", "WIN_V2_OUTFLANK", "WIN_V2_INFILT", "WIN_V2_FLANKS", "WIN_V2_PAWN_END",
      "WIN_V2_UNWIN"]
W = [int(os.environ.get(k, "0")) for k in WN]
BASE = int(os.environ.get("WIN_V2_BASE", "0"))
ON = os.environ.get("WIN_V2", "0") == "1"
CAP = int(os.environ.get("WIN_V2_CAP", "0"))
# WSF_V2 (2026-10-01): closure for the reference-form SCALE FACTOR (win_scale_adjust); features via
# _texel_win_sf_fit.scale_features, whose PASSED is a python-chess approximation of v2's passer mask ⇒ exact closure
# is claimed only with WSF_V2_PASSED=0; with it set the mismatch rate measures the definitional gap.
WSF_ON = os.environ.get("WSF_V2", "0") == "1"
WSF = {k: int(os.environ.get("WSF_V2_" + k, "0")) for k in ("BASE", "SP", "ONEFLANK", "OCB", "PASSED")}

import chess
import ChessAI
if WSF_ON:
    from _texel_win_sf_fit import scale_features


def cdiv(a, b):
    q = abs(a) // b
    return q if a >= 0 else -q


ai = ChessAI.ChessAI(None, None, chess.Board(), True)
rows, ins, tot, adj, ph = [], [], [], [], []
bad = live = 0
t0 = time.time()
with gzip.open(IN, "rt") as f:
    for i, r in enumerate(csv.DictReader(f)):
        if LIMIT and i >= LIMIT:
            break
        b = chess.Board(r["fen"])
        bd = ai.ev_breakdown(b)
        w = ChessAI.win_inputs(b.pawns, b.knights, b.bishops, b.rooks, b.queens, b.kings, b.occupied_co[True],
                               b.occupied_co[False])
        p = int(bd.get("v2_phase256", -1))
        t = int(bd["total"])
        a = int(bd.get("v2_winnab", 0))
        rows.append(i); ins.append([w[k] for k in KEYS]); tot.append(t); adj.append(a); ph.append(p)
        if ON and p >= 0:
            T = t - a
            C = sum(wi * w[k] for wi, k in zip(W, KEYS)) + BASE
            v = cdiv(C * (256 - p), 256)
            m = abs(T)
            d = max(v, -m) if T != 0 else 0
            if CAP > 0:
                d = max(-CAP, min(CAP, d))
            model = d if T > 0 else (-d if T < 0 else 0)
            bad += model != a
            live += a != 0
        if WSF_ON and p >= 0:
            T = t - a
            if T != 0:
                x = scale_features(r["fen"], T > 0)   # [1, sp, oneflank, ocb_pure, ocb_mix, pawn_end, passed, ...]
                fs = 64 + WSF["BASE"] + WSF["SP"] * x[1] + WSF["ONEFLANK"] * x[2] + WSF["OCB"] * x[3] \
                    + WSF["PASSED"] * x[6]
                fs = max(0, min(64, fs))
                model = cdiv(T * (256 - p) * (fs - 64), 256 * 64)
            else:
                model = 0
            bad += model != a
            live += a != 0
        if (i + 1) % 200000 == 0:
            sys.stderr.write("[win pass] %d rows, %.0f/s\n" % (i + 1, (i + 1) / (time.time() - t0)))
np.savez_compressed(OUT, row=np.array(rows, dtype=np.int32), inputs=np.array(ins, dtype=np.int16),
                    total=np.array(tot, dtype=np.int32), adj=np.array(adj, dtype=np.int32),
                    phase=np.array(ph, dtype=np.int16), keys=np.array(KEYS))
print("WIN PASS  %d rows  %.0fs  WIN_V2=%d" % (len(rows), time.time() - t0, ON))
if ON or WSF_ON:
    print("  closure: live rows %d · model mismatches %d  VERDICT %s" % (live, bad, "EXACT" if bad == 0 else "☠️ DIVERGES"))
print("wrote", OUT)
