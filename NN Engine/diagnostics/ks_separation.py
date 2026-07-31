# -*- coding: utf-8 -*-
"""KS separation objective (the thing a KS re-weight should maximize): how far apart is our king_safety term on
REAL attacks (fire set) vs CALM positions (suppress set)? A good model → high mean|KS| on fire, ~0 on suppress.
Avoids SF11-static contamination (no per-position SF label needed) — pure discrimination. Honors env knobs
(argv KEY=VAL before ChessAI import) so we can score any candidate config.

Run: pyrun diagnostics/ks_separation.py [KS_SAFE_CHECK=.. KS_FLOOR=.. KS_INTERACT=.. ...]
"""
import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ.setdefault('KS_SAFE_CHECK_DEF', '5')
for _a in sys.argv[1:]:
    if '=' in _a: _k, _v = _a.split('=', 1); os.environ[_k] = _v
THIS = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(THIS)); sys.path.insert(0, THIS)
import chess
from ChessAI import ChessAI

def load(name):
    p = os.path.join(THIS, "ks_sets", name)
    if not os.path.exists(p): return []
    out = []
    for ln in open(p):
        ln = ln.strip()
        if not ln or ln.startswith("#"): continue
        # lines are either "<value>\t<fen>" or a bare fen; take the part that parses as a board
        parts = ln.split(None, 1)
        fen = parts[1] if (len(parts) == 2 and _isfloat(parts[0])) else ln
        out.append(fen)
    return out

def _isfloat(s):
    try: float(s); return True
    except ValueError: return False

FIRE = load("danger.txt")                       # must-HOLD (KS v1 tuned; already fires)
FIRE_NEW = load("ks_underread_sf18.txt")         # must-FIX (SF18-confirmed; currently ~0)
# must-SUPPRESS = SF18-genuinely-safe subset (control_sf18safe.txt, materially balanced AND quiet) if built,
# else the raw (contaminated) control sets. Over-firing is only real on the SF18-safe guard.
SUPPRESS = load("control_sf18safe.txt") or (load("control_calm.txt") + load("control_eg.txt"))
ai = ChessAI(None, None, chess.Board(), True)

def mean_abs_ks(fens):
    vals = []
    for fen in fens:
        try:
            b = chess.Board(fen); bd = ai.ev_breakdown(b)
            vals.append(abs(bd.get("king_safety", 0.0) / 1000.0))   # pawns, magnitude
        except Exception:
            continue
    return (sum(vals) / len(vals) if vals else 0.0), len(vals)

fire_m, fn = mean_abs_ks(FIRE)
new_m, nn = mean_abs_ks(FIRE_NEW)
sup_m, sn = mean_abs_ks(SUPPRESS)
# Parseable line for the tuner. Objective: raise FIRENEW (currently ~0) toward danger levels, keep FIREOLD
# firing, keep SUPPRESS near 0. score = firenew + 0.5*fireold - 4*suppress (penalize calm noise hard).
score = new_m + 0.5 * fire_m - 4.0 * sup_m
print("RESULT fireold=%.4f firenew=%.4f suppress=%.4f score=%.4f" % (fire_m, new_m, sup_m, score))
print("  FIREOLD(hold) n=%d=%.3f  FIRENEW(fix) n=%d=%.3f  SUPPRESS n=%d=%.3f" % (fn, fire_m, nn, new_m, sn, sup_m))
