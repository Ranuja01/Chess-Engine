# -*- coding: utf-8 -*-
"""Coordinate-descent KS tuner. Objective (from ks_separation.py): lift FIRENEW (SF18-confirmed real attacks we
currently read ~0) while HOLDING FIREOLD (danger.txt) and SUPPRESSing control. Becomes SF-like in BEHAVIOR
(discriminative hierarchy) via our own knobs — not by copying SF constants. Each candidate config is scored by
subprocessing ks_separation.py (fresh engine per config, since Config is read at import). Hard constraints:
reject if FIREOLD drops below FLOOR_HOLD or SUPPRESS rises above SUP_CAP. Prints the progression + best config."""
import os, sys, subprocess, re
THIS = os.path.dirname(os.path.abspath(__file__))
SEP = os.path.join(THIS, "ks_separation.py")
PY = sys.executable

FLOOR_HOLD = 1.50   # FIREOLD must stay at/above this (don't break KS v1's danger set)
SUP_CAP    = 0.22   # SUPPRESS must stay at/below this (don't wake calm noise)

# Candidate values per knob (shipped default first). SF-like priorities: safe-check dominant, floor down,
# knee up (extend quadratic), attacker weights up, coffin on, magnitude up.
GRID = [
    ("KS_SAFE_CHECK", [3, 6, 9, 12, 16, 20]),
    ("KS_FLOOR",      [13, 10, 8, 6, 4]),
    ("KS_KNEE",       [12, 16, 20, 26, 32]),
    ("KS_INTERACT",   [0, 2, 4, 6]),
    ("KS_ATT_ROOK",   [3, 5, 7]),
    ("KS_ATT_QUEEN",  [5, 8, 12]),
    ("KS_ATT_KNIGHT", [2, 4, 6]),
    ("KS_DEFENDER",   [0, 1, 2]),
    ("KS_DIVISOR",    [4, 6, 8]),
    ("KING_SAFETY_MAG", [3000, 4000, 5000]),
]
SHIPPED = {k: vals[0] for k, vals in GRID}

def score(cfg):
    args = [PY, SEP] + ["%s=%s" % (k, v) for k, v in cfg.items()]
    try:
        out = subprocess.run(args, capture_output=True, text=True, timeout=180,
                             env={**os.environ, "TF_CPP_MIN_LOG_LEVEL": "3"}).stdout
    except Exception:
        return None
    m = re.search(r"fireold=([\d.]+) firenew=([\d.]+) suppress=([\d.]+) score=([\-\d.]+)", out)
    if not m:
        return None
    fo, fn, su, sc = (float(m.group(i)) for i in (1, 2, 3, 4))
    if fo < FLOOR_HOLD or su > SUP_CAP:      # hard constraints
        return (fo, fn, su, -999.0)
    return (fo, fn, su, sc)

cur = dict(SHIPPED)
base = score(cur)
print("SHIPPED: fireold=%.3f firenew=%.3f suppress=%.3f score=%.3f" % base, flush=True)
best_score = base[3]
for it in range(3):
    improved = False
    for k, vals in GRID:
        best_v, best_r = cur[k], score(cur)
        for v in vals:
            if v == cur[k]:
                continue
            trial = dict(cur); trial[k] = v
            r = score(trial)
            if r and r[3] > best_r[3]:
                best_r, best_v = r, v
        if best_v != cur[k]:
            cur[k] = best_v; improved = True
            print("pass%d %-16s -> %-5s  fireold=%.3f firenew=%.3f suppress=%.3f score=%.3f" % (
                it, k, best_v, best_r[0], best_r[1], best_r[2], best_r[3]), flush=True)
            best_score = best_r[3]
    if not improved:
        break

fin = score(cur)
print("\nBEST CONFIG:", " ".join("%s=%s" % (k, v) for k, v in cur.items() if v != SHIPPED[k]), flush=True)
print("BEST: fireold=%.3f firenew=%.3f suppress=%.3f score=%.3f  (shipped firenew=%.3f)" % (
    fin[0], fin[1], fin[2], fin[3], base[1]), flush=True)
