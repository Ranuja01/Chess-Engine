# -*- coding: utf-8 -*-
"""Trace the KS `units` distribution on real positions to CONFIRM the dead-quadratic empirically before any
curve reshape (analysis open-risk: prove live positions land past the knee, don't act on the story).

Our danger curve: `danger = units^2/KS_DIVISOR` for units <= KS_KNEE(12), linear above; then `if units <
KS_FLOOR(13) return 0`. Since FLOOR(13) > KNEE(12), the quadratic region is [FLOOR..KNEE] = EMPTY, so every
surviving position is on the linear branch. This tool prints the units histogram to show (a) how many
positions land in the deadzone (<13), (b) how many clear it and by how much (the linear range), and (c) that
essentially none sit in the would-be-quadratic band -> confirms the finding + sizes the FLOOR/KNEE fix.

Reads KSD dump lines emitted by king_safety_danger under g_capture_eval_breakdown + KS_DEBUG_DUMP.

  pyrun diagnostics/_ks_unit_trace.py [N=400]
"""
import os, sys, csv, subprocess, re

for _a in sys.argv[1:]:
    if '=' in _a:
        k, v = _a.split('=', 1); os.environ[k] = v

THIS = os.path.dirname(os.path.abspath(__file__)); ENGINE = os.path.dirname(THIS)

if os.environ.get("WORKER") == "1":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['KS_DEBUG_DUMP'] = '1'
    sys.path.insert(0, ENGINE); sys.path.insert(0, THIS)
    import chess
    from ChessAI import ChessAI
    ai = ChessAI(None, None, chess.Board(), True)
    N = int(os.environ.get("N", "400"))
    fens = []
    # ks_attack collapses = real attacked-king positions (the "lost king" cases that SHOULD light the curve)
    cp = os.path.join(THIS, "ks_sets", "collapse_dataset_classified.csv")
    if os.path.exists(cp):
        for r in csv.DictReader(open(cp, newline="")):
            if r.get("ks_class") == "ks_attack":
                f = (r.get("decision_fen") or "").strip()
                if f: fens.append(f)
    # + general game positions for contrast
    gp = os.path.join(THIS, "ks_sets", "game_regret_set.csv")
    if os.path.exists(gp):
        for i, r in enumerate(csv.DictReader(open(gp, newline=""))):
            if i % 37 == 0:
                f = (r.get("fen") or "").strip()
                if f: fens.append(f)
    fens = fens[:N]
    for f in fens:
        try:
            ai.ev_breakdown(chess.Board(f))   # emits KSD lines to stderr
        except Exception:
            pass
    sys.exit(0)

# Driver: run the worker, capture stderr KSD lines, histogram units.
env = dict(os.environ, WORKER="1")
p = subprocess.run([sys.executable, "-u", os.path.abspath(__file__)],
                   capture_output=True, text=True, cwd=ENGINE, env=env)
units = [int(m) for m in re.findall(r"units=(\d+)", p.stderr)]
if not units:
    print("no KSD lines captured (is KS_DEBUG_DUMP wired + g_capture_eval_breakdown true in ev_breakdown?)")
    sys.exit(0)
KNEE, FLOOR = 12, 13
bins = {"0 (no danger)": 0, "1-12 (quadratic-eligible IF floor<=knee)": 0, "13-24 (low linear)": 0,
        "25-49 (mid linear)": 0, "50-80 (high linear)": 0, "81+ (capped)": 0}
for u in units:
    if u == 0: bins["0 (no danger)"] += 1
    elif u <= 12: bins["1-12 (quadratic-eligible IF floor<=knee)"] += 1
    elif u <= 24: bins["13-24 (low linear)"] += 1
    elif u <= 49: bins["25-49 (mid linear)"] += 1
    elif u <= 80: bins["50-80 (high linear)"] += 1
    else: bins["81+ (capped)"] += 1
n = len(units)
print("KS units distribution over %d king-evals  (KNEE=%d FLOOR=%d => quadratic band [13..12] is EMPTY)\n" % (n, KNEE, FLOOR))
for k, c in bins.items():
    print("  %-42s %6d  (%.1f%%)" % (k, c, 100.0 * c / n))
survivors = [u for u in units if u >= FLOOR]
print("\n  clear the floor (>=13, ON the live linear branch): %d (%.1f%%)" % (len(survivors), 100.0*len(survivors)/n))
if survivors:
    survivors.sort()
    print("  surviving units:  min=%d  median=%d  p90=%d  max=%d" %
          (survivors[0], survivors[len(survivors)//2], survivors[int(len(survivors)*0.9)], survivors[-1]))
print("\n  read: if the 1-12 band is ~empty among DANGEROUS kings and survivors cluster in 13-40, the quadratic\n"
      "  never fires and the live danger is a shallow linear ramp -> confirms the dead-quadratic, sizes the fix.", flush=True)
