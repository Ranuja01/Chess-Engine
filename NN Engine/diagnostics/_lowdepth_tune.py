# -*- coding: utf-8 -*-
"""LOW-DEPTH SEARCH tuner: coordinate-descend the bounded knobs to MAXIMIZE SF-best move match at a fixed low
depth -- the Elo-aligned MOVE-CHOICE objective, not scalar static fit (which is anti-correlated with Elo).

Coarse-to-fine method: seed from the STATIC-best config (already in the right region), then sharpen on the
searched move choice. SF-best is cached once (_build_lowdepth_set.py); each candidate re-runs ONLY our
fixed-depth search (~60ms/pos at D7), so a full descent over ~500 positions is ~15-25 min.

  pyrun diagnostics/_lowdepth_tune.py [SET=ks_sets/lowdepth_tuneset.csv] [DEPTH=7] [PASSES=2] [N=0]

⚠️ FIXED depth (PRESET=LONG_FORMAT) so the objective is deterministic -- a time budget makes the baseline
wander and drowns the signal. Validate the winner on move-match target+holdout + STS, then SPRT. Low depth
is a PROXY for game depth; games decide.
"""
import os, sys, csv, subprocess

# argv KNOB=VAL -> env FIRST (before any engine import, for the worker path).
for _a in sys.argv[1:]:
    if '=' in _a:
        _k, _v = _a.split('=', 1)
        os.environ[_k] = _v

# ---------------- WORKER: one knob-config, count SF-best matches over the set at fixed depth ----------------
if os.environ.get("WORKER") == "1":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['PRESET'] = 'LONG_FORMAT'                 # fixed depth (deterministic)
    os.environ['MAX_DEPTH'] = os.environ.get('DEPTH', '7')
    os.environ['USE_OPENING_BOOK'] = '0'
    THIS = os.path.dirname(os.path.abspath(__file__)); ENGINE = os.path.dirname(THIS)
    sys.path.insert(0, ENGINE); sys.path.insert(0, THIS); sys.path.insert(0, os.path.join(ENGINE, "selfplay"))
    import chess
    from tactical_test import run_one
    SET = os.environ["SET"]; NLIM = int(os.environ.get("N", "0"))
    rows = [(r["fen"], r["sf_best"]) for r in csv.DictReader(open(SET, newline=""))
            if r.get("fen") and r.get("sf_best")]
    if NLIM:
        rows = rows[:NLIM]
    m = 0
    for fen, best in rows:
        try:
            r = run_one(fen, set())
        except Exception:
            continue
        if r["uci"] == best:
            m += 1
    print("MATCH match=%d total=%d" % (m, len(rows)))
    sys.exit(0)

# ---------------- DRIVER: coordinate descent, seeded from the static-best ----------------
THIS = os.path.dirname(os.path.abspath(__file__)); ENGINE = os.path.dirname(THIS)
PY = sys.executable
SET = os.environ.get("SET", "ks_sets/lowdepth_tuneset.csv")
if not os.path.isabs(SET):
    SET = os.path.join(THIS, SET)
DEPTH = os.environ.get("DEPTH", "7")
PASSES = int(os.environ.get("PASSES", "2"))
N = os.environ.get("N", "0")

# The screened STATIC-best (batched de-collinearisation seed). Modes pinned on; CAP/KNEE are tuned.
SEED = {"OVD_BOUNDED_MODE": 2, "OVD_CAP": 300, "OVD_KNEE": 40,
        "CENTRAL_BOUNDED_MODE": 1, "CENTRAL_CAP": 150, "CENTRAL_KNEE": 200}
# Grid AROUND the seed (sharpening, not searching) -> few candidates.
GRID = {"OVD_CAP": [200, 300, 400], "OVD_KNEE": [20, 40, 80],
        "CENTRAL_CAP": [125, 150, 175, 200], "CENTRAL_KNEE": [150, 200, 300]}


def evaluate(cfg):
    env = dict(os.environ, WORKER="1", SET=SET, DEPTH=DEPTH, N=N)
    args = [PY, "-u", os.path.abspath(__file__)] + ["%s=%s" % (k, v) for k, v in cfg.items()]
    out = subprocess.run(args, capture_output=True, text=True, cwd=ENGINE, env=env).stdout
    for line in out.splitlines():
        if line.startswith("MATCH"):
            d = dict(tok.split("=") for tok in line.split()[1:])
            return int(d["match"]), int(d["total"])
    return -1, 0


default_m, tot = evaluate({"OVD_BOUNDED_MODE": 0, "CENTRAL_BOUNDED_MODE": 0})   # shipped default (modes off)
cur = dict(SEED)
seed_m, tot = evaluate(cur)
best_m = seed_m
print("SET=%s  DEPTH=%s  N=%s" % (os.path.basename(SET), DEPTH, tot), flush=True)
print("default(modes off) match=%d/%d (%.1f%%)   static-seed match=%d/%d (%.1f%%)"
      % (default_m, tot, 100.0*default_m/max(1, tot), seed_m, tot, 100.0*seed_m/max(1, tot)), flush=True)

for p in range(PASSES):
    moved = 0
    for knob, vals in GRID.items():
        for v in vals:
            if cur.get(knob) == v:
                continue
            trial = dict(cur); trial[knob] = v
            m, _ = evaluate(trial)
            if m > best_m:
                best_m = m; cur[knob] = v; moved += 1
                print("  p%d %-14s -> %-5s match=%d/%d (%.1f%%)" % (p, knob, v, best_m, tot, 100.0*best_m/max(1, tot)), flush=True)
    print("[pass %d] %d moves, match=%d/%d" % (p, moved, best_m, tot), flush=True)
    if not moved:
        print("converged", flush=True)
        break

print("\nBEST CONFIG: %s" % " ".join("%s=%s" % (k, v) for k, v in sorted(cur.items())), flush=True)
print("match=%d/%d (%.1f%%)   [static-seed %d (%.1f%%), default %d (%.1f%%)]"
      % (best_m, tot, 100.0*best_m/max(1, tot), seed_m, 100.0*seed_m/max(1, tot),
         default_m, 100.0*default_m/max(1, tot)), flush=True)
