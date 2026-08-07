# -*- coding: utf-8 -*-
"""JOINT descent over the WHOLE eval on the 23k corpus — win%-error vs SF18, guards enforced.

Differs from `pawn_fit_shipped.py` in the two ways that matter:

1. **Seeds from the SHIPPED DEFAULTS**, not from a previous winner. `pawn_fit_shipped.py` starts at the
   iteration-2 pawn values; those were fitted to a 2,713-row corpus that no longer exists, and an optimum
   belongs to its corpus. Starting there would bias the search toward a solution for a dead objective.
2. **Spans the whole eval, not one subsystem.** Pawn credit and minor-piece value are the same ratio
   measured from opposite ends (ours 3.37-3.62 pawns per minor vs 3.60-4.01 across three references), so
   moving one while holding the other fixed is actively wrong. Kaufman is in for the same reason — it is
   the existing counterweight to the imbalance the closedness term also targets.

⚠️ One evaluation costs ~3.1 s on 23,113 rows, so a 60-knob pass is ~6 min. Runtime is NOT a reason to
narrow the knob set; an earlier plan trimmed it on an unmeasured assumption and was wrong.

🚨 A good `val` here is NOT an Elo prediction. Four pawn arms improved this objective monotonically
(-7.3 -> -13.98 -> -17.4 -> -21.85) and returned ~0 Elo. Treat the mechanism gates as the durable output
and the constants as provisional -- the eval still violates colour antisymmetry in 57.4% of positions, so
the exact values will move once that is cleaned up.

  pyrun diagnostics/joint_fit.py [PASSES=6] [GUARD_TOL=1.0] [CORPUS=...]
"""
import os, sys, subprocess

for _a in sys.argv[1:]:
    if '=' in _a:
        _k, _v = _a.split('=', 1)
        os.environ.setdefault(_k, _v)

THIS = os.path.dirname(os.path.abspath(__file__))
ENGINE = os.path.dirname(THIS)
WORKER = os.path.join(THIS, "_ks_fit_eval.py")
CORPUS = os.environ.get("CORPUS", "diagnostics/ks_sets/diverse_corpus_wide.csv")
PASSES = int(os.environ.get("PASSES", "6"))
TOL = float(os.environ.get("GUARD_TOL", "1.0"))
PY = sys.executable
TARGET = "ALL"
KNOWN_GUARDS = ["target", "calm", "working", "crowded_safe", "sts_guard", "diverse", "ks_edge",
                "passer_under_fire", "passer_blowup_guard", "passer_control"]

# Seeded EMPTY = shipped defaults. Every knob below is free to move from there.
cur = {}

# 🚨 FORCE: pin knobs ON and REMOVE them from the grid, so their sub-weights actually get tuned.
# Coordinate descent cannot discover a GATED mechanism whose weights are separate knobs: it tries the
# gate at whatever the sub-weights currently are (my hand-guessed defaults), and if that one point is
# worse than off, the gate stays off -- after which the sub-weights are INERT and show zero gradient, so
# they are never explored. For ENABLE_CLOSEDNESS it is worse than unfair: its tables default to all-zero,
# so gate-on is byte-identical to gate-off and a gain is impossible by construction.
# ⇒ To test a mechanism honestly, force it on, descend over its weights, and compare the RESULT against
# the gate-off run. "Rejected by the joint descent" is not evidence about the mechanism itself.
#   FORCE="ENABLE_WINNABILITY=1,ENABLE_CLOSEDNESS=1"
for _f in [x for x in os.environ.get("FORCE", "").split(",") if x.strip()]:
    _k, _v = _f.split("=", 1)
    cur[_k.strip()] = int(_v)

GRID = {
    # ---- NEW MECHANISMS. These gates are the durable output: their verdicts survive the pending
    # colour-symmetry cleanup far better than any fitted constant does.
    "ENABLE_WINNABILITY": [0, 1],
    "WINNAB_SCALE": [5, 10, 18], "WINNAB_BASE": [80, 110, 150],
    "WINNAB_PAWNS": [8, 12, 16], "WINNAB_NO_NPM": [30, 51, 75],
    "WINNAB_OUTFLANK": [5, 9, 14], "WINNAB_FLANKS": [12, 21, 30],
    "WINNAB_TENSION": [0, 6],                       # ours; no SF/Ethereal analogue
    "ENABLE_CLOSEDNESS": [0, 1],
    "CLOSED_N1": [-40, 0, 40], "CLOSED_N4": [-40, 0, 60], "CLOSED_N7": [0, 60, 120],
    "CLOSED_R1": [0, 40, 90],  "CLOSED_R4": [-40, 0, 40], "CLOSED_R7": [-120, -60, 0],
    "CLOSED_B4": [85, 100, 115], "CLOSED_B7": [70, 100, 130],
    "ENABLE_ENDGAME_SCALE": [0, 1],                 # BUILT, WIRED, NEVER ONCE RUN
    "PHASE_BLEND_RANGE": [24, 30],                  # 24 closes the ~20% blend step
    "PHASE_BLEND_LO": [34, 40, 46],

    # ---- NEWLY REACHABLE CONSTANTS (were hardcoded literals, never fitted)
    # ⚠️ WIDENED after pass 1: ~17 of 25 winners pinned at a grid edge, so the first run's optimum was a
    # property of my bounds, not of the objective. A pinned knob needs a plateau check before it is
    # believed (`swept-knob-needs-plateau-check`). Ranges now extend well past every previous winner.
    "EG_EXIST_KNIGHT": [0, 60, 120, 200, 300], "EG_EXIST_BISHOP": [0, 70, 150, 250, 360],
    "EG_EXIST_ROOK": [120, 250, 350, 470],     "EG_EXIST_QUEEN": [300, 500, 700, 900, 1150],
    "MG_CLAMP_KNIGHT": [3000, 3750, 4500, 5500, 7000],
    "MG_CLAMP_BISHOP_A": [3100, 3850, 4600, 5600, 7000],
    "MG_CLAMP_BISHOP_B": [3200, 4000, 4800, 5800, 7200],

    # ---- IMBALANCE / MATERIAL CONTEXT (same family as closedness; must move together)
    "KAUFMAN_SCALE": [40, 70, 100, 130, 170],
    "IMBALANCE_SCALE": [0, 1, 2, 3, 4],
    "PV_BOOST_TRIGGER": [1100, 1500, 2000, 2800, 4000],  # ⚠️ the amplifier; least trustworthy optimum
    "PV_BOOST_MAG": [0, 2500, 5000, 7500, 10000, 13000],

    # ---- PAWN SURFACE (built + gated in earlier sessions, all default-off / default-100)
    "ISOLATED_PAWN_PEN": [0, 80, 140, 200], "BACKWARD_PAWN_PEN": [0, 80, 140, 200],
    "PAWN_CLAMP_MID": [60, 100, 140, 225, 300], "PAWN_CLAMP_EG": [40, 70, 100, 175, 250],
    "SCALE_PAWN_RANK": [30, 50, 70, 100, 130], "SCALE_PASSED_RANK": [30, 50, 70, 100, 130],
    "SCALE_ENDGAME_RANK": [40, 70, 100, 130],
    "SCALE_PAWN_WALL": [40, 70, 100, 130], "SCALE_PAWN_CHAIN": [70, 100, 130, 170, 220],
    "EG_PHALANX": [40, 70, 100, 140], "EG_SUPPORT": [50, 95, 135, 180],
    "EG_DEFEND": [30, 55, 80, 115, 155], "EG_LATENT": [30, 50, 80, 120, 170],
    "STRUCT_OPPOSED_MG_PCT": [40, 60, 80, 100], "STRUCT_OPPOSED_EG_PCT": [40, 60, 80, 100],

    # ---- PASSER
    "PASSER_MAG_SCALE": [50, 80, 100, 130, 170], "PASSER_R_CAP": [256, 320, 400, 500, 640],
    "PASSER_R_MAX": [384, 448, 560, 700], "PASSER_R_FLOOR": [0, 32, 64, 110, 170],
    "PPS_OWN_BLOCK": [45, 75, 110, 160, 220], "PPS_ENEMY_BLOCK": [40, 70, 100, 140, 190],
    "PPS_OWN_ATTACK": [15, 35, 60, 90], "PPS_ENEMY_ATTACK": [15, 30, 50, 75],
    "PP_DIAG_SUPPORT": [0, 20, 45, 75, 110], "PP_FILE_CLEAR": [30, 60, 100, 150, 210],
    "PP_HORIZ_SUPPORT": [60, 110, 160, 225, 300], "PP_BLOCKADE_PEN": [30, 60, 100, 150],

    # ---- KS as a KNOB, not a seed assumption
    "MOD_KS_REALIZ": [0, 128, 256, 384, 512], "KS_REALIZ_FLOOR": [64, 128, 200, 300, 420],
    "SCALE_THREATS": [35, 55, 75, 100, 130], "THREAT_PER_TARGET_CAP": [400, 600, 800, 1100, 1500],
    "SCALE_CAPTURE_GAINS": [50, 80, 100, 120, 150],
}


def evaluate(cfg):
    args = [PY, WORKER, "CORPUS=" + CORPUS] + ["%s=%d" % (k, v) for k, v in cfg.items()]
    out = subprocess.run(args, capture_output=True, text=True, cwd=ENGINE).stdout
    d = {}
    for line in out.splitlines():
        if line.startswith("FIT"):
            for part in line.replace("FIT", "").split("|"):
                for tok in part.split():
                    if "=" in tok:
                        k, v = tok.split("=", 1)
                        try:
                            d[k.replace("train_mse", "ALL.train").replace("val_mse", "ALL.val")] = float(v)
                        except ValueError:
                            pass
    return d


def g(d, k):
    return d.get(k, float("inf"))


print("JOINT descent, %d knobs, corpus %s" % (len(GRID), os.path.basename(CORPUS)), flush=True)
base = evaluate(cur)
# ⚠️ Label honestly. When FORCE seeds gates on, `cur` is NOT the shipped default and calling it that
# invites reading a forced run's improvement against the wrong reference -- the two differ by ~10 points,
# which is larger than several real effects measured this session.
_lbl = "shipped default" if not cur else "SEED (%s)" % " ".join("%s=%d" % kv for kv in sorted(cur.items()))
print("%s : ALL.train=%.3f  ALL.val=%.3f" % (_lbl, g(base, "ALL.train"), g(base, "ALL.val")), flush=True)
if cur:
    _true = evaluate({})
    print("TRUE shipped default : ALL.train=%.3f  ALL.val=%.3f   <- compare against THIS"
          % (g(_true, "ALL.train"), g(_true, "ALL.val")), flush=True)
GUARDS = [t for t in KNOWN_GUARDS if (t + ".train") in base]
base_guard = {t: g(base, t + ".train") for t in GUARDS}
print("guards: %s  (tol %.2f)" % (",".join(GUARDS), TOL), flush=True)

best = g(base, TARGET + ".train")
for p in range(PASSES):
    moved = 0
    for knob, values in GRID.items():
        for v in values:
            if cur.get(knob) == v:
                continue
            trial = dict(cur); trial[knob] = v
            d = evaluate(trial)
            ok = all(g(d, t + ".train") <= base_guard[t] + TOL for t in GUARDS)
            if g(d, TARGET + ".train") < best - 1e-4 and ok:
                best = g(d, TARGET + ".train"); cur[knob] = v; moved += 1
                print("  p%d %-26s -> %-6s ALL.train=%.3f" % (p, knob, v, best), flush=True)
    print("[pass %d] %d moves, ALL.train=%.3f" % (p, moved, best), flush=True)
    if not moved:
        print("converged", flush=True)
        break

final = evaluate(cur)
print("\nBEST CONFIG:\n" + " ".join("%s=%d" % (k, v) for k, v in sorted(cur.items())), flush=True)
print("\nALL.train=%.3f  ALL.val=%.3f   (shipped default %.3f / %.3f)"
      % (g(final, "ALL.train"), g(final, "ALL.val"), g(base, "ALL.train"), g(base, "ALL.val")), flush=True)
print("\nPER-TIER (train / val) -- a gain bought by wrecking a guard tier must be visible here:", flush=True)
for t in GUARDS + ["ALL"]:
    print("  %-22s %9.3f  %9.3f   (base %9.3f / %9.3f)"
          % (t, g(final, t + ".train"), g(final, t + ".val"),
             g(base, t + ".train"), g(base, t + ".val")), flush=True)
