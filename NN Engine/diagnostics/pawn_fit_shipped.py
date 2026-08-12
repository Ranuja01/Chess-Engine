# -*- coding: utf-8 -*-
"""Pawn/passer win%-descent in the SHIPPED regime, on the widened corpus.

Differs from ks_fit_wholesystem.py in three ways that matter:
  1. `cur` starts from the SHIPPED DEFAULTS — no ENABLE_KS_CHECK_V2, no retuned KING_SAFETY_MAG. Earlier
     runs optimised inside a regime we do not ship, so their winning config could not be applied directly.
  2. The grid is PAWN/PASSER ONLY. The KS block is excluded so nothing outside the subsystem moves.
  3. It is seeded from the iteration-2 winners, so the descent refines a known-good point rather than
     re-deriving it — necessary because the widened corpus makes each evaluation ~2x slower.

Guards are unchanged: no tier may rise above its baseline by more than GUARD_TOL, so a pawn gain cannot be
bought by wrecking calm/working/sts_guard/passer positions.

⚠️ The corpus was rebuilt (2,713 -> 4,987 rows), so `val` here is NOT comparable to any earlier number.
An optimum belongs to its corpus.

  pyrun diagnostics/pawn_fit_shipped.py [ROUNDS=1] [GUARD_TOL=1.0]
"""
import os, sys, subprocess, re
os.environ.setdefault('KS_SAFE_CHECK_DEF', '5')
ROUNDS = int(os.environ.get('ROUNDS', '1'))
TOL = float(os.environ.get('GUARD_TOL', '1.0'))
THIS = os.path.dirname(os.path.abspath(__file__))
CORPUS = os.path.join(THIS, "ks_sets", "diverse_corpus_wide.csv")
WORKER = os.path.join(THIS, "_ks_fit_eval.py")
PY = sys.executable

# Seeded from the iteration-2 winners. Only pawn/passer knobs appear; everything else stays at the
# shipped default, which is the whole point of this script.
cur = {
    "ENABLE_PASSER_V3": 1,
    "SCALE_PAWN_RANK": 70, "SCALE_PASSED_RANK": 70, "SCALE_ENDGAME_RANK": 85,
    "SCALE_PAWN_WALL": 80, "SCALE_PAWN_CHAIN": 80,
    "RANK_DEF_R2": 125, "RANK_DEF_R3": 75, "RANK_DEF_R4": 100, "RANK_DEF_R5": 125,
    "RANK_DEF_R6": 75, "RANK_DEF_R7": 100,
    "RANK_PSD_R2": 75, "RANK_PSD_R3": 75, "RANK_PSD_R4": 75, "RANK_PSD_R5": 75,
    "RANK_PSD_R6": 125, "RANK_PSD_R7": 125,
    "RANK_EG_R2": 125, "RANK_EG_R3": 125, "RANK_EG_R4": 75, "RANK_EG_R5": 75,
    "RANK_EG_R6": 125, "RANK_EG_R7": 125,
    "CHAIN_F_A": 80, "CHAIN_F_B": 80, "CHAIN_F_C": 80, "CHAIN_F_D": 100,
    "CHAIN_F_E": 100, "CHAIN_F_F": 80, "CHAIN_F_G": 130, "CHAIN_F_H": 130,
    "WALL_F_A": 80, "WALL_F_B": 80, "WALL_F_C": 130, "WALL_F_D": 130,
    "WALL_F_E": 80, "WALL_F_F": 130, "WALL_F_G": 80, "WALL_F_H": 80,
    "EG_PHALANX": 100, "EG_SUPPORT": 95, "EG_DEFEND": 80, "EG_LATENT": 80,
    "STRUCT_R_MG_R3": 100, "STRUCT_R_MG_R4": 80, "STRUCT_R_MG_R5": 80, "STRUCT_R_MG_R6": 80,
    "STRUCT_R_EG_R3": 80, "STRUCT_R_EG_R4": 80, "STRUCT_R_EG_R5": 80, "STRUCT_R_EG_R6": 80,
    "STRUCT_OPPOSED_MG_PCT": 60, "STRUCT_OPPOSED_EG_PCT": 60,
    "PAWN_CLAMP_MID": 175, "PAWN_CLAMP_EG": 125,
    "ISOLATED_PAWN_PEN": 120, "BACKWARD_PAWN_PEN": 120,
    "PASSER_MAG_SCALE": 120, "PASSER_R_CAP": 384, "PASSER_R_MAX": 384,
    "PASSER_CONTEST_STOP": 60, "PASSER_CONTEST_PATH": 60, "PASSER_RFLOOR_R6": 130,
    "ENABLE_PASSER_DEFER_ON_FLAG": 0,
    "PP_OPP_PAWN_PEN": 160, "PP_BLOCKADE_PEN": 100, "PP_UNBLOCKED": 50,
    "PP_DIAG_SUPPORT": 50, "PP_FILE_CLEAR": 100, "PP_HORIZ_SUPPORT": 160,
    # NEVER TUNED BEFORE — the passed-pawn SUPPORT magnitudes, previously hardcoded literals.
    "PPS_OWN_BLOCK": 75, "PPS_ENEMY_BLOCK": 100, "PPS_OWN_ATTACK": 60, "PPS_ENEMY_ATTACK": 50,
    "SCALE_PASSED_PAWN": 100,
}

# Two alternatives per knob keeps one round inside the time budget on the widened corpus.
GRID = {
    "SCALE_PAWN_RANK": [55, 85], "SCALE_PASSED_RANK": [55, 85], "SCALE_ENDGAME_RANK": [70, 100],
    "SCALE_PAWN_WALL": [65, 95], "SCALE_PAWN_CHAIN": [65, 100],
    "RANK_PSD_R5": [60, 90], "RANK_PSD_R6": [110, 145], "RANK_PSD_R7": [110, 145],
    "RANK_EG_R5": [60, 90], "RANK_EG_R6": [110, 145], "RANK_EG_R7": [110, 145],
    "RANK_DEF_R2": [105, 145], "RANK_DEF_R5": [105, 145],
    "EG_PHALANX": [75, 130], "EG_SUPPORT": [75, 120], "EG_DEFEND": [60, 105], "EG_LATENT": [60, 105],
    "STRUCT_R_MG_R4": [65, 100], "STRUCT_R_MG_R5": [65, 100], "STRUCT_R_MG_R6": [65, 100],
    "STRUCT_R_EG_R4": [65, 100], "STRUCT_R_EG_R5": [65, 100], "STRUCT_R_EG_R6": [65, 100],
    "STRUCT_OPPOSED_MG_PCT": [40, 80], "STRUCT_OPPOSED_EG_PCT": [40, 80],
    "PAWN_CLAMP_MID": [140, 210], "PAWN_CLAMP_EG": [100, 155],
    "ISOLATED_PAWN_PEN": [80, 165], "BACKWARD_PAWN_PEN": [80, 165],
    "PASSER_MAG_SCALE": [100, 145], "PASSER_R_CAP": [320, 448], "PASSER_R_MAX": [448],
    "PASSER_CONTEST_STOP": [40, 90], "PASSER_CONTEST_PATH": [40, 85], "PASSER_RFLOOR_R6": [90, 170],
    "ENABLE_PASSER_DEFER_ON_FLAG": [1],
    "PP_OPP_PAWN_PEN": [125, 195], "PP_BLOCKADE_PEN": [75, 135], "PP_UNBLOCKED": [30, 80],
    "PP_DIAG_SUPPORT": [30, 80], "PP_FILE_CLEAR": [70, 140], "PP_HORIZ_SUPPORT": [120, 210],
    # The untuned support surface, opened wide because it has no prior.
    "PPS_OWN_BLOCK": [45, 110], "PPS_ENEMY_BLOCK": [65, 145],
    "PPS_OWN_ATTACK": [35, 90], "PPS_ENEMY_ATTACK": [25, 80],
    "SCALE_PASSED_PAWN": [80, 125],
}

TARGET = "ALL"
KNOWN_GUARDS = ["target", "calm", "working", "crowded_safe", "sts_guard", "diverse", "ks_edge",
                "passer_under_fire", "passer_blowup_guard", "passer_control"]


def evaluate(cfg):
    args = [PY, WORKER, "CORPUS=" + CORPUS] + ["%s=%d" % (k, v) for k, v in cfg.items()]
    out = subprocess.run(args, capture_output=True, text=True).stdout
    d = {}
    for k, v in re.findall(r"(\w+\.\w+)=([\d.]+)", out):
        d[k] = float(v)
    m = re.search(r"train_mse=([\d.]+) val_mse=([\d.]+)", out)
    if m:
        d["ALL.train"], d["ALL.val"] = float(m.group(1)), float(m.group(2))
    return d


def g(d, key):
    return d.get(key, 9e9)


print("SHIPPED-REGIME pawn descent on %d-row corpus" % sum(1 for _ in open(CORPUS)))
shipped = evaluate({})
print("shipped default : ALL.train=%.3f  ALL.val=%.3f" % (g(shipped, "ALL.train"), g(shipped, "ALL.val")))
base_d = evaluate(cur)
GUARDS = [t for t in KNOWN_GUARDS if (t + ".train") in base_d]
base_guard = {t: g(base_d, t + ".train") for t in GUARDS}
best = g(base_d, TARGET + ".train")
print("iter2 seed      : ALL.train=%.3f  ALL.val=%.3f" % (best, g(base_d, "ALL.val")))
sys.stdout.flush()

for rnd in range(ROUNDS):
    print("\n=== round %d ===" % (rnd + 1)); sys.stdout.flush()
    for knob, cands in GRID.items():
        for v in cands:
            if v == cur.get(knob):
                continue
            trial = dict(cur); trial[knob] = v
            d = evaluate(trial)
            ok = all(g(d, t + ".train") <= base_guard[t] + TOL for t in GUARDS)
            if g(d, TARGET + ".train") < best - 1e-4 and ok:
                best = g(d, TARGET + ".train"); cur[knob] = v
                print("  %-26s -> %-5s (ALL.train=%.3f)" % (knob, v, best)); sys.stdout.flush()

final = evaluate(cur)
print("\nBEST CONFIG: " + " ".join("%s=%d" % (k, v) for k, v in cur.items()))
print("ALL.train=%.3f  ALL.val=%.3f  (shipped default: %.3f / %.3f)"
      % (g(final, "ALL.train"), g(final, "ALL.val"), g(shipped, "ALL.train"), g(shipped, "ALL.val")))
print("guards: %s" % {t: (round(base_guard[t], 2), round(g(final, t + ".train"), 2)) for t in GUARDS})
