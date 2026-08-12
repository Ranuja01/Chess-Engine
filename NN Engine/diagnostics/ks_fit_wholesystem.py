# -*- coding: utf-8 -*-
"""Whole-system constrained win%-descent on the WIDE diverse corpus (build_diverse_wide.py). Minimise the KS
under-read `target` tier SUBJECT TO every guard family (calm/working/crowded_safe/sts_guard/diverse/passer_*/
ks_edge) not rising above baseline by more than GUARD_TOL -- so no term is bought by breaking another, and the
bench-proxy tiers (sts_guard/diverse) can't silently regress (the STS-blindness that sank the KS-only fit).

Actively tunes the KS + geometry (C/D) + static-safe material levers (Kaufman/capgains scale). Passer V3 is
HELD at its game-validated defaults (its knobs are game-tuned; static re-tuning them risks the proxy trap) but
passer positions are GUARDS. ENABLE_KS_CHECK_V2 + ENABLE_PASSER_V3 fixed ON. Reuses _ks_fit_eval.py worker.
  pyrun diagnostics/ks_fit_wholesystem.py [ROUNDS=2] [GUARD_TOL=1.0]
"""
import os, sys, subprocess, re
os.environ.setdefault('KS_SAFE_CHECK_DEF', '5')
ROUNDS = int(os.environ.get('ROUNDS', '2'))
TOL = float(os.environ.get('GUARD_TOL', '1.0'))
THIS = os.path.dirname(os.path.abspath(__file__))
CORPUS = os.path.join(THIS, "ks_sets", "diverse_corpus_wide.csv")
WORKER = os.path.join(THIS, "_ks_fit_eval.py")
PY = sys.executable

GRID = {
    "KS_CHK_QUEEN":     [10, 14, 20],
    "KS_CHK_ROOK":      [10, 14, 20],
    "KS_CHK_BISHOP":    [5, 7, 12],
    "KS_CHK_KNIGHT":    [6, 9, 14],
    "KS_CHK_MULTI":     [0, 4, 8],
    "ENABLE_KS_SF_WEAK":[0, 1],
    "KS_WEAK":          [2, 3, 4],
    "KS_DEFENDER":      [0, 2, 3],
    "KS_NO_QUEEN":      [6, 12, 18],
    "KS_FLOOR":         [3, 6, 9, 13],
    "ENABLE_KS_ZONE_CLAMP": [0, 1],
    "KS_ZONE_NORM":     [0, 9],
    "KS_ATT_BISHOP":    [2, 5],
    # --- attack-layer / de-king (the triple-count root) ---
    "KS_ZONE_ATTACK_PCT": [50, 75, 100],       # de-king: scales the WHOLE king-ring credit (STS+92 at 50)
    "ATTACK_OPEN_MULT": [1, 2, 3, 5],          # the x5 OPEN-SQUARE stacking specifically -- targeted de-stacking
                                               # (keep base ring credit, cut only the hole boost)
    "KING_SAFETY_MAG":  [2500, 3000, 3500],    # KS master scale: must rebalance when de-king removes OvD credit
    "SCALE_ATTACK_LAYER": [80, 100],           # uniform scale on the whole attack layer (base central + king)
    # --- OvD long-term: deliberately OPENED now that de-king removes the shared king-boost channel, so the
    # long-term OvD scaling can settle jointly with the KS scaling instead of fighting a double-counted signal.
    "IMBALANCE_SCALE":  [2, 3, 4],
    # --- previously-FAILED KS knobs, kept INERT at their current value but free to rise if the tune finds a
    # regime where they help (they may have failed only because the attack signal was triple-counted).
    "KS_OVERLOAD":      [0, 1, 2],
    "KS_ATT_PRODUCT":   [0, 1, 2],
    "KS_DYN":           [0, 64, 128],
    # --- placement (the `pieces` term was a top over-read contributor) ---
    "SCALE_PLACE_PAWN":   [90, 100, 110],
    "SCALE_PLACE_KNIGHT": [90, 100, 110],
    "SCALE_PLACE_BISHOP": [90, 100, 110],
    "SCALE_PLACE_QUEEN":  [90, 100, 110],
    "CENTER_INNER_MULT":  [150, 200],
    "CENTER_OUTER_MULT":  [100, 150],
    # --- passers (V3 fixed ON; its 9 core knobs held at game-validated values) ---
    "PASSER_RFLOOR_R5": [0, 100],              # floor-first advanced passers (6th rank)
    "PASSER_RFLOOR_R6": [0, 130],              # (7th rank)
    # --- capgains realizability: REALIZ_* must be non-zero or ENABLE_CAPG_REALIZ is INERT (factor=256=identity) ---
    "ENABLE_CAPG_REALIZ": [0, 1],
    "REALIZ_MAT_K":     [0, 200, 400],
    "REALIZ_MAT_THRESH":[0, 1000, 3000],
    "REALIZ_PHASE_K":   [0, 20],
    "REALIZ_FLOOR":     [128, 192, 256],
    "KAUFMAN_SCALE":    [90, 100, 110],
    "SCALE_CAPTURE_GAINS": [90, 100, 110],
    # --- PAWN MODEL, tuned jointly rather than in its own silo -------------------------------------
    # Measured on 1,621 real-position pawn removals: our marginal pawn value is ~+58 cp of positional
    # value where SF18's is ~-6, i.e. we overpay a pawn at EVERY rank. The whole-table SCALE_* knobs can
    # only rescale a hand-picked shape, so the per-rank knobs below let the win%-descent choose the table
    # entries themselves. All default 100 => byte-identical. Ranks 2..7 only (0 and 7 are unreachable).
    "SCALE_PAWN_RANK":   [70, 85, 100],
    "SCALE_PASSED_RANK": [70, 85, 100],
    "SCALE_ENDGAME_RANK":[70, 85, 100],
    "SCALE_PAWN_WALL":   [80, 100, 120],
    "SCALE_PAWN_CHAIN":  [80, 100, 120],
    "RANK_DEF_R2": [75, 100, 125], "RANK_DEF_R3": [75, 100, 125], "RANK_DEF_R4": [75, 100, 125],
    "RANK_DEF_R5": [75, 100, 125], "RANK_DEF_R6": [75, 100, 125], "RANK_DEF_R7": [75, 100, 125],
    "RANK_PSD_R2": [75, 100, 125], "RANK_PSD_R3": [75, 100, 125], "RANK_PSD_R4": [75, 100, 125],
    "RANK_PSD_R5": [75, 100, 125], "RANK_PSD_R6": [75, 100, 125], "RANK_PSD_R7": [75, 100, 125],
    "RANK_EG_R2":  [75, 100, 125], "RANK_EG_R3":  [75, 100, 125], "RANK_EG_R4":  [75, 100, 125],
    "RANK_EG_R5":  [75, 100, 125], "RANK_EG_R6":  [75, 100, 125], "RANK_EG_R7":  [75, 100, 125],
    # --- Passer V3 core, RELEASED from its hold. The header above kept these fixed on the grounds that
    # they are game-tuned and static re-tuning risks the proxy trap. Released deliberately: the passer path
    # is now known to be structurally bounded (R clamps to [0,384] and PASSER_R_CAP=320 => at most a 1.25x
    # multiplier), and these terms only make sense settled together with the rank tables they multiply.
    # The passer_* tiers remain GUARDS, so the release cannot be bought by wrecking passer positions.
    "PASSER_MAG_SCALE":   [80, 100, 120],
    "PASSER_R_CAP":       [256, 320, 384],
    "PASSER_CONTEST_STOP":[60, 90, 120],
    "PASSER_CONTEST_PATH":[20, 40, 60],
    # --- Structural DETECTORS, built and gated at 0. Previously judged only at hand-picked 200/100 on four
    # game venues whose 95% CIs were each +-35..67 Elo -- i.e. inconclusive, never actually tested. They have
    # never been win%-tuned at all, let alone jointly with the tables they interact with. 0 stays in the grid
    # so the descent can keep them off if they genuinely do not pay.
    "ISOLATED_PAWN_PEN": [0, 40, 80, 120],
    "BACKWARD_PAWN_PEN": [0, 40, 80, 120],
    # --- PER-FILE chain/wall shape. These base values are hand-picked (chain A 10 / B 15 / C 100 / D 150 /
    # E 150 / F 100 / G 15 / H 10) and only a whole-table multiplier ever existed, so the file shape itself
    # has never been fitted. Centre and edge files are free to move independently here.
    "CHAIN_F_A": [80, 100, 130], "CHAIN_F_B": [80, 100, 130], "CHAIN_F_C": [80, 100, 130],
    "CHAIN_F_D": [80, 100, 130], "CHAIN_F_E": [80, 100, 130], "CHAIN_F_F": [80, 100, 130],
    "CHAIN_F_G": [80, 100, 130], "CHAIN_F_H": [80, 100, 130],
    "WALL_F_A": [80, 100, 130], "WALL_F_B": [80, 100, 130], "WALL_F_C": [80, 100, 130],
    "WALL_F_D": [80, 100, 130], "WALL_F_E": [80, 100, 130], "WALL_F_F": [80, 100, 130],
    "WALL_F_G": [80, 100, 130], "WALL_F_H": [80, 100, 130],
    # --- ITERATION 2 -----------------------------------------------------------------------------
    # ENDGAME structural magnitudes. These were hardcoded literals no knob could reach, so endgame pawn
    # structure was flat in file, flat in rank, and had NEVER been fitted once -- while SF's endgame
    # connected term is its most rank-sensitive component. Most unexplored surface in the subsystem.
    "EG_PHALANX": [70, 100, 150], "EG_SUPPORT": [95, 135, 180],
    "EG_DEFEND":  [80, 115, 150], "EG_LATENT":  [25, 50, 80],
    # Per-rank structural sensitivity, an INDEPENDENT curve per phase (two evaluators => no shared shape).
    "STRUCT_R_MG_R3": [80, 100, 125], "STRUCT_R_MG_R4": [80, 100, 125],
    "STRUCT_R_MG_R5": [80, 100, 125], "STRUCT_R_MG_R6": [80, 100, 125],
    "STRUCT_R_EG_R3": [80, 100, 125], "STRUCT_R_EG_R4": [80, 100, 125],
    "STRUCT_R_EG_R5": [80, 100, 150], "STRUCT_R_EG_R6": [80, 100, 150],
    # `opposed` -- the signal getPPIncrement computes and discards. SF scales connected by (2+phalanx-opposed).
    "STRUCT_OPPOSED_MG_PCT": [60, 80, 100], "STRUCT_OPPOSED_EG_PCT": [60, 80, 100],
    # The per-pawn caps, tunable rather than assumed. Raising them alone hurt; the descent decides in context.
    "PAWN_CLAMP_MID": [175, 225, 300], "PAWN_CLAMP_EG": [125, 175, 250],
    # Realizability headroom: PASSER_R_CAP pinned itself at the internal clamp, so both must move together.
    "PASSER_R_MAX": [384, 512], "PASSER_R_CAP": [320, 384, 512],
    # Correctness gate: makes the deferred set equal the flagged set. Precondition for the PP_* knobs below,
    # since `ppInc>=100 <=> flagged` holds only while PP_BLOCKADE_PEN == 100.
    "ENABLE_PASSER_DEFER_ON_FLAG": [0, 1],
    # getPPIncrement's own hand-picked constants -- never tuned, and doubly load-bearing because ppIncrement
    # both selects the rank table and feeds the bonus via >>3 / >>2.
    "PP_OPP_PAWN_PEN": [90, 125, 160], "PP_BLOCKADE_PEN": [70, 100, 140],
    "PP_UNBLOCKED": [25, 50, 80], "PP_DIAG_SUPPORT": [50, 75, 110],
    "PP_FILE_CLEAR": [100, 150, 200], "PP_HORIZ_SUPPORT": [160, 225, 290],
}
cur = {"ENABLE_KS_CHECK_V2": 1, "ENABLE_PASSER_V3": 1,
       "KS_CHK_QUEEN": 14, "KS_CHK_ROOK": 14, "KS_CHK_BISHOP": 7, "KS_CHK_KNIGHT": 9, "KS_CHK_MULTI": 0,
       "ENABLE_KS_SF_WEAK": 0, "KS_WEAK": 2, "KS_DEFENDER": 0, "KS_NO_QUEEN": 6, "KS_FLOOR": 13,
       "ENABLE_KS_ZONE_CLAMP": 0, "KS_ZONE_NORM": 0, "KS_ATT_BISHOP": 2,
       "KS_ZONE_ATTACK_PCT": 100, "ATTACK_OPEN_MULT": 5, "KING_SAFETY_MAG": 3000,
       "SCALE_ATTACK_LAYER": 100, "IMBALANCE_SCALE": 3,
       "KS_OVERLOAD": 0, "KS_ATT_PRODUCT": 0, "KS_DYN": 0,
       "SCALE_PLACE_PAWN": 100, "SCALE_PLACE_KNIGHT": 100, "SCALE_PLACE_BISHOP": 100,
       "SCALE_PLACE_QUEEN": 100, "CENTER_INNER_MULT": 200, "CENTER_OUTER_MULT": 150,
       "PASSER_RFLOOR_R5": 0, "PASSER_RFLOOR_R6": 0,
       "ENABLE_CAPG_REALIZ": 0, "REALIZ_MAT_K": 0, "REALIZ_MAT_THRESH": 0,
       "REALIZ_PHASE_K": 0, "REALIZ_FLOOR": 256,
       "KAUFMAN_SCALE": 100, "SCALE_CAPTURE_GAINS": 100,
       "SCALE_PAWN_RANK": 100, "SCALE_PASSED_RANK": 100, "SCALE_ENDGAME_RANK": 100,
       "SCALE_PAWN_WALL": 100, "SCALE_PAWN_CHAIN": 100,
       "RANK_DEF_R2": 100, "RANK_DEF_R3": 100, "RANK_DEF_R4": 100,
       "RANK_DEF_R5": 100, "RANK_DEF_R6": 100, "RANK_DEF_R7": 100,
       "RANK_PSD_R2": 100, "RANK_PSD_R3": 100, "RANK_PSD_R4": 100,
       "RANK_PSD_R5": 100, "RANK_PSD_R6": 100, "RANK_PSD_R7": 100,
       "RANK_EG_R2": 100, "RANK_EG_R3": 100, "RANK_EG_R4": 100,
       "RANK_EG_R5": 100, "RANK_EG_R6": 100, "RANK_EG_R7": 100,
       "PASSER_MAG_SCALE": 100, "PASSER_R_CAP": 320,
       "PASSER_CONTEST_STOP": 90, "PASSER_CONTEST_PATH": 40,
       "ISOLATED_PAWN_PEN": 0, "BACKWARD_PAWN_PEN": 0,
       "CHAIN_F_A": 100, "CHAIN_F_B": 100, "CHAIN_F_C": 100, "CHAIN_F_D": 100,
       "CHAIN_F_E": 100, "CHAIN_F_F": 100, "CHAIN_F_G": 100, "CHAIN_F_H": 100,
       "WALL_F_A": 100, "WALL_F_B": 100, "WALL_F_C": 100, "WALL_F_D": 100,
       "WALL_F_E": 100, "WALL_F_F": 100, "WALL_F_G": 100, "WALL_F_H": 100,
       "EG_PHALANX": 100, "EG_SUPPORT": 135, "EG_DEFEND": 115, "EG_LATENT": 50,
       "STRUCT_R_MG_R3": 100, "STRUCT_R_MG_R4": 100, "STRUCT_R_MG_R5": 100, "STRUCT_R_MG_R6": 100,
       "STRUCT_R_EG_R3": 100, "STRUCT_R_EG_R4": 100, "STRUCT_R_EG_R5": 100, "STRUCT_R_EG_R6": 100,
       "STRUCT_OPPOSED_MG_PCT": 100, "STRUCT_OPPOSED_EG_PCT": 100,
       "PAWN_CLAMP_MID": 225, "PAWN_CLAMP_EG": 175,
       "PASSER_R_MAX": 384, "ENABLE_PASSER_DEFER_ON_FLAG": 0,
       "PP_OPP_PAWN_PEN": 125, "PP_BLOCKADE_PEN": 100, "PP_UNBLOCKED": 50,
       "PP_DIAG_SUPPORT": 75, "PP_FILE_CLEAR": 150, "PP_HORIZ_SUPPORT": 225}

# Minimize OVERALL win%-MSE subject to NO tier regressing (Pareto): de-king improves the over-read GUARDS not the
# `target` under-read tier, so a target-only objective would ignore it. `target` is guarded so KS recovery holds.
TARGET = "ALL"
KNOWN_GUARDS = ["target", "calm", "working", "crowded_safe", "sts_guard", "diverse", "ks_edge",
                "passer_under_fire", "passer_blowup_guard", "passer_control"]


def evaluate(cfg):
    args = [PY, WORKER, "CORPUS=" + CORPUS] + ["%s=%d" % (k, v) for k, v in cfg.items()]
    out = subprocess.run(args, capture_output=True, text=True).stdout
    d = {"_dir": out}
    for k, v in re.findall(r"(\w+\.\w+)=([\d.]+)", out):
        d[k] = float(v)
    m = re.search(r"train_mse=([\d.]+) val_mse=([\d.]+)", out)
    if m:
        d["ALL.train"], d["ALL.val"] = float(m.group(1)), float(m.group(2))
    return d


def g(d, key): return d.get(key, 9e9)

base_d = evaluate(cur)
GUARDS = [t for t in KNOWN_GUARDS if (t + ".train") in base_d]
base_guard = {t: g(base_d, t + ".train") for t in GUARDS}
best_tgt = g(base_d, TARGET + ".train")
print("baseline: target.train=%.3f  ALL.val=%.3f\n  guards=%s" %
      (best_tgt, g(base_d, "ALL.val"), {t: round(base_guard[t], 2) for t in GUARDS}))
for ln in base_d["_dir"].splitlines():
    if ln.startswith("DIR"):
        print("  " + ln)


def guards_ok(d):
    return all(g(d, t + ".train") <= base_guard[t] + TOL for t in GUARDS)


for rnd in range(ROUNDS):
    print("\n=== round %d ===" % (rnd + 1))
    for knob, cands in GRID.items():
        base = cur[knob]
        for v in cands:
            if v == cur[knob]:
                continue
            trial = dict(cur); trial[knob] = v
            d = evaluate(trial)
            if g(d, TARGET + ".train") < best_tgt - 1e-4 and guards_ok(d):
                best_tgt, cur[knob] = g(d, TARGET + ".train"), v
        if cur[knob] != base:
            print("  %-20s %s -> %s   (%s.train=%.3f)" % (knob, base, cur[knob], TARGET, best_tgt))

final = evaluate(cur)
print("\nBEST CONFIG:", " ".join("%s=%d" % (k, v) for k, v in cur.items()))
print("%s.train=%.3f (base %.3f)  ALL.val=%.3f (base %.3f)" %
      (TARGET, g(final, TARGET + ".train"), g(base_d, TARGET + ".train"),
       g(final, "ALL.val"), g(base_d, "ALL.val")))
print("guards:", {t: (round(base_guard[t], 2), round(g(final, t + ".train"), 2)) for t in GUARDS})
print("\nfinal direction metrics:")
for ln in final["_dir"].splitlines():
    if ln.startswith("DIR"):
        print("  " + ln)
