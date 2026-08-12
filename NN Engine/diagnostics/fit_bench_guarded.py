# -*- coding: utf-8 -*-
"""Two-stage BENCH-GUARDED tuner: the corpus win%-MSE proposes, the REAL benches dispose.

Why: the corpus fit is BLIND to move choice, so it rejected de-king (real STS +92) and selected KS_CHK cuts that
measurably HURT STS -- i.e. every fit silently discarded what we learned on the benches. The `sts_guard` corpus
tier is only SF18-labeled STS positions scored by win%-MSE; it is NOT the STS move-choice score.

Design (deliberate): the bench is a GUARD, not the objective -- STS is deterministic but JAGGED, so maximizing it
would just overfit the bench. Stage 1 ranks candidates by corpus win%-MSE (smooth gradient). Stage 2 runs the
REAL STS (+ optional WAC) on the top-K and REJECTS any candidate that regresses a bench below baseline - TOL.
The surviving best-corpus candidate wins, so bench knowledge can never be thrown away by a later fit.

  pyrun diagnostics/fit_bench_guarded.py [TOPK=6] [STS_TOL=0] [WAC_TOL=2] [CORPUS=...]
Candidate list: edit CANDIDATES below (name -> knob dict). Baseline = all-identity.
"""
import os, sys, subprocess, re

# The runner's `pyrun` sub passes KEY=VAL as ARGV, not env, so every os.environ.get below silently fell back
# to its default when invoked that way (a Phase-A run intended for the passer corpus actually scored
# diverse_corpus_wide). Fold argv into the environment first so the documented usage works.
for _a in sys.argv[1:]:
    if '=' in _a:
        _k, _v = _a.split('=', 1)
        os.environ[_k] = _v

THIS = os.path.dirname(os.path.abspath(__file__))
ENGINE = os.path.dirname(THIS)
RUNNER = os.path.join(ENGINE, "selfplay", "overnight_runner.sh")
PY = sys.executable
TOPK = int(os.environ.get("TOPK", "6"))
STS_TOL = float(os.environ.get("STS_TOL", "0"))     # allowed STS drop vs baseline (0 = none)
WAC_TOL = float(os.environ.get("WAC_TOL", "2"))     # WAC is jagged/tactical -> small slack
_corpus = os.environ.get("CORPUS", "diverse_corpus_wide.csv")
# Accept a BARE BASENAME and resolve it under ks_sets/. The engine path contains a space ("Chess Engine"),
# so an absolute CORPUS= passed through the runner splits into two argv words and silently resolves to a
# nonexistent file -- which yields 9e9 for every candidate, a degenerate tie that looks like a real ranking.
CORPUS = _corpus if os.sep in _corpus or "/" in _corpus else os.path.join(THIS, "ks_sets", _corpus)
if not os.path.exists(CORPUS):
    sys.exit("CORPUS not found: %s" % CORPUS)
WORKER = os.path.join(THIS, "_ks_fit_eval.py")

# Candidate configs to screen (knob dicts). Baseline is implicit (empty = all identity).
# Retune around PIECEVAL_RECOMPUTE_LATE (2026-08-02). The accumulators are exchange-adjusted material and
# capgains mutates them; the late recompute restores RAW material. Every consumer of that edge was fitted
# while the exchange-adjusted value was live, so a single-knob sweep cannot tell "the fix is bad" from "the
# fix needs the rest refitted around it". Baseline (empty) = shipped defaults, STS 1746 -- a candidate only
# survives stage 2 if the retune actually recovers that.
# Grids are selected with GRID=<name>; each is a self-contained arm with its own warrant. Keep dead grids
# here rather than deleting them -- a rejected grid is the evidence that stops it being re-proposed.
#
# PHASE A (2026-08-02) -- DEAD, kept as the record. Re-judged the passer R-floor on the shipped V3 baseline.
# Result: train flat (~297-299), held-out validation degraded MONOTONICALLY with floor strength (baseline
# 539.3 best, 192/256 worst at 561.1) and all benched arms rejected at -107..-149 STS. The reason is
# structural: R collapsing is CORRECT whenever the passer really is stopped, and a RANK-keyed floor fires on
# every advanced passer regardless -- the same blunt-clamp failure as KS_REALIZ_FLOOR. Do not re-propose.
GRID_RFLOOR = {
    "baseline":       {},                                                   # shipped: floors at 0
    "fl_64_96":       {"PASSER_RFLOOR_R5": 64,  "PASSER_RFLOOR_R6": 96},
    "fl_96_128":      {"PASSER_RFLOOR_R5": 96,  "PASSER_RFLOOR_R6": 128},
    "fl_128_160":     {"PASSER_RFLOOR_R5": 128, "PASSER_RFLOOR_R6": 160},
    "fl_160_192":     {"PASSER_RFLOOR_R5": 160, "PASSER_RFLOOR_R6": 192},
    "fl_192_256":     {"PASSER_RFLOOR_R5": 192, "PASSER_RFLOOR_R6": 256},   # R6=256 => rank-7 fully unconditional
    "fl_r6_only_192": {"PASSER_RFLOOR_R6": 192},                            # is it only the 7th rank that matters?
    "fl_r6_only_256": {"PASSER_RFLOOR_R6": 256},
}

# PASSER arm (2026-08-03): the SHAPED replacement for the dead floor. These knobs are keyed on the actual
# attacker/defender balance and on rear/king geometry, so unlike a rank floor they DISCRIMINATE a genuinely
# unstoppable passer from a genuinely stopped one by construction. They were set when V3 shipped and have
# never been jointly fit, while V3 simultaneously removed 8 rook-evaluator passer sites and made
# `mag x R/256` the sole payer of advanced-passer value -- so the whole channel now rests on constants that
# were only ever eyeballed. Run against passer_fit.csv (the FIT schema; passer_corpus.csv is the raw mined
# one and yields a degenerate MSE 0.000 tie for every candidate).
GRID_PASSER = {
    "baseline":        {},
    "contest_soft":    {"PASSER_CONTEST_STOP": 60,  "PASSER_CONTEST_PATH": 28},
    "contest_hard":    {"PASSER_CONTEST_STOP": 120, "PASSER_CONTEST_PATH": 55},
    "contest_stoponly":{"PASSER_CONTEST_STOP": 120, "PASSER_CONTEST_PATH": 24},  # stop square vs deep path
    "rear_strong":     {"PASSER_REAR_ENEMY": 160, "PASSER_REAR_OWN": 72},
    "rear_weak":       {"PASSER_REAR_ENEMY": 96,  "PASSER_REAR_OWN": 32},
    "king_strong":     {"PASSER_KING_FAR": 24, "PASSER_KING_HELP": 9},
    "king_weak":       {"PASSER_KING_FAR": 10, "PASSER_KING_HELP": 4},
    "rcap_up":         {"PASSER_R_CAP": 384},
    "rcap_up_mag":     {"PASSER_R_CAP": 384, "PASSER_MAG_SCALE": 110},
    "mag_up":          {"PASSER_MAG_SCALE": 115},
    "mag_dn":          {"PASSER_MAG_SCALE": 85},
}

# THREATS arm (2026-08-03): the term is DARK (ENABLE_THREATS=false), which is why `threats` reads exactly
# 0.00 in all eight win%-ranked collapse FENs while SF11 and SF15.1 both price it around -1.5 as part of the
# compensation for a material deficit. It was disabled for unbalancing other terms, not for being wrong, so
# the question is a magnitude question -- hence a scale sweep rather than a structural port. Threats are
# INVARIANT SF11->SF15.1 (constants within a few %), so the form we already have is the right form.
# NOTE: the THREAT_* constants in search_engine.h are KING-ZONE weights, a different subsystem; not these.
GRID_THREATS = {
    "baseline":     {},
    "thr_25":       {"ENABLE_THREATS": 1, "SCALE_THREATS": 25},
    "thr_50":       {"ENABLE_THREATS": 1, "SCALE_THREATS": 50},
    "thr_75":       {"ENABLE_THREATS": 1, "SCALE_THREATS": 75},
    "thr_100":      {"ENABLE_THREATS": 1, "SCALE_THREATS": 100},
    "thr_so_50":    {"ENABLE_THREATS": 1, "SCALE_THREATS": 50,  "THREATS_STANDING_ONLY": 1},
    "thr_so_75":    {"ENABLE_THREATS": 1, "SCALE_THREATS": 75,  "THREATS_STANDING_ONLY": 1},
    "thr_so_100":   {"ENABLE_THREATS": 1, "SCALE_THREATS": 100, "THREATS_STANDING_ONLY": 1},
}

# RESID arm (2026-08-03): the SHAPED replacement for the dead rank floor. `mag * R/256` means one wrong
# realizability call costs the WHOLE pawn; PASSER_RESID_PCT grants a residual share of the rank magnitude
# that R cannot remove, and — unlike a rank floor — scales it DOWN by blockade permanence, so a knight
# blockade (BLOCK 140) keeps ~no residual while a queen "blockade" (BLOCK 50, must move) keeps most of it.
# Bounds the OUTPUT, leaves the ASSESSMENT untouched. This is the discrimination the floor could not make.
GRID_RESID = {
    "baseline":   {},
    "resid_10":   {"PASSER_RESID_PCT": 10},
    "resid_20":   {"PASSER_RESID_PCT": 20},
    "resid_30":   {"PASSER_RESID_PCT": 30},
    "resid_40":   {"PASSER_RESID_PCT": 40},
    "resid_55":   {"PASSER_RESID_PCT": 55},
    "resid_70":   {"PASSER_RESID_PCT": 70},
}

# THREAT-CAP arm (2026-08-03): threats_by sums an UNBOUNDED per-target stack — MINOR 850 + ROOK 850 +
# KING 250 + SAFE_PAWN 1600 = 3550 millipawns (3.5 pawns) from ONE target — so a single misjudged piece can
# swamp the term. THREAT_PER_TARGET_CAP bounds that blast radius without changing which targets are found.
# Built on STANDING_ONLY=1, which beat full threats at every scale last night (the hanging term is volatile
# and double-counts capture_gains' en-prise sim, as the header comment at search_engine.h:665 predicted).
_TH = {"ENABLE_THREATS": 1, "THREATS_STANDING_ONLY": 1}
GRID_THREAT_CAP = {
    "baseline":        {},
    "so75_nocap":      dict(_TH, SCALE_THREATS=75),
    "so75_cap800":     dict(_TH, SCALE_THREATS=75, THREAT_PER_TARGET_CAP=800),
    "so75_cap1200":    dict(_TH, SCALE_THREATS=75, THREAT_PER_TARGET_CAP=1200),
    "so75_cap1600":    dict(_TH, SCALE_THREATS=75, THREAT_PER_TARGET_CAP=1600),
    "so50_cap800":     dict(_TH, SCALE_THREATS=50, THREAT_PER_TARGET_CAP=800),
    "so50_cap1200":    dict(_TH, SCALE_THREATS=50, THREAT_PER_TARGET_CAP=1200),
    "so100_cap800":    dict(_TH, SCALE_THREATS=100, THREAT_PER_TARGET_CAP=800),
}

# THREAT-GATE arm (2026-08-03): the capped term is a REDISTRIBUTION -- it repairs our worst decile
# (mean -28 win%^2) and taxes our best (+2.9), and the only tier it worsens is sts_guard. This sweeps a
# tension gate (g_capg_tension = pending SEE>=0 captures) to see whether the two regimes can be separated.
# ⚠️ First probe: QUIET_PCT=25 read STS 1622, WORSE than ungated 1685 -- so damping quiet positions is NOT
# a free win and the shape needs measuring rather than assuming.
_TC = dict(_TH, SCALE_THREATS=75, THREAT_PER_TARGET_CAP=800)
GRID_THREAT_GATE = {
    "baseline":      {},
    "gate_ungated":  dict(_TC),                                          # reference: STS 1685, val 263.920
    "gate_q25_hi3":  dict(_TC, THREATS_QUIET_PCT=25, THREATS_TENSION_HI=3),
    "gate_q50_hi3":  dict(_TC, THREATS_QUIET_PCT=50, THREATS_TENSION_HI=3),
    "gate_q75_hi3":  dict(_TC, THREATS_QUIET_PCT=75, THREATS_TENSION_HI=3),
    "gate_q50_hi2":  dict(_TC, THREATS_QUIET_PCT=50, THREATS_TENSION_HI=2),
    "gate_q50_hi5":  dict(_TC, THREATS_QUIET_PCT=50, THREATS_TENSION_HI=5),
    # Inverted: damp the TACTICAL end instead, in case the tax is on live positions the search resolves anyway.
    "gate_inv_q100_lo": dict(_TC, THREATS_QUIET_PCT=100, THREATS_TENSION_LO=0, THREATS_TENSION_HI=3),
}

# INPUT-ABLATION arm (2026-08-03): tests the OTHER hypothesis. Two output-bounding schemes are now dead
# (rank floor, blockade-scaled residual), both degrading held-out accuracy MONOTONICALLY -- which says R is
# well-calibrated ON AVERAGE and no global transfer change helps. But that is consistent with R's INPUTS
# being crude and their per-position error being what we actually see: any global reshape just rescales a
# noisy signal. So zero each input in turn. If R's inputs are sound, EVERY ablation should HURT. If an
# ablation HELPS, that input is miscalibrated and is shadowing the whole channel.
GRID_PASSER_ABLATE = {
    "baseline":     {},
    "abl_contest":  {"PASSER_CONTEST_STOP": 0, "PASSER_CONTEST_PATH": 0},
    "abl_stoponly": {"PASSER_CONTEST_PATH": 0},                     # keep the stop square, drop the deep path
    "abl_pathonly": {"PASSER_CONTEST_STOP": 0},                     # keep the deep path, drop the stop square
    "abl_rear":     {"PASSER_REAR_ENEMY": 0, "PASSER_REAR_OWN": 0},
    "abl_rear_own": {"PASSER_REAR_OWN": 0},                         # the half that duplicates the #16 rook channel
    "abl_king":     {"PASSER_KING_FAR": 0, "PASSER_KING_HELP": 0},
    "abl_blockade": {"ENABLE_PASSER_BLOCKADE_QUALITY": 0},
}

# JOINT arm (2026-08-03): the two surviving single-feature findings, fitted TOGETHER on the MIXED corpus.
# Rationale (owner's constraint): stacking individually-tuned pieces overfits to features AND to position
# archetypes, so the joint candidate is fitted on diverse_corpus_wide (calm / diverse / target / working /
# sts_guard / passer_* tiers, all four phase buckets), NOT on passer_fit -- fitting passers on a passer
# corpus is how the archetype overfit happens. Surviving findings folded in:
#   (a) threats with the per-target CAP (turned SCALE=100 from worst-uncapped to best-capped)
#   (b) PASSER_REAR_OWN reduced (corpus val monotone toward 0: 72->540.2, 48->539.3, 32->538.5, 0->537.3)
# The tension gate is EXCLUDED: every gated arm was worse than ungated on both axes.
_JT = dict(_TH, SCALE_THREATS=75, THREAT_PER_TARGET_CAP=800)
GRID_JOINT = {
    "baseline":        {},
    "j_threat_only":   dict(_JT),
    "j_rear32_only":   {"PASSER_REAR_OWN": 32},
    "j_rear0_only":    {"PASSER_REAR_OWN": 0},
    "j_thr_rear32":    dict(_JT, PASSER_REAR_OWN=32),
    "j_thr_rear0":     dict(_JT, PASSER_REAR_OWN=0),
    "j_thr_rear32_s50": dict(_TH, SCALE_THREATS=50, THREAT_PER_TARGET_CAP=800, PASSER_REAR_OWN=32),
    "j_thr_rear32_s100": dict(_TH, SCALE_THREATS=100, THREAT_PER_TARGET_CAP=800, PASSER_REAR_OWN=32),
}

# THREAT-TUNE arm (2026-08-03, post-SPRT): refine AROUND the game-validated config. `so75_cap800` won an
# SPRT at +45 Elo (+171 -121 =96 / 388, LLR +3.035) but its constants came off a 3-point cap grid, and
# `so100_cap800` had BETTER held-out accuracy (263.45 vs 263.92) and was never gamed. Run with
#   BASE='ENABLE_THREATS=1 SCALE_THREATS=75 THREATS_STANDING_ONLY=1 THREAT_PER_TARGET_CAP=800'
# so "baseline" here IS the shipped-candidate config and every arm is a delta on top of it.
# ⚠️ Note THREAT_SAFE_PAWN is 1600 -- above every cap tried -- so the cap is partly just re-weighting that
# one term. Separating the two needs SAFE_PAWN to become a knob; not done yet.
GRID_THREAT_TUNE = {
    "baseline":     {},                                              # == BASE (the +45 Elo arm)
    "cap600":       {"THREAT_PER_TARGET_CAP": 600},
    "cap1000":      {"THREAT_PER_TARGET_CAP": 1000},
    "cap1200":      {"THREAT_PER_TARGET_CAP": 1200},
    "s100":         {"SCALE_THREATS": 100},                          # best corpus arm, never gamed
    "s100_cap600":  {"SCALE_THREATS": 100, "THREAT_PER_TARGET_CAP": 600},
    "s100_cap1000": {"SCALE_THREATS": 100, "THREAT_PER_TARGET_CAP": 1000},
    "s125_cap800":  {"SCALE_THREATS": 125},
    "s50_cap600":   {"SCALE_THREATS": 50, "THREAT_PER_TARGET_CAP": 600},
}

# SCALE-EXTEND (2026-08-03): with the cap in place, corpus accuracy improves MONOTONICALLY with scale
# (75 -> 263.920, 100 -> 263.449, 125 -> 262.932), the exact reverse of the uncapped behaviour where
# scaling up was worst. Push until it turns, so tonight's SPRT uses the corpus optimum rather than the
# first point that happened to be tried. Same BASE as threattune.
GRID_THREAT_SCALE = {
    "baseline":     {},
    "s125":         {"SCALE_THREATS": 125},
    "s150":         {"SCALE_THREATS": 150},
    "s175":         {"SCALE_THREATS": 175},
    "s200":         {"SCALE_THREATS": 200},
    "s150_cap600":  {"SCALE_THREATS": 150, "THREAT_PER_TARGET_CAP": 600},
    "s150_cap1000": {"SCALE_THREATS": 150, "THREAT_PER_TARGET_CAP": 1000},
    "s175_cap600":  {"SCALE_THREATS": 175, "THREAT_PER_TARGET_CAP": 600},
}

# CONFIRM (2026-08-03): scale sweep at cap 800 has an INTERIOR minimum at 125 (75→263.920, 100→263.449,
# 125→262.932, 150→263.859, 175→264.782, 200→267.315). A scale x cap interaction is also visible (at 150/175
# the tighter cap 600 wins), but the diagonal at 125 was never tried. Fill it before committing tonight's SPRT.
GRID_THREAT_CONFIRM = {
    "baseline":      {},
    "s110":          {"SCALE_THREATS": 110},
    "s125":          {"SCALE_THREATS": 125},
    "s140":          {"SCALE_THREATS": 140},
    "s125_cap600":   {"SCALE_THREATS": 125, "THREAT_PER_TARGET_CAP": 600},
    "s125_cap700":   {"SCALE_THREATS": 125, "THREAT_PER_TARGET_CAP": 700},
    "s125_cap1000":  {"SCALE_THREATS": 125, "THREAT_PER_TARGET_CAP": 1000},
}

# DECONFOUND (2026-08-03): THREAT_SAFE_PAWN is 1600, above EVERY cap that helped, and is the largest single
# per-target term (next is 850). So "cap the stack" and "cut SAFE_PAWN" are confounded. Run WITHOUT the cap
# and vary SAFE_PAWN: if uncapped SAFE_PAWN=800 reproduces the capped result, the cap was never bounding a
# stack -- it was re-weighting one term, and the other threat constants are likely over-weighted too.
# NOTE: BASE sets the cap, so these arms must explicitly switch it OFF with THREAT_PER_TARGET_CAP=0.
GRID_SAFEPAWN = {
    "baseline":        {},                                                    # BASE: cap 800, SAFE 1600
    "nocap_sp1600":    {"THREAT_PER_TARGET_CAP": 0},                          # cap off, term untouched
    "nocap_sp1200":    {"THREAT_PER_TARGET_CAP": 0, "THREAT_SAFE_PAWN": 1200},
    "nocap_sp800":     {"THREAT_PER_TARGET_CAP": 0, "THREAT_SAFE_PAWN": 800},
    "nocap_sp600":     {"THREAT_PER_TARGET_CAP": 0, "THREAT_SAFE_PAWN": 600},
    "nocap_sp400":     {"THREAT_PER_TARGET_CAP": 0, "THREAT_SAFE_PAWN": 400},
    "cap800_sp800":    {"THREAT_SAFE_PAWN": 800},                             # both -> is the cap still adding?
    "cap1200_sp800":   {"THREAT_PER_TARGET_CAP": 1200, "THREAT_SAFE_PAWN": 800},
}

# PAWN-STRUCTURE arm (2026-08-04): ISOLATED_PAWN_PEN and BACKWARD_PAWN_PEN are BUILT AND GATED OFF (both 0)
# -- the same pattern as ENABLE_THREATS, MOD_KS_REALIZ and the gravity machinery, each of which turned out to
# be a real gain sitting behind a zero. SF dampens ordinary pawns hard (Isolated S(1,20), Backward S(6,19),
# both + WeakUnopposed S(15,18), plus Doubled/DoubledEarly/BlockedPawn); we have the BOOST half (chain/wall)
# and none of the DAMPEN half, so every "is this pawn actually good" judgement is forced into the passer
# layer's realizability multiplier. Ordinary pawns appear in far more positions than advanced passers, so a
# general-play-weighted corpus can resolve these where passer arms never could.
GRID_PAWNSTRUCT = {
    "baseline":     {},
    "iso_25":       {"ISOLATED_PAWN_PEN": 25},
    "iso_50":       {"ISOLATED_PAWN_PEN": 50},
    "iso_100":      {"ISOLATED_PAWN_PEN": 100},
    "bwd_25":       {"BACKWARD_PAWN_PEN": 25},
    "bwd_50":       {"BACKWARD_PAWN_PEN": 50},
    "bwd_100":      {"BACKWARD_PAWN_PEN": 100},
    "both_25":      {"ISOLATED_PAWN_PEN": 25, "BACKWARD_PAWN_PEN": 25},
    "both_50":      {"ISOLATED_PAWN_PEN": 50, "BACKWARD_PAWN_PEN": 50},
    "both_100_50":  {"ISOLATED_PAWN_PEN": 100, "BACKWARD_PAWN_PEN": 50},
}

# PAWN-STRUCTURE round 2 (2026-08-04): round 1 improved corpus val MONOTONICALLY (279.510 -> 276.372 at
# iso100/bwd50) with the SMALLEST bench footprint of the session (bwd_100 = -7 STS / -2 WAC), and the trend
# had NOT turned at the edge of the grid. Round 2 pushes further and runs on the CORRECTED backward
# definition (now also counts an enemy pawn OCCUPYING the stop square, matching SF's `leverPush | blocked`).
# ⚠️ Expect the flat knob to want a high value: SF splits these by phase (Isolated S(1,20), Backward S(6,19)
# -- nearly pure endgame terms) and ours is phase-flat, so a single constant is compensating for a missing
# phase split. A high optimum here is evidence for making them (mg, eg) pairs, not for shipping a big flat.
GRID_PAWNSTRUCT2 = {
    "baseline":      {},
    "iso150":        {"ISOLATED_PAWN_PEN": 150},
    "iso200":        {"ISOLATED_PAWN_PEN": 200},
    "bwd150":        {"BACKWARD_PAWN_PEN": 150},
    "bwd200":        {"BACKWARD_PAWN_PEN": 200},
    "i100_b50":      {"ISOLATED_PAWN_PEN": 100, "BACKWARD_PAWN_PEN": 50},   # round-1 best, re-measured
    "i150_b100":     {"ISOLATED_PAWN_PEN": 150, "BACKWARD_PAWN_PEN": 100},
    "i200_b100":     {"ISOLATED_PAWN_PEN": 200, "BACKWARD_PAWN_PEN": 100},
    "i200_b200":     {"ISOLATED_PAWN_PEN": 200, "BACKWARD_PAWN_PEN": 200},
    "i300_b150":     {"ISOLATED_PAWN_PEN": 300, "BACKWARD_PAWN_PEN": 150},
}

# ORD-FLOOR arm (2026-08-05): floor a passer at what the same pawn would earn if NOT passed. Distinct from
# every failed unconditional-base arm: those granted a share of `mag` (hundreds of mp, untethered); this
# grants at most default_midgame_pawn_rank_bonus[rank] (90 mp at rank 6) and only where the passer is
# currently scoring BELOW an ordinary pawn -- a discontinuity, not a fudge. Verified per-passer: w1 f2
# 6 -> 96, w5 c4 13 -> 73, healthy passers move ~-1% (the ordinary-pawn part is no longer amplified by
# R>256). Paired with R_CAP, the only other passer lever that has never cost STS (+4 on the mixed corpus).
GRID_ORDFLOOR = {
    "baseline":         {},
    "ordfloor":         {"ENABLE_PASSER_ORD_FLOOR": 1},
    "rcap384":          {"PASSER_R_CAP": 384},
    "ordfloor_rcap384": {"ENABLE_PASSER_ORD_FLOOR": 1, "PASSER_R_CAP": 384},
    "ordfloor_rcap448": {"ENABLE_PASSER_ORD_FLOOR": 1, "PASSER_R_CAP": 448},
    "ordfloor_mag110":  {"ENABLE_PASSER_ORD_FLOOR": 1, "PASSER_MAG_SCALE": 110},
}

GRIDS = {"rfloor": GRID_RFLOOR, "passer": GRID_PASSER, "threats": GRID_THREATS,
         "pawnstruct": GRID_PAWNSTRUCT, "pawnstruct2": GRID_PAWNSTRUCT2, "ordfloor": GRID_ORDFLOOR,
         "threatscale": GRID_THREAT_SCALE, "threatconfirm": GRID_THREAT_CONFIRM,
         "safepawn": GRID_SAFEPAWN,
         "resid": GRID_RESID, "threatcap": GRID_THREAT_CAP, "threatgate": GRID_THREAT_GATE,
         "ablate": GRID_PASSER_ABLATE, "joint": GRID_JOINT, "threattune": GRID_THREAT_TUNE}
CANDIDATES = GRIDS[os.environ.get("GRID", "passer")]


# BASE: knobs applied to EVERY arm including the baseline, so candidates are judged on top of the current
# best config rather than against a superseded default. Without this, every grid silently re-measures the old
# baseline -- e.g. the game-validated threats arm would be absent from both sides of the comparison.
#   BASE='ENABLE_THREATS=1 SCALE_THREATS=75 THREATS_STANDING_ONLY=1 THREAT_PER_TARGET_CAP=800'
BASE = {}
for _p in os.environ.get("BASE", "").split():
    if "=" in _p:
        _k, _v = _p.split("=", 1)
        BASE[_k] = _v


def withbase(cfg):
    """Arm knobs win over BASE, so a grid can still sweep a knob that BASE sets."""
    out = dict(BASE)
    out.update(cfg)
    return out


def knobstr(cfg):
    return " ".join("%s=%s" % (k, v) for k, v in sorted(cfg.items()))


def corpus_loss(cfg):
    """Stage 1: corpus win%-MSE (train/val) + per-tier, via the shared worker."""
    cfg = withbase(cfg)
    args = [PY, WORKER, "CORPUS=" + CORPUS] + ["%s=%s" % (k, v) for k, v in cfg.items()]
    out = subprocess.run(args, capture_output=True, text=True).stdout
    m = re.search(r"train_mse=([\d.]+) val_mse=([\d.]+)", out)
    tiers = {k: float(v) for k, v in re.findall(r"(\w+\.\w+)=([\d.]+)", out)}
    return (float(m.group(1)), float(m.group(2)), tiers) if m else (9e9, 9e9, tiers)


def _runner(sub, tag, cfg):
    # runs INSIDE wsl (pyrun) -> call the runner with bash directly; knobs as separate KEY=VAL argv entries
    cmd = ["bash", RUNNER, sub, tag] + ["%s=%s" % (k, v) for k, v in sorted(withbase(cfg).items())]
    return subprocess.run(cmd, capture_output=True, text=True).stdout


def bench_sts(cfg, tag):
    m = re.search(r"STS score:\s*(\d+)", _runner("sts", tag, cfg))
    return int(m.group(1)) if m else -1


def bench_wac(cfg, tag):
    m = re.search(r"Solved (\d+)/", _runner("wac", tag, cfg))
    return int(m.group(1)) if m else -1


print("STAGE 1 — corpus win%%-MSE (proposes)   grid=%s  corpus=%s"
      % (os.environ.get("GRID", "passer"), os.path.basename(CORPUS)))
if BASE:
    print("  BASE (applied to every arm incl. baseline): %s" % knobstr(BASE))
stage1 = []
for name, cfg in CANDIDATES.items():
    tr, va, tiers = corpus_loss(cfg)
    stage1.append((va, tr, name, cfg))
    print("  %-22s train=%9.3f  val=%9.3f   %s" % (name, tr, va, knobstr(cfg) or "(identity)"))
stage1.sort()

top = stage1[:TOPK]
print("\nSTAGE 2 — REAL benches on top-%d (disposes; guard = no regression vs baseline)" % len(top))
base_sts = bench_sts({}, "GUARD_BASE"); base_wac = bench_wac({}, "GUARD_BASE")
print("  baseline: STS=%d  WAC=%d" % (base_sts, base_wac))

survivors = []
for va, tr, name, cfg in top:
    if name == "baseline":
        continue
    s = bench_sts(cfg, "G_" + name.upper()); w = bench_wac(cfg, "G_" + name.upper())
    ok = (s >= base_sts - STS_TOL) and (w >= base_wac - WAC_TOL)
    print("  %-22s STS=%4d (%+d)  WAC=%3d (%+d)  corpus_val=%8.3f  -> %s"
          % (name, s, s - base_sts, w, w - base_wac, va, "PASS" if ok else "REJECT"))
    if ok:
        survivors.append((va, name, cfg, s, w))

print("\nRESULT")
if not survivors:
    print("  no candidate passed the bench guard -> keep baseline")
else:
    survivors.sort()
    va, name, cfg, s, w = survivors[0]
    print("  WINNER (best corpus among bench-clean): %s   STS=%d WAC=%d corpus_val=%.3f" % (name, s, w, va))
    print("  knobs: %s" % (knobstr(cfg) or "(identity)"))
    print("  all bench-clean candidates:")
    for va2, n2, c2, s2, w2 in survivors:
        print("    %-22s corpus_val=%8.3f  STS=%4d  WAC=%3d" % (n2, va2, s2, w2))
