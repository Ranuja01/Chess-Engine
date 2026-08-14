# Session handoff — 2026-08-12: the KS "WHEN-to-fire" reframe + the detector stack + the +15 bundle

> ## 🚨 READ THIS BLOCK FIRST
>
> **Committed + pushed `d58d47c` on `NN-ENgine` (prior checkpoint `91485c4`).** All new eval knobs default
> **byte-identical** (`250 / 35,426,396 / EBF 3.800`). Nothing is shipped as a default change.
>
> ### ★★★ STATE IN ONE PARAGRAPH
> The de-noised **OvD+central+defaware1 bundle = +15 Elo lean / ~1346-game SPRT (UNCONFIRMED, CI crosses 0)** is
> the working baseline. This session's big work + reframe was **king safety**: we built four genuine, bottom-up,
> **discrimination-validated** detector upgrades (`KS_SQC_MODE` value-aware contest, `KS_PIN_MODE` pins,
> `KS_WEAK_VAL_MODE` weak-value-coupling, `KS_FLANK_MODE` flank-breadth) that took unit **discrimination AUC
> 0.748 → 0.810** (past SF's 0.80) and the detection gap **80% → 8%** — all byte-id-off and symmetry-clean. **But
> turning them on made general move-quality WORSE.** The owner's reframe cracked why: KS is a **"WHEN-to-fire"
> (context-gating) problem, not a "how much" problem**, and we'd spent months only on "how much." The over-fire
> is the **OPENING** (phase-split: opening +0.37 / midgame +0.27 / **endgame −0.32 = KS HELPS**), i.e. a **missing
> coordination gate** (we fire on breadth/presence; SF needs a coordinated group). Detectors are a "how much"
> win that is unusable until the "when" system is fixed.
>
> ### ▶️ NEXT ACTIONS (in order)
> 1. **Coordination gate** — attacker **count × weight PRODUCT** (a lone/few-attacker position can't clear the
>    threshold), as the first "when" lever. Validate on `_ks_footprint_regret.py` **PHASE-STRATIFIED**
>    (`_ks_phase_split.py`) on BOTH cross-sets. Predict: opening improves, midgame neutral, endgame unchanged.
> 2. **Fold the whacky/variant corpus into validation** (built, still UN-USED) — see the dedicated idea below.
> 3. If the coordination gate works → the full **signed-accumulator rebuild** (§ architecture) with the
>    discrimination-validated detectors feeding the POSITIVE side.

## THE REFRAME: "WHEN to fire" vs "how much" (the session's key insight)
We asked only **"how much"** (detector magnitudes, the danger curve) and never **"WHEN should KS fire at all?"**
(the context gates). Both failure modes are "when" failures: opening over-fire (fire without coordination) and
midgame attack-chasing (fire when the king isn't decisive). Full architecture + methodology lesson:
`dev_notes/SQUARE-CONTROL-PRIMITIVE-PLAN-2026-08-12.md`; memory [[ask-when-and-how-much-validate-on-deployment]].

### SF/Ethereal architecture (source-verified) — when + how-much are ONE object
ONE signed `kingDanger` accumulator: positives (attacker weight × **count** product, weak, checks) + **large
NEGATIVE suppressors in the same units** (−873 no-queen, −shelter, −6·score/8 already-winning) → they **NET** →
one **threshold** (>100) → **SQUARE** (after all gates, so it amplifies NET danger, never proximity) → phase as
a **two-function mg/eg pair** (mg = square, eg = a different tiny linear fn) blended by **non-pawn material**.

### Ours mis-shapes all three "when" mechanisms (`cpp_bitboard.cpp`)
1. `KS_FLOOR=13` = fixed MAGNITUDE floor (chops the same off everything) vs SF's emergent NET-SIGN threshold.
2. `ks_phase_taper` = single LINEAR scale on the output vs SF's per-term two-function pair; `KS_PHASE_ZERO` =
   deep-endgame CLIFF (under-gates R/Q endgames) vs continuous material weighting.
3. attacker term = flat additive **SUM** vs SF's count×weight **PRODUCT** (no super-linearity ⇒ a lone piece
   counts as much per-unit as a coordinated group — the opening over-fire).
Plus: `KS_MIN_ATTACKERS` off + inert-with-queen; `KS_NO_QUEEN` single-digit vs SF's −873.

## THE +15 BUNDLE (the working baseline, committed)
`OVD_BOUNDED_MODE=2 OVD_CAP=300 OVD_KNEE=40 CENTRAL_BOUNDED_MODE=1 CENTRAL_CAP=150 CENTRAL_KNEE=200
KS_DEFAWARE_MODE=1`. +15 Elo lean over ~1346 games (LLR plateaued ~+0.9, never crossed +2.94 — a small effect;
CI includes 0). Not a confirmed ship; it rides forward as the base and confirms inside the bigger bundle later.
`defaware1` is the one KS lever with a sign-consistent cross-set move signal + STS +112 + symmetry-clean.

## 🌀 THE WHACKY / VARIANT CORPUS — a first-class diversity idea (owner, use at LARGE magnitude)
**Purpose:** our regret/validation sets are mined from STANDARD selfplay ⇒ they share opening/structural patterns;
a term can score by memorising known structures rather than encoding real chess. **Whacky positions break that
scaffolding** and are the sharpest test of whether a detector is genuine chess knowledge vs structure-overfit —
and SF18 (computes, not books) is a valid arbiter. **Built + smoke-tested: `_build_variant_regret_set.py`** —
piece-replacement (all/one N↔B, asymmetric one-side too) + 960-style shuffles with **castling DISABLED** (no
UCI_Chess960 needed), short random walk, SF18 multi-PV @d14, same schema, resumable, `SHARD=i/n` parallel.
- ★ **STILL UN-USED in validation** — fold it in, at a **fairly large magnitude**, for diversity beyond opening
  patterns (owner's explicit intent). Use as a THIRD cross-set + a targeted per-subsystem stress set.
- ⚠️ Use as **generalization/diagnostic + targeted-stress** signal; keep STANDARD game-representative as the
  tuning distribution + deploy arbiter (variants are far from deployment). Scale run wants `WALK_MAX≈45`.
- ★ **No-castling is the point** for KS: central/exposed kings, no clean shelter — stresses the exact when+
  central-vs-KS interaction standard flank-king positions hide.

## 🧰 DATA + TOOLS (all in `diagnostics/`)
- **Discrimination:** `_ks_discrimination.py` (unit AUC attack-vs-quiet on `ks_sets/ks_sts_corpus.csv` — the
  verify-at-each-step metric that survives the dead curve; move-regret is move-neutral there). `ks_genuine_units.py`.
- **Deployment move-regret:** `_ks_footprint_regret.py` (footprint D7 regret, base vs cand, changed-move filter,
  BOTH cross-sets `game_regret_set.csv` 15k + `game_regret_set_v2.csv` 11,940 disjoint). ⚠️ **Do NOT DEFER this
  test on a "move-neutral" assumption** (the mistake this session). `_ks_phase_split.py` = the same, STRATIFIED
  by phase — the instrument that localized the over-read to the opening.
- **Diagnosis:** `_ks_drift_analysis.py` (worst move-flips + KS units), `_ks_channel_collinearity.py` (4-channel
  collinearity — unit-KS dominant ~72%), `_ks_unit_trace.py` (unit distribution; confirmed the dead quadratic).
- **Curve knobs are env-registered before `rebuild_ks_tables`** (`KS_KNEE/FLOOR/DIVISOR`), so no code needed.

## 🧠 KEY BANKED FINDINGS (don't re-derive)
- Dead quadratic CONFIRMED (`FLOOR=13 > KNEE=12` ⇒ 100% of dangerous kings on the linear branch; units ~85%
  proximity). Compounding non-discriminating units amplifies noise (`configs A/B` failed cross-set).
- The channel-collinearity is MODERATE, not the dominant story (unit-KS dominant); the fix is unit-KS INTERNAL.
- **square_control primitive scope-corrected: KS-only** (OvD/central are NOT `attack_bitmasks` readers).
- CHECK_V2 / SC-crank / additive KS all refuted on the cross-set again (0-for-9 holds). Only subtractive/
  redistributive + now the detector *discrimination* work advanced anything.

## ☠️ METHODOLOGY LESSONS (this session, load-bearing)
- **Ask WHEN + HOW-MUCH for every eval term.** We only asked "how much" for months. [[ask-when-and-how-much-validate-on-deployment]].
- **Map the giants' INTEGRATION, not isolated pieces** — the "when+how-much are one signed object" fact is
  invisible if you study primitive/curve/channels separately.
- **Do NOT defer the deployment (mixed, move-regret, PHASE-STRATIFIED) test** on a theoretical shortcut. The
  move-regret sets ARE mixed all-phases; discrimination on a domain-specific corpus is NOT deployment validation.
- **Never conclude a cause from an eyeballed worst-N** — I called "endgames are the over-read" from a drift
  worst-20; the phase-split aggregate refuted it (endgame KS HELPS). Quantify the stratified distribution.

## STATE / BUILD / DISCIPLINES
HEAD `d58d47c` on `NN-ENgine`. Build/bench: `wsl.exe -e bash -lc "bash '<abs overnight_runner.sh>' <sub> [KEY=VAL]"`
(only the runner form auto-approves). Byte-id every build + `wac_speed` peak. |balanced STS| < ~150 UNRESOLVABLE.
Validate on MOVES/regret, not cp. Games decide, run ALONE, above the ~20-40 Elo floor. Commit only when asked,
NO footer. Discrimination corpus tiers: attack 64 / quiet_neg 128 / mid 108.
