# KS signed-accumulator rebuild — the integrated "WHEN-to-fire" object (design spec, 2026-08-13)

**Status: DESIGN FOR REVIEW.** Written overnight after Stage-1 (coordination gate) confirmed necessary-but-
insufficient in isolation (opening null, midgame −0.135 at div=4 — an isolated positive lever with no
counterbalance). This spec is the **minimum coherent unit**: net suppressors → threshold → square-after-gate →
two-function phase, all in ONE object, so a positive lever is never judged without its counterbalances.
**Magnitude choices are flagged ⚑ — those are the review points (where my stories go wrong).**

## The principle (why this object, not more levers)
SF/Ethereal fuse *when* and *how-much* in ONE signed accumulator. Every isolated lever we've tried (additive KS
0-for-9, the coupled curve, the four detectors, now the coordination gate) regresses because it acts as a
universal magnitude rule with nothing to net it below a threshold in the phases/positions where it's wrong. The
fix is not a better lever — it's building the **counterbalanced object** so each lever becomes an *input to a
contextual decision* instead of a standalone rule. Test the WHOLE loop; a positive term alone is not testable.

## ⚠️ CODE-INVENTORY CORRECTIONS (2026-08-13, from KS-CODE-INVENTORY map) — read before building
1. **DO NOT add a shelter suppressor.** `KS_SHIELD` already subtracts pre-net (cpp_bitboard.cpp:5509) → it is
   mechanically inside the accum's `net` already. Shelter is TRIPLE-channeled (flat 185/75 in
   evaluate_kings_midgame :3116-3186 + KS_SHIELD + inert ENABLE_KS_V2). A new suppressor triple-counts. Delete
   "shelter" from the suppressor list below — it is already present.
2. **The phase cliff is UPSTREAM of the accum.** `king_safety_score` early-outs `phase_score >= KS_PHASE_ZERO(104)
   → return 0` (:5724) BEFORE calling `king_safety_danger`, so the accum's eg branch is DEAD until that early-out
   is gated for KS_ACCUM_MODE. The two-function phase step MUST gate that early-out, or the endgame branch never runs.
3. **`KS_WIN_SUP` overlaps the ACTIVE `MOD_KS_REALIZ=128`** (post-map material damp, :5769-5782). Keep KS_WIN_SUP=0;
   if tested, validate JOINTLY with REALIZ (both key on the attacker material deficit). Never enable MOD_KS_BACKING
   alongside REALIZ.
4. **Linear map slope confirmed continuous:** live curve is `danger = 36 + 6·(units−12)`; `KS_ACCUM_LIN=96/16=6`
   reproduces it — magnitude continuity is real, not assumed.
5. **Retire, don't reuse:** `KS_ATT_PRODUCT` (additive-FAILED) and `KS_MIN_ATTACKERS` (hard gate) — the accum uses
   `KS_COORD_GATE_MODE` (the replace-form). Do not stack all three.
6. **Proximity has TWO homes:** `attack_count_units` in `units` (demotable via KS_ATTACK_COUNT) AND the live
   `KS_ZONE_ATTACK_PCT=50` attackingLayer channel (:9227). Demoting `units` addresses only the unit-KS home (the
   correct one for the KS accum); the attackingLayer proximity is the OvD/central lens, out of accum scope.
7. **Safe-check feeder** (step 2): `check_safe` (:5560) is crude `bm & own` AND ignores `~own_pinned` — the
   square_control swap + the pin exclusion are both one-liners.

## The object (SF form, OUR scale, all gated default-off = byte-id)
Current live scale: `units` ~13–51 for real attacks, `KS_CAP=80`, `danger = units²/KS_DIVISOR(4)` below
`KS_KNEE=12`, linear above; `KS_FLOOR=13` (fixed magnitude floor); single `ks_phase_taper` + `KS_PHASE_ZERO=104`
deep-EG cliff. The rebuild replaces the floor, the taper, and the flat attacker sum — one gate, `KS_ACCUM_MODE`.

### ★ 0. THE POSITIVE-SIDE REBALANCE IS PART OF STEP 1 (not a later step) — owner insight 2026-08-13
Thresholding the *existing* proximity-dominated `units` just thresholds PROXIMITY — a high-proximity/no-danger
king still clears the bar, so the opening over-read survives past the threshold. Therefore the positive-side
rebalance MUST be in the first accum test: **demote proximity** (so the threshold keys on danger), **elevate
safe-checks / weak** (the genuine-danger signals), and **upgrade the safe-check FEEDER** (`check_safe` still uses
the crude `bm & own` presence test — re-route through `ks_sqc_breaks` so a pawn-only "defended" check square is
still dangerous when the checker is cheaper; the un-tried, detector-info route on safe-checks, distinct from the
dead `SC=8`/`CHECK_V2` magnitude route). All these when-signals + the suppressors are **magnitude-COUPLED** — a
−35 no-queen only means something relative to the *demoted-proximity* positive size — so the **unit-trace on real
lost-king positions sizes them JOINTLY**, then validation stays incremental (attribution). Joint derivation,
attributed confirmation, never a blob.

### 1. POSITIVE side (danger sources) → summed into a signed `net`
- **attacker coordination** = count×weight PRODUCT (the Stage-1 gate) — *coordination-aware input*, NOT a
  standalone rule here. [ours-tuned divisor]
- **weak squares, value-coupled** (`KS_WEAK_VAL`) — detector, ours.
- **safe checks, per-type** (`KS_CHECK_V2`) — existing, SF-form.
- **flank breadth, contest-weighted** (`KS_FLANK_MODE=2`) — ours (measurably less over-fire than SF-raw).
- **pins** (pinned defenders dropped, `KS_PIN_MODE`) — detector, ours.
- **proximity** (`attack_count`) — KEPT but **demoted** ⚑: it's the ~85% over-read source. Candidate: it should
  contribute little to `net` directly and instead only *qualify* squares as attacked. Review: demote vs remove.

### 2. NEGATIVE side (suppressors) in the SAME units, derived on our scale
Derivation anchor: SF kingDanger ~0–1500, threshold 100, square /4096; no-queen −873 (≈58% of a 1500 max),
so a queenless "attack" nets near-silent. Porting the FRACTION to our ~0–80 unit scale:
- **no-queen** ⚑ `KS_NQ_SUP`: SF −873/1500 ≈ 58% → on our scale ≈ **−30…−46 units** (vs today's KS_NO_QUEEN=6,
  which is why queenless false-attacks survive). Keyed on the enemy-of-this-king's queen. **Primary review knob.**
- **shelter** — reuse `KS_SHIELD` (pawn shield) as a suppressor into `net`, not a separate late subtraction.
- **already-winning discount** ⚑ `KS_WIN_SUP`: SF −6·score/8. Ours: −k·max(0, own_material_edge) so a side already
  up material doesn't over-invest in a speculative king-hunt. Derived k on our millipawn edge. **Review knob.**
- **king-defended** (optional, SF −100·(N&K defend)) — defer to v2 unless cheap.

### 3. NET → THRESHOLD (replaces the fixed KS_FLOOR)
`net = positives − suppressors` (signed, may go negative). `if (net < KS_ACCUM_THRESH) danger = 0;` ⚑
Threshold derived as a fraction of typical real-attack `net`, NOT the old fixed 13. The point: the effective floor
becomes **per-position** (a sheltered queenless king self-nets below threshold; an exposed king with a queen and
open lines clears it) — this is the mechanism that kills the opening over-fire WITHOUT a blanket floor that also
chops real attacks. **This is the load-bearing "when" element.**

### 4. SQUARE AFTER the gate
`danger = (net - KS_ACCUM_THRESH)² / KS_ACCUM_DIV` ⚑ — squares NET danger (post-suppressor, post-threshold), so
super-linearity amplifies *real coordinated danger*, never proximity. This is the opposite of the dead-quadratic
trap: the old quadratic squared raw proximity-dominated units; this squares the *netted, thresholded* signal.
DIV derived so a max real attack maps into today's danger range (continuity with KING_SAFETY_MAG).

### 5. TWO-FUNCTION mg/eg PHASE (replaces single taper + KS_PHASE_ZERO cliff)
`mg = squared (above), eg = small linear fn of net`, blended by **non-pawn material** (not phase_score cliff):
`danger = (mg·npm + eg·(NPM_FULL−npm)) / NPM_FULL` ⚑. Removes the `KS_PHASE_ZERO=104` cliff that under-gates R/Q
endgames (endgame KS *helps* per the phase-split, so the eg branch must stay live, gently). npm threshold derived.

## Ours vs copied (the uniqueness ledger)
- **Copied FORM** (absolutely superior, no better known): net-sign accumulator, >threshold gate, square-after-gate,
  two-function phase. These shapes are SF's and we adopt them because there is no competitive alternative.
- **Ours**: contest-weighted flank, value-coupled weak, graded per-square defaware, the coordination divisor, the
  demote-proximity choice, and every MAGNITUDE (derived on our 0–80 scale, never ported constants, never corpus-fit).

## Knob plan (all gated, default = byte-identical)
`KS_ACCUM_MODE=0` (off) = today's path exactly. `=1` = the object. Sub-knobs: `KS_NQ_SUP`, `KS_WIN_SUP`,
`KS_ACCUM_THRESH`, `KS_ACCUM_DIV`, `KS_ACCUM_EG_K`, `KS_ACCUM_NPM`. The Stage-1 `KS_COORD_GATE_MODE` feeds the
positive side when the accum is on. defaware composes via the code fix (subtract `attacker_units` actually
computed, not the assumed flat sum). Byte-id proof: MODE=0 must reproduce `250 / 35,426,396 / EBF 3.800`.

## Validation plan (deterministic, before ANY games)
1. Byte-id at default (exact or stop).
2. Phase-stratified move-regret (`_ks_phase_split.py`) at derived defaults, BOTH cross-sets + the whacky set.
   Pass: opening improves (the suppressor+threshold now nets the proximity over-read below the bar), midgame
   NEUTRAL-or-better (the coordination product is now gated, not a standalone rule), endgame preserved (eg branch).
3. Discrimination sanity (`_ks_discrimination.py`) — AUC must not collapse; a near-inert arm in every sweep.
4. Compare vs the **defaware bundle** (deployment-relevant), not vs flat.
5. Only if move-positive on the deployment cross-sets → fold into the bundle → ONE SPRT (cross +2.94 or a
   many-game CI cleanly excluding zero). Fallback: SPRT the +15 alone.

## Open questions FOR REVIEW (do not let me settle these alone)
- ⚑ **no-queen magnitude** — 30 vs 46 units is the difference between "gently discount" and "silence queenless
  attacks"; the phase-split at a couple values decides, but the *range* is a judgment call.
- ⚑ **demote vs remove proximity** — proximity is the over-read, but it's also how we detect an attack exists at
  all; removing it may blind the detector. Likely demote (qualify-only), review.
- ⚑ **threshold derivation** — as a fraction of real-attack net; needs the unit-trace on real lost-king positions
  (`KS_DEBUG_DUMP`) to size it, which I'll run as part of the first pass.
- ⚑ **is midgame-worse fundamental or divisor-artifact** — the running sweep answers this; if div=2 is midgame-
  neutral, the coordination product needs no rescue and just feeds the positive side cleanly.
