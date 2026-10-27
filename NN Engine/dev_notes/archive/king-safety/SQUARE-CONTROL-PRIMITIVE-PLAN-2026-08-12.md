# Square-control primitive upgrade — the high-value morning candidate (2026-08-12)

**Status: PLAN ONLY, not started.** Queued for the morning after tonight's bundle SPRT resolves. This is the
natural successor to `defaware1` and the most on-thesis idea on the board: upgrade the *root primitive* that
every detector reads, so the improvement compounds across KS **and** OvD, central, passers, capgains, evasion —
not just king safety.

## ★★★★ 2026-08-12 THE REFRAME — KS is a "WHEN-to-fire" problem, and the detectors are a "how much" win
After building 4 discrimination-validated detectors (SQC/pins/weak-val/flank, AUC 0.748->0.810) they made general
move-quality WORSE (footprint +0.14..+0.54 over-fire). Diagnosis chain: over-fire is NOT endgame (phase-split:
opening +0.37 / midgame +0.27 / **endgame −0.32 = KS HELPS there**) — it's the OPENING. Root cause = the
**"WHEN-to-fire" (context gating) system**, which we under-analysed for months by only ever asking "how much".
### SF/Ethereal architecture (Opus map, source-verified): when+how-much are ONE signed object
ONE signed `kingDanger` accumulator: positives (attacker weight×COUNT product, weak, checks) + large NEGATIVE
suppressors in the SAME units (−873 no-queen, −shelter, −6·score/8 already-winning, −flank-defense) → they NET →
one threshold (>100) → SQUARE (after all gates, so it amplifies NET danger never proximity) → phase as a
**two-function mg/eg pair** (mg=square, eg=different tiny linear fn) blended by NON-PAWN MATERIAL.
### Ours gets all three "when" shapes wrong (file:line)
1. `KS_FLOOR=13` = fixed MAGNITUDE floor (chops same off all) vs SF's emergent NET-SIGN threshold. `:5571`
2. `ks_phase_taper` = single LINEAR output scale on one scalar vs SF's per-term two-function mg/eg pair. `:5617`
   `KS_PHASE_ZERO` = deep-endgame CLIFF vs continuous material weighting (under-gates R/Q endgames). `:5607`
3. attacker term = flat additive SUM (`:5372`) vs SF count×weight PRODUCT (no super-linearity → lone piece
   counts as much per-unit as a group). `KS_MIN_ATTACKERS` off+inert-w-queen `:5564`; `KS_NO_QUEEN` single-digit
   vs SF −873 `:5545`.
### Ordering (from OUR phase data, correcting the agent's phase-first suggestion)
Over-read is the OPENING = a COORDINATION problem (we fire on presence/breadth; SF needs a coordinated group).
So FIRST lever = **count×weight coordination gate**, validated on footprint STRATIFIED BY PHASE (predict: opening
improves, midgame neutral, endgame unchanged). THEN the full signed-accumulator + net-threshold + two-function
phase rebuild. Detectors (discrimination-validated) feed the POSITIVE side of that one object.
### ☠️ METHODOLOGY LESSON (why we missed it for months)
We asked only "how much" and DECOMPOSED into isolated pieces (primitive/curve/channels) — never the SYSTEM
integration. And (owner-corrected) the error was NOT "no mixed corpus" — the move-regret sets
(`game_regret_set`+`v2`) ARE mixed all-phases; it was DEFERRING that deployment move-test for pins/weak-val/flank
on a wrong "honest inputs are move-neutral" theory, validating them on the KS-specific DISCRIMINATION corpus
(AUC 0.81) only. The mixed move-regret caught the over-fire immediately once run; phase-stratification localized
it. Fixes: ask WHEN+HOW-MUCH for every term; map INTEGRATION not parts; do NOT defer the deployment (mixed,
move-regret, phase-STRATIFIED) test on a theoretical shortcut; when corpus-win but deployment-loss, change the
QUESTION not the tuning. (Whacky/variant set built but UN-USED — fold in for non-opening diversity.)

## ☠️ 2026-08-12 VERDICT — the coupled curve redesign FAILED the cross-set, and WE KNOW WHY. Bank +15, shelve.
Tested the coupled redistribute+compound (honest SQC inputs + weak/safe-check UP + proximity DOWN + raise KNEE
so the quadratic runs over the live 13-51 range, DIVISOR holds top magnitude). Two principled configs:
`A(WEAK5 SC12 ATTACKCOUNT0 KNEE40 DIV6)` and `B(milder)`. **Footprint BOTH cross-sets: A +0.038/+0.115,
B −0.018/+0.208 — both WORSE, v2 badly worse (the largest degradations measured).** 0-for-9 held.
★ **The DRIFT analysis (`_ks_drift_analysis.py`, config A on v2) gives the mechanism:**
- Root-unit inflation is UNIFORM (WORSE +3.3 vs BETTER +3.1) — compounding inflates danger BROADLY, not
  selectively on genuine danger.
- The damage pattern: the worst drift positions are almost all `base = reg 0.0 (SF18-best) → cand = reg 30-70`
  — it **BREAKS already-correct moves**, doesn't fix bad ones. ~equal flip counts (2317 worse / 2245 better)
  but the breaks are far bigger than the fixes ⇒ net-harmful.
- Move changes flow through the SEARCH (config changes LEAF KS evals → different lines), not the root — inflated
  danger pulls move selection toward ATTACK-CHASING when SF18 preferred quiet; leaks into endgames (units 26-33
  in 7-piece positions past the taper) where king ACTIVITY ≠ safety.
- ⇒ **You cannot compound units that don't discriminate.** Our units are proximity-dominated; compounding
  amplifies genuine AND false danger alike, and the false-danger amplification dominates. Weight-redistribution
  + square_control did NOT make the units discriminating enough to survive squaring. **The curve stays dead until
  the units genuinely discriminate — a deeper detector-quality problem than we can crack now.**
▶️ **For a future attempt:** the prerequisite is UNIT DISCRIMINATION (genuine-danger kings must land at clearly
higher units than proximity-only kings) BEFORE any compounding — and neither the flat weights nor the value-aware
contest achieved it. Do not re-try compounding without first proving units discriminate on the cross-set.

## ★★★ 2026-08-12 UPDATE — this is STEP 1 of a 3-step KS danger-model redesign (Opus analysis)
Two analyses (primitive map + danger-model analysis) reshaped the picture:
- **SCOPE CORRECTION:** OvD and `central` are NOT readers of `attack_bitmasks` — they are loop-siblings fed by
  `attackingLayer[colour][x][y]`. So square-control does NOT compound across the whole eval; the zero-ripple
  reroutes are **KS-internal only** (defaware `:5407`, weak `:5339`, safe-check feeder `check_safe` `:5470`) +
  the separate passer lane. It is a **KS-quality upgrade, not a whole-eval primitive swap.** Live capgains is
  population-bound and already uses SEE — leave it. x-ray/battery is population-level — **fence it**.
- ★★★ **THE DOMINANT KS GAP IS THE PRESSURE→DANGER CURVE, AND OUR QUADRATIC IS DEAD.** `rebuild_ks_tables`
  (`:412`) builds `danger = units²/KS_DIVISOR` below `KS_KNEE=12`, linear above. But `KS_FLOOR=13 > KS_KNEE=12`
  ⇒ every position that clears the deadzone floor (units ≥13) is already PAST the knee, on the **flat linear
  ramp**; the quadratic (units 0–12) is **never read in production**. Our live KS is a shallow LINEAR proximity
  meter — no compounding. SF accumulates with LARGE coefficients (400–1200) then squares above 100
  (`kingDanger²/4096`) ⇒ super-linear "this king is lost" intuition. That FLOOR≥KNEE pathology is the mechanical
  reason we don't statically sense compounding danger. (Nonlinear bolt-ons `KS_INTERACT`/`KS_ATT_PRODUCT`/`KS_DYN`
  are all additive, gated off, part of the additive-KS 0-for-9 history.)
- **WEAK-SQUARE:** form is correct (SF K/Q-only + attackedBy2) but FLAT — no per-square (attacker-value × weak)
  coupling; value enters only via the separate union term. This is really part of the curve defect (SF's weak
  compounds because it shares the squared curve). **PINS:** zero pin awareness in KS (a pinned "defender" counts
  full); SF prices +98/blocker + drops pinned pieces. Cheap separate fix (`slider_blockers`, helper exists `:8625`).

### ★★★★ 2026-08-12 CAPSTONE (Opus history+pipeline synthesis) — the curve gap is a COLLINEARITY problem
- **Live trace CONFIRMS the dead quadratic:** 100% of dangerous kings land at units 13-51 (median 18), 0% in the
  quadratic band; units are NARROW + **~85% PROXIMITY** (weak=2/safe-check=3 are ~10% of the sum).
- **`square_control` into defaware = MOVE-NEUTRAL** on the cross-sets (−0.063/+0.024, sign-flipped) — honest inputs
  are INERT on a linear curve; a gated mechanism whose value needs the compounding. Kept built + gated, not folded.
- **THE MECHANISM why compounding is 0-for-9:** the king-attack proximity is **counted ~4× at linear order** —
  `attackingLayer[x][y]` is added (i) directly to `total`, (ii) into the OvD accumulators (`:6852` comment: OvD is
  "COLLINEAR with the same attackingLayer cells already added to total"), (iii) king-sliced by `KS_ZONE_ATTACK_PCT`;
  + (iv) unit-KS reads `attack_bitmasks` proximity independently; + flat mg-shelter duplicates `KS_SHIELD`. So
  squaring `units` amplifies QUADRUPLE-COUNTED PROXIMITY NOISE, not signal. ★ **The dead quadratic has been
  PROTECTING us** — "just fix FLOOR≥KNEE" would make us worse. Every additive/compounding lever failed; every
  subtractive/de-dup lever won (de-king +7.4%, REALIZ +36.7, defaware1). Channel law governs.
- ⇒ **The redesign is CONSOLIDATE → REBALANCE → COMPOUND (order forced by the channel law):** (a) collapse the 4
  king-credit channels into ONE honest accumulator (untried `ENABLE_KS_V2` shelter re-home + partial OvD king
  de-king + keep attackingLayer @50% + defaware1); (b) widen weak/safe-check vs proximity, redistributive, on the
  SINGLE channel; (c) coordination gate + fix FLOOR≥KNEE so the quadratic squares SIGNAL. Multi-session, but each
  stage is subtractive (the winning direction) and reuses built levers.
- 🧰 **MINIMAL GO/NO-GO (zero games, deterministic):** measure the 4-channel split on attacked-king positions
  (ablate each channel, check collinearity — extend `_ks_channel_decomp.py`). If channels 3/4/5 don't materially
  duplicate unit-KS ⇒ thesis WRONG, STOP. If they do ⇒ build the consolidated single-channel config (magnitude
  held ≈constant), footprint BOTH cross-sets: sign-consistent + units distribution WIDENS (weak/check share rises)
  ⇒ proceed; regresses/inverts ⇒ channels are load-bearing (de-dup sheds signal), STOP.

### ★★★ THE 3-STEP PROGRAM (fix order is the REVERSE of impact — curve is biggest but MUST be last)
Reshaping the curve to compound pressure while the units are still DISHONEST (proximity over-read, pinned
defenders, value-blind weak) would amplify FALSE pressure = the 0-for-9 additive trap. So:
1. **`square_control` first** — honest per-square verdicts (LVA + pawn-exclusion + attackedBy2) into
   weak/defaware/`check_safe`. Validated direction (`defaware1`). Both regret sets + variant + STS + symmetry + byte-id.
2. **Pin mask** — `slider_blockers` per king: drop pinned defenders from `dm`/`dmS`; optional `+K*popcount(blockers_for_king)` term.
3. **Curve reshape LAST** — fix `FLOOR≥KNEE` so the quadratic runs; widen the discriminating weights toward SF's
   ratio; let now-honest pressure compound. Highest magnitude, highest risk ⇒ games-gated; only pays on top of 1–2.
⚠️ Before step 3, trace real lost-king positions (`KS_DEBUG_DUMP` `:5590`) to see what unit range they land in
(deadzone vs low-linear) — sizes the redesign. Curve reshape is a MAGNITUDE move ⇒ full symmetry + tournament gate.

## The thesis (why this is higher-leverage than any single detector)
`attack_bitmasks[s]` — the per-square "who attacks this square" OR-mask — is the shared root primitive. KS
(`defaware`, weak, safe-check), OvD accumulators, central, passer contest/blockade, capgain tension, and evasion
all read it. `defaware1` (validated this session) is really just *one consumer* doing a smarter read of it.
**Upgrade the primitive's notion of attack/defend and every downstream detector inherits the gain at once.**

The session result that motivates this: **the win comes from upgrading detector INFO, not magnitude** — `defaware1`
(better info, subtractive/redistributive) generalized on both cross-sets; the additive magnitude levers
(`SC=8`, `attackedBy2`-as-danger) overfit and were rejected. Making the primitive itself higher-quality is the
same move, one level deeper.

## What is crude today (never upgraded)
The per-square contest is raw piece-**count**: `popcount(attackers) − popcount(defenders)` (see
`king_safety_danger`, `overload`/`contested_zone`). It is:
- **Value-blind** — a *pawn* defender cancels a *queen* attacker 1:1. Badly wrong; a pawn guard is near-decisive.
- **Exchange-blind** — never asks *who wins the square* if pieces actually traded on it (no SEE-lite).
- **X-ray-blind** — occupancy-limited, so batteries / pieces-behind-pieces undercount (the `KS_AIM` blindness,
  which was patched additively and is dead).

## The upgrade — "defaware 2.0", a read-only `square_control(s)`
Replace count with **least-valuable-attacker vs least-valuable-defender**: the side that can occupy the square
*last and cheapest* controls it. We already split the per-square masks by type inside the zone loop
(`am & knights`, `am & queens`, …), so the cheapest attacker/defender is a few extra mask tests per square —
**no SEE simulation, cheap.** Optionally layer SF's cheap refinements: `attackedBy2` (defended-twice ⇒ safe) and
pawn-defense exclusion. Turns "2 attackers vs 1 defender" into "can the attacker *profitably* contest this."

## Observe the giants FIRST — extract FORMS, not constants (`sf-schedule-portability-heuristic`)
Send fable to map, from source:
- **SF11 / SF15.1-classical:** `attackedBy[c][pt]`, `attackedBy2`, the "safe square" notion (not attacked by an
  enemy pawn / adequately defended), pawn-attack exclusions, and where SEE gates a square's value.
- **Ethereal:** its least-valuable-attacker threat logic (`SafetyThreats`/`KingSafety`), attackedBy layering.
- Port the **shape** (least-valuable-attacker / attackedBy2 / pawn-exclusion); refit only a global scale here.
  Do NOT free-fit cells — small resolvable signal ⇒ winner's-curse risk.

## Map the consumers SECOND — so we don't break what it feeds
Before changing anything, enumerate every reader of `attack_bitmasks` and classify each:
- **POPULATION reader** (depends on HOW the mask is built — x-rays, blockers): passers, capgains, evasion.
  ☠️ **Do NOT change the population** — high ripple, all consumers move at once, un-validatable, plan-fenced.
- **Per-square contest reader** (reads the mask, computes its own attacker/defender verdict): KS/`defaware`,
  OvD, central. ✅ Safe to reroute to `square_control(s)` — KS-local per consumer.
- **The safe-check feeder (2026-08-12 insight):** the `check_safe` lambda in `king_safety_danger` decides a
  check square is "safe" by raw `bm & own` (defender *presence*). This is the SAME crude primitive. Safe-check
  failed as a MAGNITUDE lever (`SC=8`, `CHECK_V2` both refuted on the v2 cross-set), but its **input** is crude:
  a check square "defended" only by a pawn is still dangerous if the checker is cheaper. ⇒ Rerouting `check_safe`
  to `square_control(s)` is a detector-INFO upgrade of the safe-check *feeder* — the un-tried angle on safe-check,
  distinct from the (dead) magnitude route. Another consumer to reroute + validate independently.

## Hard rules (the disciplines that make this safe)
1. **Never mutate `attack_bitmasks` population.** Build `square_control(s)` as a NEW read-only function; leave
   the mask alone. This is what keeps ripple to zero.
2. **Adopt consumer-by-consumer, KS first.** `defaware` → `square_control` (zero ripple), validate; *then* OvD,
   validate; *then* central. Each hop independently gated (default byte-id) and cross-set-validated. Compound
   only proven gains ("sweep consumers first / a fix's value can be what it unblocks").
3. **NPS-bounded by construction** — static eval is already 65–84% of per-node cost; measure `wac_speed` peak,
   compare, keep the per-square cost to a few masks.
4. **Validate EACH hop** on footprint-filtered D7 regret + **both** cross-sets (`game_regret_set` +
   `game_regret_set_v2`) + balanced STS orig+mirror + colour/file symmetry + byte-id at default. Accuracy ≠ Elo.

## Sequence (morning, post-SPRT)
1. Fable maps the giants' square-control forms + our full consumer graph (POPULATION vs contest readers).
2. Build gated `square_control(s)` (least-valuable-attacker, +attackedBy2/pawn-exclusion options), default byte-id.
3. Route `defaware` to it (defaware 2.0) → footprint + both cross-sets + STS/symmetry/NPS. If it beats the
   count-contest, it replaces `defaware1` in the bundle.
4. If it generalizes, extend to OvD then central, one validated hop at a time. The compounding across consumers
   is where the "high value beyond just KS" lives.

## 🔭 Variant/960 as a structure-independent validation set (owner idea 2026-08-12)
Both regret sets (15k + v2) are mined from STANDARD selfplay ⇒ they share opening structures, so a term can
score by memorizing "in these known structures do X" rather than encoding real chess. **Variant/960 positions
break that scaffolding** ⇒ the sharpest test of whether a detector (esp. `square_control`) is genuine chess
knowledge vs standard-structure overfit. **SF18 is a valid arbiter** (it computes, not books; handles standard
pieces + `UCI_Chess960` natively).
- ✅ **Low-friction TODAY:** custom start ARRAYS that keep king-e / rooks-a&h (standard castling still legal —
  our engine already plays these, e.g. the owner's bishops-for-knights game) with the inner pieces shuffled.
- ⚠️ **Full 960 prerequisite:** verify our C++/Cython engine implements `UCI_Chess960` castling first.
- ⚠️ **Confound:** our PSTs / king-zone tables are standard-tuned, so variants also expose THOSE weaknesses ⇒
  use variant/960 as a **generalization check** (does the detector TRANSFER?), NOT a primary tuning target
  (standard chess is the deployment distribution). A detector that helps on standard AND transfers = robust.
- ▶️ Build a 960/variant regret set (varied-array selfplay + SF18 multi-PV @d14) as a THIRD cross-validation
  set for the square-control primitive + future detector upgrades.
- ★★★ **NO-CASTLING is the point, not a limitation (owner insight 2026-08-12).** Kings stay central/exposed
  with NO clean pawn shield ⇒ a KS regime our detectors NEVER trained on (`KS_SHIELD`/open-file/storm/zone are
  all calibrated on the castled-king archetype). This specifically stress-tests the two subsystems in the current
  bundle: (a) **KS** — does it read a central king's danger from general principles or memorized castled patterns?
  (b) **central-vs-KS BALANCE** — with the king in the centre a central pawn push is simultaneously a `central`
  GAIN and a king EXPOSURE, so the two terms are in direct tension; standard positions (flank king) DECOUPLE
  them, so no existing instrument can see this interaction — the variant set is the ONLY place we can measure it.
  It's also the hardest, most discriminating test for square-control's KS consumer (a central king = many
  contested squares). ⚠️ Far from the deployment distribution ⇒ generalization/diagnostic check, NOT a ship gate.
- ✅ **BUILT + smoke-tested (2026-08-12): `diagnostics/_build_variant_regret_set.py`** — generates whacky
  legal positions from piece-replacement + 960-style arrays (castling DISABLED ⇒ no UCI_Chess960 needed),
  short random walk, SF18 multi-PV @d14 label, same schema, resumable, SHARD=i/n parallel. Smoke N=20 = valid
  rows consumable by `_ks_footprint_regret.py`. ⚠️ Short walks bucket everything as "opening" — the SCALE run
  wants `WALK_MAX≈45 MAT_TOL≈700` for midgame/endgame spread. Morning: run sharded (e.g. N=8000) → then footprint
  the square-control primitive on it as the transfer test.

## Tools (all built this session, reuse)
`_ks_footprint_regret.py` (footprint D7 regret, base vs candidate, changed-move filter) · `game_regret_set.csv`
(15k) + `game_regret_set_v2.csv` (11,940 disjoint) · `_ks_unit_screen.py` / `_ks_c1_decomp.py` (firing decomp)
· `_eval_symmetry.py` (colour/file) · `overnight_runner.sh sts_suite` (balanced STS) · `wac_speed` (NPS peak).
