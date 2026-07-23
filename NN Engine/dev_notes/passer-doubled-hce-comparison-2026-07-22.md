# Passed-pawn & doubled-pawn HCE comparison: ours vs SF11 vs Ethereal (2026-07-22)

Motivation: positional collapse dossier showed our passers under-fire (~0.1 where SF Passed = 1.5-2.7) and we
over-credit doubled pawns (drawn R+doubled-P endgame read +5.8). Verified SF11 term parser is correct
(`raw_sf11_dump.py`). KEY nuance: SF isn't directionally-right via a better passer term — its Passed OVER-credits
advanced pawns too (FEN6 +1.70 wrong-way); offsetting KS/threats + search save it. So the fix is JOINT + gated,
NOT a passer boost in isolation (user's standing caution: don't blow passers up).

## OURS (the root causes) — cpp_bitboard.cpp
- **`passed_pawn_support` breakdown (`br_passed`, ~L6519) does NOT contain the passer rank value** — only
  `boost_pieces_for_supporting_passed_pawns` (small piece-path nudges) + `passer_danger` (base 3500, DEFAULT-OFF
  via ENABLE_PASSER_DANGER/ENABLE_PASSER_V2). So the breakdown line reads ~0.1 (misleading).
- **THE SMOKING GUN: passer rank value routes into `pt_pawns` via `pawn_rank_bonus` and is HARD-CLAMPED to
  ±275 millipawns (0.275p) in the MIDGAME** (L827 white / L932 black: `std::max(pawn_rank_bonus,-275)`).
  `passed_midgame_pawn_rank_bonus_base` goes up to 1360mp but is neutered to 275 midgame. Endgame path
  (`evaluate_pawns_endgame`) is UNCAPPED (fine). ⇒ #1 reason passers under-fire midgame.
- **Doubled pawns: flat 125mp boolean penalty** (`>1 pawn on file`), no phase/structure awareness. AND
  **`getPPIncrement` mis-flags a REAR doubled pawn as PASSED** (checks only enemy pawns in the span, not a
  friendly pawn ahead) → both doubled pawns collect passer credit + full PST + chain bonuses, offset by only
  125mp ⇒ `place` over-credits doubled pawns (drawn R+P endgame #1).
- Passer value is SMEARED across pt_pawns(capped), rooks (ROOK_PASSER_OWN/ENEMY), attackingLayer, central,
  kings_endgame (PASSER_KRACE_MAG) — hard to control; breakdown line captures none of it.
- Knobs: PP_OPP_PAWN_PEN=125, PP_BLOCKADE_PEN=100, PP_UNBLOCKED=50, PP_DIAG_SUPPORT=75, PP_FILE_CLEAR=150,
  PP_HORIZ_SUPPORT=225, SCALE_PASSED_RANK=100, SCALE_ENDGAME_RANK=100, ENABLE_PASSER_BLOCKADE_QUALITY=true
  (PASSER_CONTEST_PCT=30). Conditioning machinery EXISTS but the ±275 clamp overrides it.

## SF11 (public constants; matches local src) — evaluate.cpp passed() + pawns.cpp
- `PassedRank[] = {0, S(10,28), S(17,33), S(15,41), S(62,72), S(168,177), S(276,260)}` — tiny low, huge 6-7.
- Conditioned by path-safety factor **k = 35 (whole path unattacked) → 20 → 9 → 0** (path/stop enemy-controlled),
  +5/+4 if own pieces defend path/blockSq; king-race (enemy king far = boost, scaled by rank w=5r-13);
  rook/queen behind passer nullifies enemy rear file-attack (shrinks unsafeSquares). Occupied blockSq ⇒ only
  base table. So BIG bonus ONLY for a safe runner.
- **Doubled penalty S(11,56) charged ONLY if `!support`** (not pawn-defended from behind) — supported doubled =
  NO penalty; eg(56) >> mg(11) = endgame liability. Isolated/backward worse on open files (WeakUnopposed*!opposed).
- Connected[] mg-base bonus, eg derived by advancement; separate from & additive with the passer bonus.

## Ethereal — src/evaluate.c evaluatePassed() + evaluatePawns()
- `PassedPawn[canAdvance][safeAdvance][rank]` 3-way table: stop-square empty × unattacked each scale UP
  (rank-6 eg 46 → 293). King distances (friendly close good eg, enemy far good). `SafePromotionPath S(-49,57)`
  (whole file clear+unattacked → cautious mg, big eg push). Candidate-passer table (`PawnCandidatePasser`).
- **Doubled = `PawnStacked[flag][file]`: HALF penalty if it can "unstack"** (advance/capture out), full if
  frozen; eg >> mg. Backward larger on open files. Connected32 table by rank×file.

## THE COMMON DESIGN WE LACK / THE FIX
Both give huge RANK-based passer bonuses GATED by PATH-SAFETY (SF's k=0..35; Ethereal's [canAdvance][safeAdvance])
+ king-race + blockade. We have the conditioning machinery but the blunt ±275 midgame clamp throttles ALL
passers instead of only UNSAFE ones. Both condition the doubled penalty (unsupported/frozen/endgame = more).

**Levers for the joint fit (code-grounded, gated, don't-blow-up):**
1. RAISE the ±275 midgame passer clamp (L827/932) toward the real table, PAIRED with stronger path/blockade
   conditioning (PP_*) so unsafe passers stay dampened. #1 positional fix.
2. CORRECTNESS: `getPPIncrement` must disqualify a pawn with a friendly pawn ahead on the file (stop the
   rear-doubled-pawn passer mis-flag). Fixes drawn R+P over-read (#1).
3. CONDITION the doubled penalty (endgame-heavier, unsupported/blockaded-heavier) — SF `!support` / Ethereal
   unstack-flag.
4. Optionally activate `passer_danger` as the clean dedicated channel vs the smeared pt_pawns route.
All calibrated in the joint fit vs SF18 truth (diverse_corpus). Tools: `positional_collapse_dossier.py`,
`sf11_breakdown_fens.py`, `raw_sf11_dump.py`.

## IMPLEMENTATION (passer redesign under fresh gate `ENABLE_PASSER_V3`) — plan `.claude/plans/well-for-the-suppressor-wise-steele.md`
- **Round 0 (DONE):** `build_passer_corpus.py` → `ks_sets/passer_corpus.csv` (288 rows: 140 blowup_guard / 79
  control / 69 under_fire; phase-labeled; SF18 truth; dossier FENs + hand-built guards incl. the KING-RACE PAIR
  `6k1/8/8/8/2ppp3/8/6B1/6K1 b` SF18 −0.29 vs `8/8/3k4/2ppp3/8/5B2/8/6K1 b` SF18 −5.17 = same 3 passers, king
  race flips 5 pawns; OUR eval ~identical & backwards = the acceptance test). Verify harness `passer_verify.py`
  (per tier×phase over_read vs SF18 + breakdown split). BASELINE: under_fire −4.40, blowup_guard +4.19,
  control +0.40; value lands in pt_pawns, passed_supp ≈0.
- **Round 1a (DONE):** rear doubled-pawn no longer mis-flagged as passed (`getPPIncrement` ~8157, under V3:
  `infrontMask & pawns & friendly` → return 0). New gate `ENABLE_PASSER_V3` (search_engine.h + env reg
  search_engine.cpp:1071 + toggle dump). NOTE: adding an `inline` knob needs the `env_flag(...)` registration
  too or env has no effect. Effect: #1 +5.80→+5.35, #3 −5.75→−5.49 (toward SF18); small (bulk is the clamp,
  Round 2). **byte-id V3-off = 247/39,971,153 (exact).** Fresh gate is a strict superset of V2; V2 removed once
  V3 ships. Sign-smear §6-A disable folded into Round 2 (competitor disable) for a clean one-change signal.
- **Round 2b (DONE):** multiplicative R-gate on the MIDGAME rank bonus, under V3. Mechanism: (1) extracted
  `passer_realizability_R(sq, white)` from `passer_danger` (pure refactor; byte-id preserved because
  passer_danger's values are unchanged); (2) in `evaluate_pawns_midgame`, when `ppIncrement>=100`, the passed
  rank bonus is NO LONGER added in-loop (where `attack_bitmasks` is incomplete) — it is stashed in a
  reset-per-eval global `g_passer_mid_deferred[sq]` (the full unclamped value; capgains still sees it via the
  unchanged `pawn_rank_bonuses` out-param); (3) a post-loop pass beside `passer_danger` (inside the `!isEndGame`
  `br_passed` block, full attack_bitmasks) prices each deferred passer once as `bonus * mid_w/range * R / 256`,
  where `mid_w/range` mirrors the pawn-loop blend split (phase_score 40→70) so the value = pre-clamp rank bonus
  × R/256 at the correct phase weight. Non-passers and V3-off are byte-identical (`total += std::max(...,-275)`
  path retained; only ever writes the stash under the gate). **byte-id V3-off = 247/39,971,153 (exact).**
  VERIFY (V3-on vs baseline): under_fire midgame −4.19→−3.96, dossier #2 (rr4k1..., real midgame passers)
  +4.17→+3.50 (toward SF18 −2.32), blowup_guard midgame +5.33→+5.30 (guard held). Direction correct, guards
  not blown, magnitude MODEST.
  - **KEY FINDING:** the midgame ±275 clamp was real but is the SMALLER share of the passer under-fire. The
    DOMINANT under-fire mass is in the **adveg (−6.13) and endgame (−3.64) tiers**, which flow through the
    `isEndGame` block + `evaluate_pawns_endgame` (uncapped rank bonus already) + `advanced_endgame_eval` —
    NONE of which Round 2b touched (both unchanged in the verify). So the endgame passer under-read is NOT a
    clamp problem; it is a magnitude/king-race/coordination problem in the endgame path. ⇒ Round 2c/Round 3
    must extend the R-gate (or equivalent) into the endgame pawn path + the deep-endgame king-race to move the
    dominant tiers.
- **Round 3a A/B (DONE — corpus, deterministic; the finding that redirected the plan):** enabled the all-phase
  king-race delta under V3 (6892) + bypassed AE's legacy copy (4861); byte-id V3-off 247/39,971,153. Corpus
  A/B (`passer_verify` env sweep) DISPROVED the "king-race is the dominant lever" hypothesis:
  - AE→delta swap is VALUE-NEUTRAL for adveg (same formula) ⇒ adveg under_fire UNCHANGED (−6.13) — adveg
    already HAD king-race; it is NOT a coverage gap.
  - `PASSER_KRACE_MAG=400` (4×) closed adveg only −6.13→−5.31 (~13% of the gap) while blowup_guard ROSE
    +4.16→+4.50 — king-race is a WEAK, over-read-prone lever for the dominant tiers.
  - `SCALE_ENDGAME_RANK=250` (2.5× base rank magnitude) closed adveg −6.13→−4.57 and endgame −3.64→−2.60 (more
    efficient than king-race) BUT blowup_guard rose +4.16→+5.11 and control +0.42→+0.96 — uniform magnitude
    hits the SAME bidirectional wall (raises stopped passers too).
  - **⇒ VERDICT: the dominant endgame/adveg under-fire is a BASE-MAGNITUDE gap (channel #1), not king-race.
    The fix = raise the endgame passer rank magnitude GATED BY R** (extend Round 2b's R-gate to
    `evaluate_pawns_endgame`), so the increase reaches realizable passers (R high) and NOT stopped ones (R low).
    King-race = minor R-modulated add-on, kept low. This is exactly the bidirectional (over+under) discipline.
- **Round 3b (DONE) — endgame R-gate (mirror 2b in `evaluate_pawns_endgame`) + king-race owned by delta.**
  Implemented: `g_passer_end_deferred[]` (reset per eval); eg passed component (ppInc>=300) deferred and priced
  once × R in BOTH the blend post-loop (with end_w) AND the isEndGame post-loop (full weight); `!ENABLE_PASSER_V3`
  added to `passer_realizability_R`'s king-term (5720/5736) so R = pure path-safety and the delta owns king-race.
  **byte-id V3-off 247/39,971,153.** Corpus A/B (all vs 2b baseline under_fire −4.35 / blowup +4.16):
  | knobs (V3=1) | under_fire | blowup_guard | control |
  |---|---|---|---|
  | R-gate, mag100, krace=0 | −4.72 | **+3.66** | +0.31 |
  | R-gate, mag250, krace=0 | −3.89 | +4.30 | +0.68 |
  | (no-R-gate mag250, krace on) | −3.44 | +5.11 | +0.96 |
  | **C: R-gate, mag150, krace mag100/pct0** | **−4.18** | **+3.98** | +0.46 |
  - **R-gate HOLDS the guard better** (mag250: +4.30 with-gate vs +5.11 no-gate; nominal mag100 alone drops the
    guard +4.16→+3.66 = R throttling stopped passers). But raising uniform magnitude still lifts the guard
    (mag100→250 with-gate: under −0.83, blowup +0.64) because R's PATH-SAFETY doesn't capture every "stopped"
    case: e.g. guard `8/2r4k/8/6PK/6P1` (our +5.32 vs SF +0.04) = 2 king-escorted passers a rook holds; nothing
    attacks the path so R≈256 → full magnitude leaks. That "rook-behind / king-can't-escort-two" realizability
    is a DIFFERENT channel (ROOK_PASSER / king-race blockModifier), not path-safety.
  - **⇒ VERDICT (measured, not asserted): passer valuation is genuinely MULTI-CHANNEL.** No single knob (rank
    magnitude, king-race, path-safety R) closes under_fire without lifting the guard — each guard position is
    stopped by a different mechanism. Candidate **C nets positive on BOTH tiers** (under −0.17, blowup −0.18,
    control flat) = the machinery works two-sided; full closure needs the JOINT constrained fit, not a lever.
## CENTRALIZATION (2026-07-22, user-directed) — `evaluate_passers()` build log
Full design in the plan `.claude/plans/well-for-the-suppressor-wise-steele.md` (RE-SCOPE section: architecture,
taxonomy, exact deletion manifest, EXPAND/uniqueness, sequencing). Two fable consults + SF11 source extraction
drove it. Uniqueness = graded control + win%-calibration, NOT a different formula.
- **A1 (DONE): `evaluate_passers()` core** (cpp_bitboard.cpp after passer_danger). One post-loop function: loops
  `white|black_passed_pawns`, magnitude = phase-blended `passed_midgame`/`endgame` rank tables, × `min(R,320)`
  (conservative upside), exports `priced_passer[64]`, Black-positive aggregate. Replaced BOTH transitional
  stash-pricing passes (the 2b/3b `g_passer_mid/end_deferred` post-loop loops) with one call in each branch.
  **byte-id V3-off 247/39,971,153.** Corpus (V3=1, krace pct0): under_fire −4.64, blowup_guard **+3.64** (vs 2b
  +4.16 / C +3.98 — central core HOLDS THE GUARD BETTER), control **+0.18** (vs +0.42). under_fire slightly
  worse (−4.64 vs −4.35) — expected: the composite-R terms that RAISE realizable passers (rear-file, king-prox,
  graded contest) aren't added yet; §6-A/rook-additive/king-race still layered on (gate off next).
- **A2 (DONE): composite-R terms folded into `passer_realizability_R` (all gated ENABLE_PASSER_V3, byte-id
  247).** New knobs (search_engine.h + env_int reg + toggle-dump): (i) **GRADED path contest** — replaces flat
  D2 with `PASSER_CONTEST_STOP/PATH * clamp(popcount(enemy att)−popcount(own def),0,3)` per path square (stop
  weighted worst) = our unique lever using attack_bitmasks COUNTS; (ii) **rear-file control** — first heavy
  piece behind the passer on its file: enemy R/Q docks `PASSER_REAR_ENEMY`(128), own credits `PASSER_REAR_OWN`
  (48); (iii) **king-proximity to stop** — `PASSER_KING_FAR`(16)*enemyKingDist − `PASSER_KING_HELP`(6)*ownKingDist
  (dist cap 5, enemy-far dominates). Corpus (V3=1): under_fire −4.60, blowup_guard +3.71, control +0.24 —
  aggregate flat (terms REDISTRIBUTE at un-calibrated defaults) but TARGETS move right: rook-holds guard
  `8/2r4k/8/6PK/6P1` +5.35→**+4.89**, adveg guard +3.74→**+3.08**, under_fire dossier `rr4k1` +3.34→**+2.42**
  toward SF −2.32. Closing is the FIT's job; A2 = wire + prove sane direction. §6-A/rook-additive/king-race
  still layered on top (mask the clean signal — gate off in A3).
- **A3-part (DONE): §6-A gated OFF under V3.** All four condition forms (`if (square_mask & {white,black}_passed_
  pawns)` openers + `} else if(...)` else-ifs, 3-tab, incl. the kings block) wrapped with `!Config::ENABLE_
  PASSER_V3 &&` via 4 replace_all (V3-off byte-id preserved 247). Effect (V3=1 vs A2): under_fire −4.60→−4.63,
  blowup_guard +3.71→**+3.62**, control +0.24→+0.20 — SMALL (confirms §6-A was minor realizability-blind noise,
  as predicted). Channel now clean: central fn rank×composite-R, no §6-A. **Current V3 baseline: under_fire
  −4.63, blowup_guard +3.62, control +0.20** (rook-additive + king-race delta still layered; default un-fit
  magnitudes). Guard already better than 2b (+4.16); under_fire recovers via the FIT raising composite-R for
  realizable passers.
- **A3b-rook (DONE): rook-additive gated OFF under V3.** All 10 sites (4 mid `ROOK_PASSER_OWN/ENEMY` + 6 eg
  ×75 literals) wrapped `if (!Config::ENABLE_PASSER_V3)` (keeps branch structure so a passer square doesn't fall
  to the else-penalty; the non-passer ×35/×50 support/blockade credits STAY). byte-id 247. Effect negligible
  (under_fire −4.63→−4.60, guard +3.61, control +0.22) — confirms it was small/redundant with R's rear-file
  term. **CENTRALIZATION CORE COMPLETE: passer value = `evaluate_passers()` rank×composite-R; §6-A + rook-
  additive both gated off. Clean V3 baseline under_fire −4.60 / blowup_guard +3.61 / control +0.22 at un-fit
  default magnitudes** (king-race delta + boost_pieces still layered additively).
- **A3c-kingrace (DONE): king-race moved per-passer into `evaluate_passers()`, soft-gated by R.** New helper
  `passer_king_race_one(sq,white,turn)` (per-pawn extract of `passer_realizability_delta`), added in the central
  fn as `val += king_race * max(R, PASSER_R_FLOOR=64) / 256` (fable's operator fix — can't leak past a dead
  passer). Aggregate delta @7009 gated OFF under V3 (removed V3 from its guard) to avoid double-count. Now
  king-race applies to PASSERS ONLY (vs the delta's all-advanced-pawns-in-half) = more passer-correct. New knob
  `PASSER_R_FLOOR` (registered). byte-id 247. **Corpus (V3=1): under_fire −4.39, blowup_guard +3.76, control
  +0.39** — vs A3b −4.60/+3.61: king-race RECOVERS under_fire (−0.21) at small guard cost (+0.15, soft-gating
  bounds it). vs the ORIGINAL 2b baseline (−4.35/+4.16): under_fire ~matched, guard **0.40 BETTER** — the fully
  centralized eval at DEFAULT magnitudes already beats 2b on the guard. Dossier under_fire `rr4k1` +0.82 (SF
  −2.32; was +2.42 at A2); guard `8/2r4k` +4.72 (was +5.35 at 2b). NOTE: per-passer king-race has NO phase-ramp
  yet (flat PASSER_KRACE_MAG all-phase) — midgame blowup +5.44 slightly high; the fit calibrates MAG.
- **CENTRALIZATION EFFECTIVELY COMPLETE:** all passer value now flows through `evaluate_passers()` (rank ×
  composite-R + soft-gated king-race), exports `priced_passer[64]`; §6-A + rook-additive + aggregate delta all
  gated off under V3; byte-id 247. Clean, well-behaved knobs ready for the Stage-1 fit.
- **REMAINING A3 (minor, refinements):** (i) wire capgains→`priced_passer` — ordering: capgains
  @6549/6856 runs BEFORE evaluate_passers populates priced_passer, so move the passer eval earlier OR have
  capgains R-clamp its own prb (search-eval coupling correctness). (ii) soft-gate the additive king-race — move
  king-race per-pawn INTO evaluate_passers() `×max(R,R_FLOOR)` and gate off the aggregate delta @6922 under V3
  (fable's operator fix: prevents guard-leak when the fit raises magnitudes). (iii) priced-once debug assert.
  Then hygiene (corpus split + mechanism-audit 140 guards + enemy-passer tier) → **Stage-1 FIT** (the real
  closing lever: calibrate the 6 composite-R knobs + SCALE_ENDGAME_RANK to raise under_fire while holding guard).
- **NEXT after A3 (superseded plan text below kept for reference): clean the channel** — gate OFF §6-A per-piece proximity credits (inventory lines) + the
  rook-additive `ROOK_PASSER_*`/eg-×75 under V3 (delete-in-waiting); add soft-gated additive king-race + support
  (`×max(R,R_FLOOR)`); wire capgains → `priced_passer` (NOTE ordering: capgains @6549/6856 runs BEFORE
  evaluate_passers populates priced_passer — must move the passer eval earlier OR have capgains R-clamp its own);
  priced-once assert. Then hygiene (corpus split + mechanism audit + enemy-passer tier) → Stage-1 fit.

## ★★★ GAMES VERDICT (2026-07-23) — CATEGORICAL WIN, SHIP V3-DEFAULT
3 seeds × 200g SF@2400 conc3, V3-DEFAULT (`ENABLE_PASSER_V3=1`, no Stage-1 knob overrides) vs baseline.
Matched-seed `profile_collapses.py`:
| class | base avg/seed | v3 avg/seed | Δ |
|---|---|---|---|
| **positional (TARGET)** | 59.0 | **53.0** | **−6.0 (−10%)** |
| ks_attack | 17.0 | 20.7 | +3.7 (next class exposed) |
| material | 2.0 | 3.0 | +1.0 |
| ks_and_material | 0.7 | 0.7 | 0 |
| TOTAL | 78.7 | 77.3 | −1.4 (flat) |
- **Positional DOWN in ALL 3 seeds** (s0 67→56, s1 58→54, s2 52→49) = robust categorical win, not one-seed.
- TOTAL flat (SF exploits next weakness); ks_attack +3.7 = the NEXT target (mostly seed1 12→27, noisy).
- **Score +2.8%** (base 41.7% → v3 44.5%; per-seed 37.5→44.8 / 47.0→42.8 / 40.5→45.8; won 2/3 seeds). A
  categorical win PLUS a score bump — better than the score-neutral KS count-gate. On top of STS +30 bench.
- Per-seed collapses: base 91/71/74, v3 79/85/68.
- **VERDICT (KS-model): passer positional class mostly clean → SHIP V3** (flip `ENABLE_PASSER_V3` default true
  when ready), then MOVE ON to the next class = **ks_attack / OvD attack-over-read** (our eval over-reading our
  own attacks — the memory's known lead). Remaining before ship: constexpr build+byte-id, speed-reclaim (stash
  ppIncrement; contest popcounts), mixed-family win% confirm, then DELETE scattered code + V2.
- Games in `selfplay/games/{base,v3}_s{0,1,2}/`; stale prior-session base_s0 moved to `_bak_base_s0`.

## A3d + STAGE-1 SWEEP (2026-07-23) — capgains consistency, 2 new knobs, Stage-1 candidate, static ceiling
- **A3d (DONE):** capgains passer-value now clamped by `CAPG_PAWN_RANK_CLAMP` under V3 too (7889/7900,
  `ENABLE_PASSER_V2 || ENABLE_PASSER_V3`) so capture-ordering doesn't over-fear a dead passer. byte-id 247.
- **2 new knobs:** `PASSER_MAG_SCALE` (passer-specific base-magnitude scale INSIDE evaluate_passers, independent
  of the global SCALE_ENDGAME_RANK so tuning passer magnitude doesn't disturb non-passer eg pawns) and
  `PASSER_R_CAP` (upside cap on R, was hardcoded 320). All 9 composite knobs env-registered + toggle-dumped.
- **STATIC CEILING (measured):** `PASSER_R_CAP=384` changed the corpus NOTHING ⇒ no corpus passer reaches R>320
  ⇒ under_fire passers have only MODERATE realizability, so their only lever is base magnitude — which is
  COUPLED to the guard (raising it lifts both tiers; the shared dampening knobs pull both back). This IS fable's
  predicted static ceiling: the under_fire−vs−SF18-static gap is partly uncloseable statically. ⇒ stop
  fine-tuning statically; games decide (KS-model).
- **STAGE-1 CANDIDATE (best two-sided static, self-contained): `PASSER_MAG_SCALE=150 PASSER_CONTEST_STOP=140
  PASSER_CONTEST_PATH=70 PASSER_REAR_ENEMY=180`** (others default) → **under_fire −4.09, blowup_guard +3.90,
  control +0.48** (vs default −4.39/+3.76; vs 2b −4.35/+4.16; vs candidate C −4.18/+3.98 — beats all on the
  two-sided balance). byte-id V3-off 247. This is the config to GAME-TEST (multi-core, when free) for the
  KS-model categorical verdict.
- **MOVE-MATCH GUARD (GREEN):** wrote `diagnostics/passer_movematch.py` (loads `suites/passers.csv`, run_one
  vs sf_best, knobs via argv→env). Baseline V3-off **171/405=42.2%** → Stage-1 candidate V3-on
  **177/405=43.7% (+6)**. The passer redesign IMPROVES move-choice (tracks Elo better than static MSE), not
  just static eval. Stage-1 candidate wins on BOTH static (under_fire↓, guard held) AND move-match.

## ⚠️ BENCH CHECK (2026-07-23) — STAGE-1 PASSER-CORPUS TUNE OVERFIT; the CENTRALIZATION helps STS at DEFAULT
Ran the benches (were previously only byte-id V3-OFF; NEVER V3-on). Baseline V3-off: WAC 247, STS 1555/3000.
| config | WAC | STS | nodes |
|---|---|---|---|
| baseline V3-off | 247 | 1555 | 40.0M |
| **V3 DEFAULT knobs** | 242 (−5) | **1585 (+30)** | 42.6M |
| V3 + Stage-1 (MAG=150+damp) | 243 | **1531 (−24)** | 42.6M |
- **THE STAGE-1 PASSER-CORPUS TUNE WENT THE WRONG WAY:** raising passer MAGNITUDE to close corpus `under_fire`
  HURT STS (1585→1531, −54 vs V3-default). The passer-corpus under_fire metric is a MISLEADING optimization
  target — the +6 passer-move-match masked the −54 STS loss. Reconfirms [[sts-wac-tag-artifact]] / the durable
  lesson: **static corpus metrics are BLIND to move-choice; the BENCH is the guard.** DO NOT tune to minimize
  passer-corpus under_fire.
- **THE CENTRALIZATION ITSELF (default knobs) is a POSITIONAL WIN: STS +30** (1555→1585) — and STS = the
  positional collapse class we're targeting. Costs WAC −5 (tactical) + ~6% nodes (the passer eval changes move
  ordering/pruning). Net bench-positive (STS gain > WAC loss), and it's the RIGHT direction.
- **⇒ RE-ORIENT (user, load-bearing): abandon the passer-corpus-only optimization target.** Tune with STS + WAC
  + KS/OvD as GUARDS in the loop (the Stage-2 mixed/joint fit the user always specified — I skipped straight to
  a passer-only candidate). Keep passer MAGNITUDE LOW (default/less; 150 was harmful). Target = hold/raise STS
  (+30 already) while RECOVERING WAC (−5) + not regressing KS/OvD. Then the GAMES verdict. The passer corpus is
  a LEADING INDICATOR / guard-tier check only, NOT the optimizer.
- **TODO next (single-core):** (a) find the WAC/node regression cause (passer eval changing tactical choice or
  just cost); (b) mixed-family regression check (KS/OvD/total on diverse_corpus under V3-default); (c)
  STS-guided knob search (maximize STS + hold WAC, not minimize corpus under_fire).

## COMPOSITE-R SPEC (SF11 source extraction + fable second opinion, 2026-07-22) — the two missing channels
SF11 `passed()` prices a passer PER-PAWN as `PassedRank[r] + (k + kingProx)·w`, `w=5r−13`, with:
- **Rear-file control (missing #1):** `bb = forward_file_bb(Them,s) & pieces(ROOK,QUEEN)` = heavy piece BEHIND the
  passer on its file. NO enemy heavy behind ⇒ `unsafeSquares &= attackedBy[Them]` (danger shrinks, k high);
  ENEMY heavy behind ⇒ skip the shrink ⇒ whole span unsafe ⇒ k→0/9 (the "enemy rook holds it" case); OWN heavy
  behind ⇒ `k += 5`. k-ladder {35,20,9,0}·w ⇒ at r6 up to +595cp. Our R NEVER inspects the rear file = the gap.
- **King-proximity/escort (missing #2, eg-only):** `(kp(Them,blockSq)·19/4 − kp(Us,blockSq)·2)·w`, dist capped 5,
  enemy-king-distance DOMINATES; second-push dock `− kp(Us,blockSq+Up)·w`. This is what prices the
  `8/2r4k/8/6PK/6P1` guard correctly — the DEFENDING king next to the stop square kills the bonus.
- Ethereal confirms via `PassedPawn[canAdvance][safeAdvance][rank]` + per-rank friendly/enemy king-distance
  tables + `SafePromotionPath` (whole file clear+unattacked). No rook-behind term (relies on `attacked[THEM]`).

**RECONCILIATION (fable vs SF source):** fable argued "king-can't-escort-two is a SET property, not per-pawn ⇒
needs a per-side saturation cap." The SF source SETTLES it: SF handles it PER-PAWN via king-proximity-to-stop-sq
— a king can only be near one stop square, so the OTHER passer's stop square gets `kp(Them)` large ⇒ high bonus,
which is CORRECT (the unescorted passer genuinely IS dangerous). ⇒ **the guard leak is because I DISABLED R's
king term under V3 (5720/5736 edit), not because R is fundamentally per-pawn-limited. Fix = restore a proper
SF-style enemy-king-proximity-to-STOP-SQUARE term in R (not the delta's blunt square-rule, not the old coarse
D4). fable's per-side cap = held in reserve, likely unnecessary (SF has none).**

**⇒ THE COMPOSITE-R (per-pawn, additive into R∈[0,384], 256=neutral, multiplicative on rank bonus):**
1. **Rear-file dock/credit** (highest priority — the measured leak): enemy R/Q on the passer's rear file with
   clear line ⇒ `R -= PASSER_REAR_ENEMY` (~128, halves the pawn); own R/Q behind ⇒ `R += PASSER_REAR_OWN` (~48).
2. **King-proximity into R** (replaces the disabled D4/delta king dim for the V3 gate): `R += clamp(
   DEF_KING_NEAR·(enemyKingDistToStop cap5) − OWN_KING_HELP·(ownKingDistToStop cap5), −64, +96)`; SF magnitudes ⇒
   `DEF_KING_NEAR≈16`, `OWN_KING_HELP≈6` (enemy-king-far dominates). Then DELETE the separate delta king dim
   under V3 (already toggled off) to avoid double-count.
3. **(optional) graded path ladder:** dock MORE when the STOP square itself is unsafe vs only deeper path
   squares (SF's `!(unsafeSquares & blockSq)` tier), instead of our flat per-square D2.
New knobs (`PASSER_REAR_ENEMY/OWN`, `DEF_KING_NEAR`, `OWN_KING_HELP`) need `env_flag()` reg + toggle-dump.

## FABLE MEASUREMENT-HYGIENE (fold in BEFORE the Stage-1 fit — independent of the composite-R)
- **capgains inconsistency (MINE, real):** the 2b deferral keeps `pawn_rank_bonuses` full-unclamped so capgains
  is byte-clean — but capture-ordering now prices a blockaded passer at full ~1360 while eval prices it table×R.
  V2 clamped this (`CAPG_PAWN_RANK_CLAMP`); V3 needs an analog (clamp the capgains prb by R too). Deliberate.
- **§6-A: NOT sign-wrong** (a white piece near ANY passer is white-favorable = support OR restraint; I verified
  the knight site) but IS realizability-blind and LIVE during every A/B ⇒ disable under V3 + re-measure before
  fitting so the baseline isn't contaminated (ambiguous effect — corpus decides, per earlier).
- **corpus discipline:** `passer_corpus.csv` has NO train/val split (add a held-out third) + selection bias
  (built from failures we found — top up blind from `position_bank.csv`); AUDIT the 140 guards by stopping-
  mechanism (blockade/path/rear-rook/king-escort/fortress) so the fit can't hold one and leak the rest; ADD an
  ENEMY-PASSER-OVER-FEAR tier (our historical worse failure + the OvD lead — all 3 tiers are our-passer only).
- **stopping criterion (load-bearing):** `under_fire→0 vs SF18-STATIC is unreachable AND unnecessary` (collapses
  are tactical ~2.5-3× depth; SF11's own Passed is imperfect, saved by search). DONE = beats candidate C on
  corpus + holds guards + move-match → GAMES; if games neutral at ~50% closure, residual is search's job (the
  KS-count-gate lesson). Don't chase the static asymptote.
- **consolidation end-state:** one `evaluate_passers()` post-loop fn owning its own phase blend (the current
  hand-mirrored `mid_w/range` at the post-loop pass is a latent desync vs the pawn-loop blend 6285) + a written
  DELETION MANIFEST (the ~9 channels + 2 stashes + mirrored weights that must die once V3 ships).
- Doubled-pawn flat 125mp penalty (:827) still unconditioned — fell off after 1a; recondition (`!support`/phase)
  before shipping (the other half of the original dossier finding).

- **Round 3c (was NEXT) = Stage-1 constrained fit** over {`SCALE_ENDGAME_RANK`, `PASSER_KRACE_MAG`,
  `PASSER_KRACE_MG_PCT`, `PASSER_DANGER_D2`, and folding rook-behind/king-escort into the gate} minimizing
  under_fire s.t. blowup_guard held (+control, +move-match `passers.csv`), on `passer_corpus.csv`. Reuse
  `ks_fit_diverse.py`/`_ks_fit_eval.py` machinery. Then Stage-2 joint fit on `diverse_corpus.csv`. Candidate C
  (`PASSER_KRACE_MG_PCT=0 PASSER_KRACE_MAG=100 SCALE_ENDGAME_RANK=150`) is the fit's warm-start.
- **Round 2c (LATER):** disable the §6-A sign-smeared per-piece passer credits under V3 (targets the
  blowup_guard over-read +4.16) AND/OR extend the R-gate into `evaluate_pawns_endgame` + the isEndGame post-loop
  (targets the dominant adveg/endgame under-fire). Endgame extension intersects Round 3's king-race — do it
  carefully to avoid double-counting `advanced_endgame_eval`'s own passer/king-race. Tools:
  `build_passer_corpus.py`, `passer_verify.py`. Insertion pattern established by Round 2b (defer in the pawn
  fn, price post-loop with `passer_realizability_R`).

## DETECTOR AUDIT (2026-07-22) — VERDICT: (c) POOR COORDINATION, not missing detectors
We HAVE nearly every SF/Ethereal passer signal, but they aren't combined into one board-driven per-pawn GATE
on the rank bonus, and the best ones are DEFAULT-OFF:
- **Path-safety gate reading enemy PIECE ATTACKS on the path/stop-sq = the secret sauce**: EXISTS as
  `passer_danger` (cpp_bitboard.cpp:5679, reads `attack_bitmasks`, BLOCK[] stop-sq quality, −D2 path defenders,
  +D4 enemy-king dist; base {3500,2200,1200}) — **DEFAULT-OFF** (`ENABLE_PASSER_DANGER`). The active-path
  `getPPIncrement` (:8090) is occupancy/PAWN-only, NO piece-attack test.
- **Consolidation architecture EXISTS**: `ENABLE_PASSER_V2` (:823) prices a passer ONCE via realizability +
  zeroes the ~9 smeared duplicate channels — **DEFAULT-OFF** (its own comment names the exact failure mode).
- **King-race** (`passer_realizability_delta` :4696, AE king-race :4849-4949) = ENDGAME-ONLY (gate :6825);
  midgame version `ENABLE_PASSER_KRACE_MG` DEFAULT-OFF → advancing MIDGAME passers get no king-race pricing.
- **Coordinated & OK**: rook-behind (`ROOK_PASSER_OWN/ENEMY` :2168, file-aware sign-correct), blockade quality
  (`getPPIncrement` + `ENABLE_PASSER_BLOCKADE_QUALITY=true`).
- **Smoking-gun MISTUNE**: rank table (up to 1360) CLAMPED to ±275 midgame (:827/:932); the only realizability
  reaching it = `ppInc>>3` (max +50mp additive) + a binary threshold flip at ppInc=100.
- **Genuine bug (flag)**: scattered per-piece passer credits (knights :1234, bishops :1801, queens :2655, pawns
  :884/992) add to the PIECE'S OWN COLOR regardless of which side owns the passer = SIGN-SMEARED noise. Endgame
  king (:4410/:4457) + rooks are the sign-correct exceptions.
- **Genuinely MISSING (tertiary)**: candidate/wing-majority passer (`PAWN_MAJORITY_*` default 0),
  stop-square-attacked in the ACTIVE path, `!support`-gated doubled-passer penalty.

**⇒ THE FIX IS REWIRING + CALIBRATION, NOT NEW DETECTION.** Activate the consolidation (`passer_danger`/`V2`) as
the single board-driven per-pawn gate; REMOVE the ±275 clamp + smeared flat credits it replaces (so no
double-count/competition — the likely reason V2 was NO-GO before, judged on SPRT ON TOP of the old channels);
bring king-race into the midgame blend; fix the sign-smear; add the small missing bits. Calibrate
`passer_danger` constants (BASE1/2/3, D2/D4) to SF18 on diverse_corpus; judge by SF18 magnitude + collapse
profile + GAMES (not self-play SPRT). This is the central `passer_weight[sq]` gate the user wants, mostly
already present in skeleton.
