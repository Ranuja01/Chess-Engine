# Outcome-Texel retune — scoped campaign plan (2026-07-04)

Step 2 of the fixed-nodes pivot (`fable-audit-2026-07-04.md`): jointly retune the EXISTING eval SCALE constants
against GAME OUTCOMES, validated through the calibrated `node_ab` gate. Lead pre-NNUE eval campaign
([[outcome-texel-campaign]]). Escapes the per-term deadness trap: a joint fit moves correlated subspaces of
constants together (mathematically unavailable to single-knob tuning).

## What ALREADY EXISTS (reuse — scoping win)
- **`selfplay/tune_corpus.py`** — samples `(fen, result_white)` from recorded games (`game.jsonl`), labels each
  with OUR per-term static eval (`ev_breakdown`) + SF11 per-term + detectors; `--mirror` (left-right symmetry),
  `--per-game` cap (intra-game de-correlation), `--resume` (killable). **36,604 game.jsonl available** across 139
  tags → millions of positions on tap.
- **`selfplay/tune_fit.py`** — ALREADY has outcome-mode Texel: `target_sig(mode="result")` fits the game result
  through a logistic; joint multi-term SCALE fit (`--terms`), train/held-out control split, per-stratum gap
  report; scales map directly to the engine's `SCALE_*` knobs (`round(100*s)`). ALL_TERMS = the full eval family.
- **`node_ab` gate** (calibrated: reproduced C1 −202 as −194±58, ~3.6× faster than lightning) — the validation.

## GAPS to build (small, scoped)
1. **`tune_corpus --no-sf11`** — outcome-Texel fits OUR terms to `result_white`; it does NOT need SF11 (that's
   for the conditioning/distillation fit). SF11 launch per position is the dominant cost. A skip flag → a
   10×+ larger corpus for the same time. (Keep `status` strata from our own eval sign, or drop for outcome mode.)
2. **`tune_corpus --quiet-only`** — standard Texel hygiene: keep only quiet positions (static ≈ qsearch, side not
   in check, no hanging positive-SEE capture). Fable: "filter unquiet via qsearch-resolution, NOT sharpness"
   (keeps sharp-but-quiet positions, drops tactically-unresolved ones that inject label noise).
3. **`tune_fit` FITTED-K** — currently `K_PAWNS=2.0` FIXED. Add the standard Texel K-fit (minimize outcome MSE
   over K with scales fixed, then fit scales). Memory's "fitted-K discipline" — the logistic slope that best maps
   OUR eval → win% is a real, non-degenerate fit for the OUTCOME target (unlike the scale-invariant SF-total fit
   where fitting K was degenerate — that's why it was pinned).
4. **By-GAME held-out + light ridge** — split the control set by GAME not position (same leakage lesson as
   `bench_split`); optional Δscale ridge so no single term takes an extreme value (the pre-ship overfit guard).

## Corpus spec
- Source: broad sample across the 36k games (many tags/configs → outcome diversity). `--per-game 6–8`.
- Labels: `result_white` ∈ {1,0.5,0}. **λ-blend option**: outcome bias exists (self-play games reflect OUR play,
  not ground truth) → tune_fit can blend outcome with an SF-eval label to de-bias (both modes exist; add a blend).
- Size: target ~200–500k quiet positions for the fit (K + ~15 scales is low-dimensional → plenty).
- Mirror on (symmetry). Quiet-only on. No SF11 (fast).
- **DUAL-USE**: this same corpus (fen, result, + optional SF label) is the NNUE training seed. Build it large and
  clean — it funds both the immediate retune AND the eventual NNUE go/no-go. Keep the generator reusable.

## Campaign loop
1. Generate the quiet outcome-corpus (`tune_corpus --no-sf11 --quiet-only --mirror --per-game 6`).
2. Fitted-K joint Texel over the SCALE_* family (`tune_fit --target result --terms <family> --fit-k`), report
   train vs by-game held-out; reject extreme scales.
3. Candidate `SCALE_*` config → `node_ab` vs base (the calibrated gate) → if positive, lightning/blitz SPRT → ship.
4. Then Fable step 3: **margin-family re-sweep** vs the retuned eval (the eval→pruning flywheel — cleaner eval
   lets the pruning margins cut harder; `RFP_MARGIN`/futility/LMP re-tuned against the new baseline).

## OVERFITTING DISCIPLINE (user 2026-07-04 — the joint fit's power IS its overfit risk)
The joint fit escapes per-term deadness by moving many constants together — which is exactly where overfitting
bites hardest (more free parameters). Guards, in order of importance:
1. **`node_ab` + SPRT is the ONLY thing that ships.** The fit is a hypothesis GENERATOR; no fit number (train OR
   held-out error) ever flips a default. This is the −202 lesson at full strength: fit metrics ≠ strength.
2. **Regularize toward the STATUS QUO (s=1), not toward 0.** Ridge on the scale *deltas* so a constant only moves
   off its shipped value when the data strongly demands it — noise leaves constants put.
3. **Held-out by GAME, not position** (the [[collapse-eval-overread-fix]] bench_split leakage lesson); a final
   untouched holdout looked at ONCE (iterating on a holdout turns it into training data).
4. **Trust STABILITY, not point estimates.** Fit across disjoint game-splits; believe only scale moves that are
   sign+magnitude stable across splits (cf. the mobility curve: direction shard-robust, magnitude noisy → trust
   direction only). Unstable move = overfit noise → don't ship.
5. **Clusters, not kitchen-sink.** Fit small correlated clusters (rook family; threat family; …), each validated
   in games before the next. This is "one type of issue at a time" done WITHOUT collapsing back to per-term
   deadness — and it keeps the free-parameter count (hence overfit surface) low per step.
6. **Build guards incrementally.** Start with the minimal fit (fitted-K + by-game holdout), SEE what it does, then
   add ridge/cross-split only if the first fit shows instability or extreme scales — don't pre-build a
   regularization framework before a single data point (that's its own premature-optimization overfit).

## BUILD STATUS + first smoke (2026-07-04)
- BUILT: `tune_corpus --no-sf11` (skip SF11 launch, status from our eval sign) + `game` id column (by-game split);
  `tune_fit --fit-k` (Texel K-fit on the result target) + by-GAME held-out split (falls back to by-position for
  old corpora without the game column). Both compile-clean.
- **Smoke on the OLD SF-conditioning corpus** (`cond_corpus_v2.csv`, n=37222, by-POSITION, 4 default terms):
  fitted **K=2.80**; fit proposes central +39% / latent_threat +28% / capg +25% / passed 0.86 — BUT the OBJECTIVE
  barely moves: control result-loss 0.13822→0.13806 (0.1%), **result-logloss 0.5312→0.5312 (unchanged)**. ⇒ live
  demonstration of the overfit trap (confident scales, ~zero objective gain) AND a yellow flag consistent with
  [[sf11-texel-scale-invariance]] / "aggregate-calibrated" — the eval may already predict OUTCOMES near-optimally
  in aggregate (error is in the tail). NOT conclusive (wrong corpus: SF-oriented sampling, by-position/leaky, 4
  terms, likely decided-position-heavy). DECISIVE TEST = a fresh `--no-sf11` by-game corpus + full-family fit; if
  control outcome-loss still ~flat → global retune is low-yield → reprioritize to the KS/tail work sooner.
- Rule reconfirmed: trust the control-set OBJECTIVE delta (and node_ab games), NEVER the scales.

## ✅ DECISIVE RESULT (2026-07-04) — live-scalar retune is TAPPED; residual = a REPRESENTATION gap (dynamic)
Fresh `--no-sf11 --mirror` by-GAME corpus (183,574 rows / 16,659 games, 4,997 control), fitted K=2.60, joint fit
over the live positional family (capture_gains, passed_pawn_support, latent_threat, central, imbalance_w/b,
pair_bonus, piece_value_boost):
- **Control result-loss 0.13201 → 0.13198 = 0.02% = FLAT.** Train moved 15× more than control (overfit that
  doesn't generalize). Fit proposes EXTREME scales (imbalance→20%, passed→44%) chasing train noise. ⇒ **retuning
  the live scalars against game outcomes is low-yield, well-powered + by-game confirmed.** (Reconfirms the
  aggregate-calibration memory [[sf11-texel-scale-invariance]] — but see the reframe below.)
- **KEY REFRAME (user):** a flat retune is NOT "eval calibrated" — it can also mean the residual error lives in
  DIMENSIONS the live features don't span (missing REPRESENTATION SPACE). Retuning reweights existing features; it
  cannot add a dimension. Banked SF11 audit (`our_eval_reference.md` KEEP/BUILD/RETIRE): we BUILT-but-GATED
  mobility/pawn_struct/pawn_majority/outpost (=0 in corpus, untouchable by retune), NEVER built Initiative /
  trapped-rook / connected-pawns / OCB-scaling, and have imbalance as a SCALAR vs SF's TABLES.
- **RESIDUAL CORRESPONDENCE** (`tune_fit --corr`, SF11-labeled corpus): residual [SF−ours, +=we under-read]
  correlates most with **king safety 0.19**, **rooks 0.15**, **mobility 0.11**; DYNAMIC bundle 0.10 >> STATIC 0.02.
  Weak in magnitude (older by-position corpus → pointer not proof) but the DIRECTION is unambiguous: our
  irreducible error is in the DYNAMIC dimensions (king attack / mobility), not static placement.
- **TRIANGULATION:** KS-as-#1-residual converges with (a) the collapse tail (KS under-read), (b) Fable's
  independent "build gated king-danger", (c) the STS Advancement "Space+Mobility gap". Three independent methods.

## PIVOT (2026-07-04) — from RETUNE to REPRESENTATION EXPANSION
1. Retune (live scalars) = DONE, tapped. Served its purpose: proved the ceiling + pointed the residual.
2. NEXT = expand representation in the DYNAMIC dims the residual tracks, as CONDITIONAL terms ("realizability
   sliders": value × f(cheap detector) — the flat versions died; the conditioned version is untried and is how our
   one eval win, capg-tension, worked). **King safety FIRST** (triangulated), then mobility. We already have the
   machinery (`KS_INTERACT`/`KS_DYN`/danger²-curve — [[ks-detection-rebuild]]) → RE-ADJUDICATE + condition under
   the calibrated `node_ab` gate + a blitz/longer-TC venue (KS payoff grows with TC), NOT a rebuild.
3. NNUE floor — evidence-gated: if conditioned KS/mobility ALSO fail to move the dynamic tail, the residual is in
   a representation HCE can't cheaply express → the honest NNUE trigger. Not before.
Optional strengthener before committing: regenerate an SF11-labeled BY-GAME corpus + re-run `--corr` for a firmer
residual pointer (current is weak/by-position). Given the triangulation, proceeding to KS is defensible now.

## MOBILITY — deferred, but re-frame when we get there (user insight 2026-07-04)
Prior mobility attempt failed on TWO counts: (a) lowered NPS, (b) didn't cut nodes much. BUT — exactly like the 5
KS deaths — that was measured in ISOLATION with FROZEN search. It may be an isolation/coupling artifact, not a real
dud. When mobility is next (after KS): tune HOLISTICALLY — how the ENTIRE eval contributes to the mobility signal
(don't double-count with the cheap-mobility surrogates / rook activity / central), AND how the search knobs interact
(the eval↔search coupling: a cleaner mobility may only pay off after the pruning margins are re-swept for it). Same
lesson as KS: outcome-compass proposes (+0.0022, #2 lever), but conversion needs play-magnitude + coupling re-tune,
not a frozen-search isolation test. DO NOT start until KS is resolved (one thing at a time — user).

## GAP-MAP (2026-07-04, user method: mass-compare our breakdown vs SF11, classify) — `diagnostics/breakdown_gap.py`
Per SF11 feature, ours-vs-SF11 across the corpus (cond_corpus_v2, 37k, OLD/by-position — pointer not proof):
| feature | mean_sf | mean_ours | corr | fires% | class |
|---|---|---|---|---|---|
| material | 0.050 | 0.033 | 0.61 | 80 | OK |
| space | 0.015 | 0.034 | 0.60 | 99 | OK |
| **king safety** | 0.042 | **0.001** | **0.27** | **43** | **MALFORMED** |
| mobility | 0.030 | 0.005 | 0.78 | 76 | mis-scaled (tracks, 6x under) |
| **threats** | -0.002 | 0.011 | **0.23** | 66 | **MALFORMED** |
| passed | 0.008 | 0.001 | 0.35 | 51 | OK-ish |
| pawns | -0.002 | 0.002 | 0.58 | 67 | OK |
- **By CORRELATION (do we track the feature's variation): king safety 0.27 + threats 0.23 = MALFORMED** — the two
  DYNAMIC dims. Material/space/pawns fine (~0.6, no representation gap → consistent with the flat retune). Mobility
  TRACKS SF (0.78) but ~6x under-scaled (mis-scaled; but flat-mobility was dead in games → conditioned form).
- **Validates the user's realizability correction:** KS is ALREADY a board-content function (its board-dependence
  IS its realizability) — so the gap is NOT "add a slider", it's the FUNCTION COMPUTES THE WRONG VALUE (barely
  tracks SF, fires <half as often). Classification = MALFORMED contents, not missing, not mis-scaled.
- **Caveats:** old by-position corpus, coarse our->SF mapping, and these are AVERAGES (the collapse gap lives in
  the TAIL, which averages wash out). SF-agreement ≠ strength (still node_ab/SPRT-gated).
- **NEXT DATA STEP (continue the method, decide from data):** regenerate a proper SF11-labeled BY-GAME corpus +
  re-run breakdown_gap STRATIFIED by sharpness (near-equal/midgame = the decision tail) — confirm KS-malformed
  holds AND widens in sharp positions. THEN decide the KS-function fix from that.

## ✅✅ COMPASS CONFIRMED — incremental-validity screen (2026-07-04) `diagnostics/incremental_validity.py`
Fable audit #2 resolved the compass paradox: reweighting-flat (fact 2) + KS-not-in-span (corr 0.27) + KS-outcome-
load-bearing = a MISSING feature = real lever, judged by OUTCOMES not SF-agreement. Rule: **SF as feature LIBRARY
(borrow its outcome-selected term DIRECTIONS) — sound; SF as TARGET (match its magnitudes) — proven dead.**
Operationalized as incremental validity: fit sigmoid(a·our_eval + b·f) vs game result, held-out result-loss DROP
vs baseline sigmoid(a·our_eval). Baseline (our eval only) = 0.138376; flat-retune floor = 0.00003. Δ held-out:
| + SF11 feature | Δ | |
|---|---|---|
| **king safety** | **+0.003556** | ~100× floor = REAL LEVER |
| **mobility** | **+0.003005** | ~100× = REAL LEVER |
| rooks | +0.001727 | ~57× |
| material | +0.001575 | ~52× |
| threats | +0.001291 | ~43× |
| passed | +0.000739 | ~25× |
| pawns / space / imbalance | ~0 | no signal (imbalance=0 confirms our unique-lens, no SF analog) |
**KS + mobility carry large real outcome signal our eval can't synthesize.** Consistency: static features (space,
pawns) ~0 = matches the flat retune; imbalance ~0 = our own lens.
- **CAVEATS:** by-POSITION on the old corpus (intra-game correlation inflates Δ → OPTIMISTIC; by-game bootstrap is
  the rigorous version). Δlogloss screens REPRESENTATION; a feature can be outcome-predictive yet lose games (leaf
  noise / NPS) → the calibrated node_ab gate + SPRT remain the truth. Adding SF's ACTUAL column upper-bounds what
  our reimplementation achieves.
- **VERIFIED (source):** SF11 threats = simple STATIC bitboard patterns (ThreatByMinor/Rook/King, ThreatBySafePawn,
  ThreatByPawnPush, RestrictedPiece; no motif-finders; attack bitboards already computed) per sf11_eval_reference.md
  → Fable's "~a day" build is credible.
- **CORRECTION to Fable:** it deprioritized mobility as "a rescale the fit left alone" — but the decisive fit did
  NOT include mobility, and incremental validity ranks it #2 (+0.003 > threats). Mobility is a real missing feature
  (safe-mobility redefinition), higher priority than Fable placed it.

## ✅ BY-GAME CONFIRMATION (rigorous, 2026-07-04) — signal HOLDS
Fresh SF11-labelled BY-GAME corpus (36,965 rows / 7,995 games), 4 folds, whole-game holdout, coefficient-sign-
stability. Baseline (our eval) 0.132671. Mean Δ ± std, all sign-stable:
KS **+0.00311**±0.0003 · mobility **+0.00222**±0.0004 · rooks **+0.00196**±0.0004 · threats **+0.00141**±0.0002 ·
material +0.00115 · passed +0.0004 (marginal) · imbalance/space/pawns ~0 (space sign-UNSTABLE across folds =
noise, good sanity check). Δ shrank modestly vs by-position (intra-game correlation removed) but RANKING + SIGNAL
robust. **NEW: rooks = #3 lever** (our rook terms buried in `pieces`; SF's dedicated rook eval under-represented).
Confirmed priority: **KS > mobility > rooks > threats**. Screening stage of the funnel VALIDATED.

## FUNNEL (Fable, verified) — SF proposes → Δlogloss screen → node_ab gate → SPRT ships
1. **RIGOROUS confirm:** fresh SF11-labelled BY-GAME corpus → re-run incremental validity with by-GAME folds +
   coefficient sign-stability + λ-blend(SF-WDL) agreement (does the KS/mobility ranking survive intra-game shrink?).
2. **Threats FIRST** (cheap ~a day, less depth-sensitive → clean fixed-node gate = validates the whole funnel on an
   easy case) → build our threat patterns → Δlogloss(our impl) → node_ab → SPRT.
3. **KS rebuild** (biggest prize, longer build; gated-additive attack-unit form whose gate kills the midgame
   collateral that ate all 5 prior attempts) → Δlogloss → node_ab + BLITZ venue (KS pays off with depth) → SPRT.
   Why this retry differs from the 5 deaths: gated-additive (not damped-magnitude) form, fit-set (not hand-cranked)
   calibration, OUTCOME compass (not SF-agreement), calibrated gate (not ±80-floor SPRT) — all 4 causes addressed.
4. **Mobility** = safe-mobility redefine (exclude pawn-attacked squares), fit-set weight (now #2 by data).
5. **Owed regardless (cheap, gate-born):** margin-family re-sweep (all margins calibrated PRE-RFP; re-sweep after
   KS/threats shift the eval-noise profile) + **ProbCut** (still unbuilt — new MECHANISM, where search Elo comes
   from) + graveyard re-adjudication (capture-hist / 2-ply cont-hist / seecap / KS_DYN variants) now cheap on the
   3.6× gate + a properly-powered fixed-node SPSA over the margins.
6. **NNUE** = the tail residual AFTER KS/threats land (data flywheel runs free meanwhile: every node_ab game feeds
   the by-game corpus; 183k is 1-2 orders below NNUE scale but grows as a byproduct). Not now.

## Risks / discipline
- Self-play outcome bias → λ-blend + by-game held-out + node_ab SPRT truth gate (never ship on fit error alone —
  the −202 lesson: fit metrics are not strength).
- Keep it NPS-neutral (SCALE_* are magnitude knobs, no eval-cost change) → `node_ab` is EXACT for it (the calib
  regime). If any candidate changes eval cost, add a time gate.
- byte-id: SCALE_* changes are HCE value changes (SHOULD change play) — not byte-id-preserving; the fingerprint
  gate is for GATED knobs, not these. Re-baseline after shipping.
