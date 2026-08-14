# KS failure trajectory — the full history the signed accumulator must not repeat (2026-08-13)

**Purpose:** a chronological, evidence-cited map of every distinct king-safety attempt since 2026-06, the
mechanism of each failure, and what each one obliges the `KS_ACCUM_MODE` signed-accumulator rebuild
(`KS-SIGNED-ACCUMULATOR-DESIGN-2026-08-13.md`) to do differently. Every claim cites its source doc; numbers
are quoted, not re-derived. Companion canon: memory `ks-twelve-attempt-history-and-the-channel-law`,
`dev_notes/SQUARE-CONTROL-PRIMITIVE-PLAN-2026-08-12.md`, `dev_notes/SESSION-HANDOFF-2026-08-12.md`.

---

## 1. Chronological table of every distinct KS attempt

Venues: STS/WAC = deterministic benches; "games" = self-play or vs-SF game runs (the only arbiter);
regret = footprint D7 move-regret on `game_regret_set.csv` (15k) + `_v2` (11,940 disjoint).

| # | date | attempt | mechanism | result | verdict | source |
|---|---|---|---|---|---|---|
| 1 | 06-25/26 | **First attack-units KS** (`king_safety_score`: units → precomputed table, safe-checks, shield, storm) | new term ADDED beside `latent_threat` | **STS −94** (double-count); replacement-mode −26 | FAIL (additive #1) | memory `king-safety-design` |
| 2 | 06-26 | **Complement architecture** (keep latent_threat, add ONLY safe-checks) | additive, narrow slice | STS +13 → **games −0.4 ± 28.4 / 793g = dead flat** | FAIL in games (additive #2) | memory `king-safety-design` |
| 3 | 06-28 | **div6 tune** (`KS_DIVISOR=6` on a 4-theme king subset) | subset-tuned magnitude | **+73 on the subset / −198 on the full suite** (overfit); tournament pooled flat | FAIL (additive #3) | memory `king-safety-design`, OPTIMIZATION_LOG (2026-06-28) |
| 4 | 06-30 | **Detection rebuild** (`KS_ZONE2`/`KS_WEAK`/`KS_STORM` wired, per-king `KS_DYN`) | wider zone + per-king dynamic magnitude, REPLACE latent_threat | ksattack bench +70, but lightning: REPLACE-alone **−48**, bundle **−46.5 ± 22 / 1323g** — the largest KS game loss on record | FAIL (additive #4) — "park KS-as-eval" | memory `ks-detection-rebuild` |
| 5 | 07-04/05 | **Outcome-compass re-adjudication** (consolidation, MAG fits, RFP coupling sweep) | KS as isolated addition, magnitude fit on outcomes | capture ceiling 23→30%; MAG2500 lightning **−50**, MAG1000 **~−95**; deep node_ab all 5 configs within noise of 0 | FAIL — verdict "KS didn't fail from badness, it failed from **ISOLATION**" | `ks-build-plan-2026-07-04.md` |
| 6 | 07-16/17 | **KS v1 activation** (`ENABLE_KS_REPLACE_LT=1 KING_SAFETY_MAG=3000` + SF weak/safe-check defs + `KS_FLOOR=13` deadzone) | replace latent_threat; SF-defs + floor as partners | score FLAT (42.2→41.5%), total collapses flat; **KS-caused collapses 15→7 (−53%)** categorical | SHIPPED `73c7cad` — categorical, **never Elo-confirmed** | `king-safety-activation-2026-07-16.md`, `collapse-reduction-ledger.md` |
| 7 | 07-18 | **`KS_SAFE_CHECK_DEF=5`** (defensive-asymmetric safe-check) | additive defensive weight | 600g/3 seeds: total flat, **ks_attack class 70→54 (−23%)**, score −0.7% | SHIPPED `41c4123` — categorical only; first verdict was literally "DO NOT SHIP on this evidence", shipped after swapping the classifier | ledger; memory `ks-twelve-attempt…` |
| 8 | 07-20 | **Re-weight tuner** (safe-check 20, floor 10) | crank discriminating weights | FIRENEW 0.24→0.82 but SUPPRESS 2×, **WAC −13** | FAIL — "lowering the floor wakes calm noise" | `king-safety-redesign-investigation-2026-07-20.md` |
| 9 | 07-20 | **`KS_AIM`** (latent slider-through-one-blocker detector) | additive detection | initially bench-clean; final: **−103 STS and net-zero on its own class (13 fixed / 13 created)** | FAIL (additive #5) | same doc; memory `ks-twelve-attempt…` |
| 10 | 07-20 | **`KS_INTERACT` coffin** (multiplicative undefended×open×attackers) | second non-linearity on top of the square | **STS −103/−201** — double amplification SF never has | FAIL (additive #6), structurally wrong | same doc |
| 11 | 07-21 | **Floor recalibration** (`KS_FLOOR 13→6`, `KS_SAFE_CHECK=8`, `KS_ATTACK_COUNT=2`) | catch the 6–13-unit under-read band | recovers ~31% of the under-read but **STS 1555→1467 (−88)**; dissection: of 150 firing positions, **52 are SF11-static QUIET (net −50 = the whole regression)** — "SF sees no danger; we invent it" | FAIL — the 6–13 band is intrinsically ambiguous (our units AUC 0.81 ≪ 1.0) | `ks-sts-overfire-dissection-2026-07-21.md` |
| 12 | 07-21/22 | **Count-gate** (`KS_FLOOR=6 … KS_MIN_ATTACKERS=2`) | structural coordination gate silences what the low floor wakes | STS 1588 (**+33**) → **games −2.2% BOTH seeds, positional collapses +20/+8**; decoupled GATE_F13 no better | FAIL (additive #7) — "STS +33 did NOT transfer" | `ks-architecture-sf-ethereal-vs-ours-2026-07-21.md`, `collapse-reduction-ledger.md` |
| 13 | 07-23 | **`MOD_KS_BACKING`/`MOD_KS_REALIZ` as phantom-attack gate** (material-backing realizability) | damp danger by attacker's material backing | FEN-3 single-position win, but at scale **every config RAISED phantom MSE and broke real-attack guards** — phantom rows aren't under-backed | FAIL as a discriminator (the knob later wins as a blunt whole-budget damp, #16) | `king-safety-subsystem-map-2026-07-23.md` |
| 14 | 07-23 | **Triage + per-king asymmetry finding** | diagnosis, not a lever | **17/23 ks_attack collapses = our OWN attack over-read** (SF18 ≈0..+2.5 vs our deep +50..+80); FEN-1: we under-charge our OWN airy king (units 4 vs 33) — SF charges BOTH kings and nets | Direction: **strengthening KS pushes the WRONG way** | same doc; ledger |
| 15 | 07-24 | **`ENABLE_KS_CHECK_V2`** (SF15 per-type saturated safe-check table) | richer safe-check magnitude | gentle-A +2.6% score 3 seeds (general play, NOT a collapse fix); hot fit **WAC −5 / STS −42** (KS-only corpus overfit); diverse whole-system fit **rejected ALL aggressive levers**; after de-king shipped: CHECK_V2 alone = **−14 WAC / −166 STS** (was bench-neutral +2.6% before) | FAIL (additive #8) + **channel-law proof** | memory `ks-check-v2-session-2026-07-24`, `ks-twelve-attempt…` |
| 16 | 07-24 | **DE-KING** (`KS_ZONE_ATTACK_PCT=50` — halve the triple-counted attackingLayer king boost) | **SUBTRACTIVE** | **+7.4% score ALL 3 seeds (≈+50 Elo), positional collapses −22%, STS +92**; peak at 50, not 0 | ✅ WIN — the biggest KS-adjacent eval win | memory `deking-categorical-win-2026-07-24` |
| 17 | 07-25→08-01 | **`MOD_KS_REALIZ=128`** (whole-budget realizability damp, in bundle with material fix + passer V3) | **SUBTRACTIVE/conditioning** | **+36.7 Elo / 503g, SPRT H1 accepted**; same knob was **−196 STS before the material fix, +88 after** (279-pt swing); 44 sacrificial-attack regressions unresolved, NOT separable via `KS_REALIZ_FLOOR` | ✅ WIN + channel-law proof #2 | memory `ksr-bundle-game-validated-plus37`, OPTIMIZATION_LOG |
| 18 | 08-07 | **63-knob corpus joint descent** (KS knobs included) | fit magnitudes to SF18-search corpus | corpus val **−41 (best ever)** → **games −85.6 ± 74.3 Elo / 116g, H0 accepted** | FAIL — corpus fit is ANTI-correlated with Elo | memory `corpus-fit-is-anti-correlated-with-elo` |
| 19 | 08-09/10 | **Bidirectional-error measurement** | diagnosis | KS mean\|gap\| vs SF11 = **0.995 (13 over / 21 under, signed mean +0.036)** — over- and under-reads cancel | NO SCALE WORKS — every global magnitude move fails in both directions | MEMORY.md `every-eval-term-error-is-bidirectional`; `KS-MORNING-PLAN-2026-08-10.md` |
| 20 | 08-11 | **Detector decomposition + giants comparison** | diagnosis | over-read = Channel-1 **PROXIMITY** (0.897 pawns on false alarms; `KS_ATT_*` 0.440 + attack-count 0.265 + weak 0.235); **`KS_SAFE_CHECK` = 0.010** — the only discriminating sub-detector, structurally tiny; giants' safe-check:attacker ratio **15–25× (SF) / 2–4× (Ethereal) vs ours ~1× = calibrated BACKWARDS** | The fix is DISCRIMINATION, solved as ONE unit | `KS-DETECTOR-REBALANCE-PLAN-2026-08-11.md` |
| 21 | 08-12 | **`defaware1`** (`KS_DEFAWARE_MODE=1` — defender-aware contested-fraction attacker weighting) | **redistributive** (removes credit from defended attackers) | first KS lever with a **sign-consistent move signal on BOTH cross-sets** (−0.014 / −0.056), STS +112, symmetry-clean | ✅ kept — rides in the +15 bundle (UNCONFIRMED, CI crosses 0) | SESSION-HANDOFF-2026-08-12; memory `ks-twelve-attempt…` |
| 22 | 08-12 | **Additive safe-check re-refuted** (`KS_SAFE_CHECK=8` flat; `CHECK_V2` per-type) | additive magnitude | SC=8: −0.176 on 15k → **+0.078 on v2 (inverted)**; CHECK_V2: −0.14 → **+0.115 (inverted)** — fits either set, flips on the other | FAIL (additive #9) — the safe-check **MAGNITUDE route is dead**; the win is detector INFO | memory `ks-twelve-attempt…` |
| 23 | 08-12 | **Coupled curve redesign** (SQC honest inputs + weak/safe-check UP + proximity DOWN + KNEE→40 so the quadratic runs) | redistribute THEN compound | configs **A +0.038/+0.115, B −0.018/+0.208 — both WORSE, v2 badly worse** (largest degradations measured). Drift analysis: inflation is UNIFORM (worse +3.3 vs better +3.1); worst drift = **breaks already-correct moves** (base reg 0.0 → cand 30–70); pulls search toward attack-chasing; leaks into 7-piece endgames | FAIL — **"you cannot compound units that don't discriminate"** | SQUARE-CONTROL-PRIMITIVE-PLAN-2026-08-12 §VERDICT |
| 24 | 08-12 | **Four detectors** (`KS_SQC_MODE` value-aware contest, `KS_PIN_MODE`, `KS_WEAK_VAL_MODE`, `KS_FLANK_MODE`) | discrimination-validated detector upgrades | unit **AUC 0.748 → 0.810** (past SF's 0.80), detection gap 80%→8% — **but ON they make general move-quality WORSE** (footprint +0.14..+0.54); phase-split: **opening +0.37 / midgame +0.27 / endgame −0.32 (KS HELPS in endgames)** | FAIL on deployment — the **WHEN-to-fire reframe**: a "how much" win unusable until the "when" system exists | SESSION-HANDOFF-2026-08-12; SQC plan |
| 25 | 08-13 | **Stage-1 coordination gate alone** (count×weight product) | isolated "when" lever | **opening null, midgame −0.135 at div=4** — necessary but insufficient in isolation | FAIL solo — motivates the minimum-coherent-unit accumulator | KS-SIGNED-ACCUMULATOR-DESIGN-2026-08-13 |

**Score: additive/isolated KS levers 0-for-9 in games. The only Elo-confirmed wins are subtractive
(de-king +7.4%/−22% collapses; `MOD_KS_REALIZ` +36.7 in bundle) and one redistributive lean (`defaware1`,
inside the unconfirmed +15 bundle).** The two additive-direction ships (KS v1, DEF=5) rest on categorical
class counts at flat score, never Elo (memory `ks-twelve-attempt…`).

---

## 2. The channel law, and why additive KS is 0-for-9 — the mechanism

**Statement (memory `ks-twelve-attempt-history-and-the-channel-law`):** *no KS lever has an intrinsic
value; its SIGN is a function of which other king-credit channels are live.* Proven three times:

1. beside-latent −72 STS vs replace −26 (the same term, different host) — memory `ks-detection-rebuild`.
2. `MOD_KS_REALIZ=128`: **−196 STS without the material-count fix, +88 with it** — a 279-point swing from
   a change in a *different* subsystem (OPTIMIZATION_LOG 2026-07-31; memory `ksr-bundle…`).
3. `ENABLE_KS_CHECK_V2`: bench-neutral (−2/−2) and +2.6% score on 3 seeds **before** de-king; **−14 WAC /
   −166 STS after** de-king. Nothing about the knob changed — its value was consumed when de-king removed
   the double-count it was implicitly compensating for (memory `ks-twelve-attempt…`, ↩️ correction).

**Why the law holds — the channel structure** (`eval-architecture-degeneracy-map.md` §A2/§B; memory
`ks-twelve-attempt…` §FOUR CHANNELS): king-zone pressure reaches `total` through FOUR live channels off
~two signals — (1) unit-KS (`KING_SAFETY_MAG=3000`, the only phase-tapered path), (3) the attackingLayer
king-directed boost into per-piece `total` (live at 50% post-de-king), (4) the SAME accumulator re-spent
via O/D → imbalance ×3 (unconditioned, untapered), (5) flat mg-shelter 185/75 + `baseIncrement`
(duplicating `KS_SHIELD`/`KS_OPEN_FILE` inside unit-KS). Channels (3) and (4) are literally COLLINEAR
(same cells). So:

- **Any additive KS lever is a 5th credit for facts already paid ~4×.** A swept knob measures the channel
  structure, not the knob (KS-MORNING-PLAN-2026-08-10).
- **The units are ~85% proximity** (live trace: 100% of dangerous kings at units 13–51, median 18;
  weak=2/safe-check=3 ≈ 10% of the sum — SQC plan §CAPSTONE), and king-attack proximity is counted **~4×
  at linear order** (direct, OvD, king-slice, unit-KS + shelter dup). Adding or squaring therefore
  amplifies QUADRUPLE-COUNTED PROXIMITY NOISE, not signal — the mechanical reason the dead quadratic
  (`KS_FLOOR=13 > KS_KNEE=12` ⇒ the squared branch never executes) has been **PROTECTING** us, and why
  "just fix FLOOR≥KNEE" would make us worse (SQC plan).
- **The KS term error is BIDIRECTIONAL** (13 over / 21 under, signed mean +0.036, mean|gap| 0.995): the
  errors flip sign per position, so every GLOBAL magnitude move — up or down — hurts somewhere it was
  right (MEMORY.md `every-eval-term-error-is-bidirectional`).
- **The deployment direction is against strengthening:** 17/23 ks_attack collapses are our engine
  over-reading its OWN attack (`ks_collapse_triage.py`), so more danger-seeing feeds the exact failure
  mode (king-safety-subsystem-map-2026-07-23; ledger).
- The consequence chain: additive lever → inflates an already over-counted, proximity-dominated,
  bidirectional signal → over-fires on quiet/opening positions (the −88 STS floor dissection: 52/150
  firing positions SF-quiet) and pulls the SEARCH toward attack-chasing (the drift analysis) → games lose
  even when a curated bench or corpus improved. Only subtracting/redistributing credit — removing a
  double-count or moving weight without raising total firing — has ever survived games.

---

## 3. Distilled durable lessons, each tied to the failure that produced it

1. **"Compounding non-discriminating units amplifies noise"** ← the coupled-curve failure (#23): configs
   A/B inflated WORSE and BETTER positions equally (+3.3/+3.1) and broke already-correct moves (reg 0.0 →
   30–70). Prerequisite for ANY squaring: prove genuine-danger kings land at clearly higher units than
   proximity-only kings on the cross-set FIRST (SQC plan).
2. **"Corpus-fit is anti-correlated with Elo"** ← the −85.6 Elo result (#18) after the largest corpus
   improvement ever (−41 val); mechanism: scalar-distance objectives reward SHRINK, Elo comes from move
   ORDERING, and the two are decoupled (52% of large errors are move-neutral). Never derive an
   accumulator magnitude from corpus fit; the design doc's "derived on our scale, never corpus-fit" rule
   exists because of this (memory `corpus-fit…`, `eval-degeneracy-and-the-dedup-then-retune-program`).
3. **"Detectors validated on discrimination but harmful on deployment"** ← the over-fire (#24): AUC
   0.748→0.810 on the KS-specific attack-vs-quiet corpus, yet footprint +0.14..+0.54 on the mixed
   move-regret sets. Domain-specific discrimination ≠ deployment validation; do NOT defer the mixed,
   move-regret, PHASE-STRATIFIED test on a theoretical shortcut (memory `ask-when-and-how-much…`).
4. **"Ask WHEN, not only HOW MUCH"** ← months of magnitude-only work (#1–#23) while both live failure
   modes (opening over-fire, midgame attack-chasing) were *when* failures; the giants fuse when+how-much
   into ONE signed object, invisible when you study primitive/curve/channels separately
   (SESSION-HANDOFF-2026-08-12; memory `ask-when-and-how-much…`).
5. **"An isolated positive lever with no counterbalance regresses"** ← KS-in-isolation 07-05 ("failed
   from ISOLATION, not badness"), the count-gate games loss (#12), and Stage-1 solo (#25, midgame
   −0.135). A positive term alone is not testable; test the whole netted loop
   (KS-SIGNED-ACCUMULATOR-DESIGN §principle).
6. **"A fixed magnitude floor cannot resolve an overlapping band"** ← the floor dilemma (#11): attacks
   (~16u) and quiet (~8u) overlap (AUC 0.81), so floor 13 kills low-end attacks and floor 6 wakes quiet
   (−88 STS → −2.2% games). SF/Ethereal have NO magnitude floor — they net large context suppressors and
   threshold the NET (ks-architecture-sf-ethereal-vs-ours-2026-07-21).
7. **"The safe-check MAGNITUDE route is dead; the win is detector INFO"** ← SC=8 and CHECK_V2 both
   INVERTING between disjoint cross-sets (#22), versus `defaware1` (better info, redistributive)
   generalizing on both (#21). Upgrade what a detector KNOWS, not how loud it shouts.
8. **"No KS lever has an intrinsic value (channel law)"** ← the three proofs in §2. Corollary: validate
   any accumulator work with the other king-credit channels FROZEN, and expect every constant to need
   re-derivation if a channel changes (memory `ks-twelve-attempt…`).
9. **"Bench/categorical wins do not transfer"** ← STS +33 → games −2.2% (#12); ksattack +70 → −46.5 Elo
   (#4); +13 STS → −0.4 Elo (#2); the two categorical ships never Elo-confirmed (#6, #7). Games decide,
   above the ~20–40 Elo floor; |balanced STS| < ~150 is unresolvable (MEMORY.md).
10. **"Never tune on a narrow subset"** ← div6 (+73 subset / −198 full, #3) and the hot CHECK_V2 fit
    (KS-only corpus: WAC −5 / STS −42, #15). Tune/validate on broad mixed sets with guards.
11. **"Never conclude a cause from an eyeballed worst-N"** ← "endgames are the over-read" from a drift
    worst-20, refuted by the phase-split aggregate (endgame KS HELPS, −0.32) — the same instrument that
    localized the real over-fire to the OPENING (SESSION-HANDOFF-2026-08-12 §lessons).
12. **"Material backing does not discriminate phantom from real attacks"** ← the MOD_KS_REALIZ corpus
    verdict (#13): the gate never fires on phantom rows and damps real guards where it does; it later won
    only as a blunt whole-budget damp, carrying 44 unresolved sacrificial-attack regressions (#17).
13. **"Suppressors must be LARGE and in the same units"** ← our `KS_NO_QUEEN=6` (single-digit) vs SF −873
    / Ethereal −237: both giants make a queenless "attack" net near-silent; ours cannot cancel proximity
    (king-safety-subsystem-map §SF11+Ethereal models; ks-architecture doc).
14. **"Copying the giants' CONSTANTS loses; port FORMS, refit on our scale"** ← the Ethereal-raw-weighted
    proxy scored AUC 0.71 < our own units 0.81 (ks-architecture doc §validation); and the KS_INTERACT
    coffin showed inventing a *worse* structure is not "uniquely ours" either (#10).

---

## 4. Component-by-component: what each accumulator element inherits from a past failure

| accumulator component | the past failure it answers | what the failure obliges the design to do differently |
|---|---|---|
| **No-queen suppressor** (`KS_NQ_SUP` ≈ −30..−46 on our 0–80 scale) | `KS_NO_QUEEN=6` was trivial vs SF −873/Ethereal −237 — "the single biggest miscalibration" (subsystem-map 07-23); queenless false-attacks survive every gate today. Also: the 07-21 gate study found no-queen "barely helps" on a mostly-queened corpus | Size it as a FRACTION of max net (SF: 873/1500 ≈ 58%), derived from the unit-trace on real lost-king positions, **jointly** with the demoted-proximity positive side (magnitudes are coupled — a −35 only means something relative to the positive scale). Validate on the ~20% queenless slice specifically, and on the whacky no-castling set where shelter archetypes vanish |
| **Threshold replaces the fixed `KS_FLOOR`** | The floor dilemma (#11/#12): floor 13 = deadzone under-read; floor 6 = −88 STS/−2.2% games; ANY fixed cutoff over the overlapping 6–13 band trades attack-recall against quiet-precision | The effective floor must be PER-POSITION (net = positives − suppressors, then threshold) so a sheltered queenless king self-nets below the bar while a real attack clears it. ⚠️ Owner insight 08-13: **thresholding proximity-dominated units just thresholds PROXIMITY** — so the positive-side rebalance (demote proximity, elevate weak/safe-check) is part of step 1, NOT a later step (KS-SIGNED-ACCUMULATOR-DESIGN §0) |
| **Square AFTER the gate** | The dead quadratic (`FLOOR=13>KNEE=12`, never executes) has been PROTECTING us; the coupled-curve attempt to make it live (#23) was the worst cross-set degradation measured — squaring raw units amplifies 4×-counted proximity noise | Square only the NETTED, THRESHOLDED signal (`(net−THRESH)²/DIV`), and only CONDITIONAL on proven discrimination: the go/no-go is the units-distribution check (genuine-danger kings must sit clearly above proximity-only kings on the cross-set) BEFORE the square is trusted. Derive DIV for continuity with today's danger range; a magnitude move of this size is games-gated |
| **Proximity demotion** (`attack_count` kept but demoted) | Proximity is ~85% of units and the measured over-read source (0.897 pawns on false alarms, #20); but uniform REMOVAL is the de-dup-by-removal trap — de-king peaked at 50% not 0, and central-term removal was load-bearing (memory `eval-degeneracy…`) | DEMOTE, don't delete (flagged ⚑ in the design): proximity should qualify squares/detect that an attack exists, not carry magnitude. Sweep the demotion with a near-inert arm; expect an INTERIOR optimum like de-king |
| **Safe-check feeder upgrade** (`check_safe` re-routed through `ks_sqc_breaks`) | Both MAGNITUDE routes are refuted-and-inverted on cross-sets (SC=8, CHECK_V2, #22); the feeder itself still uses the crude `bm & own` presence test — a check square "defended" only by a pawn counts as safe | This is the un-tried detector-INFO route, explicitly distinct from the dead magnitude route (SQC plan §safe-check feeder). Ship it as better INPUT at held weight; if it only works with a weight crank, it is the old failure again |
| **The four detectors** (SQC / pins / weak-val / flank) feeding the positive side | Discrimination-validated (AUC 0.810) but deployment-harmful when turned on flat (#24) — over-firing the OPENING (+0.37) | They enter ONLY as inputs to the netted object (suppressors + threshold downstream of them), never as standalone adders; validation is phase-stratified move-regret on BOTH cross-sets + the whacky set, pass = opening improves / midgame neutral-or-better / endgame preserved (design §validation) |
| **Coordination count×weight product** | `KS_ATT_PRODUCT`/`KS_OVERLOAD` over-fired historically *because they ignored defenders*; `KS_MIN_ATTACKERS` is inert (queen exception drops the bar to 1); Stage-1 solo was midgame-negative (#25) | The product is a coordination-aware INPUT inside the accumulator, not a standalone rule; it composes with defender-aware weighting (defaware) so a defended "coordinated group" still nets low |
| **Two-function mg/eg phase blended by non-pawn material** | `ks_phase_taper` single linear scale + `KS_PHASE_ZERO=104` deep-endgame CLIFF under-gates R/Q endgames; and the phase-split proved **endgame KS HELPS (−0.32)** — the old "midgame-only, fade to 0" mandate (ks-build-plan 07-04) is refuted at the endgame end | Keep the eg branch LIVE and gentle (small linear fn of net), blend continuously by non-pawn material, no cliff. The curve-redesign's endgame leakage (units 26–33 in 7-piece positions where king ACTIVITY ≠ safety) is what the separate eg function must prevent |
| **Suppressors: shelter + already-winning** | Shelter is triple-counted today (flat 185/75 + attackingLayer multipliers + `KS_SHIELD` — degeneracy map §B); the missing own-initiative offset (SF −6·score/8) was identified 07-23 as why "we're up material but panic about our king" | Re-home `KS_SHIELD` as a suppressor INTO net (one owner), don't add a fourth shelter copy; the already-winning discount answers the per-king-asymmetry finding (SF charges BOTH kings and nets — subsystem-map §BREAKTHROUGH) and the 07-23 sign-flip cases |
| **All-gated, `KS_ACCUM_MODE=0` byte-id; channels frozen** | Channel law (§2): every constant's sign depends on the live channel structure; two past colour bugs rode in with game-validated ships | MODE=0 must reproduce `250 / 35,426,396 / EBF 3.800` exactly; validate with the four king-credit channels untouched; colour/file symmetry is a ship gate; compare vs the defaware bundle (deployment-relevant baseline), not vs flat |

### Cross-cutting validation obligations (from the instrument failures)
- **Games decide; deterministic instruments screen.** STS/categorical/AUC/corpus have each individually
  endorsed a games-loser (#2, #4, #6/#7, #12, #18, #24). Order: byte-id → phase-stratified footprint
  regret on BOTH cross-sets + whacky set → discrimination sanity (AUC must not collapse) + a near-inert
  arm → bundle → ONE SPRT (design §validation).
- **Do not fit the accumulator's constants on any static corpus** (#18); derive from the unit-trace
  (`KS_DEBUG_DUMP`) on real lost-king positions, joint derivation + attributed incremental confirmation
  (design §0), and remember one process per knob setting (knobs latch at init).
- **The whacky/variant no-castling set is the structure-independence check** — central exposed kings are
  the KS regime every shelter-calibrated detector never trained on; use as generalization/diagnostic at
  large magnitude, never as the ship gate (SESSION-HANDOFF-2026-08-12 §whacky).
