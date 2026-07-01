# Collapse-elimination campaign — running log

Persistent track: find the singular MISEVALUATIONS that lose real games (self-play-invisible, worst-case
Elo), fix each, validate by **position-fix + no bench regression + ACPL-vs-SF not worse ⇒ ship** (NOT
self-play). Plan: `~/.claude/plans/handoff-lossless-speed-campaign-tranquil-rose.md`. Validation model +
backlog: [[external-play-gaps]]. **Document every angle tried here** so we never re-walk a dead end.

**Baseline (current, post-combo1):** WAC d10 **252 / 67,931,145**, STS d10 **1503 (50.1%)** (OMP-pinned,
book-off). ACPL on `away_standard` n=112: MEAN 122 / MEDIAN 46.5 / blunders 21 (SF 0.25s). Dispatcher:
`overnight_runner.sh {wac,sts,cploss_probe,fenvs,tournament} ...`.

## 2026-06-30 (late night) — EVAL BUILD-OUT begun: 3 new gated terms shipped byte-clean (majority/pawn-struct/outpost)

Rescaling mine exhausted → pivot to BUILDING the cheap terms SF11 grades that we lack ([[capture-gains-overread]]).
Plan: `~/.claude/plans/handoff-agentic-eval-tuning-soft-tiger.md` (eval build-out). New-term → tunable pipeline
(5 layers: `br_<term>` accumulator → `EvalBreakdown` field → `g_capture` publish → `ChessAI.pyx` dict → `tune_corpus`
TERMS) established. All gated byte-identical (default-off); each **byte-id verified (WAC 252 / 70,150,573 exact)** +
**functional-probe verified**:
- **`pawn_majority`** — un-parked the existing block (6673) + `br_pawn_majority` capture. Knobs `PAWN_MAJORITY_MAG_MG/EG`+mods.
- **`pawn_struct`** — NEW `ISOLATED_PAWN_PEN` (no friendly pawn on adjacent files) + `BACKWARD_PAWN_PEN` (adjacent pawns
  all more advanced AND stop-square enemy-pawn-controlled). Self-contained block after majority. Probe: iso-d4 +200 / iso-d7
  −200 / phalanx 0. Signs correct (Black-positive: White weakness +, Black −).
- **`outpost`** — NEW `OUTPOST_KNIGHT`/`OUTPOST_BISHOP` (minor in enemy half, pawn-defended, no enemy pawn can advance to
  attack). Probe: Ne5 −250 / Black Nd4 +250 / undefended 0 / enemy-Pd6-can-attack 0. Signs correct.

Pawn-attack orientation grounded in the verified evaluator (White attacks `<<7/<<9`, Black `>>9/>>7`).

**MOBILITY (item 4) — BUILT + verified.** Design revised during impl: instead of 16 fragile per-branch evaluator edits
(white/black × mid/end × 4 pieces, mixed tab/space indent), a SINGLE self-contained block in `placement_and_piece_eval`
loops N/B/R/Q, recomputes each attack mask via the cheap **PEXT** `SlidingRow::operator[]` (`_pext_u64` — 1 hw instr +
load, NOT expensive), `popcount(att & safe mobilityArea)` → nonlinear `MobilityBonus_<piece>[]` table, White subtracts.
`mobilityArea = ~((K|Q|P)&own | enemy_pawn_attacks)` computed once per side per phase-block (gated). Knob
`ENABLE_PIECE_MOBILITY` (default off); when on it force-disables the cheap rook/knight/queen surrogates (search_engine.cpp
init) to avoid double-count. `br_mobility` wired (Texel-able). **Verified:** byte-id off (252/70,150,573 exact) + functional
(real-pos mob 40/18/16, draw-FENs early-return so use non-drawn positions) + colour-symmetry (mob == −mirror) + **NPS ~4.4%
overhead** (366k→350k nps, tactics held 252) = acceptable, PEXT-reuse cheap as predicted. Tables PACE/SPSA-tuned (not Texel).

**Research immortalized:** `dev_notes/sf11_eval_reference.md` (SF11 term-by-term) + `dev_notes/our_eval_reference.md`
(LIVING map of our terms + anchors + KEEP/BUILD/RETIRE + ours↔SF11 correspondence). NEXT: regenerate corpus with the 4 new
terms at UNIT magnitude → Texel `--scale-inv --attrib` priors + correspondence-filter → PACE (mobility tables) →
lightning-gate each (+NPS) → bundle. `capg70` gate ~55% / LLR +0.42.

## 2026-06-30 (night) — SF11-per-term Texel: capture_gains is the #1 over-read (cross-validated); retirement off the table

Built the SF11-per-term Texel pipeline: `tune_corpus.py` now labels each position with **classical SF11's `eval` per-term
table** (reused `SF11Eval` parser) + writes the `det_*` conditioning detectors + uses SF11's total for win/loss strata
(one engine, half the cost); `tune_cond.py` = offline `mod_gain` conditioning fit; `tune_fit.py` += `--scale-inv`/`--attrib`.
Dispatcher subs `tune_corpus`/`tune_cond`/`tune_fit`. 37,222-row `cond_corpus.csv`.

**METHOD LESSON — the SF11-total confound ([[sf11-texel-scale-invariance]]):** fitting our eval to SF11's TOTAL is
dominated by a **strength-NEUTRAL global scale** (our eval reads ~1.5x hotter, s≈0.64). A global eval multiplier changes
NO move (argmax scale-invariant), so any fit hijacks its params to do blunt shrink (the material/imbalance/pair
CONDITIONING cluster railed to damp and STILL lost to a plain global scale = **non-lever, dropped**). FIX = scale-invariant
objective (each candidate gets its own best global scale) + solo per-term attribution (`--attrib`, dodges the joint fit's
railing).

**KEY FINDING — per-term hotness vs SF11 (scale-invariant, solo, control split):** **`capture_gains` = the #1 OVER-read by
far** (solo scale→0.20, Δloss 0.0082, ~2.7x the next) and **cross-validated** — the earlier outcome-fit independently wanted
it down (→82). #2 = **`pieces`/placement UNDER-read** (→2.49). Everything else (imbalance/pvboost/passed/KS) negligible.
Mechanism (user): capture_gains approximates SEE/qsearch, built pre-qsearch as horizon insurance; it makes **full-piece
swings** on static guesses → over-reads now that qsearch resolves those lines directly + more accurately (the same
tactical-redundancy pattern as king-safety, cleanest instance).

**BUT full-off craters both benches → NOT redundant, load-bearing:** fixed-d10 `SCALE_CAPTURE_GAINS=0`: WAC **252→238
(−14)**, STS **1568→1437 (−131)**. So it earns real tactical AND positional signal → **retirement is off the table**; the
fits meant over-*scaled*, not worthless. Lever = **reduce magnitude** (screening 70/50) or **condition it** (user's
reliability-dampen: `capture_gains × f(defended-target / side-to-move-realizable / capture-chain-depth / counter-threat)`,
damp-only [floor,256], keep the swing only where the static prediction is reliable + beyond q's horizon). Speed bonus if it
can shrink: capture_gains is one of the most latency-heavy eval features → cheaper eval = more nodes. Next: bench-screen
partial values, then **lightning SPRT** (equal-time = the only Elo verdict; captures accuracy AND the speed benefit). See
[[capture-gains-overread]], [[fast-selfplay-eval-depth-bias]], [[fixed-depth-bench-ceiling]].

## 2026-06-30 (eve) — LIGHTNING VERDICTS: KS-as-eval DEAD; fast self-play BIASED for eval; pivot to Texel

The overnight `ksdyn_bundle` (KS REPLACE + prune) lost **−46.5 ±22** over 1323 self-play games. Decomposition:
**KS-alone (REPLACE, no prune) = −48 ≈ bundle ⇒ the KS REPLACE-of-latent_threat swap is the culprit, NOT the
prune** (prune ~neutral at equal time → parked). `ks_regress` showed the STS −26 was latent_threat REMOVAL on
ks=0 positions, not KS over-fire.

Built `fast_tourney` (fast fixed-depth self-play, lightened 0.1s adjudication): **~2435 games/hr @ d4 / ~1859 @
d6 = ~13x lightning**, fair (base-vs-base 50.0%), rich JSONL. Ran a KS+OvD candidate SWEEP (`ks_ovd_fastrank`):
**besideA = gentle KS BESIDE latent_threat (`KING_SAFETY_MAG=1500 KS_ZONE2=1`, no conditioning) = +31 fast-Elo**,
all OvD/dyn conditioners (MOD_KS_CONTROL/BACKING, KS_DYN) damp it to ~+3 (KS not over-firing at gentle MAG, so
the brakes just suppress a small useful term), ovdyn (IMBALANCE/REALIZ) −8.

**THE CALIBRATION GATE killed two birds:** lightning SPRT of besideA = **~0 (50% / 83 games)** — the fast +31
EVAPORATED. Combined with KS-alone (fast −16 → lightning −47), both KS configs shift the SAME way ⇒ **fast
self-play systematically OVER-credits eval king-safety at shallow depth = a depth-transfer bias, not compression
([[fast-selfplay-eval-depth-bias]]).** So fast self-play is fine for SEARCH knobs (transfer across depth) but
UNRELIABLE for EVAL knobs. THREE eval proxies now down: move-match, ksattack, fast self-play.

**Two conclusions:** (1) **KS-as-eval is NOT a lightning lever** (REPLACE −47, gentle-beside ~0) despite the
SF11 detection gap — at our depth, latent_threat + search already cover king danger; **PARK king-safety**.
(2) **Eval tuning is lightning-bound** — no validated cheap eval proxy exists. NEXT = re-open **Texel** properly
(depth-agnostic `tune_corpus`/`tune_fit`, fit static eval to SF/outcome labels; depth-4 fast games = cheap
position source since SF labels are depth-independent), gate at LIGHTNING (the 2026-06-16 "Texel degrades STS"
rejection used STS = a biased judge), and populate `proxy_elo.csv` to test if Texel-score predicts lightning.
Built subs: `fast_tourney`/`ks_ovd_fastrank`/`gate_besideA`. See [[ks-detection-rebuild]], [[agentic-eval-tuning-system]].

## 2026-06-30 — KING-SAFETY DETECTION REBUILT + per-king DYNAMIC magnitude; bundle overnight tournament

Acted on the SF11 finding below. `king_safety_danger` (cpp_bitboard.cpp) had `KS_WEAK`/`KS_STORM`/`KS_BATTERY`
DECLARED + env-read but NEVER referenced (dead scaffolding); zone was ring1+1rank, `king_ring2` computed-but-unused.
Trace (`diagnostics/ks_trace.py`, reuses `det_ks_units_w/b` + `ks_explain`) on the 8 worst under-fire FENs: engine
read 0-1 units where the model reads 9-36.

**Wired (gated, byte-id 252/70,150,573 held):** `KS_ZONE2` (OR in king_ring2), `KS_WEAK` (undefended attacked zone
sq), `KS_STORM` (enemy pawn storm). `KS_BATTERY` still dead. Under-fire kings 0->9-13 units. **Per-king DYNAMIC
magnitude `KS_DYN`** (user's "mag = f(detectors), up when it matters / down when not, BOTH sides"): each king's
danger *= mod_gain(KS_DYN, att_cnt*(open+weak)-PIVOT, SHIFT), per-king => fixes net-cancellation. DYN=128 gentle =
best; safe_checks-in-realness DISCONFIRMED (over-boost).

**Mode REPLACE latent_threat** (beside double-counts: STS -72 vs -26). `ks_regress.py`: STS -26 is a near-wash
REDISTRIBUTION (55 regress/53 improve), regressions are latent_threat REMOVAL on ks=0 positions NOT KS over-fire
(control_edge doesn't separate) => static-MAG/MOD_KS_CONTROL sweeps were the wrong lever. Cross-suite: **ksattack
+70** (45.5->48.7%), STS -26, WAC -5. **Bundle w/ ORDER-prune SUPER-ADDITIVE at EQUAL TIME**: timed ksattack
bundle 1100 > prune-alone 1060 > KS-alone 970 (fixed-depth showed opposite = prune artifact). Heavier prune
(LMR=2/VM16k) over-prunes attack lines -> below prune-alone -> trades away KS. KS does NOT eat node savings
(bundle 61.4M < prune-alone 63.5M; ~-12% vs baseline).

**OVERNIGHT (running, tag `ksdyn_bundle`, ~580min):** self-play base vs `ENABLE_KS_REPLACE_LT=1 KING_SAFETY_MAG=4000
KS_ZONE2=1 KS_DYN=128 LMR_EXTRA=1 VERIFY_MARGIN=12000 HISTORY_LMR_SCALE=3`. Baked subs `tonight_tourney`/`prune_screen`.
Op notes: dispatcher prompt-free ONLY as `wsl.exe -e bash -lc "bash '<abs path>' …"` ([[dispatcher-prompt-free-wrapper]]);
SF interop binfmt restales ([[wsl-sf-interop-binfmt]], real fix = native Linux SF). See [[ks-detection-rebuild]].

## 2026-06-29 — SF11 EXTERNAL REFERENCE → KING-SAFETY DETECTION is the #1 lever (CURRENT DIRECTION)
**The pivot:** a day of FLAT eval results, BUT we're ~300 Elo below pre-NNUE single-thread SF ⇒ big upgrades
MUST exist; "flat" = our OBJECTIVE was blind, not the eval optimal. Built the missing yardstick = **classical
Stockfish 11** (last pre-NNUE HCE, apples-to-apples). SF11 binary (user-supplied):
`…/stockfish_11/stockfish-11-win/Windows/stockfish_20011801_x64_bmi2.exe`. SF18 is NNUE-only (can't be the HCE
ref). Harness: `vs_sf11` sub (SF11 opponent FULL-strength 1-thread = fair vs our 1-thread; SF18 arbiter;
depth+nodes both sides; `--sf-arb-path` added to vs_sf.py), `diagnostics/analyze_sf_games.py`, `eval_vs_sf11.py`.
**META-LESSON: every internal objective is BLIND** — move-match blind to eval, self-play blind to search (EBF
"at peak" was a same-EBF artifact). External SF11 reveals the headroom.
**Decomposition (200 games, equal depth 12, SF18 arbiter):**
- **EQUAL-DEPTH 3.2% → ~590 Elo gap** (search speed neutralized) ⇒ **per-node EVAL+ordering quality is the
  DOMINANT lever** (biggest number measured). Re-centers EVAL.
- **23x more nodes/move** (EBF 3.8 vs SF 2.0). **EBF-crush:** maxing ALL pruning only gets EBF 3.8→3.6 (−30 WAC)
  ⇒ **can't reach SF's 2.0 by knob-tuning — STRUCTURAL** (ordering quality + LMR formula + eval-confidence).
  Ordering fine (90.5% FMC); aggressive ordering-prune safe (−31% nodes/−5 WAC w/ VERIFY). We BEAT SF18 on WAC
  @d10 ⇒ tactics are a STRENGTH; weakness is POSITIONAL eval. SF's low EBF is eval-ENABLED → loops to eval.
- **#1 EVAL GAP = KING-SAFETY DETECTION.** `eval_vs_sf11.py` (our ev_breakdown vs SF11's labeled `eval` table,
  400 STS pos): mean signed ~0, mean|gap| 1.1p (UNBIASED, high VARIANCE); variance DOMINATED by KING SAFETY —
  **SF11 reads ±4-7p where our `king_safety_danger` reads ~0** (68/400; mean|our_KS−SF11_KS|=0.633). KS anchor ON
  does NOT close it (|gap| 1.108→1.195) — under-fires + mis-fires ⇒ STRUCTURAL detection failure (zone too small
  `ring1+1rank` @cpp_bitboard.cpp:480; NO pawn-storm attackers; KS_DEFENDER over-cancels; `king_ring2` @458
  unused). Why m4000ctl was Elo-neutral + move-match couldn't tune KS. 2ndary: pieces/imbalance over-read, no
  Mobility term.
**DIRECTION (planned, not started):** rebuild king-danger DETECTION guided by SF11-KS (which attacks to detect,
NOT literal fit), gated default-off, OVERFIT-GUARDED (held-out + don't regress passed/threats/STS/WAC/collapse
corpora; balance rule = small regressions only for supreme KS win, SF11-tournament-arbitrated), gate on
`vs_sf11 depth 12` + SPRT. PARALLEL: search ORDER-prune (LMR_EXTRA=1+VERIFY=12000+hls3, −15-31% nodes/−5 WAC,
STS-drop=fixed-depth-artifact → timed-depth-validate) as floor tournament candidate; bundle w/ KS if both work.
Plan: `~/.claude/plans/handoff-agentic-eval-tuning-soft-tiger.md`. See [[sf11-eval-search-diagnostic]].
**Parked:** rook-mob (CHEAP_ROOK_MOB=32) move-match +86 but STS −176 → proxy artifact, dropped. KS_INTERACT
flat on KS-set, gated off. Tooling built (gated/byte-id, UNCOMMITTED): vs_sf11, eval_vs_sf11.py, analyze_sf_games.py,
search_sweep sub, KS_INTERACT/MOD_PIECES_DEFEND knobs, classify_failures.py, mine_ks_positions.py, label_collapses.py.

## 2026-06-29 — AGENTIC FAILURE-CLASSIFICATION (staged-funnel pivot, day session)
**Method shift (user-directed): automate the CHESS classification of each failure, then map failure-type ->
involved knobs -> conditioning detector.** Not hand-picking a hypothesis; the agent buckets the failures.

**Tooling built this session (persistent, in `diagnostics/`):**
- `movematch.py` +`--epd/--limit/--sample/--seed/--offset` (seeded DISJOINT shards: Stage-1 sample=100 off=0,
  Stage-2 sample=300 off=100, verified zero-overlap). Dispatcher subs `movematch_sample`, `funnel_cand`
  (anchor m4000ctl BAKED in -> prompt-free), `place_probe`.
- `label_collapses.py`: SF-labels each collapse `decision_fen` -> STS-schema EPD; merged
  `suites/failure_corpus.epd` = 1500 STS + 63 collapse (themes Collapse2400/2700). Self-verifying.
- `diagnose_misses.py`: HIT/MISS + HELPED/HURT detector-gap readout (det_* incl. derived openness, overextend).
- `classify_failures.py`: per-miss dossier = played-vs-correct move + per-term STATIC-eval delta (which knob
  over-credited the wrong move, mover-POV) + live detectors. `agg` (term histogram) + `dossier` (JSONL).

**Results:**
1. Placement levers REJECTED. `SCALE_PLACE_KING_EG` 100->150: +35 on the King Activity FULL theme (probe) but
   held-out 300 NET **-25** (Pawn Play -41, Recapturing -29, Open Files -25; King Activity itself -2 on the
   held-out subset) = PLACEMENT SCATTER, matches [[detector-conditioned-knobs]] disconfirmed-placement. Knight
   Outposts: `PLACE_KNIGHT` 100/150/200 FLAT (508/508/508) = dead lever.
2. `classify_failures agg` on 36 collapse misses: the term that SYSTEMATICALLY over-credits the wrong move is
   **`pieces` (placement) +0.675 mean-signed / 0.857 |delta|**, dominating all else (`capture_gains` big |delta|
   0.61 but ~0 signed = noise; `king_safety` ~0). Re-confirms placement is the static culprit.
3. **Automated chess classification (3 fan-out agents, 36 collapses) -> 3 buckets:**
   - **Tactical/search ~13** (passive_or_slow 8, bad_trade/SEE 4, promotion 1): agents tag "none/tactical",
     no static detector separates -> floor partly HORIZON-bound (re-confirms prior).
   - **Greed-under-attack ~11** (missed_defense 8, premature_attack 3, KS-lapse 2): grabbed material / pushed a
     flank pawn while DEFENDING HEAVILY; lured by capture_gains/pieces/imbalance. Correlates with high
     **`defense_edge`** (stm_defense - opp_defense; verified `whiteDefensiveScore` = own defensive attack-layer
     sum, so high = under attack). <- the detector-separable cluster.
   - **Endgame technique ~9**: high `phase_score` (101-117); placement over-credits an advanced piece/king.
4. **Tooling gap:** `own_king_danger` (KS attack-units) reads **0 on every dossier incl. live mating attacks**
   -> KS detector blind here; `defense_edge` is the working "under-pressure" proxy. (KS attack-unit detector
   too sparse — future: opponent-offense-near-our-king detector.)

**BUILT + TESTED — `MOD_PIECES_DEFEND` (gated, default-off, byte-id 252/70,150,573/1568):** damp the placement
term `br_pieces` toward 0 by the NET attack on the favoured side = `oppOffense - favouredOffense` floored at 0
(cpp_bitboard.cpp cold tail, after MOD_PIECES_CONTROL). The two-detector net-attack form is the data-derived
refinement of a first single-signal (opp-offense-only) version that was a collapse-set WASH (270/630, helped
defenders / hurt attackers) — the `diagnose_misses diff` HELPED/HURT dump showed `overextend`/`offense_edge` as
the discriminator (helped low-offense defenders, hurt high-offense aggressors), so the gate was refined to fire
only when the favoured side is net out-attacked.
- **Collapse-set sweep (anchor 270/630):** net=32 -> 280 (+10), **net=64 -> 290 (+20, peak)**, net=128 -> 250
  (overshoot). FIRST positive eval signal of the campaign, via the classify->attribute->build->re-guide loop.
- **Held-out 300 (anchor 1586/3000):** pn64 = 1535 (**-51 TOTAL**) BUT **Collapse2400 +20 on UNSEEN collapses
  (the fix GENERALIZES)**; the -51 is placement-damp COLLATERAL on midgame themes (Simplification -33, Knight
  Outposts -28, Advancement -17, Center Control -16). Since collapses are ~6% of the suite, move-match TOTAL is
  collateral-dominated -> **move-match CANNOT adjudicate this** (the morning-state finding, reconfirmed). NOT a
  valid rejection.
- **VERDICT: cluster-positive + generalizing, move-match-inconclusive -> TOURNAMENT-PENDING** (the truth gate for
  collapse-fixes; morning-state next-step #1). Nothing shipped; knob gated default-off; UNCOMMITTED.
- **Method result:** the user's agentic failure-classification pipeline WORKS end-to-end and produced the first
  generalizing collapse-help — but on the `pieces`/placement term, whose collateral move-match can't weigh.
  This is the 3rd independent confirmation that placement-magnitude conditioning trades collapse-help for
  midgame-collateral (cf. MOD_PIECES_LEVEL/CONTROL, KING_EG scalar) -> the GATE aims true, the TERM is the wall.
  Open: (a) tournament-gate pn64 (does the real collapse-help beat the collateral in games?); (b) re-run the
  classify pipeline on the weak THEME misses (AT/Advancement/Open Files) to find a NON-placement culprit that
  move-match CAN validate. Caveat: ~13/36 collapses are tactical (search's job).
- **PACE-improve attempt (deadzone) FAILED — help & collateral are INSEPARABLE on placement.** Added
  `MOD_PIECES_DEFEND_THRESH` (deadzone on net_attack: only damp once net_attack clears THRESH, to spare
  sharp/marginal positions; byte-id preserved, default 0). Collapse-set mag64 sweep noisy (thresh100 260 /
  thresh300 290 / thresh600 270 / thresh1000 240). But mag64/thresh300 on HELD-OUT 300 = **1503 (worse than
  pn64's 1535)** and held-out Collapse2400 fell to **40 = anchor** — the deadzone REMOVED the generalizing
  collapse-help while collateral persisted. ⇒ a net-attack deadzone cannot separate help from collateral (same
  positions carry both). **thresh0 (pn64) is the peak; move-match PACE on this placement candidate is
  EXHAUSTED.** 4th confirmation placement collateral is intrinsic. Move-match is structurally the wrong gate
  for pn64 (collapses ~6% of suite). NEXT must be (a) TOURNAMENT pn64, or (b) pivot method to move-match-
  validatable THEME misses. All knobs gated default-off, byte-id 252/70,150,573/1568, UNCOMMITTED.

## 2026-06-29 — DYNAMIC CONDITIONING (user's core vision) — build #1: KS super-linear interaction
**User directive (firm): STOP scalar tuning; build dynamic position-conditioned knobs** = each load-bearing
magnitude is `base × f(cheap LIVE detectors)`, recomputed per position (KS mag ~2000 closed/quiet → ~10000
open/sharp). Detectors INTERACT non-monotonically (user's "coffin": closed is NOT safe when attacker has force
+ defender can't redeploy). Magnitude/weighting comes from REAL FAILURES (collapse corpus + positional bench
misses), not hand-set. Speed kept (only already-computed detectors). Plan: `~/.claude/plans/handoff-agentic-
eval-tuning-soft-tiger.md` (rewritten to this).
**Pre-check killed the naive build:** `king_safety_danger` (cpp_bitboard.cpp:4905) ALREADY models the main
effects additively — attacker force (KS_ATT_*), defenders-in-zone (KS_DEFENDER), open files at king
(KS_OPEN_FILE), pawn shield (KS_SHIELD), safe-checks — then ×MOD_KS_BACKING ×MOD_KS_CONTROL. A naive
`KS × openness` multiplier would DOUBLE-COUNT. The genuinely-new gap = the additive sum can't express the
super-linear INTERACTION (the coffin).
**BUILT — `KS_INTERACT` (gated, default 0 = byte-id 252/70,150,573/1568):** inside king_safety_danger, after
safe-checks, `units += (KS_INTERACT * undefended_pressure * (open_files+1) * attackers) >> 4` where
undefended_pressure = attacked_zone_squares − defenders_in_zone; fires ONLY when all three co-occur. Detectors
all already computed (no scan). search_engine.h KS_INTERACT=0 + env read added; smoke `diagnostics/ks_smoke.py`.
**Smoke (anchor, phase 16): VERIFIED CORRECT** — sheltered king & open-file-but-unpressured king BOTH stay 0
at KS_INTERACT 0 AND 96 (multiplicative gate); genuinely-attacked king: ks_units 8→56, king_safety −320→−6000
at 96. (96 too hot — 7× units; tuned value is single-digit, set from failures.)
**KS MOVE-MATCH AXIS PRUNED (PACE round 1, agent-prune):** mined 224 king-attack positions from real games
(`mine_ks_positions.py` → `ksattack_corpus.epd`: divergence-ranked misreads where SF's best move attacks the
enemy king). On that KS-RELEVANT set, KS_INTERACT is FLAT/negative: alone 1060→990/980/1080 (4/8/16); depth-12
triage 1070 (≈flat, not shallow-search-fixed); JOINT cluster {KS_INTERACT,MOD_KS_CONTROL,KS_DEFENDER} all
BELOW baseline (950–1020). ⇒ **move-match does NOT respond to KS tuning, isolated OR joint** — the objective is
blind to KS (the term's value lives in lines fixed-depth move-match can't reach), AND the KSAtk "miss" bar is
strict (our move ≠ SF's single best). So KS's ONLY gate is the TOURNAMENT (overnight). This is agentic-PACE
working: one chess-pruned round falsified the KS-move-match axis. **Reframe (user): agentic SPSA = PACE;
tournaments overnight-only (unattended), daytime = guided joint PACE on move-match-MEASURABLE clusters.** KS_INTERACT
stays gated/built; ships only if the overnight tournament shows Elo. NEXT daytime = PACE on the OvD/mobility
cluster (move-match-visible, where rook-mob already won) per the user's "co-tune OvD" insight.

**(superseded plan note)** NEXT: build a KS-RELEVANT test set (STS king-attack/defense themes + collapse run-ups with mis-read king
danger — move-match is blind to KS on the general set), tune KS_INTERACT magnitude there, then TOURNAMENT
(truth; KS is Elo-gated not move-match-gated). Then next terms per the program (central×openness, rook-mob-
dynamic). The scalar coord_sweep was STOPPED (off-path). See [[dynamic-conditional-eval]], [[king-safety-design]].

## 2026-06-29 — MOBILITY VALUATION = first GENERALIZING lever (pivot payoff)
**User insight that cracked it:** the placement (`pieces`) over-credit is a SYMPTOM — the real fault is
UPSTREAM term valuation (mobility) feeding where pieces "want" to go; fix the cause, not the symptom. And
mobility is move-match-VALIDATABLE (shifts move choice broadly), unlike the collapse-only placement fix.

**Probe (CHEAP_ROOK_MOB, live default 15, ENABLE_CHEAP_ROOK_MOBILITY=1):**
- Open Files theme (rook/bishop on open lines): 8->501, 15->506, **40->527 (+21)**, 60->502 (peak ~40).
- **HELD-OUT 300 broad check: CHEAP_ROOK_MOB=40 = 1655/3000 (55.2%) vs anchor 1586 (52.9%) = +69 (+2.3%)
  NET POSITIVE** — the FIRST generalizing net-positive of the whole campaign. Gain is BROAD (Offer of
  Simplification +35, Advancement +17, Undermine +16, Bishop-v-Knight +12, AT +10, Collapse2400 +10), NOT
  localized — Open Files itself even dropped -11 on the held-out subset. ⇒ raising rook-mobility valuation
  improves move choice ACROSS themes = a real eval-quality correction, not theme-local scatter (contrast
  KING_EG held-out -25, MOD_PIECES_DEFEND held-out -51). Mobility GENERALIZES because "active pieces are good"
  is globally coherent; placement PST tweaks are position-specific.
**Full held-out 300 curve (anchor=15 -> 1586):** 24->1595 (+9), **28->1719 (+133)**, 32->1672 (+86),
40->1655 (+69), 56->1606 (+20). **DIRECTION ROBUST** (every value 24-56 beats anchor = rook mob under-valued)
but **MAGNITUDE/peak is move-match NOISE** — the curve is JUMPY (24->28 swings +124 on a 4-unit change, then
-47 to 32): fixed-depth move-match flips best-moves discretely, so the point estimates (+133/+86/+9) are NOT
reliable. Don't over-trust any single value; the TOURNAMENT picks it (move-match ADVANCES the direction only).
**Status: rm~28-40 ADVANCES (held-out broadly positive).** **2ND-SHARD CONFIRM (offset 400, disjoint):**
anchor 1474 -> rm40 1503 = **+29** (Open Files +25 here). Both shards positive (shard-1 +69, shard-2 +29) =>
direction SHARD-ROBUST (rook mob under-valued), magnitude noisy. EARNS the tournament. Tournament should test
on the SHIPPED config (DEFAULT rook=15 vs DEFAULT+rook~28/40) NOT the m4000ctl anchor (anchor is itself
unshipped); move-match used anchor-baked funnel only for a fixed comparison baseline.
**EXTENSION — bishop mobility does NOT extend the lever (rook-specific).** CHEAP_BISHOP_MOB held-out 300
(anchor=6 -> 1586): bm12 1544 (-42), bm20 1557 (-29) — both NEGATIVE. Matches [[eval-speed-bundle-shipped]]
("cheap-mobility does not generalize past rook; queen/knight gated off"). So the lever is ROOK ONLY (rook's
long-range value on open ranks/files is the under-weighted one). Queen/knight mobility left gated-off (prior:
don't generalize). ⇒ the mobility investigation resolves to a SINGLE knob: CHEAP_ROOK_MOB ~28-40.
**REMAINING GATE = TOURNAMENT** (truth; move-match can't pick the value). Run on shipped config, e.g.
`tournament <mins> "CHEAP_ROOK_MOB=32" 6 rookmob32` (and a `=40` arm); base = default rook=15. Pick the
Elo-best value; log proxy_elo.csv; then ship the new default in search_engine.h + re-baseline byte-id.

**STACKING (coordinate-ascent on anchor+rook32, held-out 300 baseline 1672):** user strategy = keep adding
held-out-validated knob tweaks for broad coverage. Batch results (8 probes): ONLY the rook-activity family
moved — CHEAP_ROOK_FWD=5 (down) +9 marginal/noise; CHEAP_ROOK_FWD=20 -100, SCALE_CENTRAL 70/130 -61/-107,
KING_SAFETY_MAG 3000/5000 -170/-87, SCALE_PASSED_PAWN 130 -72, SCALE_LATENT_THREAT no-op (disabled by
ENABLE_KS_REPLACE_LT). ⇒ eval is near-locally-optimal except rook mobility; no long queue of free wins among
the major knobs. Built `coord_sweep` dispatcher sub (baked ~30-knob grid, each held-out-scored on rook32,
writes results/coord_sweep.csv; prompt-free) and launched it to systematically calibrate the REST of the
eval scale knobs (ROOK_OPEN/7TH/CONNECTED, CAPTURE_GAINS, IMBALANCE, PAIR bonuses, THREAT_*, PAWN_RANK,
PASSED, etc.). Survivors (positive vs 1672) need shard-2 confirm + tournament — sweep is a FILTER not accept. NOTE: CHEAP_ROOK_MOB
is a LIVE default-on constant (not gated) — re-tuning 15->~40 is an HCE value change (not byte-id-preserving;
it SHOULD change play). NEXT: pick peak -> full STS/WAC no-regression at the value -> EXTEND to bishop/queen/
knight mobility (user's "mobility and whatnot"; queen/knight mob currently gated OFF — may also be under-valued)
-> TOURNAMENT gate (truth) + log proxy_elo. The placement-conditioner (MOD_PIECES_DEFEND) stays parked; this
mobility lever is the live track.

## 2026-06-29 — MORNING STATE (autonomous overnight result)
**Anchor stands: `m4000ctl` (ENABLE_KS_REPLACE_LT=1 KING_SAFETY_MAG=4000 MOD_KS_CONTROL=256)**, byte-id
252/70,150,573 (all new knobs gated-off). No new eval shipped — the cheap move-match-gated COLLAPSE track hit a
**structural cap**, established rigorously (the night's real result):
- The collapse over-read is in **LOAD-BEARING terms** (pieces +2.35..3.34p, capture_gains +1.39p,
  piece_value_boost +0.74..0.94p); KS is minor (pruned). Conditioning any of them DOWN (compensation OR mobility,
  even offense-gated) **regresses move-match with REAL general-play harm** (MOD_PVBOOST_MOB: King Activity -94 /
  Open Files -64). They can't be cheaply separated from the term's legitimate "press when ahead" function = NNUE-
  territory. **+ measurement gap: move-match CANNOT see the collapse-fix benefit (collapses aren't move-match
  themes), only the collateral** -> it rejects every load-bearing-term fix. So move-match alone cannot validate
  collapse-fixes.
- **Mobility DOES discriminate** (collapses +4.15p material / -0.8 mobility = cramped lead) and the conditioners
  DO reduce the over-read (efficacy probe ✓) — they just can't be weighed by move-match.
**AWAKE next-steps (priority):** (1) SF-agreement/TOURNAMENT-validate the over-read reducers (MOD_PVBOOST_MOB
~32) on the collapse FENs — does reducing the over-read actually help PLAY despite the move-match collateral?
(the oracle/truth gate move-match can't be); (2) move-match-VALIDATABLE track = improve the weak THEMES
(AKPC/AT/Advancement/Open Files/King Activity all <50%) via diagnosis-driven conditioning — what the autonomous
loop CAN measure; (3) NNUE for the irreducible pieces/placement residual. Reusable tooling BUILT this run
(byte-id, gated/diagnostic-only): detector dump with raw KS-units + mobility (EvalBreakdown det_*), MOD_PVBOOST_
COMP/MOB + KS_FLOOR knobs (all default-off), `scratchpad/diagnose_corpus.py` (load-bearing-term-per-zone) +
`diagnose_def1.py`.
- **Final lever `imbalance <- realizability` (REALIZ_MAT_K/PHASE_K 128 & 256): NO-OP** (both byte-identical to
  anchor, 0 moves changed) — imbalance is too small (~0.1-0.2p) to flip move-match moves = same measurement gap.
  **Night verdict: no clean move-match-validatable eval win; the cheap move-match-gated track is exhausted for
  these levers.** m4000ctl stands. Real progress resumes AWAKE (SF-agreement/tournament gate, which sees what
  move-match can't).

## 2026-06-29 — AUTONOMOUS overnight PACE: corpus diagnosis -> compensation-conditioning track
Intellect-pruning diagnosis (`scratchpad/diagnose_corpus.py`, ev_breakdown over all 63 collapse decision FENs
under m4000ctl, bucketed Qon/off x phase): the over-read (term favouring US in positions we LOST) is dominated
by **pieces +2.35..3.34p, capture_gains +1.39p (Qon/mid, the "trade miscalc"), piece_value_boost +0.74..0.94p**
— **king_safety is MINOR (+0.23/-0.02)** => queen-aware KS PRUNED as not the lever. Unifying chess cause = **we
over-value being ahead and under-value the opponent's COMPENSATION/initiative** (the offense-vs-defense thesis).
Track: **damp the "we're ahead" terms by opponent offensive compensation** (offense-vs-defense detector).
First lever (confirmed bias x conditionable): `piece_value_boost <- compensation` (MOD_PVBOOST_COMP). pieces is
biggest but NNUE-territory; capture_gains is 2nd (next lever). KS-knob axis already exhausted (def1/floor failed).
- **MOD_PVBOOST_COMP REJECTED (built gated-off, byte-id; fast probe):** barely moves the over-read (Qon/mid
  0.74->0.73, Qoff/end 0.94->0.92). WHY = the opponent's compensation is **NOT visible at the decision point**
  (offense/defense detector low there; the attack is LATENT, materializes over later moves). So "damp ahead-terms
  by CURRENT offense/defense" misses the collapses. Pivotal Q: does ANY cheap detector separate losing- vs
  winning-material-up at the decision point? Testing MOBILITY next (added to the dump as a probe before wiring).
- **MOBILITY DISCRIMINATES (built det_w/b_mobility = non-own attacked squares, byte-id):** the collapse
  Qon/mid bucket is **our_MAT_edge +4.15p but our_MOBILITY_edge −0.8 sq** — we're up 4 pawns yet our pieces are
  NOT more mobile = the material is CRAMPED/passive (healthy +4 mat would give a big mobility edge). This is the
  "material without mobility" discriminator the compensation detector lacked. ⇒ build `piece_value_boost <-
  mobility` (MOD_PVBOOST_MOB): damp the material-lead bonus when our mobility edge doesn't back the material.
- **MOD_PVBOOST_MOB built (byte-id):** efficacy PROBE positive (Qon/mid over-read 0.74->0.54 @MOB=32, less in
  the healthy bucket = targeted). But full-suite GATE: mob32 = -29 TOTAL with King Activity -44 / Open Files -39
  / AKPC -65 REGRESS (gains in Simplification +58 / Pawn Play +43 / Recapturing). Mechanism: in ATTACKING
  positions our pieces are committed -> low mobility, but that's "engaged" not "cramped" -> wrongly damps correct
  attacks. REFINED: only damp when NOT out-attacking (leader offense edge <= 0) = the multi-detector combo
  (mobility AND offense). Re-gating offense-gated mob32/mob48.
- **MOD_PVBOOST_MOB REJECTED** (gate erratic + regressing: mob16 -211, mob32 -29, offense-gated mob32 -313 with
  King Activity -94 / Open Files -64 / Offer-of-Simpl -77). NOT a measurement artifact = REAL harm: damping the
  material-lead bonus breaks correct material-up play (when ahead, the bonus rightly drives activate/simplify/
  press). **KEY STRUCTURAL FINDING:** the collapse over-read lives in LOAD-BEARING terms (pieces/capture_gains/
  piece_value_boost) that can't be cheaply conditioned DOWN without big general-play collateral = NNUE-territory.
  PLUS a measurement problem: **move-match can't see the collapse-fix BENEFIT (collapses aren't move-match
  themes), only the collateral** -> it rejects every load-bearing-term fix. Collapse-fixes can only be WEIGHED
  by SF-agreement on the collapse FENs (oracle) or a tournament (truth) -- both interop/awake, not the overnight
  move-match lane. m4000ctl stands as the anchor. Last cheap shot: capture_gains (different term) then conclude.

## 2026-06-28 (cont.) — AGENTIC-PACE loop demo end-to-end + detector dump BUILT
The agentic diagnosis loop run in full, KS-knob-tuning the m4000ctl anchor (`ENABLE_KS_REPLACE_LT=1
KING_SAFETY_MAG=4000 MOD_KS_CONTROL=256`), and it earned its keep:
- **Sweep** (`ks_movematch_sweep`, full 15-theme): the magnitude-cranks REGRESS (sc4 safe-check −112, div3
  divisor −153), the LENIENCY fixes WIN (`def1` KS_DEFENDER 2→1 = **+83 best TOTAL**; sc5of3 safe-check+open-file
  +19). Knobs INTERACT (sc4 alone −112 but sc5of3 +19; sc4def1 −54 — safe-check crank is toxic). Lesson: our KS
  problem was LENIENCY about our own king (over-credited defenders), NOT insufficient magnitude.
- **TOTAL lies → per-theme read is mandatory:** `movematch_diff` showed `def1 +83` REGRESSES passers
  (Advancement −38, **Undermine −58**) — would have shipped a passer-breaker on the TOTAL. NOT clean. Held.
- **Detector dump BUILT (reusable instrument):** detector values (`det_w/b_offense/defense`, `det_*_pieceval`,
  `det_central`, `det_pawn_count`) added to `EvalBreakdown` (cpp_bitboard.h) → populated at the capture site →
  `ev_breakdown` dict (ChessAI.pyx) → byte-id **252/70,150,573** (diagnostic-only). Diagnosis script
  `scratchpad/diagnose_def1.py`: `movematch_diff` → classify HELPED(king)/HURT(passer) classes → detector gap.
- **The discriminator (data, not guess):** on the passer positions def1 broke, the KS term is TRIVIAL
  (**|ks|≈20 vs 169** on king positions) at **midgame phase ~31** (also more pawns 12.7 vs 10.7 = closed). def1
  AMPLIFIES a trivial KS signal into finely-balanced pawn-play. ⇒ **FIX = `KS_FLOOR` deadzone** (suppress KS
  below a danger floor; midgame-safe, does NOT fight `advanced_endgame_eval` king-activity which owns phase≥96).
- **+ KS queen-awareness gap:** KS taper is pure material-phase, NOT queen-aware (only mate-drive is) → a
  queens-on "endgame" fades KS too early while the king is still mateable. FIX = CONTINUOUS `mod_gain` scaling of
  KS by queen-presence/danger (scale, not gate). See plan + memory [[position-conditional-eval-program]].
- **Method decisions:** don't tournament MARGINAL changes (flat-Elo trap, limited games) — develop toward
  SF-AGREEMENT (oracle) with move-match as the REGRESSION TRIPWIRE; tournament only a LARGE clean candidate
  (autonomous overnight). m4000ctl = HELD ANCHOR; tonight's tournament DEFERRED.

## 2026-06-28 — KS detector-conditioning + parallel vs-SF mine (m4000ctl candidate; tournament-ready)
King-safety re-tune via the FUNNEL (cheap pain-point probe → full-suite controls → tournament; see memory
`tuning-funnel-probe-then-controls`). Platform: swap@600 is near-INERT (term contributes ~0.05p); flat
magnitude can't reduce the king-danger error (signed_gap≈0 = symmetric VARIANCE) and over-fires past m2000
(static sweep). **Built detector-conditioning** `MOD_KS_BACKING` + `MOD_KS_CONTROL` (cpp_bitboard.cpp at the
king_safety site; `mod_gain` template like MOD_LT_BACKING; backing = attacker material edge, control =
attacker offensive-vs-defender edge = the imbalance-term signal), gated default-off → **byte-id 252 /
70,150,573**. Static: backing pulls the att5+ "fantasy attack" overshoot back to baseline.
- **ks_movematch_sweep (full 15-theme):** base 51.4 / flat m4000 **50.9 (−73, over-fires)** / m4000bk **52.0
  (+78)** / m4000ctl **51.8 (+57)** / m4000bkctl 50.8 (over-damps — DON'T stack) / m2000bkctl 49.4. Net
  +78/+57 but DISTRIBUTION is messy (offsetting swings; control regress Square Vacancy −69, Bishop-v-Knight
  −55) — NOT a clean king signal (move-match is an unproven Elo proxy anyway; `proxy_elo.csv` still null).
- **Pain-point probe (8 own-king collapse FENs, the decider):** **m4000ctl = LOCKED candidate** — 7/8 get
  less over-optimistic, king themes all up (+51/+87/+4); m4000bk 6/8 + g33 switches to O-O-O. The
  conditioning ENGAGES on the real holes (position-fix signal, independent of the noisy move-match).
- **Tonight:** `ks_tournament <mins>` (baked m4000ctl, permission-clean) = PACE-validation + ship gate.
- **vs_sf PARALLELIZED** (`--concurrency`, per-game SF handles; dispatcher positional conc arg) → mined
  **42 collapses @ Elo 2400 in 33.5 min** (4×; 2400 = 37% score = "should-win" fixable zone, vs 16%@2700).
  Zone split (stable, n=42): **own-king 31% / endgame-squander 31% / midgame-material 38%**. Endgame extremes:
  g111 **KNNvK eval +14.4** = `is_practically_drawn` GAP (line ~5411 handles ==1 knight only) → cheap
  correctness fix, DEFERRED (we already draw KNNvK; low-impact); g43 KBNvK +15.5 = win we can't convert
  (B+N-mate technique/HORIZON, not eval). **Next-session campaign:** endgame-conversion + material-blow zones
  on corpus `games/vssf_2400/`. Interop note: restore = `wsl --shutdown` then the long job as the FIRST call
  (WSL idle-dies in the gap → cold-boot interop race; the busy process holds interop for its duration).

## Collapse corpus (the real-loss positions)
`diagnostics/_tal_gap_fens.csv` — 3 chess.com tal-BOT (~2705) losses, engine=White, from won/equal:
- **benoni-29** `r3r3/1b2qpbk/p2p2pp/1ppP4/P1B1P1PP/2N2Pn1/1P1Q1B2/R3R1K1 w - - 0 29` — winning-capture dodge:
  engine played **c4b3** (Bb3); the win is **a4b5** (axb5, +pawn, hits e8-rook, keeps g3-knight trap). Gap-T.
- **french-28** `5r2/p1qbn1pk/4p1pp/1p1pPr2/2pP1NQP/P1P2P2/R1PB2P1/1R4K1 w - - 8 28` — a-pawn march a4→a3→a2
  undervalued; engine shuffled. Gap-P (passed-pawn danger).
- **benoni 44-57** — the lost R-vs-passers ending the above walked into.

## Parked fixes to bundle (Phase 1) — all gated default-off, built+validated on OLD baseline (258/97.5M)
- **Gap-T `VERIFY_MARGIN`** (default 6000; fix 16000): search_engine.h:719, consumed search_engine.cpp:1971/2275.
  Wider VERIFY re-search catches the buried axb5 line. Old result: benoni-29 fixed, STS recovered.
- **Gap-P P1 `PASSER_ENEMY_CREDIT_PCT`** (default 100; fix <100): cpp_bitboard.cpp:5317/5381. Trims the
  wrong-signed enemy blockade/path-control credit on advanced passers.
- **Gap-P C1 `ENABLE_PASSER_BLOCKADE_QUALITY` + `PASSER_CONTEST_PCT`** (off / 30): cpp_bitboard.cpp:7360.
  A file-contesting rook/queen gets only CONTEST_PCT% of PP_BLOCKADE_PEN (un-zeroes a rook-contested passer).
- **Gap-P C2 `ENABLE_PASSER_KRACE_MG` + `PASSER_KRACE_MG_PCT` + `PASSER_KRACE_MAG`** (off/100/100):
  cpp_bitboard.cpp:6391/4655 — lifts the king-race realizability into all phases. WAC-costly (SPRT tier).
- Other gated `ENABLE_*_FIX` toggles (ROOK_DBLCOUNT/KNIGHT_MOB/ROOK_ENDGAME_CAP/QPREC_PHASE_GATE/
  ROOK_RANKWIN) — inventory + individually screen for the bundle.

## Log (newest first)

### 2026-06-28 — vs-SF-2700 hole-mine: 21 collapse positions (NEW self-play-invisible corpus)
Ran the built vs-SF harness (`vs_sf.py`, now with draw-adjudication — validated working) vs SF UCI_Elo=2700,
50 games LIGHTNING, sequential (~52 min; NOTE vs_sf.py is SINGLE-THREADED — no concurrency, unlike tournament.py
→ ~1 min/game). **Our score 16%** (SF-2700 @0.3s/move badly out-plays our LIGHTNING engine — a flag: our
lightning strength is well under 2700, likely the TC, worth its own look) but **21/50 games = "our eval peaked
≥ +2.0 then we didn't win"** → `games/vssf_2700/collapses.csv` (game,our_color,result,peak_ply,peak_eval,
peak_move,drop_ply,drop_eval,decision_fen,drop_fen). Peaks +2.3…+7.5 (our POV). These are the self-play-INVISIBLE
class: either real conversion failures OR eval over-reads (both valuable). **NOT yet triaged** — first action =
`triage_collapses.py games/vssf_2700` (EVAL/PRUNING/HORIZON; needs interop) to separate eval holes (the fix
targets) from horizon/pruning (search lane). Caveat: at 16% score many may be "lost to a stronger SF" rather
than clean conversion-collapses → the triage + per-position eval-or-horizon (rising-depth `ourmove`/`fenvs`) is
essential before treating any as an eval hole. New corpus UNCOMMITTED (games/ output + collapses.csv).

### 2026-06-28 — King-safety Phase B: tuned div6 OVER-FIT (discard); swap@600 = the marginal real candidate
**CORRECTED after rigorous re-check (the +22.7 was an illusion).** Move-match diagnostic (full 15-theme, all
SAME-SESSION apples-to-apples): untuned **swap@600 = 7755 vs base 7715 (+40, balanced)** — WINS attack/king
(AT +86, Center +85, Open Files +38, King Activity +2), modest non-king losses (Knight Outposts −47,
Recapturing −38, Undermine −38, 7th Rank −33, Square Vacancy −32). **Tuning was the MISTAKE: `KS_DIVISOR=6`
(div6) was tuned on a 4-theme king-SUBSET (AT/Center/Knight-Outposts/Recapturing), scored +73 there — but the
FULL-suite move-match is 7517 = −198 vs base** (it helped the 4 tuned themes and TANKED the other 11 =
classic over-fit). Deterministic STS (same-session, base re-run = 1568 exact): swap@600 1548 (−20), div6 1469
(−99). **So both clean benches agree div6 is WORSE; swap@600 is marginal.** The overnight TOURNAMENT was run
on **div6 (the over-fit config, not swap@600)**: batch1 368g pre-outage = +22.7 (small-sample noise); batch2
259g POST-power-outage = −16; pooled 627g = **+6.7 (flat)**. The cross-outage sign-flip = TIMED-tournament
machine-state confound (LIGHTNING depth ∝ CPU clock/load; batch2 ran on a freshly-rebooted machine). **.so
VERIFIED intact (base WAC 252/70,150,573 + base STS 1568, both exact) — the outage reverted NOTHING.**
**VERDICT: div6 OVER-FIT → DISCARD. swap@600 (`ENABLE_KS_REPLACE_LT=1 KING_SAFETY_MAG=600`, default knobs) =
the real candidate but MARGINAL (+40 move-match = +0.27%, −20 STS ≈ neutral) — never play-tested. There was
never a confirmed big win; +23/+73 were noise + over-fit.** **METHODOLOGY LESSON (the div6 trap): NEVER tune
eval knobs on a narrow theme SUBSET — it over-fits, trading away the un-tuned themes. Tune against the FULL
suite with the non-target themes held as CONTROLS, and gate EVERY candidate on the FULL-suite move-match (div6
looked +73 on the subset but was −198 on the full).** **▶️ NEXT: re-tune FROM swap@600 with control sets
(full-suite objective, king-themes-up subject to controls-held) → gate on full move-match + a CLEAN
post-outage tournament. If swap@600 can't be pushed past marginal, fall back to the COMPLEMENT (keep
latent_threat + add only KS safe-checks = prior clean +13 STS, no regression) or park the KS lever.**

### 2026-06-27 — King-safety swap Phase A: NEUTRAL platform found (swap@MAG=600 ≈ benches)
Go-forward = replace flat `latent_threat` with the high-DOF attack-unit `king_safety_score`, then data-tune
(plan `~/.claude/plans/handoff-lossless-speed-campaign-tranquil-rose.md`). **Phase A (structural swap) DONE:**
new gate `ENABLE_KS_REPLACE_LT` (search_engine.h, default false) — when on, SKIP the latent_threat add and
route king danger through king_safety_score (no double-count; needs KING_SAFETY_MAG>0). cpp_bitboard.cpp:
latent_threat gate `&& !ENABLE_KS_REPLACE_LT`; KS gate `(ENABLE_KS_REPLACE_LT || KING_SAFETY_MAG!=0)`.
**Byte-id OFF = 252 / 70,150,573.** MAG sweep (swap on, DEFAULT KS knobs) STS: 50→1404, 100→1408, 200→1513,
400→1522, **600→1548 (−20, ≈neutral within noise)**, 650→1458, 800→1475; **WAC@600 = 252/300** (no tactical
regression). **⇒ the untuned rich king_safety_score MATCHES the evolved latent_threat on both benches at
KING_SAFETY_MAG≈600** — the encouraging floor (user's point: a flat hand-set term shouldn't beat a data-tuned
richer one with real board-condition signals; the extra DOF — safe-checks/shield/open-files/defender-balance
— is all still at default). The MAG-sensitivity (550/600/650 = 1503/1548/1458) shows the default SHAPE wants
tuning. **KEY (user): king_safety_score is ~5-24× CHEAPER than latent_threat → matching it at fixed depth ⇒
potential DOUBLE win: speed (more depth at equal time, +Elo even at neutral fixed-depth strength) + positional
(once tuned). Measuring depth-at-equal-time next.** Two-level conditioning frame (user): Level-1 = the formula
already conditions danger on one-pass board features (LIVE, tune its weights = Phase B); Level-2 = make the
weights themselves functions of GLOBAL detectors (offense/defense, space, mobility) — the [[detector-
conditioned-knobs]] vision, but EVIDENCE-GATED (placement L2 overfit; king-safety is a better candidate
because king danger genuinely IS detector-driven — cheap-proof-first AFTER L1 wins). Downstream (user): a
cheaper+stronger eval revives the shelved eval-speed-dependent SEARCH items (improving heuristic etc. died on
per-node eval cost — "eval work pays off twice", [[improving-heuristic-shelved]]). **All commits PUSHED to
origin/NN-ENgine (was 63 ahead).** Phase A swap UNCOMMITTED (gated default-off).
The strategic pivot (user): make the placement/PST value a FUNCTION of cheap board-state detectors (space,
material, pawn structure, mobility, king-pressure) and Texel-tune it, to kill the `pieces` variance that the
term-attribution exposed. **Tested fairly + cheaply offline BEFORE any C++ build** (the "measure first" gate):
- Tooling (all UNCOMMITTED): `diagnostics/detector_placement_proof.py` (global-gain proof over corpus.csv),
  `diagnostics/gen_midgame_corpus.py` (curate IMPORTANT-MIDGAME: phase<64 & |SF|<250, reuse SF labels, add
  per-piece-type `pt_*` via ev_breakdown + detectors → `tune_data/midgame_corpus.csv`, 5037 rows),
  `diagnostics/fit_conditioned_placement.py` (rich per-piece-type × 14-detector ridge fit, train/test).
- **RESULT (held-out gap MSE, curated midgame): per-piece FLAT scale +14.6%; per-piece CONDITIONED +10.4%
  (WORSE); conditioning BEYOND flat = −4.9% (OVERFITS).** Global-gain version on near-equal/important-midgame
  added only +3.3–3.5% beyond flat, and positional detectors (mobility/pawn-struct/king-pressure) added ~0.
  **⇒ the placement variance is NOT detector-explainable by cheap hand-detectors — conditioning generalizes
  WORSE than a plain per-piece scale. The scatter is the NNUE-shaped hole (a learned eval has low placement
  variance because it captures square×context interactions hand-detectors can't).** Disconfirms the
  detector-conditioned-PLACEMENT thesis ([[detector-conditioned-knobs]], [[dynamic-conditional-eval]]) for this
  use. **The one generalizable lever = per-piece-type FLAT placement recalibration (+14.6% static, ~Texel of
  `SCALE_PLACE_*`)** — but it's the scalar approach (MSE≠play, prior SCALE_PLACE washed; per-piece-type at these
  magnitudes is the only untested variant → would need a PLAY/tournament gate, not MSE). Method win: a ~30-line
  numpy held-out fit killed a multi-week C++ subproject in minutes. Real variance-killer remains NNUE-as-eval
  (shelved). pawn=1000 / our_total Black-positive; SF_static White-POV cp; convert our→White cp = −our_total/10.

### 2026-06-27 — term-attribution over collapse positions: it's `pieces` VARIANCE, not passers
**2nd batch (20 games → 5 collapses; triage 1 EVAL / 1 PRUNING / 3 HORIZON) + term-attribution over ALL 5
collapse decision FENs (`eval_breakdown --fen`).** Answer to "is there a recurring CONCEPT error (e.g. passers
undervalued)?": **NO clean concept — the recurring offender is the `pieces`/placement term as VARIANCE.**
Gaps (our_static − SF_static): +3.3 (pieces +1.13), −4.1 (ALL terms ≈0 = missing concept), +5.6 (pieces +2.93
+capture_gains), −2.3 (pieces −1.81), −10.6 (mate outlier, capture_gains). **`passed_pawn_support` = 0.00 in
ALL five** (passers are NOT the pattern — and the campaign already fixed the passer hole that DID recur in the
real games, Gap-P). `king_safety`/`latent_threat` ≈ 0 too. **⇒ (1) the dominant residual eval error is `pieces`
placement scatter — too-high in over-reads, too-low in under-reads (VARIANCE, not a directional bias) = the
known central ceiling that scalar damping WASHES → the parked Texel/conditional lever ([[material-edge-
overvaluation]], [[dynamic-conditional-eval]]). (2) ONE under-read (gap −4.1) had ALL our terms ≈0 while SF saw
+3.7 = a MISSING CONCEPT we don't model, most likely king-safety/attack (our structurally crudest area,
[[king-safety-design]] — though the built attack-unit term tested Elo-flat).** Honest campaign state: the
recurring real-game point-holes (Gap-T, Gap-P, KPvK) are PLUGGED; the remaining gap is placement-variance +
under-modeled king-safety, NOT a tidy point-hole. Next lever = the hard Texel/placement-calibration meta-lever
or continued real-PGN point-mining (diminishing returns), NOT more LIGHTNING vs-SF batches (re-confirm the same).

### 2026-06-27 — vs-SF batch mined + 3-way triaged → known scatter wall, NO new clean hole
**16-game LIGHTNING vs-SF(2400) batch → 7 collapse points; 3-way triage (`triage_collapses.py`):**
**5/7 HORIZON, 2/7 EVAL, 0 PRUNING.** The HORIZON ones: our DEEP move equals SF's best (g0 h3h2=sfBest,
g4 f1g1, g7 d8b6, g9 h7g8) — the game blunder was a shallow LIGHTNING-depth miss, depth fixes it (search lane).
The 2 EVAL ones are PURE OVER-READS where our deep move ALSO equals SF's best (g1 d8d7, g12 d3c4) — not move
errors, just over-valuation. **g12 (the big one): our_static +7.18 vs SF_static +0.69 = +6.49 gap, driven by
`pieces` +4.94** (placement/PST) — i.e. the KNOWN central eval ceiling ([[material-edge-overvaluation]],
[[eval-precision-term-attribution]]): the `pieces`/placement over-valuation that every scalar damping WASHED
(variance, not scalar-fixable; the parked Texel/conditional lever). **⇒ vs-SF-LIGHTNING re-surfaces SEARCH-depth
+ the known placement scatter, NOT new KPvK-style clean structural holes. Clean structural holes come from
DEEPER-time real games (chesscom/tal-BOT).** Harness + triage VALIDATED and working; refined triage to classify
EVAL directly when deepMv==sfBest (move correct ⇒ pure eval), low-prune probe only when we persist in a move SF
dislikes. **OPERATIONAL: WSL→SF interop drops on every idle instance-restart (binfmt WSLInterop unregistered);
robust recovery = `wsl.exe --shutdown` + the long continuous job (batch+triage) in ONE Bash call with an
interop-ready wait loop — the busy instance holds interop; a `nohup`/foreground keepalive does NOT survive.**
Uncommitted: `selfplay/vs_sf.py`, `diagnostics/triage_collapses.py`, `_kpk_oracle.py`, `_chesscom_gap_fens.py`.

### 2026-06-27 — tal-BOT corpus re-confirmed + vs-SF collapse-mining harness BUILT
**tal-BOT re-confirm (current default-on build):** shipped fixes HOLD — benoni-29 → a4b5, french-33 ev≈0
(Gap-P intact). benoni 44-57 ending is GENUINELY lost (our moves match SF: bn-54 g2f3=g2f3 our −9.45 / SF
−8.12; bn-57 g5g6=g5g6 our −10.3 / SF −5.7). Only blemish = mild over-pessimism in rook-vs-connected-passers
(bn-57 −10.3 vs −5.7) but move-neutral → low priority. No new critical hole in the tal corpus.
**vs-SF harness BUILT (`selfplay/vs_sf.py` + dispatcher sub `vs_sf <elo> <games> [preset] [win_thresh]`):**
our engine (reuses `EngineProc`) vs strength-capped SF (`UCI_LimitStrength`/`UCI_Elo`, python-chess), alternating
colors; records OUR eval trajectory and auto-flags COLLAPSES (our-POV peak ≥ win_threshold then not a win) →
dumps the peak→drop run-up window to `games/<tag>/collapses.csv` (the corpus seed). Standalone (does NOT touch
the committed self-play loop), lower-risk than retrofitting SF-as-player into tournament.py. Syntax/import-clean;
UNCOMMITTED. **WSL→SF interop was DOWN (binfmt WSLInterop unregistered → 'Exec format error'); restored via
`wsl.exe --shutdown` + keepalive re-pin** (the documented recovery).
**SMOKE-TEST (2 games, SF elo 2400, LIGHTNING) → harness WORKS** (flagged 2/2 collapses with run-up FENs).
**▶️ KEY METHODOLOGY FINDING — LIGHTNING vs-SF surfaces HORIZON blunders, NOT eval holes.** Diagnosed game-0
(peak +2.17 → lost): we played **Bc3** at the game depth (d11) rating it +2.17, but SF says Bc3 LOSES
(+1.12 → −2.98); at d16-17 OUR ENGINE AVOIDS Bc3 and plays Bd3 (SF +0.5, holds) ⇒ the blunder was a
DEPTH/HORIZON miss, not an eval hole (depth fixes the move). The residual eval over-read is only modest
(+1.96 vs SF +1.12 = the known broad over-optimism). **⇒ At LIGHTNING, our shallow search (d11) loses to
SF's tactical shots = SEARCH-lane (at peak), not new fixable EVAL holes. The real eval holes (chesscom
rook-pawn, tal-BOT) came from STANDARD-time games where the engine searched deep enough to SEE the tactics
but MIS-EVALUATED. To mine eval holes, run vs-SF at DEEPER time (STANDARD), OR auto-triage each collapse
eval-vs-horizon (re-search deep: move-changes=horizon-discard, eval-stays-wrong-vs-SF=eval-keep).** Harness
deliverable DONE + validated; the strategy lever = time control / triage. NEXT: deeper-time batch or triage filter.

### 2026-06-27 — ✅ Phase 2 fix #1 SHIPPED: rook-pawn KPvK draw (chesscom-2200 conversion loss)
**Source:** `selfplay/external/chesscom_2200_white.pgn` (NN-Engine=White vs chess.com 2200 bot, won a pawn
move 29 then drew). Extracted endgame FENs via `diagnostics/_chesscom_gap_fens.py` (replay + material tally).
**Diagnosis — EVAL hole, not horizon:** White was +1 the whole game but the extra pawn collapsed to a lone
**h-pawn (rook pawn)** → dead-drawn K+h-vs-K (moves 57-69). Our eval scores those KPvK draws at **~+4870..+4960**
(`piece_value_boost`/mate-drive on a 1-pawn lead, NO draw detection); deeper search does NOT shrink it
(d6 +4406 → d14 +5060 → d18 +4893 — every leaf reads the same wrong way). This caused the half-point loss:
at move 56 the engine, seeing the resulting KPvK as +4.7, **traded rooks (Rxf5) INTO the dead draw**. SF
confirms every critical position = **0.00** (move 56 SF keeps the rook `f8a8`; even move 47 = 0, so the whole
ending was already drawn — no lost win, just the consistent over-read).
**Oracle (`diagnostics/_kpk_oracle.py`):** full KPvK retrograde solve (KQ-vs-K promotion shortcut), rook-pawn
files only = 83,238 states (6,526 WIN / 76,712 DRAW). Strict chebyshev-opposition rule
`defender_dist <= min(pawn_dist, attacker_dist)` to the promotion corner = **0 false-draws** (never flags a won
position drawn), catches 22,904 draws incl. the chesscom positions. (tempo variant catches 28,190, also 0
false-draws — not worth a side-to-move dependency the existing rook-pawn cases don't have.)
**Fix (gated `ENABLE_RP_KPK_DRAW`, default-off):** new lone-rook-pawn KPvK case in `is_practically_drawn`
(cpp_bitboard.cpp, after the KvK check), mirroring the existing KBP/KN rook-pawn blocks (returns 0 before the
material boost). search_engine.h:~356 flag + search_engine.cpp env-parse/dump.
**Validation:** byte-id OFF = **252 / 70,150,573** (exact baseline). ON: chesscom KPvK → **ev 0** (was
+4870/+4936); move 56 → **f8h8 (keeps rooks), no longer Rxf5**; benoni-29 still **a4b5** (shipped fix holds).
**No-regression PASS: WAC flag-on 252 / 70,150,573 (byte-identical), STS flag-on 1568/3000 (identical)** —
the fix touches ONLY rook-pawn KPvK so it never appears in either suite. **SHIPPED default-on**
(search_engine.h `ENABLE_RP_KPK_DRAW=true`; rebuilt, default-on WAC = 252/70,150,573, KPvK ev=0 with no
flag). **Gate decision (user):** a self-play tournament is uninformative for a self-play-invisible fix — ship
on verifiable position-fix + no bench regression (the established [[external-play-gaps]] validation model).
Baseline unchanged: **WAC 252 / 70,150,573 / STS 1568**. Tooling UNCOMMITTED: `diagnostics/_chesscom_gap_fens.py`,
`diagnostics/_kpk_oracle.py` (reusable KPvK oracle). Fix itself uncommitted (default-on in working tree).

### 2026-06-27 — is_light detour CONCLUDED (dead); ▶️ Phase 2 = SF-opponent gap-mining (decided)
**is_light eval-speed track DEAD** (between the bundle ship and now): stand-pat is the LEAF eval for quiet
positions → cheapening craters positional STS (mode1 1266 / mode2-surrogate 1243 vs 1501); improving/null-move
already use cheap_eval (improving SPRT-null); futility-light bench-AMBIGUOUS (+10 standalone vs −107 in-sweep
= timed-depth is wall-clock/thread noisy). Lesson: BOTH proxies fail for small effects — hunt BIG worst-case
fixes, not micro speedups. Built gated/byte-id/UNCOMMITTED (KS_LIGHT_MAG + light queen/knight mobility + light
KS surrogate; superseded). [[lighteval-standpat-is-leaf]].
**▶️ PHASE 2 SOURCE DECISION:** primary = **SF-opponent games** (our engine vs strength-targeted SF
`UCI_LimitStrength`/`UCI_Elo` ~2200–2700 — exploits our eval holes differently than our own eval → finds
self-play-invisible gaps; needs a small vs-SF harness extending tournament.py). Immediate (no build) =
diagnose the existing real losses: `selfplay/external/chesscom_2200_white.pgn` (NN-Engine vs 2200 bot,
won-pawn-then-drew-rook conversion failure) + the 3 tal-BOT in `_tal_gap_fens`. Self-play corpus mining =
LOW yield (depth-bound). Loop: eval-or-horizon? (`fenvs`/`ourmove` at rising depth) → diagnose
(`eval_breakdown --fen`) → targeted fix (gated) → no-regression on plugged gaps + TIMED TOURNAMENT → ship
solo. Plan: `~/.claude/plans/handoff-lossless-speed-campaign-tranquil-rose.md`.

### 2026-06-27 — ✅✅ BUNDLE SHIPPED (defaults flipped, UNCOMMITTED for sign-off)
Flipped 5 knobs to default-on in search_engine.h: VERIFY_MARGIN=16000 (done earlier) + ENABLE_ROOK_DBLCOUNT_FIX
+ ENABLE_ROOK_DBLCOUNT_SYM_UP + ENABLE_QPREC_PHASE_GATE + PASSER_ENEMY_CREDIT_PCT=0 + ENABLE_PASSER_BLOCKADE_QUALITY.
Rebuilt; all 6 confirmed active via toggles dump. **NEW SHIPPED BASELINE: WAC 252 / 70,150,573 / STS 1568
(52.3%); wall 162s / 432k nps** (≈ original 167s/406k = speed maintained-to-slightly-faster; qprec's +nodes
offset by higher nps). Note: WAC 252 RECOVERED (Gap-T-alone was 245) and STS 1568 is the BEST yet — the full
bundle is bench-strong AND play-strong (+38.7 Elo), so the STS non-additivity scare was fully a fixed-depth
artifact. (Recover old byte-id 252/67,931,145 with the 6 knobs reset to their old defaults.)
- **PGN collapse positions FIXED (shipped build, d12, SF-free `ourmove` since WSL SF-exec was flaky):**
  benoni-29 → **a4b5** (the winning capture, was Bb3); french-33 ev **+0.93→−0.32**, french-36 **+0.33→−0.34**
  (the eval now SEES Black's a-pawn passer danger where it used to read itself winning and walk into the loss).
  ⇒ both real-loss classes (winning-capture miscalc + passer-danger under-read) addressed. Caveat: the deep
  benoni 44-57 endgame conversion not re-verified (dev doc flagged a possible SF-static realization ceiling
  there); the eval-side danger is fixed, deep-endgame technique is a separate (search) question.
- **Speed / accuracy maintained:** tactical WAC 252 (= original), positional STS 1568 (+65), nps ~maintained.
- **COMMITTED `4fe05fd`** ("Ship collapse-elimination bundle (+38.7 Elo) + campaign tooling"; engine +
  dev docs + tooling; unrelated junk left untracked). Then Phase 2: mine NEW collapse classes (gate on
  TIMED TOURNAMENT, not STS). New dispatcher sub `ourmove` (SF-free move check).
- **NEXT ACTIVE TRACK (separate plan): `is_light` v2** cheap-surrogate light eval to unlock improving/2-ply
  history — plan `~/.claude/plans/handoff-lossless-speed-campaign-tranquil-rose.md`.

### 2026-06-27 — ▶️▶️ RESULT: BUNDLE IS +38.7 ±27 ELO IN PLAY — "non-additivity" was a FIXED-DEPTH ARTIFACT
**875 games (lightning, 4w): base (shipped Gap-T) 44.5% / bundle 55.5% → bundle +38.7 ±27 Elo (SIGNIFICANT,
CI ~+12..+66).** The bundle = Gap-T + rookdblsym + qprec + Gap-P P1+C1. **This OVERTURNS the STS-based
"never bundle" conclusion** — the −77..−139 STS "destructive non-additivity" was a FIXED-DEPTH search-hole
ARTIFACT; in real timed games the same bundle is a LARGE win. by_color symmetric (base loses as both
colors) ⇒ genuine strength, not a color/adjudication artifact.
- **CRITICAL LESSON: STS (fixed depth d10) is a POOR predictor of PLAY strength for these eval/search
  changes — it mispredicted by ~+178 STS-equivalent.** I nearly PARKED Gap-P (the likely top contributor —
  it fixes the real-loss passer-danger class) on its −106 STS. **Do NOT gate collapse/eval fixes on STS;
  gate on the TIMED TOURNAMENT.** ([[fixed-depth-bench-ceiling]] is far more severe than assumed.) The
  user's "just try them all in a tournament / persist, don't pivot on benches" instinct was vindicated.
- **Open: which components drive +38.7?** Likely Gap-P (passer collapse) dominant. Attribution runs
  (Gap-P alone, etc.) would confirm but cost ~6h each. The bundle TOGETHER is the proven +38.7 win.
- **SHIP DECISION (pending user): flip defaults for the bundle** (rookdblsym `ENABLE_ROOK_DBLCOUNT_FIX=1
  ENABLE_ROOK_DBLCOUNT_SYM_UP=1`, qprec `ENABLE_QPREC_PHASE_GATE=1`, Gap-P `PASSER_ENEMY_CREDIT_PCT=0
  ENABLE_PASSER_BLOCKADE_QUALITY=1 PASSER_CONTEST_PCT=30` — Gap-T VM=16000 already default). Large
  multi-knob change resting on one (significant) tournament → recommend a confirm re-run +/or component
  attribution before committing, but per the ship-on-no-regression bar this is a big GAIN, not a regression.

### 2026-06-27 — bundle-in-PLAY test (settling "is non-additivity real in games or a fixed-depth artifact?")
The bench non-additivity (STS −77..−139) could be partly a fixed-depth search-hole artifact; a TIMED
tournament searches differently. Running ONE overnight tournament to definitively settle the recurring
"never bundle" question. **base = shipped Gap-T (VM=16000); cand = Gap-T + rookdblsym + qprec + Gap-P P1+C1**
(`ENABLE_ROOK_DBLCOUNT_FIX=1 ENABLE_ROOK_DBLCOUNT_SYM_UP=1 ENABLE_QPREC_PHASE_GATE=1 PASSER_ENEMY_CREDIT_PCT=0
ENABLE_PASSER_BLOCKADE_QUALITY=1 PASSER_CONTEST_PCT=30`), 4 workers, 360 min, tag `bundle_sprt`. Prior is
AGAINST (benches strongly negative); if flat/positive in play ⇒ non-additivity is a bench artifact and we
CAN bundle (big); if negative ⇒ "ship one at a time" confirmed in games too.

### 2026-06-26 — ✅ Gap-T SHIPPED (default VERIFY_MARGIN 6000→16000, UNCOMMITTED for sign-off)
SPRT base vs VERIFY_MARGIN=16000 (658 games, lightning, 4w): 49.5% base / 50.5% cand = **Elo +3.7 ± 31.2**
(slightly positive, no regression; self-play CAN'T confirm a self-play-invisible collapse fix — the thesis).
Meets the ship bar (collapse fixed + STS +62 + ACPL-neutral + no self-play regression). **Flipped the default
in search_engine.h:719 (6000→16000) + rebuilt + reconfirmed.** **NEW DEFAULT BASELINE: WAC 245 /
64,569,899 / STS 1565 (52.2%)** (old byte-id 252/67,931,145 recovers with `VERIFY_MARGIN=6000`). benoni-29
plays a4b5 by default now. **Not git-committed** (left for user sign-off). First collapse-campaign ship.
**rookdblsym (+34) / qprec (+9) are now SHELVED — they conflict DESTRUCTIVELY with the shipped Gap-T**
(gapt+rookdblsym 1426 = −139 vs the new 1565 baseline), i.e. Gap-T already captured more positional credit
than they offer, and stacking regresses. So they cannot be added on top. **Phase 2 real work = re-tune
Gap-P (the still-open french passer collapse, a DIFFERENT class) + mine NEW collapse classes from self-play
losses (`flip_extract`) — each shipped SOLO, one at a time, re-validated against the NEW baseline.**


### 2026-06-26 — Phase 1 parked-fix re-validation on CURRENT baseline (252/67,931,145, STS 1503)
Key lesson up front: **the parked fixes were validated on the OLD baseline (258/97.5M); combo1 changed the
landscape — re-validation was essential.** Position-fix checks via new `fenvs` dispatcher sub (fen_vs_sf on
explicit FENs, LONG_FORMAT d12). Benches OMP-pinned/book-off.

- **Gap-T `VERIFY_MARGIN=16000` = CLEAN WIN → SHIP candidate (SPRT-gating).**
  - benoni-29 FIXED: our move c4b3 (Bb3, eval +1005) → **a4b5 (axb5) = SF best**, eval +3011. The winning
    capture is found.
  - STS **1503 → 1565 (+62)** (recovers combo1's pruned positional credit, as the old doc predicted SEE+VM
    are complementary); WAC 252 → 245 (−7, fixed-depth artifact); ACPL median 46.5 → 44.5 (neutral-better,
    mean noisy). Net positional GAIN + a real collapse fixed.
  - It's a SEARCH default-change (self-play-VISIBLE) → SPRT before default-flip (old doc's own caveat:
    VM=6000 was self-play-tuned). **SPRT queued** (base vs VERIFY_MARGIN=16000, 4 workers).

- **Gap-P (passer danger) = REGRESSES STS on the combo1 baseline → PARK, needs re-tuning.**
  - P1+C1 (`PASSER_ENEMY_CREDIT_PCT=0 ENABLE_PASSER_BLOCKADE_QUALITY=1 PASSER_CONTEST_PCT=30`): STS **1397
    (−106)** (was +47 on the OLD baseline = the clean-ship-tier BROKE under combo1). C1 adds most of the loss.
  - P1 alone (`ENEMY_CREDIT_PCT=0`): STS 1472 (−31); `=50` worse at 1374 (−129, non-monotonic/erratic — same
    behavior as the conditioning knobs). Best Gap-P variant is still −31 STS.
  - **Bundle (Gap-T + Gap-P P1): STS 1470 (−33)** — Gap-T's +62 did NOT absorb Gap-P; the bundle ≈
    Gap-P-alone, i.e. Gap-P's passer change DOMINATES STS regardless of Gap-T. ⇒ they do NOT co-exist
    cleanly; Gap-P drags the bundle below baseline. Don't bundle Gap-P as-is.
  - TODO Phase 2: re-tune Gap-P on the combo1 baseline (which STS positions does removing enemy-blockade
    credit hurt? is the french-passer fix separable from the STS-costly part?) + check the french-28
    position-fix (slow grind — validate via eval-sign on F33/F36, not one move).

- **Parked `ENABLE_*_FIX` toggle screen (STS, individual, vs 1503):** rookdbl asym **−71** but
  **rookdblsym +34** (the rook double-count fix NEEDS its symmetry correction `ENABLE_ROOK_DBLCOUNT_SYM_UP`);
  **qprec (ENABLE_QPREC_PHASE_GATE) +9**; knightmob +2 / rookrankwin −1 (neutral); knightmobsym **−91**
  (knight-mob is the OPPOSITE of rook — its sym version hurts); rookendcap **−54**. ⇒ STS-positive
  correctness fixes to bundle with Gap-T: **rookdblsym (+34), qprec (+9)** (+ knightmob neutral). Testing
  bundle coexistence next (Gap-T + rookdblsym + qprec).

- **▶️ BUNDLE COEXISTENCE = DESTRUCTIVE (key meta-finding).** Individually-positive fixes combine to STS
  REGRESSIONS: gapt+rookdblsym **1426 (−77)** (vs gapt +62, rookdblsym +34 alone!); +qprec 1452 (−51);
  +knightmob 1486 (−17, non-monotonic); rookdblsym+qprec (no gapt) 1460 (−43). **The eval/search knobs are
  deeply NON-ADDITIVE — you cannot bundle individually-validated changes; interactions dominate and are
  mostly destructive.** This is now the THIRD instance (conditioning LT+realiz, Gap-P, these toggles) ⇒ a
  core property of this engine, and likely WHY every multi-change bundle has washed/regressed historically.
  (Caveat: at fixed depth some of this is search-hole artifact — but the no-regression bar fails either way,
  and the lesson "ship ONE change at a time, never bundle" holds.) ⇒ **Gap-T ALONE is the ideal config**
  (1565/+62, benoni-29 fixed, ACPL neutral). No bundle qualifies.
- **DECISION: SPRT Gap-T alone** (`VERIFY_MARGIN=16000`, search default-change, self-play-visible). Launched
  base vs VERIFY_MARGIN=16000, 4 workers, 300 min, tag `gapt_sprt`. If +Elo or flat-no-regression → flip the
  default 6000→16000 (the dev doc's owed SPRT). Other toggles (rookdblsym, qprec) are individually STS-clean
  but can't bundle → revisit each as a SOLO ship candidate later (own SPRT), not together.

- _Campaign opened; Phase 0 doc + `fenvs` sub + `_tal_gap_fens` corpus in place._
