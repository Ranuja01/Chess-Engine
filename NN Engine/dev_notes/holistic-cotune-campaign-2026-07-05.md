# Holistic king-attack cluster co-tune — campaign spec (2026-07-05)

The active eval program. Supersedes isolated single-term tuning (KS proved that fails from INCOHERENCE — see
`ks-build-plan-2026-07-04.md` verdict). Design = our reframe + Fable's two rounds, source-verified.

## Why (grounded)
Equal-depth vs SF11 = 7.5% / **−437 Elo PURE EVAL gap** + 35 collapses. Per-theme: weak on DYNAMIC/attacking
(AKPC 38%, King-Activity 43%, Open-Files 46%, Attacking 46%, Square-Vacancy 46%), strong on static/material. The
eval is one vector; tuning one term against a frozen rest makes it incoherent. Fix = co-rebalance the interacting
CLUSTER. NNUE is later, off our OWN eval (not now, not SF-distill) — there's ~437 Elo classical headroom first.

## Mechanism (settled): incoherence, pruning benign
Ruled OUT margin-noise-tax (mechanism 2): a variance tax INFLATES nodes/re-searches; KS-on DEFLATES (36.7M vs
39.1M @d10) AND improves move-choice (+1.3% STS). ⇒ benign efficiency + mild move-choice incoherence that depth
partially repairs. So margins are CONSERVATIVE now, not mis-firing → they belong in Stage 2, not the Stage-1 test.

## Objective (inner loop) — cploss-vs-SF18 in WDL space, dual-screened
- **cploss (oracle-ACPL):** per corpus position, SF18 scores its best move and our chosen move; loss = Δ. Argmax-
  based → escapes the SF-magnitude-target trap (we minimize REGRET as the judge measures it, never regress toward
  its numbers). Judge = SF18 (strongest = lowest label error/variance = the scarce resource at our throughput);
  NNUE-style-leak doesn't bite (unreachable prefs = flat gradient; gates + transfer-ratio police residual).
- **WDL/expected-score space, NOT raw cp:** convert judge evals through the win-prob model, loss = Δ(win prob).
  Caps decided-position influence, matches outcome units, fixes SF11/SF18 cp-scale mismatch. (Add to cploss_probe,
  which currently uses raw cp.)
- **DUAL COMPASS:** require the outcome-incremental-validity sign to AGREE before any candidate reaches the gate
  (kills matched-move/mismatched-plan artifacts).
- **Corpus (FROZEN per campaign, stratified):** themed + neutral-book + SF11-selfplay + the 35-collapse tail. NOT
  only our own games (cploss_probe's current bias). Freeze once — adaptive re-mining = silent leakage. Held-out
  themes + a never-touched control suite.
- **Role separation:** SF18 = judge (cploss + game arbiter). SF11 = venue (equal-depth yardstick) + feature
  library. outcomes = screen. games (node_ab/SPRT) = truth.

## ⭐ TUNE THE CONDITIONING FUNCTION, NOT BARE MAGNITUDES (user 2026-07-05)
A flat magnitude (KING_SAFETY_MAG=X) is a best-on-average compromise that's wrong in every specific position —
it over-fires on unbacked fantasy attacks (the collapse stratum: flat MAG=1500 made collapse cploss WORSE 25→48)
and under-fires elsewhere. The fix is a DYNAMIC magnitude = `value × f(board-state)`, tuning the FUNCTION f, not
the scalar. Evidence: capg-tension (our ONLY shipped eval win) was exactly this. We already have the f-hooks (mostly
gated): **KS_DYN/KS_DYN_PIVOT** (per-king attack-realness scaler), **MOD_KS_BACKING** (damp materially-unbacked
attacks = the fantasy-attack/collapse killer), **MOD_KS_CONTROL** (scale by board-control edge), REALIZ_MAT_K/
PHASE_K (imbalance realizability). These died in ISOLATION before — the untried test is tuning them INSIDE the
coherent holistic vector, where the compass rewards being right-per-position (esp. the collapse stratum flat KS
wrecked). This is the HCE way to add the position-dependent nonlinearity (before conceding NNUE). v1 spec
(`eval_cluster.json`) = magnitude anchors {KING_SAFETY_MAG, SCALE_ATTACK_LAYER, SCALE_THREATS, IMBALANCE_SCALE} +
KS conditioners {KS_DYN, KS_DYN_PIVOT, MOD_KS_BACKING, MOD_KS_CONTROL}. Larger overfit surface → lean on holdout +
transfer-ratio.

## Cluster (functional, deduped)
King-attack + piece-activity (where the gap concentrates): **KS_CONSOLIDATE (anchor member, gated-on, init
sane-not-hot) + threats + king-zone attack-layer weights + shield terms + near-king mobility + OvD imbalance** +
a **global cluster-vs-material anchor scalar** (so "the new term is right, everything adjacent shrinks 10%" is
expressible). DEDUPE double-counts first (the KS-audit representation map) so we tune real signal. Hold
static/material terms fixed. Order: king-attack FIRST (data), then structure, then endgame-scale.

## Optimizer — surrogate-assisted / pattern search, NOT SPSA
Compass is DETERMINISTIC → SPSA wastes evals averaging absent noise. Use: sample the cluster box (~30–60 compass
evals), fit a local quadratic, jump to its optimum, re-sample (or classic coordinate/pattern search w/ adaptive
steps + trust region). SPSA ONLY for the final eval+margin joint touch-up on node_ab GAMES (genuinely stochastic).
Fishtest analog: they had infinite games + NO deterministic compass; we have the compass → deterministic inner
loop → calibrated cheap games (node_ab) → SPRT ship. Don't cargo-cult fishtest.

## Overfit guards
Ridge toward STATUS QUO (Δw penalty + per-round trust region) — NOT toward SF weights (they're co-adapted to
their ecosystem; our mobility corr-0.78-but-6×-under proves SF magnitudes don't transfer; SF weights = init hint
only). Frozen corpus. Held-out at every level. **TRANSFER-RATIO habit:** after each cluster round, node_ab; track
(gate Elo)/(compass gain); stable ≈ healthy, collapse→0 = compass-saturated = STOP (that's the tapped signal, not
tune-harder). This converts "burned by fit metrics" into a self-correcting loop — the compass can mislead for only
one round before the ratio exposes it.

## The two-stage run (falsifiable)
- **Stage 1:** co-tune the eval cluster, MARGINS FROZEN → node_ab. Clean test of coherence.
- **Stage 2:** short margin sweep {futility, razor, RFP, aspiration, LMP} on the Stage-1 winner → node_ab.
- **SPRT the coherent bundle ONCE** at the end (blitz, KS's fair venue).
- **Outcomes:** S1 converts → coherence was the story. S1 flat / S2 converts → pruning-ceiling. BOTH flat →
  coupling hypothesis WRONG for KS → honest residual = "dynamic king danger needs nonlinearity the basis can't
  express" = MEASURED NNUE evidence (earned, not assumed). Every branch moves us.

## Spot-fix vs holistic (per case) — detector-precision test
For each failure family: does a cheap live detector separate the failure positions from the ~94% normal with HIGH
precision? YES → spot-fix (conditioned subtraction/gated term, bench_gate-validated — where every shipped eval win
lives: capg-tension, draw-cliffs, KPvK). NO → holistic (re-price features that fire everywhere; signatures =
single-knob collateral + themes on a Pareto frontier). Classify the 35 collapses first; expected to cluster on
king-attack mispricing → the cluster (the middle of the blast-radius dial).

## Execution order (all cheap first)
1. Classify the 35 collapses (extract FENs from selfplay/games/sf11_evalgap_d8/*.jsonl) + detector-precision test.
2. Freeze the stratified corpus (themed + neutral-book + SF11-selfplay + collapse-tail).
3. Add WDL-cploss (SF18 judge) to the cploss tool.
4. Dedupe + define the cluster knob-set (+ anchor scalar).
5. Stage-1 surrogate co-tune (margins frozen) → node_ab (transfer ratio) → Stage-2 margin sweep → SPRT bundle.

## Verified (Fable claims, source-checked)
LMR_PROFILE exists + histograms fail-low dropped moves (search_engine.cpp lmr_profile_event/dump). WDL/sigmoid
machinery in tune_fit/sprt. capg shipped w/ LMP_BASE=1 (search_engine.h:296, +39 STS) = the margin-in-vector
precedent. mobility corr 0.78 / 6×-under (breakdown_gap). node_ab = calibrated 3.6× ruler. cploss_probe = the
oracle-ACPL tool (uses raw cp + samples only our games → add WDL + stratify).

## ✅ BUILD STATUS + TOOLING (2026-07-05, all python, byte-id 245 untouched)
- **`diagnostics/build_cploss_corpus.py`** → froze `selfplay/tune_data/cploss_corpus.csv` (824 pos: collapse24 /
  sts350 / neutral200 / game250). Stratified + de-biased (STS themed + UHO neutral-book + our-vs-SF11 + EVAL-collapse
  tail from triage). Deterministic (seed 13).
- **`diagnostics/cploss_frozen.py`** + dispatcher sub `cploss_frozen <our_d=10> <judge_d=12> <lim=0> [KNOBS]` = the
  WDL-cploss compass. Loss = winpct(cp_best)−winpct(cp_after) (Lichess K=0.00368208), **SF18 judge** (fixed depth,
  deterministic), global position→cp cache (warms across candidates), per-stratum + `--shard {all,train,holdout}`.
- **`diagnostics/surrogate_tune.py <spec> <tag> [--evals 80] [--ridge 0.5]`** = pattern-search driver (Hooke-Jeeves +
  trust-region, ridge-to-status-quo), BALANCED-stratum objective (mean of 4 stratum means → collapse can't be sold
  out), optimises TRAIN shard, validates HOLDOUT, prints node_ab config. Reuses spsa `_cfg_string`/`_clamp` idea.
- **`selfplay/eval_cluster.json`** = v1 CONDITIONING spec: base `KS_CONSOLIDATE=1 ENABLE_KS_REPLACE_LT=1 KS_ZONE2=1
  ENABLE_THREATS=1`; params {KING_SAFETY_MAG, KS_DYN, KS_DYN_PIVOT, MOD_KS_BACKING, MOD_KS_CONTROL, SCALE_ATTACK_LAYER,
  SCALE_THREATS, IMBALANCE_SCALE}.
- **JUDGE ROLES (locked, user 2026-07-05):** SF18 = MOVE judge (cploss compass — drift toward perfection; regret/
  argmax, not magnitude-match). SF11 = per-term EVAL breakdown (incremental_validity dual-screen + breakdown_gap —
  apples-to-apples handcrafted) + equal-depth VENUE. Different jobs, both correct.

## ✅✅ STAGE-1 FINAL (2026-07-05, complete) — conditioning WORKS + narrow-limit CONFIRMED
**WINNER** (train_obj 30.168): `KS_CONSOLIDATE=1 ENABLE_KS_REPLACE_LT=1 KS_ZONE2=1 ENABLE_THREATS=1
KING_SAFETY_MAG=3250 KS_DYN=512 KS_DYN_PIVOT=4 MOD_KS_BACKING=256 MOD_KS_CONTROL=32 SCALE_ATTACK_LAYER=100
SCALE_THREATS=100 IMBALANCE_SCALE=3`. All 3 KS conditioners engaged (KS_DYN maxed, MOD_KS_BACKING=256, MOD_KS_CONTROL
tuned to a SMALL 32 — 256 was bad, 0 fine, 32 best = a real tuned conditioner).
**HOLDOUT GENERALIZES** (not overfit): init 26.77 → winner **22.59**. vs the **DEFAULT eval** on the same holdout:
balanced 23.83 → 22.59 = **−1.24 net** (~0.12 pp win%/move). PER-STRATUM (default→winner): **collapse 11.10→8.54
(−2.56, CRUSHED)**, sts 35.65→35.64 (flat), game 26.58→22.55 (−4.03, better), **neutral 21.99→23.63 (+1.64, WORSE)**.
- **DOUBLE finding:** (a) conditioning WORKS — collapse (the eval holes) fixed ~3× + generalizes + net-positive =
  the mechanism converts on the compass (flat KS never did); (b) the **neutral REGRESSION empirically confirms the
  make-room/over-valuation confound** (user) — narrow cluster (no capgains/placement to give way) → hotter KS
  over-reads QUIET positions. So this run validates the mechanism AND proves the broad cluster is needed. Exactly the
  diagnostic we wanted. Artifacts: results/cotune_ks1_log.csv, results/cotune_ks1_state.json.
- **NEXT (new context):** (1) node_ab winner-vs-base (transfer-ratio; KS is NPS-non-neutral + node_ab depth-biased →
  treat as directional, blitz is the fair venue); (2) Stage-2 flywheel (RFP/futility/aspiration/LMP re-tune — the
  compass can't see it); (3) the BROAD "king-attack budget" co-tune (add capgains + placement + OvD-conditioners +
  mobility; bigger 2–4k corpus; random restarts for the jagged landscape) = the make-room fix. Compass is a proxy →
  games decide (−202 lesson).

## (superseded) STAGE-1 RESULT (in progress) + THE HONEST READ
Balanced train cploss: init-cluster 40.80 → **best 31.00** (config `KING_SAFETY_MAG=3500 KS_DYN=512 MOD_KS_BACKING=256`,
rest default; robust local optimum, jagged move-flip landscape). **HONEST CALIBRATION:** the −9.8 is mostly recovery
from the bad flat-KS init (which was WORSE than default); vs the DEFAULT eval (~32 balanced) the tuned cluster is only
~1 point better (~0.1 pp win%/move). **The real win is the SIGN FLIP:** flat KS-addition was net-NEGATIVE vs default
(why isolated KS always failed); CONDITIONED KS is net-POSITIVE → conditioning made an addition work. Driven entirely
by the DYNAMIC conditioners (KS_DYN maxed + MOD_KS_BACKING=256, the fantasy-attack damper), NOT flat magnitudes —
thesis validated. `MOD_KS_BACKING` helped only once KS_DYN was high (isolation-fails-holistic-works, live). Compass is
a proxy → node_ab/SPRT decide. Pending: winner holdout + per-stratum (did it fix COLLAPSE specifically?) + gate.

## ★ KEY INSIGHTS THIS SESSION (user, load-bearing)
1. **Tune the conditioning FUNCTION, not bare magnitudes** — flat scalar is best-on-average (over-fires fantasies,
   under-fires elsewhere). value×f(board). = [[realizability-conditioning-architecture]] (banked).
2. **The BALANCE / "make-room" confound** — terms share a material-relative budget; raising KS without neighbors
   giving way makes the TOTAL over-read → KS *looks* wrong when it's fine. The co-tune HAD the make-room option
   (SCALE_ATTACK_LAYER/THREATS/IMBALANCE tunable, probed lowering them → compass declined) but a FIXED-DEPTH compass
   can't see the over-value×pruning interaction → only the JOINT config in games (Stage-2 search re-tune) arbitrates
   the true balance. Gate the JOINT, always.
3. **The narrow cluster is a real limitation** — capgains + placement + mobility + OvD-conditioners are NOT in v1
   (overfit control), yet they're exactly the make-room terms (the collapse over-read was `pieces`+`capture_gains`
   over-firing). So v1 can't find the GLOBAL balance. → the BROADER "king-attack budget" co-tune (add capgains,
   placement, OvD-conditioners, mobility; bigger 2–4k corpus; stronger holdout/transfer guards; random restarts for
   the jagged landscape) is the next/truer run.
4. **"How to get MORE points" — NOT data** (824 pos plenty for 8 knobs; data is the NNUE limiter). Missing, in order:
   BREADTH (each conditioned subsystem accumulates), the FLYWHEEL (better+faster eval → prune harder → deeper at
   equal time; INVISIBLE to a fixed-depth compass; Stage-2), and REPRESENTATION DEPTH (richer DETECTORS/shapes — our
   shelter=crude count vs SF per-file model; build shelter_storm + safe-mobility = engineering, biggest classical
   juice; SF11's +437 proves it's classically reachable). NNUE only after HCE genuinely exhausted.

## ⭐ DATA-GENERATION / FAILURE-MINING FLYWHEEL (user 2026-07-05) — feeds the campaign
Play LIGHTNING games (SF ~1s cap vs our lightning) as an ONGOING generator, not a one-off benchmark. Machinery
already exists (`vs_sf` → games + `collapses.csv`; `label_collapses` → EPD; `triage` → EVAL/PRUNING/HORIZON — this
is exactly how we mined the 35 collapses + −437). Value: (1) the one real classical "data" lever = MORE targeted
FAILURE positions → the bigger 2–4k corpus the broad co-tune wants; (2) real-TC produces a DIFFERENT failure mix
than fixed-depth — triage splits EVAL (→ eval-tune corpus) vs PRUNING/HORIZON (→ SEARCH-tune targets), so one batch
feeds BOTH lanes; (3) lightning = high throughput = ideal overnight/background generator. Loop: generate → mine
collapses+wins → triage → EVAL holes to co-tune corpus / PRUNING+HORIZON to search → tune → cleaner games → repeat.
Discipline: mined positions are OUR-GAMES-biased (failure-targeting yes, but blend with neutral strata — the
stratified corpus handles it). Opponents: SF11 or Elo-capped SF work NOW; **SF1-FULL** (2008 original, team-made, top for its era, ~2600–2700) is
the preferred ABSOLUTE-strength gauge — user is providing the binary (Linux + Windows from the archives; **use the
LINUX native ELF** → no Windows-exe/binfmt fragility, cleaner than SF11). Elo-capping is ARTIFICIAL (a strong engine
injecting deliberate blunders to hit a target = incoherent play); SF1-full plays its real, coherent (dated) game =
a clean absolute head-to-head + natural failure patterns to mine. We're ~2700 → close, meaningful match. Slots into
`vs_sf.py` exactly (opponent=SF1, arbiter=SF18); one due-diligence step = a 5-game smoke to confirm the 2008 binary
speaks UCI cleanly to python-chess. The ladder: SF1(absolute low rung)→SF11(−437 fixed-depth target)→SF18(judge).
Stay HUMBLE: SF1→SF18 ≈ 1000+ Elo; a good SF1 score = "healthy young engine on the classical path," not "near SOTA."
- **CORRECTION 2026-07-05:** the `stockfish_1` folder's `ja` builds use dotless naming — `stockfish-11-ja` = SF **1.1**
  (not SF11), and the user has the WHOLE 1.x line (1.0→1.9.1, 2010 ja builds) + SF11 + modern SF, several WITH SRC.
- **FABLE SOURCE-EVOLUTION ASK (prepare, for a later context):** we now hold SF1.x + SF11 + SF18 SOURCE. Ask Fable
  (or an analysis agent) to investigate the EVOLUTION SF1→SF11(classical)→SF18 across ALL subsystems — eval, move-gen,
  caching, search — and surface concrete "what changed / what got added" upgrades we could port, cross-referenced
  against OUR engine. Feeds the richer-DETECTORS lever (biggest classical juice) + search/movegen/caching. ADD to the
  existing Fable info bank (dev_notes/external-audit-2026-07-03.md + engine-vs-sf11-map-2026-07-04.md = the accumulated
  lever list). Source-verify every claim before acting (Fable discipline). SF1→SF11 = the CLASSICAL-eval history
  (relevant); SF12+ eval = NNUE (skip for eval, but movegen/caching/search evolution still relevant).
