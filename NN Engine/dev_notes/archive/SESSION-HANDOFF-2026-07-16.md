# SESSION HANDOFF — 2026-07-16 (overnight autonomous)

Read this first in the morning. Full detail in the linked notes. byte-id 247 intact; NOTHING committed.

## THE ARC OF THIS SESSION (what changed)
1. **Falsification (falsification-bent-ruler-2026-07-15.md):** the "eval lanes closed / load-bearing optimism
   fundamental" doctrine was a MEASUREMENT ARTIFACT of the fixed-NODE gauntlet (penalizes eval accuracy). Mobility
   fixed-node −1.05% → fixed-DEPTH +24.8 → fixed-TIME **+20 Elo self-play**. Eval lanes REOPENED (narrow claim).
2. **Mediocre yardstick built + validated (raw_uci.py; self-heals Mediocre's nondeterministic deadlock via
   kill+respawn; vs_sf.py --opponent-raw):** first REAL external opponent. Fair equal-time (our LIGHTNING ~1s vs
   Mediocre 0.5 @1s, ~2319 CCRL). **40-game baseline = 13.8%** ⇒ real strength ~1900-1950. The ~2700 self-estimate
   (and chess.com "2700 bots") = INFLATED by our over-optimistic eval. This is coherent with "can't beat SF1".
3. **Mobility transfer test = INCONCLUSIVE, not a failure** (I initially over-read it as "self-play misleading" —
   CORRECTED): vs Mediocre 11.2% vs 13.8% base over 40g, but Mediocre@13% is TOO INSENSITIVE to detect a +20 lever
   (need ~50% operating point). Neither confirms nor refutes transfer. OPEN QUESTION: does self-play Elo transfer?
   Test at a sensitive real venue. LESSON: stop turning "can't measure here" into "didn't work."
4. **Collapse autopsy (mechanism, not Elo):** diagnostics/collapse_autopsy.py — toggle-recovery labels WHY each
   loss move fails. n=39 (our-eval-drop localization, DIRECTIONAL): ~70% EVAL (46% drift + 23% mis-rank) / ~28%
   prune (mostly RFP+razoring = shallow static-eval prunes firing on the inflated eval). ⇒ root = EVAL OVER-
   OPTIMISM, mechanism-confirmed. Loss profile = "lose from EQUAL" (grinds), not "collapse from winning".
5. **Asymmetry prune-gate diff (stability-gap-audit-2026-07-15.md):** min vs max prune gates are TIGHT (only null
   3-vs-4 + a futility cur_depth>1 gate differ) ⇒ WEAK evidence for a negamax rewrite. If asymmetry matters, ALIGN
   the 2 knobs + A/B — do NOT rewrite. (Only prune gates diffed; LMR/PVS/extensions still to check.)

## STRATEGIC FRAME (grounded)
Gap to Mediocre ~350-420 Elo, but SAME HCE family (no capability gap) — closable, as many hobby HCE engines did.
Our gap is UNUSUALLY CONCENTRATED in eval CALIBRATION (systematic over-optimism), not missing features ⇒ a
FOCUSED lever: Texel-style tuning to game outcomes + subtracting the systematic over-read + plugging autopsy gaps.
Strengths to preserve: high depth (d12-14 vs Mediocre d9-12) + fast eval. Blocker: over-optimism → collapses/drift.

## TOOLCHAIN BUILT (diagnostics/collapse_autopsy.py) — the plug-the-gaps loop
- `evaldrop <game_dir> [max_games] [drop_thresh] [sf_movetime] [window]` — SF-localized gap corpus over ALL games
  (our-eval-drop finds region → SF-swing in lookback window finds TRUE blunder). Emits histogram + autopsy_corpus.csv.
- `probe <corpus.csv> KNOB=VAL ...` — test a candidate setting against the gap corpus: per category, how many gaps
  now recover SF's move. DETERMINISTIC (no game noise). THE loop: autopsy → probe candidates → keep recoverers →
  games-confirm at a sensitive venue.
- `run <collapses.csv> <game_dir>` — collapse-window variant. `worker` — internal per-config engine pass.

## OVERNIGHT PLAN (autonomous, running)
- [RUNNING] Big Mediocre mine: 300 games conc3 baseline lightning (tag mediocre_mine) → tight baseline + big corpus.
- [THEN] Refined SF-localized `evaldrop` autopsy on the 300-game corpus → robust mechanism distribution + corpus.
- [THEN] Examine eval_misrank positions; `probe` candidate eval settings against them (deterministic).
- [IF TIME] chain a 2nd mine (seed 1) for more corpus; establish a ~50% sensitive SF venue (us vs SF graded UCI_Elo).
- HOLD the self-play eval-lever gate queue (aggression confound suspected; validate vs real opponent instead).

## RESULTS LOG (updated through the night)
- **mediocre_mine 300g baseline = 14.9%** (298 decided, 36 collapses, 145 min conc3). Tight CI (SE~±2%) ⇒
  real strength ~2000 (gap ~303 Elo to Mediocre 2319). Confirms the 40-game 13.8%. Yardstick SOLID.
- **Refined SF-localized autopsy, n=290 gap positions (300g corpus) — ROBUST distribution:**
  drift 109 (38%) / eval_misrank 73 (25%) / calibration_matched 36 (12%) / rfp 21 / lmr 20 / razoring 9 /
  horizon 7 / futility 6 / lmp 5 / null 4. ⇒ **~75% directly eval; RFP+razoring+futility (~12%) are shallow
  static-eval prunes firing on the inflated eval = eval-driven too ⇒ ~87% eval-related. Only ~10% genuine
  search (LMR/LMP/null), 2% horizon.** EVAL CALIBRATION is THE lever, mechanism-confirmed at scale.
  Corpus: selfplay/games/mediocre_mine/autopsy_corpus.csv (fen, base_move, sf_best, category). 73 eval_misrank
  positions are the deterministic PROBE targets; the 109 drift = systemic calibration (Texel-to-outcomes).
- **Probe batch 1 (4 dampers vs the 73 eval_misrank gaps) — NO quick knob fix:** NPEDGE_DAMP 1/73 (34 changed),
  ENDGAME_SCALE 2/73 (31 changed), SCALE_CAPTURE_GAINS=70 0/73 (0 changed — no effect on PV), REALIZ 0/73 (0
  changed). NPEDGE perturbs 34 mis-rank moves but to OTHER wrong moves (noise, not a fix). ⇒ the eval mis-
  calibration is DIFFUSE (not one over-valued term) ⇒ the lever is FULL TEXEL-TO-OUTCOMES recalibration, not a
  knob. [batch 2 running: latent-threat / imbalance / combined global damp — testing if global de-optimism helps.]
- **Probe batch 2 (latent-threat / imbalance / COMBINED global damp):** combined (NPEDGE+ENDGAME+IMBALANCE+LATENT)
  changed 41/73 mis-rank moves but recovered only 1 to SF's ⇒ mis-rank is NOT a global-scale problem; damping
  moves us to OTHER wrong moves. **BUT `IMBALANCE_SCALE=1` recovered 10/21 RFP-prune gaps** (18 changed there) —
  over-valued imbalance inflates static eval → RFP over-fires → prunes the refutation; reduce it and RFP behaves.
  ⇒ DECOMPOSITION: damping helps the "eval feeds bad prunes" pathway (RFP ~7%) but NOT the "eval mis-ranks moves"
  pathway (~25%) nor drift (38%). CAVEAT: "recover SF's EXACT move" is a high bar (SF=NNUE); mis-rank positions are
  real ≥120cp blunders but SF-move-match may overstate HCE fixability.
- **SMALL games-confirmable candidate:** IMBALANCE_SCALE=1 (recovers 10/21 RFP gaps by de-inflating the eval RFP
  trusts). Modest (RFP bucket ~7%). Gate at a SENSITIVE real venue, not self-play/Mediocre@15%. (Imbalance was a
  "closed" eval-feature — this is corpus-mechanism evidence, not Elo; confirm with games.)
- **KEY TAKEAWAY for next session:** mechanism = eval over-optimism (~87%), and it's DIFFUSE (no single knob) ⇒
  the highest-value next lever is a proper eval recalibration fit to GAME OUTCOMES (Texel), using the Mediocre
  games as data. Search machinery is mostly fine (~10%). The autopsy corpus + probe loop is the reusable bench
  to verify any recalibration REDUCES the misrank/drift buckets before spending games.

## SENSITIVE-VENUE SCOUT (fixed-time, us LIGHTNING vs SF native-ELF, --sf-elo caps SF)
Goal: a ~50% real-opponent venue to games-confirm candidates (Mediocre@15% too insensitive; self-play suspect).
NOTE SF's UCI_Elo limiter is NOT CCRL-calibrated (plays ~150-350 below nominal). Ladder so far (40g each, seed 0):
- SF@UCI_2000: **70.0%** (us stronger ⇒ SF@2000-UCI ≈ ~1850 CCRL; too weak to gate).
- SF@UCI_2400: **53.8%** ⇒ ~50% SENSITIVE VENUE FOUND. This is the gating venue (at ~54%, a +20 Elo change
  moves the score ~+3% — detectable in a few hundred games). 14 collapses (vs 8 at SF@2000 — stronger opp exposes
  more collapses).
- Mediocre 2319 (real engine, not UCI-capped): 14.9%.
STRENGTH BRACKET: us between SF@2000-UCI (~1850) and Mediocre 2319, ~2000. **GATE candidates at SF@UCI_2400.**
[Running: 200-game SF@2400 baseline → tight reference + CROSS-OPPONENT autopsy corpus to test if eval-dominant
mechanism holds vs SF-style too (opponent-robustness of the calibration conclusion).] (Command pattern: vs_sf.py --sf-path SF18
--sf-arb-path SF18 --sf-elo <N> --sf-movetime 1.0 --games <n> --concurrency 2, tag sfelo<N>.)

## ⭐ HOW TO ESCAPE THE "eval peaked" LOOP (the crux — 2026-07-16)
RECURRING ERROR across sessions: try a single eval hole-plug → quick test reads DOWN → conclude "eval peaked".
THIS INFERENCE IS INVALID. A global optimum and a BAD LOCAL OPTIMUM have the IDENTICAL signature under single-
lever tests (every single-coordinate move reads neutral/down in BOTH). So single-lever testing CANNOT distinguish
"optimal" from "stuck in a bad basin". We are demonstrably the latter (~2000 vs a 2319 hobby HCE = massive
headroom). So "peaked" was never valid — it was the tool being blind to the distinction.
WHY hole-plugging goes down: the eval is a COUPLED system; the probe proved the miscalibration is DIFFUSE (no
single knob fixes the 73 misrank gaps). Changing one term breaks the equilibrium the other terms compensated for
→ worse locally even when the term is better. Not resistance; just the math of moving one coordinate.
THE ESCAPE (the only method that leaves a local optimum): JOINT optimization — move MANY weights at once.
  1. TEXEL tuning to GAME OUTCOMES: fit ALL eval weights simultaneously to predict win/draw/loss over a big game
     dataset (the Mediocre + SF@2400 games are the labeled data). Project has partial infra (tune_fit, outcome-
     Texel). WE HAVE NEVER RUN A JOINT EVAL FIT ⇒ zero real evidence of peaking; this is THE missing experiment.
  2. VALIDATE on the deterministic autopsy corpus FIRST (does the retune REDUCE misrank+drift counts? noise-free),
     THEN gate at SF@UCI_2400 (~50%, sensitive, real) to SPRT significance. Never a quick biased test again.
  3. BAN "peaked" from single-lever results. Only valid peaking evidence = a JOINT fit converges + can't improve
     the corpus or the SF@2400 score. Single-knob regressions are NOT ceiling evidence — wrong tool.

## 2026-07-16 EXECUTION (Fable-directed: bugs-first + Texel; ACCUMULATE)
- **Fable memo** (pressure-test) + **Mediocre-gap diagnostic** (LEARNABLE 71/224=32%, not cleanly concentrated)
  agreed: skip concept-triage, run the never-done FULL outcome-Texel, hold NNUE behind the tripwire, FIX bugs
  first, and ACCUMULATE (bundle branch) — "we never bank" is the biggest process leak.
- **Mobility at SF@2400 (200g) = 36.2% vs ~54% baseline ⇒ self-play +20 does NOT transfer; NEGATIVE at the real
  venue. DO NOT bank mobility.** (Matched 200g baseline running to pin the magnitude.) Self-play gating confirmed
  misleading at the sensitive venue.
- **Bug knobs implemented + byte-id 247 verified** (ROOT_RAZOR_CONTINUE, RESIGN_THRESHOLD, ENABLE_TT_STORE_DRAW —
  all default-off byte-identical, search_engine.{h,cpp}:~224/875/1803/2197 + cache_management.h:888/945).
  DETERMINISTIC verdicts: **ROOT_RAZOR_CONTINUE = INERT** (changed 0 moves WAC + corpus ⇒ the root-razor break
  isn't dropping moves in practice; drop). **TT_STORE_DRAW = marginal** (+2 WAC solves / +0.4% nodes, but corpus
  wash — 33 misrank changed, 1 recovered). ⇒ **Fable's "free-Elo bug list" is NOT materializing deterministically
  — reinforces EVAL/Texel as the real lever.** RESIGN leak = the one needing a games test (queued).
- **+27 eval-inflation lead DISPROVEN** (eval_breakdown): the +27 was our SEARCH finding a REAL win (SF_search=mate);
  our STATIC eval was PESSIMISTIC there (−3.48 vs SF +0.89, a −4.4 gap). Collapses = CONVERSION failures of real
  advantages (search/soft-depth), NOT eval inflation. (Supports Fable's "87%-eval is method-biased" caution.)
- **Texel feature-export scoped:** EvalBreakdown exposes per-piece-type placement (pt_pawns..pt_kings) + engine has
  SCALE_PLACE_* knobs ⇒ MIDDLE-fidelity per-piece-placement fit achievable NOW (add pt_* to corpus, fit SCALE_PLACE_*
  + term scales, fitted-K). FULL per-cell PST needs the constexpr PST tables made FILE-LOADABLE (the big build).
  Coarse term-scale fit was NULL (self-play data + too coarse); outcome_texel.py + fitted-K fitter built + validated.
- **NEXT:** matched baseline (running) → resign games-test → MIDDLE-fidelity Texel (per-piece placement + SF-WDL or
  vs-opp labels to escape the self-play-symmetry null) → if held-out signal, FULL per-cell PST build.

## COMMITTED DIRECTION (user decision, end of 2026-07-16) + WHY (the "what did Mediocre know" answer)
STOP whiteboarding features. Use DATA-BASED PERTURBATION over ALL collapse/loss data to find "strong balances we
don't currently have." = outcome-Texel: perturb eval WEIGHTS to best predict game OUTCOMES, collapse/loss positions
UPWEIGHTED, fitted-K + Δw-reg, held-out logloss kill-gate → SF@2400 games. Why the hobbyists (Mediocre 2319) beat
us WITHOUT servers: Texel tuning is CHEAP (offline logistic fit, laptop-seconds) — never needed servers; they
(a) started from COMMUNITY-PROVEN piece values + PSTs, (b) kept the eval SIMPLE (tunable), (c) executed Texel
CORRECTLY (quiet positions, fitted-K, outcome labels), (d) ACCUMULATED over years. We have (a) at the material
level (piece values are STANDARD — checked: P1000/N3150/B3250/R5000/Q9000) but our PSTs are hand-picked (they
don't cause the +27 blowups — those are real search wins squandered — but they MIS-WEIGHT MOVE SELECTION = the
mis-rank bucket) and our eval SPRAWLS (bespoke latent_threat/capture_gains/piece_value_boost = hard to tune).
Missing = correct offline PST/weight fit + simplicity + discipline, NOT cleverness/compute.

## STAGED CODE (uncommitted, byte-id 247 default) + RESUME STEPS
- 3 bug knobs in search_engine.{h,cpp}+cache_management.h (ROOT_RAZOR_CONTINUE=INERT/drop, RESIGN_THRESHOLD=untested,
  ENABLE_TT_STORE_DRAW=marginal) — all default byte-identical (verified 247/41,479,610).
- tune_corpus.py: pt_pawns..pt_kings added to TERMS (placement features). outcome_texel.py: pt_* added to fittable
  (pieces PINNED so pt_* adds only the placement DELTA), fitted-K + Δw-reg + by-game holdout built.
- vs_sf.py: writes results.csv per run now (outcome labels for the fit); --opponent-raw drives Mediocre.
- WAS RUNNING at close: matched 200g SF@2400 baseline (task baukj9uai, tag sfelo2400_base200) — READ ITS .output
  on resume for the mobility-transfer magnitude + tight baseline + (it has results.csv = clean vs-opponent fit data).
- RESUME: (1) read baseline result; (2) build a vs-OPPONENT / collapse-weighted corpus (the self-play corpus was
  NULL from symmetry) — reuse sfelo2400_base200 (flat jsonls + results.csv) + collapses.csv failures + SF-WDL labels;
  needs a flat-vs_sf-format corpus reader (tune_corpus only reads nested tournament format); (3) run outcome_texel
  over term+pt_* scales, fitted-K; KILL-GATE = held-out logloss must move (coarse self-play fit was flat = null);
  (4) if signal → apply SCALE_*/SCALE_PLACE_* knobs, validate on autopsy corpus (misrank/drift drop) → SF@2400 SPRT;
  (5) if held-out STILL flat → the linear basis is dry → FULL per-cell PST (make constexpr PSTs file-loadable) or NNUE.
- ACCUMULATE: keep every SF@2400-validated survivor ON in a bundle; gate the bundle, not single levers.

## 2026-07-16 (window 2) — FIRST NON-NULL DATA-PERTURBATION SIGNAL
Built `selfplay/vs_opp_corpus.py` (flat-vs_sf reader: collapse-fens from collapses.csv + full games from
results.csv + result_white + pt_* placement features + weight col) and added weighted-logloss + pt_* + wider
K-grid(0.5-24)/trust to `outcome_texel.py`. **Smoke corpus (mediocre_mine + sfelo2400_mob collapses + sfelo2400_mob
results.csv full games, 1705 rows) = FIRST held-out-GENERALIZING fit** (HOLD logloss −0.022 at tr .25 / −0.004 at
tr .5 once K fit). ⇒ the self-play corpus was NULL from SYMMETRY; vs-OPPONENT data has the signal, as diagnosed.
TWO findings: (1) **K wants ~23 (grid-capped) = eval ~10x OVER-CONFIDENT** (over-optimism quantified; partly
inflated by the upweighted collapse tail — extreme evals that lost). (2) **ROBUST PST rebalance: pt_pawns −24%,
pt_queens +13%, pt_bishops +11%, pt_kings −8%** (maps to SCALE_PLACE_PAWN=76/QUEEN=113/BISHOP=111/KING_EG≈92; no
SCALE_PLACE_ROOK knob). Caveat: smoke corpus = collapse-weighted + mobility-ON games; the CLEAN baseline corpus
(sfelo2400_base200, mobility-OFF, balanced W/D/L) is running for the definitive re-fit. GROUND-TRUTH test running:
SCALE_PLACE rebalance on autopsy corpus (misrank/drift recovery) + WAC (tactics/speed). NEXT: clean re-fit on the
baseline corpus → apply SCALE_PLACE_* (+ consider a K-informed global eval downscale for the RFP-overfire/over-push
magnitude, then margin re-sweep) → validate autopsy corpus → SF@2400 SPRT → bundle if it holds. byte-id 247 default.

## GROUND-TRUTH VERDICT on the placement rebalance (window 2, decisive)
Applied the fitted SCALE_PLACE_PAWN=76/BISHOP=111/QUEEN=113 → autopsy corpus probe: **eval_misrank 0/73 recovered**
(changed 40 moves to OTHER wrong moves, like the dampers) + **WAC 244/300 (−3 tactics), EBF worse.** ⇒ the PST
rebalance does NOT convert to move-quality. **KEY LESSON: held-out logloss GENERALIZING is necessary-but-NOT-
sufficient** — the fit's gain was DOMINANTLY the K term (over-confidence), and the placement WEIGHTS add ~nothing
to move selection. Over-confidence is a MAGNITUDE problem (RFP over-fire, over-push/collapse, resign — all
magnitude-dependent), NOT a move-choice problem (argmax is scale-invariant → rescaling placement changes moves
without improving them). ⇒ **the real lever the data points to = GLOBAL EVAL DOWN-SCALE + margin re-sweep (untested),
NOT PST scaling. And any Texel candidate MUST pass the autopsy-corpus + WAC ground-truth, not just held-out logloss.**
RESUME PLAN (revised): (1) let sfelo2400_base200 finish → clean re-fit for an HONEST K (the smoke K~23 was inflated
by the upweighted collapse tail); (2) test a GLOBAL eval downscale toward K~2 (e.g. scale all eval by ~1/f) WITH the
eval-denominated margins re-swept (RFP_MARGIN/FUTILITY/etc), gate at SF@2400 — this targets the collapse/RFP-overfire
magnitude directly; (3) move-quality (misrank) likely needs finer per-cell PST or is search/soft-depth — separate lane.
byte-id 247 default; nothing committed; NEW files: vs_opp_corpus.py, collapse_smoke.csv.

### 🎯 CONDITION-DISCOVERY: term-attribution of the misranks (2026-07-16, `diagnostics/misrank_attribution.py`)
Per position where our search preferred the WRONG move over SF's, diff our per-term breakdown of the two
resulting positions (our POV) → which term drove the wrong choice. mean_signed>0 = term systematically favours
the losing move. Two buckets, SHIPPED default config (byte-id 247):
- **eval_misrank (n=72, sharp mistakes):** `capture_gains` **+574** (15 drivers), `material` +472 (17), `pieces`
  ~0 signed but huge |c|=763 (NOISE — swings both ways → confirms WHY the PST/pieces rebalance was net-null),
  piece_value_boost +147, pt_pawns +122, pt_kings +86.
- **drift_no_single_blunder (n=79, slow losses):** `material` **−262** (favours the RIGHT move here → NOT the
  drift culprit), capture_gains −51, pieces −87. The only terms POSITIVE in BOTH buckets: **pt_kings** (+86/+78),
  pt_knights (+48 drift).
**Readings:** (1) sharp misranks = **capture_gains over-credit** (bespoke, ON by default, prior [[capture-gains-overread]]
history) steering us into greedy/bad captures — the #1 data-pointed CONDITIONAL-CUT candidate; (2) slow drift =
**king-placement PST (pt_kings)** consistently mis-weights across both buckets — a specific PST cell, not the
noisy bulk. **CAVEAT:** material/capg favouring the non-SF move is PARTLY correct signal (SF sacrifices), so the
lever is the bespoke `capture_gains` bias (tunable, conditionable), NOT raw material (anchor, STANDARD). **Also:
threats/king_safety/mobility/pawn_struct/outpost = 0.0 in every misrank — they're DISABLED by default**, so the
"KS when opp has mobility" idea would be ADDING a new term, a separate move from fixing the capg/pt_kings bias
that's active today. Next: capg tension/SEE-conditioned cut (revive ENABLE_CAPG_COND lane) + pt_kings placement
probe → autopsy recovery + WAC → SF@2400.

**FALSIFIED (2026-07-16, probe bj05h3bwv): capg cut does NOT recover the sharp misranks.** NOTE FIRST that
`ENABLE_CAPG_COND` is ALREADY ON (shipped +15E) — capg is already tension-conditioned (10% quiet -> 100%
tactical), so the +574 attribution is the RESIDUAL in capture-tension positions the quiet-dampener can't reach.
Probe `CAPG_HI_SCALE=50` (cut the tactical ceiling) on the autopsy corpus: eval_misrank **1/73 recovered**
(changed 39 -> shuffled among OTHER wrong moves), AND damaged eval_calibration_matched_SF **36->24** (broke 12
already-correct). **SAME failure signature as the PST rebalance (0/73).** => **SECOND independent falsification
that reweighting an ACTIVE eval term can fix the sharp misranks.** The attribution finds terms *involved* in the
wrong choice, but no reweighting lands on SF's move — SF's move sits OUTSIDE our eval's expressed features (high
changed-count + ~0 recovery = many near-tied wrong candidates, right move not considered). Matches the old
"no single knob fixes the 73 gaps." **REDIRECT:** the missing signal is likely a term our eval does NOT express —
and `threats`/`mobility`/`king_safety` are all DISABLED (0.0 in attribution). Test whether ENABLING a grounded
off-term recovers misranks (condition-discovery for a NEW feature, not a reweight) — threats probe running
(by60zjff5). If an off-term recovers a chunk WITHOUT breaking the matched-SF/WAC set, THAT is the lever. If none
do, the sharp misranks are likely SEARCH/depth (SF sees deeper), not static eval — pivot to the continuation
harness to confirm outcome impact regardless.

### ✅ CONCLUSION: sharp misranks are SEARCH/depth, NOT static eval (3 independent micro-falsifications, 2026-07-16)
Autopsy-corpus recovery of the 73 eval_misrank gaps: PST rebalance **0/73**, capg cut (CAPG_HI_SCALE=50) **1/73**,
threats-enable (ENABLE_THREATS=1) **3/73** — and each DAMAGED the eval_calibration_matched_SF set (36→24/21). Three
different static-eval interventions (reweight active term / cut bespoke term / enable a disabled grounded term) all
fail to recover the sharp misranks AND break already-correct positions. ⇒ **SF's move sits outside what our static
eval can express by ANY term change; the sharp misranks are a search-depth gap.** STOP eval-tuning this bucket.
`diagnostics/misrank_attribution.py` (term-attribution) + collapse_autopsy probe are the tools; both banked.

### 🚨 RESIGN-LEAK — the live high-value lead (2026-07-16)
base200 results.csv tally: **125/200 games = "ours resigned", = 100% of our losses (125 zeros). ZERO natural
losses — we are NEVER checkmated, we resign every lost game.** So `RESIGN_THRESHOLD=-15000` is the SOLE gate on our
loss count. Honest K=24 (clean base200 fit) ⇒ eval wildly over-confident ⇒ `sigmoid(-15/24)≈0.35` naively says our
resign point is a ~35%-score position, NOT resign-worthy (CAVEAT: outcome labels are contaminated by the resign
behavior itself, so the fit confirms DIRECTION not magnitude — needs a play test). Score = 48W/27D/125L = 30.75%
(explains 30.8% << expected ~54%). **TEST RUNNING (b9gjnenu7): resign-OFF (RESIGN_THRESHOLD=-900000) 100g SF@2400
seed0, paired vs base200 game-for-game.** If score jumps materially, resign-leak is real and the FIRST concrete
score lever this whole campaign — then sweep RESIGN_THRESHOLD to a K-aware calibrated value (do NOT touch razoring/
speed; resign is a root-only check). If flat, resign is fine (user's prior intuition confirmed) and the 30.8% is
genuine conversion weakness → continuation harness for the collapses. byte-id 247; nothing committed.

### ✅ RESOLVED: resigns are FAIR — resign-leak is NOT the story (SF audit, `diagnostics/resign_audit.py`, 2026-07-16)
SF depth-18 eval (our POV) of the base200 resign positions: **89% lost (<=-600cp), median -965cp**, 1% holdable
(-300..-150), 10% "unclear (>-150)" (one +684 = likely an extraction artifact where the last logged ply was ours,
NOT a premature resign). => **We genuinely reach lost positions and resign them correctly; the 30.8% is REAL, not
inflated by resignation.** K=24 was CIRCULAR (labels contaminated by the resign behavior) — SF ground-truth
overrides it. Resign is at most a tiny <=10% tail worth a -600 threshold tighten (being measured by b9gjnenu7).
**The disease is UPSTREAM: we walk from winning (>=+2 peak) into lost = the 91 collapses.** OPEN QUESTION the user
raises: is that an EVAL problem (our optimism picks losing plans) or a SEARCH/horizon problem (we don't see the
cliff)? Today's 3 misrank falsifications say eval-WEIGHT reweighting can't fix move choice — so if it's eval it's
OPTIMISM/calibration (magnitude), not term weights. DISCRIMINATOR: run the `triage` sub on base200 collapses.csv
(re-search each collapse DECISION_FEN deep + SF compare → labels EVAL-hole vs HORIZON). That settles eval-vs-search
for the COLLAPSES specifically, on real thrown-away wins.

### 🎯 THE DIAGNOSIS CONVERGES: collapses = EVAL OPTIMISM at critical leaves (triage beuch7gnz, 2026-07-16)
Triage re-searched base200 collapse DECISION_FENs at depth 18 + SF compare: **13 EVAL / 3 HORIZON**, **mean
optimism gap +4.86 pawns** (max +21.9), and **9/13 EVAL rows play SF's OWN best move yet our eval reads ~+5 when
SF says ~0.** ⇒ NOT a horizon/depth problem (Mediocre beats us at LOWER depth = existence proof depth isn't the
bottleneck); it's the "correct leaf, WRONG eval VALUE" case. Our eval is systematically OPTIMISTIC in the losing
positions. Because RFP/futility/razoring key off the static eval, this same optimism ALSO makes pruning unsound
(prune the refutation because we "know" we're +5) — so eval-optimism is plausibly the COMMON ROOT of both the
value error AND the ~65 `prune:*` search-corruption cases. Fixing it is UPSTREAM of both.

### 🧭 THE PROGRAM (user's methodology, 2026-07-16): characterize eval optimism in CHESS terms → conditional fix
NOT reweighting (falsified 3x) and NOT a rescale (scale-invariant no-op). The lane: find WHICH chess situations
the eval is optimistic in + WHICH sub-terms over-fire, NAME them (overextension? over-rating advanced pawns?
ignoring enemy counterattack on our king?), then design CONDITIONAL de-optimism (detector-gated, fires only when
the pattern holds) validated against a CONTROL set of calm/accurate positions so the calibrated majority is not
dampened. This is the process that originally built the engine (handmade: find failures → dissect leaves → adjust
toward chess sense w/o breaking controls) — now automated + data-scaled. The one eval change that ever shipped
Elo (capg tension-conditioning +15) came from exactly this. TOOL BUILT: `diagnostics/optimism_diag.py` — measures
optimism = our_eval − SF_eval per position, explains it BY TERM (which components are largest in over-read
positions) AND BY named chess-feature detectors (adv_own_pawns, enemy_pressure_on_our_king, overextended_pieces,
we_sacked_material, passers, enemy_queen_on) with ON-vs-OFF optimism gaps. Output = ranked NAMED biases = the
worklist for conditional terms. Running on the collapse corpus (ba6ot3emb).

### ⚠️ SCALE-INVARIANCE TRAP in the down-scale lever (mapped 2026-07-16, before building it)
The eval-denominated margins are: `RFP_MARGIN` (search_engine.cpp:3467/4087, per-ply), `FUTILITY_MARGINS[]`
(constexpr {200,450,650,950}, used :2539/:2884), razor thresholds (LITERAL 750/300/2000/1500·pow(0.75,d-4),
:2061/2065/3213/3215), `RESIGN_THRESHOLD`, `ASPIRATION_DELTA`, `VERIFY_MARGIN`, and the harness `win_thresh`
(+2.0pawns adjudication). **Alpha/beta ARE eval values, so if you scale eval by f AND scale every one of these
margins by f, EVERY search comparison is preserved exactly — it is a pure change of units, a near-no-op.** The
only things that DON'T scale are the fixed mate band (±9,999,999 / the 9000000 cutoffs), the resign threshold,
and the external adjudication win_thresh. So a *proportional* "down-scale + proportional margin scale" would
move ~nothing except the mate/resign/adjudication boundaries. **For the lever to have a pruning effect, eval must
scale DOWN while the pruning margins are held fixed or re-tuned INDEPENDENTLY (a genuine re-sweep, NOT a
proportional rescale).** Then RFP/futility/razoring fire less aggressively relative to a now-less-confident eval
= the intended anti-overfire mechanism. Design the experiment as: (a) new `EVAL_DOWNSCALE` knob (int %, applied
to the final non-mate total, byte-id default 100) + (b) an INDEPENDENT re-sweep of RFP_MARGIN/FUTILITY/razor as
its own axis — do NOT couple them 1:1. Razor literals + FUTILITY_MARGINS[] would need to become knobs first
(they are constexpr today); mind the user's razoring/resign caution when touching them.

## DISCIPLINE REMINDERS
Small samples: do NOT over-conclude from 40-game Elo (SE~±5% at 13%). Prefer the DETERMINISTIC autopsy/probe corpus
loop for lever selection; reserve games (at a ~50% venue, run to significance) for final confirmation. byte-id 247
after any build; every feature env-gated default-off; commit only when asked; no commit footer. Dispatcher form only.
Temp files to delete at session end: diagnostics/_raw_uci_smoke.py? (that's selfplay/), selfplay/_raw_uci_smoke.py,
selfplay/_raw_uci_probe.py.
