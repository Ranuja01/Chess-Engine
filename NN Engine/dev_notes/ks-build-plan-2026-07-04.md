# King-safety rebuild — plan (2026-07-04)

Funnel execution #2, the BIGGEST lever (incremental validity: sf11_kingsafety +0.0031 by-game, ~2.5× threats).
KS is a SEPARATE granular function (king-directed; `latent_threat` was always meant to be this — user). MUST be
midgame-only phase-gated (data: +0.0051 midgame → +0.0005 endgame, collapses 10× — an active endgame king is
GOOD). Fable's move = RE-ADJUDICATE the existing gated machinery under the new gate+compass, NOT rebuild from
scratch (we already have the SF-form gated-superlinear KS).

## Existing machinery (search_engine.h) — DON'T rebuild, re-adjudicate
`KING_SAFETY_MAG` (0=off; beside-mode when >0, additive beside latent_threat) · `ENABLE_KS_REPLACE_LT` (replaces
latent_threat) · `KS_ZONE2` (wider zone) · `KS_DYN`/`KS_DYN_PIVOT`/`KS_DYN_SHIFT` (per-king dynamic magnitude via
mod_gain) · `KS_INTERACT` (super-linear coffin interaction) · danger = min(units,KS_CAP)²/KS_DIVISOR (the
superlinear curve). `king_safety` is a corpus TERM column + `g_eval_breakdown.king_safety`. From ks-detection-
rebuild: the detector was correct (ksattack +70) but died at lightning — under the BROKEN proxy + worst-TC. Now
re-adjudicated under OUTCOME compass + calibrated node_ab + blitz venue.

## ✅ CHEAP-FIRST SCREEN RESULT (2026-07-04) — existing KS machinery is a REAL LEVER, 23% of ceiling
Patched with `KING_SAFETY_MAG=4000 KS_ZONE2=1 KS_DYN=128` (beside), screened `king_safety` by-game:
**our_king_safety +0.000688 (REAL LEVER, sign-stable) vs sf11_kingsafety ceiling +0.002984 = ~23% captured.**
- KEY: our existing KS machinery IS a real outcome lever — NOT dead. Vindicates Fable's "re-adjudicate not
  rebuild" + the compass thesis: the "5 KS deaths" were the broken proxy (SF-agreement + lightning-worst-TC),
  not KS being worthless. Measured vs OUTCOMES, KS carries real signal.
- BIG HEADROOM: 23% captured → ~+0.0023 still on the table (larger than the whole threats lever). KS build = the
  threats loop on the biggest prize: improve the KS function toward SF king-danger, guided by the gap, ~8-min
  patch/screen iterations, midgame-only. Config-sweep first (KS_INTERACT / ENABLE_KS_REPLACE_LT / KS_DYN variants
  — cheap, no rebuild) to find the best-capturing existing config, THEN improve the function if needed.

## CHEAP-FIRST (done) — screen the existing machinery
Patch corpus with `KING_SAFETY_MAG=4000 KS_ZONE2=1 KS_DYN=128` (beside-mode) → populates `king_safety` → run
`incremental_validity.py <corpus> king_safety` (tools now generalized to any term). Tells us how much of the
+0.0031 SF ceiling our EXISTING machinery captures. If high → just tune+gate (cheap). If low → improve, guided by
the gap (the threats loop: fast patcher, ~8-min iterations).

## ✅ KS GAP DIAGNOSIS (2026-07-04) — coverage + under-scaling (mirrors threats)
On ks_corpus.csv (our king_safety = KING_SAFETY_MAG=4000 KS_ZONE2=1 KS_DYN=128 vs sf11_kingsafety):
- corr 0.51 all / **0.62 midgame** (track big dangers, less the diffuse ones).
- **our nonzero 9,145 vs SF 30,181; SF fires where we read 0 on 22,181 positions** = huge COVERAGE gap.
- where both fire, we UNDER-score: mean |1.26| vs SF |2.13| (0.6×).
- MECHANISM: `king_safety_danger` (cpp_bitboard.cpp:4961) HAS static components (KS_SHIELD/KS_OPEN_FILE/KS_WEAK/
  KS_STORM at :5001-5032) but combines them into `units`, then `danger = min(units,KS_CAP)²/KS_DIVISOR` → **0 when
  net units ≤ 0** (a sheltered king w/ no active attackers = 0). SF assigns SOME king-safety to nearly every
  midgame king → we miss the diffuse baseline.
- TWO IMPROVEMENT LEVERS (both keep midgame-only): (1) **config re-balance** of the KS component knobs (KS_ATT_*,
  KS_WEAK, KS_OPEN_FILE, KS_SHIELD, KS_STORM, KS_DIVISOR) via SPSA/sweep screened by incremental_validity — cheap,
  NO rebuild, improves SHAPE where we fire (won't add baseline coverage but may lift the under-scaling + relative
  balance). (2) **function broaden** — give a small nonzero baseline danger for static king-weakness even w/o
  active attackers (the coverage fix; C++, gated, byte-id-safe). Do (1) first (cheap), then (2) if the gap persists.
  Expect multi-iteration like threats (3 rounds: 25%→32%→52%).

## CONFIG SWEEP RESULTS (2026-07-04, screened by incremental_validity, magnitude-invariant)
- base (MAG=4000 ZONE2 DYN=128): our_king_safety **+0.00069** (23%).
- +static-boost (OPEN_FILE=6 WEAK=5 STORM=3 SHIELD=1): **+0.00066** = NO HELP → broadening coverage of quiet-
  exposed kings is NOISE, not signal. The +0.0031 KS signal is NOT diffuse coverage; it's ACCURATELY scoring the
  REAL king attacks (where we're corr 0.62 + under-scaled). ⇒ the lever is the SEVERITY SHAPE for genuinely-
  attacked kings, not coverage.
- +KS_INTERACT=8 (super-linear multi-attacker coordination): **+0.00055 = slightly WORSE.**
- **CONCLUSION: config re-balance CANNOT lift our existing KS past ~23% capture** (static-boost no-help,
  KS_INTERACT worse). The +0.0031 KS signal is in ACCURATELY scoring real king attacks (corr 0.62, under-scaled),
  which our static machinery structurally under-represents — no knob fixes it.
- **TWO PATHS from here (decision):** (1) a genuine KS FUNCTION improvement (better real-attack severity model —
  substantial gated C++, and UNCERTAIN it can close much given the dynamic nature of king attacks); (2) recognize
  ~23% as the quantified cheap-HCE ceiling for the KS-vs-outcome signal → the remaining ~77% (accurate dynamic
  king-attack scoring) is the NNUE lever. **This is the rigorous, quantified version of the original collapse
  finding** (dynamic king-attack over-read) — now measured on the outcome compass, not asserted. Fable's tripwire:
  "the tail residual after KS/threats is the NNUE evidence" — threats captured 52%, KS existing 23% config-capped.
  A KS function-improvement attempt would test path (1); if it also plateaus, path (2) is the evidence-backed
  NNUE go. Immediate-threats (+0.00063) remains a real banked HCE win regardless.

## ✅ STAGE 0 DE-CONFOUND RESULT (2026-07-04, replace-mode, plan-approved) — clean number + screen noise floor
Re-screened the EXISTING KS machinery in **REPLACE mode** (`ENABLE_KS_REPLACE_LT=1 KING_SAFETY_MAG=4000 KS_ZONE2=1
KS_DYN=128`) via the fast patcher — no C++. In replace mode `our_total` excludes latent, so incremental_validity
measures KS over a baseline with NEITHER latent NOR KS (the cleanest isolation). Corpus `ks_replace_corpus.csv`
(36,965 rows / 7,995 games, by-game 4-fold):
- **our_king_safety +0.000945 ±0.00012 (sign-stable, REAL LEVER) = ~29% of the SF ceiling (+0.003308).**
- vs the CONFOUNDED beside-mode +0.000688 (~23%) → **the latent double-count WAS a real confound; removing it lifts
  23%→29%.** But NOT ≫23% → the KS lever is still ~71% uncaptured. Remaining suspects: (a) flat +185/+75 shelter
  still in `pieces` (Stage 1), (b) genuine function under-capture (Stage 3).
- **SCREEN NOISE FLOOR (key):** config sweep KS_SAFE_CHECK=3→10 moved our_king_safety +0.000945→+0.001054 (~32%) —
  but fold-std ≈ 0.00018, so that +0.0001 is **within ~0.5σ = NOISE.** ⇒ the by-game screen CANNOT resolve knob-
  shape gains at this granularity; hill-climbing it would p-hack the holdout. Cheap knob-shape is noise-limited
  (confirms the prior "caps ~23–32%" finding in clean replace-mode). Further capture needs STRUCTURAL steps
  (Stage 1 de-confound, Stage 3 enrichment) validated at the node_ab/SPRT GATE, not the screen.
- **NEXT-ACTION FORK (real Elo time):** (B, recommended) fit magnitude (tune_fit --target result) on the CURRENT
  replace-mode KS → node_ab + blitz SPRT = cheapest real-Elo signal on the biggest lever, ZERO new C++, tests the
  compass→Elo link that immediate-threats was too small to test; (A) build gated KS_CONSOLIDATE (Stage 1) first;
  (C) Stage 3 enrichment first. Artifacts: ks_replace_corpus.csv, ks_sc10_corpus.csv.

## ✅ STAGE 1 CONSOLIDATION RESULT (2026-07-04) — de-dup done, byte-id-safe, gap is REAL (not confound)
Built the gated `KS_CONSOLIDATE` flag (search_engine.h/.cpp + 4 gated `+185/+75` constants in evaluate_kings_midgame;
byte-id **245/39,146,294** VERIFIED gate-off). Re-screened replace-mode + `KS_CONSOLIDATE=1` (flat shelter removed
from `pieces`), corpus `ks_consol_corpus.csv`:
- **our_king_safety +0.001133 ±0.00013 (sign-stable) — up from replace-only +0.000945 (≈1.5σ, a REAL move).** The
  flat shelter WAS partly redundant with KS_SHIELD → removing it let KS absorb more signal. Consolidation validated.
- BUT baseline loss ROSE 0.13285→0.13336 (flat shelter carried real outcome signal) and the SF ceiling rose in step
  (+0.003308→+0.003735) → **capture RATIO stuck ~29→30%.** The flat shelter wasn't noise; it was crude-but-real KS.
- **VERDICT: the ~70% uncaptured king-safety signal is a genuine FUNCTION/REPRESENTATION gap, NOT a measurement
  confound.** De-confounding (latent + flat-shelter) moved absolute capture 23%→30% but the ratio ceiling is a real
  function limit. Confirms plan hypothesis (b). Next: does the CLEAN consolidated KS gain ELO at the gate (the
  compass→Elo test, now on a non-double-counted config)? → fit magnitude (tune_fit --target result on king_safety) →
  node_ab + blitz SPRT. If Elo-positive → KS bankable now, enrichment additive. If flat despite +0.00113 signal →
  the critical screen-positive-but-Elo-flat datapoint (leaf-noise/NPS/magnitude) before investing in Stage 3.
  Artifacts: ks_consol_corpus.csv; gate `KS_CONSOLIDATE` (default-off, byte-id).

## ✅ GATE ATTEMPT + DIAGNOSTICS (2026-07-04) — eval is GOOD; the −50 is magnitude/coupling, not eval-quality
Fitted MAG (tune_fit --target result on king_safety, consolidated corpus) = 0.62 → **MAG≈2500**. Lightning SPRT of
the consolidated replace-mode KS @ MAG=2500 vs base = **~−50 Elo** (stopped early at ~60 games; W19-L28). Landed near
the historical −47/−48 — BUT the mechanism is different (prior deaths were CONFOUNDED: scattered/double-counted KS
beside a live latent). Ran diagnostics to decompose the −50:
- **latent_threat vs KS incremental validity (same base):** latent **+0.000247 (7%, MARGINAL)** vs our KS
  **+0.000945 (29%, REAL)**. ⇒ **KS is a 4× BETTER king-safety representation than latent** (user's architecture
  intuition — dynamic board-state scaling > static THREAT_* scalars — CONFIRMED on the outcome compass). Replace
  mode is architecturally RIGHT (drop the near-dead latent).
- **NPS (wac_timed, depth-10):** BASE 504,137 nps / 39.1M nodes vs REPLACE-KS **520,045 nps (+3.2%) / 36.7M nodes
  (−6%)**. ⇒ **latent is the SLOWER term (user right); replace-KS is FASTER and prunes MORE → deeper at equal time.**
  The NPS-tax explanation for the −50 is DEAD — the loss happens DESPITE a speed advantage. (WAC 245→241 = benign
  tactical trade for a positional term.)
- ⇒ **The −50 is decision-quality-at-the-tuned-search: hot magnitude + the eval↔search COUPLING** ([[eval-search-
  coupling-flat-candidates]]). User's key insight: the search margins (RFP/futility/null — prune on eval VALUES) were
  tuned to the CRUDE eval's error profile = a joint local max; a better eval walks off the ridge until search is
  re-tuned. **This reframes ALL 5 prior KS deaths as possibly COUPLING artifacts (frozen-search), not eval failures.**
- **STS magnitude compass (fixed-depth, diagnostic-only — the −202 proxy caveat stands):** BASE 50.4% | MAG600 48.4
  | **MAG1000 51.7% (+1.3, PEAK)** | MAG1500 49.3 | MAG2500 50.0 (neutral). ⇒ **play-optimum ≈ MAG1000, ~2.5× gentler
  than the outcome-fit 2500.** Outcome-fit ≠ play-optimum = the KS signal is more outcome-predictive than search can
  exploit at that weight (a coupling/masking fingerprint). The −50 tested a play-HOT magnitude.
- **RUNNING:** lightning SPRT of MAG=1000 (the play-optimum) vs base — the clean before/after vs the −50@2500. If ≫
  −50 → magnitude was the culprit (KS converts with play-calibration). If still ≤0 → escalate to the SCOPED eval-
  coupled margin re-sweep (RFP/futility/null with KS on, gate the JOINT config @ blitz) = the direct coupling test.
- **Method note:** for THIS term (faster + prunes more) the EQUAL-TIME venue is correct (rewards the speed win);
  node_ab (equal-node) would UNDERSTATE it. Lightning = fast equal-time but KS's worst; blitz = fair. Artifacts:
  ks_replace_corpus, ks_consol_corpus, ks_sc10_corpus; STS tags ks_sts_*.

## ✅ COUPLING RE-SWEEP DESIGN (2026-07-04, scoped, ready to run when cores free)
MAG=1000 (STS-optimal, +1.3% equal-depth) trended ~−95 at lightning (stopped early at 15 games to free cores for
the user). Equal-depth-POSITIVE + equal-time-NEGATIVE = the eval↔search COUPLING signature (user's hypothesis). The
scoped re-sweep to test/fix it:
- **VERIFIED coupled-prune surface** (search_engine.cpp/.h): **RFP is the crux** — prunes on `rfp_static_eval +
  RFP_MARGIN*depth_remaining <= alpha` with `RFP_EVAL_MODE=0` = FULL eval, so KS is baked into every RFP decision.
  Env-sweepable: `RFP_MARGIN=1500` (per-ply, milli-pawn), `RFP_MIN_DEPTH=1`, `RFP_MAX_DEPTH=6`. Futility
  (`FUTILITY_MARGINS={200,450,650,950}`) + razoring (hardcoded 750/300, 2000/1500) are `constexpr` = NOT env-
  sweepable (defer; would need a code change). Null-move not eval-gated by default (`ENABLE_NULL_EVAL_GATE=false`).
- **MECHANISM:** KS enlarges static-eval swings in king-attack positions; RFP prunes on that eval; `RFP_MARGIN=1500`
  was calibrated to the OLD eval's distribution → it mis-prunes exactly the KS-relevant lines. The frozen local max.
- **RE-SWEEP (scoped, 2–3 knobs):** joint **`KING_SAFETY_MAG` × `RFP_MARGIN`** (± `RFP_MAX_DEPTH`), KS on
  (`KS_CONSOLIDATE=1 ENABLE_KS_REPLACE_LT=1 KS_ZONE2=1 KS_DYN=128`). Two run options:
  (a) search-lane SPSA at lightning-equal-time (`spsa` sub, spec over MAG+RFP_MARGIN) — the proper coupled venue; or
  (b) a small 3×3 grid (MAG∈{800,1200,1800} × RFP_MARGIN∈{1500,2200,3000}) via `gate`/`node_ab`, cheaper/clearer.
  Start gentle: KS makes eval hotter in king lines → RFP likely needs a BIGGER margin (prune LESS there) — sweep
  RFP_MARGIN UP from 1500.
- **HYPOTHESIS:** a (MAG, RFP_MARGIN) joint point beats base where MAG-alone (frozen RFP) didn't = the coupling
  local-max the user predicted. It also BANKS EBF (RFP re-calibrated to the cleaner+faster eval → prune harder
  elsewhere → lower EBF → deeper at equal time = the flywheel).
- **DISCIPLINE:** gate the JOINT config vs base (net is truth; eval-vs-search attribution is secondary); by-game
  holdout / blitz-gate the winner (overfit guard); keep it 2–3 knobs (don't kitchen-sink the whole margin family).
- **ALSO WORTH:** measure equal-depth positional gap-closure vs SF11 (`vs_sf11`/`pvb_delta`) for KS-on — the clean
  eval-decision metric (user: at fixed depth SF crushes us positionally, not tactically → the eval reservoir). STS
  +1.3% is a first sign; the vs-SF positional delta is the fuller compass. (Needs engine+SF → run when cores free.)

## ⚠️ STS COUPLING COMPASS = INCONCLUSIVE (2026-07-04, single-core) — proxy hit its noise floor
Fixed-depth STS, base vs KS(MAG=1200) × RFP_MARGIN{1500,2200,3000} (sub `sts_coupling`):
| | RFP1500 | RFP2200 | RFP3000 |
|---|---|---|---|
| base | 50.4 | 50.9 | 50.4 |
| ks   | 50.6 | 49.1 | 51.0 |
base FLAT (±0.5 = noise); ks ZIGZAGS 50.6→49.1→51.0 (1.9% p2p, non-monotonic = noise fingerprint). Weak +0.4
interaction hint (ks best at RFP3000) but the 49.1 mid-dip kills trust. **The naive "relax RFP unlocks KS" idea did
NOT show; and STS can't resolve conversion-scale effects (~0.5–1%) through its noise.** ⇒ single-core PROXIES are
EXHAUSTED for the KS conversion question (representation is proven good on the OUTCOME compass; conversion effects
are game-scale). Remaining KS steps ALL need games (multi-core): (1) joint MAG×RFP_MARGIN re-sweep (the real
coupling test — explore BOTH RFP directions, don't assume relax), (2) play-optimal MAG at a FAIR venue (blitz), (3)
equal-depth positional gap-closure vs SF11 (needs SF). DEFER to when cores free; do NOT over-invest in the STS proxy
(the −202 lesson). Sub `sts_coupling` added to overnight_runner.sh (baked, permission-clean, reusable).

## 🌙 OVERNIGHT RUN (2026-07-04→05, autonomous, user asleep till ~11am) — the fair-venue conversion test
Question: does the consolidated KS (better+faster rep) CONVERT to Elo at a FAIR venue? All prior tests (−50/−95)
were at LIGHTNING (KS's worst venue) + hot magnitude. Overnight battery (byte-id 245 preflight PASSED):
- **`ks_nab_battery 40 250000 4`** (new baked sub): each KS config vs base at 250k fixed nodes (~d12, deeper than
  lightning's depth-bias zone; jitter-free; conservative — equal-node ignores KS's +3.2% speed win). Grid (common:
  `KS_CONSOLIDATE=1 ENABLE_KS_REPLACE_LT=1 KS_ZONE2=1 KS_DYN=128`): MAG∈{1000,1500,2000} + MAG1000×RFP_MARGIN∈{2200,
  1000} (the coupling directions — STS compass couldn't resolve, so test both in games). ~40min each, ~3.5h total.
- **BRANCH:** any config clearly + → `gate_blitz '<cfg>' <lbl>` (BLITZ SPRT, the true fair venue, elo0=-3/elo1=0 =
  "is it non-losing"). If NONE convert even at deep+conservative node_ab → decisive: KS-addition doesn't convert →
  pivot to the bank-what-we-earned audit (capg/threats gated status). Either way the night yields a verdict.
- New subs in overnight_runner.sh: `ks_nab_battery`, `gate_blitz` (both baked/permission-clean). Discipline: only
  dispatcher launches (auto-approved), results via file-read, bounded runs (no hang), sequential (no OOM/pkill), no
  rebuilds. Interpretation guard: node_ab over-credits KS less at 250k than 150k but still somewhat (depth-bias) AND
  under-credits it (ignores speed) → treat a node_ab + as ENCOURAGING (needs blitz confirm), a clear − as decisive.

## ✅✅ DEEP-NODE BATTERY RESULT (2026-07-05, overnight) — VENUE was the catastrophe; KS ~neutral at fair depth
node_ab @250k nodes (~d12), ~230 games each (±53 Elo CI), KS(consolidated+replace) vs base:
| config | Elo |
|---|---|
| ks1000 (MAG1000, RFP default 1500) | −12.1 ±52.8 |
| ks1500 | −21.7 ±51.5 |
| ks2000 | −7.7 ±53.1 |
| ks1000 + RFP2200 (looser) | −4.6 ±53.2 |
| **ks1000 + RFP1000 (tighter) — BEST** | **−0.0 ±52.8** |
- **HUGE:** at fair depth KS is ~NEUTRAL, NOT the −50/−95 lightning disaster. **Venue/depth explained the
  catastrophe** — vindicates the coupling/venue hypothesis + "we measured wrong." All 5 within noise of 0.
- **FLYWHEEL SIGNAL:** the BEST config TIGHTENED RFP (1000, prune MORE) — the cleaner+faster KS eval lets pruning
  cut harder (the eval→EBF flywheel, user's thesis). Noise-level (±53) but right-direction + battery-best. RFP=2200
  (looser, my initial guess) was NOT best → the STS "KS likes looser RFP" hint was wrong; games say TIGHTER.
- **CAVEAT:** node_ab is CONSERVATIVE (equal-node ignores KS's +3.2% speed win) → true equal-time (blitz) should be
  a few Elo BETTER than these. And ks1000rfp1000 confounds KS with a global RFP change (needs a base+RFP1000 control
  to attribute — a follow-up; the BUNDLE-vs-base is what ships regardless).
- **DECISION:** blitz-confirm the flywheel bundle `KS_CONSOLIDATE=1 ENABLE_KS_REPLACE_LT=1 KS_ZONE2=1 KS_DYN=128
  KING_SAFETY_MAG=1000 RFP_MARGIN=1000` vs base (`gate_blitz`, the fair equal-time venue that also banks the speed
  win). If clearly + → a shippable KS+flywheel win (the first ~26-accumulation); if neutral → KS is neutral (banked
  understanding, not a lever at self-play TC); if − → move on. Games ~500, runs the night.

## 🎯 KS VERDICT + STRATEGIC PIVOT (2026-07-05) — isolation was the problem; go HOLISTIC
**KS-as-isolated-addition VERDICT: does NOT convert to Elo.** Every venue: lightning −50/−95 (hot mag, worst
venue), deep equal-node ~0±53 (noise), blitz(flywheel bundle w/ RFP1000) −36 (confounded by the global RFP change).
Consolidated + 4×-better-rep + faster + magnitude-tuned KS is neutral-to-negative. The consolidation/rep/speed wins
are REAL but don't translate. (The flywheel blitz was cut early — confounded config; a clean pure-KS blitz was not
needed given the aggregate verdict.)

**USER REFRAME (the key correction): KS didn't fail from BADNESS, it failed from ISOLATION.** The eval is ONE
vector; tuning KS alone against a frozen rest-of-eval makes it INCOHERENT (KS says "king in danger, avoid" while
threats/mobility/placement still price the position fine → bad move-choice). = eval↔eval coupling. TRUE fix =
co-rebalance the interacting CLUSTER (KS + threats + mobility + king-zone attack-layer) TOGETHER so the vector stays
coherent. This is UNTESTED (the flat outcome-Texel retune was scale-only + static + never brought gated terms online
together). Spot-fixes for specific failure PATTERNS; holistic rebalance for board-wide change.

**GROUNDING (fresh, decisive):** `vs_sf11 120 depth 8` = **our score 7.5% ≈ −437 Elo PURE EVAL gap at equal depth**
+ **35/120 collapses** (reach a winning peak then lose = EVAL mis-judgment, not search). ⇒ eval is MASSIVELY
unmined; SF11's HCE crushes ours before search. NNUE is NOT the move (user: NNUE later, off our OWN strong eval, not
SF-distill) — there's ~437 Elo of classical HCE headroom to SF11 first. Games in selfplay/games/sf11_evalgap_d8/.

**EFFICIENT METHOD (the plan):** fast deterministic compass (`movematch`, no games) = inner loop; SPRT gates only
the final coherent bundle. Loop: (1) map WHERE the gap concentrates (per-theme sts_full/movematch + the 35-collapse
corpus), (2) DEDUPE double-counts first (our representation map), (3) co-tune the CLUSTER holistically vs the compass,
(4) SPRT-gate the bundle once. FABLE question drafted: `dev_notes/fable-question-holistic-eval-2026-07-05.md`
(most efficient holistic HCE rebalance toward SF11 — objective choice, cluster vs full-vector, overfit control,
SPSA-vs-regression, spot-fix vs holistic). AWAITING: per-theme map (sts_full evalmap_base) + Fable's research.

**KS status:** consolidated machinery (KS_CONSOLIDATE, byte-id-safe) stays gated-off, ready to be brought online
AS PART OF the holistic cluster (not solo). Not shipped. Not dead — re-homed into the holistic program.

## ✅ PER-THEME EVAL-WEAKNESS MAP (2026-07-05, `sts_full evalmap_base`, depth10) — the cluster designs itself
Overall 51.0% (7653/15000). Sorted:
WEAK (dynamic/attacking): **AKPC 38%** (king-side pawn advance/attack — WEAKEST) · King Activity 43% · Knight
reposition 45% · Open Files & Diagonals 46% · AT (attacking) 46% · Square Vacancy 46% · Adv a/b/c pawns 47%.
STRONG (static/material): Recapturing 68% · Bishop-v-Knight 62% · Simplification 61% · Pawn Play Center 56% · 7th
Rank 53% · Center Control 52%.
**PATTERN (unambiguous): weak on DYNAMIC/ATTACKING themes, strong on STATIC/MATERIAL.** Matches (a) the outcome
residual (KS+mobility+rooks = the dynamic dims), (b) the −437 equal-depth collapses (we misjudge dynamic positions).
⇒ **HOLISTIC CLUSTER = dynamic king-attack + piece-activity: {KS, threats, mobility, open-files/rooks, king-zone
attack-layer}** — where we're weakest AND where terms interact (why isolated KS was incoherent: its cluster-mates
didn't move with it). Hold static/material terms ~fixed. This is the co-tune target. Baseline STS per-theme banked
for movematch_diff A/B during the co-tune.

## ⭐ NEXT SESSION FIRST STEP (user 2026-07-04) — AUDIT before more KS numbers
The 23% + config tests may be CONFOUNDED by double-counting: king-shelter is likely scored in BOTH the king
PLACEMENT functions (`evaluate_kings_midgame`/`evaluate_kings_endgame`) AND `king_safety_danger` (KS_SHIELD). The
placement copy is NOT subject to the realizability factors (KS_DYN/phase-taper). So BEFORE more KS tuning:
1. **Engine self-audit (source-read, MINE to do — no Fable):** trace every place king pawn-cover/shelter is scored
   — `evaluate_kings_midgame`/`_endgame` (cpp_bitboard.cpp), `king_safety_danger` (:4961, KS_SHIELD :5001),
   `white/black_king_shield` masks (:45,:513). Map HAVE / DON'T-HAVE / DOUBLE-COUNT. Same for open-files/storm.
2. **CONSOLIDATE (user's architecture):** LIFT shelter out of king-placement INTO KS so it's (a) counted once,
   (b) flows through the realizability conditioning (KS_DYN, phase). This de-confounds the KS screen.
3. **Research SF11 king()/shelter_storm** (dev_notes/sf11_eval_reference.md + source-verify) — the per-file shelter
   QUALITY model (advancement + storm distance + open/half-open by side) is the richness we lack; build it into
   the consolidated KS. Re-screen: does capture rise past 23%?
Only escalate to Fable on a genuine DESIGN fork from the audit (not the research/mapping — those are self-doable).

## Sequence
1. Screen existing KS machinery (cheap-first, in progress).
2. If gap: improve the KS function toward SF's king-danger (from sf11_eval_reference.md; the gated-additive
   superlinear form is right — the issue is capture, measured by the outcome screen not SF-agreement).
3. Keep MIDGAME-ONLY phase-gating (data-mandated; our latent_threat/king_safety already midgame-only).
4. Fit magnitude (tune_fit --target result on king_safety) — fit-set, not hand-cranked (the ks-detection-rebuild
   lesson: the tuning was the mistake, over-fit on a theme subset).
5. `node_ab` at a fixed-node budget + a BLITZ venue (KS payoff grows with TC; lightning was its worst venue) →
   SPRT. KS's bigger effect (+0.0031) RESOLVES cleanly at the gate — this also validates the gate-stage link
   (screen-positive → Elo) that immediate-threats was too small to test.
6. Ship on SPRT pass. Why this differs from the 5 prior KS deaths: outcome compass (not SF-agreement/theme-subset),
   fit-set (not hand-cranked), calibrated gate (not ±80-floor SPRT), blitz venue (not lightning-worst), gated-
   additive form. All causes addressed.

## Discipline
byte-id 245/39,146,294 (KS off) after any build. Screen (Δlogloss) proposes → node_ab gates → SPRT ships. Never
tune on a theme subset (ks-detection-rebuild over-fit lesson). Keep KS king-only (separate from threats).
Immediate-threats (granular fn #1, +0.00063) stays banked gated-off → bundle-gate later.
