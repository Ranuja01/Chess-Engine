# Fable consult — post-NPS-lane reassessment + the "cash-in" question (2026-07-13)

Fable, you previously produced the roadmap (`ROADMAP-2026-07-12.md`) and its reassessments
(`fable-followup-roadmap-2026-07-12.md`, `fable-question-roadmap-reassess-2026-07-12.md`). We've now
executed a chunk of it and have real results. Reassess — **but ground every claim in our actual code, not
priors. Where a question touches a mechanism, READ the cited file:lines and quote what you find.**

## What happened since your last look (all verified, committed this session)
The diagnosis held: the collapse is search-efficiency, not eval (see `collapse-search-efficiency-diagnosis-2026-07-12.md`).
The eval-FEATURE lane is closed (mobility gauntlet NO-GO). We pivoted to **Lane 2 (byte-identical NPS)** and
finished it:
- `7192005` no-copy movegen-cache probe (+6.7% NPS, byte-id 247/41,479,610)
- `bd19bd5` shipped the uncommitted statScore-LMR (+23 lightning SPRT) that was defining byte-id 247
- `1082835` committed the default-off eval/search scaffolding
- `de80343` PAWNS loop→table substitutions + branchless mailbox scatter (~+3% NPS median)
- `3adcee3` movegen blockers/checkers hoist (+2.2% NPS median)
→ **~12% cumulative byte-identical NPS**, all decision-neutral (byte-id 247 held throughout).

Remaining byte-id NPS levers are exhausted: per-slider x-ray recompute is IRREDUCIBLE (already one composite
magic-table lookup); SEE-cache was shelved (byte-id but 8.7% hit); pawn-hash is impure (reads king via
`attackingLayer`, non-pawn blockers, phase — not byte-identical); B1/B2/A3 micro-CSEs are compiler-territory.

## The measurement reality (important for your advice)
`depth_nps_bench.py` (our SPEED instrument) is **±5% run-to-run noisy**; ~12% NPS ≈ **+0.15 ply** and our
depth-at-1s median held at 12 throughout. Our calibrated gauntlet is **fixed-NODE (250k)** — NPS is invisible
there by construction; only a TIMED venue would show it.

## The "cash-in" question and a wrinkle we need you to examine IN THE CODE
The thesis motivating the NPS work: cheaper eval → afford more eval-gated pruning → bypass a negative curve
(esp. `improving`, currently `ENABLE_IMPROVING=0`, shelved). **But READ `search_engine.cpp:483-512`
(`static_eval_for_improving`) — it uses a CHEAP material+PST surrogate (`IMPROVING_CHEAP`, :489-492), NOT the
full eval.** So does our full-eval NPS speedup actually reduce improving's cost at all? If not, the flywheel
premise may be wrong as stated. Assess this against the real code.

## Read these (actual system), then answer
- `search_engine.cpp:483-512` (`static_eval_for_improving`) + `:2396-2410` and `:2730-2744` (the ENABLE_IMPROVING
  use-sites in minimizer/maximizer) — what improving actually does and costs.
- The eval-gated pruning in `search_engine.cpp`: node-entry RFP (already shipped +73 Elo), futility, razoring,
  null-move, statScore-LMR — grep `RAZOR`, `FUTILITY`, `NULLMOVE`, `RFP`/razor_threshold. Which of these consume
  a static eval, and is any bottlenecked on eval COST vs just decision quality?
- `cpp_bitboard.cpp` `placement_and_piece_eval` (the full eval) + `g_eval_light`/light-eval path — could a now-cheaper
  full eval profitably REPLACE the cheap surrogate in improving/futility (a real use of the speedup)?
- `ROADMAP-2026-07-12.md` Lanes 1 (pipeline: ordering-unfreeze + stripped-threats) and 3 (negamax-fold →
  TT-rebuild → verified TT-move → IIR → singular; pre_minimizer A/B) — still the right priority order?

## Questions
1. **Is the cash-in premise sound given the cheap-surrogate finding?** If improving uses the cheap eval, where
   (if anywhere) does ~12% cheaper full-eval actually unlock pruning? Name the concrete lever in OUR code or say
   the premise is unsound.
2. **Highest-value next lever now** — cash-in-via-pruning, a timed-gauntlet validation that NPS converts at all,
   Lane 1 (pipeline), Lane 3 (structural), or the ceiling conversation? Rank them for OUR situation.
3. **Any eval-cost-bound lever we're missing** that the cheaper eval specifically enables (e.g., full-eval-in-improving,
   tighter futility/RFP margins that were eval-cost-limited, eval-gated LMR)?
4. **Reassess the ~3100-3300 CCRL classical ceiling** given the ~12% NPS is real but only ~0.15 ply — does the
   speed change the ceiling read, or is it strictly a pruning-enabler whose value is 0 unless (3) pans out?

Be concrete and code-grounded. If a claim can't be checked against our files, flag it as a prior, not a finding.

---

## FABLE'S ANSWER (2026-07-13, code-grounded)

**Q1 — cash-in premise is UNSOUND as stated (verified in code).** `static_eval_for_improving` uses the cheap
material+PST surrogate (search_engine.cpp:486-512; `IMPROVING_CHEAP` default true, search_engine.h:337) and
"does NOT touch the full eval cache" (:489-491); `ENABLE_IMPROVING=false` (h:331). So ~12% cheaper full eval
changes improving's cost by **zero**, and improving only feeds an LMR reduce-more (:2398-2404 / :2732-2738),
never futility/RFP margins. The full-eval consumers ARE already lazy + eval-cached, so they're
decision-QUALITY-bound not cost-bound: node-entry RFP/null-eval-gate (:3266-3283, eval only when the gate
wants it, cache-probed :6065-6069, driven by RFP_MARGIN=1500), futility (:2362-2371, gated ≤4 from horizon,
cached). Razoring is NOT eval-gated (uses prev-iteration search scores, :2056); null-move needs no static eval
(ENABLE_NULL_EVAL_GATE=false). **No pruning's activation was waiting on eval cost.** Narrow real lever:
retest `ENABLE_IMPROVING=1 IMPROVING_CHEAP=0` — the full-eval path (:500-511) shares the eval cache with
RFP/futility, so on nodes RFP already evaluated it's a cache hit and the trend signal becomes real eval.

**Q2 — ranking:** (1) **Lane 3 TT-rebuild → verified TT-move → IIR with eval16-in-TT** — the roadmap already
names it (ROADMAP:92-94 "eval16 making RFP/futility free"); storing static eval in the TT entry IS the
structural version of "cheaper eval at decision sites" (node-entry eval ~free everywhere). Lane 1 is burned
down (1a flat, 1b NO-GO). (2) one **timed-venue** background run to confirm 12% converts — but +0.15 ply ≈
~+10 Elo needs thousands of games; don't gate on it. (3) the `IMPROVING_CHEAP=0` retest (one arm). (4) Lane 1
remnants only if Lane 3 stalls. (5) ceiling — nothing new.

**Q3 — eval-cost-bound levers:** eval16-in-TT (the big one; architecture not a %); IMPROVING_CHEAP=0 (real,
shares accessCacheNew); widening RFP_MAX_DEPTH/futility window is a margin-QUALITY tuning sweep not an unlock;
light-eval (`g_eval_light`) at pruning sites would strip exactly the leaf-visibility terms the collapse
diagnosis convicted — leave parked.

**Q4 — ceiling ~3100-3300 UNCHANGED.** The gap is ~2.5-3x depth-selectivity (our d18 ≈ SF11 d6) ≈ an order of
magnitude in nodes; 12% NPS ≈ +0.15 ply ≈ 1-2% of the gap, never the binding constraint. The depth is real
and worth banking, but **cash-in-via-pruning is dead as stated; the honest heir is the Lane 3 TT rebuild
(eval16).** IMPROVING_CHEAP=0 = cheap side arm; one background timed run to confirm conversion.
