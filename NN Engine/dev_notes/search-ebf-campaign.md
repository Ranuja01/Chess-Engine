# Search-efficiency / EBF-lowering campaign — dev log

## ▶️ NEXT-CHAT FIRST ACTION (2026-06-25): SEARCH LANE CLOSED → EVAL-QUALITY (material-edge calibration)
**combo1 SHIPPED default-on** (`NULLMOVE_EXTRA=2 HISTORY_LMR_SCALE=2 LMP_MAX_DEPTH=5 LMP_BASE=2`; new bench
baseline **WAC 252 / 67,931,145 nodes / STS 1503 (50.1%)**, verified). Lossless-speed lane done (movegen
−34% committed; rest below NPS noise — eval-bound). Search lane at peak: nothing stacks on combo1
(combo1+improving STS 1503→1488; combo2 benoni 0/8). **NEXT = eval-QUALITY: material-edge over-valuation**
(~+390cp up-a-rook, +231 mid→+462 end, [[material-edge-overvaluation]]). Plan
`~/.claude/plans/handoff-lossless-speed-campaign-tranquil-rose.md`. Levers already built (default-off):
`PV_BOOST_PHASE_K` (endgame damp), `MOD_MAT_PAWNS/OPPB`, `ENABLE_MATE_DRIVE_SCALE`, `ENABLE_ENDGAME_SCALE`
+ `is_practically_drawn` correctness extension. **HARD GUARD (user): do NOT regress passers
(`diagnostics/_passer_match.py passers_smoke.csv`) or trades (WAC + STS themes Recapturing/Simplification).**
Loop: knob → eval_breakdown material-bucket over-read ↓ + guards held → SPRT (material fixes self-play-
invisible → ship on bias-fix + no-regression).


**Started:** 2026-06-21. **Plan:** `~/.claude/plans/handoff-for-the-vectorized-meadow.md` (search version).
**Why:** floor is depth-bound (proven: +2 plies = −40% collapses, eval-tuning flat). Single-thread, no SMP.

## Method
Tune on the RELIABLE proxies (they PREDICT strength, unlike eval): `wac` (solves/300 + byte-deterministic
nodes + cutoff histogram), `sts` (/3000), `wac_timed_depth`/`sts_timed_depth` (depth at equal time = the
confound-free read). Read: nodes ↓ + solves/STS held + depth-at-equal-time ↑ = real win → SPRT gate.
Sweep via `wac <tag> KNOB=v OMP_NUM_THREADS=1` (deterministic; parallel-safe). PACE = perturb knobs on the
proxy, agent reads the pattern. SPSA-style breadth over ALL search knobs is the intent (decay/history/
reduction numbers + expose hardcoded formulas as knobs).

## Baseline (gold)
WAC **261/300, 134,429,469 nodes**, EBF 4.204, cutoff_hist m0=10,838,861 (96% first-move). STS **1566/3000
(52.2%)**. Engine is heavily pruning-bound/optimized → little clean headroom in the well-trodden knobs.

## Results (2026-06-21)
| candidate | WAC | nodes | STS | read |
|---|---|---|---|---|
| base | 261 | 134.43M | 52.2% | gold |
| gravity+malus (`ENABLE_HISTORY_SATURATION+MALUS`) | 256 | 124.63M (−7.3%) | 50.7% (−1.5%) | prunes content; net-neg (confirms shelved) |
| improving (`ENABLE_IMPROVING`) | 259 | 132.31M (−1.6%) | 47.6% (−4.6%) | worse; dead |
| `LMP_BASE=2` | 256 | 134.25M (−0.1%) | — | inert + −5 WAC; dead |
| `LMP_MAX_DEPTH=5` | 252 | 141.41M (+5.2%) | — | worse on both; dead |
| **`SEE_EXTEND_MARGIN=0`** | 252 | **92.07M (−31.5%)** | (pending) | **LEAD — controlled extensions** |
| `ENABLE_QCHECK_DEPTH0=1` | 261 | 134.43M (identical) | — | qsearch invisible to main NODES (see below) |

**LEAD = controlled extensions.** Default `SEE_EXTEND_MARGIN` = disabled → the check-extension extends ALL
checks → huge node adder. SEE-filtering (`=0` = only extend non-losing checks) cuts **−31.5% nodes** at −9 WAC
(some SEE-losing checks are real sacs). Frontier sweep `SEE_EXTEND_MARGIN ∈ {0,300,500,1000}` in progress to
find the margin that keeps the sacs while dropping wasted deep-losing-check extensions. Next: STS + **timed-depth**
on the best margin (the decider — does −31.5% nodes buy enough depth to beat −9 WAC) → SPRT.

## ✅ WIN: SEE_EXTEND_MARGIN=300 (controlled extensions)
Frontier: SEE=0 252/92.1M(−31.5%/−9WAC); **SEE=300 258/97.5M(−27.5%/−3WAC)** = sweet spot; SEE=500 identical;
SEE=1000 253/98.0M(−8WAC). **TIMED-DEPTH decider (equal LIGHTNING time): base 242 solves @ depth 11.58/10.23;
SEE=300 242 solves @ depth 12.33/11.04 = SAME solves, +0.75-0.81 ply deeper.** ⇒ the −3 fixed-depth WAC was
the artifact; at equal TIME it's strictly better (no tactical loss, ~0.8 ply more depth). For scale the +59-Elo
lazy+LMP ship was +0.50 ply. **STRONG SPRT CANDIDATE.** SEE=300 becomes the new baseline for the broadened SPSA.
Next: stack other node-cutters on top of SEE=300 (CHECK_EXTENSION cap, VERIFY, LMR) + broaden to all knobs +
expose hardcoded formulas → assemble → SPRT.

## STS caveat + stacks + SPRT (2026-06-21, ~11am checkpoint)
**Honest correction:** SEE=300 is tactically better at equal time (+0.8 ply, same WAC solves) BUT **STS −4.7%
at fixed depth (47.5% vs 52.2%)** — the extension filter prunes some positionally-relevant check lines. So
it's NOT a clean ship on proxies; it's depth-gain-vs-positional-accuracy = **SPRT territory** (depth usually
wins but −4.7% STS is non-trivial). **Stacks on SEE=300 all WORSE:** +VERIFY=3000 −18 WAC; +CHECK_EXT=2
+nodes/−4 WAC; +CHECK_EXT=4 −7 WAC. So SEE=300 ALONE is the candidate. **SPRT RUNNING** (`sprt_see300`, base
vs `SEE_EXTEND_MARGIN=300`, lightning, 240min) = the truth gate — read it for net Elo. If +Elo → ship SEE=300
(it's the campaign's win). If neutral/neg → the −4.7% STS positional cost outweighs the depth gain → dial the
margin up or drop it. NEXT regardless: qsearch (Phase 2, below) + broaden SPSA + expose hardcoded formulas.

## qsearch isolation (Phase 2 — the unexplored gap)
The `wac` NODES metric = main `num_iterations`, which **EXCLUDES** `qsearchVisits` (separate counter, printed
per-position "Q SEARCH VISITS" line 1249 but not surfaced by the bench). Added `qnodes=<qsearchVisits>` to the
`[search]` stderr line (search_engine.cpp ~1196, diagnostic-only, byte-id) — REBUILD owed to activate, then
the `wac` sub can sum it (add `QNODES:` grep). qsearch ordering is unoptimized (`buildNoisyMoveList` ~4710
filters the cached main-order to noisy but does NOT re-score; quiet-checks run every q-ply unless
`ENABLE_QCHECK_DEPTH0`). Levers: re-score noisy by SEE, `ENABLE_QCHECK_DEPTH0/MASK`, `MAX_QDEPTH`.

## SPSA breadth TODO (user) — beyond the hand-picked few
Sweep ALL existing search env knobs: `MAX_HISTORY`, `CONT2_GRAVITY_DIV`, `MALUS_DIV`, decay interval, VERIFY
(`VERIFY_MARGIN`/`VERIFY_RESEARCH_REDUCTION`), `ASPIRATION_*`, futility margins, `HISTORY_LMR_*`. EXPOSE the
hardcoded formulas as knobs first (the `depth²` history bonus, the per-preset `DEPTH_REDUCTION` table, decay
factor) — likely untapped headroom. Reliable proxy → SPSA converges here.

## Session 2026-06-21 outcomes (post SEE=300 ship)
- **SEE_EXTEND_MARGIN=300 SHIPPED & committed** (`5c194ae`); pooled SPRT +43.9 Elo [+22.8,+65.3]. New gold
  WAC 258/300, 97,507,126 nodes, STS 1426/3000 (47.5%).
- **Lazy/"light" eval at value sites — KILLED by measurement gate.** Eval cache hit-rate 20.6% (miss 79.4% =
  addressable) BUT the light-eval gap is ~85% `capture_gains` (median ~1.5 pawns) — the costliest tail term
  can't be skipped safely → low effective skip-rate. Built byte-id `LIGHT_GAP_PROBE`.
- **improving-cheap — REVIVED, parked for PACE+SPRT.** Standalone `cheap_eval()` (material+PST) for the
  sign-only improving trend; net-POSITIVE on WAC timed-depth (+3 solves/+0.106 ply, beats full's +0.049).
  Exposed `IMPROVING_CHEAP`/`IMPROVING_REDUCTION`/`IMPROVING_DELTA_MARGIN` (byte-id at defaults). Default-off.
- **SEE cache + qsearch SEE re-sort — CLOSED.** Phase-A: 199M see() (6.24/qnode); qsearch already well-ordered
  (qfmc 77.2%, qcut 0.31). `ENABLE_SEE_CACHE` byte-id but 8.7% hit → shelved. `ENABLE_QSEE_RESORT` −7 WAC
  fixed / timed-depth NEUTRAL → shelved. Both default-off.
- **BACKLOG — incremental-SEE rewrite (parked, safe ~3% search).** `see_impl` recomputes the FULL attacker set
  each exchange iter; knight/king/pawn attackers never change, only sliding x-rays. Quick tier (precompute
  non-sliders once, recompute sliders each iter) ~1.5–2×; full incremental x-ray ~2–2.5×. Byte-id (same SEE
  value → same WAC nodes). **Do the rewrite, NOT the cache (cache+rewrite ≈ rewrite alone, +0.5pp).**
- **Eval profiler re-run (tag sprt_see300, post surrogates):** PAWNS hottest 20–28% all phases (structural);
  ROOKS 12–21%, CAPTURE_GAINS/SEE 8–20%, BISHOPS 9–19%, LATENT_THREAT 12–13% midgame-only; ATTACK_LAYER only
  1.5–3.6% (already cheap via cache). Cross-component attacker-sharing = dead end (reverse vs forward query).
- **New byte-id diagnostic infra kept (gated/counting-only):** `SEE_COUNT`, qsearch `qfmc=`/`qcut=` on the
  `[search]` line, `LIGHT_GAP_PROBE`. All session knobs default-off; only the SEE=300 ship is committed.

## Session 2026-06-21 (pm) — formula knobs + improving SPRT candidate + old-vs-current milestone
- **Exposed 2 hardcoded formula knobs (byte-id at defaults):** `LMR_EXTRA` (extra LMR plies in
  reduced_search_depth, before the pin clamp; 0=orig) and `HISTORY_BONUS_SCALE` (% on the depth² history
  bonus `b`, all 6 sites; 100=orig). env loads + echo + [search] line.
- **PACE daytime screen → improving R2/M150 is the clean SPRT candidate.** STS grid (improving cheap):
  base 1426; R1/M0 (old default) 1392 (−34, the WORST point); R2/M0 1540; **R2/M150 1551 (+125 STS)**.
  REDUCTION=2 + DELTA_MARGIN=150 (the cheap-eval noise filter) = a strong move-quality win. Timed-depth alone:
  +5 solves, depth-flat (move choice). **LMR_EXTRA=2 = depth gamble (+0.71 ply timed-depth, STS −69);
  HISTORY×2 = −6.8% nodes but NEGATIVE interaction w/ improving → dropped; imp+LMR2 bundle STS cratered to
  1318 → ship improving ALONE, LMR_EXTRA is a separate gamble SPRT.**
- **SPRT (pm→overnight): improving R2/M150 vs baseline** — `ENABLE_IMPROVING=1 IMPROVING_CHEAP=1
  IMPROVING_REDUCTION=2 IMPROVING_DELTA_MARGIN=150`, LIGHTNING conc4 adjudicated, tags improv_r2m150_a
  (daytime) + improv_r2m150_b (1am-11am), pooled ~1500 games.
- **OLD-vs-CURRENT milestone:** rebuilt `selfplay/old CE/` at LIGHTNING (ACTIVE flip) → `tournament.py
  --p2-engine-dir "old CE"`. **current 89.9% / +380.6 ±61.5 Elo** (169 decided, 62 voids=old instability;
  prior was +176 → ~doubled). Per-phase (cur/old): depth 12.22/10.71, EBF 3.09/3.46, NPS 542/239 knps.
  Loss-opening (SF post-book eval, current POV): **14/17 losses = UHO worse-side lottery (9 < −50cp), 3/17
  genuinely outplayed (~1.8% of decided)** → current ~never loses a fair game.
- **WSL interop fix (load-bearing for adjudicated tournaments):** interop drops between short commands (idle
  distro reboot + flaky binfmt re-register post-outage). FIX: pin one interop-up instance with a long-lived
  keepalive (`nohup sleep 36000`); `wsl.exe --shutdown` first if a boot lacks it. Details in
  [[harness-concurrency-timed-uho]].

## 2026-06-23/24 — EBF Win 2: aggressive LMP + null-move crank (`lmp_nm1`)
- **Win:** `LMP_BASE=2 LMP_MAX_DEPTH=4 NULLMOVE_EXTRA=1` (UNCOMMITTED, gated, byte-id off) = **−22% nodes
  (97.5M→75.9M), wac_timed_depth +0.82 ply (12.42→13.24) +3 mates, sts_timed_depth +124 (1414→1538),
  benoni-29 (Gap-T trade) 8/8 at LIGHTNING.** Fixed-depth WAC 248 (−10, the [[fixed-depth-bench-ceiling]]
  artifact) / STS 1436 (+10). Lowers EBF AND raises the positional axis — the goal.
- **New knob `NULLMOVE_EXTRA`** (extra plies off the null-move search depth, both min/max null blocks) = the
  clean node lever; does NOT leak positional. **`LMR_EXTRA` re-confirmed to leak STS (−31)** → excluded.
- **Decomposition:** aggressive LMP ALONE (no LMR_EXTRA) already helps (STS +64 fixed / −6% nodes); the
  2026-06-21 "LMP shelved" verdict was the OLD pre-SEE_EXTEND baseline + LMP-alone — SEE_EXTEND reshaped the
  tree so LMP now wins, and the null-crank is the new multiplier.
- **`VERIFY_MARGIN`×aggression ANTI-SYNERGY on benoni at LIGHTNING:** aggr 8/8, VM 8/8, **aggr+VM 0/8** (VM
  re-searches consume the time the aggression freed → depth drops below the a4b5 threshold). So the EBF arm
  ships NO VM; aggression fixes the trade gate via the extra depth. NOTE: single-FEN LIGHTNING probes are
  TIME-NOISY → gate benoni by repeat-and-count (8×) or fixed depth.
- **Shelved sub-levers (kept gated, byte-id):** (1) **capchain guard** `ENABLE_LMR_CAPCHAIN` + leaky per-ply
  `g_captureChain` counter (+1 capture/−1 quiet, floored; `CAPCHAIN_RUN_THRESH`/`CAPCHAIN_REDUCE_LESS`) — a
  strong forcing-line de-pruner (guard-alone WAC 260/STS 1479) but a net node-ADDER (+28%), too costly to back
  aggression vs VERIFY; (2) **`PROTECT_MAX_IDX`** (index-gate PV/killer protection to early moves) — cut the
  blanket-PROTECT 384M blowup to ~baseline keeping partial STS; useful, not needed for the clean win.
- **Stretch (pending timed-depth + benoni-8×):** `lmp5_nm1` (D5,null+1: −30% nodes, STS held) / `lmp5_nm2`
  (D5,null+2: −36% nodes, STS +59 fixed). null+2 at LMP-d4 collapses STS (−99) but d5 rescues it.
- **NIGHT SPRT:** `lmp_nm1` (or validated stretch) + Gap-P eval (`PASSER_ENEMY_CREDIT_PCT=0
  ENABLE_PASSER_BLOCKADE_QUALITY=1 PASSER_CONTEST_PCT=30`) + `ENABLE_PASSER_PRUNE_EXEMPT=1` vs baseline.

## TRIED-TABLE — single-glance "what we've tried / don't re-try" (EBF campaign, baseline 258 / 97,507,126 / STS 1426)
Reliable reads = fixed-depth WAC nodes/STS (byte-deterministic) + SPRT; timed-STS is NOISE (±100–150); single-FEN LIGHTNING is time-noisy (gate benoni by 8× repeat-count).

| lever / config | what it is | result | verdict |
|---|---|---|---|
| `lmp_nm1` = LMP_BASE2 / depth4 / null+1 | aggressive LMP + 1 extra null ply | −22% nodes, STS held, benoni 8/8, **SPRT +12.4±23 (not sig, no regression)** | ✅ safe win, gated (shipped-pending) |
| `lmp5_nm2` = depth5 / null+2 | one step more aggressive | −36% nodes, STS +59, benoni 8/8 | ⭐ best escalation candidate |
| `f3` = depth6 / null+2 | deeper LMP | −35% nodes, STS +76, WAC 242 | alt (more positional, more WAC dip) |
| `NULLMOVE_EXTRA` | extra null-move reduction plies | null+1 clean; **null+2 needs depth≥5** (null+2 @ depth4 = STS −99) | ✅ clean node lever |
| `LMR_EXTRA` (reduce-more) | extra LMR plies | cheap nodes but **STS −31 (leaks positional)** | ❌ excluded |
| capchain guard (`g_captureChain` leaky counter, thresh2/rl3) | de-prune LMR/LMP on capture-heavy lines | guard-alone WAC 260/STS 1479 but **net +28% nodes** | ❌ shelved (node-adder) |
| PROTECT_PV/KILLERS — ALL moves | don't reduce PV/killer anywhere | STS +145 but **+290% nodes** | ❌ blanket too costly |
| PROTECT — top-2 (`PROTECT_MAX_IDX=2`) | protect only 2nd/3rd moves | 384M→~baseline, partial STS | ⚠️ works, not needed for clean win |
| passer-exempt (`ENABLE_PASSER_PRUNE_EXEMPT`) | keep advanced passer pushes above horizon | adds nodes; orthogonal PP safety | ⚠️ kept as insurance, not Elo-tested |
| hyper / ultra (all guards + max aggression) | everything on | **+14–16% nodes (guards add > save)** | ❌ guards backfire at scale |
| `VERIFY_MARGIN=16000` (Gap-T) | re-search near-miss reductions | fixes benoni alone; **aggr+VM benoni 0/8 (anti-synergy)** | ✅ ships solo, ❌ not with aggression |
| Gap-P eval (P1 ENEMY_CREDIT=0 + C1 blockade-quality) | passer-danger sign/blockade fix | F36 eval sign fixed, WAC 256/STS +47 | ✅ worst-case-Elo track (self-play-invisible) |
| qnode share | qsearch fraction of nodes | **~27%** (consistent across phases) | 📋 A3 target (DELTA_MARGIN/MAX_QDEPTH) |
| improving / cont-hist (prior) | per-node-eval pruning | net-negative (eval not incremental) | ❌ shelved → needs cheap-eval first |

**One-liner:** clean EBF wins = guard-free LMP-depth + null-crank (≈−35% ceiling); guards underperform; LMR_EXTRA hurts; VM≠aggression. Below −35% needs new techniques (Phase F A-set: SEE-prune / history-LMR / qsearch knobs).

## 2026-06-25 — Phase H: engine-wide LOSSLESS-speed campaign (profile search → byte-id NPS)
**Goal:** make existing code faster computing the EXACT same thing (byte-id, same nodes/moves; gate = exact WAC node match 258/97,507,126). User principle: take verified byte-id speedups. Distinct from the (failed) lossy eval-cheapening — these are zero-risk.
- **Extended the PROF framework to SEARCH (UNCOMMITTED, gated `#ifdef EVAL_PROFILE` = no-op/byte-id in production):** new `PROF_MOVEGEN`/`PROF_MAKEUNMAKE`/`PROF_TT_PROBE` IDs (cpp_bitboard.h enum + names) + `PROF_BLOCK` scopes on `make_move`/`unmake_move` (search_engine.cpp), `accessSearchEvalCache` (cache_management.h), `generateLegalMovesReordered` (move_gen.h); `eval_profile_reset/dump` auto-bracket a real search in `get_engine_move`. Build instrumented: `PROFILE_EVAL=1 python setupAI.py build_ext --inplace`; run `PRESET=LIGHTNING python main.py` → `[eval_profile] search` dump on stderr. **Production rebuilt after = byte-id 258/97,507,126 (PROF no-op confirmed).**
- **WHOLE-SEARCH PROFILE (one LIGHTNING search, absolute cycles; eval-base≈1522M):** EVAL ~1522M (~68%) · **MOVEGEN 469M (~21%, 11,415 cyc/call!)** · MAKEUNMAKE 163M (~7%) · TT_PROBE 75M (~3%). Within eval: PAWNS 312M(20%), LATENT_THREAT 260M(17%), CAPTURE_GAINS 260M(17%), ROOKS 232M(15%); SEE 148M (nested). **HEADLINE: MOVEGEN is the #1 lossless target — a previously-unmeasured ~21% (generate+score+sort+ordering-SEE per node). A 2× movegen ≈ −10% total search, byte-identical.** Then LATENT_THREAT (rescan-during-eval restructure), PAWNS (hoist attackingLayer reads), MAKEUNMAKE (per-node BoardState copy reduction).
- **MOVEGEN SUB-PROFILE (added gated `PROF_MG_GEN`/`PROF_MG_SCORE`/`PROF_MG_SORT` drill scopes inside `generateLegalMovesReordered`):** of MOVEGEN's ~12,219 cyc/call — **MG_GEN (pseudo-legal generation) 6,878 = 56%**, MG_SORT (stable_sort + Move rebuild) 2,611 = 21%, MG_SCORE (scoring loop incl. SEE) 2,026 = 17%. ⇒ **generation dominates, NOT sort/score.** Also: only ~41k movegen calls vs ~893k make-moves → **~95% of nodes hit the move-gen cache and skip generation**; movegen cost is concentrated in cache-miss nodes (12k cyc each). The plan's "iota-sort/Move-construction" targets are the SMALL 21%; the lever is the generation path.
- **🚀 WIN — hoist node-invariant king/blockers/checkers out of `generateLegalMoves` (UNCOMMITTED, byte-id):** root cause = `processMaskPairs` calls `generateLegalMoves` up to **16×/node** (12 capture pairs + 4 quiet pairs partitioned for ordering), and EACH recomputed `king`/`slider_blockers`/`attackersMask(checkers)` — all of which depend only on the board, not the per-pair from/to masks → identical across the 16 calls. Fix: extracted `generateLegalMovesPre(...)` taking precomputed `(king,blockers,checkers)` (+ reuses thread_local scratch, kills the 16×3 vector allocs/node); `processMaskPairs` derives them **once at its top** and passes them in (16×→2×/node, since processMaskPairs runs twice: captures + quiets); public `generateLegalMoves` kept as a thin wrapper for `generateLegalCaptures`. **Result (cycle profiler, the trustworthy instrument — WAC wall-time was useless session thermal drift): MG_GEN 6,878→4,501 cyc/call (−34.6%), MOVEGEN 12,219→10,095 (−17.4%) ≈ −3–4% of total search, byte-id 258/97,507,126.** Also moved S1 = thread_local scratch in `generateLegalMovesReordered` (byte-id but wall-neutral, matches the old S4 ~0.3% estimate in OPTIMIZATION_LOG — kept, free).
- **🚀 WIN 2 — MG_SORT collapse (UNCOMMITTED, byte-id):** replaced the `iota` + `stable_sort`-over-`std::vector<size_t> indices` + separate `std::vector<int> moveScores` + per-move `Move()` rebuild-via-indirection with a single `thread_local std::vector<ScoredMove{int score; uint8_t from,to,promo}>` — built inline during the score loop, `stable_sort` by `score` (equal-score order preserved == old index order, so byte-id), push `Move` directly. One fewer buffer + no index chase. **Result: MG_SORT 2,611→1,737 cyc/call (−33%); MG_SCORE 2,026→1,695 (inline build, better locality). byte-id 258/97,507,126.** (Counter-move field-compare micro NOT done: `Move cur` is reused by `killerBonus(ply,cur)` so it must be built anyway.)
- **MOVEGEN BANKED: 12,219 → 8,015 cyc/call = −34% over the session** (hoist invariants + MG_SORT collapse + thread_local scratch). NPS A/B (main.py best-of-3, production, noisy ±5%): ~420k→~438k ≈ **+3%**, consistent with the profiler's −3–4% total-search cycles. MOVEGEN went from ~22%→~18% of total search.
- **MOVEGEN ≈ OPTIMAL for lossless now.** Residual: MG_GEN 4,044 (necessary legal-gen + per-move `is_safe`), MG_SCORE 1,695 (SEE-dominated, already incremental), MG_SORT 1,737 (O(n log n) floor). Only remaining ideas: (a) skip `is_safe` for non-pin/non-king/non-ep moves in the non-check branch — standard, lossless IF it reproduces is_safe exactly, but correctness-sensitive → defer; (b) 2×→1× invariant thread-through ~0.3% → skip.
- **❌ EVAL attackingLayer-hoist — TRIED & REVERTED (proven no-op).** Hoisted the repeated `attackingLayer[0/1][x][y]` reads into locals across all per-piece evaluators (14 sites: pawns/knights/bishops/rooks/queens × W/B × mid/end). Byte-id held, but the speedup was UNVERIFIABLE (per-call costs 85–430 cyc, machine noise ±15% run-to-run swamps it). **Settled definitively by ASSEMBLY DIFF: compiled cpp_bitboard.cpp to `.s` hoisted vs one-site-reverted → 0 differing lines.** The compiler already CSEs these loads because every read PRECEDES the `update_global_central_scores` call and nothing reads the array after it (no post-call reload needed). ⇒ a source-level hoist here is a TRUE no-op (identical machine code), not even a micro-win. **Reverted all 14 (byte-id 258/97,507,126).**
- **LESSON: lossless EVAL micro-opts are doubly blocked — (1) per-call costs sit below the ±15% single-run rdtsc noise floor (only big changes like MOVEGEN's −34% read cleanly), and (2) the compiler at -Ofast already does intra-function CSE/load-hoisting, so "redundant read" patterns where all reads precede the opaque call are already optimal. A measurable lossless eval win needs PASS-LEVEL redundancy removal (eliminate a whole rescan / recompute), not load-hoisting. Verify any claimed codegen change with an `.s` diff, not the profiler.**
- **EVAL POST-PASS study (CAPTURE_GAINS + LATENT_THREAT) — NO big lossless redundancy.** Deep map: the two functions scan DISJOINT square sets (capture_gains = all non-king pieces; latent_threat = king zones only) → can't be merged; king zones already constexpr-cached; latent_threat's `attack_bitmasks` read is NOT a redundant rescan (it needs the COMPLETE post-loop mask — single read per zone square, irreducible); the piece-type if-chains duplicated 4× are a refactor, not a speed win (predictable branches). **Only genuine find: `approximate_capture_gains` computes `attack_bitmasks[r] & enemy_occ` at the popcount AND again at `get_least_valuable_attacker`, with the opaque `see()` call BETWEEN them blocking CSE → cached in `attacker_mask` (byte-id 258/97,507,126, UNCOMMITTED). Small but a REAL win (CSE genuinely blocked, unlike attackingLayer).**
- **make_move copy-reduction (byte-id):** `const BoardState& current = state_history.back()` (was a full ~104B struct copy) + `emplace_back(...)` in place of a named `newState` + lvalue `push_back`. Measured in the noise (MAKEUNMAKE is hash-map-bound, not copy-bound).
- **🟰 Per-ply MOVE-BUFFER POOL (user's lead: cache retrieval copies) — SHIPPED, byte-id, NPS-NEUTRAL.** `accessMoveGenCache` returned `vector<Move>` BY VALUE = a heap alloc + copy on every cache hit (~95% of nodes). Returning a `const&` into the cache is UNSAFE (a colliding `addToMoveGenCache` reallocs the slot mid-iterate → dangling). Safe fix = the strong-engine pattern: per-ply buffers `g_moveBuf[ply]`/`g_noisyBuf[ply]` (cpp_bitboard.cpp, sized `MOVE_POOL_PLIES=128` ≥ main+qsearch ply with a bounds-guard fallback); `buildMoveListFromReordered`/`buildNoisyMoveList` snapshot the cache into the per-ply buffer (`fillMoveGenCache`, synchronous copy — no cache ref escapes) and return `vector<Move>&` to it; deep callers (minimizer/maximizer/qsearch) bind the ref. SAFE because ply is monotonic + single-threaded → a node owns `g_moveBuf[ply]`, children use deeper buffers (invariant documented at the defs). **Clean ref-vs-value isolation A/B: ~419k→~424k mean = neutral-to-≤1%, within ±15% noise.** Verdict: move-list alloc is NOT a bottleneck here (eval=68%); kept anyway (correct, safe, real-alloc-removal, right architecture, may compound later). Built incrementally, byte-id 258/97,507,126 at steps a/b/c.
- **MEASUREMENT LESSON (recurring): every lossless micro this phase — eval hoist, make_move, capture micro, the buffer pool — lands below the ±15% single-run NPS noise floor. Only MOVEGEN's −34% read cleanly. This engine is EVAL-BOUND; lossless speed past movegen is real but unmeasurable. Band-breaker = eval-QUALITY.**
- **PHASE H EVAL VERDICT: eval is largely EXHAUSTED for lossless speed** — the compiler at -Ofast already does the intra-function CSE/hoisting, and there's no big shareable rescan. The session's real banked win is MOVEGEN (−34% cyc/call, ~+3% NPS, verified). NEXT structural candidate = MAKEUNMAKE (~7%, ~893k calls/search — per-node `BoardState` copy is real work the compiler can't elide; reduce fields copied / make-unmake incrementally). After that, lossless is done → bank it and pivot to the parked band-breaker = eval-QUALITY (material-edge calibration). Side items: `improving` audit; `is_light` (low priority).

## 2026-06-25 — Phase G+ CLOSED: eval-speed lane exhausted; SEE-incremental SHIPPED; pivot to eval-QUALITY
- **SEE-incremental SHIPPED default-on** (`ENABLE_SEE_INCREMENTAL`, search_engine.h; byte-id both ways = 258/97,507,126; free ~1.3% wall-time, identical moves, zero strength risk). Per the user's "take verified byte-id speedups" principle. UNCOMMITTED.
- **`is_light` TESTED → SHELVED.** qsearch light (`QSTANDPAT_EVAL_MODE=2`): WAC 240 (−18), **nodes +10% (106.8M)**, STS 1401, timed 12.26 (−0.16) — WORSE on every axis; the +10% nodes shows the light stand-pat is inaccurate enough to make the search BIGGER (the capture_gains≈0-at-quiescent-leaves bet was WRONG; the heavy terms are load-bearing for the stand-pat decision). futility light (`FUTILITY_EVAL_MODE=2`): WAC 250 (−8)/−4% nodes — milder but still net-negative. Knobs kept gated (default 0). Built infra: global `g_eval_light` + 4 heavy-term guards + cache-bypass + `eval_by_mode` dispatch.
- **VERDICT — eval-speed lane CLOSED.** SEE-incremental (1.3%, shipped), A-items (all net-negative, eval-screen), is_light (degrades search) all confirm: you CANNOT cheapen the eval at decision sites without hurting quality. The eval is precise-AND-slow and both are load-bearing → definitively EVAL-BOUND. **Next lever = eval QUALITY: material-edge calibration (the diagnosed +390cp up-a-rook bias, [[material-edge-overvaluation]]), NOT eval speed.**
- **combo1 status:** still the best config (bench peak STS 1503/WAC 252; leg-A SPRT +13.4 ±37, not sig). Decision: SHIP combo1 on the lean+no-regression+gap-fixes evidence and move the box to eval-quality, OR burn ~3000 games to confirm the small +13 (poor ROI). Rec = ship + pivot.

## 2026-06-24 (night) — Phase G+ eval-speed: SEE incremental + overnight screen
- **combo1 SPRT leg A: +13.4 ±37 Elo (n=467, base 48.1%, 80 adj draws). Positive lean, not sig. Banked/gated; leg B deferred.**
- **SEE incremental-x-ray rewrite (`ENABLE_SEE_INCREMENTAL`, cpp_bitboard.h see_impl, default off, UNCOMMITTED):** maintains the attacker set across exchange iters (clear used attacker + OR in x-ray sliders) instead of full `attackersMask` recompute. **BYTE-IDENTICAL confirmed (on → exact 258/97,507,126).** BUT speed only **−1.3% wall-time** (215.6→212.7s same nodes), timed-depth flat → **below Elo-noise, not worth shipping.** Lesson: per-iteration `attackersMask` is NOT the bottleneck (table-cheap + SEE exchanges short); **capture_gains cost = NUMBER of see() calls (one per attacked square), not per-call cost** → the lever is FEWER calls (`is_light` skip at quiescent leaves), not faster see(). Knob kept (harmless).
- **NEXT: `is_light` light-eval** (skip capture_gains/latent/passed-support/adv-endgame at qsearch stand-pat — where they're ~0 at quiescent leaves) = the real capture_gains lever. Careful daytime build.
- **Eval-screen RESULT (2026-06-25, `selfplay/evalscreen.csv`): A-items ALL net-negative on the combo1 base.** Base(combo1)=252/67.9M/STS1503/timed13.41 is the bench PEAK. Every config LOWERS STS+WAC for a node cut + tiny timed bump: cq 243/1485, ck 244/1413, cqk 246/1407, imp 248/1488, micro 246/**1426** (pdamp2+cutH DROPS STS — conflicts w/ history-scale's STS gain), imp_micro 244/1428, cqk_imp 243/1356, all 244/1400, all_imb 248/1429. **The "combined meta stack" REGRESSES (all=1400 STS vs 1503) → NOT an SPRT candidate.** Eval-bound reality: these are speed/approximation knobs trading accuracy for nodes; combo1 already sits at the accuracy peak. combo1 (+13.4±37 leg A) stays the best config.
- **`is_light` BUILT + byte-id confirmed (2026-06-25, UNCOMMITTED, default modes=0 → 258/97,507,126).** Global `g_eval_light` + guards on the 4 heavy terms (capture_gains/passed-support/latent_threat/adv-endgame) in `placement_and_piece_eval`; `get_board_evaluation` cache read+write bypassed when light (no pollution); `eval_by_mode(mode,...)` dispatch (0 full / 1 cheap `cheap_eval` / 2 light) at futility (`FUTILITY_EVAL_MODE`) + qsearch stand-pat/horizon (`QSTANDPAT_EVAL_MODE`). **UNTESTED** (light-mode benches + benoni/passer pending — cutoff/permission-block hit). NEXT: `QSTANDPAT_EVAL_MODE=2` wac/sts/wac_timed_depth + benoni-8× + passer-match; the bet = capture_gains≈0 at quiescent leaves so light≈full there but much faster.

## 2026-06-24 — Phase F (A-set) results (gated/byte-id, UNCOMMITTED)
- **A3 (qsearch knobs) = DEAD END.** Promoted `DELTA_MARGIN`(1500)/`MAX_QDEPTH`(10) constexprs → Config env ints (use-sites in qSearch; byte-id). Sweep: `dm700` WAC 246/qshare 24%, `dm400` 249/25%, `qd6` 253/qshare 26% (unchanged). Tightening loses WAC for ~no qnode saving; MAX_QDEPTH already effectively unbinding. The ~27% qshare is irreducible (necessary captures/checks). Knobs kept gated, not in any config.
- **A1 (`ENABLE_SEE_PRUNE`) = DISQUALIFIED.** Gate after the LMP block in both `get_score_for_*` helpers; reuses the check-ext call form `see(move.to_square, updated_state.turn, updated_state)` (post-move opponent recapture = does the moved piece hang). Knobs `SEE_PRUNE_MARGIN`/`SEE_PRUNE_MAX_DEPTH`. margin-0 d3 = −15% nodes but STS 1394 (−32); margin-0 d4 = −18%/STS 1377 (−49); **margin-2000 d3 = STS 1530 (+104)/−9% nodes/WAC −11** (conservative margin needed). BUT in the aggressive combo (`combo2`) it **breaks benoni 0/8** (prunes a resource in the a4b5 trade line) → OUT. Kept gated.
- **A2 (`HISTORY_LMR_SCALE`) = THE FIND.** Depth-scaled reduce-more for tier-0 quiets, applied at the reduced-depth call sites (`hist_delta -= min((depth_limit-cur_depth)/SCALE, SCALE_CAP)` when hist_delta<0; suppressed in pins; 0=byte-id). `HISTORY_LMR_SCALE=2` = **STS 1557 (+131)** / WAC 255 (−3) / −3% nodes — biggest positional gain measured, near-free (history bonus+decay is ~0 for targets, so scale by remaining DEPTH not history magnitude). `hls4` weaker (+52).
- **WINNER `combo1` = `LMP_BASE=2 LMP_MAX_DEPTH=5 NULLMOVE_EXTRA=2 HISTORY_LMR_SCALE=2`** (lmp5_nm2 EBF base + A2): **−30% nodes (67.9M) + STS 1503 (+77) + benoni 8/8 + WAC 252 (−6)** = dual EBF+strength win. STS gains don't fully stack (hls2 +131 → +77 atop lmp5_nm2). `combo2` (+SEE-prune) disqualified (benoni 0/8). **SPRT-PENDING (held for user go).**
- **PASSER GATE FIX:** live-SF `move_proxy`/`fen_vs_sf` over the corpus **HUNG 28 min at position 33** (SF arbiter call has no timeout; interop keepalive had dropped) → built SF-FREE `diagnostics/_passer_match.py` (runs our engine, compares to the recorded `sf_best`/`sf_cp` columns — no live SF). Use this for corpus move-match; reserve live-SF for tournaments (with keepalive pinned).
- **combo1 GATES PASSED:** passer-match baseline 84/200 (42.0%) → combo1 **87/200 (43.5%, +1.5pp = NO PP regression)**; combo1 **wac_timed_depth 13.55 NONMATE (+1.13 ply vs baseline 12.42)**. Yellow flag: combo1 passer eval-error vs SF rose 970→1470cp (sharper search scores, NOT a move regression). ⇒ combo1 = SPRT candidate (HELD for user go).
