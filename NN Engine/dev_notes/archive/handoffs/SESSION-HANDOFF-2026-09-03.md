# SESSION HANDOFF 2026-09-03 — the missing root sort; six occupancy failures; the transfer model

## READ FIRST
- ⭐ **THE FINDING: the pre-search-OFF path never sorted the root list.** Every "remove the pre-search"
  experiment in this project's history ran with **no root ordering at all**. A lane held shut by a missing
  line of code, not a missing idea. Fixing it cut the no-pre-search penalty **+38.1% → +2.7%**. See 2.
- ☠️ **Six techniques ported from strong engines failed on OCCUPANCY, not merit** — our architecture
  already fills the slot, or consumes the output differently. A reusable transfer model came out of it. See 4.
- ✅ **Root razoring VALIDATED**: worth **14.5% of nodes at no measurable accuracy cost**. Our
  abandon-the-tail approach beats SF's search-it-shallower, for us. See 3.
- ☠️ **Removing the pre-search costs 80-190 STS, measured four ways.** Revises the earlier
  "it buys nodes, not accuracy" claim — four consistent negatives are a result. See 2.
- ▶️ **The leap is not in search.** Every margin is sized for eval noise; that is the 30M barrier. See 6.

## 1. STATE
Baseline UNCHANGED and re-verified many times: **250 / 35,310,778 / EBF 3.784 / STS 1796**.
`wac_speed` reference on the current build: headline **376,007**, spread 4.2%, nodes stable over 5 reps
(⚠️ not comparable to the canonical `depth_nps_bench` 450,201 — different workload).
Everything built this session is **default-off and byte-identity verified**. Nothing committed.

## 2. THE ROOT SORT (the session's one positive)
`reorder_legal_moves`' hard-off branch returned the previous iteration's list untouched and returned
before either sort site — its own comment says *"This path does NOT sort"* (`search_engine.cpp:6194`,
branch `:6136-6198`). So move 0 was arbitrary, the one full-window PVS search went to a non-best move, and
every scout ran against a badly seeded alpha.

**`ENABLE_PRESEARCH_OFF_SORT` + `PRESEARCH_OFF_SORT_MODE`** (both default off). Mode 2 = owner's design:
strict PROVENANCE TIERS, never comparing across depths.
| tier | condition | key |
|---|---|---|
| 1 | `verified` (proved this iteration) | `top_score` desc |
| 2 | `last_real != UNPROVEN` — **REVIVAL** | `last_real_depth` desc, then `last_real` desc |
| 3 | never measured | keep move-gen heuristic order |
⚠️ Tiers 1-2 need `ENABLE_ROOT_TABLE` (which maintains `verified`/`last_real`/`last_real_depth`).
**Mode 2 without the table is byte-identical to unsorted** — verified.

**Measured (WAC d10, pre-search off + razor off; baseline 250 / 35,310,778 / 1796):**
| arm | WAC | nodes | vs base | STS |
|---|---|---|---|---|
| unsorted | 246 | 48,760,840 | +38.1% | 1684 |
| table only | 246 | 47,556,855 | +34.7% | — |
| mode 1 (`top_score` only) | 246 | 43,615,988 | +23.5% | 1612 |
| **mode 2 (tiers) + table** | 243 | **36,259,573** | **+2.7%** | **1714** |
Mode 1's −72 STS is RECOVERED by mode 2 ⇒ mode 1 was ranking fail-low BOUNDS against measured values, the
same defect as [[root-razoring-needs-scores-not-bounds-failllow-does-not-discriminate]].
**At fixed TIME** (LIGHTNING, one batch): baseline 258 WAC / 1734 STS; cascade **246 / 1743** — positional
PARITY at equal time with the pre-search off entirely, at a cost of 12 tactical solves.

☠️ **But every route to removing the pre-search costs positional accuracy**: unsorted −112 STS, mode-1 −184,
cascade −82, hybrid −125. Individually inside the ±150 floor; **four in the same direction is a result.**
⇒ **The pre-search buys ~80-125 STS points.** (This CORRECTS the mid-session claim that it buys only nodes.)

## 3. TAIL HANDLING — every variant tested, all closed
The root loop razors and `break`s, so ~23 of ~35 root moves are never touched by the main search. Remove the
pre-search and razoring must go too, so the main search pays FULL depth for all ~35 — that is where the
+38% went (main+q 21.7M → 36.3M).
| tail mechanism | result |
|---|---|
| root LMR (index-keyed), on the CASCADE | 234 / 29.66M (**−16% nodes**) but STS 1609 (−187); at fixed time WORSE than cascade alone |
| root razoring on `last_real` | 233 / 34.06M (−3.6%); largely REDUNDANT with root LMR (both together 227 / 29.56M ≈ LMR alone) |
| hybrid `PRESEARCH_OFF_FROM_DEPTH` | d4 spiked to 249/32.1M but **d3=233, d5=242, d6=238, d8=240** ⇒ a SPIKE, not a plateau; STS uniformly ~1671-1676 (−125) |
| **SF root depth decay** (`ENABLE_ROOT_ALPHA_DECAY`, new) | see 4 — inert with razoring on, HARMFUL with it off |
✅ **Root razoring is worth 14.5% of nodes at NO accuracy cost** (razor off: 251 / 40.44M / STS 1784 vs
baseline 250 / 35.31M / 1796 — both suites inside their floors). **Abandoning the tail beats searching it
shallower, in our engine.**
🐛 `OFF_FROM_DEPTH=3` came back byte-identical to full pre-search-off — a good instrument sanity check.

## 3b. ⭐ WHY THE PRE-SEARCH IS LOAD-BEARING — DECOMPOSED (this answers the standing question)
`PRESEARCH_TAIL_MODE=3` is a built diagnostic that keeps the WARMING channel (full tail search → TT /
killers / history) but overwrites the root SCORES with the fill value. Running it splits the pre-search's
value in two for the first time:
| config | WAC | nodes | STS | has |
|---|---|---|---|---|
| baseline | 250 | 35,310,778 | **1796** | scores + warming |
| **`TAIL_MODE=3`** | 246 | 42,730,538 | **1726** | **warming only** |
| cascade (mode 2 + table) | 243 | 36,259,573 | 1714 | ordering only |
| pre-search off, unsorted | 246 | 48,760,840 | 1684 | neither |
⇒ **Warming is worth +42 STS (1684→1726); SCORES are worth a further +70 (1726→1796).** ~⅔ scores, ⅓
warming, and **both are real** — every arm before this removed both at once, which is why no single
replacement ever got back to baseline.
★★★★ **AND THIS IS WHY IT IS THE BEST CONFIGURATION WE HAVE**: `TAIL_MODE=3` pays the FULL pre-search cost
(12.55M nodes) to buy only the warming half and lands at 42.7M — worse than baseline on both axes.
**The pre-search is efficient because ONE search yields BOTH channels. Splitting them costs more, not less.**
It is an economy of scope, not a mysterious dependency.

## 3c. ✅ `ROOT_PRESEARCH_REDUCTION=2` — the session's cleanest node saving (PARKED)
The knob was "invalidated" long ago by the shallower-overrides-deeper fault. `last_real_depth` was built
AFTERWARDS to carry depth with the value, so the closure deserved a re-test. Re-tested:
| `REDUCTION` | WAC | nodes | STS |
|---|---|---|---|
| 1 (baseline) | 250 | 35,310,778 | 1796 |
| **2** | 236 | **30,490,101 (−13.7%)** | 1746 (−50) |
| 3 | 219 | 29,685,309 (−15.9%) | — |
✅ **Monotonic in both directions = a real trade curve, not a spike** (contrast the `OFF_FROM_DEPTH` hybrid:
233/**249**/242/238/240). The old closure was right that the fault BITES — it is what costs R=3 its 31
solves — it was simply closed at the wrong setting. R=2 stays under it.
⚠️ With `ENABLE_ROOT_TABLE=1` the trade INVERTS (251 WAC / −3.5% nodes / STS **1658**) and the suites
disagree hard: WAC prefers the table, STS prefers no-table. The table alone with a full pre-search is bad
(243 / 39,989,229 = **+13.2% nodes**), independently confirming the 14 closed root-table arms.
☠️ **FIXED TIME KILLS IT**: baseline 258 WAC / 1777 STS vs R=2 **257 / 1680**, and R=2 searches 3.3% FEWER
nodes at equal time (lower NPS — the node mix shifts toward more expensive main-search nodes).
⇒ −13.7% ≈ 0.28 ply, under the 0.5-ply bar, exactly as the sub-35% law predicts. **PARKED, not shipped.**
🐛 My depth-gate coupling hypothesis was WRONG: `ROOT_RAZOR_MAX_DEPTH_DEFICIT=4` was byte-identical to the
default at R=3, so razoring was never refusing the shallower evidence.

## 4. ☠️★★★★ SIX OCCUPANCY FAILURES AND THE TRANSFER MODEL
Each technique is sound in its home engine and fails here because **a different component already occupies
the role or consumes the output differently.**
| technique | why it failed here |
|---|---|
| TT-move ordering | REDUNDANT — `updateMoveCacheForBetaCutoff` (`cache_management.h:1124`) already promotes the cutoff move at every cutoff. Measured **0.005%** with pre-search off |
| TT-depth LMR guard | marker was strong (AUC 0.6275, 6.2σ) but the guard's mechanism was REFUTED by counters (bet-loss rate 7.28036% → 7.28154%) and its saving absorbed by IIR |
| IIR × TT-depth guard | SAME POPULATION (both key on "has this node been searched before") — 2×2 corner −7.93% vs IIR alone −7.90% |
| killer child-clear | **INVERTED** — since `PROTECT_KILLERS` shipped, killers are an LMR/LMP EXEMPTION device, not just ordering. Clearing starved it: wrong-reductions 0.731% → 0.871%, STS −72 |
| index-keyed root LMR | the index carried no information (list was bound-sorted / unsorted). Closed by fixed-node games −25.0 ±19.6 |
| **SF root depth decay** | OCCUPIED BY RAZORING. SF needs it *because* it never prunes root moves; we razor, so it has no population — **0.012% nodes** with razoring on, and **+1.4% nodes / −71 STS** with razoring off |

**THE SIX PRE-PORT QUESTIONS** (each would have predicted one of the above):
1. **OCCUPANCY** — what already produces this technique's output DECISION here, through which table?
2. **TRIGGER OVERLAP** — what live mechanism keys on the same condition or a proxy? Run the 2×2 FIRST.
3. **MARKER ≠ GUARD** — from the marker's AUC and the guard's *directly counted* breadth, what change in
   the mechanism's own counter is predicted? (Never infer breadth from a sampler.)
4. **CONSUMER AUDIT** — list every READER of the state. Does any use it for a job the giant's engine lacks?
5. **PREREQUISITES** — check the STORE SITE of every input, not the flag.
6. **COMMENSURABILITY** — same depth, same units?
Verdict classes: REDUNDANT · INVERTED · STARVED · INCOMMENSURATE · COMPLEMENTARY.

## 5. WHAT ELSE THE SF READ FOUND (unbuilt, ranked)
- **SF does NOT order its root loop by the root table.** The `(score, previousScore)` sort only decides the
  ttMove (`rootMoves[pvIdx].pv[0]`); everything after is the ordinary staged picker, ordered by HISTORY.
  Our tier 3 keeps move-gen order, which our own comment records as measured WORSE (−12 WAC / −44 STS).
- **`lowPlyHistory` (SF17+)** — ply-indexed history, weight `8/(1+ply)` so ply 0 gets full weight, refilled
  each `go`. SF's explicit near-root cross-iteration ordering memory. **We have no analogue** (nearest is
  the `moveFrequency` PV bonus). ▶️ best candidate for our tier-3 key.
- **`ttPv` bit** — one bit per TT entry ("is or has been on the PV"), drives SF's largest LMR reduce-less
  term (SF11 annotates `r -= 2` as "~10 Elo") and gates RFP. **Our `TTEntry` has no such bit.** Needs
  `ENABLE_NODE_TT`. ▶️ measure the marker's AUC via `ENABLE_SHADOW_EVENTS` BEFORE wiring any guard.
- **Node-type flags (cutNode/allNode)** — verified ABSENT from our engine (the only grep hit is a comment).
  SF's biggest LMR terms key on them. An unoccupied slot.
- 🐛 **Our aspiration retries re-run the ENTIRE pre-search** (`alpha_beta` calls `reorder_legal_moves`
  unconditionally at `:3080`; the aspiration loop calls `alpha_beta` per attempt). SF re-searches with the
  same list. ▶️ split `g_presearch_nodes` by attempt index to size the waste.
- SF never truncates its root list on a fail-high/fail-low; ours does (D4, the `5.6 vs 13.7` starvation).

## 6. ▶️ WHERE THE LEAP IS (not search)
Every margin is sized for eval noise: `RFP_MARGIN` 1500 mp/ply, futility {200,450,650,950}, razor base 300.
Our corpus error is **245.5** vs SF11 **95.3** and SF18-static **68.9**, so we carry margins ~2.5× wider
than SF11 needs. **Wide margins ⇒ less pruning ⇒ bigger tree. That is the 30M barrier.**
This is the owner's own 2026-07-01 finding ([[eval-accuracy-payoff-is-pruning]]): *"a truer eval RAISES the
pruning ceiling... the eval's job at high strength isn't to out-guess the search — it's to make the search
CHEAPER."* Evidence: the capg70 bundle pruned **−10.6% nodes** from a cleaner eval alone.
★★★★ **MEASUREMENT RULE: gate `eval improvement + margin re-sweep` COMBINED, never a standalone eval SPRT**
— the pruning tranche only unlocks after re-sweeping FUTILITY / razor / VERIFY / LMP / ASPIRATION.
Absolute anchors: STS 54.9% vs SF18 79.3% at equal depth; **SF18 reaches our strength at ~600 nodes** vs our
~120,000 per position. Eval CONSTANTS are spent (−85.6 Elo fit, val/train 0.99); eval STRUCTURE is not
(`KS_FLOOR=13 > KS_KNEE=12` ⇒ the quadratic danger band has never been reachable — a SHAPE defect).

## 7. BUILT THIS SESSION (all default-off, byte-identity verified)
`ENABLE_PRESEARCH_OFF_SORT` + `PRESEARCH_OFF_SORT_MODE` · `PROTECT_TT_DEPTH` + `TT_GUARD_DEPTH_MARGIN` ·
`ENABLE_LMR_COUNTERS` (`[lmr_bets]` applied/researches/lost_pct) · `ENABLE_ROOT_BEST_REQUIRES_ALPHA` ·
`ENABLE_KILLER_CHILD_CLEAR` · `ENABLE_ROOT_ALPHA_DECAY` + plies/max-depth · TT markers in `[SHADOWEV]`
(`tth`/`ttdge`/`ttm`/`ttd`/`tts`) + `_shadow_auc.py` ranking both polarities.
🐛 **FIXED**: root TT stored `depth_limit` instead of the maintained-but-ignored `searched_depth`
(`search_engine.cpp:3479`) — a reduced root scout was writing a full-depth claim. Byte-identical today.
⚠️ **KNOWN, UNFIXED**: `:3517` takes `best_move` on `score > best_score` without `score > alpha`; real on
aspiration fail-low passes but currently unreachable (its only consumers are `ROOT_LMR_EXEMPT_BEST`, off,
and timed-out iterations, already guarded). `ENABLE_ROOT_BEST_REQUIRES_ALPHA` fixes it, measured INERT.

## 8. MEASUREMENT LESSONS
- ★★★★ **Every AUC is conditional on the population it was measured in.** `killer_or_cm` collapsed
  **0.750 → 0.4992** after `PROTECT_KILLERS` shipped — the guard removed its own marker from the reduced
  population. **Re-measure the whole map after every ship.**
- ★★★★ **A strong marker does not imply a useful guard.** `ttdge` was the best-evidenced marker the map has
  produced (6.2σ, replicated, LMR-specific, SF-consistent direction) and its guard still failed on every axis.
- ★★★★ **WAC alone sold three false results tonight** (the "free" sort, the d4 spike, the tail arms). It is
  solve-sanity ONLY for root work. **STS is the accuracy judge; the quiet corpus is the node judge.**
- ★★★ **A swept knob needs a PLATEAU.** `OFF_FROM_DEPTH` 233/**249**/242/238/240 is a spike ⇒ noise.
- ★★★ **Never infer a denominator from a sampler.** The `ttdge` breadth was mis-estimated twice
  (13.8% → 2.4% → actually **0.17%**) because the shadow sampler sits in an `else if` branch.
- ⚠️ **Byte-identity CANNOT see NPS cost.** Two counters were left as unconditional increments in the
  per-move helpers for part of this session; now gated at the increment. An unguarded `getenv` in a hot loop
  once cost 7.5% peak NPS at unchanged node counts. **Any per-move probe needs a `wac_speed` re-check.**
- 🐛 WSL `/tmp` is tmpfs and is wiped when the VM idle-shuts-down between calls — **copy diagnostic output
  to the scratchpad in the SAME invocation** (this destroyed one collection).
- 🐛 The `wac` sub sends engine stderr to `/tmp/wac_<tag>.err` and echoes only a SUMMARY, so
  `… 2>&1 | grep` captures nothing and exits 1 — which reads as a failed run.

## 10. ☠️★★★★ THE VENUE FINDING — WAC's NODE COLUMN REVERSES SIGN (the arc's biggest result)
Same build, same fixed depth 10. WAC300 total nodes vs `depth_nps_bench --n 60` MEDIAN nodes/position on
`cploss_corpus.csv`. Baseline: WAC 35,310,778 · **quiet 249,014**.
| config | WAC nodes | quiet nodes | agree? |
|---|---|---|---|
| `ROOT_PRESEARCH_REDUCTION=2` | −13.7% | **−22.2%** | ✅ amplified |
| cascade mode 2 | +2.7% | −1.5% | ❌ REVERSED |
| mode 3 + static fill | −5.8% | +13.6% | ❌ REVERSED |
| mode 4 (`prev_score`) | +3.0% | +16.0% | ✅ worse |
| `CONT_HIST_PIECE_KEY` | −6.14% | **+10.3%** | ❌ REVERSED |
| `PROTECT_TT_DEPTH` | −6.14% | **+2.9%** | ❌ REVERSED |
⇒ **FOUR OF SIX REVERSED SIGN.** WAC's node column is not weak here, it is close to uninformative.
**WHY**: WAC's previously-scored root prefix is **7.4%** of the root list; on quiet positions it is **24.6%**.
Anything touching root ordering, the root table, or carried-forward evidence works on a ~3× smaller
population on WAC than in real play.
⭐ **THE VALIDATING CASE**: `CONT_HIST_PIECE_KEY` was carried for weeks on its −6.14% WAC saving and measured
**−0.6 ±12.6 over 4,000 games** — an unexplained null. The quiet corpus says it **ADDS 10.3%**. The games and
the quiet corpus AGREE; **WAC was the outlier**, and a 4,000-game result is now explained.
▶️ **A WAC-only node claim is UNVERIFIED, not confirmed** — including several already quoted as closures.
⚠️ The direction is NOT predictable (R=2 amplified, the guard reversed) — measure per config.

## 11. ✅ `ENABLE_IIR=1` — THE VENUE-CORRECT CANDIDATE (the one to game-test)
| arm | WAC | quiet nodes | STS | depth@1s |
|---|---|---|---|---|
| baseline | 250 | 249,014 | 1796 | 12 (mean 12.2) |
| **IIR alone** | 245 | **209,278 (−16.0%)** | see §12 | **13 (mean 12.6)** |
| `R=2` | 236 | 193,782 (−22.2%) | 1746 | 12 (12.6) |
| R=2 + IIR (corner) | 233 | 191,534 (−23.1%) | 1697 | 13 (12.6) |
| `PROTECT_TT_DEPTH` | 254 | 256,208 (+2.9%) | 1778 | — |
| IIR + guard | 252 | 246,780 (−0.9%) | 1802 | — |
☠️ **THE BUNDLE FAILED**: the corner (−23.1%) ≈ `R=2` alone (−22.2%) ⇒ IIR contributes **1%** on top of R=2;
they are REDUNDANT, not disjoint, and the corner costs 17 WAC solves. My mechanistic disjointness argument
("R=2 is the root pre-pass, IIR is interior") predicted ≈−34.6% and was WRONG.
☠️ **The guard is worse than redundant**: adding it to IIR takes the quiet saving **−16.0% → −0.9%**. On WAC
the pair merely looked redundant — the ANTI-additivity is only visible on the correct venue.
✅ **IIR alone is the best profile of the arc**: −16.0% quiet nodes, **+1 median ply at equal time**, WAC
inside its floor. ★ I dismissed it early off its WAC number (−7.90%), which understated the real saving by
half — the venue finding's first casualty was my own triage.

## 12. THE AUDIT — 3 SELF-INFLICTED MEASUREMENT BUGS (all fixed, all verified by count)
1. 🐛 **The `[SHADOWEV]` TT probe and the `PROTECT_TT_DEPTH` guard keyed `current_state` (the PARENT) with
   the CHILD's zobrist** — the only 2 of ~31 probes to do so. `make_move_cache_key` XORs castling rights +
   ep square in, so they read a different slot than the search. Fixed to `updated_state`.
   ⇒ `ttdge` SURVIVED and strengthened: **AUC 0.6413 (~6.9σ)**; guard WAC 251 → 254.
   ★ **My 7.6% `tth` "sanity check" passed the BROKEN probe** — a plausibility check on a probe's OUTPUT
   cannot validate its KEY.
2. 🐛 **`replace_all` silently patched only the minimizer** for the killer child-clear (8-space vs 4-space
   indent). With BOTH plies clearing, the inversion is CONFIRMED and larger: wrong-reductions
   0.731% → **0.954% (3.0σ)**, STS 1757.
3. 🐛 **`_search_stability.py` reported a CRASHED arm as `0/0 (0.0%)`** — a failure that reads as perfect
   stability. Now surfaces `!! ARM FAILED` and skips.

## 13. ⭐ `last_proven` / `PRESEARCH_OFF_SORT_MODE=5` — BUILT 2026-09-03, **NOT YET MEASURED**
Modes 2 and 4 fail in OPPOSITE ways, which is why neither is usable:
| key | persistent? | proof-only? | failure |
|---|---|---|---|
| `last_real` (mode 2) | ✅ every store | ❌ fail-soft BOUNDS | width-dependent order ⇒ **the 8...Nd5 blunder** |
| `prev_score` (mode 4) | ❌ resets per call | ✅ proof-or-sentinel | too SPARSE ⇒ **+16.0% quiet nodes** |
| **`last_proven` (mode 5)** | ✅ inherited | ✅ only under `if (proven)` | ▶️ **UNMEASURED** |
`RootScore::last_proven` / `last_proven_depth`, written inside `root_table_store`'s `proven` branch,
inherited in the pre-size block beside `last_real`, consumed by mode 5 with **depth as primary key** —
preserving the owner's never-compare-across-depths invariant, now over PROOFS instead of BOUNDS.
✅ Field verified INERT when off: baseline reproduces **250 / 35,310,778 / 3.784** exactly.

### ☠️ MEASURED — MODE 5 IS A NULL, AND THE SPARSITY HYPOTHESIS IS REFUTED
🔬 Blunder FEN (validated against the PGN: `r1b1kb1r/pppp1ppp/5n2/1N6/4q3/5N2/PPPP1PPP/R1BQ1K1R b kq - 1 8`,
the position after 8.Kf1) at `PRESET=STANDARD MAX_DEPTH=64` — the TIME-BASED ~d20 regime.
| arm | delta 500/1000/1500/2000 | quiet nodes | WAC |
|---|---|---|---|
| mode 2 | `f6d5` / `f6d5` / `e4c6` / `e4c4` ☠️ | 245,259 (−1.5%) | 243 |
| mode 4 | `e4c4` ×4 ✅ | **288,842 (+16.0%)** | 242 / 36,355,504 |
| **mode 5** | `e4c4` ×4 ✅ | **288,842 (+16.0%)** | 242 / **36,297,562** |
✅ **LIVENESS PROVEN**: mode 5 and mode 4 score the same solves but differ by 58k WAC nodes on a
DETERMINISTIC bench ⇒ the code path really differs; the knob is wired, it just does not help.
☠️ **It recovered NOTHING** — the identical 288,842. `last_proven` IS strictly denser than `prev_score`
(which resets EVERY ASPIRATION ATTEMPT — `alpha_beta` rewrites it at its tail, `:3274`, once per attempt),
so the revival tier genuinely repopulated and the node cost did not move.
⇒ **The +16% is INTRINSIC to ranking the revival tier on PROOFS**, not an artifact of how many moves carry
one. Mode 2's bound+depth order is BETTER for node count while being WRONG for move choice — the two
properties are in genuine tension.
☠️ **The lesson I wrote here first ("build the field you actually want") did NOT pay off.** The field was
cheap, correct, and strictly better on the property I blamed — and changed nothing. ★ **Revised: before
building a missing field, ablate the EXISTING keys to prove the property you are blaming is the one that
costs.** That would have shown the cost tracks proof-ranking, not density, for free.
▶️ **Lane status: the presearch-removal lane stays CLOSED.** Best cascade arm is mode 2 at −1.5% vs
baseline, carrying a width-dependent blunder. There is no configuration of it worth a games night.
🐛 ⚠️ **`ourmove`/`fenvs` hardcode `MAX_DEPTH=12`** — passing `PRESET=STANDARD` alone leaves the depth cap
on and the blunder does NOT reproduce (mode 2 picks `e4c4` at d12 at every width). **Pass `MAX_DEPTH=64`
too, or the probe silently tests the wrong regime.** This is the documented trap and it caught me twice.

## 14. ⭐★★★★ THE REFERENCE MEASUREMENT — the gap is ⅔ NODE EFFICIENCY, ⅓ JUDGEMENT
🧰 NEW `diagnostics/_sts_reference.py` scores SF on **our** `sts300.epd` with **our** scoring (reuses
`sts_test.load_sts_epd`), so the SF gap is now REPRODUCIBLE instead of a quoted anchor.
| engine | @ fixed depth 10 | @ 249,014 nodes (our quiet median) |
|---|---|---|
| SF18 | 2414 (80.5%) | **2605 (86.8%)** |
| **SF11** (last fully-classical = the HCE ceiling) | **1983 (66.1%)** | **2374 (79.1%)** |
| **OURS** | **1796 (59.9%)** | 1796 (59.9%) |
| ours + `ENABLE_IIR` | 1818 (60.6%) | — |
**⇒ THE DECOMPOSITION:**
- **equal DEPTH: we are 187 pts / 6.2pp behind SF11.** Per-ply judgement is much closer than assumed.
- **equal WORK: we are 578 pts / 19.2pp behind SF11.** SF11 turns the SAME 249k nodes into several more
  plies (its own score climbs +391 pts / 13pp from d10 to that budget).
- ⇒ **~⅓ of the practical gap is judgement, ~⅔ is NODE EFFICIENCY.** This is the 30M barrier measured
  directly rather than argued, and it sizes the EBF prize at **~390 STS points — 2.6× the ±150 floor.**
✅ SF18 @ d10 = 80.5% REPRODUCES the record's long-quoted "79.3% at equal depth" anchor.
⭐ **SF11, not SF18, is the right target for this roadmap** — 90.6% of SF11 at equal depth is closeable;
618 points behind SF18 is not.
🔬 **Tension worth chasing**: our corpus error is 245.5 vs SF11's 95.3 (**2.6×**) yet the equal-DEPTH STS gap
is only 6.2pp ⇒ SF11's static-accuracy advantage does NOT convert proportionally into positional move
choice. More evidence that corpus error is the wrong objective ([[corpus-fit-is-anti-correlated-with-elo]]).
⚠️ `--depth` is NOT equal work; `--nodes 249014` is. Quote which one.
▶️ **This REFINES §6.** §6 says the leap is not in search because margins are sized for eval noise. Both hold
— the theory is that eval accuracy BUYS pruning — but the payoff now has a measured size and a venue, and
the node-efficiency term is the larger half. ★ It also means an eval fix must be gated **with** a margin
re-sweep to collect the tranche (§6's own rule), because that is where two thirds of the gap lives.

## 9. MY ERROR LOG (7 wrong direction calls, all caught by measurement, none by reasoning)
guard mechanism (refuted by counters) · TT coverage hypothesis (refuted) · `synthetic_from` inertness
(backwards) · "sorting is free" (WAC-only; STS refuted) · razoring > root LMR (backwards) · `:3517` ranked
as shipped-engine-critical (inert) · the d4 hybrid (spike, not plateau).
★ The results that survived came from the OWNER's reframings — that the pre-search's job is REVIVAL of
razored-away moves, that its consumers must be replaced rather than removed, and that testing in isolation
was hiding the system — plus count-based instruments. **Treat my hypotheses as candidates, not conclusions.**
