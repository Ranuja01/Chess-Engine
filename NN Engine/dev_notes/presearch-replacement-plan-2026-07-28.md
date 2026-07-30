# Root pre-search replacement — design plan (2026-07-28)

**Status: superseded for sequencing by `SESSION-HANDOFF-2026-07-30.md`. `ENABLE_ROOT_TABLE` is now BUILT
(2026-07-29) — see § BUILD LEDGER below.**

---
# § BUILD LEDGER — persistent root table (2026-07-29, uncommitted)

Baseline byte-identity re-confirmed after **every** build: **249 / 38,840,709 / EBF 3.934**.

## Knobs added
`ROOT_LMR_EXEMPT_BEST` · `ENABLE_ROOT_TABLE` · `ROOT_RAZOR_MAX_AGE` (default 1). All default-off/inert.
`RootScore` gained `prev_score`, `last_real`, `verified`, `age`; constants `ROOT_SCORE_UNPROVEN`,
`ROOT_AGE_NEVER`; counters `[root_table]` (slots / has_real / evidence_pct / proven_this_iter /
razor_stale_skips) and `g_root_lmr_exempt`.

## What the table does
Sized to the **full** root list before the loop, so every one of `alpha_beta`'s eight exits leaves a
complete, index-consistent table instead of a truncated stump — **the full-length invariant became
structural rather than a convention six sites had to uphold.** Entries are updated in place; nothing is
cleared or pushed. Razoring now consults `last_real`/`age` rather than a positional `synthetic_from`
cutoff, which is what makes skip-instead-of-break sound.

| step | config | solves | nodes | evidence | verdict |
|---|---|---|---|---|---|
| A1 | `ROOT_LMR_EXEMPT_BEST=1` (no table) | 243 | 37,756,371 | — | ☠️ **INERT** — bit-identical to exemption off; ex-best sorts to index 0, below `MIN_IDX`, so the guarded set is empty |
| B1 | table, store-only-verified | 252 | 45,450,867 (+17.0%) | 4.4% | ⚠️ razor starved (110,559 stale-skips) |
| B2 | + `last_real` split, fail-soft kept | 243 | 42,648,321 (+9.8%) | **92.8%** | ✅ mechanism confirmed; nodes still above base |

## Hybrid razoring arms (all with `ENABLE_ROOT_TABLE=1 ROOT_SORT_L1_LASTREAL=1`)

| arm | solves | nodes | vs base | reduced | skipped |
|---|---|---|---|---|---|
| **baseline** | 249 | **38,840,709** | — | — | — |
| L1 sort only, age 1 | 247 | 41,595,710 | **+7.1%** ← best nodes of any table arm | — | — |
| hybrid, skip > 600 | **254** ← best solves ever | 43,915,144 | +13.1% | 6,909 | 58,460 |
| hybrid, deep (B3/D150) | 250 | 44,205,186 | +13.8% | 46,576 | 0 |
| hybrid, reduce everything | 248 | 44,093,560 | +13.5% | 46,974 | 0 |
| hybrid, skip > 200 deep | 246 | 46,208,019 | +19.0% | 2,149 | 62,875 |
| hybrid + pre-search OFF | 235 | 57,288,159 | +47.5% | 49,977 | 0 |
| hybrid + p-off + deep | 241 | 55,549,863 | +43.0% | 47,238 | 0 |

☠️ **`ROOT_RAZOR_TO_LMR` adds nodes BY CONSTRUCTION.** It was justified as converting full searches into
reduced ones; it does not. The branch sits inside `if (razorable && razor fires)`, so it only touches moves
that would otherwise have been **skipped** — converting cost 0 into cost>0. The `hybrid_reduced` /
`hybrid_skipped` counters showed which bucket was being touched, which is why the error surfaced in one run.

★ **Where the cost actually lives:** `razor_stale_skips` = **45k-58k moves per bench** whose evidence was too
stale or too shallow to prune on, which therefore fell through to a **FULL-DEPTH search**. `ROOT_STALE_TO_LMR`
reduces that bucket instead — the only branch in the design that can remove work rather than add it.

## ★ Findings
1. **Slot coverage vs score coverage.** "Ours 34%, SF 100%" was *slot* coverage. **SF's SCORE coverage is
   ~3% too** — fail-soft PVS proves a value only for alpha-raisers. Slot coverage is the real defect and is
   now 100%.
2. **The pruning line is MEASURED vs FABRICATED, not proven vs unproven.** A fail-low is a real fail-soft
   value; only fill values are fabricated. Porting SF's store-only-verified literally discarded legitimate
   bounds and starved the razor. SF can afford that because **SF never razors at the root**; we do.
3. **`ROOT_RAZOR_MAX_AGE` is now the live axis** — how many iterations a move may go unsearched and still
   be razorable. This is the "reasonably recent score" dial.

⚠️ Both solve deltas (+3, −6) are inside the **no-sign-information band** (<15 solves). Only nodes and
coverage are decisive here; neither arm yet meets the shortlist rule (coverage high **and** nodes ≤ base).

---
# § EXECUTION LEDGER (2026-07-29)

Baseline commit **`0f4edc8`**. Reference fingerprint: **249 solves / 38,840,709 nodes / EBF 3.934**,
`presearch=16,051,978`. Byte-identity = solves+nodes reproducing exactly.

| step | build | result | verdict |
|---|---|---|---|
| 0 baseline | — | 249 / 38,840,709 / EBF 3.934 · FMC 87.55% · presearch 16,051,978 · `[aspiration] windows=1499 fails=2256 fallbacks=235` | ✅ fingerprint reproduced |
| 1 FMC region split | byte-id ✅ | **main FMC 86.14%** (1,046,431/1,214,868) · pre FMC 89.79% (689,666/768,109) · **pre = 38.7% of all cutoffs** | ✅ arithmetic self-check passes (1,214,868+768,109 = 1,982,977) |
| 2 prefix/tail cross-tab | byte-id ✅ | **ALL prefix_pct = 29.93%** (4,803,075 prefix / 11,244,557 tail nodes) · **tail_frac = 92.87%** (160,945 of 173,301 root moves have NO previous entry) | 🚦 see GATE 1 below |
| 2b list-1 sufficiency (quiet, d10, n=60 / 92 deep iters) | byte-id ✅ 249 / 38,840,709 | **FIRST** fed 9.58 / searched 13.68 / within 38.3% / overshoot 5.67 / head_kept 68.3% — **RETRY** fed 5.22 / searched 5.63 / within 65.6% / overshoot 1.72 / head_kept 62.5% · overshoot hist `0→44 1→6 2→0 3→8 4→7 5-7→8 8-15→11 16+→8` | ✅ **47.8% of deep iterations never leave list 1**; chunk 4 covers 70.7%, chunk 8 covers 79.3% |

## ★ FINDING 2b — the retry starves the NEXT depth, and Step 9 is a measured prerequisite

A retry searches only ~5.6 root moves, and the table a call leaves behind is whatever *it* searched. So the
next depth's FIRST attempt inherits ~5.6 instead of ~13.7. The arithmetic closes: of 60 iterations, 32 had
retries, so `(28 × 13.7 + 32 × 5.6) / 60 = 9.38` predicted against **9.58 measured**.

⇒ **Persisting the last COMPLETED root table across widenings (Step 9) raises FIRST's fed prefix from 9.58
toward the ~13.7 the search already demands**, which should collapse the `o8-15` (11) / `o16+` (8) tail —
precisely the ~21% of iterations that would make a chunked lazy scheme expensive. This also explains the
earlier oddity that short tables dominate FIRST calls rather than RETRY: the starvation propagates forward
one iteration from where it is created.

☠️ **Falsified here:** the prediction that a fail-high leaves a 1-2 entry stump and that retries would be
the expensive case for laziness. Retries are fed 5.22 and search 5.63 — narrow but not degenerate, and they
stay *within* their prefix more often (65.6%) than first attempts do (38.3%).

⚠️ **Bench speed shifted between sessions:** `depth_nps_bench` now reports ~304k NPS reproducibly (two runs)
vs 454,674 on 07-27 at *identical* median nodes (198,943). Counter conclusions are unaffected; **cross-session
timed comparisons and any SPRT against a historically-calibrated baseline are void until reconciled.**

## ★ FINDING 1 — the published FMC was inflated; the reference number is now 86.14%
`g_fh_*` counted the pre-search's own subtree (it recurses into the real `maximizer`). Splitting by region:
the **pre-search produces 38.7% of all cutoffs**, at a *higher* internal FMC (89.79%) than the main search's
**86.14%**. Every prior ordering claim quoting 87.55% was measuring tester and testee together.
⇒ **Use 86.14% as the ordering baseline from here on.**

## 🚦 GATE 1 READING — 29.93%, the REDUCED-SCOPE band

| cell | calls | prefix_pct (NODE share) | prefix/tail MOVES |
|---|---|---|---|
| FIRST len8+ (designated gate cell) | 283 | **31.67%** | 4,145 / 7,409 |
| FIRST len3-7 | 437 | 14.70% | 1,925 / 15,043 |
| FIRST len1-2 (6.6M nodes, the biggest block) | 1,070 | 6.14% | 1,336 / 40,670 |
| RETRY len1-2 (most calls) | 1,718 | 70.42% | 2,281 / 66,552 |
| RETRY len8+ | 100 | 91.34% | 1,409 / 2,813 |
| **ALL** | — | **29.93%** | **tail_frac 92.87%** |

⇒ **Ceiling on the subset scheme = 29.93% of the pre-search = ~12% of total nodes.** Per the gate that is the
**25–50% band: proceed for fixed-node-bench value only; ~12% of nodes is invisible at ±23 Elo in games.**

⚠️ **`prefix_pct` is a NODE share, not a move share.** Only ~7% of root moves are reusable, but they are the
expensive ones (move 0's full window + re-searches), which is why they carry 30% of the nodes. A first
reading of this table as move-share is wrong.

## ★★ CORPUS CHECK — WAC inflates the stump rate, but the CEILING IS ROBUST

The gate above was measured on WAC, a tactical suite. Re-measured on 60 quiet game positions
(`depth_nps_bench.py` over the `cploss_corpus` game/neutral/collapse strata, `MAX_DEPTH=10`):

| metric | WAC (tactical) | quiet game positions |
|---|---|---|
| aspiration fails per window | **1.50** | **0.60** |
| `FIRST len1-2` calls (stumps) | 1,070 (dominant) | 140 |
| `FIRST len8+` calls (healthy) | 283 | **164** (now the larger) |
| `tail_frac` | 92.87% | **82.21%** |
| `prefix_pct` ALL | 29.93% | **38.50%** |
| `FIRST len8+` prefix_pct | 31.67% | **50.79%** |
| main-only FMC | 86.14% | **82.03%** |
| **pre-search share of ALL nodes** | **41.3%** | **26.5%** |

⇒ **WAC inflates root fail-highs** (1.5 vs 0.60 per window), so stumps dominated there and healthy tables
dominate in quiet play. On the designated gate cell the prefix share crosses the full-scope line
(31.67% → **50.79%**).
⇒ **BUT THE CEILING IS ESSENTIALLY THE SAME, because two effects cancel:**
**WAC 29.93% × 41.3% = 12.4% of total nodes · quiet 38.50% × 26.5% = 10.2% of total nodes.**
**The subset scheme is worth ~10-12% of total nodes on BOTH corpora.** Corpus-robust, not an artifact.

⚠️ **`tail_frac` is still 82% in quiet play** — most root moves genuinely have no previous-iteration data.
That was NOT a suite artifact. ⇒ the lever that could lift the ceiling past 10-12% is retaining **last
iteration's PRE-SEARCH scores** for the tail (`alpha_beta` currently refills `previous_search_data` from
main-search results only and discards them). Neither audit raised this; it is the owner's idea.
⚠️ **Main FMC is LOWER in quiet positions (82.03%)** than on WAC (86.14%) — ordering is harder where there is
no forcing answer, so ordering headroom in real games is larger than the WAC-derived figure suggested.

## ★★★ ROOT CONSUMPTION — the decisive data set (2026-07-29)

**How much of the root move list does the main search actually READ?** The pre-search scores every root
move; anything beyond the consumed count is never looked at — not recomputation, pure waste.
Measured with an RAII guard on all four `alpha_beta` return paths (a missed return would bias toward the
completed case). Deep iterations only (`depth_limit >= 10`); byte-identity held at 249 / 38,840,709.

| corpus / phase | avg root moves | **avg searched** | consumed | prefix | **best_in_prefix** |
|---|---|---|---|---|---|
| WAC OPEN/MID | 41.1 | **4.00** | 9.7% | 7.4% | 91.3% |
| WAC ENDGAME | 31.2 | 3.62 | 11.6% | 8.9% | 89.7% |
| WAC ADV_EG | 16.0 | 3.69 | 23.1% | 22.1% | 84.6% |
| **QUIET OPEN/MID** | 35.1 | **11.83** | **33.7%** | 24.6% | **94.0%** |
| QUIET ENDGAME | 21.0 | 2.11 | 10.1% | 13.2% | **100%** |

⚠️ **WAC UNDER-REPRESENTS ROOT WIDTH BY ~3×.** Its forcing tactics cut the root loop early: ~4 moves
searched vs **~12** in quiet play. The owner's own observation ("I generally see 10-20 moves searched")
matches the QUIET column and falsified the WAC-derived reading. **Do not size this work from WAC.**
⚠️ On WAC `avg_searched` is ~3.6-4.0 in EVERY phase — phase changes the denominator (list length), not the
numerator. That pattern does NOT survive to quiet play, where OPEN/MID searches 11.8 and ENDGAME 2.1.

### Revised prize (quiet OPEN/MID, where nearly all nodes are)
Pre-search node split by phase (quiet): OPEN/MID prefix **38.4%** / tail 61.6% · ENDGAME prefix 42.8%.
- **Reusable prefix** ≈ 38% of pre-search nodes → the SUBSET scheme (avoid re-deriving what we knew).
- **Never read at all** ≈ 54% of pre-search nodes → the LAZY scheme (avoid deriving what we never look at).
- **Complementary, not competing.** Combined ceiling ≈ 92% of pre-search nodes ≈ **24% of total nodes**.
⇒ **Lazy generation is the larger half and was not in the original plan.** It is the owner's idea.

### The safety number
**`best_in_prefix` = 94.0% (quiet OPEN/MID), 100% (quiet endgame), 85-91% on WAC.** The move finally chosen
is nearly always already inside the previous iteration's known set ⇒ deferring the TAIL's ordering rarely
changes the outcome. This is what makes lazy generation low-risk.

⚠️ **Unanswered and load-bearing:** are the ~12 searched moves the same ones the pre-search ranked top? If
the pre-search's ordering is *why* they are the right twelve, removing it degrades the selection this whole
argument rests on. Deferring scoring does NOT defer searching — the loop still walks the full list, so
unscored moves fall back to heuristic ordering. **Needs the warming/ordering ablation before any build.**

## ☠️ MECHANISM: two wrong calls, recorded so they are not repeated
1. **"Aspiration stumps starve the prefix"** — the reason the cross-tab was built. Partly right, but the
   cross-tab showed the starvation is worst on **FIRST** calls, not RETRY.
2. **"Root razoring's `break` truncates the table"** — **FALSIFIED BY PROBE.** `ROOT_RAZOR_CONTINUE=1` is
   **byte-identical**, counters included (`tail_frac=92.8702%` to the digit) ⇒ the razor branch never
   executes. (Memory already recorded `ROOT_RAZOR_CONTINUE` as "provably inert"; it should have been checked
   before an explanation was built on it.)
3. **Current best candidate: the ROOT BETA CUTOFF** at `alpha_beta`'s root loop (`if (beta <= alpha)`), which
   under a narrow aspiration window fires on a fail-high and ends the loop. **NOT asserted — needs a
   loop-exit-reason counter before anyone relies on it.**

**The gate number is mechanism-independent**, so the go/no-go decision does not wait on this.

## ★ FINDING 2 — aspiration retries are common, not rare
`windows=1499 fails=2256 fallbacks=235` — roughly **1.5 failed attempts per window**. The stump condition is
a normal operating state, not a corner case, which is why the cross-tab (class × prev-table-length) was
built before any gate was read.

---
# ★★★ STAGE 1 MAP (2026-07-28) — the concrete, reduced version. Build from this.

Owner's design: (1) split the tail sort into two lists, trusted-first; (2) pre-search ONLY the moves missing
from the previous iteration; (3) wire ply-1 data from the real search; (4) persist the last COMPLETED
iteration's table across aspiration retries; (5) guarantee non-empty `second_moves`.

## ✅ THE INVARIANT ANY SUBSET PRE-SEARCH MUST PRESERVE (verified — it is documented in the code)
`reorder_legal_moves`' hard-off path says it outright:
> *"Every root move needs a RootScore with a NON-EMPTY legal `second_moves` list, and the **scores vector
> must stay full-length (alpha_beta indexes second_moves per root move)** — so heuristic-fill any move the
> previous iteration razored away … `second_scores` may be empty (ascending_sort only needs moves >= scores)."*

Confirmed at the call sites: `alpha_beta` indexes `current_search_data.scores[i].second_scores/.second_moves`
for **every** root move with **no bounds guard** (the razor guard does not protect it).
⇒ **This is the one true UB risk of the stage.** A subset pre-search must still emit a full-length `scores`
vector with a non-empty legal `second_moves` per move. The `!ENABLE_ROOT_PRESEARCH` path is the template.

## Item-by-item
| # | verdict | where | invasiveness |
|---|---|---|---|
| 1 split sort | **correct** | `descending_sort_wrapper` — keep the front swap, sort `[1,count)` and `[count,end)` separately, concatenate trusted-first | ~20 lines, no struct change |
| 2 missing-only pre-search | **correct, well-sited** | `reorder_legal_moves` — "has previous entry" is exactly `i < previous_search_data.scores.size()` (index alignment guaranteed: scores are pushed in `moves_list` order until the razor break). **The `!ENABLE_ROOT_PRESEARCH` path is already this shape** — swap its heuristic fill for `pre_minimizer` | medium |
| 3 ply-1 wiring | ⚠️ **OVER-SCOPED — verify, do NOT build** | `ascending_sort` already sorts only `moves[0 : values.size())`, i.e. the searched prefix, leaving the unsearched tail untouched. **There is no ply-1 analogue of the root's mixed-tail sort**; shallow ply-1 scores never compete with deep ones. `out_entry` plumbing already carries real ply-1 data on the prefix | ~zero |
| 4 persist completed table | **correct, and a HARD PREREQUISITE for item 2** | `get_engine_du move`'s aspiration loop — snapshot `preliminary_search_data` when an attempt resolves in-window, restore before each same-depth retry (mirror the existing `completed_move`/`completed_score` pattern) | small-medium |
| 5 non-empty guard | **correct** | guard at the top of `minimizer`'s `cur_depth==1` branch, or fill at source in `reorder_legal_moves`. ⚠️ the early draw return at `cur_depth==1` exits **before** `out_entry.second_moves` is assigned ⇒ the hazard exists TODAY, independent of this work | ~10 lines |

## Byte-identity
| sub-step | byte-id? |
|---|---|
| all counters | ✅ YES |
| item 1 gated off | ✅ off / ❌ on — **even at reduction=1** (the merged sort currently lets a pre-pass score outrank a real one; the split forbids it) |
| item 2 gated off | ✅ off / ❌ on (by design — it saves nodes) |
| item 3 | ✅ (no code) |
| item 4 gated off | ✅ off / ❌ on (aspiration retries occur at default config) |
| item 5 guard | ✅ **iff** the incidence counter reads 0 on the bench — so count first, then guard |

## Is item 1 a prerequisite for item 2?
**Sort policy: no. Data plumbing: yes — they are the same restructure.** Once the pre-search covers only the
tail, `count = min(main.scores, pre.scores)` alignment is meaningless (pre entries no longer index-correspond
to the front). Build 1+2 as one code path with **two separate gates** so games can attribute them.

## ⚠️ Unsafe-alone combinations
- **item 2 without item 4** — an aspiration stump shrinks the "prefix" to 1-2 entries, so the subset
  pre-search covers nearly everything and the saving evaporates *precisely on the expensive iterations*.
- **lowering `ROOT_PRESEARCH_REDUCTION` without item 1** — shallow scores outrank deep ones in the merged sort.
- **item 2 without the full-length / non-empty invariant** — unguarded `scores[i]` indexing ⇒ UB.

## Pre-flight counters — (b) corrected, (d) added
- **(a) prefix-vs-tail node split** of `g_presearch_nodes`, attributed by `i < previous_search_data.scores.size()`;
  also log the missing-tail LENGTH per iteration. ✅ byte-id. **This is the go/no-go gate — it bounds item 2's
  maximum saving.**
- **(b) TT hit rate inside the pre-search** — ⚠️ **do NOT reuse the `tt_visits`/`tt_probes` delta**: `tt_probes`
  counts a slot hit (not a depth-sufficient usable hit), and `pre_minimizer`'s LMR arm probes the same position
  again incrementing `tt_probes` **without** `tt_visits`, so the ratio can exceed 100%. Use dedicated counters at
  `pre_minimizer`'s main probe site (probe / `using_tt` after `use_tt_entry`). ✅ byte-id.
- **(c) table-warming write volume** (TT stores, killer/history/countermove, moveGenCache promotions during the
  pre-search). ✅ byte-id — but **correlation only.**
- **★ (d) WARMING ABLATION — the premise test, and NOT byte-identical.** A gated variant that runs the
  pre-search but **suppresses its stores/updates**, compared on `g_fh_first/g_fh_total` and a fixed-node bench.
  Counter (c) cannot answer the causal question. **If warming carries a large share of the 87.55% FMC, item 2's
  saving is partly an illusion** (the tail you stop pre-searching also stops being warmed) and the stage must be
  re-scoped before the merge code is written.

## Recommended order
1. **Counters (a), (b-dedicated), (c), empty-`second_moves` incidence** → verify WAC 249 / 38,840,709 unchanged.
   Read (a): if the prefix share is small, the ceiling is low — **decide before building.**
2. **Ablation (d)** — the premise test.
3. **Item 5** (guard, informed by the incidence counter) — independent risk reduction.
4. **Item 4** (gated) — may be a small win alone, since retries currently rebuild ordering from a stump.
5. **Items 2+1 together, two gates, at reduction=1, item 4 ON** — expect ~neutral Elo with a node saving;
   a regression here means warming mattered (cross-check step 2).
6. **Lower `ROOT_PRESEARCH_REDUCTION`** with the split sort on — the payoff sweep, plateau-checked.

⚠️ **Validation reality:** at ±23 Elo (1200 games) steps 4 and 5 may individually read as noise. **The
fixed-node bench and the counter deltas are what will show the mechanism fired; games only gate against harm.**

---
# ★★ EARLIER AUDIT — SUPERSEDES §3 AND §4 OF THE ORIGINAL PLAN BELOW

## ✅ VERIFIED FIRST-HAND (not taken on trust): the merge ALREADY EXISTS
`descending_sort_wrapper` (`search_engine.cpp`) — its own comment states it:
> *"mainSearchData carries the previous iteration's full move list with its real searched scores (cutoff
> length ≤ N); preSearchData holds a fresh shallow pre-pass score for every move. **Keep main's first
> `count` real entries, fill the rest of the tail from the pre-pass**, then sort the tail."*

with `count = min(main.scores.size(), pre.scores.size())`, and `reorder_legal_moves` at k>1 doing
`returnData = previous_search_data;` then `descending_sort_wrapper(current_search_data, returnData)`.

⇒ **At iteration k>1 the previous iteration's REAL scores already win for the searched prefix; the
pre-search supplies only the tail.** Two consequences:
1. **The pre-search's prefix scouts have their scores DISCARDED every iteration.** That work is the
   candidate saving.
2. **Stage 2 of the plan below ("flip selection to prefer previous-iteration data") is already the current
   behaviour for the prefix.** The plan's §2-row-4 "freshness for scout-refuted prefix moves" is NOT what
   the code does — a scout-refuted prefix move keeps its previous fail-soft value.

## ★ The project restated
Not "replace the pre-search with previous-iteration data" (the prefix already works that way) but:
**stop paying for the prefix scouts whose scores are thrown away**, while keeping (a) tail refresh,
(b) second-level lists for tail moves, (c) whatever the discarded prefix work delivers via **side effects** —
`pre_minimizer` recurses into the REAL `maximizer`, so the whole pre-search subtree warms the same TT,
killer/history/countermove tables and moveGenCache promotions the real search then uses. **(c) is the
unmeasured wildcard, and our 87.55% FMC is partly a product of it.**

## ★ THE MOST LIKELY FAILURE MODE: the razored tail
Root razoring defaults to `break` (`ROOT_RAZOR_CONTINUE=false`), so once one stale-low move trips the razor
**every remaining root move goes unsearched that iteration** ⇒ those moves get **no** previous-iteration data
ever. The only thing re-scoring them at current depth is the pre-search. "Feed ordering from the previous
iteration" is **undefined for exactly this population**, which can be most of the move list. Ossifying root
pruning on stale shallow scores would be gradual and invisible to benches.
(This is also the population the owner's "pre-search only the subset" idea should target — the razored tail
and never-searched moves, NOT "bound-valued" moves, which already ride on previous data.)

## ★ STAGE 0 AS WRITTEN IS THE WRONG GATE — replace it
"Fraction of root moves with an EXACT previous score" is answerable in advance: under PVS every root move
except the PV is bound-valued, so coverage is ~1/N and a >70% gate fails trivially — falsely killing a plan
whose prefix half already runs on those bounds. **Use instead (all byte-identical counters):**
1. **Tail fraction** — `moves_list.size() - previous_search_data.scores.size()` per iteration (how much the
   razor `break` leaves unsearched).
2. **Prefix/tail split of the 41.3%** — a second `[node_split]`-style delta around the pre-search loop, split
   at `i == count`. **The prefix share is the theoretical maximum saving.**
3. **Bound class per retained root score** — needs an (unread, byte-id-safe) provenance field on `RootScore`.

**Gate: proceed only if the prefix share is large (>20 of the 41.3 points) AND the tail is not so dominant
that the tail pre-search must keep running at near-full cost.**

## ⚠️ Latent traps found (PASSED ON, not yet verified by me — check before relying)
- **Sentinel leak:** ply-1 `minimizer` with an empty `second_moves` falls through its loop, misses
  `is_checkmate`/`is_stalemate`, and returns `9999999 - plies` — a near-mate score. Any partial scheme must
  guarantee a non-empty legal list. (This is why pre-search depth is clamped ≥ 2.)
- **Time-up on iteration 1:** `reorder_legal_moves` returns the (empty) `previous_search_data`, and
  `alpha_beta`'s time-up path then indexes `moves_list[0]`/`scores[0]` on empty vectors. Reachable today via
  `NODE_LIMIT` tripping inside the depth-3 pre-search.
- **Order-of-operations:** `alpha_beta` selects the razor-threshold branch by
  `previous_search_data.moves_list.empty()` *before* clearing it two lines later. Moving the clear breaks
  first-iteration razoring silently.
- **ply-1 bypasses the promotion channel:** the real search's `cur_depth == 1` branch iterates
  `second_level_moves_list` directly and contains **no** `updateMoveCacheForBetaCutoff` call — so §5's
  "moveGenCache promotion already covers ply-1" only becomes true if ply-1 is switched to ordinary movegen,
  trading score-sorted ordering for heuristic ordering. A real testable question, not a freebie.
- **`second_scores` is used only for ordering** (the ply-1 razoring consumer is commented out) ⇒ droppable;
  `second_moves` is the load-bearing half.
- **Aspiration:** `previous_search_data` means "the previous `alpha_beta` CALL", including failed aspiration
  attempts. A fail-high attempt beta-cuts the root loop early and can leave a ONE-ENTRY "previous iteration".
  Today the re-run pre-search papers over this. Any rework needs a **monotone** root table keyed on
  `{score, depth, window, bound}` that a failed attempt cannot overwrite.

## Audit verdict
**Hard-adjacent moderate, and smaller than the plan implies** (Stages 1–2 largely re-build what
`descending_sort_wrapper` does). Upside is capped by the prefix share — unknown, measure first — and 41% off
still leaves ~36× the SF node count. Byte-identity dies at the first behavioural step, leaving games-only
validation at ±37 Elo/456g.
**Recommendation: run the three Stage-0 counters (cheap, byte-identical) because the prefix-share number is
worth having regardless — but do NOT sequence this ahead of the eval lane.** Prior evidence: the pre-search
is comparatively *efficient* per our own equal-node test (240 vs 222 solves), search is 0-for-12 this month
against +83 Elo from five eval ships, and this project's validation instrument is the most expensive we have.

---

## 1. Why

Measured, not estimated: the root pre-search (`reorder_legal_moves` → `pre_minimizer`) accounts for
**16,051,978 of 38,840,709 nodes = 41.3%** of everything we search (counter: `[node_split]`, a before/after
delta on the shared node counter across the call, so it includes its own qsearch).

It runs **every iteration** at `depth_limit - max(1, ROOT_PRESEARCH_REDUCTION)`, and **again on every
aspiration widening / full-window fallback**, because those re-enter `alpha_beta`.

Against Stockfish on identical FENs (`depth_race_vs_sf.py`), at nominal depth 10 we need **61.6× the nodes**
(133,090 vs 2,160) while being only **2.6× slower per node** — so the time-to-depth gap is dominated by node
count, not eval speed. Our per-ply growth is close to SF's (1.72 vs 1.50); we start ~17.6× higher at shallow
depth. **It is a fixed-overhead problem, and the pre-search is the largest identified component.**

### Why this is NOT the same as the node-cuts that failed today
Every node reduction tried on 2026-07-28 removed **information** and was punished: `LMR_EXTRA=2` (−55 Elo in
games), quiet-check safety filters (solves down), `MAX_QDEPTH` 8/6/4 (**worse on BOTH axes** — fewer solves
AND more nodes, because noisier leaves cost the main search more than qsearch saved).
The pre-search is different in kind: at iteration *k>1* it **re-derives data of the same nominal depth the
previous iteration already produced**. That is recomputation, not information. This is the one node-reduction
on the board with a mechanism story that does not require trading accuracy away.

## 2. What the pre-search actually provides (all four must be replaced)

| # | Output | Consumer | Replacement |
|---|---|---|---|
| 1 | Root move ordering (scores per root move) | root move loop | previous iteration's root scores (already retained in `previous_search_data`) |
| 2 | **Second-level move lists + preliminary scores** | `minimizer` at `cur_depth == 1` (its signature takes `second_level_moves_list` / `second_level_preliminary_scores`) | **OPEN — see §5** |
| 3 | Root-razoring reference scores | root razoring | previous iteration's scores — ⚠️ the `ENABLE_ROOT_PRESEARCH=0` fallback fills `top_score = 0`, and razoring then computes `alpha - 0 > threshold` and **breaks the entire root loop**. That artifact (57/300 solves) is why the existing off-switch is not a valid experiment. |
| 4 | **Freshness for scout-refuted moves** | ordering quality | ⚠️ genuine loss. Previous-iteration scores for scout-refuted moves are **bounds, not exact values**. This is the pre-search's only irreplaceable contribution and the whole reason a naive swap degrades. |

## 3. The core design: a quality-aware merge

Per root move, record `{score, depth, bound_type, source}` and select by quality rather than by origin:

- **EXACT** previous-iteration score at depth ≥ candidate → **prefer previous iteration**.
- Previous score is a **BOUND** (scout-refuted / fail-low) → prefer the pre-search value if present.
- **No previous entry** (first iteration, new move) → pre-search, else plain move-gen ordering.

⚠️ **This is why `ROOT_PRESEARCH_REDUCTION` was never a valid experiment:** a shallower pre-search produces
data *worse* than the previous iteration's, yet it still **overrides** it. The dial degrades the ordering
source while leaving it in charge. Any reduction must come *after* the merge, never before.

**Node savings only arrive once the pre-search stops running.** Building the merge alone saves nothing —
it is the precondition that makes skipping it safe.

## 4. Staged plan (each stage independently measurable; do not skip gates)

**Stage 0 — measure the gap the pre-search actually fills.** Instrument: of the root moves at iteration
*k>1*, how many have an EXACT previous-iteration score at sufficient depth vs a BOUND vs nothing?
*Byte-identical (counters only). Gate: if EXACT coverage is high (say >70%), the merge is worth building; if
most root scores are bounds, the pre-search is doing real work and this whole plan weakens.*
**This is the cheapest and most decisive step. Do it first.**

**Stage 1 — build the merge structure, preserving current behaviour.** Track depth/bound/source per root
move; selection still prefers the pre-search. *Gate: byte-identical (249 / 38,840,709). Commit.*

**Stage 2 — flip selection to prefer EXACT previous-iteration data.** Pre-search still runs, so **no node
saving yet**; this isolates the ordering-quality question. *Gate: FMC and solves hold (FMC baseline 87.55%).
If ordering degrades here, stop — the rest cannot work.*

**Stage 3 — stop running the pre-search where coverage is sufficient.** Keep it on iteration 1 (no prior
data, and that iteration is cheap). **This is where the 41% comes back.** *Gate: games (SPRT, candidate as
p1). Also re-check root razoring is fed real scores, not zeros.*

**Stage 4 — resolve the second-level lists (§5).**

**Stage 5 — re-tune.** ⚠️ **Blast radius:** LMR, LMP, futility, razoring and aspiration were all calibrated
against the *current* node distribution and ordering source. After Stage 3 those calibrations are stale.
Budget a re-tune campaign; do not judge the knobs on their pre-rework values.

## 5. Open question — are the second-level lists still needed?

`minimizer` special-cases `cur_depth == 1` to consume pre-search-generated child lists. But
`updateMoveCacheForBetaCutoff` (`cache_management.h:1033`) is an **ungated, position-keyed hash-move
mechanism** that already promotes the previous cutoff move to the front of the moveGenCache entry on every
revisit — i.e. we already have TT-move-quality ordering at ply 1 from a different channel.

**If ordinary move generation + moveGenCache ordering suffices at ply 1, the entire "lists of lists" problem
dissolves** and this becomes a moderate project instead of a hard one. Worth testing early and cheaply.

## 6. Risks / what would kill this
- **Stage 0 shows low EXACT coverage** ⇒ the pre-search is genuinely producing new information, not
  recomputation, and the premise fails.
- **Stage 2 degrades FMC** ⇒ previous-iteration ordering is materially worse; no node saving is worth it.
- **Byte-identity dies at Stage 2**, removing our only cheap regression guard for everything downstream.
- **The re-tune (Stage 5) could consume more time than the depth gain is worth.**
- Validation is games-only at ~±37 Elo per 456 games, so small regressions may hide.

## 7. Honest alternative
The eval lane has 5 ships and **+83.4 Elo** measured, with ~18pp of *classical* headroom at equal depth
(NNUE is worth only 2.7pp over SF15-classical). Search produced **zero Elo across ~12 attempts** on
2026-07-28. If forced to bet on the next 20 Elo, eval is the better bet.
This plan is worth doing because it is the first *quantified structural* inefficiency found — everything
else in search has been knob-guessing — but it should be sequenced honestly against that.

## 8. Questions for Fable
1. Is Stage 0's EXACT-coverage metric the right gate, and what threshold would you want?
2. Is the merge rule (prefer EXACT-and-deeper, fall back to pre-search on bounds) correct, or is there a
   standard formulation we should copy instead?
3. Can the ply-1 second-level lists be dropped in favour of ordinary movegen + moveGenCache promotion?
4. What does a reference engine do at the root that we are approximating with this pre-search — is the right
   end state "no pre-search at all", or "IID/IIR at the root only when the TT is cold"?
5. Given the Stage 5 re-tune blast radius, is this worth doing before the eval lane?
