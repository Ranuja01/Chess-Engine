# The iteration-to-iteration carry-over system (map, 2026-09-02)

**Why this exists.** Every presearch/root experiment so far has moved ONE knob and read ONE bench. But the
root is a *system*: several producers write state at iteration N, several consumers read it at N+1, and they
do not share a convention for what a root move's "value" or "position" means. This maps the whole surface so
changes can be reasoned about jointly rather than piecemeal.

**Objective (owner's plain statement, 2026-09-02):** *protect good nodes, remove bad nodes* — reach the
giants' ordering quality so the pre-search becomes unnecessary, lowering EBF while holding accuracy.
⚠️ Two claims must stay separate. **(A)** "the giants pay no pre-search nodes and still order better, from
main-search nodes alone" — TRUE, and it is the GOAL. **(B)** "therefore removing ours saves nodes today" —
FALSE, it costs +38% ([[presearch-economics-38-percent-of-nodes-returning-2-to-1]]). Measuring B says
nothing about A. A is not refuted by anything measured to date.

---

## 1. The carry-over surface: producer → state → consumer

| # | state | written by | read by | knob |
|---|---|---|---|---|
| 1 | `moves_list` (full ordered root list, length N) | presearch `reorder_legal_moves` + root sort | root loop order; **root LMR's `i`** | `ENABLE_ROOT_PRESEARCH` |
| 2 | `top_score` (sentinel when unproven) | this iteration's root search | sort level 1 | — |
| 3 | `prev_score` (SF `previousScore`) | last iteration's `top_score` | sort level 2 | `ENABLE_ROOT_TABLE` |
| 4 | `last_real` + `last_real_depth` | last search that PROVED a value | **root razoring** (with `ROOT_RAZOR_MAX_AGE`) | `ENABLE_ROOT_TABLE` |
| 5 | `synthetic_from` (real/fill boundary) | how far real scores reach | razor guard **and root LMR's gate** | `PRESEARCH_OFF_FILL` |
| 6 | `second_moves` / `second_scores` (ply-1 replies) | `minimizer` | next iteration's ply-1 ordering hint | — |
| 7 | aspiration window centre | previous iteration's returned score | next iteration's `alpha`/`beta` | `ASPIRATION_DELTA` |
| 8 | TT (score/depth/bound; `move` only under NODE_TT) | every search | cutoffs; ordering **(measured null)** | `ENABLE_NODE_TT`, `ENABLE_TT_MOVE` |
| 9 | move-gen cache + promoted best move | `updateMoveCacheForBetaCutoff` at EVERY cutoff (`i != 0`) | move-list build; lazy re-sort | `ENABLE_LAZY_RESORT` |
| 10 | history / killers / counters / cont-hist | cutoffs | ordering, LMR `statScore`, LMP | many |
| 11 | `root_prev_best` | last iteration's best move | root-LMR best exemption | `ROOT_LMR_EXEMPT_BEST` |

★ #9 is our de-facto best-move memory, and it already works. Adding a SECOND such channel via the TT
measured **0.005%** with the pre-search off ⇒ ordering is NOT the pre-search's contribution.

---

## 2. Where the conventions actually conflict (the defects)

**D1 — index vs score.** Root LMR keys on **list index `i`**; the root table re-sorts by `prev_score`.
Turn both on and `i` no longer means what the reduction schedule assumes.
**Measured 2026-09-02 (WAC, nopre+norazor):** table alone **246** · root-LMR alone **205** · **both 162**.
The table is free alone and *harmful* on top of root LMR ⇒ a real antagonism, not noise.

**D2 — two notions of "this move's value".** The sort ranks on `top_score`/`prev_score`; razoring reads
`last_real`+`age`. A move can be ranked by one and pruned by the other. `ROOT_SORT_L2_LASTREAL` exists
precisely because `top_score`'s unproven sentinel makes the fail-low block degenerate (every chronic
fail-low move ties on both levels, so the stable sort freezes the old order).

**D3 — `synthetic_from` changes meaning when the pre-search is off.** Root LMR is gated on
`i < synthetic_from`, i.e. it only reduces moves with REAL scores — and real root scores are the
pre-search's product. So root LMR's *eligibility* is defined in terms of a partition only the pre-search
creates. ⚠️ I predicted this would make root LMR inert with the pre-search off; **measured wrong** — it
fires hard (205, −22.6% nodes). The dependency is real but does not disable it; needs re-derivation.

**D4 — aspiration retries truncate the table and starve the NEXT depth.**
From [[presearch-list1-sufficiency]]: a retry searches only **~5.6** moves and leaves whatever *it*
searched, so the next depth's first attempt inherits **~5.6 instead of ~13.7**. Arithmetic closes
(28×13.7 + 32×5.6 over 60 ⇒ 9.38 predicted vs 9.58 measured). The named fix is persisting the last
COMPLETED table across widenings — "a hard prerequisite, not hygiene" — which is exactly what
`ENABLE_ROOT_TABLE` provides ("all eight of `alpha_beta`'s exits leave a complete table rather than a
truncated stump"). ⇒ **#7 and #3/#4 are one defect, not two.**

**D5 — coverage.** `alpha_beta` clears the caller's scores and re-pushes only the moves it searched, so a
move scored at iteration 5 and razored at 6 loses its score permanently: **~34% root coverage vs SF's 100%**
(`search_engine.h:524-529`).

---

## 3. What tonight established

| arm (WAC d10) | solves | nodes | vs baseline |
|---|---|---|---|
| baseline (pre-search on) | 250 | 35,310,778 | — |
| nopre + norazor | 246 | 48,760,840 | +38.1% |
| nopre + norazor + table | 246 | 47,556,855 | +34.7% |
| nopre + norazor + **root LMR** + exempt | **205** | 37,748,252 | **+6.9%** |
| nopre + norazor + table + LMR + exempt | **162** | 35,877,346 | **+1.6%** |

⭐ **THE NODE GAP IS CLOSABLE.** A root loop that REDUCES instead of abandoning takes the no-pre-search
penalty from +38.1% to +6.9%, and to +1.6% with the table. The "find 38% elsewhere" framing was wrong —
root reductions close it directly.
☠️ **ACCURACY IS THE BLOCKER**, and it is now the whole problem: −41 solves from coverage, −43 more from D1.
⇒ Restated in the owner's terms: **"remove bad nodes" is SOLVED; "protect good nodes" is not.**

⚠️ 162 and 205 are catastrophic-ablation territory ⇒ **bug report, not verdict.** The header claims *"a
reduced scout that beats alpha is re-searched at full depth, so a reduction can never decide a move"*
(`search_engine.h:514`). Losing 41 solves at NORMAL node count is what a violated re-search property looks
like. Diagnose before concluding anything architectural.

---

## 4. Open questions, in dependency order

1. **Does the root re-search property actually hold?** Instrument root-level reductions the way
   `ENABLE_SHADOW_EVENTS` does interior ones: count reduced root moves whose honest score would have
   changed the pick. A count has no noise floor. **This is the blocker; everything else waits on it.**
2. **Fix D1**: make root LMR key on something order-invariant — e.g. the previous iteration's **score gap
   from best** (available free from iterative deepening, no pre-search, no index). ⚠️ HYPOTHESIS ONLY;
   check against the 14 closed root-table arms first.
3. **Re-derive D3**: what is `synthetic_from` with the pre-search off, and what should gate root LMR then?
4. **Then** re-run the bundle, and judge it on the QUIET CORPUS, not WAC.

## 5. Instrument rules that bind this lane
- **WAC is solve-sanity ONLY for root work.** Its scored prefix is 7.4% of the root list vs 24.6% on quiet
  positions, so root-level NODE effects invert between venues (this reversed one verdict twice in a
  session). **STS = accuracy judge · quiet corpus (`depth_nps_bench`) = node judge.**
- Node-savers are judged at **fixed TIME**; below ~35% they are Elo-neutral by construction.
- **A 2×2 judged on ONE suite is not a 2×2 result.**
- ★ **"It was tested" ≠ "it is closed"** — ask which REGIME, and whether the mechanism's own prerequisites
  were present. `nopre+norazor+ROOT_LMR = 181/300` was run with the table OFF and no best-exemption, i.e.
  SF's mechanism missing two of its three supports.
