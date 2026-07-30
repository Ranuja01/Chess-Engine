# Root pre-search — verified control flow and insertion points

Read first-hand from `reorder_legal_moves`, `descending_sort_wrapper` and `alpha_beta`'s root loop.
Everything below cites symbols, not line numbers. **Nothing here is implemented yet.**

Goal restated: **near-or-better ordering accuracy at materially fewer nodes**, and — because the node
distribution itself changes — a re-opened search-feature test matrix.

---

## 1. What actually happens today (verified)

```
reorder_legal_moves(alpha, beta, depth_limit, ..., previous_search_data, ...)
  moves_list = previous_search_data.moves_list        <- CARRIED FORWARD, not regenerated
              (else buildMoveListFromReordered)          when the previous table is empty

  if (!ENABLE_ROOT_PRESEARCH)  -> hard-off path (see §2, this is the structural template)

  move 0:  pre_minimizer(1, depth, alpha, beta, ...)     full window, seeds alpha
  i=1..N:  pre_minimizer(1, depth, alpha, alpha+1, ...)  scout
           if (alpha < score < beta) re-search full window
           addToSearchEvalCache(...)                     root-frame TT write per move
           alpha = max(alpha, highest_score)             <- tightens EVERY later scout

  returnData = previous_search_data                      (previous REAL scores)
  returnData.moves_list = moves_list
  descending_sort_wrapper(current_search_data /*pre-pass*/, returnData /*previous real*/)
```

`descending_sort_wrapper`:
- `common = min(main.scores.size(), main.moves_list.size())` — the previously-searched prefix.
- Max-scoring entry **within that prefix** is swapped to the front.
- `count = min(main.scores.size(), preSearchData.scores.size())` — normally the prefix length,
  because the pre-pass covers every move.
- Tail = `main.moves_list[1..]` paired with `main.scores[1..]` (previous REAL, `count-1` of them)
  **concatenated with** `preSearchData.scores[count..]` (pre-pass, for moves with no previous entry).
- **One `sortSearchDataByScore` over that concatenation.**

### Two confirmed consequences

1. **The mixing is real.** Previous-iteration real scores and shallow pre-pass scores are compared
   against each other in a single sort. This is the accuracy question.
2. **The pre-search's output for indices `< count` is DISCARDED in full** — score *and*
   `second_moves`. The wrapper reads `preSearchData.scores` only from `count` onward. That discarded
   work is measured at **38.4% of pre-search nodes**, and it is pure recomputation.

⚠️ Note what is *already* efficient and must not be "fixed": **the root move list is carried forward,
not regenerated.** The move-acquisition saving is already realised at the root; the prize is node count.

---

## 2. The hard-off path IS the implementation template

`!ENABLE_ROOT_PRESEARCH` already does exactly what a subset/lazy scheme must do:

```
reuse = min(previous_search_data.scores.size(), moves_list.size())
for i < reuse:      rd.scores.push_back(previous_search_data.scores[i])       // trust previous
for i >= reuse:     rs.top_score = 0
                    rs.second_moves = buildMoveListFromReordered(state_history, zobrist, 1, move)
                    rd.scores.push_back(rs)                                    // heuristic fill
```

**`buildMoveListFromReordered(..., 1, move)` is the cheap ply-1 ordering fallback, already written and
already in use.** This removes the main structural obstacle to laziness: we do not need to invent a way
to supply `second_moves` for a move we declined to pre-search — the call exists and is exercised.

**The invariant it protects** (documented in `reorder_legal_moves`, and load-bearing):
> `scores` must stay **full length**, and every `RootScore.second_moves` must be **non-empty and legal**,
> because `alpha_beta` indexes `scores[i].second_moves` for **every** root move with no bounds guard.

---

## 3. Insertion points, in dependency order

### A. `ENABLE_ROOT_SORT_SPLIT` — ☠️ BUILT AND MEASURED: **DOMINATED, closed**

| arm | WAC solves | WAC nodes | quiet nodes (d10) |
|---|---|---|---|
| gate OFF | **249/300** | **38,840,709** (byte-id ✅) | 198,943 |
| gate ON (naive trusted-first) | **233/300** | 41,294,984 (**+6.3%**) | 201,390 (**+1.2%**) |

**Worse on both axes, in both venues.** The predicted failure mode is what happened: trusted-first puts
*every* previously-searched move ahead of *every* unsearched one, **including previously-searched moves
that scored badly**. On tactical positions the winning move is frequently one the previous iteration never
searched, and the split shoves it behind ~9 trusted entries — which is why WAC (tactical) is punished ~5×
harder than the quiet corpus. **Today's single mixed sort gets this right by accident:** a genuinely bad
trusted score sinks below a promising pre-pass score, which is the correct ordering.

⇒ **Provenance alone is the WRONG sort key; bound type is the missing information.** A trusted EXACT or
LOWER-bound score deserves to outrank an unknown move; a trusted UPPER-bound (failed-low) score does not —
it is evidence of badness and belongs mixed in with the pre-pass group. **This promotes the bound field on
`RootScore` from bookkeeping hygiene to a hard prerequisite for the accuracy half of the design.**

The knob and `sortSearchDataRange` are kept (default off, gate-off byte-identical) as the vehicle for the
provenance-aware version — only the sort key needs replacing, not the plumbing.

### A′. `ENABLE_ROOT_SORT_SPLIT` — original rationale (superseded by the result above)
**Site:** `descending_sort_wrapper`, the single `sortSearchDataByScore(sub_data)` call.
**Change:** sort `[0, real_len)` and `[real_len, end)` separately, where
`real_len = clamp(mainSearchData.scores.size() - 1, 0, sub_data.moves_list.size())`.
Needs a new `sortSearchDataRange(SearchData&, lo, hi)` mirroring `sortSearchDataByScore`'s
index-permutation form.
**Risk: minimal.** Same moves, same scores, same lengths — only the order changes. The invariant is
untouched. Independently shippable.

⚠️ **Do not assume trusted-first is strictly better.** Under PVS a previously-searched move that failed
low carries an upper bound near alpha — it is *known bad*, and putting it ahead of an unknown move the
pre-pass rates highly is a downgrade. Today's mixed sort gets that case right by accident. If the naive
split measures flat or negative, the next form is **provenance-aware**: exact / lower-bound trusted
first, then failed-low trusted entries mixed with pre-pass entries — which requires a bound field on
`RootScore`.

### B. `ENABLE_PRESEARCH_SUBSET` — stop re-deriving the prefix
**Site:** the `i = 1..N` loop in `reorder_legal_moves`.
**Change:** for `i < presearch_prev_len`, skip both `pre_minimizer` calls and push a placeholder
`RootScore`. Safe *because the wrapper discards those entries anyway* (§1.2) — but only while
`ENABLE_ROOT_SORT_SPLIT`'s boundary and `count` agree, so **B must not ship without A's boundary logic
being correct**.
**Expected saving: ≈38.4% of pre-search nodes ≈ 16% of total.**

⚠️ **Three confounds, all real:**
1. **Alpha seeding.** The prefix pre-search raises `alpha`, which narrows the scout window for every
   *tail* move. Skipping it leaves alpha at entry value ⇒ wider tail scouts ⇒ **the tail gets more
   expensive**, partly cancelling the saving. Mitigation: seed alpha from the max of
   `previous_search_data.scores[..].top_score`. **Measure the tail's node count, not just the total.**
2. **Root-frame TT writes.** Each loop iteration calls `addToSearchEvalCache` at the root frame.
   Skipping the prefix drops those writes. Whether they matter is the warming question (§4).
3. **`updatePV` / `highest_score`** no longer see prefix moves; the returned `highest_score` changes
   meaning. Verify nothing downstream reads it as "best root score".

### C. `PRESEARCH_TAIL_FILL` — the lazy half, WITHOUT restructuring
**Site:** the same loop.
**Change:** for `i >= presearch_prev_len + CHUNK`, skip `pre_minimizer` and use the hard-off path's
fill (`top_score` seeded from the trusted floor rather than 0 — see the `top_score = 0` artifact below,
plus `buildMoveListFromReordered(..., 1, move)`).

**This is the important simplification:** true on-demand laziness would require `alpha_beta`'s root loop
to call back into the pre-search, which is invasive. But we measured that **47.8% of deep iterations
never leave the prefix at all** and a chunk of 4 covers **70.7%**. A fixed `prefix + CHUNK` window
captures nearly all of the available saving with **no restructuring and no callback** — the only loss
versus true laziness is that in the ~21% of iterations running 8+ past the edge, those moves are
heuristically ordered instead of pre-searched. Which is precisely what the warming/ordering ablation is
measuring anyway.
**Expected saving: most of the ≈54% never-read share.**

⚠️ **`top_score = 0` is a razoring hazard, not a neutral placeholder.** `alpha_beta` razors on
`alpha - scores[0].top_score > razor_threshold`, and the historical `ENABLE_ROOT_PRESEARCH=0` "57/300
collapse" was this artifact, not a measurement of the technique. Fill from the trusted floor.

### D. Persist the completed root table — **measured prerequisite**
**Site:** `get_engine_move`'s aspiration loop, mirroring the existing `completed_move` /
`completed_score` pattern: snapshot `preliminary_search_data` when an attempt resolves in-window;
restore before each same-depth retry **and before the `MAX_WIDENINGS` fallback**.
**Why it is a prerequisite, with numbers:** a retry searches only ~5.6 root moves and a call hands
forward whatever *it* searched, so the next depth's FIRST attempt inherits ~5.6 instead of ~13.7
(predicted 9.38 vs **measured 9.58**). Under B and C that short prefix directly forces re-derivation on
the expensive iterations.

---

## 4. What must be measured BEFORE any of B/C

**`PRESEARCH_NO_WARM`** — suppress the pre-search's persistent writes (`addToSearchEvalCache`,
killer/countermove/history, `updateMoveCacheForBetaCutoff`) when `g_in_presearch`.
`pre_minimizer` recurses into the real `maximizer`, so the pre-search warms the very tables the main
search then consumes. Compare on **main-only FMC** — raw FMC is unusable, it counts the pre-search's own
subtree.

⚠️ **Bounded interpretation:** suppressing stores also kills the pre-search's *own* internal reuse.
Report `g_presearch_nodes` in the ablation arm: flat ⇒ the FMC delta is attributable to warming;
ballooning ⇒ entangled, and the result is **inconclusive, not negative**.

**`PRESEARCH_PLY1_DISCARD`** — keep root scores, discard `second_moves`/`second_scores`, let `minimizer`
order ply 1 natively. `updateMoveCacheForBetaCutoff` is an ungated position-keyed hash move that already
exists, so much of the ply-1 ordering may be free. If this is cheap, the "lists of lists" contract can be
dropped entirely and C gets much simpler.

---

## 5. Sequencing

| stage | gate | judged on | ships alone? |
|---|---|---|---|
| 0 | `PRESEARCH_NO_WARM`, `PRESEARCH_PLY1_DISCARD` | main-only FMC, `g_presearch_nodes` | no — ablations |
| 1 | `ENABLE_ROOT_SORT_SPLIT` | main-only FMC, solves, nodes | **yes** |
| 2 | persist completed table | RETRY `avg_prevlen` → FIRST's level | yes |
| 3 | `ENABLE_PRESEARCH_SUBSET` | `[node_split] presearch=`, **tail nodes separately** | with 1+2 |
| 4 | `PRESEARCH_TAIL_FILL` + CHUNK sweep | nodes, main-only FMC, overshoot hist | with 1+2+3 |
| — | latent UB fix | ships regardless of every gate | **yes** |

**Byte-identity (WAC 249 / 38,840,709) with every gate off, after every build.** One process per knob
setting — Config latches at extension init.

**Verification rule:** the mechanism counter must move as predicted *before* any games are spent.
For stage 3 that means `[node_split] presearch=` falling to ≈ the tail share; if it doesn't, the skip
isn't skipping what we think.

---

## 6. The re-test matrix (deferred, but the point of the exercise)

Every search feature in the 0-for-13 ledger was tuned against a node distribution in which the
pre-search is 41.3% of all nodes and warms the TT/killer/history. Changing that changes:
- what the main search finds already in the TT on first visit,
- the effective time-to-depth budget at fixed time,
- root-level LMR/LMP thresholds, which key on **move index** — and the root list's ordering provenance
  is exactly what B/C change.

⇒ **Re-testing ProbCut, `ENABLE_QCUT`, malus, `SEE_PRUNE_CAPTURES` and the LMR family after this lands
is justified** — but as a separate campaign with its own baseline, **not bundled into these gates**.
Bundling would make a null result uninterpretable.

⚠️ Standing caution: search is **0-for-13**. A node saving is not Elo. Judge at fixed TIME (node-saving
changes flatter themselves at fixed depth), and via `gate` (SPRT, p1 = candidate) so the sign is
unambiguous. Same-session baseline arm — never compare a timed number across sessions.
