# S3 root audit: our root today, against the giants, and a phased path by measurement (2026-11-13)

Input for the owner's decision on search v2 stage S3 ("the root by previous-iteration scores beside the pre-search",
`SEARCH-V2-PLAN-2026-11-10.md` §3) and for the build. The giants' roots are in
[`reference/root-giants-2026-11-13.md`](../reference/root-giants-2026-11-13.md) ("the giants note" below).

Read-only: nothing was built or run. Line numbers are the working tree of 2026-11-13 (after commit `e7c254c`); the symbol
is named with each so the reference survives edits. "INFERRED" marks a conclusion of mine that the code does not state.
Measured numbers quoted here come from the memory notes and are dated; none was re-measured.

## 0. Summary

1. Our root is a different system from the giants', not a variant of theirs. Each `alpha_beta` call first runs a
   full-width PVS search of every root move one ply shallower (the pre-search), then searches root moves in score order
   at full depth **until one falls more than a margin below alpha, and abandons the rest** (root razoring). The giants
   search every root move every iteration and make the tail cheap with reductions (the giants note §4.1 rows 2-4, 15).
2. With the pre-search on, its scores are **discarded for every move the previous main search reached**; only the tail
   (moves the previous iteration razored away) is ordered and razored on pre-search scores. For the prefix the
   pre-search contributes transposition-table entries, history, killers and nothing else (§3.3).
3. The pre-search's "scores" for non-best moves are null-window fail-soft values, bounds and not exact scores (§3.2).
   The same is true of the previous iteration's main-search scores. Root razoring therefore compares alpha with bounds
   one ply shallower.
4. Most of the giants' root parts already exist here behind knobs that are off (`ENABLE_ROOT_TABLE`,
   `ENABLE_ROOT_LMR`, `ROOT_LMR_EXEMPT_BEST`, `ENABLE_PRESEARCH_OFF_SORT`, `ENABLE_ROOT_BEST_REQUIRES_ALPHA`,
   `ENABLE_ROOT_ALPHA_DECAY`), wired into today's root. The September attempts to replace the pre-search **did not save
   nodes**: the best case was parity (+2.7% nodes, WAC −7), and root LMR without the pre-search lost −25.0 ± 19.6 Elo at
   equal nodes (§5). This note therefore proposes no removal. §6 is a phased path decided by measurement: port our root
   as it is (R0), measure what each of our root features does today (R1), add standard parts as options beside ours,
   smallest first (R2), and let games decide (R3).
5. The time manager reads three things from the root, none of which needs the pre-search: the best move's share of
   main-search nodes, how many iterations the best move has held, and the score drop over three iterations (§3.8). The
   giants' time managers need no all-move scores either (the giants note §3).
6. Six observations in our code look like defects or traps (§7). None changes play at the shipped defaults in a way I
   can show from reading; two matter for the options of §6.5.

## 1. What is on at runtime

The shipped preset block (`search_engine.cpp:2678-2728`, the `V2_SHIP_GEN` ladder under `V2_PRESET`) sets **no** root,
pre-search, aspiration or time-manager knob. Every knob below therefore runs at its header default unless the
environment or the UCI `Knobs` option overrides it (start-up reads: `search_engine.cpp:2070-2076`, `:2135-2174`,
`:3236-3279`). The UCI front end sets `TIME_CHECK_NODES = 2048` (`uci_main.cpp:90`); the Cython path keeps 200,000
(`search_engine.h:16`, `:3291`).

| knob | default | where | state |
|---|---|---|---|
| `ENABLE_ROOT_PRESEARCH` | true | `search_engine.h:1492` | **ON** |
| `ROOT_PRESEARCH_REDUCTION` | 1 | `:1396` | pre-search depth = iteration depth − 1 |
| `PRESEARCH_OFF_FROM_DEPTH` | 0 | `:1475` | off (never switches the pre-search off) |
| `PRESEARCH_TAIL_MODE` / `PRESEARCH_CHUNK` / `PRESEARCH_TAIL_REDUCTION` | 0 / 4 / 2 | `:1485-1487` | off (full pre-search of the tail) |
| `ENABLE_PRESEARCH_SUBSET` | false | `:1476` | off |
| `PRESEARCH_OFF_FILL`, `ENABLE_PRESEARCH_OFF_SORT`, `PRESEARCH_OFF_SORT_MODE` | 0, false, 1 | `:1508`, `:1515`, `:1520` | read only when the pre-search is off |
| `ENABLE_RAZORING` and `ENABLE_ROOT_RAZOR` | true, true | `:235`, `:1470` | **ON** (root razoring needs both) |
| `RAZOR_BASE_FIRST` / `RAZOR_FLOOR_FIRST` / `RAZOR_BASE` / `RAZOR_FLOOR` / `RAZOR_DECAY_PCT` | 750 / 200 / 300 / 100 / 75 | `:248-252` | live |
| `ROOT_RAZOR_CONTINUE` | false | `:236` | off (break) |
| `ENABLE_ROOT_TABLE` | false | `:1427` | off; with it `ROOT_RAZOR_MAX_AGE` 1, `ROOT_RAZOR_MAX_DEPTH_DEFICIT` 2, `ROOT_SORT_L1/L2_LASTREAL`, `ROOT_RAZOR_TO_LMR`, `ROOT_STALE_TO_LMR` are all inert |
| `ENABLE_ROOT_LMR` (`MIN_IDX` 3, `BASE` 1, `DIV` 4) | false | `:1412-1415` | off |
| `ROOT_LMR_EXEMPT_BEST` | false | `:1420` | off |
| `ENABLE_ROOT_BEST_REQUIRES_ALPHA` | false | `:1527` | off |
| `ENABLE_ROOT_ALPHA_DECAY` (2 plies, below depth 14) | false | `:1543-1545` | off |
| `ENABLE_ROOT_SORT_SPLIT` | false | `:3074` | off |
| `ENABLE_SIMPL_BIAS` (`SIMPL_AHEAD_THRESH` 1500, `SIMPL_MARGIN` 0) | false | `:2918-2920` | off |
| `HONEST_ROOT_TT` | true | `:3387` | ON |
| `ENABLE_TT_DEPTH_FIX` | true | `:3178` | ON (pre-search stores its real depth) |
| `ASPIRATION_DELTA` / `MIN_DEPTH` / `MAX_WIDENINGS` / `WIDEN_PCT` | 500 / 5 / 3 / 200 | `:3398-3401` | **ON** |
| `ENABLE_NODE_TT`, `ENABLE_TT_MOVE` | false, false | `:3069`, `:1622` | off: no node stores its own best move |
| `SEARCH_ARM` | 0 | `:3473` | 1 = the root calls search v2 below ply 1 |
| `TM_MODE` / `TM_SCALE` / `TM_NORM` / `TM_OPT_PLY` / `TM_START` / `TM_SINGLE` / `TM_PARTIAL` / `TM_LOWCLOCK` | 1 / 7 / 95 / 1 / 175 / 1 / 0 / 0 | `:3437-3527` | clock games through the UCI binary only |

## 2. Where the root lives

| piece | symbol | lines |
|---|---|---|
| iterative-deepening driver, aspiration, time factors | `get_engine_move` | `search_engine.cpp:4176-4660` |
| root move loop (ply 0, maximizes) | `alpha_beta` | `:4795-5372` |
| pre-search driver and the no-pre-search path | `reorder_legal_moves` | `:8150-8622` |
| pre-search ply 1 | `pre_minimizer` | `:8624-9002` |
| main-search ply 1 (root bookkeeping branch `cur_depth == 1`) | `minimizer` | `:6361`, branch from `:6543` |
| merge of previous-iteration and pre-search data | `descending_sort_wrapper`, `sortSearchDataByScore`, `sortSearchDataRange` | `:9444-9516`, `:9339-9407`, `:9414-9442` |
| ply-1 reply ordering | `ascending_sort` | `:9518` |
| root record | `RootScore`, `SearchData` | `search_engine.h:3626-3699` |
| root table writer (knob-off) | `root_table_store` | `search_engine.cpp:905-942` |
| root TT flag | `root_tt_flag` | `:598` |
| node share counters | `g_root_move_nodes`, `g_root_main_nodes`, `root_nodes_add` | `:275-282` |
| clock budget | `cmd_go` | `uci_main.cpp:159-239` |

Search v2 does not contain the root. With `SEARCH_ARM=1` the old root calls v2 at three sites only: ply 1 scores its
moves with `sv2::score_move<false>` (`search_engine.cpp:6611-6613`), and `pre_minimizer` calls `sv2::negamax<true>` for
its ply-2 searches (`:8847`, `:8877`, `:8905`). The plan records the decision (`SEARCH-V2-PLAN-2026-11-10.md` §7e, "The root
stays as it is in S1").

## 3. Our root as built

### 3.1 The driver (`get_engine_move`)

- **First iteration is depth 3**, full window (`:4272-4286`). There are no depth 1 and 2 iterations: the pre-search of
  depth 3 is a depth-2 search and `pre_minimizer` returns an empty reply list below depth 2 (`:8162-8168`). Plan §6.1
  measured this first iteration at 766 nodes against Stockfish 11's 228 at its depth 3.
- **Loop condition** (`:4321`): depth cap, `may_start_iteration`, `score < 9000000` (stop on a found mate), not timed out.
  `may_start_iteration` (`:4312-4320`): under a soft limit, `spent + TM_START% x last iteration <= TM_SOFT_LIMIT x
  g_tm_scale`; otherwise the preset's `MOVE_TIMES[depth]` rule.
- **Aspiration** (`:4367-4416`): from depth 5, when the previous score is not a mate score. Window `prev_score ± 500`
  millipawns. On a fail the failing side's half-width is doubled and the window is rebuilt around the **same**
  `prev_score` (`:4401-4410`); after 3 widenings one full-window search (`:4391-4398`). The other bound is not moved, the
  depth is not reduced after a fail high, and the centre is the previous iteration's score, not an average.
- **Every aspiration pass re-enters `alpha_beta`**, which runs the whole pre-search again with the new window
  (`:4808-4813`; the comment says so).
- **A timed-out iteration is not adopted** (`:4418-4449`); with `TM_PARTIAL=1` (off) its best completed move is used when
  it beat the pass's entry alpha (`:4426-4439`).
- After a completed iteration the PV's moves get a `moveFrequency` bonus (`:4482-4501`).

### 3.2 The pre-search (`reorder_legal_moves`, `pre_minimizer`): ON

- **Which moves.** All of them, in the previous call's list order, or move generation's order on the first call
  (`:8176-8183`, loop `:8462`).
- **Depth.** `depth_limit - 1`, never below 2 (`:8166-8168`). It is a real search, not a static pass: `pre_minimizer` is
  a ply-1 minimizing node that calls the ordinary interior search for ply 2 (`:8847-8848`, `:8877-8878`, `:8905-8906`), ends
  in the q-search (`:8652-8698`), probes and writes the TT (`:8760-8768`, `:8857-8923`), and applies LMR to late quiet
  replies with the interior's schedule (`:8816-8825`).
- **Windows.** Move 0 with the caller's window, which is the aspiration window (`:8442`). Every later move with a null
  window at the running alpha, re-searched with `(alpha, beta)` when it lands inside (`:8555-8563`).
- **What a score is.** For move 0 and any move that raised alpha: a value inside the window. For every other move: the
  fail-soft return of a null-window search, the value of the first reply that brought the move to alpha or below
  (`pre_minimizer` returns at its beta cutoff, `:8949-8973`). So the pre-search gives every root move a **number**, but
  for the non-best moves it is an upper bound, not a score (INFERRED from the windows; the memory note
  `root-razoring-needs-scores-not-bounds-failllow-does-not-discriminate` describes the pre-search as "an actual searched
  score for EVERY root move", which is true of coverage and not of exactness).
- **Other products.** For every root move: the ply-1 reply list in the order searched and the scores of the replies
  searched before the cutoff (`pre_moves_list`, `preliminary_scores`, `:8713-8714`, `:8937`), stored in the move's
  `RootScore` (`:8460`, `:8566`). TT entries for each root child at the pre-search depth (`:8446`, `:8570`). History, killer,
  counter-move and move-cache updates at ply-1 cutoffs (`:8954-8971`). The global `is_draw` flag when ply 1 is a
  repetition or fifty-move draw (`:8632-8643`), which the driver reads (`:4324`).
- **Cost.** Counted as `g_presearch_nodes` (`:4811-4813`). Memory (09-02, v1 eval, WAC d10): 13.6 M of 35.3 M nodes,
  38.5%.

### 3.3 The merge: which pre-search scores survive

After the pre-search, on every call but the first, `returnData` starts as the previous call's data and
`descending_sort_wrapper(current_search_data, returnData)` merges (`:8612-8618`, `:9444-9516`):

1. The previous call's `scores` holds one entry per move the main search **reached** before it razored (`count` entries,
   `:9474`). The best of those goes to index 0 (`:9457-9471`).
2. The rest of that prefix keep their **previous main-search** scores and reply lists (`:9478-9479`). The pre-search's
   entries for them are dropped: only `preSearchData.scores[count ..]` is appended (`:9480-9481`).
3. The whole tail, prefix leftovers and pre-search entries together, is sorted by `top_score` in one sort (`:9506`,
   `sortSearchDataByScore`), an unstable `std::sort` when the table is off (`:9386-9390`).

So with the pre-search on:

| root move | score used for order and razoring | reply list used at ply 1 |
|---|---|---|
| reached by the previous main search | previous main-search score (depth D−1, mostly a null-window bound) | previous main search's |
| not reached (razored away last time) | this call's pre-search score (depth D−1, mostly a null-window bound) | this call's pre-search's |
| any move, first call of the search | pre-search score (depth 2), sorted (`:8604-8610`) | pre-search's |

Both sources are one ply shallower than the search about to run, so the mixed sort compares like with like in depth.
The header says the same thing about the prefix in its own words ("the work is re-derived and thrown away",
`search_engine.h:1402-1406`), and `ENABLE_PRESEARCH_SUBSET` exists to skip it. Memory (07-28, quiet corpus): 47.8% of
deep root iterations never search past the prefix, so in those the pre-search's scores are not read at all.

On an aspiration re-search the "previous call" is the failed pass at the same depth: `alpha_beta` clears and refills
`previous_search_data` on every call (`:4854-4855`, `:5020-5021`, `:5257-5258`), so a pass that failed high after k moves
hands the next pass a prefix of k+1 entries.

### 3.4 The main root loop (`alpha_beta`)

- Move 0: full window, full depth (`:4908`). Its result seeds alpha (`:4962`).
- Moves 1..: razor test first (§3.5), then a null-window search at full depth, a full-window re-search when the score
  lands inside the window (`:5207`, `:5221-5226`). **No reduction** at defaults (`root_r = 0`, `:5168-5205`).
- Ply 1 (`minimizer`, `cur_depth == 1`): sorts the reply list by the supplied reply scores, ascending (`:6563`), then
  scores each reply through the ordinary interior path (`:6611-6616`). The replies searched and their scores go back to
  the root in `out_entry` (`:6564`, `:6637`, `:6661`).
- Best move: `score > best_score` (`:5285`), not `score > alpha` (§7.2).
- Root beta cutoff returns at once (`:5325-5337`).
- The table handed to the next call holds the searched moves only (`push_back`, `:5020-5021`, `:5257-5258`): the header
  puts its coverage at about 34% of root moves (`search_engine.h:1421-1423`).

### 3.5 Root razoring: ON

- **Threshold** (`:4843-4851`): `max(RAZOR_BASE x 0.75^(depth−4), RAZOR_FLOOR)` = 225 at depth 5, 168 at 6, 126 at 7,
  **100 millipawns from depth 8 on**; on the first call of a search `max(750 x 0.75^(depth−4), 200)` = 1000 at depth 3.
  If move 0's new score beats its own reference by more than the threshold, the threshold grows by that amount
  (`:4976-4977`).
- **Test** (`:5036-5079`): `alpha - scores[i].top_score > threshold`, with `alpha < 9000000`. `razorable` is
  `i < synthetic_from` (`:5054`), which is always true with the pre-search on (`synthetic_from` stays `SIZE_MAX`,
  `search_engine.h:3696`).
- **Action** (`:5123-5126`): `break`. Every remaining root move is skipped for this call. Because the list is sorted by
  the same score, the first trip covers the rest.
- **Data it prunes on:** the table in §3.3. A depth-(D−1) number, usually a bound, from the previous main search or
  this call's pre-search, against an alpha that move 0 has just set at depth D.
- Memory (08-28, v1 eval, WAC d10): 3,586 fires; the previous call's razor line excluded the eventual best move in
  16.4% of the audited iterations; root razoring worth about 19% of nodes with the pre-search and 1.2% without.

### 3.6 Knob-off parts

- **Root table** (`ENABLE_ROOT_TABLE`, `:4980-5017`, `root_table_store` `:905-942`): pre-sizes the table to the full list,
  writes a real `top_score` only when the move was first or beat alpha, the sentinel `ROOT_SCORE_UNPROVEN` otherwise,
  and keeps `prev_score`, `last_real` (written on every store, bounds included), `last_real_depth`, `last_proven`
  (proofs only), `verified`, `age`. This is Stockfish's record minus `averageScore`, `effort` and the PV. It also
  changes the razor to an evidence test (`:5055-5078`). Memory (07-29): 14 arms, none below baseline nodes; kept as a
  correctness item.
- **Root LMR** (`ENABLE_ROOT_LMR`, `:5168-5216`): reduction by **list index**, `BASE + (i − first) / DIV` from index 3 (4
  while nothing beat the entry alpha), re-search at full depth on `> alpha`. Not the interior's formula and no history
  term. `ROOT_LMR_EXEMPT_BEST` exempts the previous call's best move only (`:5177`).
- **Alpha decay** (`ENABLE_ROOT_ALPHA_DECAY`, `:5302-5310`): Stockfish's depth cut after an alpha raise, in SF18's form
  (2 plies below depth 14). SF19's is 3 plies at depths 4-11 (the giants note §2.1). Never measured here as far as the
  memory notes show.
- **Simplification bias** (`ENABLE_SIMPL_BIAS`, `:4955-4957`, `:5279-5320`): when clearly ahead, a root move that captures
  a non-pawn piece and scores within `SIMPL_MARGIN` of the best replaces it. It is the only root feature that compares
  the **scores of two different root moves**, so it needs a real score for the alternative (§6.2).
- **No-pre-search path** (`:8190-8393`): reuses the previous table, fills unscored moves with a floor or a static eval,
  and sorts only under `ENABLE_PRESEARCH_OFF_SORT` (modes 1-5).

### 3.7 The root and the transposition table

- The root never probes the TT for itself or its children; `alpha_beta` has no probe. The root position is never stored
  (`store_node_tt` needs `ENABLE_NODE_TT`, off, and is not called by `alpha_beta`).
- Each root child is stored after its search with the depth actually searched and a flag from the window
  (`:4924`, `:5237`, `root_tt_flag` `:598`), and again by the pre-search at its depth (`:8446`, `:8570`).
- So the previous best move reaches the next iteration through the root list (§3.3), not through the TT. That matches
  Stockfish's choice (root list) and not Reckless's (TT move).

### 3.8 What the time manager reads from the root

All in `get_engine_move`, after a completed iteration, from depth 5 (`:4444-4476`):

| factor (`TM_SCALE` bit) | formula | root data |
|---|---|---|
| nodes (1) | `max(0.563, 0.4499 + 2.2669 x (1 − share))` (`:4457-4459`) | `g_root_move_nodes[from][to]` of the best move over `g_root_main_nodes`: main-search nodes under root moves, summed over the whole search, **pre-search nodes excluded** (`root_nodes_add` is called only around `minimizer`, `:4909`, `:5239`) |
| stability (2) | `1.311 − 0.0533 x stability`, stability capped at 10 (`:4450`, `:4460`) | best move identity per completed iteration |
| score drop (4) | `clamp(0.1127 + 0.00262 x drop3, 0.5028, 1.6561)`, `drop3` = score three iterations ago minus now (`:4455-4461`) | the best score per iteration |

The product times `TM_NORM`% scales the soft limit (`:4462-4469`). The budget itself is in `cmd_go`
(`uci_main.cpp:195-221`): optimum from a 50-move horizon with a ply-dependent share, hard limit `min(5 x optimum, 0.8 x
clock − overhead)`, and a forced move stops after the first iteration (`TM_SINGLE`).

Not read by the time manager: best-move changes inside an iteration, the previous search's score, any second-best
score, anything from the pre-search.

### 3.9 Everything that consumes the pre-search

| consumer | what it takes | where | on at defaults |
|---|---|---|---|
| root order of the tail | pre-search `top_score` for moves the last main search did not reach | `:9480-9481`, `:9506` | yes |
| root order on the first call | pre-search `top_score` for all moves | `:8604-8610` | yes |
| root razoring | the same scores as the reference for tail moves | `:5041-5043`, `:5079` | yes |
| razor threshold bump | move 0's reference | `:4976-4977` | yes |
| ply-1 reply order | reply list and reply scores for tail moves, and for all moves on the first call | `:6563`, `:8460`, `:8566` | yes |
| time-out fallback | `moves_list[0]`, `scores[0].top_score` | `:4839-4840`, `:4967-4968` | yes |
| interior warming | TT entries, history, killers, counter moves, move-cache promotions written during the pre-search | `:8857-8923`, `:8954-8971` | yes |
| draw exit | `is_draw` set at pre-search ply 1 | `:8641` → `:4324` | yes |
| root LMR | index order (the reduction is keyed on it) | `:5178-5187` | no |
| the time manager's node share | nothing (pre-search nodes are outside both counters) | §3.8 | n/a |

Memory's measured split of where the value is: the root score channel is **not** it (a table at 93% evidence left the
removal cost at +54.7% against +56%, 07-29); what remains is interior warming (the pre-search subtree produced 38.7% of
all beta cutoffs) and the right to razor. Those were v1-eval, fixed-depth numbers.

## 4. Ours against the consensus

Rows are the giants note §4.1.

| # | consensus rule | ours today | knob that moves us toward it |
|---|---|---|---|
| 1 | one record per root move for the whole search | searched moves only, about 34% | `ENABLE_ROOT_TABLE` |
| 2 | every root move searched every iteration | no: razor and break | `ENABLE_ROOT_RAZOR=0` |
| 3 | first move full window, rest null window | yes | — |
| 4 | late root moves reduced by the ordinary LMR | no reduction; the knob-off version is index-keyed, not the interior formula | `ENABLE_ROOT_LMR` (different shape) |
| 5 | score kept only for the first move or an alpha raise | every searched move keeps its fail-soft value | `ENABLE_ROOT_TABLE` |
| 6 | stable sort after every root search | a merge with the pre-search, one unstable sort | none (the off-path sort is `ENABLE_PRESEARCH_OFF_SORT`) |
| 7 | previous best first, then the ordinary picker | previous best first, then by previous / pre-search score | none |
| 8 | no pruning at the root | razoring is root pruning | `ENABLE_ROOT_RAZOR=0` |
| 9 | small delta growing with the score, geometric widening, no full-window fall-back | fixed 500, doubling, full window after 3 | `ASPIRATION_*` (shape fixed) |
| 10 | one ply less after a fail high | no | none |
| 11 | beta comes down after a fail low | no | none |
| 12 | root result in the TT | no (children only) | `ENABLE_NODE_TT` (interior only) |
| 13 | best-move changes counted for the time manager | no | none |
| 14 | optimum scaled by root factors, hard maximum inside | yes (three factors) | `TM_SCALE` |
| 15 | no separate ordering pass | the pre-search | `ENABLE_ROOT_PRESEARCH=0` |

We agree on 3 and 14, partly on 7. Rows 1, 2, 4, 5, 6, 8, 15 are one system, and the record shows what happens when
they are switched one at a time.

## 5. The record, read against §3 and the giants

| attempt (memory note) | what it ran | what was missing, by the giants' design |
|---|---|---|
| 07-28/29 tail skip, subset, root LMR, razor off (`presearch-tail-pays-for-itself`) | parts of the pre-search removed, consumers kept | rows 2, 4, 6 |
| 07-29 root table, 14 arms (`root-table-lane-closed-14-arms`) | the record, with the pre-search and razor still on | the table has no job while the pre-search supplies the tail; rows 2, 8 unchanged |
| 08-26 no pre-search, no razor, index root LMR = 181/300 (`root-techniques-all-depend-on-presearch-ordering`) | reduction by list index on an **unsorted** list (found 09-02), table off, no exemption | rows 1, 5, 6; and the giants reduce by move number in picker order with a history term, not by root-list index |
| 08-28 razor on carried bounds (`root-razoring-needs-scores-not-bounds…`) | pruning on fail-low bounds | row 8: nobody prunes at the root |
| 09-02/03 sorted off-path, tiers, then root LMR at fixed nodes: −25.0 ± 19.6 and −11.7 ± 30.0 (`presearch-economics…`) | the first fair run of **our** root LMR: `DIV=8` index schedule, exempt-best, sort mode 1 | the reduction formula (index, no history, no PV terms), the depth cut after an alpha raise, ply-1 ordering still fed by stale lists, iterations from depth 3 |

**The history, plainly** (numbers from `presearch-economics-38-percent-of-nodes-returning-2-to-1`, WAC d10, v1 eval,
baseline 250 / 35,310,778 / STS 1796):

| arm, pre-search and root razor off | WAC | nodes | vs baseline | STS |
|---|---|---|---|---|
| unsorted | 246 | 48,760,840 | +38.1% | 1684 |
| sorted by `top_score` (mode 1) | 246 | 43,615,988 | +23.5% | 1612 |
| provenance tiers + root table (mode 2) | 243 | 36,259,573 | **+2.7%** | 1714 |
| sorted + root LMR `DIV=16` / `DIV=8` | 230 / 218 | 36,919,792 / 34,647,701 | +4.6% / −1.9% | 1612 / — |

- **No replacement arm saved nodes at equal strength.** The best was parity: +2.7% nodes with WAC −7 and STS −82. The
  one arm below baseline nodes (root LMR `DIV=8`) lost 32 WAC solves.
- **Root LMR without the pre-search lost in games at equal nodes**: −25.0 ± 19.6 over 1,668 games at 150k nodes a move,
  −11.7 ± 30.0 over 713 at 400k (09-03). With the pre-search on, root LMR was +4.7 ± 20.3 over 819 timed games (07-29),
  unresolved.
- The pre-search costs 13.6 M of 35.3 M nodes (38.5%) and removing it grew the main search from 21.7 M to 48.8 M on an
  unsorted list. The note's "20-25% of total nodes from one change" was an estimate of a prize. It was never achieved
  and is not an expectation.
- Root razoring with the pre-search on: turning it off cost +24% nodes (39.34 M against 31.76 M, 08-26) for +4 WAC
  solves at fixed depth; at fixed time (07-29) STS was flat (1687 against 1688) and WAC −12 (240 against 252).

Two things the record establishes that any new measurement must respect: the off-path's cost is nodes, not accuracy;
and fixed-depth WAC and STS each miss the other's effect here, so only games decide.

One thing the record does not contain: any test with search v2 below the root, with the v2 eval, or after the seven
search-correctness fixes and the TT flag fix. Every number above is v1-era.

## 6. A phased path, decided by measurement

Nothing in this section recommends removing a feature. Our root stays whole and stays the default at every phase;
standard parts enter as options beside it and are judged in games.

### 6.1 The four phases

| phase | content | gate |
|---|---|---|
| **R0 port** | `alpha_beta`, the ply-1 branch of `minimizer`, `reorder_legal_moves` and `pre_minimizer` re-expressed in search v2's negamax form with **every** system kept: pre-search, razoring and its hybrids, the root table, root LMR, the merge and its sorts, simplification bias, aspiration, every knob as it is read today | node-for-node identity on the six sets of `diagnostics/_sv2_identity.sh` under `SEARCH_ARM` 0 and 1 (S1's method, plan §7e), including the knob-off parts under at least one non-default setting each |
| **R1 measure** | the inventory of §6.2: for each of our root features, a counter or an ablation that says whether its information is new or already known from the previous iteration, and its node share | numbers recorded here before any option is built; no behaviour change, bench fingerprint unchanged |
| **R2 options** | standard parts added as knobs beside ours, default off and byte-identical, smallest and reversible first (§6.5) | each option: fingerprint unchanged when off; changed-rate shown when on |
| **R3 games** | each option, and each of our features where R1 raises a question, at equal time | the ship rule: self-play ≥ +2σ, SF18 at equal nodes > −1σ, Mediocre not negative |

R0 has one decision inside it: the port can be a line-for-line re-expression (cheapest to prove identical) or a v2 root
node that calls today's helpers. Identity is the gate either way. Two of the §7 observations (1 and 5) are
result-identical to fix and can ride with R0; the others change behaviour and belong to R2 as options.

### 6.2 Inventory: each of our root features, and how to tell whether it is still needed

"New information" below means: something the root would not already know from the previous iteration's main search.
Counter lines are the diagnostic prints at the end of `get_engine_move`.

| feature | supplies → consumers | on at defaults | what would show whether it is needed | already measurable with |
|---|---|---|---|---|
| **Pre-search** (`reorder_legal_moves`, `pre_minimizer`) | (a) a depth D−1 value for every root move → order and razoring of the **tail** only (§3.3); (b) ply-1 reply lists and reply scores → ply-1 order for tail moves, all moves on the first call; (c) TT, history, killer, move-cache writes → the whole interior; (d) the first call's order; (e) `is_draw` | **yes** | its node share; how often the root loop reads a pre-search entry at all (searches past the prefix); how often the pre-search's best differs from the previous best, and how often that difference becomes the iteration's result; the split between its score channel and its warming; the cost per aspiration re-search | `[node_split] presearch=` (`g_presearch_nodes`, `:4618-4620`); `[tail_mode] full / reduced / heuristic` (`:4624-4630`); `PRESEARCH_TAIL_MODE=3` (same search, score replaced: isolates the score channel, `:8525-8538`); `ENABLE_PRESEARCH_SUBSET` + `prefix_skipped` (`:8469-8478`); `ROOT_PRESEARCH_REDUCTION`; `PRESEARCH_OFF_FROM_DEPTH`; the 07-28 "fed head = final pick" and prefix-overshoot counters (memory `presearch-list1-sufficiency`; I did not find their print in the current file). **Missing:** a rank-agreement counter between the pre-search's tail order and the order the main search then confirms |
| **Root razoring** (break) | skips every root move past the first whose reference is more than the threshold below alpha → node saving; the root table's short length follows from it | **yes** | how often a skipped move becomes best later (same search, and by an outside judge); nodes saved; equal-time games against "reduce" and "search all" (§6.3) | `[tail_mode] razor_fires` (`:4629`); `[razor_audit] iters_after_razor / winner_was_razored / avg_depth_past_razor` (`RootRazorAudit`, `:949`, print `:4631-4635`; the last figure is a list-index distance, not plies); `ENABLE_ROOT_RAZOR=0`; `RAZOR_BASE`, `RAZOR_FLOOR`, `RAZOR_DECAY_PCT` |
| **Razor hybrids** (`ROOT_RAZOR_TO_LMR`, `ROOT_STALE_TO_LMR`, `ROOT_RAZOR_CONTINUE`) | reduce a razor-condemned move instead of skipping it; reduce a stale one; skip one move instead of the rest | no (the first two need the root table) | reduced searches that came back above alpha (the razor would have lost them) | `[root_table] hybrid_reduced / hybrid_skipped / stale_reduced / razor_stale_skips` (`:4636-4643`); `g_root_lmr_researches` (`:5212`). Memory: stale→LMR collapsed to 170-180 / 300 (07-29) |
| **Root table** (`ENABLE_ROOT_TABLE`) | a full-length record with provenance (`verified`, `last_real`, `last_proven`, `age`) → the evidence test for razoring, the tiered sorts, a sound `ROOT_RAZOR_CONTINUE` | no | evidence coverage; how many razor decisions the evidence test changes | `[root_table] slots / has_real / evidence_pct / proven_this_iter` (`:4636-4639`); `[draw_empty_ply1] root_scores_short` (`:4623`). Memory: 93% evidence, +7.1% nodes at best, +16% under gravity (07-29) |
| **Root LMR** (`ENABLE_ROOT_LMR`, index schedule) | fewer plies for late root moves → node saving; consumes the list order | no | reductions, re-searches, and whether a reduced move was the eventual best | `[root_lmr] reduced / exempt_best` (`:4644-4645`), `g_root_lmr_researches`; `ROOT_LMR_MIN_IDX / BASE / DIV`, `ROOT_LMR_EXEMPT_BEST`. Memory: +4.7 ± 20.3 with the pre-search (07-29); −25.0 ± 19.6 without it at equal nodes (09-03) |
| **Ordering between iterations** (`descending_sort_wrapper`, `sortSearchDataByScore`) | previous best to the front; prefix by previous main-search values, tail by pre-search values, one sort → the search order and the razor's cut point | **yes** | where in the list the iteration's best move sat (index histogram); how often the best came from the prefix or from the tail; whether separate sorts change it | `ENABLE_ROOT_SORT_SPLIT` (`:9496-9503`); the depth ≥ 13 per-move print with `REAL / SYNTH` (`:5261-5276`) and `Best:` index (`:5363-5367`). **Missing:** an index histogram of the root's best move, split prefix / tail |
| **Aspiration at the root** | a narrow window → cutoffs; its failed passes re-run the pre-search and shorten the table | **yes** | fail rate per window, re-searches per iteration, nodes spent in failed passes | `[aspiration] windows / fails / fallbacks` (`:4542-4545`); `ASPIRATION_DELTA=0` = the full-window control; `[iter]` under `ENABLE_ITER_LOG` (`:4515-4516`). **Missing:** nodes inside failed passes, and the pre-search's share of them |
| **Ply-1 branch of `minimizer`** | sorts the carried reply list by the carried reply scores; returns the replies searched → next iteration's ply-1 order | **yes** | first-reply cutoff rate at ply 1 with the carried order against the interior's own order for the same position | the global first-move-cutoff counters (`g_fh_first / g_fh_total`, `:6654-6657`) are not split by ply. **Missing:** the ply-1 split, and a shadow comparison with `buildMoveListFromReordered`'s order |
| **Simplification bias** | prefers a non-pawn capture within a margin when clearly ahead → the move played | no | fires per game, and the regret of the preferred move against the best | no fire counter found. Needs an exact value for the second move, which a null-window root search does not give (§3.2) |
| **Depth-3 start** | the first iteration is depth 3 with a depth-2 pre-search | **yes** | nodes and time of the first iteration; whether depths 1-2 as iterations give the same first order | `[iter]`; plan §6.1 (766 nodes against SF11's 228) |
| **Root child TT stores** (`HONEST_ROOT_TT`, `ENABLE_TT_DEPTH_FIX`) | an entry per root child after each search → transpositions into root children | **yes** | hits on those entries | none specific |
| **Node-share counters** (`g_root_move_nodes`) | the time manager's nodes factor | **yes** (clock games) | the factor's value per move, with and without pre-search nodes in the share | `TMITER … share= share_all=` under `FIX_COUNT_LOG` (`:4299`, `:4470-4475`); `g_tm_cut`, `g_tm_cut_differs` (`:4420-4423`) |
| **Alpha decay** (`ENABLE_ROOT_ALPHA_DECAY`) | shallower search of the remaining root moves after an alpha raise | no | fires; later moves that would have won at full depth | `g_root_alpha_decays` (`:5309`) |
| **PV bonus to `moveFrequency`** | ordering input for the next iteration and the next move | **yes** | not a root-decision feature; listed for completeness | — |

What R1 has to add, all behind the existing diagnostic guards and none changing a node: (1) the root's best-move index
histogram, split prefix / tail; (2) pre-search order against the confirmed order for the tail; (3) for each razored
move, whether it is the final best move of the search and how many iterations later it was found; (4) nodes in failed
aspiration passes; (5) the ply-1 first-reply cutoff split.

### 6.3 Root razoring as a candidate to keep: cost and benefit against "reduce, never skip"

Razoring at the root is ours and no giant has it (the giants note §4.1 row 2). The record says it pays in nodes (§5,
last bullet) and has not been beaten in games. What has never been measured is the like-for-like comparison at equal
time on the current engine. A design:

| arm | setting | what it is |
|---|---|---|
| A | defaults | skip the rest past the razor line (ours) |
| B | the razor's condition kept, the action changed to "search at reduced depth, re-search on `> alpha`" | the giants' rule applied to exactly the moves we skip. `ROOT_RAZOR_TO_LMR` with `ROOT_RAZOR_SKIP_MARGIN=0` does this (`:5084-5100`), but it needs `ENABLE_ROOT_TABLE`, which also changes the razor's evidence test and the sort (`:5055-5078`, `:9355`). A clean arm B needs a small knob that changes the action only |
| C | `ENABLE_ROOT_RAZOR=0` | search every root move at full depth: the upper bound on what skipping can cost in accuracy, and on what it saves in nodes |
| D | `RAZOR_FLOOR` / `RAZOR_BASE` swept (e.g. 50, 100, 200, 400) | the threshold's plateau; the header notes the margins were calibrated before eval magnitudes changed (`search_engine.h:245-247`) |

- **Benefit**: nodes and time to depth of A against B and C on the quiet corpus (`depth_nps_bench`, `_ebf_per_iter.py`),
  never on WAC nodes (memory venue rule).
- **Cost**: counter (3) of §6.2, and an outside judge on the positions where it fires: SF18's value of our move in A
  against C on the regret set, restricted to positions with a razor fire in the final iteration
  (`narrow-items-are-judged-where-they-fire`).
- **Decision**: games at equal time through the UCI binary, A against B and A against C, plus fixed nodes at two
  budgets, and SF18 at equal nodes. If A is level or better, razoring is kept on evidence rather than on belief; if B
  is better, the razor's condition survives as the reduction's trigger.
- The pre-search and razoring were measured as a pair (08-26: both 31.76 M, pre-search only 39.34 M, neither 45.03 M),
  so the same three arms should be read once with today's pre-search and once under any §6.4 option that changes the
  razor's reference.

### 6.4 The owner's cascading tables: options

The idea (owner, 09-03; memory `presearch-economics…`): the main search is the primary source of root ordering, and the
pre-search fills the gap for moves the main search dropped. Order by provenance: **verified this iteration > last
measured at the deepest depth > heuristic**. In code it exists for the pre-search-off path only:
`PRESEARCH_OFF_SORT_MODE` 2 (tiers on `last_real`), 4 (`prev_score`), 5 (`last_proven`, proofs only), with
`ENABLE_ROOT_TABLE` (`:8256-8357`). Its one measurement is the +2.7% nodes / WAC −7 / STS 1714 row of §5. Options, none
recommended over another:

| option | what it is | what exists | what it needs | what to watch |
|---|---|---|---|---|
| C1 tiers with the pre-search **on** | the provenance sort applied to today's merge, so the pre-search's entries are the "heuristic" tier's better substitute | the sort keys and the table | the tiered comparator in `descending_sort_wrapper`; mode 5's `last_proven` key | the table's evidence test moves the razor (07-29: +16% nodes); read with §6.3 |
| C2 pre-search **narrowed to moves without a fresh score** | pre-search only the moves the previous main search did not reach, plus a small chunk | `ENABLE_PRESEARCH_SUBSET` skips the prefix (`:8469-8478`), `PRESEARCH_SUBSET_ALPHA_MARGIN`; `PRESEARCH_TAIL_MODE` / `PRESEARCH_CHUNK` do the opposite cut | a definition of "fresh" by age and depth (`age`, `last_real_depth` in the table) | the prefix's pre-search output is discarded but its warming is not: memory `presearch-is-self-financing` found every removal variant node-neutral or worse, and the skipped prefix's alpha seeding was the hard part |
| C3 a **cheaper or shallower** pre-search | depth D−2 or less, or reduced for the tail only | `ROOT_PRESEARCH_REDUCTION`, `PRESEARCH_TAIL_MODE=1` + `PRESEARCH_TAIL_REDUCTION` | the razor's reference becomes two plies shallower than alpha: `ROOT_RAZOR_MAX_DEPTH_DEFICIT` and `last_real_depth` were added for exactly that, and pre-search values would have to be written into the table with their true depth (memory `root-table-lane-closed-14-arms`, "untried, principled fix") | the header records that a reduced pre-search hit the "shallower overrides deeper" fault (`search_engine.h:3649-3652`); memory 08-26: `RPR=2` with a fed TT gave 231 against 235 |
| C4 pre-search only in **early iterations** | on below depth N, tiers after | `PRESEARCH_OFF_FROM_DEPTH` (`:8190`) | the off-path sorted (mode 2/5) and the razor rule at the switch | untested with a sorted off-path |
| C5 pre-search as a **time-management signal only** | a shallow pass whose only output is the gap between the best and second root values ("easy move"), or its disagreement with the previous best | the pass itself; `TM_SCALE` has room for a fourth bit | a predictor test first: does the gap predict that the best move will not change (the `TMITER` log already carries share, stability and drop per iteration, `:4470-4475`) | the second value is a null-window bound (§3.2), so the gap is a lower limit; no giant uses a gap (the giants note §3), so there is no reference constant |

C1 and C2 keep everything we have and change what the pre-search is asked to do. C3 trades the pre-search's depth
against the razor's soundness. C5 is independent of the others. Each is one or two knobs on the R0 port.

### 6.5 Standard parts as options beside ours (phase R2), smallest first

Each is a knob on the R0 port, default off and byte-identical, working **with** the pre-search, razoring and the merge
unless the row says otherwise. Ordered from the smallest and most reversible. "Giants" cites the giants note.

| # | option | giants | what we have | touches our systems? |
|---|---|---|---|---|
| P1 | count best-move changes inside an iteration; key the per-move node counter on the whole move | §4.1 row 13 | stability and share exist (`:4450`, `:4457`) | no: measurement and a time-manager input only |
| P2 | a root move must beat alpha to become best | SF19 `:1471`, SF11 `:1248` | `ENABLE_ROOT_BEST_REQUIRES_ALPHA` | no |
| P3 | aspiration: after a fail low bring beta down; after a fail high search one ply less; no jump to the full window | rows 9-11 | `ASPIRATION_*`, fixed shape | no; fewer failed passes also means fewer repeated pre-searches |
| P4 | aspiration centre and width from the record: an average instead of the last score, a delta that grows with the score | §4.2 | — | no |
| P5 | play the best completed move of a cut iteration | SF19 `:403-409` | `TM_PARTIAL` | no |
| P6 | depths 1 and 2 as iterations before the first pre-search | all (SF11 aspirates from 4) | the driver starts at 3 | the first call's order would come from iterations instead of a depth-2 pre-search |
| P7 | the complete record with the store rule "first or beat alpha" and a stable sort | rows 1, 5, 6 | `ENABLE_ROOT_TABLE`, `ROOT_SORT_L1/L2_LASTREAL` | yes: the razor's evidence test and the sort key change with it; §6.4 C1 is this option in the owner's form |
| P8 | reduce instead of skip, on the razor's own condition | row 4, applied to our trigger | `ROOT_RAZOR_TO_LMR` (needs the table) | yes: §6.3 arm B |
| P9 | the interior's LMR for late root moves, in place of the index schedule | row 4 | `ENABLE_ROOT_LMR` (index schedule); v2's interior LMR | yes: consumes the root order; see §6.6 item 1 |
| P10 | depth cut after an alpha raise (SF19 only); never reduce a former best (SF11 only) | §4.2 | `ENABLE_ROOT_ALPHA_DECAY`, `ROOT_LMR_EXEMPT_BEST` | only with P8 / P9 |
| P11 | ply 1 ordered by the interior's own source instead of the carried reply lists | all | the carried lists (§3.4) | yes: a consumer of the pre-search gets a second supplier |
| P12 | the giants' whole root as one arm: P7 + P9 + P11 with the pre-search and razoring off | §4.1 rows 1-8, 15 | the off-path (`:8190-8393`) | an arm for comparison only; §5 is its record so far (parity at best, −25 Elo with root LMR) |

P1-P5 change nothing our root depends on and can be tested alone. P6-P11 each change one supplier or one consumer and
are read against the R1 numbers. P12 is listed so the comparison exists, not as a goal; if it is run, it must be run
whole (§6.7), which the September arms were not.

### 6.6 Prerequisites for the larger options (P8, P9, P11, P12)

1. **A reduction the root can trust.** The giants reduce root moves with the interior formula, by move number in picker
   order, with history in it and strong PV-node relief (SF19 `search.cpp:1316-1318`, `:1352`, `:1369`; Reckless
   `search.rs:884-885`, `:919`). Ours at the interior is `reduced_search_depth(depth_limit, cur_depth, pin, i, state)`
   (called at `search_engine.cpp:8825`): keyed on the move index and the iteration depth. Before the root uses it, decide
   what "PV node at the root" does to it. This is the part the 09-03 closure names: "do not re-open without a mechanism
   that is not index-keyed root reduction". The giants' mechanism is also keyed on move number; what differs is the
   order the number counts in and the history and PV terms. If S5's LMR table with a history adjustment comes first,
   the root inherits it; if the root comes first, it runs on today's schedule and the test reads the schedule too.
2. **Ordering at ply 1 without carried lists.** Today's ply-1 order for tail moves comes from the pre-search. P11
   needs the interior's source to be good at ply 1: the move cache with promote-to-front covers positions seen
   before (every ply-1 position was seen in the previous iteration), and a node's own TT best move is the giants'
   answer. `ENABLE_NODE_TT` is off and was measured as harmless-to-null in v1 (`node-tt-dependency-starved…`).
3. **The record's sort key.** "Sentinel unless proven" needs the stable sort and the best-requires-alpha rule together
   (memory `root-lmr-exempt-best-is-inert-until-the-table`: each rule creates the other's failure when ported alone).
4. **A per-move exchange test and picker stages** are S3's other half (the move source). The root does not need them,
   but the giants count the reduction's move number in picker order, so P9 reads better once they exist.
5. **Depth-1 and depth-2 iterations** must not hit the q-search precaution branches that assume `depth_limit >= 6`
   (`:6434-6441`, `:8657-8664`) in a new way. They already run at depth 3-5, so this is a check, not a build.

### 6.7 Only for an arm that runs without the pre-search (P12): which consumers lose their supplier

The September arms switched the supplier off and left consumers running. Any arm without the pre-search has to settle
every row below first.

| consumer (§3.9) | without the pre-search | the giants' substitute |
|---|---|---|
| root order of the tail | previous order kept by the stable sort; unscored moves in picker order | the same; they never have a score for the tail |
| root razoring | removed | none; the tail is reduced, and in SF19 searched 3 plies shallower after the first alpha raise |
| razor threshold bump | removed with it | — |
| ply-1 reply order | the interior's own ordering | TT move, history, killers |
| time-out fallback | the record's first entry | the same |
| interior warming (TT, history) | gone as a separate pass | none dedicated: the previous iteration and the reduced searches of the tail are the warming |
| draw exit flag | must be set by the new ply 1 | n/a |
| simplification bias (off) | has no score for the alternative move (sentinel) | n/a; it would need its own verification search of the candidate |
| root LMR index schedule (off) | replaced by the interior formula | — |

The fourth and sixth rows are where the measured cost of removing the pre-search sat (memory: 38.7% of beta cutoffs came
from the pre-search subtree; with it off the main search grew 125%). Whether the reduced tail searches replace that
warming in our engine is the open empirical question; it cannot be read from code.

### 6.8 Phase R3: how the games are read

1. **One option per test**, on the R0 port at defaults, except where two parts only make sense together (P7 with a
   sort key; P8 with P7 until a clean action-only knob exists). Disjoint parts may share a bundle
   (`bundle-to-save-box-time-only-when-the-parts-stay-readable`).
2. **Venues, per the ship rule**: timed self-play through the UCI binary; fixed nodes at two budgets (the 09-03 run's
   150k and 400k are the comparison points); SF18 at equal nodes; Mediocre not negative. Fixed-depth WAC and STS only
   as sanity and fingerprints; node effects on the quiet corpus, not WAC.
3. **Registered before each run**: the R1 counter the option is expected to move, time to depth 13
   (`_ebf_per_iter.py`), and the predicted sign.
4. **Interactions**: the pre-search and razoring are a measured pair, so an option that touches either is read as a 2×2
   with the other (`search-changes-are-antagonistic-not-additive`). The plan's 2×2 with the move source (cache /
   staged, plan §3 S3) applies to P9 and P11.
5. **A level result keeps ours.** An option that is not better in games stays off; a feature of ours that R1 shows
   carries no new information is still kept unless an option beats it in games.
6. **One process per setting**, and fresh seeds for a candidate built on an earlier winner.

Two further pre-search options that fit neither §6.4 nor the table above: running it only after an aspiration fail low,
when the previous best has just collapsed and the alternatives need re-ranking (my suggestion; in no engine read), and
as the source of several root values in analysis (the giants use a MultiPV loop of full root searches, SF19
`search.cpp:360-503`; the pre-search's values for non-best moves are bounds).

## 7. Observations that look like defects or traps

1. **A dead PV update in the pre-search.** `reorder_legal_moves` tests `if (score > alpha) updatePV(moves_list[0], 0)`
   (`search_engine.cpp:8455`) while `score` still holds its initial `-99999999` (`:8160`); move 0's result is in
   `highest_score`. The branch never runs. Harmless today because `alpha_beta` clears the PV table after the pre-search
   (`:4831-4832`).
2. **`best_move` can be taken by a move that did not beat alpha** (`:5285`; the fix `ENABLE_ROOT_BEST_REQUIRES_ALPHA` is
   off, `search_engine.h:1521-1527`). On a fail-low pass a later move with a higher fail-soft bound replaces move 0. At
   defaults the pass is always re-searched or discarded, so the played move is not affected (INFERRED); but it is the
   value handed in as `root_prev_best` (`:4827`) and compared for `tm_stability` only after a resolved pass. Option P7
   needs P2 with it (§6.6 item 3).
3. **The root's own clock checks return without setting `time_up`** (`:5024-5025`, `:5339-5348`). The interior sets the
   flag at the same limit but only every `TIME_CHECK_NODES` nodes (`:6378-6385`; 2,048 in the UCI binary, 200,000 in the
   Cython path). When the root's check fires first, the driver sees a completed iteration (`:4444`): it adopts the move,
   updates the time factors and emits an `info` line for a depth at which only part of the root list was searched
   (INFERRED; not reproduced). The move itself is a sound best-so-far, so this is an accounting and reporting
   inconsistency rather than a playing error, and it contradicts the stated rule at `:4289-4292`.
4. **Every aspiration re-search repeats the full pre-search** (`:4808-4813`) and inherits the failed pass's truncated
   table (`:4854-4855`). Known (memory `presearch-list1-sufficiency`); listed because option P3 (fewer failed passes) reduces its cost
   without touching the pre-search.
5. **`g_root_move_nodes` is indexed by from and to squares only** (`:275-282`), so the four promotions of one pawn move
   share a counter, and the time manager's share for a promotion best move includes its under-promotions (`:4457`). Small;
   option P1 keys the counter on the whole move.
6. **The tail sort is an unstable `std::sort`** when the table is off (`:9386-9390`, called from `:9506` and `:8610`). Root
   moves with equal scores are ordered by the standard library's algorithm, so the root order, and with it the bench
   fingerprint, depends on the library implementation as well as the code (INFERRED; relevant to the release-readiness
   rule that only result-identical choices may depend on the platform).

Two stale comments, for whoever edits next: `search_engine.h:1407-1411` says the root-LMR start rule
`moveCount > 1 + rootNode + (rootNode && bestValue < alpha)` is "invariant SF11->SF18"; SF19 has neither that rule nor
the best-move exemption (the giants note §2.1). `search_engine.h:1536-1537` quotes the depth cut as `depth > 2 && depth <
14 ... depth -= 2`; SF19 reads `depth > 3 && depth < 12 ... depth -= 3`. SF18 was not re-read, so both comments may be
right for SF18.

## 8. Not verified

- Nothing was run: every "at defaults" statement is from the header, the preset block and the start-up reads.
- Stockfish 18 (cited by our comments) and Stockfish 11's `timeman.cpp` were not read.
- Whether SF19's ProbCut fires at its root in practice (the text has no root condition).
- Triumviratus's interior reduction code was read only at its root conditions; `HISTORY.md` was searched, not read.
- Reckless's commit hash was taken from the 11-11 movegen note, not re-derived.
- The memory notes' numbers are v1-eval and pre-date search v2, the correctness fixes and the TT flag fix.
- `search_v2.cpp` was read only at its entry points from the root.

## 9. Open questions for the owner

1. **R0's form.** A line-for-line re-expression of the four root functions in negamax (cheapest identity proof), or a
   v2 root node that calls today's helpers? Identity is the gate either way.
2. **R1's scope.** Are the five new counters of §6.2 the right set, and should R1 be read on the quiet corpus and the
   games-derived set as well as WAC (the venue rule says yes for node questions)?
3. **Razoring (§6.3).** Is a small action-only knob for arm B (reduce on the razor's own condition, without the root
   table) worth building, so that skip against reduce is read without the table's side effects?
4. **Cascading tables (§6.4).** Which of C1-C5 first? C1 and C2 are the closest to the 09-03 design; C5 is independent
   and has a cheap predictor test from logs.
5. **Order of the R2 options.** P1-P5 need nothing from R1 and could run while R1 is read. Is that acceptable, or
   should every option wait for the R1 numbers?
6. **P9 and S5.** The interior's LMR is keyed on the move index and the iteration depth today. Test it at the root on
   today's schedule, or wait until S5 gives the interior a history-adjusted table?
7. **Is the giants' whole root (P12) wanted as a comparison arm at all**, given the September record (parity at best,
   −25 Elo with root LMR at equal nodes), or only if the smaller options point that way?
8. **The observations in §7:** fix 1 and 5 with R0 (result-identical), and treat 2, 3 and 6 as options with their own
   reads?
