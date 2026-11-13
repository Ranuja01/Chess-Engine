# ROOT SEARCH MODEL — how our root works, what depends on what, and every configuration tried

As of 2026-11-13 (production gen 7.3, the MOVEGEN_FAST build). Read this before changing the root. Detail with file:line
citations lives in three working documents; this one is the summary that stays true and is kept up to date:

- `active/S3-ROOT-AUDIT-2026-11-13.md`: our root as built, the per-feature inventory, the phased plan, the R1 results (§10)
  and the owner's rules (§11).
- `active/S3-ROOT-WITHOUT-PRESEARCH-2026-11-13.md`: how Stockfish 19, Stockfish 11, Reckless and Triumviratus run a root
  without a pre-search, against ours, with ranked hypotheses and tests.
- `reference/root-giants-2026-11-13.md`: the giants' roots as written.

How to judge any change here: [`JUDGING-SEARCH-IDEAS.md`](JUDGING-SEARCH-IDEAS.md). The owner's standing rules: never rip a
part out alone (the capture-gains lesson), keep v1 and the owner's systems in the code behind switches, and judge by headroom
as well as by the first result (memory `search-v2-traditional-beside-ours-with-switches`, `keep-v1-search-for-hybrid-ideas`).

## 1. The system as built (defaults)

Each iterative-deepening iteration (`get_engine_move` → `alpha_beta`):

1. **Pre-search** (`reorder_legal_moves` → `pre_minimizer`): every root move is searched one ply shallower, full width,
   with PVS. A non-best move's value is a null-window fail-soft bound, not an exact score. It produces a value per root move,
   a reply list and reply scores per root move, and TT / history / killer / move-cache writes.
2. **Merge** (`descending_sort_wrapper`): the previous iteration's best move goes first. Moves the previous main search
   reached keep the main search's values; the pre-search's values for them are discarded. Only the moves the previous main
   search did not reach (the "tail") take pre-search values. The rest of the list is sorted in one pass.
3. **Main loop** (`alpha_beta`): move 0 gets a full window; the others get a null window and are re-searched when they beat
   alpha. There are no reductions at the root.
4. **Root razoring** (`alpha_beta`, `ENABLE_ROOT_RAZOR`): when a move's stored value is more than the threshold below alpha,
   the loop ends and every later move is skipped. The threshold starts at `RAZOR_BASE` 300 (750 on the first iteration),
   decays by `RAZOR_DECAY_PCT` 75% per ply beyond depth 4, and floors at `RAZOR_FLOOR` 100 (200 on the first iteration). It
   widens when alpha jumps past move 0's stored value.
5. **Ply 1** (`minimizer`, the `cur_depth == 1` branch): replies are searched in the carried order, sorted by the carried
   reply scores. The node has no null move, no reverse futility and no other node-level cut; a ply-1 razoring block exists in
   the code but is commented out. The first reply that refutes ends the node.
6. **Aspiration** (`get_engine_move`): `ASPIRATION_DELTA` 500 around the previous score. The failing side widens ×2
   (`ASPIRATION_WIDEN_PCT` 200) up to `ASPIRATION_MAX_WIDENINGS` 3 times, then the full window is used. Every pass re-runs the
   pre-search.
7. **Interior null move** starts at ply 3 on the opponent's side and ply 4 on ours (`NULLMOVE_CURDEPTH_MINI` 3 / `_MAXI` 4,
   tuned with the pre-search present). Plies 1 and 2 never null-move.

Off at defaults, but built and wired: the root table (`ENABLE_ROOT_TABLE`), root LMR (`ENABLE_ROOT_LMR`,
`ROOT_LMR_EXEMPT_BEST`), the pre-search-off path with sorts (`ENABLE_ROOT_PRESEARCH=0`, `ENABLE_PRESEARCH_OFF_SORT`,
`PRESEARCH_OFF_SORT_MODE`), the razor hybrids, alpha decay, the pre-search subset (`ENABLE_PRESEARCH_SUBSET`), and since
11-13 `ROOT_RAZOR_REDUCE`, `ASPIRATION_RECENTER`, `PLY1_NODE_PRUNE` and `ROOT_TABLE_REPLY_SCORES`.

## 2. Dependencies: what the pre-search supplies, and to whom

| consumer | what it takes from the pre-search | measured weight (11-13, depth 10) |
|---|---|---|
| root razoring | the stored value of tail moves, as the razor's reference | the razor's whole decision on the tail; see §3 |
| order of the tail | pre-search values | 9-13% of iterations' winners carry a pre-search value |
| first-iteration order | pre-search values for every move | — |
| ply-1 reply order | reply lists and scores for tail moves, and for every move on the first call | — |
| interior warming | TT, history, killers, counter moves, move-cache writes | prefix warming no longer pays (subset arm −1 to −3.5% nodes at equal accuracy); tail warming unmeasured |
| time-out fallback | move 0 and its value | — |
| draw exit | `is_draw` set at pre-search ply 1 | inert at the shipped `FIX_DRAW_EXIT=1` |

**The pre-search and root razoring are one mechanism.** The razor discards moves that the search later plays: the played
move had been razored at some iteration in 57-80% of searches. The pre-search is what brings those moves back. With the
razor off, the pre-search falls to 10-12% of nodes and supplies 0.6-3% of winners. Removing it as well then costs fewer nodes
than keeping it.

## 3. Measured costs (2026-11-13, gen 7.3, depth 10, LONG_FORMAT; queues #226 / #227)

| | WAC | STS300 | quiet corpus | game set |
|---|---|---|---|---|
| pre-search share of all nodes | 40.3% | 24.6% | 23.8% | 23.4% |
| iterations whose best move changed | 28.6% | 34.8% | 27.2% | 26.5% |
| winner's value from: index 0 / main search / pre-search | 79 / 9 / 12% | 67 / 20 / 13% | 75 / 16 / 9% | 76 / 15 / 9% |
| searches whose played move was razored at some iteration | 80% | 72% | 60% | 57% |
| nodes in failed aspiration passes (pre-search's part of them) | 36% (52%) | 9% (29%) | 12% (41%) | 14% (28%) |

## 4. How the strongest engines do the same jobs

Read in Stockfish 19 and 11, Reckless and Triumviratus (§1 of the without-pre-search doc):

- Every root move is searched every iteration; none is skipped. The list is stable-sorted after each pass, and only the
  first move or a move that beats alpha keeps a value. Root reductions are small: in Stockfish 19 they work out to 0-3 plies.
- **A late root move is dismissed cheaply at its ply-1 reply node.** That node is an ordinary cut node whose first act is a
  null move at about `depth − (7 + depth/3)`, followed by reverse futility and the rest of the interior's pruning. Our ply 1
  has none of these.
- The previous iteration leaves a best move in the TT for every node it visited. Ours stores no node best move
  (`ENABLE_NODE_TT` off); the move cache plays that role.
- No giant's time manager needs a value for every root move.

**So our pre-search plus root razoring, and the giants' cheap ply-1 cut nodes, are two answers to the same job: making bad
late root moves cheap.** Ours skips them, using stored values, and revives them through the pre-search. Theirs searches each
one with a quick null-move test. A root that removes ours without adding theirs is an incomplete replacement.

## 5. Configuration register: every root configuration tried

No configuration below is rejected for good. Each non-winner has a written re-test trigger, and the triggers are reread at
the start of each search stage. "Untuned" means its margins were not refitted. "Incomplete" means a consumer from §2 has no
supplier.

| config | knobs (on top of defaults) | depth-10 screen (WAC / STS / quiet nodes vs A0) | games | status | re-test trigger |
|---|---|---|---|---|---|
| **A0 today** | — | 249 / 2037 / — | — | default | — |
| A1 pre-search values replaced (diagnostic) | `PRESEARCH_TAIL_MODE=3` | 245 / 1954 / −4.0% (WAC +23% nodes) | — | diagnostic: the values matter on tactics | — |
| A2 razor off | `ENABLE_ROOT_RAZOR=0` | 257 / 2079 / +7.2% (WAC +27%) | — | trades nodes for accuracy | when a cheap ply-1 node exists (H1) |
| A3 pre-search subset | `ENABLE_PRESEARCH_SUBSET=1` | 250 / 2040 / −2.9% (WAC −3.5%) | not yet | **candidate** (fewer nodes, equal accuracy) | the next free games slot |
| A4 cascading tables | `ENABLE_ROOT_PRESEARCH=0 ENABLE_ROOT_RAZOR=0 ENABLE_PRESEARCH_OFF_SORT=1 PRESEARCH_OFF_SORT_MODE=2 ENABLE_ROOT_TABLE=1` | **258 / 2086** / +7.4% (WAC +18%) | #228: −14.9 ± 17.3 at 1,280 of 2,000 g (final ~19:00 11-13) | INCOMPLETE (no ply-1 cut node) and UNTUNED | H1's games; again after retuning |
| A5 pre-search one ply shallower | `ROOT_PRESEARCH_REDUCTION=2` | 243 / 1962 / −22.6% (WAC −13%) | not yet | open: fewer nodes, less accuracy | an equal-time read |
| Sept: root LMR with the pre-search off | (09-03 arms) | — | −25.0 ± 19.6 at equal nodes (1,668 g), −11.7 ± 30.0 (713 g) | INCOMPLETE (no root table, no best-move exemption, no ply-1 cut node), v1 eval | after H1, with C4's supports |
| Sept: tiered sort with the pre-search off | `PRESEARCH_OFF_SORT_MODE=2` + table | v1: +2.7% nodes, WAC −7, STS −82 | — | superseded by A4 on today's engine | — |
| C1 razor reduces instead of skipping | `ROOT_RAZOR_REDUCE=1` | #230 | — | pending | — |
| C2 / C3 aspiration re-centred | `ASPIRATION_RECENTER=1` / `2` | #230 | — | pending | — |
| C4 A4 + root reductions | A4 + `ENABLE_ROOT_LMR=1 ROOT_LMR_EXEMPT_BEST=1` | #230 | — | pending | — |
| C5 A4 + re-centred aspiration | A4 + `ASPIRATION_RECENTER=2` | #230 | — | pending | — |
| H1 / H2 A4 + ply-1 null move (+ reverse futility) | A4 + `PLY1_NODE_PRUNE=1` / `3` | #230 | — | pending: the direct test of §4 | — |
| H3 today + ply-1 null move | `PLY1_NODE_PRUNE=1` | #230 | — | pending: a hybrid of both answers | — |
| D1 A4 + reply scores kept | A4 + `ROOT_TABLE_REPLY_SCORES=1` | #230 | — | pending (defect fix) | — |
| owner hybrid: history-signalled razoring | not built | — | — | parked | when the history stage (S5) starts |
| owner hybrid: razor as a reduction | C1 is its first form | — | — | see C1 | — |

## 6. Claims, and what would falsify them

| claim | falsified by |
|---|---|
| The pre-search and root razoring are one mechanism (the pre-search's main job is reviving razored moves). | With the razor on, a sorted pre-search-off path keeps the played-move-razored rate and the node count level. |
| A4's deficit comes mainly from the missing ply-1 cut node. | H1 does not reduce A4's nodes under failing late root moves (`late_faillow_nodes`), or H1 stays behind A0 at equal time. |
| The pre-search's prefix warming no longer pays. | The subset arm (A3) loses at equal time. |
| Failed aspiration passes are a real cost (9-14% of nodes on game positions). | A re-centred window (C2 / C3) cuts failed passes without cutting nodes. |

## 7. Counters to read for any root change

The `[root_r1]` line (`get_engine_move`'s end-of-search prints; counting only) gives:
- iterations, best-move changes, and where the winner's value came from (index 0 / main / pre-search / synthetic);
- root moves searched after index 0, and those raising alpha, by the same source split;
- searches whose played move had been razored;
- aspiration nodes, the nodes in failed passes, and the pre-search's part of those;
- nodes under late root moves, and under those that failed low (`late_nodes`, `late_faillow_nodes`);
- the ply-1 cuts (`ply1_rfp_cuts`, `ply1_null` cuts / tries).

Read it together with `[node_split]` (the pre-search's node total), `[razor_audit]` and `[aspiration]`. The suite totals are
on the last line of a single-process run. Node totals come from summing `(nodes=` in the run's stderr; the `[node_split]
total` field is per position only.
