# Sweep 3: time manager, ply-1 entry, carried reply lists (2026-11-20)

A read-only audit of three areas the 11-18 sweeps left open (`SEARCH-DEFECT-REGISTER-2026-11-18.md` §8), on gen 7.7
(`V2_PRESET=shipped`, `SEARCH_ARM 1`). Nothing was run: no build, no engine, no script. Every count below is a proposal,
not a measurement. Items already in the register (S2-1 … S2-33, PS-3 … PS-10, C6, C9) are not repeated; four of them are
sharpened in §3.

**Result in one paragraph.** No live defect that changes a search decision (class D) was found in these three areas. The
ply-1 node and the carried reply lists are sound. The time manager's arithmetic is sound and cannot lose on time by its own
rules. What was found: one aspiration-window shape that wastes passes exactly where mate scores appear (S3-1), one missing
term in a borrowed time formula (S3-2), a low-clock path that can play a move with no search at all (S3-3), and seven
small or latent items.

Terms used:
- **Soft limit**: the time a move is meant to take; tested only between iterations. **Hard limit**: the absolute stop,
  tested inside the search every 2,048 nodes.
- **Scale**: the product of the three `TM_SCALE` factors times `TM_NORM`; it multiplies the soft limit (range 0.21 … 5.6).
- **Pre-search**: the one-ply-shallower pass over every root move that runs before each main root pass
  (`reorder_legal_moves` → `pre_minimizer`).
- **Prefix / tail**: root moves the previous main pass reached (they keep its values) / the rest (they take pre-search values).
- **Carried reply list**: for each root move, the opponent's replies and their last scores (`RootScore::second_moves` /
  `second_scores`), handed to the ply-1 node on the next pass.
- **Aspiration window**: a narrow (alpha, beta) around the previous iteration's score; a result outside it is a "fail" and
  the pass is repeated with a wider window.
- Classes: D = a search decision changes, V = a wrong value read later, L = latent, E = efficiency, T = time management.
- VERIFIED = I read the code that executes at shipped settings. REPORTED = not checked to that standard.

Line numbers are of today's files; the symbol is given beside each so the reference survives edits.

---

## 1. New findings, ranked by likely impact

| # | defect | class | status |
|---|---|---|---|
| **S3-1** | An aspiration fail with a mate-range score is widened three times from the old score before the full window is tried | E / T | VERIFIED (mechanism); size unknown |
| **S3-2** | The score-drop factor sees only a drop inside this search, never the drop since the previous move | T (design) | VERIFIED |
| **S3-3** | On a very low clock the hard limit is 10 ms, which can cut the first pass and play an unsearched move | T | VERIFIED |
| S3-4 | The start rule tests the scaled soft limit, not the hard limit, so it can start an iteration it has itself predicted cannot finish | T, low | VERIFIED |
| S3-5 | The pre-search leads with the dethroned move after a best-move change | E, low | VERIFIED |
| S3-6 | A negative clock value falls through to the 45 s preset | T, latent | VERIFIED (code); trigger REPORTED |
| S3-7 | `go` in a position with no legal move reads an empty list | L | VERIFIED (by reading) |
| S3-8 | The node counter is a 32-bit `int` | L | VERIFIED |
| S3-9 | `go infinite` can answer before `stop`; `go depth` / `go nodes` can answer from the book | L (protocol) | VERIFIED |
| S3-10 | The ply-1 exit's history add ignores `SV2_HIST_ALLNODE_GATE` | L (knob coverage) | VERIFIED |

### S3-1: a mate-range aspiration fail is widened from the old score, three times

- **Defect.** After a failed aspiration pass the window grows ×2 from the *previous iteration's* score (500 → 1,000 →
  2,000 → 4,000 millipawns), then falls back to the full window. When the failed pass already returned a mate-range score
  (|score| ≥ 9,000,000), none of the three widened windows can contain it, so three passes are certain to fail again.
- **Mechanism.** `get_engine_move`, the aspiration loop (`search_engine.cpp:4525-4585`): `alpha_win = prev_score - lo` /
  `beta_win = prev_score + hi` (`:4575`, `:4583`). `ASPIRATION_RECENTER` would re-centre on the returned score, but it
  excludes mate-range scores on purpose (`score > -9000000` at `:4567`, `score < 9000000` at `:4580`), and it is 0 at
  shipped values anyway. Every repeated pass re-runs the pre-search over all root moves. Wherever a window bound inside
  such a pass is itself mate-range (alpha after move 0 returned the mate, and every window derived from it), the usual
  savers are off by their own guards: the root razor (`alpha < 9000000`, `:5321`), reverse futility and its lazy eval
  (`beta < 9000000`, `search_v2.cpp:507`, `:524`).
  - Fail high (we found a mate): each doomed pass searches move 0 and, because of S2-6, move 1.
  - Fail low (we are being mated): each doomed pass is a whole fail-low pass over the root moves the razor leaves.
  - The transposition table answers part of each repeat, so the cost may be small. That is why it must be counted.
- **Reference.** SF11 re-centres on the returned value: `alpha = max(bestValue - delta, -VALUE_INFINITE)` and
  `beta = min(bestValue + delta, VALUE_INFINITE)` (`stockfish_11/.../search.cpp:483`, `:491`), so a mate-range fail is
  resolved by the next pass.
- **Relation to known items.** S2-33 part (a) asks for "aspiration windows and re-searches around mate-range scores" to
  be looked at; this is one concrete shape there. It is NOT shown to be the cause of the 200× iteration: that needs an
  ablation, not a reading. C2 / C3 (`ASPIRATION_RECENTER` 1 / 2, no gain at depth 10) do not cover it, because the knob
  skips mate-range scores.
- **Count (fixed depth, safe beside games).** In the aspiration loop (once per pass, not hot code, so plain counters):
  `asp_fail_mate` = failed passes with |score| ≥ 9,000,000 while `attempt < ASPIRATION_MAX_WIDENINGS`, and
  `asp_mate_wasted_nodes` = nodes from such a fail to the pass that resolves it. Print on the `[aspiration]` line. WAC
  d10 has 63-65 searches ending on a mate-range score (register §5b), so it will fire there; game12 / eg60 give the game
  rate. The existing `fallbacks=` field already counts windows that needed the full window.
- **Fix sketch.** Knob `ASP_MATE_FULL` (default 0 = byte-identical): when a failed pass's score is mate-range, set the
  failing bound to the full bound at once (`alpha_win = -9999998` or `beta_win = 9999999`) and keep the other side.
- **Expected effect.** Bench nodes −0 … −2% on WAC (mate-heavy), about 0 on game positions; no strength change expected
  in ordinary games. Its value is the tail: time saved on the move where a mate first appears, in a won or lost position.

### S3-2: the score-drop factor cannot see a drop that happened between moves

- **Defect.** The factor that adds time when the score falls reads only `tm_iter_scores[k - 3] - score`, the fall over
  the last three iterations of *this* search (`search_engine.cpp:4640-4647`). If the opponent's reply refuted our last
  move, every iteration of the new search already shows the lower score, the fall over three iterations is 0, and the
  factor sits at its floor (0.5028): the engine spends its *least* time right after being surprised.
- **Reference.** The constants are Berserk's (`clamp(0.113 + 0.026·Δ3iter + 0.026·Δprevmove, 0.50, 1.66)`, per
  `TIME-MANAGER-RESEARCH-2026-10-25.md` §2.2); ours kept the first term and dropped `Δprevmove`. SF18's main term is the
  same idea: `fallingEval = (11.85 + 2.24 * (bestPreviousAverageScore - bestValue) + 0.93 * (iterValue - bestValue)) / 100`
  (`src(SF18)/search.cpp:490`). Obsidian has it too (`0.025·Δmove`).
- **Class.** T, a design omission in a borrowed formula, not a coding error.
- **Count (no engine run needed).** From any finished 10+0.1 PGN set with scores: the share of our moves whose score is
  ≥ 340 millipawns below our previous move's score (340 is where the factor would reach 1.0). That is the share of moves
  the term would lengthen.
- **Fix sketch.** Knob `TM_DROP_PREV` (default 0): keep the previous move's final score in a file-scope variable (cleared
  on `ucinewgame` and when the side changes), and use `drop = max(0, drop3) + max(0, prev_move_score - score)` in
  `f_drop`. Clock games only; fixed-depth and node-limited results are untouched, so the bench stays identical.
- **Expected effect.** Small: −3 … +10 Elo at 10+0.1. `TM_NORM` may need re-levelling so the mean time per move stays equal.

### S3-3: a 10 ms hard limit can cut the first pass, and the move is then not searched at all

- **Defect.** `hard = max(0.01, min(5 * opt, 0.8 * rem - mo))` (`uci_main.cpp:217`, `cmd_go`). When the optimum collapses
  the hard limit is 10 ms, about 16,000 nodes. The first pass (a depth-2 pre-search of every root move, then the depth-3
  main pass) is not exempt from the clock. If it is cut during the pre-search, `reorder_legal_moves` returns the empty
  previous data (`search_engine.cpp:8836-8839`), `alpha_beta` returns 0 with `best_move` untouched (`:5052-5056`), and
  `get_engine_move` plays the first move of the move-ordering list (`:4895-4900`). That move has had no search.
- **When.**
  - No increment and `rem` below about `52 × Move Overhead` (1.56 s at the default 30 ms): the known weakness 1 of
    `TIME-MANAGER.md` §6, which says "plays at depth ~3". This finding is the worse case under it: sometimes no depth at all.
  - Any clock under about 60 ms, increment or not.
  - New detail: the collapse point scales with the UCI option. A network GUI that sets `Move Overhead` to 300 … 1,000 ms
    (common for online play) moves it to 15 … 52 s of clock in a no-increment game.
- **Reference.** Reckless never stops before depth 1 is complete (`reckless_src/src/time.rs:74-76`,
  `if td.completed_depth == 0 { return false; }`). SF18 shrinks the horizon under one second
  (`src(SF18)/timeman.cpp:92-94`, `centiMTG = scaledTime * 5.051`), so its reserve shrinks with the clock.
- **Count.** Fixed depth: a histogram of `num_iterations` after the first `alpha_beta` call (`:4433`), on game12 / eg60 /
  WAC; the share above 16,000 nodes is the share of positions where a 10 ms limit cuts the first pass. In games: two
  fields on the existing `TMCNT` line, `firstpass_cut` (`time_up` already true at `:4446`) and `fallback_first_move`
  (the branch at `:4895`).
- **Fix sketch.** Two independent parts.
  - `TM_LOWCLOCK=1` already exists and passed its test (10+0: +20.4, 95% +1.1 … +39.9, 800 games, 0 forfeits, 11-05);
    its record says "replicate, then ship" and it is still 0. This is a park trigger that has not been reread.
  - Knob `TM_FIRST_PASS` (default 0): the clock does not cut the first pass when `rem` is at least a few times its
    cost (say 100 ms); below that keep today's behaviour.
- **Expected effect.** No change with an increment larger than the overhead (all internal tests). Matters in sudden-death
  and online play.

### S3-4: the start rule ignores the hard limit

- **Defect.** `may_start_iteration` starts an iteration when `spent + 1.75 × last ≤ TM_SOFT_LIMIT × g_tm_scale`
  (`search_engine.cpp:4458-4464`). The scale reaches 5.6, and the hard limit is `min(5 × opt, 0.8 × rem − mo)`. When the
  hard limit is the smaller number, the rule starts an iteration that its own prediction says ends after the hard stop.
  That iteration is cut and its work thrown away (`TM_PARTIAL` 0).
- **When.** (a) `movestogo` ≤ about 5: hard / opt is 1.0 at 1 move to go, 1.7 at 2, 2.5 at 3, 4.2 at 5, so any scale above
  that ratio fires it; at 1 move to go, any scale above 1. (b) Playing on the increment with under about 0.3 s: opt is
  capped at `0.2 × rem`, hard is `0.8 × rem − 0.03`. (c) Scale above 5.
- **Class.** T, low: time is wasted, the clock is never overrun.
- **Count.** On the `TMITER` line (already behind `FIX_COUNT_LOG`): flag iterations where `soft × scale > TIME_LIMIT`,
  and how many of them are then cut. Needs a 40-moves-in-N run to see (a).
- **Fix sketch.** Knob `TM_START_HARD` (default 0): compare against `min(TM_SOFT_LIMIT × g_tm_scale, TIME_LIMIT)`.
- **Expected effect.** None at 10+0.1; a little time kept near a time control.

### S3-5: the pre-search opens with last iteration's first move, not its best move

- **Defect.** `alpha_beta` saves the root order as it was searched (`previous_search_data.moves_list =
  current_search_data.moves_list`, `search_engine.cpp:5200`). The next call's pre-search walks that order
  (`reorder_legal_moves`, `:8569`) and gives its one open-window search to index 0 (`:8829`). The swap that brings the
  best move to the front (`descending_sort_wrapper`, `:9852-9866`) runs only after the pre-search (`:9006`). So after an
  iteration whose best move changed (26-35% of iterations, `ROOT_SEARCH_MODEL.md` §3), and after every aspiration fail
  high, the pre-search's open window goes to the dethroned move and the real best move is scouted, fails high, and is
  searched again.
- **Class.** E, low. Pre-search values of prefix moves are discarded by the merge, and the best move sits in the prefix
  before any tail move, so tail values are not affected. The cost is nodes, softened by the transposition table. It also
  places the open-window pre-search defects (S2-4, S2-10) on a move that is no longer the principal variation.
- **Count (fixed depth).** In `reorder_legal_moves`, once per call: `ps_lead_not_best` = calls where the largest
  `top_score` of the previous prefix is not at index 0, and the nodes of the pre-search's first two searches in those calls.
- **Fix sketch.** Knob `PS_LEAD_BEST` (default 0): do the front swap on the previous data before the pre-search loop
  (moves and scores together, so index pairing holds); the later swap then finds index 0 already largest.
- **Expected effect.** Bench nodes −0 … −1%. Take it only as part of a pre-search change.

### S3-6: a negative clock value is treated as "no clock"

- **Defect.** `wtime` / `btime` start at −1 as "not given" (`uci_main.cpp:170`) and the clock branches test
  `mine >= 0` (`:201`, `:228`). A real negative value skips both, and with no depth or node limit `g_cfg` stays the base
  preset (`:189`): STANDARD, `TIME_LIMIT` 45 s (`search_engine.h:138-140`). The engine thinks for seconds on a clock that
  is already below zero.
- **Trigger (REPORTED, not checked in fastchess's source).** A match runner with a time margin may send the overdrawn
  value; our queues run `timemargin=250` (`selfplay/_queue_284.sh:182`). Cutechess does send negative values. It needs
  the engine to have overdrawn first, which the hard limit makes rare, so this is latent in practice.
- **Count.** `grep` the fastchess logs for `time -`; no engine work.
- **Fix sketch.** A `have_clock` flag set by the parser, and `mine = max(mine, 0)`. No knob needed: fixed-depth,
  node-limited and positive-clock behaviour is unchanged.

### S3-7 … S3-10: latent and robustness

| # | defect | where | fix sketch |
|---|---|---|---|
| S3-7 | **`go` on a mated or stalemated position.** `reorder_legal_moves` reads `moves_list[0]` of an empty list and makes that move. Neither `cmd_go` nor `get_engine_move` tests for it (`cmd_go` builds the legal list only inside the `TM_SINGLE` branch and only tests size 1). Undefined behaviour; `make_move` aborts on an empty move per the comment at `search_engine.cpp:4892-4894`. Analysis GUIs can send it. | `search_engine.cpp:8565`, `8783-8803`; `uci_main.cpp:221-226` | in `cmd_go`: no legal move → `bestmove 0000`, no search |
| S3-8 | **The node counter is `int`.** `num_iterations` overflows at 2,147,483,647 nodes, about 22 minutes at 1.6 M nodes/s. Reachable by `go infinite`, and by a 90+30 game's hard limit (1,580 s in `TIME-MANAGER.md` §2) if an S2-33-type iteration runs to it. After the wrap: negative `info nodes` / `nps`, the history decay test `num_iterations % DECAY_INTERVAL` misfires, `NODE_LIMIT` never trips. | `search_engine.cpp:4418`, `increment_node_count_with_decay` `:10284-10289`, `uci_main.cpp:191` | `long long` counter (a speed check is needed: it is touched at every node) |
| S3-9 | **Protocol.** (a) `go infinite` sends `bestmove` on its own when the loop ends (a mate-range score, or the depth cap `MAX_PLY − 4`); UCI wants it held until `stop`. (b) The book probe is skipped only for `infinite`, so `go depth N` / `go nodes N` from the start position return a book move with no search unless `OwnBook` is false. (c) `Move Overhead` is used only by the `TM_MODE ≥ 1` branch; `go movetime` and the old rule use a fixed 30 ms. | `uci_main.cpp:184-188`, `193-200`, `238-242`; `search_engine.cpp:4467` | wait for `stop` under `infinite`; skip the book when a depth or node limit is given |
| S3-10 | **Knob coverage at ply 1.** The no-cutoff exit of `minimizer` adds `depth_limit − cur_depth` to the best reply's history as a raw add and does not read `SV2_HIST_ALLNODE_GATE` (search v2's copy does, `search_v2.cpp:951-957`). So the C4 arm covered plies ≥ 2 only, the same shape as S2-31. One or two writes per root pass: negligible. | `search_engine.cpp:7752-7754` | read the knob there when C4 is reopened |

---

## 2. Checked and clean (do not re-read)

### Time manager and driver (`uci_main.cpp` `cmd_go`; `search_engine.cpp` `get_engine_move`)

- **The budget cannot overrun the clock.** The hard limit is at most `0.8 × rem − Move Overhead` in the `TM_MODE 1`
  branch, in every case (sudden death, increment, movestogo 1 … 50). After a hard stop at least `0.2 × rem` plus the
  overhead remains, except under the 10 ms floor (S3-3), where nothing could help.
- **Hard-stop latency.** The clock is tested every 2,048 node entries in every node function: the three that run at
  shipped settings (`minimizer` at ply 1 `:6668`, `sv2::negamax` `search_v2.cpp:439`, `sv2::qsearch` `:1394`) and the two
  v1 ones (`maximizer` `:7807`, `qSearch` `:9415`). That is about 1.3 ms. After `time_up` every node returns at entry and
  every loop tests it after each child. `pre_minimizer` has no test of its own (S2-20) but finishes its loop in
  microseconds.
- **Time not seen by the engine's clock.** Between `go` arriving and `search_start_time`: `stop_search`, the book probe
  (start-position games, first 30 plies), a legal-move list, a copy of the game history. `update_cache` returns at once
  (`PRINT_CACHE_STATS 0`). All small against the 30 ms overhead. The one large uncounted cost is `clearAllCaches` when
  the side to move differs from the previous `go` (`uci_main.cpp:246`); it never happens inside a game played for one side.
- **Movestogo.** Clamped to 50; the share is `0.95 / h`; time after the control is not counted; at one move to go the soft
  and hard limits are both `0.8 × rem − mo`. Same structure as SF18 (`timeman.cpp:126-128`).
- **Increment.** Counted for `h − 1` future moves; with any increment above the overhead `left` cannot collapse. On the
  increment alone the clock settles at a few seconds (worked from the formula).
- **Start rule and factors.** `elapsed` is the last iteration's whole duration, failed aspiration passes included. The
  three factors index correctly: the drop is read against the score three completed iterations back and is 0 until depth 6;
  stability counts completed iterations only; mate-range scores are excluded from the drop; the node share is cumulative
  over the search and counts main-loop nodes only (a known choice, `ROOT_SEARCH_MODEL.md` H2). `g_tm_scale` is reset to 1
  per search and changes only on a completed iteration from depth 5.
- **A cut iteration is never played.** `completed_move` / `completed_score` change only when `time_up` is false
  (`:4626-4631`), apart from the C9 exits (§3).
- **`go movetime`**: the search runs to the limit less 30 ms, no soft logic, as designed. **`go depth` / `go nodes`**: no
  clock (`TIME_LIMIT` 1e9), limits restored on every `go`. **`go nodes`** stops inside the iteration and discards it, as
  designed. **One legal move**: soft limit 0, one pass.
- **Ponder.** Not advertised, not parsed; `ponderhit` is an unknown command. Nothing half-built.
- **`stop`.** Applied after the first completed pass only; `g_uci_stop`, `g_iter_done` and `time_up` are reset in an
  order that leaves no window in which a `stop` is lost or carried into the next search. All `Config` writes happen
  after the previous search thread is joined.
- **Nothing is carried from move to move** in the root data: `preliminary_search_data` is local to one `get_engine_move`.

### Ply 1 (`minimizer`, `cur_depth == 1`, `search_engine.cpp:6833-7177`, and its exit `:7749-7782`)

- **Entry pruning.** Reverse futility and the null move exist only inside the `PLY1_NODE_PRUNE` block (`:6861`), 0 at
  header and never set by the preset or start-up code (`:2212`). No other node-level cut exists; the ply-1 razor is in a
  comment block. Confirmed.
- **Move-loop pruning** comes from `sv2::score_move<false>`: late-move pruning at remaining depth ≤ 5, so iteration depth
  ≤ 6 (reply index ≥ 5 / 10 / 17 / 26 at iterations 3 / 4 / 5 / 6); losing-capture SEE pruning at remaining depth ≤ 3,
  so iteration depth ≤ 4. History pruning and quiet SEE pruning are off. From iteration 7 nothing is pruned at ply 1.
  Confirmed.
- **A6 (child futility off at ply 1).** The gate is `(RootSide || cur_depth > 1)` (`search_v2.cpp:1190`). With the other
  conditions (`rd ≤ 4`, `depth_limit ≥ 5`) it removes futility at ply 1 only when the line's depth is exactly 5:
  iteration 5, or iteration 4 under a check extension. One shallow iteration per search: inert as an asymmetry.
  ☠️ But the gate is load-bearing. The ply-1 loop has no `using_fp` branch (`:6998-7034`): it takes whatever
  `score_move` returns as a searched score. A futility-pruned reply returns its static eval, which can lie below the
  node's current minimum when no reply has yet come under beta, and it would then become the node's value and best reply.
  Anyone making A6 "one rule for both sides" must add the `using_fp` branch (as `sv2::negamax` has, `search_v2.cpp:839-847`)
  in the same change.
- **A pruned reply cannot become a score.** A pruned reply returns the −MATE sentinel, which is +9,999,999 at ply 1 and
  never below `lowest_score`. It is stored in the carried scores and sorts that reply last on the next pass.
- **The "all moves pruned" return** (`:7169-7173`) is unreachable at ply 1: reply 0 is always searched, so the node's
  minimum always moves; the branch would need an empty reply list at a non-terminal node, and no path produces one (the
  draw returns refill an empty list, `:6843-6846`, `:9028-9029`).
- **Mate and stalemate at ply 1** are detected (`:7154-7168`). Mate-distance pruning (`FIX_MDP`, off at shipped values)
  also skips ply 1, so the node always fills its output.
- **The cutoff-history block** (`:7050-7147`) matches search v2's `quiet_cutoff_history` / `capture_cutoff_history`
  (`search_v2.cpp:240-315`) statement for statement: same bonus, same gravity path at shipped values
  (`ENABLE_HISTORY_SATURATION` on), same malus. The two extra branches it carries (`ENABLE_CUTCAL_LOG`, `ENABLE_QCUT`)
  are off. Only known difference: S2-31. It does not update the move cache, which
  search v2 nodes do (`:909-914`); the carried list does that job for prefix moves.
- **The exit** (`:7749-7782`): one raw history add (S3-10); the correction-history block and `store_node_tt` are off at
  shipped values (`CORR2` 0, `ENABLE_CORR_HIST` false, `ENABLE_NODE_TT` false). The best reply is always set when the exit
  is reached.
- **Time-up** returns 0 before any output is written; the root discards the pass.

### Carried reply lists (`SearchData` / `RootScore`)

- **Pairing holds on every path.** The pre-search indexes its results in the previous pass's root order
  (`:8569`, `:8847`, `:8953`). The merge swaps the best prefix entry to the front as a (move, entry) pair, keeps main
  entries for the prefix, appends pre-search entries for indices ≥ the prefix length, which the swap never touches, then
  sorts (move, entry) pairs (`:9846-9914`). Checked for: a normal iteration, a razor break, a root fail high, a fail low
  and its retry, the C9 exits, `ROOT_M0_CUT`, and time-up (the previous data is returned whole).
- **`minimizer` takes the lists by value**, sorts its own copy, and returns the sorted list and the scores of the replies
  it searched, in the same order (`:6853-6854`, `:7024`, `:7048`, `:7176`). The scores are a prefix of the list;
  `ascending_sort` permutes only that prefix.
- **A re-search of the same root move** (scout, then full window) starts from the same carried input and replaces the
  output entry whole (`entry = RootScore{}`).
- **Reply scores are read in one place only**, the ply-1 sort. Nothing prunes on them.
- **Staleness is bounded by design.** A reply list belongs to a position, which does not change. For a prefix move the
  order behind the last searched replies is the order of the last pass that searched them; for a tail move the pre-search
  regenerates the list every pass.
- **Mate at ply 1** gives an empty list, handled.
- The pre-search's `updatePV` for its first move tests a variable that is never assigned (`:8842`); harmless, the PV table
  is cleared straight after the pre-search (`:5049-5050`).

---

## 3. Known items, sharpened (no new ids)

- **S2-33 part (b): why one iteration could take 150 s.** The hard limit is `5 ×` the *unscaled* optimum
  (`uci_main.cpp:217`), while the search aims at `optimum × scale`. In the positions the factors judge easiest (best move
  stable for 10 iterations, ≥ 95% of the nodes, steady score) the scale is `0.563 × 0.778 × 0.503 × 0.95 = 0.209`. So the
  hard limit there is `5 / 0.209 ≈ 24×` the intended think. Game 56's 46.Kxg6 fits: optimum 29.9 s, intended about 6 s,
  stopped at 149.6 s. SF18's ratio is about 10× (maximum up to 6.7× optimum, typical total 0.6-0.7× optimum).
  - Fix sketch, knob `TM_HARD_SCALE` (default 0): the limit for the iteration about to start is
    `min(hard, K × TM_SOFT_LIMIT × g_tm_scale)`, K around 6-8, set where `may_start_iteration` says yes. With K = 8 the
    46.Kxg6 stop would have come at about 50 s, with K = 4 at about 25 s.
  - Count first, from existing PGNs: per 1,000 moves, moves whose time exceeds 4× and 8× the scaled soft limit (`TMITER`
    gives the scale), next to the hard-stop count (`TMCNT cut=`).
- **C9: the 0.94 per 1,000 moves is a floor, not the rate.** It was counted at 10+0.1 (#273), where the start rule makes
  hard-limit cuts rare. The root's clock tests run after every root move, the in-search test every 2,048 nodes; when root
  moves are cheaper than 2,048 nodes (late moves refuted from the table) the root test fires first. Where every move ends
  on the limit (`go movetime`, the UI presets) C9's share should be much higher. What it does there is adopt the partial
  pass when its score is inside the window, which is `TM_PARTIAL=1`'s rule, measured negative (−0.61 pp, the adopted move
  better in 42.2%, research doc step 3). Count with the existing fields: `root_clock_later / (cut + root_clock_later)` on
  the `TMCNT` line under `go movetime 1000`. If material, the fix is one line at each exit: set `time_up` before returning
  (`search_engine.cpp:5263-5267`, `:5628-5638`).
- **Weakness 1 of `TIME-MANAGER.md`** (no increment, low clock): see S3-3 for the no-search case, the `Move Overhead`
  scaling, and the unshipped `TM_LOWCLOCK`.
- **A6**: inert as an asymmetry (iteration 5 only), but its gate protects the ply-1 loop; see §2 before changing it.

---

## 4. Notes for whoever measures

- **Ply count in test games (REPORTED).** `TM_OPT_PLY` takes the ply from the position's fullmove number
  (`uci_main.cpp:213`). If the opening file gives no move number (plain EPD), a test game starts at ply 0 where a real game
  from the start position is at ply 16-20, and the optimum's share is about 30% lower in the first moves after the book
  (1.9% against 2.8% of `left`), 10% lower by ply 60. Check one line of the opening file before reading time-per-move
  numbers as deployment numbers.
- **Node-limited judges.** Under `go nodes N` every move ends in a cut iteration that is thrown away. Both arms carry the
  same handicap, so arm comparisons hold; the absolute level against SF18 / SF1.1 is understated by that waste.
- **Counting TM items needs a clock,** so most cannot run beside timed games. S3-1, S3-3 (the first-pass histogram) and
  S3-5 are fixed-depth counts and can. S3-2 and S3-6 need only existing PGNs and logs.
