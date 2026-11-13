# The root of the search in the reference engines (2026-11-13)

How Stockfish 19, Stockfish 11, Reckless and Triumviratus 8.1 run plies 0-1 of the search and the iterative-deepening
driver: the root-move record, ordering between and inside iterations, reductions and pruning at the root, aspiration
windows, the root's use of the transposition table, and what the root hands to the time manager. Written for search v2
stage S3 (a standard root beside ours, behind a switch). Our own root is audited in
[`active/S3-ROOT-AUDIT-2026-11-13.md`](../active/S3-ROOT-AUDIT-2026-11-13.md).

Read-only research: nothing was built or run. Every statement about an engine carries a file:line for the version below.
"INFERRED" marks a conclusion that is mine and not written in the source. Constants are quoted as written; each engine
has its own score unit, so constants are not comparable across engines without conversion.

## 0. Sources and versions

| engine | version read | where | files read |
|---|---|---|---|
| Stockfish 19 | `version = "19"` (`misc.cpp:57`) | `stockfish_19_linux/src/` | `search.cpp` (root driver `:195-632`, `search()` `:720-1643`), `search.h:138-166`, `timeman.cpp` (all), `thread.cpp:300-323`, `movepick.cpp:248-249`, `history.h:42,132` |
| Stockfish 11 | release 11 | `stockfish_11/stockfish-11-win/src/` | `search.cpp` (`:326-586`, `:596-1300`), `search.h:59-76`, `thread.cpp:55-63`. `timeman.cpp` NOT read |
| Reckless | the tree on disk (the movegen note of 11-11 records commit `aec5cb2`) | `E:\chess_data\engines\reckless_src\src\` | `search.rs:55-1079`, `time.rs` (all), `thread.rs:387-417`, `movepick.rs:55-109`, `types/score.rs:8-14` |
| Triumviratus | 8.1 | `E:\chess_data\engines\triumviratus_src\` | `docs/project/NOVELTIES.md` (all), `HISTORY.md` (searched, not read in full), `source/search/14_deepen.inc` (all), `15_threads.inc:380-485`, `13_search.inc:1780-1970` and its root conditions by search, `01_params.inc` (the `Asp*`, `Tm*`, `Root*` rows), `threads.h:70-85` |

Not independent votes. Reckless credits Stockfish and shares an author with it. Triumviratus started from Stockfish 19's
algorithms and rewrote them in its own code in 8.1; §2.4 lists only what it does differently, and where it matches
Stockfish 19 it adds no evidence. Stockfish 11 against Stockfish 19 is the one real "does it survive nine years of
testing" comparison in this note.

Stockfish 18 was not read for this note. Several comments in our code cite SF18 line numbers for root rules
(`search_engine.h:1407-1411`, `:1534-1539`); two of those rules read differently in SF19 (§2.1, LMR and the depth cut after
an alpha raise).

## 1. The common shape, in one paragraph

All four keep one record per legal root move for the whole search and never drop a move from it. Each iteration searches
**every** root move: the first with the full aspiration window, the rest with a null window and the ordinary late-move
reduction, re-searched when they beat alpha. Only the first move and moves that raise alpha keep a score; every other
move's score is set to minus infinity, and a **stable** sort after every root search therefore lifts the new best move
to the front and leaves all the others in the order they already had. The move loop at the root is the ordinary move
picker with the previous best move forced first; the root list decides which moves are allowed, not the order they are
tried in. There is no move-count, futility, history or exchange pruning at the root, no singular search, no null move.
None of the four prunes a root move, and none runs a separate ordering pass. The time manager reads only facts about
the best move (its score trend, how long it has been best, how often the best move changed, its share of the nodes).

## 2. Engine by engine

### 2.1 Stockfish 19

**Record.** `RootMove` (`search.h:138-166`): `effort` (nodes spent under the move, whole search), `score`,
`previousScore`, `averageScore`, `meanSquaredScore`, `uciScore`, two bound flags `inexactLower` / `inexactUpper`,
`previousScoreExact`, `selDepth`, `tbRank`, `tbScore`, `pv`, `previousPV`. Sort order: `score` descending, ties by
`previousScore` (`search.h:149-151`). The list is built once from the legal generator in generation order
(`thread.cpp:319-321`) and copied to each thread.

**Driver.** `iterative_deepening` (`search.cpp:270-632`).
- Iterations start at depth 1 (`rootDepth++` from 0, `:333-336`).
- At the start of each iteration every move's `score` is copied to `previousScore` and `pv` to `previousPV`
  (`:347-352`).
- Aspiration, every iteration and every PV line: `delta = 5 + threadIdx % 8 + abs(meanSquaredScore) / 10193`, window
  `averageScore ± delta` (`:376-379`). `meanSquaredScore` starts at `-VALUE_INFINITE * VALUE_INFINITE`
  (`search.h:157`), so the first iteration's delta is larger than the score range and the window is full (INFERRED from
  the arithmetic).
- Search depth of a pass: `max(1, rootDepth - failedHighCnt - 3 * (searchAgainCounter + 1) / 4)` (`:392-393`). One ply
  less per consecutive fail high; `searchAgainCounter` counts iterations the time manager asked to repeat at the same
  depth (`:356-357`, set at `:613`).
- After each pass: `std::stable_sort(rootMoves.begin() + pvIdx, rootMoves.begin() + pvLast)` (`:403`).
- Fail low: `beta = alpha; alpha = max(bestValue - delta, -VALUE_INFINITE); failedHighCnt = 0` (`:420-428`).
  Fail high: `alpha = max(beta - delta, alpha); beta = min(bestValue + delta, VALUE_INFINITE); ++failedHighCnt`
  (`:429-434`). Then `delta += 47 * delta / 128` (`:438`). There is no cap on the number of widenings and no jump to a
  full window.
- After the PV line: a second stable sort of the lines searched so far (`:493`).
- Mate bookkeeping: a mate found in an earlier iteration and lost in this one, or an exact loss score from an aborted
  pass, is replaced by the last good line (`:505-547`).
- MultiPV: an outer loop over `pvIdx`; each line is a full root search restricted to `rootMoves[pvIdx, pvLast)`
  (`:360-503`, filter at `:1134`). Skill level forces at least 4 lines (`:318-319`).

**The root inside `search()`.** `rootNode` is a template constant (`:725`).
- First move: `ttData.move = rootNode ? rootMoves[pvIdx].pv[0] : ...` (`:820`). The TT's own move is ignored at the root.
- Order of the remaining moves: the ordinary `MovePicker` (`:1111-1112`), with a low-ply history table that adds to quiet
  scores at plies 0-4 (`history.h:42`, `movepick.cpp:248-249`; filled with 102 at each search, `search.cpp:326`; updated at
  `:2052`). Moves not in the root list's active range are skipped (`:1134`). The root list's order is not used to order
  the loop.
- Skipped at the root: repetition and draw exits and mate-distance pruning (`:737`, `:785-801`), tablebase probe
  (`:923`), all shallow-depth move pruning (step 15, `:1166`), singular extension (`:1246`). The TT cut is `!PvNode`
  (`:874`). Razoring is `!PvNode` (`:991`). Child futility needs `!ss->ttPv`, and a PV node sets `ttPv` (`:822`, `:996`).
  Null move needs `cutNode` (`:1011`), and the root is called with `cutNode = false` (`:395`). Internal iterative
  reduction needs `!ss->followPV`, and the root sets `followPV` (`:772`, `:1051`).
- ProbCut (`:1057-1096`) and the small TT-based ProbCut (`:1101-1104`) carry **no** `PvNode` or `rootNode` condition in
  this text. As written they can return from the root without scoring a root move; whether that happens in practice was
  not checked.
- LMR at the root: the ordinary rule, `depth >= 2 && moveCount > 1` (`:1362`). There is no root-specific start, and no
  "never reduce a move that was recently best" rule: Stockfish 11's two root terms (§2.2) are gone. What the root gets
  instead comes from being a PV node on the PV: `r += 929` for `ttPv` (`:1161-1162`), then
  `r -= 3023 + PvNode * 1004 + (ttData.value > alpha) * 885 + (ttData.depth >= depth) * (816 + cutNode * 940)`
  (`:1316-1318`), `r -= moveCount * 65` (`:1323`), history through `statScore` (`:1342-1352`), and one ply back for PV
  nodes, `d = max(1, min(newDepth - r / 1024, newDepth + 2)) + PvNode` (`:1369`). A reduced search that beats alpha is
  re-searched, one ply deeper or shallower by its margin (`:1377-1387`).
- After a move raises alpha without failing high: `if (depth > 3 && depth < 12 && !is_decisive(value)) depth -= 3`
  (`:1534-1536`). No root condition, so at root depths 4-11 every later root move is searched three plies shallower
  after the first alpha raise. (Our header comment quotes SF18 as `depth > 2 && depth < 14 ... depth -= 2`,
  `search_engine.h:1536-1537`; SF19's numbers differ.)
- Recording a searched root move (`:1437-1506`): `effort += nodes` (`:1441`); `averageScore` and `meanSquaredScore` are
  moving averages whose weight is the share of the move's total effort spent in this search, between 12/32 and 24/32
  (`:1446-1468`); `score`, `selDepth`, bound flags and the PV are written only when `moveCount == 1 || value > alpha`
  (`:1471-1493`), otherwise `score = -VALUE_INFINITE` (`:1505`). `++bestMoveChanges` when a move other than the first
  takes over, first PV line only (`:1498-1499`).
- TT store of the root: written unless this is a second or later PV line (`:1621-1627`).

**Time manager.** Budget in `timeman.cpp`: `optimumTime = optScale * timeLeft` with
`optScale = min(0.012112 + pow(ply + 3.22713, 0.46866) * optConstant, 0.19404 * time / timeLeft) * originalTimeAdjust`
(`:112-124`), `maximumTime = min(0.8097 * time - moveOverhead, maxScale * optimumTime)` (`:136-138`). After each completed
iteration (`search.cpp:569-614`):

| factor | formula | needs |
|---|---|---|
| falling eval | `(11.48 + 2.30 * (bestPreviousAverageScore - bestValue) + 1.1 * (iterValue[iterIdx] - bestValue)) / 100`, clamp `[0.576, 1.728]` (`:573-576`); `iterValue` holds the last four iterations' scores (`:616-617`) | best score only |
| stability | `timeReduction = clamp(interpolate(rootDepth - lastBestMoveDepth, 4.96, 18.79, 0.639, 1.712), 0.629, 1.544)`; `reduction = (1.468 + previousTimeReduction) / (2.284 * timeReduction)` (`:579-584`) | best move identity |
| best-move changes | `1.077 + 2.229 * totBestMoveChanges / threads` (`:586`), the count halved each iteration (`:341`) | the event "a later root move raised alpha" |
| effort | `nodesEffort = rootMoves[0].effort * 100000 / nodes`; `clamp(interpolate(nodesEffort, 75800, 104510, 0.969, 0.714), 0.693, 0.838)` (`:571`, `:588-589`) | a node count for the best move |

Stop when elapsed exceeds `min(optimum * the four factors, maximum)` (`:591-611`); with one legal move the total is
capped at 500 ms (`:594-596`); past half of the total, the next iteration repeats the depth (`:613`). Inside the search,
`check_time` stops at `maximum` (`:2103-2130`). The driver stores `bestPreviousScore` and `bestPreviousAverageScore` for
the next move's search (`:247-248`).

### 2.2 Stockfish 11, where it differs

- **Record.** `score`, `previousScore`, `selDepth`, `tbRank`, `bestMoveCount`, `tbScore`, `pv` (`search.h:59-76`). No
  average, no effort.
- **Aspiration.** Only from `rootDepth >= 4`; below that the window stays infinite (`search.cpp:346-347`, `:433`).
  `delta = 21 + abs(previousScore) / 256`, centred on `previousScore` (`:435-438`). Fail low:
  `beta = (alpha + beta) / 2; alpha = max(bestValue - delta, -VALUE_INFINITE)` (`:480-487`). Fail high:
  `beta = min(bestValue + delta, VALUE_INFINITE); ++failedHighCnt` (`:489-493`); alpha is not moved.
  `delta += delta / 4 + 5` (`:500`). Depth of a pass `max(1, rootDepth - failedHighCnt - searchAgainCounter)` (`:453`).
- **`bestMoveCount`.** Incremented for the line's first move each time a pass resolves inside the window (`:496`).
- **Root LMR starts later and exempts recent best moves.** `depth >= 3 && moveCount > 1 + rootNode + (rootNode &&
  bestValue < alpha) && (!rootNode || thisThread->best_move_count(move) == 0)` (`:1117-1119`; `thread.cpp:57-63`). So at
  the root the second move is never reduced, the third is not reduced while nothing has reached alpha, and a move that
  has ever ended an iteration as best in this search is never reduced.
- **PV re-search at the root even on a fail high.** `value > alpha && (rootNode || value < beta)` (`:1222`).
- **Pruning at the root.** Shallow pruning `!rootNode` (`:997`), singular `!rootNode` (`:1043`), razoring `!rootNode`
  (`:823`), futility, null move and ProbCut `!PvNode` (`:832`, `:839`, `:891`). Internal iterative deepening needs
  `!ttMove` (`:932`), and the root's `ttMove` is always set (`:695`). There is no depth cut after an alpha raise
  (`:1272-1292`).
- **Root record update.** The same rule as SF19 without averages (`:1242-1270`).
- **Time manager.** Three factors, no effort term: `fallingEval = (332 + 6 * (previousScore - bestValue) + 6 *
  (iterValue[iterIdx] - bestValue)) / 704`, clamp `[0.5, 1.5]` (`:539-541`); `timeReduction = lastBestMoveDepth + 9 <
  completedDepth ? 1.94 : 0.91`, `reduction = (1.41 + previousTimeReduction) / (2.27 * timeReduction)` (`:544-545`);
  `bestMoveInstability = 1 + totBestMoveChanges / threads` (`:553`). One legal move stops at once (`:556`). Past 0.6 of
  the total the next iteration repeats the depth (`:566-569`). The budget formula (`timeman.cpp`) was not read.
- **Contempt.** The dynamic contempt is set from the root move's previous score (`:440-444`). A root-fed evaluation
  input that SF19 replaces with optimism (`search.cpp:382-383` in SF19).

### 2.3 Reckless

**Record.** `RootMove` (`thread.rs:387-399`): `mv`, `score`, `previous_score`, `display_score`, `upperbound`,
`lowerbound`, `sel_depth`, `nodes`, `pv`, `tb_rank`, `tb_score`. No average in the record; the driver keeps one running
average per PV line (`search.rs:63`).

**Driver.** `start` (`search.rs:55-298`).
- Iterations from depth 1 (`:78`). `previous_score = score` for every move at the start (`:94-96`).
- Aspiration from depth 1: `delta = 23 - min(eval_stability, pv_stability, 7)` (`:98`), then
  `delta += average * average / 26394` (`:115`); window `average ± delta` (`:117-118`). `average` starts as the previous
  search's best score (`:63`, `:297`; zero in a fresh thread, `thread.rs:272`) and after each resolved pass becomes
  `(average + score) / 2` (`:158-162`).
- Fail low: `alpha = max(score - delta, -INFINITE); beta = min(alpha + delta, beta); delta += 26 * delta / 128`
  (`:142-146`). Fail high: `alpha = max(beta - delta, alpha); beta = min(score + delta, INFINITE)`; `reduction += 1`
  unless the score is decisive; `delta += 60 * delta / 128` (`:147-156`). The depth reduction is declared per iteration
  (`:99`) and is not reset by a fail low. Depth of a pass `max(depth - reduction, 1)` (`:133`).
- Sort after each pass, by score descending (`:135`; Rust's `sort_by_key` is stable), and again over the searched lines
  (`:173`).
- `eval_stability` counts iterations whose best score stayed within 12 of the average (`:185-189`); `pv_stability` counts
  iterations with the same best move (`:191-195`). Both feed the aspiration delta and the time manager.
- The same mate bookkeeping as SF19 (`:197-222`).

**The root inside `search`.** `NODE::ROOT` is a compile-time constant (`:34-52`).
- First move: the TT move of the root position (`:378-384`, `:765`). There is no override from the root list. The root
  writes the TT whenever a move raises alpha (`:1069-1071`) and at the end (`:1190-1192`), so the TT move is the previous
  best unless the entry was overwritten (INFERRED).
- Order of the rest: the ordinary picker, with one root difference: at the root the picker re-scores the remaining
  captures and quiets before every pick (`movepick.rs:72-74`, `:92-94`), so the history learnt under earlier root moves
  orders the later ones.
- Skipped at the root: upcoming-repetition and draw exits, mate-distance pruning (`:326`, `:342-358`), tablebase probe
  (`:412`), eval-difference history update and hindsight depth changes (`:492`, `:506`), all move-loop pruning
  (`:793-851`), singular search (`:711`). TT cut `!NODE::PV` (`:386`), razoring `!NODE::PV` (`:538`), RFP `!tt_pv` with
  `tt_pv` true at PV nodes (`:374`, `:549`), null move and ProbCut `cut_node` (`:567`, `:638`), and the root is called with
  `cut_node = false` (`:133`).
- LMR at the root: the ordinary rule, `depth >= 2 && move_count >= 2` (`:861`). PV nodes get
  `reduction -= 519 + 437 * (beta - alpha) / root_delta` (`:884-885`) and two plies back,
  `reduced_depth = clamp(new_depth - reduction / 1024, 1, new_depth + 2) + 2 * PV` (`:919`). The deeper / shallower
  adjustment after a reduced search beats alpha is **not** applied at the root (`:927-930`).
- Root record (`:1011-1044`): `nodes += ...` (`:1015`); score, bounds, PV only when `move_count == 1 || score > alpha`
  (`:1017-1036`), else `score = -INFINITE` (`:1042`); `best_move_changes += 1` when a later move takes over, first line
  only (`:1038-1040`).

**Time manager.** Budget (`time.rs:31-58`): soft bound `(0.0594 - 0.0492 * exp(-0.0386 * fullmove)) * (main - overhead) +
0.75 * inc`, hard bound `0.7281 * (main - overhead) + 0.75 * inc`. After each iteration the soft bound is multiplied by
five factors (`search.rs:245-263`):

| factor | formula | needs |
|---|---|---|
| nodes | `max(3.1838 - 2.6554 * fraction, 0.5460)`, `fraction = root_moves[0].nodes / nodes` | a node count for the best move |
| score trend | `clamp(0.7426 + 0.0480 * (previous_best_score - score), 0.7214, 1.4031)` | best score, last search's score |
| PV stability | `max(1.2881 - 0.0440 * pv_stability, 0.7160)` | best move identity |
| eval stability | `max(1.2664 - 0.0416 * eval_stability, 0.8642)` | best score against the running average |
| best-move changes | `1.1500 + ln(1 + 0.2526 * best_move_changes)`, the count reset each iteration (`:89`) | the alpha-raise event |

The hard bound is checked every 2,048 nodes and never before depth 1 completes (`time.rs:73-83`).

Our earlier brief (`reckless-search-brief-2026-11-06.md` items 9 and 10) was checked against these lines and matches:
the delta shape (`:98-115`), one ply less after a fail high (`:147-154`), five time factors (`:245-263`).

### 2.4 Triumviratus 8.1: what differs from Stockfish 19

The record (`threads.h:70-85`), the store-only-if-first-or-raised-alpha rule, the effort-weighted averages
(`13_search.inc:1787-1816`, with `RootAvgChi 3`, `RootAvgMin 12`, `RootAvgMax 24`, `RootMsqMax 16`,
`01_params.inc:1637-1640`, the same numbers as SF19's `search.cpp:1447-1456`), the stable sort
(`14_deepen.inc:326-332`, `:688`, `:719`), the forced first move (`13_search.inc:625-626`), the root exemptions (pruning
`:1286`, singular `:1713`, tablebases `:772`, repetition `:590`), the mate bookkeeping (`14_deepen.inc:756-785`) and the four
time factors (`:232-249`) are Stockfish 19's design. They add no independent evidence. The differences:

| area | Stockfish 19 | Triumviratus 8.1 |
|---|---|---|
| aspiration base | `5 + threadIdx % 8 + abs(meanSq) / 10193` | `AspBase 6` + thread offset + `abs(mean_sq) / AspMeanSqDiv 10050` (`14_deepen.inc:660`; `01_params.inc:276-277`) |
| widening | `47 / 128` both sides | `AspGrowLo 50 / 128` after a fail low, `AspGrowHi 52 / 128` after a fail high (`:699-716`; `01_params.inc:283-284`) |
| written, off | none | `AspStable` (narrow the window while the best move repeats, `:663-667`), `AspLoKeep` (beta keeps part of the window after a fail low, `:701`), `AspFhStep` (give fail-high plies back gradually, `:705`): all 0 (`01_params.inc:286-287`, `:1676`) |
| time budget | share of the clock by game ply (`timeman.cpp:112-124`) | own: moves left estimated as `21.1 + 1.09 x pieces - 0.087 x ply` from its games, share grows with log of the reserve (`14_deepen.inc:95-111`) |
| factor constants | §2.1 | re-tuned (`TmFallBase 1170`, `TmFallPrev 240`, `TmFallIter 114`, `TmInstabBase 1091`, `TmInstabMul 2120`, `TmEffortLo 82932`, `TmEffortHi 101788`, `TmDeepenShare 528`; `01_params.inc:1322-1329`, `:1402`) |
| **surprise rule** | none | `x 1.2` on the move's total time when the opponent did not play the expected reply (`14_deepen.inc:824-840`, `TmSurpriseScale 1200`, `01_params.inc:1346`); silent while our clock is behind the opponent's (`TmSurpriseBehind 1`, `:1363`) |
| expected reply | none | written, off (`TmExpectScale 1000`, `01_params.inc:1335`): less time while the search still returns the predicted move at a score within `TmExpectMargin` (`14_deepen.inc:819-823`) |
| defence | none | `x 1.5` when the root score is between `-178` and `-1424` (`TmDefendMul 500`, `:841-856`; `01_params.inc:810-812`) |
| jump guard | none | a jump up of 209 printed centipawns or more against the previous move never cuts the time (`TmJumpGuard`, `:237-241`; `01_params.inc:837`) |
| budget corrections | none | less time with at least 24 pieces (`TmEarlySave`, `:187-191`), smaller share when behind on the clock (`TmBehindMul`, `:180-186`), per-expert shares (off) |
| root moves in analysis | none | analysis-only rules (`AnRootAlt` reduces ranks 2-3 less, `13_search.inc:2197`; `AnShadow`, `AnDirect`); `an_bits = 0` in games (`14_deepen.inc:532`) |

**The surprise rule in full.** At the end of a search the engine stores the position key expected after its move and
the PV's reply, the predicted next move, and the score and depth, but only when the PV has at least three moves
(`15_threads.inc:459-467`). At the start of the next search `expected = key matches`, `surprise = a key was stored and
it does not match` (`:399-400`). On a surprise the total time of the move is multiplied by 1.2 at every iteration's
verdict (`14_deepen.inc:831-839`). Its own record: `+6.23 ± 6.75` over 2,174 games at 40+0.4, `+0.7 ± 9.0` at 16+0.16, and
"neutral at 16+0.16, positive at 40+0.4" (`01_params.inc:177-180`, `:1338-1350`;
`docs/development/archive/DEVELOPMENT_8.0.md:857-882`, `:2369`). A check against Stockfish 19 with the rule on and off read
`-14.5 / -8.6` over about 120 games a side (`DEVELOPMENT_8.0.md:2398`), which resolves nothing. `NOVELTIES.md` does not
list the rule; the README does (`README.md:126`). The rule reads one fact from the root: the best line's first two
moves.

## 3. What the root feeds the time manager

| signal | SF19 | SF11 | Reckless | Triumviratus | data it needs from the root |
|---|---|---|---|---|---|
| score trend (falling eval) | yes | yes | yes | yes | the best move's score per iteration, and the last search's |
| best move stable for N iterations | yes (ramp) | yes (step at 9) | yes | yes | best move identity |
| best-move changes inside iterations | yes | yes | yes | yes | a counter at the alpha-raise site |
| best move's share of nodes | yes | no | yes | yes | a node counter per root move (read for the best only) |
| eval stability | no | no | yes | no | best score against a running average |
| expected reply / surprise | no | no | no | yes | the PV's first two or three moves, kept across moves |
| defence, jump guard | no | no | no | yes | best score |
| one legal move | cap 500 ms | stop | not seen in the lines read | cap `TmSingleMs 500` | the root list's length |

**None of these needs a score for every root move.** Each reads the best move's score, identity, PV or node count, or
counts an event. No engine here computes a best-against-second gap ("easy move"): the non-best moves carry minus
infinity, so the number does not exist in their record (INFERRED from the store rule; the old Stockfish "easy move"
code is not in either version read).

## 4. Consensus and disagreement

### 4.1 What Stockfish 19, Stockfish 11 and Reckless all do

Candidates under "adopt only what is universally superior". Triumviratus agrees on every row and is not counted.

| # | rule | SF19 | SF11 | Reckless |
|---|---|---|---|---|
| 1 | One record per legal root move, kept for the whole search | `search.h:138-168` | `search.h:59-78` | `thread.rs:387-399` |
| 2 | Every root move is searched in every iteration; none is pruned or skipped | no root prune `:1166`; loop `:1120-1552` | `:997`; `:964-1300` | `:793`; `:770-1079` |
| 3 | First move full window, the rest null window, re-search on `> alpha` | `:1362-1423` | `:1117-1228` | `:861-1003` |
| 4 | Late root moves are reduced by the ordinary LMR and re-searched when they beat alpha | `:1362-1392` | `:1117-1198` | `:861-937` |
| 5 | A score is kept only for the first move or a move that raised alpha; the others get minus infinity | `:1471-1505` | `:1248-1269` | `:1017-1043` |
| 6 | Stable sort after every root search, so the best goes first and the rest keep their order | `:403`, `:493` | `:462`, `:506` | `:135`, `:173` |
| 7 | The loop's first move is the previous best; the rest come from the ordinary picker, not from the list order | `:820`, `:1111` | `:695`, `:949` | `:380`, `:765` |
| 8 | No move-loop pruning, no singular search, no null move, no razoring, no TT cut at the root | §2.1 | §2.2 | §2.3 |
| 9 | Aspiration with a small delta that grows with the size of the score, widened geometrically with no fall-back to a full window | `:376-438` | `:433-500` | `:98-156` |
| 10 | One ply less for the next pass after a fail high | `:392-393`, `:433` | `:453`, `:492` | `:133`, `:151` |
| 11 | After a fail low the upper bound comes down toward the old alpha | `:422` | `:482` | `:144` |
| 12 | The root's own result goes into the TT (first PV line only) | `:1621` | not read at this depth | `:1190` |
| 13 | Best-move changes are counted at the alpha-raise site and feed the time manager | `:1498`, `:586` | `:1262`, `:553` | `:1038`, `:260` |
| 14 | Time: an optimum scaled by root factors, tested between iterations; a hard maximum tested inside the search | `:569-614`, `:2103` | `:535-572` | `:245-280`, `time.rs:73` |
| 15 | No separate ordering pass over the root moves, and no pruning on a previous iteration's root scores | whole driver | whole driver | whole driver |

Row 15 is an absence, so it is evidence only that the giants' system does not need one, not that one cannot pay in
another system.

### 4.2 Where they differ (each a legitimate option)

| choice | SF19 | SF11 | Reckless |
|---|---|---|---|
| aspiration centre | effort-weighted average of the move's scores | previous iteration's score | running mean `(avg + score) / 2`, seeded from the last search |
| first aspirated depth | 1 (full by arithmetic at depth 1) | 4 | 1 |
| delta | `5 + meanSq / 10193` | `21 + abs(prev) / 256` | `23 - stability (max 7) + avg² / 26394` |
| growth | `x (1 + 47/128)` | `+ delta / 4 + 5` | `26/128` after low, `60/128` after high |
| beta after a fail low | `= alpha` | `(alpha + beta) / 2` | `min(new alpha + delta, beta)` |
| fail-high depth cut resets on a fail low | yes | yes | no |
| root LMR starts | move 2 | move 3, move 4 while nothing reached alpha | move 2 |
| never reduce a move that was best | no | yes (`bestMoveCount`) | no |
| PV credit in the reduced depth | `+1` | none (`r -= 2` for ttPv) | `+2` |
| deeper / shallower re-search at root | yes | not in this version | no |
| depth cut after an alpha raise | `-3` at depths 4-11, root included | none | none seen |
| ProbCut at the root | not excluded in the text | excluded | excluded (`cut_node`) |
| root's first move from | the root list | the root list | the TT |
| picker at the root | ordinary + low-ply history | ordinary | ordinary, re-scored before every pick |
| node share in time | yes | no | yes |
| eval stability in time and aspiration | no | no | yes |

Two readings of this table. First, Stockfish dropped both of its own root-LMR safeguards between 11 and 19 (the later
start and the best-move exemption), while adding strong PV-node terms to the general formula; the protection moved from
"root rules" to "PV rules". Second, every version reduces by **move number in picker order**, with the history score in
the formula. The reduction is not keyed on the root list's order and not on a root score.
