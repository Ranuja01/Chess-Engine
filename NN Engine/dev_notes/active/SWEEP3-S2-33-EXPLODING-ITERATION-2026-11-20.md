# Sweep 3, S2-33: an iteration that explodes and runs to the hard stop (2026-11-20)

A read-only analysis. Nothing was run (timed games were on the box) and no source file was changed. Every code claim
carries a file and line as read on 2026-11-20 plus the symbol name, because line numbers move.

Each finding is marked:
- **VERIFIED** = I read the code that executes at shipped settings (`V2_PRESET=shipped`, `SEARCH_ARM 1`).
- **REPORTED** = inferred from the code and the game record; it needs the run in §5 to confirm.

Register entry: `SEARCH-DEFECT-REGISTER-2026-11-18.md` §6d.

---

## 0. Short version

1. **Most likely cause (search).** A mate score is never written to the transposition table (TT, the cache of searched
   positions) or to the qsearch cache. `FIX_TT_MATE` is 0 in production. Until a mate is in view every result is an
   ordinary score, the TT merges the many move orders that reach the same position, and a pawn ending costs almost
   nothing (23 iterations in 2.2M nodes). In the first iteration that reaches the mate, every position inside the mate
   proof has a mate value, none of them is stored, and the proof is searched as a tree with no merging. Its size is
   then (defender's legal moves) to the power (defender's plies).
2. **The game supports it.** The first-mate iterations of the next three moves cost 172.3M, 42.9M and 21.6M nodes for
   mate in 12, 10 and 9: a factor of 2.0 per extra mate move. That line gives about 345M nodes (118 s) for mate in 13
   and 690M (235 s) for mate in 14. The hard stop came at 149.6 s.
3. **Four things make it worse, each correct on its own:** move-loop pruning is off while a node's best is mated
   (`FIX_MATE_GATE`), reverse futility is off against a mate bound, the null move is refused in pawn endings, and the
   aspiration loop walks four narrow windows before the full one with no shortcut for a mate-sized fail-high.
4. **Cheapest decisive test:** the position after 46...Kb3 at `go depth 22`, production against `Knobs FIX_TT_MATE=1`.
   Production is known to need 172M nodes there and to finish. Predicted with the knob: under 10M.
5. **Time manager.** No reference engine stops an iteration between the soft limit and the maximum either. They are
   safe only because their iterations do not jump. Proposed safety net `TM_ITER_CAP`: stop a running iteration at
   `max(optimum, 3 × scaled soft limit)` while nothing in that iteration has put the current best move in doubt. On
   46.Kxg6 it stops at 29.9 s instead of 149.6 s. Predicted to fire 1 to 5 times per 1,000 moves at 10+0.1.
6. **29.Rg5 is a different mechanism.** It was the time manager's own scale factors, reacting to a falling score, that
   allowed two extra iterations. The iteration finished 52 s inside the hard limit, and 217M nodes for depth 18 is what
   normal growth from that game's depth-16 costs predicts.

---

## 1. What happened (facts)

| fact | value | source |
|---|---|---|
| position | `8/6K1/6p1/6Pp/2k2P1P/2P5/8/8 w - - 1 46`, White to play 46.Kxg6 | `ui_games/ui_games.pgn`, game 56 |
| the game's record of the move | depth 25, 1,228 nodes, 149.6 s, 1:25 left afterwards | PGN comment at move 46 |
| the cold replay | depths 3 to 25 in 0.68 s and 2.2M nodes; depth 26 never finished | register §6d |
| optimum / hard limit | 29.9 s / 149.6 s | recomputed below |
| the next moves | 47.f5 mate in 12, depth 22, 172.3M nodes, 58.9 s; 48.Kh7 mate in 10, depth 20, 42.9M, 13.3 s; 49.g6 mate in 9, depth 19, 21.6M, 7.5 s; 50.f6 mate in 9, depth 16, 1.2M, 0.4 s | PGN comments |
| the move before | 45.g5: depth 27, 62.4M nodes, 21.8 s, +25.43, no mate | PGN comment |

**The limits, recomputed from `cmd_go` (`uci_main.cpp:207-220`).** With 224 s left, a 10 s increment and game ply 90:
- `left = 224 + 10 × 49 − 0.03 × 52 = 712.4`
- share `= 0.012 + 93^0.45 × 0.0039 = 0.0420`
- optimum `= 29.9 s`
- hard `= min(5 × 29.9, 0.8 × 224 − 0.03) = 149.6 s`

Both match the game. VERIFIED.

**Why the game shows 1,228 nodes and the replay 2.2M.** The TT is kept between our moves. `clearAllCaches` runs only on
`ucinewgame` or when the side we search for changes (`uci_main.cpp:246`), and `update_cache` only prints
(`search_engine.cpp:4198`). The search for 45.g5 reached depth 27, which is depth 25 seen from two plies later. So in
the game every iteration up to 25 was answered from the table, and depth 26 was the first new work. The replay started
cold, paid 2.2M nodes for the same 23 iterations and hit the same wall. **The wall does not depend on the table being
warm.** VERIFIED for the mechanism; the two node counts are as recorded.

**Speed was normal.** The neighbouring moves ran at 2.9 to 3.2M nodes per second (172.3M / 58.9 s, 42.9M / 13.3 s,
21.6M / 7.5 s). So the 150 s was about 440M nodes of real search, not a slow loop.

**One caution.** The depth-26 iteration printed nothing, so nobody has seen its score. That it is "the iteration where
the mate first appears" is an inference from the next move. REPORTED; §5 tests it.

---

## 2. How the search runs here (the parts that matter)

Terms, once:
- **Iteration:** one pass of iterative deepening at a fixed nominal depth.
- **Aspiration window:** a narrow score window around the previous iteration's score. A result outside it "fails high"
  or "fails low" and the iteration is searched again with a wider window.
- **LMR (late-move reduction):** moves late in the ordering are first searched to a smaller depth (a "scout").
- **Fail-soft:** a node may return a score beyond the bound it was asked about.
- **Mate score:** `9,999,999 − (game ply at which the mate lands)`. Anything at or above 9,000,000 in size is a mate.

### 2.1 The driver

`get_engine_move`, `search_engine.cpp:4467-4703`:
- Iterative deepening runs while `score < 9000000`, so it stops at the first mate score (`:4467`).
- Aspiration runs when the previous score is not a mate (`:4518`). Shipped values: `ASPIRATION_DELTA 500`,
  `ASPIRATION_MIN_DEPTH 5`, `ASPIRATION_MAX_WIDENINGS 3`, `ASPIRATION_WIDEN_PCT 200`, `ASPIRATION_RECENTER 0`
  (`search_engine.h:3535-3543`; none is set by the preset).
- With a previous score of +25.43, the windows at depth 26 are (24.93, 25.93), then beta 26.43, 27.43, 29.43, then the
  full window. That is **up to five full `alpha_beta` calls at depth 26**. Every call re-runs the pre-search at depth
  25 (`reorder_legal_moves`, `:5030`, `:8553`).
- A fail-high whose value is a mate takes the same slow ladder. `beta_win = prev_score + hi` is used when recentring is
  off, and also when it is on and the score is a mate (`:4580-4583`). The search depth is never lowered on a fail-high.

VERIFIED.

### 2.2 The root

`alpha_beta`, `search_engine.cpp:5013-5662`:
- After move 0 fails high, move 1 is still searched before the first beta test (S2-6; `:5183`, `:5269`, `:5614`).
- With a mate-sized alpha that search has a mate-range null window, and the root razor is off, because it requires
  `alpha < 9000000` (`:5303`, `:5321`).

VERIFIED.

### 2.3 Interior nodes: where the TT is read and written

`sv2::negamax` and `sv2::score_move`, `search_v2.cpp`:
- A node does not probe its own entry. The parent probes the child (`:1060-1072`) and the parent stores the child's
  result (`tt_store`, `:1084`, `:1275`, `:1300`, `:1368`).
- `store_node_tt` is inert: `ENABLE_NODE_TT` is false (`search_engine.h:3095`, `search_engine.cpp:5689`).

VERIFIED.

### 2.4 Mate scores never enter a table

- `addToSearchEvalCache` returns early when `tt_score_storable` is false (`cache_management.h:1195`).
- `tt_score_storable` is false for any score of size 9,000,000 or more unless `FIX_TT_MATE` is set
  (`cache_management.h:1171-1177`).
- `FIX_TT_MATE` has header value 0 (`search_engine.h:3315`), is not set by the preset (the gen 6.6 note at
  `search_engine.cpp:2739-2742` says it stays off) and is read from the environment at `:3323`.
- `addToQCache` refuses mate scores too (`cache_management.h:1078`).
- A score of exactly 0 is refused as well (`ENABLE_TT_STORE_DRAW` false, `cache_management.h:1195`).

VERIFIED.

### 2.5 Pruning inside a mate proof

| rule | what happens | where |
|---|---|---|
| Reverse futility | needs `beta` below mate range | `search_v2.cpp:524` |
| Null move | refused for a side with only pawns, or with fewer than 7 men and no queen, or fewer than 4 with a queen (`NULL_NPM_GUARD 2`). Black never qualifies in this ending; White only after promoting | `isUnsafeForNullMovePruning`, `search_engine.cpp:10386-10411` |
| Late-move, history, SEE and futility pruning | all off while every move searched so far at the node is mated (`FIX_MATE_GATE`, shipped gen 6.5). In a lost position that is every defender node | `search_v2.cpp:1024`, `:1136`, `:361` |
| Mate-distance pruning | off (`FIX_MDP 0`) | `search_v2.cpp:448` |

VERIFIED.

### 2.6 LMR in light endgames

- With no pieces the phase score is 128, so `reduced_search_depth` returns `base` straight away
  (`search_engine.cpp:10472-10475`, `FIX_LMR_EG 0`).
- `base` is `DEPTH_REDUCTION[depth_limit]`, which is **12 for every depth from 14 up** (`search_engine.h:172-175`).
  This is an absolute target depth, not a number of plies removed.
- So at iteration 26 in a pawn ending every reducible late move is scouted to absolute depth 12, whatever the move
  number or the ply. From ply 11 on the scout is a bare qsearch.
- Re-search rules (`search_v2.cpp:1308-1327`):
  - a scout that beats the node's bound is searched again at full depth (`FIX_RESEARCH`);
  - a fail-low within `VERIFY_MARGIN` (16,000 millipawns, `search_engine.h:3221`) is searched again at
    `depth_limit − 2` (`VERIFY_RESEARCH_REDUCTION 2`).

VERIFIED.

**Consequence.** "Depth 26" in a pawn ending is not a 26-ply full-width search. It is a full-depth first-move chain
plus depth-12 scouts plus re-searches. Its cost is set by how many re-searches fire and how many of them the TT
answers, not by the depth. That is why depths 14 to 25 are nearly free, and why the cost can jump by orders of
magnitude when one class of re-search switches on everywhere at once.

---

## 3. Hypotheses, ranked

### H1. Mate scores are not storable, so the mate proof is searched without transpositions

**Mechanism VERIFIED, cause REPORTED.**

Inside the proof every node has a mate value. For those nodes:
- The parent's store is refused (§2.4), whether the value is exact or a bound.
- The defender must be shown to lose after every legal move. Each late defender move goes through this sequence:
  1. it is scouted to depth 12, which cannot see a 25-ply mate and comes back with an ordinary score that looks like an
     escape;
  2. it therefore "beats the bound" and is searched again at full depth (`was_reduced && rs <= ra`,
     `search_v2.cpp:1315`);
  3. that search finds the mate, and the result is refused by the TT.
- The same defender position reached by another king route is searched from scratch every time.

The proof tree is therefore about `b^n` nodes, with `b` the defender's legal moves and `n` the defender's plies. With a
working table it is the number of distinct positions, a few thousand in a pawn ending.

Evidence from the game:

| move | mate in | first-mate iteration | nodes | ratio to the row below |
|---|---|---|---|---|
| 47.f5 | 12 | depth 22 | 172.3M | 4.0 over two mate moves |
| 48.Kh7 | 10 | depth 20 | 42.9M | 2.0 |
| 49.g6 | 9 | depth 19 | 21.6M | — |
| 50.f6 | 9 | depth 16 | 1.2M | (off the line) |

- 172.3M nodes over 11 to 12 defender plies is an effective defender branching of about 5, the legal-move count of a
  king with a pawn.
- Each of these searches started with a table full of ordinary results from the move before and still paid tens of
  millions of nodes. The mate proof is rebuilt every move.

Why depths up to 25 are cheap: no line reaches a mate, so every value is ordinary and storable.

### H2. The mate-range pruning gates make the proof full width

**VERIFIED code, size REPORTED.**

The four rules of §2.5 all switch off inside the proof. The largest is `FIX_MATE_GATE`. Late-move pruning normally
keeps 2 moves at remaining depth 1 and 5 at remaining depth 2 (`LMP_BASE 1`, `LMP_SCALE 1`: the threshold is
`1 + rd²`). A mated defender node keeps all of them.

Each rule is right, and Stockfish has the same guards. They are listed because they set `b` in H1 to the full legal
count. This is a multiplier on H1, not a separate cause.

### H3. The re-search rules have no mate guard

**VERIFIED code, firing REPORTED.**

- **The wasted scout.** In a mated defender node the depth-12 scout of every late move is pure cost. It cannot confirm
  the mate, and the bound it stores is useless to the full-depth search that follows.
- **A fragile proof.** For the attacker, a late move's depth-12 scout cannot see the mate either. It fails low against
  a mate-sized alpha by millions of millipawns, so no near-miss re-search follows (`(ra − rs) < VERIFY_MARGIN` is
  false, `:1312`). A mate is therefore found only when the mating move is searched first. A node that proved a mate on
  one visit can fail to on the next if the ordering changed, and an inconsistent result provokes more re-searches.
- **Mate against mate is always a "near miss".** Two mate scores differ by plies (1 to 60) while the margin is 16,000.
  A defender move whose scout does return a mate is searched again at `depth − 2` (`:1315-1316`). If that result lands
  on the other side of the bound, `SV2_VERIFY_PROMOTE` adds a full-depth search (`:1351-1362`). A mate found by a
  shallow search is already decisive, so the deeper look is waste.

How often the third case fires here is unknown. From ply 11 on the scouts are qsearches and rarely return a mate. §4
counts it.

### H4. The aspiration ladder and the root

**VERIFIED code; a multiplier of at most about 5.**

- Up to four narrow passes fail before the full window (§2.1). Each is a complete depth-26 `alpha_beta` call with its
  own depth-25 pre-search.
- Each failing pass also searches root move 1 in a mate-range null window (§2.2).
- Stockfish 11 and 18 and Reckless open the window from the returned value, which for a mate goes straight to
  infinity, and lower the search depth on each fail-high (SF11 `search.cpp:453`, `:491`; SF18 `search.cpp:372-373`,
  `:412`; Reckless `search.rs:133`, `:147-155`).
- Whether our narrow passes are cheap (they fail high on ordinary scores, which are storable) or expensive is not
  knowable by reading. The per-pass log in §4 answers it.

### H5. No mate-distance pruning

**VERIFIED code; expected to be a factor of a few here.**

Mate-distance pruning stops a line once it is deeper than a mate already found. Without it, a line that can no longer
mate faster is still searched to full depth, plus up to three check extensions (`CHECK_EXTENSION 3` per path,
`search_v2.cpp:1035`).

The first mate is found near the horizon, so the saving is the last one to four plies of the sibling subtrees. It grows
on the later moves, where the depth is far beyond the mate.

### H6. Qsearch against a mate bound

**VERIFIED code; a constant factor per horizon node.**

- The stand-pat cut cannot fire against a mate-sized beta.
- A node in check searches every evasion down to `MAX_QDEPTH 10` (`search_v2.cpp:1435-1467`).
- Promotions are exempt from delta pruning (S2-32).
- Mate results are not kept in the q-cache (§2.4).
- It is bounded: the attacker's qsearch moves are captures and promotions only, so checks do not chain.

### Checked and not the cause

| candidate | finding |
|---|---|
| A depth- or ply-indexed table ending near 25 or 26 | None. `MAX_PLY` is 64 and the iteration cap is 60 (`search_engine.cpp:4455`). `DEPTH_REDUCTION` has 64 entries with a guard beyond (`depth_reduction`, `:495`). `LMR_PRODUCT_RED[64][64]` is clamped (`:10484`). `FUTILITY_MARGINS_EFF[4]` is indexed only at remaining depth 2 to 4. `MOVE_POOL_PLIES` is 128 with a fallback buffer. Extensions need `depth_limit + 2 < MAX_PLY`. The only table edge nearby is the plateau that starts at 14 (§2.6). VERIFIED |
| The 32-bit node counter (`int &num_iterations`) | Not here: about 440M nodes is far below 2^31. **Latent for long controls:** at 3M nodes per second it overflows after about 12 minutes, and the hard limit at 90+30 is 630 to 1,580 s (`TIME-MANAGER.md` §2). Signed overflow is undefined behaviour and feeds `num_iterations % DECAY_INTERVAL` (`:10289`) and the nps line. A `long long` counter is a release-readiness item |
| S2-2, the endgame qsearch skip | Not involved. It needs a line target below 6 (`search_v2.cpp:462`). At iteration 26 the smallest targets are 12 (LMR) and 24 (re-search), and the null move is refused. VERIFIED |
| S2-5, the stuck history decay interval | Not a cause. The interval is 200,000 in this ending by its own rule (`search_engine.cpp:4396-4399`). Decay runs about 2,000 times in a 440M-node iteration, which hurts ordering a little |
| Passed-pawn extension, pawn-ending prune gate | Off: `PE_PASSER_EXT 0`, `PE_PRUNE_GATE false` (`search_engine.h:2893-2896`), not set by the preset. VERIFIED |
| Repetition and 50-move handling | No loop. One relative of H1: a value of exactly 0 is never stored, so a node whose value is a repetition draw is searched again on every transposition. Rare at +25; counted in §4 |
| Pre-search and root razor at depth 26 | The pre-search runs once per aspiration pass, so up to five times, each a depth-25 search that the table mostly answers. The razor is switched off by a mate-sized alpha (§2.2). Both are multipliers inside H4 |

---

## 4. Count-only instrumentation plan

**Rules.**
- Counters change no decision.
- Hot-path counters go behind the existing counter-build macro family (`FIXCNT_*`, compiled only with
  `EXTRA_FLAGS=-DFIXCNT_BUILD`, `search_engine.cpp:185-221`).
- Cold counters (per pass, per iteration) go behind the existing `tm_log` flag, which is read once into a `static const
  bool` from `FIX_COUNT_LOG` (`:4445`), or behind `Config::ENABLE_ITER_LOG`.
- No bare `getenv` anywhere in search code.
- The counter build's bench must equal production (254 / 35,922,538).

### 4.1 Free today, no code

- `ENABLE_ITER_LOG=1` prints `[iter] d= cum_nodes=` per iteration (`search_engine.cpp:4701`).
- `FIX_COUNT_LOG=1` prints `TMITER` per iteration and `TMCNT` per move (`:4657`, `:4983`).
- `ENABLE_LMR_COUNTERS=1` gives scouts, full re-searches, near-miss re-searches and the nodes under them, by side
  (`sv2::print_counters`).
- The stats line already has `qnodes`, and the `[aspiration]` line has windows, fails and fallbacks.

### 4.2 New counters

| # | counter | site | what the outcome means |
|---|---|---|---|
| C1 | **Per-pass log** (cold): depth, attempt, window, returned score, nodes in the pass, pre-search nodes in the pass, seconds | the aspiration loop in `get_engine_move`, after each `alpha_beta` call (`:4529`, `:4558`), behind `tm_log` | Says which pass exploded. Narrow passes cheap and the full-window pass huge with a mate score: H1 to H3. Every pass large: H4 is a real factor of 5. A huge pass with an ordinary score: not a mate effect at all, so H1 is wrong for this position |
| C2 | **Heartbeat** (cold): every 10 s inside a pass, the root move index, nodes so far and best so far | the root loop in `alpha_beta` after each root move (`:5518`), behind `tm_log` | An exploding iteration is silent today, since `info` lines print only on completion. Shows whether the time went into move 0 or the later moves |
| C3 | `tt_refused_mate[flag]` and `tt_refused_zero`, against all store attempts; `q_refused_mate` | `addToSearchEvalCache` and `addToQCache` (`cache_management.h:1195`, `:1078`), `FIXCNT` macro | H1's precondition. Expect about 0% of stores refused in iterations up to 25 and a large share (above 20%) in the mate iteration |
| C4 | **Repeated mate subtrees:** a count-only side table (2^20 keys, counter build only) of children whose store was refused for a mate, with depth. Count refusals whose key is already there at equal or greater depth, and sum the nodes spent under them | around the child searches in `sv2::score_move` (`:1080`, `:1268`, `:1293`, `:1344`) | The direct measure of H1: the share of the iteration's nodes spent re-proving a mate already proved. Above 80%: H1 is the cause. Below 30%: look at H3 and H4 |
| C5 | Nodes by window class at node entry: ordinary null window; ordinary open window; alpha in mate range (the side to move must mate faster); beta in losing-mate range | `sv2::negamax` after the horizon test (`:472`), `FIXCNT` macro | Where the nodes are. Mostly in mate-range windows: the mate machinery. Mostly ordinary: H4's narrow passes or something else |
| C6 | Pruning gates: reverse futility refused only by the mate guard; move-loop prunes refused only by `best_mated`; futility prunes taken with a mate-sized alpha | `search_v2.cpp:524`, `:1136-1151`, `:1202`. `FIXCNT_PRUNE` exists for the v1 path and is not ported to search v2 | Sizes H2. If the `best_mated` refusals are a small share of the moves in mated nodes, H2 is minor |
| C7 | Re-searches by class: full re-searches whose scout was ordinary and whose result is a mate (the wasted scout); near-miss re-searches where both values are mates; promote searches; nodes under each | `sv2::score_move` `:1308-1370`, beside the `ENABLE_LMR_COUNTERS` counters | Sizes H3. A large mate-against-mate count means a mate guard on the margin test is worth a knob |
| C8 | Nodes that mate-distance bounds alone would cut | `sv2::negamax` at `:447`. `FIXCNT_NODE(mdp_cut)` exists in the v1 `minimizer` (`search_engine.cpp:6681`) and is not ported | Sizes H5. Above 30% of nodes: `FIX_MDP` is a large factor. Below 5%: it is not |
| C9 | Maximum ply reached per iteration; maximum `depth_limit` seen; qsearch nodes in check; maximum q-depth | `sv2::negamax` and `sv2::qsearch` entries, `FIXCNT` macro | Rules extension or qsearch runaway in or out (H6). Expect a maximum ply of 26 + 3 + 10 |
| C10 | Root: the razor refused only by `alpha >= 9000000`; root passes whose move 0 failed high with a mate score, with move 1's nodes | `alpha_beta` `:5303`, `:5321`, `:5183`, `:5518` (cold) | Sizes the root part of H4, and what `ROOT_M0_CUT` saves in this case |
| C11 | **Exploding iterations in games:** iterations whose nodes exceed 20× the previous iteration's (previous at least 10,000 nodes), and moves ended by the hard stop, per 1,000 moves, on the `TMCNT` line | the iterative-deepening loop after an iteration (`:4693`), cold | The register's "count next". Says how rare S2-33 is in ordinary play, and so how to rank the fixes |

### 4.3 Cheapest decisive experiments (fixed depth, node-bounded, one process per setting)

All are `position fen …` then `go depth N` with a `nodes` bound, so they are deterministic. Run them when no timed
games are on the box.

**E1. The one to run first.** It completes, so it gives full numbers.
- Position after 46...Kb3: `8/8/6K1/6Pp/5P1P/1kP5/8/8 w - - 1 47`, `go depth 22`, cold table.
- Production is known to take 172.3M nodes (from a warm table in the game, so the cold number may differ somewhat).
- Arms:
  1. production, with `ENABLE_ITER_LOG=1` and `ENABLE_LMR_COUNTERS=1`;
  2. `Knobs FIX_TT_MATE=1`;
  3. `Knobs FIX_TT_MATE=1,FIX_MDP=1`;
  4. `Knobs FIX_MDP=1`.

**Registered prediction for E1:**

| arm | prediction |
|---|---|
| 2 (`FIX_TT_MATE=1`) | the mate iteration under 10M nodes, a cut of more than 15× |
| 3 (with `FIX_MDP=1`) | a further 1.2× to 3× |
| 4 (`FIX_MDP=1` alone) | under 2× |

If arm 2 saves less than 3×, H1 is wrong and the C1 to C7 counters decide.

**E2. The incident itself.**
- `8/6K1/6p1/6Pp/2k2P1P/2P5/8/8 w - - 1 46`, `go depth 26 nodes 700000000`, production and `FIX_TT_MATE=1`.
- The bound is about 4 minutes of search, well below the counter's 2^31.
- Prediction: production does not finish inside 440M nodes; the knob arm finishes depth 26 inside 20M and prints a
  mate score. That would also confirm depth 26 is the mate iteration.

**E3. One-knob ablations on E1, to split the multipliers.**

| knob | what it tests |
|---|---|
| `ASPIRATION_DELTA=0` | full window from the start: H4's share |
| `FIX_MATE_GATE=0` | pruning back on in mated nodes: H2's share. Diagnostic only, it is unsound |
| `VERIFY_MARGIN=0` | no near-miss re-searches: H3's third part |
| `ROOT_M0_CUT=1` | the root part of H4 |
| `FIX_LMR_EG=225` | an ordinary LMR shape in light endgames in place of the depth-12 plateau: is the plateau part of the jump? |

---

## 5. The time manager

### 5.1 What we do today

VERIFIED, `uci_main.cpp:201-227` and `search_engine.cpp:4458-4467`:
- The **optimum** (soft limit) is tested only between iterations. Another iteration starts when
  `spent + 1.75 × last iteration <= optimum × scale`.
- The **hard limit** `min(5 × optimum, 0.8 × clock)` is tested every 2,048 nodes inside the search
  (`search_v2.cpp:439-444`).
- Nothing lies between the two.

On 46.Kxg6:
- The scale after depth 25 was at its floor: `0.563 × 0.778 × 0.503 × 0.95 = 0.21`, for a dominant, stable best move
  with no score drop. That makes the scaled soft limit 6.3 s.
- The start rule predicted the next iteration at 1.75 × a few milliseconds and started it at 0.68 s. It then ran to
  149.6 s.
- A predictor built on the last iteration cannot see a 200× jump.

### 5.2 The reference engines

| engine | stop or not start between iterations | inside an iteration | maximum |
|---|---|---|---|
| Stockfish 11 | stop when `elapsed > optimum × fallingEval × reduction × bestMoveInstability` (`search.cpp:535-565`). Past 0.6 × that, the next iteration repeats the same depth (`increaseDepth`, `:566-571`, `:453`) | only `elapsed > maximum − 10 ms`, in `check_time` (`search.cpp:1740`) | the same share formula with `MaxRatio 7.3`, `StealRatio 0.34` (`timeman.cpp:36-37`, `:124-128`): about 6.5× the optimum |
| Stockfish 18 / 19 | stop when `elapsed > min(totalTime, maximum)`, with `totalTime = optimum × fallingEval × reduction × instability × highBestMoveEffort` (SF18 `search.cpp:485-528`). No deeper iteration past 0.5 × `totalTime`. SF19 also stops on a mate in 3 or better (`search.cpp:602-603`) | only `elapsed > maximum` (SF19 `search.cpp:2126`) | `min(0.825 × clock − overhead, maxScale × optimum)`, `maxScale` up to 6.68 (SF18 `timeman.cpp:114-134`); SF19 6.87 and 0.81 (`timeman.cpp:118-138`) |
| Reckless | stop when `elapsed >= soft × (node share × pv stability × eval stability × score trend × best-move stability)` (`search.rs:245-266`, `time.rs:64-70`) | only the hard bound, every 2,048 nodes, and never before depth 1 completes (`time.rs:73-82`) | `0.7281 × clock + 0.75 × increment` (`time.rs:36-41`): at this clock about 171 s against a soft bound of about 19 s, nine times the soft bound |

**What they share:**
- None has a rule between the soft limit and the maximum. An iteration already far past the optimum runs to the
  maximum in all of them. Reckless's maximum is looser than ours.
- They are safe for a different reason: their iterations do not jump. Mates are stored in the table. On a fail-high
  they open the window from the returned value and search one ply shallower each time (§3 H4).

So a guard against an exploding iteration is not something to copy. It would be ours, and its right size depends on
how often our iterations explode.

### 5.3 Proposed rule: `TM_ITER_CAP`

**The idea.** The hard limit exists for a move in trouble. An iteration in which nothing has yet put the current best
move in doubt should not get more than a few times the time at which the manager would have stopped anyway.

**Knobs** (both default to byte-identical behaviour):
- `TM_ITER_CAP` (percent of the scaled soft limit). 0 = off. Suggested 300. A negative value means count only.
- `TM_ITER_FLOOR` (percent of the unscaled optimum that the cap never goes below). Suggested 100.

**The cap,** in seconds since the search started, set when an iteration starts:

```
cap = max(TM_ITER_FLOOR / 100 × optimum,
          TM_ITER_CAP  / 100 × optimum × g_tm_scale)        # never above the hard limit
```

**The cap is lifted** (back to the hard limit, as today) for the rest of the iteration as soon as that iteration shows
doubt about the move:
1. an aspiration pass of this depth completed as a fail-low;
2. root move 0 (the previous best) completed at or below the pass's alpha;
3. a later root move beat alpha (a new best is being established).

**When the cap fires,** the iteration is cut exactly as the hard limit cuts it today. The last completed iteration's
move is played, and `TM_PARTIAL` and `ASP_KEEP_FAILHIGH` apply unchanged.

**On 46.Kxg6:**

| floor | cap | time saved of the 149.6 s |
|---|---|---|
| 100 | `max(29.9, 3 × 6.3) = 29.9 s` | 119.7 s (80%) |
| 50 | 18.8 s | 130.8 s (87%) |

None of the three doubt conditions can have been true: the move was the only capture, stable for many iterations, and
the score was rising.

**Why normal moves are not touched.**
- An iteration starts only when `spent + 1.75 × last <= soft`, and `spent >= last`, so `last <= soft / 2.75`.
- To reach `3 × soft` the iteration must cost more than `2 × soft + 1.75 × last`, which is at least 7.25 times the
  previous iteration, with no fail-low and no new best move.
- Measured growth is a median of 1.93× per iteration, with the next iteration's cost at the 90th percentile equal to
  1.87× the time already spent (`TIME-MANAGER-RESEARCH-2026-10-25.md` §1).

**Pseudo-diff (NOT applied):**

```
search_engine.h, namespace Config
+   inline int TM_ITER_CAP = 0;      // percent of the scaled soft limit an iteration may reach while the best move
+                                    // is not in doubt; 0 = off; negative = count only
+   inline int TM_ITER_FLOOR = 100;  // percent of the optimum the cap never goes below

search_engine.cpp, file scope (beside g_tm_scale)
+   static double g_tm_iter_limit_s = 1e18;   // seconds since the search started; 1e18 = only the hard limit
+   long g_tm_cap_fired = 0, g_tm_cap_would = 0, g_tm_cap_would_changed = 0;
+   double g_tm_cap_saved_s = 0;
+   // Whether the search clock has run out: the hard limit, or the iteration cap while it is armed.
+   static inline bool search_clock_expired(const TimePoint &t0)
+   {
+       const double el = std::chrono::duration<double>(Clock::now() - t0).count();
+       if (el >= Config::ACTIVE->TIME_LIMIT) return true;
+       if (el < g_tm_iter_limit_s) return false;
+       if (Config::TM_ITER_CAP > 0) { ++g_tm_cap_fired; return true; }
+       /* count only: note the first crossing of this iteration, keep searching */
+       return false;
+   }

every in-node clock test (sv2::negamax search_v2.cpp:442, sv2::qsearch :1397, minimizer search_engine.cpp:6672,
and the two other v1 node-entry tests)
-   if (std::chrono::duration<double>(Clock::now() - t0).count() >= Config::ACTIVE->TIME_LIMIT)
+   if (search_clock_expired(t0))
        time_up.store(true, std::memory_order_relaxed);
    (alpha_beta's own two tests at :5263 and :5628 stay on TIME_LIMIT: they return without setting time_up, which
     is the open C9 item, and must not gain a new way to fire)

get_engine_move, the iterative-deepening loop, right after `t0 = Clock::now();` (:4502)
+   g_tm_iter_limit_s = 1e18;
+   if (Config::TM_ITER_CAP != 0 && Config::TM_SOFT_LIMIT > 0)
+       g_tm_iter_limit_s = std::max(Config::TM_ITER_FLOOR / 100.0 * Config::TM_SOFT_LIMIT,
+                                    std::abs(Config::TM_ITER_CAP) / 100.0 * Config::TM_SOFT_LIMIT * g_tm_scale);

lifting the cap (three one-line sites)
    aspiration loop, the fail-low branch (:4564)                    + g_tm_iter_limit_s = 1e18;
    alpha_beta, after move 0, when score <= root_alpha_entry (:5181) + g_tm_iter_limit_s = 1e18;
    alpha_beta, a later move beats alpha (:5582)                     + g_tm_iter_limit_s = 1e18;

after the loop and on the TMCNT line (:4983)
+   " cap_fired=" … " cap_would=" … " cap_would_changed=" … " cap_saved_s=" …
```

The clock test runs once per 2,048 nodes, so the extra comparison costs nothing measurable. Being new code in the
search file it still needs the usual bench-time check.

**Count-only first** (`TM_ITER_CAP=-300`). It changes no move, so it can ride along any timed queue. It records:
- how often the cap would have fired;
- how many of those iterations, left to run, ended with a different best move (the rule's real cost);
- the seconds between the would-fire moment and the iteration's end (the rule's gain).

**Registered predictions:**

| quantity | prediction |
|---|---|
| fire rate, 10+0.1 self-play | 1 to 5 per 1,000 moves (central 2) |
| the same order as | today's hard-stop rate, to be read from `TMCNT cut=`. The C9 count of 0.94 root-clock exits per 1,000 moves is a floor |
| would-fire iterations that end with a different best move | under 20% (the doubt conditions remove the fail-low cases) |
| time saved per fire | 2 to 4 × the optimum |
| Elo at 10+0.1 | −3 … +3 (it is a safety net) |
| long-control games that reach a mate through a simple ending, until H1 is fixed | about one fire per game |

**Order of work.** Fix the search cause first (§7). The cap is the net under it: other iteration explosions will exist
that are not this one.

---

## 6. Is 29.Rg5 the same mechanism?

**No. REPORTED.** The figures are from the PGN; the per-iteration split needs the `TMITER` log of a replay.

| | 46.Kxg6 | 29.Rg5 |
|---|---|---|
| phase | pawn ending, mate in view | middlegame (rooks, bishop, knight), +2.18 → +1.70 |
| the long iteration | never finished, cut by the hard stop at 149.6 s | **finished** depth 18 at 145.4 s; the hard limit was 5 × 39.5 = 197.7 s |
| cost against normal growth | more than 200× the 23 iterations before it | 217.1M nodes for depth 18. The neighbouring moves reached depth 16 in 17.9 to 58.4M nodes; two more plies at the measured 1.93× per ply is 67 to 218M. **Inside normal growth** |
| why the time was spent | nothing limits an iteration once it has started | the manager chose to go deeper. After a fall of 0.48 pawn the score-drop factor is `0.1127 + 0.00262 × 480 = 1.37`, a best-move change sets the stability factor to 1.31, and a low node share adds up to 2.7. The scaled soft limit can reach 3× the optimum or more (about 120 s), so the start rule allowed depths 17 and 18 |
| mate scores, table refusals | yes (H1) | none |

What the two share is only the weakness already listed in `TIME-MANAGER.md` §6.3: the soft limit is tested between
iterations.

Two separate questions for 29.Rg5:
- **Should three scale factors multiply up to 5.6 with only the hard limit above them?** Stockfish allows the same
  size of excursion (its `totalTime` is capped by a maximum of about 6.7× the optimum). It is a tuning question for the
  final pass, not a defect.
- **Was the last iteration unusually expensive?** The start rule assumes 1.75×, and a fail-low iteration costs more.
  The count-only cap answers this for free: condition 1 or 2 lifts the cap in a fail-low iteration, so 29.Rg5 would
  not be cut by it, and the log will show how such iterations are distributed.

---

## 7. Fix candidates this points to

Each goes behind a knob, default 0, and is judged by games.

1. **`FIX_TT_MATE=1`** (exists). The register already lists the mate bundle as due for a re-test, because its gen 6.6
   verdict was measured while phantom mates could be stored as real ones and C8 has since shipped. S2-33 is a concrete
   position where it should be worth more than 15×. Test it alone first, then with `FIX_MDP=1`.
   - Note for the games: the bundle raised bench nodes by about 3% in gen 6.6, so the node-increasing-fix rule applies.
   - A second effect: iterative deepening stops at the first mate score and the table is kept between our moves, so
     with mates stored the proof survives to the next move instead of being rebuilt every move (172M, 43M, 22M nodes
     in this game).
2. **A mate shortcut in the aspiration loop** (new; `ASP_MATE_FULL`). A pass that fails with a mate-sized score goes
   straight to the full window. That saves up to three depth-26 passes with their pre-searches. Related form to test
   once: one ply less on each fail-high re-search, as the references do.
3. **A mate guard on the near-miss re-search** (new; `VERIFY_MATE_GUARD`). No `depth − 2` re-search when both values
   are mates. And no depth-12 scout for the moves of a node whose best is already mated, since the full-depth search
   is certain to follow.
4. **`ROOT_M0_CUT=1`** (exists, #283). It also removes the mate-window search of root move 1 in every failing pass.
5. **`TM_ITER_CAP`** (new, §5.3), count-only first.
6. **Release-readiness:** a 64-bit node counter (§3, latent).
7. **Not proposed:** changing `FIX_MATE_GATE` or the mate guards on reverse futility and the null move. They are
   correct, and their cost disappears once the proof is stored.

---

## 8. What this analysis could not establish

- The score, the pass and the root move the depth-26 search was in when it was cut (C1 and C2 give it).
- Whether the four narrow aspiration passes are cheap or expensive in the mate iteration (C1).
- How often the mate-against-mate re-search of H3 fires (C7).
- The split of the explosion between H1 and its multipliers H2 to H6 (E1 arm 2 against E3).
- How often an iteration costs more than 20× the previous one in ordinary games (C11), and so how much S2-33 matters
  outside long-control endgames.
- The per-iteration timeline of 29.Rg5 (a replay with `FIX_COUNT_LOG=1`).
