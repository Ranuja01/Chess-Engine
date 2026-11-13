# S3 root: how the giants run without a pre-search, and what our no-pre-search arm (A4) is missing (2026-11-13)

Cross-examination for the owner's question: what does a strong engine do, in mechanism, so that a root with no pre-search
works, and which of those supports does arm A4 lack. Companion to
[`S3-ROOT-AUDIT-2026-11-13.md`](S3-ROOT-AUDIT-2026-11-13.md) ("the audit") and
[`reference/root-giants-2026-11-13.md`](../reference/root-giants-2026-11-13.md) ("the giants note"); neither is repeated.

Read-only: nothing was built or run. Line numbers are today's working tree. "INFERRED" marks a conclusion of mine that
no line states. Nothing here is measured; every size is an estimate and says so.

File names used below: `se.cpp` = `search_engine.cpp`, `se.h` = `search_engine.h`, `sv2.cpp` = `search_v2.cpp`,
`cm.h` = `cache_management.h`; `SF19` = `stockfish_19_linux/src/search.cpp`, `SF11` = `stockfish_11/stockfish-11-win/src/search.cpp`,
`RK` = `E:\chess_data\engines\reckless_src\src\search.rs`, `TV` = `E:\chess_data\engines\triumviratus_src\source\search\13_search.inc`.

A4 = `ENABLE_ROOT_PRESEARCH=0 ENABLE_ROOT_RAZOR=0 ENABLE_PRESEARCH_OFF_SORT=1 PRESEARCH_OFF_SORT_MODE=2 ENABLE_ROOT_TABLE=1`.

## 0. Summary

1. **The giants' substitute for our razor is not at the root. It is the null move (and the other node-level cuts) at
   ply 1.** A late root move is searched with a null window; its child is an ordinary cut node, and the first thing that
   node does is ask "can the opponent even pass?" at a depth cut by 7 + depth/3 plies (SF19 `:1011-1028`). A bad root
   move is dismissed for about the price of a q-search. Our ply 1 is a separate code branch with **no** null move, no
   reverse futility and no node-level cut of any kind (`se.cpp:6636-6883`), and the null-move gate forbids plies 1 and 2
   anyway (`se.h:3184-3185`). The razor has been doing that job. A4 switches the razor off and puts nothing in its
   place, so every root move pays a full-depth ply-1 node in every pass. §1.2, §4 H1.
2. **The time manager's node-share factor was fitted on the pre-search's node accounting.** The share is "best move's
   main-search nodes / all main-search nodes", pre-search excluded (`se.cpp:273-282`, `:4510-4512`). In A4 the main
   search covers every root move, so the denominator grows, the share falls, and `f_nodes` rises (INFERRED; not
   measured). That changes time per move in clock games only, which is the venue where A4 is behind while being ahead at
   fixed depth. §4 H2. Readable from logs that already exist.
3. **One defect in the off path:** the root table carries a root move's reply list across a pass that did not search the
   move, but drops the reply scores (`se.cpp:5074-5075`), so the next pass cannot put that move's refutation first
   (`se.cpp:6656`, `:9618-9646`). It bites after every aspiration fail high. §3 D1. Four smaller items in §3.
4. **SF19's own root reductions are small** (0 to 3 plies by its formula at game depths, INFERRED in §1.3). The idea that
   the giants make the tail cheap "by root LMR" is mostly wrong for SF19; the tail is cheap because of what ply 1 and
   ply 2 do. That also explains why our index-keyed root LMR lost: it removed depth where the giants remove none, and
   left the expensive part (ply 1) untouched.
5. A4's measured cost is nodes (+3 to +18%), not accuracy. On the usual rate of roughly 100 Elo per doubling at this
   time control, +7.4% nodes on quiet positions is about −10 Elo (INFERRED), which is the size of the last read
   (−12.8 ± 17.8). So the deficit needs no further explanation beyond "A4 has no cheap dismissal of the tail"; H2 and D1
   are on top of it.

## 1. What plays each of the pre-search's roles in the giants

### 1.1 The data that flows from one iteration to the next

| role of our pre-search | SF19 | what carries it |
|---|---|---|
| order of all root moves | previous best first (`:820`); the rest in **picker order**: captures, then quiets by butterfly + continuation history + a low-ply history table indexed by ply 0-4 (`movepick.cpp:248-249`, `history.h:42`, filled at `search.cpp:326`, written at `:2052`) | history tables. The root list's order is not used inside the loop (giants note §2.1) |
| order at ply 1 under each root move | the ply-1 node's own TT entry gives the reply that refuted the root move last iteration (`:817-820`); then the picker, whose continuation history is keyed on the root move just played | TT move + history. `TTEntry::save` keeps the old move when a later store has none (`tt.cpp:104-106`), so a fail-low visit does not erase the refutation |
| warming of each subtree | nothing dedicated. Every node of iteration D−1 left an entry: key, move, value, static eval, depth, bound, PV flag (`tt.cpp:101-121`; eval written even when nothing else is known, `search.cpp:853`). Iteration D re-walks the same tree with a TT move at nearly every node, and nodes without one are searched a ply shallower (IIR, `:1051`) | the TT |
| revival of a root move that dropped | no move ever drops: all are searched every pass, and a reduced scout that beats alpha is searched again (`:1362-1392`) | the search itself |
| the alpha the first move sets | the first move with the aspiration window; then at root depths 4-11 every later move is searched 3 plies shallower after the first alpha raise (`:1534-1536`) | — |

SF11, Reckless and Triumviratus are the same in every row (TT move at ply 1, picker order, no dedicated pass); the
differences are in §5.

### 1.2 The part that matters: what a late root move costs them

Take a root move that is not best at root depth D. In SF19:

1. The root calls its child with a null window and `cutNode = true` (`:1372`).
2. The child probes the TT (`:817`). The entry from iteration D−1 is one ply too shallow to cut (`:874`), but it supplies
   the refuting reply and, when its bound agrees, replaces the static eval as the node's estimate (`:843-845`).
3. **Null move** (`:1011-1028`): if the static eval is at least `beta − 13·depth + 365` (minus 47 when improving), the
   opponent passes and the root side is searched at `depth − R`, `R = 7 + depth/3 + max((eval − beta)/256, 0)`. At
   depth 11 that is `R ≥ 10`: a 1-ply search, in practice a q-search. If it still holds, the node returns at once; the
   verification search exists only at depth 16 and above (`:1027`).
4. Otherwise, child futility for nodes that are not on the PV, up to depth 18 (`:996-1008`), then ProbCut, then the TT
   move, which usually cuts.
5. If the TT move fails, the remaining replies are reduced hard: a cut node adds 4026/1024 (about 4 plies) to every
   late move's reduction (`:1327-1328`).

So the giants' test for "this root move is not worth a search" is **dynamic**: a shallow search of the position as it
is now, against the alpha of this pass. Ours is **static on stale data**: the razor compares alpha with a number from
the previous iteration or the pre-search. Both dismiss the tail cheaply. A4 has neither.

Steps 3-5 are the same in SF11 (null move at any non-PV node with `eval >= beta`, `R = (854 + 68·depth)/258 + min((eval −
beta)/192, 3)`, verification from depth 13; `SF11:838-881`), Reckless (`cut_node`, `r ≈ 4.3 + depth/4 + …`, `RK:567-616`,
RFP at `RK:549-564`) and Triumviratus (`n.cut`, `TV:996-1052`). None of them excludes ply 1.

### 1.3 How big their root reductions are (INFERRED from the formulas)

SF19, late root move, root depth 12, move number 10: `reduction()` gives `reductions[12]·reductions[10] − 577 + 982 ≈
3210` (`:713`, `:1885-1887`; delta equals rootDelta at the root). On the PV: `+929 − 3023 − 1004` (`:1161-1162`,
`:1316-1318`), `+697`, `−65·10` (`:1321-1323`). Net about +160 in 1024ths: zero plies, and `d = newDepth − 0 + PvNode`
(`:1369`) is one ply *more* than the nominal depth. At depth 20, move 30, not improving, the same sum is about 3100:
3 plies off, 2 after the PV credit. History and the eval-to-alpha term move it by about one ply either way
(`:1352-1355`). So SF19's root reduces a late move by 0-3 plies at game depths. SF11 reduces more at the root (its
formula gives about 3 at depth 12 move 10, less 2 for the PV, `SF11:1117-1150` per the giants note) but exempts moves
that were ever best.

Share of root work spent on late moves: not computable from the text. What the formulas do say is that the saving is
at plies 1-2, not in the root's own reduction.

### 1.4 Aspiration and the tail

- Their windows are narrow (SF19 delta `5 + meanSq/10193`, giants note §2.1) and a failed pass is cheap to repeat:
  every ply-1 node was stored at this depth in the failed pass, and after a fail high the next pass is one ply
  shallower (`:392-393`), so those entries cut at node entry (`:874`).
- Ours: delta 500, each pass re-enters `alpha_beta` (`se.cpp:4410`); the root never probes a child's entry before
  calling ply 1 (audit §3.7), so each root move's ply-1 loop runs again and is rescued one ply lower by the per-reply
  probe (`sv2.cpp:911-918`). Close to equivalent in nodes when the first reply's entry still decides; not equivalent
  after D1 (§3).

## 2. Our side with the pre-search off, support by support

| support (§1) | in A4 | where |
|---|---|---|
| null move at ply 1 | **missing**. Ply 1 is its own branch of `minimizer` (`se.cpp:6636-6883`): draw test, sort the carried replies, loop. The null move lives in the other branch (`se.cpp:6947`) and in `sv2::negamax` (`sv2.cpp:437`), both gated `cur_depth >= NULLMOVE_CURDEPTH_MINI/MAXI` = 3 / 4 (`se.h:3184-3185`). The header records that 3 / 4 was tuned at fixed depth with the pre-search and razor on (`se.h:3178-3181`) | H1 |
| reverse futility / futility at ply 1 | **missing** for the node (same branch). Child futility is also switched off for ply-1 replies: `(RootSide \|\| cur_depth > 1)` (`sv2.cpp:1037`) | H1 |
| TT cut at ply-1 node entry | missing at the node; present one ply lower per reply (`sv2.cpp:911-918`) | minor |
| refutation first at ply 1 | **present, by a different carrier**: the root record's reply list sorted by last search's reply scores (`se.cpp:6656`). As good as a TT move while the scores exist; see D1 for when they do not | D1 |
| picker order for the other replies | **weaker**. The reply list is generated once, in the first call's fill loop (`se.cpp:8337`), and never again: with the table `reuse` equals the list length from the second call on (`se.cpp:8288-8290`, `:5054-5055`). Replies searched are re-sorted by score; replies never searched keep the order move generation gave them at depth 3. Ply-1 LMR is keyed on that index (`sv2.cpp:1060`). With the pre-search on, every tail move gets a fresh list each iteration (`se.cpp:8808-8809`) | H4 |
| a best move at every interior node | **weaker**. `ENABLE_NODE_TT` and `ENABLE_TT_MOVE` are off (`se.h:3074`, `:1627`), so `store_node_tt` returns at once (`se.cpp:5492-5493`). The carrier is the move-generation cache: the whole ordered list per position, 2^20 slots, direct-mapped, promoted on a cutoff by a move other than the first (`sv2.cpp:791-793`, `cm.h:1284-1348`). The TT is 2^22 (CLAUDE.md, gen 6). A position whose list was overwritten has a bound but no order. September's measurement that a TT move adds nothing with the pre-search off (memory `presearch-economics…`) pre-dates search v2 | not ranked |
| IIR | reads **off**: header default false (`se.h:2914`), no assignment in the shipped preset (`se.cpp:2702-2755`), only the start-up read (`se.cpp:3103`). INFERRED from those three sites; the brief assumed it on | open question |
| static eval kept in the TT | missing (`cm.h:1218-1226` stores score, depth, flag, window, move); a separate eval cache exists | not root-specific |
| history tables | present, saturating with malus (`sv2.cpp:157-210`, `se.h:1645-1646`); the pre-search's own updates use the older unbounded form (`se.cpp:9052-9066`), so A4 removes a stream of large raw bonuses from ply 1. No low-ply table | not ranked |
| depth cut after an alpha raise | knob, off: `ENABLE_ROOT_ALPHA_DECAY` (`se.cpp:5394-5402`), defaults 2 plies below depth 14 (`se.h:1549-1550`); SF19 is 3 plies at depths 4-11 | §4b |
| root reduction | knob, off: `ENABLE_ROOT_LMR`, index schedule (`se.cpp:5264-5274`) | §4b |
| narrow aspiration, one ply less after a fail high | missing (audit §4 rows 9-11); `ASPIRATION_RECENTER` exists (`se.cpp:4442-4458`) | not A4-specific |
| first iterations | A4's first call has nothing: order = move generation at ply 0 (`se.cpp:8271`), every entry synthetic with score 0 (`se.cpp:8336`), no sort (`reuse > 1` false, `se.cpp:8349`), depth 3, full window. Cheap in nodes; the order entering depth 4 is depth-3 scout bounds | minor |
| node share for the clock | **changed meaning**, see H2 | H2 |

## 3. The off path, line by line

Checked: the fill loop, synthetic entries, the tiered sort, the table under aspiration re-searches, reply lists,
time-outs, and differences from the on path.

**D1. Reply scores are dropped for a root move a pass did not reach (defect).** When the table is sized at the start of
the root loop, each slot inherits `second_moves` but not `second_scores` (`se.cpp:5074-5075`; the slot is a fresh
`RootScore{}` from `:5055`). `root_table_store` refills both only for a move that is searched (`se.cpp:930-933`). A pass
that ends early (root beta cutoff, `se.cpp:5418-5430`, which is every aspiration fail high) leaves all later moves with
a reply list and no scores. In the next pass `ascending_sort` sorts a prefix of length `values.size()` = 0
(`se.cpp:9620-9643`), so the replies are tried in the order they were *searched* last time, not best-first. If the
refutation was found at reply j > 0 last time, it is tried j-th again, behind replies known to fail. The table-off
path loses the same data another way (the move falls out of `scores` and is refilled from move generation,
`se.cpp:8311-8341`). With the pre-search on, the tail gets fresh lists and scores every pass, so this loss does not
exist there. Fix: one line, inherit `second_scores` beside `second_moves`. Result changes, so it needs a knob.
Size: small per event (only moves whose refutation changed last pass), on every fail-high re-search.

**D2. A root move's re-search starts from the old reply order.** The scout's `entry` is discarded and the full-window
re-search is given the same `current_search_data.scores[i]` lists (`se.cpp:5296-5312`). The scout has just measured
every reply; its order is thrown away, and the PV search of a new best move opens with the reply that has just failed
to refute it. Same in the on path, and the giants have the same gap (their ply-1 store after a fail low carries no
move). Listed because it is free to fix (pass the scout's `entry` lists into the re-search) and is paid on exactly the
moves that change the result.

**D3. The revival tier ranks on bounds taken against different alphas.** Mode 2 orders unverified moves by
`last_real_depth`, then `last_real` (`se.cpp:8422-8424`). `last_real` is written on every store (`se.cpp:940-941`) and is
the fail-soft return of a null-window scout. Within one pass, moves searched after an alpha raise are measured against
a higher alpha and can return higher bounds than moves searched before it, so they sort above them next iteration for
no reason. The code's own comment says the order becomes a function of the widening history (`se.cpp:8370-8379`). In
A4 every move is searched every pass, so `last_real_depth` ties and the whole tail is re-sorted by these bounds each
iteration. The giants never rank the tail on a number: a stable sort leaves it in "most recently best first" order.
Modes 4 and 5 exist for this (`se.cpp:8380-8389`); mode 5 is the closest to Stockfish's behaviour and is untested on
today's engine.

**D4. `best_move` drifts on a fail-low pass** (`se.cpp:5375-5381`, `ENABLE_ROOT_BEST_REQUIRES_ALPHA` off): known (audit
§7.2). In A4 it is harmless at defaults: the pass is always searched again and the sort keys do not read `best_move`.
It would matter with `ROOT_LMR_EXEMPT_BEST`.

**D5. `ROOT_RAZOR_REDUCE` cannot be reached with the table on.** The branch order is hybrid, then table skip, then
`ROOT_RAZOR_REDUCE` (`se.cpp:5152`, `:5171`, `:5179`). With `ENABLE_ROOT_TABLE=1` the razor's "reduce" form is
`ROOT_RAZOR_TO_LMR=1` with `ROOT_RAZOR_SKIP_MARGIN=0` (`se.h:1451-1454`). Relevant to any "A4 + razor as a reduction"
arm (queue #229 should be checked against this).

Checked and clean:
- The fill loop runs once per search in A4 (first call only), so September's ~12% speed penalty does not apply. Per
  pass the off path copies every `RootScore` twice (`se.cpp:8303`, `:5075`) and `minimizer` takes both lists by value
  (`se.cpp:6454`): about 3N small vector copies a pass, negligible.
- `synthetic_from` is 0 on the first call and N afterwards (`se.cpp:8483`); nothing in A4 reads it wrongly.
- Time-outs: every return either precedes the table sizing at the very end of the search or leaves a full-length table
  (`se.cpp:4901-4908`, `:5032-5037`, `:5327-5336`). A cut pass is not adopted (`se.cpp:4467-4493`).
- Move 0 is always stored as verified (`se.cpp:5084`), so after any pass the next one opens with the previous best or
  the fail-high move (`se.cpp:8401-8411`). This matches Stockfish.
- Root child TT flags are computed against the right window for scout, re-search and fail high (`se.cpp:5323`, `:598`).

Only in the on path: the `is_draw` flag at pre-search ply 1 (`se.cpp:8736`; the main ply 1 sets it too, `se.cpp:6650`),
unbounded history / continuation-history bonuses at ply-1 cutoffs (`se.cpp:9055-9061`), and a move-cache promotion at
ply 1 (`se.cpp:9049-9050`). The main ply 1 never promotes (`se.cpp:6745-6855`), so in A4 the ply-1 positions' cached
lists are never updated; harmless while the root record carries the order.

## 4. Ranked hypotheses for A4's deficit in games

**H1. No cheap dismissal of bad root moves (missing SUPPLIER). Largest.**
- Mechanism: §1.2. The giants refute a late root move with a null move at ply 1; we skipped it with the razor; A4 does
  neither, so each of the N−1 scouts runs a full ply-1 loop whose first reply opens an all-node at ply 2 with no null
  move there either.
- Evidence: `se.cpp:6636-6883` (no node-level cut at ply 1), `se.h:3184-3185`; SF19 `:1011-1028`, SF11 `:838-881`,
  `RK:567-616`, `TV:1014`. A4's cost is nodes, not accuracy (audit §10). The razor's line is 100 millipawns from depth 8
  (audit §3.5): it was acting as a very aggressive reverse-futility test at ply 1.
- Cheapest test: no existing knob reaches it (`NULLMOVE_CURDEPTH_MINI=1` does nothing, the ply-1 branch has no
  null-move code). A small new knob, default off, e.g. `ROOT_CHILD_NULL`: in the ply-1 branch, for a null-window call
  only, not in check, null move allowed by `isUnsafeForNullMovePruning`, static eval at or below alpha in the root
  frame; search the null move with the interior's reduction at `(alpha, alpha+1)`; if it comes back at or below alpha,
  return it with `out_entry.second_moves` set to the carried list and no scores (after D1's fix the old scores
  survive). Counters first: fires, and nodes under root moves that failed low as a share of main nodes (the bound on
  what H1 can save; `root_nodes_before` at `se.cpp:5248` already has the number).
- A wider variant: send scouts of moves 1.. through `sv2::negamax` as an ordinary interior node with the gate at 1, and
  use the root's ply 1 only for the re-search of a move that beat alpha (option P11 of the audit plus the null move).
- Expected size: unknown until the counter is read. If fail-low scouts are 40-60% of A4's nodes (my guess, not
  measured) and the null move clears most of them, A4 moves from +7% to well below today's nodes.
- Risk on record: tactical losses at the root (zugzwang, quiet threats); verification from a depth as SF does.

**H2. The clock's node-share factor is mis-scaled in A4 (TUNING mismatch). Timed games only.**
- Mechanism: `share = g_root_move_nodes[best] / g_root_main_nodes` (`se.cpp:4510-4511`); both count main-search nodes
  only (`se.cpp:273-282`, `:4976`, `:5325`). At defaults the main search touches about a third of the root moves (audit
  §3.4) and the pre-search's 24-40% is outside the ratio. In A4 every move's full-depth scout is inside it. The share
  falls, `f_nodes = max(0.563, 0.4499 + 2.2669·(1 − share))` rises (`se.cpp:4512`), and the soft limit is multiplied by
  it (`se.cpp:4515-4521`, `TM_SCALE` 7, `se.h:3521`). The constants and `TM_NORM` 95 were fitted on the default root.
- Evidence: INFERRED from the code; the direction is forced, the size is not known. It is the one mechanism here that
  exists at 10+0.1 and not at fixed depth, which is the shape of the result (ahead at d10, behind on the clock).
- Cheapest tests, none needing new code: (a) read time per move and depth per move by arm from queue #228's PGNs;
  (b) `FIX_COUNT_LOG=1` on both arms over the game set and compare `share=` and `fn=` in the `TMITER` lines
  (`se.cpp:4524-4528`); (c) games at fixed nodes (the 09-03 `node_ab` form): if A4 is level there and behind on the
  clock, it is the time manager; (d) A4 against defaults with `TM_SCALE=6` on both sides.
- Expected size: a few Elo to about 15; cannot be larger than what time allocation is worth.

**H3. Reply scores lost across fail-high passes (DEFECT, D1).** Test: the one-line fix behind a knob; read nodes in
failed aspiration passes (`[root_r1]`) and total nodes on the quiet corpus. Expected size: 1-3% of nodes (guess).

**H4. Ply-1 reply lists frozen at the first iteration (missing SUPPLIER).** Mechanism in §2. With the pre-search on,
about two thirds of root moves get a newly generated, newly searched reply list each iteration; in A4 none does after
depth 3, and ply-1 LMR is keyed on the frozen index. Test: a knob that rebuilds each root move's list as "carried
refutation first, then `buildMoveListFromReordered`'s current order" once per iteration (N make / unmake pairs).
Counter first: the ply-1 first-reply cutoff rate in A4 against defaults (the audit's missing counter (5)). Size: small.

**H5. Tail order by bounds (TUNING, D3).** Test: existing knob, `PRESEARCH_OFF_SORT_MODE=5` and `=4` in A4, fixed depth
on the quiet corpus and STS. Size: small without root reductions (order then only moves the moment alpha rises); it
becomes important the day a reduction is keyed on root order.

**H6. Margins fitted around the pre-search's warming (TUNING).** RFP 600, the LMR product schedule, history-LMR and
null-move depths were all tuned with a pre-search filling the TT and histories one ply ahead. Not testable cheaply;
it is the reason to re-sweep one or two interior margins on A4 if H1 brings it level.

### 4b. The owner's hypothesis: a worse base that opens the door to more reduction

A4 searches every root move at full depth with a complete record, and is more accurate per depth (+9 WAC, +49 STS).
That surplus is unspent.

1. **What the giants run at plies 0-1 that needs every root move searched.** All of it does, and none of it needs a
   root *score*: the reductions read move number, history and PV flags, and the reduced scout itself is the safety net.
   Root LMR with re-search on a beat (SF19 `:1362-1392`); the 3-ply cut after an alpha raise (`:1534-1536`); one ply less
   after a fail high (`:392-393`); null move, futility and cut-node reductions at ply 1 (§1.2). None reads the complete
   root record; the record serves ordering and the clock.
2. **What we have behind knobs.** `ENABLE_ROOT_ALPHA_DECAY` with `ROOT_ALPHA_DECAY_PLIES` / `_MAX_DEPTH` (never
   measured, audit §3.6). `ENABLE_ROOT_LMR` + `ROOT_LMR_EXEMPT_BEST` (index schedule; lost in September without the
   table and with `DIV=8`). The razor as a reduction with the table: `ENABLE_ROOT_RAZOR=1 ROOT_RAZOR_TO_LMR=1
   ROOT_RAZOR_SKIP_MARGIN=0` on top of A4. This one is new in kind: in A4 every move has a fresh main-search bound from
   the previous iteration, so the table's evidence test (`se.cpp:5128`) passes for all of them and the owner's razor
   condition gets its data from the main search instead of the pre-search; a reduced move that comes back above alpha
   is searched again (`se.cpp:5296-5302`). A move reduced by 2 or more is "stale" next iteration and is searched in
   full (`ROOT_RAZOR_MAX_DEPTH_DEFICIT` 2), which refreshes it. **What we lack:** the null move and futility at ply 1
   (H1), the interior LMR formula at the root (audit P9), one ply less after a fail high.
3. **How to trade the surplus for depth**, cheapest first: the alpha-raise cut at SF19's numbers; the razor as a
   reduction (above); then the ply-1 null move (H1), which is where the giants spend the same surplus. Index root LMR
   last: §1.3 says the giants take almost nothing there.

## 5. Cross-reference

| support | SF19 | SF11 | Reckless | Triumviratus |
|---|---|---|---|---|
| null move at ply 1 under a late root move | yes, cut nodes (`:1011`) | yes, any non-PV node (`:838-847`) | yes, cut nodes (`RK:567`) | yes (`TV:1014`) |
| verification only at high depth | ≥ 16 (`:1027`) | ≥ 13 (`:869`) | ≥ 16 (`RK:615`) | own depth (`TV:1045`) |
| reverse futility at ply 1 | not on the PV, depth < 19 (`:996`) | non-PV, shallow only | not on the PV, quadratic margin, any depth (`RK:549-564`) | not read |
| refutation at ply 1 from | TT move | TT move | TT move | TT move (as SF19) |
| root tail order | picker + low-ply history | picker | picker, re-scored before every pick (giants note §2.3) | as SF19 |
| root LMR size | 0-3 plies (§1.3) | larger, former best moves exempt | PV credit of 2 plies (giants note §2.3) | as SF19 |
| cut after an alpha raise | 3 plies, depths 4-11 | none | none seen | not read |
| node share in the clock | all nodes are main-search nodes | none | `root_moves[0].nodes / nodes` | as SF19 |

They agree on every support of H1 and none contradicts SF19. Reckless is the strongest confirmation of the direction:
its reverse futility has no depth limit, so its ply 1 dismisses a bad root move on the static eval alone more often
than Stockfish's does. Reckless and Triumviratus are not independent of Stockfish (giants note §0). Ethereal was not
read.

## 6. Tests, ranked

| # | test | new code | reads |
|---|---|---|---|
| 1 | Time per move and depth per move by arm from queue #228's PGNs; `TMITER share / fn` on both arms (`FIX_COUNT_LOG=1`) | none | H2 |
| 2 | Counter: nodes under root moves that failed low, as a share of main nodes, A4 and defaults | counter only | the ceiling of H1 |
| 3 | A4 + `ENABLE_ROOT_ALPHA_DECAY=1 ROOT_ALPHA_DECAY_PLIES=3 ROOT_ALPHA_DECAY_MAX_DEPTH=12` | none | §4b |
| 4 | A4 + `ENABLE_ROOT_RAZOR=1 ROOT_RAZOR_TO_LMR=1 ROOT_RAZOR_SKIP_MARGIN=0` (the razor as a reduction, fed by the main search) | none | §4b; check #229 ran this form (D5) |
| 5 | A4 with `PRESEARCH_OFF_SORT_MODE=5`, and `=4` | none | H5 |
| 6 | Inherit `second_scores` in the table (D1), behind a knob | one line | H3 |
| 7 | `ROOT_CHILD_NULL` (null move at ply 1 for scouts), with a fire counter | small | H1 |
| 8 | A4 against defaults at fixed nodes, two budgets | none | separates decisions from speed and the clock |
| 9 | Pass the scout's reply order into the root re-search (D2) | small | both arms |
| 10 | Rebuild ply-1 lists each iteration (H4), after the ply-1 cutoff counter | small | H4 |

Tests 1-6 fit one fixed-depth queue plus a log read. Test 7 is the one that answers the owner's question; it goes to
games only with 2 in hand.

## 7. Not verified

- Nothing was run. H2's direction is from the code; whether the share really falls, and by how much, is test 1.
- SF19's root reduction sizes (§1.3) are my arithmetic on its formulas, with `improving` at the root assumed either way.
- Whether SF19's futility and null move fire at ply 1 as often as the text allows.
- IIR reading off at defaults: three sites read, the UCI front end's own settings not checked.
- Which interior (`SEARCH_ARM` 0 or 1) queue #228's binaries use; the ply-1 branch is the same in both.
- Reckless and Triumviratus were read only at the lines cited. SF11's reduction code and Ethereal were not read.
- The share of A4's nodes spent on fail-low scouts is a guess until test 2.

## 8. Open questions for the owner

1. Is a null move at ply 1 acceptable as an option beside the razor (it is the giants' form of the same decision), and
   should it use the interior's reduction or its own?
2. If test 1 shows A4 spends its clock differently, re-fit `TM_NORM` / the node factor for A4 before the next games,
   or count the pre-search's nodes into the share at defaults so the two arms mean the same thing?
3. D1 and D2 change results. Knobs for each, or fix D1 inside the A4 arm only (it does not exist with the pre-search on)?
4. Is IIR meant to be off in the shipped preset? The memory index lists it as a confirmed node saver.
5. Should "the razor as a reduction fed by the main search" (test 4) be treated as the owner's cascading-tables idea in
   its complete form, and get games if the screen is clean?
