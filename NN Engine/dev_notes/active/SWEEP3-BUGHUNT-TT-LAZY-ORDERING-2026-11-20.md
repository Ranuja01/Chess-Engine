# Sweep 3: transposition table and caches, lazy-eval internals, move scorer and history (2026-11-20)

A read-only audit of three areas the 11-18 register (`SEARCH-DEFECT-REGISTER-2026-11-18.md` §8) lists as not yet audited:

1. the transposition table (TT, the cache of searched positions) and the three other caches in `cache_management.h`: the
   qsearch cache (q-cache), the eval cache and the move-list cache;
2. the lazy-eval request (`lazy_request`, the eval's early exit) and every caller's threshold;
3. the move scorer in `move_gen.h` and the history tables (sizes, caps, update rule, decay).

Nothing was run and no source file was changed. Every statement about our code was read in the code that executes at
the shipped settings (the UCI binary, `V2_PRESET=shipped`, gen 7.7, `SEARCH_ARM 1`). Each finding is marked:

- **VERIFIED**: I read the lines that execute and they do what the finding says.
- **REPORTED**: arithmetic or inference on top of verified code; the size is not measured.

Classes, as in the register: **D** a search decision changes, **V** a wrong value that is read later, **L** latent (the
knob is off today), **E** efficiency (same decisions, more work, or ordering only).

New ids start at S3-20 (S3-1 … S3-10 are in `SWEEP3-BUGHUNT-TM-PLY1-2026-11-20.md`).

**Not re-reported here** (already on record; I read each and agree unless a line below says otherwise): S2-9, S2-16,
S2-26, S2-27, S2-28, S2-29; the 10-22 audits' TT1 to TT5 and G1 to G11; MOVEGEN C1 to C7; "mates are never stored" as
the cause of S2-33 (`SWEEP3-S2-33-EXPLODING-ITERATION-2026-11-20.md` H1, `SWEEP3-TRIAGE-2026-11-20.md` §4); the 32-bit node
counter; "a negative history value never decays back to 0" and "the continuation table decays every 16 intervals"
(`SWEEP3-S2-5-S2-2-2026-11-20.md` §1.1, §4.3).

---

## 0. Bottom line

- **The tables are sound.** Frames, keys, flags, depth labels, the mate and draw refusals and the time-up paths are all
  consistent. I found no wrong-value defect in the TT, the q-cache, the eval cache or the move cache beyond the ones the
  register already holds.
- **The lazy-eval request is sound.** All three sites ask with the right bound and sign, every lazy return is consumed
  by the cut it was asked for, and the estimate reaches a table only on the two paths S2-9 already names (plus one
  small third path, §3).
- **What is new is ordering and sizing**, all in code that runs today:
  - under-promotions are ordered near the front and are never reduced or pruned (S3-20);
  - the move cache's hit path makes two assumptions about the cached list that stop holding once a node has had two
    different cutoff moves (S3-21, S3-22);
  - the three 2^20 caches do not grow with `Hash`, so long-control searches turn them over many times (S3-25);
  - two small ones: a q-cache hit is written back with a weaker flag (S3-23), and the capture tier mixes two scales
    (S3-24).
- Two latent items for knobs that are off (S3-26, S3-27).

---

## 1. New findings, ranked by likely impact

| # | defect, one line | class | status |
|---|---|---|---|
| **S3-20** | Rook, bishop and knight promotions are ordered above most quiet moves and are never reduced or pruned | E, D in endgames | VERIFIED (code); size unknown |
| **S3-25** | The eval, q and move caches stay at 2^20 entries whatever `Hash` is; at long controls each is overwritten many times inside one search | E | VERIFIED (sizes); turnover REPORTED |
| **S3-21** | The "best two quiets" re-rank on a move-cache hit swaps instead of inserting, and it scores losing captures with the quiet scorer | D (ordering, LMP index) | VERIFIED |
| **S3-22** | After two different cutoff moves at a node, killers are inserted ahead of that node's remaining good captures | E (ordering) | VERIFIED |
| S3-24 | In the good-capture tier an equal trade scores 0 by one rule while a dearer attacker on the same victim scores the full exchange value by another | E (ordering) | VERIFIED |
| S3-23 | A q-cache hit at the qsearch root is written back with a flag recomputed against the probing window, so an EXACT entry becomes a bound | E | VERIFIED |
| S3-26 | `TT_WAYS` > 1 (buckets) has no age, so its deep slot fills with old searches' entries and never frees | L | VERIFIED (code) |
| S3-27 | Lazy move generation's "same order as today" claim fails once a quiet scores 200,000 or more, which `moveFrequency` reaches at depth | L | VERIFIED (code); the depth REPORTED |

### S3-20: under-promotions are ordered high and never reduced

**Defect.** At any node where a pawn can promote, the generator emits four moves (queen, rook, bishop, knight). The
three under-promotions are scored above killers or level with them, and the search treats every promotion as a move
that may not be reduced or pruned.

**Mechanism (VERIFIED).**
- Four promotions per push or capture: `move_gen.h:929` (comment and code of `processMaskPairsFast`), `:609-617`,
  `:663-671`. The promotion code is 1 = none, 2 knight, 3 bishop, 4 rook, 5 queen.
- The scorer adds `(promotion − 1) × 5000`: `move_gen.h:1346` (quiet branch), `:1300` (capture branch), `:1532`
  (`score_quiet`). So a quiet promotion scores queen 20,000, rook 15,000, bishop 10,000, knight 5,000, plus the history
  terms, which are keyed by from and to squares and are therefore shared by all four.
- For scale: killer 10,000 / 9,000 (`cache_management.h:1357-1367`), counter move 8,000 (`move_gen.h:1344`), a typical
  history value far below either. The rook promotion sorts above both killers; the bishop promotion ties the first killer.
- `lmr_eligible` requires `move.promotion == 1` (`search_v2.cpp:1091`). A promotion is therefore never `do_lmr`
  (`:1127`), and late-move pruning (`:1140`), history pruning (`:1153`), the quiet exchange prune (`:1162`), futility
  (`:1190`) and the reduction (`:1215`) all require `do_lmr`. Each under-promotion gets a full-depth scout (`:1281-1305`).
- qsearch searches all four as well (that half is S2-32).

**Why it matters.** A rook or bishop promotion is better than the queen only to avoid stalemate. At a node that does not
cut on the queen promotion (a fail-low node, or a PV node), the search pays three more full-depth subtrees, and the TT
cannot merge them with the queen's subtree because the positions differ. In pawn races both sides reach such nodes.
The second half of the same finding: a quiet **queen** promotion sits at 20,000 in the quiet scale, below every
threat-escape move (32,500 to 100,000, S3 scale table in §4) and below a PV move's `moveFrequency` (S2-28).

**Reference (REPORTED from `dev_notes/reference/movegen-giants-2026-11-11.md`, one line VERIFIED).** Stockfish 18 puts
the queen promotion with the captures and the three under-promotions with the quiet moves, where they are scored,
reduced and pruned like any quiet (`stockfish_1/src(SF18)/movegen.cpp:113-120`, read).

**Count (fixed depth, counter build).** In `sv2::score_move`, behind the `FIXCNT_*` macro family
(`search_engine.cpp:185-221`, compiled only with `-DFIXCNT_BUILD`): moves with promotion 2, 3 or 4 that reach a child
search; the nodes spent under them (node counter before and after, as `g_verify_nodes` does at `search_v2.cpp:1347`);
how many became the node's best. Read it on eg60 and on the S2-33 position. Material if the node share is 1% or more
on eg60.

**Fix sketch (knob `UNDERPROMO_LATE`, default 0 = byte-identical).** 1: the scorer gives rook and bishop promotions no
bonus and a fixed penalty that places them last among quiets, the knight promotion a small bonus only; `lmr_eligible`
accepts promotions 2 to 4, so they are reduced and pruned as late quiets. 2: also drop rook and bishop promotions from
qsearch (S2-32's fix). A separate value for the quiet queen promotion (the capture tier, as the references do) is a
second knob.

**Expected effect (estimate).** Middlegame bench: under 0.3% of nodes. Endgame sets: −1 to −5% of nodes. Games: −3 to
+8 Elo. Risk: a stalemate-avoiding under-promotion found one or two plies later.

### S3-25: three caches are fixed at 2^20 entries; long searches overwrite them many times

**Defect.** From gen 6 the shipped preset sizes the TT at 2^22 entries and the eval cache, q-cache and move cache at
2^20 each. The UCI `Hash` option changes the TT only. All four are direct-mapped (one slot per index) and a different
position always takes the slot. The sizes fit 10+0.1; they do not fit the owner's long games.

**Mechanism (VERIFIED).**
- Sizes: `search_engine.cpp:3411-3417` sets `TT_LOG2 22`, `EVAL_LOG2 20`, `QCACHE_LOG2 20`, `MGCACHE_LOG2 20` as
  environment defaults; `initTableSizes` (`cpp_bitboard.cpp:165-207`) reads them once.
- `Hash` → `TT_LOG2` only: `uci_main.cpp:316-326`; default 128 MB (`:522`).
- Entry sizes, by the struct layouts: `TTEntry` 32 bytes (`cache_management.h:113-127`; 128 MB / 2^22 confirms it),
  `QTTEntry` 16 (`:177-182`), `EvalEntry` 16 (`:77-83`), `MoveEntry` 48 plus one heap vector per slot (`:85-93`).
  **Record correction:** the 10-22 audit (TT5) says about 40 bytes for the TT entry; it is 32, two per cache line.
- Always-replace on a different key: TT `cache_management.h:1218` (the `updatedKey != entry.key` half of the test);
  q-cache `:1082-1086`; eval cache `:745-748`, `:763-767`; move cache `:988-999`, `:1297-1303`.
- Stores into the TT happen for every child searched at remaining depth 2 or more (`search_v2.cpp:1083`, `:1271`,
  `:1296`, `:1364`); one q-cache store per qsearch root (`:229-234`); one eval-cache store per eval-cache miss
  (`search_engine.cpp:11409-11423`); one move-cache store per generated list.

**Size (REPORTED: arithmetic, the write shares are assumed, not counted).** A search of N counted nodes writes on the
order of N/4 to N/3 entries to each table. At 10+0.1 (about 360,000 counted nodes per move, register §6b) that is
about 100,000 writes: 2 to 3% of the TT and about 10% of each small cache per move, so nothing is overwritten inside
a search. At a 30-second think (about 48 million counted nodes) it is on the order of 12 to 16 million writes per
table: the TT is overwritten about 3 times and each 2^20 cache 10 to 15 times **inside one search**. A deep TT entry
written early in such a search survives to the end with a probability of a few percent. The move cache loses the
node-local order (the last cutoff move first) that the 10-22 audit calls its one strength.

**Count (free, no code).** The stats line already prints eval visits and eval-cache hits, move-generation visits and
hits (`search_engine.cpp:4940-4943`) and `qstore` (`:4780`). Misses divided by 2^20 is the turnover. Read them at
`go depth 12`, `18` and `22` on a few game positions. A TT store counter is counter C3 of the S2-33 plan.

**Fix sketch.** No engine change is needed to test it: `Knobs TT_LOG2=25,EVAL_LOG2=23,QCACHE_LOG2=23,MGCACHE_LOG2=22`
against the default, time to depth 20 on a few positions, one process per setting, when the box is free. If it pays:
(a) derive the three cache sizes from `Hash` (for example Hash/8 each for eval and q, Hash/4 for the move cache);
(b) give the long-control launcher a larger `Hash`. The eval values, q values and move lists are position-pure, so a
larger table changes hit rates and node counts but no stored value.

**Expected effect (estimate).** None at 10+0.1 (the reason gen 6 could shrink the tables without loss). At 30 s per
move: a few percent to 15% less time to depth, larger in endgames, where transpositions carry the search. RAM is the
limit: the sizes above are about 1.6 GB for one process.

### S3-21: the "best two quiets" re-rank swaps moves out of place and scores captures as quiets

**Background.** On a move-cache hit the node replays its cached list. If the last cutoff at this node came from index 3
or later (`last_cutoff_index > LAZY_RESORT_MIN_CUTOFF_IDX`, 2), the two quiets with the highest current score are
moved to the front of the "quiet tail" (`PROMOTE_TOP_K` 2). `ENABLE_LAZY_RESORT` is on (header `search_engine.h:1594`,
start-up read `search_engine.cpp:2260` keeps the header value).

**Defect (VERIFIED, `search_engine.cpp:10734-10803`).**
1. **A swap, not an insert.** The selection uses `std::iter_swap` (`:10796`). The move that was at the front of the tail
   is exchanged with the chosen one and lands wherever that one was, possibly 20 places down. This is the shape
   `FIX_KILLER_SPLICE` fixed for killers in gen 6.5 ("swapped in instead of inserted"), not ported to this loop.
   - The tail starts at `quiet_start = firstNonCapture + indexIncrement` (`:10739`), and `indexIncrement` counts the
     killers placed but not the counter move (`:10583`, `:10612`). So the counter move, just inserted there, is the
     first candidate to be swapped away. It usually comes back in the second pass (its score includes the 8,000
     counter bonus), and then the next move in the node's own cached order is the one thrown down the list.
   - The header comment of this block says why that matters: the cached order carries the node's own earlier cutoff
     moves and "ranks quiets better here than a global history table does" (`:10729-10733`). The swap sends exactly
     those moves to indices where late-move pruning (by index: 2, 5, 10, 17, 26 at remaining depth 1 to 5,
     `search_engine.h:1383-1385`) removes them.
2. **Captures in the tail are scored by the quiet scorer.** The loop scores every move from `quiet_start` to the end
   (`:10760-10771`) with `score_quiet`, whose header says "captures are never passed here" (`move_gen.h:1518`). They
   are: a capture with a negative exchange value is given a low score on purpose and sorts among the quiets
   (`move_gen.h:1337`), and S3-22 leaves good captures in the tail too. `score_quiet` reads the quiet history tables
   for a capture (about 0). When every quiet ahead of it has a negative history value, the capture counts as "best"
   and is moved to the front of the quiets; in the other direction a good capture sitting at the front of the tail can
   be swapped far down.

**Count.** In the top-K branch, behind `FIXCNT_*`: fires per move-cache hit; for each swap, the distance the displaced
move travels and whether its new index is past the late-move-pruning limit at this node's remaining depth; displaced
moves that are the counter move; chosen or displaced moves that are captures.

**Fix sketch (knob `TOPK_INSERT`, default 0).** 1: `std::rotate` instead of `iter_swap` (the chosen move is inserted,
the rest shift down one), and only non-captures are scored and moved. Result-changing; gates on bench and game12.

**Expected effect (estimate).** Bench nodes −0.5 to +0.5%; games −3 to +6. Small, but it removes a pruning of the
node's own remembered moves that nobody chose.

### S3-22: with two remembered cutoff moves, killers go ahead of the good captures

**Defect (VERIFIED).** `updateMoveCacheForBetaCutoff` moves the cutoff move to the front of the cached list on every
cutoff that is not at index 0 (`cache_management.h:1347`, `promoteMoveToFront`, `cpp_bitboard.h:809-816`: erase and
insert at the front). The previous front move shifts to index 1. After two different cutoff moves the list reads
`[newest cutoff] [older cutoff] [the rest, captures first]`.

`insert_killers_after_captures` (`search_engine.cpp:10584-10613`) assumes one leading move followed by the captures:
it starts at index 1 and walks forward while the move is a capture (`:10587-10591`). If index 1 holds the older cutoff
move and that move is a quiet, the walk stops at once and both killers and the counter move are inserted at indices 1
to 3, in front of every capture that follows.

Example: generated `[c1 c2 c3 q1 q2 …]`; a cutoff by q2 gives `[q2 c1 c2 c3 q1 …]`; a later cutoff by c2 gives
`[c2 q2 c1 c3 q1 …]`; the next visit searches `[c2, killer 1, killer 2, counter, q2, c1, c3, …]`.

Captures are not pruned by index, so nothing is lost; the cost is winning captures tried after up to four quiets when
the lead move fails. It also feeds S3-21 (2): the "quiet tail" then contains good captures.

**Count.** On move-cache hits, behind `FIXCNT_*`: lists where index 1 is a non-capture and a capture with a
non-negative exchange value sits behind it; of those, hits where the cutoff move turned out to be one of those
captures.

**Fix sketch (knob `SPLICE_SKIP_LEADS`, default 0).** 1: the walk skips leading non-captures that are not killers
until the first capture, then passes the captures (the insertion point becomes "after the last leading good capture").

**Expected effect (estimate).** Bench nodes 0 to −0.5%. Bundle with S3-21: disjoint lines of one function, both
readable from the same counter build.

### S3-24: the good-capture tier mixes two scales

**Defect (VERIFIED, `move_gen.h:1293-1339`).** A capture whose victim is worth at least its attacker takes the fast
path and scores `200000 + victim − attacker` (`:1307-1309`). Any other capture runs the exchange evaluator and scores
`200000 + exchange value` (`:1324`, `:1337`). The fast path's number is a lower bound on the exchange value, the slow
path's is the value itself, and both land in one tier:

| capture | path | score above 200,000 |
|---|---|---|
| rook takes an undefended rook (wins 5,000) | fast | 0 |
| queen takes the same undefended rook | slow | 5,000 |
| queen takes an undefended queen (wins 10,000) | fast | 0 |
| king takes an undefended queen | slow (the king's value is 12,000, `cpp_bitboard.h:153`) | 10,000 |
| pawn takes a defended pawn (even) | fast | 0 |
| queen takes an undefended pawn | slow | 1,000 |
| pawn takes knight | fast | 2,250 |

So every same-piece capture (pawn takes pawn, knight takes knight, rook takes rook, queen takes queen) scores 0, the
bottom of the tier, whether it is an even trade, a recapture that wins the piece back or the capture of a hanging
piece. Ties are broken by capture history (halved every 35,000 nodes, so small), `moveFrequency`, and then build order,
which is minor takes minor, pawn takes pawn, rook takes rook, queen takes queen (`move_gen.h:1147-1169`): the queen
recapture is the last of the ties. The same order is the qsearch order.

The 10-22 ordering audit records "ours: `victim − attacker`" against the references' "most valuable victim first"
(§3.2 there) but did not list it as a gap or note the two scales.

**Reference (VERIFIED).** Stockfish 11: `6 × victim + capture history` (`stockfish_11/.../movepick.cpp:110-111`).
Stockfish 18: `7 × victim + capture history` (`stockfish_1/src(SF18)/movepick.cpp:155-156`). No attacker term.

**Count.** On move-generation misses, behind `SV2_MOVE_PROBE` (a probe build already exists, `move_gen.h:1311-1322`): at
nodes whose previous move was a capture and that have a capture back on the same square, the index of the first such
recapture among the captures; at nodes that cut on a capture, the index of the cutting capture among the captures,
split by fast and slow path.

**Fix sketch (knob `CAPTURE_KEY`, default 0).** 1: both paths score `200000 + 8 × victim − attacker / 8` once the
capture is known not to lose (the slow path keeps its test of the sign). 2: today's key plus `victim` when the target
square is the previous move's target (a recapture bonus). Test the form once (owner 11-19), constants later.

**Expected effect (estimate).** Bench nodes −0.5 to −2%, most of it in qsearch; games −2 to +6.

### S3-23: a q-cache hit is written back with a weaker flag

**Defect (VERIFIED).** `qsearch` probes the q-cache at every node, the qsearch root included (`search_v2.cpp:1426-1428`),
and returns the stored score on a hit. `qsearch_entry` cannot tell a hit from a search: it recomputes the flag from the
returned score and the current window and stores again (`:224-234`). An EXACT entry (a value proved inside an open
window) probed by a null-window search is always "at or below alpha" or "at or above beta" for that window, so it is
written back as an upper or lower bound. From then on it no longer answers an open window, and it no longer answers a
null window on the other side of its value. Bound entries are written back unchanged. The older `get_q_search_eval`
has the same shape (`search_engine.cpp:11204-11223`). The main TT does not: a hit returns before any store
(`search_v2.cpp:1068-1072`).

**Size.** EXACT q entries come only from qsearch roots reached with an open window, so they are few. Small.

**Count.** In `qsearch_entry`, behind `FIXCNT_*`: stores where the slot already holds this key with flag EXACT and the
new flag is not EXACT.

**Fix sketch (knob `QCACHE_KEEP_EXACT`, default 0).** `addToQCache` keeps a same-key EXACT entry when the new score is
equal and the new flag is a bound. One compare in the store, none in the probe.

**Expected effect (estimate).** qsearch nodes 0 to −0.5%. A correctness-of-bookkeeping item; take it when the q-cache
is next touched (S2-16's fix lands in the same function).

### S3-26 (latent): buckets without an age

`TT_WAYS` 2, 4 or 8 turns the TT into buckets (`cache_management.h:1141-1148`, `:1230-1255`). The store takes an empty
slot, else the same position, else the shallowest entry of the bucket (`:1235-1241`). There is no age or generation
field. Deep entries of earlier searches are never the shallowest, so they are never evicted; over a game every
bucket's "deep" slots fill with positions that can no longer occur, and the bucket ends as one always-replace slot
plus dead weight. Any test of `TT_WAYS` without an age understates buckets. Fix before testing: a search counter
stored per entry and a victim rule of `depth − k × age` (every reference engine, 10-22 audit §1b). This belongs in
S2-29's run, which already has to choose a same-position rule.

### S3-27 (latent): lazy generation is not order-identical once a quiet scores 200,000

Lazy generation (`SV2_LAZY_GEN`, off) searches "the leading good captures" first and promises that every move keeps
today's index (`search_v2.cpp:113-168`). It builds that lead from a captures-only list (`move_gen.h:1183-1191`,
`:1433-1438`). The promise holds only while no quiet move scores 200,000 (`MG_CAPTURE_TIER`, `move_gen.h:1048`) or
more, because such a quiet would sort in front of the captures in today's full list. Without `moveFrequency` the
largest possible quiet score is about 150,000 (§4). With it, the bound is passed at depth (next section). The probe
that checked the promise (`quiet_above_good_capture`, `move_gen.h:1491-1508`) was read at depth 10, where it cannot
fire. Fix before enabling at long controls: cap the `moveFrequency` term below the tier, or build the lead from the
full scores.

---

## 2. Known items, sharpened (no new ids)

- **S2-28 (`moveFrequency` unscaled): it accumulates, and it passes the capture tier from about iteration 17
  (REPORTED, arithmetic).** The bonus `(d−1)² × (L−i)²` is added after every iteration (`search_engine.cpp:4668-4687`),
  the table is divided by 4 once per search (`:4431`) and again before iterations 13, 18, 23, … (`:4504-4505`). With a
  PV as long as the depth, the first PV move holds about 49,000 after iteration 12 (not the 17,000 of one iteration),
  about 245,000 after iteration 17 and 400,000 to 800,000 between iterations 20 and 22. The register's 144,000 "at
  depth 20" is one iteration's bonus. Consequence at long controls: the quiet first PV move is ordered in front of
  every winning capture (200,000 + at most 9,000 + promotion) at every node where that from-to move is legal for that
  side, not only on the PV. Real PVs are often shorter than the depth (a TT cut ends them), so these are upper
  figures. Count: the largest `moveFrequency` value per iteration on the `[iter]` line (cold, per iteration).
- **S2-9 (lazy estimates stored): a third path, and one that is nearly empty.**
  - Third path (VERIFIED): a qsearch root's lazy stand-pat estimate is returned to the parent at remaining depth 1,
    which does not store it (`search_v2.cpp:1083`: `cur_depth < depth_limit − 1`) but folds it into its own fail-soft
    value; the grandparent stores **that** in the TT at depth 2 (`:1084`, `:1275`, `:1300`). #277 counted the
    reverse-futility returns and the q-cache stores, not this one. It is the cheapest kind of entry to redo, so it is
    small; `LAZY_FAIL_HARD` covers it.
  - Nearly empty (VERIFIED): lazy stand-pats **below** the qsearch root. A qsearch node below the root is reached by a
    capture, a promotion or a check evasion; the gate refuses the lazy exit after a capture (`lazy_gated`,
    `search_engine.cpp:1622-1632`), so only promotions and evasions remain. #277's q-cache count is therefore close to
    complete.
- **S2-29 / TT1 (replacement).** The same-position rule is the register's. The different-position rule ("always") is
  not a problem at 10+0.1 (2 to 3% of the table written per search) and is one at long controls (S3-25). An age is not
  needed for today's direct-mapped table, since a different position always takes the slot; it becomes necessary with
  buckets (S3-26).
- **S2-12 (qsearch lazy request has no mate guard).** I agree with the triage: not real. With beta at or above
  9,000,000 the exit cannot fire; with beta at or below −9,000,000 the full eval would cut too. The futility site is
  the same: with a mate-sized alpha every late quiet is pruned with or without the lazy exit (`search_v2.cpp:1202`).
- **The history tables run on three clocks (known facts, one consequence).** Butterfly and capture history halve every
  `DECAY_INTERVAL` counted nodes; the continuation table every 16 intervals, which a 360,000-node search never
  reaches (560,000); `moveFrequency` per search. So at 10+0.1 the butterfly table is halved about ten times per move
  and the continuation table not at all: the quiet score and the reduction's history sum (`history_lmr_delta`,
  `search_engine.cpp:1837-1859`, both weights 1) are carried by the continuation table. S2-5's stuck interval moves
  this balance by a factor of 4 to 6 in the butterfly table's favour. Any S2-5 arm changes the mix of the two tables,
  not only their age.

---

## 3. Lazy eval: sites, bounds and what each caller trusts (all VERIFIED)

The request is `lazy_request(up, need, site)` (`search_engine.cpp:1634-1643`): `need` is in the search frame (root side
positive); `up` says whether the site cuts above or below it; the function converts to the eval's Black-positive frame
with `Config::side_to_play`. The exit is in `placement_and_piece_eval_v2` (`eval_v2.cpp:5153-5169`): estimate = cheap
part + the parent's positional part when known; margin `M` = 1,500 (parent known) or 3,000, times the site scale (100
at all three).

| site | where | fires when (side to move's view) | the caller then | what leaves the node |
|---|---|---|---|---|
| reverse futility | `search_v2.cpp:507-521` | estimate ≥ beta + `RFP_MARGIN` × remaining depth + M; non-PV, beta not a mate score, depth 1 to 6, no threat gate | always cuts (`:524-539`; `g_lazy_nocut` checks it) | the estimate, as the node's value (S2-9, S2-26) |
| stand-pat | `search_v2.cpp:1471-1492` | estimate ≥ beta + M; null window only (guard bit 2); not in check | always cuts | the estimate (S2-9) |
| futility of a child | `search_v2.cpp:1194-1213` | the child's estimate ≤ alpha − margin − 1 − M; null window; late quiet, remaining depth 2 to 4 | always prunes | nothing at shipped settings: the value only feeds `best_early_eval` (unreachable, S2-3) and `g_sv2_fut_bound` (`SV2_FUT_BOUND` off) |

Checked, each clean:

- **Signs and frames** at all three sites, for both sides and both root colours.
- **No request outlives its eval call:** set, one `eval_by_mode` call, cleared; no return in between.
- **A lazy value is never put in the eval cache** (`search_engine.cpp:11358-11366` returns before the store), and the
  eval-cache hit path ignores the request and returns the full value.
- **The parent record is keyed.** `g_lazy_ply` is indexed by history size and checked against a 64-bit key of the
  parent's bitboards, side, castling rights and en passant square (`:1586-1611`). A stale record cannot be used.
- **The null move** reuses the node's history slot; the record is saved and put back (`search_v2.cpp:548`, `:601-602`),
  and the gate refuses the exit in the null child (`lazy_gated`, the equal-side test).
- **The special evals** (drawn class, rook against minor, bare king) return before the exit, as full values, and leave
  no positional part (`eval_v2.cpp:5021-5093`).
- **`g_lazy_fired` cannot be stale** where it is read: `lazy_request` clears it and each reader follows its own eval call.
- **The eval is a function of the position alone** (`moveNum` is stored in the context and never read; no
  `side_to_play` in `eval_v2.cpp`), so a kept record and a cached positional part stay correct. `TEMPO_V2` is 0, so
  the parent's value carries no tempo of the wrong side.
- **A promotion is not gated** (documented in LAZY-EVAL.md §3). The estimate then errs toward the side that promoted
  (the parent's positional part still holds the passed pawn's bonus), and the two sites in the child belong to the
  other side, so the error makes them fire less. Conservative. REPORTED (reasoning, not measured).

One exposure, not a defect, **REPORTED:** the margins are absolute (1.5 and 3 pawns) while part of the positional part
is multiplicative: the winnability scale is applied to the finished total, material included (`eval_v2.cpp:5431-5437`),
and in pawn endings the race term reads the side to move (`:5422-5428`). Step 0b's error figures (median 0.35, p99
2.6 pawns) come from sets that are mostly middlegames. The probe rows carry the phase (`lazy_probe_features`,
`eval_v2.cpp:4968-5000`), so the error by phase can be read from the existing rows in `E:\chess_data\lazy\` with the
existing reader, no engine run. Do it before the lazy margins are next touched.

---

## 4. The quiet-move score: sizes of the terms (VERIFIED from the code; "typical" is REPORTED)

`move_gen.h:1341-1407` on a move-cache miss; `score_quiet` (`:1521-1574`) for the top-K re-rank on a hit (the same
terms without the threat term).

| term | range | where it is bounded |
|---|---|---|
| butterfly history | ±16,384 by gravity; typical values small (bonus = remaining depth², halved every 35,000 nodes); the raw exit credit can pass the bound until the next halving | `hist_update` `cache_management.h:1513-1518`; raw add `search_v2.cpp:957` |
| continuation history (1 ply) | ±16,384 by gravity; raw adds from the pre-search (S2-23) | same |
| killer 1 / killer 2 | 10,000 / 9,000 | `cache_management.h:1357-1367` |
| counter move | 8,000 | `move_gen.h:1344` |
| promotion | 5,000 / 10,000 / 15,000 / 20,000 (S3-20) | `move_gen.h:1346` |
| threat order, escape | + value × 10: knight 32,500, bishop 34,500, rook 50,000, queen 100,000 | `move_gen.h:1388-1391` |
| threat order, stepping onto an attacked square | − value × 9.5: 30,875 to 95,000 | same |
| `moveFrequency` | unbounded; about 49,000 at depth 12, above 200,000 from depth 17 (§2) | `cache_management.h:1400-1416` |
| good capture | 200,000 + 0 to 10,000 + promotion + capture history + `moveFrequency` | `move_gen.h:1309`, `:1337` |

Readings:

- **The largest quiet score without `moveFrequency` is about 150,000** (queen escape + both history caps + killer +
  counter), below the capture tier. Only `moveFrequency` can cross it.
- **The threat term decides the order whenever it applies.** Every move of a piece attacked by a cheaper one that does
  not land on such a square gets the same bonus, three to ten times the killer. Stockfish 18 uses the same sets and
  nearly the same factors (20 and −19 times the piece value, `movepick.cpp:174-175`, read), but against history sums
  that reach well past 100,000, so there the term competes; here it rules. `THREAT_ORDER_W` was swept (50, 100, 200;
  100 best), so this is a description, not a defect. If the history bonus scale is ever raised (G4), re-sweep W with it.
- **Killers and the counter move outrank history in practice**, as in a staged picker. Consistent.
- **Overflow: none.** `h × |delta|` in `hist_update` stays below 2^31 unless a value passes 131,072 with the largest
  delta (it cannot: deltas are at most about 3,800 and raw adds are halved long before). `moveFrequency` sums stay
  below 2^28 at the depth cap. Sort keys map the full `int` range (`move_gen.h:265-278`). The only 16-bit store in
  these areas is the eval cache's positional part, which is range-checked (`search_engine.cpp:11418`).

---

## 5. Checked and clean (do not re-read)

**Transposition table**
- **Frames.** The TT holds root-frame scores; search v2 converts on every store and probe (`root_lo`, `root_hi`,
  `root_flag`, `tt_store`, `tt_use`, `search_v2.cpp:172-199`). The flag of a parent-side fail-high / fail-low is right
  for both sides.
- **Root side changes.** The UCI layer clears every table when the side it searches for changes and on `ucinewgame`
  (`uci_main.cpp:246`, `:528`); `bench` clears the TT and q-cache per position (`:406-407`). No root-frame value
  survives a colour change.
- **Key and collisions.** The full 64-bit key (position, side, castling rights, en passant square) is stored and
  compared in all four tables (`cache_management.h:1136`, `:1056`, `:734`, `:899`). A false match needs 42 further
  bits to agree: about 2 × 10⁻⁷ per million probes. The Zobrist update handles castling, en passant and promotion
  (`:516-638`).
- **Depth labels.** Stores and probes use the same "parent's remaining depth" convention, including after a check
  extension (`search_v2.cpp:1057`, `:1065`, `:1084`) and at the null move's probe (`:575`).
- **Flags.** The first move is flagged against the window it was searched with; a scout stores bounds only; a
  re-search is flagged against the drifted window (`ENABLE_TT_FLAG_FIX`, `:1367`).
- **What a probe trusts.** Depth at least the depth needed, then EXACT at any window, a lower bound at or above beta, an
  upper bound at or below alpha (`use_tt_entry`, `search_engine.cpp:10193-10229`). The stored window fields are dead.
  The null move's probe accepts EXACT entries only (`search_v2.cpp:575`).
- **Mates and zeros.** Never stored (`cache_management.h:1171-1177`, `:1195`). With `FIX_TT_MATE` on, the conversion to
  and from distance is consistent at every store and probe I read, the null probe and `tt_entry_cuts` included (latent).
- **Time-up.** A partial value is 0 or is discarded before a store at every search v2 site (the exception is S2-21).
- **A TT hit stores nothing** (`search_v2.cpp:1068-1072`), unlike the q-cache (S3-23).

**q-cache**
- Probed at every qsearch node, stored only at the qsearch root. Stored values do not depend on the q-depth (no quiet
  checks enter qsearch at shipped settings), so a root's entry is valid for a deeper node.
- "Never aged" is sound for pure entries: the value depends on the position only. Age would not help; the impure
  entries are S2-9 and S2-16, and a 2^20 always-replace table forgets them in about ten moves at 10+0.1.
- Mates are refused (`cache_management.h:1078`); a time-up result and a root repetition are refused
  (`search_v2.cpp:227-229`).
- With the stand-pat seed, a delta-pruned capture raises the returned bound to what the prune assumed (`:1518-1523`), so
  the stored upper bound is honest.

**Eval cache**
- Key = position key with castling and en passant, salted by the eval arm (`search_engine.cpp:1557-1572`); value in the
  fixed Black-positive form, so right for either root colour; checkmates not cached (`:11413`).
- `static_eval_for_improving` reads it and does not write it, so no entry is written without its positional part.

**Move cache**
- Legal moves depend only on what the key holds. Nodes iterate a private copy.
- qsearch's captures-only generation is never stored as a full list (`search_engine.cpp:11021-11029`).
- A capture-only list left by lazy generation is marked and regenerated by every other reader (`:10635`, `:11019`).

**Scorer and history**
- Read and write indices agree for butterfly, continuation and capture history (`move_gen.h:1305`, `:1396`, `:1405`
  against `search_v2.cpp:250-251`, `:300`).
- The gravity update holds the bound for bonus and malus alike, with symmetric rounding (`cache_management.h:1513-1518`).
- Killers and counter moves are cleared per search (`search_engine.cpp:4322-4323`); killer reads are guarded past ply 63.
- The keyed sort is exactly the stable order: keys are unique, lists stay under 256 moves, the 8 padding keys sort last.
- The 2-ply continuation term in the reduction's history sum is inert while its table is off (the move stack is
  written only under the knob).
- Dead code confirmed, do not cite: `cmh_ref_bb` has no caller; the history writes at `search_engine.cpp:9158-9198`
  are inside a comment block; `HIST_DIRTY_GATE` and `MF_PRESENCE` are 0.

---

## 6. For whoever builds the next counter build

All of the counts above are fixed-depth and fit one build beside the triage's "cnt20" list:

| counter | site | finding |
|---|---|---|
| under-promotions searched, nodes under them, times best | `sv2::score_move` | S3-20 |
| eval / q / move cache misses per search (already printed) at depth 12, 18, 22 | stats line | S3-25 |
| top-K fires; displaced distance; displaced past the pruning limit; captures chosen or displaced | `buildMoveListFromReordered` hit path | S3-21 |
| hits with a non-capture at index 1 and a good capture behind it | same | S3-22 |
| EXACT q entries overwritten by a bound of the same key | `qsearch_entry` | S3-23 |
| index of the recapture / the cutting capture among captures, by scorer path | scorer, `SV2_MOVE_PROBE` build | S3-24 |
| largest `moveFrequency` value per iteration | iteration log (cold) | S2-28 |

The counter build's bench must equal production (254 / 35,922,538). No counter needs a bare `getenv` in search code.
