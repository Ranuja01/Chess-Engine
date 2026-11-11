# Move generator and move ordering: audit of our pipeline (2026-11-11)

Input for search v2 stage S3 (the staged picker beside our cached lists). It traces what runs today, lists the waste with
sizes, flags correctness concerns, and sets out options for a "best of all worlds" discussion. Nothing here is a decision.

Companion: [`reference/movegen-giants-2026-11-11.md`](../reference/movegen-giants-2026-11-11.md) (how Stockfish 19 / 11,
Reckless, Ethereal and Berserk do it). Not repeated here: the history tables' keys and update rules
([`research-2026-10-22/SEARCH-AUDIT-MOVE-ORDERING.md`](research-2026-10-22/SEARCH-AUDIT-MOVE-ORDERING.md)), and the speed
items F4, F5, F7, F16, F17, F27-F30 of [`owner-cpp-ideas.md`](owner-cpp-ideas.md), which are cited by number.

Read-only: nothing was built, run or measured. Code facts carry file:line from the working tree of 2026-11-11 (HEAD
`4c2a321`). "Inference" marks my own conclusions. Shares are % of search time from the 11-11 profile of the v2 path
(plan §6.2b: perf, 24 game positions to depth 12, no PGO). Knob values are read from the header and the shipped preset
block (`search_engine.cpp:2678-2719`); the `[toggles]` dump is the authority if they disagree.

## 0. Bottom line

1. **Legality is sound by code read.** The generator is the python-chess design (blockers, evasions, a pin mask and a
   rebuilt-occupancy test for en passant, castling path attack tests). I found no legality defect. Two **latent** defects
   sit beside it, both off in the shipped engine: the per-ply move buffer is overwritten when singular extensions re-enter
   the same ply (C1), and quiet checks in qsearch are found only on move-cache hits once noisy-only generation is on (C2).
2. **Most of the 20% is not "generation" in the narrow sense.** `generateLegalMovesReordered`'s own 10.5% is scoring,
   sorting and list handling; the two generator functions are 9.7%. Three earlier attempts to trim the generator with the
   result unchanged measured nothing (08-25), and collapsing the 16 passes measured +0.25% ± 0.5% (F7). What is left is
   work that should not happen at all, not work to do faster.
3. **Three kinds of work that should not happen**: (a) moves generated, legality-tested, scored and sorted at nodes that
   cut on an early move; (b) late quiet moves that are made, probed and unmade only to be pruned (already logged as G20 /
   I6 in the pruning audit); (c) exchange evaluations computed twice, or computed when the answer is known, in qsearch.
   (a) and (b) change the move order or need care to keep it; (c) is result-identical.
4. **The hash move is not used.** `ENABLE_TT_MOVE` is off (`search_engine.h:1622`); the head of a cached list is our hash
   move, and it exists only on a cache hit. The hit rate in our notes is 12% (10-22) or 24% (08-25), both stale, and the
   move cache now has a quarter of the TT's slots (gen 6 sizing, `search_engine.cpp:2650-2652`). So at most interior nodes
   the first move tried is the best capture by victim and attacker, even when the TT remembers a best move (inference from
   those numbers; one counter read settles it, §7).
5. **Our exchange evaluator answers for a square, not for a move** (`see(to, side, state)`, `cpp_bitboard.h:1968`). Known
   since the pruning audit (G13 / G14). Its ordering consequence is new here: when a cheap and a dear piece attack the same
   defended pawn, the dear capture inherits the cheap capture's exchange value and can sort above it (C3). A staged picker's
   good / bad capture split needs a per-move test in any case.

## 1. The pipeline, traced

### 1.1 Data types

- `Move`: three bytes, from / to / promotion, no piece, no capture or en passant flag (`search_engine.h:3582-3598`).
  Promotion is 1 for "none", 2-5 for knight to queen.
- `BoardState`: six piece bitboards, two colour bitboards, occupancy, promoted, turn, castling rights (rook squares), en
  passant square, clocks; no piece-on-square array (`search_engine.h:3512-3565`). "Which piece is on this square" is a chain
  of up to six tests wherever it is needed (`cpp_bitboard.h:1733-1749`, `:1461-1495`, `move_gen.h:380-393`).
- Slider attacks: PEXT into three flat tables, rank / file / diagonal (`cpp_bitboard.h:43-54`, F5 done). A rook needs two
  lookups, a queen three (`cpp_bitboard.h:1447-1455`).
- Move cache: 48-byte slots holding a 64-bit key and a `std::vector<Move>` whose data is on the heap
  (`cache_management.h:85-91`); the key is compared in full (`:924`).
- Per-ply buffers `g_moveBuf[128]`, `g_noisyBuf[128]` (`cache_management.h:411-414`, `cpp_bitboard.h:22`).

### 1.2 An interior node (v2 path, `search_v2.cpp`)

In order, for one node:

1. Copy the state (112 bytes) (`search_v2.cpp:215`). Check test: one attacker lookup on our king (`:232`).
2. Static eval, reverse futility, null move (`:239-336`). The move list does not exist yet, which is right.
3. `buildMoveListFromReordered` (`:343`; `search_engine.cpp:10195`): copy the state again (`:10200`), compute the cache key,
   read the slot, copy its list into `g_moveBuf[ply]` or clear the buffer (`cache_management.h:920-929`).
4. **Cache miss** (`search_engine.cpp:10425-10448`): `generateLegalMovesReordered` (`move_gen.h:704`):
   - king square, blockers, checkers, once (`:754-756`);
   - 12 capture passes and 4 quiet passes, each a (mover set, target set) pair (`:758-854`). Every pass calls
     `generateLegalMovesPre` (`:218`): generate pseudo-legal moves into three scratch byte-vectors, then test each with
     `is_safe` and push the survivors onto three more byte-vectors (`:229-272`). In check the pass generates evasions
     (`:625-675`). The capture pairs are disjoint and complete; the quiet pairs put the king first in endgames (`:816-834`);
   - build a second `BoardState` for the value and exchange helpers (`:857-873`);
   - the lesser-attacker sets for threat-aware ordering, one lookup per enemy knight, bishop and rook (`:893-903`;
     `THREAT_ORDER_W` is 100 in the shipped preset, `search_engine.cpp:2698`);
   - score every move (`:905-1000`). Capture: 200000 + victim − attacker + promotion + move frequency + capture history, or
     the exchange value when the victim is cheaper than the attacker; a negative exchange value drops the 200000
     (`:909-931`). Quiet: butterfly + killer bonus + counter bonus + continuation history (1 ply, from×to, a 134 MB table,
     `cache_management.h:200`) + threat term + promotion + move frequency (`:933-999`);
   - one stable sort of all moves (`:1013-1036`, `:174-212`), then convert to `Move` (`:1038-1040`);
   - copy the list into the cache slot (`cache_management.h:981-985`).
5. **Cache hit** (`search_engine.cpp:10214-10423`): find the first non-capture after the head, insert killer 1, killer 2 and
   the counter move there (`:10164-10193`; `FIX_KILLER_SPLICE` is 1 in the shipped preset, `:2703`); if the last cutoff here
   came from index 3 or later, re-score the whole quiet tail and bring the best two forward (`:10311-10380`).
6. Move loop (`search_v2.cpp:417-517`). Per move: is it en passant, is it a capture (`:423-424`); update the key (finds the
   moving piece again); `make_move` (copies the state, finds the piece again, `search_engine.cpp:3920-4026`); then
   `score_move` (`:577`): two more state copies (`:579`, `:587`), repetition test, check test for the extension (`:599`), TT
   probe of the child (`:615`), a second check test on the same position (`:641`), **then** late-move pruning, history
   pruning and exchange pruning (`:691-710`), then the child search.
7. Cutoff (`:494-516`): if the cutting move was not first, promote it to the front of the cached list
   (`cache_management.h:1261-1324`); history updates; TT store.
8. No move improved: mate / stalemate test (`:520-529`).

The TT is probed for the **child** inside the parent's loop, so a node never sees its own TT entry before it builds its
list (except the singular probe, off by default).

### 1.3 A qsearch node (`search_v2.cpp:860-1009`)

1. Depth cap, q-cache probe, check test (`:874-906`).
2. **In check**: the full main-search list through `buildMoveListFromReordered` (`:910`), every evasion searched. This
   writes the move cache.
3. **Not in check**: stand pat first (`:949-961`), then `buildNoisyMoveList` (`:964`; `search_engine.cpp:10575`):
   - copy the state; read the move cache (a main-search visit may have left the full list);
   - on a miss, generate noisy moves only (`QGEN_NOISY_ONLY` 1): the 12 capture passes and one pawn pass restricted to
     promotion pushes and en passant (`move_gen.h:795-804`), scored and sorted by the same code as the main list; not stored;
   - filter (`search_engine.cpp:10629-10744`): keep promotions, en passant, and captures whose exchange value is ≥ 0. The
     exchange value is computed here for **every** capture (`:10651`). Quiet moves are dropped (`:10671-10672`): neither
     quiet-check knob is on by default (`search_engine.h:3059`, `:3100`).
4. Loop: per-capture delta pruning, make, recurse, unmake (`:965-1006`). No history is read for pruning or written.

### 1.4 The root and the pre-search

- `reorder_legal_moves` (`search_engine.cpp:8136`) takes the root list from the previous iteration, or builds it once with
  `buildMoveListFromReordered` at ply 0 (`:8162-8169`).
- With the pre-search on, each root move is made and `pre_minimizer` (`:8610`) searches the reply position at reduced
  depth. It builds the reply list at ply 1 (`:8699`) and keeps a copy per root move (`second_moves`), which the real
  iteration then uses as the order of replies. These are whole-vector copies (`:8699-8700`, `:8230`), a few dozen per
  iteration: not a hot path.
- The root order is by pre-search score; the generator's own order is used only for the first list and for ties.
- In S1 the root, ply 1 and the pre-search still run today's code and enter v2 at ply 2 (plan §7e).

## 2. Correctness

### 2.1 Legality, case by case (code read; no defect found)

| case | how it is handled | where |
|---|---|---|
| pinned piece | blockers of our king computed once per node; a blocker may move only along the line through the king | `cpp_bitboard.h:892-934`, `:1033-1036` |
| king steps (not in check) | attackers of the target with the king still on the board; correct because no slider attacks the king | `:1011-1027` |
| king steps (in check) | the whole line through king and each checking slider is removed first, then the attacker test | `move_gen.h:631-656` |
| double check | only the king moves | `move_gen.h:658-660` |
| single check | other pieces may only capture the checker or block | `move_gen.h:662-666` |
| en passant, pinned capturer | pin mask from the king through the capturer | `cpp_bitboard.h:1029-1031`, `:1070-1110` |
| en passant, rank discovery (both pawns leave the rank) | occupancy rebuilt without both pawns, rank and diagonal slider test on the king | `cpp_bitboard.h:1112-1143` |
| en passant as a check evasion | generated when the checker is the pawn that just advanced two | `move_gen.h:668-673` |
| castling | rights are rook squares; path empty; king's square, path and target not attacked; not generated in check | `move_gen.h:551-600`, `:664` |
| castling rights lost | cleared when the king, the rook, or a capture touches the squares | `cpp_bitboard.h:1569-1576` |
| promotions | four per push or capture; kept in the qsearch list | `move_gen.h:454-461`, `:508-515` |
| each move generated exactly once | capture pairs disjoint; castling and en passant arise only in the quiet pass of their piece | `move_gen.h:758-780`, `:555`, `:603` |

Evidence beyond the read: perft matched the published counts for the start position, Kiwipete and position 3 on 10-23
(`owner-cpp-ideas.md` F5 row), and the port leaves this code untouched. Any new generator for S3 should pass the same
perft set plus positions built for pinned en passant captures and castling through attack, before any search test.

### 2.2 Concerns

**C1. A singular extension overwrites the list its own node is iterating (latent; `ENABLE_SINGULAR` is off,
`search_engine.h:2900`).** The node holds a reference to `g_moveBuf[ply]` and a reference to the current move inside it
(`search_v2.cpp:343`, `:419`). The exclusion search re-enters the same node function at the **same ply**
(`search_v2.cpp:437`; today's search `search_engine.cpp:7075`, `:7814`), and that call's `buildMoveListFromReordered`
refills the same buffer (`search_engine.cpp:10211-10213`). The buffer comment states the invariant that is broken
("deeper nodes use deeper buffers, so it is never overwritten while we iterate it", `:10207-10210`). If the cached order
changed in between (a promotion through a transposition, an eviction and regeneration, the lazy tail re-score reading
newer history), the outer loop continues over a different order: a move can be searched twice and another skipped, and
the extension can land on a move that is not the TT move. All moves remain legal, so nothing crashes. Two consequences:
S5 plans singular extensions, so the v2 move source must give the exclusion search its own list; and the "singular is
neutral" verdict on v1 may have been measured with this live (check whether the per-ply buffers predate that test:
`git log -S g_moveBuf`).

**C2. Quiet checks in qsearch would be found only on cache hits (latent knob interaction).** With `QGEN_NOISY_ONLY` 1 the
miss path generates no quiets (`search_engine.cpp:10602`), so `ENABLE_QCHECK_FULL` / `_MASK` can only see quiet checks
when a full list happens to be cached. Either knob would then behave differently by cache state. Nothing guards the
combination. The plan's table (§4) lists "qsearch with checks" as a current feature; at the defaults read here no quiet
check enters qsearch (`:10668-10672`).

**C3. The exchange value is per square (known), and it mis-orders one capture shape (new).** `see(to, side, state)` starts
with the cheapest attacker of the square (`cpp_bitboard.h:1859-1862`). The scorer calls it for a capture whose victim is
cheaper than its attacker (`move_gen.h:923-929`). Example: our pawn and our queen both attack a pawn defended once. Pawn
takes pawn scores 200000 + 0 by victim − attacker. Queen takes pawn asks the square, gets the pawn capture's value
(+1000, or 0) and scores 200000 + that, so it sorts level with or above the pawn capture, in the good tier, although it
loses the queen for a pawn. The same value admits that capture to qsearch (`search_engine.cpp:10651`) and exempts it from
the losing-capture prune (`search_v2.cpp:708-709`). How often this fires is unmeasured (§7). A per-move test fixes all
three sites and is a behaviour change.

**C4. En passant is ordered as a quiet and searched as a capture.** The scorer's capture test is "target square holds an
enemy piece" (`move_gen.h:909`), so en passant takes the quiet branch and reads butterfly and continuation history; the
search flags it as a capture (`search_v2.cpp:423-424`) and writes capture history on a cutoff (`:509-512`). Its quiet
entries are never written, so it sorts as an unknown quiet. Rare; cost is a late en passant now and then.

**C5. A cached list is trusted on key equality alone.** No move of a cached list is checked against the position. With
the full 64-bit key this is the same risk every engine takes with its TT. It becomes a real requirement if a later TT
layout stores a short key (F8's 16-byte entry): then a stored move needs a "fits this position" test before it is played,
which we do not have (today the legal list itself is the filter: `promoteMoveToFront`, `cpp_bitboard.h:809-816`).

**C6. Ply 128 and beyond share one fallback buffer** (`search_engine.cpp:10211-10212`, `:10582-10593`). Two such plies would
overwrite each other. Not reachable at today's depth caps; worth an assert in debug builds.

**C7. IIR's "first visit" test is the move cache** (`search_v2.cpp:339-341`; `ENABLE_IIR` off at the header default,
`search_engine.h:2909`). A picker arm without the cache needs another trigger (the references use "no hash move"), so the
S3 arms would differ in IIR as well as in ordering unless the trigger is made common first. Qsearch evasion nodes also
write the cache (§1.3), so the signal is not purely "the main search has been here".

## 3. Ordering accuracy: what is new since the 10-22 audit

The 10-22 audit's table of terms and tables stands. Changes and additions:

- The killer splice defect it reported (§4.2 there) is fixed in the shipped preset (`FIX_KILLER_SPLICE` 1, gen 6.5).
- Threat-aware quiet ordering has shipped (gen 6.3). Ours uses the same lesser-attacker sets as Stockfish 19
  (`move_gen.h:893-903` against SF19 `movepick.cpp:201-210`); Reckless adds safe attacking squares and checking squares
  (reference note §2.3).
- qsearch generates noisy moves only (the audit's item 3).
- New: C3 (a dear capture inheriting a cheap one's exchange value) and C4 (en passant).
- New: a king capture always runs the exchange evaluator, because the king's value is never ≤ the victim's
  (`move_gen.h:923-929`). Harmless to the order, a small cost.
- Unchanged and still unmeasured: `moveFrequency` is added to captures and quiets without a bound (`move_gen.h:918`, `:999`),
  so a quiet can in principle outscore the 200000 capture base. This matters for any lazy scheme (§6, option B).
- Still true: quiet scores are frozen at the node's first visit and replayed from the cache. The frozen local order beat a
  re-sort by current history when tested (comment at `search_engine.cpp:10306-10310`), which the 10-22 audit reads as a
  symptom of slow history rather than a law.

## 4. Speed: the 20%, function by function

| function | share | what is in it | algorithmic waste | C++ overhead |
|---|---|---|---|---|
| `generateLegalMovesReordered` | 10.5% | pass set-up, second `BoardState`, threat sets, the scoring of every move, the stable sort, conversion to `Move` (the scorer and sort are inlined into it: inference from the symbol list) | scoring and sorting moves that are never searched; a continuation-history read from a 134 MB table per quiet; exchange values the qsearch filter recomputes | three byte-vectors → scored structs → `Move` vector → cache slot: four forms of one list |
| `generatePseudoLegalMoves` | 5.9% | piece moves, pawn moves, castling, en passant for one pass | quiets generated at nodes that cut on a capture; the piece type found by a test chain per piece per pass | three `push_back`s per move |
| `generateLegalMovesPre` | 3.8% | the per-move legality filter and the second set of pushes | none in the test itself (two operations on the common path) | the scratch-then-filter double buffering (measured at about 1%, 08-25) |
| `buildNoisyMoveList` | 1.2% | cache read, the filter loop | an exchange evaluation per capture, part already known | state copy |
| `buildMoveListFromReordered` | 0.9% | cache read, splice, lazy tail re-score | — | state copy; three linear searches |
| `__memmove` | 0.7% | list copies out of and into cache slots | a full copy when only the head is needed | heap list behind the slot: a second cache miss per probe; only the slot is prefetched (`search_engine.cpp:3917`) |
| `see_impl` | 3.1% | exchange evaluation: ordering, qsearch filter, pruning | the duplicate in qsearch (W3) | — |
| `attackersMask` | 1.5% | check tests, king-step legality, castling | the same position's check status computed up to four times (F16, F27) | — |

What earlier measurements already rule out (so they are not on the waste list):
- **Collapsing the 16 passes into one**: built in full, identical lists, +0.25% ± 0.5% (F7, parked).
- **A per-node attack memo, direct emission, a short legality path**: all identical, none faster (08-25; the comments at
  `move_gen.h:34-37`, `:223-225` and `search_engine.h:1588-1595` record them).
- **Fixed arrays for the existing generator alone**: the allocation traffic is already gone (`SCRATCH_REUSE`, F4); what is
  left is capacity checks.

Those results say the generator's inner loops are not where a result-identical gain is. They do not bound a picker that
skips generation, scoring and sorting altogether.

## 5. Waste list

Size is % of search time. "Identical" = the search tree does not change if done exactly.

| # | what | evidence | size | identical? | belongs to |
|---|---|---|---|---|---|
| W1 | **Whole list built at nodes that cut early**: quiets generated, filtered, scored and sorted, then never looked at | code §1.2 step 4; 08-25 probe: 41% of generated moves never examined at generating main-search nodes (v1, old tree) | ≤ 0.4 × the main search's part of the 20%; my guess 3-6% (inference) | **no**: quiets would be scored after the captures' subtrees have changed the history tables | S3 picker |
| W2 | **Late quiets made only to be pruned**: key update, `make_move`, two state copies, repetition test, two check tests and a TT probe before late-move pruning returns | `search_v2.cpp:447-450` then `:588-641`, `:691-697`; the limit is index 1 + depth², so index 2 at remaining depth 1 (`LMP_BASE` 1, `search_engine.h:1374-1377`), on the side that takes the pruning path there (asymmetry A3 searches every move at that depth on the other side, `search_v2.cpp:631-639`); pruning audit G20 / I6 | unknown; `make_move` + `update_state` are 9.0% and the child probe about 6% for all moves together. A large part at frontier nodes (inference) | yes if the repetition, TT and check tests are reproduced from the child key before the move is made; no if pruning simply moves ahead of them | S3 picker (a "stop quiets" signal also saves their generation and scoring) with F27; or alone as I6 |
| W3 | **Exchange value computed twice, or needlessly, in qsearch**: the filter recomputes it for every capture (`search_engine.cpp:10651`). For victim ≥ attacker the answer is always ≥ 0 (the first capture cannot lose more than its attacker); for the rest the scorer has just computed it (`move_gen.h:928`) | code; `see_impl` 3.1% over all callers | 0.5-1.5% (inference) | **yes** (the filter uses only the sign; a verify mode can compare) | independent speed fix |
| W4 | **Check status recomputed**: node entry (`search_v2.cpp:232`), twice in the parent for the same child (`:599`, `:641`), again as `checkers` in the generator (`move_gen.h:756`) | code; `attackersMask` 1.5% + inlined copies | 1-2% (F27's estimate) | yes | S2 (F16 / F27) |
| W5 | **The piece on a square is rediscovered** in the scorer (`move_gen.h:914-915`, `:976-979`), the key update, `update_state` (`cpp_bitboard.h:1540`, `:1555-1567`), `is_en_passant` + `is_capture` in every loop | code | 1-2% (F29's estimate) | yes with an exact conversion at the boundaries | S2 / S3 types (F17 or F29) |
| W6 | **A full stable sort where usually one or two picks are needed** | code `move_gen.h:1013-1036`; share inside the 10.5% unknown (08-25 profile, v1: sort 5.1%, before the current sort routine) | 1-3% (inference) | no as a picker; yes as F30 (one 64-bit key per move, about half the compares) | S3 picker, or F30 alone |
| W7 | **A 134 MB table read per quiet scored** (`cmh_get_bb`, `move_gen.h:997`): a probable cache miss each, at every miss node | code; `cache_management.h:200` | 1-3% (inference) | no (lazy scoring, or the piece×to key: −0.6 ± 12.6 Elo over 4,000 games, −6.1% nodes) | S3's own tables (owner 11-10) |
| W8 | **Threat sets built for a noisy-only list**: the only non-captures there are pawn moves, which the threat term never scores (`move_gen.h:893-903`, `:970-984`) | code | 0.1-0.3% | yes | independent, with W3 |
| W9 | **Cache slot = two cache misses and a copy**: slot, then the heap list; a full copy on every hit (`cache_management.h:925`) | code; `__memmove` 0.7% | under 1% | yes (list stored inside the slot) | independent (F4's long-term item), or moot if the cache becomes option C |
| W10 | **Hit-path rework**: scan for the first quiet, three linear searches, a second slot probe for the lazy re-score (`search_engine.cpp:10313-10314`), and a re-score of the whole tail when the last cutoff was late | code; inside `buildMoveListFromReordered` 0.9% | under 0.5% | yes | S3 |
| W11 | **State copies in the list builders and the per-move scorer** (`search_engine.cpp:10200`, `:10591`; `move_gen.h:857-873`; `search_v2.cpp:579`, `:587`) | code | about 1% (F28's estimate, overlapping) | yes | S2 (F28) |
| W12 | **Four forms of one list** (scratch vectors → filtered vectors → scored structs → `Move` vector) | code; direct emission measured 0-3%, unresolved (08-25) | under 1% on its own | yes | falls out of S3's fixed arrays (F4) |

Not waste, though it looks like it: the 16 passes (F7), and the per-move legality test (two operations; the references do
the same test, only later).

**Reading the list.** The result-identical items (W3, W4, W5, W8, W9, W11, part of W6) add up to a few percent, the size
of speed batches 7 and 8. W1 and W2 are the larger part and both are what a picker is for; neither can be had from the
generator alone.

## 6. Options for discussion ("best of all worlds")

Options, not decisions. Each says what it keeps of ours.

**A. A result-identical batch now, independent of S3.** W3 + W8 (qsearch exchange values and threat sets), the duplicate
check test inside `score_move` (the first half of W4), W9 if the cache is staying for now. Each has a natural verify mode.
Expected: 1-3% together, judged on real-play nps like batches 7-8. Keeps everything.

**B. Two-phase list building beside the cache: the smallest step toward a picker.** Generate and score captures; search the
good ones; only if none cuts, generate and score the quiets, merge them with the losing captures by the same stable sort,
and store the finished list in the cache as today. This keeps the capture tiers, the pass order as the tie-break, losing
captures among the quiets, killers after the captures, the cache and promote-to-front. It differs from today in exactly
two ways: quiet scores are read after the capture subtrees have run, and a quiet can no longer sort above a good capture
(today possible through the unbounded move-frequency bonus; how often is one counter, §7). The plan (§6.2b) says today's
order cannot be reproduced lazily; this is nearly so, and the two differences can be measured before any games. It
isolates "lazy quiets" from "a different ordering scheme", which keeps S3's 2×2 readable.

**C. Keep the idea of the move cache, drop its weight: a small refutation slot.** What is uniquely ours is not the stored
tail but the head: the position's own recent cutoff moves in most-recent-first order, more than the single move a TT
keeps. A slot of key + the first two to four moves (16 bytes with 16-bit moves, against 48 bytes + a heap list) keeps
that, costs one cache line, and feeds a picker's first stage: play the stored moves after a "fits this position" test,
then generate. Promote-to-front becomes a shift of four moves. It could also live inside a wider TT entry. Open
question: how much of the cache's value is in entries 2-4 rather than the head; one counter (cutoff index on cache hits)
answers it.

**D. The staged picker (S3's arm B) shaped to keep our parts.** Stages: stored refutations / hash move → good captures →
killers and counter (kept as stages, so protect-killers keeps its meaning; Stockfish 19 and Reckless removed them,
Ethereal and Berserk keep them) → quiets by history + threat term + move frequency → losing captures. Choices that are
ours to make, each a knob:
- losing captures after the quiets (all four references) or merged among them by score (ours today);
- the tie-break: our phase-dependent pass order as an explicit small key (F30's layout), or plain generation order;
- stop generating quiets once late-move pruning has started (W2), with the pruning tests moved ahead of `make_move`;
- the pre-search order at the root stays whichever way the 2×2 goes.
Needs first: a per-move exchange test (C3), a "fits this position" test for stored moves, an exclusion-safe list for
singular (C1), a common IIR trigger (C7), a legal-move count for mate and stalemate if generation becomes pseudo-legal.

**E. Legality: stay legal.** Our generator already produces legal moves with a two-operation test, and Reckless shows a
fully legal generator at the top of the lists. Pseudo-legal generation with a late test would save the test only for
moves never played, and the test is not where the time is. If the new generators are written for fixed arrays anyway,
pin masks applied at generation (Reckless's form) remove the filter pass and the double buffering (W12) for free.

**F. Per-position facts stored once (F16 / F27), and whether to store the opponent's attack set.** Checkers and blockers
are an easy yes: four readers today. The attack set is a real choice on which the references split (reference note §4.2):
storing it turns king-step legality and castling tests into one AND and gives the threat term its inputs for nothing,
but every node pays for it, including nodes that return on a TT hit or stand pat. We already build three quarters of it at
every list build for threat ordering (`move_gen.h:893-903`). A middle form: compute it lazily, at the first reader, and
keep it for the node. One detail if it is shared: legality wants it computed with our king removed, the ordering term
today has the king in place, and the two differ on squares behind the king.

**G. Move representation for v2: 16 bits (F17) or 32 bits with the piece (F29).** Three references use 16 bits and look
the piece up in a 64-entry board array, which we do not have; Berserk carries the piece in a 32-bit move and needs no
lookup. Without a board array, F29 removes more of our work (W5) than F17. Adding a board array to the state would also
do it, at 64 more bytes per copied state. Either way the en passant and capture flags end the repeated
`is_en_passant` / `is_capture` calls and fix C4 by construction.

**H. History tables for the picker.** The from×to continuation table makes each quiet score a likely cache miss (W7), and
every reference keys by piece and target square in a table of about a megabyte. The piece×to form is already built and
measured Elo-neutral in games; S3's separate tables (owner 11-10) are the place to start from it, with continuation at
plies 2, 4 and 6 as later knobs.

**Most promising, in my reading:** B (cheap, readable, keeps nearly everything), C (keeps the distinctive part of the
cache at a fraction of its cost), and W2 inside D (pruning before `make_move`, with the picker told to stop producing
quiets), with A as a warm-up that does not wait for anything.

## 7. Measurements that would settle the open sizes

None was run for this audit. Each is one probe or one profile read.

| question | measurement |
|---|---|
| How is `generateLegalMovesReordered`'s 10.5% split between scoring, sorting and the rest? (W1, W6, W7) | `perf annotate` of that function on the existing 11-11 perf data (the build has line info), as was done for `make_move` in plan §6.4 |
| How much of the 20% is qsearch and how much main search? | cachegrind with the two list builders as collection toggles, or the `MG_COUNT` counters (`search_engine.cpp:10432-10433`) extended to the qsearch site |
| What is the move-cache hit rate today, and how often does the TT hold a move when the cache misses? (finding 4) | `move_gen_visits` / `move_gen_cache_hits` already exist (`search_engine.cpp:10199`, `:10216`); add one counter for "TT entry with a move, cache miss" |
| How many generated moves are never examined, per site? (W1) | rerun the 08-25 consumption probe (generated, examined, per generating node) on the v2 path |
| How many moves are made and then pruned? (W2) | two counters in `score_move`: entries, and returns at `search_v2.cpp:697`, `:702`, `:706`, `:710`; by remaining depth |
| How many exchange evaluations come from the qsearch filter, and how many are for victim ≥ attacker? (W3) | `SEE_COUNT` exists (`cpp_bitboard.h:1969`); split by call site |
| How often does a dear capture inherit a cheap one's exchange value? (C3) | counter in the scorer: the square has a cheaper attacker of ours than this move's piece |
| How often does a quiet outscore a good capture? (option B) | counter after the sort: a non-capture ahead of a capture scoring ≥ 200000 |
| Where in the cached list do cutoffs come from on a hit? (option C) | histogram of the cutoff index on cache hits (`g_cutoff_histogram` exists; split by hit / miss) |
| Does C1 fire with singular on? | debug check: after the exclusion search, compare the buffer with a copy taken before it |

## 8. Notes that looked stale while reading

- "The move cache served 12% of list builds" (10-22) and "24%" (08-25) predate the gen 6 table sizes and noisy-only
  qsearch generation. The 07-15 speed note's "91.9%-hit moveGenCache" is the first-move cutoff rate, not a hit rate.
- Plan §4 lists "qsearch with checks"; at the defaults read here qsearch admits no quiet checks (C2).
- Plan §6.2b: "a losing capture scores near zero and sorts among the quiets" is given as a reason today's order cannot be
  built lazily. That part can (merge at the quiet stage, option B); the history timing is the part that cannot.
- The 10-22 ordering audit's line numbers are from gen 5; its killer-splice defect and its qsearch finding are both fixed.
- `owner-cpp-ideas.md` F5's text still describes nested vectors; the status table records the flat tables as built.
