# S3 move source: design concept (2026-11-11, for owner review)

Search v2's stage S3 (plan §3), with S2 folded in. Inputs: `MOVEGEN-AUDIT-2026-11-11.md` (our pipeline, the waste list,
options A-H) and `../reference/movegen-giants-2026-11-11.md` (SF19, SF11, Reckless, Ethereal, Berserk; the §5 checklist).
Goal (owner 11-11): a technically strong move source that is cache friendly and fast, built so that the open choices,
where the strong engines disagree and where ours differs from all of them, are switches that can be compared on cycles,
time and ordering quality before games decide.

## 1. What is being decided, and what is only being compared

**Fixed by the evidence (all five references agree, and nothing in ours argues against it):**
- check and pin facts computed once per position and stored for the node;
- captures and quiet moves generated in separate calls, into fixed arrays, with no heap traffic;
- the stored move(s) tried before anything is generated, after a test that they fit the position;
- quiet moves generated only when the stored moves and the captures fail to cut, and never once late-move pruning has
  started;
- no full sort of the list;
- a per-move exchange test with a threshold (it also ends the per-square mis-valuation, audit C3).

**Compared as arms, because the engines disagree or ours is unique** (§5): how much of a position's move knowledge is
remembered (today's full cached list, a small refutation slot, a TT move, nothing); killers and counter as a stage or not;
losing captures after the quiets or merged among them (ours); quiet selection by partial sort or by scan; history keys.

## 2. Data, laid out for the cache

| item | form | size | why |
|---|---|---|---|
| move in a list | 32-bit: from 6, to 6, moving piece 3, captured piece 3, promotion 3, kind 3 (normal, double push, en passant, castle) | 4 B | scoring, history indexing, SEE and make never look a piece up (our `BoardState` has no square-to-piece array: audit W5, option G). Berserk's form |
| move in the TT / refutation memory | 16-bit: from, to, promotion | 2 B | half the space; expanded and checked by the fits test on use |
| scored move | move + 32-bit score | 8 B | the giants' form; a list of 256 is 2 KB |
| per-ply list | one fixed array of 256 scored moves per ply, beside the v2 `Stack`, not in the call frame | 2 KB / ply | contiguous, reused, no allocation; keeps search frames small (plan §6.3 target < 1 KB per frame) |
| per-node facts (in the `Stack` entry) | checkers, our pinned pieces (blockers), the pinners, check squares per piece type, enemy attack set (lazy, at its first reader) | ~100 B | four readers recompute checkers today (W4); "gives check" before making the move; the attack set is where the references split, so lazy (option F) |
| histories (v2's own, owner 11-10) | butterfly [side][from][to] int16 (8 KB); continuation via a per-ply pointer to a [piece][to] block (SF's form), int16; capture history [piece][to][captured] | the continuation block a node reads is 1.5 KB | today a quiet's continuation read is a probable cache miss into a 134 MB table (W7); with the per-ply pointer, every quiet at a node reads the same small block |

## 3. Generation

Legal by construction, as today (our generator is already legal and sound, audit §2.1; Reckless shows a legal generator at
the top of the lists; option E): pin masks applied at generation from the stored facts, check target masks in single
check, king only in double check, the king's destinations masked by the enemy attack set computed with our king removed.
Two entry points, `gen_noisy` (captures, queen promotions, en passant) and `gen_quiet` (the rest, including
under-promotions), each appending to the per-ply array. An empty list after both proves mate or stalemate, as today.
Verification before anything uses it: the union of the two calls equals today's legal list as a set, on every position of
a large set (WAC, STS, the game sets, random legal positions, perft trees), and perft node counts are unchanged.

## 4. The picker (one object per node, stages)

1. **Remembered moves** (arm, §5): play each after the fits test, with duplicates suppressed later.
2. **Good captures:** `gen_noisy`, score = victim value × k + capture history (+ promotion bonus); pick by scan for the
   maximum; a capture failing `see_ge(move, threshold)` is swapped aside for stage 5. The threshold: flat 0 (Ethereal), or
   loosened by the score (SF, Berserk, Reckless), a knob.
3. **Killers and counter** (arm): played after the fits test, without generating, if this arm is on.
4. **Quiets** (skipped when the search says stop): `gen_quiet`, score = butterfly + continuation (plies as knobs) + our threat
   term + our move frequency (+ check bonus, knob); selection by SF's depth-scaled partial insertion sort, or by scan
   (arm).
5. **Losing captures:** after the quiets (all references), or merged among the quiets by score (ours today) (arm).

Modes from the same object: qsearch (stages 1-2, with today's admission rule: promotions and captures with exchange ≥ 0),
ProbCut (stage 2 with a threshold), evasions (all legal evasions: captures of the checker by victim, then the rest by
history). "Does this move give check" comes from the stored check squares and blockers before the move is made, so
late-move pruning, check extension and LMR eligibility are decided before `make_move` (audit W2).

## 5. The comparisons (owner 11-11: "compare different methods to see if we save cycles or time")

**Remembered move knowledge**: the two questions the owner raised, made measurable:
- R0: none (generate every time);
- R1: the TT move only (16-bit in the TT entry);
- R2: a refutation slot, the position's last 2 or 4 cutoff moves, most recent first (16 B), our promote-to-front idea at
  a fraction of the cost (option C);
- R3: today's full cached legal list, reordered at cutoffs (ours today), fed to the picker as stage 1 plus a "skip
  generation" path.
This answers "does caching save more time than its ordering costs" (R3 vs R0 / R1) and "is updating a stored list better
than a TT move" (R2 / R3 vs R1).

**Other arms:** killers and counter as a stage (on / off); losing captures (after / merged); quiet selection (partial
sort / scan); continuation plies (1 / 1-2 / 1, 2, 4, 6); butterfly key (from×to / piece×to).

**How each arm is read, cheapest first:**
1. Cost: instructions per node (cachegrind, load-insensitive) and cycles per node with cache misses (perf), on the
   24-position game set at depth 12.
2. Ordering: first-move cutoff rate, mean cutoff index, nodes to depth (`_ebf_per_iter.py`, its per-iteration form).
3. Time to depth on the same positions (a quiet box, several runs).
4. Games at equal time for the survivors, then the plan's 2×2 against the root ordering (pre-search vs the previous
   iteration's scores).
A cheaper move source that orders worse can lose on (2) what it wins on (1); only (3) and (4) combine them, which is why all
are recorded.

## 6. Build order (each step checked before the next)

0. **Measure today first** (no behaviour change; the audit's §7 counters on the v2 path): cache hit rate and how often the TT
   holds a move on a miss; cutoff index on cache hits; how often a quiet outscores a good capture; moves made only to be
   pruned; how often the per-square exchange value misleads. This sizes the arms before they are built.
1. Infrastructure in `search_v2`: the move type, per-ply arrays, per-node facts, `see_ge`, the fits test, v2's history
   tables (first with today's formulas, so the first arm differs from today only in the move source).
2. The new generator, verified for set equality and perft against today's.
3. The picker with R3 + today's scoring as its first configuration (closest to today), then the other arms.
4. The reads of §5, then games, then the 2×2.

## 6b. Step 0 results (11-11 evening; probe build `-DSV2_MOVE_PROBE`, `sv2_probe.h`, bench unchanged 247 / 39,861,964)

Search v2 path (`SEARCH_ARM=1`). WAC = the 300-position bench at depth 10; G24 = 24 game positions (`game_regret_set`, every
625th row) at depth 12. Output `E:\chess_data\s3_step0\`.

| measure | WAC d10 | G24 d12 | reading |
|---|---|---|---|
| interior move lists built | 3,209,184 | 1,476,855 | |
| served by the move cache | **43.1%** | **39.3%** | |
| first-move cutoffs, cache hit | **94.8%** | **94.2%** | of nodes that cut |
| first-move cutoffs, cache miss | **82.5%** | **81.6%** | (first visits mostly: different node populations) |
| cutoffs at index 1-4 on a hit | 4.2% | 4.8% | the cached tail's share (R2's question) |
| a remembered move exists at a cache miss (TT-sized 2^22 table) | **0.12%** | **0.45%** | misses are first visits, not evictions |
| on a hit, the remembered move = the cached head | 98.5% | 99.2% | |
| moves made, then pruned at once | **63%** (22.7 M of 36.1 M) | **56%** (7.8 M of 14.0 M) | LMP 87% of them, futility 9%, losing-capture SEE 4% |
| exchange-branch captures where a cheaper piece of ours also attacks the square (C3) | **17.3%** | **14.9%** | the per-square value describes another capture |
| qsearch lists | 4.31 M, 1.39 moves each | 1.83 M, 1.14 each | |

What it says for the arms:
- **R1 (TT move) as an addition to the cache: almost never fires** (a remembered move exists on 0.1-0.5% of misses). As a
  *replacement*, it would keep nearly all of the hit nodes' first move (= the cached head 98.5-99.2%), lose the cached tail
  (~4-5% of cutoffs on hits) and regenerate lists on revisits: a cost / ordering trade for the §5 reads.
- **W2 is the largest single item**: over half of all made moves are pruned immediately. Built first, result-identical:
  `SV2_PRUNE_EARLY` (`sv2::lmp_before_make`), verified on 19,756,269 WAC prunes with 0 mismatches; queue #208 runs SW1 / SW2
  and the instruction count.
- **C3 is frequent** (15-17% of exchange-branch captures): the per-move exchange test is worth having early.
- **A quiet never sorts above a good-tier capture** (score ≥ 200000): 0 of 2,566,032 full lists (WAC d10), 0 of 1,153,459
  (G24 d12). So option B (captures first; quiets generated only if no capture cuts, merged with the losing captures by the same
  stable sort) reproduces today's order exactly but for ONE thing: the quiets' history scores are read after the good
  captures' subtrees have run. That makes B the cleanest first picker arm: its only difference from today is the timing of
  the history read.
- Not measured yet: the main vs qsearch split of generation time.

**Built 11-11 evening (both knobs, default off, byte-identical at 0):**
- `SV2_PRUNE_EARLY` (search v2): late-move pruning decided before `make_move` (`sv2::lmp_before_make`): today's tests in
  today's order, the child's key / castling rights / en passant square / halfmove clock built from the move, the repetition
  test with the child pushed on the path as `make_move` pushes it (`is_repetition_key`, refactored out of `is_repetition`),
  the TT cut test without its side effect, `moveGivesCheckFast` for the check test; castling, near-fifty-move children and the
  passer knobs go the normal way. Verify build (`-DSV2_PRUNE_VERIFY`): 19,756,269 early prunes on the WAC bench, **0
  mismatches**, bench 247 / 39,861,964. Queue #208 (11-11 20:28): **SW1 all 6 sets SAME, SW2 12 of 12 games SAME**;
  cachegrind on #204's 12 positions, the same 3,179,520 nodes: **−3.16% instructions** (17,829,490,063 → 17,266,969,539;
  registered −3 to −8%, at its low end: the pre-make tests still need the child key, the repetition test, the TT probe and a
  check test). Real-play speed with PGO: queue #209.
- `SEE_PER_MOVE` (bitmask; 1 = capture ordering, 2 = the qsearch filter, 4 = v2's losing-capture prune): `see_move`, the
  exchange value of the capture itself (see_impl's incremental path with the first capture forced), in `move_gen.h`. Verify
  build (`-DSV2_SEE_VERIFY`): where the mover is the side's least valuable attacker, `see_move == see` on 9,779,141 checks, **0
  mismatches**. Fingerprints (WAC d10): 1 = 248 / 38,680,259; 3 = 249 / 37,279,382; 7 (arm 1) = 247 / 39,685,447. Games: queue
  #209 (`SEE_PER_MOVE=3` vs 0, 2,000 games, rule registered in its header).

**Queue #209 results (11-12 02:20-07:56; PGO `sv2_pgo1pe` trained with `SEARCH_ARM=1 SV2_PRUNE_EARLY=1`; gates: default and
arm 1 + early both 247 / 39,861,964, `SEE_PER_MOVE=3` 249 / 37,279,382, as expected):**
- **Early pruning, real play: level, not faster.** `_ebf_per_iter.py` (the #207 set, 40 per phase, seed 1, d13), `sv2_pgo1`
  (S1) vs `sv2_pgo1pe` with `SV2_PRUNE_EARLY=1`, alternated 3×: same tree True (119 positions at d13 in every run); arm 1 −
  arm 0 = +0.56%, −1.98%, +1.43%, mean **+0.01% (sd 1.77)**. Registered prediction −3 to −8% ⇒ **MISSED**. The −3.16%
  instructions (#208) did not become time. Not explained; candidates, none tested: the saved work (make / unmake of a move
  that is then pruned) is cheap, cache-warm, well-predicted code, while the costly part, the child's TT probe, is done on
  both paths; the two binaries also differ in PGO profile and layout (sv2_pgo1 from #207's sources, sv2_pgo1pe from today's).
  A cycles-and-cache-misses read (perf, identical trees) would separate these. **The knob stays (result-identical, default
  off); it is not counted as a speed win.** Lesson for the S3 reads: instructions alone over-promise; the §5 order (cost →
  ordering → time) needs the cycles read before time is predicted.
- **`SEE_PER_MOVE=3` vs 0, 2,000 games** (10+0.1, UHO, seed 209, both `SEARCH_ARM=1 SV2_PRUNE_EARLY=1`): **+7.3 ± 13.7 Elo**
  (nElo +8.1 ± 15.2; 801 W / 759 L / 440 D; ptnml [154, 158, 346, 176, 166]); 0 warnings, 0 disconnects / stalls, 0 time
  forfeits; nps 1,642k vs 1,632k, mean depth 11.6 both. About +1.05σ: inside ±2σ. Prediction +0 to +8 ⇒ inside. **Not the
  verdict**: registered before any game, the verdict is read on #209 + #210 combined (queue #210, 800 more games, seed 210).

## 7. What stays ours

The threat-aware quiet term, move frequency, the killer protection in LMR, the root pre-search (until its own 2×2), and the
remembered-move idea (R2 / R3) if the comparison keeps it. Where the measurement says ours is better, it stays; where it
says otherwise, the switch records why.
