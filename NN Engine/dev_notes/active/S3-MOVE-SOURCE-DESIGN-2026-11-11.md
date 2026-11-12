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

## 7. What stays ours

The threat-aware quiet term, move frequency, the killer protection in LMR, the root pre-search (until its own 2×2), and the
remembered-move idea (R2 / R3) if the comparison keeps it. Where the measurement says ours is better, it stays; where it
says otherwise, the switch records why.
