# Fable question — root pre-search vs node-keyed TT (search architecture), 2026-07-28

## What we're asking

Our engine orders root moves with a **full-width shallow pre-search run every iteration**, and our
transposition table is **child-keyed and written by parents**, with node functions returning a bare `int`
(the best move never leaves the function). Four standard search features that assume the opposite
(node-keyed entries holding a best move) have all underperformed here.

We are considering a substantial rework and want an outside read before committing a day+ to it. **We are
not looking for validation** — if the pre-search is defensible, or if a hybrid is better than either
extreme, we would rather know now.

---

## Engine context

- **~2000 Elo HCE engine**, C++ behind a Cython entry point. Not NNUE.
- **Non-negamax**: separate `minimizer()` / `maximizer()` / `pre_minimizer()` rather than one negamax
  routine. Evaluation is absolute (Black-positive) with a single root flip by side to move.
- Present: PVS, iterative deepening, aspiration, TT, null-move, LMR, LMP, futility, razoring, qsearch,
  killer/history/counter-move, SEE capture ordering.
- **NPS ~497k** vs SF11 2.53M; static eval is **65-84% of per-node cost**.
- At equal depth 10 on full STS we score 54.9% vs SF11 65.5% / SF15-classical 73.3% / SF18 79.3%
  ⇒ ~18pp of **classical** headroom; NNUE is worth only 2.7pp over SF15-classical, so NNUE is not our gap.

## The architecture in question

**1. Root pre-search.** `reorder_legal_moves()` → `pre_minimizer()` runs a **full-width search of the root at
`depth_limit - 1`, every iteration**, purely to order root moves and generate the second-level move lists.
The code comments estimate its cost at ~1/EBF of the tree (~25%).

**2. Child-keyed TT.** `alpha_beta()` probes and stores using `updated_state` — the **child** position after
making a move. Parents store their children's scores.
- `minimizer()` (≈890 lines): **zero** `addToSearchEvalCache()` calls. It probes three times and stores
  nothing. Returns `lowest_score`.
- `maximizer()`: 2 stores. `pre_minimizer()`: 4 stores. `alpha_beta()`: 13 stores.

**3. No best-move return.** `minimizer`/`maximizer` return `int`. `TTEntry` HAS a `move` field, but it is
written at only two places (the beta-cutoff sites), both gated behind a default-off flag, and via a probe
(`accessSearchEvalCache`) for the node's OWN key — which on a first visit returns `nullptr`, because under
parent-stores-child the node's entry does not exist yet.

**4. Downstream consequence — `minimizer` also depends on the pre-search structurally**, not just
behaviourally: its signature takes `std::vector<int> second_level_preliminary_scores` and
`std::vector<Move> second_level_moves_list` as parameters.

---

## Measurements (WAC 300-position suite, fixed depth 10; base = 249 solved / 38,840,709 nodes / EBF 3.934)

| config | solves | nodes | EBF | note |
|---|---|---|---|---|
| base | 249 | 38,840,709 | 3.934 | FMC (first-move cutoff) = **87.55%** |
| `ENABLE_ROOT_PRESEARCH=0` | **57** | 57,598,769 (**+48%**) | 3.705 | ⚠️ CONFOUNDED — see below |
| `ROOT_PRESEARCH_REDUCTION=3` | 222 | 31,790,180 (−18%) | **3.562** | shallower pre-search |
| `ROOT_PRESEARCH_REDUCTION=5` | 198 | 33,404,303 (−14%) | **3.432** | shallower still |
| `LMR_EXTRA=2` | 240 | 31,970,295 (−17.7%) | 3.859 | pruning lever, for contrast |

⚠️ **The `=0` row is not a fair test of "no pre-search"** — the rest of the search consumes pre-search output
through `minimizer`'s parameters, so disabling it starves a system built around it rather than testing an
alternative design. We are treating that 57 as uninformative.

**Two things we think this shows:**
- **EBF is dominated by the pre-search, not by pruning.** Shrinking the pre-search moves EBF 3.934 → 3.432,
  while a comparable node cut from reduction depth (`LMR_EXTRA=2`, −17.7%) moves it only to 3.859.
- **The pre-search is load-bearing for ordering.** It is very likely *why* FMC is 87.55% — root moves are
  ordered by an actual shallow search rather than by heuristics.

## Features that underperform, and may share one root cause

| feature | observed | assumes |
|---|---|---|
| `ENABLE_TT_MOVE` | fires **36 times** across a 300-position bench; 243 solves (−6) | node-keyed entry with a stored best move |
| `ENABLE_IIR` | 247 WAC (−2) | a TT move's presence/absence as its trigger |
| `ENABLE_SINGULAR` | 246 WAC (−3) | a TT move to exclude and verify |
| `ENABLE_CORR_HIST` | wired at the RFP site only, on raw eval | a `staticEval` assignment point everything inherits |

We had recorded these as four independent negative results. We now suspect they may be one structural
result. **Is that reading correct, or are we pattern-matching?**

## What we already ruled out (so you needn't re-derive it)

- **Ordering is not our bottleneck at the margin.** A static placement-table tiebreaker for history-thin
  quiets fired on ~1 eligible quiet per node (40.8M eligible / 23.1M non-zero) and moved FMC 87.55% → 87.57%.
  A 33× weight sweep was flat. With FMC already at 87.55% there is ~12.5% total headroom.
- **EBF is insensitive to depth-local pruning** (long-standing finding, re-confirmed above).
- **Qsearch quiet checks**: we found qsearch had literally never searched one (a `bool turn` passed by value
  meant `is_check` tested the mover). Fixing it costs ~42 Elo at equal time (+27% nodes) — parked.
- **No corpus metric we own predicts Elo.** Games are our only instrument (SPRT via `selfplay/sprt.py`).

---

## Questions

1. **Is the child-keyed TT + no-best-move-return the root cause** of the four features underperforming, or
   are there other explanations we should test first?
2. **Is a full-width root pre-search ever justified in a modern engine**, or is it always dominated by
   TT-move + IID/IIR? Note ours is not IID — it runs unconditionally, every iteration, full width.
3. **Is there a hybrid worth keeping?** e.g. a much shallower or first-iteration-only pre-search, or one
   retained only at the root while node-keyed TT ordering serves everywhere else. We are open to the answer
   being "keep a reduced form" — the dial (`ROOT_PRESEARCH_REDUCTION`) already exists.
4. **Does the non-negamax split force any of this?** Is child-keyed storage a natural consequence of
   separate minimizer/maximizer, or incidental and fixable without unifying to negamax?
5. **What is the minimal change** that would make TT-move / IIR / singular testable — best-move return plus
   node-keyed stores, or is more required (e.g. proper bound flags, exclusion handling)?
6. **What EBF should we expect** after such a rework, given FMC ~87.55%? Is our 3.934 even comparable to the
   ~2 quoted for strong engines, or are we measuring differently (ours may fold in aspiration re-searches
   and iterative-deepening overhead)?
7. **What breaks** when converting parent-stores-child to node-keyed storage? We expect trouble around
   aspiration re-search bound flags (`HONEST_ROOT_TT` exists because root stores hardcoded `EXACT`) and
   around the singular-exclusion path.
8. **Migration order**: what sequence keeps the engine testable at each step?

## Constraints on any recommendation

- Validation is **games only**; a 456-game tournament gives ±37 Elo, so we cannot resolve small effects
  cheaply. Byte-identity will not survive this change, so our usual regression guard is unavailable.
- We can build two `.so` variants and A/B them directly (`tournament.py --p1-engine-dir/--p2-engine-dir`),
  most likely via a git worktree.
- Rollback point committed at `da76e2f` (all current scaffolding gated default-off, byte-identical).

## A note on the pre-search

The root pre-search is an original design of this engine's author, not a port. It is **load-bearing and
demonstrably effective** — it is the most likely reason FMC sits at 87.55%, and removing it naively
collapses the engine. If the answer is that a properly-implemented node-keyed TT dominates it, we will
retire it as a technique that did its job and was outgrown, and record it as such. We are asking whether
that is true, not looking for permission to delete it.
