# Search architecture — Fable review CORRECTS our analysis, 2026-07-28

Independent review of `fable-question-search-architecture-2026-07-28.md`. **Our central claim was wrong.**
Read this before acting on anything in that document or in the 07-28 handoff's architecture section.

## ✅ What was verified TRUE
- **No store call site passes a move.** `addToSearchEvalCache`'s `Move move = Move()` parameter is inert at
  every call site; `TTEntry::move` is written only at the two in-node beta-cutoff sites, gated
  `ENABLE_SINGULAR || ENABLE_TT_MOVE`, via a probe of the node's OWN key — which exists only on a revisit.
  ⇒ move-field population needs: searched once → revisited → revisit reaches a cutoff. 36 firings/300
  positions is the predictable result.
- **The root pre-search runs every iteration** at `max(2, depth_limit - ROOT_PRESEARCH_REDUCTION)`.
  **Plus a cost we missed:** every aspiration widening and the full-window fallback RE-CALL `alpha_beta`,
  so a fail-high/fail-low iteration pays the pre-search two or three times.

## ☠️ What was WRONG

**1. "`minimizer` contains ZERO stores; min nodes never store" — FALSE.**
`minimizer` delegates every child search to **`get_score_for_minimizer()` (`search_engine.cpp:2564`)**, which
holds the probe (`:2598`) and **seven stores** (2640, 2822, 2827, 2863, 2868, 2909). Symmetrically
`get_score_for_maximizer()` (`:2933`) holds a probe and seven stores. **Every interior node, min and max,
gets a TT entry and probes before searching a child.** The TT is neither starved nor half-empty.
⇒ **Root-cause error:** we inferred function bodies from the line ranges BETWEEN function *declarations*,
never noticing the helper functions in between, and attributed their stores to `alpha_beta`.

**2. "Child-keyed rather than node-keyed" — mechanically true, semantically misleading.**
Stores happen in the parent's frame using the post-make_move zobrist, but the entry is **keyed by the
position it scores**. In content the table is position-keyed and correct; score reuse works today.
The real cost is exactly two things: (a) the child's best move is out of scope at the store site, so
`TTEntry::move` cannot be filled there; (b) a node's entry exists only after it returns, so during its own
first execution a self-probe is `nullptr`.

**3. ★ WE ALREADY HAVE A POSITION-KEYED HASH MOVE — and missed it.**
`updateMoveCacheForBetaCutoff` (`cache_management.h:1033`) runs **UNGATED** at every cutoff with `i != 0`
(`search_engine.cpp:4089` + the maximizer twin) and promotes the cutoff move to the front of the
moveGenCache entry for that position. Every revisit searches the previous cutoff move first. **This is
functionally a TT move for ordering**, is a co-author of the 87.55% FMC, and predicts correctly that
`ENABLE_TT_MOVE` adds nothing even when it fires — it is REDUNDANT, not broken.
⚠️ The memory `tt-move-vs-cutoff-promotion` ALREADY said this ("our de-facto hash move is the moveGenCache
cutoff-move promotion"). It was read during this session and its significance still missed.

**4. "One root cause explains four features" — 2 of 4, and the weaker 2.**
| feature | verdict |
|---|---|
| `ENABLE_SINGULAR` | ✅ genuinely starved — its probe needs a node entry WITH a move at node entry |
| `ENABLE_TT_MOVE` | ⚠️ firing count explained; the −6 solves is REDUNDANCY (see #3), not brokenness |
| `ENABLE_IIR` | ❌ unrelated — its trigger is a **moveGenCache miss** (`:3826-3828`), not TT-move absence |
| `ENABLE_CORR_HIST` | ❌ unrelated — a staticEval-plumbing question |

**5. ★ THE EBF METRIC IS BROKEN — every EBF conclusion this session is void.**
Printed EBF is `pow(num_iterations, 1.0 / depth_limit)` (`:2119`), where `num_iterations` is **cumulative
across all ID iterations, all aspiration attempts, the pre-search, qsearch, and TT-hit bookkeeping**.
Literature "~2" is per-iteration `nodes(d)/nodes(d-1)`. On our formula SF11 at d10 would print ~2.9-3.2 —
**the "3.9 vs 2" gap was largely an artifact.** The rows are also mutually inconsistent: an 18% node cut
"moved" EBF 3.934→3.562 (implying ~63% as a 10th root) while `LMR_EXTRA=2`'s 17.7% cut gave 3.859, exactly
the arithmetic prediction. ⇒ **"EBF is dominated by the pre-search" rests on a broken instrument.**
Do not use this EBF for decisions. For a real one, log per-iteration node counts and take the ratio.

**6. The `ENABLE_ROOT_PRESEARCH=0` collapse is a DIAGNOSABLE ARTIFACT, not evidence.**
The fallback fills unrazored/first-iteration root scores with `top_score = 0` (`:4913`); root razoring then
computes `alpha - 0 > threshold` and **breaks out of the ENTIRE root loop** (`:2406`) whenever alpha is
large — constant on a tactical suite. The 57 measures that, not the pre-search's absence. A fair "no
pre-search" A/B must first fix the fallback's razor interaction.

**7. ★ The pre-search is comparatively EFFICIENT — the opposite of our framing.**
At equal node cost (~−18%): `LMR_EXTRA=2` kept **240** solves, `ROOT_PRESEARCH_REDUCTION=3` kept **222**.
The pruning lever DOMINATES shrinking the pre-search on solves-per-node.

## Answers to the questions that matter
- **Q4 — non-negamax does NOT force child-keyed storage.** Incidental and fixable in place. Absolute
  Black-positive scoring makes a shared TT *simpler* than negamax (no sign flip; `use_tt_entry` already
  ignores its `is_maximizing` arg for bounds). The split costs code duplication, not TT semantics.
  **Do NOT unify to negamax** — maximal risk, zero measured payoff.
- **Q2/Q3 — verdict on the pre-search: "reduced form, not removal — and not urgently."** As permanent
  every-iteration furniture it is non-standard (the previous ID iteration already searched every root move
  at d−1, and `alpha_beta` retains those scores). Its unique contributions are freshness for scout-refuted
  moves plus two side products: root-razoring reference scores and the ordered ply-1 lists `minimizer`
  consumes at `cur_depth==1`. **Natural hybrid: pre-search on iteration 1 only, previous-iteration scores
  thereafter, keeping ply-1 list generation.**
- **Q7 — the migration trap is the DEPTH OFF-BY-ONE.** All stores/probes use parent-frame remaining
  (child-remaining + 1). A self-store using the node's own frame is 1 less and would **silently** never
  satisfy existing probes. Normalize first. Mate scores (|score| ≥ 9M rejected) and root bound flags
  (`HONEST_ROOT_TT`, `ENABLE_TT_DEPTH_FIX`, both default true) are already safe.

## Recommended path (half a day, not a day)
1. **Normalize the depth convention** (shift store and probe together) — provably behaviour-identical, so
   byte-identity survives. Verify, commit.
2. **Gated node-exit self-stores with the move** (`ENABLE_NODE_TT`, default off), suppressed under
   `excluding`. Fire counter. **No signature plumbing needed** — the move only has to reach the node's own
   store, which is in the same frame.
3. **Retest `ENABLE_SINGULAR` (and TT-move) in games** — the actual experiment this enables.
4. Later/optional: remove parent-side child stores (gated, expect ~0); pre-search first-iteration-only knob,
   SPRT'd separately so the two never confound; corrhist independent of all of it.
⇒ **Then spend the remaining time on EVAL.** The 18pp equal-depth gap is evaluation, and the best search
result this month was a soundness fix (qdelta), not plumbing.

## What would falsify this assessment
- Node-exit self-stores produce many first-visit TT-move promotions **and** SPRT shows a clear gain ⇒ the
  redundancy argument (moveGenCache ≈ TT move) is weaker than claimed.
- A corrected per-iteration EBF shows growth far above ~3 while reference engines show ~2 ⇒ the
  pre-search/ordering story reopens.
- Singular SPRTs clearly positive after step 2 ⇒ "one root cause" earns partial credit and deeper TT work
  (always-store policies, aging) moves up the queue.
