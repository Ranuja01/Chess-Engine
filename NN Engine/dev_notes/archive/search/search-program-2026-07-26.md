# Search program — options for 2026-07-26 (research-grounded, costed)

Written overnight 2026-07-25 after the +83 Elo eval-arc tournament. Every claim below is either a MEASUREMENT
from this session or a CITATION from local engine source (`stockfish_11/`, `stockfish_17/`). Nothing here is
recalled-from-memory design.

## Where we actually stand

| fact | evidence |
|---|---|
| Eval arc = **+83.4 ±32.1 Elo** | 620g tournament, equal clock, colour-symmetric |
| We are **10.6pp below SF11**, **18.4pp below SF15-classical** at equal depth 10 | `sf_bench_ceiling.py` |
| **NNUE is worth only 2.7pp** over SF15-classical at d10 | same |
| EBF stuck ~3.667; **invariant** to every depth-local prune (26% node cut ⇒ −0.063) | 11 configs this session |
| Better eval did **NOT** improve ordering, EBF, or reduction safety | old-vs-new eval A/B, search held fixed |

**The synthesis:** eval accuracy buys *permission* to prune, but permission only becomes depth if the search has
machinery that CONVERTS it. SF has that machinery; we mostly don't. That is the gap, and it is buildable.

---

## Option A — Reduction shape: absolute target → remaining-depth (HIGHEST EBF LEVERAGE)

**Our code.** `reduced_search_depth()` (`search_engine.cpp`):
```cpp
int base = Config::ACTIVE->DEPTH_REDUCTION[depth_limit];   // indexed by ITERATION depth
int r = static_cast<int>(base - (move_factor / scale)) - Config::LMR_EXTRA;
return std::max(r, 2);                                     // an ABSOLUTE target depth
```
`cur_depth` is never read (outside the rare `is_in_relavent_pin` branch). `DEPTH_REDUCTION[10] = 8`, so at d10 the
base reduction is 2 plies for EVERY node in the iteration — a node at cur_depth 2 and one at cur_depth 7 are
reduced to the same absolute depth. Null-move does the same (`search_engine.cpp` ~4194).

**SF11** (`stockfish_11/src/search.cpp:74-78, 197, 1126`):
```cpp
Reductions[i] = int((24.8 + std::log(Threads.size()) / 2) * std::log(i));
Depth reduction(bool i, Depth d, int mn) {
    int r = Reductions[d] * Reductions[mn];
    return (r + 511) / 1024 + (!i && r > 1007);
}
Depth r = reduction(improving, depth, moveCount);   // then newDepth - r
```
`d` = **remaining depth at that node**; `r` is SUBTRACTED from it. Reduction ≈ `log(remaining) × log(moveNumber)`,
applied per-node, everywhere in the tree. Note `improving` (an eval-derived flag) feeds it directly.

**Why this explains our three symptoms at once:**
1. **EBF can't move** — a front-loaded reduction doesn't compound per-ply, and EBF is a per-ply ratio.
2. **`LMR_EXTRA` blunders** (−9 solves at 1) — lowering an ABSOLUTE target guts the SHALLOW region where lines
   resolve, instead of trimming the deep tail.
3. **"Ordering-limited" (2026-07-14) may be a misdiagnosis** — we may have concluded ordering can't support harder
   reduction when the reduction was simply applied in the wrong SHAPE.

**Proposal.** `ENABLE_LMR_RELDEPTH` (default off ⇒ byte-id 248/44,038,704): `r = cur_depth + f(rem, moveNumber)`
with `f` a log×log table mirroring SF11, plus `LMR_RELDEPTH_DIV` to scale. Keep the existing path as the else-branch
(same pattern as `ENABLE_STATSCORE_LMR`).
**Judge on equal-accuracy-at-fewer-nodes, NOT on EBF alone** — this session proved EBF can fall for bad reasons
(an over-optimistic eval buys wrong cutoffs; de-king legitimately ADDED 10% nodes).
**Risk:** highest of the three. It rewrites the hottest path in the search and changes every reduced-node depth.
**Cost:** ~half a day to build + byte-id, then a games run. This is a campaign, not an afternoon.

**✅ VERIFIED END-TO-END (2026-07-25).** `search_engine.cpp:2683` passes `reduced_depth` as the CHILD's
`depth_limit`: `maximizer(cur_depth + 1, reduced_depth, alpha, alpha + 1, ...)`. Corroborated at :2671, which must
compute remaining depth as `reduced_depth - cur_depth` for the TT-depth test. So the effective reduction
(`depth_limit - reduced_depth`) is **CONSTANT across `cur_depth`** for a given move number, and a node at
`cur_depth >= reduced_depth - 1` has nothing left to reduce. **The hypothesis is now a confirmed defect, not a
guess** — Option A no longer needs a verification step before building.

**★ Bonus finding:** `ENABLE_IMPROVING` already IS the SF eval→reduction wire (`:2654-2660`,
`reduced_depth = clamp(reduced_depth - IMPROVING_REDUCTION, 2, depth_limit)`) — but it nudges the ABSOLUTE target,
and measured **−7 solves** (241 vs 248) this session. Same shape defect. ⇒ **fixing the shape may rescue
`improving` for free**, which would be independent corroboration that the shape (not the idea) was the problem.

---

## Option B — Correction history, applied the way SF actually applies it (BEST FIT TO OUR SITUATION)

**SF17** (`stockfish_17/src/search.cpp:85-152, 792, 812, 822, 1581, 1593`):
```cpp
int correction_value(...) {
    pcv   = pawnCorrectionHistory[pawn_structure_index<Correction>(pos)][us];      // PAWN STRUCTURE
    micv  = minorPieceCorrectionHistory[minor_piece_index(pos)][us];
    wnpcv/bnpcv = nonPawnCorrectionHistory[non_pawn_index<COLOR>(pos)][COLOR][us];
    cntcv = (*(ss-2)->continuationCorrectionHistory)[piece_on(m.to_sq())][m.to_sq()];
    return 7685*pcv + 7495*micv + 9144*(wnpcv+bnpcv) + 6469*cntcv;
}
Value to_corrected_static_eval(v, cv) { return clamp(v + cv / 131072, ...); }
```
Applied **where `ss->staticEval` is assigned** — search Step 6 (:812, :822) and qsearch Step 4 (:1593, :1606) —
so RFP, null-move, futility, LMP, `improving` and stand-pat ALL inherit it automatically.
**The TT stores the UNADJUSTED eval** (:825 comment) ⇒ the correction is re-derived per visit, never compounded.
Update (`update_correction_history`, :132) writes a bonus from the search-vs-static difference into all five tables.

**Why this is the best fit for us right now:**
- **Our previous attempt failed in exactly the two ways this shows are wrong:** applied at RFP sites ONLY (not at
  the staticEval assignment), and on raw `evalCacheNew` (risking compounding). Mis-integration, not a bad idea.
- **It is the missing eval→search converter.** This session proved a +83 Elo eval produced zero search benefit
  because nothing reads the eval. Corrhist is precisely the wire.
- **★ The pawn-structure table is the user's passer/pawn-hash intuition, and WE ALREADY HAVE THE KEY.**
  `generatePawnKey` (`cache_management.h:521`) survived the pawn-hash cull. That lane was closed on PURITY (our
  pawn terms read king position + non-pawn blockers ⇒ a cached VALUE would be wrong). **Correction history has no
  purity requirement** — it is a learned adjustment, not a cached value. The objection that closed the lane does
  not apply. This is a genuine re-opening, not a retry.
- **Contradicts nothing.** The old "load-bearing optimism is FUNDAMENTAL" verdict was already broken by de-king
  (a big over-read removed, STS +92). And corrhist is TARGETED (per-structure signed correction), not the UNIFORM
  global shrink that flattens the landscape ([[corpus-fit-flattens-eval]]).

**Proposal.** `ENABLE_CORRHIST` (default off ⇒ byte-id): start with the pawn-structure table alone, keyed on the
existing pawn key, applied at our staticEval assignment sites in min/max/qsearch, TT storing the raw eval.
**Risk:** low-to-moderate. Additive, gated, and the failure mode is a wrong nudge, not a structural break.
**Cost:** ~half a day for the single-table version.

---

## Option C — Continuation-history re-key from×to → piece×to (LOWEST RISK)

The 2026-07-14 prescription that was **never executed** (we pivoted to the eval lane instead — correctly, it paid
+83 Elo, but EBF was never going to move as a result). SF keys continuation history by **piece×to**; ours uses
from×to (`cont_ctx_key`/`cont_ent_key`). from×to conflates different pieces making the same geometric move.

**Why it's attractive:** ordering changes cannot hurt the mean — they only reorder the search — so there is no
load-bearing-optimism trap. It is the cheapest of the three.
**Why I rank it third despite that:** our first-move cutoff is already **86.3%**, and the historical instrument
reads 90–93% (`g_fh_first`, a DIFFERENT counter from the `cutoff_histogram` I divided — do NOT treat 86.3 vs 92.1
as a regression until measured with the same instrument). Either way the remaining ordering headroom is a few
points, and Option A says reduction SHAPE — not ordering quality — is what blocks EBF.
**Cost:** ~2-3 hours. **Risk:** low.

---

## Recommendation

**B first, then A, with C as filler.** Reasoning:
- **B** directly repairs the defect this session actually demonstrated (eval gains cannot reach the search), is
  the lowest-risk of the two structural items, re-opens the pawn-hash lane on legitimate grounds, and lands on the
  passer/pawn-structure thread the user independently proposed.
- **A** has the highest EBF leverage and the best root-cause story, but it is the riskiest change in the codebase
  and its hypothesis is still UNVERIFIED end-to-end (I traced `r` through `reduced_search_depth` and both call
  sites, but not into the child's search call — **verify that first, it is 20 minutes**).
- **C** is cheap and safe but targets the smallest measured gap.

**Do NOT bundle.** Each is gated default-off, byte-id verified at 248/44,038,704, and judged separately —
`RAZOR_FLOOR=800` this session was a reminder of what happens when a swept number is trusted without a
neighbourhood check.

## Also open (eval side)
- **Passer V3 + `PASSER_RFLOOR_R5/R6`** — categorical game WIN on the OLD baseline, bench-negative on the new one
  ⇒ probably overlaps de-king. Needs a 3-seed re-judge on the current baseline. (`vs_sf` seed threading still
  needs verifying before that run — it is not a positional arg the way `gauntlet`'s is.)
- **Safe-check table** — dead as ported (−14 WAC / −166 STS unconfounded), but the PRINCIPLE is sound: SF15 really
  does weight checks per-type. Suspected cause is scale mismatch (SF's weights live in a quadratic king-danger
  formula; ours is a linear knee-12 budget). A correct port would re-derive the weights in our units, not transplant.
- **Capgains realizability** — still the identity function; needs a different signal, not opened knobs.
