# Fable question — most EFFICIENT way to holistically rebalance our HCE eval toward SF11 (2026-07-05)

(Draft for the user to relay. Verified-facts-first; we source-verify every claim Fable makes before acting.)

## Who we are (verified)
Non-negamax C++ hand-crafted-eval engine (~2700), separate minimizer/maximizer, absolute eval Black-positive,
driven via a Cython entry point, run under WSL. Full self-play A/B + SPRT infra (lightning/blitz presets, SF18
arbiter), fixed-node gate (`node_ab`, calibrated), deterministic themed move-match (`movematch`), an equal-depth
vs-SF11 harness (`vs_sf11`), an outcome-incremental-validity screen, and a static outcome-Texel fitter. We do NOT
want to distill SF; NNUE is a LATER step trained off our OWN strong eval, so the current goal is to push our
HAND-CRAFTED eval closer to SF11's (classical, pre-NNUE).

## The grounding datum (fresh, 2026-07-05)
At EQUAL fixed depth 8, us vs classical SF11 (both 1-thread, SF18 arbiter): **our score 7.5% / 120 games ≈ −437
Elo of PURE EVAL gap** (search neutralized), with **35/120 "collapses"** (we reach a winning peak then lose — an
eval mis-judgment, not a search horizon). ⇒ eval is a massively unmined surface; SF11's hand-crafted eval is far
better than ours before search enters. (Search/EBF is a SECOND, larger gap on top — not the question here.)

## Where the eval gap concentrates (per-theme, fresh)
Themed move-match of OUR eval at depth 10 (STS 15-theme, 15k positions), sorted: WEAK = **AKPC 38%** (king-side pawn
advance/attack), King Activity 43%, Knight-reposition 45%, Open Files/Diagonals 46%, Attacking-themes 46%, Square
Vacancy 46%, a/b/c-pawn advance 47%. STRONG = Recapturing 68%, Bishop-v-Knight 62%, Simplification 61%, Pawn-play-
center 56%. **Pattern: we are weak on DYNAMIC/ATTACKING themes, strong on STATIC/MATERIAL** — consistent with the
outcome residual (king-safety + mobility + rooks are the dynamic dims) and the 35 equal-depth collapses (dynamic
mis-judgment). So our candidate co-tune CLUSTER = {king-safety, threats, mobility, open-files/rooks, king-zone
attack-layer}, holding static/material fixed.

## What we've established (verified, hard-won)
1. **Single-term eval ADDITIONS don't convert to Elo for us.** Our biggest, most-triangulated lever — king safety
   (outcome-incremental-validity +0.0031 by-game, ~2.5× threats) — after we CONSOLIDATED it (it was triple-counted
   across `latent_threat` + a midgame king-shield bonus + the attack-layer; deduped, gated, byte-id-safe), proved a
   4× better king-danger REPRESENTATION than the term it replaced AND ~3% FASTER — yet came back **neutral-to-
   negative at every venue** (lightning −50/−95 with hot magnitude; deep equal-node ~0 ±53; blitz −36). Magnitude
   and venue explained the *catastrophic* losses (lightning is KS's worst venue; the outcome-fit magnitude was too
   hot for play) but even done right it does not become a *win*.
2. **Our hypothesis (needs a method, not just a term):** we tuned KS *in isolation against a frozen rest-of-eval*,
   and the eval is ONE vector — moving one component alone creates an internal contradiction (KS says "king in
   danger, avoid" while threats/mobility/placement still price the position as fine → incoherent move-choice →
   neutral-to-negative). I.e. **eval↔eval coupling**: terms must be co-rebalanced to stay coherent.
3. **Our only shipped eval win was a conditioned SUBTRACTION** (scaling down an over-read term by a cheap live
   detector), not an addition. And a **flat outcome-Texel retune of the live SCALE constants came back "tapped"**
   (held-out result-loss ~flat) — BUT it was scale-only, STATIC, fit-to-outcomes, and never brought the *gated*
   terms (king-safety, mobility, threats) online *together*. So "holistic" in the true sense is untested.
4. **We have a representation map** (which terms double-count which board features) from the KS audit, so we can
   dedupe before rebalancing.

## The QUESTION (method design — where we want your research)
Given eval is the surface (−437 equal-depth) but single-term additions don't convert (isolation/incoherence), what
is the MOST EFFICIENT way to holistically rebalance our hand-crafted eval toward SF11's equal-depth strength — with
LIMITED game throughput and strong overfit risk? Specifically:

A. **Objective for the fast inner loop.** We have a deterministic, no-games themed move-match compass and an
   equal-depth vs-SF11 harness. Is equal-depth MOVE-AGREEMENT with SF11 the right inner objective — or does that
   re-introduce the "SF as magnitude TARGET" trap that already burned us (fitting our eval to SF's *totals* was
   confounded by a strength-neutral global scale; argmax-invariant)? Move-agreement is argmax, not magnitude — is
   that the escape, or does it have its own failure mode? Would a per-position "does our best move match SF11's at
   equal depth" objective, optimized directly, be sound? Or cploss-vs-SF11? Or a blend with game-outcomes?

B. **Cluster vs full-vector.** Should we co-tune the WHOLE eval vector at once (full SPSA), or coordinate-descent
   over CORRELATED CLUSTERS (e.g. the king-attack cluster = KS + threats + king-zone attack-layer + mobility-near-
   king, tuned together; then structure; then endgame-scale)? How to identify the clusters — from the term
   correlation structure, from SF11's own term grouping, or from where the equal-depth gap concentrates?

C. **Overfit control with few games.** Many params + limited SPRT throughput = overfit risk (we've been burned by
   fit-metrics that weren't strength). If the inner loop is a fast deterministic compass (move-match), how do we
   keep the final SPRT-gated result from overfitting the compass? Held-out themes? A structured prior toward the
   status quo? Regularize toward SF11's *relative* term weights (as a prior, not a target)?

D. **Is SPSA even the right optimizer** given we have fast deterministic compasses, or is a
   regression/Texel-on-move-agreement / structured convex fit more efficient than noisy stochastic search? What did
   SF's own dev process actually do (fishtest = per-patch SPRT at scale — which we can't match on throughput), and
   what's the best *low-throughput* analog?

E. **Spot-fix vs holistic.** We have a 35-position collapse corpus (acute equal-depth eval failures) AND per-theme
   systematic weakness data. When is a targeted subsystem fix (a specific failure pattern) the right move vs a
   board-wide holistic rebalance? How to tell which regime a given weakness is in?

## What would change our mind / what we're NOT asking
Not asking "should we do NNUE" (later, off our own eval — not now). Not asking to distill SF's eval. We want the
most efficient PATH to close the equal-depth HCE gap, given the tooling above and our isolation-failure finding.
