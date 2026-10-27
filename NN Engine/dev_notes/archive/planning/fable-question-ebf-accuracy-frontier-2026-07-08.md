# Fable brief — EBF-reduction lane hit an accuracy frontier; is Lane 2b the wrong lane right now? (2026-07-08)

## What you're adjudicating
We ran the EBF-reduction / node-balancing campaign (your Lane 2b) for a session. Every pruning lever we
tested reduces nodes but **costs bench accuracy** in a place our self-play venue structurally cannot see.
We may be spending the accuracy statScore-LMR just bought. We need you to tell us whether to keep pushing
pruning, re-weight our metrics, or pivot the campaign. Concrete data below.

## Engine / frame (recap)
Non-negamax C++ engine, ~2700 Elo, pre-NNUE. Absolute eval (Black-positive), separate minimizer/maximizer,
single root flip. millipawn scale (pawn=1000). Shipped baseline this era: **statScore-LMR** (continuous
reduce-less LMR, +23 lightning SPRT / +33 node_ab) — it de-buries LMR-reduced refutations (accuracy up) but
paid **+6% nodes**. Byte-id baseline **247/300 WAC, 41,479,610 nodes, STS 1550/3000**. Root-cause frame from
your prior audits: our SEARCH invents a ~+3.9 phantom over a well-calibrated static eval (buries opponent
refutations at shallow depth). Campaign thesis (Lane 2b): claw back statScore's +6% via **position-specific**
prunes (history reduce-more is dead — context-blind global [from][to], 5 failures).

## The user's key insight this session (please pressure-test it)
**Bench tactical/strategic loss is self-play-INVISIBLE.** If both engines share a blind spot, self-play
never steers into the position that exposes it → self-play Elo cannot punish an accuracy regression, and
symmetrically cannot reward an accuracy gain. Implication: a lever that is "+X Elo in lightning self-play"
but "−Y on a static accuracy bench (WAC/STS)" may be a **mirage** — it beats an equally-blind copy of itself
by searching deeper while actually choosing worse moves. Is this reasoning sound? How should we weight a
static accuracy bench (STS) vs self-play Elo when they disagree?

## The data (all single-core, sequential, deterministic; baseline 247 WAC / 1550 STS / 41.48M nodes)

| Lever | WAC solved | STS /3000 | Nodes | Venue-game result |
| --- | --- | --- | --- | --- |
| **statScore-LMR** (shipped baseline) | 247 | 1550 | 41.48M | +23 lightning SPRT |
| **null_eval_gate** (only attempt null-move when static eval past bound) | 247 (0 loss) | **1477 (−73)** | −2.3% | ~+14 lightning SPRT (directional, ~127g); prior +11.3 node_ab |
| **ProbCut** (self-verifying, margin=1500) | 236 (−11) | — | −8.2% | node_ab **+17.5** (equal-node) but lightning SPRT **~−83** (~34g) |
| **passer-danger** (eval accuracy term) | — | **1511 (−39)** | — | prior handoff claimed STS +78 — does NOT reproduce on post-statScore baseline |
| SEE-prune quiets (best margin) | 241 (−6) | — | −2.9% | not gated |
| null-gate + progressive + passer (combined) | 239 (−8) | 1536 | −3.2% | not gated |

Readings we've drawn:
- **null_eval_gate**: WAC-neutral but **−73 STS** = strategic-accuracy regression. Its +14 lightning fits the
  "mirage" hypothesis exactly (deeper but worse decisions). Same shape as **malus** (calibration-greenlit,
  STS-regressed → we killed it).
- **ProbCut**: the clean venue split — its shallow verification search inherits our search's phantom
  miscalibration; deep (equal-node) it's trustworthy (+17.5), shallow (lightning, ~1-ply verify) it's blind
  (−83). We banked it as a "late lever" (pays once search soundness/eval improve). Correct call?
- Every node-reducer trades against accuracy somewhere self-play can't see.

## The strategic questions
1. **Is Lane 2b the wrong lane right now?** If pruning inherently trades against accuracy (and self-play
   can't police it), is "claw back the +6% via pruning" self-defeating pre-NNUE? Should the campaign pivot
   its PRIMARY effort to the two accuracy-neutral levers — **SPEED** (pawn-hash NPS: more depth, byte-
   identical decisions, zero accuracy cost) and a genuine **ACCURACY** term (threats, resolves in 1-2 plies)
   — and treat pruning as secondary/parked?
2. **Metric adjudication:** when STS and lightning self-play disagree (null_eval_gate: −73 STS / +14 Elo),
   which is the strength truth for a pre-NNUE engine, and why? Is the user's self-play-blind-spot argument
   correct, or is STS itself a poor proxy (fixed-depth, strategic themes) for lightning strength?
3. **ProbCut:** is "late lever, banked default-off, revisit after search-soundness + eval improve" right, or
   is there a near-term way to make its verification trustworthy at shallow depth (empirical margin/MIN_DEPTH
   calibration — require child ≥3-4 ply)?
4. **passer-danger:** an accuracy term that is −39 STS but was theorized to be self-play-invisible hidden
   strength. Given the user's own insight, do we trust STS (−39 → drop it) or the invisibility argument
   (test it at vs_sf1, the asymmetric venue)? How to resolve "accuracy term fails the accuracy bench"?
5. **Meta:** are we actually lost, or is this the expected shape of a pre-NNUE engine at its pruning/accuracy
   frontier — where the only Pareto-improving moves left are speed and eval-accuracy, not more pruning? If
   so, what is the correct ordered program for the next N sessions?

## What we have ready
statScore shipped; ProbCut + null-R built (default-off, byte-id clean); Phase A levers wired; shadow-price /
cutcal / see-selfcheck diagnostics; the self-play harness (node_ab / lightning SPRT / vs_sf1 asymmetric /
STS / WAC). We can build the threats term and pawn-hash NPS. Time this session: a few hours.
