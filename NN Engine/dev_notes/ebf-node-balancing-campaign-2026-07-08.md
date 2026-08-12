# EBF-Reduction / Node-Balancing Campaign (Lane 2b) — 2026-07-08

## Why
statScore-LMR shipped (+23 SPRT / +33 node_ab; byte-id **247 / 41,479,610**) but spent **+6% nodes**
(reduce-less widens the tree). The "reduce-more via history" half is **DEAD** — the decoupled malus/Q-cut
(`ENABLE_QCUT`) STS-regressed (1550→1483→1439 monotone with λ, no EBF gain), 5th malus failure, root cause
= **context-blind global `[from][to]` indexing**. Lesson: **history reduces LESS only; node clawback must
come from POSITION-SPECIFIC signals** — eval margins, exact SEE, self-verifying search.

Enabling shift: the eval is more trustworthy now (SEE fix +18.2, `ENABLE_CAPG_COND` balanced, cheaper
bishop/rook mobility, attack-layer caches, collapse fixes). Fable: **all pruning margins are eval-coupled**
("cleaner eval raises the pruning ceiling"). Goal = **node-neutral-or-better at higher accuracy** (the
node-balance: statScore's +6% offset by position-specific cuts = one budget). Fable Lane 2b.

## Measured baseline (shadow re-price, shipped engine)
LMP **1.23%** wrong-enter (dominant ~24M/suite, near error ceiling → DON'T tighten) · Futility **0.27%**
(slack). ProbCut absent. Tool = `ENABLE_PRUNE_SHADOW` (the wrong-prune pricer, already built).

## Phases (gate each: shadow-price → WAC/STS floor → node_ab → SPRT + 1 TC spot-check; byte-id 247 off)
- **A — drawer wins (no build, node_ab-away):** `ENABLE_SEE_PRUNE`(+margin/depth), `SEE_PRUNE_CAPTURES`,
  `ENABLE_NULL_EVAL_GATE` (already +11.3), `NULLMOVE_PROGRESSIVE`. Exact/eval-gate = eval-independent = safe.
- **B — ProbCut + eval-scaled null-move R (BUILD, single-core):** self-verifying → eval-independent.
  Knobs `ENABLE_PROBCUT` (MARGIN=2200 OUR units NOT SF 189, MIN_DEPTH=5, DEPTH_REDUCTION=3, CANDIDATES=3),
  `ENABLE_NULLMOVE_EVAL_R` (R_DIV=1920, R_CAP=3). **Non-negamax trap: max pushes β UP, min pushes α DOWN.**
  Insert node-level (non-PV, not-in-check, !g_in_verify) after null-move/OTV, before the move loop
  (max ~cpp:3773, min ~cpp:3292); null-R after `reduced_depth -= NULLMOVE_EXTRA` (min 3212 / max 3700).
  Full spec in the approved plan (~/.claude/plans/handoff-search-soundness-campaign-shiny-nygaard.md).
- **C — margin retune on the upgraded eval (sweeps):** widen futility (has slack), tighten RFP; NOT LMP.
- **D — eval feeders (parallel/pointers):** SPEED pawn-hash NPS (prior +8.6%=+11) + eval32-TT →
  `dev_notes/sf-source-evolution-bank-2026-07-06.md`; ACCURACY threats term →
  `dev_notes/eval-lane-strategy-fable-q6-2026-07-06.md`.

## Sequencing DECIDED: pruning-first + eval-speed parallel
ProbCut/SEE-prune are eval-independent → push pruning to the **current-eval ceiling** first; eval-accuracy
(threats) DEFERRED (bigger build, unproven post-statScore conversion — passer went neutral). **Empirical
trigger for eval work = when Phase-C margin sweeps stop improving on node_ab (the eval wall).** Build Phase
B now (single-core) + pawn-hash background; run the node_ab gauntlet (A→B→C) when multi-core opens.

## Parked
Malus/history reduce-more (dead, context-blind, 5×); LMP tighten (error ceiling); Lane-2 TT/negamax-lite +
IIR/singular/hindsight (need TT-move/eval32); correction-history (parked); KS-final-form + space (post-search); NNUE.
