# Collinearity: why the eval can't be tuned, and how to fix it

Canonical reference. When two (or more) eval terms are the **same function of the same underlying data**,
the eval becomes *degenerate* — mathematically un-tunable — and every attempt to fit it flattens. This doc
states the mechanism precisely, shows how it appears in our engine, and records the fix (bounded/saturating
re-expression) with the evidence that motivated it (2026-08-10 session).

## The mechanism (the math)

Suppose two terms both contribute to `total` as a linear function of the same signal `A`:

    total += w1 * A          (term 1)
    total += w2 * A          (term 2)   =>   total += (w1 + w2) * A

The output depends only on the **sum** `w1 + w2`. Any split with the same sum — `(1,2)`, `(2,1)`, `(1.5,1.5)`
— produces *identical* predictions on every position. So a fit that tunes `w1` and `w2` from data has
**infinitely many equally-good solutions**: it can determine the sum but never the individual weights. The
parameters are **non-identifiable**. Geometrically the loss surface has a flat valley; the optimiser slides
along it and stops arbitrarily (or wherever the regulariser pushes).

### Two consequences

1. **Un-tunable.** No amount of data fixes it — it's structural, not statistical. `val/train ≈ 0.99` on our
   23k corpus (more data does not help) is the fingerprint of this, not of under-fitting.
2. **No situational granularity.** Because the two terms are locked in lock-step (same shape), you can
   *never* make one fire without the other firing proportionally. But the whole point of having two terms
   (e.g. immediate placement vs "who wins the pressure battle") is that they should respond *differently*
   depending on the position. Collinearity destroys exactly that ability — the two ideas are physically
   incapable of separating when the position demands it.

## Why fitting then makes it *worse* (the flattening spiral)

We hit three stacked flattening pressures, which is why every retune we ran washed:

1. **Objective:** we fit to eval-*distance* from a stronger reference (SF18). Our eval is systematically
   larger, so *any shrink* reduces the distance — the objective *rewards* flattening. (Proven: `PV_BOOST_MAG=0`
   scored best on the cp-distance metric and was move-neutral.)
2. **Conditioning:** a degenerate system is ill-conditioned, so we stabilised the fit with **ridge**. But
   ridge resolves collinearity *by shrinking coefficients toward zero* — it "fixes" the valley by flattening
   the terms. It bakes in the very thing we're trying to avoid.
3. Result: a shrinkage-biased objective, on an ill-conditioned system, stabilised by a shrinking regulariser.
   It could not have produced a non-flat result. The best proxy gain we ever recorded (−41 val, 63 knobs)
   measured **−85.6 Elo in games** — the flattening spiral, not bad luck.

★ **The correct fix for collinearity is STRUCTURAL (de-duplicate / re-shape), not regularisation.** Ridge is
the wrong tool: it trades identifiability for flattening. Remove the redundancy structurally and the fit is
identifiable *without* ridge — so no forced shrink.

## The fix: bounded / saturating re-expression (not deletion)

Deleting a collinear term is usually wrong — collinearity is a **tuning** problem, not automatically a
**play** problem. A redundant-for-tuning term can still be net-positive at the current fixed weights
(measured: `SCALE_CENTRAL=0` is net-negative in play — central helps despite being collinear). So we keep
the term but change its **functional shape** so it stops being proportional to the base:

    linear (collinear):   term = w * diff              -- proportional to the base magnitude
    bounded (identifiable): term = CAP * diff/(diff+K)  -- saturates; NOT proportional to the base

Once the term **saturates**, it responds differently to the input than the linear base does: the base keeps
rising while the term plateaus. In the regime where they diverge, the fit can finally tell them apart (they
have different shapes) *and* they can behave independently by situation. Same fix, both payoffs
(identifiability + situational granularity). This is why SF expresses comparative ideas as bounded quantities
(Space = bounded count, Imbalance = bounded table), never as linear re-sums.

⚠️ Saturation removes the **collinearity** (proportionality), NOT the two-path structure — the signal still
travels both paths. That's fine and correct: a bounded term is a genuinely *different-shaped* signal, not a
proportional second copy. To eliminate the two-path structure entirely you'd delete information; not needed.

## Where it lives in our engine (2026-08-10 audit)

`central_score` and the O/D-imbalance term both re-read the same `attackingLayer`/placement cells already
added to `total` per-piece; king-zone pressure is credited via four channels. Full map:
`dev_notes/eval-architecture-degeneracy-map.md`. Screening each on move-match (target=collapse decisions,
holdout=general) classified them:

| term | screen | verdict |
|---|---|---|
| central differential | already clamped+phase-tapered; removal net −4 target | LOAD-BEARING → keep |
| king-zone boost (post de-king 50%) | removal net −3 target | LOAD-BEARING → keep |
| capgain→material contamination | the "fix" (`PIECEVAL_RECOMPUTE_LATE=1`) net −4 | contamination HELPS → keep |
| **O/D imbalance (linear)** | removal net 0 target / **+5 holdout**; **changed ~87 moves for net 0** | **NOISE — the one collinear term not earning its keep** |

The linear O/D reshuffling ~87 moves for zero net signal is the empirical signature of collinearity:
maximal activity, no information.

## Evidence the fix works

O/D re-expressed as a bounded differential (`ovd_imbalance`, `cpp_bitboard.cpp`, gated `OVD_BOUNDED_MODE`,
default 0 = byte-identical):
- **MODE 1** (dominance ratio `CAP·diff/(off+def)`): balanced STS **−217** at seed — the ratio rescales the
  whole magnitude and mis-seeds. Rejected.
- **MODE 2** (dynamic KNEE by phase `CAP·diff/(diff + KNEE·(128−phase)/128)`): balanced STS **−25** (neutral)
  at an *untuned* seed, and it **reduces the colour skew** (orig/mirror 1681/1666 vs baseline 1631/1741),
  colour-symmetry unchanged (11 violations = baseline). ⇒ a clean, bounded, identifiable replacement.

★ The bounded refactor is **not a standalone Elo win** (O/D is small-signal). Its value is that it makes the
term **identifiable**, so it can be tuned in a joint retune without the flattening spiral. De-dup/re-shape is
the *prerequisite* for a fit that doesn't flatten — not itself the Elo lever.

## Cite this doc when

- Proposing a joint retune (first make the system identifiable, or it will flatten).
- Considering ridge / L2 on eval params (it trades identifiability for flattening — prefer structural fix).
- Adding a comparative/dominance term (make it bounded/saturating, never a linear re-sum of the base cells).
- Explaining why `val/train ≈ 0.99` and why more corpus won't help.
