# The tail screen is falsified — 2026-07-27

## Result

The fixed-node tail screen **cannot distinguish two configs that are 53.8 Elo apart**. Run on the graded
corpus (`cploss_corpus_wide.csv`, whole corpus, seed 12345, `PRESET=LONG_FORMAT NODE_LIMIT=250000`):

| config | mean | >5% | >10% | **>20%** | p95 | **p99** | n |
|---|---|---|---|---|---|---|---|
| base (shipped default) | 25.25 | 15.7% | 5.2% | **1.0%** | 10.2 | **20.1** | 9823 |
| `ENABLE_QDELTA_PERMOVE=0` | 25.03 | 15.2% | 5.1% | **1.0%** | 10.2 | **19.9** | 9838 |

Default beats qdelta-OFF by **+53.8 ±37.5 Elo** (456 games, colour-symmetric, commit `029f619`).

- **The tail is flat.** `>20%` is identical at 1.0%; p99 differs by 0.2 (noise).
- **The mean is inverted.** qdelta-OFF scores *better* (25.03 vs 25.25), so ranking on the mean would have
  rejected the shipped +54 Elo change.

**Control passed:** `[qdelta_permove] seen=0 fires=0` in the qdelta-OFF run confirms the prune genuinely
did not run. This is not a mistyped/ignored knob — see `env-knob-name-verify`.

## What this retracts

The 2026-07-26 finding that fixed nodes + the tail resolves the qdelta gap was measured on the **OLD**
corpus at **n=1477** (`>20%` 0.5% vs 0.8%, p99 17.0 vs 18.4). At n=9838 on the graded corpus that
separation **does not replicate**. The original signal was ~7 vs ~12 tail events — inside its own stated
±10 Poisson bound. It was noise, and the handoff's own caveat ("run the full 9,000 before trusting a
verdict") was the correct instinct.

⇒ **The screen can neither promote nor kill a candidate.** It is not a triage layer.

## What survives

- **The ruler is still right.** Fixed nodes is deterministic (two runs byte-identical) and fair to
  node-cutting configs. The defect is in the *statistic*, not the budget. Fixed TIME remains void
  (σ≈1.0, sign flips); fixed DEPTH remains biased against node-savers.
- **The corpus is still good.** 77,432 games → 4.29M unique positions, tiered off the engine's own eval
  trace rather than off cploss (so no selection-inflation). Reusable by any future metric.
- **The error profile still holds** and still inverts intuition: `inaccuracy` has the highest mean loss
  (27.59) and `blunder` the lowest (22.68). We bleed in quiet subtle positions, not where we once blundered.

## The standing problem, restated

Every metric we have tried now fails to predict Elo:

| metric | verdict |
|---|---|
| WAC solves (fixed depth) | +1 solve for +54 Elo; blind to node savings |
| STS (fixed depth / equal time) | +107 and +93 for ~0 Elo |
| cploss mean (fixed time) | σ≈1.0, sign-flips between runs |
| cploss mean (fixed nodes) | **inverted** on the one known-Elo pair |
| cploss tail `>20%` / p99 (fixed nodes) | **flat** on the one known-Elo pair |

The mean-vs-tail hypothesis was the best remaining explanation for the week's contradictions. It is now
tested and dead. **Games remain the only instrument that has ever tracked Elo.**

Note the asymmetry that makes this expensive: a 456-game tournament gives ±37 Elo, so games can only
resolve candidates around the qdelta scale (+50). Most of the remaining queue is plausibly +5–20, which
neither games at this budget nor any screen we own can currently resolve.

## Consequences for the queue

Candidate selection falls back to **prior warrant** (cross-engine presence per
`sf-schedule-portability-heuristic`) rather than to any local measurement, with games as the only gate.
That ordering favours **ProbCut** (present in SF11/15/16/17/18 *and* Ethereal) over corpus-fit ideas.

Open question worth more than another metric attempt: whether an SPRT with a proper stopping rule buys
more resolution per hour than fixed-length tournaments, given that everything left is small-effect.
