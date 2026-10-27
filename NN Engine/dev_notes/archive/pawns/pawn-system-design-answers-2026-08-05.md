# The pawn system — design answers, grounded in SF18 ground truth

Answers the owner's four design questions using `pawn-truth-findings-2026-08-05.md` (2,817 samples,
ranks 5-7) plus a file-contrast run (3,350 samples, ranks 4-6, files a vs d). All figures in **cp**;
engine units are millipawns (`values[PAWN] = 1000`), so **1 cp = 10 engine units** and the **225 clamp
is 22.5 cp per pawn**.

## THE MEASURED ORDERING OF FACTORS
| factor | size | resolves? |
|---|---|---|
| rank | **+27 (4→5), +108 (5→6), +300 (6→7)** | yes, decisively |
| obstruction | **~3-5× span** at fixed rank (r6 isolated: passed +271 · piece_blocked +166 · pawn-blocked +94) | yes |
| structure (strong−weak) | **+4 (r4), +13 (r5), +27 (r6)** | mostly NOT resolved |
| file (a vs d) | mean **+2 ±12 (unresolved)**, median ~15 | weakest measured |

## ❓Q1 — ABOLISH FILE SCORING, OR ADD A RANK FACTOR? → **KEEP IT, DON'T GROW IT, DON'T PROMOTE IT**
Do **not** abolish. Edge-vs-centre is real in the median (~15 cp) though the mean does not resolve, and our
`pawn_chain_file_bonus` already spans **10→150 mp = 1→15 cp**, which is the right ORDER for a ~15 cp effect.
The file scaling is not the problem and removing it would be discarding a small real signal.
⚠️ But **file is the wrong second axis for the table.** It is the smallest of the four factors; obstruction
is 3-5× larger and rank ~10× larger. The brief proposed **file × rank**; the data says **rank × obstruction**,
with file left where it is — a minor modifier on the structural component only.
⚠️ One caveat on FORM: ours is a 15× MULTIPLICATIVE ratio on the chain bonus, while the measured effect
looks like a small roughly-constant offset. That difference is below what this run resolves; do not act on
it yet.

## ❓Q2 — REPLACE THE PER-RANK BOOSTS, OR ADD TO THEM? → **NEITHER: REFINE THE SELECTOR**
We already have three rank tables — `default_` / `passed_` / `endgame_pawn_rank_bonus`. That is *already* a
rank × obstruction table with a **2-level** obstruction axis (passed vs not) and a phase axis. The data says
obstruction has **~5 distinguishable levels spanning 3-5×**.
⇒ Do not add a new competing term, and do not replace the tables. **Grade the selector**: choose which rank
table (or which interpolation between them) a pawn indexes into, based on passed / contested / piece-blocked
/ pawn-blocked / opposed. Same mechanism, more resolution, **nothing new entering the clamp**.
★ And fix the SHAPE. Ours steps ×1.34, ×1.29, ×1.25 — **decelerating**. The truth steps ×1.9 then ×2.2 —
**accelerating**. We flatten exactly where value takes off:
| rank | SF18 positional (measured − 100 cp material) | our `passed_midgame` | |
|---|---|---|---|
| 5 | ~+43 to +59 | 62.5 cp | ≈ right |
| 6 | ~+165 to +198 | 84.0 cp | **~2× low** |
| 7 | ~+489 to +601 | 108.5 cp | **~5× low** |

## ❓Q3 — BUILT FROM PARTS, OR BASELINE-HIGH-THEN-SCALE? → **BASELINE-THEN-SCALE, AND EXTEND REALIZABILITY**
The obstruction effect is **multiplicative, not additive**: at fixed rank, passed/opposed is a *ratio*
(~2× at r5, ~3× at r6, ~4.8× pooled), not a constant offset. That is exactly the `mag × R/256` shape we
already use for passers.
⇒ The owner's intuition is right, and the answer to *"for non-passers there is no realizability"* is that
**there should be**. The obstruction axis IS realizability for an ordinary pawn. One form for both:
`value = rank_magnitude(rank, phase) × obstruction_factor`, where `obstruction_factor` is today's `R` for
passers and a new, cheaper obstruction term for everyone else.
🚨 **WITHHOLD, NEVER PENALISE.** A blockaded 7th-rank pawn measures **+227..+399** — still far above a free
rank-5 pawn (**+143..+159**). So the scale must be **floored**, never driving an advanced pawn below a less
advanced one. This matches SF (the whole safety analysis sits inside `if (empty(blockSq))`; SF withholds the
advancement bonus and never subtracts) and it is an argument against our `BLOCK[]` docking.

## ❓Q4 — THE 225 CAP AND "TWO CENTRAL PAWNS = A MINOR" → **THE CAP IS CORRECTLY SIZED. DON'T RAISE IT.**
The cap is **22.5 cp per pawn**. The measured strong-vs-weak structural axis is **+4 to +27 cp**.
⇒ **The clamp is calibrated almost exactly to the true total value of pawn structure.** The owner's
empirical fix — derived from central pawns growing until two equalled a minor — lands within ~10% of what
SF18 independently prices the structural axis at. It is doing real work and should stay.
⚠️ Raising it is the wrong lever twice over: structure is not worth more than ~30 cp, and
[[pawn-clamp-is-spent-on-placement-not-structure]] shows **141 of the mean 211 raw is `positional`**
(placement + attacking layers), so raising the cap hands ~2/3 of the new room to terms nobody wanted to grow.
✅ **The 5× shortfall is NOT in the structural path — it is in the PASSER path**, which is deferred and
priced *outside* the clamp. So: **raise the passer ceiling, leave the structural cap alone.** That is
precisely the separation that prevents a repeat of the two-pawns-equals-a-minor failure, because that
failure came from structural terms, not passer terms.
✅ **"Dangerously passed" is exactly the right criterion** and the data defines it: danger = **advanced AND
unobstructed**. Free r7 **+589..+701** vs blockaded r7 **+227..+399** vs free r5 **+143..+159**. A passed
but undangerous pawn should land near an ordinary pawn of its rank — which is what a floored multiplicative
`R` already produces.

## ▶️ WHAT THIS IMPLIES, IN ORDER OF EXPECTED VALUE
1. **Fix the rank curve's shape** (accelerating, not decelerating) and raise the top two ranks on the
   PASSER path. Largest measured discrepancy by far (~5× at r7), and it lives outside the clamp.
2. **Grade the obstruction selector** from 2 levels to ~5. Second largest factor, reuses existing tables.
3. **Extend realizability to non-passers** as a floored multiplicative factor.
4. Leave the structural cap and the file scaling alone.
⚠️ Ranks 2-3 are still unmeasured, so the brief's early-push risk is unjudged — measure before touching the
low ranks, since raising the top of an accelerating curve is exactly what makes early pushes look attractive.
