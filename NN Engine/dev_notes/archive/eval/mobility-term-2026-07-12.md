# Mobility term — the first clean eval candidate (2026-07-12)

## How we got here (understand-the-static-failure, offline-first)
The Lane-1 pipeline offline tests: **stage 4 (leaf visibility) CONVICTED** — walking each over-push line 4
plies forward THROUGH the refutation (`overpush_refutations.csv`), our static leaf reads **+253cp** while
SF11 reads **+51cp** (gap +202, 72% over-read; `leaf_visibility_test.py`). The proposed medicines FAILED:
stripped/full threats (+249/+255, no move), KS (no move), OvD/imbalance cranked (WORSE, +303 — wrong-signed).
Decomposition (`leaf_breakdown.py`): the over-read = **pieces(PST placement) +268 + material +131**, our
activity term (OvD) barely fires (+11). ⇒ our PST placement is **context-blind**: credits our pieces' squares
while the opponent out-activates us; SF11's MOBILITY tracks the opponent's compensation and nets to +51.

## ⭐ Mobility VALIDATED before building (`mobility_proxy.py`)
Correlation of our attacked-square edge with the leaf over-read: **−0.47 (attacked-sq) / −0.53 (legal-moves)**.
Over-read = **+320cp when the opponent is more mobile** vs **+74cp when we are** (n=26/24). ⇒ we over-read
exactly when the opponent out-activates us. First eval mechanism this campaign PROVEN before writing C++.

## The term (built, byte-id 247 preserved)
`ENABLE_MOBILITY` / `MOBILITY_SCALE` (search_engine.h ~811). In `placement_and_piece_eval` right before the
npedge damp (cpp_bitboard.cpp ~6823): count squares each side attacks from the already-built `attack_bitmasks`
(excluding own-occupied), `total += (bmob - wmob) * MOBILITY_SCALE` (Black-positive). ADDITIVE (shifts the
feature ratio, not a damp) + SLOW (attacked-square count moves WITH the material resolution) => the
volatility rule holds. Default-off = byte-identical. Reuses the pv_wmob/pv_bmob loop pattern (pvb block).

## Offline screens
- **Move-lens (`moves_dump`+`move_flip_report`, 816-pos stratified, SF18 judge), per-stratum Δ (− = mobility
  improves move-selection):**

  | scale | collapse | sts (tactical) | neutral | game | ALL |
  |-------|----------|----------------|---------|------|-----|
  | 100 | +0.42 | +0.04 | −0.02 | +0.23 | +0.09 |
  | **150** | −0.36 | **−0.03** | +0.05 | −0.19 | −0.07 |
  | 250 | +0.57 | −0.02 | +0.07 | −0.07 | 0.00 |
  | 500 | +0.57 | +0.48 (scatter) | +1.17 | +0.34 | +0.61 |

  **⭐ ROBUST: mobility HOLDS `sts` at all sane scales (100-250: −0.03..+0.04) — the FIRST eval term all
  campaign to clear the volatility screen** (every prior term scattered sts +0.35..+0.48). **NOT robust: the
  net move-lens BENEFIT** — 150's −0.07 is a fragile, few-flip-dominated peak (100=+0.09, 250=0.00 bracket
  it). Honest read: mobility is volatility-SAFE but ~NEUTRAL on the move-lens at sane scales; 500 over-weights
  and scatters. So it's *safe, not clearly helpful offline* — the gauntlet must decide.
- **Over-push avoidance (`our_depth_avoid`, d10, baseline 42%):** scale 150 = 43% (+1), 500 = 47% (+5). The
  narrow over-push metric is MODEST at the clean scale; the broad move-lens (game −0.19) is the stronger
  signal (but cploss-style = ANTI-PREDICTIVE for ship — see [[compass-context-fragility]]).

## ⭐ GAUNTLET VERDICT: NO-GO (mean −1.05%, regression-to-the-mean)
Paired baseline vs mobility@150, 400g/seed, conc4:

| seed | baseline | mob@150 | Δ |
|------|----------|---------|-----|
| s0 | 46.9 | 49.5 | +2.6 |
| s1 | 53.5 | 44.6 | −8.9 |
| s2 | 46.5 | 50.5 | +4.0 |
| s3 | 50.0 | 48.1 | −1.9 |
| (s4 base) | 50.9 | (not run) | — |
| **mean (4 pairs)** | **49.2** | **48.2** | **−1.05** |

Sign-flipping, POSITIVE on low-baseline seeds (s0/s2 ~46-47%) and NEGATIVE on high-baseline seeds (s1/s3
~50-53%). Paired ⇒ the s1 −8.9 is a real per-seed effect, not baseline luck. **Mobility REGRESSES our play
toward the mean** (baseline spread 7pt → mobility spread 5.9pt, pulled toward ~48 everywhere): it makes us
play solid/less-sharp, helping the bad seeds and hurting the good ones, net slightly negative. Same
load-bearing-optimism wall as npedge ([[collapse-fix-load-bearing-optimism]]); the gauntlet confirms the
move-lens (cploss) anti-predicted.

## ⭐⭐ THE STRATEGIC LESSON (bank this)
Mobility was the FIRST eval term all campaign to (a) have a mechanism PROVEN before building (corr −0.5),
(b) pass the volatility screen (holds sts) — and it STILL failed at the gauntlet via regression-to-mean.
⇒ **The eval-FEATURE lane is closed for SCORE (not just for the collapse):** our eval sits at a peaky-but-
effective local optimum, and ANY broad feature change (even validated, even volatility-safe) reduces
variance without raising the mean, because our aggressive optimism IS our strength. This closes the
"understand-the-static-failure → build the missing feature" path — not for lack of the feature (mobility IS
missing) but because adding it dampens the load-bearing optimism.
⇒ **The remaining strength levers are the ones that DON'T change eval FEATURES:** (1) **Lane 2 SPEED/NPS**
— makes the SAME eval faster → more depth, so it is IMMUNE to regression-to-mean (the eval is byte-identical,
just deeper); this is why the roadmap made it "always-on, funds every lane," and per the SF11 depth curve
depth IS a proven collapse-killer. (2) **Lane 1/3 SEARCH** (ordering/TT/IIR) — changes what's searched, not
eval features. The method (offline-first, understand-then-validate, volatility rule) WORKED — it just proved
the eval-feature lane is a dead end for this engine, which is itself a durable, expensive-to-learn result.

## Status + next
**DONE — mobility is NO-GO (banked default-off; byte-id 247 preserved).** The eval-feature lane is closed for
score (above). NEXT = the non-eval-feature lanes: **Lane 2 NPS/speed** (make the same eval faster → depth,
regression-immune; pawn-hash has prior scar tissue — the incremental pawnKey bug class, cache impurity — so
approach carefully) + the **depth-at-fixed-movetime instrument** to measure it; and **Lane 1/3 search**
(ordering unfreeze / TT-rebuild → TT-move → IIR). The inject-ceiling 2×2 remains a valid offline experiment
but its stage-4 arm (mobility) is now known gauntlet-neutral, so its value dropped. Knobs `ENABLE_MOBILITY`/
`MOBILITY_SCALE` stay compiled default-off. Tools built this campaign: `leaf_visibility_test.py`,
`leaf_breakdown.py`, `mobility_proxy.py`, `make_refutation_map.py`, `our_depth_avoid.py`, `sf11_overpush_test.py`,
`triage_slice.py`, `diagnose_collapse_moves.py`. byte-id 247/41,479,610; nothing committed.
