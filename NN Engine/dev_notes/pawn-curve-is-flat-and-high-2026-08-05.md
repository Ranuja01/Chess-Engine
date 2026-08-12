# ★★★ Our pawn value curve is FLAT and HIGH; SF's is steep and low-based

The single largest measured discrepancy in the pawn subsystem, and it reframes both the passer work and the
"two central pawns equal a minor" problem as **the same defect**.

Method: `pawn_truth_ours.py` recomputes OUR marginal pawn value on the exact FENs the SF18 ground truth was
measured on, so every term our eval already pays (placement, attacking layers, chain/wall, capture gains,
realizability) is subtracted out. Both columns include the pawn's own material. 6,229 samples, medians.

## THE CURVE
| rank | SF18 | ours | gap (cp) | gap (mp) |
|---|---|---|---|---|
| 2 | +96 | +249 | **−153** | −1532 |
| 3 | +67 | +221 | −154 | −1544 |
| 4 | +56 | +225 | **−169** | −1693 |
| 5 | +80 | +232 | −152 | −1525 |
| 6 | +190 | +245 | −55 | −550 |
| 7 | +424 | +340 | **+83** | **+831** |
**Ours spans 221→340 = 1.5×. SF spans 56→424 = 7.6×.**
⇒ **We barely distinguish a back pawn from a passer on the 7th.** We overpay every pawn on ranks 2-5 by
~150 cp and underpay the 7th by ~83 cp. The crossover is between rank 6 and rank 7.

## ★★★ WHY THIS REFRAMES THE PASSER LANE
Six passer mechanisms have been killed (rank floor · blockade-scaled residual · flat unconditional base ·
ordinary-pawn floor · rear reduction · joint-with-threats), and **every one of them tried to RAISE THE TOP**.
The measurement says the top is only ~83 cp low while the BOTTOM is ~155 cp high. So raising the top alone:
- leaves the RATIO wrong (what matters for choosing between pawns), and
- adds material inflation on top of an already-inflated base.
⇒ **The correction is a SLOPE correction, not a level correction.** Lower ranks 2-5, then raise rank 7.
That is a different intervention from all six that failed, and it is the first one the data actually asks for.

## ★★ AND IT IS THE SAME DEFECT AS "TWO CENTRAL PAWNS EQUAL A MINOR"
At ~250 cp per pawn, two pawns = ~500 cp — more than a minor (325) and near a rook (500). The owner
observed exactly this and capped structural+positional at 225 mp (22.5 cp). But **most of the overpayment is
OUTSIDE that clamp** (rank bonus, capture gains, king safety, central score), which is why the cap reduced
the symptom without removing it.

## WHAT THIS DOES NOT SAY
- It does not say our material value for a pawn is wrong (`values[PAWN]=1000` with N 3250 / B 3450 / R 5000
  are sane ratios). The inflation is in the POSITIONAL terms that fire per pawn.
- It does not localise WHICH term overpays. The gap is the sum over every term that changes when a pawn is
  added. ▶️ Next step is `ev_breakdown` on the same paired FENs to attribute the ~150 cp per term.
- The obstruction gaps (blocked −138, opposed −152, passed −137, contested −70, piece_blocked −105) are all
  the same order, so the flatness is broad, not one context.

## ✅ ATTRIBUTION — WHICH TERMS CARRY IT (`pawn_gap_attribution.py`, 6,986 samples, medians, White-POV cp)
| term | r2 | r3 | r4 | r5 | r6 | r7 |
|---|---|---|---|---|---|---|
| `pt_pawns` (INCLUDES 100 cp material) | 105 | 115 | 122 | 125 | 124 | 105 |
| `passed_pawn_support` | 21 | 18 | 24 | 14 | 12 | **118** |
| `piece_value_boost` | 39 | 33 | 35 | 34 | 45 | 51 |
| `kaufman_imbalance` | 17 | 17 | 17 | 16 | 14 | 16 |
| **ours** | 251 | 222 | 225 | 233 | 245 | 340 |
| **SF18** | 100 | 74 | 55 | 80 | 190 | 424 |
**Only FOUR terms respond to adding a pawn.** Three findings:
1. `pt_pawns` contains the 100 cp of material, so the **entire** placement + rank + structural signal in it
   is **5-25 cp and does NOT rise with rank**.
2. **`passed_pawn_support` is the ONLY rank-responsive term, and it is a STEP FUNCTION** — flat 12-24
   through rank 6, then 118 at rank 7.
3. `piece_value_boost` + `kaufman_imbalance` contribute **~50-68 cp of RANK-INDEPENDENT inflation** per pawn.
★★★ **This explains why the DETECTOR dominates.** `passed_pawn_support` is the only term that scales with
rank, so a pawn the detector does not flag gets a flat ~5-25 cp regardless of how advanced it is. It also
explains why realizability's 1.25× cap binds so hard: R modulates only this one term.
⚠️ These are medians pooled over all five obstruction contexts, so per-context behaviour is hidden — the
per-cell table shows we OVERPAY phalanx passers and UNDERPAY lone ones at rank 7.

## ▶️ ORDER OF WORK
1. **Attribute the ~150 cp** per-term via `ev_breakdown` on the paired FENs. Do NOT reshape before knowing
   which terms carry it — a uniform shrink flattens the eval (already a recorded lesson).
2. Then correct the SLOPE: reduce ranks 2-5, raise rank 7, keeping the material scale fixed.
3. Only then grade the obstruction selector.
⚠️ This is a large eval change and therefore a real ship candidate (it should clear the 20-40 Elo floor),
unlike the constant-sized variants that have dominated recent testing.
