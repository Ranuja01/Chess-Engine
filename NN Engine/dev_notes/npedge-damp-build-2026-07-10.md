# npedge realizability damp — build + offline screen (2026-07-10 overnight)

The first eval term built under the full realizability discipline (contrastive fit -> mechanism gate ->
slice screens -> score-primary gauntlet). Target: the fantasy-vs-real collapse over-read. byte-id 247
preserved (all knobs default-off).

## What was built
A clamped-ramp damp on the pawn-placement credit (`br_pt_pawns`), keyed on **non-pawn (piece) material
edge** for the favoured side, midgame-only. cpp_bitboard.cpp right after the `MOD_PIECES_DEFEND` block;
knobs in search_engine.h + env-load/echo in search_engine.cpp.

- `ENABLE_NPEDGE_DAMP` (bool, default false)
- `NPEDGE_DAMP_LO=800`, `NPEDGE_DAMP_HI=2500` (engine units, pawn=1000): ramp
  `fnp = clamp((npedge-LO)/(HI-LO),0,1)`; `npedge<=LO` full damp, `>=HI` no damp.
- `NPEDGE_DAMP_MAX=90` (/256): damp depth `= MAX*(256-fnp)/256`; retained `gain=256-damp`;
  `total += (br_pt_pawns*gain>>8) - br_pt_pawns`. MAX=90 -> floor retains ~65% (Fable's "trim, not erase").
- npedge computed from masks: `popcount(piece & occupied_side)*values[pt]` for N/B/R/Q, favoured side =
  `(br_pt_pawns>0)?(bnp-wnp):(wnp-bnp)`. pawns/king excluded (npedge = NON-pawn edge).

## Correction to Fable's design (data-driven, verified)
Fable specified a `counter_pressure` co-condition (opp offense - our defense) to spare real wins + close
the KPK misfire. **Our corpus REFUTES counter_pressure as a discriminator** (`npedge_cocond.py`):
- Within the low-npedge danger zone, counter_pressure AUC=0.46 (no separation, wrong direction: fantasy
  -813 vs real -651). No cheap detector cleanly rescues the 15 low-npedge real wins (best `off_us` ~0.67).
- **Substituted `!isEndGame` as the co-condition.** It closes the KPK misfire structurally (a pure pawn
  endgame is an endgame -> damp off -> the draw-scale lane owns it) AND matches the two-lane plan, WITHOUT
  the false counter_pressure gate. Confirmed by the screen: all EG slices Δ=0.
- Also corrected: pvb does NOT cascade from pt_pawns (independent code path, keys off piece material) but
  is already npedge-self-limited, so leaving it alone is right anyway.

## Offline slice screen (`screen_slices.py` on corpus_ks.csv, vs cached SF18)
Residual = our_static - sf18_static (cp). PASS = FANTASY-MG drops, REAL-WIN* held ±20, EG slices ~0.

| slice (n)            | old  | MAX=90 | MAX=180 | note |
|----------------------|------|--------|---------|------|
| FANTASY-MG (24)      | +472 | +413 (−59) | +354 (−118) | target: DROP ✓ |
| REAL-WIN (45)        | +332 | +333 (+2)  | +335 (+3)   | HOLD ✓ (high-npedge spared, f=1) |
| REAL-WIN-MG (28)     | +78  | +81 (+3)   | +83 (+6)    | HOLD ✓ |
| REAL-WIN-LOWNP (15)  | −58  | −95 (−38)  | −133 (−75)  | the only collateral (already under-read) |
| COLLAPSE-EG (24)     | +439 | +439 (−0)  | +439 (−0)   | gate ✓ (midgame-only) |
| CONTROL-MG (299)     | −5   | −2 (+4)    | +2 (+7)     | HOLD ✓ |
| CONTROL-EG (68)      | +42  | +42 (−0)   | +42 (−0)    | gate ✓ |

Frontier ratio ~fantasy:collateral = 1.55:1, fixed by the ramp (MAX scales both). Dramatically better
targeted than l1a (which hit the whole REAL-WIN set -> −80 Elo). Only the 15 low-npedge reals take
collateral, and per the co-cond fit those convert by ATTACK (off_us +738), not placement.

## WAC tactical tripwire (Fable's floor)
Baseline 247/300, 41,479,610 nodes. MAX=90 -> **244** (−3), +7% nodes. MAX=180 -> **241** (−6). The damp
DOES change move-choice (not a no-op — answers Fable's caveat), at a mild, scaling tactical cost. MAX=90
is the safer arm.

## Decision
- Primary ship candidate: **MAX=90** (fantasy −59, minimal collateral, WAC −3).
- Secondary probe: **MAX=180** (fantasy −118, but 2× collateral + WAC −6).
- Gauntlet SCORE primary, ≥3 seeds, paired vs banked baseline (byte-id 247 identical when off, so banked
  s0-3 = valid references). No fully-closing the +365 over-read is possible via pt_pawns alone (floor
  bounds it); the bet is that trimming the fantasy tail nets positive without l1a's collateral.

## GAUNTLET RESULT (2026-07-11): NO-GO, net −1.8% (neutral within noise)
3-seed paired, 400g each, same byte-id-247 .so (damp env-gated):

| seed | baseline | damp MAX=90 | Δ | collapse base→damp |
|------|----------|-------------|-----|--------------------|
| s0 | 46.9% | 48.9% | +2.0 | 22.3% → 14.8% |
| s1 | 53.6% | 45.2% | −8.4 | 20.0% → 19.3% |
| s2 | 46.5% | 47.5% | +1.0 | 25.5% → 17.3% |
| **mean** | **49.0%** | **47.2%** | **−1.8** | 22.6% → 17.1% |

Seed0 in isolation was a false positive (the ≥3-seed rule caught it, as with singular/null-gate). Collapse
rate fell ~5.5pt but score didn't follow — the "collapse-rate down ≠ score up" lesson for the 3rd time
(l1a broad −80, npedge targeted −1.8).

## ROOT CAUSE OF THE NULL (move-selection diagnostic, `moves_dump` + `move_flip_report` on the 824-pos
stratified corpus, SF18 judge): THE LEAK IS TACTICAL, NOT BROAD-POSITIONAL.
Per-stratum move-selection Δ (win%-loss; − = damp improves move choice):

| stratum (n)   | ungated | TQUIET=2 | TQUIET=4 |
|---------------|---------|----------|----------|
| collapse (24) | −0.65 ✓ | +0.76 ✗  | +1.71 ✗  |
| sts (350)     | +0.35 ✗ | −0.11 ✓  | +0.11    |
| neutral (200) | −0.06   | −0.25    | −0.24    |
| game (244)    | −0.25   | −0.00    | −0.32    |
| ALL (818)     | +0.04   | −0.09    | −0.06    |

The ungated damp IMPROVES positional strata (collapse, game) but WORSENS the tactical `sts` stratum
(+0.35, 45% flip rate). Net cploss ≈ 0 — a cancellation. But in games a tactical error costs a whole
result while a quiet gain rarely swings one, so "flat mean cploss" = net-negative games (and seed1's
sharp openings ate the −8.4). Same root as the +7% nodes / −3 WAC: perturbing the eval scatters the
search's TACTICAL RESOLUTION.

## ⭐ STRUCTURAL CONCLUSION (rigorous): the collapse over-read is a TACTICAL problem; static damping can't
fix it. `tension_by_stratum.py`: collapse positions have mean 5.6 SEE-captures, 92% ≥2, 75% ≥4 —
STATISTICALLY IDENTICAL to the `sts` (5.7/98%/82%) and real-`game` (5.4/92%/74%) strata, and far above
`neutral` (3.1/76%/36%). So the collapses ARE tactical (pawn-up-WITH-counterplay). The quiet-gate
(`NPEDGE_DAMP_TQUIET`) fixes the sts leak but destroys the collapse fix, because tension can't separate
"helpful firing" (collapse) from "harmful firing" (sts) — they occupy the same tension regime.
⇒ **A static positional-eval damp is the WRONG TOOL for this collapse class.** You cannot lower the
over-read without disturbing the search's tactics in exactly the positions that matter. SF11 reads these
low WITHOUT losing tactics because its eval is coherent with its search; our bolt-on damp fights ours.
⇒ **Redirect:** the collapse is a SEARCH / counterplay-blindness problem at the tactical decision point,
not a static-magnitude miscalibration. This is the 3rd independent confirmation that reducing the static
over-read does not convert to score. The eval-damp line for collapses is banked/exhausted.

## Knobs (all default-off, byte-id 247 preserved): ENABLE_NPEDGE_DAMP, NPEDGE_DAMP_LO/HI/MAX,
NPEDGE_DAMP_TQUIET (tactical-tension gate). Tools added: npedge_hist.py, npedge_cocond.py,
screen_slices.py, movediff_fantasy.py, tension_by_stratum.py; moves_dump.py made pyrun-env-compatible.

## OPERATIONAL LESSON (recurring-mistake candidate)
**Do NOT run `build` while a gauntlet is running.** Rebuilding ChessAI.so mid-run corrupted in-flight
engine process loads -> games 68-84 of gnt_base_nightA errored ("engine exited unexpectedly"). Games
outside the build window are clean and still scored (errored games are excluded from the denominator, not
counted as losses). Sequence builds BEFORE launching game runs; once built + byte-id-checked, drive all
arms via env knobs on the stable .so with no further rebuilds.
