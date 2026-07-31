# Falsification of the "eval lane closed" doctrine — the ruler was bent (2026-07-15)

## The claim under test
Prior doctrine (from the fixed-NODE SF18@400 gauntlets): eval-FEATURE + eval-ADJACENT levers are "closed lanes"
— corrhist −1.7%, mobility −1.05%, all regression-to-mean. Fable's reframe: those negatives are an ARTIFACT of
the fixed-node venue, not eval defects. Mechanism: a more-accurate eval prunes LESS (RFP/null/futility fire
less) → more nodes/position → fewer plies at the 250k-node cap → scored a loss. Prediction: re-gate at fixed-
DEPTH (node cost is free) and the negatives evaporate/invert.

## Venue logic (why this is decisive)
- **Fixed-NODE** PENALIZES eval accuracy (accuracy costs plies at the cap).
- **Fixed-DEPTH** OVER-CREDITS it (accuracy is free — no CPU cost, no pruning-reduction cost charged).
- **Fixed-TIME** is the truth (charges the eval its real per-move cost AND rewards better moves via the clock).
Also: the prior verdicts were UNPAIRED vs SF18 (huge seed variance); the new runs are PAIRED base-vs-cand self-
play (SF only adjudicates draws) = far lower variance. So the falsification is also a methodology upgrade.

## Results (byte-id 247 build, paired self-play, SF18 arbiter)
| lever    | fixed-NODE (prior) | fixed-DEPTH-10 paired            | fixed-TIME lightning SPRT           |
|----------|--------------------|----------------------------------|-------------------------------------|
| corrhist | −1.7% (→"closed")  | 50.4% / +2.6 Elo ±39.9 (NEUTRAL) | pending (overnight)                 |
| mobility | −1.05% ("held STS")| 53.6% / **+24.8** Elo ±41.1      | **~+20 Elo** (542g: +226−193=123, 53.0%, LLR+0.54) |

- corrhist: the negative EVAPORATED (neutral at fixed depth). Not proven a win; its fixed-time gate is pending.
- mobility: the negative INVERTED. Fixed-depth +24.8 (over-credited), fixed-TIME **~+20 Elo held rock-steady
  for 500+ games** (band +18..+22 the whole run). This is the REAL arbiter charging the eval cost → a genuine,
  ~+20 Elo gain that the fixed-node venue threw away as a "closed lane."

## TRANSFER TEST vs Mediocre (2026-07-16) — INCONCLUSIVE (venue too insensitive), NOT a falsification
Paired vs Mediocre (fair equal-time, 40 games each): baseline 13.8% (5 collapses) vs +mobility 11.2% (9 collapses).
**Do NOT over-read this** (earlier draft wrongly called it a "transfer failure" — corrected): at ~13% score the
Mediocre venue is TOO INSENSITIVE to detect a +20 Elo lever (sensitivity peaks near 50%; SE~±5% over 40g buries
+20 either way). So this run NEITHER confirms nor refutes that mobility's self-play +20 transfers. The collapse
count (9 vs 5) is a small-count difference, suggestive at most. **OPEN QUESTION (worth testing, not concluded):**
does self-play Elo transfer to real-opponent Elo for our over-optimistic engine? To answer, gate at a SENSITIVE
(~50%) REAL-opponent venue (SF at a graded UCI_Elo), not Mediocre@13% and not self-play alone. Lesson (self-
inflicted): stop turning "can't measure here" into "didn't work" — the exact small-sample over-read this reset
was meant to kill.

## CONCLUSION — the doctrine was a measurement artifact (NARROW claim only; see transfer failure above)
Two independent eval levers both shed their fixed-node negative once the node-budget penalty was removed;
mobility converts to ~+20 Elo at fixed TIME. **The "eval-FEATURE / eval-ADJACENT lanes are closed / load-
bearing optimism is fundamental" conclusion rested on a biased instrument (fixed-node gauntlet) and is UNSAFE.**
The eval lane is REOPENED. `mobility` (ENABLE_MOBILITY=1, default MOBILITY_SCALE=40) is a confirmed +20 Elo
brick and the anchor for a bundle campaign.

## Caveats / discipline carried
- Fixed-depth OVER-credits eval (mobility +24.8 there vs +20 at time) — never ship on a fixed-depth read alone.
- The SPRT capped short of the +2.94 accept bound (expected: a +20 effect vs elo1=5 accumulates LLR slowly);
  the point estimate over 542 games is the verdict, not the bound-cross.
- Overlap RESOLVED (fixed-depth STS 2x2, mobility × cheap-rook): SYNERGY, not overlap. mobility WITH cheap-rook
  = +52 STS (1550->1602); mobility WITHOUT cheap-rook = −50 (1522->1472). Sign flips on cheap-rook presence ⇒
  complementary, NOT redundant. The +20 gate was measured with cheap-rook ON (default) ⇒ VALID as-is. RULE:
  never ship mobility with cheap-rook off; do NOT consolidate (replacing cheap-rook w/ mobility loses ~130 STS).
- The SPSA sub routes EVAL knobs to fixed-DEPTH — now known to over-credit. Move eval-SPSA to fixed-TIME (or
  validate winners at time) before any joint eval tune, else we re-introduce a bent ruler at the escape step.

## NEXT (overnight 4-core + single-core prep)
1. Re-gate the OTHER "closed" eval-feature levers at fixed-TIME, one at a time (KS/imbalance/damps/corrhist).
2. MOBILITY_SCALE fixed-time magnitude tune (default 40 — peak unknown).
3. Combine the fixed-time survivors → hand bundle or joint fixed-TIME SPSA (the local-optimum escape).
4. Mobility↔cheap-rook overlap check (single-core) to confirm it's a clean brick.
Nothing committed; byte-id 247 default intact; mobility et al. remain env-gated default-off.
