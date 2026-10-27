# KS architecture: SF11 & Ethereal vs ours — why a single-count floor misfires (2026-07-21)

**Motivation (user):** our KS gates danger with a blunt magnitude FLOOR on a single accumulated count; lowering
it to catch low-end attacks wakes quiet positions (STS −88). SF and Ethereal don't use a magnitude floor at all
yet stay quiet on quiet positions AND fire on low-attacker-count real threats. What's their mechanism?

## The three structures side by side
**Ours** (`cpp_bitboard.cpp` king_danger, ~5024-5168):
```
units = KS_ATT_* · attacker_counts + KS_ATTACK_COUNT · attacked_zone_squares(PROXIMITY)
        + KS_WEAK·weak + KS_OVERLOAD·overload − KS_DEFENDER·def − KS_SHIELD·shield − KS_NO_QUEEN
if (KS_MIN_ATTACKERS>0 && att_pieces<gate) return 0     # gate exists but default OFF
if (units < KS_FLOOR) return 0                           # MAGNITUDE FLOOR = the blunt instrument
danger = min(units,cap)² / KS_DIVISOR                    # already convex
```
Positive side DOMINATED BY PROXIMITY; suppressors single-digit units; gate default-off; magnitude floor is the
sole quiet-silencer.

**Ethereal** (`evaluateKings`): NO magnitude floor. Two structural devices:
- COORDINATION-COUNT GATE: `if (kingAttackersCount > 1 − popcount(enemyQueens))` → needs ≥2 attackers, or ≥1
  with an enemy queen. Below that, king safety is not computed at all.
- SIGNED BALANCE → SOFT ZERO-FLOOR: `safety = attackerWeightSum + 45·scaledCount + 42·weak
  + SafeChecks(N112 R90 Q93 B59) + pksafety(shelter/storm) − 237·noQueen − 74·base`; then
  `eval += −mg·MAX(0,mg)/720` (quadratic mg, linear eg). MAX(0,mg) = danger only when positives beat
  suppressors; no hand-set threshold.

**SF11** (`evaluate.cpp` king(), 378-461): NO floor; `kingDanger =` count·weight + 185·weakRing + 148·unsafeChecks
+ 98·blockers(pins) + 69·kingAttacks + 3·flank²/8 + (mobility[Them]−mobility[Us]) + safeChecks(R1080 N790 Q780
B635) − 873·noQueen − 100·(N&K defend) − 6·score/8 − 4·flankDef + 37; then `if (kingDanger>100) score −=
kingDanger²/4096`. Safe checks are the DOMINANT positive; suppressors are LARGE and CONTEXT-SENSITIVE.

## The two mechanisms we lack (the answer)
1. **Gate on COORDINATION COUNT, not magnitude.** "How many distinct pieces coordinate" (structural), not "how
   much accumulated pressure." A lone piece / diffuse proximity → not an attack, regardless of unit total.
2. **A context-sensitive SIGNED COUNTERBALANCE that reaches zero naturally.** Strong suppressors (no-queen
   −237/−873, good shelter, king-defended, king-mobility, base bias) OUTWEIGH the positives in quiet positions
   so danger = max(0, …) = 0 WITHOUT a threshold. The effective floor is per-position (a sheltered queenless
   defended king is pushed far below 0; an exposed king with a queen sits near/above 0). Our fixed floor chops
   the same amount off every position and our suppressors are too weak to cancel proximity.

## Queen used TWICE (both engines) — not redundant
- GATE threshold: enemy queen lowers the attacker-count bar (2→1) — a BINARY "is this a king-attack?" decision.
- MAGNITUDE suppressor (−237 / −873): no queen scales danger DOWN — a CONTINUOUS "how big?" decision.
Same fact, two roles (when-to-fire vs how-much). We use it once, tiny (`KS_NO_QUEEN`), and not in a gate.

## Structural-gate validation on the SF11-anchored corpus (`ks_structural_gate.py`)
| tier | n | gate fires | mean attackers | queenless% |
|---|---|---|---|---|
| attack | 64 | 82.8% | 1.14 | 4.7% |
| quiet_neg | 128 | 46.1% | 0.66 | 20.3% |
- **COUNT-GATE alone zeroes 53.9% of quiet_neg vs only 17.2% of attacks (~3:1 asymmetry)** — the structural
  "suppress when not needed" mechanism works, magnitude-independent.
- BUT my Ethereal-raw-weighted proxy scored AUC 0.71 < our units 0.81 → copying Ethereal CONSTANTS blindly
  loses; the gain needs re-tuning the combined structure (gate + suppressors + threat-weights) together.
- No-queen barely helps here (both classes mostly have queens); it bites only the ~20% queenless quiet set.

## Concrete lever (testing now, existing knobs)
`KS_FLOOR=6` (low, catch low-end attacks) + `KS_MIN_ATTACKERS=2` (gate silences the quiet positions the low
floor would wake) ± `KS_ATTACK_COUNT=1` (demote proximity). STS direct + deterministic; if the gate recovers the
−88 while keeping attack capture, it's the shippable structural fix (and reframes the old "KS_MIN_ATTACKERS
NO-GO" as "NO-GO in isolation on a proximity base + magnitude floor", not NO-GO with a lowered floor).

## Full redesign (if the knob test is promising but partial)
Restructure king_danger to the SF/Ethereal shape, all gated default-off: (1) COUNT-gate replaces magnitude
floor; (2) re-weight positives toward safe-checks + weak-squares, demote KS_ATTACK_COUNT proximity; (3) strong
context suppressors (scale up no-queen; add defended-king, king-mobility, base bias); (4) danger = max(0,
positive − suppressors) then the existing quadratic. Tune the whole structure jointly (ks_fit_sts) with an STS
guard; validate on an SF18-anchored (non-circular) corpus before games.

Tooling: `ks_structural_gate.py`, `ks_sf_feature_sep.py`, `ks_ours_units_auc.py`, `build_ks_sts_corpus.py`,
`ks_fit_sts.py`. Companion: `ks-sts-overfire-dissection-2026-07-21.md`.

## Joint diverse-corpus study outcome (2026-07-21, fit-only, double-count-safe)
Built a DIVERSE SF18-anchored corpus (`build_diverse_corpus.py` → `ks_sets/diverse_corpus.csv`, 563 rows:
243 target / 71 working / 41 calm / 21 crowded_safe / **187 sts_guard** = the STS "prone-to-break" positions
this session exposed, added as a protected guard tier; SF18 truth CAPPED at the SF11-static ceiling; stratified
train/val). Joint constrained win%-descent (`ks_fit_diverse.py`, OvD triple + shelter HELD FIXED to avoid the
king-zone attacker−defender double-count; only CLEAN KS levers in the grid).
- **Fit picked `KS_FLOOR=6 KS_SAFE_CHECK=8` (magnitude) and did NOT select the count-gate** (MSE can't see it).
- **VALIDATION (STS300): fit-best NO gate = 1497 (−58 vs 1555) despite the sts_guard MSE tier "holding".**
  → definitive: static win%-MSE ≠ move-choice; MSE-fitting is necessary-not-sufficient for KS. Adding the
  gate: 1497→1566; the hand config with proximity kept high (`KS_ATTACK_COUNT=2`) is the STS best at **1588**.
- Target capture on the DIVERSE (SF18) corpus is intrinsically low (12–14%, |SF11tgt|=2.23) — the KS under-read
  is largely SEARCH-BOUND (ceiling-capped), reinforcing "not a static-eval discrimination failure".
- **SHIP CANDIDATE (double-count-safe, deterministically validated): `KS_FLOOR=6 KS_SAFE_CHECK=8
  KS_ATTACK_COUNT=2 KS_MIN_ATTACKERS=2`** (STS 1588, +33; controls held). Suppressors (KS_NO_QUEEN/KS_DEFENDER)
  were not MSE-selected and were marginal on top of the gate → gate is the dominant lever, suppressors optional.
- **METHOD LESSON (bank):** for KS specifically, select on the STS MOVE bench + games, NOT corpus win%-MSE.
  The diverse corpus + guards are still valuable — they calibrate magnitude and prevent gross control breakage —
  but the gate/move-choice lever must be chosen by the bench.
- New tooling: `build_diverse_corpus.py`, `ks_fit_diverse.py`, `ks_ours_units_auc.py`, `ks_structural_gate.py`.
  Night game matrix (4-core): baseline (Kaufman+CAPG_PIN) vs +count-gate KS [± KS_NO_QUEEN=12].
