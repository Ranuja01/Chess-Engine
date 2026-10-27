# King-Safety Activation — SF11/Ethereal-aligned (2026-07-16)

**Status: BUILT + deterministically validated + GAME-GATE POSITIVE (categorical). Nothing committed. byte-id 247 on defaults.**

## GAME GATE RESULT (2026-07-17, paired full-game A/B vs SF@2400, 200g each, seed0, conc3)
- Baseline (KS off): **42.2%**, 83 collapses. Bundle (KS on): **41.5%**, 83 collapses. Raw score + total collapses FLAT (expected — KS trades aggression for safety; SF is stronger so fixing one class exposes the next).
- **CATEGORICAL VERDICT (the real test): KS-caused collapses 15 → 7 (−53%)**, other-class 68 → 76 (+8), total +0. `diagnostics/ks_collapse_attribute.py` (SF11 King-safety ≤ −1.5 on each collapse decision_fen). KS-caused collapses more than HALVED; replaced by the newly-exposed other class. Guard clean: KS-caused DROPPED (KS not causing king-danger collapses); the +8 are non-king-danger (exposed, not caused). Aligns with the deterministic over-read −0.85. Small-count (15 vs 7 ≈ 2 SE) but directionally clear + consistent with two other evidence lines.
- **RECOMMENDATION: commit KS v1** (ship config below). This is the first of the accumulating collapse-fixes; next = diagnose the now-dominant "other" class (the +8/76). Nothing committed — awaiting user.

## Why
Collapses (winning→lost vs SF@2400) are eval OPTIMISM (deep-triage: we play SF's move at d18 yet mis-value by mean +4.9). Static-vs-SF11-static confirms ~90% is a static-eval flaw. Autopsy localized the dominant, statically-fixable cause: **king safety disabled** (`KING_SAFETY_MAG=0`). 29% of over-read collapse decisions are king-danger blindness (mean over-read +5.6). We HAVE a full SF-style `king_safety_danger` but it was off AND mis-defined vs SF11.

## Root cause of prior KS failure (component dump on 27-pos danger set, SF11 KS mean −4.0)
Detector under-fired (26% coverage) at the shipped weights because:
- **Defender over-subtraction**: defpc(3.8) × KS_DEFENDER erases ~76% of attacker units. SF11 has NO blanket defender term (defense is IMPLICIT).
- **Weak squares ~0** (our def = zero-defenders; SF = under-defended, ≤1 K/Q defender).
- **Safe checks ~0** (our clause = zero-own-coverage; SF adds overwhelmed-defender: weak AND doubly-attacked).
- Missing **no-enemy-queen discount** (SF −873 / Ethereal −237, on their scale).
Color-symmetry was already perfect (wrong-sign = accuracy, not asymmetry). KS operating MAG is ~thousands not 100 (percent × small base).

## Change (all gated default-off = byte-identical; 3 localized inserts in `king_safety_danger`)
- New knobs (search_engine.h): `KS_NO_QUEEN=0`, `ENABLE_KS_SF_WEAK=false`, `ENABLE_KS_SF_SAFECHECK=false` (+ env parse).
- **weak** (~4995): SF def = `popcount(dm)<=1 && (dm & (knights|bishops|rooks|pawns))==0`.
- **safe-check** (3 loops): `check_safe` lambda adds `(weakS && popcount(amS)>=2)` to the baseline `dmS==0`.
- **no-queen** (before clamp): `if (KS_NO_QUEEN && !(queens & enemy)) units -= KS_NO_QUEEN`.

## Key finding: SF-defs need the DEADZONE as partner
SF-defs alone RAISE sensitivity everywhere (calm false-fire 12→18) — they're a superset, so at unchanged weights they add units on calm too. But SF-defs + `KS_FLOOR` deadzone SEPARATE cleanly: SF-defs make calm danger SMALL (clippable), real danger stays LARGE. The baseline (proximity danger) could never be floored — its noise sat above any floor. Deadzone is the SF-defs' partner, not a substitute.

## SHIP CONFIG (knob bundle, no default change)
`ENABLE_KS_REPLACE_LT=1 KING_SAFETY_MAG=3000 KS_DEFENDER=0 ENABLE_KS_SF_WEAK=1 ENABLE_KS_SF_SAFECHECK=1 KS_FLOOR=13 KS_NO_QUEEN=6`
(replaces latent_threat.) MAG=3000 chosen over 4000 via gate B: WAC 248 (+1 vs base) vs 243@4000, smaller offensive double-count, keeping ~85% of defensive gain.

## Deterministic gates (ALL PASS)
| gate | baseline | F13@3000 |
|---|---|---|
| byte-id (defaults) | 247/41.48M | 247/41.48M ✅ |
| firing (danger/calm/eg false-fire) | 59%/12/3 | 67%/7/2 ✅ dominates |
| color-symmetry | 0 | 0 ✅ |
| over-read king-danger | +5.57 | +4.72 (−0.85) ✅ |
| over-read total | +3.37 | +3.14 (−0.23) ✅ |
| over-read placement (offensive) | +3.69 | +3.84 (+0.15) ⚠️ small |
| WAC tactics | 247 | 248 ✅ |
| nodes | 41.48M | 42.27M (speed ~neutral at MAG=3000) |

## DESIGN GUARD-RAIL (user, confirmed): KS stays CONDITIONAL — NO blanket central-king penalty
Central kings are NOT inherently bad; a king is only in danger when the enemy has real attacking potential. Burning
tempo to castle when no threat exists is worse than developing/attacking. SF11 already encodes this well via its
conditional attacker/safe-check/weak-square logic — that IS the good logic, so we replicate the MECHANISM, never a
positional "king off back rank = bad" term. Our KS satisfies this: every unit needs actual enemy presence (attackers
in zone / weak squares / safe checks), and the `KS_FLOOR` deadzone is a FEATURE (only fires on substantial, real
danger). The central-king residual (Kd2/Ke8 misses) is UNDER-SCORING of a genuine threat (safe-check weighting too
small vs SF), NOT a missing central-king penalty. v2 = boost DEFENSIVE safe-check sensitivity (still check-gated) +
trim attackingLayer's redundant offense. Never a blanket king-square term.

## Open / next
- **Offensive `attackingLayer` double-count** (placement +0.15): KS's offensive half overlaps the existing king-attack placement bonus. Small; deferred v2 lever = trim `attackingLayer` king-attack if games show attacking over-optimism. Do NOT perturb attackingLayer (load-bearing ~400E) on a 0.15 signal.
- **Residual king-danger over-read +4.72** is largely the OTHER over-read classes (material/placement over-scaling) that co-occur on those positions, NOT KS-still-missing — the next collapse class to bridge.
- **Game gate (A)**: measure CATEGORICAL KS-collapse reduction (not total rate; SF stronger ⇒ exposes next class). Single-core collapse-continuation replays; 4-core SF@2400 overnight on user signal. Guard: new collapses exposed not caused (autopsy+symmetry+control clean).

Tooling: diagnostics/ks_firing_profile.py, ks_component_dump.py, ks_symmetry.py, build_ks_sets.py, sf11_collapse_gap.py (--engine-env), king_safety_probe.py. See [[eval-collapse-diagnosis-method]], [[collapse-categorical-verification]].
