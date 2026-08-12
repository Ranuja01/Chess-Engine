# KS/STS over-fire dissection — why the KS recalibration is more accurate yet regresses STS (2026-07-21)

**Question (user):** the KS recalibration (`KS_FLOOR=6 KS_SAFE_CHECK=8 KS_ATTACK_COUNT=2`) recovers ~31% of
the KS under-read on the attack corpus — measurably MORE accurate eval there — yet drops STS300 1555→1467
(−88). STS is a positional suite. Where is it shifting move-choice, and is it tuning or a detection gap?

## Method (single-core, deterministic; games not needed to localize)
1. Per-position STS diff KS-off vs KS-on (`sts_ks_diff.py`, reads `results/sts_results_<tag>.csv`).
2. Raw-unit band scan over all 300 (`ks_band_scan.py` off/on + `ks_band_join.py`): cross-tab KS activation
   transition × STS outcome.
3. Decisive fork (`ks_overfire_vs_sf11.py`): on positions where WE fire, does SF11-**static** also fire?
   SF11-static is the same no-search level as our eval, so it isolates DETECTION from search.

## Findings
- **Regression is entirely in central/pawn/maneuvering themes**, not king-safety themes: Pawn Play in the
  Center −66, Center Control −38, Undermine −22, Offer of Simplification −21, Knight Outposts −13,
  King Activity −11. (Per-theme delta, test−base.)
- **Two damage mechanisms, roughly equal:**
  - `newly-fires` (floor 13→6 wakes units in the old 6–13 deadzone): **131/300 positions newly get a KS
    term**, net −35. The band signal is NOISY not uniformly wrong (26 regressed / 22 gained / 83 same) =
    the discrimination gap — units 6–13 mix real danger with ordinary middlegame pressure.
  - `already-on` (heavier weights over-amplify positions already firing): 45 positions, **net −39** — the
    single largest bucket. So it is NOT just the floor; `KS_SAFE_CHECK 3→8` + `KS_ATTACK_COUNT 1→2` inflate.
  - `stays-zero`: −14 (search-side leakage: KS changes leaf evals elsewhere in the tree).
- **DECISIVE (SF11-static comparison).** Of the 150 positions where our ON-config KS fires:
  - **52 → SF11-static QUIET** (|SF11_KS|<0.3 where we fire ±0.4–1.8): **net −50**. This is the whole
    regression. SF sees no danger; we invent it.
  - 87 → SF11 fires same sign (both see danger): net −10.
  - 11 → SF11 fires opposite sign: net +11.
  - ⇒ **SF11-static DISCRIMINATES "do not apply KS here" where we cannot.** Not a search gap (SF11-static
    has no search). Not primarily floor/weight tuning (where SF agrees, we're only −10). It is a
    **detection-architecture gap**: SF's static KS detector knows *when* to fire; ours counts king-zone
    attackers without SF's gating (safe checks that actually work, undefended attacked squares, king
    mobility, no-queen gate, etc.). Confirms the memory's "discrimination gap, not range gap" INDEPENDENTLY
    of the position bank.

## Re-tune as the verdict (in progress)
Per user: mix the over-fire negatives (target = SF11-static KS ≈ 0) and the genuine-attack positives
(target = SF11-static KS high) into ONE fit and let the constrained descent find a discriminating zone.
- Corpus `build_ks_sts_corpus.py` → `ks_sets/ks_sts_corpus.csv`: 300 STS positions, `target_ks` = SF11-static
  King-safety, `target_total` = our OFF total with only KS swapped to SF11's (win%-fit drives KS alone).
  Tiers: quiet_neg 128 (|sf11_ks|<0.3, GUARD: stay ~0), attack 64 (|sf11_ks|≥0.75, TARGET: fire), mid 108.
- Fit `ks_fit_sts.py`: coordinate-descend KS knobs INCLUDING the discrimination gates
  (`KS_MIN_ATTACKERS`, `KS_OVERLOAD`, `KS_ATT_PRODUCT`, per-piece `KS_ATT_*`) to cut `attack` win%-error
  SUBJECT TO `quiet_neg`/`mid` not rising. **Outcome IS the diagnosis:**
  - fires on attack without waking quiet_neg → shippable discriminating zone (tunable).
  - cannot → knob space can't express SF's discrimination → detection rebuild (SF-style per-piece gating).

## Tooling added (all in diagnostics/)
`sts_ks_diff.py`, `ks_band_scan.py`, `ks_band_join.py`, `ks_sts_probe.py`, `ks_overfire_vs_sf11.py`,
`build_ks_sts_corpus.py`, `ks_fit_sts.py`. Corpus `ks_sets/ks_sts_corpus.csv`. Result CSVs
`results/ks_band_{off,on}.csv`, `results/sts_results_{ksoff,kson}.csv`.

## DEFINITIVE: which detection logic we lack (SF11 sub-signal separation)
Enumerated SF11 `king()` kingDanger sub-signals (evaluate.cpp:378-461) and measured, on the corpus, which
SEPARATE `attack` (SF fires, n=64) from `quiet_neg` (SF quiet, n=128) via rank-AUC (`ks_sf_feature_sep.py`,
pure python-chess geometry, scores the more-endangered king). Ranking:

| SF sub-signal | attack | quiet | AUC |
|---|---|---|---|
| flank_attack / flank² | 5.0 | 3.5 | **0.719** |
| ring_attacks (density) | 1.73 | 0.85 | 0.699 |
| attackers_count | 1.14 | 0.66 | 0.679 |
| attackers_product (count×wt) | 68 | 38 | 0.628 |
| weak_ring | 0.72 | 0.40 | 0.619 |
| blockers (pins) | 0.27 | 0.08 | 0.594 |
| enemy_has_queen | 0.95 | 0.80 | 0.578 |
| safe_check_any / _wt | 0.17 | 0.04 | **0.558** |
| mobility_diff | 1.06 | 0.37 | 0.543 |

**Findings (robust):**
1. **Our KS detection is NOT grossly broken.** Our raw units rank the attack tier above quiet_neg at
   **AUC 0.813** (ON config) / 0.789 (OFF) — `ks_ours_units_auc.py`, attack_mean 16.4 vs quiet_mean 7.9 units.
2. **The regression is a floor lowered into a genuinely OVERLAPPING band.** attack(~16u) and quiet(~8u)
   distributions overlap (AUC 0.81 ≪ 1.0), so ANY hard floor trades attack-recall against quiet-precision:
   floor 13 (default) zeroes quiet (good STS 1555) but also kills low-end attacks (the under-read); floor 6
   catches them but wakes quiet (−88). The units 6–13 band is intrinsically ambiguous.
3. **Our danger curve is ALREADY convex** (`ks_safety_table[u] = min(u,cap)²/KS_DIVISOR` up to KS_KNEE, then
   linear; cpp_bitboard.cpp:361-383). So "add a quadratic" was a NON-fix — we have one. (AUC is invariant to a
   monotone square anyway; the quadratic is a magnitude/scaling detail, not a discriminator.)
4. **Top reconstructable SF single feature = flank-attack breadth (AUC 0.72)**; ring-density 0.70, count 0.68;
   **safe-checks RULED OUT (0.56)**. But the full SF-feature repro only reached AUC 0.80 combined.

**CRITICAL CAVEAT (circularity — why this is not yet the definitive answer):** the corpus tiers are DEFINED by
SF11-static KS (`attack=|sf11_ks|>=0.75`, `quiet_neg=|sf11_ks|<0.3`), so SF11's real KS separates them at
AUC≈1.0 BY CONSTRUCTION. My python SF-feature reconstruction hitting only 0.80 means the repro is INCOMPLETE,
not that SF is 0.80. Measuring "which logic SF has that we lack" against an SF11-static target is circular and
cannot cleanly name the gap.

**The genuinely definitive next step (non-circular):** re-anchor the corpus to **SF18-SEARCH** (or game/WDL
outcome) labels, then (a) our-units AUC vs that honest target, (b) which SF-reconstructed features (flank, …)
add separation toward it. Outcomes:
- our-units AUC vs SF18 is HIGH and flat with extra features → detection fine; the KS under-read (6–13 band)
  is intrinsically ambiguous statically → chasing more KS capture via floor/curve is a DEAD END; SF's edge is
  search / other terms, not static KS.
- a specific feature (e.g. flank) lifts AUC vs SF18 → that IS the missing logic; add it gated + fit + STS-guard.
- Tooling: `ks_sf_feature_sep.py`, `ks_ours_units_auc.py`, `build_ks_sts_corpus.py`, `ks_fit_sts.py`,
  `ks_logic_probe.py`. SF18 labeler pattern: `add_sf18_labels.py`.

## Discipline note
This was localized DETERMINISTICALLY on a single core (no games) — STS diff + SF11-static comparison. Games
still decide shipping, but the mechanism (SF-quiet over-fire) is established without them.
