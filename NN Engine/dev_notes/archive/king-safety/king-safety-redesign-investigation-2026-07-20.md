# King-safety redesign investigation (2026-07-20)

Follow-on from the Kaufman work: the eval-vs-SF11 gap analysis flagged KING SAFETY as a big divergence. This
note consolidates the KS deep-dive: SF11's model, our comparison, SF18 re-validation, and why a simple
re-weight is NOT the fix. Read with `kaufman-imbalance-2026-07-20.md` and the collapse ledger.

## SF11 king() recipe (source study, `stockfish_11/.../evaluate.cpp` king<Us>())
`kingDanger` = Σ (per-piece weighted attackers: N=81/B=52/R=44/Q=10) + 185·weak-ring-sqs + 148·unsafe-checks +
98·pins + 69·king-adjacent-attacks + safe-checks (**R=1080 N=790 Q=780 B=635**) + flank − 873·(no enemy queen)
− 100·knight-defender − shelter − flank-defense + 37. Transform: `if kingDanger>100: score -= kingDanger²/4096`
(mg), `/16` (eg). Pawn shelter/storm is a separate cached Score. **Scale reaches 1000-2000 on a real attack.**

## Our model (`king_safety_danger`, cpp_bitboard.cpp) — we HAVE every ingredient
`units` = KS_ATT_KNIGHT(2)·#N + KS_ATT_BISHOP(2)·#B + KS_ATT_ROOK(3)·#R + KS_ATT_QUEEN(5)·#Q +
KS_ATTACK_COUNT(1)·attacked_sqs + KS_WEAK(2)·weak_sqs − KS_DEFENDER(0)·defenders − KS_SHIELD(2)·shield_pawns
+ KS_OPEN_FILE(2)·open + KS_STORM(1)·storm + KS_SAFE_CHECK(3)·safe_checks − KS_NO_QUEEN(6)·(no enemy queen).
Then `if units < KS_FLOOR(13): return 0` (hard cliff); `danger = min(units,CAP)²/KS_DIVISOR(4)` quadratic up to
KS_KNEE(12) then LINEAR; × KING_SAFETY_MAG(3000) × phase-taper. Coffin `KS_INTERACT`=0 (off).
**So it is the SAME mechanism** (per-piece attackers, safe-checks, weak, shield, no-queen, quadratic). The
differences vs SF are (a) flat weight HIERARCHY (safe-check=3 ≈ everything, vs SF's safe-check≫all), (b) a hard
floor cliff vs SF's smooth >100 threshold, (c) a knee→linear cap SF lacks, (d) coffin off. NOT missing parts.

## SF18 re-validation of the "KS under-read" dossier — 64% was SF11-static contamination
The dossier (`ks_underread_vs_sf11.txt`, 36) was built on SF11-STATIC, which is invalid on sharp positions
(user caught #3: SF11-static −1.53 but SF18-search 0.0 = OURS correct). `ks_sf18_revalidate.py` re-scored with
SF18-search: **only 13/36 GENUINE**; 23 dropped as artifacts. Genuine set -> `ks_sets/ks_underread_sf18.txt`.
LESSON reaffirmed: validate KS gaps with SF18-SEARCH, never SF11-static.

## The genuine gap is TWO halves (`ks_genuine_units.py`, raw units, floor off)
Of the 13 SF18-confirmed genuine KS gaps (unsafe-king units):
- **6/13 merely FLOORED** (units 10-17, killed by KS_FLOOR=13) — re-weightable.
- **7/13 DETECTION under-fire** (units 0-3 on a real +3p attack — e.g. `6k1/q4p2/...` SF18 −2.39, uW=**0**).
  **No re-weighting fixes a 0** — our detector misses these attack patterns entirely.

## Re-weight tuner = NO-GO (partial + risky)
`ks_separation.py` (objective: FIRENEW↑ hold FIREOLD/SUPPRESS) + `ks_tune.py` (coordinate descent). Baseline:
FIREOLD(danger.txt)=1.90, FIRENEW(genuine)=0.24, SUPPRESS(control)=0.10. Best re-weight (KS_SAFE_CHECK=20
KS_FLOOR=10 KS_KNEE=32): FIRENEW 0.24→**0.82** (only the floored half), but SUPPRESS 0.10→**0.21** (2×, near cap)
and **WAC 243→230 (−13)** (STS neutral 1590→1590). So lowering the floor wakes calm noise (the collapse cause)
and dents tactics, for a partial fix. **NO-GO as a standalone lever.**

## BUILD (2026-07-20 cont.): Front A aim detector = bench-clean win; coffin DROPPED (structural)
- **Front A — latent king-AIM (`ENABLE_KS_AIM`, built, gated, byte-id):** catches enemy sliders ALIGNED with
  the king through EXACTLY ONE blocker (the discovered/latent line our line-of-sight zone-scan misses). Reuses
  the empty-board king rays (as `slider_blockers`) + `betweenPieces`; iterates only the 0-3 aligned sliders (no
  new per-piece loop; O(aligned) — cheaper than SF's per-piece ring test). Weights `KS_AIM_BISHOP/ROOK/QUEEN`.
  RESULT (aim 6/8/11, no coffin): FIRENEW 0.24→**0.71** (3× detection), **STS 1606 held, WAC 243 held**,
  suppress-on-SF18-safe 0.074→0.169 (modest over-fire, benches unaffected). **Bench-clean; game-testing.**
- **METHODOLOGY: SF18-validate the CONTROL set too.** Of 200 old control positions, only **66 SF18-genuinely-
  safe**; **105 SF18-actually-dangerous** (many mates/±8 — our KS firing there is CORRECT, not noise). Guard =
  the 66 (`control_sf18safe.txt`), not the contaminated 200. (Same lesson as the SF11-static dossier: old sets
  aren't truth; SF18-search is.)
- **Coffin (`KS_INTERACT`) DROPPED — structurally wrong, not just mis-tuned.** It TANKED STS (1606→1503 at
  KS_INTERACT=3, →1405 aggressive) while aim-alone held STS. Code review: our base is already SF-shaped
  (additive `units` → ONE quadratic `units²/KS_DIVISOR`, super-linearity from the single square). The coffin
  adds a SECOND, MULTIPLICATIVE non-linearity (`undefended×(open+1)×attackers`) that is then SQUARED by the
  transform = double amplification SF never has → volatile over-read on ordinary middlegames. Also its `(open+1)`
  defangs the open-lines gate (never zeroes). **Redundant (quadratic already gives co-occurrence) + unstable.**
  LESSON: "our own, not a copy" = our detectors/weights on a SOUND (SF-shared) architecture, NOT a novel worse
  structure. The aim term is the right kind of flair (additive signal into the architecture that works).
- **Coffin ALTERNATIVE (SF-aligned, additive):** if we want more "undefended holes" sensitivity, raise
  `KS_WEAK` (strict `KS_SF_WEAK`, currently only 2 vs SF's heavy 185-relative) — additive, stable, amplified by
  the existing quadratic. To try after the aim-only game.

## Conclusion + next direction
The KS gap is NOT a flat-mis-tune (the shipped model separates its own corpus 19.5×). It is:
1. **DETECTION** — 7/13 genuine attacks under-fire (units 0-3). Need to find WHAT our zone/attacker/safe-check
   detection misses on those patterns (design + code, not a knob).
2. **SIGNAL-TO-NOISE** — can't lower the floor without waking calm noise; needs the multiplicative coffin
   (KS_INTERACT, off) or better discrimination so genuine attacks separate from calm.
Both are DESIGN tasks (user-steered). The simple re-weight is shelved. Kaufman remains the shippable win.
Corpora/tools built: `ks_sf18_revalidate.py`, `ks_genuine_units.py`, `ks_separation.py`, `ks_tune.py`,
`ks_gap_dossier.py`; `ks_sets/ks_underread_sf18.txt` (13 genuine), `ks_underread_vs_sf11.txt` (36 raw).
