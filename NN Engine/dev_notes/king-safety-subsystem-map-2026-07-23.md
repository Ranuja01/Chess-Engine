# King-Safety subsystem MAP (2026-07-23) — the reference for the KS over-read / consolidation work

Built like the passer inventory: a bill-of-materials for every place king-danger is credited, the double/
triple-count map, and the consolidation decision. Sources: fable KS-map agent + direct code verification +
the FEN-3/4 `KING_SAFETY_MAG=0` differential. **Confidence tags: [E]=empirically confirmed, [C]=code-verified,
[F]=fable-read (plausible, not independently re-verified).**

## Live-vs-dead (the framing correction — I had this wrong at first)
- **DEAD: `get_latent_threat_score`** (~cpp_bitboard.cpp:5338-5586, the `black_increment−white_increment` fn with
  the `+4` presence offset, `×2` sparse-defender doubling, `÷3`, `+75` central-file bonus). **KS v1
  (`ENABLE_KS_REPLACE_LT=true`, search_engine.h:677) SKIPS it** at ~6708. **[E]** proof: an `ENABLE_KS_DEBUG`
  dump placed in this function NEVER fired. ⇒ my earlier "+4 offset / ×2 doubling" analysis was of DEAD code.
- **LIVE: unit-KS** = `king_safety_danger` (5038-5259) → `king_safety_score` wrapper (5267-5279). **[E]** proof:
  `KING_SAFETY_MAG=0` zeroes exactly the +7.20 `king_safety` term.

## Layer inventory (LIVE king-danger contributions)
| # | Layer (symbol, line) | What it credits | Scale / key knobs | Phase |
|---|---|---|---|---|
| 1 | **unit-KS** `king_safety_danger`→`king_safety_score` (5038-5279); call 6733 | attack-units on each king zone → `ks_safety_table[units]` (units²/`KS_DIVISOR=4` to knee 12, then linear, cap 80), netted white−black | **×30** (`KING_SAFETY_MAG=3000`/100) [C]; att N2/B2/R3/**Q5**, `KS_ATTACK_COUNT=1`, `KS_WEAK=2`, `−KS_SHIELD=2`, `KS_OPEN_FILE=2`, safe-check `KS_SAFE_CHECK=3`/`_DEF=5`, `KS_NO_QUEEN=−6`; `KS_FLOOR=13` deadzone; **`KS_MIN_ATTACKERS=0` ⇒ a LONE QUEEN is never gated out** [C] | tapered: full ≤48, 0 ≥104 (`ks_phase_taper`) — the ONLY real KS taper [F] |
| 3 | **king mg shelter** `evaluate_kings_midgame` (~2875-3024) | flat shelter **185mp/shield-pawn**, **75mp** partial; + `attackingLayer`-derived `baseIncrement` (≤×4) shelter/exposure per ring square; writes O/D scores | 185/75 flat (killed only by `KS_CONSOLIDATE`, default false = LIVE); baseIncrement UNGATED [F] | mg branch only, hard switch |
| 4 | **`attackingLayer` + OvD** `setAttackingLayer` (~7993); OvD imbalance (~6794) | attack map is **king-centric**: +inc per king-2-ring sq, **×`ATTACK_OPEN_MULT=5`** on open sq near king; added per-piece into `total` AND into off/def scores; OvD then re-spends `(off−max(def,0))×IMBALANCE_SCALE=3` | ATTACK_OPEN_MULT=5, IMBALANCE_SCALE=3 [C]; UNCONDITIONED, UNTAPERED [F] | separate mg/eg tables hard switch; OvD all-phase |
| — | modulators (all default OFF): `MOD_KS_BACKING=0` (damp-only material backing), `MOD_KS_CONTROL=0` (scale by O/D control edge — note: circular, that signal is built from the same king-zone attackingLayer), `KS_LIGHT_MAG=0` | | | |
| — | KEEP piece-local: endgame king PST `whitePlacementLayer[KING]` (~4420, genuine centralization); static threats (piece-on-piece); passer king-race (passer subsystem) [F] | | | |

## THE OVER-READ IS TRIPLE-COUNTED (the prize) [E on the numbers, F on the mechanism]
A queen near an exposed central king credits the SAME facts in three LIVE, unconditioned places:
- **unit-KS** (layer 1) → FEN-3 `king_safety +7.20` (×30, lone queen fires since KS_MIN_ATTACKERS=0)
- **attackingLayer** (layer 4) → per-piece king-zone pressure into `total`, ×5 on open squares
- **OvD imbalance** (layer 4) → re-spends the O/D scores → FEN-3 `imbalance_white +2.16`
None asks "does the attack convert." This IS the fantasy-attack signature (matches memory's ks_attack triage:
"our own attacks over-read", and the c5/OvD lead).

## FEN-3 vs FEN-4 DIFFERENTIAL (empirical, `KING_SAFETY_MAG=0`) [E]
| | with KS | KS=0 | SF18 | ideal KS | read |
|---|---|---|---|---|---|
| #3 `3r4/ppp3Q1/nq2k2p/7N/3r2P1/2N1B2P/PP3P2/5bK1 w` (over-fire) | +10.75 | **+3.55** | +3.04 | ≈0 | KS is the ENTIRE over-read; killing it ≈ SF |
| #4 `2n2b1r/1pB1k3/1p4Qp/1N6/4P3/Pn2P3/1q2BP1P/5K2 w` (correct) | +10.52 | **+7.82** | +8.69 | ≈+0.9 | KS is a minor bonus; MATERIAL/imbalance carry it |
⇒ Our KS over-credits attacks broadly; the "correct" case survives only because material/pieces carry it, NOT
because KS is right. #3 lives ENTIRELY on phantom KS.

## CONSOLIDATION recommendation (fable + my double-check)
- **Yes to one phase-blended `evaluate_king_safety()`** — BUT the big lever is **layer 4 (attackingLayer ×5 +
  OvD re-spend), not unit-KS.** Merging only unit-KS + shelter (finishing the `KS_CONSOLIDATE` stub, which today
  kills only the 185/75 flat and leaves baseIncrement + O/D writes live) = the SAME incomplete-consolidation trap
  that sank PASSER_V2.
- **The real prize = TUNABILITY:** today you can't lower the ~3-pawn fantasy reading without touching
  KING_SAFETY_MAG + ATTACK_OPEN_MULT + IMBALANCE_SCALE in three places. Consolidate → one place → one
  **realizability gate on the WHOLE king-danger budget** (not one third).
- **RISK / TRAP:** the mg-king shelter baseIncrement feeds `whiteDefensiveScore`/`whiteOffensiveScore` used by
  OvD ELSEWHERE — you must DECOUPLE the O/D accumulators from `total` before lifting shelter (the clamp+smear
  that killed PASSER_V2).
- **SOBER CAVEAT:** KS re-work has been GAME-NEUTRAL every time. Expect cleanliness/tunability, NOT direct Elo.
  Verify byte-id gate-off + per-class collapse profile, exactly like PASSER_V3.

## THREE PATHS (decision pending)
- **A — full KS consolidation** (one home; de-king attackingLayer OR route OvD-king-zone through one realizability
  gate). Cleanest/biggest; riskiest (O/D decoupling); likely game-neutral + tunability upside.
- **B — cheap targeted gates NOW** (`KS_MIN_ATTACKERS≥2` to kill lone-queen fantasy + `MOD_KS_BACKING` on).
  30-min A/B, but only touches the unit-KS third.
- **C — go at the biggest channel** — the OvD/attackingLayer king-zone re-count (the 07-22 OvD lead),
  realizability-condition THAT.
- Lean: **B as a cheap probe** — if gating the KS third is game-neutral again, the money is in layer 4 (C) and we
  scope the consolidation (A) around layer 4, not unit-KS.

## STEP-0 PROBE (2026-07-23) [E] — the budget responds; material-backing is the right gate, control is circular
User's key refinement: **FEN-3's KS is DIRECTIONALLY CORRECT** — SF18 +3.04 sees white better DESPITE being down
material, i.e. the attack genuinely beats the material deficit. So the goal is **magnitude/blowup CONTROL toward
SF's modest positive, NOT killing KS** (keep #4 fully credited, keep #3 positive). Probe on the current binary
(dissect_fen, env works):
| config | FEN3 (SF +3.04) | FEN4 (SF +8.69) |
|---|---|---|
| base | +10.75 | +10.52 |
| KS_MIN_ATTACKERS=2 | +10.75 (NO-OP, multi-attacker) | +10.52 |
| **MOD_KS_BACKING (300/600)** | **+7.15 (−3.6)** | **+9.17 (−1.35)** |
| MOD_KS_CONTROL (300/600) | +17.95 (WORSE) | +13.22 |
- **`MOD_KS_BACKING` DISCRIMINATES** (contra my earlier worry that material can't tell 3 from 4): it damps the
  phantom (#3, −3.6) MORE than the real one (#4, −1.35) because #3's attacker is down MORE material, and it keys
  on the DEGREE (`min(0, threat_side_edge)`). #4 lands +9.17 ≈ SF. Saturates by 300 (can't fully fix #3, still
  +7.15) ⇒ need a STRONGER/FINER version acting on the WHOLE budget, not just unit-KS.
- **`MOD_KS_CONTROL` = WRONG DIRECTION** (amplifies) — confirms fable's circularity (built from the same king-
  zone attackingLayer). Exclude, or invert.
- **`KS_MIN_ATTACKERS≥2` = no-op for multi-attacker phantoms** (still useful for lone-queen ones).
- ⇒ The realizability gate should be **material-backing-based** (MOD_KS_BACKING-like), stronger, on the whole
  king-danger budget. This is the gate the consolidation should build. Budget responds ⇒ consolidation viable.

## Housekeeping
- The `ENABLE_KS_DEBUG` dump (search_engine.h + reg) is currently placed in the DEAD `latent_threat` fn → never
  fires. To use it, relocate the dump into the LIVE `king_safety_danger` (per-king unit sub-components).
