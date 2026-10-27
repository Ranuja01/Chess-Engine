# Pawn over-read REDIRECT + two new gated levers (2026-07-19)

**Big finding this session: the "pt_pawns / passed-pawn over-read" collapse class was MIS-SCOPED. The
dominant over-read is over-positive ENDGAME NON-PASSED pawn PLACEMENT, not passed-pawn realizability.**
Read with `collapse-reduction-ledger.md` and `passed-pawn-subsystem-map-2026-07-18.md`.

## What shipped first this session
- **DEF-5 committed `41c4123`** (`KS_SAFE_CHECK_DEF=5` default; narrow defensive-KS-only commit; capgains
  pin/tempo stayed shelved). byte-id changed to DEF-5-active; WAC node total 39,914,378 is the new default.

## The diagnostic redirect (how the target moved)
Built passer-v2 (below), then ran a deterministic firing screen on the passer exemplars (fen3/P2/P3) and
found the deflation was tiny. Decomposition via the SCALE_* knobs on `pt_pawns`:
- **`SCALE_PASSED_RANK=0` → NO change** to fen3's `pt_pawns` (+4.97). The passed-pawn rank boost contributes
  ~**0** to the over-read.
- All rank tables together ≈ 0.58; structure (wall/chain) = 0; `SCALE_PLACE_PAWN=0` no change;
  `SCALE_ATTACK_LAYER=50` only −0.25. The **~+4.4 residual is general pawn PLACEMENT** (attacking-layer
  central control + the hardcoded structural chaining bonuses +100/+135/+50 in `evaluate_pawns_endgame`
  ~3053-3060 + the endgame rank table), NOT the passer machinery.
- **Root of the mis-scope:** the class was diagnosed from *"our `pt_pawns` +5..9 vs SF's `Passed` ~0."* But
  `br_pt_pawns` (accumulator at 6253/6606) is the ENTIRE pawn-evaluator output, while SF's `Passed` is passers
  only — apples-to-oranges, so a big gap is guaranteed regardless of passer accuracy.

## SF11 pawn study (agent, `stockfish_11/.../src/pawns.cpp`)
- SF gives **~0 raw advancement/placement to a NON-PASSED pawn.** Endgame non-passed value is dominated by
  **penalties** (Doubled `S(11,56)`, WeakLever `S(0,56)`, Backward `S(9,24)`, Isolated `S(5,15)` — all
  eg-heavy). The only positive term, Connected `{0,7,8,12,29,48,86}`, is **mg-heavy** (`v*(r-2)/4` eg ramp) —
  only leaks into eg for ADVANCED pawns. A lone advanced non-passed pawn gets **no positive placement**.
- Ethereal matches qualitatively (`PassedPawn[canAdvance][safeAdvance][rank]` for passers; structure penalties
  eg-heavy; connected rank-scaled mg-dominant). Ethereal source NOT in workspace (values from knowledge).
- **Our design is the opposite:** large POSITIVE endgame non-passed placement. That is the over-read.
- Design implication (how to still encourage advancing without SF's deep search): SF relies on
  search + big PASSED bonus + CANDIDATE-passer half-bonus. We have shallower search (~2000, EBF~3.2) so we
  keep SOME advancement heuristic, but it should be **candidate-conditioned** (reuse `getPPIncrement`) and
  much smaller, not a flat rank table. (Table reshape = the deferred "second measure"; not built yet.)

## Two new GATED levers built this session (both default-off = byte-identical; WAC 39,914,378 unchanged)
1. **`ENABLE_PASSER_V2`** — passer-realizability consolidation. Two complementary pillars: `passer_danger`
   (blockade D1 + path-attack D2, both phases; **D4 king term zeroed**) + `passer_realizability_delta`
   (all-phases king-race, sole king authority; AE deep-eg copy auto-skips via `ENABLE_PASSER_KRACE_MG`).
   Also drops `evaluate_kings_endgame` king-attacks-passer ±150/±125, and clamps the capgains pawn-rank read
   (`CAPG_PAWN_RANK_CLAMP=275`). Firing screen: small correct deflation, P3 holds. (The scattered midgame
   ±100 dedup was NOT done — the over-read turned out endgame/placement, so it was moot.)
2. **`ENABLE_NPEDGE_DAMP_EG`** — endgame extension of the existing `NPEDGE_DAMP` (which damps the
   pawn-placement claim `br_pt_pawns` when a pawn lead is NOT backed by non-pawn material). Base is midgame-only
   because a pure pawn endgame (KPK) has npedge~0 yet converts. **Endgame guard:** only damp when the DEFENDER
   still holds >= `NPEDGE_EG_PIECE_FLOOR` (=3250, one minor) of non-pawn material — i.e. there is a piece to
   over-value our pawns against (user's target: "up pawns but the opponent's bishop actually compensates").
   Pure pawn endgames spared. Reuses `NPEDGE_DAMP_LO/HI/MAX/TQUIET`.

## Deterministic screen (our-POV totals; SF18 target)
| FEN | baseline | eg-damp default (MAX90) | eg-damp strong (MAX200/HI4000) | SF18 |
|---|---|---|---|---|
| fen3 (deflate) | +5.58 | +3.83 | +1.70 | ~0 |
| P2 (deflate) | +4.91 | +2.41 | -0.65 | ~0 |
| P3 (hold, +4.84) | +9.72 | +6.86 (holds) | +3.36 (OVER-damped) | +4.84 |

Default mag (MAX90) is the operating point: deflates over-reads, P3 stays winning. Strong over-damps P3.
The damp is blunt (P3's black has R vs white's B → npedge reads "unbacked" though white wins) — games decide net.

**`passers.csv` no-regression guard (150 pos, move-match / mean|our-sf| cp):** baseline 74/150 (49.3%)/582;
**eg-damp 76/150 (50.7%)/552** (improves both); passer-v2 76/150 (50.7%)/651 (move-match holds). Both pass.

## Nighttime A/B battery (RUNNING, seed 0, 200g each vs SF@2400, conc3, ~5.3h)
tags: `ab_base_s0` (shipped DEF-5), `ab_damp_s0` (`ENABLE_NPEDGE_DAMP_EG=1`), `ab_v2_s0` (`ENABLE_PASSER_V2=1`),
`ab_both_s0` (both). Judge: collapse count/profile + score + no KS/DEF-5 resurface. Seed 1 + magnitude tuning
(damp MAX ~120-140, floor) tomorrow. Nothing committed beyond DEF-5.

## Next
Read the 4 `games/ab_*_s0/collapses.csv`+`results.csv` on completion; pick winner(s); seed-1 confirm; tune damp
magnitude; consider the candidate-conditioned table reshape as the second measure. Update the ledger each round.
