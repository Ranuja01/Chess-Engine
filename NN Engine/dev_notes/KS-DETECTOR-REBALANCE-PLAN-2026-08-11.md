# KS detector rebalance — the plan (2026-08-11)

The KS lever, derived today from a decomposition + a source-verified comparison to the giants (SF11, SF15.1
classical, Ethereal). **Nothing built yet — this is the execution plan.**

## The diagnosis (data, not theory)
- `_ks_channel_decomp.py` (92 `ks_attack` over-read collapses vs 171 `positional` controls): the KS over-read
  is **Channel-1 (`king_safety`) proximity**, not collinearity. king_safety fires **0.897 pawns on the
  false alarms** vs OvD-king-leak 0.08 and central ~0 (central ruled out — a castled king isn't central).
- `_ks_c1_decomp.py` (sub-decomposition): **PROXIMITY dominates** — attacker weights `KS_ATT_*` = 0.440
  (half the over-firing), attack-count 0.265, weak 0.235; **`KS_SAFE_CHECK` = 0.010 (~nothing)** on the false
  alarms and is the ONLY sub-detector that fires *less* on over-reads than control ⇒ it's the one genuinely
  discriminating signal, and it's structurally tiny.
- ⇒ **We fire on mere *proximity* (pieces near the king) even when the squares are defended, and the
  genuine-threat detector is drowned.** It's a DISCRIMINATION failure (`every-eval-term-error-is-bidirectional`:
  KS 13 over / 21 under), not a magnitude one — global scale moves fail in both directions.

## The giants (source-verified, `_task` fable report 2026-08-11) — we're calibrated BACKWARDS
| | per-attacker weight | safe-check weight | ratio |
|---|---|---|---|
| SF11 / SF15.1 | N81/B52/R44/Q10 | Q780–1292 R1080–1886 | safe-check **15–25×** |
| Ethereal | N48/B24/R36/Q30 | Q93/R90/B59/N112 | **2–4×** |
| **ours** | N2/B2/R3/Q5 | `KS_SAFE_CHECK=3` | **~1×** |

- In both giants **genuine threat dominates**; in ours it's co-equal with proximity. That inversion IS the
  over-read, measured against the reference.
- Their defender-awareness is **binary event-gating**: `attackedBy2` (a square defended twice is *never*
  weak — SF & Ethereal), double-pawn-defended squares excluded from the zone, safe-check requires
  under-defense. **We ported the weak "K/Q-only defender" part but NOT the `attackedBy2` part.** Standard, missing.
- Coordination: Ethereal hard-gates ≥2 attackers; SF uses `count × weight` **product** (1 attacker small, 2
  jumps super-linear). Ours is a flat additive sum, gate off (`KS_MIN_ATTACKERS=0`, and it's **inert anyway**
  — the queen exception drops the bar to 1 and over-reads have queens). So we over-credit lone attackers.
- SF's decade of classical-KS work (SF1→SF11) added **discrimination detectors** (safe-check taxonomy, weak
  algebra, pins, flank, no-queen, mobility-in-danger) — **not bigger proximity weights.** Direction confirmed.

## ★★★ THE PRINCIPLE: solve the KS detectors as ONE coherent unit
The detectors are an **interdependent system, mis-calibrated as a whole.** One-at-a-time is proven to
mislead — `sc4 alone −112 but sc5of3 +19`; `CHECK_V2` +2.6% → −166 across de-king (channel law: a lever's
sign depends on the other live channels). Cranking safe-check ALONE is toxic (adds danger to an over-reading
stack = the additive-KS **0-for-9** trap); the giants' safe-check dominance works only *because* proximity is
correspondingly small — **it's the RATIO, moved together.** ⇒ Build a coherent structural UNIT and validate it
as a whole; do NOT tune N knobs independently, and do NOT game-test each piece.

## The unit (build all behind ONE gate, default off = byte-identical)
1. **Defender-aware attacker weighting (the unique lever, KS-local).** Replace `KS_ATT_type × presence` with
   `KS_ATT_type × contested-fraction` — the zone squares a piece attacks where `attackers > defenders`
   (per-square `overload` at `king_safety_danger`, already computed at weight 0) over its footprint. A
   fully-defended proximity attacker decays → 0; a converging/breakthrough attack keeps full weight. This is
   **graded per-square contest — finer than the giants' binary `attackedBy2`** (uniquely ours), it's
   **redistributive** (dodges 0-for-9), per-square (fixes over AND under), and a *read* of the existing
   bitmask (no ripple). **Build BOTH forms** (contested-fraction vs breakthrough-count) — cannot pick on theory.
2. **`attackedBy2` in the weak-square test** (standard; both giants). One added condition in `king_safety_danger`.
3. **Ratio rebalance**: shift safe-check up / proximity down toward the giants', HELD so total firing on the
   92/171 set stays ≈constant (redistributive, NOT additive).
- ⏸️ Coordination product/gate (`KS_ATT_PRODUCT`, `KS_MIN_ATTACKERS`) — LATER / maybe. Product over-fired
  before; the gate is inert with queens. The defender-aware weighting subsumes much of their intent.

## Screen order (fast deterministic FIRST, games LAST)
1. **`_ks_c1_decomp.py` / `_ks_channel_decomp.py`** — does the unit selectively DROP the over-read firing while
   sparing control/genuine? If not, iterate the *design* (this is the cheap inner loop).
2. Move-match (`_move_match_arms.py`, target+holdout) + **held-regret** (`_regret_tune_broad.py` seed eval on
   the game set) + balanced STS (orig+mirror) + colour/file symmetry (`_eval_symmetry.py`) + byte-id at default.
3. Small internal calibration of the unit's ratio (a few of its own knobs, on held-regret/firing-decomp).
4. Fold into the OvD+central bundle; confirm the BUNDLE in one tournament.

## Ripple / consumer caveats (from the fable map)
- **DO NOT change how `attack_bitmasks` is POPULATED** (x-rays etc.) — it feeds passers, capgains, evasion
  (high ripple). **Reading it inside KS is KS-local (safe).** The whole unit above is reads only.
- `attackingLayer` changes move OvD + central + placement at once (the de-king channel) — leave it.
- The dead `num_attackers[]`/`num_supporters[]` arrays: NOT needed (the live per-square popcount over
  `attack_bitmasks` already gives the contest); don't pay the per-loop population cost for KS alone.

## DO-NOT (from `ks-twelve-attempt-history-and-the-channel-law`)
Additive KS is 0-for-9. No one-at-a-time knob sweeps. No `KS_MIN_ATTACKERS` (inert w/ queen). No
`KS_ATT_PRODUCT`/`KS_OVERLOAD` as a blanket weight (over-fire). No porting SF verbatim — inspired, uniquely
ours. Validate with the four king-credit channels frozen.

## Tools
`_ks_channel_decomp.py` (channel firing over/control), `_ks_c1_decomp.py` (Channel-1 sub-decomposition) —
both deterministic, no SF. Pattern to copy for the bounded/gated build: `ovd_imbalance` / `central_bounded`
in `cpp_bitboard.cpp`.
</content>
