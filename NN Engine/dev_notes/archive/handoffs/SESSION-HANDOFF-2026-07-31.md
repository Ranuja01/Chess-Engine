# Session handoff — 2026-07-31 (eval arc: the material bug and what it unblocked)

**Read this top block first.** Previous handoff: `SESSION-HANDOFF-2026-07-30.md` (search arc, 0-for-9).

---

## 🏆 HEADLINE — STS 1746, the highest ever recorded (prior best 1647)

```
ENABLE_MATERIAL_COUNT_FIX=1 ENABLE_PASSER_V3=1 MOD_KS_REALIZ=128
```
**STS 1746 (58.2%) · WAC 246 · nodes 35,089,668 (−2.5% vs default)**
Default baseline for comparison: **254 / 35,982,407 / STS 1629**.

## 🏆🏆 GAME-VALIDATED — SPRT PASSED (2026-07-31)
```
+213 -160 =130 of 503  (55.3%)   elo ~ +36.7 +/- 35.7   LLR +3.034 (bound +2.944)
elo0=0 elo1=20, LIGHTNING, conc 4, tag `sprt_ksr128`
```
**Largest confirmed eval win to date** (gravcap +33.0). ⚠️ The CI is WIDE — the claim is the SPRT decision
(≥20 Elo at α=0.05), not the point estimate. ☠️ At **game 47 this run read −87 Elo** and only crossed zero
at ~g128 ⇒ **never peek early.**
★★ **STS (+117) and games agree while the BANK dissented** (`ks` class +14.3 worse) ⇒ on eval changes weight
**games > STS > bank MSE**; the bank measures ACCURACY, this bought DISCRIMINATION.
▶️ **IN FLIGHT:** arm B `sprt_candbase` = `ENABLE_MATERIAL_COUNT_FIX=1 ENABLE_PASSER_V3=1` alone vs defaults.
**A−B splits the credit.** If most of the +37 is B ⇒ `MOD_KS_REALIZ` carries the sacrificial-attack risk for
little return (consider dropping). If the lever carries it ⇒ the **smooth-form KS redesign is the priority lane.**

## ✅✅ SHIPPED 2026-08-01 — this bundle is now the DEFAULT BUILD
`ENABLE_MATERIAL_COUNT_FIX=1` · `ENABLE_PASSER_V3=1` · `MOD_KS_REALIZ=128` are now the code defaults in
`search_engine.h`. **New baseline fingerprint `246 / 35,089,668 / EBF 3.846 / STS 1746`**, verified on the
shipped build (WAC + nodes reproduce the env-knob run **to the node**, STS to the point).
⚠️ **WAC 254 → 246 is the INTENDED trade** (WAC at ceiling; STS carries the positional gap) — not a bug.
▶️ Old default reachable without a rebuild: `ENABLE_MATERIAL_COUNT_FIX=0 ENABLE_PASSER_V3=0 MOD_KS_REALIZ=0`.
▶️ NOT yet done: NPS re-measure (median-of-3) and a d12 fingerprint; both stale for the new default.

### Final arm table (all vs code defaults, same venue)
| arm | STS | games |
|---|---|---|
| **A** = fix + V3 + lever (**SHIPPED**) | 1746 | **+36.7, SPRT H1 accepted, 503g** |
| B = fix + V3 | 1658 | +25.4 [−7,+58], 439g |
| B + `RFP_MAX_DEPTH=8` | 1689 | +19.5 ±35.8, 500g |
| lever alone, no fix | 1467 | **0.0 ±35.8, 500g** |
☠️ **The 279-pt bench swing does NOT reproduce in games** — the lever alone is NEUTRAL, not harmful, and
A−B = +11 is unresolvable (needs ~4000 games). Only the bundle has a verdict.
🚨 **Forward consequence: big negative STS OVERSTATES harm** ⇒ candidates killed on bench negatives alone
(search 0-for-13 / 0-for-9) may deserve re-testing.

## 🔬 The causal chain (this is the important part)

| config | STS |
|---|---|
| default | 1629 |
| material fix only | 1634 (neutral) |
| V3 only | 1663 |
| **candidate baseline** (V3 + material fix) | 1658 |
| V3 + `MOD_KS_REALIZ=128`, **NO** fix | ☠️ **1467** |
| V3 + fix + `MOD_KS_REALIZ=128` | 🏆 **1746** |

**The same lever is −196 STS without the material fix and +88 with it — a 279-point swing.**
`MOD_KS_REALIZ` was banked as "DISPROVEN"; it was judged against a material edge wrong in 46% of positions.

★ **The owner predicted this before any data** — *"it reopens some items that were formerly utilizing the
double counting"* — at a point when the fix measured **neutral alone (+5 STS)** and I had reported it as
"not measurably helping". **A fix's value can be entirely in what it unblocks; testing it in isolation
understates it to zero.**

## 🌙 AUTONOMOUS OVERNIGHT PLAN (~10h, sequential — games must run ALONE)
Measured rate: **119 games/hour at conc 4** ⇒ 500 games ≈ 4h 15m. All three arms use the existing
prompt-free `gate` sub (p1 = candidate, p2 = code defaults), so no new tooling is needed.
☠️ **NO COMMITS overnight** (owner cannot respond). ☠️ Never run a bench beside a game run — LIGHTNING is
time-controlled, so a concurrent process changes the result.
⚠️ **NOT included by design: the direct A-vs-B lever run.** Its expected effect is ≈+11 and 500 games
resolves only ±45 ⇒ it would cost 4h to return "inconclusive". Resolving it needs ~4000 games.

### ✅ Run 1 RESULT (2026-08-01, 500 games, 4h14m) — RFP8 does NOT convert
```
brfp8 vs base:  +213 -185 =102 of 500  (52.8%)   elo ~ +19.5 +/- 35.8   INCONCLUSIVE (hit cap)
```
Common scale vs code defaults: **A +36.7 (503g) · B +25.4 (439g) · B+RFP8 +19.5 (500g)**.
⇒ B+RFP8 lands **BELOW** B (−5.9, inside noise) ⇒ deeper reverse-futility is **neutral in games**.
☠️ **The +31 STS was noise, exactly as the ~100-pt jaggedness band predicted.** Three independent
signals agree it was never real: (1) it sat inside the noise band, (2) the plateau check could not
corroborate it because `RFP_MAX_DEPTH` **saturates at ≥8 at d10**, (3) games came back neutral.
▶️ **DECISION: ship B, PARK the pruning lane.**
★ Side effect: this WEAKENS the case against the lever. The `RFP_MAX_DEPTH=8` interaction (−167 STS on A)
was the main live objection to `MOD_KS_REALIZ` — but if deeper RFP is worth nothing in games, the lever
blocking it costs nothing real. Lever ledger is back to: **≈+11 marginal, CI spans 0, no game-level side
effect found** ([[snapshot-signature-mining-overcounts-30x]]).
▶️ Run 3 (RFP 6 vs 8 at d12) is now **near-pointless as a ship question** — keep only for depth-scaling
interest.

### Run 1 (~4h15m) — the SHIP-DECISION arm [SPEC AS PLANNED]
```
gate 'ENABLE_MATERIAL_COUNT_FIX=1 ENABLE_PASSER_V3=1 RFP_MAX_DEPTH=8' brfp8 sprt_brfp8 500 20 4
```
Does the +31 STS from deeper pruning on the clean eval convert to Elo? Compare against the SAME opponent
as the other arms: **A = +36.7 (503g) · B = +25.4 (439g)**.
▶️ **≥ +35** ⇒ B+RFP8 matches/beats A **without** the lever's liabilities ⇒ **ship B+RFP8, drop the lever.**
▶️ **≈ +25** ⇒ RFP8 adds nothing in games; the +31 STS was noise ⇒ ship B, park pruning.
▶️ **< +15** ⇒ deeper RFP actively hurts in games despite STS ⇒ another instance of bench≠games.

### Run 2 (~4h15m) — the INTERACTION PROOF
```
gate 'MOD_KS_REALIZ=128' leveralone sprt_leveralone 500 20 4
```
The lever WITHOUT the material fix. Bench says **−196 STS (1467)**; if games agree it is clearly negative
while A is +36.7, the interaction is proven **in games** from three directions — no component carries the
result alone, only the combination does. That is the strongest possible form of the
"a fix's value is entirely in what it unblocks" claim.
▶️ Clearly negative ⇒ proof complete. ▶️ Neutral/positive ⇒ the whole causal story needs revisiting.

### Run 3 (~1h30m, single core, ONLY after the game runs finish)
```
sts rfp6_d12 ENABLE_MATERIAL_COUNT_FIX=1 ENABLE_PASSER_V3=1 RFP_MAX_DEPTH=6  MAX_DEPTH=12
sts rfp8_d12 ENABLE_MATERIAL_COUNT_FIX=1 ENABLE_PASSER_V3=1 RFP_MAX_DEPTH=8  MAX_DEPTH=12
```
d10 CANNOT discriminate here — `RFP_MAX_DEPTH` **saturates at ≥8** (8/10/12 all = 1689 exactly), so the
d10 sweep cannot tell "8 is optimal" from "the bench went blind". d12 is the first depth where the cap
binds; real games reach **d16-22**.

### Chaining
Launch each as a background task and let the completion notification trigger the next — `ScheduleWakeup`
is unreliable here. Read every result with the Read tool (shell `grep`/`cat` prompts the owner).

## 🐛 The bug itself
`cpp_bitboard.cpp` ~L6605: the phase-blend path calls **both** `evaluate_X_midgame` and `_endgame` on the
same square. The returned scores are blended; the `whitePieceVal/blackPieceVal +=` **side effect is not**,
so pawns/rooks/queens/kings are counted **twice** when `phase_score > 40`. Blend sites: pawns 6605, rooks
6682, queens 6731, kings 6765/7055. Knights/bishops are single-call. Audit: **277/600 positions wrong**,
per-side excess ~3.8 pawns (max 30); 323 exact because symmetric counts cancel.
Fix = recompute both sums from the bitboards after the piece loops (L6812/L7117), before every consumer.
Live consumer = **`piece_value_boost`** (`PV_BOOST_MAG=10000`, a RATIO of those accumulators).
✅ OvD does **not** double-count (48 midgame sites, 0 endgame). Kaufman reads bitboards directly.

## ⚠️ The guard check does NOT confirm the STS gain
Bank (1634 SF18-anchored), candidate → +KSR=128: **MAE 12.546 → 12.543 (flat)**, **MSE 347.0 → 350.3**.
`ks` class **367.1 → 381.4 (+14.3 WORSE)** · `open` 368.4 → 378.4 · `mid` 277.2 → 273.6 ·
`end` **377.5 → 377.5 IDENTICAL** ✓ (validity: `KS_PHASE_ZERO=104` already zeroes KS there).
⇒ Move choice up, eval accuracy flat/slightly down. **STS is one corpus; games are the arbiter.**

## ★★ The regression is a diagnosed TAIL of SACRIFICIAL ATTACKS
MAE flat + MSE up ⇒ tail. Confirmed: **44 worse / 50 better / 1540 unchanged**; the 44 contribute
**+16058 MSE** vs −10687. Regressed profile: mean `|KS|` 2.37→1.58, **mean phase 24 (opening)**.
Every top regression: **the attacking side is DOWN 3-5 material** — i.e. a **sacrifice**, where the attack
is real compensation and damping it is exactly wrong. `MOD_KS_REALIZ` damps on `threat_side_edge < 0`.
★ **`KS_REALIZ_FLOOR=128` is BINDING on these** (KS −4.68 → −2.34 = exactly 0.5×) ⇒ the surgical lever.
▶️ **IN FLIGHT: `KS_REALIZ_FLOOR` sweep 160/192/224** at `MOD_KS_REALIZ=128`.

## ✅ THE GAIN IS AN EVAL EFFECT, NOT EXTRA SEARCH (owner's catch)
At fixed depth KSR used **+3.8% nodes** (33,795,226 → 35,089,668), so the +88 was confounded with search
thoroughness. Re-measured at **fixed nodes (`NODE_LIMIT=100000`, both arms)**:

| regime | baseline | +KSR=128 | Δ |
|---|---|---|---|
| fixed depth | 1658 | 1746 | +88 |
| **fixed nodes (100k)** | **1498** | **1621** | **+123** |

⇒ **Removing the node advantage makes the gain BIGGER.** Eval quality dominates when node-starved — which
is the real-game (clocked) condition. **Two independent regimes agree in sign and magnitude = replication.**

## ✅ CONTROL: it is NOT just "less KS"
`KING_SAFETY_MAG` with **no** KSR: 3000 → **1658**, 2400 → **1549**, 2000 → **1649**. Uniform magnitude cuts
land at/below baseline while conditional damping gives 1746 ⇒ **the conditionality (damp only when the
attacker is materially BEHIND) is doing the work.**
⚠️ **But that sweep also exposes STS's jaggedness: a 109-point drop then a 100-point recovery on a smooth
knob.** Treat any STS delta under ~100 with suspicion. (The fixed-node replication is why +88/+123 survives
this caveat.)

## ☠️ THE TAIL CANNOT BE SEPARATED VIA THE FLOOR
`KS_REALIZ_FLOOR` at `MOD_KS_REALIZ=128`: 128 → **1746**, 160 → 1669, 192 → 1655, 224 → 1622
(baseline, i.e. no damping, 1658). **Non-monotone, and only the MOST aggressive setting wins** — mild
damping is worse than none. ⇒ the +88 and the 44 sacrificial-attack regressions come from the *same*
aggressive damping; the floor is a blunt clamp and cannot tell a fantasy attack from a real sacrifice.

## ★★ REPRESENTATION: our jagged knob responses are a CLAMP-STACK problem (owner's hypothesis, supported)
Our KS pipeline composes **hard cutoff → integer lookup → integer shifts → two clamps**:
`KS_PHASE_ZERO=104` (hard 0 above) · `ks * ks_phase_taper[phase]/256` · `mod_gain(...) >>8/>>12` ·
`MOD_FLOOR=128`/`MOD_CEIL=512` · `KS_REALIZ_FLOOR=128` (**observed binding**: KS −4.68 → −2.34 = exactly 0.5×).
SF15.1's entire transform is **one gate then a smooth polynomial**:
`if (kingDanger > 100) score -= make_score(kingDanger*kingDanger/4096, kingDanger/16);`
Each clamp creates a plateau (zero derivative) and an edge (discontinuity); stacked, they produce knobs that
do nothing across a range then jump. ★ Proven instance: `LMR_REMDEPTH_SCALE` was literally a two-valued step
function from integer division (200 and 250 byte-identical).
▶️ **Design direction: one gate + smooth polynomial, phase-blend the OUTPUT rather than clamping intermediates.**
That is also the only way to distinguish "under-backed because thin" from "under-backed because sacrificed".

## ✅ `MOD_KS_BACKING` RESOLVED — same lever, weaker floor, do NOT stack
| config (on candidate baseline 1658) | STS |
|---|---|
| `MOD_KS_BACKING=256` | 1705 (+47 — **under the ~100 jaggedness band, not a claim**) |
| `MOD_KS_BACKING=512` | 1575 |
| `MOD_KS_REALIZ=128` | 🏆 **1746** |
| **both** | ☠️ **1651 = −95 vs `MOD_KS_REALIZ` alone** |

Both damp on the same `min(0, threat_side_edge)` in `evaluate_king_safety` (backing ~L5384, realiz ~L5399,
applied sequentially to the same `ks`) ⇒ stacking multiplies the damps (~0.25×) and over-damps.
★★ **At strength 512 the two are BYTE-IDENTICAL** — 251/300 solves, 34,876,906 nodes, EBF 3.771, cutoff
histogram equal to the last digit. Both saturate to the same 0.5× floor (`MOD_FLOOR=128`/256 vs
`KS_REALIZ_FLOOR=128`/256) and `sig ≤ 0` forbids boosting. ⇒ **one lever, two floors** — exactly what
`search_engine.h` L1130-1134 documented; `MOD_KS_REALIZ` exists to cut below backing's 0.5× wall.
★ Another clamp-stack instance: two nominally distinct knobs collapse to one behaviour because a clamp eats
the difference. Guard concurs (BACKING=256: MSE 350.9 / `ks` 385.1 / `end` 377.5 identical).
⇒ **`MOD_KS_REALIZ=128` alone remains the best config. Leave `MOD_KS_BACKING=0`.**

## ☠️ SAFE-CHECKS ARE *NOT* REOPENED BY THE FIX
The safe-check table died from a **UNITS error** (SF15+ ~356-385 internal per displayed pawn vs `PieceValue`
126-208), not from the material edge — and it does not read the accumulators. **Only the material-edge
consumers are reopened:** `MOD_KS_REALIZ` ✅ done · **`MOD_KS_BACKING` ▶️ untested** · `ENABLE_CAPG_REALIZ`
("identity" on the bad input) · imbalance realizability (L6946/6954). Do not re-test the whole KS backlog on
this — that is the "reintroduce a knob that failed for a reason" trap.

## ▶️ Next steps
1. ✅ DONE — `KS_REALIZ_FLOOR` sweep (128 wins) and `MOD_KS_BACKING` (rejected, see above).
2. **`ENABLE_CAPG_REALIZ`** + imbalance realizability (L6946/6954) — the last two reopened consumers.
3. Joint tune (`fit_bench_guarded.py`: corpus proposes, bench disposes) over `piece_value_boost` + KS +
   V3 + OvD **together**. ⚠️ Raw `ks_fit_wholesystem` is **4-for-4 bench-negative** — do not run it blind.
4. **Games**: ledger venue (200g SF@2400 conc3, **3+ seeds** — 2 was underpowered).

## Tooling built this session (all in `diagnostics/`, uncommitted)
`_material_audit.py` · `_taper_headroom.py` · `_v3_gap.py` · `_collapse_full_probe.py` ·
`_collapse_v3_causal.py` · `_collapse_class.py` · `_class_guard.py` · `_ksr_tail.py` ·
`_ledger_report.py` · `_ledger_validity.py`. `refresh_bank_ours.py` now takes `KEY=VAL` (2.7s for 3726 rows).

## ☠️ Corrections banked this session
- **V3's "bench-negative on the NEW baseline" is WRONG** — read off WAC (−12); STS is **+34**.
- **Piece-value tapering measured at ~0.06pp** over 1634 positions ⇒ **not a lever** (ours flat 3.25 ≈
  Ethereal's ENDGAME N/P ratio 3.30; SF 6.20→4.11, Ethereal 5.20→3.30, both taper via the PAWN rising).
- **SF15.1 agrees with SF18 where SF11 does not** ⇒ **always use two classical witnesses**.
- V3 removes **~52%** of passer-specific bank error (854 positions, bit-identical no-passer control) and is
  **eval-positive on the collapse positions themselves** (95 closer / 58 further) ⇒ it does NOT cause the
  collapses; the earlier "V3 fails the ledger" call was underpowered (Poisson ±8 on counts of ~60) and is
  **retracted**.
