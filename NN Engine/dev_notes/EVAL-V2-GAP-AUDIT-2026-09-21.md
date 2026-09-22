# EVAL v2 — SYSTEMATIC GAP AUDIT vs SF11 / SF15.1 / Ethereal / Weiss  ★ 2026-09-21

Sources read directly (not summaries) by the `engine-contrast` agent: SF11 and SF15.1 local trees;
Ethereal and Weiss fetched from GitHub **`master`** — ⚠️ NOT the register's pinned commits (@0e47e9b /
@c735b8f), so their constants and line numbers may differ from the pins.

**Why this exists.** Two gaps were found by ACCIDENT this month — KS has no shelter/storm, and v2's passer
carries SF's rank-weight multiplier `w` but not the LADDER it multiplies. Both were **mechanisms, not
bonuses**. This is the systematic sweep for the rest.

**Counting rule:** SF11 + SF15.1 are ONE lineage. "3/4 [2 designs]" = three engines, two independent designs.
**M** = mechanism (changes how something is valued, or conditions other terms). **B** = bonus (a constant on
a detector). ☠️ **A missing MECHANISM outranks a missing BONUS** — mechanisms unblock other terms.

⚠️ **Excluded as already measured null for us** (do not re-surface): central · bishop pair · space · threats
as a whole · Kaufman · tier-2b · rook files · weak-unopposed · material taper · long diagonal ·
`MOB_V2_SAFE` · tempo · connected pawns (rank-keyed AND rank-flat). Plus the register's DEAD list.

---

## ★★★★ TIER 1 — ARCHITECTURAL. One change unblocks a cluster.

### A1 — **v2 has no top-level `(mg,eg)` pair.** 4/4 references have one.
Every v2 term blends internally with `phase256` and returns a scalar; `total` is already blended
(`eval_v2.cpp:889, 1062, 1228, 1352, 1610, 1884`). All four references accumulate packed `S(mg,eg)` and
interpolate **once** (SF11 `evaluate.cpp:817-820`).
**Blocks, simultaneously:**
- **I1/A2 — the ENDGAME SCALE FACTOR. 4/4 UNIVERSAL, and v2 cannot express it at all.** Our only
  drawishness is `draw_class` → `total = 0`: binary. We cannot say "won but hard to convert" or "drawish".
  SF11 `evaluate.cpp:743-761`, SF15.1 `:908-947`, Ethereal `evaluateScaleFactor`, Weiss `ScaleFactor`.
- **L2/A3 — a TAPERED PST. 4/4.** `whitePlacementLayer` is one `int` per square
  (`cpp_bitboard.cpp:350-359`) ⇒ no piece can have a different endgame square preference; **no endgame
  king centralisation**.
- **K2/A8 — king safety's two-leg transform. 3/4.** SF `S(kD²/4096, kD/16)`; ours is one curve, unblended.
- **Any eg-LEG-ONLY term** — SF's `minPawnDist`, and the winnability/complexity family wanted for a
  reworked OvD. ★ Ethereal's `evaluateComplexity` is **king-free** (confirmed by full line read) which is
  exactly what de-duplicates it from KS — but it is DEFINED on the eg leg.
☠️ **Interpolating once CANNOT be byte-identical** — see the design block at the top of `eval_v2.cpp`.
✅✅ **STATUS: PLUMBING COMPLETE 2026-09-21.** `EvalPair` · `eval_blend` · `Config::EVAL_V2_PAIR` (default
0 = the shipped path). All nine scorers converted, caller wired, ONE interpolation at the end; phase-flat
terms (Kaufman, KS-A) enter both legs identically.
| gate | result |
|---|---|
| mode 0 | **`250 / 49,440,513 / EBF 4.031` — BYTE-IDENTICAL** |
| mode 1 | `246 / 50,631,624 / EBF 4.031` |
| bound, 4,000 positions | **max 3.00 mp** vs a 16 mp bound, **0 violations** (42% differ, mean 0.50 mp) |
| colour symmetry, mode 1 | **0/800** |
☠️ **The symmetry gate earned its keep:** the single blend first used `>> 8`, which rounds toward negative
infinity, so it broke `eval(mirror) == -eval` on the signed total — **474/800 violations, all 1 mp**. Fixed
to `/ 256` (truncates toward zero, sign-symmetric). ★ The per-term `>> 8` sites are CORRECT and must stay:
they blend non-negative per-side magnitudes and difference afterwards. **Once a quantity is SIGNED, `>> n`
is not a safe divide.**
⚠️ Blend-site count corrected to ~**16** (material is censused per TYPE, 6 — not per piece, 32).
▶️ **NEXT, now unblocked:** the endgame scale factor (I1/A2) · tapered PST (L2/A3) · KS phase legs (K2) ·
the winnability/OvD eg-leg direction. ⚠️ Mode 1 is a real behaviour change (−4 WAC solves, +2.4% nodes);
it needs its own measurement before it becomes the shipped default — do NOT flip it silently.

### A4 — no LAZY-EVAL BAND. 2/4 [1 design].
SF11 one threshold (`:790-793`), SF15.1 two (`:997`, `:1017`). ★ Matters beyond speed: the winnability
layer is *defined* on the near-balanced band; without one, any port of it fires everywhere.

### A5 — the replacement layer is DRAW-ONLY.
SF's material probe can return a VALUE function or a per-side SCALING function; v2 has `draw_class` (zero)
and the dead tier-2b. ⇒ cannot express "won but needs technique".

### A6 — no pawn / king-safety / material cache. 4/4 have one.
Speed only. ☠️ Never justify an eval change on NPS — listed because the Layer A split already exists and
the cache was scheduled (`eval_v2.cpp:2709-2711`).

### A7 — cross-term wiring. ⚠️ NOT a gap to close: "the wiring thesis was tested and is unsupported".

---

## ★★★ TIER 2 — 4/4 MECHANISMS, independent of A1

### P1 + P2 — the PASSER PATH-SAFETY LADDER. 4/4 [3 designs].
SF `evaluate.cpp:611-635`: `unsafeSquares` / `squaresToQueen` / `blockSq` empty ⇒ `k = 35/20/9/0`, `+5` for
own R/Q behind or a defended stop square, **all multiplied by `w = 5r−13`**. Ethereal
`PassedSafePromotionPath` + `PassedPawn[canAdvance][safeAdvance][rank]`. Weiss `PassedFreeAdv` /
`PassedBlocked`.
☠️ **v2 has the multiplier and not the ladder.** `passer_value_mp` (`:1013-1065`) is rank table + eg
king-distance + 50% candidate scaling, **unconditional** — it never asks whether the pawn can actually go.
⇒ This is why **P3 (rook/queen behind a passer, 3/4)** has nowhere to attach: in SF it is a RUNG on this
ladder, not a standalone bonus. Build the ladder first or the port is a third, invented form.
Inputs: attack maps exist but are built only inside the `:2664` gate and are not passed to the passer scorer.

✅✅ **STATUS: BUILT 2026-09-21** — `Config::PASSER_V2_PATH_PCT` (default 0 = absent). `SideAttacks wa, ba`
hoisted out of the attack-build block (the same move `MobAcc` already needed for the placement pass) and
passed to `passer_value_mp` as nullable pointers, so with the build off the ladder is ABSENT rather than
reading uninitialised stack.
☠️ **THE FREE VERSION DOES NOT EXIST, and the reason is worth keeping.** `PawnEntry` already carries
`blocked` (stop square holds an enemy PAWN) and `stop_held` (stop square attacked by an enemy PAWN), so the
obvious cheap ladder is to gate the passer on those — no new inputs, no hoist. It is **VACUOUS BY
CONSTRUCTION**: `passed` means no enemy pawn anywhere in the forward three-file span, and both of those
predicates require an enemy pawn INSIDE that span. They are identically zero on every passed pawn, and live
only on the candidates the scorer also prices. ⇒ the ladder for real passers MUST come from piece attacks.
★ Caught at design time by asking what already owns the signal — the gate would have been unfireable on the
very term it gated (`a-detector-gate-passes-vacuously-unless-the-term-is-proved-to-fire`).
⚠️ **Deviation from SF, deliberate:** we intersect with the SHARED attack map, which under `KS_V2_XRAY`
(on in the shipped config) sees bishops through queens and rooks through queens + own rooks. SF uses a
PLAIN `attackedBy`. Our `unsafe` set is therefore strictly larger and our `k` at most one rung more
pessimistic. Reusing the shared map beats a second build for one term; if the ladder ever reads too weak,
test `KS_V2_XRAY=0` first.
| gate | result |
|---|---|
| byte-identity at default | **`250 / 49,440,513 / EBF 4.031`** ✅ (and `250 / 59,549,832 / 4.080` without `RFP_MARGIN`, also exact) |
| colour symmetry at `PATH_PCT=100` | **0 / 800** ✅ |
| fire rate, play distribution, 4,000 pos | **25.3%** · median **93 mp** · p90 **662 mp** · max **2,069 mp** |
| signed mean over all positions | **−4.9 mp** ⇒ reshapes without shifting the mean |
☠️☠️ **VERDICT 2026-09-21: REJECTED ON MEASUREMENT. Seven arms, three channels, null-to-negative in all of
them.** This is the strongest-supported item the audit had — 4/4 universal, three independent designs, a
MECHANISM conditioning a large existing term rather than a bonus on a detector, ported from source rather
than from a summary, and it fires on a quarter of real positions with a 93 mp median swing.

d7 footprint regret, 6,000 play-distribution positions, read against a neutral arm (`ASPIRATION_DELTA=300`,
delta **−0.1522**, win% 50.0) because the tool's null is not zero:
| arm | regret delta | vs neutral | win% |
|---|---|---|---|
| `PATH_PCT=25` | −0.1154 | +0.037 worse | 50.2% |
| `PATH_PCT=50` | −0.2236 | −0.071 better | 50.8% |
| `PATH_PCT=100` | −0.1213 | +0.031 worse | 51.1% |
| `PATH=100 MAG=20` | **+0.0644** | **+0.217 worse** | **48.1%** |
| `PATH=100 MAG=30` | −0.0993 | +0.053 worse | 49.8% |
| `PATH=50 MAG=40` | +0.0229 | +0.175 worse | 50.1% |
Node channel: WAC `246 / 50,795,598 / EBF 4.032` vs base `250 / 49,440,513 / 4.031` ⇒ **−4 solves and +2.7%
MORE nodes** — the opposite of the pruning-headroom prediction. STS **1765 vs 1854 (−89)**, inside the ±150
floor but not positive.

★★★ **THE INFORMATIVE HALF: at CONSTANT passer mass the ladder is WORSE than flat, and that is the finding.**
The first three arms raise total passer weight (the ladder more than TRIPLES a rank-7 passer, faithfully to
SF's own 3.6× ratio) while `PASSER_V2_MAG=60` was fitted to a ladder-LESS passer — so they conflate SHAPE
with MASS. The bottom three hold mass roughly constant and isolate the shape. **All three are worse than
neutral, and the worst reading of the whole sequence is the most mass-neutral one.** ⇒ the mild win%
ordering in the first ladder came from the extra weight, not from the discrimination. **The shape carries
no information at our search depth.**
⇒ Leading hypothesis, untested: our search already resolves path safety tactically, so a static verdict on
it is redundant with search and, being static, sometimes overrides what the search would have found. Compare
`a-feature-measured-where-it-is-redundant-looks-worthless` — ask which REGIME would close it, not whether
the references carry it.
⚠️ One live alternative before this is called permanent: our `unsafe` set uses the shared `KS_V2_XRAY` map
and is strictly larger than SF's plain `attackedBy`, so every `k` may sit one rung low. `KS_V2_XRAY=0`
tests it — but that knob also moves KS, so it is not a clean single-term test.
⚠️ Do NOT read the `cr4_CRITICAL` rows (n = 5-12). They swing −14.5 to +5.7 and are pure noise.

☠️ **P3 (rook/queen behind a passer) IS CLOSED BY THIS**, not merely still blocked: P3 lives inside this
ladder in SF (enemy R/Q behind keeps the span unsafe; own R/Q behind is the `+5`). Both halves shipped in
these arms and the result is null-to-negative. Reviving P3 as a standalone bonus would be the third invented
form the audit warned about, now with a measured null behind the mechanism it belongs to.

★★★★ **AND THE AUDIT'S OWN RANKING HEURISTIC IS DAMAGED BY THIS.** The tiering assumed reference-count and
mechanism-vs-bonus predict payoff. This was the top-ranked non-architectural item on both axes and it is
null — the **thirteenth** consecutive move-null concept. Twelve were constants on detectors, which is why
they were dismissed as a selection artefact; this one was not. ⇒ treat the remaining tiers as UNRANKED
until something re-establishes that either axis predicts anything.

### K1 — SHELTER / STORM. 4/4. Spec already delivered (see the register).
★ SF **double-wires** it: seeds `king()` directly AND feeds `kingDanger` as `−6·mg/8`. "Build shelter" is
really "choose one channel or both".
☠️ The storm half **is** OvD by v2's own design comment ⇒ the ownership call blocks it.

---

## ★★ TIER 3 — 3/4, inputs already present

| id | gap | note |
|---|---|---|
| **P4** | **Rear doubled passer over-credit** | v2 flags passed on span-clear alone (`:702-703`); a friendly pawn AHEAD on the file does not demote it, so **both stacked passers are paid in full**. 3 references fix it 3 ways. ★ Closer to a defect than a gap; one AND (`ps_nfill(front) & own`) |
| **T1** | Threats against the enemy QUEEN (knight-on-queen / slider-on-queen) | 3/4 [2 designs]; mobility area + attack maps already exist |
| **K3** | KS DEFENDER COUNT | 3/4 [2 designs]. ⚠️ The register's "v2's KS-A has no defender count so the slot is empty" is only half true — defence already enters via the `weak` set (`:529`) and safe-check set (`:543`). Overlap is **partial, not empty** |
| **S4** | Doubled-isolated reclassification | 3/4 [3 forms]; currently BOTH fire on the same pawn |
| **P7** | Square rule / unstoppable passer | 2/4 [2 designs]; v2's exact KPK covers KPvK only |
| **S1/S2** | Blocked pawns → closedness → piece-value adjustment | 3/4 as an input, 1/4 as Ethereal's explicit form; `e.blocked` exists and is consumed only by one bad-bishop form |

---

## ★ CHEAP — single-lineage but the inputs are ALREADY COMPUTED
- **K4 blockers-for-king** (SF `:449`, ×98 into danger) — the set already exists from `MOB_V2_PIN`.
- **K5 unsafe checks** (SF, ×148) — same rays as our safe checks; softens our safe/unsafe binary.
- **K8 king-line danger** (Weiss) — one queen-ray popcount; the one structural ALTERNATIVE to storm.

---

## ✅ NO STRUCTURAL GAP — subsystem closed
**MOBILITY.** Every reference exclusion already exists in v2 as a knob (`MOB_V2_EXCL_QUEEN`,
`MOB_V2_EXCL_LOWRANK`, `MOB_V2_PIN`, `KS_V2_XRAY`); only defaults differ. No queen-specific handling in any
reference; no trapped-piece term beyond rook (SF's `CorneredBishop` is Chess960-only). ⇒ stop looking here.

---

## RECORD CORRECTIONS THIS AUDIT PRODUCED
1. "exclude shelter pawns attacked by enemy pawns" is **2/4, not 3/4** (SF15.1, Weiss; **SF11 and Ethereal
   do not**) — the SF lineage splits internally.
2. Rook-behind-passer max increment is **85, not ~110** (5 × w, w = 17 at the 7th rank).
3. KingProtector's "empty slot" premise is **partial**, not empty (above).
4. Ethereal has **no** rook-behind-passer term — UNRESOLVED closed by a full read of `evaluatePassed`.
5. The "rung 6 collision" (rook-behind × rook-on-file) applies **only to the enemy-passer case**; the
   own-passer case is mutually exclusive by construction.
6. ⚠️ Ethereal's `KingShelter` is indexed by **ABSOLUTE file** — a verbatim port fails our file-mirror
   symmetry gate, exactly as `PawnIsolated` did. SF's `map_to_queenside` tables are symmetric by design.

## UNRESOLVED
- Ethereal/Weiss read at `master`, not the pinned commits; `distanceBetween` assumed Chebyshev.
- v2's castling-rights bit layout for a K-side vs Q-side shelter test was not located.
- Whether v1's placement layers are a midgame table or a phase compromise (affects how costly A3 is).
