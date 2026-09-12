# Eval v2 — rung 1: king safety (KS-A)

**Scope.** KS-A = king zone · attacker accounting · safe checks · danger→score transform.
⚠️ **KS-B (shelter / pawn storm) is NOT in this rung** — it reads pawn structure, which does not exist
until rung 2. Splitting there is what allows KS to go first at all; only shelter was ever blocked.

**Why KS is rung 1** (owner's call, and the reasoning is the strongest argument in the program): if v1's KS
failures come from feeders shaped around the old system and from values used two and three times over,
then measuring KS anywhere except in a barebones eval measures the contamination rather than the subsystem.
This is the first time KS can be measured with nothing else around it.

---

## What v1 does, and the four defects we are not reproducing

| # | defect | evidence |
|---|---|---|
| 1 | ☠️ **The danger transform is a four-stage clamp stack** — `KS_FLOOR` deadzone → `KS_KNEE` quadratic band → `KS_DIVISOR` → `KS_CAP` | **`KS_FLOOR`=13 sits ABOVE `KS_KNEE`=12, so the entire quadratic growth band is inside the deadzone and never fires.** Recorded refutation, not a suspicion |
| 2 | `KS_CAP` inert above 40 | measured |
| 3 | ~15 mutually-exclusive default-off experiment modes inside `king_safety_danger` (>200 dead lines in one 554-line function) | counted |
| 4 | attacker weights assigned by **piece value** (queen highest, knight lowest) | ⚠️ a design split, not a defect — see below |

📌 Record to respect: **12 attempts, additive 0-for-11, only SUBTRACTIVE changes have ever won**
(de-king ≈ +50 across 3 seeds; `MOD_KS_REALIZ` inside the +36.7 bundle). ⇒ A rung that merely *adds* a
conventional KS term is starting from an 0-for-11 prior. What has never been tried is a **re-shape**.

⚠️ And the gate cannot see this subsystem well: last night `MOD_KS_REALIZ=0` — ablating a term that shipped
at **+36.7 Elo** — read −1.1pp against a 1.3pp spread, i.e. invisible. KS's signature is a **tail**; the
d7 regret gate measures a **mean**. ⇒ **KS-A must be confirmed with node-limited games, not with the gate.**

---

## The design

### 1. Danger transform — bounded BY CONSTRUCTION, one function

Replace `FLOOR → KNEE → DIVISOR → CAP` with a single monotone saturating curve:

```
danger(u) = KS_MAX * u^2 / (u^2 + KS_HALF^2)
```

| property | why it matters here |
|---|---|
| `danger(0) = 0` | no deadzone needed ⇒ defect #1 cannot recur |
| `≈ KS_MAX·u²/KS_HALF²` for small u | keeps the classical quadratic onset everyone uses |
| `→ KS_MAX` as u grows | the cap is a *limit*, not a `std::min` ⇒ defect #2 cannot recur |
| two constants, both physical | `KS_MAX` = worst-case danger; `KS_HALF` = attack units at half of it |
| smooth, no cliff | a sweep of either constant is continuous, so it cannot read flat from saturation |

★ This is the clamp-policy decision applied: the ceiling is visible in the constants and *moves when you
tune them*, instead of being an opaque post-hoc `min` that makes the knob step-shaped.

### 2. Attacker accounting

`u = Σ (weight[piece] × attacks into the king zone)`, plus the safe-check channel below.

⚠️ **Attacker weights are a DESIGN SPLIT, not a correction** (standing rule: references disagree ⇒ ours is
legitimate, theirs is a candidate):

| | knight | bishop | rook | queen |
|---|---|---|---|---|
| SF11 | **81** | 52 | 44 | 10 |
| Ethereal | **48** | 24 | 36 | 30 |
| Weiss | 36 | 22 | 23 | **78** |
| **ours** | 2 | 2 | 3 | **5** |

⇒ **Default keeps our ordering** (queen-high, which Weiss shares). `KS_ATT_PROFILE=1` selects the
knight-high shape (SF/Ethereal) as a **candidate arm**. This is a *shape* experiment — and shape is the one
class of change our ~15 failed SF ports never tested, since every one of them ported a NUMBER into our
existing shape.

### 3. Safe checks — a SEPARATE channel ✅ universally superior

SF and Ethereal both price safe checks **independently of attacker count**, and Weiss carries a separate
`CheckPower[]` alongside `AttackPower[]`. All three references separate the channels ⇒ adopt the shape.

Rationale that makes it more than convention: a lone queen that can give a safe check is dangerous *without
a second attacker*, and an attacker-count model cannot express that. It is also where SF puts most of the
queen's danger — which is precisely why SF can afford to weight the queen at 10 in the count.

### 4. King zone

One definition, computed once, published in the breakdown as a detector so "does it fire on the right
squares" is auditable without any eval measurement (detector/transformation split).
⚠️ v1 has THREE zone tables (`king_zones`, `_lean`, `_clamped`) of which one is live. v2 has one.

---

## Deliberately NOT in KS-A

heat map / attacking layer (retired — KS is one of the five jobs it was fusing) · shelter and storm
(rung 2, needs pawns) · `KS_DYN`, `KS_INTERACT`, `KS_OVERLOAD`, `KS_ADJACENCY`, `KS_BATTERY`, `KS_FLANK`,
`KS_AIM`, `KS_ZONE2`, the `KS_ACCUM_*` family and every other v1 experiment mode · `MOD_KS_REALIZ`
(realizability is a *conditioner*; it belongs after the thing it conditions exists and measures).

---

## Knobs — all default to OFF or identity

| knob | default | meaning |
|---|---|---|
| `KS_V2_MAX` | 0 | ☠️ **0 = KS-A absent = rung 0.5 byte-identical.** The rung IS this knob |
| `KS_V2_HALF` | tbd | attack units at half of `KS_V2_MAX` |
| `KS_V2_ATT_PROFILE` | 0 | 0 = our weights · 1 = knight-high (SF/Ethereal) |
| `KS_V2_CHECK` | tbd | safe-check channel weight |

---

## Measurement plan

1. **Rung-level (is KS worth having?):** STS, v2+KS vs v2 alone. Expect hundreds of points — ⚠️ §H of
   `INSTRUMENT-MAP.md` puts the early-rung STS floor at ~70 points from chaotic sensitivity, so only a
   large move is readable.
2. **Constants:** ☠️ NOT on STS. The d7 regret gate, `BASE_KNOBS` = the previous rung.
3. ★ **Confirmation: node-limited games, ≥2 budgets.** KS's signature is a tail and the gate measures a
   mean, so the gate can neither promote nor refute this rung. Node limits avoid thermal/throttle
   corruption; ⚠️ node-limit equal-work is **step-shaped**, so one budget can land on a plateau.

## Pre-registered predictions (before any KS-A run)

1. KS-A at sensible constants moves STS by **> 100 points** over rung 0.5. If it moves < 70 it is inside
   the chaotic floor and the rung is unproven regardless of sign.
2. The **regret gate reads it as null** — same as `MOD_KS_REALIZ`. That is a prediction about the
   *instrument*, not the feature; if the gate DOES see KS-A clearly, my tail-vs-mean model is wrong.
3. The knight-high profile is **within noise** of ours at the gate, and needs games to separate. Two
   references against one is not enough to predict a winner.
4. ⚠️ At least one constant will read flat over a 2x sweep. Under the old clamp stack that meant
   saturation; under this curve it means genuinely no signal — ★ which is exactly the ambiguity the
   bounded-by-construction design exists to remove, so it is a test OF the design as well as of KS.

---

## Complete component audit vs SF11 (`evaluate.cpp:446-461`) — added after owner asked "what am I missing?"

SF's entire `kingDanger` sum, verbatim, with our verdict on each:

| # | SF component | value | KS-A verdict |
|---|---|---|---|
| 1 | `kingAttackersCount x kingAttackersWeight` | **a PRODUCT** | ✅ IN — ★ but see shape decision A |
| 2 | weak squares in the king ring | `185 x popcount` | ✅ IN |
| 3 | attacks on squares adjacent to the king | `69 x kingAttacksCount` | ✅ IN |
| 4 | safe checks by type | R 1080 / N 790 / Q 780 / B 635 | ✅ IN (separate channel) |
| 5 | no enemy queen | **`- 873`** | ✅ IN — huge, cheap, and v1 already has `KS_NO_QUEEN`/`KS_NQ_SUP` |
| 6 | base offset | `+ 37` | ✅ IN (absorbed into the curve's shape) |
| 7 | unsafe checks | `148 x popcount` | ⏸️ DEFER — secondary channel, add after 1-5 are measured |
| 8 | pieces pinned/blocking near our king | `98 x popcount(blockers_for_king)` | ⏸️ DEFER — needs pin machinery |
| 9 | king flank attack | `3 x flank^2 / 8` | ⏸️ DEFER |
| 10 | king flank defense | `- 4 x kingFlankDefense` | ⏸️ DEFER |
| 11 | knight defending next to our king | `- 100 x bool(...)` | ⏸️ DEFER — tiny |
| 12 | **mobility difference feeds kingDanger** | `+ mg(mobility[Them] - mobility[Us])` | ☠️ **EXCLUDED — ALREADY REFUTED FOR US.** We built exactly this link at exactly this site and it read **null across a 4x range including cranked** ([[the-wiring-thesis-was-tested-and-is-unsupported]]). Do not rebuild it |
| 13 | shelter feeds back into danger | `- 6 x mg_value(score) / 8` | ⏸️ KS-B (rung 2, needs pawns) |
| 14 | shelter / storm itself | `pe->king_safety<Us>()` | ⏸️ KS-B (rung 2, needs pawns) |

⇒ **KS-A carries 6 of 14; 7 are deferred with a named reason; 1 is excluded because WE already refuted it.**

## ★ Two SHAPE decisions this audit surfaced, both bigger than any constant

### A. Attacker term: PRODUCT (`count x weight`) vs SUM (`Σ weight`)
SF multiplies **count by total weight**, so a single heavy attacker contributes little until a second piece
joins — **coordination is built into the shape**. v1 SUMS weights, so one queen alone already scores.
⇒ This is very likely more important than the knight-vs-queen weight argument, and it is untested here.
**KS-A default = product** (SF and the coordination logic agree); `KS_V2_ATT_SUM=1` selects our legacy sum
as the control arm. ⚠️ Note this interacts with the weights: under a product, a queen-heavy weighting
double-counts far less, which may be why SF can afford `queen = 10`.

### B. The transform is PHASE-SHAPED, not just phase-scaled
`score -= make_score(kingDanger^2 / 4096, kingDanger / 16)` — **QUADRATIC in the midgame, LINEAR in the
endgame.** Ours applies one curve and tapers its magnitude by phase.
⇒ Our single Hill curve `KS_MAX * u^2/(u^2 + KS_HALF^2)` is quadratic-then-saturating in BOTH phases.
**Decision: ship the single curve for KS-A** (one shape, two constants, easy to read), and register
"separate eg exponent" as an explicit later candidate rather than silently matching SF.

### ⚠️ And a correction to this document's own framing
SF has a deadzone too — `if (kingDanger > 100)`. So a floor is NOT unusual and our `KS_FLOOR` was never the
defect by itself. **The defect was the floor sitting ABOVE the knee (13 > 12), which made the quadratic
band unreachable.** The Hill curve removes the possibility rather than the feature: it is ~0 for small `u`
by construction, so it behaves like a soft deadzone without a hard threshold that can be mis-ordered
against a knee that no longer exists.

---

## ★ NO PHASE GATE — KS stays alive in the endgame (owner, 2026-09-11)

⚠️ This was MISSING from the design above and is now a first-class decision.

**v1 hard-zeros king safety in the deep endgame:** `KS_PHASE_FULL=48`, `KS_PHASE_ZERO=104`,
`KS_PHASE_FLOOR=0` ("taper value /256 AT/ABOVE KS_PHASE_ZERO; 0 = hard-0"), plus a deep-endgame early-out
that skips the computation entirely (`cpp_bitboard.cpp:5906`).

**KS-A does NOT do this.** Danger is computed in every phase. Owner's reasoning, which is correct:
> pressure scales down in endgames INHERENTLY because there are fewer pieces — that still means TRUE
> danger can be detected, instead of purely turning off.

Three independent supports:
1. ✅ **SF has no phase gate.** `kingDanger` is always computed; phase-dependence lives entirely in
   `make_score(danger^2/4096, danger/16)`, whose **endgame leg is LINEAR but NON-ZERO**. The reference
   arrives at the owner's design independently ⇒ adopt under the universality rule.
2. ★ **The product form reinforces the natural decay.** Under `count x weight`, three attackers down to one
   drops the term ~3x rather than linearly in weight — so endgame danger self-attenuates HARDER under the
   shape we are already adopting, making an explicit gate doubly redundant.
3. ★ **It is a SUBTRACTIVE change.** The KS record is 12 attempts, additive **0-for-11**, with only
   SUBTRACTIVE changes ever winning (de-king ≈ +50). Removing a gate is the one category with a winning
   prior.

⚠️ **What we give up, stated honestly:** the early-out was a speed optimisation — KS cost nothing in deep
endgames because it was skipped. Computing it always costs NPS in exactly the positions where nodes are
cheapest and most numerous. ⇒ **Measure the NPS price of removing the gate**, and if it is material, take
it back as a cheap *attacker-count* precondition (skip when the enemy has <2 attackers in the zone) rather
than as a PHASE gate — that preserves detection of real endgame danger while skipping the empty cases.
📌 Real endgame king danger this restores: back-rank mates, mating nets, an exposed king in a queenless
middlegame that our `phase_score` already classifies as "endgame" — and note our phase is a 3-way boolean
that cliffs at one minor trade, so the gate fires far earlier than "deep endgame" suggests.

---

## ✅ AS BUILT — final configuration and what changed from this design

```
KS_V2_MAX=4000  KS_V2_HALF=600  KS_V2_ONSET=450  KS_V2_COORD=256
KS_V2_WEAK=57   KS_V2_ADJ=61    KS_V2_NO_QUEEN=321
KS_V2_CHK_Q=126 KS_V2_CHK_R=122 KS_V2_CHK_B=80  KS_V2_CHK_N=152
KS_V2_ZONE_SF=1 KS_V2_XRAY=1    KS_V2_PAWN_ATT=0
```
Cross-corpus: **mean −1.44% eval error, worst +0.48%** (6 corpora, worst-case rule).

### Where this design was WRONG
| design said | reality |
|---|---|
| "no deadzone needed — the curve is ~0 at small u" | ☠️ **FALSE.** At u=100/HALF=300 the curve returns 400mp. Without `ONSET` the rung was a REGRESSION (+5.7% to +20% on general corpora). Both references suppress to EXACTLY zero. I removed the mechanism because v1's implementation of it was broken |
| attacker weights are "a first-class rung-one experiment" | measured **0.13%** — at the floor. Ordering is nearly empty; BALANCE is what matters (4.83%) |
| safe checks are "universally superior" | true of the CONCEPT, but SF's magnitudes in our formula made the channel NET HARMFUL. Fixed by taking Ethereal's ratios — our channel set matches Ethereal's, not SF's |
| 6 of 14 SF components ⇒ "we are missing a lot" | wrong comparison. Against ETHEREAL we have nearly all. 4 of the missing are SF-ONLY ⇒ skip; shelter is universal ⇒ rung 2 |
| feeders deferred as "not important" | ☠️ they measured NEUTRAL at the old constants and clearly BETTER once ONSET was re-derived — the absorption signature. Zone + x-ray are COMPLEMENTARY: x-ray alone regresses `_v2` +1.85%, the zone turns it −0.48% |

### Predictions scorecard
1. `>100 STS` — ✅ +148 (but STS later shown unfit for this question)
2. gate reads KS-A null — ✅ **confirmed on BOTH corpora** (49.2% in a 47.9-51.8 band with 57% of moves changed)
3. knight-high within noise — ✅ 0.13%
4. some constant reads flat over a 2x sweep — ✅ the COORD interior (64/128/192 all ~1440)

### ▶️ Open for a rung-1 revisit
- ★ **Curve shape.** Ours SATURATES at MAX; SF and Ethereal stay quadratic. We therefore cannot exceed 4
  pawns by construction, while SF11 reads 7.62 in some positions. "Quadratic longer before saturating" is
  untried and is the structural half of the under-read finding.
- `KS_V2_CHK_COUNT=1` (Ethereal per-square form, matching the constants we now use) — built, untested.
- KS-B shelter/storm — universal across all three references, needs rung 2's pawn structure.
