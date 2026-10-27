# Eval v2 — a priori constants and rung content, from the reference engines

**Purpose.** The v2 ladder adds one feature per rung with **a priori** constants and measures it *untuned*
first (see the escalation ladder in the plan: untuned → tune its own constants → retune neighbours, where
that last step is a ☠️ RED FLAG rather than a win). This file is where those a priori constants come from,
so no rung starts from a number someone invented.

Source: `stockfish_11/stockfish-11-win/src/` (`evaluate.cpp`, `pawns.cpp`, `types.h`). ⚠️ SF11 and SF15c,
not SF18 — a hand-written eval is the target, and SF11 hand-written already captures ~90% of SF15c's gain.

---

## ☠️ THE UNIT CONVERSION IS PHASE-DEPENDENT — get this wrong and every endgame constant is off by 1.7x

SF scores are `S(mg, eg)` pairs in units where (`types.h:182`):

| | SF11 | ours |
|---|---|---|
| pawn, midgame | **128** | 1000 |
| pawn, endgame | **213** | 1000 (flat) |

⇒ **midgame:** `millipawns = sf_mg * 7.81` · **endgame:** `millipawns = sf_eg * 4.69`

★ There is no single scalar. SF's pawn *gains* 66% of its value going into the endgame; ours is flat, and
our `EG_EXIST_*` tapers the PIECES up with no pawn entry, so our pawn effectively **loses** ~5.8% relative
where SF's gains ~52%. A one-factor conversion silently misprices every endgame constant by ~1.67x, and it
would do so in the direction that makes endgame terms look too big — which is exactly the shape of a result
we would then "fix" by retuning neighbours.
⇒ **Rung rule: convert mg and eg legs separately, always.** Record both, never collapse to one number.

---

## Mobility (`evaluate.cpp:93-107`, `:222-230`, `:287-289`)

★ **The mobility AREA matters more than the table.** SF does not count attacked squares; it counts
attacked squares inside a restricted area (`:230`):

> `mobilityArea[Us] = ~(b | pos.pieces(Us, KING, QUEEN) | pos.blockers_for_king(Us) | pe->pawn_attacks(Them))`
> where `b` = our pawns that are **blocked or on the first two ranks**.

So excluded: our own blocked/back pawns, our king and queen squares, our pinned-to-king blockers, and
everything the enemy's pawns control. ⚠️ Our current concept — "attacked squares, plus squares jumpable
from there" — has **no exclusion set at all**, which is the single biggest structural difference and the
most likely reason a naive mobility term reads as noise: squares an enemy pawn covers are not mobility.

`MobilityBonus[PieceType-2][attacked]`, mg/eg, SF units:
- **Knight** (0..8): `(-62,-81) (-53,-56) (-12,-30) (-4,-14) (3,8) (13,15) (22,23) (28,27) (33,33)`
- **Bishop** (0..13): `(-48,-59) (-20,-23) (16,-3) (26,13) (38,24) (51,42) (55,54) (63,57) (63,65) (68,73) (81,78) (81,86) (91,88) (98,97)`
- **Rook** (0..14): `(-58,-76) (-27,-18) (-15,28) (-10,55) (-5,69) (-2,82) (9,112) (16,118) (30,132) (29,142) (32,155) (38,165) (46,166) (48,169) (58,171)`
- **Queen** (0..27): `(-39,-36) (-21,-15) (3,8) (3,18) (14,34) (22,54) (28,61) (41,73) (43,79) (48,92) (56,94) (60,104) (60,113) (66,120) (67,123) (70,126) (71,133) (73,136) (79,140) (88,143) (88,148) (99,166) (102,170) (102,175) (106,184) (109,191) (113,206) (116,212)`

★ Note the shape: **strongly negative at low mobility**, not merely small. A trapped knight is worth
-62mg ≈ **-484 millipawns** — half a pawn. Mobility here is mostly a *penalty for immobility*, which a
"bonus per attacked square" formulation cannot express. ⚠️ Rook mobility is the most eg-weighted of the
four (`-76 → +171`), matching the folklore that rook activity is an endgame quantity.

## Space (`evaluate.cpp:662-691`)

```
if (non_pawn_material < SpaceThreshold /* 12222 */) return 0;      // midgame-only, by construction
SpaceMask   = CenterFiles & (ranks 2,3,4 for White)                // files C..F only
safe        = SpaceMask & ~our pawns & ~enemy pawn attacks
behind      = our pawns, smeared 1 and 2 ranks back
bonus       = popcount(safe) + popcount(behind & safe & ~enemy attacks)
weight      = our piece count - 1
score       = make_score(bonus * weight * weight / 16, 0)          // ⚠️ EG LEG IS ZERO
```
★ Three things our `SPACE_MAG` formulation does not have: it is **restricted to the four centre files**,
it is **quadratic in piece count** (`weight * weight / 16` — space matters only when you have pieces to
use it), and its **endgame leg is exactly zero**. ⚠️ Our `SPACE_MAG`/`SPACE_KNIGHT_MAG` are a different
computation entirely, and their recorded null was measured on the pre-08-14 contaminated harness ⇒ that
null is **UNREADABLE**, not a refutation.

## Pawns (`pawns.cpp:34-43`, `:132-139`)

| term | SF11 | → mg mp | → eg mp | ours today |
|---|---|---|---|---|
| Isolated | S(5,15) | 39 | 70 | `ISOLATED_PAWN_PEN = 0` |
| Backward | S(9,24) | 70 | 113 | `BACKWARD_PAWN_PEN = 0` |
| Doubled | S(11,56) | 86 | **263** | — |
| WeakLever | S(0,56) | 0 | 263 | — |
| WeakUnopposed | S(13,27) | 102 | 127 | — |

★ **These are much smaller than intuition suggests** — an isolated pawn costs SF about **4% of a pawn** in
the midgame. Any v2 rung that seeds them larger is asserting something SF's tuner disagrees with.

**Connected pawns** (`:43`, `:135-138`):
```
Connected[RANK_NB] = { 0, 7, 8, 12, 29, 48, 86 }
v      = Connected[r] * (2 + bool(phalanx) - bool(opposed)) + 21 * popcount(support)
score += make_score(v, v * (r - 2) / 4)
```
⚠️ **SUPERSEDED — see "NOT universal" below: Ethereal DOES weight connected pawns by file, rising toward
the centre, so this does not settle the question.** As written SF is still rank-only:
☠️ **KEYED ON RANK ALONE.** Every modifier is structural — phalanx, opposed, support count — and there is
**no file term and no central multiplier anywhere in it**. ⇒ This is direct evidence for the owner's own
counter-argument against our "chains are stronger into the centre" bonus: SF prices *advancement*, and
prices centrality once, elsewhere (space / mobility / PST). Our central-chain bonus is a candidate
double-count of central scoring.
⚠️ Note the eg leg `v * (r-2)/4`: connected pawns are worth **nothing in the endgame below rank 3** and
scale steeply above it. Our pawn tables have no such asymmetry.

## King safety (`evaluate.cpp:81-87`, `:279-285`, `:376-413`)

```
KingAttackWeights[PIECE_TYPE_NB] = { 0, 0, 81, 52, 44, 10 }    // -, pawn, N, B, R, Q
QueenSafeCheck = 780 · RookSafeCheck = 1080 · BishopSafeCheck = 635 · KnightSafeCheck = 790
```

⚠️ **SUPERSEDED — "inverted" overstates it. Weiss ranks the QUEEN top exactly as we do, so this is a
two-against-one design split, not a defect on our side. See "NOT universal" below. Against SF alone:**
☠️★ **OUR ATTACKER WEIGHTS ARE INVERTED RELATIVE TO SF'S.**

| attacker | SF11 weight | ours (`KS_ATT_*`) | ranking |
|---|---|---|---|
| knight | **81** | 2 | SF highest / ours lowest |
| bishop | 52 | 2 | |
| rook | 44 | 3 | |
| queen | **10** | **5** | SF lowest / ours highest |

SF prices the **knight** as the most dangerous king attacker and the **queen** as nearly irrelevant *to the
attacker count*, because the queen's danger is priced through a different channel entirely — `QueenSafeCheck
= 780` plus the no-queen suppressor. We price by piece value, which double-counts the queen (it is already
the most valuable piece everywhere else) and under-prices the knight, whose whole point near a king is that
it cannot be blocked or driven off.
⇒ ★ **A first-class rung-one experiment — a CANDIDATE to test, not a correction to apply.** It is a *shape* difference, and
`ks-twelve-attempt-history-and-the-channel-law` says additive KS changes are 0-for-11 while only
SUBTRACTIVE ones have won — a re-ranked attacker weight is neither; it is a re-pricing, which has not been
tried.

⚠️ Safe-check ordering also differs: SF has **rook highest** (1080), then knight (790), queen (780), bishop
(635). Ours (`KS_CHK_*`) ties queen and rook at 14, knight 9, bishop 7 — so we under-rank the rook check.

---

## ▶️ What this fixes about the ladder

1. Every constant above enters a rung as an **a priori** value, converted mg/eg separately. A rung that
   pays untuned tells us the feature carries signal; one that needs its own constants tuned is still fine;
   one that needs its NEIGHBOURS retuned is the degeneracy signature and gets recorded as such.
2. Three of these are **shape** differences rather than magnitude differences — the mobility *area*, space's
   quadratic piece-count weight and zero eg leg, and the KS attacker ranking. ★ Shape differences are the
   ones a magnitude sweep can never find, which is consistent with our record: ~15 SF ports read null, and
   every one of them ported a *number* into our existing shape.
3. The connected-pawn finding is a **subtraction**: it argues against building the central-chain bonus into
   v2 at all, pending the changed-set overlap test against central scoring.

---

# Material — three engines, two lineages (added 2026-09-11)

⚠️ **Scales differ; RATIOS transfer.** We are in millipawns (pawn = 1000); SF is in its own units, Ethereal
in another. Never port a constant — port the ratio and the phase SHAPE.

## Piece values, normalised to pawn = 1.00

| engine / phase | pawn | knight | bishop | rook | queen |
|---|---|---|---|---|---|
| classical | 1.00 | 3.00 | 3.00 | 5.00 | 9.00 |
| **OURS (flat, both phases)** | **1.00** | **3.25** | **3.45** | **5.00** | **10.00** |
| Ethereal endgame | 1.00 | 3.30 | 3.54 | 5.58 | 11.27 |
| SF1 (2008, both phases) | 1.00 | 4.08 | 4.08 | 6.30 | 12.55 |
| SF11 endgame | 1.00 | 4.01 | 4.30 | 6.48 | 12.59 |
| Ethereal midgame | 1.00 | 5.20 | 5.38 | 7.65 | 15.76 |
| SF11 midgame | 1.00 | 6.10 | 6.45 | 9.97 | 19.83 |

Raw: SF1 mg/eg P 204/256, N 832/832, B 832/832, R 1285/1285, Q 2560/2560 (`value.h:62-70`) ·
SF11 P 128/213, N 781/854, B 825/915, R 1276/1380, Q 2538/2682 (`types.h:182-186`) ·
Ethereal P 82/144, N 426/475, B 441/510, R 627/803, Q 1292/1623 (`evaluate.c`).

## ★★ THE TWO FINDINGS

**1. Our values ARE the consensus ENDGAME values — applied in every phase.** Ours sit within a few percent
of Ethereal's endgame row. ⇒ We are not using "wrong" values; we are using endgame values in the midgame.

**2. The pawn gains faster than the pieces into the endgame — in all three, across two independent
lineages and eleven years.**

| engine | pawn mg→eg | pieces mg→eg |
|---|---|---|
| SF1 | +25% | flat |
| SF11 | +66% | +8-9% |
| Ethereal | +76% | +11-28% |
| **ours** | **flat** | **UP** (`EG_EXIST_*`, and with no pawn entry at all) |

⇒ Likely consequence, and it is TWO errors in opposite directions, so no uniform rescale can find it:
**pieces under-valued in the midgame** (40-60% low against all three references) and **over-valued in the
endgame** (our base already equals consensus endgame, then `EG_EXIST_*` adds more).
📌 Consistent with the recorded observation that odds-losses trace to material overvaluation.

## ⚠️ NOT universal — do not treat these as settled

**King-attack attacker weights.** Two engines rank the KNIGHT top; one ranks the QUEEN top, as we do.

| | knight | bishop | rook | queen |
|---|---|---|---|---|
| SF11 | **81** | 52 | 44 | 10 |
| Ethereal | **48** | 24 | 36 | 30 |
| Weiss (`AttackPower`) | 36 | 22 | 23 | **78** |
| ours (`KS_ATT_*`) | 2 | 2 | 3 | **5** |

⇒ Our ordering matches Weiss. The knight-high variant is a CANDIDATE to test, **not** a correction to make.

**Central weighting of connected pawns.** SF keys `Connected[]` on RANK ALONE. ☠️ Ethereal does NOT:
`PawnConnected32[32]` is indexed by rank AND file and rises toward the centre — at rank 7 it runs
`108 → 214 → 216 → 233`, the largest on the d/e files, more than double the a/h value.
⇒ A central chain bonus is **not** self-evidently a double-count of central scoring. Decide it by
changed-set overlap, not by appeal to SF's design. (An earlier note here claimed SF settled it; it does not.)

⚠️ **Weiss material values unresolved** — defined as macros (`P_MG`, `N_MG`, ...) elsewhere. Its KS weights
and pawn tables above are read directly; its piece scale is not yet a third data point.
