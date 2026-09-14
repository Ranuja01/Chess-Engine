# Eval v2 — SLICE 1: DRAW CLASSIFIER + CONVERTIBILITY SCALE

@author: Ranuja Pinnaduwage (maintained with Claude)

Date: **2026-09-13**. Status: ✅ **BUILT AND GATED — all four gates pass, and ENABLED in the shipped v2 config (`DRAW_V2_CLASS=1`) on the owner’s conditional
sign-off on the gate.** ⚠️ This document records the design AS IT EVOLVED through the day. Where an early section and a
later measurement disagree, **the LATER measurement stands** — read §2b-§2d and the final tables before trusting §1.
History that governs this: `EVAL-V2-CURRENT-CONFIG.md` §6 · memory [[endgame-draw-detection]].

★ **One sentence:** v1 already ships a binary draw detector and a built-but-disabled graded scale; the v2
job is not to invent them but to **re-derive them zero-global AND move several cases from the binary
detector into the graded scale**, because they are not ground truth and the binary form cannot express them.

---

## 0. RECORD-CHECK — this lane has THREE months of history, and I missed it once

⚠️ On 2026-09-13 I discussed draw detection as new work. It is not. The root cause was an unindexed
memory; `endgame-draw-detection` is now in MEMORY.md. What already exists:

| already done | where |
|---|---|
| `is_practically_drawn` | `cpp_bitboard.cpp:6769`, **LIVE and unconditional** in the endgame branch, called at `:7941` |
| KPvK rook-pawn case | `ENABLE_RP_KPK_DRAW=true` (`search_engine.h:1173`), shipped 2026-06-27 from a real external loss |
| R+N-vs-R, KRKN, KRKB | shipped 2026-06-08; the `+1N` endgame over-read went 461 → 142 |
| **`endgame_convertibility_scale`** | `cpp_bitboard.cpp:7112`, **BUILT, `ENABLE_ENDGAME_SCALE=false`** (`search_engine.h:1180`) |
| 🧰 `diagnostics/_kpk_oracle.py` | 83,238 states, 0 false-draws — the oracle tooling EXISTS |
| ⭐ the governing GATE (owner, June) | *"a self-play tournament is uninformative for a self-play-invisible fix → ship on verifiable position-fix + no bench regression"* |

★ **That gate IS the "verification is proportionate" rule.** The owner set it three months before I
restated it as new.

---

## 1. WHAT v1 ACTUALLY DOES (read from source, not from the notes)

`is_practically_drawn(int pieceNum)` — ⚠️ **`pieceNum` is UNUSED** (signature + one commented-out print
only). It reads eight globals (`occupied`, `kings`, `pawns`, `knights`, `bishops`, `rooks`,
`occupied_white`, `occupied_black`) and returns a **hard `return 0`** from the whole eval.

✅ Confirmed the zero is total, not partial: `search_engine.cpp:9043` sets
`total = placement_and_piece_eval(...)` — that IS the entire eval, so material is zeroed too.

### ⭐ PROVEN LIVE IN THE SHIPPED ENGINE, not only in the Python mirror (2026-09-13)
Owner asked whether `is_practically_drawn` is active or forgotten. **Active** — unconditional at
`cpp_bitboard.cpp:7941`, and demonstrated by execution rather than by reading: `ChessAI.ev` on shipped v1
(`EVAL_ARM=0`, no knobs):

| position | v1 eval (mp) | |
|---|---|---|
| KvK / KBvK / KNvK | 0 / 0 / 0 | rule fires |
| KBvKB asym / KNvKN asym | 0 / 0 | rule fires (asymmetric placement, so the 0 is the rule, not cancellation) |
| KR vs K (a win) | −10037 | read as won (v1's `piece_value_boost` doubles v2's −5000) |
| KQ vs K (a win) | −20034 | read as won |
| **K+R+B vs K+R `8/8/8/8/8/4k3/5r2/4KRB1 w`** | **0** | ☠️ **tablebase: WIN in 21. The false positive is live in the shipped engine.** |

### ☠️ And its companion is the FORGOTTEN one: `endgame_convertibility_scale` was reverted the day it shipped
| commit | date | `ENABLE_ENDGAME_SCALE` |
|---|---|---|
| `046a17f` "Ship continuation-aware LMR + endgame convertibility scale (default-on)" | 2026-06-09 | true |
| `ab070b7` **"Revert endgame scale to default-off (suite regression)"** | 2026-06-09 | **false** |
| `cdebc8a` (line touched, value unchanged) | 2026-08-07 | false |

⚠️ `STRENGTH_BACKLOG.md:41` still records it as "**shipped** (default-on `046a17f`)" — stale for three
months. Same failure class as the unindexed draw memory: the header is not the record.
⇒ **Before v2 carries this scale's constants into the OCB / pawn-ending tiers, it must answer what regressed
in `ab070b7` and why.** A scale reverted for a suite regression is not a validated prior.

#### What regressed, per `ab070b7`'s own message — and why it does not settle anything
> *"A scale-ON STS bench (which we never ran before shipping it) shows the endgame convertibility scale
> changes far more leaf evals than the 5 FEN spot-checks implied: **-3.3 STS / -3 WAC vs scale-off at d10**,
> for only -3% nodes -- a net suite regression. Kept as a knob; redo with tighter targeting."*

☠️ **By today's instrument record that is an UNRESOLVED NULL, not a refutation:**
- **WAC −3** is inside WAC's ±5–6 floor, and WAC at d10 **does not discriminate strength at all**
  ([[wac-at-d10-does-not-discriminate-strength]]).
- **STS −3.3** is inside the ±150 arm-vs-arm floor however it is read — as points trivially, as pp ≈ 99 of
  3000. ⚠️ The record even disagrees with itself: the commit body says −3.3, the knob comment says −3.9.
- Same class as most of the eval failure record ([[the-eval-failure-record-is-mostly-unresolved-nulls-not-refutations]]).

✅ **One part of the revert reasoning IS a real mechanism and should be kept:** it "changes far more leaf
evals than the 5 FEN spot-checks implied" — the scale is BROAD. That is exactly why v2 changes its FORM:
v1 multiplied the WHOLE `total` behind a boolean `isEndGame`; v2 scales the **endgame leg only, inside the
continuous blend** (§2), which is narrower by construction.
⇒ **The June measurement transfers in neither direction** — it neither vindicates nor condemns v2's version,
which is a different mechanism. v2's scale gets measured on its own.

### The cases, and ☠️ THE CRITICAL SPLIT

| # | case | line | ground truth? |
|---|---|---|---|
| 1 | `no_king_mask == 0` (KvK) | `:6803` | ✅ **PROVABLE** |
| 2 | KB vs K / KN vs K | `:6780-6784` | ✅ **PROVABLE** |
| 3 | equal count, only bishops or only knights (KBvKB, KNvKN) | `:6774-6777` | ✅ **PROVABLE** |
| 4 | K + lone rook pawn vs K, chebyshev opposition | `:6813` | ☠️ **NOT clean as shipped — measured 6.2% FP** despite the June `_kpk_oracle.py` claim; the race test ignores side to move. v2: OFF in `DRAW_V2_KPK`; fix = exact KPK bitbase |
| 5 | wrong-coloured bishop + rook pawn, opposition | `:6838` | ☠️ **NOT ground truth — measured 9.1% FP** (see below) |
| 6 | bishop vs lone pawn (can't promote) | `:6929` | ⚠️ 0/62 at first, then **1/80 (1.2%) on a fresh seed**; 0/304 with v2’s tempo term. OFF in `DRAW_V2_KPK` |
| 7 | knight vs lone pawn | `:6978` | ⚠️ unproven — my Python mirror is approximate here |
| 8 | **R+B vs R** | `:6882` | ☠️ **NOT ground truth** — the code's own comment says *"known theoretical draw in most cases"* |
| 9 | **R+N vs R** | `:6899` | ☠️ **NOT ground truth** — same comment |
| 10 | **KRKN / KRKB** (bare rook vs bare minor) | `:6917` | ☠️ **NOT ground truth** — "the lone minor + king holds" is true only under correct defence |

### ⭐⭐ PROVEN, NOT ARGUED — cases 8 and 9 are false-positive generators
Queried the **Lichess 7-piece tablebase** (`tablebase.lichess.ovh`, authoritative WDL/DTM, no download):

| position | material | v1 verdict | ☠️ TRUTH |
|---|---|---|---|
| `8/8/8/8/8/4k3/5r2/4KRB1 w` (Ke1 Rf1 Bg1 vs Ke3 Rf2) | K+R+B vs K+R | **case 8 ⇒ hard 0** | **`"category":"win"`, DTM 21** |
| `8/8/8/8/8/4k3/5r2/4KRN1 w` (Ke1 Rf1 Ng1 vs Ke3 Rf2) | K+R+N vs K+R | **case 9 ⇒ hard 0** | **`"category":"win"`, DTM 25** |
| `8/8/8/8/4k3/8/4KR2/6b1 w` (Ke2 Rf2 vs Ke4 Bg1) | K+R vs K+B | case 10 ⇒ hard 0 | `"category":"draw"` ✅ |

Both winning positions satisfy their case's condition exactly
(`popcount(no_king_mask)==3 && popcount(rooks)==2 && popcount(bishops or knights)==1`, opposite sides),
so `is_practically_drawn` returns true and **the eval reports exactly 0 for a forced win.**

### ⭐⭐⭐ THEN MEASURED PROPERLY — 🧰 `diagnostics/_draw_oracle.py` (built 2026-09-13)
Mirrors every case in Python from the C++, generates random legal positions per material signature, and
checks each flagged position against the tablebase. First run (n≈12/case, seeded):

**Two independent seeded runs.** Headline = the n=50 run; the n=12 run is the replication check.

| case | n=50 FP | **rate** | n=12 FP | verdict |
|---|---|---|---|---|
| `eq_only_minor` (KBvKB, KNvKN) | **0** | **0%** | 0 | ✅ **0 for 62** |
| `rookpawn_KPvK` — the oracle-validated one | **0** | **0%** | 0 | ✅ **0 for 62** |
| `B_vs_P` | **0** | **0%** | 0 | ✅ **0 for 62** |
| `wrongB_rookpawn` | 5 | **10.0%** | 9.1% | ☠️ refuted |
| `RB_vs_R` | 14 | **28.0%** | 8.3% | ☠️ refuted |
| `RN_vs_R` | 11 | **22.0%** | 25.0% | ☠️ refuted |
| `R_vs_minor` (KR vs KB) | 12 | **24.0%** | 41.7% | ☠️ refuted |
| `R_vs_minor` (KR vs KN) | 14 | **28.0%** | 16.7% | ☠️ refuted |
| `N_vs_P` | — | — | 16.7% | ⚠️ my mirror is approximate; **do not quote** |

⚠️ **My first headline of 41.7% for KRvKB was small-sample noise** (n=12). The four rook cases converge to
**~22-28%** at n=50 and that is the figure to use. The three clean cases are **0 for 62 across two seeds**.
Example counterexamples: `8/8/8/1r5k/1R3B2/1K6/8/8 w` (RB_vs_R, win) · `8/7K/8/8/3b3R/1k6/8/8 w`
(R_vs_minor, win) · `5B2/8/7K/8/8/8/P4k2/8 w` (wrongB_rookpawn, win — the chebyshev-opposition test passes
and White still wins, because the defending king must also dodge the bishop and the attacking king).

☠️ **I had `wrongB_rookpawn` down as "provable (classic)" in the first draft of this table. It is not.**
Counterexample `8/8/8/2B5/5k2/K7/P7/8 b` (Ka3 Bc5 Pa2 vs Kf4): the opposition test passes — defender
distance 5 ≤ min(pawn 6, attacker 5) — and White still **wins**. Equidistance is not sufficient; the
defender can be outmanoeuvred. ★ **Only the case that already had an oracle survived contact with one.**
That is the lesson, and it is the owner's proportionate-verification rule paying off literally.

⚠️ **Bound this before it gets over-read.** Uniform random placement **over-samples loose pieces**, and the
tactical subset never reaches the eval's verdict — a hanging bishop is simply captured, reaching KRvK, which
is not flagged. ⇒ **These rates are an UPPER BOUND on real harm.** The residual that actually bites is the
*quietly won by technique* class — which is exactly KRB/KRN-vs-KR at DTM 21-25, where no capture rescues us.

⚠️ Epistemics: **one counterexample refutes a universal claim; one confirmation proves nothing.** Every
☠️ case above is refuted as ground truth. The ✅ cases are *unrefuted at this sample size*, not proven —
`eq_only_minor` and the trivial insufficient-material cases are provable by argument; `B_vs_P` is not, and
wants a larger sweep before it stays binary.

★ **And this vindicates why the cases were added rather than condemning it.** The 2026-06-08 entry shipped
them to fix a real over-read (the `+1N` endgame read went 461 → 142). The direction was right: R+B vs R is
NOT worth its nominal material edge. But the binary detector was the only tool available, so the fix
overshot from *"+2 pawns"* to *"exactly 0"* when the truth is *"often winning, hard to convert."*
⇒ **The graded scale is the correct magnitude for precisely these cases.** This is not a nice-to-have
addition; it is the right home for content the binary form could never express.

☠️ **v1 carries this defect today** — but v1 is the frozen byte-identical control. **Do not fix it there.**
It is an argument for v2 landing, and it is recorded here so the behaviour is not mistaken for correct.

★★ **This is the finding that shapes the v2 design.** Cases 8-10 are *heuristics wearing a classifier's
clothes*. Flagging them as a hard `0`:
- ☠️ violates the owner's own asymmetric requirement — **prove NO FALSE POSITIVES**, because a won
  position flagged drawn is catastrophic while a missed draw only forfeits an opportunity;
- and it is *unnecessary*, because the graded scale can express "usually drawn" exactly.

⇒ **v2 splits them.** Cases 1-7 stay a binary hard zero. Cases 8-10 move to the scale.
★ This is precisely what the June note anticipated — *"a graded drawishness scale … the cases the binary
detector cannot express"* — with one addition: **some cases currently IN the binary detector belong in
the scale.** That is a migration, not just an addition.

### `endgame_convertibility_scale` (`cpp_bitboard.cpp:7112`) — built, off, and well-calibrated
`double` in `[DRAW_SCALE_FLOOR=0.25, 1.0]`, multiplied into `total` when `ENABLE_ENDGAME_SCALE && isEndGame`:
- lead ≤ one bishop, 0 winner pawns → `0.25`; ≤2 pawns → `0.45 + 0.18 * pawns`
- opposite-coloured bishops (one each, opposite colours, nothing else) → `0.35 + 0.09 * totalPawns`
- a winner's passer pulls back: adv ≥ 6 → `1.0`; ≥ 5 → ≥ `0.85`; ≥ 4 → ≥ `0.65`

---

## 2. THE REFERENCE COMPARISON — and it changes the FORM, not the constants

**SF11** `evaluate.cpp:816-818`:
```cpp
ScaleFactor sf = scale_factor(eg_value(score));
v =  mg_value(score) * int(me->game_phase())
   + eg_value(score) * int(PHASE_MIDGAME - me->game_phase()) * sf / SCALE_FACTOR_NORMAL;
```
and the rules, `:749-758`:
```cpp
if (sf == SCALE_FACTOR_NORMAL) {
    if (pos.opposite_bishops() && pos.non_pawn_material() == 2 * BishopValueMg)  sf = 22;
    else  sf = std::min(sf, 36 + (pos.opposite_bishops() ? 2 : 7) * pos.count<PAWN>(strongSide));
    sf = std::max(0, sf - (pos.rule50_count() - 12) / 4);
}
```

### ★★ THREE STRUCTURAL LESSONS

**(1) Scale the ENDGAME LEG ONLY, inside the phase blend — do not multiply the total.**
SF multiplies `eg_value` before interpolation, so the damp **fades in continuously with phase** and has
zero effect in the midgame *by construction*. v1 instead multiplies the whole `total` behind a boolean
`isEndGame` gate — a cliff.
⇒ v2 must use SF's form. We locked `MG_LIMIT/EG_LIMIT` to a continuous `phase256` precisely to kill
cliffs like this; scaling the eg leg is the form that respects it. **This is a form change, and it is the
single most important thing in this document.**

**(2) SF separates SPECIFIC from GENERAL — which independently confirms §1's tier split.**
`me->scale_factor()` (material table) can return `SCALE_FACTOR_DRAW = 0` for known-dead endings; the
general heuristics only ever *reduce toward* a floor. Two mechanisms, two epistemic statuses. ★ I derived
the same split from v1's own "in most cases" comments before reading this; SF agreeing is corroboration.

**(3) ⚠️ SF couples the scale to the FIFTY-MOVE COUNTER; we have nothing like it.**
`sf -= (rule50_count - 12) / 4`. As the counter climbs, convertibility falls. This is exactly the owner's
stated motivation — *"a practical playing need to avoid going into drawn positions"*.
☠️ **Not buildable today without plumbing**: the eval never receives `rule50`, and
`generateZobristHash` excludes it, so the eval cache would collide across different counters — the same
defect class already recorded for castling rights and en passant. **Flag as a follow-up, do not build in
this component.**

### Constants — near-identical, independently derived (a good sign for v1's scale)
| concept | SF11 | v1 |
|---|---|---|
| opposite-coloured bishops, bare | `22/64` = **0.344** | **0.35** + 0.09/pawn |
| per-pawn climb | `7/64` = **0.109** | **0.09** |
| general floor with 0 pawns | `36/64` = **0.5625** | 0.25 (but only at edge ≤ 1 minor) |

⇒ Carry v1's constants as priors; they sit inside SF's band. The **form** is what changes.

---

## 2b. ☠️ SF HAS FOUR TIERS, NOT TWO — AND TIER 2b CORRECTS THIS DOCUMENT

Prompted by the owner asking whether SF uses "exact algorithms or approximators with convertibility scales
like v1". It is neither/both, and the missing middle is exactly what our failing cases need.

| tier | mechanism | members (SF11 `endgame.h:36-63`) |
|---|---|---|
| **1** | **exact bitbase** | `KPK` only — `endgame.cpp:184` `if (!Bitbases::probe(...)) return VALUE_DRAW;` else `VALUE_KNOWN_WIN + PawnValueEg + rank`. ★ Classification AND magnitude |
| **2a** | value fn: **known win + drive to mate** | `KXK`, `KBNK`, `KQKP`, `KQKR`, `KNNKP` — `VALUE_KNOWN_WIN` + corner gradient |
| ★ **2b** | value fn: **TECHNIQUE GRADIENT** | **`KRKB`, `KRKN`, `KRKP`** — a SMALL value replacing material entirely |
| **3** | **scaling fns** — keep the eval, multiply the EG leg | `KRPKR` (Philidor), `KBPsK`, `KPsK`, `KBPKB`, `KPKP`, … |
| **4** | generic heuristics | `evaluate.cpp:749-758` — OCB 22/64, pawn-count ramp, rule50 damp |

### The whole of SF's KRKB handler (`endgame.cpp:241-248`) — our 24%-FP case
```cpp
Value result = Value(PushToEdges[pos.square<KING>(weakSide)]);
return strongSide == pos.side_to_move() ? result : -result;
```
`PushToEdges` = **20 (centre) → 100 (corner)** ⇒ **~94-470 mp in our units**, against a nominal R−B edge of
**1550 mp**: roughly **6-30% of it**. `KRKN` adds `PushAway[distance(bksq,bnsq)]`, widening to ~5-54%, and
SF says why: *"the attacking side has slightly better winning chances than in KR vs KB, particularly if the
king and the knight are far apart."* ✅ Our own measurement agrees — KRvKN 28% won vs KRvKB 24%.

★★ SF does THREE things at once: **discards the material read** (it is not worth +1.5), **keeps a non-zero
value** (it is sometimes won), and **shapes it as a gradient** pointing at the winning technique.

### ⇒ THIS CORRECTS §1 AND §3 OF THIS DOCUMENT
I recommended migrating cases 8-10 into the convertibility scale. **That is wrong, mechanically:**
- a **scale MULTIPLIES** the existing eval, so 0.25 × 1550 = 387 mp is the right ballpark but **FLAT** —
  it gives search **no gradient**;
- v1's hard **0** is worse still: zero magnitude *and* zero gradient, so even in a won position the engine
  has nothing to steer by.
★ **KRKB/KRKN wins are found by TECHNIQUE, and technique needs a slope.** A magnitude alone cannot express
"drive the defending king to the edge"; only a shaped replacement value can.

| cases | correct mechanism |
|---|---|
| KvK, KBvK, KNvK, KBvKB, KNvKN | binary hard 0 — ✅ built, `DRAW_V2_CLASS` |
| KPvK family | **exact bitbase** (tier 1) — replaces the 0.6%-FP heuristic |
| **R+B-vs-R, R+N-vs-R, KRvKB, KRvKN** | ★ **replacement value + corner gradient** (tier 2b), **NOT the scale** |
| OCB, pawn endings, general convertibility | the graded scale (tiers 3-4) — where `endgame_convertibility_scale` already correctly sits |

## 2c. DOES v1 HAVE DRAW "EXTRAS" THE GIANTS LACK? — No. Same cases, worse mechanisms.

Owner's question (2026-09-13): *"the extra stuff v1 has which the giants don't in terms of draws — do they
have a place in v2 if clean, accurate and efficient?"* Checked against SF11 source first:

| v1 case | SF11 mechanism | why SF's does not false-positive |
|---|---|---|
| KvK, KBvK, KNvK, KBvKB, KNvKN | generic rule `material.cpp:198` — pawnless, edge ≤ B, own npm < R ⇒ `SCALE_FACTOR_DRAW` | keyed on material invariants, not hand cases |
| KB vs KP, KN vs KP | same generic rule — **the minor side only** is scaled to 0; the pawn side is untouched | one-sided: the side that cannot win is zeroed, the side that might is not |
| lone rook pawn KPvK | **exact bitbase**, `endgame.cpp:184` | exact by construction |
| **wrong-coloured bishop + rook pawn** | named `KBPsK`, `endgame.cpp:356-358` | ★ see below |
| R+B vs R, R+N vs R | generic rule ⇒ **14/64 ≈ 22%** scale | keeps a fraction, never zero |
| KR vs KB, KR vs KN | named `KRKB`/`KRKN` — technique gradient | shaped replacement value |

### ★★ The wrong-bishop case shows WHY: v1 tests a RACE, SF tests a FORTRESS ALREADY REACHED
```cpp
// SF11 endgame.cpp:349-358 (KBPsK)
if ((pawnsFile == FILE_A || pawnsFile == FILE_H) && !(pawns & ~file_bb(pawnsFile))) {
    ...
    if (opposite_colors(queeningSq, bishopSq) && distance(queeningSq, kingSq) <= 1)
        return SCALE_FACTOR_DRAW;
}
```
- **v1:** `defender_dist <= min(pawn_dist, attacker_dist)` — a race the defender wins on paper and can still
  be outmanoeuvred in. **Measured 10% false positives.**
- **SF:** the defending king is **already within one square of the queening corner** — nothing left to race.
  It also covers several pawns on one rook file, and fires when the weak side has pawns of its own.
★ **General lesson: prefer a STATIC fortress condition to a DYNAMIC race condition in any binary draw
test.** A race depends on tempo, move order and interference; a fortress already reached does not. Every
v1 case that failed the oracle today was a race.

### ⇒ So where does v1 genuinely exceed the giants?
Not in classification — in the **scale-and-drive** space:
- `endgame_convertibility_scale`'s **passer pull-back** (an advanced winning passer lifts the scale toward 1);
  SF's generic scale has no passer-advancement term.
- The **queen-gated mate drive**, `cpp_bitboard.cpp:5183` (`defender_has_queen ? 0.0 : ...`).
- `advanced_endgame_eval`'s **king races**.

### Policy (agreed — see `EVAL-V2-CURRENT-CONFIG.md` §5)
**Classifications are governed by oracles, not consensus.** An ours-alone classifier that passes the
tablebase belongs in v2 with no reference support. ⚠️ But ours-alone classifiers carry the highest prior
risk, and the oracle is the bar. ☠️ Magnitude extras — the passer pull-back, the mate drive — are measured
like any positional term.

**Adopt next, oracle-first:** SF's fortress form of the wrong-bishop test, and Weiss's two cases v2 misses
— **KB vs KN** and **KNN vs K**.

✅ **RESULT (2026-09-13) — all three ADDED to `draw_class`, behind `DRAW_V2_CLASS` (default off):**

| case | uniform | corner-biased (EDGE=1) | total | short FP | long FP |
|---|---|---|---|---|---|
| KB vs KN | 0/160 | 0/240 | **0/400** | 0 | 0 |
| KNN vs K | 0/160 | 0/240 | **0/400** | 0 | 0 |
| SF fortress wrong-bishop | 0/54 | 0/232 | **0/286** | 0 | 0 |
| (shipped) `eq_only_minor` B + N | 0/382 | 0/480 | **0/862** | 0 | 0 |

⚠️ They pass the **proposed** DTM-weighted gate (§2d), which awaits the owner's sign-off — not literal zero-FP,
which the constructed KNvKN mate-in-1 shows no sampler can certify.
✅ **KNN vs K: the references AGREE** — SF’s named `Endgame<KNNK>` returns `VALUE_DRAW`, Weiss draws it. ⚠️ *Corrected
2026-09-13*: this row first said SF “scales to 4/64” — that is only the generic material rule, which a named handler
overrides. Lesson: check for a NAMED handler before citing SF’s generic rule for any specific ending.
★ The wrong-bishop case enters in SF's **fortress** form. v1's **race** form of the same idea measured 10% FP. ✅ Every reference draws KNN vs K: SF’s named `Endgame<KNNK>` returns `VALUE_DRAW` (SF11 `endgame.cpp:329`,
SF15.1 `:313`) and overrides the generic scale; Weiss draws it too. ⚠️ *Corrected 2026-09-13* — this line first
claimed SF scales it to 4/64, having read only the generic material rule.

## 2d. ☠️ "ZERO FALSE POSITIVES" IS UNATTAINABLE FOR MINOR-PIECE DRAWS — PROPOSED: WEIGHT BY DTM

**Found 2026-09-13, after the three reference candidates each came back clean on uniform sampling**
(KBvKN 0/160 · KNNvK 0/160 · SF fortress wrong-bishop 0/54).

KBvKB, KNvKN, KBvKN and KNNvK each contain a **handful of genuine tablebase wins** — forced mates where a
king is boxed into a corner, often by its own piece. At that frequency **uniform random placement
essentially never generates one.** So:
- ⚠️ The shipped set's **"0 FP in 382 samples" was TRUE AS STATED and NOT PROOF OF CLEAN** — the same trap
  as the rook-pawn rule at 0-for-62, which a third seed broke at 6.2%.
- Every reference draws these anyway: SF's generic rule zeroes KBvKN/KBvKB/KNvKN (`material.cpp:199`,
  own npm < rook), and Weiss's `TrivialDraw` zeroes all four.

### ★★ Why they are nearly harmless — and what the gate should actually protect
**Search detects checkmate itself.** A forced mate inside the horizon is found whatever the leaf eval
says, so a hard 0 on a mate-in-4 costs nothing. The false positives that cost games are **long-technique
wins beyond the horizon** — exactly today's refuted cases, K+R+B vs K+R at DTM 21 and K+R+N vs K+R at 25.

⇒ **PROPOSED gate (refines the owner's asymmetric rule — needs sign-off):**
| false positive | definition | verdict |
|---|---|---|
| SHORT | `abs(dtm) <= SHORT` plies (default 12; our search reaches d12 in ~1 s) | reported, **tolerated** |
| LONG | `abs(dtm) > SHORT`, or DTM unknown | ☠️ **the gate** — one LONG FP keeps a case out |

⚠️ The honest trade: this admits rule-level false positives that the literal "prove NO false positives"
standard would forbid. It is defensible only because search closes the short ones; if SHORT is set above
what search reliably reaches, it stops being defensible. Treat SHORT as a search-depth claim and keep it
conservative.

### ★ And a sampling fix that the gate needs to be meaningful
🧰 `_draw_oracle.py EDGE=1` forces one king onto a corner or corner-adjacent square, where boxed-king mates
live. **A clean result under uniform sampling says nothing about a rare-FP class; only a sampler aimed at
where the FPs live can clear it.** The shipped `eq_only_minor` set is re-run under EDGE=1 as well, not just
the new candidates.

### ⚠️ The gate's premise, checked against the search code — STRUCTURAL, not yet demonstrated
The DTM-weighted gate tolerates SHORT false positives on the claim that **search finds forced mates itself**.
That claim fails if a draw rule's static 0 lets eval-driven pruning cut the mating line before search reaches
the mate. Checked each pruner (2026-09-13):

| pruning | can a false draw-0 cut a short mating line in these endings? |
|---|---|
| **null move** | ✅ **No — it is OFF there.** `isUnsafeForNullMovePruning` (`search_engine.cpp:8164`) returns unsafe when the side to move has **no queen and fewer than 7 non-king pieces** — true of every draw case. `ENABLE_NULL_EVAL_GATE=false` besides |
| **RFP** | ✅ **No.** Inside a flagged subtree every non-terminal leaf scores 0 and the only other scores are mates. The guard `beta > -9000000 && beta < 9000000` (`:5979`, mirror `:5294`) disables RFP at mate bounds, and near a bound of 0 the test `0 ∓ RFP_MARGIN·d` (1500/ply) cannot fire on a zero window |
| **futility** | ✅ **No**, by the same 0-or-mate argument |
| **LMP** | ⚠️ keyed on move index and depth, not eval — it can drop a late quiet mating move at shallow depth. **But it does so in ANY position**, so it is not a hazard the draw flag introduces |

⇒ The premise holds **structurally** for minor-piece endings.
### ✅ DEMONSTRATED at the root (2026-09-13) — interior pruning still owed
**First, the premise that such positions exist is confirmed.** Even the EDGE=1 sampler produced no SHORT
false positive, so one was constructed and checked against the tablebase: `6nk/8/6K1/4N3/8/8/8/8 w`
(White Kg6 + Ne5 vs Black Kh8 + Ng8) — **TB: "win", DTM 1, Nf7#** (g7/h7 covered by the king, g8 held by
Black's own knight, a knight check cannot be blocked). v2's `eq_only_minor` rule flags it drawn.
⇒ Rare forced mates in KNvKN are real, and **both uniform and corner-biased random sampling missed them.**

**Then the search**, full rung-2 v2 config, d12, via the (validated) harness `diagnostics/_draw_short_mate_demo.py`
(the rule-fires / controls gate is `diagnostics/_draw_v2_verify.py`):

| `DRAW_V2_CLASS` | static eval | search score | move | nodes |
|---|---|---|---|---|
| 0 (off) | **−50** | 9999997 ✅ mate | e5f7 | 85 |
| 1 (on) | **0** — the rule fired | 9999997 ✅ mate | e5f7 | 78 |

★ With the draw rule scoring the position 0, the search still plays **Nf7#** and scores it as mate. The OFF
arm's −50 proves the ON arm's 0 is the rule, not material cancelling.
✅ The harness was validated first on a known mate-in-1 (back-rank Ra8#: 9999997, a1a8) plus a non-mate
control (−10, "no"), so a "no" would have meant the search, not the parser.

⚠️ **What this does NOT show:** a mate-in-1 is played at the ROOT, and the root is never pruned. The worry the
structural table above answers — RFP/futility/null-move cutting a mating line INSIDE the tree — is only
exercised by a mate at least 3 plies deep. **Still owed: a draw-flagged, TB-confirmed mate with DTM ≥ 3.**
Construct it and TB-verify it; do not trust hand geometry — even this mate-in-1 needed the tablebase.

### ✅ …and the search for one came back EMPTY, which is itself the answer (2026-09-13)
🧰 `diagnostics/_draw_deep_mate_finder.py`: built the geometry these mates need (defending king in a corner, boxed by its
OWN minor, attacking king at distance 2, attacker to move), restricted to endings the SHIPPED C++ flags
(KNvKN, KBvKB), and asked the tablebase. An offline dry run first confirmed the generator is sound: 82.7% legal,
every legal candidate flagged by v2 — so an empty result means "none exist here", not "the generator was broken".
**Result: 0 wins with DTM ≥ 3 in 300 queries (844 candidates).**

Then the whole tablebase cache from the day, tabulated by material and DTM:

| material | positions | decisive | decisive by DTM |
|---|---|---|---|
| **KBvKB** | 746 | **27** | **all 27 at DTM 1** |
| **KNvKN** | 446 | 0 | — |
| KBvKN | 400 | 0 | — |
| KNNvK | 400 | 0 | — |
| (refuted) KRBvKR · KRNvKR · KRvKB · KRvKN | 62 each | 15 · 14 · 17 · 16 | DTM not stored (legacy entries) — rates reproduce the 22-28% refutation |

★★ **Across 1,592 minor-piece positions — heavily biased toward exactly where boxed-king mates live — every
forced win found was a mate in ONE ply. Not one was deeper.** The chess reason: the defender's own minor can
always break the box, so these mates exist only when they are immediate.
⇒ For the endings the shipped rule covers, **every false positive observed is a mate-in-1 at the position itself**,
which the root demonstration above shows search plays. The interior-pruning scenario — a flagged mate ≥ 3 plies
deep — **did not occur**, so the premise now rests on data as well as on the structural argument.
⚠️ Absence in a sample is not proof of absence; the claim is "none in 1,592 targeted queries", not "impossible".
⚠️ A mate-in-1 reached AT THE SEARCH HORIZON with a quiet mating move is missed — but it is missed without the draw
rule too (static eval doesn't see mates either; the leaf just reads 0 instead of ~−50). Not a hazard the rule adds.
⚠️ And "0/862 FP" for the shipped set was always a statement about the SAMPLER: under the self-block generator,
**27 of 746 KBvKB positions (3.6%) are mate-in-1 false positives.** All SHORT, so the proposed gate tolerates them.

☠️ **A run that looked like this demonstration and was not:** an inline `wsl.exe -e bash -lc "... DRAW_V2_CLASS=\$DC ..."`
from PowerShell. PowerShell does not treat `\` as an escape, so `$DC` expanded (empty) on the Windows side,
and the leftover `\` escaped the NEXT space — `DRAW_V2_CLASS` received the garbage value `" USE_OPENING_BOOK=0"`,
which is truthy, so **both "arms" ran with the rule ON** and showed static 0 twice. Caught only because the
header echoed the malformed value. The script-file form above is the valid run.
⚠️ And note the premise is **ending-specific**: it leans on null move being disabled by the <7-piece guard. A
draw rule for a position WITH a queen, or with 7+ pieces, would not inherit this safety.

## 3. DESIGN FOR v2

**Two components, deliberately separate, because their verification differs.**

### A. `draw_class(const V2Context&)` → bool — the binary detector
⚠️ **Membership revised after measurement — this is the corrected list, not the first draft's.**

⚠️ **Membership revised TWICE by measurement. This is the final list.** `DRAW_V2_CLASS` (default off):

| in A (hard 0) | why it qualifies |
|---|---|
| KvK, KBvK, KNvK | insufficient material — **provable by argument**, and `python-chess` agrees |
| `eq_only_minor` (KBvKB, KNvKN) | **0 FP in 382 samples across 5 seeds**; also provable by argument |

☠️☠️ **THE LONE-PAWN CASES WERE PULLED OUT TOO** — see below. They live behind their own default-off knob
`DRAW_V2_KPK` and are **not part of what ships**.

☠️ **Migrated to the scale** (component B): `wrongB_rookpawn` (10% FP), `RB_vs_R` (28%), `RN_vs_R` (22%),
`R_vs_minor` both forms (24% / 28%). `N_vs_P` is **undecided** until the C++ tail past `:6978` is mirrored.

### ☠️☠️ THE CASE WITH THE DOCUMENTED ORACLE WAS THE ONE THAT FAILED

`rookpawn_KPvK` came back **0/62 on two seeds** and I put it in the shipped set. A third seed at n=80 found
**5 false positives — 6.2%**. Combined: **5/142 ≈ 3.5%**.

★★ Two lessons, both mine:
1. **0-for-62 was not proof, and I shipped on it anyway** after writing "unrefuted, not verified" in this
   very document. A null at small n is not a clean bill — it is the noise-floor error in a new costume.
2. ⚠️ **The June claim — *"validated against a full KPvK retrograde oracle: no won position is flagged
   drawn"* — does NOT hold for the rule as shipped.** Whatever that oracle checked, it was not this
   condition. **A documented validation does not transfer to a changed rule.**

**Root cause, and it is shared by both pawn cases:** the chebyshev-opposition test **ignores whose move it
is**. `8/8/8/8/8/2k1K2P/8/8 w` — all three distances are 5, so `defender <= min(...)` fires, and White wins
because White moves first.

**Fix attempted and measured:** add the missing tempo (the pawn's side on move gains one square).

| case | v1 form | **+ tempo term** |
|---|---|---|
| `rookpawn_KPvK` | 6.2% FP | **0.6%** (2/320, 4 seeds) |
| `B_vs_P` | 1.2% FP | **0/304** ✅ |

A **10× improvement — and still not zero**, so it fails the asymmetric gate and stays off. Survivors like
`8/8/8/8/8/PK1k4/8/8 b` have the defender equidistant *and* on move and are still lost: chebyshev distance
simply does not capture KPvK.

### ★ THE REAL FIX: AN EXACT KPK BITBASE, NOT A BETTER HEURISTIC
Zero false positives **by construction**, and it covers **all** of KPvK rather than only rook pawns.
- SF ships one: `stockfish_11/stockfish-11-win/src/bitbase.cpp` — ~24 KB packed, generated at init in ms.
- We already own the retrograde tooling: 🧰 `diagnostics/_kpk_oracle.py`, 83,238 states.
⇒ Build that, then fold `DRAW_V2_KPK` into `DRAW_V2_CLASS`. Until then the lone-pawn cases do not ship.
★ This is "converge on the giants in SPIRIT" working as intended — we take their *method*, not their table.

⚠️ **Cross-effect to remember:** `DRAW_V2_KPK` is the only other place in v2 that reads `turn`, so enabling
it **degrades the tempo identity gate from exact to statistical** (`EVAL-V2-SLICE1-TEMPO-DESIGN.md` §4).
That caveat was written before this component existed and fired on the very next one.

Pure on `c.pawns/knights/bishops/rooks/queens/kings/white/black`; zero globals; no `pieceNum`. Returns a
hard 0 for the whole eval.
**Knob** `DRAW_V2_CLASS` (0 = off, byte-identical).

☠️ **Breakdown contract when the rule fires — got wrong once, fixed 2026-09-13.** The first version published
NOTHING (`terms_valid = 0`) so consumers would KeyError rather than read the previous position's stale terms.
It broke the gate it was meant to protect: `eval_symmetry.py` reads `bd["total"]` and crashed with
`KeyError: 'total'` the moment one of its 600 positions was draw-flagged — and so would any of the ~172
breakdown consumers. ★ **`total` was never unknown here — it is exactly 0. Only the TERM fields are.** Fix:
publish `total = 0` with **only `EB_TOTAL`** set. Stale term fields stay hidden; every consumer that needs the
score gets the true one.
★ **General rule: omit a breakdown field only when its value is genuinely unknown — never omit one whose value you
know, just to force an error elsewhere.**

### B. `convertibility_256(const V2Context&)` → int in `[FLOOR_256, 256]` — the graded scale
v1's three rules, **plus migrated cases 8-10**, in `/256` fixed point (v2 has no doubles).
Applied **to the endgame leg only**, SF-style, inside the existing blend.
**Knobs** `DRAW_V2_SCALE` (0 = off), `DRAW_V2_FLOOR_256` (prior 64 ≈ 0.25).

⚠️ **Ordering matters and must be stated:** A runs first and short-circuits; B never sees a position A
flagged. So the two can never double-damp.

---

## 4. VERIFICATION — proportionate, and asymmetric

Per the owner's June gate, **A needs no games attribution**:

| component | gate |
|---|---|
| **A (binary)** | ⭐ **oracle proof of NO FALSE POSITIVES.** Extend 🧰 `_kpk_oracle.py` (it exists; do not fork). For each case sample legal positions and assert: flagged-drawn ⇒ not won under optimal play. Plus firing rate on the corpora and NPS. **Ship on that + no bench regression.** |
| **B (graded)** | ☠️ a MAGNITUDE ⇒ no oracle exemption. §I + STS attribution, then games inside the slice |
| both | arm-0 byte-identity; `[toggles]` echo; non-zero changed-move rate |

★ A's asymmetry is the whole point: **a won position flagged drawn is catastrophic; a missed draw only
forfeits an opportunity.** Tune the oracle to hunt false positives, not coverage.

### 🧰 THE ORACLE IS AVAILABLE WITH NO DOWNLOAD — use it
We have **no Syzygy data files on disk** (only SF's `syzygy/` *source* dirs), and `python-chess` ships
`chess.syzygy`/`chess.gaviota` but they need data. ⇒ Use the **Lichess tablebase HTTP API**, authoritative
to 7 pieces, which is how cases 8 and 9 were refuted above:
```
https://tablebase.lichess.ovh/standard?fen=<FEN with spaces as underscores>
```
Returns `category` (`win`/`draw`/`loss`/`cursed-win`/`blessed-loss`), `dtz`, `dtm`, plus per-move rows.
⚠️ Every case in §1 is **≤5 pieces**, so coverage is total — this oracle can settle the binary detector
completely. Be a good citizen: sample (a few hundred positions), don't sweep exhaustively, and cache
results to a local file so a re-run costs nothing. ☠️ Do NOT download the 5-man tablebases: C: is at
~70 GB free and the repo already strains OneDrive sync.

## 5. REGISTERED PREDICTIONS (before building)
1. **A fires on well under 1% of corpus positions** — it needs ≤3 non-king pieces. ⇒ invisible to STS and
   to games, exactly as the June gate says. Its value is correctness, not Elo.
2. **B is where any measurable Elo lives**, because it fires on the far larger class of "technically
   winning but hard to convert" endings.
3. ⚠️ **Migrating cases 8-10 out of the binary detector will CHANGE PLAY** and is not a pure refactor —
   positions that read exactly 0 will read a damped-but-nonzero value. It must be measured, not assumed
   free ([[a-correctness-fix-into-absorbed-tuning-is-not-free]]).
4. The mate drive (`cpp_bitboard.cpp:5183`, `drive_scale`, `defender_has_queen ? 0.0 : ...`) is the June
   note's candidate for being **subsumed** by B. Build B first, then test whether the mate drive still
   pays on top.

## 6. OPEN, FLAGGED, NOT IN SCOPE
- ☠️ **rule50 → convertibility** (SF `:757`). Needs `rule50` plumbed into the eval **and into the eval
  cache key**, or positions differing only in the counter collide. Same defect class as castling rights.
- v1's `is_practically_drawn` dead `pieceNum` parameter — v1 is the frozen control; **do not touch**.
