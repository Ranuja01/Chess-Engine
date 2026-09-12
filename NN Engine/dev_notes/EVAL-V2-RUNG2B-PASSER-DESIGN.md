# Eval v2 — Rung 2b design: passed pawns

@author: Ranuja Pinnaduwage (design captured with Claude)
Status: DESIGN. Nothing built. Written 2026-09-12 after the four mandatory pre-rung scans.

⚠️ Prerequisites: `EVAL-V2-RUNG2-PAWN-DESIGN.md` (2a shipped doubled+isolated+backward at −0.66% mean /
worst −0.00%; connected/support PARKED) and the memory
`passer-law-multiplicative-vs-additive-and-valuation-graveyard`, which carries a **standing owner
instruction: "do not propose a 5th passer VALUATION mechanism."**

---

## 0. THE FOUR SCANS

| scan | finding |
|---|---|
| 1 — consumers by DATA dependency | **35 functions** touch passer state. All ten piece evaluators take `white_passed_pawns`/`black_passed_pawns` as parameters, plus `boost_pieces_for_supporting_passed_pawns`, `passer_danger`, `evaluate_passers`, `position_complexity`, `endgame_convertibility_scale`, `advanced_endgame_eval`, `approximate_capture_gains`, `rook_tension_scale` and the probe surface |
| 2 — knobs AND hardcoded magnitudes | **48 passer knobs.** ⚠️ Hardcoded magnitudes included this time (the doubled miss): `passer_block_quality` is entirely hardcoded (50/60/63/100/110, **zero knobs**); `passer_danger` carries a bare `4000`; `passer_king_race_one` carries `2000` |
| 3 — producers / side effects | `getPPIncrement` writes `white_passed_pawns`, `black_passed_pawns`, `candidate_passed_pawns`. `passer_realizability_R` writes `g_passer_rawR`. ☠️ `passer_king_race_one` writes **`whiteKingSquare`, `blackKingSquare`, `whiteMat`, `blackMat`, `g_ae_matedrive`** — a *scoring* helper mutating board-level globals |
| 4 — clamps / shared budgets | `PASSER_R_MAX=384` (inside `passer_realizability_R`), then `PASSER_R_CAP=320`, then `PASSER_R_FLOOR=64` on the king-race leg, plus two `std::clamp(..., 0, 96)` on the danger term and a `std::clamp` on path contest. **A four-stage clamp chain on one quantity.** |

### Dead vs live (from the defaults)
**LIVE:** `ENABLE_PASSER_V3=true`, `ENABLE_PASSER_BLOCKADE_QUALITY=true`, `ENABLE_RP_KPK_DRAW=true`,
`PASSER_CONTEST_STOP=90`, `PATH=40`, `REAR_ENEMY=128`, `REAR_OWN=48`, `KING_FAR=16`, `KING_HELP=6`,
`MAG_SCALE=100`, `R_CAP=320`, `R_FLOOR=64`, `R_MAX=384`, `CONTEST_PCT=30`, `DANGER_D2=70`, `D4=24`,
`PP_*` (6 knobs), `PPS_*` (4 knobs), `ROOK_PASSER_OWN/ENEMY`, `KRACE_MAG=100`.
**DEAD (0/false):** `ENABLE_PASSER_DANGER`, `ENABLE_PASSER_DETECT_SF`, `ENABLE_PASSER_V2`,
`ENABLE_PASSER_ORD_FLOOR`, `ENABLE_PASSER_KRACE_MG`, `ENABLE_PASSER_DEFER_ON_FLAG`,
`ENABLE_PASSER_PRUNE_EXEMPT`, `PASSER_BLOCK_ADV`, `PASSER_CANDIDATE_DOCK`, `PASSER_ENEMY_CREDIT_PCT`,
`PASSER_RESID_PCT`, `PASSER_RFLOOR_R5/R6`.
⇒ **12 of 48 are switched off**, and four of those are the *repair attempts* for the defect below.

---

## 1. ☠️ THE DEFECT, CONFIRMED FROM SOURCE (not inherited from the memory)

```c
int base = mag * Config::PASSER_RESID_PCT / 100;   // PASSER_RESID_PCT = 0  =>  base = 0
int val  = base + (mag - base) * R / 256;          // => val = mag * R / 256, PURELY MULTIPLICATIVE
```
`R` is clamped to `[0, 320]` and can reach ~0. v1's own comment records the measured consequence:

> *"once R falls below ~21/256 the passer collects LESS than the same pawn would have earned for NOT being
> passed (measured: w1's f2 on the 6th took 6 mp where `default_midgame_pawn_rank_bonus[6]` = 90)"*

★ **Being recognised as passed can make a pawn worth less.** Both repairs (`PASSER_RESID_PCT`,
`ENABLE_PASSER_ORD_FLOOR`) are **default-off because they measured worse inside v1** — the
accidental-load-bearing signature: the fix fails because every neighbouring constant was fitted around the
broken mechanism. ⇒ It cannot be repaired in place, which is precisely why it belongs to the rebuild.

### What the references do instead
| | form |
|---|---|
| **SF11 / SF15.1** | `bonus = PassedRank[r]` granted **UNCONDITIONALLY**; the entire safety analysis sits inside `if (pos.empty(blockSq))`, so SF **WITHHOLDS upside when the stop square is occupied and never SUBTRACTS**. It has no blockade penalty at all |
| **Ethereal** | same shape: table granted, modifiers additive |

⇒ **2b IS THE STRUCTURAL REPLACEMENT, NOT A 5TH MECHANISM.** The graveyard names exactly this as what
remains genuinely unbuilt: *"pricing a near-promotion passer under a SINGLE owner in SF's ADDITIVE form —
replacing the multiplicative form, not re-crediting on top of it."*

---

## 2. ARCHITECTURE — where 2b lands relative to 2a

### Layer A (detector, PURE, cacheable) — EXTENDED, not replaced
`passed[2]` and `candidate[2]` join the existing masks. ✅ Both are pure functions of the two pawn
bitboards, so the cacheability contract is preserved exactly.
☠️ **The detector RETURNS them.** v1's `getPPIncrement` PUBLISHES them as a side effect, which is what makes
a pawn hash unsafe there — ten piece evaluators read those masks, and a cache hit would skip the write.

### Layer C (passer value, piece/king dependent, NOT cacheable) — NEW
```
value = PassedRank[r]                        // UNCONDITIONAL. never scaled, never zeroed
      + free_to_advance_bonus                // only when the stop square is empty (SF's gate)
      + path_safety_bonus                    // upside only
      + king_proximity_bonus(ours) - (theirs)
      + rook_behind_bonus
```
**No multiplier anywhere. No R. No clamp chain.** Every modifier is additive with its own sign, exactly as
both references do it.
⚠️ `passer_realizability_R` and its four-stage clamp chain are **NOT ported**. Neither is `passer_danger`
(a 5th mechanism by definition), nor `PASSER_ENEMY_CREDIT_PCT` — ☠️ re-enabling that is an **UN-FIX**: its
zeroing was "Gap-P P1", a component of a bundle that shipped **+38.7 ±27 Elo over 875 games**.

### ☠️ `passer_king_race_one` must be rewritten, not ported
Scan 3 caught it writing **`whiteKingSquare`, `blackKingSquare`, `whiteMat`, `blackMat`, `g_ae_matedrive`**
— a scoring helper mutating board-level globals. That is incompatible with v2's zero-global contract and
would silently corrupt any caller that reads those afterwards. The king-race CONCEPT survives; the
implementation does not.

---

## 3. ▶️ THE TWO QUESTIONS 2a HANDED TO 2b (owner, 2026-09-12)

### Q1 — does a CONNECTED / phalanx passer earn extra? The references DISAGREE.
| | a connected passer receives |
|---|---|
| SF11 | passer bonus **+** connected bonus, ADDITIVE (`Connected[]` applies to every connected pawn) |
| Ethereal | passer bonus **ONLY** — `else if` explicitly denies connected to passed pawns |

★ The owner's chess intuition (connected passers, and phalanxes of passers, are far stronger) backs SF.
⇒ Genuine two-way split under `adopt-reference-methods-only-if-universally-superior` ⇒ **ours to measure.**
⇒ **This is also the PARKED connected term's re-try.** It is retested HERE, conditioned on passed-ness,
not as a standalone repeat of the 2a sweeps.

### Q2 — chain or sum?
Ethereal's pawn terms are an if/else-if CHAIN (mutually exclusive categories); SF stacks additively.
⚠️ The 2a exclusivity test was a NO-OP because `backward`/`connected` and `isolated`/`connected` are
already disjoint **by construction** in our detector. With `passed` present the question becomes real:
does a passed pawn still collect `isolated` / `doubled` / `backward`?
⇒ `PS_V2_CONN_EXCL=2` already reserves "also exclude passers"; 2b implements and measures it.

---

## 4. ⚠️ THE COUPLING 2a ALREADY MEASURED
`isolated ↔ passed` lift **2.57×**; `weak_unopposed → passed` **4.27×** (which is why weak-unopposed was
dropped from 2a). ⇒ An isolated PENALTY and a passer BONUS fire on the same pawn far above chance, in
opposite directions. That is the components-cancel pattern, and 2b is where it becomes live.
★ **Re-run the overlap + lift matrix with `passed` populated before choosing any constant** — the 2a matrix
could not include it.

---

## 5. ORDER OF WORK
| step | content | gate |
|---|---|---|
| **2b-1** | `passed` + `candidate` into the detector; port SF's candidate cases verbatim (`ENABLE_PASSER_DETECT_SF` is OFF in v1, so we miss **~14% of SF's passers**) | 🧰 `passer_detector_diff.py` measures the gap as a pure predicate count, no engine. Then the mask ORACLE, extended |
| **2b-2** | re-run the overlap/lift matrix WITH `passed` | decides whether isolated/weak terms need conditioning |
| **2b-3** | Layer C in ADDITIVE form, single owner | §I, then the 2a+2b games run |
| **2b-4** | Q1 + Q2: connected×passed, and chain-vs-sum | §I; the parked connected term lives or dies here |
| **2b-5** | pawn hash folded in over 2a+2b together | byte-identity cached vs uncached |

⚠️ **Screening discipline from the graveyard:** validate on the de-biased regret set + games, **never**
`passer_corpus.csv under_fire` — tuning to it is anti-correlated with Elo (proven: `MAG=150` read as a win
on corpus and movematch while costing −54 STS).
⚠️ **The phantom-knob law:** `SCALE_PASSED_PAWN` covers 18% of passer mass, `PASSER_MAG_SCALE` 32%, ~68%
under neither ⇒ **every past passer ablation measured a FRACTION; none of their nulls are nulls.** Quote
this whenever a v1 passer result is cited as evidence.

---

## 6. OPEN
1. ⏳ Does our `passed` detector need the rear-doubled exclusion (v1's `ENABLE_PASSER_V3`)? A pawn with a
   friendly pawn ahead on its file can never promote — but our 2a `doubled` mask flags the REAR pawn, so
   the interaction with 2b's `passed` needs stating explicitly rather than inheriting.
2. ⏳ `passer_block_quality` is **entirely hardcoded** (50/60/63/100/110, no knobs). It is our answer to
   `passer-defect-is-blockade-cost-blindness` and worth keeping — but as SF-style WITHHELD UPSIDE, never as
   a subtraction, and with its magnitudes promoted to knobs.
3. ⏳ Which of the ten piece evaluators genuinely need the passer masks in v2? In v1 all ten take them; that
   is the coupling that makes the pawn layer un-cacheable, and most of it is probably rook-behind-passer
   (rung 6) plus blockade (here).

---

## 7. FIVE-ENGINE COMPARATIVE REVIEW (2026-09-12)

Sources: SF1 (2008), SF11, SF15.1 read LOCALLY; Ethereal and Weiss FETCHED (see
[[reference-engine-sources]] -- they are not on disk). Conversions: SF mg x7.81 / eg x4.69, Ethereal
mg x12.20 / eg x6.94. ⚠️ **Weiss is UNCONVERTED** -- its piece values are macros we have never resolved,
so only its SHAPE is comparable, never its magnitudes.

### FORM -- and it is UNANIMOUS
| engine | base | granted how | safety handled by |
|---|---|---|---|
| SF1 (2008) | formula `20*tr` mg / `10+10r^2` eg, `tr = max(0, r(r-1))` | unconditional | additive king-distance, scaled by `tr` |
| SF11 | `PassedRank[rank]` | **unconditional** | upside WITHHELD inside `if (empty(blockSq))`, never subtracted |
| SF15.1 | `PassedRank[rank]` | **unconditional** | same |
| Ethereal | `PassedPawn[canAdvance][safeAdvance][rank]` | **unconditional** | ★ safety SELECTS WHICH TABLE, never scales |
| Weiss | `PawnPassed[rank]` | **unconditional** | additive `PassedBlocked[4]` / `PassedFreeAdv[4]` |
| **v1** | `passed_rank[r]` **x R/256** | ☠️ **MULTIPLIED**, R in [0,320] | multiplicative realizability |
| **v2 (proposed)** | `table[rank]` | unconditional | additive |

★ ★ **4 of 4 references grant unconditionally and modify ADDITIVELY. NOT ONE multiplies by a
realizability factor.** Universal under `adopt-reference-methods-only-if-universally-superior` => adopt
without further debate. This confirms the graveyard's law from FIVE independent sources rather than one.
★ Ethereal's 2x2 table indexing is a fourth distinct way to express safety and the most interesting:
**safety picks a table, it does not scale a value** -- so a "stopped" passer still gets a full, separately
tuned number rather than a fraction of another one.

### MAGNITUDE -- in our milli-pawns
| rel rank | SF11 mg/eg | Ethereal [1][1] mg/eg | Weiss (RAW, unconverted) | **v1 mg/eg** |
|---|---|---|---|---|
| 2 | 78 / 131 | **-342** / 160 | **-14** / 25 | **65** / 90 |
| 3 | 133 / 155 | **-488** / 243 | **-19** / 40 | **160** / 210 |
| 4 | 117 / 192 | **-671** / 416 | **-72** / 115 | **285** / 360 |
| 5 | 484 / 338 | 98 / 618 | -38 / 146 | 625 / 750 |
| 6 | 1312 / 830 | 1159 / 1152 | 60 / 175 | 840 / 990 |
| 7 | **2156** / 1219 | 1513 / **2033** | 311 / 218 | **1085** / 1260 |

☠️ ★ **Ethereal AND Weiss both price an UNADVANCED passer as a MIDGAME LIABILITY** (negative through
ranks 2-4/5). Mechanically sensible: it is a pawn no neighbour can defend, so it is a target, and it is far
from promoting. SF11 is mildly positive. **v1 pays 65/160/285 at exactly those ranks, and then UNDER-pays
at the top (1085mg at rank 7 where SF11 gives 2156).**
=> **v1's passer taper is COMPRESSED AT BOTH ENDS: too generous where two references go negative, too
stingy where all of them go large.** This is INDEPENDENT of the multiplicative defect and would still be
wrong after fixing it. => v2 seeds a taper that is flat-or-negative at low rank and steep at high rank.

### ★ ★ Q1 ANSWERED -- and Weiss supplies a better form than either engine previously checked
| engine | does a connected / defended PASSER earn extra? |
|---|---|
| SF11 | YES -- `Connected[]` applies to every connected pawn including passers, additively |
| Ethereal | NO -- the `else if` explicitly denies connected to passed pawns |
| **Weiss** | **YES -- via a DEDICATED `PassedDefended[RANK_NB]` table**: `S(0,0) S(0,0) S(3,-14) S(3,-10) S(0,33) S(32,103) S(158,96) S(0,0)` |

=> **2 of 3 say YES**, so the owner's chess intuition holds and Ethereal is the dissenter.
★ ★ More useful than the vote: **Weiss owns the interaction in a SEPARATE TERM** rather than reusing
the connected bonus. That is a third design neither SF nor Ethereal offers, and it fits our one-owner rule
exactly -- **the parked connected term STAYS parked, and a new `PassedDefended[rank]` owns the
passer-x-defence interaction outright.** It also sidesteps the no-5th-valuation-mechanism instruction,
because it is not a passer VALUATION mechanism: it is a detector-conditioned table, like every other one.
⚠️ Note the shape: Weiss's PassedDefended is ~0 below rank 4 and explodes at 5-6. Defence only matters
once the pawn is close enough for the defence to be decisive.

### Also worth stealing / noting
- **Weiss `PassedRookBack = S(21,46)`** -- a dedicated rook-behind-passer term. v1 has `ROOK_PASSER_OWN=50`
  / `ROOK_PASSER_ENEMY=25` inside the ROOK evaluator. => v2 owns it at rung 6, reading 2b's `passed` mask.
- **Weiss `PassedSquare = S(-26,422)`** -- the square rule (unstoppable passer), a near-pure ENDGAME term.
  v1's `passer_king_race_one` is our version and it ☠️ writes board globals (scan 3); the concept
  survives, that implementation does not.
- **SF1's `tr = max(0, r(r-1))` scaling of the king-distance terms** -- king proximity matters more the
  further advanced the pawn. All later engines keep this shape; ours does not express it.

---

## 8. FULL PASSER MECHANISM REVIEW (owner, 2026-09-12) -- beyond the base table

⚠️ **Correction to section 1 and section 7: v1 is RICHER on these dimensions than I implied. It has every
one of these concepts.** Its problems are FORM (multiplicative) and MAGNITUDE, not absence.

### 1. CONNECTION between passers
| | how |
|---|---|
| SF11 | no term in `passed()`, but `Connected[]` from pawns.cpp applies to passers too. Plus `bonus /= 2` if the square ahead is not itself passed |
| Ethereal | NO -- excluded. `if (several(forwardFileMasks(US, sq) & myPassers)) continue;` is an ANTI-DOUBLE-COUNT for two passers on one file, not a bonus |
| Weiss | YES -- `PassedDefended[rank]`, passer defended by a friendly PAWN |
| **v1** | **YES** -- `PP_DIAG_SUPPORT=75`, `PP_HORIZ_SUPPORT=225` (phalanx), `PP_FILE_CLEAR=150`, inside `getPPIncrement` |

★ ★ **MAGNITUDE is the finding.** Weiss rank 6: `PassedDefended` 158 vs `PawnPassed` 311 => **defence
adds ~51% on top of the base passer bonus** (~53% at rank 5). That RATIO transfers even though Weiss's
units do not. In v1, `PP_HORIZ_SUPPORT=225` raises `ppIncrement`, which reaches the score only as
`(ppIncrement >> 3)` ~ **+28mp on a base of 840-1085mp: about 3%.**
=> **v1 UNDER-PRICES connected passers by roughly 17x.** The owner's intuition is not just right, it is
quantifiable, and it explains why the concept reads as absent despite being implemented.

### 2. SUPPORT FROM FRIENDLY PIECES -- NOT universal
| | definition of "support" |
|---|---|
| SF11 | our ROOK/QUEEN BEHIND on the file **OR** ANY friendly piece attacking the STOP SQUARE -> one flat `k += 5`, then scaled by `w = 5r-13` |
| Ethereal | **NONE AT ALL** |
| Weiss | `PassedRookBack` -- friendly RO✅ anywhere behind on the file (via `Fill`). Rooks only, not queens |
| **v1** | richest: `PPS_OWN_BLOCK=75`, `PPS_OWN_ATTACK=60`, rank-scaled, plus `ROOK_PASSER_OWN=50` |

### 3. ☠️ BLOCKADING -- I OVER-CLAIMED IN SECTION 1. THE REFERENCES DISAGREE THREE WAYS.
| | how |
|---|---|
| SF11 | **WITHHOLDS** -- the whole `k` bonus is inside `if (pos.empty(blockSq))`; never subtracts. Also: an enemy ROOK/QUEEN BEHIND our passer makes the ENTIRE span count unsafe |
| Ethereal | **SELECTS A DIFFERENT TABLE** -- `canAdvance` is an INDEX, so a blocked passer gets its own separately tuned number |
| Weiss | **SUBTRACTS** -- `PassedBlocked[4] = S(1,-4) S(-6,6) S(-11,-11) S(-54,-52)`, genuinely negative when advanced |
| **v1** | subtracts -- `PP_BLOCKADE_PEN=100`, `PPS_ENEMY_BLOCK=100`, `passer_block_quality` |

=> Section 1 said "SF WITHHOLDS and never SUBTRACTS" as though it settled the design. **It does not: Weiss
subtracts, exactly as v1 does.** Three-way split = ours to choose by measurement, not to inherit.

### 4. KING PLACEMENT -- UNIVERSAL, and v1 GETS THIS RIGHT
| | form | enemy : friendly weight |
|---|---|---|
| SF11 | `(enemyDist(blockSq)*19/4 - ourDist(blockSq)*2) * w`, **ENDGAME ONLY**, plus a second-push term | **2.4x** |
| Ethereal | `dist * PassedFriendlyDistance[rank]` and `* PassedEnemyDistance[rank]` (distance to the PAWN) | enemy larger |
| Weiss | `Dist(forward, ourK) * PassedDistUs[r]`; `(rank-3) * Dist(forward, theirK) * PassedDistThem` | enemy larger |
| **v1** | `PASSER_KING_FAR=16`, `PASSER_KING_HELP=6` | **2.7x** ✅ |

★ All four weight **"their king cannot get there" far above "our king escorts"**. ✅ v1's 16:6 already
matches. ⚠️ It only PARTLY absorbs piece support: SF applies king distance to the STOP SQUARE and in the
ENDGAME ONLY, while piece support applies in both phases.

### 5. ★ WHAT WAS MISSING FROM MY EARLIER REVIEW
- ★ ★ **RANK GATING is universal and we violate it.** SF11 gates every extra at `r > RANK_3`; Weiss
  does `if (rank < RANK_4) continue;`. **Low-rank passers get the base table and NOTHING else.** v1 runs the
  entire `R` machinery -- contest, path, king race, blockade -- on every passer at every rank.
- ★ **`PassedFile`** (SF only): `- S(11,8) * map_to_queenside(file)` => **EDGE passers are worth MORE
  than central ones**, because the defending king is usually central. v1 has no passer file term.
- ★ **Enemy ROOK/QUEEN BEHIND our passer** (SF only) -- flips the whole span to unsafe.
- ★ **The SQUARE RULE**: Weiss `PassedSquare = S(-26,422)`, gated to PAWNLESS endgames. v1's
  `passer_king_race_one` is the analogue; scan 3 caught it writing board globals.
- ★ **SF's candidate halving**: `bonus /= 2` when the pawn ahead is not itself passed.

### => CONSEQUENCES FOR THE 2b DESIGN
1. **Rank-gate the extras** (universal, and we violate it): base table at all ranks, machinery only at 5+.
2. **Connected-passer bonus seeded at ~50% of the base at ranks 5-6** (Weiss ratio), not v1's ~3%.
3. **Blockade is an OPEN CHOICE, not settled** -- withhold (SF) vs table-select (Ethereal) vs subtract
   (Weiss/v1). Add it to the 2b experiment list.
4. **Piece support is NOT universal** -> build it, but screen it; Ethereal wins without it entirely.
5. **King distance: port v1's ratio (2.7x), not its placement** -- and make it endgame-weighted like SF.
6. ⏳ `PassedFile` (edge > centre) is SF-only -> CANDIDATE, not adoption.

---

## 9. DECISION TABLE + VERIFICATION PLAN (2026-09-12)

### The ordering principle
Rung 2a produced the only predictor we have validated: **a rung pays when the core has NO REPRESENTATION of
that information.** KS paid (+101 Elo) because v2 could not express "this king is in danger". Connected
failed because square-quality was already represented. Everything below is ranked by what v2 currently
cannot express at all.

### TIER 1 -- build, high confidence
| item | why | risk |
|---|---|---|
| **Detection correctness** (~14% SF gap) | if a passer is not detected, no valuation matters. ★ It is also OUTSIDE the valuation graveyard -- the graveyard names DETECTION as one of two genuinely-untested boxes | none: a predicate, verified against the oracle with no constant involved |
| **Unconditional additive base table** | 4/4 references, no dissent. v2 has ZERO passer representation today | magnitude, not form |
| **King distance, both kings, enemy ~2.5x** | 3/3 universal; v2 has NO king-pawn interaction of any kind | low -- v1's 16:6 already matches, so port the RATIO not the placement |
| **Rank gating of the extras** | 2/2 universal (SF `r > RANK_3`, Weiss `rank < RANK_4 continue`); v1 violates it | none -- a restriction, can only reduce firing |

### TIER 2 -- build, then SCREEN. The references disagree, so nothing here is inheritable.
| item | split |
|---|---|
| Blockade | withhold (SF) / table-select (Ethereal) / subtract (Weiss + v1) |
| Connected passer | yes (SF, Weiss) / no (Ethereal) -- use **Weiss's SEPARATE `PassedDefended[rank]`**, which keeps connected parked and gives the interaction its own owner |
| Piece support | SF lumped / Weiss rook-only / Ethereal NONE |
| `PassedFile` (edge > centre) | SF only. Lowest priority |

### TIER 3 -- ☠️ cheap AND invisible, which is the trap
**Computationally all of 2b is cheap**: rung 1 already pays for the attack maps, so path safety, blockade
and support are lookups, and king distance is two `distance()` calls per passer.
★ ★ But **"small aggregate effect" and "invisible to §I" are the SAME SET**. `doubled` fires on 1.48%
of pawns (~4mp aggregate) yet is SF's LARGEST pawn penalty. The **square rule** is the extreme: Weiss gates
`PassedSquare = S(-26,422)` to PAWNLESS endgames -- it almost never fires and is DECISIVE when it does.
=> Judge rare-but-decisive terms by **whether the position class exists**, never by aggregate accuracy.
Otherwise we park exactly the terms that win won endgames.

### HARMFUL PATTERNS -- from measurement, not speculation
| pattern | evidence |
|---|---|
| multiplicative forms that can reach zero | v1's `R` makes a passer worth LESS than a non-passer (6mp vs 90mp, measured). 4/4 references avoid it |
| terms duplicating existing representation | connected at 2a: harmful at every magnitude AND both shapes |
| magnitudes imported without conversion | connected 13x; rung 1 safe checks 7-12x |
| re-crediting what another term already docked | `PASSER_ENEMY_CREDIT_PCT` -- its zeroing shipped **+38.7 Elo**. Turning it on is an UN-FIX |
| clamps on sums of competing quantities | `PAWN_CLAMP` masked two terms 14x/23x, signs INVERTED |

### ☠️ THE DOUBLE-COUNT MAP
1. ★ ★ **Passer table x PST rank ramp -- the connected failure, scaled up 10x.** The PST already
   prices advancement; a passer table of 1085-2156mp on top is the same double-count that killed connected,
   at an order of magnitude more money.
   ✅ **Mitigating fact: SF's `PassedRank[]` is ALREADY a delta over SF's own psqt**, so porting it as an
   additive bonus over our PST is structurally correct -- **PROVIDED our PST's pawn rank ramp is comparable
   in scale to SF's.** ⚠️ That comparison is open item #2 from the 2a design, which I SKIPPED, and which
   would have predicted connected's failure. **It is now a Tier-1 pre-check, run before any constant.**
   ✅ v1 already solves its own version via `ENABLE_PASSER_V3` deferral (a flagged passer does not collect
   the ordinary rank bonus). v2 must make the same decision explicitly.
2. **Passer x isolated** -- 2.57x lift, opposite signs, same pawn. Decide explicitly (Q2); do not let it
   happen by default.
3. ★ **Blockade x free-to-advance are the SAME BINARY.** Rewarding "free" AND penalising "blocked"
   prices one question twice. Weiss does both but in an `if / else if`, i.e. ONE two-valued term. Additive
   both would be the convolution.
4. **Piece support x king distance** -- partial overlap (SF's king term is endgame-only and keyed to the
   stop square; piece support is both-phase). Keep both, measure the overlap once both exist.
5. **Rook-behind x rook-on-open-file** -- a scheduled collision with rung 6, not a 2b problem.

### THE ANTI-CONVOLUTION RULE
**Price each QUESTION once, and put the answer in exactly ONE table.**
- *How advanced is this pawn?* -> PST **or** passer table, never both.
- *Is it defended?* -> one term (`PassedDefended`), not connected-plus-support.
- *Can it advance?* -> ONE two-valued branch, not a reward and a penalty.
- *Where are the kings?* -> one term per king.
★ And this is now CHECKABLE rather than asserted: the detector exposes every firing set as a bitboard,
so the overlap matrix answers "do these two terms answer the same question" BEFORE a constant is chosen.
⚠️ The missing instrument is the same overlap check run against the **PST's contribution** -- exactly the
comparison that would have predicted connected's failure.

### VERIFICATION -- three layers (owner, 2026-09-12)
| layer | what it proves |
|---|---|
| **A. Pawn-rule correctness** | the detector is right under ALL legal pawn configurations, not just typical ones. Mask-for-mask against the Python oracle over the corpora, PLUS hand-built edge cases: rank-2 and rank-7 pawns, fully blocked chains, doubled passers, a passer with an enemy pawn on an adjacent file behind it. ⚠️ en passant does NOT affect passed-ness (it is a capture right, not a structural property) -- state it so nobody "fixes" it later |
| **B. Aggregate accuracy** | §I over the six general corpora. ☠️ Ranks arms; does NOT decide -- it mispriced KS by two orders of magnitude |
| **C. ★ Passer corpus as a FALSIFIER** | `ks_sets/passer_corpus.csv`, 288 positions: `blowup_guard` 140 / `control` 79 / `under_fire` 69, spread opening->adveg |

☠️ **The passer corpus is a GUARD, NOT AN OPTIMIZER.** The graveyard proves tuning to `under_fire` is
anti-correlated with Elo (`MAG=150` read as a win on corpus AND movematch while costing -54 STS).
★ ★ **But that warning is about MAGNITUDE knobs, which trade `under_fire` against `blowup_guard` by
construction (two-sided error). 2b is a STRUCTURAL change, and a structural fix CAN improve both tiers at
once where no magnitude knob can.**
=> **"Does it improve `under_fire` AND `blowup_guard` TOGETHER?" is a question only a genuine structural fix
can answer yes to.** That is a falsifier for the structural claim, not a tuning target -- a legitimate and
stronger use of the corpus than aggregate §I, and it does not violate the standing rule.
⚠️ A change that improves `under_fire` while worsening `blowup_guard` is the KNOWN TRAP and is rejected
regardless of its mean.

---

## 10. TIER-1 PRE-CHECK RUN (2026-09-12) -- and it corrects the rung-2a diagnosis

This is open item #2 from the 2a design, which I SKIPPED there and which was supposed to predict
connected's failure. Run properly now, before any 2b constant.

### Neither PST ramps with rank
```
SF11 PBonus (pawn PST), mg mean by rank:   r2 ~+9  r3 ~+5  r4 ~+5  r5 ~0  r6 ~-6  r7 ~-3
OUR  whitePlacementLayerBase[0], by rank:  0, 13.8, 11.8, 23.0, 25.5, 25.0, 25.0, 0
```
Both price FILE / CENTRE, not advancement; ours is flat at ~25mp from rank 3 and **zero at rank 7**.
✅ => **The passer x PST double-count I called the biggest risk DOES NOT EXIST.** Porting `PassedRank[]` as
an additive bonus over our PST is structurally correct -- and for a STRONGER reason than section 9 gave:
not "SF's is already a delta", but "neither PST prices advancement at all".

### ☠️ ☠️ This corrects the rung-2a diagnosis of CONNECTED
The 2a log says connected was "~13x our scale", comparing `PS_CONN_RANK_MP` (1343mp) against v1's
non-passed rank bonus (105mp). ⚠️ **That table lives in `evaluate_pawns_*` and DOES NOT EXIST IN v2.**
Against v2's actual pawn placement signal (~25mp) connected was **~50x**.
=> **Connected did not DOUBLE-COUNT. It OVERWHELMED** -- one pawn relation outweighing the entire placement
layer of a three-term eval.

### ★ ★ THE LAW THAT EXPLAINS BOTH RUNGS: magnitude x FIRING RATE
| rung | magnitude | firing rate | outcome |
|---|---|---|---|
| KS (rung 1) | up to 4000mp | gated by ONSET to positions with real danger | **+101 Elo** |
| connected (2a) | up to 1343mp | **~45% of pawns** (phalanx 30.6% u supported 26.7%) | harmful at every magnitude |
| passers (2b) | up to 2156mp | **11.72%**, and the universal RANK GATE cuts it much further | ? |

=> A thin eval has no other terms to absorb a large, FREQUENT contribution. Large-and-rare is fine
(KS proves it); large-and-frequent is not (connected proves it).
✅ **For 2b the two constraints CONVERGE**: the rank gate every reference applies is also exactly what
solves our thin-eval magnitude problem, and it cuts firing precisely where the table is largest. First time
a universal design choice and a v2-specific need have pointed the same way.

### Registered prediction
The base passer table may survive at NEAR-REFERENCE magnitude **because of the rank gate**. ⚠️ Sweep
`PASSER_MAG` from the first run regardless -- I have now been wrong three times about reference magnitudes
transferring (SF's doubled taper, the 2a magnitude optimum, rank-flat connected).

---

## 11. 2b BUILT AND MEASURED (2026-09-12)

### Gates -- all pass
arm 0 byte-identical `250 / 35,310,778 / EBF 3.784` - 2b off `249 / 62,014,960` - 2b on
`250 / 60,474,175` (differs => live) - **detector oracle 11/11 predicates, 66,000 masks, ZERO mismatches**
- colour invariance **0/800 at TOL=0** - file mirror `21 (3.2%) worst 5mp`, unchanged from baseline.

### Detection gap CLOSED and quantified
`candidate` fires on **1.54%** of pawns against `passed` at **11.72%** => the SF rule finds **13.1% more
passers**, independently reproducing the recorded ~14% figure by a different route.

### ☠️ The oracle earned itself
`initialize_attack_tables()` runs in `ChessAI.__cinit__` -- at ENGINE CONSTRUCTION, not module import. The
module-level probe read `passed_span_white` as ALL ZEROS, so `stoppers` was empty and **every pawn was
flagged passed, back-rank pawns included**: 2,735 `passed` + 362 `candidate` mismatches from one cause.
★ A detector reading a zero mask does not crash and does not look wrong -- more passers, higher eval,
plausible numbers. Recorded as [[runtime-tables-are-empty-outside-an-engine-instance]].

### ☠️ The MIDGAME leg is the harmful half
| arm | mean% | worst% |
|---|---|---|
| mag 6, both legs | -0.15 | +0.52 |
| **mag 6, eg only** | -0.15 | **+0.11** |
| mag 12, both legs | -0.25 | +1.11 |
| **mag 12, eg only** | -0.29 | **+0.25** |

Killing the mg leg roughly HALVES the worst case at equal mean. => `PASSER_V2_MG_PCT` now defaults to 0.
⚠️ Cause: our converted table is midgame-heavy BY CONSTRUCTION. SF's `S(276,260)` becomes mg 2156 /
eg 1219 because SF's endgame pawn is worth more (213 vs 128) while **ours is FLAT at 1000 in both phases**.
The conversion is arithmetically faithful; it inherits a phase relationship our eval does not have.

### The eg-only frontier
| mag | 6 | 12 | 20 | 30 | 45 | **60** | 110 |
|---|---|---|---|---|---|---|---|
| mean% | -0.15 | -0.29 | -0.44 | -0.58 | -0.71 | **-0.72** | -0.02 |
| worst% | +0.11 | +0.25 | +0.47 | +0.81 | +1.46 | +2.27 | +6.16 |

★ ★ **Same shape as rung 1's ONSET frontier -- which we MEASURED to be worth +0.1 +/- 50 Elo, i.e.
FLAT.** The rung-1 law applies: *adding the channel pays; tuning where it fires does not.*
=> **Stop optimising this frontier. Pick a conservative point and go to games.**

### ☠️ The corpus split, which is the largest we have seen
At full magnitude the SAME term is **-9.27% on `game_regret_set_uho`** (monotone, still improving) and
**+16.74% on `game_regret_set`**. Both large, opposite signs. ⚠️ Unexplained -- I stopped theorising
after two hypotheses failed to survive contact (phase distribution, baseline fit). It is a real,
systematic, reproducible split and it is the most interesting open question of the rung.

### Predictions scorecard
WRONG -- "the base table may survive at near-reference magnitude because of the rank gate". The optimum is
mag ~20-60 eg-only, i.e. **a fifth to a half of reference scale**. Fourth consecutive miss in the same
direction: assuming reference magnitudes transfer.
☠️ **The rank-gate test was a NO-OP** -- `PASSER_V2_MIN_RANK=0` produced output IDENTICAL digit-for-digit
to the default, because SF's own `w = 5r - 13` is already <= 0 below the 4th rank and we clamp `w < 0 -> 0`.
The gate is a SPEED early-out, not a behavioural one. **Second time I built a test that could not
discriminate** (the 2a exclusivity test was the first). => Before running an arm, check whether the knob
can change the output AT ALL.
RIGHT -- king distance helps (removing it costs 0.9pp) and the candidate discount helps slightly (0.17pp).

### => RECOMMENDED and OPEN
Ship **eg-only at mag 20-30** (worst +0.47 to +0.81%) and take 2a+2b to GAMES. §I demonstrably cannot price
this class: it read KS at -1.44% and games said **+101 Elo**.
⚠️ 2b never reaches worst <= 0, unlike 2a (-0.66% / -0.00%). That is a real difference and the games run
must be read with it in mind.

---

## 12. ★ ★ HOLISTIC PAWN CLOSE (owner, 2026-09-12) -- the material TAPER dominates the rung

Prompted by the owner: "since it's all related, do a final look at pawns holistically."

### The observation that started it
**Every pawn term that SHIPS is endgame-weighted; every midgame-heavy one FAILED.**
| term | phase weighting | outcome |
|---|---|---|
| doubled 86mg / 263eg | eg-heavy | SHIPS |
| isolated 0mg / 100eg | eg-only | SHIPS |
| backward 0mg / 113eg | eg-only | SHIPS |
| connected (rank table, both legs) | mg-heavy | FAILED at every magnitude |
| passers, both legs (2156 / 1219) | mg-heavy | FAILED |
| passers, eg-only | eg-only | WORKS |

Six terms, no exceptions. Mechanism: **our pawn is FLAT at 1000 in both phases while `EG_EXIST_*` tapers
PIECES up in the endgame with no pawn entry** -- so relative to pieces our pawn gets CHEAPER toward the
endgame where SF's gets 52% DEARER (128:764 -> 213:848).

### The test, and the parked-feature law firing exactly as written
`EVAL_V2_PAWN_MG` was built at rung 0.5, measured **NULL**, and parked with an explicit trigger:
*"re-test after rung 2 -- a thin eval has no pawn structure for it to act on."* We are at rung 2.

| arm | mean% | worst% |
|---|---|---|
| pawn layer alone (2a + 2b@25) | -0.97 | +1.86 |
| taper alone (MG 900) | **-1.67** | +0.47 |
| taper alone (MG 700) | -4.21 | +2.20 |
| layer + taper 900 | -2.77 | **-0.72** |
| layer + taper 700 | -5.57 | -1.81 |
| layer + taper 570 | -6.81 | -0.80 |
| **layer@60 + taper 600** | **-6.59** | **-2.05** |

★ ★ ★ **The taper alone outweighs the ENTIRE pawn layer** (-1.67 vs -0.97), and they are
**SYNERGISTIC, not redundant** -- additive would be -2.64, actual -2.77. I predicted redundancy if the
terms were compensating for the missing taper; **wrong, and informatively so.**
★ **The taper FIXES the layer's worst case**: layer alone +1.86% (on `game_regret_set`), with taper
-0.72%. => **The pawn layer was being blamed for a MATERIAL defect.** The +16.74% on `game_regret_set` at
high passer magnitude -- the corpus split section 11 could not explain -- was the FLAT PAWN VALUE amplified.
★ **And the correction is MUTUAL**: taper alone is -5.07% mean but **+3.47% worst** (it hurts UHO); the
pawn layer fixes that. Each patches what the other breaks.
★ **With the taper in, the passer term wants to be BIGGER**: at MG 600, mag 25 gives -6.57/-1.19 and
mag 60 gives -6.59/**-2.05**. Its optimum was being SUPPRESSED by the material defect.
✅ **A reference magnitude finally transferred**: SF's pawn ratio is 128/213 = **0.60**, Ethereal's
82/144 = **0.57**, and the measured sweet spot is **570-600**. First transfer after four straight misses.

### ☠️ ☠️ THE CAVEAT, and it is the big one
| gate | result |
|---|---|
| arm 0 | 250 / 35,310,778 ✅ byte-identical |
| colour invariance | 0/800 at TOL=0 ✅ |
| file mirror | 21 (3.2%) worst 5mp, unchanged ✅ |
| **WAC** | **245/300, 68.8M nodes** (rung 1: 250 / 35.3M; 2a+2b: ~250 / 60M) |

**The config with the best §I accuracy solves FIVE FEWER WAC positions and uses 11% more nodes.**
A -6.59% corpus improvement paired with a tactical-suite regression is EXACTLY the
`corpus-fit-is-anti-correlated-with-elo` signature. ⚠️ This is simultaneously the strongest candidate the
rebuild has produced AND the one most likely to be a corpus artifact. **Games decide. Nothing here ships on
§I.**
⚠️ `EVAL_V2_PAWN_MG` changes MATERIAL, which moves the eval's SPREAD far more than any positional term --
so `RFP_MARGIN`, `FUTILITY_MARGINS` and the razor constants are now mis-sized against it. Per the plan,
quote **eval + margins together** and re-sweep margins at the checkpoint. The WAC node rise may be exactly
this rather than a decision-quality loss.

### => RECOMMENDED for the 2a+2b GAMES RUN
```
EVAL_ARM=1  PS_V2_MAG=100  PASSER_V2_MAG=60  EVAL_V2_PAWN_MG=600
(+ the rung-1 KS config)
```
with a second arm at `EVAL_V2_PAWN_MG=1000` (taper off) to separate the taper from the pawn layer, because
§I says the taper is the larger half and that claim has never seen a game.
