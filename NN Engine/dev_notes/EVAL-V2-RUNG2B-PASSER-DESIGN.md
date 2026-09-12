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
