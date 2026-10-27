# King-Safety Phase Transition — SF11 / SF15.1 / Ethereal vs. ours (2026-08-14)

**Question.** Where and when does king safety turn on/off across the midgame→endgame transition in the
reference engines, and what is our phase/king-safety model missing or misdefining? Sources read directly:
SF11 classical (`stockfish_11_linux/.../src/`), SF15.1 classical (`stockfish_15_linux/.../src/`), Ethereal
master (`src/evaluate.c`, fetched 2026-08-14), and our `cpp_bitboard.cpp` / `search_engine.h` /
`move_gen.h` / `search_engine.cpp` as of this working tree. Line numbers are as-of-today; symbols are the
durable cite.

**Established diagnosis this maps onto** (given): move-regret shows KS HURTS at npm≤12, HELPS at npm≥13;
harm is MATERIAL-located, queens a secondary modulator; owner worry = phase definition may mislabel
(a) queenless middlegames as "endgame" and (b) piece-heavy queen endgames as "midgame".

---

## 1. Side-by-side: phase + KS transition mechanisms

### 1.1 Phase definition

| | Inputs | Formula | Range / granularity | Queen's share |
|---|---|---|---|---|
| **SF11** | total non-pawn material, Mg values (N 781, B 825, R 1276, Q 2538 — `types.h:182-186`) | `gamePhase = (clamp(npm, 3915, 15258) − 3915) × 128 / 11343` — `material.cpp:132-135` (`Material::probe`), limits `MidgameLimit=15258, EndgameLimit=3915` (`types.h:188`) | **continuous** 0..128 (128 = midgame) | one queen ≈ 2538/11343 ≈ **22%** of the ramp |
| **SF15.1** | identical | identical (`material.cpp:140-143`); only `PawnValueMg` changed 128→126 (`types.h:189`), pawns don't enter phase anyway | continuous 0..128 | same |
| **Ethereal** | piece **counts**: `phase = 4·Q + 2·R + 1·(N\|B)` | tapered eval `(mg·phase + eg·(24−phase)·factor/SCALE_NORMAL) / 24` (`evaluateBoard`, src/evaluate.c) | **continuous** 0..24 (24 = midgame) | one queen = 4/24 ≈ **17%** |
| **Ours** | same counts as Ethereal: `phase = 4·Q + 2·R + 1·(N\|B)`, `MAX_PHASE=24` (`cpp_bitboard.h:101`) | `phase_score = 128·(24−phase)/24` (`placement_and_piece_eval`, cpp_bitboard.cpp:7178-7183) — note **inverted**: 0 = opening | numerically continuous, but **consumed as a 3-way boolean**: midgame `≤64`, endgame `65..96`, advanced `>96` (cpp_bitboard.cpp:7185-7196) | one queen = 4/24 ≈ 21 phase_score points |

Key observations:

- **Our phase *number* is Ethereal's exact formula.** The scalar is industry-standard, not outdated.
  What differs is *consumption*: Ethereal interpolates every term through it; we collapse it to
  `isEndGame` and select between two disjoint evaluator bodies.
- **Neither SF nor Ethereal distinguishes "queenless" from "low material" *in the phase scalar*** —
  both treat the queen as just the biggest material/count weight. The queenless distinction lives
  entirely **inside the king-safety term** (per-side queen gates, §1.4), never in phase.
- SF taper vs ours on the same position, queenless-full-material (2R+2B+2N each): SF npm 11528 →
  gamePhase ≈ 86/128 = **67% midgame**; ours phase 16 → phase_score 42 → also ≈ 67% "midgame-ness" —
  the scalars agree almost exactly. The divergence is downstream.

### 1.2 Detectors (what fires king danger)

| | SF11 `Evaluation::king()` (evaluate.cpp:372-474) | SF15.1 `king()` (evaluate.cpp:534-628) | Ethereal `evaluateKings`/safety block | Ours `king_safety_danger` (cpp_bitboard.cpp:5323-5715) |
|---|---|---|---|---|
| attacker count/weight | `kingAttackersCount × kingAttackersWeight`, weights `{N81,B52,R44,Q10}` (evaluate.cpp:81), seeded by pawn attacks on the ring (evaluate.cpp:243) | same product, weights retuned `{N76,B46,R45,Q14}` (evaluate.cpp:201) | `kingAttackersWeight` sum of dual-valued `Safety*Weight`: N S(48,41), B S(24,35), R S(36,8), Q S(30,6); plus `SafetyAttackValue S(45,34) × scaledAttackCounts` (attacks-into-area count normalized to area size ×9) | flat sum `KS_ATT_{KNIGHT..QUEEN} = {2,2,3,5}` per attacker present (5433-5437) + `KS_ATTACK_COUNT=1` per attacked zone square (5410) |
| weak squares | `185 × popcount(kingRing & weak)`, weak = attacked, not double-defended, defended at most by K/Q (387-389, 447) | `183 × …` (549-551, 601) | `SafetyWeakSquares S(42,41) × weak-in-area`, weak uses `attackedBy2` both sides | `KS_WEAK=2` per zone square attacked with no own defender (SF-style under-defended and attackedBy2 variants gated OFF: `ENABLE_KS_SF_WEAK`, `ENABLE_KS_WEAK_ATT2` — 5380-5391) |
| safe checks | scalar per type: Q780 R1080 B635 N790 (84-87), `safe` = not our-attacked or overwhelmed (392-393) | `SafeCheck[pt][single/multi]`: N{790→730,1128} B{650,984} R{1071,1886} Q{805,1292} (205-207) — **multiplicity added** | `SafetySafe*Check` per checking square: Q S(93,83), R S(90,98), B S(59,59), N S(112,117) | `KS_SAFE_CHECK=3` per safe-check square (5541-5597); per-type V2 (`ENABLE_KS_CHECK_V2`) built but OFF |
| unsafe checks | `148 × popcount(unsafeChecks)` (448) | same (602) | — | — |
| pinned defenders / blockers | `98 × blockers_for_king` (449) | same (603) | (weak-square logic) | `KS_PIN_MODE` excludes own-pinned defenders (5342-5344), default OFF |
| shelter / storm | `pe->king_safety` seeds the score (384); shelter is **mg-only** (`ShelterStrength`/`UnblockedStorm` applied to mg half, pawns.cpp:206-211); shelter mg feeds back into danger: `−6·mg_value(score)/8` (455) | identical structure (546, 609) | `SafetyShelter[2][8]`, `SafetyStorm[2][8]` dual-valued, fed through pawn-king hash `pksafety` into the **safety accumulator itself** | `KS_SHIELD=2` per shield pawn (5507-5509), `KS_STORM=1` per advanced-rank storm pawn (5524-5539), `KS_OPEN_FILE=2` (5511-5519) |
| flank breadth | `kingFlankAttack` quadratic `3·f²/8` + separate `FlankAttacks` penalty (439-457, 468) | same (593-611, 622) | — | `KS_FLANK_MODE` built (5484-5505), default OFF |
| mobility coupling | `mg_value(mobility[Them] − mobility[Us])` inside danger (452) | same (606) | — | — |

### 1.3 Thresholds, floors, transform

- **SF11 = SF15.1, unchanged across five years:** fire only if `kingDanger > 100`; then
  `score −= make_score(kingDanger²/4096, kingDanger/16)` (SF11 evaluate.cpp:460-461; SF15.1:614-615).
  **Quadratic mg, small linear eg.** The eg leak `kingDanger/16` means a real attack still counts in
  the endgame — tapered by gamePhase but *never structurally zero*.
- **Ethereal:** entry gate `kingAttackersCount[US] > 1 − popcount(enemyQueens)` — needs **≥2 attackers
  without an enemy queen, ≥1 with one**. Transform
  `eval += MakeScore(−mg·MAX(0,mg)/720, −MAX(0,eg)/20)`: quadratic mg, **linear eg**, each clamped
  to penalty-only. Then the whole eval is phase-interpolated. Three phase-coupling layers: dual-valued
  detector constants, asymmetric transform, global taper.
- **Ours:** `units` clamped ≥0 (5651); deadzone `KS_FLOOR=13` → 0 (5687, search_engine.h:1380);
  quadratic table `danger = min(units,KS_CAP=80)²/KS_DIVISOR=4` up to `KS_KNEE=12`, linear above
  (`rebuild_ks_tables`, cpp_bitboard.cpp:412-424). Output is a **single scalar** — no mg/eg pair —
  multiplied by `ks_phase_taper` and `KING_SAFETY_MAG=3000` (5734, 5797; search_engine.h:1165).

### 1.4 The queen: feeder AND gate (all three references), only feeder-plus-nudge for us

| | Feeder (queen as attacker) | Gate (attacker has no queen) |
|---|---|---|
| SF11 | weight 10 + QueenSafeCheck 780 | **`− 873 × !pos.count<QUEEN>(Them)`** (evaluate.cpp:453) — vs the fire threshold of 100 and typical real-attack danger ~1500-3000, this zeroes most queenless "attacks" |
| SF15.1 | weight 14 + SafeCheck[Q] {805,1292} | same −873, annotated **"(~24 Elo)"** (evaluate.cpp:607) — the single biggest annotated term in the danger sum |
| Ethereal | `SafetyQueenWeight S(30,6)`, `SafetySafeQueenCheck S(93,83)` | **double gate**: entry needs 2 attackers when queenless, *plus* `SafetyNoEnemyQueens S(−237,−259)` inside the sum |
| Ours | `KS_ATT_QUEEN=5` (search_engine.h:1193) | `KS_NO_QUEEN=6` units (5638; search_engine.h:1387) — **6 on a 0..80 unit scale, pre-square.** Relative to SF: −873/≈2500 typical ≈ −35% of a real attack; ours −6 vs a firing sum ≈ 20-40 units ≈ −15..30% *of units*, but because the map squares units it never zeroes anything the way −873 crossing the 100-threshold does. `KS_MIN_ATTACKERS` (Ethereal-style count gate with queen relaxation, 5680-5684) and `KS_NQ_SUP` (large suppressor inside `KS_ACCUM_MODE`, 5663) both exist but **default OFF** |

### 1.5 Guards/gates: hard vs soft

- **SF11/SF15.1:** *no phase gate at all* on `king()`. It runs at every non-lazy eval, bare-kings
  included. Phase only interpolates the mg/eg halves at the very end
  (`v = mg·phase + eg·(128−phase)·sf/64; v /= 128` — SF11:817-820, SF15.1:950-952).
- **Ethereal:** one *structural* gate — the attacker-count/queen condition. No phase gate; the eg
  half of safety survives to bare-kings, damped only by the global taper and `factor`.
- **Ours:** the primary gate is the **hard phase boolean**: the KS call sits inside `if (!isEndGame)`
  (cpp_bitboard.cpp:7203 → call at 7571-7576). `ks_phase_taper` (full ≤`KS_PHASE_FULL=48`, zero at
  `KS_PHASE_ZERO=104`, linear between — 425-437) was designed as the soft fade, but the branch
  truncates it: at phase_score 64 the taper still passes **≈71%** (256·(104−64)/56 = 182/256), and at
  65 the branch kills the term. **A 71%→0 cliff on a one-point phase move** (e.g. one minor trade).
  `KS_EXTEND_EG` (search_engine.h:1260; endgame-branch call at cpp_bitboard.cpp:7877-7885) was built
  2026-08-13 to complete the fade; **default 0 = the cliff is live**. So "KS_PHASE_ZERO is dead code"
  is now precisely: *the fade's tail (65..104) is dead unless KS_EXTEND_EG=1*.

### 1.6 Score modulation / final blend

- SF both: single tapered blend of the whole Score, with **endgame scale factors** applied to the eg
  half only (`sf/SCALE_FACTOR_NORMAL`). SF11 `scale_factor()` (evaluate.cpp:743-761): OCB 22,
  pawn-count cap `36+7·pawns`, rule50 decay. SF15.1 `winnable()` (evaluate.cpp:871-952) is much
  richer — see §1.7.
- Ethereal: `factor = evaluateScaleFactor(...)` applied to the eg half in the same shape.
- Ours: **no final tapered blend exists.** `total` is a flat sum; each term does its own ad-hoc
  phase handling (passer blend at 6491, central buckets at 7588-7601, OvD knee at 6974,
  `PHASE_BLEND_LO` placement ramps at 7227+ — which carry their own documented incomplete-ramp bug,
  an 80%→100% step at the same isEndGame boundary). No winnability/scale-factor layer at all
  (`ENABLE_ENDGAME_SCALE` gated off, 8126).

### 1.7 SF11 → SF15.1 evolution narrative (what five years changed)

The **transform, threshold, taper and the −873 queen gate did not move** — the "where/when"
architecture was considered right in 2019 and was still right at the last classical SF. What evolved:

1. **Detector refinement, not gating:** attack weights retuned ({81,52,44,10}→{76,46,45,14} — queen
   up 40%, minors down ~7-12%); safe checks gained **multiplicity** (`SafeCheck[pt][more_than_one]`,
   e.g. rook 1071→1886 for two safe check squares — evaluate.cpp:205-207, 563-587).
2. **Latent aim entered the model:** `RookOnKingRing` / `BishopOnKingRing` — a slider whose file/
   diagonal merely points through pawns at the ring scores even with zero direct ring attacks
   (evaluate.cpp:424-428). (Our equivalent: `ENABLE_KS_AIM`, 5605-5618, default OFF.)
3. **The endgame side got smarter about winnability, not about danger:** `initiative()` →
   `winnable()`, and the generic scale factor grew special cases — notably
   **queen-vs-no-queen: `sf = 37 + 3 × minors-of-the-queenless-side`** (evaluate.cpp:937-939), i.e.
   a *material-shape-specific* endgame damping of the eg score (~58-77% of normal), plus OCB-by-
   passers, rook-endgame 36, single-flank −4 (912-947). This is where SF handles "piece-heavy queen
   endgame": the danger model runs untouched; the *eg realization* is scaled by material shape.
4. Context: in SF15.1 the classical eval is the fallback only —
   `useClassical = !useNNUE || (count > 7 && |psq| > 1760)` (evaluate.cpp:1059).

---

## 2. "Where and when" in chess terms

What actually flips king safety on/off in the mature engines is **not the game phase**:

1. **WHEN = can the attack mate?** Two conditions, both per-side and material-shape-based:
   - **The attacking side's queen.** All three references make queen-absence the dominant off-switch
     (SF −873 crossing the fire threshold; Ethereal's harder entry bar + −237/−259). Chess content:
     without a queen, sustained mating attacks against a healthy king are rare; danger should collapse
     toward (not exactly to) zero. Crucially this is *per attacker side* — a queenless middlegame
     turns KS mostly off **for the correct reason and side**, at any phase.
   - **Coordination.** SF's `count × weight` product plus the >100 threshold; Ethereal's ≥2-attackers
     entry gate. One piece near a king is not an attack.
2. **WHERE (phase) = how much does danger convert to points?** — a **continuous magnitude dial,
   never a switch.** The mg half (quadratic — attacks compound) fades with material; a **small linear
   eg half survives to bare kings** (SF `kingDanger/16`, Ethereal `−eg/20`): even in an endgame, an
   exposed king bleeds a little, and a queen endgame with a real attack bleeds meaningfully.
3. **A third, separate dial: winnability of the resulting score** (scale factors / `winnable`),
   keyed on material *shape* (OCB, pawnless, queen-vs-no-queen, single-flank) — damping the eg score
   after the danger model has spoken.

How this dodges the owner's two mislabel traps:

- **(a) Queenless middlegame** (say 2R+2B+2N each, pawns on): phase says ~2/3 midgame in every
  engine including ours. SF/Ethereal keep the KS machinery running but the queen gate cuts it to a
  remnant — which is *correct*: rook+minor attacks still exist, they're just rarer. The phase scalar
  is never asked to encode "queenless".
- **(b) Piece-heavy queen endgame** (Q+R or Q+minor each, npm high in queen terms but *count* low):
  SF's phase is material-weighted, so Q+R each = npm 7628 → gamePhase ≈ 42/128 (33% mg) — still a
  third middlegame; and regardless, safe-check/weak-square danger with the eg leak keeps firing, with
  `winnable`'s queen-endgame scale factor managing conversion. Ethereal: queen present → 1 attacker
  suffices to enter; eg half alive. Nobody switches KS off because "it's an endgame".

**The transition, stated once:** *king safety is turned off by the absence of the attacker's queen and
of coordination; it is only turned down — continuously, and never fully — by material leaving the board.*

---

## 3. Our engine mapped against them

### 3.1 The mechanism inventory (verified in-tree)

- Unit accumulator `king_safety_danger` (5323-5715): zone = ring+own-rank extension, optionally
  clamped/widened; attacker weights, zone-square count, weak squares, shield, storm, open files, safe
  checks; floor 13; quadratic-to-knee table; per-side. A large set of SF/Ethereal-shaped upgrades
  (SQC, pins, defaware, flank, check-V2, accum-mode, min-attackers, EG material gate, EG extension)
  are all **built but default-OFF** (byte-id discipline).
- `king_safety_score` (5723-5735) nets the two kings, applies `KS_DEF_MAG`, multiplies by
  `ks_phase_taper[phase_score]`.
- `evaluate_king_safety` (5746-5798): material-backing modulators (`MOD_KS_*`), then
  `KING_SAFETY_MAG(=3000) · ks / 100`.
- Called **only** inside the `!isEndGame` branch (7571-7576); the endgame twin call exists behind
  `KS_EXTEND_EG=0` (7882-7885).

### 3.2 Trap (a): the queenless middlegame

At *full* queenless material we do **not** mislabel (phase 16 → phase_score 42 → midgame, taper 100%).
The mislabel begins one trade-pair later:

| Material (each side) | phase (both) | phase_score | our label | our KS weight | SF gamePhase (mg %) | SF KS state |
|---|---|---|---|---|---|---|
| 2R+2B+2N | 16 | 42 | midgame | 100% | 86 (67%) | runs, −873 remnant + eg leak |
| 2R+B+N | 12 | 64 | midgame (boundary) | **71%** | 61 (48%) | same |
| 2R+B or R+B+N | 8-10 | 74-85 | **endgame** | **0%** | 27-38 (21-30%) | still runs: mg-tapered remnant + `kingDanger/16` |
| R+B vs R+B | 6 | 96 | endgame | 0% | 12 (9%) | eg leak persists |

So for queenless positions our error is **not** premature shutdown at full material (the owner's fear
(a) as stated doesn't occur there); it is (i) the missing queen *gate* in the band we do evaluate —
`KS_NO_QUEEN=6` under-suppresses exactly where SF would cut ~35% and Ethereal would demand a second
attacker — and (ii) the **cliff at one trade past the boundary**, where R+B+N-style attacks (which DO
mate) drop from 71% credit to zero between two evals.

### 3.3 Trap (b): the piece-heavy queen endgame — REAL and worse than feared

| Material (each side) | phase (both) | phase_score | our label | our KS | SF |
|---|---|---|---|---|---|
| Q+R | 12 | 64 | midgame boundary | 71% | 33% mg + eg leak + queen safe-checks |
| Q+minor | 10 | **74** | **endgame** | **0** | ~14% mg + eg leak, queen present, `winnable` q-vs-q sf |
| Q only | 8 | **85** | **endgame** | **0** | 10% mg + eg leak — queen attacks vs open kings are the classic danger case |

Because a queen is only 4 of 24 count-points, **two queens plus one minor each is already "endgame"
to us and king danger ceases to exist** — precisely the positions where a centralized queen against a
bare-ish king is most lethal and where SF keeps safe-check danger fully armed. This is the exact
inversion of the channel law: we keep KS strong where the giants suppress it (queenless midgame band)
and delete it where they keep it (queen endgames).

### 3.4 A tension the regret data now needs to answer

Under **defaults**, the unit-KS block *cannot execute* at npm≤12: phase ≥ 12 (the midgame condition)
requires ≥ ~27 npm-equivalent (cheapest phase point = a minor at 3 npm), so every npm≤12 position
takes the endgame branch where the term is absent. Therefore the measured "KS hurts at npm≤12" verdict
cannot indict the unit-KS term as wired today unless the measurement arms had `KS_EXTEND_EG` (or
similar) live. Either (i) the harm flows through the **other, endgame-live king-credit channels** —
the endgame `attackingLayer` proximity tables (`setAttackingLayer(10, isEndGame)` at 7643; tables at
9181-9203), the OvD accumulators, flat shelter — which matches the channel law and the de-king win, or
(ii) the arms extended KS into the endgame and measured the *extension* hurting. These imply different
fixes (subtract from proximity credit vs. shape the extension), so pin down which before acting.

### 3.5 Structural absences vs the references (summary)

1. **Hard boolean phase gate** on the entire king-model (and the mg evaluator generally) — the giants
   have zero phase gates on KS.
2. **No eg channel**: single scalar output, so there is no "small linear endgame leak"; danger is
   all-mg by construction.
3. **Queen gate under-scaled ~5-10×** relative to the accumulator it modifies, and no
   attacker-count entry gate at defaults.
4. **No winnability layer** (scale factors on the eg score; SF15.1's queen-vs-no-queen sf is the
   direct counterpart of the owner's case (b)).
5. **Incomplete tapers**: both the KS taper (71%→0) and the placement `PHASE_BLEND_LO` ramps
   (80%→100% step, 7244-7252 et al.) break at the same boundary — the boolean bites every ramp that
   crosses it.

---

## 4. Is the phase DEFINITION the root issue? Blast radius

**Verdict: the scalar definition is fine — it is Ethereal's formula verbatim. The root issues are
(i) boolean consumption of that scalar (branch-select instead of blend) and (ii) the absence of
per-side queen/coordination gating inside KS.** Redefining `phase_score` itself (e.g. SF-style
material-weighted) is neither necessary nor sufficient: it would move the (b)-trap boundary a little
(Q+minor each would read ~mid) but the cliff and the missing eg channel would remain, and it would
perturb every consumer below.

Full consumer map of `phase_score` / `isEndGame` (the blast radius of any redefinition):

**Evaluation — `cpp_bitboard.cpp`:**
- Evaluator body selection: `if (!isEndGame)` splits into two disjoint piece-eval loops (7203 / 7643
  region) — the biggest structural consumer.
- `isNearGameEnd` / `advanced_endgame_eval` gate — documented UB history, now pinned `true` (7148-7157).
- KS taper + early-out (`king_safety_score` 5723-5734; `KS_PHASE_FULL/ZERO`).
- Passer blend `evaluate_passers` (6491-6492); passer king-race `kpct` (7915).
- OvD: `realizability_factor` `REALIZ_PHASE_K` (6946); `ovd_imbalance` knee (6974).
- Central: `central_bounded` (6996-7010) and legacy buckets at 20/31/45 (7588-7601).
- Placement `PHASE_BLEND_LO/RANGE` ramps (7227, 7320, 7376, 7417, 7748).
- `boost_pieces_for_supporting_passed_pawns` rank gates on `isEndGame` (6181, 6249).
- `MOD_PIECES_*` gates (7985-8017); NPEDGE damp mg/eg split (8058-8072); endgame scale (8126);
  `PV_BOOST_PHASE_K` (8137); pawn-majority mg/eg blend (8218); space gate `SPACE_PHASE_MAX` (8319).
- `setAttackingLayer(…, isEndGame)` — two different king heat tables (9151-9227).

**Move generation/ordering — `move_gen.h`:** quiet-move *strategy* switches on `isEndGame`/
`isNearGameEnd` (655-666→727-739; 1000-1011→1045-1057); king-PST ordering gate (77-82).
⚠️ Two latent defects found while mapping: the midgame boundary here is **62**, not the eval's 64
(a 2-point band where eval and ordering disagree about the phase), and `isNearGameEnd` is
**uninitialized on the 63..96 path** (declared 648/993, assigned only in the >96 branch, read at
739/1057) — the same UB shape as the documented eval one, still live in move_gen. Flag, don't touch:
the eval precedent showed "fixing" UB is a behavior change needing games.

**Search — `search_engine.cpp`:** `lmr_phase_bucket` at 24/64/96 (675-686); decay-interval /
q-precautions at 96/117 (2402-2421); a phase-bucketed scale in `get_engine_move` at 24/64/96/117
(7203-7226). Thresholds are mutually inconsistent by design-drift (62 / 64 / 24-64-96 / 96-117).

Changing the *definition* touches all of the above at once; changing the *consumption at the KS site
only* touches nothing else. That asymmetry dictates the fix order below.

---

## 5. Recommendations (prioritized, evidence-tied)

All are KS-local (no phase redefinition), consistent with the channel law (redistribute/gate, don't
add) and the additive-hurts-critical verdict.

1. **Give the queen gate reference-scale teeth (WHEN, not how-much).** Raise the no-queen suppression
   from `KS_NO_QUEEN=6` to something that, like SF's −873-vs-threshold-100, actually zeroes typical
   queenless sums — the machinery already exists three ways: `KS_MIN_ATTACKERS=2` (Ethereal's entry
   gate, queen-relaxed, 5680-5684), `KS_NQ_SUP` under `KS_ACCUM_MODE` (5663), or simply a large
   `KS_NO_QUEEN` relative to `KS_FLOOR=13`. Evidence: SF15.1 annotates −873 as ~24 Elo, its largest
   danger term; Ethereal gates entry on it; our 6/80 is the single clearest under-scaling found.
2. **Delete the cliff: `KS_EXTEND_EG=1` + `KS_EG_MAT_GATE=1` as one unit.** The extension completes
   the designed 71%→0 fade over phase_score 65..104 (7877-7885); the material gate (npm-based,
   1246-1267) simultaneously cuts the deep-endgame band the regret maps call harmful. Ship them
   together: extension alone is additive KS in the endgame (0-for-9 territory); the gate alone can't
   fire (§3.4). Resolve §3.4's arm-configuration question first.
3. **Add the eg leak (mg/eg split of the KS output).** Return `danger` as (mg, eg_small) with
   eg ≈ mg/16-style linear remnant that bypasses `ks_phase_taper` but respects the material gate —
   the direct port of `kingDanger/16` / `−eg/20`. This is what makes queen endgames (trap b) safe to
   keep KS alive in without re-arming the deep-endgame harm: the eg channel is linear and small.
4. **Queen-endgame winnability, not queen-endgame danger** for case (b)'s conversion side: SF15.1's
   `sf = 37 + 3·minors` queen-vs-no-queen scale (evaluate.cpp:937-939) is the model — an eg-score
   damp keyed on material shape. We have no scale-factor layer; a bounded, KS-independent damp on
   `total` in queen-imbalance endgames is the smallest faithful port. Lower priority: fires rarely.
5. **Unify the phase thresholds and pin the move_gen UB** (62 vs 64; uninitialized `isNearGameEnd`) —
   correctness hygiene with measured, not assumed, behavior change (games per the eval-UB precedent).
6. **Long-term only:** if the boolean split is ever retired, do it as *taper completion* (extend the
   existing `PHASE_BLEND_*` ramps to 100% and let one evaluator body cover 0..128), not as a phase
   redefinition — §4's map is the checklist of consumers that must each be re-validated.

---

*Verified against: SF11 `evaluate.cpp` (`king()`, `initiative()`, `scale_factor()`, `value()`),
`material.cpp` (`Material::probe`), `types.h`; SF15.1 same files (`winnable()`, `SafeCheck`,
`RookOnKingRing`); Ethereal master `src/evaluate.c` (fetched 2026-08-14, safety block + constants +
tapered eval); ours `cpp_bitboard.cpp`, `search_engine.h`, `move_gen.h`, `search_engine.cpp` at the
2026-08-14 working tree. No engine code was modified.*

---

## CLEAN-DATA REVISION 2026-08-14

**What changed since the sections above were written — the measurement, not the code.** The
"KS hurts at npm≤12" verdict this doc's §5 priorities were partly built on has been proven a
**diagnostic-harness contamination artifact**: in-process position benches (run_one + `_ks_phase_split`)
carried move-ordering history (historyHeuristics/counterMoveHeuristics/moveFrequency — move-indexed,
not position-keyed) across unrelated positions, worst at low material. Fixed with a per-position
`clearSearchTables()` (search_engine.cpp; diagnostic-only, default on, `DIAG_NO_CLEAR=1` opts out);
the game engine is byte-identical. §3.4's tension flag was the correct instinct and is now resolved
as option **neither**: the harm was ~85% ghost. The unit-KS is provably inert in the endgame at
defaults — `KING_SAFETY_MAG` is read only inside `evaluate_king_safety`, called at
cpp_bitboard.cpp:7574 inside `!isEndGame` and at :7884 behind `KS_EXTEND_EG=0`.

**Clean KS on/off ruler** (`KING_SAFETY_MAG` 0 vs default, 3 corpora, per-position clear):

- Low-material "KS hurt" is a **ghost** — clean, KS is help/neutral at low material on all sets
  (low-material "changed" count collapsed 734→103).
- KS **helps critical**: enriched lichess −2.01 win% on 616 critical positions.
- KS **helps queen-present high-material** attacks: lichess 28+ Q-on −2.13 (708 pos).
- **The one confirmed defect: KS over-reads QUEENLESS positions at mid-high material.** Two
  independent regret sets agree: Qless 13-27 +0.25/+0.37, Qless 28+ +0.19/+0.26 (n≈195-360);
  Q-on helps. This is exactly §1.4's diagnosed under-scaled queen gate — SF's −873 analog.

### Re-ranked recommendations (supersedes §5's ordering)

1. **(was #1 — now the PRIMARY, confirmed lever) Reference-scale no-queen suppressor.** The only
   clean cross-validated harm signal lands precisely on the mechanism §1.4 flagged as under-scaled
   5-10×. Concrete design below. Note this is **subtractive** — consistent with the channel law
   (only subtractive KS changes have ever won in games).
2. **(was #2 — DEMOTED to parked) `KS_EXTEND_EG` + `KS_EG_MAT_GATE`.** This pair was aimed at the
   npm≤12 harm, which is now a contamination ghost; the term it would gate cannot execute in the
   endgame at defaults, and clean data says low-material KS is help/neutral. The 71%→0 taper cliff
   (§1.5) remains a real *smoothness* defect on paper, but there is no longer any measured harm for
   the extension to fix — extending KS into the endgame is additive KS (0-for-9 territory) with no
   clean-signal justification. **Drop from the active queue.**
3. **(was #3 — demoted, blocked on data) The eg leak (mg/eg split).** Its motivation was trap (b),
   the queen-heavy endgame band — which the clean corpora **cannot test** (see "unresolved" below).
   Keep as a designed hypothesis; do not build until a queen-endgame-enriched regret set exists.
4. **(was #4 — unchanged, low) Queen-endgame winnability scale factor.** Same data blockage as #3.
5. **(was #5 — unchanged, independent) Phase-threshold unification + move_gen `isNearGameEnd` UB.**
   Correctness hygiene; never depended on the contaminated measurement.

### No-queen suppressor design on OUR scale

**The pipeline the suppressor lives in** (all verified in-tree):

- Units accumulate 0..`KS_CAP=80` (search_engine.h:1379); `KS_NO_QUEEN=6` is subtracted at
  cpp_bitboard.cpp:5638 **before** the floor check at :5687 — so it directly raises the effective
  fire bar from `KS_FLOOR=13` (search_engine.h:1380) to `13 + KS_NO_QUEEN`.
- The danger map (`rebuild_ks_tables`, cpp_bitboard.cpp:412-424) is quadratic only to `KS_KNEE=12`
  — **below the floor**. So every position that fires sits on the linear segment:
  `danger = 36 + 6·(units − 12) = 6·units − 36`. Marginal effect of one suppressor unit on a
  still-firing attack = **6 danger = 180 internal cp = 0.18 pawns** (via
  `KING_SAFETY_MAG=3000 → ×30`, search_engine.h:1165, before taper/netting in
  `king_safety_score` cpp_bitboard.cpp:5723-5734).
- Typical queenless firing sums at defaults: attackers (N2/B2/R3) ~5-8 + zone-square count ~8-12 +
  weak ~4-6 + open/semi-open file 2-4 + storm 1-3 + safe checks 3-10 − shield 4-6 ≈ **20-32 units**
  → danger 84-156 → up to ~2.5-4.7 pawn-equivalents pre-taper. Current `KS_NO_QUEEN=6` raises the
  bar only to 19 and shaves ~1.1 pawn-eq off still-firing sums — a haircut, not a gate.

**Reproducing SF's effect.** SF's −873 vs threshold 100 zeroes any queenless danger sum under 973 —
in practice *most* queenless "attacks", while the biggest coordinated ones survive at a remnant.
The equivalent statement on our scale is: pick `KS_NO_QUEEN = S` so that `13 + S` sits **above the
typical queenless sum but below the extreme ones**. With typical sums 20-32:

- `S = 20` → fire bar 33: zeroes most queenless attacks, extremes (35-45+, e.g. 2R+B with storm
  and multiple safe checks) survive reduced. **Recommended center.**
- `S = 28` → bar 41: near-total zeroing; closest in spirit to SF where only overwhelming queenless
  attacks show remnant danger.
- `S = 35` → bar 48: matches the already-derived `KS_NQ_SUP=35` from the accum-mode design
  (search_engine.h:1230, "SF −873 is ~58% of its 1500 max") — the SF-faithful full-zero arm.
- Do **not** scale by SF's suppressor/threshold ratio (8.7× → S≈113): SF's threshold is 15-30×
  below its typical attack magnitude, ours only ~2×; ratio-porting the threshold relationship
  instead of the attack-fraction relationship would zero the entire queenless channel including
  the cap.

**Candidate: sweep `KS_NO_QUEEN ∈ {12, 20, 28, 35}` (single flat knob, no rebuild needed — read
live at :5638), center 20, plateau-check per the swept-knob rule.** Port the FORM, refit the
CONSTANT — the sweep is the refit.

**Flat vs material×queen conditioned?** Start **flat**. Reasons: (i) both regret sets show the
over-read at similar magnitude across both queenless bands (13-27 and 28+), so one constant likely
covers the live range; (ii) the term only executes in `!isEndGame` anyway, so the low-material end
where a bigger discount might be wanted is largely outside its domain; (iii) the suppressor is
already correctly **per-king** (keyed `!(queens & enemy)` at :5638), so mixed-queen positions get
the asymmetric treatment for free; (iv) one knob = one clean arm. Build the material-conditioned
variant (discount grows as npm falls, SF15.1-`winnable`-flavored) only if the flat winner leaves a
residual over-read in the Qless 13-27 band on re-measurement. `KS_MIN_ATTACKERS=2` (queen-relaxed
entry gate, :5680-5684) remains a complementary *coordination* gate — orthogonal signal, separate
arm, not part of this candidate.

### What the clean data does NOT resolve

- **Queen-present at mid material (13-27 Q-on):** the two regret sets disagree in sign — unresolved;
  more standard-set runs will not settle it (the ±2-3 flip on ~20 standard-set positions was shown
  to be sampling variance, not contamination).
- **Extreme-critical sign under queenless tuning:** the enriched lichess corpus is attack-puzzle /
  queen-heavy, so it can confirm "KS helps critical" for queen attacks but **cannot test whether the
  suppressor's zeroing hurts the rare critical queenless attack** — needs a queenless-enriched
  critical set before/alongside any games gate.
- **Trap (b) queen-heavy endgames** (§3.3): still untested by any clean instrument — the corpora
  under-sample Q+minor-each endgames, and the term is phase-gated off there anyway. The eg-leak and
  winnability designs stay parked on this data gap, not refuted.

---

## ATTACK-SIGNAL UTILIZATION (STRENGTH/PHASE-AWARENESS) 2026-08-14

**Reframe this section answers.** Three-way triangulation (our `ev_breakdown` KS vs SF11-static KS vs
SF18) REFUTED "our KS magnitude is inflated at the root": on queenless mid-material positions our KS
term is 0..~1 pawn — comparable to or smaller than SF11's 0.12..2.19 — with live units ~14 (danger
~48cp), nowhere near `KS_CAP=80`. The surviving hypothesis is **utilization shape**: we collect the
king-attack signals too FLATLY as a function of *available attacking strength* and *phase*, and across
a search tree that flat marginal price flips quiet moves (search-integrated over-read, not static).
Below: exactly how the references make the same signals strength/phase-aware, where we are flat, and
which flatness most plausibly produces the confirmed queenless mid-material over-read.

### A. The references' apparatus (constants + cites; sources re-read today)

**A1. Safe checks — piece-TYPED, huge, saturating, and gated on true safety.**

- **SF11** (`stockfish_11_linux/stockfish-11-linux/src/evaluate.cpp`): scalar per type —
  `QueenSafeCheck 780 · RookSafeCheck 1080 · BishopSafeCheck 635 · KnightSafeCheck 790` (:84-87).
  Each type contributes its scalar **at most once** — the test is `if (rookChecks) kingDanger +=
  RookSafeCheck` on the whole bitboard (:401-402, :414-415, :424-425, :432-433) — no per-square
  stacking at all. "Safe" = `~pos.pieces(Them)` AND (`~attackedBy[Us][ALL]` OR weak-and-double-attacked
  "overwhelmed", :392-393). Existence-gated by construction: the square must be in
  `attackedBy[Them][ROOK]` etc. — no rook, no rook check. **Priority de-duplication**: queen checks are
  counted only from squares that are NOT rook checks and not defended by our queen (:408-412); bishop
  checks only if not queen checks (:419-422). Failed-safety checks are not dropped — they become
  `unsafeChecks`, priced `148 × popcount` (:404, :427, :435, :448).
- **SF15.1** (`stockfish_15_linux/stockfish_15.1_linux_x64/src/evaluate.cpp`): the scalars became a
  **multiplicity table** — `SafeCheck[PieceType][single/multiple] = { Q {805,1292}, B {650,984},
  R {1071,1886}, N {730,1128} }` (:205-207), indexed `more_than_one(checksBB)` (:563, :572, :579, :587).
  So the shape is: **once per type, +~50-75% if that type has ≥2 safe squares, hard-saturated there**.
  Note the ordering: rook ≥ queen > knight ≈ bishop — the *check* danger is not queen-dominated;
  the queen's uniqueness is handled by the −873 gate instead.
- **Ethereal** (master `src/evaluate.c`, fetched 2026-08-14): per-square (`popcount`) but **typed**:
  `SafetySafeKnightCheck S(112,117) · SafetySafeRookCheck S(90,98) · SafetySafeQueenCheck S(93,83) ·
  SafetySafeBishopCheck S(59,59)`, each × `popcount(<type>Checks)` where
  `knightChecks = knightThreats & safe & ei->attackedBy[THEM][KNIGHT]` (existence-gated) and
  `safe = ~board->colours[THEM] & (~ei->attacked[US] | (weak & ei->attackedBy2[THEM]))` — the same
  undefended-or-overwhelmed definition as SF.

**A2. Attacker weights — count×weight product (SF) / dual-valued constants (Ethereal), and the
ordering is INVERTED vs ours.**

- **SF11** `KingAttackWeights = { N 81, B 52, R 44, Q 10 }` (:81); **SF15.1** `{ N 76, B 46, R 45,
  Q 14 }` (:201). Accumulated once per piece that bears on the ring (:280-285), seeded by pawn attacks
  on the ring (:243), then consumed as the **product** `kingAttackersCount × kingAttackersWeight`
  (SF11:446, SF15.1:600 "(~10 Elo)") — super-linear in coordination. **The queen has the SMALLEST
  presence weight (10/14 vs knight 81/76).** SF gives the queen almost no credit for *being near* the
  king; her danger is carried by what she can *do* (SafeCheck[Q], weak-square eligibility via the
  `attackedBy[Us][QUEEN]` carve-out in `weak`, :387-389) and gated by whether she *exists* (−873).
  Adjacency intensity is priced separately: `69 × kingAttacksCount` (attacks on squares adjacent to
  the king, SF11:450).
- **Ethereal**: `SafetyKnightWeight S(48,41) · SafetyBishopWeight S(24,35) · SafetyRookWeight S(36,8)
  · SafetyQueenWeight S(30,6)`, accumulated per piece into `kingAttackersWeight` (in each piece
  evaluator, masked `& kingAreas[THEM] & ~pawnAttacksBy2[THEM]`), and `safety` is **initialized from**
  that sum. Intensity separately: `SafetyAttackValue S(45,34) × scaledAttackCounts` where
  `scaledAttackCounts = 9.0 × kingAttacksCount / popcount(kingArea)` (area-normalized). **The phase
  awareness is in the constants themselves**: rook S(36,8) and queen S(30,6) collapse ~4-5× from mg
  to eg while knight S(48,41)/bishop S(24,35) persist — heavy-piece *proximity* is priced as a
  midgame-only signal at the individual-detector level, before any global taper.

**A3. Available-strength modulation beyond the no-queen gate.**

- **SF (both)**, inside the danger sum (SF11:446-457, SF15.1:600-611 with Elo annotations):
  `+ mg_value(mobility[Them] − mobility[Us])` — the attacker's *general piece activity* (a proxy for
  spare capacity to prosecute the attack) feeds danger directly; `− 100 × bool(our knight defends
  near king)` — a specific defensive-resource discount; `− 6 × mg_value(shelter)/8` — shelter quality
  feeds back; `+ 98 × blockers_for_king` (pinned defenders can't defend); `+ 148 × unsafeChecks`
  (latent strength: checks that exist but aren't safe *yet*). Entry threshold `> 100` (SF11:460)
  means small residues never convert at all.
- **Ethereal**: the **entry gate is itself strength-conditioned** —
  `if (kingAttackersCount[US] > 1 − popcount(enemyQueens))`: two coordinating attackers required when
  queenless, one when a queen exists. Inside: `SafetyNoEnemyQueens S(−237,−259)` (boolean) and
  `SafetyAdjustment S(−74,−26)` (constant offset ≈ SF's threshold in effect). Conversion clamps to
  penalty-only and splits shape by phase: `eval += MakeScore(−mg·MAX(0,mg)/720, −MAX(0,eg)/20)` —
  **quadratic mg, linear eg, each independently floored at 0**.
- **SF15.1 winnability**: `winnable()` scale factors keyed on material *shape* (queen-vs-no-queen
  `sf = 37 + 3·minors`, :937-939) damp the eg half after danger has spoken (§1.7.3).

**A4. Phase interaction.** SF: single-valued danger units → `(d²/4096, d/16)` Score → global taper
`v = (mg·phase + eg·(128−phase)·sf/64)/128`. Ethereal: **every safety constant is dual-valued**, the
accumulator is a packed (mg,eg) pair throughout, transformed asymmetrically, then globally tapered.
Both therefore have **two nested phase couplings** (per-signal constants/split + taper); neither has
any phase *gate*. The signal-level content: heavy-piece proximity ~vanishes in eg, checks persist,
and the eg lane is linear (never compounds).

### B. Ours, side-by-side (all defaults verified in-tree today)

| Signal | References | Ours (`king_safety_danger`, cpp_bitboard.cpp:5323-5715) |
|---|---|---|
| Safe check | typed 635..1886 (SF), once-per-type + multiplicity-saturated; typed per-square S(59..117) (Eth); safe = undefended OR overwhelmed; existence-gated per type | **flat `KS_SAFE_CHECK=3` per SQUARE, typeless** (:5573-5595; search_engine.h:1313) — a knight check square = a queen check square; unbounded per-square stacking within the cap; baseline safe = zero own coverage only (SF's overwhelmed variant behind `ENABLE_KS_SF_SAFECHECK`, off :5560-5565). Typed V2 built, OFF: `ENABLE_KS_CHECK_V2` with `KS_CHK_Q14/R14/B7/N9 + KS_CHK_MULTI` once-per-type + second-square bump (:5589-5593; search_engine.h:1314-1320) |
| Attacker presence | count×weight product, queen LOWEST (10-14 of 81-76 scale); Eth dual-valued, R/Q eg-collapse | **flat SUM `{N2,B2,R3,Q5}` — queen HIGHEST** (:5433-5437; search_engine.h:1190-1193); product form built OFF (`KS_COORD_GATE_MODE` :5422-5430, `KS_ATT_PRODUCT` :5642-5648); single-valued, no per-signal eg shape |
| Zone/adjacency intensity | SF `69×kingAttacksCount`; Eth area-normalized `SafetyAttackValue×9·count/area` | flat `KS_ATTACK_COUNT=1` per attacked zone square (:5410; norm built OFF `KS_ZONE_NORM` :5411-5414) |
| Weak squares | SF 185/183 per ring-weak; Eth S(42,41) | flat `KS_WEAK=2`, typeless (value-typed variant `KS_WEAK_VAL_MODE` built OFF :5391, :5441) |
| No-queen | SF −873 vs threshold 100 (~24 Elo); Eth entry-gate relaxation + S(−237,−259) | flat `KS_NO_QUEEN=6` on the 0..80 unit scale (:5638); `KS_MIN_ATTACKERS` Ethereal gate built OFF (:5680-5684) |
| Strength feedback | mobility差, knight-defender, shelter-feedback, unsafeChecks, blockers | none live (pins `KS_PIN_MODE` OFF; defaware OFF; no mobility coupling, §1.2) |
| Phase shape | dual constants / (mg,eg) split → taper | single scalar × one **global** `ks_phase_taper` applied after netting (:5734) |

**The unifying divergence:** every reference prices a unit of king-attack signal by **what piece
generates it and what phase it lands in**; we price every unit at a flat, type-blind, phase-flat
count and let one global knob (`ks_safety_table` slope 6/unit ≈ 0.18 pawn on the live linear
segment, ×taper) convert. On the linear segment the *marginal* price of +1 unit is identical whether
the unit is a queen safe-check or a third bishop-covered zone square.

**Why this specifically explains a search-integrated queenless mid-material over-read.** In a
queenless position our unit stream barely changes character: attackers R3+B2+N2 ≈ 7 vs
Q5+R3 ≈ 8 with a queen; zone-square and weak counts are type-blind; and **safe checks are where it
inverts** — a rook+bishop attack often generates *more* distinct check squares than a lone queen
(two rays each, several landing squares), so flat 3/square can charge the queenless attack MORE
check units than the queen attack, at 0.54 pawn per 3 squares (9 units × 6 × 30 /1000, pre-taper).
SF prices the same configuration: once-per-type (no square stacking), then −873 shears ~35-60% of
the whole sum. So our danger-vs-available-strength curve is nearly FLAT where theirs is steep, and
in search every quiet move that adds/removes 1-2 type-blind units repriced at ~0.18-0.36 pawn flips
move ordering — precisely the confirmed clean-regret signature (Qless 13-27/28+ over-read, Q-on
helps). The static magnitude can look SF-comparable at the root while the *derivative* along quiet
lines is wrong.

### C. Candidates — minimal, subtractive/reshaping, gated, validated on the clean regret instrument

Ranked. Each adopts a reference PRINCIPLE (shape), never their constants; each is byte-id off at
defaults; each is a single arm for `_regret_tune.py` on `game_regret_set.csv` with the Qless-band
split read out. Per the channel law, all three REDISTRIBUTE or SUBTRACT — none adds a new source.

1. **Typed, saturating safe checks — flip `ENABLE_KS_CHECK_V2=1` (already built, :5589-5593).**
   Divergence targeted: A1 vs the flat per-square typeless 3 (the single sharpest inversion — the
   only signal where queenless attacks can out-score queen attacks). V2 is *reshaping-subtractive
   in exactly the harm band*: once-per-type + optional single second-square bump deletes the
   per-square stacking that over-charges multi-check-square R/B/N attacks, while a lone queen/rook
   check is sized to clear `KS_FLOOR=13` on its own merit (the SF15 `SafeCheck[pt][more_than_one]`
   principle, our own scale). Sweep `KS_CHK_{Q,R,B,N}` around {14,14,7,9} with `KS_CHK_MULTI ∈
   {0,4,7}` — keep our knight below rook/queen initially (SF and Ethereal disagree on knight rank;
   let regret decide). Plateau-check; mirror test (ship gate).
2. **Strength-conditioned count channels — scale the TYPE-BLIND subtotal when the attacker is
   queenless.** Divergence targeted: A3/B — our zone-count + weak units are pure counts with no
   available-strength conditioning, and `KS_NO_QUEEN=6` is a flat pre-floor nudge that cannot
   reshape them. Minimal form: one new percentage knob applied to
   `(attack_count_units + KS_WEAK·weak)` when `!(queens & enemy)` (insertion point = the existing
   :5638 branch), sweep {100=byte-id, 75, 50, 35}. This is the *multiplicative* analog of SF's −873
   (shears the sum proportionally rather than shifting the bar) and Ethereal's per-constant eg
   collapse — subtractive only, queenless band only, and **orthogonal to the additive `KS_NO_QUEEN`
   sweep already queued** (run as a separate arm; do not stack winners without a joint arm).
3. **Demote queen PROXIMITY credit — sweep `KS_ATT_QUEEN` 5→{3,2}** (and optionally
   `KS_MIN_ATTACKERS=2`, the built Ethereal entry gate, as its companion coordination arm).
   Divergence targeted: A2 — the references' inverted ordering says presence-near-the-king is a
   minor-piece signal; the queen's danger belongs in her *typed checks* (candidate 1) and the
   *exists* gate, not in standing nearby. Purely subtractive, pairs naturally with candidate 1
   (which re-homes the queen's signal), and matches the standing triage that 17/23 ks_attack
   collapses are our own over-read. Read the Q-on bands as the guard: this must NOT degrade the
   confirmed Q-on help; if it does, the redistribution (1+3 jointly) is the arm, not 3 alone.

*Not proposed now:* dual-valued (mg,eg) constants per signal (Ethereal's deepest phase device) —
that is the parked eg-leak/§CLEAN-DATA #3 architecture change, still blocked on a queen-endgame
instrument; and any new detector (0-for-9 law).

*Sources re-read for this section: SF11 evaluate.cpp :81-87, :225-248, :280-285, :370-474; SF15.1
evaluate.cpp :195-207, :530-628; Ethereal master src/evaluate.c (fetched 2026-08-14, WebFetch);
ours cpp_bitboard.cpp :5390-5715, search_engine.h :1190-1326, :1387. No engine code modified.*
