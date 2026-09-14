# Eval v2 — SLICE 1 / COMPONENT 1: TEMPO

@author: Ranuja Pinnaduwage (maintained with Claude)

Date: **2026-09-13**. Status: ☠️ **BUILT, VERIFIED CORRECT, PARKED AT 0 ON MEASUREMENT** — see §6.
Register entry: `EVAL-V2-CURRENT-CONFIG.md` §2. Slice plan: same doc §5.

★ **One sentence:** a constant keyed on the side to move, phase-blended, added to v2's Black-positive
total — the only eval term we have whose correctness is checkable as an **exact identity** rather than a
statistic, because v2 is otherwise 100% side-to-move-blind.

---

## 0. RECORD-CHECK (run first, per the standing discipline)

**Verdict: NEVER BUILT, NEVER MEASURED.** No knob, no arm, no measurement has ever existed.

| record | what it actually says |
|---|---|
| `hce-eval-mechanisms-2026-07-15.md:72` | "Tempo bonus \| **SKIP** \| Flat feature-add; exactly the refuted lane." |
| ☠️ but that inherits from | `collapse-campaign.md:20` "flat SF-parity terms are neutral in games (search masks the accuracy at our depth)" |
| ☠️ and THAT thesis was reversed | MEMORY.md: "07-25's 'eval→EBF DISCONFIRMED' was a **REGIME error** that steered two months"; whole eval = +7pp |
| `archive/fable-question-counterplay-initiative-2026-07-11.md:71-74` | "The eval is deliberately COLOR-SYMMETRIC with NO tempo/initiative" — absence is **documented design**, not oversight |
| `memory/pre-nnue-strength-roadmap.md:26` (June) | proposed "symmetric tempo"; the mirror-assert half shipped, the tempo half never did |

⇒ Classification: **PROPOSED ONLY / CLOSED ON A PROXY.** The proxy (a bundle of other flat terms) has
since been invalidated, so nothing here closes tempo. It is genuinely new work.

⚠️ **Name collisions that are NOT this term.** `ENABLE_CAPG_TEMPO` (`search_engine.h:1366`, default
false) is a capture-gains evasion rule. The `tempo` comments at `cpp_bitboard.cpp:5088` and `:5286` are a
passer king-race **polarity bug fix**, not a bonus.

---

## 1. THE FIVE-ENGINE TABLE — built from source, in OUR units (pawn = 1000 mp)

| engine | tempo | form | per-mover **mg** | per-mover **eg** | mg:eg |
|---|---|---|---|---|---|
| SF 1.1 | `TempoValueMidgame=50 / Endgame=20` | **phased pair**, applied PRE-interpolation and signed by stm | **123** | **39** | 3.1 |
| SF 11 | `Tempo = Value(28)` | flat, POST-interpolation and POST-flip | **219** | **131** | 1.67 |
| SF 15.1 | ☠️ **REMOVED** | — | — | — | — |
| Ethereal | `Tempo = 20` | flat, POST-interp + POST-flip | **244** | **139** | 1.76 |
| Weiss | `Tempo = 18` | flat, POST-interp + POST-flip | **173** | **88** | 1.96 |

Source lines (all verified against source this session, not quoted from our own notes):

```
stockfish_1/stockfish-1.1_ja/src/value.h:95-96      const Value TempoValueMidgame = Value(50);  ... Endgame = Value(20);
stockfish_1/stockfish-1.1_ja/src/position.cpp:909   mgValue += (sideToMove == WHITE)? TempoValueMidgame : -TempoValueMidgame;
stockfish_1/stockfish-1.1_ja/src/position.cpp:1872  result += (side_to_move()==WHITE)? TempoValueMidgame/2 : -TempoValueMidgame/2;
stockfish_11/stockfish-11-win/src/evaluate.h:32     constexpr Value Tempo = Value(28); // Must be visible to search
stockfish_11/stockfish-11-win/src/evaluate.cpp:833  return (pos.side_to_move()==WHITE ? v : -v) + Eval::Tempo;
stockfish_15/stockfish_15.1_win_x64/src/evaluate.cpp:1040   v = (pos.side_to_move() == WHITE ? v : -v);   // nothing added
Ethereal src/evaluate.c   const int Tempo = 20;   return Tempo + (board->turn == WHITE ? eval : -eval);
Weiss    src/evaluate.c   const int Tempo = 18;   return (sideToMove == WHITE ? eval : -eval) + Tempo;
Weiss    src/types.h      P_MG = 104, P_EG = 204,
```

Unit conversion: divide by that engine's own pawn (SF11 128/213 · Ethereal 82/144 · Weiss 104/204 ·
SF1.1 `PawnValueMidgame=0xCC=204 / Endgame=0x100=256`), then multiply by our 1000. SF1's swing is
`TempoValue` and its per-mover half is `TempoValue/2`, which is what `position.cpp:1872/1893` writes
literally; SF11/Ethereal/Weiss add the whole constant to the mover, so their swing is `2 x Tempo`.

### 🐛 OUR OWN FACT SHEET WAS WRONG, AND IT WOULD HAVE FLIPPED THE VERDICT

`sf-evolution-fact-sheets-2026-07-05.md:15` records SF 1.0 as having "**NO tempo**". It has one —
`value.h:95`. The fact sheet surveyed **`evaluate.cpp`**; SF1's tempo lives in **`position.cpp`**, on the
incremental mg/eg accumulators.

☠️ Textbook instance of our own standing rule: **enumerate by DATA DEPENDENCY, not by file or by name.**
It matters because the adoption test turns on the count: "SF1 and SF15.1 both lack it" (3/5, a real split)
versus the truth (**4/5**, and the sole abstainer is SF15.1's vestigial classical eval behind NNUE).

### ★★ THE NON-OBVIOUS RESULT: DO NOT COPY THE FLAT FORM

Three engines write **one flat constant**. But their pawn is worth more in the endgame (128→213,
82→144, 104→204), so that flat constant silently becomes **~1.7-2.0x more pawns in the midgame**. They
never chose that taper; the unit handed it to them. **Our pawn is FLAT at 1000 in both phases**, so
transcribing their code would give us a flat-in-pawns tempo — which is what *none* of the four has.
And SF1, the only engine that chose the phase profile **deliberately**, went **steeper** (3.1) and
**smaller** (123 mg).

⇒ To converge in spirit we take **SF1's shape with the others' magnitudes**: a phased pair on v2's
continuous `phase256`. This is [[matching-a-reference-term-is-not-being-right]] applied exactly — shape
and channel set transfer, scale does not.

⚠️ The same trap already bit us once on this exact axis: `search_engine.h:808`'s `PASSER_V2_MG_PCT`
comment records the passer table inheriting SF's phase-dependent pawn, "which we do not have".

### Adoption verdict under [[adopt-reference-methods-only-if-universally-superior]]

4/5 have it; the abstainer removed it from **eval and search together** (SF15.1 `search.cpp:1461` is a
bare `-(ss-1)->staticEval` where SF11 had `+ 2 * Eval::Tempo`) — a coherent deliberate removal, in the
generation where NNUE, which is natively side-to-move-aware, became the real eval.

⇒ **CANDIDATE, not a mandate.** Our way (absent) is legitimate. Build it gated and measure.

---

## 2. THE FOUR MANDATORY SCANS

### (1) Consumers, by data dependency

v2's output reaches, via the **single** flip at `search_engine.cpp:9045-9046`
(`if (Config::side_to_play) total = -total;`):

| site | mechanism |
|---|---|
| `search_engine.cpp:4245 / 4649` | futility (min / max) vs `FUTILITY_MARGINS_EFF` |
| `search_engine.cpp:5294 / 5979` | **RFP** vs `RFP_MARGIN` (1500 **per remaining ply**) |
| `search_engine.cpp:5311 / 5996` | null-move eval gate |
| `search_engine.cpp:5328 / 6013` | null-move R modulation (`NULLMOVE_R_DIV`, `_CAP`) |
| `search_engine.cpp:5639 / 6281` | OTV vs `OTV_MARGIN` (1750) |
| `search_engine.cpp:7528-7547` | **qsearch stand-pat + delta** vs `DELTA_MARGIN` (1500) |
| `search_engine.cpp:5827 / 6526` | corrhist update |
| `search_engine.cpp:1203-1218` | `static_eval_for_improving` |

☠️ **`Config::side_to_play` is the ROOT colour, latched once per search (`search_engine.cpp:1524`) — it
is NOT a per-node side-relative conversion.** This is not negamax. So a tempo term must be written in
**absolute Black-positive space keyed on the node's own `turn`**, and the constant flip then carries it
correctly to every site above.

✅ Razoring and probcut compare **search** scores, not static evals (`:5066`, `:5458`, `:6143`) — unaffected.
✅ `static_eval_for_improving` compares evals **2 plies apart**, where an alternating constant **cancels**.

### (2) Knob inventory, INCLUDING hardcoded magnitudes

No existing tempo knob anywhere. The magnitudes that matter are the ones a constant shifts against:
`RFP_MARGIN=1500`/ply (`search_engine.h:2346`), `DELTA_MARGIN=1500` (`:282`), `OTV_MARGIN=1750` (`:2673`),
plus `FUTILITY_MARGINS` and the null-move R divisors.

⚠️ **A 200 mp tempo is ~13% of the smallest per-ply margin**, and these were all fitted against a
side-to-move-blind eval. See §5 — this is the term's real risk, and it is a *confound*, not a feature.

### (3) Producers / side effects

`eval_v2.cpp` is zero-global and pure. ★ **`turn` is ALREADY received and stored** — `eval_v2.cpp:1003`
takes `bool turn`, `build_context` writes `c.turn` at `:159`, and **nothing ever reads it** (dead
storage). Adding tempo introduces no new input, no new side effect, no new allocation.

✅ Eval cache is safe by construction: the key includes side-to-move via `zobristTurn`
(`cache_management.h:429-431`), so v1/v2 stm variants never collide.

⚠️ `generatePawnKey` / `corr_xor_mask` deliberately EXCLUDE stm (`cache_management.h:586-589`) — correct,
and a reason tempo must never be folded into the (planned) pawn hash.

### (4) Clamps and shared budgets

None. v2 applies **no clamp to its total** (verified), and a constant participates in no shared budget.

### ✅ HAZARD RAISED AND CLOSED

I flagged `KS-3STAGE-AUDIT-2026-08-14.md:251` — v1's KS takes `defensive` = the side-to-move's own king
and prices safe checks `KS_SAFE_CHECK_DEF=5` vs `KS_SAFE_CHECK=3` (`cpp_bitboard.cpp:5731`), so "the SAME
king's danger oscillates ±2 units/check-square (±60 mp each) with ply parity inside qsearch. Unmeasured —
flagged." If v2's KS inherited that, an explicit tempo would land on an uncontrolled implicit one.

✅ **It did not.** v2's `ks_units(c, wa, ba, true/false)` keys on the king's **colour**, not on the side to
move. **v2 has zero side-to-move conditioning of any kind.** Tempo is therefore genuinely additive here in
a way it would NOT have been in v1.

✅ Also closed: the SF11/Ethereal `+ 2 * Tempo` null-move eval-inheritance correction is **not needed** —
our only static-eval negation is `search_engine.cpp:6626` under `PRESEARCH_OFF_FILL == 2`, and the default
is **0** (`search_engine.h:1005`), so that path is dead.

---

## 3. DESIGN

```cpp
// gated on TEMPO_V2_MG | TEMPO_V2_EG, both default 0
const int t = (Config::TEMPO_V2_MG * c.phase256
             + Config::TEMPO_V2_EG * (256 - c.phase256)) >> 8;
total += c.turn ? -t : t;      // Black-positive: + when BLACK is on move
```

**Sign derivation** (the one thing that must not be got wrong, see §4):
`turn == true` means **White to move** — established independently at two v1 sites:
`cpp_bitboard.cpp:5911-5916` (`danger_white` is the defensive king when `turn`) and `:5078` (a Black
passer's catcher is White, credited `pawnDist + 1` when `turn`). v2's output is **Black-positive**
(`eval_v2.h:17-20`). Tempo rewards the mover ⇒ White to move must push the total **down**.

**Constants.** Median of the four references: **`TEMPO_V2_MG = 200`, `TEMPO_V2_EG = 110`** (ratio 1.8,
inside the 1.67-3.1 observed band; mg inside 123-244, eg inside 39-139). Defaults ship at **0**.

**Placement.** After the rung-2 block in `placement_and_piece_eval_v2`, before the breakdown publish.

**No breakdown field.** ⭐ Deliberate, and it is the proportionate-verification rule, not laziness: §4's
identity check reads the term off `total` exactly, so a dedicated field would add struct churn and a
Cython reader change to verify something already verifiable. Recorded so it is not mistaken for an omission.

---

## 4. ⭐ THE GATE — AN EXACT IDENTITY, NOT A STATISTIC

☠️ **First, the trap.** `_eval_symmetry.py`, the gate the ladder protocol runs on **every** rung, is
**BLIND to this term.** It mirrors the board *including* `turn` (`DIAGNOSTICS-TOOLKIT.md:94`), so a tempo
term with the sign **backwards** — penalising the mover — is still perfectly mirror-symmetric and passes
clean. Our highest-value early guard cannot catch the one defect this term is prone to.

✅ **The right instrument already exists**: `diagnostics/eval_symmetry.py` (note: the OTHER file), whose
TEMPO test flips `flipped.turn = not flipped.turn` (`:188`) and reports
`swing = e_white_stm - e_black_stm` in White-POV pawns, `>0 => eval favours the side to move` (`:195`,
printed `:213`). Extend the canonical tool; do not fork it.

★★ **Because v2 is otherwise 100% side-to-move-blind, the gate is an exact identity, position by
position — not a mean, not a distribution:**

| arm | required TEMPO swing | what a deviation proves |
|---|---|---|
| v2 **before** tempo | **exactly 0.000 on every position** | a non-zero swing means v2 reads `turn` somewhere it should not — pre-check that the blindness claim is true |
| v2 **with** tempo | **exactly `2t/1000` pawns**, where `t` is the phase blend at that position | wrong sign / wrong magnitude / wrong phase curve, each distinguishable |

⇒ One run validates **sign, magnitude AND the phase blend simultaneously**, deterministically. This is an
oracle-grade check on a *magnitude* term, which we have not had before.

⚠️ Do run the v1 arm as the control so a harness change cannot masquerade as a result.

### ☠️ THE GATE'S EXACTNESS IS CONDITIONAL — IT EXPIRES SILENTLY
v2 is side-to-move-blind **in the shipped configuration**, not **structurally**. `eval_v2.cpp:1062` is
today the ONLY read of `c.turn` in the whole file, and that is the sole reason the swing is an identity
rather than a distribution. **The moment any later component reads `turn`** — a draw/convertibility rule
keyed on who moves, a threat term, an SF-style capgains successor — **this gate silently degrades from an
exact identity to a statistic**, and a sign error in tempo stops being detectable by it.
⇒ Any future v2 component that reads `turn` MUST re-establish the baseline: re-run the tempo-off arm and
confirm what the new non-zero swing is, before trusting this gate again. Treat a non-zero tempo-off swing
as a **finding to explain**, never as noise.

Plus the standard rung gates: `wac arm0` byte-identity at default (250 / 35,310,778 / EBF 3.784), the
knob echoed in `[toggles]`, and a non-zero changed-move rate proving the knob is live.

---

## 5. ⚠️ REGISTERED PREDICTION (before any build — the standing discipline)

**A constant keyed on side to move is identical across every sibling at a node**, because all siblings
lead to the same side to move. ⇒ **It cannot reorder moves at fixed depth.** Its only channels are:

1. comparisons between plies of different parity (extensions, reductions, qsearch boundaries);
2. comparisons against the **absolute** margins listed in scan (1).

Therefore I predict:

- **§I eval accuracy: NULL.** A constant offset per stm class has zero positional variance — it is a pure
  mean correction, and [[every-eval-term-error-is-bidirectional]] says *add signal, don't correct the mean*.
- **STS: inside its ±150 floor.** Fixed-depth, and parity effects largely cancel.
- **Games: |Elo| < 10**, i.e. **below single-slice resolution** (one night resolves ~+20). That is an
  argument for tempo riding **inside** a bundle, not for it leading one.

☠️ **And the honest caveat on my own slice-plan wording.** `EVAL-V2-CURRENT-CONFIG.md:115` bills slice 1
as "order-INVARIANT terms that cannot cancel". Tempo is order-invariant **within a node** but it is **not
margin-invariant**: at ~200 mp against a 1500 mp/ply RFP margin it is a ~13% perturbation of a system
fitted to a stm-blind eval. Whatever games measure will be **partly a margin-fit artifact, not evaluative
content** — precisely the spread-coupling confound the rebuild plan warns about.

★ ⇒ **The instrument matched to the actual mechanism is NODE COUNT, not §I.** `wac_speed` node count and
depth@1s directly measure the margin interaction. If nodes barely move, the term is near-inert and shipping
it on 4/5 reference convergence is low-risk. If nodes move a lot, the margin coupling is the story and
tempo must not be judged until margins are re-swept at a checkpoint. **Run that screen before anything else.**

---

## 6. ⭐ RESULT (2026-09-13) — CORRECT, AND PARKED

### Controls, both clean
| control | required | got |
|---|---|---|
| arm 0 byte-identity | 250 / 35,310,778 / EBF 3.784 | ✅ exact |
| arm 1 tempo-off STS | the recorded rung-2 value, 1698 | ✅ **1698** exact |

⇒ Adding the translation-unit content and the branch perturbed neither arm.

### The identity gate — PASSED AT ZERO TOLERANCE
600 positions from `game_regret_set.csv`, sampled for a full phase spread (0-14 non-pawn pieces).
`diagnostics/eval_symmetry.py` TEMPO swing, White-POV pawns:

| config | swing (mean / median / max) | required | |
|---|---|---|---|
| v1 arm 0 (harness proof) | +1.339 / +0.053 / 21.213 | large and ragged — v1's emergent stm effect | ✅ harness lives |
| **v2, tempo OFF** | **+0.000 / +0.000 / 0.000** | **exactly 0** | ✅ **blindness PROVED, not assumed** |
| v2, 200/200 (constant) | +0.400 / +0.400 / 0.400 — **zero spread** | exactly 0.400 | ✅ |
| v2, 110/110 (constant) | +0.220 / +0.220 / 0.220 — **zero spread** | exactly 0.220 | ✅ |
| v2, 200/0 (mg leg) | +0.206 / +0.210 / **0.400** | ceiling exactly 0.400 | ✅ |
| v2, 0/110 (eg leg) | +0.106 / +0.102 / **0.220** | ceiling exactly 0.220 | ✅ |
| v2, 200/110 (ref) | +0.312 / +0.314 / 0.400 | — | ✅ |

★ The legs **add exactly**: `0.206 + 0.106 = 0.312`, the combined run's mean to three decimals. Sign,
magnitude and the phase blend are each verified as identities rather than statistics.

### The verdict — two instruments agreeing
| arm | STS | nodes (WAC, fixed depth) |
|---|---|---|
| off | **1698** | 63,216,318 |
| ref 200/110 | 1631 (**−67**, ☠️ inside the ±150 floor — UNRESOLVED) | 70,058,283 (**+10.82%**) |
| crank 4x 800/440 | **1312 (−386**, far outside the floor — clearly harmful) | 64,992,326 (+2.81%) |

☠️ **Do not read the WAC solve counts** (246 / 252 / 240): the floor is ±5-6 and the metric does not
discriminate strength at all. "+6 solves at ref" is exactly the reading this project has twice got wrong.

1. **STS is monotone downward with magnitude, with no local maximum** ⇒ the optimum is at or below 0.
   Only the crank point is resolvable, and it is harmful. Same shape as connected pawns at rung 2.
2. **The node cost is 100% confound by construction.** Tempo has **zero positional variance**, so unlike a
   real term none of its node movement can be "a better eval prunes better" — all of it is margin/parity
   interaction. And it is **NON-monotonic** (+10.8% at 1x, +2.8% at 4x): the step-shaped signature of a
   constant crossing fitted absolute thresholds, not a smooth evaluative effect.

⇒ Corroboration rule fires: **two instruments agree that there is no evaluative content here. Spend no
games.** PARKED at 0; re-test trigger is the checkpoint margin re-sweep, because margin coupling is the
only channel it ever demonstrated.

### ⭐ THE RUNG-GRADIENT TEST (owner's hypothesis, run 2026-09-13)

**Owner's question:** *"every engine has tempo, so I wonder why it doesn't work for us. Perhaps it needs the
other things we've yet to add?"* — i.e. tempo corrects the **odd-even effect**, whose size is proportional to
how much a single quiet move can change the eval; on a thin eval that distortion is small, so a
reference-scaled constant would be an over-correction that shrinks as the eval thickens.

**Test:** the same tempo at three rungs. Each "off" arm doubles as a control.

| rung | tempo off | tempo 200/110 | Δ |
|---|---|---|---|
| 0 — material + PST | **1364** ✅ | 1336 | **−28** |
| 1 — + king safety | **1480** ✅ | 1378 | **−102** |
| 2 — + pawns (shipped) | **1698** ✅ | 1631 | **−67** |

✅ All three controls reproduce their recorded values exactly, and rung 2 reproduces **1631** from an
earlier, independent run — the harness is stable and the table is reproducible.

☠️ **The hypothesis is NOT supported.** The gradient is **non-monotonic** (−28, −102, −67): there is no
trend of tempo becoming less harmful as the eval thickens. And **all three deltas sit inside the ±150
floor**, so the rungs cannot be distinguished from one another at all — only their common sign is readable.

★ By our own rule — *three unresolvable signals the same way are a result* — what IS supported is
**"tempo is harmful-or-neutral at every rung."** What is NOT supported is any dependence on eval richness.
⇒ Hypothesis 1 (**our margins were fitted without tempo, SF's were fitted with it**) carries the
explanation alone. That is [[a-correctness-fix-into-absorbed-tuning-is-not-free]] exactly.

⚠️ **Pre-registered caveat, which stands:** rung0→rung2 is a weak richness axis. PSTs already respond to
quiet moves, and rung 2 adds **pawn structure — the LEAST quiet-move-responsive thing available**. The
honest test is re-running this after slice 2 lands **mobility**, the biggest missing responder.
★ And the null is the informative direction here: the eval-spread confound biases *toward* the hypothesis
(a fixed 200 mp is relatively larger on a thinner eval), so a null despite that bias is real evidence.

### ☠️☠️ THE MAGNITUDE WAS CONVERTED WRONG — 200 mp IS 8× v2's WHOLE POSITIONAL SPREAD

Measured 2026-09-13 by probing `ChessAI.ev` directly on hand-built positions (v2 rung 2):

| position | v2 eval (mp) |
|---|---|
| knight on the rim (a3) | **−5** |
| knight central (e5) | **−35** |
| Italian, Bc4 → Bf1 (passive) | **25** |
| sharp Sicilian middlegame | **−5** |
| up a clean rook | −5000 |

⇒ **v2's entire midgame positional signal lives in 5-35 mp.** Rim-vs-centre for a knight — about the
largest single placement decision a PST expresses — is **30 mp**.

★★ **So `TEMPO_V2_MG = 200` is roughly 8× the biggest positional differential in the whole eval.** §1's
conversion (SF11 `Tempo=28` ÷ `PawnValueMg=128` × 1000) is arithmetically impeccable and **positionally
meaningless**: a reference engine's positional constants are a large fraction of *its* pawn, while ours are
a few percent of ours. The "small nudge" SF intends became the largest positional term we own.

⚠️ **Third instance of one pattern.** `PASSER_V2_MG_PCT` (`search_engine.h:808`) records the passer table
inheriting SF's phase-dependent pawn we do not have; §1 above records the flat-tempo taper the same way.
★ **A faithful unit conversion can still carry a relationship we do not share.** Convert a POSITIONAL term
against the POSITIONAL spread, not against the pawn — and **measure our spread first**, which is a
60-second probe.

⚠️ **This does NOT resurrect the term, and I am not rationalising the null back to life.** It corrects the
*explanation*, and it makes a testable prediction registered before the run: at ~25 mp tempo should become
**inert** (inside the ±150 floor) rather than −67/−102. But note what that means — a tempo correctly sized
to v2's positional scale is **below every instrument we own** (STS ±150, §I 0.05%, games ±25 Elo).
★ **"Correctly sized" and "demonstrable" are different claims.** A term can be both right and unshippable
on evidence, and that is the honest end state here.

#### The magnitude sweep, run to settle it

| tempo mg/eg | STS | Δ | |
|---|---|---|---|
| 0 / 0 | **1698** | — | |
| **25 / 14** (matched to the 5-35 mp positional spread) | 1656 | **−42** | **inert — inside the floor, AS PREDICTED** ✅ |
| 50 / 28 | 1511 | **−187** | outside the floor |
| 100 / 55 | 1504 | **−194** | outside the floor |
| 200 / 110 (reference-converted) | 1631 | −67 | inside the floor |

☠️☠️ **The response is NON-MONOTONIC and erratic** — −42, −187, −194, −67. There is no optimum; the *worst*
points sit in the middle and the largest magnitude reads better than half of it. **A term with genuine
evaluative content produces a smooth curve with a single peak.** This is threshold-crossing behaviour, and
it is the **second independent line of evidence for margin coupling** after the non-monotonic node screen
(+10.8% at 1×, +2.8% at 4×).

⇒ The positional-scale correction fixes my REASONING and changes NOTHING about the verdict. At the
correctly-sized 25 mp the term is merely **inert** — the best case available, and not a reason to ship.
**PARKED at 0, final.**

⚠️ A transferable caution: five magnitudes of one term produced a curve that is not even ordered. **Do not
rank on a single STS arm-vs-arm delta**, and prefer a magnitude LADDER over a point comparison whenever the
question is "how much" — a point comparison here would have supported almost any story you wanted.

### ✅ Registered prediction, graded
| predicted | outcome |
|---|---|
| STS inside its ±150 floor at ref | ✅ **correct** (−67) |
| cannot reorder at fixed depth; only channels are parity + absolute margins | ✅ **correct** — the node result is unexplainable any other way |
| §I null | ⚪ not run; the STS + node corroboration settled it more cheaply |
| games \|Elo\| < 10 | ⚪ not spent, and now should not be |

⚠️ What I got WRONG in the plan that sent tempo here: `EVAL-V2-CURRENT-CONFIG.md` §5 billed slice 1 as
"order-INVARIANT terms that cannot cancel". Tempo is order-invariant **within a node** and genuinely
cannot cancel with the other members — **and that was never the relevant risk.**
★ **Generalise: "cannot cancel with its slice-mates" is NOT "has no confound of its own."** Every
remaining slice-1 member must also be checked against the absolute margins, not only against each other.

---

## 7. APPENDIX — the pre-registered park criteria, and how each graded

Written before the build, graded after:

| criterion | outcome |
|---|---|
| TEMPO swing is not exactly `2t/1000` ⇒ stop, the implementation or the blindness claim is wrong | ✅ **did not fire** — exact at zero tolerance on all six configs |
| Node count moves >10% at the reference magnitude ⇒ a margin change wearing an eval costume; park | ☠️ **FIRED** — +10.82% |
| Null at cranked magnitude (4x) on every instrument ⇒ refuted for this core | ⚠️ **worse than null** — STS −386 at crank, far outside the ±150 floor |

⚠️ **Honest note on the 10% number.** I set it without calibrating against our own recorded transfer
function ([[node-savings-below-35-percent-are-elo-neutral]]), so the threshold itself was arbitrary and
probably too tight. Its *purpose* — detect a margin change wearing an eval costume — was nonetheless met,
and independently of where the line sat, by two things the threshold did not depend on: the
**non-monotonicity** in magnitude, and the fact that a zero-variance term can have **no other channel**.
★ The lesson is to pre-register the *mechanism* the criterion is testing for, not only a number.
