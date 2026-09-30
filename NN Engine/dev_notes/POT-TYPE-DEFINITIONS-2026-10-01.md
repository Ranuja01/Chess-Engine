# POT — transformation TYPE definitions, for owner review (2026-10-01)

**POT (Potential) — "OvD reworked"**: the owner's v1 long-term-pressure invention, redefined 10-01 (C3 doc §17) as the
potential of either side to STRUCTURALLY transform the position so that another subsystem later "shines through".
Sources: `POT-TRANSFORMATION-KNOWLEDGE-2026-10-01.md` (taxonomy), C3 doc §17-18a (definition, T1 study).
Status: DESIGN FOR REVIEW — nothing built. Every type below is measured before any C++ exists.

## 0. Rules every type obeys
1. **Gate = PRECURSORS present ∧ RESULT absent**, both judged from STRUCTURE. Never from any subsystem's score (owner,
   10-01: a quiet subsystem may be kinetic with nothing to say).
2. **One owner per result.** POT carries only the LATENT part of the owner's future score and falls silent when the
   result appears (the owner takes over) or when the precursors die (break permanently blocked, king safe in an intact
   shelter, majority crippled).
3. **Both sides, tempo to the mover** (owner, 09-27 / 10-01).
4. **A state of the king is KS's, not POT's.** Anything that describes how safe a king IS (castling delay, shelter)
   goes to KS as a FEEDER, never as a POT score (owner, 10-01). POT may only say how the structure can CHANGE.
5. **Measure first:** coverage (gate on-rate) → event rate and precursors → signal on the DEPTH residual (SF18 −
   our d10-12 SEARCH: what our search still misses) → BEYOND-controls (owner term + KS channels + C1/C3 + stm). A type
   that fails beyond-controls is dropped, whatever its raw signal.
6. **Bundle for the fit, verify per type:** nested held-out value per type, a fire-gated game split, and a per-type
   ablation at 250k vs SF18 (parts can cancel — Fit K).

Shipped owners referenced below: KS (+ KS-B shelter/storm, built at 0) · pawn structure `PS_V2` (doubled / isolated /
backward / weak-unopposed) · passers + candidates `PASSER_V2` (**endgame leg only**) · mobility · placement (outposts,
bad bishop, trapped rook) · tapered PST · winnability scale factor `WSF_V2` (in gate).

## T1. Central opening vs an uncastled king
- **Result:** a file next to a still-central king (d/e, ±1 within c-f) loses all the defender's pawns.
- **Owner:** KS (the open lines then feed its attack/zone inputs).
- **Gate:** defender's king on d/e ∧ every adjacent c-f file still holds a defender pawn ∧ a central lever exists or is
  one supported push away (for the attacker).
- **Precursors (for the fit):** central levers · supported lever-pushes · central rams (−) · heavy pieces behind the
  break · the king's distance to safety as a GATE modifier only (rule 4).
- **POT scores:** the latent KS gain of the side that can open it.
- **Status (§18/18a):** levers predict that lines open (29→48%), but carry NO signal vs the STATIC residual, raw or
  beyond controls. **Re-test on the depth residual before judging.** Its castling-delay signal went to KS (feeder lead).
- **Overlap risk:** medium (KS). Controls: all KS channels + KS-B cells.

## T2. Flank storm vs a castled king — only with a closed centre
- **Result:** a storm pawn makes contact with the shelter (a lever on the shelter files), then opens a file.
- **Owner:** KS-B storm (built at 0) scores storm pawns by rank and blocking — **that is already a latent-looking term.**
- **Gate:** kings castled ∧ centre CLOSED (central rams, no central lever — Nimzowitsch's condition) ∧ storm pawns not yet
  in contact ∧ unblocked (SF `BlockedStorm`: a rammed storm pawn is no threat).
- **POT could add only:** the closed-centre condition (a storm is real only if the centre cannot counter-strike).
- ⚠️ **Overlap risk: HIGH.** If KS-B's storm, once priced, already carries this, T2 is dropped. **My recommendation:
  measure it after KS-B is re-priced; do not build a POT T2 unless it shows beyond-KS-B signal.**

## T3. Chain-base attack
- **Result:** the base of a pawn chain becomes backward/isolated, or the chain dissolves with a half-open file.
- **Owner:** pawn structure (backward/isolated/weak-unopposed).
- **Gate:** a ram on c-f with a chain behind it (the defender's pawn diagonally supporting the rammed pawn) ∧ an attacker
  lever against the BASE reachable (existing or one supported push) ∧ the base not yet weak.
- **Precursors:** base lever reachable · pieces aimed at the base (rook/queen on its file, minor attacking it) ·
  chain direction (Nimzowitsch: play on the side the chain points to) · the defender's ability to support the base.
- **POT scores:** the latent structure penalty the base would take, to the side holding the lever.
- **Overlap risk:** low-medium (pawn structure scores the base only once it is weak; §14's lever features tested the
  ungated version).

## T4. Majority → passer
- **Result:** a passed pawn is created from a flank majority.
- **Owner:** passers + CANDIDATES — v2 already scores candidates (`passer_value_mp` reads `passed | candidate`), but
  **endgame leg only**.
- **Gate:** a healthy flank majority (not doubled, not fixed by a ram) ∧ no passer on that flank yet.
- **POT could add only:** the majority BEFORE any single pawn qualifies as a candidate, and its middlegame value
  (the owner term has no mg leg).
- ⚠️ **Overlap risk: HIGH** with candidates; §14's `mobile_majority` was null and owned. And the §14 lead ("we OVER-rate
  the leader's mg edge when passers are on the board") points the other way. **My recommendation: fold this into the
  passer re-tune (next step 3 in the handoff order) rather than a POT type**, unless you see it differently.

## T5. Minority attack / fixing a weakness
- **Result:** a backward or isolated pawn is fixed on a half-open file (Carlsbad c6).
- **Owner:** pawn structure (backward/weak-unopposed); the half-open file (rook files were closed null 09-20).
- **Gate:** on one flank the attacker has fewer pawns ∧ a lever against the majority is reachable ∧ a half-open file
  toward it ∧ the target not yet weak.
- **POT scores:** the latent weakness penalty, to the side with the minority attack.
- **Overlap risk:** low (the weakness is scored only after it exists).

## T7. Trade-down conversion → WINNABILITY (endgame hand-over)
- Already built as the reference-form scale factor `WSF_V2` (C3 §16), in the gate now. POT's mg types fade with phase,
  and winnability takes the endgame (owner, 10-01).

## Deferred (reasons)
- **T6 freeing break:** its result is mostly "equality" (mobility/space) — low value, high overlap with mobility.
- **T8 bad bishop / outpost:** owners shipped (outpost, bad bishop); a latent form ("rams forming on the bishop's
  colour") is possible later.
- **T9 second weakness / T10 dynamic→static / T11 exchange sac / T12 blockade / T13 tension choice:** kinetic, too rare,
  or needs search.

## Proposed first bundle
**T1 + T3 + T5**: the three with a clear latent/kinetic line and owners that only score the result.
T2 and T4 are measured but defaulted OUT (high overlap) unless they show beyond-owner signal.
**Measurement order:** (a) coverage and event rates on our games (pure Python, now); (b) after the gauntlet: our d10-12
search pass on the labelled mg rows (the depth residual) + more SF18 labels on gated rows + SF18 self-play games for
precursors; (c) the bundled fit with per-type verification.

## Open questions for the owner
1. Is T2 (storm) KS-B's alone, i.e. POT out unless measurement says otherwise?
2. T4: fold it into the passer re-tune, or keep it as a POT type?
3. Any transformation you've seen matter that is not on this list (your v1 games — e.g. the Bg7/Ba3 diagonal)?

## Owner review, round 1 (2026-10-01)
1. **T2 storm:** KS-B owns king-directed storms. Owner asks whether POT's storm should be the NON-king kind — pawns
   advancing toward the centre or the enemy structure to create a future break. Proposal: yes, as the general
   "lever REACH" precursor (a pawn that can march to make contact in 2-3 moves), feeding T1/T3/T4/T5 rather than a type
   of its own. King-directed storm stays KS-B (with the closed-centre condition as a KS-B feeder candidate).
2. **T4 majority → passer IS POT** (owner): the POTENTIAL to break out into a passer, before any pawn is a candidate;
   it turns down as it becomes kinetic (candidate/passer ⇒ the passer terms own it). ⇒ T4 back IN, gated on "no
   candidate/passer on that flank yet". (§14's `mobile_majority` null was ungated and on the static residual — re-test.)
3. **The scoring is REALIZABILITY OF POTENTIAL** (owner): POT = Σ_k P(reaching state k, alone or combined) × the
   advantage the side would have AT that state. The type descriptions state end results only to define k; the score is
   the chance of getting there × what it is worth there, discounted by how unresolved it still is.
**New types from the owner:**
- **T8' Centre liquidation → open board (piece activity):** potential = heavy pieces aligned behind own pawns on files
  that would open; owner of the result = mobility / placement. ★ Directly relevant to §18a's finding that we already
  OVER-credit heavy pieces on closed king files (−5.0σ beyond controls) — an existing term scores that pressure as
  kinetic. ⇒ T8' cannot be ADDED on top until that term is found; it may be the correct SHAPE (credit ∝ P(open)) for
  pressure some term now scores in full.
- **T9' Fortress / blockade creation:** the defender's potential to build an unbreakable wedge. Proposal: model it as
  the PRECURSOR-KILL side of the attacker's types (it lowers P_k), plus an endgame winnability input; static fortress
  detection is notoriously hard ⇒ later.
- **T6' Exchange sac / piece transformation:** the move is search's; the static part is the value of minor-piece
  anchors (holes, outposts, colour complexes) when lines stay closed = a closedness-conditioned minor-vs-rook imbalance
  (SF weighs space by `blockedCount`; Kaufman imbalance parked). ⇒ a separate imbalance lead, probably not POT.
**Potential → kinetic per subsystem (owner question):** not all-or-nothing and not global. Per type k and per REGION
(king file group, flank, centre), U_k ∈ [0,1] is CONTINUOUS (the winnability lesson: no cliffs) and structural:
the expected remaining change of owner_k's INPUTS over the next N plies, regressed on structure (rams, levers, lever
reach, open/half-open files, candidate status …) from game sequences. Examples of the natural limits: T1 U falls as
king-adjacent files open (each open file moves it toward KS); T4 U falls as a pawn becomes a candidate, then 0 at a
passer; T3/T5 U falls when the lever is played (tension) and 0 once the target is weak; T8' U falls per central file
opened. P_k (reach) and V_k (value at the state) are fitted from data too; V_k is measured as the eval/outcome gain
after reaching k, not hand-set.

## Coverage study (2026-10-01, `diagnostics/_pot_coverage.py GAMES=1500 N=20`, 43,539 middlegame positions)
Predictions: any gate > 50% ✓ · T4/T5 fire least ✗ (they fire MOST) · T3 lowest event rate ✗ (T4 lowest).
| type | gate on (share of positions) | result within 20 plies (of gated side-rows) |
|---|---|---|
| T1 central opening | 26.0% | 31.6% (13,665) |
| T3 chain base | 18.4% | 35.4% (8,954) |
| T4 majority → passer | 31.6% | 16.4% (13,002) |
| T5 minority attack | 38.5% | 21.7% (16,110) |
**ANY gate: 70.2%** of middlegame positions. Top combinations: none 29.8 · T4+T5 13.7 · T5 12.0 · T1 10.5 · T4 7.2 · T3 5.5.
Reading: the concept is NOT narrow in coverage — unresolved structure is the normal middlegame state. T4+T5 co-fire
because they are the two sides of one flank (A's majority = D's minority) — POT nets both sides by design. These gates
are LOOSE (definitions, not detectors): next is PRECISION — the event rate with the gate ON vs OFF (lift), per type,
then the depth residual. Value per position is expected small; the fit shrinks it (owner, 10-01).

## Gate precision (2026-10-01, `_pot_coverage.py MODE=lift GAMES=1500 N=20`)
Event rate WITH the key precursor vs a near-miss population WITHOUT it (same gate minus that precursor).
Predictions: all lifts > 1.2× ✗ · T1/T3 highest ✗ (they are the lowest).
| type | key precursor | with | without | lift |
|---|---|---|---|---|
| T1 | central lever reach ≤ 2 | 31.6% (13,665) | 32.7% (1,210) | 0.97× (−0.8σ) |
| T3 | lever reach vs the chain base | 35.4% (8,954) | 37.6% (18,076) | 0.94× (−3.5σ) |
| T4 | flank majority | 16.4% (13,002) | 9.9% (77,409) | **1.65× (+18.8σ)** |
| T5 | lever reach on the minority flank | 21.7% (16,110) | 19.3% (9,253) | 1.13× (+4.7σ) |
Reading: the MAJORITY precursor is real (Kmoch/Shereshevsky confirmed on our games); lever REACH as defined does not
raise T1/T3 events. Two explanations, not yet separated: (a) my EVENTS are too generic ("a file lost D's pawns", "the
base square changed" happen for many reasons: D's own captures, advances, piece trades) — an event should require the
transformation to be caused by the lever (the lever pawn captures / is captured), and (b) d6 games — levers get played
or ignored at random. Next: causal event definitions for T1/T3 (the lever exchange happened), re-run; strong-game
(SF18 self-play) sequences when the engine is free. T4 is the first POT type with a demonstrated precursor.
