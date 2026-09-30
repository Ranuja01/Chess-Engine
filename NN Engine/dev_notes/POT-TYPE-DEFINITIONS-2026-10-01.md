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
