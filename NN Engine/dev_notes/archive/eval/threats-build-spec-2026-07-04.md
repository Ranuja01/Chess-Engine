# Threats term — implementation spec (2026-07-04)

## ⭐ PHASE DECISION (data-driven, `diagnostics/phase_screen.py`, 2026-07-04)
Incremental validity stratified by phase (by-game Δ held-out result-loss):
| feature | MIDGAME Δ | ENDGAME Δ |
|---|---|---|
| king safety | +0.00508 | +0.00051 (collapses 10× → GATE OFF in endgame) |
| threats | +0.00179 | +0.00100 (stays substantial → ALL-PHASE) |
| mobility | +0.00329 | +0.00028 (midgame, like KS) |
| our_threats(v2) | +0.00086 | 0.00000 (midgame-only → we capture NONE of the +0.0010 endgame threat signal) |
**⇒ per-function phase treatment DIFFERS (the granularity payoff): KS + mobility = midgame-weighted, gated off/faded
in the advanced endgame (active king is GOOD — user's nuance, data-confirmed). THREATS = ALL-PHASE (endgame
immediate threats are real + outcome-predictive). Enabling our immediate-threats in the endgame is data-justified
(recovers up to +0.0010 we currently leave at 0).** The KS rebuild MUST keep midgame-only phase-gating.

## ⭐ ARCHITECTURE DECISION (user, 2026-07-04) — GRANULAR threat functions, NOT SF's monolith
- `latent_threat` = KING SAFETY (latent threats toward the king; badly named). The fast KS replaced it. Keep it
  KING-ONLY — do NOT fold general threats into it. KS is its own separate function/lever (+0.0031).
- SF crams many threat TYPES into one `threats()`. We deliberately DO NOT replicate the monolith. Split by TYPE
  into separate granular functions, each tuned separately with its OWN realizability detector (the realizability
  differs per type: immediate=SEE-like "can I win it"; pawn-push="can the pawn safely advance"; restricted="are its
  squares truly denied"). = the [[position-conditional-eval-program]] applied to threats.
- ⇒ our `get_static_threats_score` = granular function #1 = IMMEDIATE piece-on-piece threats. Its 32%-of-SF capture
  is NOT a bug — it's ONE component of SF's bundle; the other ~68% (latent/pawn-push/restricted) are SEPARATE
  functions, not missing coverage of ours. "Chase SF's 90% monolith" was the wrong target.
- **Plan:** complete immediate-threats for its type (endgame-enable — immediate threats are all-phase), bank it
  gated. Build the rest as granular functions (latent-pawn-push, restricted — each own realizability). Because each
  piece is small, GATE THE BUNDLE (efficient node_ab), not each tiny piece. Order independent functions by EV →
  **KS next (biggest lever)**; latent-threat family as follow-on. See memory [[fable-audit-2026-07-04-fixed-nodes-pivot]].



## ✅ IMMEDIATE-THREATS = COMPLETE granular fn #1 (2026-07-04)
Screen trajectory (by-game Δ held-out, sf11_threats ceiling +0.00121): v1 +0.00033 (25%, strict weak-def) → v2
+0.00043 (32%, SF-aligned weak=not-pawn-defended) → **v3 +0.00063 (52%, ALL-PHASE) = REAL LEVER, sign-stable.**
Each data-guided fix ~doubled capture. The remaining ~48% is OTHER threat TYPES (pawn-push/restricted/
knight-slider-on-queen) = SEPARATE granular functions per the architecture, NOT missing coverage of this one. So
immediate-threats is DONE for its type: validated, byte-id-safe (245/39,146,294 off), banked gated-off. Smoke:
midgame hanging-Q −2350, endgame rook-on-knight −900, startpos 0. NEXT: fit SCALE_THREATS (tune_fit --target
result) for the ship value; gate as a BUNDLE (too small solo) with the next granular threat fns OR alongside KS.
Funnel + fast-patcher fully proven (screen caught 2 impl gaps, ~8-min iterations). Corpus: threats_corpus_v3.csv.

## ENDGAME-ENABLE (2026-07-04, path A) + a bug found
- Eval is phase-branched: `phase_score<=64` midgame / `<=96` normal-endgame / `>96` advanced (cpp_bitboard.cpp
  :6073-6084). The `if(!isEndGame)` region holds the MIDGAME-only king terms (latent_threat/central); `br_pieces`
  + `capture_gains` + `passed` run in BOTH branches (two structural copies ~6327 midgame / ~6603 endgame).
- **FIX:** moved the threats block OUT of the midgame-only region (after latent_threat) INTO the both-phase region
  right after `capture_gains` in BOTH branches (relocate, not duplicate-logic — one `total += th` per eval).
  byte-id safe when off (adds 0; br bookkeeping net-neutral). Now threats fires all-phase → captures the +0.0010
  endgame signal it was leaving at 0.
- **⚠️ PRE-EXISTING BUG FOUND (not ours, flag separately):** `isNearGameEnd` (cpp_bitboard.cpp:6048) is only
  initialized in the `phase_score>96` branch (:6083) — UNINITIALIZED for midgame + normal-endgame = undefined
  behavior (likely why `advanced_endgame_fired` read True on normal-endgame test positions). Fix = init
  `isNearGameEnd=false` at declaration. Out of scope for threats; worth a separate byte-id-checked fix.

## BUILD STATUS (2026-07-04)
- **IMPLEMENTED** (all gated, default-off): knob `ENABLE_THREATS=false`/`SCALE_THREATS=100` (search_engine.h) + env
  read/echo; `EvalBreakdown.threats` (cpp_bitboard.h) + `br_threats` publish; `ChessAI.pyx` struct+dict `threats`;
  `tune_corpus.py` TERMS. Function `get_static_threats_score()` = `threats_by(false) - threats_by(true)` using
  `attackersMask` for weak-piece detection; v1 patterns = Hanging + ThreatByMinor + ThreatByRook + ThreatByKing +
  ThreatBySafePawn (pawn-attacks-piece). Integrated gated after the latent_threat block (~:6299).
- **SMOKE PASSED:** startpos threats=0; black Q hanging to a pawn = **−2350** (favours White); white Q hanging =
  **+2350** (exact mirror-symmetric → no color bias); decomposes as SafePawn 1600 + Hanging-queen 750. Signs right,
  color-symmetric, silent on quiet positions.
- **byte-id** (ENABLE_THREATS off): verifying (must = 245/39,146,294).
- **v1 SCREEN (by-game, threats_corpus): our_threats Δ +0.00033 vs sf11_threats ceiling +0.00131 = captured ~25%.**
  The Δlogloss screen CAUGHT the incomplete impl in ~35 min BEFORE any gate time (the funnel working as designed).
  Diagnosis: corr(our,SF)=0.52; **SF fires on 33,083 positions, we fired on only 7,063** (23,047 where SF flags a
  threat we read 0 = 21% coverage), and 2× hot where we did fire. ⇒ a COVERAGE gap.
- **v2 FIX:** the weak-piece test was too strict (required under-defended). SF "weak" = attacked by us AND NOT
  defended by an enemy PAWN (piece-defended pieces still count). Changed the skip to `if (defenders & enemy_pawns)
  continue;` + Hanging = undefended OR out-attacked (na>nd). Broadens coverage toward SF's 33k. Rebuild → byte-id →
  smoke → regen corpus → re-screen (target: our_threats Δ approaching the +0.00131 ceiling).
- FUNNEL LESSON: the screen is the cheap iteration loop for the impl. **Fast-iteration patcher built**
  (`diagnostics/patch_our_cols.py` — recompute only OUR columns, keep SF11 labels → ~5 min vs 30 min regen).
- **v2 SCREEN: our_threats +0.000425 (up from v1 +0.00033) vs sf11_threats ceiling +0.00131 = ~32% captured.**
  Coverage 19%→30% (SF ~90%). Improved but still a big gap. Remaining gap sources: (a) our threats is MIDGAME-only
  (SF scores all phases; ~30% of positions are endgame where we read 0); (b) missing patterns ThreatByPawnPush /
  RestrictedPiece (fires broadly) / Knight-Slider-OnQueen; (c) weak/scoring still diverges from SF.
- **STRATEGIC NOTE (2026-07-04):** threats is the SMALLEST confirmed lever (ceiling +0.00131 vs KS +0.0031 =
  2.4×). It was chosen to VALIDATE THE FUNNEL cheaply — which it DID (the screen caught the incomplete impl before
  any gate time; the patcher proved fast iteration). Closing threats to its ceiling needs several more iterations
  (endgame-enable + pattern additions). Decision point: perfect threats (small prize) vs bank the funnel-validation
  and move to KS (2.4× lever) now that funnel + tooling are proven. Even a partial threats (+0.00043) may gate
  positive; but the marginal effort/EV favors KS. USER DECISION.

---


First feature build of the SF-library funnel (chosen: threats first = cheap funnel-validator). Confirmed real
lever by the by-game incremental-validity screen (+0.00141 held-out, sign-stable). Represents SF11's **piece-on-
piece static threats**, which we genuinely LACK — our `latent_threat` (get_latent_threat_score) is KING-DIRECTED
zone pressure, a different thing (that is why our "threats" corr-with-SF is 0.23).

## What to compute (SF11 threats(), static relationships only — NO motif finders)
For each side (attacker = us, targets = them), scored White-POV, mirrored for both colours:
- **weak enemy** = an enemy NON-PAWN piece that we attack AND is (undefended OR attacked-by-more-than-defended).
- **Hanging** `~S(69,36)`: weak AND (undefended OR attacked by a pawn) → the big one.
- **ThreatByMinor[target]**: our knight/bishop attacks a weak enemy piece; value scales by TARGET type
  (pawn small … rook/queen large). `S(6,32)..S(90,119)` family.
- **ThreatByRook[target]**: our rook attacks a weak enemy piece; scales by target.
- **ThreatByKing** `~S(24,89)`: our king attacks a weak enemy piece.
- **ThreatBySafePawn** `~S(173,94)`: a SAFE pawn of ours attacks an enemy piece (nearest thing to a pawn-fork,
  scored per target, not as a motif). "safe" = the pawn's square is not attacked by the enemy, or is defended.
- **ThreatByPawnPush** `~S(48,39)`: a pawn that can safely push one square to ATTACK an enemy piece next move.
- **RestrictedPiece** `~S(7,7)`: enemy piece whose only moves are into squares we control (lower priority — skip v1).
- **KnightOnQueen/SliderOnQueen** `~S(16,12)/S(59,18)`: our knight/slider attacks the enemy queen (lower prio v1).
Use MG values first (we have a phase blend; start MG-only, add EG scaling if the screen likes it). Our engine unit
= milli-pawn (pawn 1000); SF S(x,y) are centipawn-ish → multiply by ~10 then let the fit set the scale.

## Reusable primitives (cpp_bitboard.h/.cpp — confirmed present)
- `attackersMask(colour, square, occupied, q_and_r, q_and_b, kings, knights, pawns, occupied_co)` → bitboard of
  `colour`'s pieces attacking `square`. Use it to get OUR attackers of an enemy square AND THEIR defenders (call
  with the other colour) → weak-piece test + which attacker type.
- `BB_KNIGHT_ATTACKS[sq]`, `BB_KING_ATTACKS[sq]`, `BB_PAWN_ATTACKS[colour][sq]`, `sliding_attacks(sq,occ,deltas)` /
  `BB_DIAG_ATTACKS`/`BB_RANK_ATTACKS` for slider attack sets.
- `lowest_value_attacker(attackers, colour)` for the attacked-by-lower-value logic (already used in SEE-ish checks).
- `scan_reversed`/`__builtin_ctzll` bit iteration; piece-type helpers as used across placement_and_piece_eval.

## Plumbing (the `br_<term>` → breakdown pipeline, byte-id gated)
1. **Knob** (search_engine.h, near the eval SCALE block): `inline bool ENABLE_THREATS = false;` +
   `inline int SCALE_THREATS = 100;` (or a single `THREATS_MAG=0` default). Default OFF/0 ⇒ byte-id.
2. **Env read** (search_engine.cpp initialize_engine): `Config::ENABLE_THREATS = env_flag(...)` + SCALE + echo.
3. **Function** `int get_static_threats_score()` in cpp_bitboard.cpp (near get_latent_threat_score ~:5116),
   returns Black-positive milli-pawns like the other terms (White threat = negative contribution).
4. **Integrate**: in placement_and_piece_eval, behind `if (Config::ENABLE_THREATS)`, `total += threats;` and
   `br_threats = total - br_run; br_run = total;` (mirror the br_latent pattern @ ~:6299).
5. **Publish**: add `int threats;` to `EvalBreakdown` (cpp_bitboard.h ~:342) + `g_eval_breakdown.threats = br_threats;`
   (~:6916 area) + the `threats` key in ChessAI.pyx ev_breakdown dict + add `"threats"` to tune_corpus.py TERMS.
6. **byte-id**: ENABLE_THREATS=false ⇒ the block is skipped, br_threats=0, total unchanged ⇒ WAC 245/39,146,294.

## Build → verify → screen → gate (the funnel)
1. Build → **byte-id WAC must = 245/39,146,294** (knob off).
2. Smoke: `ourmove ENABLE_THREATS=1 '<fen with a hanging piece>'` + `eval_dump` → the threats field is nonzero and
   the right sign on a known hanging-piece position; 0 in a quiet position.
3. Regenerate a small SF11 by-game corpus WITH the new `threats` column (ENABLE_THREATS=1 so our term is populated)
   → run `incremental_validity.py` adding OUR `threats` term: does our reimplementation have held-out Δ > floor
   (does it capture the SF signal we screened)? This is the "our impl, not SF's column" confirmation.
4. `node_ab` vs base at a fixed-node budget (ENABLE_THREATS=1 + a fit-set SCALE) → is it +Elo at equal nodes?
5. Lightning/blitz SPRT → ship on pass. (SCALE fit via tune_fit --target result on the threats column.)

## Discipline
- v1 = the high-signal subset (Hanging + ThreatByMinor + ThreatByRook + ThreatBySafePawn); add pawn-push / king /
  queen-attack / restricted only if the screen wants more. Simpler = fewer bugs (eval bugs are costly here).
- Gated-additive, default-off, byte-id — same safety as every prior eval term.
- SCALE is fit-set (tune_fit), never hand-cranked; node_ab + SPRT decide; the Δlogloss screen only proposes.
