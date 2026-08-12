# Overnight autonomous plan — 2026-07-22 (~10h, 4 cores free after user's gaming)

**Trigger:** user will SIGNAL to start (after ~1h gaming). Until then, do NOT launch 4-core game runs.
**Mode:** ADAPTIVE — launch a run → wait for the completion notification → READ results → DECIDE the next run.
Do NOT blindly sweep. Kill/skip any arm the data disproves; don't burn cores on it.

## Budget & venue
- conc3 (~4 cores) = ONE 200g run at a time, ~76 min each → **~7-8 run-slots in 10h.**
- Venue per run: `pyrun selfplay/vs_sf.py --sf-elo 2400 --games 200 --concurrency 3 --our-config '<KNOBS>'
  --tag <tag> --adjudicate-draw --seed <s>` via the LITERAL runner wrapper.
- **Metric priority:** (1) CATEGORICAL collapse verdict — per-CLASS (ks_attack) trend from
  `collect_collapses.py`+`classify_collapses.py` is the leading indicator (STABLE across seeds where score is
  noise); (2) score as secondary (seed variance ≈13% ≫ 3.5% SE → never trust one seed's score).
- Byte-id already good (working tree 247/39,971,153; no rebuild — knobs go via --our-config). VERIFY the first
  run's stderr toggle-dump shows the knobs applied (KS_MIN_ATTACKERS=2 etc.) before trusting the batch.

## Configs
- **BASE** = current default (Kaufman ON + CAPG_PIN ON, KS floor-13): `--our-config ''`
- **GATE** (the deterministically-validated ship candidate, STS 1588):
  `--our-config 'KS_FLOOR=6 KS_SAFE_CHECK=8 KS_ATTACK_COUNT=2 KS_MIN_ATTACKERS=2'`
- **PIN_OFF** (to A/B the pending CAPG_PIN commit): `--our-config 'ENABLE_CAPG_PIN=0'`
- (reserve) **GATE_NQ** = GATE + `KS_NO_QUEEN=12` (suppressor refinement).

## ROUND 1 (4 slots, ~5h) — the primary test: does the count-gate help in GAMES?
Paired GATE vs BASE, seeds 0 and 1. Launch sequentially (one at a time), tag `base_s0/gate_s0/base_s1/gate_s1`:
1. BASE seed 0  2. GATE seed 0  3. BASE seed 1  4. GATE seed 1.
After all 4: `pyrun diagnostics/collect_collapses.py` then `classify_collapses.py` on the new dirs; read the
game CSVs + task .output with the Read tool.
**PRINCIPLE (user):** the GATE is deterministically SOUND (chess-correct coordination gate, double-count-safe,
STS 1588, closes half the KS↔SF11 gap). A game-NEUTRAL result is therefore INFORMATIVE, not a reason to
discard — the good elements stay; we DISSECT the transfer gap. Only a clear game REGRESSION with an understood
mechanism justifies dropping.
**DECISION:**
- GATE ks_attack collapse class DOWN on BOTH seeds AND score within-noise-or-up → **GATE is a keeper.**
  ROUND 2 = confirm seed 2 + probe GATE_NQ (does the suppressor add on top?).
- GATE ks_attack flat/up OR score neutral → **DO NOT DISCARD — DISSECT WHY the deterministic gain didn't
  transfer.** ROUND 2 = the dissection loop below (not a pivot away). Keep the gate as the working KS.
- Mixed (1 each) → seed noise; ROUND 2 = GATE vs BASE seed 2 tiebreak, THEN dissect if still flat.

## ROUND 2 — DISSECTION loop (when GATE is deterministically sound but game-flat)
Goal: explain the transfer gap; the gate is presumed-good until a mechanism says otherwise.
1. **Vanish attribution:** `classify_collapses.py --vanish base_s0 gate_s0 --seed 0` (deterministic games ⇒
   same (family,seed,game,color) = SAME game across runs) → exactly which collapses VANISHED vs APPEARED under
   the gate, per class. A DOWN in ks_attack masked by another class rising = the gate IS working (next class
   exposed), NOT a failure.
2. **Does the gate even FIRE on the collapse decision-FENs?** Dump KS units on the collapse `decision_fen`s
   with gate-off vs gate-on (reuse `ks_units_dump.py` / `ks_subcomponent_dump.py`). If the game collapses are
   MULTI-attacker positions the gate doesn't touch (gate only zeroes LONE-attacker cases), that explains a
   null: the STS gain was on quiet positions that don't become SF@2400 collapses. → the gate is correct but
   orthogonal to this opponent's collapse mode; keep it, and the NEXT lever targets the real collapse class.
3. **Is the collapse mode even KS?** SF18-classify the surviving collapse FENs (SF11-static is contaminated
   for attacks). If they're OvD/tactical/endgame, KS tuning can't move them — points to the OvD realizability
   over-read (c5 dossier) or elsewhere.
Outcome of the dissection decides whether GATE ships as-is (sound, neutral, next-class work), needs a tweak
(e.g. gate threshold), or its benefit is real-but-elsewhere. THEN spend remaining slots on the CAPG_PIN A/B.

## ROUND 2 (≈4 slots, ~5h) — dictated by Round 1
- **If GATE winning:** GATE vs BASE seed 2 (confirm, 2 slots) + GATE_NQ vs BASE seed 0 (2 slots).
- **If GATE not winning:** PIN_OFF vs BASE seeds 0 & 1 (4 slots) — validate the pending CAPG_PIN commit
  (pin-on = BASE; if BASE ≥ PIN_OFF on collapse/score, CAPG_PIN holds).
- **If mixed→tiebroke:** finish whichever direction the seed-2 tiebreak indicated, then spend leftover slots on
  the CAPG_PIN A/B.

## COMMIT DISCIPLINE
- Do **NOT** auto-commit anything. If CAPG_PIN A/B holds, or GATE proves out, WRITE UP the result and
  RECOMMEND the commit — leave the actual `git commit` for the user (standing rule: commit only when asked).

## Bank as you go
Update `collapse-reduction-ledger.md` (new row per fix with venue/seeds/class-signal) + this note's results
section + a handoff. Save FENs of any fresh collapses for the user to eyeball.

## Deterministic side-tasks (fill idle gaps / if a run is between launches)
- OvD c5-over-read dossier: gather more "OvD credits an unrealizable king attack" FENs (like
  `5q1k/7p/2ppRp2/p5p1/2P3P1/Q7/PP3PPK/3r4 b`); check whether a realizability-conditioned OvD (not a blanket
  IMBALANCE_SCALE cut — that was debunked) reduces the over-read without breaking controls. See
  `game-analysis-2026-07-22.md`.

## Guardrails (what NOT to do — disproven / out of scope)
- No blanket IMBALANCE_SCALE↓ (debunked = SF18-target noise).
- No KS coordination product / KS_OVERLOAD / floorless-without-strong-suppressors (all NO-GO or worse than
  the gate deterministically).
- No new C++ this block (fit-only decision). No Threats/Mobility enable (untested).
- Don't judge by raw total collapse count or a single seed's score — per-CLASS categorical verdict decides.
