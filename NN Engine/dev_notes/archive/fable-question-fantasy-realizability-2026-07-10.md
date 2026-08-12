# Fable consult — we found the fantasy-vs-real discriminator (piece-material backing); how to build the damp WITHOUT re-killing real wins? (2026-07-10)

*Self-contained; repo access + pointers in §5. Follows the 2026-07-09/10 consults; this is the RESOLUTION step.*

## 0. One-paragraph recap (venue trust unchanged)
Custom C++ HCE engine. The ~22% gauntlet "collapse" (eval peaks ≥+2p then draws/loses) = a CONDITIONAL static
over-read (+368cp; calibrated elsewhere; classical-solvable — SF11 reads them ~0). Venue: external GAUNTLET (our
engine vs throttled SF18, ≥2 seeds) = truth; lightning/compass/node_ab ANTI-PREDICT. We built a FAST offline
"class-residual screen" (recompute our eval under any knob vs cached SF, seconds) that gates before games.

## 1. What just happened — a NEGATIVE that taught us the objective
We localized the over-read to `pt_pawns` (+172, corr 0.95 with pawn_lead) and screened existing convertibility
hooks; a candidate cleared the offline screen (collapse residual −100..−170, control held). **GAUNTLET: it
HALVED the collapse rate but scored −11.6% (≈−80 Elo), seed-robust.** Lesson: **collapse-rate is the wrong
objective; SCORE is.** Our optimism is LOAD-BEARING (drives active play); broadly damping it makes us passive
and loses winnable games. Also: **61% of "collapses" are real LOSSES**, 39% draws (so it IS worth fixing — we
just fixed it wrong).

**The circling root (self-diagnosed):** we were fitting "collapse vs CONTROL", which finds features that
CORRELATE with collapse but are SHARED with real wins (material-lead) → damping them kills wins. Predictable in
hindsight.

## 2. ⭐ The decisive fit — fantasy-vs-real SEPARATES (AUC 0.85), discriminator = PIECE-material backing
We re-fit the RIGHT contrast, both "we think we're winning": FANTASY = our_static ≥ +150cp but SF18 ≤ +50cp
(over-read wins); REAL = our_static ≥ +150 and SF18 ≥ +150 (SF-confirmed wins). **HOLDOUT AUC = 0.85** (38
fantasy / 45 real) with cheap classical detectors ⇒ we are NOT missing the feature; a targeted realizability
EXISTS. The dominant discriminator:
- **`npedge` (NON-PAWN / piece material edge, mover-POV): REAL +437cp (up ~a piece) vs FANTASY +12 (no piece
  edge — pawn-only / positional).** Secondary: `counter_pressure` (opp offense − our defense; fantasy more),
  `def_us`, `off_opp`.
So: **we over-credit a PAWN/positional lead when it is NOT backed by a piece-material edge** (and opponent has
counter-pressure). Real wins are piece-backed; fantasy wins aren't. This is exactly where our failed candidate
went wrong — it damped the pawn-material lead, which is present in BOTH classes, so it hit the real wins too.

## 3. What we intend to build (want you to pressure-test)
A realizability damp on the positional-optimism cluster (`pt_pawns` placement + `capture_gains` + the material-
lead boost), keyed on `f(unbacked = low/neg npedge, counter_pressure)`, so it fires on fantasy (pawn-up-sharp,
no piece) and is ~1 (no damp) on real (piece-up) wins. Centered f=1 at detector-neutral, one-sided clamp
(damp-only). Screen offline on the fantasy-vs-real corpus: must drop the FANTASY residual while holding BOTH
CONTROL and the REAL-WIN positions within ±20cp; then gauntlet ≥2 seeds with SCORE as the primary metric.

## 4. Questions
1. **Form of the gate.** The discriminator (piece-backing) is nearly BINARY (up-a-piece vs not). Is a smooth
   `f(npedge)` (centered/clamped) right, or given the near-binary nature is a firmer gate ("damp only when
   |npedge| below ~⅓ minor AND counter_pressure present") cleaner + less collateral? How do we keep it from
   catching the real piece-up wins at the boundary?
2. **Which term(s) to multiply.** The over-read lives in `pt_pawns` (pawn placement) but `pvb`/`capg` co-inflate.
   Damp the pawn/placement credit specifically, or the aggregate optimism, given npedge already excludes the
   real (piece-up) cases? Interdependence worry: does damping placement when unbacked mis-fire in, e.g., a sound
   pawn-up endgame that DOES convert?
3. **Endgame vs midgame.** npedge separation is strong; but the endgame collapses (drawn passers) are a
   different beast (pawn-up KPK draws). Does the same npedge-gated damp handle both, or does the endgame need
   the separate draw/scale lane (which we showed is score-safe because a drawn KPK stays a draw)?
4. **Validation gate.** Given l1a (collapse-rate down but score down), we'll make SCORE the gauntlet primary and
   also verify the damp holds the REAL-WIN residual offline. Is "drop fantasy residual + hold control + hold
   real-wins offline → then gauntlet SCORE ≥2 seeds" the right acceptance test? Anything else to guard against
   (e.g., move-choice effects the static residual can't see)?
5. **The counter_pressure secondary.** Should counter_pressure/def_us enter the gate (co-occurrence with
   unbacked = "unbacked AND under pressure"), or is npedge alone enough? The AUC is driven mostly by npedge.

## 5. Repo pointers
- Fits/tools: `diagnostics/fit_fantasy.py` (this AUC-0.85 result), `fit_convert.py`, `pawn_gap.py`, `ks_gap.py`,
  `wdl_collapse.py`, `screen_knob.py` (fast offline screen), `verify_triage_static.py --dump`; corpus
  `diagnostics/corpus_ks.csv`. Handoff `dev_notes/SESSION-HANDOFF-2026-07-08.md` pt.9-11. Memory
  `[[collapse-fix-load-bearing-optimism]]`, `[[eval-accuracy-payoff-is-pruning]]`, `[[capg-tension-conditioning]]`,
  `[[external-gauntlet-calibrated]]`.
- Eval knobs (existing convertibility conditioners, all default-off): `MOD_PIECES_CONTROL/DEFEND/LEVEL`,
  `MOD_MAT_PAWNS/OPPB`, `MOD_PVBOOST_COMP`, `ENABLE_ENDGAME_SCALE` in `search_engine.h`; `mod_gain`/
  `realizability_factor` pattern in `cpp_bitboard.cpp` (:5983/:5996); `endgame_convertibility_scale` :5925.
