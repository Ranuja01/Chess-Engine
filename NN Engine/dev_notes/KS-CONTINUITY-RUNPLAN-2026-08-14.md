# KS Continuity / Shape — tonight's run-plan (2026-08-14)

**Purpose:** turn the 3-stage KS audit into an executable, sequenced queue for the cores-free night
block. Everything here validates on the CLEAN regret instrument + ours/SF11/SF18 triangulation, passes
the colour-symmetry mirror gate, and is decided in GAMES. One process per knob setting (knobs latch at
init). Games run ALONE. JOBS ≤ 4, one engine-loading job at a time.

## GROUND TRUTH (read the code, not the stale comments) — corrects the handoff

The attack-unit KS term is **LIVE at the shipped default**, not off:
- `KING_SAFETY_MAG = 3000` (search_engine.h:1165), `ENABLE_KS_REPLACE_LT = true` (search_engine.h:1170)
  ⇒ evaluate_king_safety REPLACES the dead latent_threat and is the sole king-danger term.
- The cpp comment at cpp_bitboard.cpp:7568 ("Default-off … byte-identical") is STALE — do not trust it.

⇒ **Continuity/shape changes to this term are NOT byte-id-free.** Every arm below that touches the live
base term changes shipped play → it MUST pass `_eval_symmetry.py` (mirror gate) and be decided in games,
not just on the regret ruler. Only the additive/gated knobs (KS_ACCUM_MODE=0, KS_MIN_ATTACKERS=0,
KS_INTERACT=0, KS_BATTERY unwired, ENABLE_KS_CHECK_V2 off) are byte-id at default.

Two LIVE discontinuities (confirmed at default):
- **Floor step** — KS_FLOOR=13 (search_engine.h:1380) is a hard `if (units < KS_FLOOR) return 0`
  deadzone (cpp_bitboard.cpp:5687). A quiet move tipping across it jumps the static eval by
  `ks_safety_table[13] * KING_SAFETY_MAG/100`. (Read the exact step off ks_safety_table[13] before
  quoting a number — do NOT reuse the ~1260 mp figure without confirming KS_KNEE/KS_DIVISOR.)
- **Phase cliff** — KS is called only under `!isEndGame` (phase_score ≤ 64; call at cpp_bitboard.cpp:7574,
  gate at :7203), but its taper is built to fade KS_PHASE_FULL=48 → KS_PHASE_ZERO=104. So at phase 64→65
  KS drops ~71% → 0 in one phase point, at every depth. KS_EXTEND_EG=0 leaves the designed smooth endgame
  fade off (the endgame re-call is coded at cpp_bitboard.cpp:7882).

The continuity toolkit already exists as gated knobs — tonight is mostly a SWEEP, not new C++:
- `KS_EXTEND_EG=1` — run the SAME KS term in the endgame branch so the taper fades it 64→104 smoothly
  (kills the cliff). Coded, off.
- `KS_ACCUM_MODE=1` (+ KS_ACCUM_THRESH, KS_ACCUM_LIN, KS_ACCUM_DIV) — signed-accumulator object:
  net = positives − suppressors, per-position threshold, then a linear (or squared) map. This REPLACES
  the hard KS_FLOOR with a threshold+ramp = continuous gate. Coded, off (cpp_bitboard.cpp:5655).
- `KS_MIN_ATTACKERS=2` — Ethereal-style attacker-count gate; lets KS_FLOOR come down without waking calm
  positions (0-1 attackers → 0). Coded, off (cpp_bitboard.cpp:5680).
- `ENABLE_KS_CHECK_V2` — typed saturating safe-checks (Stage-2 shape). Built, off.
- `KS_ATT_QUEEN` (default 5) — proximity weight; SF weights queen LOWEST for proximity danger (queen
  danger lives in CHECKS). De-invert = drop to 3/2.

Genuine NEW-CODE items (author with attention, not blind mid-game):
- **Floor ramp** — only if NOT going the KS_ACCUM route: replace the hard `units<KS_FLOOR → 0` with a
  short linear ramp over [KS_FLOOR-Δ, KS_FLOOR]. (KS_ACCUM_MODE already gives a ramp-from-threshold, so
  try that FIRST and skip this.)
- **KS_BATTERY wiring** — declared (search_engine.h:1335) but UNWIRED (cpp_bitboard.cpp:5293). Stage-1
  feeder: a Q-behind-R battery currently counts as one attacker. Wire into the attacker-unit sum.
- **Open-file predicate** — "open" tests OWN pawns only, so an enemy-rammed file reads as fully exposed
  (fires on every castled king after any pawn trade). Fix to require both sides' pawns absent for "open",
  own-only for "semi".

## PHASE 0A — eval-loss PER FAILURE-PATTERN (static; do this FIRST — it decides the whole regime)

The strategic question is not "which lever" but "which REGIME are we in": is the KS eval-loss concentrated
in a few CORRECTIVE feeder bugs (→ targeted fixes, fast, high prior) or DIFFUSE across many patterns (→ a
clean-room re-shape, slow)? Measure before betting. Pure-static chain, no search:

    # 1. tag every collapse FEN with its structural KS pattern(s) — pure python-chess, no engine
    pyrun diagnostics/_ks_pattern_classify.py --in ks_sets/collapse_dataset_classified.csv \
        --fen-col decision_fen --color-col our_color \
        --out ks_sets/collapse_ks_patterns.csv --emit-fens-dir ks_sets/kspat
    # 2. per-pattern eval-loss (win% error, ours vs SF11) — one static gap run per pattern .fens file
    #    (loads the engine ONCE for static ev_breakdown; NOT a search — light)
    pyrun diagnostics/sf11_collapse_gap.py --fens-file ks_sets/kspat/_kspat_us_QUEENLESS_ATTACK.fens
    pyrun diagnostics/sf11_collapse_gap.py --fens-file ks_sets/kspat/_kspat_us_BATTERY.fens
    #    …one per pattern; also _collapse_leverage.py for points-forfeited if the corpus is game-tagged.

Read-out: rank the 9 patterns by mean win%-gap × count (leverage). CONCENTRATED in 1-3 (esp. the corrective
ones — OPENFILE_MISREAD, BATTERY, PINNED_DEFENDER) ⇒ targeted lever/feeder fixes, and "understand tonight /
ship tomorrow" is realistic. DIFFUSE across many ⇒ the shape is wrong ⇒ clean-room re-shape (tuned on
REGRET, never SF-fit — the −85.6 trap). ⚠️ The classifier is structural only; the OVER-vs-UNDER direction
comes from the sign of the sf11_collapse_gap KS term, not the pattern flag. Validate the classifier first:
`pyrun diagnostics/_ks_pattern_classify.py --selftest`.

## PHASE 0B — prune-transmission diagnostic (zero code; run after 0A)

Size how much of the queenless/mid over-read is transmitted through the prune gates (Stage-3) vs the leaf.
Existing knobs, no build:

    ks_phase SET=ks_sets/game_regret_set.csv KSMAG_TEST=1 QSPLIT=1                        # default (control)
    ks_phase SET=ks_sets/game_regret_set.csv KSMAG_TEST=1 QSPLIT=1 FUTILITY_EVAL_MODE=2 RFP_EVAL_MODE=2

Decision: if the over-read (queenless / mid-high-material buckets) SHRINKS materially under mode=2, the
defect is prune-transmitted ⇒ prioritize CONTINUITY (Phase 1) over leaf re-weighting. If it barely moves,
the defect is in the leaf value ⇒ prioritize Stage-2 SHAPE (Phase 2) first. Read QSPLIT per-side-queen ×
material; report the resolvable buckets only.

## PHASE 1 — continuity (existing knobs; touches LIVE term ⇒ symmetry gate + games)

Regret ruler first (all vs the Phase-0 control), then MIRROR gate, then games on the survivor(s). Run each
as its own process.

- **Arm A — kill the phase cliff:** `KS_EXTEND_EG=1`
- **Arm B — replace the hard floor with a ramp:** `KS_ACCUM_MODE=1 KS_ACCUM_THRESH=13 KS_ACCUM_LIN=<t>`
  (sweep KS_ACCUM_LIN so the mid derivative matches the current linear-segment slope; keep KS_NQ_SUP /
  KS_WIN_SUP at their tuned values). This is the continuous-gate design.
- **Arm C — floor down behind an attacker gate:** `KS_MIN_ATTACKERS=2` with KS_FLOOR lowered (e.g. 6),
  so calm positions stay 0 but the 0→step vanishes.
- **Arm A+B combined** (only if both survive the ruler + mirror individually).

Gate: each arm must pass `_eval_symmetry.py N=800 <its knobs>` (these change the live eval). Any arm that
fails the mirror is OUT regardless of ruler gain.

## PHASE 2 — Stage-2 shape (existing knobs; live ⇒ symmetry gate + games)

- **Typed safe-checks:** `ENABLE_KS_CHECK_V2=1` (a queenless R+B should NOT out-score a queen attack).
- **De-invert proximity:** `KS_ATT_QUEEN=3` (and a 2 arm). SF weights queen LOWEST for proximity.
- (Optional) multiplicative no-queen shear — separate arm from the flat KS_NO_QUEEN unit-subtraction.

## PHASE 3 — Stage-1 feeders (NEW CODE; the OPPOSITE under-read tail)

Author carefully (colour-symmetry is easy to break here — non-mirrored file/rank windows). Each is its own
gated knob, default byte-id, then ruler + mirror + games.
- Wire `KS_BATTERY` (x-ray/battery sight into the attacker-unit sum).
- Fix the open-file predicate (both-sides-absent = open, own-only = semi).
- Corner-zone shrink (clamp built, off) and pinned-defenders-count if time.

## SCREENING TIERS — regret → fast A/B (rank) → lightning (ship)

Three tiers, cheap→expensive. The middle tier (owner's idea) is ALREADY BUILT — no harness work.
1. **Regret ruler** (`ks_phase`) — widest + cheapest (~1h/knob), + per-position diagnostic (where/why).
   Screens MANY knobs → shortlist.
2. **Fast A/B games — RANK the shortlist on the real objective.** Existing subs:
   `fast_ab <depth> <p1cfg> <p2cfg>` (fixed depth, e.g. depth 7) or `node_ab <nodes> …` (fixed-NODE,
   lower variance). Precedent: `ks_ovd_fastrank` used exactly this. ⚠️ **Fast-depth Elo COMPRESSES ~3×
   and RANKS only** — it orders candidates, it is NOT the ship number. ⚠️ **EVAL knobs only** — the harness
   itself flags that fixed-depth/node is "blind to a change that alters eval COST," so SEARCH knobs
   (node/speed changers) skip this tier and go straight to the time gate.
3. **Lightning gate — the ship decision.** `gate`/`gate_blitz` (SPRT) or `tournament` (timed A/B). The only
   venue that yields the real, uncompressed Elo. Run ALONE.

Flow: regret shortlist → `fast_ab`/`node_ab` rank (eval knobs) → lightning SPRT on the 1-2 best. Bonus:
regret-vs-fast_ab disagreement directly tests whether the regret proxy predicts the objective (cheap).

## GENERALIZATION GATE — the variant set (owner-requested diversification)

`ks_sets/variant_regret_set.csv` exists and is DROP-IN (identical schema to `game_regret_set.csv`:
SF18 multi-PV labeled; 960/piece-swap shuffled starts = structure-independent). Use it as a HELD-OUT
generalization check, NOT a tuning target:
- Tune every KS candidate on the standard D7 regret set; then RE-SCORE the survivor on the variant set.
- Ship signal: helps standard AND holds on variant = reading real danger. Helps standard but flat/negative
  on variant = OVERFIT to standard-chess structure (the exact failure this set is built to expose).
- Do NOT add it to the tuning corpus (tuning on it chases 960 quirks and destroys the independence).
- Especially apt for KS: piece-swap / no-knight geometry is all-diagonal king attacks — a fix that reads
  danger rather than memorized patterns generalizes here.
- ⚠️ Tonight sanity-check before trusting it: (a) **phase distribution** — sampled rows are all `opening`
  phase_bucket, and the generator's known caveat is "short walks bucket everything as opening"; if it's
  opening-heavy it is WRONG for KS (a midgame concern) and needs a re-gen with `WALK_MAX≈45`; (b) row count
  adequate; (c) SF18 label depth matches the standard set (schema matches; depth unverified). The classifier
  `_ks_pattern_classify.py` runs variant-agnostically (verified on the no-knights game), so per-pattern
  leverage can be split by set too.

## ARM SPECS — copy-paste, ONE process per arm (verified knob defaults from KS-CODE-INVENTORY)

All arms run on the CLEAN regret ruler. Base command per arm (control = no extra KS knobs):

    ks_phase SET=ks_sets/game_regret_set.csv KSMAG_TEST=1 QSPLIT=1 [ARM KNOBS]

and every survivor is RE-SCORED on the variant set (generalization gate) by swapping
`SET=ks_sets/variant_regret_set.csv` (verify its phase spread first — see that section).

| # | Arm | Extra knobs (default → arm) | Class | Notes |
|---|---|---|---|---|
| C0 | control | (none) | baseline | the clean default; every delta is vs this |
| 0B | prune-transmission | `FUTILITY_EVAL_MODE=2 RFP_EVAL_MODE=2` | diagnostic | KS-free margins; does the over-read shrink? |
| A | phase-cliff → taper | `KS_EXTEND_EG=1` | continuity | kills the 71%→0 cliff at phase 64→65 |
| B | floor → ramp (accum) | `KS_ACCUM_MODE=1 KS_ACCUM_THRESH=13 KS_ACCUM_LIN=96` | continuity | LIN=96 reproduces the live slope 6; sweep LIN {64,96,128} |
| B2 | accum + no-queen gate | `KS_ACCUM_MODE=1 KS_ACCUM_THRESH=13 KS_NQ_SUP=35` (sweep {30,35,40,46}) | redistributive | the real no-queen GATE (vs the flat −6 haircut) |
| C | floor-down behind gate | `KS_MIN_ATTACKERS=2 KS_FLOOR=6` | continuity | calm stays 0; the 0→step vanishes |
| S1 | typed safe-checks | `ENABLE_KS_CHECK_V2=1` | reshape | queenless R+B must stop out-scoring a queen attack |
| S2 | de-invert proximity | `KS_ATT_QUEEN=3` (and a `=2` arm) | reshape | SF weights queen LOWEST for proximity |
| NQ | flat no-queen sweep | `KS_NO_QUEEN=20` (sweep {12,20,28,35}) | subtractive | cheapest; the confirmed clean-regret defect (ranked #1) |
| ALL | all-feeders-on (owner's idea, as a MEASUREMENT not a fit) | `KS_SQC_MODE=1 KS_PIN_MODE=1 KS_WEAK_VAL_MODE=1 KS_FLANK_MODE=2 ENABLE_KS_CHECK_V2=1` | confirmation | expect over-fire (2026-08-12 detector lesson); watch it fail/pass on the CLEAN ruler |
| P | pins-only feeder fix | `KS_PIN_MODE=1` | corrective | phantom-defender fix; batch feeder-hygiene |

⚠️ These touch the LIVE KS term ⇒ each must pass `_eval_symmetry.py N=800 <its knobs>` BEFORE it earns a games slot. Byte-identity does NOT cover them (real-change deltas). Corrective FEEDER fixes needing code (open-file predicate, storm-blockage, KS_BATTERY wire) are NOT knob-only — stage them separately once 0A says they have leverage. The KS-relevant parked-lever re-open (`ENABLE_KS_CHECK_V2`) IS arm S1 above (corroborated as a contamination-suspect kill). The other 12 re-openable levers (eval + search) live in `KS-PARKED-LEVER-RESCREEN-QUEUE-2026-08-15.md` as a LOWER-PRIORITY queue — invalid nulls to clear, not expected wins; run in leftover engine hours, prioritizing rows 7/8/9 (history/ordering knobs = the contaminated mechanism). Do NOT let them crowd the KS critical path.

## DECISION RULES / DISCIPLINE

- Regret ruler is the SCREEN (searched move, clean harness); |balanced STS| < ~150 is unresolvable; GAMES
  decide above the ~20-40 Elo floor.
- Every live-term arm passes the mirror gate BEFORE it earns a game. Byte-identity does NOT cover these —
  they are real-change deltas.
- One process per knob (latch at init). Games ALONE. JOBS ≤ 4, one engine-job at a time.
- Confirm each build's fingerprint against the register; re-read wac_speed peak after any probe.
