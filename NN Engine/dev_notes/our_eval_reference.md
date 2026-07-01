# Our evaluation — living reference (terms, anchors, status, SF11 correspondence)

Map of OUR static eval (`cpp_bitboard.cpp::placement_and_piece_eval`, Black-positive milli-pawns, pawn=1000). **Living
doc — update as the eval evolves.** Companion: [[sf11_eval_reference.md]] (SF11's algorithms). Sign convention: eval is
Black-positive, so a WHITE advantage is NEGATIVE; White contributions subtract from `total`, Black add.

## Per-term breakdown (`EvalBreakdown` / `ChessAI.ev_breakdown`)

| our term | what it computes | anchor (cpp_bitboard.cpp) | status | SF11 correspondence |
|---|---|---|---|---|
| `material` | piece-value sum (`whitePieceVal-blackPieceVal`) | per-piece evaluators | KEEP | = SF Material |
| `pieces` | per-piece PST/placement + attacking-layer, per-piece-type (`pt_*`) | `evaluate_*_midgame/endgame` | KEEP (weak) | **NOT SF placement** — corr **0.70 w/ SF *Material***, ~0 w/ SF per-piece. A material-scaled "general good-squares" helper, not SF-style placement |
| `capture_gains` | static SEE/qsearch approximation (horizon-effect guard): per attacked square, `see()` + alternating-capture sim | `approximate_capture_gains` ~7008 | RETUNE (over-read) | ~ SF Threats (corr 0.24). #1 OVER-read ([[capture-gains-overread]]) — cut ~0.7×. Latency-heavy |
| `passed_pawn_support` | passed/candidate pawn support (`getPPIncrement`, spans) | pawn evaluators / `getPPIncrement` ~7469 | KEEP | = SF Passed |
| `latent_threat` | **king-directed** latent threats (two king-zones, per-zone-square attacker scan) — expensive | `get_latent_threat_score` ~5062 | KEEP (live!) | ~ SF King safety (corr 0.27) + Threats. NOT deprecated (`ENABLE_KS_REPLACE_LT` default-off). 4-8× heavier than KS |
| `king_safety` | attack-unit king danger (zone attackers/defenders, weak sq, open files, safe checks, pawn storm) | `king_safety_danger` ~4907 | PARKED as eval | = SF King safety. Rebuilt, but not a lightning lever (search covers it, [[ks-detection-rebuild]]) |
| `central` | centre control | central-score feed | KEEP | **= SF Space** (corr 0.60 — the one clean positional match). Under-read (attribution →1.9), small |
| `imbalance_white/black` | **OvD** = offensive-vs-defensive pressure imbalance (`whiteOffensiveScore` vs `blackDefensiveScore` × `IMBALANCE_SCALE`) | ~6304-6318 | KEEP (ours-unique) | **NO SF analog** (SF "Imbalance" = Kaufman material-combo, different). Our own lens — corr ~0 w/ everything |
| `pair_bonus` | bishop/knight pair existence bonus | ~6517-6536 | KEEP | part of SF Bishops (SF has a synthetic bishop-pair in Imbalance) |
| `piece_value_boost` | material-DOMINATION boost: `(matDiff/leaderMat)×PV_BOOST_MAG`, escalates as board thins | ~6605-6663 | KEEP | ≈ a lead-converter; distinct from SF Imbalance (which is combination-valuation). corr 0.62 w/ SF Material |
| `pawn_majority` | wing pawn-majority (candidate passer) bonus, phase-blended + modulators | ~6673-6705 | NEW (gated) | ~ candidate-passer flavour of SF Passed/Initiative |
| `pawn_struct` | **isolated** (no adjacent-file friendly pawn) + **backward** (adjacent pawns all ahead + stop-square enemy-pawn-controlled) | pawn-struct block after majority | NEW (gated) | = SF pawns.cpp isolated/backward (we also have doubled hardcoded @731) |
| `outpost` | knight/bishop in enemy half, pawn-defended, no enemy pawn can advance to attack | outpost block after pawn_struct | NEW (gated) | = SF Outpost |
| `mobility` | per-piece `popcount(attacks & safe mobilityArea)` → non-linear `MobilityBonus[piece]` table (PEXT-reuse) | mobility block after outpost | NEW (gated) | = SF Mobility. Replaces the cheap rook/knight/queen surrogates when on |
| `advanced_endgame` | mate-drive (king-to-edge), passer king-race, KPvK/insufficient-material draws | `advanced_endgame_eval` ~4686 | KEEP | ~ SF endgame scale factors (but binary draw vs SF's continuous scale) |

Also: rook activity is RICH (13 terms — `ROOK_OPEN/7TH/CONNECTED/SEMI/PASSER_*/OWN_PAWN/MINOR_BLOCK/ROOK_BLOCK`, ~2010) —
≥ SF in most respects; cheap bishop-colour-complex (`get_bishop_colour_complex_score`, bad-bishop analog).

## KEEP / BUILD / RETIRE (from the SF11 gap analysis)

- **KEEP (ours, valid — not deficient):** OvD/`imbalance` (no SF analog — our unique lens), `piece_value_boost`,
  `latent_threat` (live, king-directed), rich rook suite, `central`↔Space. SF has **no named-tactic detectors** → do NOT
  build fork/skewer finders.
- **BUILD (cheap gaps SF grades, we lacked):** `mobility` (proper per-piece — DONE, gated), `pawn_struct`
  (isolated/backward — DONE) + un-park `pawn_majority` (DONE), `outpost` (DONE). Smaller/later: doubled/isolated as tunable,
  connected/phalanx, trapped-rook (needs castling-rights threaded into eval — not currently available), OCB endgame
  draw-scaling, Initiative term.
- **RETUNE:** `capture_gains` (#1 over-read); the cheap knight/queen mobility surrogates (retire if full mobility wins).

## Method notes

- New term → tunable: `br_<term>` accumulator → `EvalBreakdown` (cpp_bitboard.h) → `g_capture_eval_breakdown` publish →
  `ChessAI.pyx` cdef-struct + dict → `tune_corpus.py` TERMS. Gate: base knob 0 (or SCALE 100) = byte-identical.
- Tuning: scalar terms → Texel (`tune_fit --scale-inv --attrib`, corpus columns) + PACE; **mobility** = PACE/SPSA on the
  non-linear tables (not Texel). Correspondence-filter (`tune_fit --corr`) before gating — attribution finds collinear
  proxies (that's how `pieces`↔material was caught). Lightning SPRT decides; NPS is a first-class gate.
- Fitting our eval to SF11's TOTAL is confounded by a strength-neutral global scale (~1.5× hotter) → use scale-invariant
  ([[sf11-texel-scale-invariance]]). Depth: tactical terms (capture_gains, KS) are search-redundant at depth; strategic
  terms (structure/placement/material) transfer ([[fast-selfplay-eval-depth-bias]]).

See dev log `collapse-campaign.md`. Pointers: [[capture-gains-overread]], [[sf11-texel-scale-invariance]], [[ks-detection-rebuild]].
