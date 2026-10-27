# Our evaluation — living reference (terms, anchors, status, SF11 correspondence)

Map of OUR static eval (`cpp_bitboard.cpp::placement_and_piece_eval`, Black-positive milli-pawns, pawn=1000). **Living
doc — update as the eval evolves.** Companion: [[sf11_eval_reference.md]] (SF11's algorithms). Sign convention: eval is
Black-positive, so a WHITE advantage is NEGATIVE; White contributions subtract from `total`, Black add.

---

# ▶️▶️ EVAL v2 (ARM 1) — TERM MAP vs SF11 / SF15.1-classical  ★ added 2026-09-19

☠️ **THE TABLE FURTHER DOWN IS v1's (`EVAL_ARM=0`). DO NOT APPLY IT TO v2.** Several fields carry a
*different quantity* per arm, which is this project's most-repeated failure (4 recorded instances).

**Units:** ours is Black-positive millipawns; SF's trace is White-POV pawns ⇒ `white_pawns = -v / 1000`.

| SF trace row | v2 field(s) | quality of the correspondence |
|---|---|---|
| **Material** | `material` **+** `pieces` | ✅ **EXACT.** SF folds PSQT into Material; v2 splits them (`material` = piece values, `pieces` = PST only), so the SUM is the right comparand — never `material` alone |
| **Mobility** | `mobility` | ✅ **cleanest single pair** (the record's own recommendation for where to start) |
| **King safety** | `king_safety` | ✅ (+ `det_ks_units_w/b` for the detector/transformation split) |
| **Pawns** | `pawn_struct` | ✅ isolated/doubled/backward/phalanx/supported… |
| **Passed** | `v2_passers` | ✅ the WHOLE passer value |
| **Knights / Bishops / Rooks / Queens** | `v2_placement` (+ `v2_rookfile`) | ⚠️ **AGGREGATE vs SF's FOUR rows.** Compare sum-to-sum only; there is no per-piece split on our side |
| **Imbalance** | `kaufman_imbalance`, `pair_bonus` | ☠️ **PARKED** — absent by default |
| **Threats** | `threats` | ☠️ **PARKED** — absent by default |
| **Space** | `space` | ☠️ **PARKED** — absent by default |
| **Winnable** / Initiative | — | ☠️ **not rebuilt** (3 structural faults). ⚠️ Also never parsed by ANY tool we own — no `sf11_winnable` column exists |
| **Total** | `total` | ✅ |

### ☠️ TRAPS — every one of these has already cost the project something
1. **`imbalance_white/black` MEAN DIFFERENT THINGS PER ARM.** v1 = **OvD** (offensive-vs-defensive, no SF
   analogue). v2 = **signed per-side holdings** `-(w_mat + w_pst)` / `(b_mat + b_pst)`, an exact partition of
   `total`. Reading v1's mapping here would be a textbook wrong-field read.
2. **`det_w/b_mobility` is arm-dependent too** — v2 = area-filtered squares summed over N/B/R/Q; v1 = squares
   attacked that are not its own.
3. ★ **v2 FIXED v1's material-inclusion defect.** v1's `pieces`/`pt_*` bundled each piece's VALUE with its
   placement, so pairing them against SF's placement-only rows made us look 3-5 pawns over-read whenever we
   were simply ahead on material — **two headline patterns were withdrawn over this (2026-08-02), and a
   "placement is 8-16× SF11" claim over the same (09-06)**. v2's `pieces` is PST-only, so the pairing above
   is sound. ⚠️ `dossier_overread.py` and `overread_term_attribution.py` still contain the v1 defect.
4. **ABSENT ≠ ZERO.** Every v2 field publishes only when its owner actually ran. A parked term is missing
   from `ev_breakdown`, not zero — so a consumer using `.get(key, 0)` silently converts "we don't carry this"
   into "we measured zero here".
5. **`v2_rookfile` is absent in the shipped config** — `ROOKFILE_V2_OPEN`/`SEMI` are both 0, so v2 currently
   scores **no rook-on-open-file bonus at all**, while SF scores it inside its Rooks row. That is a genuine
   gap to note, not a publication bug.
6. **The SF parsers keep the MG column only** (`eval_vs_sf11.py:82` discards EG), so nothing in our cached
   corpora can answer an MG-vs-EG question even for SF.

### ⚠️ BEFORE READING ANY GAP AS A FINDING
- **Imbalance / Threats / Space / Winnable will show as MISSING.** All four were *measured and parked* — they
  are pre-answered, not discoveries. → `EVAL-V2-REBUILD-LOG.md` slice-3 closure.
- **The aggregate mean-gap form of this comparison is a CLOSED NULL** (2026-08-19: `capture_gains` −0.08 in
  collapses *and* −0.08 in the quiet control ⇒ "the eval-calibration-by-aggregate lane is CLOSED"), and
  tuning a term toward SF's magnitude is a resolved NEGATIVE (KS, 09-11: best fit, **+9.70% worst-case
  error**). ★ The ONE triangulation result that converted to Elo (+45, threats) came from a **silence** —
  "we read ~0 where SF reads large on a class that matters" — not from a mean. **Look for zeros, not gaps.**

### Publication status (2026-09-19)
v2 publishes **17 of 48** breakdown fields under the shipped config (was 14/45). Added this session:
`pawn_struct`, `v2_passers`, `v2_placement`, `v2_rookfile`, plus the four parked terms when enabled.
★ Publication is inside `if (g_capture_eval_breakdown)`, which is false during search ⇒ **byte-identical**:
re-verified v2 `250 / 49,440,513 / EBF 4.031` and v1 `250 / 35,310,778 / EBF 3.784`, both exact.

---

## Per-term breakdown (`EvalBreakdown` / `ChessAI.ev_breakdown`) — ⚠️ **v1 / `EVAL_ARM=0` ONLY**

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

## Per-theme STS diagnostic findings (2026-07-01, living)

Method + tooling: `sts_full` (per-theme scoreboard) → `sts_gap`/`diagnostics/sts_sf11_gap.py` (our-vs-SF11 per-term on a theme's failures) → `diagnostics/pvb_delta.py` (played-vs-best per-term DELTA = the decisive culprit; aggregate signed means are noise). Scoreboard 2026-07-01 = **51.2%**; weakest: Advancement 39.7%, AKPC 40.2%, King Activity 43.1%, Open Files 43.8%.

- **Advancement of a/b/c pawns (39.7%) — HARD, parked.** Best move is a flank pawn advance (b2b4/a4a5/…); we play a piece maneuver/capture. pvb_delta: culprit is **`pieces` +0.30 + `material`/`capture_gains`** (piece-activity + material grabs over-favored), **NOT** the pawn terms — `pt_pawns`/`passed_pawn_support`/`pawn_struct`/**`pawn_majority`(enabled)** all ~0 delta (reactivating pawn_majority DISCONFIRMED here). The push-value SF reads = **Space + Mobility**, a GAP we don't compute; some "misses" are legit material-winning captures (STS is thematic → true weakness < 39.7%). Fix needs a deliberate space-on-advance term or broad `pieces` dampening — not a knob-tweak. See [[eval-theme-diagnostic-loop]].

See dev log `collapse-campaign.md`. Pointers: [[capture-gains-overread]], [[sf11-texel-scale-invariance]], [[ks-detection-rebuild]], [[eval-theme-diagnostic-loop]], [[eval-accuracy-payoff-is-pruning]].
