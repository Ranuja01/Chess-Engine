# Texel Fit C — fit every linear v2 parameter jointly, then price new detectors

@author: Ranuja Pinnaduwage (maintained with Claude)

Status: DESIGN, awaiting owner review (2026-09-27). Built from a read-only code audit of `eval_v2.cpp`. Predecessor: Fit
A (PST only, shipped as `PST_V2_TAPERED=2`, +38 vs SF18). Data: `fitC_std_d6` (30,000 shipped self-play games from a
frozen engine, the 14,784-opening `openings_uho_ext.txt`) plus `fitB_variant_d6` (4,000 variant-start games).

## 1. Term inventory (shipped; `EVAL_V2_PAIR=0`, so every scorer blends at its own site)
| scorer | fit parameters | linear? | treatment |
|---|---|---|---|
| material `v2_piece_value` | `values[]` (also SEE, ordering, null-move, phase) | linear, but a gauge with the PST means | **FIXED** |
| phase | `EVAL_V2_MG/EG_LIMIT` | hyperparameter | **FIXED** (Fit A2: flat) |
| PST `rung0_tapered_pst` | 384 tied cells | linear | mean pinned per (piece, leg) |
| pawn structure `pawn_structure_mp` | doubled mg/eg; isolated per file-class mg/eg (fitted in mp, pct folded in); backward; weak-unopposed | linear | ~14 scalars |
| passers `passer_value_mp` | eg rank table, mg rank table (now free), king-distance weights | linear with CAND_PCT fixed | ~14 |
| mobility `mobility_mp` | SF11-shape tables per (type, n), MAG / EG_PCT folded in, in mp | linear once the magnitude is folded in | 132; occupancy-weighted mean pinned per (type, leg); smooth along n |
| placement `placement_mp` | outpost N/B, behind-pawn, bad-bishop classes, trapped rook, weak queen (per leg) | linear | ~18 |
| king safety `ks_units` → danger | W[4], WEAK, ADJ, CHK[4], NO_QUEEN, ONSET, MAX (HALF pinned) | **NONLINEAR** (hinge + saturating curve) | C2 |
| draw classifier / KPK / tier2 | — | early-return rows | **EXCLUDE** via a flag |

## 2. Engine side: one source of truth
- `template<bool FEAT> eval_v2_body(..., V2FeatSink*)`. Each scorer adds `if constexpr (FEAT) fs->add(PID, side, count)`
  where it consumes a value. The shipped path is the `<false>` instance and compiles to today's code. Verify with the
  WAC fingerprint and NPS.
- API: `v2_features(board) → {id[], x_mg[], x_eg[], ks channels, block_mp[], total, phase256, flags}`, plus
  `v2_param_count/name/start`. **θ₀ is exported from C++** so that no unit conversion (mobility /95 ·128/213 ·1.25,
  percent knobs) lives in Python: the record has three unit-scale errors from exactly that.
- Shipping path: the fitted constants move into init-once static tables with a compiled-header mode and a
  `V2P_FILE` / `V2P_DUMP` override (the `v2_pst_init` pattern). Mobility in mp is not byte-identical (per-entry
  rounding), so it is gated by a ≤1 mp-per-piece bound.

## 3. Fitter generalisation
- Model: E = fixed + Σ_blocks X_b θ_b (+ g_KS in C2). Per-block priors: λ_b = 1/(2σ_b²), with a mean pin where a gauge
  exists and smoothness where the table is ordered.
- Keep the bootstrap STAB rule. Add a **minimum-support freeze**: a parameter active in fewer than ~2k training games
  stays at θ₀.
- **Gates:**
  - per-block residual: `block_mp` vs X_b θ₀,b within the truncation budget (median ≈ 0, p99 ≲ 5 mp);
  - **closure**: load θ̂, re-pass 50k rows, engine == fixed + X θ̂;
  - symmetry, colour 0/800 and file 0/651;
  - held-out val_hash / val_run;
  - then SPRT → replication → the SF18 gauntlet (paired) → the variant gate;
  - and the RFP re-sweep at the end of eval.

## 4. Stages
- **C1**: the linear tables that exist today, ~562 parameters.
  - C1a: PST frozen at fit A; measures the other blocks' incremental value.
  - C1b: everything jointly.
- **C2**: KS reproduced EXACTLY in Python from exported per-king channels (gate: equal to `king_safety` to the mp on
  100% of rows).
  - C2a: fixed shape, fitted scale per leg (also delivers the missing KS endgame leg).
  - C2b: joint L-BFGS on the unit weights with HALF pinned and strong relative L2.
  - The variant data weighs more here, since KS fires more there. ⚠️ Additive KS changes are 0-for-11 in the record.
- **C3**: new detectors at weight 0: shelter/storm tables, pawnless flank, KingProtector, defender count. Plus the
  existing zero-weight detectors (connected, path ladder, rook files, threats, reach, long diagonal, latent, tempo,
  bishop pair).
  - Kaufman stays out: collinear with material.
  - Gates: a Python oracle for each mask, detector antisymmetry, and fire rate / support before any fit.

## 5. Risks
- Pruning coupling: margins are in absolute mp. Re-sweep after the eval block.
- The Texel "already winning" confound: advanced-piece features.
- Self-play exploitation: the external gauntlet is mandatory.
- Collinearity: pawn mg PST ↔ passer mg; weak-unopposed ⊂ isolated|backward; mobility ↔ PST; KS ↔ king mg PST. L2
  resolves the fit, but individual parameters then lose meaning.
- Sparse parameters (trapped rook, weak queen, high queen mobility, KS checks) are handled by the support freeze.

## 6. C1 RESULT (2026-09-27) — built, gated, NULL in games; not shipped
**Engine:**
- Extractor (`13e3096`): per-block gate on 3.69M rows, max residual 4.1 mp.
- Fitted-value path `C1_V2_FIT` (`3066167`): default byte-identical; identity load within 6 mp of shipped (mean
  0.9); closure 2 mp.
- Engine-server logging fix (`c561d42`): load failures were being swallowed.

**Fit (`_texel_c1_fit.py`, 3.69M rows, λ₂ 3e-12):**

| variant | val_hash | val_run |
|---|---|---|
| PST-only control | −0.19% | −0.18% |
| C1-only | −0.50% | −0.50% |
| joint | −0.76% | −0.74% |
| joint, bootstrap-filtered (c1b; 126 of 488 supported cells) | −0.68% | −0.48% |

- The fit still hit its iteration cap at 1,500 (ill-conditioned directions: overlapping blocks).
- Largest moves: immobile endgame rook −0.36 → −1.70 pawns; rook mobility steeper; weak-unopposed mg −0.22; knight
  outposts 0.47 → 0.29; passer mg rank 7 −0.30, offsetting fit A's pawn PST.
- Gates: closure 2 mp, colour 0/800, WAC 249 (−3.9% nodes).

**SPRT** `sprt_c1b` vs shipped (NODE_LIMIT=50000, seed 70, elo0 0 / elo1 10): **H0**, `+890 -929 =639 / 2458 (49.2%)`,
**−5.5 ± 16.1**.

**Reading:**
- A ~0.5% held-out gain from re-weighting EXISTING SF-derived tables does not turn into Elo; fit A's 2.75% did. The
  PST was the one badly sized block.
- C1 mostly traded value between collinear blocks (mobility ↔ PST, passer ↔ pawn PST).
- ⇒ Stop re-tuning existing tables. The infrastructure's value now lies in **C3 (pricing NEW detectors: coverage)**
  and **C2 (KS: missing eg leg + untuned nonlinear internals)**.
- Untested alternatives, if ever revisited: the unfiltered joint fit (−0.74%), C1-only with PST fixed.
