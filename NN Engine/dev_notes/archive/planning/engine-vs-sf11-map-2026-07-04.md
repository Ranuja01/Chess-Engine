# Engine vs SF11 (classical) — full audit + prioritized gap map (2026-07-04)

Synthesis of a 3-agent research sweep (our eval audit, SF11 eval design, our search vs SF11) + our measured
outcome-compass data. SF11 (classical, ~3450 CCRL) is ~750 Elo above us with a HAND-CRAFTED eval → the gap is
classical-reachable, not an NNUE requirement. **The ~590-Elo EQUAL-DEPTH gap is EVAL-dominated** (search speed
neutralized) → per-node eval + ordering quality is the dominant lever, NOT search depth. So: eval first.

## ★ HEADLINE FINDING — our king-safety is architecturally BROKEN (and it confounded the KS screen)
The eval audit found king-shelter is **TRIPLE-COUNTED live** even with `king_safety` gated off:
1. `evaluate_kings_midgame` explicit shield bonuses (+185/+75), 2. `setAttackingLayer` reducing enemy attack value
on shield squares (→ `pieces`), 3. `latent_threat` defender-presence increments. Weak-squares-near-king are
DOUBLE-counted (setAttackingLayer + latent_threat); king-zone attacker pressure overlaps too. **Every LIVE king
signal is FLAT, un-realizability-conditioned, midgame-only.** The ONLY site with realizability (`KS_DYN`) + a
proper phase-taper is `king_safety_danger` — which is **gated off** (`KING_SAFETY_MAG=0`).
⇒ (a) This IS the "illusory-activity static over-read" collapse mechanism, structurally. (b) The 23% KS
incremental-validity screen was **CONFOUNDED** — enabling `KING_SAFETY_MAG` stacked a 4th shelter copy on the
triple-counted live one, so 23% never measured a clean term. (c) The user's CONSOLIDATION instinct is exactly
right and now evidence-backed.

## EVAL TERM MAP (ours) — LIVE / GATED, verdict, measured capture where known
LIVE: `pieces` (PST + attack-layer; corr 0.70 w/ SF **Material** not placement — a material-scaled good-squares
helper), `capture_gains` (**RETUNE #1 over-read**, live+conditioned via capg-tension), `passed_pawn_support` (=SF
Passed, KEEP), `latent_threat` (king-directed, midgame-only, part of the triple-count), `central` (=SF **Space**,
corr 0.60, KEEP under-read), `imbalance_white/black` (**OvD — NO SF analog, ours-unique, KEEP**), `pair_bonus`
(KEEP), `piece_value_boost` (KEEP), `advanced_endgame` (binary draw-detect ≈ crude SF scale-factor), rich 13-term
rook suite (**≥ SF, KEEP**), cheap bishop-colour complex (bad-bishop analog).
GATED-OFF: `threats` (our immediate-threats fn #1, **+0.00063 outcome = 52% of SF threats ceiling**, banked),
`king_safety` (realizability-conditioned fn, the consolidation target), `mobility` (per-piece non-linear table =
SF Mobility; **incremental validity +0.0022 = real lever**), `space` (dead-as-flat), `pawn_struct`/`outpost`/
`pawn_majority` (built, un-park candidates), `rook_cond`, placement conditioners (MOD_PIECES_*, MOD_LT_BACKING).

## KEEP / BUILD / RETUNE (mapped to SF11 + outcome data)
- **KEEP (valid, ≥ SF or ours-unique):** OvD/imbalance (no SF analog), rich rook suite, material, piece_value_boost,
  central↔Space, pair_bonus. SF has NO named-motif detectors → do NOT build fork/skewer finders.
- **BUILD (SF grades these, we lack or gate them) — ranked by measured/est. outcome value:**
  1. **King-safety CONSOLIDATION + shelter_storm + kingDanger² severity** — the #1 eval lever. Details below.
  2. **Mobility** (un-gate the per-piece table; +0.0022 real lever) — retire the cheap surrogates if it wins;
     tune the non-linear tables via PACE/SPSA, midgame-weighted.
  3. **Threat family** (immediate DONE +0.00063; add pawn-push / restricted as separate granular fns → bundle-gate).
  4. **Continuous endgame scale-factor** (SF ×scaleFactor/64) — replace our binary `advanced_endgame` draw-detect
     with a continuous down-scaler (OCB / few-pawn / wrong-rook-pawn). Cheap, corrects endgame over-valuation.
  5. **Initiative** term (whole-board winnability nudge) — small, endgame.
  6. **Trapped-rook** — BLOCKED (needs castling rights threaded into eval; not currently available). Defer.
- **RETUNE:** `capture_gains` (#1 over-read, cut ~0.7× / condition) — but note the global scalar outcome-Texel
  retune was TAPPED (aggregate-calibrated); capture_gains conditioning is already live (capg-tension).

## ★ THE `latent_threat` POV ERROR (user insight 2026-07-04) — we optimized king-safety through a "threats" lens
`latent_threat` is NAMED + knob-tuned as a threats term (`THREAT_ATTACK_MULT`/`THREAT_PRESENCE_MULT`/`THREAT_PAWN/
KNIGHT/...`) but FUNCTIONALLY scans the KING ZONES = it's a KING-SAFETY signal in threat clothing. SF11's
`threats()` is a DIFFERENT concept (piece-on-piece). So historically we read/optimized a king-safety term from the
wrong (threats) POV. Implications: (1) `latent_threat` conceptually BELONGS INSIDE the consolidated king-safety
function — it should be ABSORBED by the KS consolidation, not left as a standalone "threat" term. (2) Our granular
split finally put the REAL threats concept (piece-on-piece = our new `threats` term, +0.00063) where it belongs,
separate from KS. (3) The THREAT_* knobs on latent_threat are really KS knobs — rename/re-home during consolidation.
⇒ KS consolidation TARGET = fold {latent_threat king-directed scoring + evaluate_kings_midgame shield/exposure +
setAttackingLayer king-zone credit} into `king_safety_danger` (the realizability+taper site), dedupe, THEN enrich.

## ★ KING-SAFETY REBUILD — the reframed plan (was "23% ceiling", now "broken + confoundable")
1. **CONSOLIDATE + DEDUPE (do FIRST, de-confounds everything):** lift the scattered king-shelter / weak-square /
   king-exposure scoring OUT of `latent_threat` + `evaluate_kings_midgame` + `setAttackingLayer`, into the single
   realizability-conditioned `king_safety_danger`. Remove the triple/double counts. Gated + byte-id-safe (a big,
   careful refactor — the shield/weak-square credit must move, not duplicate). THEN re-screen `king_safety` for a
   CLEAN capture number (the 23% was confounded).
2. **ADD the missing richness — per-file shelter_storm** (SF's biggest KS ingredient, ABSENT from our docs AND our
   code — only a crude shield COUNT exists). Need SF11 `pawns.cpp evaluate_shelter`: `ShelterStrength` (shelter-pawn
   rank/advancement per file), `UnblockedStorm`/blocked-storm (enemy storm-pawn distance-to-king), open vs half-open
   file (by side) penalties, best-of-king-file-and-castling-squares selection, `min(rank, RANK_5)` storm clamp.
   Reconstruct source-verified (I can do it; escalate to Fable ONLY if a design fork).
3. **SEVERITY (the under-scaling half):** SF's `kingDanger` is super-linear — `kingDanger²/4096` when >100; safe-
   check weights R1080/Q780/N790/B635; weak-sq 185, unsafe-check 148, knight-blocker 98, king-attacks 69, flank²/8;
   `−873×(no enemy queen)`. Our danger=units²/KS_DIVISOR is the same FAMILY; align the component weights + the
   no-queen discount + safe-check weights (our KS_ATT_*/KS_SAFE_CHECK are much smaller). This addresses the "we
   under-score real attacks (mean 1.26 vs SF 2.13)" half of the gap.
4. **Phase-gate midgame-only** (data: KS +0.0051 mid → +0.0005 end, collapses 10×; the `ks_phase_taper` already does
   this). Fit magnitude (outcome), node_ab + BLITZ (KS pays off with TC), SPRT.
5. NNUE evidence ONLY if a FAITHFUL SF11-form KS (post consolidate + shelter_storm + severity) ALSO plateaus.

## SEARCH MAP (ours vs SF11) — secondary (gap is eval-dominated)
HAVE (modern, complete): ID+aspiration, PVS, TT (depth-preferred), null-move (depth-adaptive R), LMR (history/
cont-aware), LMP, futility, razoring, **RFP (shipped +73)**, check extensions, qsearch (delta-pruned), killers/
history/counter, 1-ply cont-hist, SEE ordering. Knob sweeps are TAPPED (RFP was a new MECHANISM, not a sweep).
GATED-OFF: SEE-prune, capture-hist, 2-ply cont-hist, null-eval-gate, TT-move ordering, check-order, null-
progressive, improving, etc. (all screened dead/marginal).
**MISSING + the structural blocker:** `TTEntry` has **NO best-move field** → blocks **Singular Extensions, IID/IIR,
Multi-cut** (all need a TT hash move). **ProbCut** is the one missing mechanism NOT blocked by it. Only checks are
extended (no recapture/promotion forcing extensions). EBF ~3.8 vs SF ~2.0 (23× node gap = these missing mechanisms).
TOP SEARCH LEVERS (for later): (1) **add a TT best-move field** — unlocks singular ext + IID + cleaner interior
ordering, the highest-leverage structural search change; (2) **ProbCut** (buildable now, reuses the RFP node-entry
eval); (3) forcing extensions. But eval work outranks all of these (590 Elo equal-depth is eval).

## PRIORITIZED ACTION LIST (Elo-per-effort, outcome-data-informed)
1. **KS consolidate+dedupe → re-screen clean → shelter_storm + severity align** (biggest eval lever, and fixes the
   collapse mechanism at the root; the confounded 23% becomes measurable).
2. **Un-gate + tune mobility** (+0.0022 real lever, cheap-ish, midgame-weighted).
3. **Threat family**: bundle-gate immediate-threats (+0.00063) + build pawn-push/restricted granular fns.
4. **Continuous endgame scale-factor** (replace binary draw-detect; cheap correctness).
5. **Search (later): TT best-move field → ProbCut / singular ext.** Structural, secondary to eval.
KEEP as-is: OvD, rook suite, material. RETUNE only opportunistically (global Texel is tapped).

## Method carried
byte-id 245/39,146,294 after any build. Outcome compass (incremental_validity) proposes → node_ab gates → SPRT
ships. Consolidation must MOVE credit, not duplicate (verify byte-id + re-screen). SF = feature LIBRARY (borrow
directions/algorithms), never a magnitude TARGET. Reconstruct SF algorithms source-verified; Fable only on a
genuine design fork.
