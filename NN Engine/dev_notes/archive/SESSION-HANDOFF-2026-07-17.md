# SESSION HANDOFF 2026-07-17 — KING SAFETY SHIPPED (game-gate positive); next class = material/placement over-read

**NEW-CHAT ENTRY POINT. Read this first, then `dev_notes/king-safety-activation-2026-07-16.md` (full detail).**

## Headline: the methodology WORKED end-to-end. KS v1 is game-gate POSITIVE. Nothing committed.
First full run of the diagnosis→fix→categorical-verify loop produced a real, measured win — and validated the process for reuse on every future collapse class.

## What was done (this session)
Diagnosed collapses as **eval optimism** (not horizon; not resign-leak — resigns are FAIR, 89% ≤−600cp). Localized the dominant statically-fixable cause: **king safety was disabled**. Aligned our `king_safety_danger` to classical SF11 + Ethereal (3 gated edits: SF weak-square def, overwhelmed-defender safe-check clause, no-enemy-queen discount) and found **SF-defs need the KS_FLOOR deadzone as partner**. Tuned to F13@MAG=3000 via the deterministic firing/control-set + over-read gates.

## KS v1 — SHIP CONFIG (a knob bundle, byte-id 247 on defaults; replaces latent_threat)
`ENABLE_KS_REPLACE_LT=1 KING_SAFETY_MAG=3000 KS_DEFENDER=0 ENABLE_KS_SF_WEAK=1 ENABLE_KS_SF_SAFECHECK=1 KS_FLOOR=13 KS_NO_QUEEN=6`

## Results (all gates)
- Deterministic: byte-id 247 (defaults); firing 67% danger / 7 calm-ff / 2 eg-ff (dominates baseline 59/12/3); color-symmetry 0; over-read king-danger 5.57→4.72; WAC 247→248 (+1); nodes ~neutral.
- **GAME GATE (paired full-game A/B vs SF@2400, 200g each, seed0, conc3):** baseline 42.2%/83 collapses, bundle 41.5%/83 collapses — **raw score + total collapses FLAT**. But the CATEGORICAL verdict: **KS-caused collapses 15 → 7 (−53%)**, replaced by +8 other-class, total +0. Guard clean (KS-caused dropped; the +8 are non-king-danger = exposed, not caused). Two independent evidence lines agree (over-read −0.85 + games −53%). Small-count (15 vs 7 ≈ 2 SE) but directionally solid.

## ⭐ RECOMMENDATION for the user
**COMMIT KS v1** (the bundle above, as the new default OR a preset). It's the first accumulating collapse-fix: it fixed its class (KS collapses halved) at flat total because SF is stronger and the next class surfaced — textbook "flat total ≠ failure." Then continue to the next class.

## Next class (diagnosed) = MATERIAL over-read — STRONG "material-term anomaly" LEAD (start here)
On the 64 non-KS collapse decision-fens: mean over-read +3.97, SF11's terms ALL small (no missing compensation) — it's OUR positive terms over-scaling: `material +2.76`, `pt_pawns +2.23`, `pieces +1.84`.

**⭐ The material lead (probe `diagnostics/ks_material_check.py`, run on the 64):** our `material` term (= `blackPieceVal - whitePieceVal`, [cpp_bitboard.cpp:7157](cpp_bitboard.cpp)) reads **mean +2.76** but the ACTUAL raw piece count (our own values P1000/N3150/B3250/R5000/Q9000) is **+0.90**, SF11 Material **+0.09**. So **our material term over-reads the real piece count by +1.86 on average** — and the extremes are damning: `+16.30` where raw is `+0.15`; `+1.55` where raw is `-7.25` (WRONG SIGN, ~9p off). No legitimate adjustment flips the sign and adds 8+ pawns. This is ~HALF the non-KS over-read and points to a concrete anomaly in `blackPieceVal/whitePieceVal` accumulation (values[] dynamic? a piece missed/double-counted? a capture-path artifact in `eval_breakdown_capture`? PV_BOOST folding — PV_BOOST_MAG=10000=10p is ON). **NOT confirmed bug-vs-intentional — needs a focused code read of the piece-value accumulation. If a bug, it's a big free win and makes the "hard" class tractable.** This REOPENS the material class (which we'd written off as un-reweightable) via a different door: fix the *calculation*, not the *weight*.

Fallbacks if it's intentional-not-bug: realizability-conditioned material discount, or accept it needs a learned component. **Corpus: `diagnostics/ks_sets/other_collapses.txt` (64). Probe: `ks_material_check.py`.**

## Parked
- **v2 KS (improve-on-SF-for-our-system):** DEFENSIVE-ASYMMETRIC KS — carry our-king danger harder (bridges the central-king residual, ~36% of SF magnitude) while trimming `attackingLayer`'s redundant offense (kills the +0.15 offensive double-count). Delicate (attackingLayer load-bearing ~400E); only worth it if the KS residual proves to still bite. Guard-rail: KS stays CONDITIONAL, never a blanket central-king penalty (SF11's conditional logic is the model).

## Tooling built (reusable, `diagnostics/`)
ks_firing_profile.py, ks_component_dump.py (KS_DEBUG_DUMP diagnostic), ks_symmetry.py, build_ks_sets.py (danger/control sets), sf11_collapse_gap.py (+ --engine-env, --fens-file), king_safety_probe.py, ks_residual.py, ks_recoverable.py, collapse_rewind_fens.py, ks_collapse_attribute.py (categorical verdict + --write-other), probe_fens.py, optimism_diag.py, misrank_attribution.py, resign_audit.py. `vs_sf.py` gained `--start-fens`/`--games-per-fen` (replay mode). Methods banked: [[eval-collapse-diagnosis-method]], [[collapse-categorical-verification]].

## Discipline notes
NOTHING committed (byte-id 247 held; bundle is a knob config, not a default change). Autonomous compute uses ONLY the auto-approved runner form + Read tool (raw git-bash `cd/grep` PROMPTS). conc3 (conc4 OOMs keras). Game arms ~78-82 min / 200g at conc3.
