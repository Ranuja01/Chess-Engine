# SESSION HANDOFF 2026-09-06 — the eval lane priced, both structural leads closed, and a new instrument

## READ FIRST
- ⭐ **THE EVAL LANE IS NOW PRICED AND VALIDATED.** Substituting SF's eval into OUR search:
  **SF11 +174 STS @d6 / +104 @d8; SF15-classical +219 @d8; SF15-NNUE +397.** A better eval is worth
  **3-4 plies** — SF15c's eval at d8 beats OUR eval at d10. 📄 [[sf-oracle-eval-substitution-is-built-and-sf11-buys-174-sts-at-d6]]
- ⭐⭐ **AND IT BUYS PRUNING HEADROOM — the owner's reframing, measured.** Cranking `RFP_MARGIN` 1500→400
  collapses OUR eval by **−207 STS**, does **not degrade SF11 at all (+78)**, and costs SF15c only −124.
  **SF11 @ rfp400 = −40% nodes AND +98 STS vs our baseline.** ⇒ a truer eval's value is largely
  **tolerating a harder crank**, not being better at our current margins.
  📄 [[a-truer-eval-buys-pruning-headroom-the-crank-result]]
- ☠️ **BOTH STRUCTURAL LEADS ARE NOW CLOSED ON CLEAN INSTRUMENTS.** KS `FLOOR<KNEE` (the audit's #1
  never-run item) is WORSE at every lowered floor; the pawn taper's regret win REVERSED SIGN on the v2
  cross-set. 📄 [[taper-and-ks-floor-knee-both-refuted-on-clean-instruments]]
- 🔬 **NO MISSING TERM FOUND** — SF15c is TWO-SIDED against us (185 better / 159 worse) and no term
  separates the tails. 📄 [[eval-disagreement-mining-no-missing-term-but-our-components-are-10x]]
- ☠️ **AND ITS "8-16× placement / noisy-not-biased" COROLLARY IS WITHDRAWN (same day, by code read)**:
  `pt_*` accumulates the per-piece evaluator's raw return, which opens with `total -= values[X]`, so it is
  **placement + full face-value material** while SF11's rows are placement-only. A knight's non-material
  credit is clamp-bounded to **+500mp**, so the measured 1683 could never have been placement.
  📄 [[pt-star-is-material-inclusive-so-the-10x-claim-was-invalid]]
- 📊 **THE FAILURE RECORD IS MOSTLY UNRESOLVED NULLS**: ~85 attempts (not 9-11); of "additive KS 0-for-11",
  only ~4 are resolved negatives. 📄 [[the-eval-failure-record-is-mostly-unresolved-nulls-not-refutations]]

## 1. THE MEASURED PICTURE (all reproducible)
| | ours | SF11 | SF15c | SF18 |
|---|---|---|---|---|
| STS @ d10 | 1796 (59.9%) | 1983 (66.1%) | — | 2414 (80.5%) |
| STS @ 249,014 nodes | 1796 | **2374 (79.1%)** | — | 2605 (86.8%) |
| nodes to depth 10 | 249,014 | **26,265 (9.48× fewer)** | — | — |
| EBF (venue-matched) | **1.914** | **1.452** | — | — |
| depth at OUR node budget | 10 | **15** | — | — |
| eval swapped into OUR search @d8 | 1676 | 1780 | **1895** | 2073 |
⇒ time-to-depth = nodes × NPS ⇒ **53× slower at d10, 144× at d15** (EBF compounds, NPS is a flat 5.6×).
⇒ **~⅓ of the gap is per-ply judgement, ~⅔ is tree size**, and eval accounts for 3-4 of the ~5 plies.
🎯 **SF11/SF15c are the targets, not SF18.** ⚠️ Whole-position evals are NOT comparable across SF
generations (modern SF saturates large advantages) — read `PieceValue` from source instead.

## 2. SEARCH — PARKED, with one candidate
Four lanes closed with mechanisms this arc: margins (52 configs), LMR magnitude (saturated AT the optimum),
node type (AUC 0.568 vs move_index 0.815), futility calibration (small-population). Plus the presearch
cascade (never was a saving — it is damage control on a +38% increase).
✅ **`ENABLE_IIR=1` is the standing candidate**: quiet **−16.0%** nodes, **+1 median ply**, STS 1818,
**games +7.7 ±17.7 / 2400g** (both segments positive), unresolvable at any affordable N.
☠️ **0-for-10 at bundling** — every partner tried was another node-saver (same population). Default-off.
▶️ Untested, both cheap: `lowPlyHistory` (ordering class; spec in `LOWPLYHISTORY-PORT-SPEC-2026-09-04.md`)
and **selective root IIR** (reduce presearch depth only for root moves lacking evidence — the presearch×IIR
tie-in; the machinery exists, the uniform version `R=2` was a fixed-time null).
⚠️ **Every search closure was measured against our CURRENT eval.** An eval change invalidates the margin
calibration (measured: a foreign eval costs +3.6% to +53.7% nodes at our margins until re-cranked), so the
52-config sweep must be redone and IIR re-screened after any eval work.

## 3. THE NEW INSTRUMENTS (all reproducible, all extensions of existing tools)
| tool | what it answers |
|---|---|
| `ENABLE_ORACLE_EVAL` + `ORACLE_SCALE` + `ORACLE_ENGINE_PATH` | put SF's eval inside OUR search. ~6,200 evals/sec, **fixed-depth only** (NPS drops 14×). Sign verified vs `probe_fens`. |
| `_sf_node_bench.py --engine sf11` | SF's nodes AND depth on the IDENTICAL 60 quiet-corpus FENs |
| `_sts_reference.py --engine sf11 [--nodes N]` | SF scored on OUR sts300 with OUR scoring |
| `_search_stability.py VS=1 DUMP=` | **cross-arm** move diff + dump of flipped positions |
| `_ks_footprint_regret.py CAND_KNOBS= DUMP=` | "when it changes our move, is it better?" — SF18 regret on the changed set. **The deployment gate.** |
| `_tail_term_stats.py A= B=` | per-term SE (is a separation real?) and mean \|value\| (what signed means hide) |
⚠️ **THE CRANK TEST is the cheap eval screen**: a real eval improvement tolerates a harder margin crank.
Hours, fixed-depth, no games. It is what separated SF11/SF15c from ours when STS could not.

## 4. ✅ THE THREE-STAGE QUESTION — ASKED AND ANSWERED (09-06, two-agent code read)
All three stages came back clean, and the finding that motivated them was mine, not the eval's:
1. **FEEDERS** — base placement cells are ≤60mp (pawn) / ≤40 (knight) / ≤35 (bishop) / ≤65 (queen) at
   default scale 100 ⇒ **≤0.065 pawn per cell, 3-10× SMALLER than SF11's PSQT**. Each piece reads its table
   **once, at its own square**. The per-attacked-square accumulation is `attackingLayer`, a different table.
2. **TRANSFORMATION** — **no double-booking in the score.** `material` (`:8566`) is never added to `total`;
   `pieces` = Σ`pt_*` is the superset and `material` a sub-view of it. But `pt_*` DOES include face-value
   material, which is what made it incommensurable with SF11's rows. `det_w_pieceval` ≈ 44.4 is explained by
   the king's 12000 plus post-capture-simulation mutation — a field read as its writer did not intend.
3. **OUTPUT** — **nothing shrinks placement at defaults.** All three placement damps and the NPEDGE pawn
   damp are 0/false, and `br_pieces` is snapshotted at `:7650` *before* that whole region, so the exported
   `pieces` is the raw pre-adjustment sum in any configuration.
📄 [[pt-star-is-material-inclusive-so-the-10x-claim-was-invalid]]

## 4b. ▶️ THE LEAD IT SURFACED — our phase taper runs OPPOSITE to SF's
`EG_EXIST_{KNIGHT 200, BISHOP 250, ROOK 350, QUEEN 900}` (`search_engine.h:1161-1164`) is added by the
endgame evaluators only — and **there is no `EG_EXIST_PAWN`**. Effective pawn:knight goes `0.308` (mg) →
`0.290` (eg): the pawn **loses 5.8%** relative to pieces. SF11's goes `0.164 → 0.249`: the pawn **gains
52%**. Opposite sign. ⚠️ The 09-06 taper refutation swept `SCALE_PAWN_RANK` — the rank-PLACEMENT bonus —
as a **proxy** for pawn value; the value taper itself is untested and has a direct env-wired lever.
🔍 Same direction, second escalator: `piece_value_boost` (`:8355/8373`, live above `|total| >= 1500`) adds
**+2381mp — 47.6% of a rook again** — in a K+R+4P vs K+4P endgame; `PV_BOOST_PHASE_K` (default 0) exists to
damp exactly this. 🎮 Both bear on [[odds-losses-are-material-overvaluation-not-passers]].
⚠️ `EG_EXIST_KNIGHT/BISHOP=0` were old corpus-fit "switch off" winners ⇒ SUSPECT, i.e. **unclosed**, not
evidence. Gate any arm as **eval + re-cranked margins together**, never a standalone STS read.

## 5. ⚠️ INSTRUMENT LAWS EARNED THIS ARC
- **CRANK UNTIL IT BREAKS** — a margin question asked at gentle settings wanders inside the ±150 floor and
  reads null; cranked hard the same question resolves at −207.
- **Flip RATE does not rank evals** (oracle 63.5%, a known non-improvement 40%, pure noise 20.8%). Only
  regret on the flipped set ranks them.
- **Byte-identical-to-control is the signature of a silent 100% fallback** (the SF15 arms, from a
  colon-only parser). **A guard whose output the harness discards is not a guard** — `sts`/`wac` `2>/dev/null`.
- **Match the instrument AND the statistic before diagnosing a regression** — canonical NPS 450,201 is a
  LIGHTNING **mean**; the same script at LONG_FORMAT reads ~386k and fakes a −14% regression.
- **Signed means hide magnitude; n=40 differences need an SE.** The KS "4.4× separation" was −0.367 ± 0.517.
- **A knob essential in one experiment can be harmful in the next** (`ORACLE_SCALE=200` read −0.0177 where
  raw read −0.6681).

## 6. STATE
`273aebf` committed the search instrumentation + SF reference tooling (engine unchanged, all knobs
default-off, fingerprint verified). Since then, uncommitted: `ENABLE_ORACLE_EVAL`/`ORACLE_SCALE`/
`ORACLE_CLASSICAL`, `EVAL_NOISE_SIGMA` (parked; injected in the SEARCH eval path so it cannot be calibrated
against the corpus metric), the `_search_stability` VS mode, the `_ks_footprint_regret` arm override, and
`_tail_term_stats.py`. Baseline re-verified repeatedly: **250 / 35,310,778 / 3.784 / STS 1796 / quiet
249,014 / depth@1s 12**.
⚠️ Several dev_notes the memory index cites are still UNTRACKED (`SEARCH-SWEEP-2026-08-25.md`,
`SESSION-HANDOFF-2026-08-24/-27.md`, `PASSER-REARCHITECTURE-SPEC-2026-08-22.md`,
`SEARCH-INFRA-FLAVORS-2026-08-23.md`).
