# Search lane — REOPENING ANGLES (2026-07-24)

## The governing principle (user)
**Every feature we "closed" exists in every strong engine (SF11/15/18, Ethereal, Obsidian, Caissa).** So a failed
attempt is evidence about **our angle**, not about the feature. Past failures are valuable *because* they tell us
which angle is already excluded — they are not permanent verdicts. Re-approach with that knowledge.

## Why the closures are re-openable NOW (they were all measured against the OLD eval)
Several failure DIAGNOSES explicitly blamed eval quality:
- **ProbCut**: *"its shallow verification search inherits our search's PHANTOM SCORES"* — its whole premise is
  "shallow + margin predicts deep". A systematic attack over-read poisons that predicate. De-king removes a large
  part of that over-read ⇒ ProbCut was judged on a broken predicate.
- **Prune-push / LMR-push**: *"pushing reduction blunders"*, `LMR_EXTRA=3` → −4.9%. Pruning is unsafe when the eval
  mis-ranks. Better eval ⇒ safer pruning. NOTE the roadmap's own thesis: the ordering foundation is
  **NEUTRAL SOLO — "Elo lives only in the PAIR"** {ordering} × {pushed LMP/futility + steeper reduction}. The pair
  was never tested against a trustworthy eval; both halves were measured solo (and solo was predicted neutral).

## ★ The unified theory that emerged (reconciles corrhist vs de-king)
Corrhist's post-mortem declared **"load-bearing optimism … FUNDAMENTAL"** (de-biasing damps the optimism that
drives active play ⇒ −Elo). **Our de-king result CONTRADICTS that**: it removed a large systematic over-read and
STS went **UP +92**. The refinement:

| change type | effect on eval landscape | outcome |
|---|---|---|
| **UNIFORM de-bias / global magnitude shrink** (corrhist@RFP; `SCALE_*` cuts the corpus fit keeps picking) | FLATTENS → sibling moves less distinguishable | HURTS |
| **TARGETED double-count removal** (de-king: the `attackingLayer` king-boost counted in `pieces` + OvD + KS) | removes redundancy, keeps discrimination | HELPS |

⇒ "Load-bearing optimism" is not fundamental; it describes the *flattening* kind of change only. Prefer
**structural de-duplication** over blanket scaling. (Same lesson as `corpus-fit-flattens-eval`.)

## Concrete re-approach angles (per feature, informed by the prior failure)
- **Correction history** — prior build applied the correction at **the two RFP sites ONLY** (`evalCacheNew kept
  raw`). Strong engines apply the corrected static eval **broadly** (every margin-based decision + stand-pat).
  Correcting only the prune predicate ⇒ pruning got more conservative (nodes +11.9%) with zero move-quality gain.
  **New angle:** apply broadly, and retest AFTER de-king ships (residual bias smaller + more SYMMETRIC, so the
  uniformly-downward correction that caused nodes↑ should shrink).
- **Threat-conditioned ordering** — both re-key attempts died of **history UNDER-FILL** (any table fragmentation
  starves at our node budgets). Two untried angles: (a) Ethereal actually **SUMS 3 history sources** so
  fragmentation in one is compensated — we only tested *additive-on-top* (double-count) and *replacement*, never
  as one term of a summed ensemble; (b) **Obsidian's STATIC threat ordering term** — fixed piece-value-scaled
  bonus/malus from threat bits, **needs NO table filling ⇒ dodges the under-fill trap entirely**. Identified in our
  own docs, never built.
- **ProbCut** — retest on the fixed eval (margin units were already correctly converted: 2200 ours ≈ 189 SF).
- **Singular** — `SINGULAR_MARGIN=2` was copied RAW from SF11 (SF units, pawn=128) while ProbCut's margin WAS
  converted (~11.6×). Converted equivalent ≈ **23**. Measurable via existing `g_sing_fire`/`g_sing_gatepass`.
  Caveat: weak foundation (no real TT best-move field).
- **Pre-search removal** — the naive ablation (`ENABLE_ROOT_PRESEARCH=0`) gives **WAC 64/300 with nodes UP** and is
  documented INVALID (*"`cur_depth==1` hard-depends on pre-search `second_moves`; removal breaks ≥3 entangled
  things"*). ROOT BUG: the real search writes back **the re-sorted pre-search INPUT list**, not its own real
  2nd-level results ⇒ nothing real to reuse. **New angle:** make the real search PERSIST its real 2nd-level
  ordering (additive, nothing consumes it until tested) — that is the actual prerequisite, not a TT rewrite.
- **Pawn-hash eval cache** — NOT built (only `pawn_key` use is corrhist logging). Blocker is ENTANGLEMENT (pawn
  eval sets globals KS/imbalance/passer read); safe boundary already designed: `pawnKey → {score, passed masks,
  pawn_rank_bonuses[64]}` + re-emit globals. Eval profile PAWNS = 22.4%, est. **~10-15% NPS**.

## Measurement traps (do not re-learn these)
- **Fixed-NODE gauntlets CANNOT see a speed win** (same nodes = same result). Speed ⇒ depth@movetime / TIME-based
  games only.
- **Regression-to-mean at our budget**: CHECK_ORDER 4-seed = +8.7 / −5.7 / +3.1 / −0.3 → mean **+1.45%, within
  noise** (95% CI ≈ ±5.6%; a real +1.5% needs ~10 seeds). Levers help when the baseline seed is low and hurt when
  high. ⇒ For small effects use **SPRT**, not fixed 3-seed blocks. **Sign-consistency across seeds is stronger
  evidence than the mean** (e.g. gentle-A was +1.2/+4.8/+1.7 = all-positive, unlike CHECK_ORDER's mixed signs).
- **Fixed-depth accuracy LIES** for node-savers (LMR_EXTRA taught this) — it is not a ship gate.
- **Corpus win%-MSE fits are bench-NEGATIVE 4/4 times** — always bench-guard (`fit_bench_guarded.py`).

## Dependency-ordered slate (post de-king)
1. **Pawn-hash speed** — the only structurally regression-immune lane; measure in TIME.
2. **Prune-push retest (the PAIR)** — where the roadmap says the Elo actually lives, now that eval is better.
3. **Corrhist v2** — applied broadly, post-de-king.
4. **Static threat ordering** (under-fill-proof) / **summed-ensemble history**.
5. **Persist real 2nd-level ordering** → then a fair pre-search test.
6. **ProbCut retest**, **singular margin conversion**.
