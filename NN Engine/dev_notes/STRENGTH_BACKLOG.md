# Strength Backlog — search / eval improvement candidates

Living tracking table for the pre-NNUE strength track. Source-of-truth for *queued / in-flight* work;
**shipped** items graduate to `OPTIMIZATION_LOG.md` (STRENGTH TRACK), strategic context lives in
`DEPTH_AND_POSITIONAL_NOTES.md`. Derived from the deep-research roadmap (memory
`pre-nnue-strength-roadmap`) + the tournament/eval diagnostics (memory `tournament-color-bias-diagnostics`).
Last updated: 2026-06-06.

Status tags: `idea` → `discussed` → `designed` → `testing` → `shipped` / `parked`.
Discipline: env-toggle default-off (off = byte-identical) → STS300 gate + WAC over-correction guard →
overnight adjudicated self-play tournament as judge. Constants are SHAPES — SPSA-tune, don't hand-pick.

| Item | Reuses | Status | Risk | Notes |
| --- | --- | --- | --- | --- |
| **Color-symmetry fix** | `placement_and_piece_eval`, `ev_breakdown` probe, `eval_symmetry.py` | **designed** (ACTIVE) | med | Eval not mirror-symmetric (startpos −0.5 Black; `pieces` +0.17p residual). Placement tables / attacking-layer usage / pin term EXONERATED by reading → asymmetry is in a per-piece `evaluate_*` branch or attack-map. Localize via per-piece-type mirror probe (rebuild), fix (White canonical), prove `eval(P)=−eval(mirror(P))`, **re-baseline WAC/STS**. Breaks byte-identity. |
| **History-scaled LMR (reduce-less)** | `reduced_search_depth`, composite ordering score (`historyHeuristics`+killer+counter+`moveFrequency`) | **designed** (next) | med | Only DECREASE reduction for high-ordering-score quiets; never increase (reduce-more arm deferred). Keyed off the *full* ordering composite, not raw history. Env `ENABLE_HISTORY_LMR` default-off. Should also cut VERIFY_MARGIN re-search load. Risk = the prior over-pruning regression — hence reduce-less only first. |
| **History-scaled LMR (reduce-more)** | same as above | parked | high | Reduce MORE for low-score quiets (EBF/speed). The regression-risky arm; only after reduce-less stabilizes, separate env + test cycle. |
| **Continuation history** | `counterMoves` infra, history update sites | discussed | med | Extend 1-ply counter-move into multi-ply (prev→cur) scored table; use for ordering AND LMR scaling. Biggest modern EBF lever after plain history. New table state. |
| **SEE pruning (main search)** | SEE in `move_gen.h` | discussed | med | Skip quiets with SEE < −margin·depth (quadratic) + losing captures (linear margin). Prior `SEE_EXTEND_MARGIN` attempt cut a sac-mate → guard hard on WAC. |
| **Late move pruning (LMP)** | move index in search loop | discussed | med | After N quiets at low depth, skip the rest. Risk = over-pruning quiet wins (our weakness) → conservative + gate on improving. |
| **Improving heuristic** | per-node static eval (+ small per-ply eval stack) | discussed | low | `staticEval > staticEval[2-ply]` → modulate LMR/futility (prune less when not improving... safe direction). Cheap; feeds the LMR work. |
| **SEE-ignores-pins fix** | SEE / capture eval (`move_gen.h`), `get_relevant_pin` | idea | med | From game review (`jun6_standard/game_006` move-55 double-blunder): a pinned-but-winnable piece reads as defended → easy 2-ply material wins missed. Localize with `line_probe`. |
| **Incremental eval** | board make/unmake, eval accumulator | parked (E3) | high | Update PST/material incrementally + pawn-hash; debug recompute-assert (WAC harness). Biggest nps lever. Major refactor — defer. |
| **Lazy eval** | eval term structure, in-window bounds | parked | med | Early-exit with material+margin before expensive terms. Pairs with incremental eval. |
| **Staged / partial move generation** | `generateLegalMovesReordered` cache | discussed | med | Generate captures→killers→quiets→bad-captures lazily; stop on cutoff. Promote-to-front already gives ordering win → this is an *nps* win. |
| **SPSA + SPRT tuning loop** | self-play harness (`tournament.py`) | discussed | high | THE multiplier — turns every constant above into a measured Elo gain (Fishtest-lite: SPRT early-stop A/B + SPSA param optimization). Data-hungry. |

## How to use
1. Promote `idea`→`discussed`→`designed` as we scope; one `designed` item enters `testing` at a time.
2. A `testing` item gets env-toggled (default-off), runs STS300+WAC, then an overnight adjudicated tournament.
3. `shipped` → move the row's result to `OPTIMIZATION_LOG.md` STRENGTH TRACK; `parked` → note blocker + date here.
