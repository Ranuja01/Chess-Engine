# Strength Backlog — search / eval improvement candidates

Living tracking table for the pre-NNUE strength track. Source-of-truth for *queued / in-flight* work;
**shipped** items graduate to `OPTIMIZATION_LOG.md` (STRENGTH TRACK), strategic context lives in
`DEPTH_AND_POSITIONAL_NOTES.md`. Derived from the deep-research roadmap (memory
`pre-nnue-strength-roadmap`) + the tournament/eval diagnostics (memory `tournament-color-bias-diagnostics`).
Last updated: 2026-06-08.

Status tags: `idea` → `discussed` → `designed` → `testing` → `shipped` / `parked`.
Discipline: env-toggle default-off (off = byte-identical) → STS300 gate + WAC over-correction guard →
overnight adjudicated self-play tournament as judge. Constants are SHAPES — SPSA-tune, don't hand-pick.

| Item | Reuses | Status | Risk | Notes |
| --- | --- | --- | --- | --- |
| **Color-symmetry fix** | `placement_and_piece_eval`, `ev_breakdown` probe, `eval_symmetry.py` | **shipped** (ae8a462/b9c27bc) | med | 12 non-mirrored items fixed; corpus mirror residual 0.000; WAC d10 re-baselined **263,422,651**, STS300 50.1%. |
| **History-scaled LMR (reduce-less)** | composite ordering score | **dropped** (harmful) | — | Standalone reduce-less (CAP≥1) cost ~+10% nodes, STS flat-to-down; the reduce-MORE arm shipped instead. Revisit only with continuation history. |
| **History-scaled LMR (reduce-more)** | `reduced_search_depth`, `lmr_hist_tier` | **shipped** (default-on 2026-06-08) | med | `ENABLE_HISTORY_LMR=1 CAP=0 MORE_CAP=1` — reduce never-cut (tier-0) quiets 1 ply more. STS300 50.1→51.7%, −1.3% nodes @d10; self-play +25.5±65 (not sig, no regression). Knob retained for override. |
| **Continuation history** | `counterMoveHeuristics` (1-ply), `history_lmr_delta` | **CH-A shipped** (default-on `046a17f`) | med | **CH-A:** continuation-aware LMR — a tier-0 quiet with a strong 1-ply `counterMoveHeuristics` score is NOT reduced-more (cancel −1, back to base). `CONT_HIST_LMR_THRESH=2000` (d10 sweep optimum: **−6.2% nodes**, WAC 262, STS neutral; 1000 over-fires, rebounds to −2.7%). **Next (deferred):** full 2-ply table (thread `previousMove2` through 21 minimizer/maximizer sites — BoardState stores no move) + compact `[piece][to]` reindex (134MB→~2.4MB, de-thrash L3). |
| **SEE pruning (main search)** | SEE in `move_gen.h` | discussed | med | Skip quiets with SEE < −margin·depth (quadratic) + losing captures (linear margin). Prior `SEE_EXTEND_MARGIN` attempt cut a sac-mate → guard hard on WAC. |
| **Late move pruning (LMP)** | move index in search loop | discussed | med | After N quiets at low depth, skip the rest. Risk = over-pruning quiet wins (our weakness) → conservative + gate on improving. |
| **Improving heuristic** | per-node static eval (+ small per-ply eval stack) | discussed | low | `staticEval > staticEval[2-ply]` → modulate LMR/futility (prune less when not improving... safe direction). Cheap; feeds the LMR work. |
| **SEE-ignores-pins fix** | SEE / capture eval (`move_gen.h`), `get_relevant_pin` | idea | med | From game review (`jun6_standard/game_006` move-55 double-blunder): a pinned-but-winnable piece reads as defended → easy 2-ply material wins missed. Localize with `line_probe`. |
| **Incremental eval** | board make/unmake, eval accumulator | parked (E3) | high | Update PST/material incrementally + pawn-hash; debug recompute-assert (WAC harness). Biggest nps lever. Major refactor — defer. |
| **Lazy eval** | eval term structure, in-window bounds | parked | med | Early-exit with material+margin before expensive terms. Pairs with incremental eval. |
| **Staged / partial move generation** | `generateLegalMovesReordered` cache | discussed | med | Generate captures→killers→quiets→bad-captures lazily; stop on cutoff. Promote-to-front already gives ordering win → this is an *nps* win. |
| **Endgame dead-draw detection** | `is_practically_drawn` | **shipped** (default-on 2026-06-08) | low | Extended with R+N-vs-R / KRKN / KRKB (pawnless, exact-count, returns 0 before `piece_value_boost` runs). Found via `pattern_diag` as the #1 endgame over-read (R+N-vs-R −684→0; `+1N` endgame bias 461→142). Leaves the boost's trade-down logic intact everywhere it's a real win. |
| **Graded drawishness scale (endgame scale factor)** | `is_practically_drawn`, material/pawn signals, `is_white_square` | **shipped** (default-on `046a17f`) | med | `endgame_convertibility_scale()`: continuous [0.25,1] damp of an *unconvertible* endgame lead — bare-minor-up / opposite-bishops, pulled to 1 by an advanced winning passer; mates (edge>minor)→s=1. Scales `total` + the `piece_value_boost` term at the boost block. **Validated by FEN spot-checks** (mates/wins byte-identical off→on, OCB drawish −0.64→−0.34 toward SF); the `pattern_diag` aggregate is the wrong lens (the scale correctly leaves benign saturated wins at s=1). Constants are SPSA-shapes for later tuning. |
| **Mate-drive material scale** | `advanced_endgame_eval` | parked-knob (`ENABLE_MATE_DRIVE_SCALE`, default-off) | med | Scale the king-drive by winner's material margin (defender-queen→0). Suite-neutral, no definitive self-play; its R+N-vs-R target now handled by is_practically_drawn. Preserved as a knob; redo with a defender-bare gate inside the graded scale above. |
| **`pattern_diag.py` (miseval miner)** | `deep_diag`/`term_diag` helpers | **shipped** (tooling 2026-06-08) | — | Offline: material-config offender clusters / signed-bias-by-material-delta (lex-oriented, fixable-vs-scatter) / transformation-leak scan. Trust-gated (NNUE-static valid only where it tracks SF search). The systematic replacement for eyeballing top-20 offender lists. |
| **Harness concurrency + timed + UHO book** | `tournament.py`/`sprt.py`, `gen_openings.py` | **shipped** (0e02285) | — | `--concurrency N` (measured **6.5×** on 8c/16t; per-thread SF arbiter, main-thread bookkeeping), `--max-minutes` greedy timed mode, UHO unbalanced-book generator (`openings_uho.txt`). The throughput unlock for sound SPRT. |
| **SPSA + SPRT tuning loop** | self-play harness (`tournament.py`, `sprt.py`) | discussed (**throughput prereq SHIPPED**) | high | THE multiplier — SPRT early-stop A/B + SPSA param optimization. Data-hungry; the prerequisite **game concurrency (6.5×) + timed mode + UHO book** now shipped (0e02285), so ~1000-game runs are feasible overnight. |

## How to use
1. Promote `idea`→`discussed`→`designed` as we scope; one `designed` item enters `testing` at a time.
2. A `testing` item gets env-toggled (default-off), runs STS300+WAC, then an overnight adjudicated tournament.
3. `shipped` → move the row's result to `OPTIMIZATION_LOG.md` STRENGTH TRACK; `parked` → note blocker + date here.
