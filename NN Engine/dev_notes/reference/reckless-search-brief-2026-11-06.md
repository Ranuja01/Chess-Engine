# Reckless: engine brief for the search arc and the NNUE stage (2026-11-06)

Reckless is a Rust NNUE engine, rated second on CCRL by its own README. This brief records what it is, what its source shows and what may be worth testing in search v2.

Everything below was read from the source at commit `aec5cb2` (2026-10-28), cloned to `E:\chess_data\engines\reckless_src`. It has not been built or run. Line numbers refer to that commit.

**None of this has been checked line by line against our code.** The "our record" notes come from memory and header defaults, and most of those verdicts are from `EVAL_ARM=0`. Run `record-check` and `knob-audit` before building anything here.

## Facts

| | |
|---|---|
| Source | github.com/codedeliveryservice/Reckless |
| Nets | github.com/codedeliveryservice/RecklessNetworks; this commit's build uses `v60-7f587dfb` |
| Licence | AGPL-3.0 (`LICENSE` at this commit; some older pages say MIT). Ideas only, never port its code |
| Tools | OpenBench with fastchess for testing, Bullet as the NNUE trainer, an `spsa` build feature for tuning |
| Team | Arseniy Surkov (lead), Shahin M. Shahin (search, a prolific Stockfish and Fishtest contributor), Styx (NNUE, testing, hardware), plus contributors and donated compute |
| Size | 10,994 lines of Rust in `src/`; `search.rs` is 1,490 lines, `movepick.rs` 216 |

**Results:** in TCEC Season 29 it was second in the Premier Division (Stockfish 35, Reckless 34, Lc0 31) and lost the Superfinal to Stockfish 59–41 (+41 =36 −23).

**Rating history, from its README (self-reported, not checked on the lists themselves):**

| Version | Date | SPCC | CCRL Blitz | CCRL 40/15 |
|---|---|---|---|---|
| v0.1.0 | 2023-05 | 2005 | – | – |
| v0.3.0 | 2023-11 | 2616 | 2617 | – |
| v0.5.0 | 2024-02 | 3240 | 3211 | – |
| v0.7.0 | 2024-08 | 3500 | 3422 | – |
| v0.8.0 | 2025-08 | 3777 | 3765 (#2) | 3609 (#3) |
| v0.9.0 | 2026-03 | 3833 (#2) | 3767 (#2) | 3612 (#2) |

**It did not come from nowhere.** It took nearly three years. The author says it began as a solo learning project, and that the turning point was working on modern Stockfish search during the FIDE x Kaggle event, where the team formed. After that "we quickly went from fringes of top 50 to top 10 and beyond."

**Not an independent vote.** The README credits Stockfish, PlentyChess, Ethereal and Berserk, and a co-author is a Stockfish contributor. The search reads as modern Stockfish lineage. A March 2026 forum post called it a clone; the replies rejected that (it picks different moves, and shared ideas are normal), and I found no evidence for it. Under [[adopt-reference-methods-only-if-universally-superior]], count it as correlated with Stockfish.

## Why it is useful to us

- **A short, readable modern search.** `search<NODE>` with node-type generics and a per-ply stack is the shape `SEARCH_ARM=1` wants (owner-cpp-ideas F18), in about a third of the lines one might expect.
- **A staged picker in 216 lines**, the traditional alternative to our move cache (memory `search-v2-traditional-beside-ours-with-switches`).
- **A small team's path** on donated compute rather than fishtest's scale.

## What does not transfer

Margins are SPSA-tuned against an NNUE, on a pawn of 109 (N 403, B 435, R 679, Q 1242) against our 1000, so about ×9.2 at the pawn. Only shapes and signals transfer, never magnitudes ([[matching-a-reference-term-is-not-being-right]]). These parts are co-designed, so test stacks first ([[search-changes-are-antagonistic-not-additive]]).

## Candidates for search v2

**1. Static terms in quiet ordering** (`movepick.rs:145-215`). The clearest fit, because it needs no eval and no filled history.

Beside its history terms, each quiet gets:
- a bonus for escaping an attack by a lesser piece, per piece type;
- a penalty for landing on a square attacked by a lesser piece;
- a bonus for reaching a checking square;
- a bonus for "offense" squares: safe squares that attack a more valuable enemy piece, plus safe pawn attacks and rook moves onto the enemy king's file;
- a penalty for moving a pawn next to our own king while it is on its home rank.

Our record already pointed this way: threat ordering died of history under-fill, so try a static term (`ordering-and-reduction-idea-backlog`). Gen 6.3's `THREAT_ORDER_W` covers part of it. Offense squares, checking squares and wall pawns look new; check first.

**2. Using |correction| as an uncertainty signal.** The size of the correction loosens pruning and reduces less wherever the eval has lately been unreliable: RFP (`search.rs:556`), futility (`:810`), LMR (`:865`, `:943`), singular margins (`:732`), qsearch SEE (`:1340`). Our correction-history lane closed on 08-24 because keyed tables did not *correct* the eval better than a global slot (`corrhist-signal-is-position-local-not-structural`). This is a different use of the table, and that verdict was on v1. It also fits [[eval-accuracy-payoff-is-pruning]]: prune harder only where the eval has earned it. Its keys are pawn, non-pawn per colour, and continuation at 2 and 4 plies, bucketed by the fifty-move clock (`:1403`).

**3. Reduction signals from the line itself.**
- `last_critical_ply` (`:882`): reduce less the longer the line has gone since it last took a reduced or non-first move, up to 8 plies.
- `laterality` (`:887`, updated at `:1472`): a running sum of log(move number) along the line; non-PV lines far from first choices get reduced more.
- `cutoff_count` of the child (`:903`).
- `tt_move_score − singular_score` (`:908`): the gap between how the TT move scored and what the others could reach.
- A small deterministic dither on the reduction (`:917`), which spreads fractional reductions across the depth step.

The first two are the least familiar to me, and I have not checked them against other engines.

**4. Softening fail-high returns with `lerp`.** RFP (`:563`), ProbCut (`:702`), multi-cut (`:744`), stand-pat (`:1302`) and qsearch cutoffs (`:1390`) return a blend of score and beta, not the raw score. One line each, and it limits how far an overshoot propagates. Relevant now that `RFP_MARGIN` is 600.

**5. The fifty-move clock in the TT key** (`board.rs:75-83`). The hash XORs in a bucket of the clock (one per 8 plies past 8), so entries do not alias across very different rule-50 states; there is also a guard against TT cutoffs at clock 90 or more (`search.rs:404`). `correct_eval` also decays the eval by `(200 − clock) / 200`. Compare with `FIX_HALFMOVE` and `FIX_MATE_50`.

**6. Null move details** (`:567-633`). Cut nodes only; skipped when a singular search is likely, and when the TT move is a lower-bound capture of a knight or better; the null window drops to `tt_score` when that is a lower bound below beta; verification only at depth 16 or more.

**7. Singular details** (`:711-756`). Single, double and triple extensions with margins that shift with PV status, quietness and |correction|; multi-cut; dropping the TT move when the exclusion search beat it; a −3 negative extension; a low-depth extension on cut nodes whose eval is below alpha. Our singular is banked as neutral on v1, with a known TT dependency (`singular-banked`, `node-tt-dependency-starved-singular-and-ttmove`).

**8. History handling.**
- Quiet history keyed by whether from and to are attacked (`history.rs:83`).
- Continuation history at 1, 2, 4 and 6 plies with separate weights (`movepick.rs:205-208`). Ours has `ENABLE_CONT_HIST_2PLY` off after a fixed-depth loss at equal weight, which is not a games verdict.
- Pawn-keyed history.
- Malus that shrinks with position in the tried list (`search.rs:1115-1125`).
- A bonus for the opponent's previous move when this node fails low (`:1151-1181`), and one from the eval swing between plies (`:492-503`).

**9. Pruning and root details, smaller items.** Futility on bad captures (`:822`), a history term inside the LMP limit (`:799`), aspiration delta that narrows with score and PV stability and widens with score squared (`:98-115`), one ply less after a fail high (`:147-154`), and reductions even on the full-depth re-search (`:939-993`).

**10. Time management, for later** (`:245-263`). Five factors multiply together: the best move's share of nodes, score trend, PV stability, eval stability, and a log of best-move changes. Compare with `active/TIME-MANAGER-RESEARCH-2026-10-25.md`.

## For the NNUE stage

- **Architecture at this commit** (`nnue.rs`): 10 king buckets with horizontal mirroring, piece inputs **plus threat inputs** in a separate accumulator, then 768 → 16 → 32 → 1 with 8 output buckets. Threat inputs are relevant to us: our heat map and attack layer already compute this kind of information. See the `Full_Threats` section of the Stockfish NNUE doc in memory `nnue-future-references`.
- **Public net history stops at v10** (2025-03-30), while the build is at v60, so the later nets are undocumented in that changelog. Through v10 (Elo at 8+0.08, each from its own SPRT, not comparable across rows):

| Net | Change | Elo |
|---|---|---|
| v2 | hidden 128 → 256 | +81 |
| v3 | → 384 | +21 |
| v5 | regenerated data with the latest engine | +14 (at 20 s) |
| v7 | → 512 | +24 |
| v8 | final WDL 0.5 → 0.6 | +5 |
| v9 | king mirroring | +11 |

- **Data provenance is not public** in anything I read. The engine repo has no data generator, the changelog's only data note is v5's regeneration, and the author's interview does not cover it. Worth asking them directly if we get there.
- **Measurement:** those SPRTs ran up to about 18,000 games with error bars near ±3 Elo, far beyond what one box can do routinely.

## As an opponent

It could join the gauntlet at a node limit, like SF18 @ 1,100. It needs Rust 1.88 or later, and clang for Syzygy unless built with `make no-syzygy`; the build downloads its net. Not needed today.

## Sources

- [Reckless repository](https://github.com/codedeliveryservice/Reckless) · [network repository and changelog](https://github.com/codedeliveryservice/RecklessNetworks)
- [Interview with Arseniy Surkov, Chessdom](https://www.chessdom.com/interview-with-arseniy-surkov-author-of-reckless/)
- [Reckless reaches the Superfinal, Chessdom](https://www.chessdom.com/reckless-breaks-a-duopoly-in-a-historic-computer-chess-shake-up/) · [Stockfish wins Season 29, Chessdom](https://www.chessdom.com/stockfish-claims-its-19th-tcec-title-defeating-reckless-in-the-superfinal/)
- [The clone claim and its replies, TalkChess](https://www.talkchess.com/viewtopic.php?p=990858)
