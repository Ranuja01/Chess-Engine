# External-play gaps — worst-case-Elo hole-plugging track

A track distinct from the self-play SPRT campaign. These are concrete failures observed in real games
vs the chess.com `tal-BOT` (2705) where the engine had a winning/equal position and lost. They are
**self-play-invisible** (the base won't exploit them, the ship won't enter them, and they're rare per
game), so the PACE→SPRT loop can't surface them.

**Preset note (important for diagnosis):** these games were played on the **STANDARD** preset
(~6s/move, iterative depth to the ~12 cap), NOT LIGHTNING. The tactics in both gaps are visible within
the depth STANDARD *and* LIGHTNING reach (~10–15 ply), so the engine *should* see them at either TC.
That points the finger primarily at **EVAL** (the danger / resulting position is mis-scored), though
**SEARCH may be complicit** (the saving/winning line is being pruned away before it's evaluated).
Disentangling eval-vs-pruning is the dedicated session's first job. Ideal outcome: a fix that holds on
**both** STANDARD and LIGHTNING.

## Validation model (user-set, important)
- Anchor each gap on the **concrete position** (replay the FEN through the engine + analysis tools to see
  *why* it chose wrong — eval term attribution / search horizon / SEE).
- **Provable fix on the position + no bench regression (WAC/STS byte-deterministic) ⇒ ship**, even when
  self-play Elo is flat. The point is to raise **worst-case Elo**, which lifts win-rate vs *external*
  opponents even though it barely moves a shared-eval A/B.
- Bonus criterion: a fix that solves both the *failure* (we miss the danger) AND the *exploitation* (we
  see it when colors are swapped) is the strongest — ship it if benches are clean.

## Theme #1 — opponent passed-pawn danger systematically undervalued
The recurring root. The latent danger of an enemy passer (proximity-to-promote × stoppability, and in
pawn races) is weighted too low. Connects to the long-standing **true-passed-mask refinement** backlog
(the ADV=5 rank-proxy over-fired and was reverted) and [[collapse-eval-overoptimism]] /
[[dynamic-conditional-eval]] / [[material-edge-overvaluation]].

### Game B — 2026-06-22, Benoni (A56), engine = White, 0-1
Final: `8/8/3r1Pp1/8/4K2k/8/8/8 w - - 0 58`. The engine walked into a lost R-vs-passers ending,
misjudging the pawn race / defensive difficulty of the opponent's passer.
```
1. d4 Nf6 2. c4 c5 3. d5 g6 4. Nc3 Bg7 5. e4 d6 6. Nf3 O-O 7. h3 a6 8. a4 e6 9. Bd3 exd5 10. cxd5 Nbd7
11. O-O Re8 12. Bf4 Qc7 13. Re1 Nh5 14. Be3 b6 15. Ng5 Ne5 16. Be2 h6 17. Nf3 Nxf3+ 18. Bxf3 Nf6 19. Qd2 Kh7
20. Bf4 Nd7 21. Be2 Bb7 22. h4 Nf6 23. Bc4 Nh5 24. Be3 Qe7 25. g3 Nf6 26. f3 Nh5 27. g4 Ng3 28. Bf2 b5
29. Bb3 Be5 30. Kg2 c4 31. Bxg3 cxb3 32. axb5 axb5 33. Rxa8 Rxa8 34. Nxb5 Rc8 35. Nd4 h5 36. gxh5 gxh5
37. Nf5 Qf6 38. Bxe5 dxe5 39. Kh1 Rc2 40. Qd1 Bc8 41. f4 Bxf5 42. Qxh5+ Kg7 43. exf5 e4 44. Qg5+ Qxg5
45. fxg5 Rxb2 46. Rxe4 Rc2 47. f6+ Kg6 48. Re8 b2 49. Rg8+ Kf5 50. Rb8 Rc1+ 51. Kg2 b1=Q 52. Rxb1 Rxb1
53. d6 Rd1 54. Kf3 Rxd6 55. Ke3 Kg4 56. Ke4 Kxh4 57. g6 fxg6 0-1
```

### Game A — 2026-06-16, French Winawer (C18), engine = White, 0-1
Final: `8/6pk/4p1pp/3pPR2/2pP1q2/2P2P2/2P1K1r1/4Q3 w - - 0 48`. Black's queenside passer marched
a4→a3→a2 while the engine shuffled; the promotion threat eventually forced material loss.
```
1. e4 e6 2. d4 d5 3. Nc3 Bb4 4. e5 c5 5. a3 Bxc3+ 6. bxc3 Ne7 7. Qg4 O-O 8. Bd3 c4 9. Bh6 Ng6 10. Bxg6 fxg6
11. Be3 Nc6 12. h4 Rf5 13. Ne2 Qa5 14. O-O b5 15. Rfb1 Rb8 16. Rb2 Bd7 17. f3 Qb6 18. Bf2 Qa5 19. Rbb1 Rbf8
20. Be3 Qb6 21. Bf2 Qa6 22. Re1 Ne7 23. Be3 h6 24. Reb1 Qa4 25. Ra2 Qa5 26. Bd2 Qc7 27. Nf4 Kh7 28. a4 bxa4
29. Be3 Rb8 30. Rd1 Qa5 31. Bd2 Qb6 32. Kf1 Qd8 33. Raa1 Rb2 34. Rdc1 Qa5 35. Ke1 a3 36. Kd1 a2 37. Ke1 Qb6
38. Kf1 Rb1 39. Rxa2 Rxc1+ 40. Bxc1 Qb1 41. Qg3 Qxc1+ 42. Qe1 Qxf4 43. Rxa7 Rh5 44. Rxd7 Rxh4 45. Ke2 Nf5
46. Rf7 Rh2 47. Rxf5 Rxg2+ 0-1
```

## Theme #2 — winning-capture / trade miscalculation (Game B, move 28...b5 → 29.Bb3)
Bishop on c4; `28...b5` attacks it. The engine retreated `29.Bb3` instead of `29.axb5 axb5 30.Bxb5`,
which wins a pawn, hits the e8-rook, and keeps the g3-knight trap live — clearly winning, a sub-1300
sees it. The engine valued Bb3 higher: candidate causes = a horizon tactical ghost, SEE/capture-ordering
miscalc, or the resulting-position eval undervaluing up-a-pawn-with-initiative. **First step: replay the
FEN after 28...b5 through the engine, read the eval breakdown + PV for axb5 vs Bb3.** Distinct from the
passer theme; possibly relates to [[material-edge-overvaluation]] (which says material edge is *over*-valued,
so dodging a winning pawn is surprising → likely search/SEE, not the material scalar).

## Future item — deep-LMR + forcing-line guards (from the Gap-T diagnosis)
Architectural finding (grep-confirmed): move-loop LMR/LMP fire ONLY at the root (`alpha_beta`) and the
preliminary pass (`pre_minimizer`); the deep recursive `minimizer`/`maximizer` apply no move-loop LMR
(only null-move). So the reduction surface is tiny — likely why past LMR/ordering experiments were
marginal. **Opportunity:** extending LMR (+LMP) into the deep search is a large missing EBF/node lever.
**Safety (build proactively, per user):** thread a `capture_run` counter down the recursion (consecutive
captures / within a window) and make the reduction `f(capture_run, cur_depth, remaining_depth)` —
taper/kill LMR in forcing capture sequences; compose with the existing passer/check exemptions.
`last_move_was_capture` is already threaded → small extension. Build the guard even if a strong settings
set ships first, so future LMR/LMP increases are safe by construction. Separate project, after the Gap-T/P
fixes. (Memory: deep-lmr-opportunity.)

## Gap-T RESOLVED (2026-06-22) — VERIFY_MARGIN 6000 -> 16000 (existing knob, no new code)
Mechanism: history-LMR over-reduces a (only mildly-reduced) quiet move whose scout fails low by > the flat
VERIFY_MARGIN=6000, so VERIFY skips it and the buried score corrupts the axb5-vs-bishop choice. The fix is
NOT reduction-coupling (the `VERIFY_REDUCTION_K` candidate I built was REJECTED — the buried move is only
mildly reduced, so coupling-to-reduction misses it; STRIP that knob). A plain wider flat margin catches it.
**VERIFY_MARGIN=16000 results (same build, USE_OPENING_BOOK=0):** benoni-29 LIGHTNING f2g3->a4b5 (fixed);
WAC 258/300 held (nodes 97.5M->114M, +17% fixed-depth); STS 1426->1566 (47.5->52.2%, +140 — recovers the
STS that SEE_EXTEND_MARGIN=300 traded away); wac_timed_depth 242/12.447->242/12.445 (neutral);
sts_timed_depth 1411(47.0%)/d11.207 -> 1486(49.5%)/d11.153 (+75 STS at equal time, depth -0.05 = negligible).
So it's a net positional WIN at equal time, not a fixed-depth mirage; +17% fixed-depth nodes are free in play.
CAVEAT: overrides a play-tuned default (VM=6000 was self-play-validated) and STS != Elo -> the DEFAULT CHANGE
needs an SPRT (queue for overnight bundle). Note SEE=300 + VM=16000 are complementary (SEE cut nodes/STS,
VM buys the STS back). Threshold: VM=10000 did NOT fix benoni-29 d9; 16000 does (full STS recovery).

## Gap-P RESULT (2026-06-22/23) — conditional passer scorer, gated, byte-id
Knobs built (all default-off/100 = byte-id; corpus `diagnostics/suites/passers.csv`, 405 SF-labeled via `gen_passer_corpus.py`):
- **P1 `PASSER_ENEMY_CREDIT_PCT`** (default 100): scales ONLY the enemy-defender term in `boost_pieces_for_supporting_passed_pawns` (the wrong-signed blockade/path-control credit); own-support − self-block (`black_adjustment`) untouched (verified separable via isolated own-block vs enemy-block FENs). =0 fully removes the wrong-sign.
- **C1 `ENABLE_PASSER_BLOCKADE_QUALITY` + `PASSER_CONTEST_PCT`(30)** in `getPPIncrement`: only a SECURE blockade (enemy minor on the stop square) gets full `PP_BLOCKADE_PEN`; a rook/queen merely contesting the file ahead gets `PASSER_CONTEST_PCT`% — un-zeroes a rook-contested advancing passer (the French case). Also recovers the WAC that P1 alone costs.
- **C2 `ENABLE_PASSER_KRACE_MG` + `PASSER_KRACE_MG_PCT`(100) + `PASSER_KRACE_MAG`(100)**: lifts the `advanced_endgame_eval` king-race (extracted to `passer_realizability_delta()`, de-duped vs the AE copy) into all phases, phase-ramped, + a magnitude knob (the king-race was calibrated too weak — ~0.4 for a queening passer).

Deterministic isolation (baseline WAC 258 / STS 1426): **P1 only** = 250(−8)/1563(+137); **P1+C1** = 256(−2)/1473(+47) ⇒ **CLEAN SHIP TIER** (F36 +0.33→−0.05 sign fixed, F33 +0.93→+0.42); **+C2 mag150..300** = 250..253 / +127..+102, F36→−0.44 ⇒ stronger but WAC-costly = SPRT-gated. **benoni-47 endgame only partly helped** (SF *static* +0.13 there = NNUE/search-realization ceiling). LIGHTNING move-match too noisy to tune on (±4) → deterministic WAC/STS/gap-FEN-eval.

## CORRECTION + capture-chain LMR guard (2026-06-23)
**Deep LMR ALREADY EXISTS** (misread earlier): `minimizer`/`maximizer` delegate to `get_score_for_minimizer`/`maximizer` which hold the full LMR. So the EBF lever = forcing-line GUARDS that let LMR be pushed harder, not "add deep LMR". **Built `ENABLE_LMR_CAPCHAIN`** (default off; threads `last_move_was_capture` into the helpers + ANDs into `base_lmr` → skip LMR on a quiet move when the parent move was a capture). **v1 is TOO BROAD: benoni-29 → a4b5 (fixed) but ~17× nodes (7.85M @d9) → depth collapses at equal time.** Needs v2 (int `capture_run` run-length: only in real recapture chains, OR reduce-LESS not skip, gated on remaining-depth). See [[deep-lmr-opportunity]].

## Status (2026-06-23)
E0 cleanup DONE (dead `VERIFY_REDUCTION_K` + passer debug prints stripped; byte-id 258/97,507,126 confirmed). Gap-T (`VERIFY_MARGIN=16000`) + Gap-P (`P1+C1`) = bench-validated clean ship tiers, gated, NOT yet default — **SPRT-gate the bundle overnight**. Capture-chain guard v1 too broad (v2 next). All knobs default-off; commits held for sign-off. Tools: `fen_vs_sf.py`, `eval_breakdown.py`, `movematch`, `wac`/`sts` (+timed), `gen_passer_corpus.py`.
