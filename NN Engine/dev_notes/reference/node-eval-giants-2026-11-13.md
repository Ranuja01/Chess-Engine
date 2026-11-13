# Node-entry static eval, improving and correction history in the reference engines (2026-11-13)

Created 2026-11-13 for search v2 stage S4 (`dev_notes/active/SEARCH-V2-PLAN-2026-11-10.md` §3). A read of source only;
nothing was run. Our side and the options are in `dev_notes/active/S4-NODE-EVAL-AUDIT-2026-11-13.md`.

**Not repeated here** (read there): the per-version pruning formulas SF11 to SF18 and Ethereal
(`sf-pruning-schedules-comparison.md` §1.3-1.13), the SF evolution sheets (`sf-evolution-fact-sheets-2026-07-05.md`),
Reckless's search in general (`reckless-search-brief-2026-11-06.md`), the root (`current/ROOT_SEARCH_MODEL.md`).
This file adds: SF19's exact constants (newer than those docs), the plumbing of the static eval (where it is computed,
stored, corrected), improving and its fallbacks, and the correction-history tables.

Sources (line numbers as read on 2026-11-13):

| tag | path |
|---|---|
| SF19 | `Chess Engine/stockfish_19_linux/src/` (`search.cpp`, `history.h`, `evaluate.cpp`) |
| SF11 | `Chess Engine/stockfish_11/stockfish-11-win/src/search.cpp` (the path `stockfish_11/src/` does not exist) |
| RK (Reckless) | `E:\chess_data\engines\reckless_src\src\` (`search.rs`, `history.rs`, `evaluation.rs`) |
| TV (Triumviratus) | `E:\chess_data\engines\triumviratus_src\` (`source/search/*.inc`, `docs/project/NOVELTIES.md`) |

Units: SF19 / TV internal units (a pawn is about 200-210 of them in SF19's search; not centipawns). Ours are millipawns.
Never transplant a constant (memory `convert-reference-constants-by-positional-scale-not-by-the-pawn`).

---

## 1. Stockfish 19

### 1.1 Where the static eval comes from and where it lives

`Search::Worker::search`, step 5 (`search.cpp:825-855`). Four cases, in this order:

| case | `ss->staticEval` | `eval` (the value pruning reads) | line |
|---|---|---|---|
| in check | copy of `(ss-2)->staticEval` | same; all early pruning is skipped (`goto moves_loop`, :975) | :829-830 |
| excluded move (singular probe) | unchanged, the value the node set before the probe | same | :831-832 |
| TT hit | `to_corrected_static_eval(ttData.eval or evaluate(pos), correctionValue)` | the TT value instead, when its bound points the right way: `ttData.bound & (ttData.value > eval ? BOUND_LOWER : BOUND_UPPER)` | :833-846 |
| TT miss | `to_corrected_static_eval(evaluate(pos), correctionValue)` | same | :847-855 |

- **Two values per node.** `ss->staticEval` is always the corrected static eval. `eval` is that, or the TT's search
  value when it is a tighter bound. Razoring and reverse futility read `eval`; the null-move gate, futility of moves,
  ProbCut's threshold, improving and the correction update read `ss->staticEval`.
- **Stored in the stack** (`ss->staticEval`), at every node, PV included. The sentinel frames before the root hold
  `VALUE_NONE` (:297).
- **Stored in the TT uncorrected** (`unadjustedStaticEval`): on a miss an eval-only entry is written at once
  (`VALUE_NONE`, `BOUND_NONE`, `DEPTH_UNSEARCHED`, :853-854), and every later store carries it (:1090, :1622-1627,
  :1752, :1878). The correction is applied fresh at each read, so a drifting correction is never frozen into a table.
- **Clamp**: `to_corrected_static_eval(v, cv) = clamp(v + cv / 131072, VALUE_TB_LOSS_IN_MAX_PLY + 1, VALUE_TB_WIN_IN_MAX_PLY - 1)` (:105-107).
- **50-move scaling** is inside `evaluate()`: `v -= v * pos.rule50_count() / 199` (`evaluate.cpp:63`). So the raw eval
  stored in the TT already carries the damping of the moment it was stored. The TT *value* is adjusted separately by
  `value_from_tt(.., pos.rule50_count())` (:821).
- **Quiescence** does the same in small (:1713-1760): in check, no eval (`bestValue = futilityBase = -VALUE_INFINITE`,
  and `ss->staticEval` is not written); otherwise corrected eval from the TT's raw eval or `evaluate`, the TT value as
  a better stand-pat when its bound allows and it is not decisive (:1733-1735). Stand-pat cutoff returns a blend,
  `(441 * bestValue + 583 * beta) / 1024` (:1748). `futilityBase = ss->staticEval + 306` (:1759).

### 1.2 Improving and opponent worsening

```
improving         = ss->staticEval > (ss - 2)->staticEval;      // :863
opponentWorsening = ss->staticEval > -(ss - 1)->staticEval;     // :864
...
improving |= ss->staticEval >= beta;                            // :1046, after the null move, before ProbCut and the move loop
```

- Strict `>`. Computed at every node, PV included, from ply 0 (the sentinels' `VALUE_NONE` is a large positive number,
  so at plies 0 and 1 improving is false; INFERRED from the enum, not traced).
- **In check**: the node copies `(ss-2)->staticEval` into its own slot (:830). Its own improving is therefore false,
  and the node two plies later compares against the last eval made out of check. That copy is the whole fallback;
  there is no `(ss-4)` ladder as in SF11.
- `opponentWorsening` has no special in-check path; after an in-check parent it compares against the parent's copied
  value.

### 1.3 Every consumer, with its formula

| consumer | formula | reads | line |
|---|---|---|---|
| hindsight depth | `priorReduction >= 3 && !opponentWorsening` → `depth++`; `priorReduction >= 2 && depth >= 2 && ss->staticEval + (ss-1)->staticEval > 166` → `depth--` | staticEval, parent's | :867-870 |
| quiet ordering from the eval change | parent move quiet, parent not in check: `evalDiff = clamp(-((ss-1)->staticEval + ss->staticEval), -189, 194) + 60`; `mainHistory << evalDiff * 11`; pawn history `<< evalDiff * 13` when no TT hit and the mover was not a pawn or promotion | staticEval, parent's | :979-986 |
| razoring | `!PvNode && eval < alpha - 482 * depth * depth` → return qsearch | eval | :991-992 |
| reverse futility ("futility pruning: child node") | gate `!ss->ttPv && depth < 19 && eval >= beta && (!ttData.move \|\| ttCapture) && !is_loss(beta) && !is_win(eval)`; `futilityMult = min(45 + depth * 4, 85) - 20 * !ss->ttHit`; `futilityMargin = futilityMult * depth - (2789 * improving + 335 * opponentWorsening) * futilityMult / 1024 + abs(correctionValue) / 198435`; cut when `eval - futilityMargin >= beta`; returns `(661 * beta + 363 * eval) / 1024` | eval, improving, opponentWorsening, \|correction\| | :996-1008 |
| null move | gate `cutNode && ss->staticEval >= beta - 13 * depth - 47 * improving + 365 && !excludedMove && non_pawn_material(us) && ss->ply >= nmpMinPly && beta >= -2000`; `R = 7 + depth / 3 + max((ss->staticEval - beta) / 256, 0)`; verification search at `depth >= 16` | staticEval, improving | :1011-1044 |
| ProbCut | `probCutBeta = beta + 241 - 64 * improving`; skipped when a valid TT value is below it; capture picker threshold `probCutBeta - ss->staticEval`; `probCutDepth = depth - (improving ? 5 : 3)`; qsearch first, then the reduced search | staticEval, improving | :1057-1096 |
| LMR base | `reduction(i, d, mn, delta) = reductionScale - delta * 577 / rootDelta + !i * reductionScale * 197 / 512 + 982`, `reductionScale = reductions[d] * reductions[mn]`, `reductions[i] = int(2872 / 128.0 * log(i))` (1024ths of a ply) | improving | :713, :1885-1888 |
| LMR, eval term | quiet moves, alpha not decisive: `r += 3 * clamp(alpha - eval, -64, 96)` | eval | :1354-1355 |
| LMR, uncertainty | `r -= abs(correctionValue) / 26310` | \|correction\| | :1324 |
| move-count pruning | `moveCount >= (3 + depth * depth) / (2 - improving)` → skip remaining quiets | improving | :1169-1170 |
| futility of a capture | `!givesCheck && lmrDepth < 8`: `ss->staticEval + 234 + 247 * lmrDepth + PieceValue[captured] + 134 * captHist / 1024 <= alpha` → skip | staticEval | :1181-1188 |
| futility of a quiet ("parent node") | `!ss->inCheck && lmrDepth < 12`: `ss->staticEval + 119 * lmrDepth + 90 * (ss->staticEval > alpha) + 164 <= alpha` → skip, and `bestValue` is raised to that value; `lmrDepth` here already includes `history / lmrDivisor` | staticEval | :1213-1224 |
| singular margins | `corrValAdj = abs(correctionValue) / 198368` subtracted from the double and triple extension margins; `singularBeta` itself comes from the TT value, not the static eval | \|correction\| | :1250-1263 |
| multi-cut → correction | on a multi-cut return, `!ss->inCheck && value > ss->staticEval`: correction update with `clamp((value - ss->staticEval) * singularDepth * 177 / 1024, ±CORRECTION_HISTORY_LIMIT / 4)` | staticEval | :1281-1287 |
| fail-low bonus for the parent's move | `bonusScale += 142 * (!ss->inCheck && bestValue <= ss->staticEval - 106)`; `+= 159 * (!(ss-1)->inCheck && bestValue <= -(ss-1)->staticEval - 68)` | staticEval, parent's | :1585-1586 |
| correction update | §1.4 | staticEval | :1631-1638 |
| qsearch futility | `futilityValue = futilityBase + PieceValue[captured]`; `<= alpha` → skip; SEE test `see_ge(move, alpha - futilityBase)`; only the first two moves pass move-count pruning | staticEval | :1790-1812 |

Shape notes that matter for a port:
- Both futility prunes of *moves* use the **parent's** static eval and are decided **before** the move is made. No
  eval of the child is computed to prune a move.
- The null-move gate reads the corrected static eval only (not the TT value), fires only at expected cut nodes, and has
  no ply or iteration-depth condition apart from `nmpMinPly` (verification).
- Reverse futility runs to depth 18 with a margin of at most 85 units per ply, and only when there is no quiet TT move.

### 1.4 Correction history

**Tables** (`history.h`):
- One shared array `UnifiedCorrectionHistory` = `DynStats<MultiArray<CorrectionBundle<i16, 1024>, COLOR_NB>, CORRHIST_BASE_SIZE>`
  (:189-191), `CORRHIST_BASE_SIZE = UINT_16_HISTORY_SIZE` (:40), sized with the thread count and shared between
  threads (:208, `Shared = true`). A bundle holds four `i16` fields: `pawn`, `minor`, `nonPawnWhite`, `nonPawnBlack`
  (:158-162). Each key selects a bundle and reads **its own field** of it:
  `pawn_key()`, `minor_piece_key()`, `non_pawn_key(WHITE)`, `non_pawn_key(BLACK)`, each `& sizeMinus1`, then `[us]`
  (:227-248).
- Continuation correction: `CorrectionHistory<Continuation>` = `[piece][to]` → `[piece][to]` (:179-183). Each stack
  frame holds a pointer to the sub-table of the move made from it (`search.cpp:669-670`); a null move points it at
  `[NO_PIECE][0]` (:678).
- Limit `CORRECTION_HISTORY_LIMIT = 1024` (:41). Cleared with the fill value −5 (`search.cpp:696`; INFERRED from the
  argument of `clear_range`).

**Read** (`correction_value`, `search.cpp:85-101`), with `m = (ss-1)->currentMove`:

```
cv = 15341 * pawn + 10569 * minor + 12906 * (nonPawnWhite + nonPawnBlack)
   + (m.is_ok() ? 8761 * (contCorr[ss-2][pc][to] + contCorr[ss-4][pc][to]) : 64049)
corrected = raw + cv / 131072
```

With every entry at ±1024 the sum is 69,244 × 1024 / 131072 ≈ ±541 units; the pawn table alone ≈ ±120. After a null
move the continuation part is the constant 64049 (≈ +0.49 unit).

**Update rule** (`StatsEntry::operator<<`, `history.h:70-74`): `bonus` clamped to ±D (D = 1024), then
`entry += bonus - entry * |bonus| / D` (gravity).

**When and how much** (`search.cpp:1629-1638`, end of the node, cut or not):

```
if (!ss->inCheck && !(bestMove && pos.capture(bestMove)) && (bestValue > ss->staticEval) == bool(bestMove))
    bonus = clamp((bestValue - ss->staticEval) * depth * (bestMove ? 12 : 18) / 128, -256, +256);
    update_correction_history(.., 1061 * bonus / 1024);
```

- The third condition is the bound guard: a node with a best move (score above alpha: fail high or exact) teaches only
  "eval too low"; a node without one (fail low) teaches only "eval too high".
- Fail lows weigh 18, the others 12. The residual is taken against the **corrected** eval.
- Per-table scale of the bonus (`update_correction_history`, :109-131): pawn ×1, minor ×150/128, each non-pawn
  ×186/128, continuation `(ss-2)` ×130/128, `(ss-4)` ×70/128.
- A second, smaller teacher: the multi-cut return (§1.3).

**Consumers of the correction's size** (|cv| as "the eval is unreliable here"): reverse-futility margin wider
(/198435), LMR smaller (/26310), singular double / triple extensions easier (/198368).

---

## 2. Stockfish 11 (classical eval, no correction history)

`search.cpp:786-829`. Already tabled in `sf-pruning-schedules-comparison.md`; only the plumbing here.

- In check: `ss->staticEval = eval = VALUE_NONE`, `improving = false`, straight to the move loop (:787-792).
- TT hit: `tte->eval()` or `evaluate(pos)`; the TT value replaces `eval` when its bound allows (:793-807).
- TT miss: `evaluate(pos) + bonus` with `bonus = -(ss-1)->statScore / 512` (:810-814): the parent move's history shifts
  the static eval, a forerunner of correction history, and it is this shifted value that goes into the TT (:819).
  After a null move no eval is computed: `-(ss-1)->staticEval + 2 * Eval::Tempo` (:817).
- Improving, with the in-check ladder (:828-829):
  `(ss-2)->staticEval == VALUE_NONE ? (ss->staticEval >= (ss-4)->staticEval || (ss-4)->staticEval == VALUE_NONE) : ss->staticEval >= (ss-2)->staticEval`.
  Note `>=` (SF19 uses `>`) and the default **true** when both earlier evals are missing.
- Consumers: razoring `depth < 2 && eval <= alpha - 531` (:823-826); reverse futility `!PvNode && depth < 6 &&
  eval - 217 * (depth - improving) >= beta`, returns `eval` (:831-836); null move `eval >= beta && eval >= ss->staticEval
  && ss->staticEval >= beta - 32 * depth + 292 - improving * 30`, `R = (854 + 68 * depth) / 258 + min((eval - beta) / 192, 3)`
  (:839-852); ProbCut `raisedBeta = beta + 189 - 45 * improving` (:895); move-count `(5 + depth * depth) * (1 + improving) / 2 - 1`
  (:81-83); LMR `+ (!improving && r > 1007)` (:76-79); quiet futility `ss->staticEval + 235 + 172 * lmrDepth <= alpha` (:1019).
- Improving is worth one ply of reverse-futility margin (217 units, about one pawn) in SF11. In SF19 it is worth
  `2789 / 1024` ≈ 2.7 plies of the per-ply multiplier.

---

## 3. Reckless

General search: `reckless-search-brief-2026-11-06.md`. Node eval specifics (`search.rs:447-535`):

- In check: `raw_eval = eval = Score::NONE` (:453-455). Excluded: the stack's value. TT entry: its `raw_eval` or the
  network; miss: network, and an eval-only TT write (:466). `stack[ply].eval = eval` (:484), corrected.
- `correct_eval` (`evaluation.rs:3-13`): `(raw * (21032 + material) + optimism * (1548 + material)) / 27015`, then
  `eval * (200 - fiftymove_clock) / 200`, then `+ correction`, then the TB clamp. Material scaling and the 50-move
  damping are applied at correction time, so the TT's raw eval carries neither.
- `estimated_score` = the TT score when its bound allows, else `eval` (:471-482); razoring, reverse futility and the
  null move read it.
- **Improvement is a number**, not only a flag (:525-535): 0 in check; else `eval - stack[ply-2].eval`; if that is
  missing `eval - stack[ply-4].eval`; else 0. `improving = improvement > 0`.
- Consumers that use the number: reverse-futility margin `1140 * d * d / 128 - 120 * improvement / 1024 + 22 * d +
  669 * |corr| / 1024 - 54 * (no threats against us) - 19`, floor 2, returns `lerp(estimated, beta, 0.6945)` (:549-564);
  null-move gate `estimated >= beta + max(-9 * d + 110 * tt_pv - 94 * improvement / 1024 - 21 * (cutoff_count < 2) + 337, 2)`,
  `r = (4407 + 917 * improving + 265 * d + 477 * clamp(estimated - beta, 0, 1187) / 128) / 1024` (:567-589);
  late-move pruning `move_count >= (2818 + 78 * improvement / 16 + 1351 * d * d + 74 * history / 1024) / 1024` (:799);
  LMR `reduction -= clamp(425 * improvement / 128, -241, 1155)` and `-= 3417 * |corr| / 1024` (:864-865; a second site
  :942-943 with 366 / 2255).
- Other consumers: razoring `estimated < alpha - 237 - 254 * d * d` (:538-546); ProbCut `beta + 254 - 85 * improving`,
  base depth `depth - 4 - improving` (:636-658); quiet futility `eval + 79 * d + 55 * history / 1024 + 77 * (eval >= beta)
  + 555 * |corr| / 1024 - 127 <= alpha`, `depth < 14` (:806-819); bad-noisy futility `eval + 84 * d + 82 * history / 1024 + 24`
  (:822); hindsight `reduction >= 2249 && eval_delta < 0` → `depth += 1`, `!tt_pv && depth >= 2 && reduction > 0 &&
  eval_delta > 57` → `depth -= 1` (:506-517); quiet ordering from the eval change, `812 * (-(eval + parent eval)) / 128`
  clamped to [−144, 324] (:492-503); qsearch SEE threshold `(alpha - eval) / 8 - min(|corr|, 68) - 74 - history / 48` (:1340).
- **Correction history** (`search.rs:1403-1452`, `history.rs:159-219`): pawn, non-pawn white, non-pawn black, and two
  continuation terms (ply−2, ply−4); no minor table. Read = plain sum `/ 64`, no per-table weights. The keyed tables
  are `[16 fifty-move buckets][side to move][65536]`, limit 14605; the continuation table is keyed
  `[in_check][capture][piece][to][piece][to]`, limit 16418. Update: `bonus = clamp(148 * depth * diff / 128, -4678, 2496)`
  (an asymmetric clamp: "eval too high" may teach almost twice as hard), same bonus to every table, gravity
  `e += bonus - |bonus| * e / MAX`. Skipped in check, when the best move is noisy, and when the bound contradicts the
  residual (`Upper && best >= eval`, `Lower && best <= eval`) (:1194-1199).

---

## 4. Triumviratus (SF19 base; differences only)

Node eval (`13_search.inc:808-861`, `node_evaluate`) is SF19's step 5 line for line: in check copies `f[-2].static_eval`;
TT value as a better eval; eval-only TT write on a miss; `opp_worse = static_eval > -f[-1].static_eval` (:839);
`improving = static_eval > f[-2].static_eval` (`node_trend`, :875-882); `improving |= static_eval >= beta` (:1125).

**Correction weights** differ from SF19's (`01_params.inc:730-773`, `:1628`): `CorrPawnW 15562`, `CorrMinorW 9744`,
`CorrNonPawnW 10452`, `CorrContW 7220`, `CorrNoMoveW 80695`, `CorrEvalDiv 131072`; a gain per material band
`CorrGain0-3`, all `1024` (`:395`). Read in `corr_signal` (`09_history.inc:602-635`).

### 4.1 CorrPhase (correction history per material band)

- **What it changes** (`09_history.inc:539-548`, `:577-587`, `:607`): one 64-bit constant per band is XORed into the
  **pawn** key and the **minor-piece** key before the table lookup: `kp = pawn_key ^ phk ^ fk`, `km = mm_key ^ phk`.
  The two non-pawn keys are not salted (`kw = np_key[white] ^ fk`). Bands by pieces on the board: ≤ 9, 10-15, 16-23,
  ≥ 24. Same tables, same size, same update; one pawn structure now owns up to four slots, one per band.
- **Why it claims a gain** (`NOVELTIES.md` §3, :133-149): its network has one expert per band, and the systematic
  error on a given pawn or minor structure differs between experts. A diagnostic build reports the corrections
  removing 58.8% of the eval error with it against 57.3% without, the gain coming from positions of 9 pieces or fewer.
  Games: **+3.2 ± 3.8** over 8,634 at 12+0.12, adopted 2026-10-06. That is inside its own error bar; by our ship rule
  it would read "not harmful".
- History in the parameter file: a first test read −5.10 ± 8.13 over 1,976 games before a prefetch problem in the
  make was fixed (`01_params.inc:199-204`; only the excerpt was read). Now fixed code, no option (`:957`, "CorrPhase
  retired 28/10"; INFERRED that "retired" means the switch, since `corr_phase_key` is unconditional).
- **Variants tried around it** (all in `09_history.inc` / `01_params.inc`, default off unless said):
  `CorrPhaseQueen` (band × queens on board, 8 regimes, :573-587); `CorrC2Key` (salt by the 8.1 network's expert, or
  the pair, :550-571; SPRT pending); a key per cell of 32 lost −21.3 ± 16.8 over 458 games ("each cell's corrections
  learn from an eighth of the data", :580-582); `CorrFiftyStep` (salt by fifty-move band, 0 = off, :589-600);
  `CorrExpertW` (a separate per-band pawn table **added** to the shared one, entering the eval but not the
  uncertainty signal, 0 = off, :637-655); `CorrCont6W`, `CorrContOppW` (more continuation plies, :617-623);
  `CorrPairCont` and `CorrPkW` closed negative and removed (:625, :649).
- The lesson for a keyed table: splitting a key multiplies the slots a structure needs, and a split that is too fine
  starves each slot (the 32-cell loss). Four bands was the finest split that held.

### 4.2 Per-expert margins

`by_expert(n.expert, v0, v1, v2, v3)` (`13_search.inc:110-115`) adds a per-band offset to a rule's margin; `n.expert`
is the same piece-count band (`search_band`, `09_history.inc:398-402`). Where it is read and the shipped values:

| rule | parameter | bands ≤9 / 10-15 / 16-23 / ≥24 | line |
|---|---|---|---|
| reverse-futility multiplier (units per ply) | `RfpExp0-3` | 3 / 1 / −3 / 3 | `13_search.inc:943`, `01_params.inc:869-872` |
| quiet futility margin | `FutExp0-3` | −3 / −14 / −2 / 8 | `:1182`, `:873-876` |
| LMR (1024ths of a ply) | `LmrExp0-3` | −25 / 102 / −6 / 101 | `:445`, `:877-880` |
| null-move floor | `NmpExp0-3` | 0 / 0 / 0 / 0 | `:997`, `01_params.inc:471` |
| move-count threshold scale | `LmpExp0-3` | 0 / 0 / 0 / 0 | `:1252`, `01_params.inc:549` |
| razoring, ProbCut, capture futility / SEE, singular | `RazorExp`, `ProbCutExp`, `CaptFutExp`, `CaptSeeExp`, `SingExp` | not read (defaults not checked) | `:915`, `:1066`, `:1220`, `:1229`, `:1723` |

The shipped offsets are small against the base margins (the reverse-futility multiplier is 45-85 per ply in SF19's
form). The header comment calls them an "extension of the per-expert corrections, CorrPhase +3.2" (`:110-114`); no
separate game result for them was found in the files read.

### 4.3 Other differences in the trend signals

- Reverse futility (`reverse_futility`, `13_search.inc:933-984`): SF19's shape with `RfpImproving 2852`,
  `RfpOppWorse 335`, `RfpCorrDiv 198435` (`01_params.inc:497-499`). Levers at 0 that are worth knowing as ideas:
  `ImpSlopeRfp` (margin follows the *continuous* improvement, clamped to ±`ImpAdjCap` 160, `:869-873`, `:949`);
  `RfpCorrSign` (the correction's sign, not only its size); `RfpBehind` / `RfpAll`, built on a prune log that found
  reverse-futility cuts wrong 2.1× / 2.9× / 5.1× as often (depths 1-2 / 3-5 / 6-12) when the side's static eval is
  below −700, and 2.73× in ALL nodes (`:961-971`); `RfpDisagree` (wider margin when the network's two outputs
  disagree, `:972-979`).
- `C2SameRegImp` (off, SPRT pending, `:309-316`, `:840-849`, `:877-881`): improving, opp_worse and the hindsight step
  compare evals of different plies; when a capture between them moved the position to another expert, the difference
  measures the experts and not the position, so the comparison is made only inside one regime (else four plies back).
  An earlier form, `PhaseEdgeImproving`, was stopped at −14.1 ± 19.4 over 296 games (not a measurement).
- Order: both eval gates give their verdict, then the eval change teaches the parent's move, then a gate acts
  (`eval_gates`, `:909-931`); reverse futility before razoring.

---

## 5. Side by side

| | SF11 | SF19 | Reckless | Triumviratus |
|---|---|---|---|---|
| eval at every node out of check | yes | yes | yes | yes |
| stored | stack + TT (with the history bonus) | stack (corrected) + TT (raw) | stack (corrected) + TT (raw) | as SF19 |
| in check | `VALUE_NONE` | copy of `(ss-2)` | `NONE` | copy of `f[-2]` |
| TT value as better eval | yes | yes (not for null gate, futility of moves) | yes (`estimated_score`) | yes |
| improving | `>=`, ladder to `(ss-4)`, default true | `>`, no ladder; `\|= staticEval >= beta` later | continuous, ladder to ply−4, default 0 | as SF19 |
| opponent worsening | no | yes | via `eval_delta` (hindsight only) | yes |
| correction tables | none (`statScore / 512` on a TT miss) | pawn, minor, non-pawn ×2, continuation ×2 | pawn, non-pawn ×2, continuation ×2, × 16 fifty-move buckets | SF19's, pawn and minor keys salted by material band |
| \|correction\| as a signal | - | RFP, LMR, singular | RFP, futility, LMR, qsearch SEE | RFP, LMR, singular |
| futility of a move | parent eval, before the move | parent eval, before the move | parent eval, before the move | as SF19 |
| null gate | `eval >= beta` and a static floor | static floor only, cut nodes only | `estimated` floor, cut nodes only | as SF19 + per-band term (0) |

What all four share, and we do not (details in the audit): one static eval per node, kept where the next two plies
can read it; improving in at least reverse futility, move-count pruning, LMR, the null gate and ProbCut; futility of
moves decided on the parent's eval before the move is made; the null move gated on the static eval.

## 6. Not verified

- SF19: `VALUE_NONE`'s numeric value and so improving at plies 0-1; the meaning of `clear_range(-5, ..)`; the size
  multiplier of the shared correction array per thread; `search.cpp:1390-1560` was not read (no static-eval consumer
  was found there by search).
- Reckless: `fiftymove_clock_bucket()`'s bucket width; the second LMR site's context (:942-943).
- Triumviratus: defaults of `RazorExp`, `ProbCutExp`, `CaptFutExp`, `CaptSeeExp`, `SingExp`, `RfpCorrSign`,
  `RfpBehind*`; `13_learn.inc` (`Learner::corrections`, the update) was not read, so its bonus formula is assumed to
  be SF19's; whether the per-expert margins have their own game result.
- SF11's file was read at :66-85 and :780-870 only; the other cited lines come from a search listing.
