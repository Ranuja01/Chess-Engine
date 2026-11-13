# Pruning, reductions, extensions and history tables in the giants (2026-11-13)

Created 2026-11-13 for stage S5 of `active/SEARCH-V2-PLAN-2026-11-10.md`. A read of the sources on disk; nothing was
built or run. Our side of the comparison is in `active/S5-PRUNING-AUDIT-2026-11-13.md`.

**What this adds to the docs already on disk.** `sf-pruning-schedules-comparison.md` has every schedule for SF11 to SF18
and the "what scales with what" section; `reckless-search-brief-2026-11-06.md` lists Reckless's ideas without formulas;
`root-giants-2026-11-13.md` and `current/ROOT_SEARCH_MODEL.md` cover the root. This file adds: the SF19 column, the
history tables in full (the older doc has only their keys), how each engine keeps the singular exclusion search from
disturbing its own node, Reckless's exact formulas, and Triumviratus's HistGrad and causal reduction.

**Sources and units.**

| engine | path | units |
|---|---|---|
| SF19 | `stockfish_19_linux\src\` (`search.cpp`, `history.h`, `movepick.cpp`) | reductions in 1/1024 ply; eval internal (pawn 208, as SF18) |
| SF11 | `stockfish_11\stockfish-11-win\src\` | reductions in plies; pawn 128 mg / 213 eg |
| Reckless | `E:\chess_data\engines\reckless_src\src\` (`search.rs`, `history.rs`), commit `aec5cb2` per the brief | 1/1024 ply; pawn 109 |
| Triumviratus | `E:\chess_data\engines\triumviratus_src\source\search\` (`13_search.inc`, `13_learn.inc`, `01_params.inc`), `docs\project\NOVELTIES.md` | 1/1024 ply; SF19's scale INFERRED from equal constants |

Only forms transfer; every constant below is on its engine's scale and against an NNUE (see the comparison doc §4).
Line numbers are as read today.

---

## 1. Stockfish 19, technique by technique

Node flags used below: `PvNode`; `cutNode` (passed by the parent); `allNode = !(PvNode || cutNode)`
(`search.cpp:726`); `ss->ttPv` (`:822`); `improving = staticEval > (ss-2)->staticEval` (`:863`);
`opponentWorsening = staticEval > -(ss-1)->staticEval` (`:864`). After the null move, `improving |= staticEval >= beta`
(`:1046`), so move-loop consumers see a boosted flag. In check, all of steps 8 to 12 are skipped (`:975-976`).

### 1.1 Node-level pruning (before the move loop)

| technique | formula | gates | reads | cite |
|---|---|---|---|---|
| Hindsight depth | `depth++` if `priorReduction >= 3 && !opponentWorsening`; `depth--` if `priorReduction >= 2 && depth >= 2 && staticEval + (ss-1)->staticEval > 166` | all nodes | the parent's applied reduction, two static evals | `:867-870` |
| TT cutoff | `ttData.depth > depth - (ttData.value <= beta)`, bound matches, `cutNode == (ttValue >= beta) \|\| depth > 4`; at `depth >= 7` the TT move is made and the child's entry must agree | `!PvNode && !excludedMove`, `rule50 < 96` | TT | `:874-912` |
| Razoring | `eval < alpha - 482 * depth * depth` returns `qsearch` | `!PvNode` | eval (TT-refined) | `:991-992` |
| Reverse futility | `futilityMult = min(45 + 4*depth, 85) - 20 * !ttHit`; margin `= futilityMult * depth - (2789 * improving + 335 * opponentWorsening) * futilityMult / 1024 + \|correctionValue\| / 198435`; return `(661 * beta + 363 * eval) / 1024` | `!ttPv`, `depth < 19` (6 when seeking mate), `eval >= beta`, `!ttMove \|\| ttCapture` | eval, correction | `:996-1008` |
| Null move | gate `staticEval >= beta - 13 * depth - 47 * improving + 365`; `R = 7 + depth / 3 + max((staticEval - beta) / 256, 0)`; child searched as non-cut (`cutNode = false`) | `cutNode`, `!excludedMove`, non-pawn material, `ply >= nmpMinPly`, `beta >= -2000` | static eval only | `:1011-1044` |
| Null verification | at `depth >= 16`: re-search `depth - R` with null move off until `ply + 3 * (depth - R) / 4` | | | `:1027-1042` |
| IIR | `depth--` | `!followPV && !allNode && depth >= 6 && !ttMove` | TT move, last iteration's PV | `:1051-1052` |
| ProbCut | `probCutBeta = beta + 241 - 64 * improving`; picker threshold `probCutBeta - staticEval`; qsearch first, then `depth - (improving ? 5 : 3)`; returns `value - (probCutBeta - beta)`; writes the TT at `probCutDepth + 1` | `depth >= 3`, not a TT value below `probCutBeta`, skips `excludedMove` | capture history, SEE | `:1057-1096` |
| Small ProbCut | TT lower bound `>= beta + 428` at `ttDepth >= depth - 4` returns `beta + 428` | also in check | TT only | `:1101-1104` |

Newer than SF18 (comparison doc §1): the eval-excess term is back in the null move's R (SF18 had removed it) and
`eval >= beta` is not required, only the static-eval gate; reverse futility's multiplier now grows with depth and its
depth cap is 19; IIR is restricted to nodes off the previous PV; ProbCut's depth is a two-step on `improving`, not SF18's
eval-dependent clamp; razoring lost its constant term.

### 1.2 Move-loop pruning (Step 15)

Common gate: `!rootNode && non_pawn_material(us) && !is_loss(bestValue)` (`:1166`). `r = reduction(improving, depth,
moveCount, delta)`, plus `929` when `ttPv` (`:1157-1162`); `lmrDepth = newDepth - r / 1024` (`:1173`).

| technique | formula | applies to | cite |
|---|---|---|---|
| LMP | `moveCount >= (3 + depth * depth) / (2 - improving)` tells the picker to skip quiets | every move type triggers it; only quiets are skipped | `:1169-1170` |
| Capture futility | `staticEval + 234 + 247 * lmrDepth + PieceValue[captured] + 134 * captHist / 1024 <= alpha` | captures and checks, `!givesCheck && lmrDepth < 8` | `:1181-1188` |
| SEE, captures and checks | `!see_ge(move, -(177 * depth + captHist * 34 / 1024))`, with the stalemate-sacrifice guard | captures and checks | `:1192-1195` |
| Continuation-history pruning | `contHist[0] + contHist[1] + pawnHist < -4136 * depth` | quiets, not on the followed PV at PV nodes (`!followPV \|\| !PvNode`) | `:1200-1206` |
| History into lmrDepth | `history += 69 * mainHistory / 32`; `lmrDepth += history / lmrDivisor[min(depth, 16) - 1]` | quiets | `:1208-1211` |
| Quiet futility | `staticEval + 119 * lmrDepth + 90 * (staticEval > alpha) + 164 <= alpha` with `lmrDepth < 12`, `!inCheck`; raises `bestValue` to the futility value | quiets | `:1213-1224` |
| SEE, quiets | `!see_ge(move, -23 * lmrDepth * lmrDepth)` after `lmrDepth = max(lmrDepth, 0)` | quiets | `:1226-1230` |

`lmrDivisor` is a 16-entry table by depth: `{3637, 2787, 2761, 2939, 3171, 3347, 3147, 2762, 2772, 3106, 3107, 3060,
3112, 2991, 3090, 3542}` (`:55-56`). SF18 used one constant (3208).

### 1.3 LMR in full

Base (`:1885-1888`, table at `:713`):

```
reductions[i]  = int(2872 / 128.0 * log(i))
reductionScale = reductions[depth] * reductions[moveCount]
r = reductionScale - delta * 577 / rootDelta + !improving * reductionScale * 197 / 512 + 982
```

`depth` is the node's remaining depth and `delta = beta - alpha`. Adjustments, in code order (positive = reduce more):

| term | value | cite |
|---|---|---|
| ttPv, before pruning (so `lmrDepth` sees it) | `+929` | `:1161-1162` |
| ttPv, after the move is made | `-(3023 + PvNode * 1004 + (ttValue > alpha) * 885 + (ttDepth >= depth) * (816 + cutNode * 940))` | `:1316-1318` |
| base offset | `+697` | `:1321` |
| move count | `-65 * moveCount` | `:1323` |
| eval uncertainty | `-\|correctionValue\| / 26310` | `:1324` |
| cut node | `+4026 + 933 * !ttMove` | `:1327-1328` |
| TT move is a capture | `+1079` | `:1331-1332` |
| child fail-high count `(ss+1)->cutoffCnt > 1` | `+264 + 1095 * (cutoffCnt > 2) + 1138 * allNode` | `:1335-1336` |
| else, the move is the TT move | `-2179` | `:1339-1340` |
| statScore, capture | `873 * PieceValue[captured] / 128 + captureHistory` | `:1342-1344` |
| statScore, quiet | `(2252 * mainHistory + 1126 * contHist[0] + 1093 * contHist[1]) / 1024` | `:1346-1349` |
| statScore into r | `-statScore * 439 / 4096` | `:1352` |
| quiet, alpha against eval | `+3 * clamp(alpha - eval, -64, 96)` (new in SF19) | `:1354-1355` |
| all node | `r += r * 276 / (256 * depth + 268)` | `:1358-1359` |

Applied when `depth >= 2 && moveCount > 1` (`:1362`): `d = max(1, min(newDepth - r / 1024, newDepth + 2)) + PvNode`,
searched as a cut node with a null window; `ss->reduction = newDepth - d` is left for the child's hindsight step.
Captures and checks are reduced by the same code; only the statScore differs.

Re-search (`:1377-1391`): on `value > alpha`, `newDepth += (d < newDepth && value > bestValue + 53) - (value < bestValue
+ 8)`; a null-window search at `newDepth` runs only if `newDepth > d`; then the continuation histories of the move get
`+1334`.

When LMR does not apply (the first move at non-PV nodes, or `depth < 2`): `r += 1127` without a TT move, then
`newDepth - (r > 5234) - (r > 5487 && newDepth > 2)` (`:1395-1404`). So even "full-depth" null-window searches can lose
one or two plies.

After a move raises alpha without cutting: `depth -= 3` when `3 < depth < 12` (`:1535-1536`), which shrinks every later
move of the node. Fail-high value is blended toward beta: `(bestValue * depth + beta) / (depth + 1)` (`:1559-1560`).

### 1.4 Extensions

SF19 has **no check extension**, no recapture or passed-pawn extension. The whole set:

1. **Singular** (`:1246-1303`). Gate: `!rootNode && move == ttMove && !excludedMove && depth >= 6 + ttPv`, a valid
   non-decisive TT value with a lower bound, `ttDepth >= depth - 3`, `!is_shuffling`, `!seekMate`.
   `singularBeta = ttValue - (59 + 66 * (ttPv && !PvNode)) * depth / 63`; `singularDepth = newDepth / 2`; the exclusion
   search keeps the node's `cutNode`.
   - `value < singularBeta`: `extension = 1 + (value < singularBeta - doubleMargin) + (value < singularBeta -
     tripleMargin)`, with `doubleMargin = -2 + 204 * PvNode - 152 * !ttCapture - corrValAdj - 1175 * ttMoveHistory /
     114178 - 38 * (ply > rootDepth)` and `tripleMargin = 70 + 279 * PvNode - 188 * !ttCapture + 81 * ttPv - corrValAdj
     - 43 * (ply > rootDepth)`, `corrValAdj = |correctionValue| / 198368`. Also `depth++` for the whole node.
   - **Multi-cut**: `value >= beta` (not decisive) returns `value`; `ttMoveHistory << -421 - 110 * depth`; the
     correction history learns `value - staticEval` when positive.
   - **Negative extension**: otherwise, if `ttValue >= beta || cutNode`, `extension = -3`.
2. **LMR going negative**: up to `newDepth + 2`, plus one at PV nodes (§1.3).
3. **Deeper re-search** after a reduced fail-high: `+1` (§1.3).
4. **Hindsight** `depth++` (§1.1).
5. **PV TT move before qsearch**: `newDepth = max(newDepth, 1)` when the TT entry has depth above 1 or a decisive value
   (`:1417-1420`).

`is_shuffling` (`:153-160`) blocks the singular test when a piece is moving back and forth late in a long line.

### 1.5 How the exclusion search avoids disturbing its node

SF19 calls `search<NonPV>(pos, ss, ...)` on the **same stack frame** with `ss->excludedMove` set (`:1253-1255`). The
comment calls the mechanism "a bit fragile" (`:1241-1242`). What makes it work:

- **The move list is a local object.** `MovePicker mp` is constructed inside each call (`:1111`), so the inner call has
  its own list and stage; the outer picker is untouched.
- The static eval is reused, not recomputed: `else if (excludedMove) unadjustedStaticEval = eval = ss->staticEval`
  (`:831-832`); `ss->ttPv` is kept (`:822`).
- Gated on `!excludedMove`: the TT cutoff (`:874`, `:914`), tablebases (`:923`), the null move (`:1011`), the singular
  test itself (`:1246`, so no recursion), and the **TT write** (`:1621`).
- The move loop skips the excluded move (`:1124`), and so does ProbCut (`:1069`).
- No legal move left returns `alpha` (`:1566`), a fail low, so a forced TT move reads as singular.
- Not gated (INFERRED from the absence of a test): reverse futility, IIR, ProbCut, history updates, correction
  history, and the `(ss+2)->cutoffCnt = 0` reset.

Reckless takes a different route: the singular test runs **once, before the move loop and before the node's picker
exists** (`search.rs:711-752`, picker at `:765`), with the exclusion in a per-ply array `td.excluded[ply]`. The
extension is then applied only to the first move (`:857`). See §4.

---

## 2. History tables in SF19

One update rule for every table (`history.h:70-77`), the "gravity" form with a per-table limit `D`:

```
clamped = clamp(bonus, -D, D);   entry += clamped - entry * |clamped| / D
```

| table | index | D | initial fill | cite |
|---|---|---|---|---|
| main (butterfly) | `[colour][move.raw()]` (from, to and flags: 16 bits) | 7183 | -5 | `history.h:128`, `search.cpp:691` |
| low-ply | `[ply < 5][move.raw()]` | 7183 | 102, refilled every search | `history.h:132`, `search.cpp:326` |
| capture | `[piece][to][captured type]` | 10692 | -742 | `history.h:135`, `search.cpp:692` |
| continuation | `[inCheck][capture][piece][to]` of the earlier move, then `[piece][to]` of this move | 30000 | -586 | `history.h:138-143`, `search.cpp:667-668`, `:704` |
| pawn | `[pawn_key & mask][piece][to]`, base size 8192 per thread | 8192 | -1338 | `history.h:146`, `:220-222` |
| TT-move | one counter per thread | 8192 | 0 | `history.h:196` |
| correction | pawn key, minor-piece key, non-pawn key per colour (each `[key][colour]`), and continuation `[piece][to][piece][to]` at plies 2 and 4 | 1024 | -5 / 5 | `history.h:157-194`, `search.cpp:85-131` |

No killers and no counter-move table (removed in SF17).

### 2.1 Updates

`update_all_stats`, run when a best move raised alpha (`search.cpp:1957-2012`):

```
bonus = min(133 * depth - 81, 1487) + 364 * (bestMove == ttMove) + (ss-1)->statScore / 28
malus = min(968 * depth - 235, 2244)
if (!PvNode) bonus += bonus * (quietsSearched + capturesSearched) / 256
```

- Quiet best move: quiet set `<< bonus * 899 / 1024`. Each earlier quiet takes a **decaying** malus: start
  `malus * 1159 / 1024`, multiply by `921 / 1024` per move, so later-tried moves are punished less (`:1986-1991`).
- Capture best move: capture history `<< bonus * 1427 / 1024`.
- Every earlier capture: `<< -malus * 1489 / 1024` whatever the best move was.
- The parent's quiet move, if it was the first move tried (or second after a TT hit): continuation histories of
  `ss-1` `<< -malus * 713 / 1024` (`:2002-2003`).

The quiet set, `update_quiet_histories` (`:2045-2058`): main `<< bonus`; low-ply `<< bonus * 712 / 1024` when
`ply < 5`; continuation `<< bonus * 750 / 1024` through the rule below; pawn `<< bonus * (bonus > -4 ? 1104 : 459) /
1024` (rewards weigh more than penalties).

`update_continuation_histories` (`:2017-2041`): plies `{1, 2, 3, 4, 5, 6}` with weights `{520, 390, 145, 251, 66,
209}`; in check only plies 1 and 2. A running count of the entries already positive selects a multiplier from
`{94, 103, 110, 106, 119, 126, 121}`, so agreement between plies strengthens the update:
`entry << bonus * weight * multiplier / 65536 + 73 * (i < 2)`.

Other write sites:

| event | update | cite |
|---|---|---|
| TT cutoff above beta with a quiet TT move | quiet set `<< min(112 * depth, 695)`; parent's early quiet move (`moveCount < 5`) continuation `<< -2210` | `:880-889` |
| static eval change after the opponent's quiet move | `evalDiff = clamp(-((ss-1)->staticEval + staticEval), -189, 194) + 60`; opponent's main `<< evalDiff * 11`; pawn `<< evalDiff * 13` when no TT hit | `:979-986` |
| reduced move beat alpha (after the re-search) | continuation `<< 1334` | `:1390` |
| node failed low, prior move quiet | `bonusScale = -241 - (ss-1)->statScore / 98 + min(59 * depth, 420) + 186 * ((ss-1)->moveCount > 9) + 142 * (bestValue <= staticEval - 106) + 159 * (bestValue <= -(ss-1)->staticEval - 68)`, floored at 0; `scaledBonus = min(150 * depth - 85, 1337) * bonusScale`; continuation `<< scaledBonus * 263 / 16384`, main `<< scaledBonus * 215 / 32768`, pawn `<< scaledBonus * 324 / 8192` | `:1579-1601` |
| node failed low, prior move a capture | capture history `<< 892` | `:1604-1609` |
| node outcome, non-PV | `ttMoveHistory << (bestMove == ttMove ? 918 : -747)` | `:1574-1575` |
| multi-cut | `ttMoveHistory << -421 - 110 * depth` | `:1279` |

### 2.2 Where each table is read

| reader | tables | cite |
|---|---|---|
| quiet ordering | `2 * main + 2 * pawn + contHist[0] + [1] + [2] + [3] + [5]` (ply 5 is skipped), `+ 8 * lowPly / (1 + ply)` when `ply < 5`; plus a check bonus 16384 and `20 * PieceValue * (escapes a lesser attacker - moves into one)` | `movepick.cpp:231-249` |
| capture ordering | `captureHistory + 7 * PieceValue[captured]`; good or bad by `see_ge(move, -value / 18)` | `movepick.cpp:225-226`, `:307` |
| evasions | captures by victim; quiets `main + contHist[0]` | `movepick.cpp:254-257` |
| quiet sort depth | partial insertion sort down to `-3560 * depth`; "good quiet" above `-14000`, the rest after the bad captures | `movepick.cpp:280`, `:324`, `:331`, `:352-354` |
| continuation pruning | `contHist[0] + contHist[1] + pawn` | `search.cpp:1200-1206` |
| lmrDepth shift (futility, SEE) | the above `+ 69 * main / 32` | `:1208-1211` |
| capture futility and SEE margin | capture history | `:1178-1195` |
| LMR | statScore (§1.3): main, contHist 0 and 1; capture history for captures | `:1342-1352` |
| history bonus size | `(ss-1)->statScore / 28` | `:1973` |
| singular double margin | `ttMoveHistory` | `:1261` |
| reverse futility, LMR, singular margins | `\|correctionValue\|` | `:1004`, `:1324`, `:1259` |

Pruning reads continuation plies 1 and 2 only; ordering reads 1, 2, 3, 4 and 6; updates write all six.

---

## 3. Stockfish 11: where it differs

The schedules are in the comparison doc. Additions for S5:

- **Extensions SF11 has and SF19 dropped**: singular gives at most `+1` (`search.cpp:1059`); a check extension
  (`:1075`); a passed-pawn push that is the first killer (`:1078-1081`); one more at `:1086`; castling (`:1089`).
  The conditions at `:1060-1090` were not re-read line by line here.
- **History**: `stat_bonus(d) = d > 15 ? -8 : 19 * d * d + 155 * d - 132` (`:86-87`), quadratic in depth with a
  cut-off; a larger bonus when `bestValue > beta + PawnValueMg` (`:1610-1611`). Continuation plies `{1, 2, 4, 6}`
  with equal weight (`:1648`). The reverse of a quiet best move takes a malus (`:1670`). Killers and a counter-move
  table `[piece][to]` exist (`:1658-1675`).
- **Ordering**: `main + 2 * contHist[0] + 2 * contHist[1] + 2 * contHist[3] + contHist[5]` (`movepick.cpp:114-118`);
  killers and the counter move are a separate "refutations" stage (`:62`, `:182-188`).
- **Malus**: flat `-bonus2` for every earlier quiet (`:1621-1622`), no decay by position.
- No low-ply, pawn, TT-move or correction history; no multi-cut history side effects.

---

## 4. Reckless: exact forms (the brief has the list; these are the formulas)

Flags: `improvement = eval - stack[ply-2].eval` as a **number**, not a flag (`search.rs:525-535`); `estimated_score` is
the eval refined by the TT (above the lines read). `history` for a quiet `= quiet_history + conthist(1) + conthist(2)`,
for a capture the noisy history (`:786-791`).

| technique | formula | cite |
|---|---|---|
| Hindsight | `depth += 1` if the parent's `reduction >= 2249 && eval_delta < 0`; `depth -= 1` if `!tt_pv && depth >= 2 && reduction > 0 && eval_delta > 57` | `:506-517` |
| Razoring | `estimated_score < alpha - 237 - 254 * depth^2`, non-PV, `alpha < 2048`, TT move not quiet, TT bound not lower | `:538-546` |
| Reverse futility | `estimated_score >= beta + max(1140 * depth^2 / 128 - 120 * improvement / 1024 + 22 * depth + 669 * \|corr\| / 1024 - 54 * (no piece of ours under threat) - 19, 2)`; returns `lerp(score, beta, 0.6945)`; `!tt_pv`, no depth cap | `:549-564` |
| Null move | cut node; `!potential_singularity`; `estimated_score >= beta + max(-9 * depth + 110 * tt_pv - 94 * improvement / 1024 - 21 * (cutoff_count[ply+1] < 2) + 337, 2)`; not when the TT move is a lower-bound capture of a knight or better; `r = (4407 + 917 * improving + 265 * depth + 477 * clamp(score - beta, 0, 1187) / 128) / 1024`; the window drops to `tt_score` when that is a lower bound below beta; verification at `depth >= 16` | `:567-633` |
| ProbCut | `probcut_beta = beta + 254 - 85 * improving`; cut node only; needs `tt_score >= probcut_beta` (or `eval >= beta` without a TT score) and a non-quiet TT move; qsearch, then `probcut_depth = clamp(base - (score - probcut_beta) / 319, 0, base)` with `base = depth - 4 - improving` and a raised beta `+197` per ply saved; returns `lerp(score, beta, 0.2695)` | `:636-705` |
| LMP | `move_count >= (2818 + 78 * improvement / 16 + 1351 * depth^2 + 74 * history / 1024) / 1024`: improvement and history sit **inside** the limit | `:795-803` |
| Quiet futility | `eval + 79 * depth + 55 * history / 1024 + 77 * (eval >= beta) + 555 * \|corr\| / 1024 - 127 <= alpha`, `depth < 14`; sets `skip_quiets` | `:806-819` |
| Bad-capture futility | `eval + 84 * depth + 82 * history / 1024 + 24 <= alpha`, `depth < 11`, only in the bad-noisy stage; ends the loop | `:822-834` |
| History pruning | `depth < 5 && history < -948 * depth` | `:837-839` |
| SEE | quiet `min(-12 * depth^2 + 56 * depth - 27 * history / 1024 + 27, 0)`; capture `min(-7 * depth^2 - 36 * depth - 39 * history / 1024 + 14, 0)` | `:842-850` |

Reckless keys its shallow pruning on raw `depth`, not on a reduced depth; history enters each margin directly.

**LMR** (`:861-937`), base as read at `:862`: `reduction = 269 * ilog2(depth)`. There is **no move-number logarithm** in
the base at this commit; the move number enters through `laterality`, a running sum along the line of
`max(ilog2(move_count) - 1, 0)` (`:1472-1473`). Terms: `- clamp(425 * improvement / 128, -241, 1155)`;
`- 3417 * |corr| / 1024`; `+ 1412` once the node already has an exact bound; `+ 464 * (tt_score <= alpha)`;
`+ 326 * (tt_depth < depth)`; quiet `+ 2432 - 179 * history / 1024 + 418 * clamp(alpha - estimated_score, -65, 91) /
128`, capture `+ 1687 - 130 * history / 1024`; `- 128 * min(ply - last_critical_ply, 8)`; PV `- 519 - 437 * (beta -
alpha) / root_delta`, non-PV `- 96 + 32 * laterality`; tt_pv `- 333 - 611 * (tt_score > alpha) - 685 * (tt_depth >=
depth)`, else cut node `+ 1852 + 2204 * (no TT move)`; gives check `- 955`; child cutoff count above 2
`+ 1151 + 400 * allNode`; `+ clamp(496 * (tt_move_score - singular_score - 185) / 128, 0, 2021)`; a parent reduced
much more `+ 136`; dither `(nodes + 27 * id) % 128 - 59`. Depth `clamp(new_depth - reduction / 1024, 1, new_depth + 2)
+ 2 * PV`. Re-search: `new_depth += (score > best + 57) - (score < best + 9)`. The full-depth path has its own
reduction formula with steps at 2621 and 5579 (`:939-993`).

**Singular** (`:711-756`), before the move loop: gate `depth >= 5 + tt_pv`, `tt_depth >= depth - 3`, bound not upper
(`:519-523`); margin `depth` (or `ceil(depth / 4)` on an exact bound) `+ depth * (tt_pv && !PV)`; depth
`(depth - 1) / 2`. Outcomes: `+1 / +2 / +3` with `double_margin = 195 * PV + 48 * (PV && !tt_was_pv) - 16 * quiet -
16 * |corr| / 128` and `triple_margin = 230 * PV + 56 * (PV && !tt_was_pv) - 19 * quiet - 15 * |corr| / 128 + 36`;
multi-cut returns `lerp(score, beta, 0.4027)`; if the exclusion search beat the TT score, the TT move is **dropped**
(`:745-746`); negative extension `-3` when `tt_score >= beta || cut_node`. Without a singular candidate: `+1` at
`depth <= 7` on a cut node whose score is at least 25 below alpha (`:754-756`).

**History** (`history.rs`), same gravity rule (`:77-80`):

| table | index | limit | cite |
|---|---|---|---|
| quiet | `[stm][from threatened][to threatened][from][to]` | 8192 | `:82-100` |
| pawn | `[pawn_key & 511][piece][to]` | 8192 | `:108-127` |
| noisy | `[piece][to][captured type][to threatened]` | 12800 | `:135-151` |
| continuation | `[in_check][capture][piece][to][piece][to]` | 15320 | `:227-249` |
| correction | `[fifty-move bucket of 16][stm][key & 65535]`; continuation correction as the continuation table | 14605 / 16418 | `:159-219` |

Updates (`search.rs:1092-1181`): `quiet_bonus = min(184 * depth, 1742) - 72 - 42 * cut_node`; `quiet_malus = min(171 *
depth, 1099) - 46 - 31 * quiets_tried`; `cont_bonus = min(97 * depth, 1098) - 74 - 48 * cut_node`; `cont_malus =
min(414 * depth, 949) - 49 - 17 * quiets_tried`; noisy `min(96 * depth, 885) - 43 - 87 * cut_node` and
`min(175 * depth, 1252) - 58 - 16 * noisy_tried`. The malus of the i-th tried quiet is scaled by
`1024^2 / ((1024 + 45 * i)^2 / 1024)`, an inverse-square decay. Continuation plies `{1, 2, 4, 6}`, equal weight
(`:1454-1461`). A quiet best move that needed more than one search and cut gets an extra continuation bonus
`min(233 * depth - 86, 1550)` (`:1145-1148`). Fail-low nodes (cut or PV) reward the prior quiet move with a factor
built from its move count, whether it was the TT move, and the two eval drops (`:1151-1181`). The eval-difference
update is `clamp(812 * -(eval + parent eval) / 128, -144, 324)` on the opponent's quiet history (`:492-503`).

---

## 5. Triumviratus: where it differs from SF19

The reduction and learning code is SF19's shape with every number turned into a named parameter
(`13_search.inc:434-498`, `13_learn.inc`). Parameters read: `LmrLogMul 2872`, `LmrNotImproving 197`, `LmrOffset 697`,
`LmrAllNode 276` equal SF19's; `LmrBase 998` (SF19 982), `LmrCutNode 4093` (4026), `LmrTTMove 2037` (2179),
`LmrStatDiv 377` (439) differ (`01_params.inc:656-684`). INFERRED: an SPSA re-tune of SF19's vector. Structural
differences:

- **Per-expert levers.** Many parameters have one value per material band ("expert": up to 9, 10-15, 16-23, 24+
  pieces): LMR scale and offset (`13_search.inc:441-445`), history bonus, malus, decay and gravity
  (`13_learn.inc:38-48`, `:173-175`). The stated reason: "in an endgame a quiet move that cuts among few alternatives
  says more than one among forty in the opening" (`13_learn.inc:165-166`).
- **Long-time ramp.** Selected levers shift with root depth (`lt_update`, `13_search.inc:387-407`); off when the
  offsets are 0.
- **History divisor as a curve.** SF19's 16-entry `lmrDivisor` is replaced by a parabola in depth centred on depth 8
  (`HistDivMid 3113`; `13_search.inc:174-177`). Their note: the inherited table "zigzagged by +-8% around 3083".
- **Passed-pawn pushes** are reduced less: `PassedPushRed 1125` (`13_search.inc:478-482`, `01_params.inc:843`).
- **Re-search feedback scaled by the reduction overcome**: a reduced move that clears alpha earns `FollowFhBonus +
  FollowFhRed * min(plies overcome, 8)`, and `FollowFhMiss` if the re-search refutes it (`13_learn.inc:266-273`).
- **Main history carries a threat context** (`qh_push(..., ctx, ...)`, `13_learn.inc:72`), as Reckless's quiet table.
- Built and off: `LmrGives 0` (less reduction for checks), `SingDropHash 0` (no TT-move LMR bonus when the exclusion
  search beat the TT value) (`01_params.inc:1658`, `:1675`).

### 5.1 HistGrad: the history update as a listwise ranker

`hist_grad_factors` (`13_learn.inc:132-156`); `NOVELTIES.md` §4. When a quiet move cuts, take the **ordering scores**
of the best move and of the quiets tried before it (`quiet_score[]`, `best_score`, `13_search.inc:45-46`), and form a
softmax with temperature `HistGradT = 3000` (score units):

```
w_i = exp((score_i - top) / T),  z = sum of w over the best move and the tried quiets,  ranked = their count
factor(tried quiet i) = clamp(1024 + Mix * (ranked * w_i / z - 1), 0, 3072)
factor(best move)     = clamp(1024 + Mix * (2 * (1 - w_best / z) - 1), 0, 3072)
```

with `HistGradMix = 256` of 1024 (`01_params.inc:1622-1623`). The factor multiplies the ordinary bonus or malus
(`13_learn.inc:172-178`). Effect: a best move the ordering already ranked first learns little and a surprising one
learns up to three times as much; a tried move the ordering rated low is barely punished, one it rated high is punished
in full. The factors average 1 for a uniform softmax, so the total update size is kept. Their measurement: a
non-regression pass over 11,166 games at 5+0.05; Mix 512 "did not hold" (`NOVELTIES.md:61`, `:159-160`). It is a
no-loss result, not a measured gain.

### 5.2 Causal reduction of quiet moves

`13_search.inc:122-129`, `:156-159`, `:483-486`, `:2264-2265`. When one of our quiet moves fails low, the child's
cutoff move (`f[1].refute`) is the refutation. `causal_mask` is the refuting piece's from square, its to square, and
the squares it attacks from there. Each later sibling quiet whose from and to squares are both outside the mask is
reduced `CausalRed = 512` more (half a ply). Excluded: tactical moves, checks, the hash move, passed-pawn pushes. The
mask is replaced by each newer failing quiet's refutation and cleared per node (`:2434`). Never a prune: a reduced
move that beats alpha is re-searched as usual. Their measurement: `+4.27 +- 4.13` over 6,594 games at 25+0.25
(`01_params.inc:142-145`). Extensions that failed and were sent back to SPSA: more reduction when sibling refutations
agree on a square (`CausalAgree`), and seeding the mask from the null move's refutation (`CausalNull`)
(`DEVELOPMENT_8.0.md:2082-2083`). Their own stated precedent: the null-move threat used by older engines to protect
connected moves.

Relevance to us: it needs only the child's cutoff move per ply, which our killer write already has at the cutoff site,
and it is a reduce-more rule, the direction our record says works (`history-lmr-reduce-more`).

Not verified here: the "SPSA-tuned order" of the ordering terms (`PR4`) was not read; the null-move and singular
parameters of Triumviratus were not read in context.

---

## 6. Cross-engine summary for S5

| question | SF19 | SF11 | Reckless | Triumviratus |
|---|---|---|---|---|
| LMR base | `ln(depth) * ln(moveCount)`, remaining depth | same, plies | `ilog2(depth)` + line `laterality` | SF19's |
| shallow pruning keyed on | reduced depth (`lmrDepth`) shifted by history | reduced depth | raw depth, history inside each margin | SF19's |
| null move | cut nodes, `R = 7 + d/3 + excess/256`, static-eval gate | non-PV, eval gate, statScore gate | cut nodes, skipped before a likely singular test | SF19's, per-expert levers |
| check extension | none | yes | none (checks reduce less: `-955`) | none (`LmrGives` off) |
| singular | per move in the loop, `+1..+3`, multi-cut, `-3` | `+1`, multi-cut | before the loop, `+1..+3`, multi-cut, `-3`, TT-move drop | SF19's |
| exclusion isolation | local picker; TT write, null move, TT cut gated | same idea | test finishes before the node's picker exists | SF19's |
| continuation plies written / read in pruning | 1-6 weighted / 1, 2 | 1, 2, 4, 6 / 1, 2, 4 | 1, 2, 4, 6 / 1, 2 | SF19's with weights per expert |
| malus decay over tried moves | geometric `921/1024` | none | inverse square | geometric per expert, times HistGrad |
| killers / counter move | none | yes | none seen in `search.rs` (picker not re-read) | none seen in the files read |
| extra main-history key | move flags | none | from / to under threat | threat context |
