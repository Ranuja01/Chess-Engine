# Stockfish pruning/reduction schedules — cross-version comparison (SF11 / SF15.1 / SF16 / SF17-dev / SF18 / Ethereal)

Date: 2026-07-25. Extracted by reading the actual sources on disk; every formula cites file:line.

## Source inventory (IMPORTANT corrections)

| Label | Path (abbrev. below) | Actual identity |
|---|---|---|
| SF1.1 | `stockfish_1\stockfish-11_ja\src\` | **SF 1.1 (2008) IS on disk here** — the folder's "11" means "1.1", not 11. Confirmed by Glaurung-era files absent from SF11: `book.cpp`, `color.cpp`, `direction.cpp`. Binary used by the `vs_sf1` runner sub: `stockfish_1\stockfish-11_ja\16-9\stockfish11_win32_ja.exe` (path valid). Already documented in `sf-source-evolution-bank-2026-07-06.md:5`. **This table's SF1 column was NOT extracted — re-run against this path to fill it.** |
| ~~`stockfish_1\src`~~ | `stockfish_1\src\search.cpp` | **Decoy: this top-level tree is Stockfish 18**, not SF1 (`misc.cpp:43` `version = "18"`; search.cpp 2210 lines, matching stockfish_18_linux). Ignore it; see `sf-evolution-fact-sheets-2026-07-05.md:5`. |
| SF11 | `stockfish_11\stockfish-11-win\src\search.cpp` | genuine SF11 (last pre-NNUE classical) |
| SF15 | `stockfish_15_linux\stockfish_15.1_linux_x64\src\search.cpp` | genuine SF15.1 (last classical-eval-capable) |
| SF16 | `stockfish_16\src\search.cpp` | genuine SF16 |
| SF17 | `stockfish_17\src\search.cpp` | **SF17.1-dev snapshot**, not the 17.0 release (already has correction-history families, 1024-scaled reductions, `risk_tolerance`, `priorReduction`) |
| SF18 | `stockfish_18_linux\src\search.cpp` | SF18 snapshot (adds NUMA-shared history, `ttMoveHistory`, `is_shuffling`) |
| Ethereal | fetched from github.com/AndyGrant/Ethereal `src/search.c` + `src/search.h` | WebFetch succeeded; code text verified across two fetch passes but **line numbers were inconsistent between passes → Ethereal line numbers below are approximate**; constants come from the stable `src/search.h` block |

**Unit trap #1 (reduction units):** SF11–SF16 `reduction()` returns **plies**; SF17/SF18 return **1/1024 ply** (`newDepth - r/1024`). Ethereal's LMRTable is in plies.
**Unit trap #2 (eval units):** see EVAL SCALE section — internal "pawn" ≠ displayed cp ≠ constant across versions.

---

## 1. Per-mechanism comparison

### 1.1 Late move reductions — table init

| Ver | Formula | Cite |
|---|---|---|
| SF11 | `Reductions[i] = int((24.8 + log(Threads.size())/2) * log(i))` | SF11 search.cpp:190-195 |
| SF15 | `Reductions[i] = int((20.26 + log(Threads.size())/2) * log(i))` | SF15 search.cpp:158-162 |
| SF16 | `reductions[i] = int((18.79 + log(Threads)/2) * log(i))` | SF16 search.cpp:496-497 |
| SF17 | `reductions[i] = int(2954 / 128.0 * log(i))` (=23.08·ln i; thread term gone) | SF17 search.cpp:596-597 |
| SF18 | `reductions[i] = int(2747 / 128.0 * log(i))` (=21.46·ln i) | SF18 search.cpp:606-607 |
| Ethereal | `LMRTable[depth][played] = 0.7844 + log(depth)*log(played)/2.4696` (64×64, plies) | Ethereal search.c ~149-156 (approx) |

All versions: base reduction = product of logs, `reductions[d] * reductions[mn]` (SF) or `log(d)*log(mn)/k` (Ethereal).

### 1.2 LMR — `reduction()` helper

| Ver | Formula | Cite |
|---|---|---|
| SF11 | `r = Reductions[d]*Reductions[mn]; return (r+511)/1024 + (!i && r > 1007)` | SF11 search.cpp:76-79 |
| SF15 | `return (r + 1642 - delta*1024/rootDelta)/1024 + (!i && r > 916)` — **window-width term new** | SF15 search.cpp:72-75 |
| SF16 | `return (r + 1118 - delta*793/rootDelta)/1024 + (!i && r > 863)` | SF16 search.cpp:1621-1624 |
| SF17 | `return r - delta*764/rootDelta + !i * r*191/512 + 1087` (1024ths; !improving now multiplicative +37%) | SF17 search.cpp:1755-1758 |
| SF18 | `return r - delta*608/rootDelta + !i * r*238/512 + 1182` (!improving +46.5%) | SF18 search.cpp:1735-1738 |
| Ethereal | no helper; table lookup + additive adjustments (below) | — |

`delta = beta - alpha` at the node; `rootDelta` = root aspiration window width (SF15 search.cpp:598/982).

### 1.3 LMR — gate and in-search adjustments to r

| Ver | Gate | Adjustments (sign: − = search deeper) | Cite |
|---|---|---|---|
| SF11 | `depth>=3 && moveCount > 1+rootNode+... &&` (captures only under escape clauses: moveCountPruning, `staticEval+PieceValue[EG][captured]<=alpha`, cutNode, low ttHitAverage) | ttHitAverage high −1; other-thread-marked +1; ttPv −2; `(ss-1)->moveCount>14` −1; singularLMR −2; quiets: ttCapture +1, cutNode +2, move-escapes-capture (`!see_ge(reverse_move)`) −2, statScore step rules ±1, `r -= statScore/16384`; captures: `depth<8 && moveCount>2` +1. Clamp `d = clamp(newDepth-r, 1, newDepth)` — **never extends** | SF11 search.cpp:1117-1195 |
| SF15 | `depth>=2 && moveCount > 1+(PvNode && ply<=1) && (!ttPv \|\| !capture \|\| (cutNode && (ss-1)->moveCount>1))` | ttPv&&!likelyFailLow −2; `(ss-1)->moveCount>7` −1; cutNode +2; ttCapture +1; PvNode −(1+11/(3+depth)); singularQuietLMR −1; `depth>9 && threatenedPieces&from` −1; child `cutoffCnt>3` +1; `r -= statScore/(13628 + 4000*(7<depth<19))`. Clamp `d = clamp(newDepth-r, 1, newDepth+1)` — **may extend by 1**. Adaptive re-search: `doDeeperSearch = value > alpha+64+11*(newDepth-d)` | SF15 search.cpp:1125-1204 |
| SF16 | `depth>=2 && moveCount > 1+rootNode` | ttPv −(1+(ttValue>alpha)+(tteDepth>=depth)); cutNode +(2−(tteDepth>=depth&&ttPv)); ttCapture +1; PvNode −1; repetition (`move==(ss-4)->currentMove && has_repeated`) +2; child cutoffCnt>3 +1 else ttMove r=0; `r -= statScore/14189`. `d = clamp(newDepth-r, 1, newDepth+1)` | SF16 search.cpp:945-947, 1101-1148 |
| SF17 | `depth>=2 && moveCount>1` | pre-prune: `r -= 32*moveCount`, ttPv +979; post-do_move: ttPv −(2381+PvNode*1008+(ttValue>alpha)*880+(tteDepth>=depth)*(1022+cutNode*1140)); `r += 306 − 34*moveCount`; `r -= |correctionValue|/29696`; PvNode&&\|bestValue\|<=2000 `r -= risk_tolerance(...)`; cutNode +(2784+1038*!ttMove); ttCapture&&!capture +(1171+(depth<8)*985); child cutoffCnt>3 +(1042+allNode*864) else ttMove −1937; `r -= statScore*1582/16384`. `d = clamp(newDepth-r/1024, 1, newDepth + !allNode + (PvNode&&!bestMove))` | SF17 search.cpp:1038-1285 |
| SF18 | `depth>=2 && moveCount>1` | pre-prune: ttPv +946; post: ttPv −(2719+PvNode*983+(ttValue>alpha)*922+(tteDepth>=depth)*(934+cutNode*1011)); `r += 714`; `r -= 73*moveCount`; `r -= |correctionValue|/30370`; cutNode +(3372+997*!ttMove); ttCapture +1119; child cutoffCnt>1 +(256+1024*(cnt>2)+1024*allNode); ttMove −2151 (plain if, stacks); `r -= statScore*850/8192`; allNode `r += r/(depth+1)`. `d = max(1, min(newDepth-r/1024, newDepth+2)) + PvNode` — up to +2 extension | SF18 search.cpp:1040-1047, 1191-1261 |
| Ethereal | quiets (and a flat noisy branch), after LMP checks | quiet: `R = LMRTable[d][played] + !PvNode + !improving + (inCheck && king move) − (stage < STAGE_QUIET) − hist/6167`; noisy: `R = 3 − hist/4952 − !!kingAttackers`; clamp `MIN(depth-1, MAX(R,1))` — **never extends, never <1** | Ethereal search.c ~1040-1060 (approx) |

Non-LMR reduced path (SF16+): when LMR doesn't apply, `newDepth − (r > T1) − (r > T2 && newDepth > 2)` — SF16 search.cpp:1173-1180 (`r>3`), SF17:1289-1297 (3495/5510), SF18:1264-1272 (3957/5654); `!ttMove` adds to r first (SF17 +1156, SF18 +1140).

### 1.4 Late move pruning / move-count pruning

| Ver | Formula (quiet-skip threshold) | Cite |
|---|---|---|
| SF11 | `(5 + d*d) * (1 + improving) / 2 − 1` | SF11 search.cpp:81-83, applied :1002 |
| SF15 | `improving ? 3 + d*d : (3 + d*d)/2` | SF15 search.cpp:77-80, applied :990 |
| SF16 | same as SF15 | SF16 search.cpp:63-65, applied :953-955 |
| SF17 | `(3 + d*d) / (2 − improving)` (same values, closed form) | SF17 search.cpp:81-83, applied :1054-1056 |
| SF18 | inlined `(3 + d*d) / (2 − improving)` | SF18 search.cpp:1053-1055 |
| Ethereal | table: `LMP[0][d] = 2.0767 + 0.3743*d²`, `LMP[1][d] = 3.8733 + 0.7124*d²`, only for `depth <= 8` | Ethereal search.c init ~149-156, use ~571-575; search.h `LateMovePruningDepth = 8` |

Common gate: `!rootNode && non_pawn_material(us) && bestValue not lost` (SF11:997-999, SF15:985-987, SF16:951, SF17:1052, SF18:1051). Effect = tell MovePicker to stop generating quiets, not `continue`.

### 1.5 Futility pruning — node-level (this IS reverse futility / static null move in SF)

`futility_margin`:

| Ver | Margin | Cite |
|---|---|---|
| SF11 | `217 * (d − improving)` | SF11 search.cpp:69-71 |
| SF15 | `165 * (d − improving)` | SF15 search.cpp:65-67 |
| SF16 | `futilityMult = 117 − 44*noTtCutNode; futilityMult*d − 3*futilityMult/2*improving` | SF16 search.cpp:57-61 |
| SF17 | `futilityMult = 110 − 25*noTtCutNode; futilityMult*d − improving*2*futilityMult − oppWorsening*futilityMult/3` | SF17 search.cpp:72-79 |
| SF18 | lambda: `futilityMult = 76 − 23*!ttHit; futilityMult*d − (2474*improving + 331*opponentWorsening)*futilityMult/1024 + |correctionValue|/174665` | SF18 search.cpp:879-885 |

RFP block:

| Ver | Condition → return | Cite |
|---|---|---|
| SF11 | `!PvNode && depth<6 && eval − margin >= beta && eval < VALUE_KNOWN_WIN` → `return eval` | SF11 search.cpp:831-836 |
| SF15 | `!ttPv && depth<8 && eval − margin − (ss-1)->statScore/303 >= beta && eval>=beta && eval<28031` → `return eval` | SF15 search.cpp:780-787 |
| SF16 | `!ttPv && depth<11 && eval − margin − (ss-1)->statScore/314 >= beta && eval>=beta && (!ttMove \|\| ttCapture)` → `return (eval+beta)/2` | SF16 search.cpp:761-767 |
| SF17 | `!ttPv && depth<14 && eval − margin − (ss-1)->statScore/301 + 37 − |correctionValue|/139878 >= beta && ...` → `return beta + (eval−beta)/3` | SF17 search.cpp:860-865 |
| SF18 | `!ttPv && depth<14 && eval − margin >= beta && eval>=beta && (!ttMove \|\| ttCapture)` → `return (2*beta+eval)/3` (statScore term gone; correction inside lambda) | SF18 search.cpp:887-889 |
| Ethereal | `!PvNode && !inCheck && !excluded && depth<=8 && eval − 65*max(0, depth−improving) >= beta` → `return eval` (search.h `BetaPruningDepth=8, BetaMargin=65`) | Ethereal search.c ~339-343 (approx) |

### 1.6 Futility pruning — per-move (quiets)

`lmrDepth = newDepth − reduction(...)` (SF11:1008, SF15:993) or `newDepth − r/1024` (SF17:1059, SF18:1058); SF16 `newDepth − r` (:958).

| Ver | Formula | History role | Cite |
|---|---|---|---|
| SF11 | `lmrDepth<6 && !inCheck && staticEval + 235 + 172*lmrDepth <= alpha && historySum < 25000` → continue | hard **veto** | SF11 search.cpp:1016-1024 |
| SF15 | `!inCheck && lmrDepth<13 && staticEval + 106 + 145*lmrDepth + history/52 <= alpha` → continue | continuous **margin term** | SF15 search.cpp:1024-1028 |
| SF16 | `!inCheck && lmrDepth<15 && staticEval + (bestValue < staticEval−57 ? 144 : 57) + 121*lmrDepth <= alpha`; earlier `lmrDepth += history/6437` | shifts lmrDepth | SF16 search.cpp:992-999 |
| SF17 | `futilityValue = staticEval + (bestMove ? 48 : 146) + 116*lmrDepth + 103*(bestValue < staticEval−128)`; `!inCheck && lmrDepth<12 && futilityValue<=alpha` → raise bestValue to futilityValue, continue; `lmrDepth += history/3593` | shifts lmrDepth | SF17 search.cpp:1094-1108 |
| SF18 | `futilityValue = staticEval + 42 + 161*!bestMove + 127*lmrDepth + 85*(staticEval>alpha)`; `!inCheck && lmrDepth<13 && futilityValue<=alpha` → raise bestValue, continue; `lmrDepth += history/3208` | shifts lmrDepth | SF18 search.cpp:1084-1109 |
| Ethereal | `!inCheck && eval + 77 + 52*lmrDepth <= alpha && lmrDepth<=8 && hist < {14296,6004}[improving]` → skipQuiets (search.h FutilityMarginBase/PerDepth/HistoryLimit) | hard veto (like SF11) | Ethereal search.c ~585-596 (approx) |

Continuation-history-only pruning of quiets: SF11 `contHist[0]<0 && contHist[1]<0` at `lmrDepth < 4+...` (SF11:1010-1014, threshold `CounterMovePruneThreshold=0` search.h:35); SF15 `lmrDepth<5 && history < −3875*(depth−1)` (SF15:1017-1020); SF16 `lmrDepth<6 && history < −4211*depth` (:987); SF17 `history < −4348*depth` no depth cap (:1089); SF18 `history < −4083*depth` (:1088-1089). Ethereal: 2-ply table `ContinuationPruningDepth[]={3,2}` / limits `{−1000,−2500}` (search.h).

### 1.7 Futility pruning — per-move (captures) — victim credited?

| Ver | Formula | Victim credited? | Cite |
|---|---|---|---|
| SF11 | not present (SEE-only: `!see_ge(move, −194*depth)`) | — | SF11 search.cpp:1030-1031 |
| SF15 | `!givesCheck && !PvNode && lmrDepth<7 && !inCheck && staticEval + 180 + 201*lmrDepth + PieceValue[EG][victim] + captHist/6 < alpha` | **YES** (EG value) | SF15 search.cpp:998-1005 |
| SF16 | `staticEval + 277 + 292*lmrDepth + PieceValue[victim] + captHist/7 < alpha`, `lmrDepth<7 && !inCheck` | YES | SF16 search.cpp:963-972 |
| SF17 | `staticEval + 242 + 230*lmrDepth + PieceValue[victim] + 133*captHist/1024 <= alpha`, `lmrDepth<7 && !inCheck` | YES | SF17 search.cpp:1068-1074 |
| SF18 | `staticEval + 232 + 217*lmrDepth + PieceValue[victim] + 131*captHist/1024 <= alpha`, `lmrDepth<7` (inCheck gate dropped) | YES | SF18 search.cpp:1066-1073 |
| Ethereal | no per-capture futility in main search (SEE margins only: `SEENoisyMargin=−20`, `SEEQuietMargin=−64`, `SEEPruningDepth=10`) | — | Ethereal search.h |

SEE pruning of quiets in main search: SF11 `−(32−min(lmrDepth,18))*lmrDepth²` (:1027); SF15 `−24*lmrDepth² − 15*lmrDepth` (:1031); SF16 `−26*lmrDepth²` (:1004); SF17 `−27*lmrDepth²` (:1113); SF18 `−25*lmrDepth²` (:1114). Captures: SF16 `−197*depth` (:975); SF17 `−154*depth − clamp(captHist/32, ±13x*depth)` (:1077-1078); SF18 `−max(166*depth + captHist/29, 0)` with stalemate-sac guard `(alpha >= VALUE_DRAW || non_pawn_material != PieceValue[movedPiece])` (:1077-1080).

### 1.8 Razoring

| Ver | Formula | Verified? | Cite |
|---|---|---|---|
| SF11 | `!rootNode && depth<2 && eval <= alpha − 531` → return qsearch(alpha,beta) | no | SF11 search.cpp:67-68, 822-826 |
| SF15 | `eval < alpha − 369 − 254*d²` → `value = qsearch(alpha−1,alpha); if (value<alpha) return value` — no depth/node-type gate | **yes** | SF15 search.cpp:770-778 |
| SF16 | `eval < alpha − 438 − (332 − 154*(childCutoffCnt>3))*d²` → verified qsearch | yes | SF16 search.cpp:748-757 |
| SF17 | `!PvNode && eval < alpha − 461 − 315*d²` → return qsearch(alpha,beta) | no | SF17 search.cpp:855-856 |
| SF18 | `!PvNode && eval < alpha − 485 − 281*d²` → return qsearch(alpha,beta) | no | SF18 search.cpp:873-874 |
| Ethereal | "alpha pruning": `!PvNode && !inCheck && depth<=4 && eval + 3488 <= alpha` → `return eval` (giant constant margin, no qsearch drop — a hopeless-node bailout, not a razor) | — | Ethereal search.c ~710-717 (approx), search.h `AlphaPruningDepth=4, AlphaMargin=3488` |

### 1.9 Null-move pruning

| Ver | Gate | R | Verification | Cite |
|---|---|---|---|---|
| SF11 | `!PvNode && (ss-1)->move != NULL && (ss-1)->statScore < 23397 && eval>=beta && eval>=staticEval && staticEval >= beta − 32*depth + 292 − improving*30 && !excluded && nonPawnMat && nmpMinPly-ok` | `(854 + 68*depth)/258 + min((eval−beta)/192, 3)` | `depth>=13`: re-search depth−R with `nmpMinPly = ply + 3*(depth−R)/4` | SF11 search.cpp:838-886 |
| SF15 | as SF11 but statScore<17139, `staticEval >= beta − 20*depth − improvement/13 + 233 + complexity/25` | `min((eval−beta)/168, 7) + depth/3 + 4 − (complexity > 861)` | `depth>=14`, same nmpMinPly | SF15 search.cpp:789-836 |
| SF16 | `!PvNode && (ss-1)->statScore<16620 && eval>=beta && eval>=staticEval && staticEval >= beta − 21*depth + 330 && ...` | `min((eval−beta)/154, 6) + depth/3 + 4` | `depth>=16` | SF16 search.cpp:769-808 |
| SF17 | **`cutNode`** (not just !PvNode) `&& eval>=beta && staticEval >= beta − 19*depth + 418 && ...` (statScore gate dropped) | `min((eval−beta)/232, 6) + depth/3 + 5` | `depth>=16` | SF17 search.cpp:868-906 |
| SF18 | `cutNode && staticEval >= beta − 18*depth + 350 && ...` (even `eval>=beta` dropped) | **`7 + depth/3`** — eval-vs-beta term removed entirely | `depth>=16` | SF18 search.cpp:893-925 |
| Ethereal | `!PvNode && !inCheck && !excluded && eval>=beta && (ns-1)->move != NULL && depth>=2 && nonPawnMat && TT-not-refuting` | `4 + depth/5 + min(3, (eval−beta)/191) + (ns-1)->tactical` | **none** | Ethereal search.c ~719-736 (approx) |

Post-NMP in SF17/18: `improving |= staticEval >= beta (+94 in SF17)` (SF17:908, SF18:927) — downstream LMP/reduction see a boosted improving.

### 1.10 ProbCut

| Ver | probCutBeta | Depth gate / search depth | Extras | Cite |
|---|---|---|---|---|
| SF11 | `beta + 189 − 45*improving` | `depth>=5`; qsearch verify then search at `depth−4` | move cap `probCutCount < 2 + 2*cutNode`; no TT store; returns value raw | SF11 search.cpp:888-928 |
| SF15 | `beta + 191 − 54*improving` | `depth>4`; `depth−4` | no move cap; TT-skip condition; **TT store** at depth−3 (:885); plus in-check small probcut `beta+417` (:907-919) | SF15 search.cpp:839-889 |
| SF16 | `beta + 181 − 68*improving` | `depth>3`; `depth−4` | returns `value − (probCutBeta − beta)` (:873); small probcut `beta+452` in-check (:884-888) | SF16 search.cpp:828-873 |
| SF17 | `beta + 185 − 58*improving` | `depth>=3`; `max(depth−4, 0)` | `!PvNode` gate removed; small probcut `beta+415` unconditional (:982-985) | SF17 search.cpp:919-931 |
| SF18 | `beta + 235 − 63*improving` | `depth>=3`; **`clamp(depth − 5 − (staticEval−beta)/315, 0, depth)`** — eval-dependent | returns `value − (probCutBeta−beta)`; small probcut `beta+418` (:985-989) | SF18 search.cpp:935-989 |
| Ethereal | `beta + 100` (`ProbCutMargin`) | `depth>=5`; search at `depth−4`; qsearch pre-verify only when `depth >= 10` | returns value raw; TT store depth−3 | Ethereal search.c ~738-776 (approx), search.h |

All versions: MovePicker probcut mode with SEE threshold `probCutBeta − staticEval` (captures that can plausibly reach the raised beta).

### 1.11 Singular extensions

| Ver | Entry gate | singularBeta / singularDepth | Extension outcomes | Cite |
|---|---|---|---|---|
| SF11 | `depth>=6 && move==ttMove && (bound & LOWER) && tteDepth >= depth−3` | `ttValue − 2*depth`; `depth/2` | ext ∈ {0, **1**}; sets `singularLMR` (LMR −2); multicut `singularBeta >= beta → return singularBeta`. No negative ext | SF11 search.cpp:1041-1090 |
| SF15 | `depth >= 4 − (prevDepth>24) + 2*(PvNode && tte->is_pv())`, wrapped in `ply < rootDepth*2` | `ttValue − (3 + (ttPv && !PvNode))*depth`; `(depth−1)/2` | ext ∈ {−2,−1,1,**2**}; double if `!PvNode && value < singularBeta−25 && doubleExtensions<=9`; neg: `ttValue>=beta → −2`, `ttValue<=alpha && ttValue<=value → −1` | SF15 search.cpp:1036-1106 |
| SF16 | `depth >= 4 − (completedDepth>30) + ttPv` | `ttValue − (60 + 54*(ttPv && !PvNode))*depth/64`; `newDepth/2` | ext up to **3** (`2 + (value < sB−78 && !ttCapture)`), cap `multipleExtensions<=16`; neg −1/−2/−3; recapture ext (:1076-1080) | SF16 search.cpp:1023-1080 |
| SF17 | `depth >= 6 − (completedDepth>29) + ttPv` | `ttValue − (59 + 77*(ttPv && !PvNode))*depth/54`; `newDepth/2` | `ext = 1 + (v < sB−doubleMargin) + (v < sB−tripleMargin)`; margins built from PvNode/ttCapture/ttPv/|correctionValue|; multicut `value>=beta → return value`; neg −3/−2; plus `depth++` on success | SF17 search.cpp:1132-1183 |
| SF18 | `depth >= 6 + ttPv && !is_shuffling(...)` | `ttValue − (53 + 75*(ttPv && !PvNode))*depth/60`; `newDepth/2` | as SF17, margins add `−897*ttMoveHistory/127649` and ply-vs-rootDepth terms; multicut also penalizes ttMoveHistory (`<< max(−400−100*depth, −4000)`) | SF18 search.cpp:1129-1181 |
| Ethereal | on ttMove at sufficient tte depth (singularity() helper) | `rBeta = max(ttValue − depth, −MATE)`; verify at `(depth−1)/2` | ext ∈ {−1,0,1,**2**}; double if `!PvNode && value < rBeta−16 && dextensions<=6`; multicut = abandon node (`STAGE_DONE`) when `value>=rBeta && rBeta>=beta` | Ethereal search.c ~1290-1325 (approx) |

### 1.12 History: keying and feed into reduction

Tables (declaration cites):

| Table | SF11 | SF15 | SF16 | SF17 | SF18 | Ethereal |
|---|---|---|---|---|---|---|
| main/butterfly `[color][from_to]` | movepick.h:89 (cap 10692) | movepick.h:90 (cap **7183**) | movepick.h:117 | history.h:107 | history.h:135 (keyed `move.raw()`) | yes |
| continuation `[inCheck][capture][pc][to] → [pc][to]` | thread.h:75 | thread.h:77 | movepick.h:127/133 | history.h:118/124 | history.h:145/150 | CMH+FMH only (no inCheck/capture split) |
| capture `[pc][to][capturedType]` | yes (10692) | yes | movepick.h:124 | history.h:115 | history.h:142 (10692) | yes |
| pawn history (pawn-key bucket) | — | — | movepick.h:136 (512) | history.h:127 | history.h:153-154 (Dyn, shared) | — |
| low-ply history | — | — | — | history.h:111-112 (4 plies) | history.h:139 (5 plies) | — |
| counterMoves / killers | yes | yes | yes | **removed** | removed | yes |
| correction history | — | — | 1 table `[color][pawnKey]`, **quadratic** apply `v += cv*|cv|/12475` (search.cpp:67-72) | **5 families** pawn/minor/nonPawn(W,B)/cont, linear `v + cv/131072` (search.cpp:85-97, 128-130) | as SF17 + 2nd cont ply (ss−4), NUMA-shared (search.cpp:80-100) | — |
| ttMoveHistory | — | — | — | — | history.h:216 | — |

statScore (the history→reduction channel):

| Ver | statScore | Feed | Cite |
|---|---|---|---|
| SF11 | quiets only: `mainHist + cont{0,1,3} − 4926`, clamped to 0 if all components >=0; step rules vs `(ss-1)->statScore` give ±1 | `r -= statScore/16384` | SF11 search.cpp:1165-1186 |
| SF15 | all moves: `2*mainHist + cont{0,1,3} − 4433` | `r -= statScore/(13628 + 4000*(7<d<19))` | SF15 search.cpp:1168-1175 |
| SF16 | `2*mainHist + cont{0,1,3} − 4392` | `r -= statScore/14189` | SF16 search.cpp:1130-1136 |
| SF17 | 3-way: capture `846*PieceValue[victim]/128 + captHist − 4822`; inCheck `mainHist + cont0 − 2771`; quiet `2*mainHist + cont{0,1} − 3271` | `r -= statScore*1582/16384` | SF17 search.cpp:1232-1246 |
| SF18 | capture `868*PieceValue[victim]/128 + captHist`; quiet `2*mainHist + cont{0,1}` — **no constant offset** | `r -= statScore*850/8192` | SF18 search.cpp:1215-1224 |
| Ethereal | summed quiet hist (main+CMH+FMH) | quiet `R -= hist/6167`; noisy `R = 3 − hist/4952` | Ethereal search.c ~1040-1060 |

statScore elsewhere: gates NMP (SF11:841 `<23397`, SF15:792, SF16:772); shifts RFP margin (SF15:784 `/303`, SF16:763 `/314`, SF17:862 `/301`); SF11 even biases static eval (`bonus = −(ss-1)->statScore/512`, SF11:812).

### 1.13 Qsearch delta/futility pruning — IS THE VICTIM CREDITED?

**Yes, in every SF version, per move.** The invariant expression across SF11→SF18:
`futilityValue = futilityBase + PieceValue[(EG)][pos.piece_on(to_sq(move))]`

| Ver | futilityBase | Per-move test & extras | Cite |
|---|---|---|---|
| SF11 | `bestValue + 154` (:1445) | victim EG credited (:1477); if `futilityValue<=alpha` raise bestValue & skip; secondary `futilityBase<=alpha && !see_ge(move, 1)`; guards `!inCheck && !givesCheck && !advanced_pawn_push`; **no move cap** | SF11 search.cpp:1461-1502 |
| SF15 | `bestValue + 153` (:1477) | same expression (:1522); **NEW hard cap `moveCount > 2 → continue`** (:1519-1520); recapture exempt (`to_sq != prevSq`); promotions excluded; qsearch contHist pruning (`cont0<0 && cont1<0`, :1550-1555); quiet-check-evasion cap (:1559-1563) | SF15 search.cpp:1510-1563 |
| SF16 | `staticEval + 206` (:1475) | victim credited (:1516); moveCount>2 cap (:1513); 3 SEE branches incl. `!see_ge(move, (alpha−futilityBase)*4)` (:1516-1540); flat SEE `−74` (:1555) | SF16 search.cpp:1475-1555 |
| SF17 | `staticEval + 359` (:1624) | victim credited (:1662); moveCount>2; SEE-vs-gap `!see_ge(move, alpha − futilityBase)` (:1673); contHist+pawnHist threshold `<=6290` (:1682-1687); flat SEE `−75` | SF17 search.cpp:1624-1690 |
| SF18 | `staticEval + 351` (:1604) | victim credited (:1641); moveCount>2 (:1637-1639); SEE-vs-gap with `bestValue = max(bestValue, min(alpha, futilityBase))` (:1653-1657); then **hard `if (!capture) continue`** (:1660-1662); flat SEE `−80` (:1665) | SF18 search.cpp:1604-1666 |
| Ethereal | node-level, not per-move: `if (max(QSDeltaMargin=142, moveBestCaseValue(board)) < alpha − eval − QSSeeMargin=123) return eval` — best possible victim credited **once at node entry**, whole node bails | Ethereal search.c ~992-995 (approx), search.h | |

Note: `pos.piece_on(to_sq)` under-credits en-passant (empty square → 0) — guarded by `advanced_pawn_push` (SF11) / promotion exclusion (SF15+); intentional conservative underestimate.

---

## 2. WHAT SCALES WITH WHAT (functional forms, constants stripped)

This is the load-bearing section. `d` = remaining depth, `mn` = moveCount, `Δe = eval − beta`.

- **LMR base**: `r = A·ln(d) · A·ln(mn)` — a **product of logs**. Universal from SF11 through SF18 and Ethereal. Then:
  - `− c·(beta−alpha)/rootDelta` (window-width: wide/PV-ish window ⇒ less reduction; SF15+),
  - `+ improving` correction (step in SF11–16, **multiplicative fraction of the base** in SF17/18, additive +1 in Ethereal),
  - `− c·statScore` (linear in summed history; every version; divisor order 10⁴ in ply units),
  - discrete node-character offsets: −ttPv/PvNode/ttMove, +cutNode/allNode, +ttCapture, +child-fail-high-density (cutoffCnt), −|correction| (SF17/18).
  - Form: **reduction = f(ln d · ln mn) − g(window) − h(history) ± node-type offsets**, clamped so reduced depth >= 1; since SF15 the clamp allows small extensions (never in Ethereal or SF11).
- **LMP threshold**: **quadratic in depth**, doubled by improving: `mn_max ≈ (c₀ + d²) / (2 − improving)`. Identical shape SF11→SF18 and Ethereal (Ethereal: `c₀ + c₁·d²` per improving flag, only to depth 8). No history/eval input.
- **RFP (node futility)**: prune when `eval − M(d) >= beta` with **M linear in depth**: `M = m·(d − improving·k)`, optionally minus opponent statScore (SF15–17), minus opponent-worsening (SF17/18), plus |correction| (SF17/18: uncertain eval ⇒ prune less). Depth cap grew 6→8→11→14. Return value migrated `eval` → blend toward beta `(2β+e)/3`.
- **Per-move quiet futility**: prune when `staticEval + c₀ + c₁·lmrDepth (+ history term) <= alpha`. **Linear in *reduced* depth (lmrDepth), not raw depth** — the key coupling: moves already slated for big reduction get pruned sooner. History migrates from hard veto (SF11, Ethereal) → additive margin (SF15) → shifts lmrDepth itself (SF16+).
- **Per-move capture futility**: `staticEval + c₀ + c₁·lmrDepth + Value(victim) + c₂·captHist <= alpha` — linear in lmrDepth, **victim always credited** (absent in SF11/Ethereal).
- **Razoring**: prune when `eval < alpha − M`. M: constant+shallow-only (SF11, depth<2) → **quadratic in depth with no depth cap** (SF15+: `c₀ + c₁·d²`, the quadratic self-limits). Drops to qsearch; SF15/16 verify the qsearch result, SF17/18 trust it.
- **NMP**: gate = `staticEval >= beta − a·d + b` (linear-in-depth slack: deeper ⇒ allowed further below beta). R = `base + d/3 (+ min(Δe/c, cap))` — **linear in depth**, plus a capped eval-excess term that SF18 finally deleted (R = 7 + d/3, pure depth). Verification only at high depth (13–16+) via re-search at d−R with NMP locked out for ~¾ of that subtree depth. Ethereal: shallower R (4 + d/5), keeps the Δe term, no verification.
- **ProbCut**: threshold `beta + c₀ − c₁·improving`; try captures with SEE >= threshold−staticEval at depth ≈ d−4 (SF18: d−5−Δstatic/315). Scales with improving and (SF18) staticEval-vs-beta.
- **Singular margin**: `singularBeta = ttValue − c·d` — **linear in depth** (c ≈ 1–4 internal units/ply everywhere, incl. Ethereal's exact `ttValue − depth`), verified at half depth. Extension count = staircase in (singularBeta − value): margins are constants ± node-type ± |correction| ± ttMoveHistory.
- **Qsearch futility**: `futilityBase = standPat + c` (c ≈ 0.7–1.7 pawns); per move `futilityBase + Value(victim) <= alpha ⇒ prune`. Scales **only** with stand-pat, a constant, and the victim's value — never with depth. Plus a hard capture-count cap (>2) since SF15.
- **statScore**: linear sum of history tables (weights 1–2×), fed **linearly** into r. Nothing nonlinear anywhere in the history→reduction path.
- **Correction history (SF16+)**: error `bestValue − staticEval` accumulated per structural key, scaled by depth/8; applied to static eval quadratically (SF16) then linearly (SF17/18); |correction| then modulates RFP margin, LMR, singular margins (uncertainty ⇒ prune/reduce less).

## 3. PORTABILITY (what a simpler engine can take as-is)

**Self-contained (need only depth, moveCount, staticEval, alpha/beta):**
- LMR log-product table + clamp — needs nothing else; improving/history adjustments are optional add-ons.
- LMP quadratic move-count threshold (improving optional: use the `/2` non-improving arm).
- Razoring (quadratic form, verified variant SF15:770-778 is the safest to port).
- RFP linear-margin form (SF11:831-836 is the dependency-free version).
- NMP with R = base + d/3 and the depth>=N verification re-search — fully self-contained (SF18:893-925); `nmpMinPly` needs only a per-thread int.
- Qsearch per-move futility **with victim credit** (SF11:1471-1492 form) + moveCount>2 cap (SF15+) — needs only stand-pat, PieceValue, and SEE-optional. **This is the piece our engine's node-level delta prune lacks** (cf. speed-and-qsearch-findings-2026-07-24: our SE:5383/5392 node-level prune skips the hanging-queen recapture; every SF since at least 11 credits the victim per move).

**Needs moderate machinery:**
- `improving` flag: requires storing staticEval on the stack and comparing to ss−2/ss−4 (and SF17/18 mutate it after NMP — skip that nuance).
- Per-move futility keyed to **lmrDepth**: requires computing the LMR reduction before the pruning block (SF17/18 ordering) — worth it; it is the main pruning/reduction coupling.
- statScore→reduction: requires main + continuation history; the continuation tables need [piece][to] stacks. A butterfly-only variant (Ethereal `hist/6167`) is the minimal port.
- ProbCut: needs a captures-above-SEE-threshold generator and TT store; moderate.
- Capture futility with captureHistory: works without the history term (drop `captHist/1024`).

**Depends on heavy/entangled machinery (port last or not at all):**
- Singular extensions: needs reliable TT depth/bound discipline, excludedMove plumbing, and (SF15+) double-extension accounting; interacts with LMR (`singularLMR`) and multicut. Ethereal's minimal form (rBeta = ttValue − depth, ext ∈ {−1..2}, dextensions cap) is the porting template.
- Correction history (SF16+ only): five keyed tables in SF17/18; feeds RFP/LMR/singular. SF16's single pawn-key quadratic table (search.cpp:67-72, 1343-1350) is the minimal viable version.
- ttPv/cutNode/allNode-conditioned offsets: require propagating expected-node-type and TT-PV bits; large Elo in SF but meaningless without an accurate TT/ordering substrate.
- cutoffCnt (child fail-high density), priorReduction hindsight (SF17:847-850, SF18:753-757), risk_tolerance (SF17 only, deleted in SF18) — SF-internal feedback loops, skip.
- SF16's thread-count term in the reductions table (search.cpp:496-497) — SMP-specific, dropped by SF17; skip.

## 4. EVAL SCALE (the constant-transplant trap)

| Ver | Internal pawn | Normalization to displayed cp | Cite |
|---|---|---|---|
| SF11 | `PawnValueMg = 128, PawnValueEg = 213` | none — raw internal shown | SF11 types.h:182 |
| SF15.1 | `PawnValueMg = 126, PawnValueEg = 208` | `NormalizeToPawnValue = 361`; `cp = v*100/361` | SF15 types.h:189; uci.h:38; uci.cpp:319 |
| SF16 | `PawnValue = 208` | `NormalizeToPawnValue = 356`; `cp = 100*v/356` | SF16 types.h:161; uci.cpp:48,327 |
| SF17 | `PawnValue = 208` | material-dependent WDL fit, anchor a ≈ 377 at material 58 | SF17 types.h:176; uci.cpp:516-519, 556-564 |
| SF18 | `PawnValue = 208` | same scheme, a ≈ 385 | SF18 types.h:185; uci.cpp:510, 550-559 |
| Ethereal | `SEEPieceValues[] = {103, 422, 437, 694, 1313, ...}` — pawn ≈ 103, effectively cp-native | — | Ethereal search.h |

Consequences:
- SF margins from SF15 onward are in units where **~356–385 internal ≈ 1 displayed pawn**, while `PieceValue` (used in SEE/futility victim credit) says a pawn is 126–208. These are *different* scales living in the same file; a margin like SF18's razor `485 + 281·d²` is ≈ 1.3 + 0.73·d² "displayed pawns" but 2.3 + 1.35·d² "PieceValue pawns".
- SF11 margins (razor 531, RFP 217/ply) are on the classical 213-eg-pawn scale ⇒ razor ≈ 2.5 pawns, RFP ≈ 1 pawn/ply. The *same-looking* SF15 RFP constant 165 is only ≈ 0.46 displayed pawns/ply. **Constants that look similar across versions can differ ~2× in real terms.**
- Ethereal ÷ SF18 unit ratio ≈ 2.0 (pawn 103 vs 208): divide SF18 eval-margin constants by ~2 for Ethereal scale. Reduction constants: SF17/18 are in 1/1024 ply — divide by 1024 to get plies (SF11–16 and Ethereal already in plies).
- For our engine: re-fit every margin to our own eval's pawn unit and observed eval spread; only the **forms** in §2 transfer. (This is the same failure mode as our search margins being fit to the old inflated eval.)

## 5. EVOLUTION (SF11 → SF15 → SF16 → SF17/18) and what it says about load-bearing shape

SF 1.x: source not on disk — trajectory below starts at SF11. (not found)

- **LMR**: the log·log product is invariant across 8 versions and Ethereal ⇒ **the product-of-logs is the load-bearing shape**. What evolved: (1) resolution — plies → 1/1024 plies (SF17), turning coarse ±1 tweaks into a continuous score; (2) ever more *node-character conditioning* (ttPv, cutNode/allNode, cutoffCnt, correction); (3) reduction allowed to go negative into small extensions (SF15+). Lesson: get the log·log base + history term first; the conditioning zoo is refinement.
- **LMP**: `~d²/(2−improving)` essentially frozen since SF11 (5+d² → 3+d²). A solved shape; don't innovate here.
- **RFP**: margin stayed **linear in depth** forever; what grew is the **depth cap** (6→8→11→14) — trust in the static eval rising with eval quality — and the return value softening from `eval` to `(2β+e)/3`. Lesson: linear margin is right; the aggressiveness knob is the depth cap, and it should track eval trustworthiness.
- **Razoring**: constant-margin shallow-only (SF11) → **quadratic-margin uncapped** (SF15+), with the verification qsearch appearing (SF15/16) then being dropped again (SF17/18). The quadratic replaced the depth gate. Least stable mechanism across versions — consistent with our finding that razoring has the worst node/solve ratio; SF keeps re-litigating it too.
- **NMP**: gate slack `beta − a·d + b` stable; R drifted from mostly-flat (SF11 ≈ 3.3+0.26d) to steep (d/3 + 5) and the Δe term **shrank then vanished** (cap 3 → 7 → 6 → gone in SF18). SF18's `R = 7 + depth/3` gated on `cutNode` is the simplest it has ever been ⇒ depth-linear R + verification is the load-bearing part; the eval-excess term was tuning noise. Ethereal keeps the Δe term — both work.
- **ProbCut**: shape (raised beta, SEE-thresholded captures, ~d−4 verify) unchanged since SF11; refinements are TT interplay and eval-dependent depth. A stable, portable design.
- **Singular**: margin `ttValue − c·d` invariant; the *extension ladder* inflated (1 → ±2 → ±3 with double/triple margins) and gained dampers (doubleExtensions cap → removed → ttMoveHistory/correction damping). The verification-at-half-depth trick is the core; the ladder is SF-specific tuning surface.
- **History→search**: monotone enrichment: butterfly+cont (SF11) → +statScore-as-margin instead of veto (SF15) → +pawnHistory, correction history (SF16) → 5 correction families, lowPly, capture-statScore (SF17/18), while **killers/counterMoves were deleted** (SF17). Direction of travel: fewer special-case move slots, more continuous scores feeding one linear channel into r.
- **Qsearch**: the per-move victim-credited futility (`futilityBase + PieceValue[victim] <= alpha`) is **byte-for-byte the same idea from SF11 to SF18** — the single most conserved schedule in the whole file. Additions: hard moveCount>2 cap and recapture exemption (SF15), SEE-vs-gap test, and SF18's blunt "skip all non-captures". Conservation ⇒ this is load-bearing; our node-level delta prune diverges from all six engines examined.

Cross-cutting: every mechanism that survived unchanged is a *simple monomial in depth* (linear for margins, quadratic for move counts/razor, log·log for reductions) with history/eval-certainty as **linear modulators**. Everything SF churned version-to-version was constants and node-type conditioning — exactly the parts that are least portable.
