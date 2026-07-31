# Caissa search-mining brief — 2026-07-14

Source: github.com/Witek902/Caissa @ master, `src/backend/{Search.cpp, MoveOrderer.cpp/.hpp, MovePicker.cpp, NodeCache.cpp, TranspositionTable.cpp}`. NNUE/eval files skipped. Emphasis on DIVERGENCES from the Obsidian/SF brief (staged movepicker, gravity, static threat quiet-ordering bonuses, correction history, history pruning, qsearch move-cap, cutNode-gated NMP already covered there — not re-reported except where Caissa's variant differs).

---

## 1. Movepicker / ordering / history scheme

**Staged picker** (`MovePicker::PickMove`): TT → generate+pick captures (score-sorted, promotions folded in) → killer (ONE per ply, not two) → counter-move → generate+pick quiets. Killer/counter yielded as standalone stages with fixed scores (`KillerMoveBonus=1'000'000`, `CounterMoveBonus=999'999`), removed from the quiet list afterward. Killers are cleared for `ply+1` at every node entry (`ClearKillerMoves(node->ply+1)`) — killers never leak across siblings' subtrees.

**Quiet history — THE key divergence** (`MoveOrderer.hpp`):
```
quietMoveHistory[2 stm][2 fromThreatened][2 toThreatened][64*64 from×to]
```
Butterfly from×to keying is KEPT, but split into **4 threat-context buckets** using the precomputed per-node threat bitboard (`node.threats.allThreats`, all squares attacked by the opponent): whether the from-square and the to-square are attacked. So "Nf3-e5 while f3 is hanging" and "Nf3-e5 into a defended e5" and the quiet same move in calm context all learn independently. This is contextual specificity WITHOUT piece-keying — it distinguishes *tactically bad/good instances of the same from×to move* rather than reordering by piece identity.

**Continuation history** (`continuationHistory[2 prevIsCapture][2 prevStm][2 curStm][6 prevPiece][64 prevTo] → [6*64 pieceTo]`):
- Keyed piece×to on the current move (SF-style) BUT with an extra **prevIsCapture dimension** — quiet replies to captures vs to quiets learn in separate tables.
- Pointers cached per node for plies 1..6 (`InitContinuationHistoryPointers`); **ordering** sums plies 1,2,4,6 (indices 0,1,3,5) with tuned weights /1024: 1.0, `ContWeight1=1019`, `ContWeight3=555`, `ContWeight5=582`. **Updates** hit plies 1,2,3,4,6 with weights 1.0, 1014, 300, 978, 978 (/1024). Update set ≠ read set (ply-3 updated but not read; ply-2 read at weight ~1 but statScore uses only 1,2,4).
- `moveStatScore` (used by pruning/LMR) = threat-bucketed butterfly + contHist plies 1,2,4 unweighted.

**Bonus/malus formula — scoreDiff-scaled** (`UpdateQuietMovesHistory`):
```
histBonus = min(-113 + 164*depth + 148*scoreDiff/64, 2178)
histMalus = -min(-51 + 160*depth + 155*scoreDiff/64, 1844)
```
where `scoreDiff = min(bestValue - beta, 256)` — the **margin of the fail-high scales the update**, both bonus AND malus. Separate cont constants (~same magnitudes). Gravity form `counter += delta - counter*|delta|/16384` (16384 = larger cap than SF's ~1024-style; slower saturation). Skips update entirely when `numMoves<=1 && depth<2` ("don't update uncertain moves").

**History lifecycle divergence**: `Clear()` initializes tables to **positive constants**, not zero — `QuietMoveHistoryClear=802`, `ContinuationHistoryClear=762`, `CapturesHistoryClear=346` (tuned!). Fresh moves start mildly optimistic. `NewSearch()` (each `go`) only **scales by 7/8** (`ScaleDownHistoryCounter`), keeping cross-search memory; killers zeroed.

**Capture ordering**: `WinningCaptureValue=20M` if attacker<victim, `GoodCaptureValue=10M` if equal or SEE>=0, else INT16_MIN; plus `MVV*4096` and captureHistory `[stm][piece][captured][to]` (no from-square). Underpromotions get −30M/−40M.

**Static threat escape/enter bonuses** (same family as Obsidian's, constants for reference): minors ±4000 vs pawn-attacks, rooks ±8000 vs minor-attacks, queens ±12000 vs rook-attacks — escape-from bonus and enter-to malus symmetric per piece class.

**Prior counter-move bonus on fail-LOW** (Search.cpp end of NegaMax): if node fails low and `previousMove` was quiet, bonus `min(1200, depth*120 - 100)` to the *parent's* continuation history for that move — rewards the opponent move that neutralized us. (SF has a version; Obsidian brief didn't list it.)

## 2. Reductions & pruning (constants; SIMPLER/DIFFERENT flags)

**LMR** (fixed-point, LmrScale=1024): two log-log tables, quiets `1024*(0.553 + 0.430*ln(d)*ln(m))`, captures `(0.673 + 0.420*ln(d)*ln(m))`. Additive terms (in 1024ths): nonPV +240, ttCapture +1168, killer/counter (moveScore>=CounterMoveBonus) −2688, cutNode +2928, !improving +608, givesCheck −1136; captures: winning −1008, bad −(−192), cutNode +1296. statScore term: `r -= (moveStatScore + 6877)/15` (offset ≈ their equivalent of our STATSCORE_OFFSET; div 15 on 1024-scale ≈ ours/2048 idea). **DIFFERENT**: PV low-ply term `r -= 1024*depth/(1+ply+depth)` — reduces less near root *as a smooth function of ply/depth ratio*, not a table. `r -= 208` if `ttEntry.depth >= depth` (trusted TT = calmer position). Clamp `r∈[0,newDepth]` — never drops into qsearch from LMR.
**Post-LMR**: if reduced search beats alpha: `newDepth += (score > bestValue+85) && ply < 2*rootDepth` (do-deeper), `newDepth -= (score < bestValue+newDepth)` (do-shallower), re-search only if `newDepth > lmrDepth`. Re-search is full-depth like ours otherwise.

**LMP** — SIMPLER than SF: threshold `4 + d²` improving else `4 + d²/2`, with **PV bonus applied as +2 depth inside the formula** (`depth + 2*isPvNode`). Applied on quietMoveIndex; when it fires during the quiets stage it `break`s (skips all remaining, not continue).

**Futility (per-move)** — uses **lmrDepth² not depth**: prune if `staticEval + 32*lmrDepth² + moveStatScore/383 < alpha` (depth<9, not in check). On trigger calls `movePicker.SkipQuiets()` but still lets the first quiet through (`if quietMoveIndex>1 continue`). lmrDepth = depth minus the *table* reduction for this moveIndex — pruning thresholds pre-account for the reduction the move would get.

**History pruning**: `quietMoveIndex>1 && depth<9 && moveStatScore < -(234*lmrDepth + 148*lmrDepth²)` → skip. (Obsidian had history pruning; Caissa's is lmrDepth-based and quadratic.)

**SEE pruning — cheap gate divergence**: only runs at all if `move.ToSquare() & node->threats.allThreats` (target square attacked; otherwise SEE can't lose material — skips the SEE call). Captures: depth<=5, threshold `-120*depth`, only for moveScore<GoodCapture. Quiets: depth<=9, threshold `-49*lmrDepth - moveStatScore/134` (history feeds the SEE margin).

**RFP**: depth<=6, margin `83*d + 0*d² - 145*(improving && !OppCanWinMaterial)`, floor 16. `OppCanWinMaterial` = threat bitboards show a higher piece attacked by lower (queen by rook, Q/R by minor, majors+minors by pawn) — the improving discount is **canceled when material is en prise** (threat-conditioned RFP). Return value **blended toward beta**: `(eval*(1024-525) + beta*525)/1024` — fail-soft smoothing (see §3).

**Razoring**: depth<=4, `eval + 22 + 158*d < beta` → qsearch, with **win guard** `beta < 1200` (comment: dropping to qsearch when clearly winning can lose a forced mate).

**NMP** — DIFFERENT in two ways: (1) gated `node->isCutNode` (Obsidian noted cutNode gating) AND `eval>=beta+16` only at depth<4; (2) **no separate verification search** — on null fail-high at depth>=10 it reduces the *current node's* depth by 5 and **falls through into the normal move loop** (`node->depth -= 5; if <=0 return qsearch`), i.e. verification = continue searching this node shallower. R = `3 + depth/3 + min(3, (eval-beta)/85) + improving`. Disallowed if parent or grandparent was null.

**Probcut**: depth>=5, probBeta=beta+133, captures-only picker, pre-filtered by `SEE(move, probBeta - staticEval)`; qsearch verify then reduced search `depth-4`. TT guard: skip if TT depth>=depth−3 with score<probBeta. Plus SF's in-check TT probcut (beta+329).

**Singular** — with a **negative-extension ladder**: singularDepth = `(59*depth - 215)/128` (≈ d/2 − 1.7, not SF's (d−1)/2); singularBeta = ttScore − depth (full depth, no /8 scaling!). Extension ladder: +1; +1 more if `< sBeta − 14 − 256*isPv`; +1 more if `< sBeta − 51 − 256*isPv` (PV strongly discouraged from multi-extending). Negative: multicut return is **blended** `(singularScore*singularDepth + beta)/(singularDepth+1)`; else ttScore>=beta → ext = −2−!isPv; else cutNode → −2; else ttScore<=alpha → −1. Extension budget guard `ply < 2*rootDepth`. Root-level: a time-triggered **root singularity check** (obvious-move early stop) — search root at depth/2 with window around bestScore−threshold (407→204 as depth grows, step 24/ply) excluding the best move; if it fails low, stop the search and play the move.

**IIR**: depth>=3, (cutNode || PV), reduce 1 if no TT move OR `ttDepth+4 < depth` (stale-TT also triggers, not just missing).

**Qsearch**: TT-bound-improved stand-pat (use ttScore as better eval when bounds allow); stand-pat beta-cutoff returns **blended** `(v*(1024−519)+beta*519)/1024`; move-count prune after 3 moves (when not in check & not losing); capture futility `futilityBase = standPat + 77` with SEE(1) test and `to != prevSquare` recapture exemption; bad-capture break on SEE<0; **only one check evasion tried after a best move is found in check**; captures tried are recorded (max 8) and capture history updated on cutoff *inside qsearch*.

**Aspiration**: window `6 + |prevScore|/17`, growth `w += w/3`, max 547 then jump to full. On fail-low: `beta = (alpha+beta+1)/2` (pulls beta DOWN toward alpha) and reset depth; on fail-high: `depth--` (re-search shallower, min guard `depth + 5 > targetDepth`). Score carried across iterations as a running average `avg = (avg + 3*new)/4` and the aspiration center uses that average, not the raw last score.

**In-node depth decay on alpha improvement**: every time a move raises alpha (non-cutoff), `if (depth > 2) node->depth--` — remaining later moves at this node are searched progressively shallower. Tiny code, direct tail-compressor.

**Fail-high value smoothing** (fail-soft "adjusted beta"): at every fail-high, returned value is `(bestValue*depth + beta)/(depth+1)` — deep nodes return near-true fail-soft values, shallow nodes return near-beta. Same pattern in RFP (525/1024 toward beta), qsearch stand-pat (519), qsearch final (540). Reduces score swings feeding parent TT/aspiration.

**TT**: clusters, relevance = `depth − age` replacement; same-key non-exact writes with `depth < prevDepth − 4` are REJECTED (keeps deep entries) but still refresh the move if missing; TT cutoff requires `ttDepth >= depth + (ttScore>=beta)` (one extra depth for fail-high cutoffs) and `halfMoveCount < 80` (no TT cutoffs near 50-move rule). Eval writes: static eval stored at depth 0 immediately upon evaluating any node (bound Lower, score −Inf) so siblings/re-visits skip Evaluate().

## 3. Caissa's distinctive ideas ("secrets") — neither SF-classical nor in the Obsidian brief

1. **Threat-bucketed butterfly history** (§1). Context-splits from×to by (fromAttacked, toAttacked). 4× table size, zero per-probe cost beyond two bit tests against an already-computed threat mask.
2. **NodeCache** (NodeCache.cpp; used at `ply<3`): a small hash table of near-root positions storing **per-move subtree node counts** across iterations. Two uses: (a) **ordering** — quiets get `+4096 * nodesSearched/nodesSum` (a move that ate a big subtree last iteration is ordered earlier — "effort = evidence of resistance"); (b) **time management** — fraction of nodes spent on the best root move feeds soft-limit scaling. Persistent across iterative-deepening iterations, generation-aged, halved on overflow.
3. **Tuned nonzero history INIT + 7/8 cross-search decay** (§1) — history priors are an SPSA-tuned parameter, and knowledge persists (decayed) between `go` commands.
4. **scoreDiff-scaled bonus/malus** — update magnitude ∝ how hard the fail-high beat beta (capped 256cp), on both bonus and malus, for butterfly and contHist.
5. **In-node depth decay on alpha raise** — `node->depth--` per alpha improvement (floor depth 2).
6. **Fail-soft beta-blend returns everywhere** (fail-high `(v*d+beta)/(d+1)`, RFP/stand-pat ~51% toward beta) — systematic score-swing damping.
7. **NMP "verification by continuation"** — no second null-verification search; reduce current depth by 5 and keep searching the same node.
8. **prevIsCapture dimension in contHist** — replies-to-captures learn separately.
9. **Threat-gated SEE pruning** — skip the SEE call entirely when the target square isn't attacked.
10. **Root singularity early-stop** — time-based obvious-move detection via exclusion search at the root.
11. **Aspiration beta-pull on fail-low + depth-drop on fail-high + averaged window center**.
12. **draw-by-repetition upper-bound trick** (`CanReachGameCycle`, SF has cycle detection too): in non-PV nodes with alpha<0, if a repetition is *reachable*, raise alpha to 0.

## 4. Portability table

| Idea | Verdict | Reasoning vs (a) min/max split (b) HCE (c) piece×to lesson (d) tuning coupling |
|---|---|---|
| Threat-bucketed from×to quiet history | **PORT** | (a) tables already stm-indexed; our two functions index by side — fine. (b) needs a threats bitboard per node; we have attack_bitmasks in eval — need a cheap movegen-side opponent-attack mask (we may already compute it for king safety/LMP ordering-safety). (c) **exactly matches the lesson**: keeps from×to, adds context that *demotes tactically-bad instances* (moving a piece into attack) instead of reordering strategic quiets by piece identity. (d) self-contained; init values can start 0. |
| Threat escape/enter static bonuses | ADAPT (already briefed by Obsidian) | Same threat mask prerequisite; constants above are a working starting point. |
| scoreDiff-scaled bonus/malus | **PORT** | (a/b) trivially; scoreDiff = bestValue−beta available in both minimizer/maximizer (mind the sign in the minimizer: use `beta−bestValue` equivalent margin in our absolute-score frame — compute margin as distance past the bound, always positive). (c) sharpens malus on decisive refutations = better demotion of bad quiets. (d) 4 knobs, tunable independently. |
| In-node depth decay on alpha raise | **PORT** | (a) works per-function; "alpha raise" = bound improvement in maximizer, symmetric in minimizer. (c) not an ordering change; compresses tail directly (EBF lever). (d) 1 knob (floor). Risk: interacts with our LMP/futility depth conditions — test at fixed nodes. |
| Fail-soft beta-blend returns | ADAPT | Only meaningful if we're fail-soft; if our routines are fail-hard this is a no-op/rework. Sign-symmetric blend needed per function. Low priority. |
| NodeCache (near-root node-count ordering) | **ADAPT — high interest** | (a) fine — keyed by position hash. (b) eval-free. (c) safe by the lesson: it *promotes* moves proven expensive to refute, using search effort not piece identity; near-root only (ply<3) so it cannot reorder deep strategic tails. (d) self-contained. Also gives bestMoveNodeFraction for time management later. |
| prevIsCapture contHist dimension | ADAPT | Cheap (2× table); keeps our from×to keying if we apply the dimension to OUR contHist. Untested combination (Caissa uses it with piece×to) — offline A/B. |
| Nonzero history init + 7/8 cross-search decay | ADAPT | Trivial; our engine may already decay between moves. The *tuned positive prior* is the novel bit; SPSA later. |
| NMP verification-by-continuation | SKIP for now | We'd first need to know our NMP fail-high handling; behavioral change to NMP at high depth, and NMP wasn't flagged as our pain. |
| Threat-gated SEE pruning | **PORT (micro)** | Pure speed: skip SEE when to-square unattacked. Needs the same threat mask. Byte-id risk: changes nothing semantically IF our SEE≥threshold is guaranteed true when target unattacked — verify (en-prise from-square nuance: Caissa's gate is to-square only). |
| History-fed SEE margin (`−statScore/134`) & futility statScore term (`/383`) | PORT | We already have statScore; feeding it into futility/SEE margins = "prune harder when history says the move is bad" = demote-bad-quiets applied to PRUNING not ordering — aligned with the lesson and with the goal (harder lossy prunes gated by signal). |
| lmrDepth-based pruning thresholds | ADAPT | Pre-computing the would-be reduction for futility/history-prune thresholds; we have the LMR table, cheap. |
| Singular negative-ext ladder / blended multicut | ADAPT later | We have singular banked-neutral; the ladder + `ttScore−depth` margin (no /8) is a different tuning point to revisit when singular is reopened. |
| Root singularity early-stop | SKIP | Time-management feature; our venue is fixed-node/fixed-time gauntlets — low strength ROI now. |
| Razoring win-guard (`beta<1200`) | PORT (trivial safety) | One condition; prevents qsearch-drop in winning positions. |
| TT: no-overwrite same-key if `depth<prev−4` (non-exact), cutoff `ttDepth>=depth+(ttScore>=beta)`, halfmove<80 guard | ADAPT | Self-contained TT polish; check against our TT map doc first (tt-and-cache-architecture). |
| Aspiration beta-pull / averaged center | ADAPT | Self-contained root change; low risk, small gain. |
| Killer cleared for child ply at node entry | PORT (trivial) | Prevents stale killers from sibling subtrees; check whether ours already does. |

## 5. Top 3 to try first

**#1 — Threat-bucketed quiet history (`[fromThreatened][toThreatened]` split of our butterfly + optionally our contHist).**
The single best fit for our stated problem: adds tactical context that *demotes bad quiets* (into-attack moves) while leaving from×to strategic ordering intact — the piece×to failure mode (reordering strategic quiets into the reduced tail) should not appear, because the bucket only re-ranks a move when it is tactically marked, and STS-type quiet plans mostly live in the unattacked-unattacked bucket which behaves exactly like today's table.
*Test*: build gated (`ENABLE_THREAT_BUCKET_HIST`), fixed-depth WAC (expect ↑ or =, nodes ↓) + STS move-match (**must not drop below 51.7 baseline** — kill criterion); then equal-node gauntlet vs SF18@400 vs the 51% baseline. Prerequisite: a per-node opponent-attack bitboard — measure its NPS cost separately first (byte-id lane discipline).

**#2 — statScore-fed pruning margins (futility `+statScore/383`-style term, SEE quiet margin `−statScore/134`) + lmrDepth-based thresholds.**
This is "compress the tail → prune harder safely" WITHOUT touching ordering at all: moves history already hates get pruned harder, moves it likes get protected. Directly attacks EBF; no reduction-reordering trap by construction (only demotes, via prune margins).
*Test*: three knobs off/on independently; fixed-node node-count + WAC + STS; then equal-node gauntlet. STS drop = kill (would mean strategic quiets carry low statScore — which the Step-2 decomposition data can predict in advance from our prune logs).

**#3 — In-node depth decay on alpha improvement (`depth--` per alpha raise, floor 2) + scoreDiff-scaled history bonus/malus.**
Two tiny, orthogonal, self-contained levers. The depth decay is a pure EBF/tail lever (later siblings of an improving node get shallower); scoreDiff scaling sharpens the existing bonus/malus without re-keying anything.
*Test*: each behind its own flag. Depth decay: fixed-TIME depth reached + collapse-rate on the avoidance corpus + WAC/STS; it removes nodes, so equal-node gauntlet is mandatory arbiter. scoreDiff scaling: WAC/STS ordering harness (cutoff-index histogram if ENABLE_CUTOFF_CLASS is revived).

**Runner-up worth queuing:** NodeCache near-root node-count ordering — cheap, lesson-safe (promotes by proven search effort, ply<3 only), and doubles as the instrument for best-move-node-fraction time management later.
