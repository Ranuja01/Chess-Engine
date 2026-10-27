# Ethereal search mining brief — 2026-07-14

Source: Ethereal master (AndyGrant/Ethereal), files `src/search.c`, `src/search.h`, `src/movepicker.c`, `src/history.c/.h`. Ethereal is pure-HCE lineage; current master search is eval-agnostic. All constants quoted verbatim from source. Emphasis on DIVERGENCES from the SF/Obsidian brief we already have.

## 1. Movepicker + history scheme

### Stages (movepicker.c)
TT → generate noisy → GOOD_NOISY (SEE ≥ threshold, threshold=0 in main search) → KILLER_1 → KILLER_2 → COUNTER_MOVE → generate quiet → QUIET (history order) → BAD_NOISY → DONE. `skip_quiets` jumps GOOD_NOISY → BAD_NOISY directly (killers/counter skipped too — diverges from SF, which still tries killers). Dedup vs tt/killers/counter by equality check.

### History tables (history.c) — THE headline divergence
- **Butterfly**: `history[turn][threat_from][threat_to][from][to]` — from×to butterfly, but **split 4 ways by whether the from-square and to-square are currently attacked** (`testBit(board.threats, from/to)`). `threats` = squares attacked by the opponent.
- **Capture history**: `chistory[piece][threat_from][threat_to][to][captured]` — also threat-indexed. EP/promos treated as pawn captures.
- **Continuation**: `(ns-1)->continuations[0][piece][to]` (counter) + `(ns-2)->continuations[1][piece][to]` (followup). Note: continuation IS piece×to here — but it's only 2 of 3 summed terms, and butterfly stays from×to (threat-split). This is a middle ground vs SF's everything-piece×to.
- **Quiet score** = plain unweighted sum: `cmhist + fmhist + butterfly` (`return *histories[0] + *histories[1] + *histories[2];`). No shifts/weights — simpler than SF.
- **Capture score** = `64000 * (MovePromoPiece(move)==QUEEN) + chistory` — NO MVV-LVA term in current master ordering of noisies within the good/bad split (SEE does the good/bad split; history orders inside). Divergent: SF keeps an MVV base.
- **Update (gravity)**: `delta = good ? stat_bonus(depth) : -stat_bonus(depth); *current += delta - *current*abs(delta)/16384;` — same gravity form as Obsidian/SF, `HistoryDivisor = 16384`.
- **stat_bonus**: `depth > 13 ? 32 : 16*depth*depth + 128*MAX(depth-1, 0)` — **unusual: bonus COLLAPSES to 32 above depth 13** (SF caps at a large plateau; Ethereal actively distrusts very-deep cutoffs as history signal).
- **Malus**: in `update_quiet_histories`, only the last (cutoff) move gets bonus; **every earlier tried quiet gets full-size malus** (same magnitude, sign-flipped) on all three tables. No separate malus constant — symmetric.

### Threat semantics (the "secret")
Because butterfly is keyed `[threat_from][threat_to]`, "this quiet historically works" is learned SEPARATELY for (a) escaping an attacked square, (b) moving into an attacked square, (c) neutral. It's a *learned* version of Obsidian's static ±16k/32k threat ordering — the demotion of into-attack quiets is data-driven per from/to, not a fixed constant. Eval-agnostic; needs only an "opponent attacks" bitboard (we have attack_bitmasks).

## 2. Reductions & pruning (search.c, constants search.h)

- **LMR table**: `LMRTable[d][p] = 0.7844 + log(d)*log(p)/2.4696` (64×64). Application (quiets):
  `R = LMRTable[min(d,63)][min(played,63)]; R += !PvNode + !improving; R += inCheck && moved-onto-piece==KING; R -= (mp.stage < STAGE_QUIET); R -= hist/6167;`
  Divergences: simpler than SF (no cutNode, no ttPv, no statScore complexity term); `hist/6167` = continuous history-LMR like our statScore-LMR; `stage < QUIET` = "move came from killer/counter/noisy stage → reduce less" (stage-as-signal, distinctive).
- **LMP**: depth ≤ 8; `counts[0][d] = 2.0767 + 0.3743*d²; counts[1][d] = 3.8733 + 0.7124*d²` (improving index). Quadratic in depth — SF-family uses (3+d²)/(2−improving), essentially same shape; Ethereal's is SPSA-tuned.
- **Reverse futility (BetaPruning)**: depth ≤ 8, `eval − 65*MAX(0, depth − improving) ≥ beta → return eval`. Margin 65/ply, improving folds into the DEPTH not the margin.
- **AlphaPruning** (distinctive, rarely seen): depth ≤ 4 and `eval + 3488 <= alpha → return eval` — a giant-margin "hopeless node" razor that returns eval directly, no qsearch.
- **Futility (child, quiets)**: `fmpMargin = 77 + 52*lmrDepth`; skipQuiets if `eval + fmpMargin <= alpha && lmrDepth <= 8 && hist < FutilityPruningHistoryLimit[improving]` with limits `{14296, 6004}`. **Divergence: futility is GATED BY HISTORY** — a quiet with high history is exempt from futility. Plus a second threshold `FutilityMarginNoHistory = 165` for the no-history variant.
- **Continuation-history pruning**: `stage > STAGE_COUNTER_MOVE && lmrDepth <= {3,2}[improving] && MIN(cmhist,fmhist) < {-1000,-2500}[improving] → continue`. Prunes on the MIN of the two continuation entries — "both plies say this followup is bad".
- **SEE pruning**: depth ≤ 10, `seeMargin[noisy]= -20*d²`, `seeMargin[quiet]= -64*d`; threshold softened by history: `see(move, seeMargin[isQuiet] − hist/128)` — history buys SEE slack (divergent detail).
- **NMP**: depth ≥ 2 (very low!), `R = 4 + depth/5 + MIN(3, (eval−beta)/191) + (ns-1)->tactical`; guard `!ttHit || !(ttBound&UPPER) || ttValue >= beta`. The `+ (ns-1)->tactical` term (reduce more if the previous move was a capture/promo) is Ethereal-specific.
- **ProbCut**: depth ≥ 5, `rBeta = beta + 100`, verify with qsearch then depth−4 search. Standard.
- **Singular**: depth ≥ 8, ttDepth ≥ depth−3, LOWER bound. rBeta test at reduced depth; result:
  `double_extend = !PvNode && value < rBeta−16 && (ns-1)->dextensions <= 6` → +2; `value < rBeta` → +1; `ttValue >= beta` → −1 (negative extension); `ttValue <= alpha` → −1. Multicut: `value >= rBeta && rBeta >= beta → mp.stage = STAGE_DONE` (stop the whole node). The **dextensions budget counter (≤6 per line)** is a clean explosion guard.
- **IIR**: `depth >= 7 && (PvNode || cutnode) && (ttMove==NONE || ttDepth+4 < depth) → depth−1`. Note: also fires when ttMove EXISTS but is stale (ttDepth+4 < depth) — divergent from plain "no ttMove" IIR.
- **Qsearch**: delta prune `MAX(142, moveBestCaseValue(board)) < alpha − eval → return eval` (best-case-capture bound, cheap board scan, not per-move); QSSeeMargin = 123 (only try captures with SEE > margin... generous positive margin = only clearly-winning captures late).
- **Aspiration**: WindowDepth 4, WindowSize 10, `delta += delta/2` growth; on fail-low `beta=(alpha+beta)/2` (shrink beta toward alpha — divergent nicety), depth reset to full; on fail-high depth is decremented unless near-mate.
- **Extensions**: check extension OR singular result only. No SF menagerie — much simpler.

## 3. Ethereal's distinctive ideas (neither SF-classical nor in the Obsidian brief)

1. **Threat-split butterfly/capture history** `[threat_from][threat_to]` — learned threat-conditioned ordering (Obsidian's static ±16k/32k, made adaptive). 4× table size, zero eval dependence.
2. **History-gated futility & history-softened SEE pruning** (`hist < limit[improving]` gate; `seeMargin − hist/128`) — prunes only quiets the history ALREADY calls bad. Exactly the "demote bad quiets, protect good ones" direction we want: the prune consults the ordering signal instead of position count alone.
3. **stat_bonus collapse to 32 above depth 13** — deep-node cutoffs deliberately barely move history. Protects strategic ordering from being overwritten by deep tactical noise.
4. **AlphaPruning** (eval + 3488 ≤ alpha at depth ≤ 4 → return eval, no qsearch).
5. **Stage-aware LMR** (`R -= mp.stage < STAGE_QUIET`) — trust the picker stage as a reduction signal.
6. **NMP `+ (ns-1)->tactical`** and **NMP from depth 2**.
7. **Double-extension budget `dextensions <= 6`** per line + multicut that kills the whole node via the picker stage.
8. **Continuation pruning on MIN(cmhist, fmhist)** with tiny depth caps {3,2}.

## 4. Portability table

| Idea | Verdict | Reasoning vs (a) non-negamax split (b) HCE (c) piece×to lesson (d) self-contained |
|---|---|---|
| Threat-split butterfly history | **PORT (top)** | (a) tables already side-indexed by `turn`; each of minimizer/maximizer updates its own side — no sign symmetry needed. (b) needs only attack_bitmasks (have). (c) DEMOTES into-attack quiets adaptively without re-keying good strategic quiets — orthogonal to the piece×to trade. (d) self-contained: table dims ×4 + two testBits. |
| History-gated futility (`hist < {14296,6004}`) | **PORT (top)** | Pure guard on an existing prune; protects high-history strategic quiets from LMP/futility — directly attacks our "reordered strategic quiets get pruned" failure mode. Offline-testable with prune_verify harness. |
| History-softened SEE prune (`− hist/128`) | PORT | Same character, tiny patch. |
| stat_bonus deep-collapse (32 above d13) | **PORT (cheap)** | One-line change to our bonus formula; guards strategic history from deep tactical overwrite — matches our tactical↑/strategic↓ diagnosis. |
| Continuation pruning MIN(cm,fm) < {-1000,-2500} | ADAPT | We have 2-ply contHist parked; keying from×to vs piece×to changes the constants, retune. Depth-capped {3,2} = low risk. |
| Stage-aware LMR (`stage < QUIET → R−1`) | ADAPT | We don't have a staged picker; approximate as "move is killer/counter/TT/good-capture → reduce less" (we already sort in one pass — flag those moves). |
| AlphaPruning (margin 3488, d ≤ 4) | ADAPT | Margin is eval-scale-dependent; our absolute-Black-positive eval means the alpha comparison must be done in the side-relative frame each of minimizer/maximizer already uses. Careful but simple. |
| NMP `+ prev-move-tactical`, NMP from d2 | ADAPT | R formula portable; our NMP conditions differ; test via node counts. |
| Singular double-ext + dextensions budget + stage-DONE multicut | ADAPT/LATER | We have singular banked-neutral; the budget counter and negative extension (ttValue≥beta → −1) are the parts worth stealing when we revisit. |
| Capture ordering = pure chistory (no MVV base) | SKIP for now | Requires trained capture history to be good first; our capture-hist is parked/untuned. Revisit after capture-hist turn-on. |
| Continuation piece×to keying | SKIP | Directly contradicts our measured piece×to strategic loss; Ethereal's mitigations (threat-split butterfly dominating the sum) don't transfer cleanly. |
| Aspiration beta-shrink on fail-low | PORT (trivial) | Root-only, engine-structure-agnostic. |
| Qsearch moveBestCaseValue delta prune | ADAPT | Needs a best-case-gain board scan; cheap; eval-scale constants retune. |

## 5. Top 3 to try first

1. **Threat-split butterfly history** — `history[turn][attacked(from)][attacked(to)][from][to]`, opponent-attack mask from existing attack_bitmasks at node entry. Keep everything else identical (same bonus/malus/gravity). Offline test: WAC (expect ↑ or flat — escaping/into-threat quiets order better) + **STS must not drop below 51.7%** (kill criterion); node counts at fixed depth for tail compression; then equal-node gauntlet vs SF18@400 vs 51% baseline. This is the adaptive replacement for the piece×to idea: it adds tactical discrimination WITHOUT re-keying the strategic from×to signal.
2. **History-gated futility/LMP exemption** — add `hist < LIMIT[improving]` (start {14296,6004}-scaled to our history range) as an extra condition on futility skipQuiets, and optionally exempt top-history quiets from LMP. Offline test: prune_verify labeled-corpus harness (does the gate specifically rescue moves the prune got wrong?), then STS (expect ↑ — it protects strategic quiets), WAC, equal-node gauntlet. This is the direct antidote to the LMP/LMR-eats-strategic-quiets mechanism we proved.
3. **stat_bonus deep-collapse** — change our history bonus to `depth > 13 ? small : current formula`. One line, byte-A/B-able. Test: STS (hypothesis: ↑ strategic move-match stability), WAC, fixed-depth node deltas; gauntlet only if offline signal.

Also cheap enough to bundle anytime: history-softened SEE prune margin (`− hist/128`), aspiration fail-low beta-shrink.
