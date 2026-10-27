# SEARCH AUDIT: PRUNING (2026-10-22)

**Status:** read-only reference audit. I built nothing, ran nothing and edited no other file, because an overnight queue was running.
Tags: **[E]** = evidence, with file:line or URL+function; **[S]** = my inference.
- Our line numbers are from the tree as of 2026-10-22 (HEAD `875326a`, gen 5).
- Local Stockfish paths are relative to `C:\Users\Kumodth\OneDrive\Desktop\Programming\Chess Engine\`.
- I fetched the other engines' sources from `raw.githubusercontent.com` (default branch) on 2026-10-22.

---

## 0. BOTTOM LINE

1. **Null move is our least consensus-shaped pruner.** All nine reference engines require `eval >= beta` (or a stronger static gate), non-PV (SF18/19 and Obsidian: cutNode only), an npm guard, and **no** "previous move was a capture" ban. **Ours has the opposite on every axis:**
   - no eval gate (`ENABLE_NULL_EVAL_GATE=false`);
   - it fires at PV nodes;
   - it is forbidden after any capture;
   - its material guard is type-blind;
   - R is keyed on iteration depth, with no eval term;
   - there is no verification. [E]
2. **The `NULL_NPM_GUARD` "PARKED (leans −)" verdict tested the wrong form.** Queue #90 ran `NULL_NPM_GUARD=1`, which *replaces* the legacy count rule. That **also enables null move** in K+R+P-type low-material endings, where it is refused today. The pure SF addition is mode `2` (legacy OR npm==0), and it was never run. [E `_queue_pe_rules.sh:16`, `search_engine.cpp:8811-8831`]
3. **`improving` is absent from all pruning.** 9/9 references use it, or a graded `improvement`, as a modulator: LMP count, RFP margin, futility, null gate, ProbCut margin. Its main effect in modern forms is **more** pruning when not improving (LMP count roughly halves). Our LMP count `1+rd²` ≈ SF's *improving* count at every node. [E]
4. **Our pruning depth caps are far below consensus.** In remaining-depth plies:
   - RFP ≤ 6 (consensus 7–19);
   - LMP ≤ 5 (SF none, Ethereal 8);
   - futility 2–4 (consensus lmrDepth ≤ 8–13);
   - capture SEE ≤ 3 (SF none, Ethereal 10).

   At a d10 iteration, most interior nodes sit at rd 5–9, where only null move and LMR act. [E]/[S] This is the most plausible pruning-side contributor to "SF11 d15 vs ours d10 at equal nodes". [S]
5. **Every move-loop prune runs *after* `make_move`.** That covers LMP, history, quiet SEE, capture SEE and futility (`search_engine.cpp:6844` then `:6849`). Our futility also evaluates the *child*. Every reference prunes before making the move, and LMP and futility become a cheap `skipQuiets` flag. Movegen and make are ~41% of v2 node cost, so this is a speed item as well as a shape item. [E]/[S]
6. **Mate and npm gates are missing on move-loop pruning.** 9/9 references gate shallow pruning on `bestValue > mated`. The npm(us) gate on move-loop pruning is SF-only (and Obsidian), **not** universal. So closing `PE_PRUNE_GATE` (−2σ kp) is consistent with Ethereal, Weiss, Berserk, Stormphrax and Gyatso, which lack it. [E]
7. **Closures that still hold:**
   - futility-eval-mode (cost arithmetic);
   - root razor margins;
   - the RFP margin *value* (v2 re-swept on 09-17, 1000 shipped).

   **Closures that do not hold:**
   - null eval gate and eval-R: v1, pre-A2, tested as a bundle-mate;
   - improving: v1, as a lone LMR term;
   - ProbCut: v1, non-SF form, no fire count;
   - history pruning: never measured;
   - quiet SEE pruning: v1, flat margin, per-square SEE;
   - `RFP_MAX_DEPTH=8`: v1, +19.5 ± 35.8, unresolved;
   - `NULL_NPM_GUARD`: wrong mode;
   - `RFP_RETURN_BLEND`: never run.
8. **Consensus margins in pawn units are now far smaller than ours.** RFP is 0.24–0.85 pawn per ply in modern engines; SF11 was ~1.0 eg-pawn per ply; v2 is 1.0. The "a truer eval buys pruning headroom" result says tighter margins become affordable only as eval error falls. So margins are an SPSA job **after** the structural forms land, not a first move. [E]/[S]
9. **Plan.**
   - Stage 0: per-family fire, cut and wrong-rate instruments. Most of the plumbing exists (`ENABLE_PRUNE_LOG`, `ENABLE_PRUNE_SHADOW`, `prune_verify.py`).
   - Stage 1: cheap correctness and gating (mate gate, null PV gate, npm mode 2).
   - Stage 2: null-move reform as a 2×2.
   - Stage 3: `improving` infrastructure, then LMP/RFP modulation.
   - Stage 4: history and continuation-history pruning, lmrDepth-keyed SEE.
   - Stage 5: razoring and modern ProbCut.
   - Stage 6: joint SPSA.

   Each change gets its own default-off byte-identical knob, is judged at equal time, and goes through a 2×2 before any bundle.
10. **Expected magnitude [S]:**
    - SF's own source tags sum to ~200 Elo for shallow move-loop pruning, ~40–93 for null move, ~25–50 for RFP and ≤10 for ProbCut/razoring. We have most of these mechanisms in *some* form, so the reachable gain is the **shape gap**, not the full numbers.
    - Realistic per-stage gains are +5…+25. Stage 2 (null) and Stage 3 (improving/LMP) are the likeliest to clear the ~35% node-saving visibility bar.

---

## 1. UNITS ("ratios transfer, scales never")

| engine | pawn (internal) | × to our mp (pawn = 1000) | eval type |
|---|---|---|---|
| SF11 | mg 128 / eg 213 (`types.h:182`) | mg ×7.81 / eg ×4.69 | HCE [E memory reference-engine-sources] |
| SF15.1 | mg 126 / eg 208 (`stockfish_15/stockfish_15.1_win_x64/src/types.h:189`) | ×7.94 / ×4.81 | hybrid NNUE+classical [S] |
| SF18 / SF19 | `PawnValue = 208` (`stockfish_18_linux/src/types.h:185`, `stockfish_19_linux/src/types.h:199`) | ×4.81 | NNUE |
| Ethereal | 82 / 144 (second-hand, `EVAL-V2-RUNG-PRIORS.md:169`) | ×12.2 / ×6.94 | master may be NNUE [S] |
| Weiss | unresolved (macros) | treated as ~cp ⇒ ×10 [S] | HCE [S] |
| Berserk, Obsidian, Stormphrax, Gyatso | NNUE, roughly cp-scaled [S] | ×10 [S] | NNUE |

⚠️ Which SF11 column applies depends on the phase. Our v2 pawn is ~1000 in both phases (`EVAL_V2_PAWN_MG=999`, `search_engine.cpp:2173`). I quote both columns. The v2 RFP 1000 mp/ply sits exactly at SF11's **eg** conversion of 217.

---

## 2. FEATURE × ENGINE TABLE

`d` is remaining depth in plies. `lmrD` is the depth after the would-be LMR reduction. `imp` is the improving flag. Converted margins are in our mp; the NNUE ×10 conversions are [S].

### 2.1 Reverse futility pruning (static null / "child futility")

| engine | conditions | margin | return | cite |
|---|---|---|---|---|
| **OURS** (v2 shipped) | non-PV (β−α==1), !check, β not mate, rd 1–6 | **1000·rd** (v1: 1500) | raw eval (blend knob untested) | `search_engine.cpp:5891-5905` (min), `:6576-6590` (max); `:2138` |
| SF11 | !PV, d<6, eval<KNOWN_WIN | 217·(d−imp) → mg 1695 / eg 1018 per ply | eval | `stockfish_11/stockfish-11-win/src/search.cpp:69-70, 831-836` |
| SF15.1 | !ttPv, d<8, eval≥β | 165·(d−imp) − (ss−1)->statScore/303 → ~794/ply | eval | `stockfish_15/.../search.cpp:64-66, 780-787` |
| SF18 | !ttPv, d<14, eval≥β, (!ttMove ‖ ttCapture), !loss(β), !win(eval) | (76 − 23·!ttHit)·d − (2474·imp + 331·oppWorsening)·mult/1024 + \|corr\|/174665 → ~365/ply | (2β+eval)/3 | `stockfish_18_linux/src/search.cpp:876-890` |
| SF19 | !ttPv, d<19 (6 if seekMate), eval≥β, (!ttMove ‖ ttCapture) | min(45+4d, 85) − 20·!ttHit per ply, − imp/oppW terms, + \|corr\| → ~240–410/ply | (661β+363·eval)/1024 | `stockfish_19_linux/src/search.cpp:994-1008` |
| Ethereal | !PV, !check, d≤8 | 65·max(0, d−imp) → mg 793/ply | eval | `ethereal search.c` step 7 (~484-492); `search.h:49-50` |
| Weiss | !PV, d<7, eval≥β, ttMove hist>6450 or no ttMove | 77·(d−imp) + (ss−1)->histScore/131 | eval | `weiss src/search.c:367-372` |
| Berserk | !PV, d≤10, eval≥β, ttMove hist>10790 or none | 83·d − 148·(imp ∧ !oppEasyCapture) − 25·oppDeclining | (eval+β)/2 | `berserk src/search.c:524-529` |
| Obsidian | !PV, d≤11 | max(87·(d−imp), 22) | (eval+β)/2 | `Obsidian src/search.cpp:43-45, 862-868` |
| Stormphrax | !PV, d≤12 | 85d + 7d² − 75·imp + complexity term | lerp toward β (711/1024) | `Stormphrax src/search.cpp:838-853`; `tunable.h:209-213` |
| Gyatso | !PV, d≤12, \|β\|<mate | 75d + 15d² − clamp(improvement/2, ±80) | eval − margin | `GyatsoChess search.nim:244-254` |

### 2.2 Null-move pruning

| engine | gate | R | verification | npm guard | cite |
|---|---|---|---|---|---|
| **OURS** | ply ≥ 3 (min) / 4 (max), iteration DL ≥ 5, **!prev capture**, !prev null, !check, **PV allowed**, **no eval gate** (knob off) | child target = `DEPTH_REDUCTION[DL]` − 2 − (DL≥10). SF-equivalent R = 4 at DL 5–9, 6 at DL 10–14, then +1 per iteration (DL 20 → 12). **Independent of rd, no eval term** (knob off) | none | legacy: own non-king men (incl. pawns) < 7, or < 4 with a queen | `search_engine.cpp:5908-5995`, `:6593-6675`, `:8781-8838`; `search_engine.h:76-95` |
| SF11 | !PV, prev ≠ null, (ss−1)->statScore < 23397, eval ≥ β, eval ≥ staticEval, staticEval ≥ β − 32d + 292 − 30·imp, npm(us) | (854+68d)/258 + min((eval−β)/192, 3) | d ≥ 13 (nmpMinPly) | npm(us) > 0 | SF11 `search.cpp:838-886` |
| SF15.1 | same family; staticEval ≥ β − 20d − improvement/13 + 233 + complexity/25 | min((eval−β)/168, 7) + d/3 + 4 − (complexity>861) | d ≥ 14 | npm | SF15.1 `:789-838` |
| SF18 | **cutNode**, staticEval ≥ β − 18d + 350, !loss(β) | 7 + d/3 | d ≥ 16 | npm | SF18 `:892-925` |
| SF19 | cutNode, staticEval ≥ β − 13d − 47·imp + 365, β ≥ −2000 | 7 + d/3 + max((staticEval−β)/256, 0) | d ≥ 16 | npm | SF19 `:1010-1046` |
| Ethereal | !PV, !check, eval ≥ β, d ≥ 2, prev ≠ null, TT not an upper bound below β | 4 + d/5 + min(3, (eval−β)/191) **+ (ss−1)->tactical** | none | npm | `ethereal search.c` step 9 (~504-527) |
| Weiss | !PV, eval ≥ β, eval ≥ static, static ≥ β + 138 − 13d, (ss−1)->histScore < 28500 | 4 + d/4 + min(3, (eval−β)/227) | none | nonPawnCount > (d > 8) | `weiss search.c:374-395` |
| Berserk | !PV, d ≥ 5, prev ≠ null, **!opponentHasEasyCapture**, eval ≥ β | 4 + 0.36d + min(9(eval−β)/1024, 4) | d ≥ 14 | HasNonPawn | `berserk search.c:542-585` |
| Obsidian | cutNode, eval ≥ β, static + 22d − 208 ≥ β | min((eval−β)/147, 4) + d/3 + 4 + ttMoveNoisy | none | hasNonPawns | `Obsidian search.cpp:870-890` |
| Stormphrax | !PV, d ≥ 4, static ≥ β + max(213 − 10d − 41·imp, 0), TT not upper < β | 6 + d/5 | d > 14 | nonPk | `Stormphrax search.cpp:864-913` |
| Gyatso | d ≥ 3, static ≥ β, !check, prev ≠ null | 2 + d/4 (+1 ply) | d > 14 | hasNonPawnKingPiece | `search.nim:256-292` |

### 2.3 Razoring (in-tree; distinct from our root razor)

| engine | form | cite |
|---|---|---|
| **OURS** | **none in-tree.** "Razoring" here means root razoring on presearch scores | `search_engine.h:235-252` |
| SF11 | d < 2, eval ≤ α − 531 → return qsearch (~1 Elo) | SF11 `:822-826` |
| SF15.1 | eval < α − 369 − 254d² → qsearch; return if < α | `:770-778` |
| SF18 / SF19 | !PV, eval < α − 485 − 281d² (SF19: α − 482d²) → return qsearch | SF18 `:870-874`; SF19 `:989-992` |
| Ethereal | "alpha pruning", d ≤ 4, eval + 3488 ≤ α → return eval (~3 Elo) | step 8 |
| Berserk | d ≤ 5, eval + 146d ≤ α → qsearch-verified | `:531-540` |
| Obsidian | !PV, α < 2000, eval < α − 352d → qsearch-verified | `:852-860` |
| Stormphrax | d ≤ 4, \|α\| < 2000, (!ttMove ‖ noisy), static + 347d ≤ α → qsearch-verified | `:855-862` |
| Weiss, Gyatso | none | — |

### 2.4 Move-loop pruning

| engine | LMP count (quiets) | futility (parent) | history / cont-hist prune | quiet SEE | capture SEE / futility | gates | cite |
|---|---|---|---|---|---|---|---|
| **OURS** | index i ≥ 1 + rd², rd 1–5, **i is 0-based over all moves**. After make_move. Killers/counter exempt at i ≤ 8 | **child** eval + {450, 650, 950} at rd 2/3/4 (the 200 entry is dead), DL ≥ 5, not in a "relevant pin", no history gate. After make_move. Returns the child eval | `ENABLE_HIST_PRUNE` **off**: butterfly < −512·rd, rd ≤ 4 | `ENABLE_SEE_PRUNE` **off**: post-move per-**square** SEE > 0, rd ≤ 3 | per-square SEE < −1000, rd ≤ 3 (on); no capture futility | none (PE gate off); none on mate; min side skips all at rd 1 (A3) | `cpp:5170-5255` (max), `:4743-4816` (min); `h:34, 332-339, 1354-1370, 1530-1539` |
| SF11 | (5+d²)(1+imp)/2 − 1 → skip quiets (~moveCount) | lmrD < 6, static + 235 + 172·lmrD ≤ α ∧ hist sum < 25000 (~5) | contHist[0], [1] < 0 at lmrD < 4 + … (~20) | −(32 − min(lmrD, 18))·lmrD² (~20) | −194·d (~25) | npm(us) ∧ best > mated | SF11 `:81-83, 996-1032` |
| SF15.1 | (3+d²)/(2 − imp) (~7) | lmrD < 13, static + 106 + 145·lmrD + hist/52 ≤ α (~9) | contHist sum < −3875(d−1) at lmrD < 5 (~2) | −24·lmrD² − 15·lmrD (~3) | capture futility static + 180 + 201·lmrD + victim + capHist/6 (~0); SEE −222d (~9) | npm ∧ best > TB-loss; step total ~98 | `:77-79, 984-1028` |
| SF18 | (3+d²)/(2 − imp), no depth cap | lmrD (+ hist/3208) < 13, static + 42 + 161·!bestMove + 127·lmrD + 85·(static > α) ≤ α; raises bestValue | contHist + pawnHist < −4083·d (no lmrD cap) | −25·lmrD² | capture futility lmrD < 7: static + 232 + 217·lmrD + victim + capHist; SEE −(166d + capHist/29) | npm ∧ !loss(best) | `:1049-1116` |
| SF19 | same | lmrD < 12, static + 164 + 119·lmrD + 90·(static > α); quiets skipped only off the followPV line | < −4136·d | −23·lmrD² | lmrD < 8, 234 + 247·lmrD; SEE −(177d + capHist·34/1024) | same | `:1164-1230` |
| Ethereal | d ≤ 8: [2.08 + 0.37d², 3.87 + 0.71d²] (~80) | lmrD ≤ 8: eval + 77 + 52·lmrD ≤ α ∧ hist < [14296, 6004] → skipQuiets; plus +165 without history (~5.5) | min(cm, fm) < [−1000, −2500] at lmrD ≤ [3, 2] (~10) | −64·d − hist/128, d ≤ 10 | −20·d² − hist/128 (~42 for both SEE prunes) | best > −TBWIN | `search.c:158-159, 595-641`; `search.h:49-80` |
| Weiss | > (imp ? 2 + d² : d²/2) | none | histScore < −1024·d at lmrD < 3 | −73·d at lmrD < 7 (all moves) | same | !loss(best) | `search.c:462-480` |
| Berserk | [1.43 + 0.33d², 1.77 + 0.98d²] | lmrD < 10: eval + 66 + 44·lmrD ≤ α → skipQuiets | hist < −2850(d−1), lmrD < 4, not killer/counter | −13.8·lmrD² | −108·d | best > −TB | `search.c:46-62, 646-667` |
| Obsidian | (d² + 3)/(2 − imp) | lmrD ≤ 10, static + 159 + 153·lmrD ≤ α → skipQuiets | hist < −7471·d → skipQuiets | −21·lmrD² | −96·d | npm ∧ best > TB-loss | `search.cpp:40-68, 993-1023` |
| Stormphrax | (3+d²)/(2 − imp) + history·985/2²³ | lmrD ≤ 8, \|α\| < 2000, !directCheck: static + 274 + 68d + hist/div ≤ α | quiet hist < −2242·d − 1315 at lmrD ≤ 5; noisy-history prune d ≤ 4 | −20·lmrD² | min(−111d − hist/65, 0) | !loss(best) | `search.cpp:42-50, 1030-1060`; `tunable.h:234-249` |
| Gyatso | 3 + d² at d ≤ 7, !PV | d ≤ 8: static + 75 + 75d ≤ α (not TT move or killers) | none | −50·d² | −(12d² + 41d) | \|α\| < mate | `search.nim:325-365`; `searchparams.nim:5-79` |

### 2.5 ProbCut

| engine | conditions | margin | depth | pre-verify | cite |
|---|---|---|---|---|---|
| **OURS** (off) | non-PV, !check, rd ≥ 5, finite bounds, ≤ 3 per-square-SEE ≥ 0 captures/promos | **2200** | child rd − 4 | none; no TT gate | `cpp:6707-6745` (max), `:6020-…` (min); `h:2864-2869` |
| SF11 | !PV, d ≥ 5 | β + 189 − 45·imp → mg 1476 / eg 887 | d − 4 | qsearch first; ≤ 2 + 2·cutNode tries | `:888-930` (~10) |
| SF15.1 | + TT gate (ttDepth ≥ d − 3 ∧ ttValue < pcβ ⇒ skip) | 191 − 54·imp | d − 4 | qsearch; stores TT | `:841-889` (~4) |
| SF18 / SF19 | d ≥ 3, any node, TT gate | 235 − 63·imp (SF19: 241 − 64·imp) | SF18 clamp(d − 5 − (static − β)/315); SF19 d − (imp ? 5 : 3) | qsearch; returns value − (pcβ − β); plus "small ProbCut" on a TT lower bound ≥ β + 418/428 | SF18 `:935-983`; SF19 `:1054-1104` |
| Ethereal | !PV, d ≥ 5, TT gate | +100 (×12.2 → 1220) | d − 4 | qsearch only at d ≥ 10 | step 10 (~9) |
| Weiss | !PV, d ≥ 5 | +200 | d − 4 | qsearch; returns score − 160 | `:397-433` |
| Berserk | !PV, d ≥ 6, TT gate | +168 | d − 4 | qsearch | `:589-620` |
| Obsidian | !PV, d ≥ 5, TT gate | +190 | d − 4 | qsearch; SEE ≥ (pcβ − static)·10/16 | `:897-935` |
| Stormphrax | !ttPv, d ≥ 7, (!ttMove ‖ noisy), TT gate | 307 − 10·imp | max(d − 3, 1) − 1 | qsearch | `:920-968` |
| Gyatso | none | — | — | — | — |

### 2.6 `improving`

| engine | definition | consumers |
|---|---|---|
| **OURS** | `ENABLE_IMPROVING=false`; cheap material+PST surrogate within 6 plies of the leaf; LMR only | LMR (off) — `cpp:5270-5277`, `h:1332-1346` |
| SF11 | static ≥ (ss−2) (or (ss−4) when ss−2 was in check) | RFP, null gate, ProbCut, LMP, LMR (`:828-829`) |
| SF15.1 | graded `improvement` = static − (ss−2 or ss−4), else 168; improving = > 0 | + null gate uses improvement/13 (`:765-768`) |
| SF18 / SF19 | static > (ss−2); + `opponentWorsening` = static > −(ss−1); `improving \|= static ≥ β` after null | RFP (with oppW), ProbCut, LMP, LMR, null gate (SF19) |
| Ethereal / Weiss / Berserk / Obsidian / Stormphrax / Gyatso | static > (ss−2), some falling back to ss−4 | LMP everywhere; RFP in 5/6; futility history limit (Ethereal); cont-pruning depth (Ethereal); LMR |

---

## 3. PER FEATURE: CONSENSUS, DISAGREEMENT, EVOLUTION, REPORTED ELO

Elo figures are the engines' **own in-source annotations** [E]; I did not cross-check them against fishtest. They are measured at their authors' TC on their own eval. They give an order of magnitude only.

### 3.1 RFP
- **Consensus (9/9) [E]:**
  - non-PV only;
  - margin linear in depth, sometimes plus a small quadratic term;
  - margin reduced by `improving` (8/9: everyone except Berserk's variant form, and ours);
  - mate / decisive guards.
- **Disagreements:**
  - return value: eval (SF11/15, Ethereal, Weiss) vs pulled toward β (SF16+, Berserk, Obsidian, Stormphrax);
  - depth cap: 6–19;
  - TT-move guard: SF18/19 skip if a quiet TT move exists; Weiss/Berserk require the TT move's history to be high;
  - context terms: SF adds |correction| and `opponentWorsening`; Weiss/SF15 subtract the parent's statScore.
- **Evolution SF11→19 [E]:**
  - cap d<6 → <8 → <14 → <19;
  - margin 1.0 eg-pawn/ply → ~0.8 → ~0.37 → ~0.24–0.41;
  - return eval → blended toward β (SF16+).
  - Trend: **deeper, tighter, softened return**.
- **Elo:** SF11 "~50"; SF15.1 "~25"; Ethereal "~32" [E in-source]. Our own shipped measurement: +73 (v1, 659 g) [E `external-audit-2026-07-03.md:6`].

### 3.2 Null move
- **Consensus (9/9) [E]:**
  - eval/static ≥ β (Gyatso: static ≥ β);
  - prev ≠ null;
  - npm guard;
  - R grows with depth (and with eval−β in 6/9);
  - non-PV (8/9; SF18/19 and Obsidian go further to cutNode only);
  - unproven mates clamped to β.
- **Verification** is in 5/9 (SF, Berserk, Stormphrax, Gyatso), always only at d ≥ 13–16.
- **Disagreements:**
  - Ethereal *adds* R after a tactical previous move;
  - Berserk *skips* null when the opponent has an easy capture;
  - Weiss forbids null at d > 8 with only one piece.
  - Nobody forbids null after any capture as we do.
- **Evolution [E]:** SF11 → SF15 added complexity/improvement terms. SF18 dropped `eval ≥ β` in favour of a stricter staticEval gate, made null **cutNode-only**, and raised R to 7 + d/3. Verification moved d13 → d14 → d16.
- **Elo:** SF11 "~40", SF15.1 "~22", Ethereal "~93" [E in-source].

### 3.3 Razoring
- **Consensus:** weak. 7/9 have some form; Weiss and Gyatso have none.
- The modern form verifies with qsearch and uses a quadratic SF or linear Obsidian/Berserk/Stormphrax margin.
- **Elo:** SF11 "~1", Ethereal alpha-pruning "~3".
- **Evolution [E]:** SF went from d<2 only to a quadratic margin at any depth (SF15+), dropped its constant term in SF19, and keeps it non-PV.

### 3.4 Futility (parent)
- **Consensus (7/9; Weiss has none; ours uses a child form):**
  - parent static eval + base + slope·**lmrDepth** ≤ α;
  - history folded in (SF15+, Ethereal, Stormphrax), either as a gate or as an lmrDepth shift;
  - not in check;
  - typically implemented as `skipQuiets`, so it is done before move generation for later quiets.
- **Disagreements:** whether to raise bestValue to futilityValue (SF18/19 do); the lmrDepth cap (6 → 13).
- **Evolution [E]:**
  - lmrD < 6 → < 13;
  - slope 172 mg → 145 → 127 → 119 (smaller per lmr-ply);
  - history gate → history inside lmrDepth;
  - capture futility added in SF15.
- **Elo:** SF11 "~5", SF15 "~9", Ethereal 14A+14B "~5.5" [E in-source]. A small feature everywhere.

### 3.5 LMP / move-count pruning
- **Consensus (9/9):** an a + b·d² count, split by `improving` (8/9; Gyatso has no split). The non-improving count is about half the improving count.
- **Disagreements:**
  - depth cap: SF none, Ethereal ≤ 8, Gyatso ≤ 7;
  - history scaling of the count (Stormphrax);
  - PV exemption (Gyatso only).
- **Evolution [E]:** SF11 `(5+d²)(1+imp)/2−1` → SF15+ `(3+d²)/(2−imp)`. Almost identical in shape; slightly more aggressive.
- **Elo:** SF15.1 "~7" (move-count alone); Ethereal "~80" [E in-source]. The gap reflects what each engine's other pruners already cover.

### 3.6 History / continuation-history pruning
- **Consensus (7/9; Gyatso and our default lack it):** prune a quiet whose continuation or combined history is below −k·depth, at small lmrDepth.
- **Disagreements:** which tables (SF: cont[0] + cont[1] (+ pawnHist); Ethereal: min(cm, fm); Weiss/Berserk/Obsidian: combined); whether to skipQuiets or continue.
- **Evolution [E]:** SF11 hard-coded `< 0` on two tables (~20 Elo) → SF15 −3875(d−1) (~2) → SF18 −4083·d with no lmrD cap.
- **Elo:** SF11 "~20", Ethereal "~10".

### 3.7 SEE pruning
- **Quiets:** margin ∝ −lmrD² in SF/Berserk/Obsidian/Stormphrax/Gyatso (d²); ∝ −d in Ethereal/Weiss.
- **Captures:** ∝ −d (SF, Berserk, Obsidian, Stormphrax, Weiss) or −d² (Ethereal); history-adjusted in SF18+, Stormphrax and Ethereal.
- **Depth:** no cap (SF), ≤ 10 (Ethereal), lmrD < 7 (Weiss).
- SF18+ has a stalemate guard (do not prune the sacrifice of the last piece when α < draw).
- **Elo:** SF11 quiet ~20, capture ~25; SF15 ~3 / ~9; Ethereal both ~42 [E in-source].
- Every reference uses a **per-move** SEE. Ours is per-square [E memory `speed-and-qsearch-findings-2026-07-24.md:73` via preplan §1 #12].

### 3.8 ProbCut
- **Consensus (8/9; Gyatso has none):** non-PV, d ≥ 3–7, qsearch pre-verify, TT gate (6/8), depth d − 4, margin β + ~1.0–1.5 pawn, improving-scaled in SF and Stormphrax.
- **Evolution [E]:** SF added the TT gate (SF15), TT store, depth ≥ 3 and a dynamic depth (SF18), plus a second "small ProbCut" on a TT lower bound.
- **Elo:** SF11 "~10", SF15 "~4", Ethereal "~9".

### 3.9 Mate / npm gates
- `bestValue > mated` (or `!is_loss`) on move-loop pruning: **9/9** [E].
- `npm(us)` on null: **9/9** [E].
- `npm(us)` on move-loop pruning: **SF11/15/18/19 and Obsidian only**; not Ethereal, Weiss, Berserk, Stormphrax or Gyatso [E]. This is SF-family style, not consensus.

---

## 4. OURS VS CONSENSUS: GAP BY GAP, AND WHICH CLOSURES STILL HOLD

| # | gap | consensus | ours | closure on record | premise now | verdict |
|---|---|---|---|---|---|---|
| G1 | **Null eval gate** | 9/9 | none (`ENABLE_NULL_EVAL_GATE=false`) | 07-03 "bundle-mate dud (node-adder)" on v1 [E external-audit:6]; gauntlet neutral, sign-unstable (preplan #10) | v1 eval; tested **as a bundle-mate**; pre-A2. With no gate, null runs where it almost surely fails: wasted nodes, no unsound cuts [S] | **REOPEN, H** |
| G2 | **Null at PV nodes** | 8/9 forbid (2/9 cutNode only) | allowed | never considered | — | **OPEN, H** (also removes A4's PV-max wrong-bound issue) |
| G3 | **"no null after capture"** | 0/9 have it; Ethereal does the opposite (R+1) | `!last_move_was_capture` | never tested | an eval gate makes it mostly redundant [S]; it also blocks null in every recapture-resolved quiet node | **OPEN, M**: test as a pair with G1 |
| G4 | **Null R** keyed on iteration depth, flat, no eval term | depth-scaled + eval term (6/9) | `DEPTH_REDUCTION[DL]`, constant over rd; eval-R knob off | eval-R tested in the 07-08 campaign (v1); `NULLMOVE_PROGRESSIVE` dead knob [E] | v1; R not keyed on rd is unique to us | **REOPEN, M**: replace with R = a + rd/b + min((eval−β)/c, 3) |
| G5 | **Null npm guard** | npm(us) > 0 | type-blind count rule | `NULL_NPM_GUARD=1` "PARKED, leans −" (#90) | **wrong form**: mode 1 also *drops* the count rule, so it newly allows null in low-material endings with no verification. Mode 2 untested | **REOPEN, H**: mode 2 first; mode 1 only paired with verification |
| G6 | **Null verification** | 5/9, only at d ≥ 13–16 | none (`VERIFY_*` is LMR near-miss verification, not null) | — | rarely fires at our depths (iteration ~12) [S] | **L**, but needed if G5 mode 1 or 3 is ever used |
| G7 | **RFP depth cap 6** | 7–19 | 6 | `RFP_MAX_DEPTH=8` v1: −25 STS, +19.5 ± 35.8 / 500 g (unresolved) [E OPTIMIZATION_LOG:1034, 1844] | v1; never with improving | **REOPEN, M** (with G9) |
| G8 | **RFP return value** | SF16+, Berserk, Obsidian, Stormphrax blend toward β | raw eval; `RFP_RETURN_BLEND` built, **never run** [E preplan #9] | — | — | **OPEN, M**: cheapest arm in the report |
| G9 | **`improving`** | 9/9, modulating RFP, LMP, null, ProbCut, futility | absent from pruning | shelved 3× as an LMR-only term on v1 with full/cheap eval costs [E memory improving-heuristic-shelved] | **changed**: v2 eval is 11.5% of node cost (v1 32.8%); the closure tested a different consumer (LMR) | **REOPEN, H**, as a modulator (LMP first) |
| G10 | **LMP count** | (3+d²)/(2−imp), no or ≤ 8 cap | `1 + rd²`, cap 5, 0-based index, after make_move | `LMP_BASE=0` games +2.6 ± 40 (v1); `LMP_HIST_EXEMPT` loses [E SEARCH-SWEEP §5] | our count ≈ consensus *improving* count; the non-improving halving is missing | **REOPEN, H** via G9 |
| G11 | **Futility form** | parent eval + lmrDepth slope + history, skipQuiets, before make | child eval, 3 rd buckets, after make, no history | `FUTILITY_EVAL_MODE` closed (cost arithmetic); margin non-monotone [E SEARCH-SWEEP §5] | cost-mode closure **holds**. The *form* (parent + history + lmrDepth, before make) was never tested | **OPEN, M** as a *second* gate in front of the child-eval one (idea I5) |
| G12 | **History pruning** | 7/9 | off, butterfly | **never measured** in games [E preplan #14] | — | **OPEN, M**: needs the piece×to cont-hist |
| G13 | **Quiet SEE pruning** | lmrD² margin, per-move SEE | off; flat 0 margin rd ≤ 3; per-square | v1: −7.1% nodes, 240 WAC; seep_d5/d7 −315/−483 STS [E memory margin-sweep] | the closure measured a *harsher* form than consensus (0 margin at rd 3, where SF allows −225 cp-equivalent) | **REOPEN, M**, after a per-move SEE |
| G14 | **Capture SEE** | −k·d, depth uncapped or ≤ 10, history-adjusted | flat −1000, rd ≤ 3, per-square (least-valuable attacker ⇒ optimistic ⇒ under-fires [S]) | shipped (+18 bundle) | holds as "keep"; the shape gap is open | **M** (SPSA later) |
| G15 | **Capture futility** | SF15+, (Gyatso via SEE) | none | — | ~0 Elo in SF15.1 | **L** |
| G16 | **In-tree razoring** | 7/9 | none | root razor margin closure **holds** (different mechanism) [E SEARCH-SWEEP §6b] | — | **OPEN, L-M** (qsearch-verified form is sound) |
| G17 | **ProbCut form** | qsearch pre-verify, TT gate, margin ~1.0–1.5 pawn | no pre-verify, no TT gate, 2200 | v1 "phantom scores", no fire counter ever read [E preplan #4] | self-verifying ⇒ the eval-noise premise never applied; margin ~1.5× too wide | **REOPEN, M** |
| G18 | **Mate gate on move-loop pruning** | 9/9 | none | never considered | correctness | **OPEN, H** (cheap) |
| G19 | **npm gate on move-loop pruning** | SF and Obsidian only | `PE_PRUNE_GATE` closed (−2σ kp) | #90 fixed-node move test | consistent with 5/9 references lacking it | **HOLDS** |
| G20 | **Prune before make_move** | 9/9 | after make_move (`cpp:6844` → `:6849`) | — | cost only: ~41% of node cost is movegen/make [E memory the-nps-gap…] | **OPEN, M** (speed; node-identical if done right) |
| G21 | **Min/max asymmetry at rd 1** | n/a | minimizer skips all move-loop pruning at rd 1 (A3) [E SEARCH-CORRECTNESS §A3] | `FIX_MIN_RD1=1` −15 WAC (fixed depth only) | productive at fixed depth; equal time unmeasured | every pruning change acts **only on the max side at rd 1** — read fire counts per side |

**Closures that still hold:**
- `FUTILITY_EVAL_MODE` (cost arithmetic);
- `QSTANDPAT_EVAL_MODE`;
- root razor margins;
- `ROOT_RAZOR_CONTINUE` (provably inert);
- `NULLMOVE_PROGRESSIVE` (dead knob);
- the RFP margin *value* at the current form (1000, v2-measured 09-17);
- `PE_PRUNE_GATE` (and it is not a consensus feature anyway).

**Closures that do not hold, and why:**
- **v1-era:** null eval gate, eval-R, improving, ProbCut, quiet SEE prune, `RFP_MAX_DEPTH=8`, `LMP_BASE`.
- **Measured pre-A2** (the minimizer scout had the wrong bound): everything above.
- **Dead or starved mechanism:** ProbCut had no fire counter ever read.
- **Wrong form:** `NULL_NPM_GUARD=1`; quiet SEE with a flat margin; improving as LMR-only; null gate tested as a bundle-mate.
- **Never measured:** `RFP_RETURN_BLEND`, `ENABLE_HIST_PRUNE`.

---

## 5. IMPROVEMENT IDEAS [S], each with its test

**I1. Consensus null-move core: eval gate + non-PV + remove the after-capture ban.**
- Gate on `eval_side(rfp_static_eval) ≥ bound`. Today `rfp_static_eval` is computed only at non-PV rd 1–6, so the gate needs the node-entry eval at every null-eligible node. v2 makes this cheap.
- Gate on `(beta − alpha) == 1`.
- Replace `!last_move_was_capture` with the eval gate.
- **Prediction:** −8…−20% quiet nodes at fixed depth; null attempts −40…−70% with a higher cut rate; WAC within ±3; equal-time Elo +0…+15.
- **Test:** fire counters (attempts, cuts, by eval-gate side), quiet-corpus nodes, a 2×2 against the capture-ban removal, then equal-time fastchess.

**I2. `NULL_NPM_GUARD=2`** (the SF addition alone: legacy OR pawns-only).
- **Prediction:** fires only in pawn-only nodes with ≥ 7 pawns; divergence mostly on kp sets; move test better ≥ worse.
- **Test:** `_fix_firecount.py`-style divergence plus an SF18 d16 move test on kp_dense/eg_critical (the #90 protocol), then equal-time games from K+P starts.
- Separately, mode 1 plus verification at rd ≥ 8 is a second arm. Never mode 1 alone.

**I3. `improving` as a pruning modulator** (one node-entry eval per non-check node, stored in `g_evalStack`).
- **Frame caveat:** improving must be computed in the **side-to-move frame**: max side "higher is better", min side "lower is better". The `static_eval_for_improving` sign bug of 06-13 is the warning [E memory].
- Consumers, one knob each:
  - (a) LMP count `(BASE + rd²)` for improving, `(BASE + rd²)/2` otherwise;
  - (b) RFP margin `RFP_MARGIN·(rd − imp)`;
  - (c) ProbCut margin;
  - (d) null gate offset.
- **Order:** (a) first. It is the only *pruning-increasing* consumer and the likeliest to clear the 35% bar.
- **Prediction for (a):** −10…−25% nodes, WAC −0…−5, equal-time +5…+20.
- **Test:** the LMP shadow-search wrong-rate (`ENABLE_PRUNE_SHADOW`) split by improving, then equal-time games.

**I4. Raise the RFP cap to 8–10 with improving.**
- Only after I3(b). Pair it with `RFP_RETURN_BLEND` 50/67 (the cheapest arm here; run it first, alone).
- **Test:** `prune_verify` wrong-rate at rd 5–8. It must stay under ~1% (rd 3 is 1.0% today [E memory rfp-is-99.8]). Then equal time.

**I5. Two-stage futility (original to our engine): a parent-eval pre-gate *before* make_move, in front of the existing child-eval gate.**
- Parent `static + A + B·lmrD + hist/C ≤ bound` ⇒ skipQuiets with no make_move.
- Survivors keep today's child-eval test.
- It keeps the accuracy of our child-eval futility (the record says it is more accurate per fire) and buys the giants' cost profile.
- **Test:** fire counts for each gate; quiet nodes and NPS; equal time.

**I6. Move LMP and futility before make_move.**
- LMP's decision does not need the child position; `move_is_check` is the only child-side input. Compute gives-check from the parent, or keep LMP after make for checking candidates only.
- Purely speed if node-identical.
- **Test:** byte-identical node counts, then NPS on quiet d12. ⚠️ Our pruned moves still enter some counters and the movegen cache; verify the fingerprint.

**I7. Continuation-history pruning** on the piece×to re-keyed tables (preplan #15 adopts them on engineering merit).
- Form: `cont1 + cont2 < −k·rd` at rd ≤ 3–4, killers/counter exempt (as Berserk).
- **Test:** wrong-rate via shadow search; nodes; equal time. Never bundle with I3(a) until the 2×2: both act on the quiet tail [E memory ordering-does-not-buy-index-keyed-pruning].

**I8. Per-move SEE, then lmrDepth-quadratic quiet SEE pruning** (−k·lmrD² with k ≈ SF18's 25/208 ⇒ ~120 mp·lmrD²), plus depth-scaled capture SEE (~−800·d mp, d ≤ 6).
- **Test:** perft-style SEE unit tests against python-chess on 10k moves first; then nodes and the WAC veto; then equal time.

**I9. qsearch-verified in-tree razoring on both halves** ("bad-side RFP"): non-PV, rd ≤ 3, eval far past the node's losing bound ⇒ return qsearch if it confirms.
- Our max/min structure makes it the mirror of RFP_MIN/RFP_MAX in the opposite direction.
- Low expected Elo; cheap.
- **Test:** fire counts, `prune_verify`, nodes.

**I10. Modern ProbCut:**
- qsearch pre-verify;
- TT gate, using the child-probe TT already used by singular (`accessSearchEvalCache`);
- margin 1200–1500;
- per-move SEE ≥ (pcβ − static) filter;
- return value − (pcβ − β).
- **Test:** the fire counter (never read before) → sweep the margin on quiet nodes → equal time.

**I11. Mate gate** (`best > −MATE_BOUND` on max, `< +MATE_BOUND` on min) on all move-loop pruning; RFP and null already have one.
- **Prediction:** rarely fires; byte-identical on WAC except mate positions.
- Ship on correctness (gen-5 style) if no harm.

**I12. (Original) Phase-conditioned margins for our known eval weakness.**
- The autopsy shows 136/174 SF18 losses decided in the early middlegame, positional, eval-caused [E SEARCH-CORRECTNESS #88]. SF18/19 widen RFP by |correctionValue|; Stormphrax by "corrplexity".
- Without corrhist, a cheap analogue is phase-scaled RFP and futility margins: wider at full material, tighter in endgames, where the v2 static eval beats SF11 (0.87×).
- **Test:** fit by SPSA on two parameters; judge on the SF18 move test at early-middlegame divergences and equal-time games. Corrhist (if reopened) would supersede it.

**I13. (Original) Use the presearch's per-root-move score drop as a pruning confidence signal at ply 1–2.**
- When a root move's presearch score sits far below the best, its subtree could take a tighter RFP/LMP. That is a soft form of root razoring, which the record says earns its nodes.
- **[S], speculative.** Test only after I1–I4. Fire counts first.

**I14. Interaction with depth caps (the SF11 depth gap).**
- Our caps (LMP 5, RFP 6, futility 4, SEE 3) are fixed in rd. At iteration 10–14, the rd 5–9 band gets only null and LMR.
- After I3/I4, raise caps one at a time (LMP 5 → 8 with improving, capture SEE 3 → 6).
- **Test:** depth reached at 100k/250k/1M nodes on the preplan §5 instrument, not d10 benches.

---

## 6. STAGED BUILD + TEST PLAN

**Rules for every arm:**
- one knob, default = today's behaviour, byte-identical fingerprint at 0 (v2 shipped gen 5; v1 control `250 / 35,310,778`);
- write both halves (min and max) in the same commit;
- symmetric by construction, checked with a colour-mirror search test on the K+P and quiet sets;
- register predictions before running;
- equal-**time** fastchess through the UCI binary is the deciding instrument; fixed-depth benches are a cost bound and a tactical veto only;
- **2×2 before any bundle** [E memory search-changes-are-antagonistic-not-additive];
- pooled fresh seeds;
- ship rule: self-play ≥ +2σ, SF18 > −1σ pooled, Mediocre not negative.

**Stage 0 — instruments (no games).**
1. Extend `[PRUNEFIRE]` and the fire counters to null (attempt / cut / TT-hit / eval-side), LMP, futility, quiet and capture SEE, history prune and ProbCut, **per side** (min/max) and per rd.
   - `ENABLE_PRUNE_LOG`, `log_prune_fire` and `ENABLE_PRUNE_SHADOW` / `shadow_record` already exist at the LMP and futility sites [E `cpp:5183-5190, 5240-5253`].
2. Run `prune_verify.py` wrong-rates for each family on gen 5. The 99.8% RFP figure is v1 at margin 1500.
3. Quiet-corpus nodes and depth at 100k/250k/1M (preplan §5).
- **Prediction:** null attempt→cut rate < 50% (no eval gate); LMP wrong-rate < 2%; futility fires on 86–88% of what it examines (as on v1).

**Stage 1 — cheap correctness / gating (each alone, then as a bundle only after pairwise 2×2s).**
- I11 mate gate.
- G2 null non-PV (subsumes A4).
- I2 `NULL_NPM_GUARD=2`.
- I6 prune-before-make (node-identical speed item).
- **Predictions:** I11 ≈ 0; G2 −1…−5% nodes, ≈ 0 Elo; I2 ≈ 0 general, kp move test better > worse; I6 identical nodes with +3…+10% NPS.

**Stage 2 — null-move reform.**
- 2×2 of {eval gate} × {drop after-capture ban}.
- The winning cell then goes through a 2×2 with {rd-keyed R + eval term (G4)}.
- Verification only if G5 mode 1/3 is pursued.
- **Prediction:** the gate cell gives −8…−20% nodes and +0…+15 Elo at equal time; the corner is ≤ the best cell (anti-additivity prior).

**Stage 3 — `improving`.**
1. Infrastructure: node-entry eval at all non-check nodes. This costs NPS; measure it alone, expected −2…−5%.
2. Then I3(a) LMP, which must recoup the infrastructure cost.
3. Then I3(b) RFP − imp.
4. Then I4 cap and blend. Run `RFP_RETURN_BLEND` 50/67 standalone at any stage; it is the cheapest arm.
- Each step is judged against the previous step, not against gen 5.

**Stage 4 — move-loop shape.**
- I7 cont-hist pruning; I8 per-move SEE → lmrD² quiet SEE + depth-scaled capture SEE; I5 parent-eval futility pre-gate.
- 2×2 of I7 against I3(a).

**Stage 5 — razoring / ProbCut.**
- I9 and I10. Fire counters first; margin sweep on quiet nodes; equal time.

**Stage 6 — joint SPSA.**
- Over the surviving forms' constants: RFP slope/imp/cap, LMP base/imp divisor, null gate offset/R terms, futility A/B/C, SEE k's.
- Run in **games at equal time**. Convergence statistics alone are not evidence [E memory spsa-tuning-needs-replication-not-convergence-stats].
- Replicate on fresh seeds before shipping.
- Then re-crank margins whenever the eval changes [E memory a-truer-eval-buys-pruning-headroom…].

**Ordering caveat:**
- A2 (`FIX_MIN_SCOUT`) is unconfirmed (+10 ± 25 / 600) and stays off. Every stage is measured on gen 5 with A2 off.
- If A2 is later shipped, Stages 2–3 must be re-read (pruning on the minimizer's PV nodes changes with the scout bound).
- If the negamax port lands first (preplan §3), implement Stages 1–5 there once instead of twice. The forms in this report are written negamax-style for that reason.

---

## Evidence index
- **Ours:**
  - `search_engine.h`: :34, :76-95, :229-264, :332-339, :1332-1370, :1527-1539, :2814-2869, :2991-2995, :3139-3172.
  - `search_engine.cpp`:
    - preset :2079-2192 (RFP 1000 at :2138);
    - min node entry :5868-6024;
    - max node entry :6553-6745;
    - min loop pruning :4743-4816;
    - max loop pruning :5168-5255;
    - make before prune :6844/:6849;
    - null guard :8781-8838.
  - `selfplay/_queue_pe_rules.sh:16`.
- **Stockfish (local):**
  - SF11 `stockfish_11/stockfish-11-win/src/search.cpp` :60-90, :785-935, :996-1032; `search.h:35`;
  - SF15.1 `stockfish_15/stockfish_15.1_win_x64/src/search.cpp` :60-80, :760-905, :980-1030; `types.h:189`;
  - SF18 `stockfish_18_linux/src/search.cpp` :715-760, :870-983, :1030-1116; `types.h:185`;
  - SF19 `stockfish_19_linux/src/search.cpp` :860-870, :989-1104, :1140-1230.
- **Fetched 2026-10-22 (raw.githubusercontent.com, default branch):**
  - Ethereal `src/search.c` (`search()` steps 6-15) + `src/search.h:41-80`;
  - Weiss `src/search.c` (`AlphaBeta`, `Quiescence`);
  - Berserk `src/search.c` (`Negamax`, `InitPruningAndReductionTables`);
  - Obsidian `src/search.cpp` (`negamax`, params :40-68);
  - Stormphrax `src/search.cpp` (`search`) + `src/tunable.h:209-249`;
  - Gyatso `Gyatso/src/search.nim` (`negamax`) + `searchparams.nim`.
  - Local copies: scratchpad `refsrc/`.
- **Record:**
  - `research-2026-10-08/SEARCH-TRANSITION-PREPLAN.md` §1, §6;
  - `SEARCH-CORRECTNESS-PLAN-2026-10-19.md`;
  - `SEARCH-SWEEP-2026-08-25.md` §5-6b;
  - `OPTIMIZATION_LOG.md:1034, 1844`;
  - `external-audit-2026-07-03.md:6`.
  - Memory: sf-source-paths…, reference-engine-sources, search-changes-are-antagonistic…, a-truer-eval-buys-pruning-headroom…, eval-accuracy-payoff-is-pruning, half-a-ply…, margin-sweep-52-configs…, v2-margin-handicap…, rfp-is-99-8-percent-correct…, improving-heuristic-shelved.
- **Unverified:**
  - all Elo figures (in-source annotations, not fishtest);
  - Ethereal's/Weiss's current eval type and pawn scale;
  - NNUE engines' ×10 conversion.
