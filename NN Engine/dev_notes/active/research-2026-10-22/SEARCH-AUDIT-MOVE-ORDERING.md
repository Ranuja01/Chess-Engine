# SEARCH AUDIT: MOVE ORDERING AND HISTORY HEURISTICS (2026-10-22)

**Status:** read-only reference audit. I built nothing, ran nothing and edited no other file, because an overnight queue was running.

Tags: **[E]** = evidence, with file:line or URL+function; **[S]** = my inference.
- Our line numbers are from the tree as of 2026-10-22 (HEAD `875326a`, gen 5).
- Local Stockfish paths are relative to `C:\Users\Kumodth\OneDrive\Desktop\Programming\Chess Engine\`:
  - SF11 = `stockfish_11/stockfish-11-win/src/`
  - SF15.1 = `stockfish_15/stockfish_15.1_win_x64/src/`
  - SF18 = `stockfish_1/src(SF18)/`
  - SF19 = `stockfish_19/stockfish/src/`
- Fetched engines (`raw.githubusercontent.com`, default branch, 2026-10-22):
  - Ethereal (`AndyGrant/Ethereal@master`), Weiss (`TerjeKir/weiss@master`) and Berserk (`jhonnold/berserk@main`) were fetched verbatim with curl.
  - Obsidian (`gab8192/Obsidian@main`), Stormphrax (`Ciekce/Stormphrax@main`) and GyatsoChess (`GyatsoYT/GyatsoChess@main`, Nim) came through WebFetch's summariser. Their formulas are therefore **quoted fragments, not line-verified**, and line numbers were not available. Treat those three columns as [E-] (evidence, weaker).
- Sibling audits: `SEARCH-AUDIT-PRUNING.md` covers history pruning in depth, and `SEARCH-AUDIT-REDUCTIONS-EXTENSIONS.md` covers LMR. This document covers only how histories feed those.

---

## 0. BOTTOM LINE

1. **Ordering at the head is not our gap.** First-move cutoff (FMC) is 87–92%, which is in the normal range for strong engines [E memory `fmc-headroom…`, `tt-and-cache-architecture`].
   - The gaps are the **cost** of producing the order and the **content** of the history tables.
2. **The `moveGenCache` design is a memory and speed liability with one real asset.**
   - The asset is node-local cutoff promotion, which acts as our de facto hash move [E `cache_management.h:1134-1186`; memory `cached-move-order-beats-a-fresh-history-sort`].
   - Every miss pays the full price: legal generation with `is_safe`, scoring every move (SEE on unclear captures), then a `stable_sort` of the whole list (`move_gen.h:658-934`). Every reference engine stages the work instead.
3. **★ NEW: qsearch builds the FULL legal list, quiets included, and then throws the quiets away.**
   - `buildNoisyMoveList` (`search_engine.cpp:9364-9420`) calls `generateLegalMovesReordered` on a cache miss. That generates and scores all quiets, using DRAM-cold 134 MB `counterMoveHeuristics` lookups. The full list is **never cached**.
   - Arithmetic from two of our own notes suggests about half of all movegen calls happen outside the main-search miss site [S, §4.1].
   - The 08-25 staged-generation closure instrumented only the main-search site. **That closure did not size this.**
4. **★ NEW: the gravity update is only half-applied.**
   - Presearch cutoffs (`cpp:7942-7952`) and every node-exit "best move" write (`cpp:6405, 7108, 7980`) use raw, unbounded `+=` with no malus. The presearch generates ~38.7% of all cutoffs [E memory `fmc-counter-includes-presearch`].
   - Separately, our bonus `rd²` is **~0.6% of MAX_HISTORY at rd 10**. SF's is ~30%. Our tables therefore adapt ~50× more slowly, and nothing ages them between moves [E/S §3.4].
5. **★ NEW defect: the cache-hit killer splice uses `iter_swap`, not insert** (`cpp:9022-9089`). This has two effects:
   - It demotes the best-history quiet to the killer's old slot.
   - When a promoted quiet sits at index 0, `firstNonCapture` = 0 is reset to 1. The killers are then swapped in **ahead of every capture**, and the best capture is sent deep into the tail [E code-read; frequency unmeasured].
6. **Consensus items we lack** (support out of 7 reference engines + SF):
   - piece×to continuation history at plies 1, 2, 4 and usually 6 (7/7);
   - capture history keyed by victim (7/8; Gyatso has none);
   - **capture malus when the best move is quiet** (8/8, ours never does this);
   - a history-dependent good/bad capture SEE threshold (SF11+, Berserk, Obsidian, Stormphrax);
   - threat-aware quiet scoring (SF15+, Obsidian and Berserk add a score term; Ethereal, Stormphrax, Gyatso and Berserk index the table by threat);
   - history in futility, SEE and LMP margins (see the pruning audit).
7. **What we have that no reference has:**
   - `moveFrequency`, an unbounded PV-move bonus added to captures *and* quiets (`move_gen.h:843,908`; `cpp:3703`);
   - the frozen cached order.
   - Neither has ever been ablated [S]. Ablating them is cheap and should come first.
8. **Closures re-examined:**
   - Static ordering: **HOLDS** (games null).
   - Piece-key continuation history: **HOLDS as Elo-neutral**, adopt for memory.
   - TT move: **HOLDS only while the cache promotion exists**; it becomes mandatory in a staged picker.
   - Threat history: **CLOSED ONLY IN ITS INDEX FORM on v1**. The SF15+ *score-bonus* form needs no learning and was never tried.
   - Continuation history 2-ply: **closed in from×to form only**. Piece×to plies 2/4 were never measured.
9. **Elo scale to expect.**
   - SF annotates history-LMR at ~30 Elo, countermove pruning at ~20 (SF11), threat/eval-diff ordering at ~3 and continuation pruning at ~2 (SF15.1) [E].
   - Our own wins: gravcap +33, statScore-LMR +23, PROTECT_KILLERS +12.4 [E memory].
   - ⇒ Pure ordering tweaks are worth a few Elo each and sit below our games resolution. **The levers are speed (qsearch generation, staged picker) and history-into-pruning.**
10. **Recommended order** (§6):
    1. ablation and probe stage (no tree change);
    2. qsearch captures-and-checks-only generation (result-identical);
    3. the history-content fixes;
    4. the staged picker inside the negamax port.

---

## 1. WHAT OURS IS (code map)

| element | ours (default) | cite |
|---|---|---|
| **Pipeline** | `buildMoveListFromReordered`, then a snapshot copy of `moveGenCache[key]` into a per-ply buffer.<br>**Hit (≈24%):** find the first non-capture, `iter_swap` killer1, killer2 and counter into it, lazy top-K bubble (`PROMOTE_TOP_K=2` when the last cutoff index > 2), optional TT move.<br>**Miss:** full legal generation, score all, `stable_sort`, store. | `search_engine.cpp:8998-9260`; `cache_management.h:781-887` |
| **Generation** | 12 attack mask-pairs plus 4 quiet mask-pairs (13.8–14.7 passes per call), with pseudo-legal moves filtered by per-move `is_safe()` (`MG_ISSAFE` ≈ 2× `MG_PSEUDO`). | `move_gen.h:175-232, 699-794`; memory `movegen-is-36-percent…`, `the-nps-gap-is-mostly-not-eval` |
| **Capture score** | `200000 + victim − attacker + promo + moveFrequency + capHist` when `victim ≥ attacker`. Otherwise SEE: SEE ≥ 0 gets the `200000+SEE` tier, SEE < 0 gets just `SEE+…` and can sit **below** quiets. | `move_gen.h:834-856` |
| **Quiet score** | `butterfly + killer(10000/9000) + counter(8000) + cmh(ply-1, from×to) + [cont2] + [check] + [pcont] + [static] + promo + moveFrequency`. | `move_gen.h:858-909` |
| **Killers** | 2 per ply (`killerMoves[ply][2]`), shift insert. | `cache_management.h:1188-1205` |
| **Counter move** | `counterMoves[prevFrom][prevTo]` (from×to key, no piece). | `cache_management.h:153`; `cpp:6296` |
| **Butterfly** | `historyHeuristics[stm][from][to]` | `cache_management.h:156` |
| **Continuation ply 1** | `counterMoveHeuristics[2][4096][4096]` (from×to × from×to, 134 MB). `CONT_HIST_PIECE_KEY=0` → piece×to 2 MB, off. | `cache_management.h:155, 243-265`; `h:1584` |
| **Continuation ply 2** | `contHist2` (134 MB), `ENABLE_CONT_HIST_2PLY=false`, `CONT2_GRAVITY_DIV=4`. | `h:1578, 1617` |
| **Capture history** | `captureHistory[stm][from][to]` (no victim). `CAPTURE_HIST_VICTIM=0` → `[stm][attacker][victim][to]`. | `cache_management.h:277-303`; `h:1585-1586` |
| **moveFrequency** | `[stm][from][to] += (d−1)²·(L−i)²` per PV move per completed iteration. `>>2` at search start and at some iterations. No reference engine has this. | `cpp:3697-3713, 3577, 3626`; `cache_management.h:1317` |
| **Bonus** | `b = rd²·HISTORY_BONUS_SCALE/100` (=rd²), with rd = remaining depth. | `cpp:6297, 5770ff, 7003ff` |
| **Update rule** | `hist_update`: `h += b − h·|b|/16384`. Malus `−b/MALUS_DIV` (=−b) to tried quiets, **main search cutoffs only**. Continuation history ply-1 gets the same `b` (the raw path uses 4b). | `cache_management.h:1310-1315`; `cpp:6300-6344` |
| **Raw (non-gravity) writes** | presearch cutoff: `hist += rd²`, `cmh += 4rd²`, `capHist += rd²`, no malus (`cpp:7942-7952`).<br>Node exit when a quiet is best: `hist += rd` (`cpp:6405, 7108, 7980`). | [E] |
| **Capture updates** | Capture cutoff: bonus to the move, malus to other tried captures. **Quiet cutoff: no capture malus.** | `cpp:6346-6365` |
| **Aging** | **None.** `decayHistoryHeuristics` calls are commented out (`cpp:5506, 6468`). `clearSearchTables` runs only on ucinewgame or a side change. The move-gen cache also persists across moves. | `cpp:3441-3476` |
| **Ordering → LMR** | `statScore = hist + cmh + ch2` (weights 1/1/1), `delta = statScore/683`, clamped to ±2 plies. Applies to quiets only; captures are never LMR'd.<br>⚠️ The shipped reduce-more (`HISTORY_LMR_MORE_CAP=1`) is **unreachable** because the statScore branch returns first (`cpp:1283-1308` vs `:1341`). | `cpp:1278-1345, 4688` |
| **Ordering → pruning** | LMP keys on the move **index** `i ≥ 1+rd²`, rd ≤ 5. Killers and counter are exempt at i ≤ 8 (PROTECT_KILLERS). `ENABLE_HIST_PRUNE`, `ENABLE_SEE_PRUNE` and `LMP_HIST_EXEMPT` are off. All of it runs **after** `make_move`. | `cpp:4686-4790`; pruning audit §0.5 |
| **Root** | Presearch scores order the root. move_gen order at the root measured worse than arbitrary. | memory `movegen-ordering-is-not-useful-at-the-root` |
| **qsearch** | Full list (cached or freshly generated), then filtered. Keeps promotions, captures with SEE ≥ 0 (SEE computed **again**, `cpp:9411`) and quiet checks at **every** q-depth (`ENABLE_QCHECK_DEPTH0=false`). No qsearch history updates. | `cpp:9364-9440`; `h:3002` |

---

## 2. FEATURE × ENGINE TABLE

### 2a. Picker stages and generation

| engine | stages (in order) | sort | generation and legality |
|---|---|---|---|
| **Ours** | one list: [captures by MVV-LVA or SEE] [quiets by score] [SEE<0 captures interleaved by score]; cached order plus killer/counter splice | full `stable_sort` on a miss | **legal**: 16 mask-pair passes plus per-move `is_safe` |
| SF11 | TT → captures (`see_ge(−55·v/1024)` decides good) → killer1, killer2, counter → quiets → bad captures | partial selection for captures; `partial_insertion_sort(−3000·d)` for quiets | pseudo-legal, staged; `pos.legal()` in the search loop (`movepick.cpp:55-265`) |
| SF15.1 | same; good-capture threshold `−69·v/1024` | same | same (`movepick.cpp:194-232`) |
| SF19 | TT → good captures (`see_ge(−v/18)`) → **good quiets (v > −14000)** → bad captures → bad quiets. **No killers, no counter** | partial insertion `−3560·d` | same (`movepick.cpp:280-354`) |
| Ethereal | TT → good noisy (SEE ≥ 0) → K1 → K2 → counter → quiets → bad noisy | selection | pseudo-legal; legality checked **after** make (`apply`) |
| Weiss | TT → good noisy (`>11046` auto-good, `≤−9543` auto-bad, else SEE) → 1 killer → quiets → bad noisy | partial insertion (`> −750·d`) | pseudo-legal; `MoveIsLegal` before make |
| Berserk | hash → good noisy (`SEE(−score/2)`) → K1 → K2 → counter → quiets → bad | selection | pseudo-legal; `IsLegal` before make |
| Obsidian [E-] | TT → good captures (`seeGe(−score/32)`) → killer → counter → quiets → bad | selection | pseudo-legal; `isLegal` before make |
| Stormphrax [E-] | TT → good noisy (`see(−score/4+72)`) → quiets → bad noisy. **No killers, no counter** | selection | **fully legal** generator |
| Gyatso [E-] | TT → good noisy (SEE ≥ 0) → quiets (killers scored inside) → bad | selection | **fully legal** |

### 2b. History tables

| engine | butterfly key | continuation (plies) | capture key | other | bonus(d) / malus(d) | limit | aging |
|---|---|---|---|---|---|---|---|
| **Ours** | [stm][from][to] | from×to (1); 2 off | [stm][from][to] | moveFrequency, killers 2, counter | rd² / −rd²; presearch and node-exit raw | 16384 | **none** |
| SF11 | [c][from_to] | piece×to (1, 2, 4, 6), indexed [inCheck][capture] | [piece][to][captured] | counter [piece][to]; reverse-move malus | `19d²+155d−132` (d ≤ 15); same malus | 10692 / 29952 cont | none (clear on new game) |
| SF15.1 | same | (1, 2, 4, 6); only 1–2 when in check | same | eval-diff history bonus (~3 Elo) | `min((12d+282)d−349, 1594)` | 7183 / 10692 / 29952 | none |
| SF19 | [c][move.raw] | piece×to (1, 2, **3**, 4, 5, 6) with weights 520/390/145/251/66/209 | same | **pawn history** [pawnKey][piece][to]; **lowPly** (ply < 5); ttMoveHistory; eval-diff into main and pawn | `min(133d−81, 1487)` + TT-move and statScore terms, scaled by moves searched; malus `min(968d−235, 2244)`, decaying per move | 7183 / 10692 / 30000 / 8192 | **main ×729/1024 per search**; lowPly refilled per search |
| Ethereal | [c][threat_from][threat_to][from][to] | (1, 2) [piece][to] | [pt][thr_f][thr_t][to][captured] | killers 2, counter | `16d²+128(d−1)` (32 if d>13); −same | 16384 | none |
| Weiss | [stm][from][to] | (1, 2, 4) | [piece][to][captured] | **pawn history**; 1 killer | `min(2418, 251d−267)` / `−min(693, 532d−163)` | per table 4373–16384 | none |
| Berserk | [stm][!thr_f][!thr_t][from_to] | (1, 2, 4, 6), weights 36/35/19/17 | [piece][to][defended][captured] | killers 2, counter | `min(1708, 4d²+191d−118)`; malus shrinks with quiet count | 16384 | none |
| Obsidian [E-] | [c][from_to] | (1, 2, 4, 6); updates halved at 4 and 6 | [piece·to][captured] | **pawn history**, 1 killer, counter, eval-diff | `min(175d+15, 1409)` / `min(196d−25, 1047)` | 16384 | none |
| Stormphrax [E-] | [stm][from][to][fromAtt][toAtt] + pieceTo[thr] | (1, 2, 4, 6) with **shared-base gravity** | [from][to][captured][toDefended] | no killers, no counter | `339d−313 ≤ 2663` / `−(384d−177 ≤ 1318)` | ~16.3k | **butterfly and pieceTo ×750/1024 per search** |
| Gyatso [E-] | [stm][from][to][fromAtt][toAtt] | (1, 2) | — | killers 2 | `min(1500, d²+2d)` | 16384 | butterfly ×3/4 per search |

### 2c. How histories feed LMR and pruning

| engine | LMR | history pruning | futility, SEE and LMP use of history |
|---|---|---|---|
| **Ours** | quiets: `±min(2, statScore/683)`; captures never reduced | off | none |
| SF11 | `r −= statScore/16384` (~30 Elo); compares against `(ss−1)->statScore` (~10 Elo) | countermove prune: cont[0] and cont[1] < 0 at lmrD < 4–5 (~20 Elo) | futility requires `hist sum < 25000` |
| SF15.1 | `/ (13628 + 4000·[7<d<19])` (~30) | `cont sum < −3875(d−1)`, lmrD < 5 (~2) | `+ history/52` in the futility margin |
| SF19 | `r −= statScore·439/4096` (1024 = 1 ply); **capture statScore** = MVV + capHist | `cont+pawn < −4136·d` | `lmrDepth += history/divisor`; capture futility and SEE use capHist |
| Ethereal | `−hist/6167` quiet; noisy `3 − hist/4952` | `min(cm, fm) < {−1000, −2500}` | futility gate `hist < {14296, 6004}`; SEE margin `−hist/128` |
| Weiss | `−hist/8870`; lmrDepth uses `hist/9164` | `hist < −1024·d` at lmrD < 3 | parent histScore feeds RFP and NMP |
| Berserk | `−8·hist/65536` | `< −2850(d−1)` at lmrD < 4 → skip all quiets | — |
| Obsidian [E-] | `−hist/9621` quiet, `/5693` capture | `< −7471·d` → skip | through lmrDepth |
| Stormphrax [E-] | weighted history ×423 or 447/4096 | quiet `< −2242d−1315`; noisy `< −1006d²−1121` | LMP count `+ hist·985/2²³`; futility `+ hist/102`; noisy SEE `− hist/65` |

---

## 3. PER FEATURE: consensus, disagreement, evolution, Elo

### 3.1 Staged picker vs a sorted list
- **Consensus (9/9 references):** generate captures first and pick the next one by partial selection. Generate quiets only after the captures (and the killers/counter where present) fail. Delay bad captures. Allow `skipQuiets` so that LMP and futility stop quiet generation entirely [E §2a].
- **Disagreement:**
  - SF19 split quiets into good and bad around the bad captures (−14000).
  - Stormphrax and SF19 dropped killers and counter stages altogether, with threat- or pawn-indexed histories taking over.
  - Legality: legal generators (Stormphrax, Gyatso) versus pseudo-legal checked before make (Weiss, Berserk, Obsidian, SF) versus checked after make (Ethereal).
- **Evolution:**
  - SF11 → 15: threat-escape and check bonuses added to the quiet score.
  - SF15 → 19: killers, countermoves and refutation stages removed. Added: pawn history, lowPly, a plies-1–6 continuation table, ttMoveHistory, and a good-quiet/bad-quiet split.
  - Elo for killer removal is not annotated in source (unverified).
- **Elo:** no reference annotates the picker itself. Its value is speed.

### 3.2 Capture ordering
- **Consensus:**
  - MVV (no LVA) + capture history at a scale comparable to piece values (SF `7·PV + capHist`; Weiss, Berserk, Obsidian, Stormphrax likewise).
  - Good/bad is decided by SEE with a **history-dependent threshold** (SF11 `−55·v/1024`, SF19 `−v/18`, Berserk `−v/2`, Obsidian `−v/32`, Stormphrax `−v/4+72`).
  - Ethereal and Gyatso use a flat 0; Weiss lets extreme history override SEE.
- **Ours:**
  - MVV-LVA (`victim − attacker`) on a 200000 base. SEE only when victim < attacker.
  - Capture history is added but keyed [from][to].
  - `moveFrequency` is added to captures.
- **Evolution:** SF11 already had `[piece][to][captured]` and the history threshold. The key has not changed since.

### 3.3 Quiet ordering terms
- **Consensus:** `butterfly + Σ continuation(piece×to)` at plies 1 and 2 (7/7), plus 4 (5/7) and 6 (4/7). Weights are close to 1, with ply 1 weighted highest.
- **Threat terms:**
  - SF15.1 adds an escape bonus of 50000/25000/15000 (`movepick.cpp` score).
  - SF19 uses `20·PV[pt]·(escape − enter)` against lesser attackers.
  - Obsidian and Berserk use a ±16k score term.
  - Ethereal, Stormphrax, Gyatso and Berserk index the butterfly table by from/to-attacked.
  - Weiss has none.
- **Check bonus:** SF15+ (16384, with SEE ≥ −75 in SF19) and Stormphrax (9994). Others have none.
- **Ours:** butterfly + ply-1 from×to + flat killer/counter bonuses + moveFrequency. No threat term and no check term. (Ours is `ENABLE_CHECK_ORDER`, off as a mirage on v1.)

### 3.4 Update formula and dynamics (the quantitative gap)
- **Consensus:**
  - Gravity `h += b − h|b|/MAX` with MAX ≈ 7k–30k (all 8).
  - Bonus is roughly linear in depth and capped at **~1000–2700**, so **b/MAX ≈ 10–35%** at mid depth.
  - Malus applies to every tried quiet (8/8) and to every tried capture, **including when the best move is quiet** (SF11 `search.cpp:1634-1639`; SF19 `:2004-2010`; Ethereal, Weiss, Berserk; Obsidian likely; Stormphrax yes).
  - Updates happen once per node, at node end with the best move (SF) or on fail-high.
- **Ours:**
  - `b = rd²`, so 100 at rd 10 and 25 at rd 5, against MAX 16384: **b/MAX ≈ 0.15–0.6%**.
  - Gravity's fixed point does not depend on step size. Its *memory* does: about MAX/b updates, i.e. ~160 at rd 10 and ~650 at rd 5.
  - With **no aging across moves** (refs: SF19 ×0.71 per search, Stormphrax ×0.73, Gyatso ×0.75; Ethereal, Weiss, Berserk and Obsidian have none, but their b/MAX is large), our tables are a near-static game-long average [S, arithmetic E].
  - This fits the measured fact that "node-local cached order beats a fresh history sort" [E memory]: our global history is a weak, slow signal, so the frozen local order wins by default. **That closure may be a symptom of slow history, not a law** [S].
- **Raw writes bypass gravity:**
  - The presearch writes rd² and 4rd² with no malus (`cpp:7942-7952`).
  - Node exit writes `+rd` to the "best" quiet even at fail-low nodes (`cpp:6405, 7108, 7980`), where the best move is noise. SF updates only when `bestMove` exists, i.e. alpha was raised (SF11 `:1322-1330`).
- **Elo:** gravity+malus+capHist+recalibration = **+33 ±17** for us [E memory `gravcap`]. SF annotates the TT-move history updates at ~1–3 Elo (SF15.1 `search.cpp:637-650`).

### 3.5 Killers and counter moves
- **Consensus pre-2023:** 2 killers + a counter keyed [piece][to] (SF11, Ethereal, Berserk). Weiss and Obsidian use 1 killer.
- **Modern:** SF19 and Stormphrax have none.
- **Ours:** 2 killers + counter keyed [from][to] with no piece (a coarser identity). Killers are protected from LMR (+12.4 Elo, our one structural-marker win).

### 3.6 Low-ply and root history, pawn history, eval-diff history
- lowPly: SF12–14 had it, SF15.1 does not, SF19 does (ply < 5) [E SF19 `history.h:42,132`; `movepick.cpp:245`]. No other reference has it.
- Pawn history: SF19 (`history.h:146`), Weiss, Obsidian.
- Eval-diff history: SF15.1 ~3 Elo (`search.cpp:754-759`), SF19 (`:978-986`) and Obsidian credit the parent's quiet move from `−(staticEval[−1] + staticEval)`. This needs a node-entry static eval, which our min/max split does not keep at every node (see the pruning audit's `improving` item).

### 3.7 Histories in LMR and pruning
- History-LMR is universal (8/8). SF annotates it at ~30 Elo (SF11 `:1186`, SF15.1 `:1174`).
- Capture LMR with capture history: SF19, Ethereal, Obsidian, Stormphrax. **We never reduce captures.**
- History or continuation pruning: 7/9 (pruning audit §3.6), ~20 Elo in SF11.

---

## 4. OURS VS CONSENSUS, GAP BY GAP

### 4.1 The cached ordered list vs a staged picker — strength or liability?

**Strengths [E]:**
- **Node-local promotion.** A revisited node puts its own last cutoff move first, a perfect node-local hash move. Re-sorting by global history *lost* nodes (+7.3% at 4 reuses) [memory `cached-move-order…`].
- **TT-move redundancy.** Feeding the TT move on top of the promotion measured −3 WAC [memory `node-tt-dependency…`].
- **Skipped generation.** A hit skips generation (24% hit rate).

**Liabilities:**
1. **Memory.**
   - Slots: `CACHE_SIZE = 1<<23` (`cpp_bitboard.h:24`) × `MoveEntry` (key + `std::vector` + 2 ints + bool, ~48 B) = **~384 MB of slots**.
   - Each occupied slot also has a heap vector (≈ 30–40 moves plus malloc overhead, unmeasured).
   - Nothing clears it between moves, so it is our memory-growth source [E `cache_management.h:43-49`; memory `tt-and-cache-architecture`].
   - A staged picker needs **zero** persistent memory; its ordering state is the history tables (SF: a few MB).
2. **Cost on a hit [E code]:**
   - a full vector copy (`fillMoveGenCache`, `cache_management.h:823-832`);
   - an `is_capture` scan to find the first quiet;
   - three linear `std::find`s for killers and counter;
   - an optional top-K re-score.
   - `updateMoveCacheForBetaCutoff` takes the list **by value** (`cache_management.h:1134`), so every non-first cutoff copies it again, and `promoteMoveToFront` does an erase+insert.
3. **Cost on a miss:** the whole list is generated legal, scored and sorted. 87–92% of cutoffs need only move 0, and most moves are never examined.
   - The 08-25 probe found 59% consumption at *generating main-search nodes* [E memory `movegen-is-36…`].
4. **★ qsearch is outside the cache economy.**
   - `buildNoisyMoveList` generates the **full** list (quiets, `is_safe`, quiet scoring with 134 MB cmh lookups) and **does not store it** (`cpp:9381-9389`).
   - It needs quiets only to find quiet checks, which it then tests one at a time.
   - **Sizing arithmetic [S]:** the movegen profile counted **4,320,438 movegen calls** on WAC d10. The staged-gen probe counted **2,061,579 main-search miss generations** on the same baseline (`243 / 31,764,817`) [E both in memory `movegen-is-36-percent…`]. The ~2.26M difference is qsearch + presearch + other sites.
   - So **roughly half of all generation happens where the 08-25 "staged generation closed (5–8% ceiling)" verdict never looked.** That verdict's own scope note ("the move-gen cache is part of the baseline") does not cover qsearch, which does not use the cache's store path.
   - ⚠️ The two counts come from different builds (profile vs probe). Confirm with one probe run before believing the split.
5. **Stale order is permanent.** A slot written at a shallow presearch depth keeps that order for the rest of the game unless a cutoff refreshes the front, and no aging exists.
   - The full re-sort lost, but a *bounded* refresh (e.g. on a new search ID) was never tried [S].
6. **Tie-break by generation order.** Zero-history quiets order by mask-pair sequence (minor → pawn → queen → rook/king in the middlegame). This sequence is phase-dependent (`move_gen.h:744-793`) and is effectively a hand-written static prior.
   - Static ordering's +9/+34 bench and null games say the prior is weak either way.

**Verdict [S]:**
- Compared on ordering quality, the cache is about a staged picker with a TT move. Our FMC matches theirs.
- Compared on memory and speed, it is a clear liability: hundreds of MB, copies, and full generation on every miss.
- The single feature to preserve in a redesign is **"the node's own last best move first"**. A staged picker gets exactly that from a **node-entry TT move**, which our child-probe TT lacks [E memory `tt-and-cache-architecture`].
- ⇒ The picker redesign and the node-entry TT are **one change**. Neither is a win alone, which is precisely why TT-move tested negative while the cache existed.

### 4.2 The killer splice defect (new, code-read)
`buildMoveListFromReordered` hit path (`cpp:9022-9089`):

```cpp
for (i...) if (!is_capture(...)) { firstNonCapture = i; break; }
if (firstNonCapture == 0) firstNonCapture = 1;
... std::iter_swap(km1, cached_moves.begin() + firstNonCapture);
```

- **(a) The swap demotes a move.** `iter_swap` *exchanges* the killer with the first quiet. The previous first quiet, which is often the promoted cutoff move or the top-history quiet, drops to wherever the killer was. SF and others *emit* killers as a stage and leave the rest of the order intact.
- **(b) Killers can jump ahead of captures.**
  - If the list starts with a quiet at index 0 (after `promoteMoveToFront` of a quiet cutoff), `firstNonCapture` is 0 and is forced to 1. Index 1 is then usually the **best capture**.
  - The killer swaps into index 1, and the best capture goes to the killer's old slot, possibly deep in the quiet tail, where index-keyed LMP can prune it.
  - The same happens when `firstNonCapture` legitimately is 0 because there are no captures. That case is harmless.
  - **(The "if no captures" intent was to protect index 0 — but it also fires when the front move is a promoted quiet with captures behind it.)**
- **Frequency is unmeasured.** It needs a counter on the taken branch: index-1 move is a capture AND a swap happened [S].
- Prediction: fixing it lowers WAC/quiet-corpus nodes slightly. This is a tree change.

### 4.3 History content gaps (ranked by consensus × cheapness)

| gap | consensus | ours | note |
|---|---|---|---|
| G1 capture malus on a quiet best move | 8/8 | never | one loop at 3 sites; tree change |
| G2 capture-history key with victim | 7/8 | [from][to]; victim key built (`CAPTURE_HIST_VICTIM`), −4.1% nodes, flat STS, **never games-tested alone** | re-test after G1 |
| G3 piece×to continuation, plies 2 and 4 (and 6) | 7/7 (4/7 for ply 6) | ply 1 only; ply 2 tested only from×to | `CONT_HIST_PIECE_KEY=1` + ply 2 never measured together |
| G4 bonus scale and aging | 8/8 bonus ~10–35% of MAX; 3/8 age per search | 0.6%; no aging | `HISTORY_BONUS_SCALE` exists (h:3267), unswept; aging knob absent |
| G5 gravity on all writes | 8/8 | presearch and node-exit raw | presearch = 38.7% of cutoffs |
| G6 history-dependent good/bad SEE threshold | 5/8 | binary victim ≥ attacker, then SEE<0 → low tier | needs SEE on all captures (cost) |
| G7 threat-aware quiet scoring | 7/8 in some form | none (`ENABLE_THREAT_HIST` index form off) | the score-bonus form needs no learning; see 4.4 |
| G8 capture LMR with capHist | 4/8 | captures never reduced | LMR audit territory |
| G9 history in futility, SEE and LMP margins | 6/8 | none | pruning audit |
| G10 eval-diff history, pawn history, lowPly | 3/8, 3/8, 1/8 | none | lowPly overlaps our presearch/root mechanism; low priority |
| G11 counter key [piece][to] | 6/6 that have a counter | [from][to] | small |

### 4.4 Past closures — do they hold?

| closure | record | still holds? |
|---|---|---|
| **Static ordering** (`ENABLE_STATIC_ORDER`) | bench +9 WAC/+34 STS; **games +0.4 ±20 / 1600**; anti-additive with LMP | **HOLDS.** No reference engine uses a placement prior; consensus fills history-silent moves with continuation/pawn/threat history. Do not reopen. |
| **Threat history** (`ENABLE_THREAT_HIST`, `move_gen.h:118-156`) | +5% nodes, STS −110, v1 | **PARTIAL.** What was tested *replaces* butterfly with a 4× split table (`move_gen.h:897`), i.e. 4× sparser at a time when history is coverage-starved [memory `history-sparsity…`]. That is the Ethereal/Stormphrax form, which works for them with ~50× larger bonuses.<br>The SF15+/Obsidian/Berserk **static score term** (escape from a lesser attacker: +; enter: −) needs no data and was never built. **Reopen in score form.** |
| **Continuation history 2-ply** | from×to at b/4: WAC −4…−8, STS −88 | **PARTIAL.** Only the 134 MB from×to form was tested. Its context space (4096²) makes it the sparsest table we own. The piece×to ply 2/4 the consensus uses was never measured.<br>Run it **with** `CONT_HIST_PIECE_KEY=1`. |
| **Piece-keyed continuation history** | −6.1% nodes, STS +17, **games −0.6 ±12.6 / 4000, replicated** | **HOLDS as Elo-neutral.** Adopt on memory merit (268 MB → 4 MB) and as the base for the item above. Note the noted erosion of PROTECT_KILLERS STS in the 2×2. |
| **Capture-history victim key** | −4.1% nodes, STS flat; anti-additive WAC −7 with piece-key (suites disagree) | **OPEN.** Never games-tested. The consensus key also differs: [piece][to][captured], no colour beyond the piece, no from-square; ours is [side][attacker][victim][to], which is equivalent. |
| **TT move ordering** | −3 WAC vs NODE_TT control; promotion redundant | **HOLDS in the cached architecture** — they compete. **INVERTS in a staged picker**, where the TT move is the only node-local signal. |
| **Lazy full re-sort** (`RESORT_AFTER_REUSES`) | every firing dose worse | **HOLDS as measured**, but it is evidence about our *slow* history (§3.4), not about history in general. Re-test after G4/G5. |
| **Staged/lazy generation** (08-25) | 5–8% ceiling, closed | **SCOPE GAP.** It did not include qsearch generation or pin-aware generation (§4.1). Re-size. |
| **Check ordering bonus** | mirage, v1, bonus 6000 | **PARTIAL.** SF15+ uses 16384 but gated by SEE ≥ −75 (SF19). Ours had no SEE gate. Low priority. |
| **History-pruning closure** | none (never measured) | open; see the pruning audit |

### 4.5 moveFrequency (ours only)
- PV moves of each completed iteration add `(d−1)²(L−i)²` to [stm][from][to]. That is ~17k at d = 12 for the first move, more than the killer bonus. It accumulates over iterations, is decayed `>>2` per search, and is added to **captures and quiets** (`move_gen.h:843, 908`; `cpp:3697-3713`).
- It is a root-PV memory that leaks into every interior node with the same from/to.
- No reference engine has it. SF's nearest analogue is lowPly history, confined to plies < 5. It has never been ablated [S; no record found by grep].
- Arm: `MOVE_FREQ_W=0`. Prediction: ±2% nodes, sign unknown. If it is neutral, delete it, because it complicates any picker port.

---

## 5. IMPROVEMENT IDEAS (fitting our engine)

Every item is [S] unless marked otherwise. The test ladder is: (i) fire counter on the taken branch; (ii) FMC% and `[cutoff_histogram]`; (iii) quiet-corpus nodes at fixed depth plus `NODE_LIMIT` equal-work WAC/STS; (iv) games.

1. **QS-GEN: captures/promotions/checks-only generation in qsearch.**
   - **Result-identical** if it emits exactly the moves the noisy filter keeps, in the same relative order:
     - captures and promotions via `to_mask = opponents | promo-rank`;
     - quiet checks via direct check squares per piece type plus quiets of discovered-check candidates (our pieces in `slider_blockers(enemy king)`).
   - Relative order is preserved because `stable_sort` among the kept subset ranks by the same scores, and ties keep generation order *within* the subset. Verify with a debug assert comparing the two lists on a corpus.
   - Gate: byte-identical node counts / solves / STS, with only NPS moving.
   - Pre-registered: qsearch-side movegen cycles fall ≥ 50%; total NPS +5–12%. Under +5% ⇒ record the null and the measured qsearch generation share.
2. **Cheap hit-path speedups (result-identical):**
   - pass `moves` by reference in `updateMoveCacheForBetaCutoff`;
   - replace `promoteMoveToFront`'s erase+insert with `std::rotate`;
   - store `firstNonCapture` in `MoveEntry` to remove the per-hit `is_capture` scan;
   - reuse the qsearch `see()` from scoring.
   - Expected 1–3%, below the bench floor. Pool them with item 1 under one byte-identity gate.
3. **SPLICE-FIX (tree change).**
   - Insert killers and counter by `rotate` at the true first-quiet index, never ahead of a capture. Equivalently, emit them as a "virtual stage" ahead of the quiet region.
   - Probe first: count of hits where the move swapped into index 1 was a capture.
   - Prediction: nodes −0.5…−3% at fixed depth, STS flat. Games only if bundled with G1–G5 as "history hygiene".
4. **HISTORY HYGIENE bundle, as a 2×2-able pair of knobs:**
   - (a) `HIST_GRAVITY_ALL=1`: route presearch and node-exit writes through `hist_update` with the same `b`, plus presearch malus.
   - (b) `HISTORY_BONUS_SCALE` sweep {100, 400, 1600} with a **consensus-shaped** bonus `min(c1·rd + c0, cap)` (new knob `HIST_BONUS_FORM=1`).
   - ⚠️ statScore-LMR divisor 683 is calibrated to the current distribution [E memory `gravity-is-built…`: "revert together or never"]. Each arm must re-derive `STATSCORE_DIVISOR` from `ENABLE_STATSCORE_PROFILE` (P95 rule, `h:379`), or the LMR consumer is mis-scaled.
   - Prediction: the zero-history bucket shrinks (`ENABLE_CUTCAL_LOG`, current 75% at d10), and the lazy re-sort stops being harmful.
5. **AGE-PER-SEARCH:** `hist = hist·HIST_AGE/1024` at search start (SF19 729, Stormphrax 750). Pairs with item 4b: only meaningful once `b` is large. Run as a 2×2 with 4b.
6. **CAP-MALUS (G1):** malus to `searched_captures` on a quiet cutoff (8/8 consensus). One loop at 3 sites. Run the 2×2 with `CAPTURE_HIST_VICTIM` (G2).
7. **THREAT-SCORE (G7, score form, no table):**
   - Add `THREAT_ORDER_W·PV[pt]·(escape − enter)` versus lesser attackers (SF19 form, `movepick.cpp:200-244`).
   - The opponent attack maps already exist in v2's threat detector, but the ordering path has no `BoardState` — check currency [memory warns about stale attack layers].
   - It fills exactly the history-silent population that static ordering targeted. Unlike PST, it carries *tactical* information.
   - Test FMC on the quiet corpus, then nodes.
8. **CONT-PLY2/4 on piece key (G3):** `CONT_HIST_PIECE_KEY=1` + ply-2 (+ ply-4 new table) at SF-like weights (ply 1 > 2 > 4).
   - Since piece-key alone is Elo-neutral, use piece-key as the *control* arm.
9. **Original idea — "promotion-aware TT move" for the transition.**
   - In the negamax port, give the node-entry TT entry the cached list's *front move* semantics: store the node's own best move at node exit (`store_node_tt` already does it, `ENABLE_NODE_TT`).
   - Then let a staged picker emit TT → good captures → killers → quiets, with **no move-gen cache**.
   - Equivalence test before games: FMC and fixed-depth nodes vs current within ±2%. That shows the cache's ordering value is fully carried by the TT move, which lets the 384 MB+ cache be deleted.
10. **Original idea — a "cache order as history" bridge (if the cache is kept).**
    - On a cutoff at index k > 0, the node already maluses the k tried quiets.
    - Additionally re-rank the cached tail by *live* score **only when history is fresh** (a per-entry search-ID stamp). This resolves the 08-26 finding without discarding node-local promotions.
    - Lower priority than item 9.
11. **moveFrequency ablation** (§4.5): `MOVE_FREQ_W ∈ {0, 50, 100}`. A 3-point plateau check, nodes plus STS.
12. **Capture LMR with capHist (G8):** the LMR audit's domain. Ordering supplies the statScore = `7·PV[victim] + capHist`, SF19 form (`search.cpp:1343`).

---

## 6. STAGED BUILD AND TEST PLAN

Rules carried in from the record:
- knobs default off and byte-identical (`250/35,310,778` v1 · **v2 gen 5 fingerprint from `baseline-fingerprints-register`**), verified BEFORE flipping;
- result-identical speed work is separated from tree-changing work;
- fire counters go on the taken branch;
- predictions are registered before running;
- 2×2s are used for any pair touching the same consumer (statScore/LMR, index-keyed LMP);
- games decide.

**Stage A — probes, no tree change (one `build_profile` + counters, all `#ifdef EVAL_PROFILE`):**
- A1: movegen calls split by site (main miss / qsearch / presearch / root) and qsearch quiet moves generated vs kept. **Prediction:** qsearch ≥ 35% of movegen calls; kept-quiet fraction < 5%.
- A2: splice-defect counter (§4.2). **Prediction:** fires on ≥ 2% of cache hits.
- A3: history-write census: raw vs gravity writes by site; |h| distribution; fraction of |h| > MAX. **Prediction:** presearch ≥ 30% of writes; some entries above MAX.
- A4: `moveFrequency` share of quiet score at cutoff moves.
- **Stop rule:** if A1 shows qsearch < 20% of calls, item 1 drops to "pool with item 2".

**Stage B — result-identical speed (byte-identical node counts required):**
- B1 = item 1 (QS-GEN).
- B2 = item 2 (hit-path copies, SEE reuse).
- Each is gated on identical `solves / nodes / EBF / STS` plus perft-style list equality on a 2k-FEN corpus. NPS via `depth_nps_bench` median-of-5 idle, and only after the queue finishes.
- Compose freely (node-identical changes are exempt from anti-additivity).
- Ship on byte-identity + NPS ≥ +3%, combined.

**Stage C — ordering hygiene (tree-changing; fixed depth → equal-work → games):**
- C1 SPLICE-FIX (item 3) alone.
- C2 = CAP-MALUS × CAPTURE_HIST_VICTIM, as a 2×2.
- C3 = HIST_GRAVITY_ALL × (bonus form/scale + re-derived STATSCORE_DIVISOR), as a 2×2. Then C3b: AGE-PER-SEARCH on the winner.
- Instruments: FMC%, cutoff histogram, zero-bucket share (`ENABLE_CUTCAL_LOG`), quiet-corpus nodes, `NODE_LIMIT` WAC + STS at 2 budgets.
- **Prediction:** C3 reduces the zero bucket ≥ 5pp at d10, and makes `RESORT_AFTER_REUSES=16` non-negative. If not, the slow-history hypothesis (§3.4) is refuted; record it.
- Games: one bundle "C-hygiene" (the winners) vs gen 5, ≥ 2,400 games UHO, varied seeds, run alone. Expected +5…+15 [S]. Accept per the ship rule (self-play ≥ +2σ, SF18 > −1σ, Mediocre not negative).

**Stage D — content additions (each alone, after C):**
- D1 THREAT-SCORE (item 7).
- D2 piece-key + ply-2/4 (item 8; control = piece-key alone).
- D3 moveFrequency ablation (item 11).
- Each starts with a fire counter, then FMC, nodes, STS@equal nodes, then games if ≥ bench noise.
- Never bundle D with LMP changes (anti-additive mechanism, memory `ordering-does-not-buy-index-keyed-pruning`).

**Stage E — inside the negamax port (with pin-aware legal generation and the node-entry TT):**
- E1: staged picker + node-entry TT move; **no move-gen cache**. Equivalence gate: FMC within 1pp, fixed-depth nodes within ±3% of the cached build at identical pruning. Then fixed-time games.
- E2: history into pruning (continuation/history prune, futility `+hist/k`, SEE `−hist/k`), SPSA'd jointly with the pruning audit's items, never one at a time.
- Memory target: drop the 384 MB+ cache and the 2×134 MB from×to tables. Resident footprint goes from ~1.2 GB to TT + small tables.

---

### Evidence index
- Ours:
  - `move_gen.h:59-156, 175-232, 658-934, 942-995, 1154-1172`
  - `cache_management.h:43-49, 152-303, 781-887, 1134-1325`
  - `search_engine.cpp:1278-1345, 3441-3476, 3577, 3626, 3697-3713, 4686-4790, 5648, 6005, 6268-6365, 6400-6405, 7108, 7609, 7925-7982, 8998-9260, 9364-9440`
  - `search_engine.h:332-404, 1314-1369, 1530-1619, 2980-3052, 3257, 3267`
  - `cpp_bitboard.h:24-28, 798-805`
- SF11: `movepick.h:89-104`; `movepick.cpp:55-265`; `search.cpp:84-87, 1005-1027, 1160-1186, 1205-1216, 1600-1668`; `search.h:35`; `thread.cpp:67-81`.
- SF15.1: `movepick.h:90-106`; `movepick.cpp` `score()`, `:194, 200, 232`; `search.cpp:83-85, 754-759, 1010-1032, 1168-1175, 1716-1724`.
- SF18: `history.h:42, 135-150, 216`; `search.cpp:314, 318-319, 1833-1835`.
- SF19: `history.h:38-42, 70-74, 128-146, 196`; `movepick.cpp:33, 194-262, 280-354`; `search.cpp:326-330, 690-710, 978-986, 1178-1229, 1338-1352, 1960-2057`.
- Ethereal: `src/movepicker.c` (`select_next`, `init_picker`, `init_noisy_picker`), `src/history.c` (`get_capture_histories`, `get_quiet_histories`, `update_quiet_histories`, `stat_bonus`), `src/search.c` (pruning/LMR), `src/types.h:100-105`.
- Weiss: `src/movepicker.c` (`NextMove`, `ScoreMoves`, `SortMoves`), `src/history.h` (`HistoryBonus`, `UpdateQuietHistory`, `GetQuietHistory`), `src/search.c:304, 451, 459-566`, `src/threads.h:30-39`.
- Berserk: `src/movepick.c` (`NextMove`, `ScoreMoves`), `src/history.c` (`HistoryBonus`, `UpdateHistories`, `UpdateCH`), `src/search.c:52-64, 632, 640-781`, `src/types.h:198-204`.
- Obsidian [E-]: `src/movepick.cpp` (`MovePicker::nextMove`, scoring), `src/history.h` (`addToHistory`), `src/search.cpp` (`negamax`, `updateMoveHistories`).
- Stormphrax [E-]: `src/movepick.h/.cpp` (`scoreNoisies`, `scoreQuiets`), `src/history.h` (`HistoryEntry::update`, `updateWithBase`, `age`), `src/search.cpp`, `src/tunable.h`.
- Gyatso [E-]: `Gyatso/src/moveorderer.nim`, `history.nim`, `search.nim`, `movegen.nim`.
- Memory notes:
  - `movegen-is-36-percent-of-search-not-a-non-lever`, `the-nps-gap-is-mostly-not-eval`, `tt-and-cache-architecture`
  - `cached-move-order-beats-a-fresh-history-sort`, `capture-history-victim-rekey-built-minus-4-percent-nodes`, `cont-hist-piece-rekey-wins-and-the-two-rekeys-are-anti-additive`
  - `gravcap-shipped-plus33-elo`, `history-sparsity-is-coverage-not-architecture`, `ordering-retest-queue-under-gravcap`, `fmc-headroom-ordering-not-bottleneck`
  - `static-eval-move-ordering-is-the-08-24-candidate`, `ordering-does-not-buy-index-keyed-pruning`, `tt-move-vs-cutoff-promotion`, `node-tt-dependency-starved-singular-and-ttmove`
  - `protect-killers-games-result…`, `shadow-auc-evidence-map…`, `presearch-value-is-not-root-ordering`, `movegen-ordering-is-not-useful-at-the-root`, `ordering-history-design-ideas`.
- Docs: `SEARCH-TRANSITION-PREPLAN.md` §1 #6, #14-18, §4, §6 #15; `SEARCH-SWEEP-2026-08-25.md` §2-5, §7, §12.2.
