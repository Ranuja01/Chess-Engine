# CORRECTION HISTORY — reference audit vs our 08-24 closure (2026-10-22)

READ-ONLY audit. Nothing was built or run. Tags: **[E]** = evidence with a citation (local `file:line`, or a URL plus the
function or commit); **[S]** = my inference. Local Stockfish paths are relative to `C:\Users\Kumodth\OneDrive\Desktop\Programming\Chess Engine\`.
GitHub sources were fetched from `raw.githubusercontent.com/<repo>/HEAD/...` on 2026-10-22. HEAD commits: Berserk `32628515` (2026-09-07),
Obsidian `720614a` (2025-09-27), Stormphrax `d1468d9` (2026-08-29), PlentyChess `9ca0057` (2026-10-21), Weiss `c735b8f`
(2026-08-13), Caissa `324f1f9` (2026-10-19), Ethereal `0e47e9b` (2024-06-11).

---

## BOTTOM LINE

1. **The 08-24 closure does not hold for the modern form. The form we closed is one no strong engine still uses.** [E+S] Our build was one
   pawn-keyed table, an EMA with a fixed 1/64 rate, no depth weighting, no bound guards, trained against the RAW eval, and applied to the
   RFP eval only. Every engine audited (SF16.1→19, Berserk, Obsidian, Stormphrax, PlentyChess, Weiss, Caissa) does something
   different on most of those points (§2).
2. **Two engines measured the change from our form to the modern form in games, and both gained.** [E]
   - **Caissa** started as a 1/256 EMA against the raw eval. Replacing it with "SF-style correction history" (gravity, depth-scaled bonus,
     `[stm]` tables) measured **+20.1 ± 6.8 Elo** at STC; the LTC test was unfinished at +30 ± 12 (commit `190abaa7`).
   - **Berserk** started with a plain EMA (+1.0 STC, LLR not passed / +4.0 LTC). It then measured **depth-weighting
     +4.1 / +3.2** (`d7b13294`) and **gravity +2.7 / +3.4** (`f5033c4b`).
   Our offline replay ranked depth-weighting and guards as **worse** (gravity was never replayed). Its MAE metric points the opposite
   way to these game results.
3. **The 08-24 gate cannot carry the "whole family is dead" verdict.** [E code + S] It tested combining keys as an
   **intersection** (one tuple key), but every engine **sums** separate tables. It never logged a continuation key. It scored the "guards"
   arm on the very records the guards discard, and only implemented half the guard. It used 200 unrelated positions, while games re-visit
   the same pawn structures for dozens of moves. It ran on v1. And it measured MAE, which no Elo-producing consumer reads.
4. **The reference Elo is real but modest at the top, and larger for a hand-crafted eval (HCE).** [E] Stockfish's own SPRTs work out to about
   +1–2 Elo per corrhist component at LTC (§3; derived from the published W/L/D). **Weiss, an HCE engine like ours,** shows pawn corrhist
   +4.9 STC / **+11.7 LTC**, and its later tables add roughly +2 to +8 each (§3). SPRT stops inflate these numbers (memory
   `sprt-point-estimates-inflate`).
5. **What to build first (§7):** the consensus minimum, with every knob default-off and byte-identical:
   - tables: pawn + non-pawn[White] + non-pawn[Black], indexed by the real side to move, plus a 2-ply continuation table in stage 2;
   - update: int16 gravity entries, bonus ∝ (best − *corrected* eval) × remaining depth, clamped, with bound-consistency guards;
   - application: once at node entry, so it reaches **every** eval consumer we have, including qsearch stand-pat.

   Judge it on equal-time games at two time controls plus the SF18 move test at divergence points, not on an offline MAE.

---

## 1. COMPARISON TABLE (feature × engine)

| | **Ours (08-24 build)** | **SF16.1** | **SF17.1** | **SF18** | **SF19** | **Berserk** | **Obsidian** | **Stormphrax** | **PlentyChess** | **Weiss (HCE)** | **Caissa** |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Tables | pawn | pawn | pawn, minor, nonPawn W/B, cont | same + cont(ss−4) | same | pawn, cont ×2 | pawn, nonPawn W/B, cont | pawn, nonPawn W/B, major, cont ×3 | pawn, nonPawn W/B, minor, major, cont | pawn, minor, major, nonPawn W/B, cont ×6 | pawn, nonPawn W/B, cont ×2 |
| Side index | `maxbit` (root side vs not) | `[stm]` | `[stm]` | `[stm]` | `[stm]` | stm folded into pawn key; cont no stm | `[stm]` | `[stm]`; cont no stm | `[stm]` (cont too) | `[stm]` | `[stm]` |
| Continuation key | — | — | (piece,to) of ss−1 under move at ss−2 | + under ss−4 | same | ss−1 under ss−2 and ss−3 | ss−1 under ss−2 | **zobrist XOR zobrist N plies ago**, N=1,2,4 | ss−1 [piece][to][threat-from][threat-to] | ss−1 under ss−2..ss−7 | ss−1 × ss−2, ss−1 × ss−4 |
| Size | 2×16384 int32 | 2×16384 | 32768×2 | 65536 bundles ×2 (×threads) | same | 131072 pawn | 32768×2 | 16384×2 + 32768 cont | 16384×2 | 2×16384 | 2×16384 |
| Update rule | EMA `e += (r−e)>>6`, clamp ±2000 mp | **gravity** D=1024 | gravity | gravity | gravity | gravity D=16384 | gravity 1024 | gravity 1024 | gravity 1024 | gravity, per-table D 1063–1651 | gravity 1024 |
| Residual r against | **raw** eval | corrected | corrected | corrected | corrected | corrected | corrected | corrected | corrected | corrected | corrected |
| Bonus | none (fixed 1/64) | `clamp(r·d/8, ±256)` | `clamp(r·d/8, ±256)` × per-table 111–162/128 | `clamp(r·d/(bestMove?10:8), ±256)` | `clamp(r·d·(bm?12:18)/128, ±256)·1061/1024` | `clamp(3–4·r·d, ±4096)` | `clamp(r·d/8, ±256)` | `clamp(r·d/8, ±256)` | `clamp(r·d/100·118/1024, ±256)` (d in 1/100 ply) | `clamp(r·d/4, −172, +289)` | `clamp(r·d/4, ±249)` |
| Guards | not check, best quiet, not mate | not check, best quiet-or-none, **bound-consistent** | not check, not capture, `(b<se && b<β) ‖ (b>se && bestMove)` | `(b>se) == bool(bestMove)` | same + **multicut update** | bound-consistent | bound-consistent | bound-consistent, not singular-excluded | bound-consistent + multicut update | bound-consistent | bound-consistent; **bad captures allowed** |
| Combine / apply | ×0.75 | `cv·|cv|/12475` (quadratic) | Σ wᵢ·eᵢ / 131072 | Σ / 131072 | Σ / 131072 | Σ / 8192 | Σ (each /512) | Σ / 2048 | Σ / 65536 | Σ / 131072 | Σ / 512 |
| Where applied | RFP eval (+null gate); qsearch optional | the one `staticEval`: all of search + qsearch | same | same | same | same + qsearch | same + qsearch | same + qsearch | same + qsearch | same + qsearch | same + qsearch |
| TT stores | n/a (no eval field); q-cache cached corrected values | **raw** | raw | raw | raw | raw | raw | raw | raw | raw | raw |
| \|corr\| as complexity | — | — | futility margin, LMR, singular margins | same | same | — | — | RFP margin, singular | LMR | — | — |
| Rule50 | none | none in corrhist | none | none | none | ×(200−fmr)/200 after | ×(200−hmc)/200 before | ×(200−hm)/200 before | ×(293−r50)/293 before | ×(256−r50)/256 if r50>7, after | ×(256−hmc)/256 after |

**Ethereal:** no correction history. Confirmed [E]: zero `corr` hits in `src/search.c`, `src/history.c`, `src/history.h` and `src/thread.h` at HEAD
`0e47e9b` (2024-06-11, the repository's last commit).

---

## 2. SOURCE DETAIL

### 2.0 Ours [E]
- **Table:** `pawnCorrHist[2][CORR_SIZE]`, `CORR_SIZE = 16384`, int32 (`cache_management.h:167-168`, `cpp_bitboard.cpp:180`).
  Index = `generatePawnKey(...) & (CORR_SIZE-1)` (`search_engine.cpp:363-366`). The pawn key has no side-to-move bit (`cache_management.h:571-583`).
- **Read:** `entry · CORR_W / CORR_DIV`, i.e. 192/256 = 0.75 (`search_engine.cpp:368-371`; knobs `search_engine.h:2970-2973`).
- **Update:** `e += (best − raw_static − e) >> CORR_SHIFT` with shift 6, clamped to ±`CORR_MAX` = 2000 mp (`search_engine.cpp:373-381`).
  - It is called at node exit in the minimizer (`:6403-6427`) and maximizer (`:7109-7127`).
  - Conditions: best move not a capture, `|score| < 9,000,000` (not mate), node not in check.
  - There is **no** bound/direction guard and **no** depth term.
  - If the node had no node-entry eval, a **fresh eval is paid** (`:6418-6421`, counted by `g_corrhist_eval_computed`).
- **Where applied:**
  - Only `rfp_static_eval` (`:5886-5887`, `:6571-6572`). That eval exists only when `rfp_want_eval` is true: a null window with remaining depth in
    [`RFP_MIN_DEPTH` 1, `RFP_MAX_DEPTH` 6], or `ENABLE_NULL_EVAL_GATE`, which defaults to false (`h:2845-2857`).
  - So in practice it reaches **RFP at rd 1–6 plus `NULLMOVE_EVAL_R`/OTV when those are on**.
  - The v2 shipped preset sets `RFP_MARGIN = 1000` mp per ply (`search_engine.cpp:2138`).
  - With `ENABLE_CORRHIST_QSEARCH` it also reaches qsearch stand-pat and the horizon return (`:8030-8032`, `:8136-8138`).
- **Frame:** our evals are root-relative and the table is split by `maxbit`. Residuals are stored root-relative (`:303-307` comment).
- **Offline gate** `diagnostics/corrhist_signal.py`:
  - ORACLE (`evaluate`, `:66-102`) and REPLAY of the EMA (`replay`, `:107-132`).
  - The GLOBAL single-slot arm is `single_slot=True` (`:113`) and the random-bucket arm is at `:191-194`.
  - The "guards" arm drops only `ft == 2 and bs <= se` (`:122-123`).
  - "Combining" keys is `(mb, k[0], k[1])`, a tuple **intersection** (`:207-208`).
  - Kill criteria are at `:243-250`.
  - Its record source logs pawn/minor/major/nonPawn keys but **no move or continuation key** (`search_engine.cpp:316-332`).

### 2.1 Stockfish 16.1 (`stockfish_16/src` is 16.1 — `misc.cpp:78 version = "16.1"`) [E]
- **Table:** one table, `CorrectionHistory = Stats<int16_t, 1024, COLOR_NB, 16384>`, indexed `[us][pawn_key & 16383]` (`movepick.h:37-53, 138-140`).
- **Gravity:** `entry += bonus − entry·|bonus|/D` with D = 1024 (`movepick.h:71-75`).
- **Apply:** `v += cv·|cv| / 12475`, which is **quadratic**, then clamp out of the TB range (`search.cpp:67-72`).
  - It is applied where `ss->staticEval` is assigned (`:712`, `:721`) and in qsearch (`:1445`, `:1459`).
  - So improving, razoring, futility, null move and the eval-diff history bonus (`:727-733`) all read the corrected eval.
- **TT:** stores `unadjustedStaticEval` (`:723-725`, `:1334-1340`, `:1611-1615`).
- **Update** (`:1342-1350`): `!inCheck && (!bestMove || !capture(bestMove)) && !(best ≥ β && best ≤ staticEval) && !(!bestMove && best ≥ staticEval)`;
  bonus `clamp((best − ss->staticEval)·depth/8, ±256)`. `ss->staticEval` is the **corrected** value.
- **Origin:** PR #4950 by Vizvezdenec, merged 2023-12-31 (https://github.com/official-stockfish/Stockfish/pull/4950) [E].
  16.1 is the first release that has it [S: PR date vs. 16.0 in June 2023].

### 2.2 Stockfish 17.1 (`stockfish_17/src`, `misc.cpp:43 version = "17.1"`) [E]
- **Tables:** `correction_value` (`search.cpp:85-97`) = `7685·pawn + 7495·minor + 9144·(nonPawnW + nonPawnB) + 6469·cont`.
  - `cont` = `(ss−2)->continuationCorrectionHistory[piece_on(prev.to)][prev.to]`, i.e. a 2-ply move pair.
  - Apply `v + cv/131072`, **linear** (`:126-130`).
- **Keys:** `minorPieceKey` = knights + bishops, no kings. `nonPawnKey[c]` = all non-pawns of colour c **including the king** (`position.cpp:350-367`).
  - Table size 32768, int16, limit 1024 (`history.h:37-63, 129-167`).
  - Pawn table initialised to 6 (`search.cpp:582`).
- **Update:** `update_correction_history` (`:132-152`) splits one bonus across tables: pawn ×111/128, minor ×146/128, nonPawn ×162/128, cont ×143/128.
  - Guard (`:1494-1502`): `!inCheck && !(bestMove && capture) && ((best < se && best < β) || (best > se && bestMove))`.
  - Bonus `clamp((best − se)·depth/8, ±256)`.
- **Application:** search (`:812`, `:822`) and qsearch (`:1593`, `:1606`). The TT stores the raw eval (`:824-826`, `:1486-1492`, `:1616`).
  - Qsearch **does** write TT values derived from the corrected stand-pat (`:1608-1617`).
- **Complexity use:**
  - futility margin `+ |cv|/139878` (`:860-864`);
  - singular double/triple margins `|cv|/248873`, `/255331` (`:1147-1148`);
  - LMR `r −= |cv|/29696` (`:1211`).
- **History** [E PR pages, merge dates from the GitHub close events]:
  - material corrhist #5556 (2024-09-09);
  - minor/major/non-pawn #5598 (2024-09-17);
  - continuation #5617 (2024-10-12).
  17.1 has **no material and no major** table [E absence in `search.cpp:85-97`], so both were added and later removed. The removal PR was not located [unverified].
  [S] SF17.0 (2024-09-06) predates these PRs, so the multi-table form first shipped in 17.1.

### 2.3 Stockfish 18 (`stockfish_18_linux/src`, `misc.cpp:43 version = "18"`; `stockfish_1/src(SF18)` has the same corrhist code) [E]
- **Tables:** one `UnifiedCorrectionHistory` of `CorrectionBundle{pawn, minor, nonPawnWhite, nonPawnBlack}` × `[COLOR]`.
  - Size `CORRHIST_BASE_SIZE = 65536` × thread count. It is shared across a NUMA node's threads (`history.h:39-41, 156-260`).
  - Each field is read at **its own key's** index.
- **Correction:** `10347·pawn + 8821·minor + 11665·(npW + npB) + 7841·(cont[ss−2] + cont[ss−4])`. When there is no previous move, cont = 8 (`search.cpp:80-94`).
- **Update** (`:102-123`):
  - pawn ×1, minor ×156/128, nonPawn ×178/128, cont(ss−2) ×127/128, cont(ss−4) ×59/128;
  - guard `!inCheck && !(bestMove && capture) && (best > se) == bool(bestMove)`;
  - bonus `clamp((best − se)·depth/(bestMove ? 10 : 8), ±256)` (`:1473-1481`).
  - [S] This guard is *directional*: a positive correction only when alpha was raised, a negative one only on fail-low.
- **Consumers of the corrected `ss->staticEval`/`eval`:**
  - improving and opponentWorsening (`:750-751`);
  - the hindsight reduction adjustment (`:756`) and the eval-diff ordering bonus (`:862`);
  - futility (RFP) (`:878-889`, plus the margin term `+|cv|/174665`);
  - null-move condition (`:893`) and probcut (`:947-948`);
  - child futility (`:1068`, `:1097-1098`) and history bonus scaling (`:1430`);
  - qsearch stand-pat and `futilityBase` (`:1563-1604`).
- **Complexity use:** singular `|cv|/230673` (`:1142`) and LMR `r −= |cv|/30370` (`:1197`).

### 2.4 Stockfish 19 (`stockfish_19_linux/src` = `stockfish_19/stockfish/src`, `misc.cpp:57 version = "19"`) [E]
- **Correction:** `15341·pawn + 10569·minor + 12906·(npW + npB) + 8761·(cont[ss−2] + cont[ss−4])`. With no previous move the cont term is 64049 (`search.cpp:85-101`).
  - Update weights: pawn 1, minor 150/128, nonPawn 186/128, cont 130/128 and 70/128 (`:109-131`).
- **Changes from 18:**
  - bonus `clamp((best − se)·depth·(bestMove ? 12 : 18)/128, ±256)` then ×1061/1024 (`:1629-1638`);
  - **a new update at the multicut return** with `value − se` × singularDepth × 177/1024 (`:1280-1287`);
  - correction-table prefetch in `do_move` (`:643-655`);
  - tables cleared to −5 (`:696`);
  - `correction_value` computed before the TT probe (`:812`);
  - futility margin `+|cv|/198435` (`:1004`), singular `/198368` (`:1259`), LMR `/26310` (`:1324`).
- `CorrHistType` now holds only `PieceTo` and `Continuation`; pawn/minor/non-pawn live in the bundle (`history.h:152-170`).

### 2.5 Berserk (`src/search.c`, `src/history.[ch]`, `src/types.h`) [E]
- **Tables:** `pawnCorrection[131072]`, keyed by a pawn zobrist that **includes stm** (`types.h:40-43, 91, 203`). `contCorrection[12][64][12][64]` (`types.h:204`).
- **Read:** `GetCorrectionScore = (31·pawn + 17·cont(ss−3) + 46·cont(ss−2)) / 8192`, where each cont entry is indexed by the `(ss−1)` move (`history.h:79-85`).
- **Update:**
  - gravity `e += inc − e·|inc|/16384` (`history.h:64-66`);
  - pawn bonus `clamp(3·(real − se)·depth, ±4096)`, cont `4·…` (`history.c:92-110`);
  - guard `!inCheck && !IsCap(bestMove) && (bound & (best ≥ se ? LOWER : UPPER))` (`search.c:857-861`).
- **Application:** search (`:479`, `:488`) and qsearch (`:913`, `:922`). The TT stores `rawEval` (`:490-491`, `:855`). FMR scaling is applied to `eval` after correction (`:86-88`).
- **Lineage** (GitHub commit messages):
  - `8c72a1d8` (2024-01-03) EMA 255/256 vs **raw** eval, applied at 50%: +1.02 STC (LLR 1.59, did not pass), +4.03 LTC;
  - `d7b13294` "Utilize depth when updating pawn correction history": **+4.13 STC / +3.17 LTC**;
  - `525dab99` continuation corrhist: +2.75 / **+5.46**;
  - `f5033c4b` "Utilize Gravity for Correction Histories", which also moved the residual to `ss->staticEval` (corrected): **+2.70 / +3.42**;
  - `bf918b53` cont from the opponent's prior 2 moves: +2.44 / +0.90;
  - `fd4cc597` corrhist-based reduction: +3.94 / +1.36;
  - `ba2b7bab` enlarged the pawn table "such that depth 24 bench success hitrate is >95%" — sizing chosen by hit rate.

### 2.6 Obsidian (`src/search.cpp`, `src/history.h`) [E]
- **Tables:** pawn, nonPawn W/B `[CORRHIST_SIZE = 32768][2]` and a cont `[pieceTo][pieceTo]` table (`history.h:7-43`). Gravity limit 1024.
- **`adjustEval`** (`search.cpp:374-388`): FMR scaling `(200 − hmc)/200` **first**, then `+ 30·pawn/512 + 35·npW/512 + 35·npB/512 + 27·cont(ss−2)/512`.
  It is used in search (`:824`) and qsearch (`:543`). The TT stores `rawStaticEval` (`:553`, `:631`).
- **Update** (`:1223-1239`): not in check, best not a capture, `canUseScore(bound, best, se)` (bound-consistent, `:415-417`), bonus `clamp((best − se)·depth/8, ±256)`.
- **Elo:** `302ec639` "Add correction history" +2.03 STC / +5.61 LTC; `a5219f6e` cont +2.82 STC.

### 2.7 Stormphrax (`src/correction.[ch]`, `src/search.cpp`, `src/eval/eval.cpp`, `src/tunable.h`) [E]
- **Tables:** `SidedTables{pawn, blackNonPawn, whiteNonPawn, major}[16384]` per stm, plus `m_cont[32768]`.
  - The cont table is keyed by **`pos.key() ^ keyHistory[size − N]`** for N = 1, 2, 4. That is a *position-pair* key, not a move key.
  - Entries are atomic int16 with gravity limit 1024 (`correction.h`, `correction.cpp`).
- **Bonus and weights:** bonus `clamp((score − se)·depth/8, ±256)`. Weights 173/167/127/152/214/144 (`tunable.h:174-180`). Read as `correction/2048`, applied after material scaling and `×(200 − halfmove)/200` (`eval.cpp:32-66`).
- **Guard** (`search.cpp:1433-1440`): not excluded, not in check, best null or quiet, and `exact || (upper && best < se) || (lower && best > se)`.
- **Complexity:** `|correction|` feeds the RFP margin (`:844`) and the singular double/triple margins (`:1098-1111`).
- **TT:** stores the raw eval (`:786`, `:1443`).

### 2.8 PlentyChess (`src/history.[ch]pp`, `src/search.cpp`) [E]
- **Tables:** pawn, nonPawn W/B, minor and major `[stm][16384]`, plus cont `[stm][piece][to][threatened(from)][threatened(to)]` for the previous move only
  (`history.h:16-78`; `search.cpp:559`).
- **Factors** 6252/5916/4109/2627/6019, read as `/65536` after rule50 scaling `×(293 − r50)/293` (`history.cpp:44-50, 151-166`). Gravity limit 1024 (`:168-197`).
- **Guards:** `!checkers && (!bestMove || !capture) && (!failHigh || best > se) && (!failLow || best ≤ se)`. Bonus `clamp(r·depth/100·118/1024, ±256)`
  (`search.cpp:1269-1273`). There is also an update at multicut (`:1051-1055`).
- **Complexity:** LMR `−|cv|/142k` (`:1099`).

### 2.9 Weiss (HCE — `EvalPosition(pos, thread->pawnCache)`) (`src/history.h`, `src/search.c`, `src/threads.h`) [E]
- **Tables:** pawn, minor, major, `nonPawn[colour]` `[stm][16384]`, plus cont at offsets **2..7** (`threads.h:31-85`; `history.h:33-54`).
- **Gravity:** per-table limits 1651/1142/1222/1514/1063 (`history.h:43-60`).
- **Bonus:** `clamp((score − eval)·depth/4, −172, +289)`. The clamp is asymmetric (`history.h:71-73`).
- **Read:** `GetCorrectionHistory = (5868·pawn + 7217·minor + 4416·major + 7025·(npW + npB) + 4060/3235/2626/3841/3379/2901·cont[2..7]) / 131072`
  (`history.h:155-168`).
- **`CorrectEval`:** adds the correction, then `×(256 − r50)/256` if r50 > 7 (`search.c:63-69`). Used in qsearch (`:152`) and search (`:347`).
  The TT stores `unadjustedEval` (`:636-641`).
- **Guard** (SF16.1 form): `!inCheck && !capturing(bestMove) && !(best ≥ β && best ≤ se) && !(!bestMove && best ≥ se)` (`search.c:643-648`).

### 2.10 Caissa (originator, Oct 2023) (`src/backend/Search.[ch]pp`) [E]
- **Tables:** pawn and nonPawn W/B `[stm][16384]`, plus cont `[stm][pieceTo][pieceTo]` for (ss−1, ss−2) and (ss−1, ss−4) (`Search.hpp:256-298`).
- **Gravity and bonus:** gravity 1024 (`Search.cpp:1168-1171`); bonus `clamp(r·depth/4, ±249)` (`:136-141`, `:2368`).
- **Apply:** `AdjustEvalScore` adds `(53·pawn + 65·npW + 65·npB + 76·cont…)/512`, then rule50 scaling (`:1173-1196`). Used in search (`:1696`) and qsearch (`:1301`).
  The TT stores the raw `node->staticEval` (`:1255`, `:1290`).
- **Guard:** not in check; best move invalid, quiet, **or a SEE-losing capture**; directional `(best < se && best < β) || (best > se && bestMove)` (`:2363-2366`).
- **Lineage** (commit messages):
  - `403c8ad4` material-key EMA, 1/256 blend vs raw eval: +3.66 STC / +8.99 LTC (LTC LLR not passed);
  - `aff75594` + pawn-structure EMA: +1.74 STC;
  - **`190abaa7` "SF-style correction history"** — EMA → gravity, add non-pawn W/B, drop material, `[stm]` tables: **+20.10 ± 6.82 STC**,
    LTC +30.3 ± 12.2 (LLR 1.95, unfinished);
  - `6e715458` continuation: +3.09;
  - `39a3b7fa` update-condition tweak: +1.12;
  - `54aef4aa` allow bad-capture updates: +2.63;
  - `24fde494` tables shared across threads (4 threads): +6.12;
  - `b02c47fa` prefetch: +2.40 (a speed gain).

---

## 3. REPORTED ELO

| Source | Change | STC | LTC |
|---|---|---|---|
| SF PR #4950 [E W/L/D; Elo derived by me — S] | first pawn corrhist | 128,672 g W32757 L32299 ⇒ **≈ +1.2** | 97,422 g W24626 L24178 ⇒ **≈ +1.6** |
| SF PR #5556 | material table | ≈ +1.0 | ≈ +1.1 |
| SF PR #5598 | minor + major + non-pawn | 23,328 g ⇒ **≈ +4.4** | 10,626 g ⇒ **≈ +8.7** |
| SF PR #5617 | continuation | ≈ +0.8 | ≈ +1.6 |
| **Weiss `85910015`** (HCE) | **pawn corrhist** | **+4.87 ± 3.10** | **+11.70 ± 4.81** (40+0.4) |
| Weiss `5d9ce65e` | material | +3.02 | +7.88 |
| Weiss `3f9eb096`, `3fc8367f`, `c62cb0fb`, `51b94fbf`, `938d4949` | cont −2 … −7 (five commits; `3fc8367f` also has other tweaks) | +1.8, +3.0, +1.4, +1.9, +3.3 | +1.7, +4.6, +4.1, +2.0, +4.0 |
| Weiss `54e8064e` | non-pawn | +2.80 | +6.84 (60+0.6) |
| Weiss `8048bb98` | material → minor/major | +1.75 | +2.25 |
| Caissa `190abaa7` | **EMA → SF-style** | **+20.1 ± 6.8** | +30.3 ± 12.2 (unfinished) |
| Berserk `d7b13294` / `f5033c4b` | depth weighting / gravity | +4.1 / +2.7 | +3.2 / +3.4 |
| Obsidian `302ec639` | pawn + non-pawn + cont | +2.0 | +5.6 |

- SF Elo was derived from W/L/D as `400·log10(s/(1−s))`. The PR pages quote only LLR and W/L/D (fetched pages for
  https://github.com/official-stockfish/Stockfish/pull/4950, /5556, /5598, /5617).
- The CPW page https://www.chessprogramming.org/Static_Evaluation_Correction_History states "non-linear scaling" with TC and quotes no Elo [E].
- Weiss URLs are in the commit messages (`chess.grantnet.us/test/38269`, `/38270`, …) [E].
- **Reading** [S]:
  - Gains are larger at longer TC; Weiss pawn goes from 4.9 to 11.7.
  - Gains are larger for an HCE engine (Weiss) than for SF's NNUE.
  - The Weiss family sums to roughly +24 STC / +45 LTC, but SPRT stops inflate each term (memory `sprt-point-estimates-inflate…`).
  - **A realistic prior for us is +5 to +20 Elo at equal time, larger at longer TC, with zero not excluded.**

---

## 4. CONSENSUS FORM, AND WHERE ENGINES DISAGREE

### 4.1 What nearly all of them do the same way [E §2]
1. **Several separate tables, summed with weights.** None of them intersects keys into one tuple.
   - Pawn: 10/10 implementations, counting SF16.1→19 as four.
   - Non-pawn per colour: SF17+, Obsidian, Stormphrax, PlentyChess, Weiss, Caissa.
   - A continuation/last-move table: everyone except SF16.1.
2. **Index by side to move**, with the value stored relative to the side to move.
3. **Gravity update** with an int16 entry, a limit (usually 1024) and a bonus clamped to about ¼ of the limit.
   Berserk used EMA, then measured gravity better. Caissa used EMA, then measured SF-style +20.
4. **Bonus ∝ (best − corrected staticEval) × remaining depth.** The table learns the *remaining* error after correction, and deep nodes count more.
5. **Guards:** not in check; best move is quiet or there is none; and **bound consistency** — a fail-high only teaches "eval too low", a
   fail-low only teaches "eval too high".
6. **Applied once, where staticEval is assigned**, so every consumer inherits it, **including qsearch stand-pat**.
7. **The TT stores the raw eval.** The correction is re-derived on each visit, so it never compounds.
8. **Linear application.** SF16.1's quadratic form was dropped in 17.

### 4.2 Where they disagree
- **Extra keys:** minor (SF, Weiss, Plenty), major (Weiss, Stormphrax, Plenty; SF added it and later removed it), material (Caissa and Weiss had it, then removed it).
- **Continuation key:** move-pair at ss−2 (SF, Obsidian), ss−2 + ss−4 (SF18+, Caissa), ss−2..ss−7 (Weiss), ss−1 with threat bits (Plenty),
  position-XOR at N = 1, 2, 4 (Stormphrax).
- **Guard detail:** symmetric bound-consistency (most) versus SF18/19's *directional* `(best > se) == bool(bestMove)`. Caissa also allows SEE-losing captures.
- **Extra update sites:** multicut (SF19, Plenty).
- **|correction| as a complexity signal:** used in LMR, RFP margin and singular margins by SF17+, Stormphrax and Plenty; not by Berserk, Obsidian, Weiss or Caissa.
- **Rule50 scaling:** before vs after the correction, and its constants.
- **Learning-rate constants:** per-table limits (Weiss), bonus divisors /4, /8, /10.

---

## 5. OUR GAPS vs THE CONSENSUS — and could each explain the 08-24 negative?

| # | Gap | Could it explain 08-24? |
|---|---|---|
| G1 | **Applied only to RFP at rd 1–6** (plus the null gate, which is off by default). Not applied to child futility, qsearch stand-pat/delta, `improving` (off), NULLMOVE_EVAL_R, probcut-like or razoring sites. | **Yes, for the STS/games flatness.** [S] With `RFP_MARGIN = 1000` mp/ply, a correction of a few hundred mp flips RFP mainly at rd 1–2. The giants' Elo flows through every pruning decision plus stand-pat. A one-consumer wiring has a fraction of the leverage. |
| G2 | **One pawn table** instead of a weighted sum of several | **Yes.** [S] Summing pawn + non-pawn W/B + cont gives each node several partially independent estimates. When one slot is starved, the others still contribute. Starvation (median 5 updates/slot) is exactly what summing tables mitigates. |
| G3 | **Fixed-rate EMA (1/64), no depth term**, vs gravity with a depth-scaled bonus | **Yes.** [E Berserk/Caissa] Both engines measured this change as a game gain. [S] At depth 8, SF's summed correction moves about 0.5× the residual per update. Our single table moves 1.6%. The gate itself found "faster EMA helps monotonically", which is the same direction. |
| G4 | **Residual against the RAW eval** | **Partly.** [S] Fitting the raw residual and applying 0.75× has roughly the same fixed point, but there is no feedback from the tables' combined output. With several summed tables, fitting against the corrected eval is required; otherwise each table learns the full error and the sum over-corrects. |
| G5 | **No bound guards** | **Possibly.** [S] Fail-soft bounds teach the wrong direction. Our 08-24 "guards are worse" result is invalid (§6). |
| G6 | **`maxbit` index with root-relative values** | Not a cause of the 08-24 result (a fixed root colour inside a test). **A latent hazard:** if a process keeps tables while the root colour changes (analysis, or a process playing both colours without `ucinewgame`), slots written for one colour are read for the other (§8). |
| G7 | **The offline MAE gate as the kill** | **Yes — it is the main reason the lane closed** (§6). |
| G8 | **No continuation key** | Never tested. It is the most universal non-pawn table (§4.1). |
| G9 | **No complexity channel** | Not part of the 08-24 build. It is an extra Elo channel in SF/Stormphrax/Plenty; stage 3 candidate. |
| G10 | **The qsearch extension "harmful" (07-30)** | **That verdict was measured in `DISABLE_QCACHE`, a regime our own memory later ruled invalid** ("NOT A VALID MEASUREMENT REGIME", memory `corrhist-qsearch-harmful…`). Cached it was +1 vs flat. The −49 does not stand. [E memory] Also, SF's qsearch *does* store values derived from the corrected stand-pat in the TT (SF17.1 `search.cpp:1608-1617`). Caching corrected-eval-derived *scores* is normal; only the *static-eval field* must be raw. [E] |
| G11 | **Extra eval paid at update-only nodes** (`g_corrhist_eval_computed`) | Costs NPS. [S] Probably small next to the gains, but it must be counted, since memory says eval is about a third of node cost. |
| G12 | **v1 eval** | **Yes, in principle.** v2 is a different function, with different residual structure and magnitude. |

---

## 6. DOES THE 08-24 CLOSURE SURVIVE?

### Still holds [E/S]
- **The exact artefact we built** (one pawn table, EMA shift 6, ×0.75, RFP-only, v1) is not worth shipping. STS 1630 vs 1629 cached was flat [E memory].
  Keep that build closed.
- **On v1, against MAE, the keyed EMA replay did not beat a running single global slot.** As a statement about that metric on that data, it is a correct measurement [E SEARCH-SWEEP §1].
- **The method lessons stand:** use GLOBAL and random controls, and compare ORACLE against REPLAY.

### Does not hold
1. **"Multi-keying is dead / combining keys is worse."**
   - The combining arm built **one tuple key** `(mb, pawn, minor[, major])` (`corrhist_signal.py:207-208`). Its coverage collapse (93.9% → 47.9%) is
     the expected cost of intersecting keys [E].
   - No reference engine combines that way; all sum separate tables [E §2].
   - **The refutation tested a design nobody uses.** [S]
2. **"Equal per-family ORACLE scores mean the signal is position-local."**
   - Equal scores across pawn/minor/major/non-pawn on a 200-position, single-search-per-position corpus are expected *whatever* the
     structure. Within one search, almost every key is constant for most nodes, so each key acts as a position ID. [S]
   - The test cannot tell "position-local" from "structural but confounded by the corpus design". Memory's own rule applies: "ELEGANCE OF
     EXPLANATION IS NOT EVIDENCE".
3. **"Sample starvation is structural to the keying; logging more won't help."**
   - It was measured on **200 unrelated positions**. In a game, consecutive searches share pawn and non-pawn structure for dozens of moves,
     and the table persists across moves (it is cleared only in `clearSearchTables`, `search_engine.cpp:3462-3463`).
   - Berserk sized its pawn table by **hit rate in bench** (`ba2b7bab`), a different notion of occupancy.
   - **The occupancy that matters is per-game, at the slots that are READ at decision sites.** Never measured. [S]
4. **"Weiss guards are worse."**
   - The replay drops guarded records from *learning* but still *scores* them (`corrhist_signal.py:117-123`: base and corrected are appended
     before the `continue`). The guarded arm is graded on exactly the bound records it refuses to chase.
   - It also implements only the fail-high half (`ft == 2 and bs <= se`), not the fail-low half.
   - The MAE target is itself a fail-soft **bound** on those records, not a true value.
   - **Invalid comparison.** [E code + S]
5. **"Depth weighting is worse."**
   - MAE weights every record equally, and the noisiest records (rd 1–2 fail-soft) dominate the count. Depth weighting deliberately
     down-weights them, so MAE on the same noisy targets penalises it by construction. [S]
   - Berserk measured depth weighting in games at **+4.1 / +3.2**. [E]
6. **"Do not propose gravity updates / continuation corrhist."**
   - **Neither was ever measured.** The replay has no gravity arm and the log has no move key [E `corrhist_signal.py:107-132`;
     `search_engine.cpp:329-331`]. The memory's "each is measured or refuted above" is false for these two items.
7. **The kill instrument was MAE.**
   - No consumer of the eval reads MAE. Elo comes from decisions flipping across margins at RFP, futility and stand-pat, and from tree shape.
   - Memory already records that corpus fit is anti-correlated with Elo, and the 08-24 note itself warns not to over-read the GLOBAL "+8.23%".
     **The kill criteria were applied in the one metric the lab distrusts.** [S]

**Verdict** [S]: 08-24 closed one implementation, correctly. It did not close correction history. The "whole family closes" corollary rests on
items 1–7, and each is either an untested design or an invalid comparison. **Re-open the lane, building the consensus form**, judged by games.

---

## 7. STAGED BUILD + TEST PLAN (nothing runs until the overnight queue is done and the owner says go)

Constraints:
- Every knob defaults to 0 and stays byte-identical to the shipped v2 gen-5 fingerprint.
- One process per setting (knobs latch at init).
- No side runs while a queue saturates the box.
- PLAY GAMES after search changes. Use a 2×2 against any concurrent search change (the anti-additivity law).

### Stage 0 — instruments, no games (≈ ½ day)
1. **New knob family `CORR2_*`.** Leave the old `ENABLE_CORR_HIST` path untouched as the "old form" arm.
2. **Frame-safe storage** (§8): index by real colour `turn`, store values stm-relative.
3. **Colour-mirror test of the search:** run a position and its colour-flip in fresh processes with CORR2 on. The `[turn]` tables must be
   exact mirrors, and the root scores must negate. This catches sign errors in the minimizer/maximizer halves.
4. **Counters** (counted on the taken branch, printed only under a log flag):
   - updates/node, and how many updates each guard rejects;
   - per consumer, how often the corrected eval **flips the decision** versus the raw eval (RFP, stand-pat, horizon, futility);
   - extra evals paid (`computed` vs `reused`);
   - **occupancy of slots at READ time, logged over a sequence of game moves, not a corpus.**
5. **(Optional, informative only, never a kill gate)** repair `corrhist_signal.py`:
   - score only bound-consistent or exact records;
   - add a **sum-of-tables** arm with a gravity replay against the corrected value;
   - add the continuation key to `[CORRLOG]`;
   - drive it from game-sequence logs (consecutive moves of real games), not 200 independent positions.

### Stage 1 — consensus minimum (`CORR2 = 1`)
- **Tables:** pawn, nonPawn[W], nonPawn[B]. Indexed `[turn][key & (N−1)]`, N = 16384 (a knob: 16k/64k). Entries int16, limit 1024.
  - All three keys exist: `generatePawnKey`, `generateNonPawnKey` (`cache_management.h:571, 617`).
- **Read:** `corr_mp = (Wp·pawn + Wn·(npW + npB)) / 1024`. Weights are in mp at saturation (§8), starting at `Wp ≈ 350`, `Wn ≈ 400`, with a
  global scale knob for sweeps (×½, ×1, ×2).
- **Update** at node exit:
  - guards: not in check, |score| not mate, best move not a capture (or there is no best move), node not singular-excluded if we
    have that, and **bound-consistent in the root frame**:
    `(best ≥ β_orig ⇒ best > se_corr) ∧ (best ≤ α_orig ⇒ best < se_corr)`, with exact accepted.
    This root-frame form is frame-invariant (§8).
  - `bonus = clamp(resid_stm_mp · rd / CORR2_BDIV, ±256)`, with `resid_stm` = (best − *corrected* eval) converted to stm sign.
    Start `CORR2_BDIV ≈ 38` (§8). Then gravity `e += bonus − e·|bonus|/1024` on all three tables.
- **Apply once** at node entry, to every consumer of `rfp_static_eval`: RFP, null gate, NULLMOVE_EVAL_R, OTV. Also to qsearch stand-pat
  and horizon (`CORR2_QS`, default **on** within CORR2, with an off arm).
  - The update needs a node-entry eval. Compute it where missing, accept the counted cost.
- **TT/q-cache:** our TT has no eval field. The q-cache stores scores, and storing scores derived from a corrected stand-pat matches SF (§5 G10).
- **Arms** (fresh seeds; [[gate-new-candidates-on-fresh-seeds-not-ship-seeds]]):

  | Arm | What it is |
  |---|---|
  | A | CORR2 |
  | B | the old `ENABLE_CORR_HIST` form — re-measures 08-24 in games |
  | C | **one GLOBAL running slot** per colour, same machinery — tests the 08-24 headline directly |
  | D | A without qsearch |

- **Instruments:**
  1. byte-identity at defaults;
  2. mirror test;
  3. counters: flip rates, eval cost, NPS on the quiet corpus;
  4. **SF18 move test at divergence points**, where A's root move differs from the control (`_fix_firecount.py`-style divergence move test);
  5. **equal-time fastchess self-play at two TCs** (corrhist scales with TC), using UHO openings and a varied seed;
  6. SF18 gauntlet (~1100) plus Mediocre "not negative".

  Ship rule as registered: self-play ≥ +2σ, SF18 > −1σ pooled, Mediocre not negative.
- **Register before running** [S predictions]:

  | ID | Prediction |
  |---|---|
  | P1 | Read-time occupancy in game sequences is far above the 08-24 corpus median of 5 (p50 ≥ 20). |
  | P2 | A beats control by **+5 to +20 Elo** at equal time (central +10), larger at the longer TC. |
  | P3 | C (GLOBAL slot) is within ±5 (no gain). |
  | P4 | B is within ±5 (replicates the flat 08-24 result). |
  | P5 | D ≤ A (qsearch application helps or is neutral, contradicting 07-30). |
  | P6 | NPS cost < 3%. |
  | P7 | RFP flip rate under A is a few % of RFP evaluations, concentrated at rd 1–2. |

  **Park** (register entry with a re-open trigger, not "closed") if A < 0 at the pooled −2σ bound on both instruments.
- **Then re-sweep RFP_MARGIN combined with A** (the "eval-accuracy payoff is pruning" law): a truer eval buys margin headroom.

### Stage 2 — continuation table (`CORR2_CONT`)
- 2-ply move-pair `[turn][piece,to of the previous move][piece,to of the move before]`, SF/Obsidian form. Then optionally add ss−4 (SF18, Caissa).
- Use our existing prev-move plumbing (`g_searchStack`).
- Same instruments. Prediction: +2 to +5.

### Stage 3 — reach and extras (each its own arm)
- (a) Correct the **child eval at the futility site** (child keys are cheap).
- (b) `|corr|` as complexity: RFP margin `+|c|/k` and LMR `−|c|/k` (SF/Stormphrax/Plenty).
- (c) minor/major tables.
- (d) SF18-style directional guard vs the symmetric guard.
- (e) multicut/OTV update site.
- (f) Rule50 scaling (we have none anywhere — a separate lane).

### Sequencing note
- Corrhist interacts with the dynamic-eval retune and with the search-correctness FIX_* knobs, so settle it **before** the dynamic-lane
  retune (as PREPLAN §2.2 already says).
- After the negamax port it becomes simpler: `[us]` tables and one `ss->staticEval`. **Do not wait for the port:** the root-frame form
  below is exact, and stage 1 measures the idea now.

---

## 8. FRAME OF REFERENCE (non-negamax, Black-positive absolute eval, millipawns)

- **Sign** [S, derived]: our search scores are root-relative after the single `Config::side_to_play` flip. Convert at the edges only:
  - Let `root_side_to_move = maxbit` (maximizer = root side). Then `resid_stm = maxbit ? (best − se) : −(best − se)`.
  - Store `resid_stm` in `table[turn]`.
  - When reading, `corr_root = maxbit ? v : −v`.
  - Indexing by real `turn` instead of `maxbit` makes entries mean the same thing for any root colour.
- **Bound semantics in the root frame** [S, derived]:
  - In both halves, `best ≥ β_orig` is a root-frame **lower** bound (true ≥ best). A maximizer cut and a minimizer whose every child came back
    ≥ β both give a lower bound. `best ≤ α_orig` is a root-frame **upper** bound.
  - So the guard "lower bound ⇒ only teach upward; upper bound ⇒ only teach downward" can be written once in the root frame. It is
    invariant under the stm negation, because both the bound type and the direction flip together.
  - SF17/18's `bestMove`-based forms are stm-specific. Prefer the bound form (Berserk, Obsidian, Stormphrax).
  - ⚠️ Our `[CORRLOG] failtype` column is root-frame (`search_engine.cpp:328`). It is correct for this purpose, but the gate implemented only
    one of the two clauses.
- **Scales — ratios transfer, scales never** [S, with E anchors]:
  - SF `PawnValue = 208` (`stockfish_18_linux/src/types.h:185`). The UCI/WDL "pawn" is larger: `a(material)` ≈ 350–390 internal per
    `uci.cpp:501-513`. So any SF→mp conversion is uncertain by about 1.7×.
  - SF18 pawn table at saturation contributes `1024·10347/131072 ≈ 81` internal, which is 0.39 PawnValue or ≈ 0.22 WDL-pawn. All tables
    saturated in the same direction ≈ 455 internal (≈ 1.2–2.2 pawns).
  - Per update, a bonus at the clamp moves the pawn table by about 20 internal (≈ 0.05–0.1 pawn).
  - In our units: **Wp ≈ 250–400 mp, Wn ≈ 300–450 mp per non-pawn table at saturation**, and a summed ceiling of about 1.5 pawns
    (our current single-table ceiling is 2000·0.75 = 1500 mp). Hence the global scale knob in stage 1.
  - Bonus conversion: SF's bonus at d = 8 for a 1-PawnValue residual is about 208. In mp, `bonus ≈ resid_mp·rd/38`. Our rd may not equal
    SF depth (our d10 is shallower in practice), so `CORR2_BDIV` is a knob, starting at 38.
- **Eval-mode frames:** the old code learns against `RFP_EVAL_MODE` and applies to `QSTANDPAT_EVAL_MODE`. These agree only while both are 0
  (`search_engine.h:2922-2923`). Keep that assertion in CORR2.

---

## 9. FILES READ
- Ours:
  - `search_engine.cpp`: 300-381, 3440-3467, 5867-5925, 6400-6428, 6560-6590, 7109-7127, 8020-8040, 8125-8140, 2120-2149;
  - `search_engine.h`: 2904-2973;
  - `cache_management.h`: 167-168, 571-623;
  - `diagnostics/corrhist_signal.py` (all);
  - `dev_notes/SEARCH-SWEEP-2026-08-25.md` §1;
  - `dev_notes/research-2026-10-08/SEARCH-TRANSITION-PREPLAN.md`: 76-193, 456;
  - memory: `corrhist-pawn-structure-reopens-pawn-hash.md`, `corrhist-qsearch-harmful-and-qcache-masks-eval.md`,
    `corrhist-signal-is-position-local-not-structural.md`, `sf-source-paths-and-pruning-shapes.md`.
- Stockfish (local): `stockfish_16/src` (16.1), `stockfish_17/src` (17.1), `stockfish_18_linux/src` (18), `stockfish_19_linux/src` (19).
  Files: `search.cpp`, `history.h` / `movepick.h`, `position.cpp`, `types.h`, `uci.cpp`, at the lines cited above.
- GitHub (HEAD): Berserk, Obsidian, Stormphrax, PlentyChess, Weiss, Caissa and Ethereal sources as cited.
- Commit messages and diffs via `api.github.com/repos/<repo>/commits/<sha>` for the SHAs in §2/§3.
- SF PR pages #4950, #5556, #5598, #5617.
- CPW Static_Evaluation_Correction_History.
