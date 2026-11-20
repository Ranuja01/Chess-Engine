# Sweep 3: triage of the open search-defect items (2026-11-20)

A read-only pass over every open item of `SEARCH-DEFECT-REGISTER-2026-11-18.md` and of the plan's §0b
(`SEARCH-V2-PLAN-2026-11-10.md`), written to answer the owner's question of 11-20: which items are truly worth a
2,000-game run (about 5.6 hours of machine time each).

**How this was made.** Code was read, nothing was run (timed games were playing on the box). Every claim below is marked:

- **VERIFIED** = I read the code path and it executes at the shipped settings (`V2_PRESET=shipped`, gen 7.7,
  `SEARCH_ARM 1`).
- **REPORTED** = taken from the register or reasoned from the code, not proven by a count. Numbers marked "estimate" are
  my judgement, not measurements.

Line numbers are those of the working tree on 11-20; the function or knob name beside each is the stable reference.
Not read: the ply-1 branch of `minimizer` beyond its null move, the root razor's body, `uci_main.cpp`'s time manager.
Not covered here because other readers have them: S2-1 (check extension), S2-2 + S2-5, S2-33, and the fresh bug hunt.
They are placed in the run order (§6) only.

**Terms used.**
- *Self-play run*: 2,000 games at 10 seconds + 0.1 per move against production. Its 95% interval is about ±11.5 Elo, so an
  effect below roughly 6 Elo cannot be told from zero in one run.
- *Fail-soft*: a search that returns the real value it found even when that value lies outside the window it was asked
  about. The value's size then travels up the tree and into the transposition table (TT).
- *Scout*: a cheap search with a window one unit wide that only answers "is this move better than the bound, yes or no".
- *LMR* (late-move reduction): later moves in the list are first searched to a reduced depth.
- *Pre-search*: our root's own shallower pass (`reorder_legal_moves` → `pre_minimizer`) that orders the root moves and
  their replies before the main pass. About 38% of all nodes.
- *Count-only build*: a binary with counters added and no behaviour change; it runs at fixed depth, which is safe beside
  the owner's gaming but not beside timed games.

---

## 1. Summary

**Triage counts (49 item rows in §2a-2c, plus 11 parked items re-read in §2d):**

| verdict | rows | meaning |
|---|---|---|
| GAMES NOW | 2 | worth a run as soon as a slot is free |
| COUNT FIRST | 12 | a fixed-depth counter decides; four of them will probably lead to a run |
| FIX-ONLY | 11 | trivially correct or result-identical; no solo run (bundle, or ship on identity) |
| PARK | 17 | with a named trigger |
| NOT REAL | 6 | with the proof |
| already done | 1 | LAZY-EVAL W1 (read on 11-19) |

(The plan's "TT replacement: no aging" bullet is the same item as S2-29 and is not counted twice.)

**Runs I recommend (mine only; the other readers' items are in §6):**
1. The mate-score bundle on gen 7.7 (`FIX_TT_MATE`, `FIX_MDP`, `ENABLE_NULL_MATE_CLAMP`, plus the S2-14 guard). Trigger
   fired on 11-19. One run. §4.
2. TT replacement (S2-29). A count picks the form, then one run. §2.
3. The pre-search pair S2-4 + S2-10, shaped by #284's result. One run. §3.
4. The PV block S2-7 (+ S2-18, sized by S2-28's count). One run, after its count.
5. One "hygiene bundle" of the small fixes that change the bench (S2-8, S2-11, S2-16, S2-17, S2-32), judged on the
   not-harmful rule that shipped gen 6.5's seven fixes. One run.

**What I would not spend games on now:** the return-form × margin question (§5: one run cannot resolve the size of effect
it could plausibly have; it belongs at the head of the C7 block), S2-25, S2-30, S2-22, S2-23, S2-31, PS-6, A6.

**Where the register is wrong or overstated** (details in §7): S2-10's "nearly every reduced move is searched twice",
S2-7's one-line fix, S2-11's distance, S2-12 as a separate defect, PS-3 as a defect rather than a heuristic, PS-6 as an
open item, A6 as an item at all, and S2-25, where the kink is the small part of a larger design fact.

**One finding for the S2-33 reader** (VERIFIED as code, REPORTED as a cause): with `FIX_TT_MATE` off, **no mate score is
ever written to the TT** (`tt_score_storable`, `cache_management.h:1171-1177`), and no exact 0 either (`:1195`). An
iteration that first meets mate scores therefore re-searches every mate-scored subtree on every visit, while reverse
futility is also off in mate-range windows (`search_v2.cpp:524`) and move-loop pruning is off while every move so far is
mated (`FIX_MATE_GATE`). That is a candidate mechanism for the 200× iteration of game 56. A replay of that position with
`FIX_TT_MATE=1` is a one-position, fixed-depth check.

---

## 2. The triage table

Expected effects are estimates. "Bundle" means the hygiene bundle of §6, run 9.

### 2a. Sweep items (register §5, §6, §6b, §6c)

| # | verdict | checked | reason | expected effect |
|---|---|---|---|---|
| **S2-7** stale PV tails feed move ordering | **COUNT FIRST**, then one run | VERIFIED: `updatePV` (`search_engine.cpp:10513`) copies `pv_length[cur_depth + 1]` moves from the row below; `pv_length` is cleared only at `alpha_beta` entry (`:5023`, `:5049`). A child that returns without entering a node (a TT hit or a draw inside `sv2::score_move`, `search_v2.cpp:1026-1072`) leaves the row below holding another subtree's line, and `negamax` then calls `updatePV` (`:871-872`). The PV feeds `moveFrequency` after every iteration, cut or complete (`search_engine.cpp:4668-4676`). | Real and frequent in mechanism. The count is free: the UCI layer already replays the PV and cuts it at the first illegal move, so "PV length written vs PV length legal" per iteration sizes it. The fix changes ordering at every node, so it needs a run. **The register's fix is incomplete** (§7). | −5 … +8 Elo; unknown sign because `moveFrequency` is our own term and was tuned with the stale tails in it |
| **S2-8** delta pruning skips checking captures | **COUNT FIRST**, then bundle | VERIFIED: `search_v2.cpp:1513-1524` exempts promotions and en passant only. Stockfish also exempts moves that give check. | Count the share of per-capture delta prunes that give check, and the bench cost of exempting them (an in-check qsearch node searches every evasion, `:1435-1467`). | 0 … +4 Elo, +0.5 … +2% nodes |
| **S2-10** pre-search scouts at the wrong bound | **COUNT FIRST**, then the pair with S2-4 | VERIFIED: `pre_minimizer` scout at `:9236`, re-search test at `:9260`. Design in §3. | Real, but smaller than the register says: a fail-soft scout that already returns ≥ beta skips the re-search today. Four counters decide the size. | −1 … −5% bench nodes (estimate); −3 … +5 Elo |
| **S2-11** repetition detection crosses a null move | **FIX-ONLY** (bundle) | VERIFIED: the path scan in `is_repetition_key` (`search_engine.cpp:10328-10338`) stops only at an irreversible move; the null move is made in place (`search_v2.cpp:549-552`) and pushes nothing, so nothing marks it. | Rare and nearly harmless: the false draw needs both sides to cooperate in undoing moves, and the side that would have to cooperate is the one that does not want the draw. No count needed; stop the scan at the null. | ≈ 0 |
| **S2-12** qsearch lazy request has no mate-range guard | **NOT REAL** | VERIFIED: `search_v2.cpp:1471-1492`. With beta ≥ 9,000,000 the lazy exit can never fire (the estimate would have to exceed a mate score). With beta ≤ −9,000,000 it fires and returns an ordinary eval, which is a legitimate stand-pat fail-high there. Reverse futility's guard protects the *prune*, not the lazy request. | Nothing mate-specific goes wrong. What remains is S2-9 (an estimate returned as a value). | none |
| **S2-13** the seed-off family | **NOT REAL now** | Register: closed by C8, shipped in gen 7.7 (`ENABLE_QSTANDPAT_SEED`, `search_v2.cpp:1499`). | — | — |
| **S2-14** null move with a losing-mate beta | **FIX-ONLY**, rides in the mate bundle | VERIFIED: the null-move condition (`search_v2.cpp:544-545`) has no test on beta; reverse futility has one (`:524`). | A reduced null search "proves" the side is not mated in N. It only fires in mate-range windows, so it belongs with the three mate knobs and with S2-33. | ≈ 0 Elo; shorter mates |
| **S2-15** losing checking captures are SEE-pruned | **NOT REAL** | VERIFIED: `search_v2.cpp:1171-1179`. Stockfish also prunes losing captures that give check. Only the comment is wrong. | Fix the comment. Coupled to S2-1 (its `!extend` test), so it moves with that item. | — |
| **S2-16** the q-cache stores path-dependent draws | **COUNT FIRST**, then bundle | VERIFIED: `addToQCache` (`cache_management.h:1075-1087`) refuses mates only; `qsearch_entry` (`search_v2.cpp:227-228`) refuses a 0 only when the q-root itself is a repetition. A q-root in check searches every evasion (`:1435-1467`), each child tests repetition (`:1385`), so a perpetual at the horizon returns 0 from below the q-root and is stored, then served on other paths. | Count q-cache stores with value 0. Fix: skip an exact 0, as the TT does. | 0 … +3 Elo, in perpetual-check lines |
| **S2-17** qsearch has no 50-move test | **FIX-ONLY** (bundle) | VERIFIED: `qsearch` entry (`:1385`) tests repetition only. | Every non-check q-ply is a capture, which resets the clock; only one evasion ply can cross the limit. | ≈ 0 |
| **S2-18** PV rule differs by side | **FIX-ONLY**, shares S2-7's run | VERIFIED: `search_v2.cpp:871`. In the root frame both sides test "score above root alpha", so the minimizer writes a PV row on its fail-lows and the maximizer on its cutoffs. The true PV (exact nodes) is written correctly by both. | It has no effect of its own beyond the rows S2-7 copies. | — |
| **S2-19** the "deeper" rule ignores a deeper TT answer | **FIX-ONLY**, inside C7 | VERIFIED: `first_depth = reduced_depth` (`:1250`) even when `tt_use` answered the scout (`:1258-1264`). | Narrow: an entry at full depth is already used by the first probe (`:1060-1067`), so only an entry exactly one ply deeper than the verify depth is overruled. | < 2 Elo |
| **S2-20** `pre_minimizer` never checks `time_up` | **FIX-ONLY** | VERIFIED: no test in the loop (`search_engine.cpp:9105-9370`). After the stop, every remaining child returns 0, the loop takes min(…, 0), may cut, and writes killer, history and the move-cache order (`:9347-9366`). | Result-identical for any search that completes. It fires on **every move of a node-limited game** (the SF18 and SF1.1 judges), once per move. Ship on bench identity. | ≈ 0; cleans the outside judges |
| **S2-21** a partial qsearch value through a null cut | **FIX-ONLY** | VERIFIED: the qsearch loops do not test `time_up` after a child (`search_v2.cpp:1456-1465`, `:1531-1545`); the null cut returns the value (`:603-610`) and the parent stores it (`:1083-1085`). | One entry per stopped search. One line: test `time_up` after the null search. Result-identical for complete searches. | ≈ 0 |
| **S2-22** the ep square enters every key | **PARK** | REPORTED (make / update code is in `cpp_bitboard.h`, the owner's file). | A position after a double push can never repeat as such, so the cost is one late repetition in a cycle that starts right after a double push, and a rare split transposition. Trigger: the release sweep, or any change to the key functions. | ≈ 0 |
| **S2-23** raw history adds outside search v2 | **PARK** | VERIFIED: `pre_minimizer` `:9355-9361`, `:9393` (raw `+=`, the continuation add at 4×). | Ply 1 only, bounded by the decay. Trigger: history stack A, when every history write goes through one function. | ≈ 0 |
| **S2-24** iterative deepening stops on any mate-range score | **NOT REAL now** | Register §5b: counted, no difference with the seed on or off; C8 removed the phantom source. | The remaining case (an unproven mate from a null search ending the loop) is `ENABLE_NULL_MATE_CLAMP`'s, in the mate bundle. | — |
| **S2-25** the null-move target table is non-monotone | **PARK** | VERIFIED: `DEPTH_REDUCTION[9] = [10] = 8` (`search_engine.h:85-86`), minus `NULLMOVE_EXTRA` 2, minus 1 from depth 10 (`search_v2.cpp:554-557`). Also verified: the table is flat at 12 from index 14 (`:90-93`). | The kink is one ply at one depth. The larger fact is the form: the null child's target is an absolute ply, so from iteration 14 every node deeper than ply 8 gets a null search of zero depth (qsearch only). That is a design to test against the standard relative form, not a bug. Trigger: Stack E's null-move work. | kink ≈ 0; the form unknown and possibly large |
| **S2-27** null cuts stored at full depth | **NOT REAL** as a defect | Standard practice in every engine read. | The origin tag (searched / static cut / null cut) is an instrument worth building when S2-26 or S2-9 is reopened. | — |
| **S2-28** `moveFrequency` enters ordering unscaled | **COUNT FIRST** (with S2-7) | VERIFIED: added raw to capture and quiet scores (`move_gen.h:1302`, `:1573`); the bonus is (depth − 1)² × (PV length − i)² per iteration (`search_engine.cpp:4674`), about 17,000 for the first PV move at depth 12 alone, against a history cap of 16,384; decay is a shift by 2 at each search start and every 5 iterations from depth 13 (`:4431`, `:4504`). | Our own term: size it, do not remove it. Its scale is a constant, so by the 11-19 rule it waits for the tuning pass unless the count shows a large step. | unknown |
| **S2-29** TT replacement is strictly depth-preferred | **GAMES NOW** (count picks the form) | VERIFIED: `addToSearchEvalCache` replaces a same-position entry only when `entry.depth <= depth_used` (`cache_management.h:1218`); another position always takes the slot (`TT_WAYS` 1, `search_engine.h:3551`); no age. | Touches every store. A deep entry whose bound does not cut blocks the fresh result of every shallower search of that position (every reduced scout of a position the last iteration searched in full). Stockfish replaces unless the new entry is 4 or more plies shallower, or when the new bound is exact (from memory, REPORTED). A fixed-depth bench cannot show table pressure, because the bench clears the table per position, so the count only picks the form; games decide. | −3 … +12 Elo |
| **S2-30** the null move has no eval gate | **PARK** | VERIFIED: `ENABLE_NULL_EVAL_GATE` false (`search_engine.h:2909`); the node-entry eval exists only at remaining depth ≤ 6 (`search_v2.cpp:494-497`). | Needs the node-entry eval at every depth. Trigger: S4, then Stack E (already planned there). | — |
| **S2-31** the ply-1 malus ignores `SV2_MALUS_SEARCHED_ONLY` | **PARK** | VERIFIED: the knob has no read site in `search_engine.cpp` except its start-up read (`:3329`). | Only matters to C5's re-test. Trigger: history stack A. | — |
| **S2-32** qsearch searches all four promotions | **COUNT FIRST**, then bundle | REPORTED (noisy list builder not read). | Count under-promotion children per qsearch node on the endgame set. Keep the knight (it can give check). | 0 … +1 Elo; a small endgame speed gain |
| **PS-3** front root move picked by upper bounds after an aspiration fail-low | **COUNT FIRST** | VERIFIED: `descending_sort_wrapper` swaps the largest `top_score` of the searched prefix to the front (`search_engine.cpp:9850-9866`); `ENABLE_ROOT_TABLE` is off (`search_engine.h:1444`), so after a failed-low pass every `top_score` is a fail-soft upper bound. | This is a heuristic, not plainly a defect: after a fail-low the old best move has just been refuted, and "the move whose bound is highest" is a guess at its successor. The count must classify the outcome: of the re-search passes whose front changed, how often did the promoted move end as best, the old best, or a third move. | −3 … +6 Elo, at critical moments |
| **PS-6** S2-3 reaches the root razor | **PARK** | REPORTED. | Already judged: #280 played `SV2_FUT_BOUND=1`, which changes exactly the bounds the razor reads, and read +1.4 (−10.7 … +13.5). Trigger: the root-razor work, with S2-3. | inside #280 |
| **PS-7** S2-2 inside the pre-search | **COUNT FIRST** (the S2-2 reader's) | VERIFIED: `pre_minimizer` horizon, `:9041-9058`, same gate as the main search. | Split S2-2's counter by pre-search and main. | with S2-2 |
| **PS-8** `PRESEARCH_TAIL_MODE` 1-3 escape the depth fix | **PARK** | VERIFIED off: header 0 (`search_engine.h:1511`). | Latent. Fix before the knob is used. | — |
| **PS-9** `PLY1_NODE_PRUNE` returns a static eval as a depth-D result | **PARK** | VERIFIED off: header 0 (`:3211`). | Latent, and it taints H1 / H2's earlier results (ROOT_SEARCH_MODEL §5). Fix before H2 is compared again. | — |
| **PS-10** three root stores are safe only because a 0 is refused | **PARK** | VERIFIED: the refusal is `cache_management.h:1195`; `ENABLE_TT_STORE_DRAW` false. | Latent. It becomes live the day draws are stored. | — |

### 2b. "Open beyond C7-C9" (plan §0b)

| item | verdict | reason |
|---|---|---|
| LAZY-EVAL W1 overdue | **already done** | Read in #274 on 11-19: SF18 control − `LAZY_V2=0` = −0.47 pp (z −0.30), lazy eval stays (OPTIMIZATION_LOG, gen 7.7 entry). Due again after the next pruning ship. |
| `ENABLE_NULL_MATE_CLAMP` never played alone | **GAMES NOW**, in the mate bundle | §4. |
| TT cutoffs at open windows (22% of TT cuts) | **COUNT FIRST** | VERIFIED that they exist: `use_tt_entry` cuts on an exact entry at any window (`search_engine.cpp:10203`). They are also what truncates the PV (S2-7). Expect a park: not cutting adds nodes. Read it in S2-7's count build. |
| Child futility skipped at ply 1 on the non-root side (A6) | **NOT REAL** | VERIFIED: `search_v2.cpp:1190` needs remaining depth ≤ 4 and `depth_limit ≥ 5`; at ply 1 that is iteration depth 5 only. Games play at depth 11.7. |
| C3 and C5 screened only on gen 7.4 | **PARK** | C5 was played on gen 7.6 (#275, −3.0, neutral). C3 (`SV2_STATIC_CUT_STORE`) was never played; it is the "reduced-depth store" form that S2-9's and S2-26's re-test triggers ask for. It joins the C7 block (§5), after a fixed-depth check on gen 7.7. |
| SW4: shadow logger, line tracer, prune log not ported to v2 | **PARK** | Tooling, no games. Trigger: before Stack E reshapes pruning (the false-cut audits need it). |
| Stack 0 counters last read on gen 7.3 | **COUNT FIRST** | A free read at the next count build. |
| A3 pre-search subset | **COUNT FIRST** | Its −2.9% nodes was measured before S2-10; the two overlap (both remove open-window pre-search work). Re-read its saving on the base that has S2-10. Alone it cannot pass a ≥ +2σ rule (savings under ~35% have been Elo-neutral here); its use is as the node-saving partner of a parked node-adder. |
| 32-bit node counter | **FIX-ONLY** | Overflows after about 22 minutes at 1.6 M nodes per second (analysis mode only). Result-identical; a wide mechanical diff, so check speed. Release sweep. |
| `FIX_REP_KEY` bit 2 never judged where it fires | **COUNT FIRST** | VERIFIED live path: `is_repetition_key`, `search_engine.cpp:10341-10352`; shipped value 1, so bit 2 is off. Count repetitions whose earlier occurrence is before the root; if material, judge it on a repetition-heavy venue, not on general games. |
| Qsearch Q5: in check at the q-depth cap | **PARK** | VERIFIED: `search_v2.cpp:1401-1419`; `MAX_QDEPTH` 10, so it needs ten q-plies. Count it for free with the others. |
| TT replacement: no aging | **= S2-29** | Same run. |
| Unstable root `std::sort` | **FIX-ONLY** | Changes the fingerprint once. Release sweep. |
| MOVEGEN C2 / C4 / C6 | **PARK** | C2 latent (two knobs off), C4 rare (en passant ordered as a quiet), C6 unreachable (an assert). |
| Speed: W3, W8, batch 9 (b) / (c), F27-F29 | **FIX-ONLY** | Result-identical. The speed track: identity on the bench sets, then time to depth. |

### 2c. Latent items (register §6, §6b)

| item | verdict | trigger |
|---|---|---|
| ProbCut's child bypasses `score_move` | **PARK** | before Stack G turns ProbCut on (VERIFIED: `search_v2.cpp:712` calls `negamax` directly; `ENABLE_PROBCUT` false) |
| Singular exclusion with one legal move | **PARK** | before Stack F |
| `see()` scores en passant one pawn low | **PARK** | with S2-1's `see()` work |
| `bench` state leaks into `go`; `position` before `isready`; `go depth 1/2` searches depth 3; Resign plays `legal[0]` | **FIX-ONLY** | release sweep (UCI layer, no search change) |
| Correction-history update at a fail-low; `moveGivesCheckFast` on captures | **PARK** | before correction history or a capture caller is turned on |

### 2d. Parked items: is any re-test trigger met?

| parked item | trigger | met? |
|---|---|---|
| Mate-score bundle (gen 6.6) | "re-test on top of C8" | **YES, since 11-19.** §4. |
| `ENABLE_SEE_PRUNE` (void) | the `see()` call fixed | **Fires when S2-1 is resolved.** |
| C6 `ASP_KEEP_FAILHIGH`, C9 root clock exits | aspiration or the time manager's cut reshaped | **Fires if S2-33's time-manager part ships.** A 100-game re-count, not a run. |
| `FIX_LMR_EG` (−19.8 on gen 6.4, v1) | "base changed since" | **Loosely yes.** VERIFIED that it reaches search v2: `reduced_search_depth` (`search_engine.cpp:10472-10476`) is what `score_move` calls (`search_v2.cpp:1215`). In light endgames every late quiet move is scouted at one absolute target, which S2-2 then turns into a static eval. It belongs to the endgame cluster (S2-2, PS-7, S2-33), not to a run of its own. |
| S2-9 mode 2, S2-26 | a return form paired with a margin re-sweep; a reduced-depth store | The owner asked for the design: §5. |
| `RFP_THREAT_GATE` | whenever pruning is reshaped | Not yet. It fires if `RFP_MARGIN` is re-swept (§5), as an arm of that sweep. |
| C4 history gate / bound | C7 or a history-scale change | Not yet (C7 is last). |
| C5 | S2-31 fixed | No. |
| S2-3 | futility reshaped, or the root-razor work | No. |
| S2-9 mode 1 | mode 2 passes, or a node saver frees ~25% | No. |
| `FIX_TT_R50`; D7 = 2; lazy generation; early LMP; the root configurations (ROOT_SEARCH_MODEL §5) | as written there | No. |

---

## 3. S2-10: the pre-search scouts at the wrong bound

### What the code does (VERIFIED)

`pre_minimizer` is the reply node of a root move in the pre-search: a minimizing node with window (alpha, beta). It
lowers beta as replies come in (`beta = std::min(beta, lowest_score)`, `search_engine.cpp:9344`). For a late quiet
reply it:

1. scouts at reduced depth with the window **(alpha, alpha + 1)** (`:9236`);
2. if the scout's score is below beta, searches again at full depth with (alpha, beta) (`:9260-9267`).

A minimizing node's question about a late move is "is this reply better for me than my best so far", which is the window
(beta − 1, beta). The scout at (alpha, alpha + 1) asks instead "does this reply refute the root move outright". When
the call's window is one unit wide the two questions are the same, so only **open-window calls** are affected: root
move 0 of each pass (`:8829`, with the root's aspiration window, 1 pawn wide: `ASPIRATION_DELTA` 500) and any root move
whose scout landed inside the window (`:8949`).

### Why it is smaller than the register says

The register says "nearly every reduced move is searched twice". Not so: the child is fail-soft, so the scout often
returns a lower bound that is already ≥ beta, and then step 2 is skipped. The waste is confined to replies whose
scout bound lands between alpha + 1 and beta, which are the replies close to the best one. Early quiet replies whose
reduction is zero are a second case: they are searched twice at the same depth with two windows.

### The knob (pseudo-diff, NOT applied)

```cpp
// search_engine.h, namespace Config, beside PS_TT_FLAG_FIX
// PS_SCOUT_FIX (S2-10): pre_minimizer's reduced scout asks the minimizing node's own question, (beta - 1, beta) with
// beta as narrowed so far, instead of (alpha, alpha + 1). The same search in a null-window call. 0 = byte-identical.
inline int PS_SCOUT_FIX = 0;

// search_engine.cpp, start-up reads, beside PS_TT_FLAG_FIX
Config::PS_SCOUT_FIX = env_int("PS_SCOUT_FIX", Config::PS_SCOUT_FIX);

// search_engine.cpp, pre_minimizer, inside `if (do_lmr)`, before the scout
+   const int sa = Config::PS_SCOUT_FIX ? beta - 1 : alpha;      // the scout window is (sa, sa + 1)
    ...
-   score = ... sv2::negamax<true>(cur_depth + 1, reduced_depth, alpha, alpha + 1, ...)
+   score = ... sv2::negamax<true>(cur_depth + 1, reduced_depth, sa, sa + 1, ...)
    if (cur_depth < reduced_depth - 1)
    {
-       if (score <= alpha)          { flag = UPPERBOUND; addToSearchEvalCache(..., alpha, alpha + 1, ...); }
-       else if (score >= alpha + 1) { flag = LOWERBOUND; addToSearchEvalCache(..., alpha, alpha + 1, ...); }
+       if (score <= sa)             { flag = UPPERBOUND; addToSearchEvalCache(..., sa, sa + 1, ...); }
+       else if (score >= sa + 1)    { flag = LOWERBOUND; addToSearchEvalCache(..., sa, sa + 1, ...); }
    }
    if (score < beta) { /* full-depth (alpha, beta) search: unchanged */ }
```

Values: 0 (today) and 1. The exact-only TT probe before the scout and the re-search test stay as they are. Gates: the
default bench must be 254 / 35,922,538; `PS_SCOUT_FIX=1` must change the node count (proof that it is live).

### Count first (four counters in a count-only build, open-window calls only)

`ps_scout` (reduced scouts), `ps_scout_skip` (scout ≥ beta, no re-search), `ps_research` (re-searches), and
`ps_research_fh` with its nodes (re-searches that came back ≥ beta: the work the fixed scout would mostly have
settled at reduced depth). Registered reading: if the nodes under `ps_research_fh` are below 1% of all nodes on game12,
S2-10 has no run of its own and simply rides with S2-4.

### Predicted effect (estimate)

Bench nodes −1% … −5%, centre −3%; game depth +0.0 … +0.05 ply; Elo −3 … +5. The risk is the supplier role: the
pre-search's reply scores order the ply-1 replies of the main pass, and more of them become reduced-depth bounds.

### Pairing with S2-4: neither a blind bundle nor a full 2×2 in games

S2-4 (`PS_TT_FLAG_FIX=1`, #284, +1.8% bench nodes) and S2-10 act on the same nodes and are not independent: S2-10
removes most of the full-depth re-searches whose stores S2-4 relabels (`:9272-9287`), so with S2-10 on, S2-4 has less to
fix. That rules out reading a bundle as the sum of its parts. A full 2×2 in games would cost two more runs and could not
resolve an interaction of the expected size (three arms against one base give the interaction a 95% interval near ±20
Elo).

So: **the 2×2 at fixed depth, one more run in games, chosen by #284.**

1. Fixed depth, all four cells (base, S2-4, S2-10, both): bench and game12 nodes, solved counts, `g_ps_exact_mislabel`,
   the four counters above, and time to depth. This gives the attribution that games cannot.
2. Games:

| #284 (S2-4 alone) reads | next run | ship rule |
|---|---|---|
| passes (≥ +2σ) | S2-4 ships; then S2-10 alone on the new base | S2-10: not negative and time to depth not worse |
| neutral (the registered expectation) | **the pair, both on, against gen 7.7** | the pair is a correctness fix plus its node saver, which is the owner's park-and-pair rule of 11-18: read as one item |
| negative (≤ −2σ) | S2-4 parked as a productive bug; S2-10 alone | as the first row |

A second pair run (S2-10 alone) is added only if the pair reads below −1σ, to find which half carries it.

---

## 4. The mate-score bundle

### Do the three knobs still reach gen 7.7's code? Yes, all three (VERIFIED)

| knob | what it does | where it acts today |
|---|---|---|
| `FIX_TT_MATE` | stores a mate score in the TT as its distance from the stored position, and converts it back at the reader's ply | `tt_score_storable` / `tt_score_to_entry` / `tt_score_from_entry` (`cache_management.h:1171-1189`), used by `addToSearchEvalCache` (`:1195-1197`, every store of every search, search v2 included) and by every reader: `use_tt_entry` (`search_engine.cpp:10202`, reached from search v2's `tt_use`), `tt_entry_cuts` (`search_v2.cpp:322`), the null-move probe (`:577`), the pre-search's exact probe (`search_engine.cpp:9227`) |
| `FIX_MDP` | mate-distance pruning: a node whose window the mate bounds alone settle returns the bound | `sv2::negamax` entry (`search_v2.cpp:448-454`); the ply-1 `minimizer` excludes itself (`search_engine.cpp:6684`) |
| `ENABLE_NULL_MATE_CLAMP` | a mate score returned by a null-move cut becomes the bound | `search_v2.cpp:607-608`; ply 1 `search_engine.cpp:6946`. Its header value is `true` but the start-up read defaults it to false (`:2153`) and it is read again after the preset (`:2851`), so it is off in production |

The stand-pat seed does not touch any of the three. It does change what they were measured against: on gen 6.6, 5-10%
of qsearch visits returned a mate-sized value while not in check, and `FIX_TT_MATE` made those values storable at full
depth. So the register's "doubtful, re-test on top of C8" is right in mechanism.

### What is at stake

With `FIX_TT_MATE` off, a mate score is **never** stored (`tt_score_storable` returns false). Every subtree that
returns a mate is searched again on every visit. That costs nothing in an ordinary middlegame and a great deal in a won
ending or a mating attack, which is where S2-33's 149-second iteration sits. Of the three knobs this is the one with
playing value; the other two are hygiene.

### The record (OPTIMIZATION_LOG 11-04 / 11-05)

Gen 6.6 against gen 6.5: self-play pooled ≈ −5.7 ± 7.0 (passes not-harmful), SF18 at equal nodes −3.60 pp (z −1.43) and
−3.10 pp with the flag fix (z −1.22), both under the −1σ bar; +6.2% nodes per fixed depth on the bench, with game depth
11.2 against 11.3. The note left open "which of the three carries the SF18 cost, and whether it is the +6.2% nodes".

### A clean re-test

1. **Fixed depth first, per knob** (free): bench and game12 for each knob alone and for all three on gen 7.7; the 64 WAC
   mates' distances (the sweep reported 3 scored one move short); and the S2-33 position replayed with `FIX_TT_MATE=1`.
   The bench is mate-heavy (63 of 300 searches end on a mate score), so its node change overstates the cost in play.
2. **Games: one bundle run, not three.** The three knobs act only on lines that contain a mate score, they share one
   subject, and none of them alone can move 2,000 games by a resolvable amount. Add the S2-14 guard (no null move when
   beta is a losing mate) as a fourth switch in the same bundle. The fixed-depth split of step 1 gives the attribution.
   Registered expectation (estimate): −6 … +8 Elo.
3. **Split only on a failure.** If the bundle reads below −1σ in self-play or again below −1σ against SF18, run
   `FIX_TT_MATE` alone (the knob with the value and the likeliest carrier of the node cost), and keep the rest parked.
4. **Venue.** General self-play barely exercises these knobs. If S2-33's reader builds a won-ending or mate venue, the
   bundle should also be read there (memory `narrow-items-are-judged-where-they-fire`).

Ship rule: this is a correctness item, so the rule that applied on 11-05 applies again: not harmful in self-play, and
SF18 above −1σ. Under the present trial (outside judges only after a self-play pass) "pass" has to be defined for
correctness bundles; I suggest "self-play above −1σ", as on 11-05. That choice is the owner's.

---

## 5. The pattern: two forms that shrink a static cut's return both read below zero

### The two results

| run | change | self-play | bench nodes | game depth |
|---|---|---|---|---|
| #281 | `LAZY_FAIL_HARD=2`: a lazy estimate returns "estimate − margin" at reverse futility and at the stand-pat cut | −10.9 (−22.4 … +0.5) | +3.9% | 11.7 vs 11.6 |
| #282 | `RFP_RETURN_BLEND=67`: reverse futility returns a value two-thirds of the way from the eval to beta | −4.0 (−15.3 … +7.3) | +0.34% | 11.7 vs 11.7 |

Each is inside noise on its own. Averaged they give about −7.4 ± 8.1 (z ≈ −1.8), which is suggestive and not
established; the two changes are also not independent tests of one thing.

### Were the margins tuned around the raw return? How each was set

| margin | value | how and when it was set | set against |
|---|---|---|---|
| `RFP_MARGIN` (reverse futility: prune when eval − margin × remaining depth ≥ beta) | 1500 (v1) → 1000 (09-18) → **600 (gen 6.4, 10-30)** | 1000: a WAC node sweep (−17.0% nodes at identical solves) plus a margin A/B in games (EVAL-V2-REBUILD-LOG, "2026-09-18 — MARGIN A/B"). 600: a games 2×2 with threats in the eval (600 alone −27.5 ± 19.1 on gen 6.3, with the threats term +73.0), then a plateau check at 800 and 400 (#116; 400 alone −5.2 ± 18.9). Tried since: 500 with two eval arms on 10-31 (+7.0, +15.3), 500 with CORR2 on 11-10 (+6.9, −7.6 … +21.5). (OPTIMIZATION_LOG 10-29 … 11-10.) | games at 10+0.1, 1,000 per arm (±19), on a 200-mp grid, always with the raw return and the full-depth store. Not re-checked since the gen 7 retune, lazy eval, D7, C2c, PROMOTE or the seed |
| Lazy margins `LAZY_V2_M` 1500 / `LAZY_V2_M_CHEAP` 3000 | since gen 7.1 (11-09) | from the miss rate per lazy exit (a miss = the estimate says "cut", the full eval says "no cut") and the SF18 leg at equal nodes; the narrower 1000 / 2000 pair failed that leg (#184, −4.2 pp) (LAZY-EVAL.md) | the *cut decision*, never the returned size; every game validation had the raw estimate returned |
| `VERIFY_MARGIN` 16000, `VERIFY_RESEARCH_REDUCTION` 2 | v1 era, never refit on v2 (REPORTED, from the knob's comment, `search_engine.h:3213-3221`, `:3317-3322`) | widened from 6000 for one game's buried capture line; STS +62, self-play +3.7 ± 31 | fail-low sizes with D7's defect in place (the reason C7 exists) |

**Reading.**
- `RFP_MARGIN`: yes in the plain sense. Every value was chosen in games with the raw return in place.
- `VERIFY_MARGIN`: no, in practice. At 16 pawns it already re-searches nearly every reduced fail-low, so a shift of 1 to
  3 pawns in returned sizes does not change which moves it verifies. The register's own ablation agrees (+26.8% nodes
  for mode 1 with the verify off, against +28.9% with it on).
- Lazy margins: they were set on a different quantity, so they are not the coupling either.

**And a caution about the hypothesis itself.** The blend moved the bench by 0.34% of its nodes. A change that touches so
little of the tree cannot have displaced a margin optimum that was itself located on a 200-mp grid at ±19 Elo. For
S2-26 the simpler readings are "about zero" or "a small per-depth cost", not "a mis-set margin". S2-9 mode 2 is the
larger change (+3.9%, most of it at the stand-pat cut), and its registered first step, the split by site, is still
undone.

One consumer that reads returned sizes with a *tight* margin was not ablated in either run: the **root razor**, which
drops a root move when alpha − its bound exceeds 100 to 300 mp (`search_engine.cpp:5061-5069`, `:5321`). Root bounds are
built from these fail-soft returns. Whether the razor fires more or less under each form is a fixed-depth counter
(`g_razor_fires`), and the project rule is to ablate, not to read, when explaining why a number moved.

### The smallest paired test

**Step 0, no games** (fixed depth, about ten bench runs): a 3 × 3 grid, return form {raw, blend 67, lazy mode 2} ×
`RFP_MARGIN` {600, 500, 400}, on the bench and game12, recording nodes, solved, `g_razor_fires`, the verify re-search
counters and TT cuts. Plus the S2-9 split by site. Registered reading: if the margins are tuned around the raw return,
the node response to the margin differs between forms by more than 1%; if the three forms respond alike, the hypothesis
has no mechanism and stops here.

**Step 1, one run (5.6 h), only if step 0 shows a form × margin difference:** a head-to-head on one binary,
**blend 67 at `RFP_MARGIN` 500 against raw at `RFP_MARGIN` 500**. Read beside #282, which is the same pair at 600
(−4.0). The direction is 500, not 700: the raw return makes later TT probes cut at margin zero, so removing it makes
pruning less aggressive, and the compensating move is a tighter margin.

- "Tuned around the raw return" predicts the form's effect rises clearly at 500 (registered: ≥ +8, a swing of 12 or more).
- "The raw value is simply the better estimate" and "noise" both predict −10 … +4.

**What this can and cannot resolve.** The difference between two independent 2,000-game results has a 95% interval near
±16 Elo. Step 1 answers the question only if the interaction is that large. For comparison, the one margin interaction
on record that was resolved, threats × RFP on 10-30, was +62 ± 39.

**Step 2, only if step 1 passes +2σ:** blend 67 at 500 against gen 7.7, the ship question.

**The other half, the store rather than the value.** If step 0 shows no form × margin difference, the remaining live
hypothesis is the register's own: the defect is the full-depth TT store of an unproven size. The knob for that exists
(`SV2_STATIC_CUT_STORE`, value 3 = stored one ply shallower; `search_v2.cpp:41-45`) and was never played. It needs no
margin partner and is one run.

**Recommendation.** Do step 0 at the next free fixed-depth window. Do not spend a run on step 1 now: by the 11-19 rule
both forms have been tested once at a sensible value and read neutral, and the plan already has C7 retuning `RFP_MARGIN`
together with the verify margins. Step 1 or the store form goes at the head of that block.

---

## 6. Recommended order for the rest of phase H

Each self-play run is 5.6 h. Outside judges (SF18 and SF1.1, about 5 h) follow only a pass. Count builds and benches are
fixed-depth and fit the owner's gaming windows, never beside a timed queue.

**Ordering rules used:** (1) a fix that changes what everything else was tuned around goes early; (2) larger expected
effect first; (3) a result that fires another item's trigger goes before that item; (4) C7 last, by rule.

| slot | run | items sharing it | hours | why here |
|---|---|---|---|---|
| in flight | #283 `ROOT_M0_CUT=1` (S2-6) | — | 5.6 | queued |
| queued | #284 `PS_TT_FLAG_FIX=1` (S2-4) | — | 5.6 | queued; its reading picks slot 6 |
| free work A | count build "cnt20" + benches | counters for S2-10, S2-29, S2-7 / S2-28, S2-8, S2-16, S2-32, PS-3, PS-7, open-window TT cuts, Q5, `FIX_REP_KEY` bit 2, Stack 0; the mate knobs split; the S2-33 replay with `FIX_TT_MATE=1`; the §5 grid | ~1 h fixed depth | decides half of the list below before any game is spent |
| 1 | S2-5 decay interval (other reader) | — | 5.6 | all tuning has lived in the stuck setting; settle it before anything is tuned |
| 2-3 | S2-1 check extension, two arms (other reader) | — | 11.2 | the largest expected effect; its result fires `ENABLE_SEE_PRUNE`'s trigger |
| 4 | **mate-score bundle** | `FIX_TT_MATE` + `FIX_MDP` + `ENABLE_NULL_MATE_CLAMP` + the S2-14 guard | 5.6 | trigger fired 11-19; may also carry S2-33's search half |
| 5 | S2-33 (other reader) | its time-manager half fires C6 / C9's re-count (100 games, 0.3 h) | 5.6 | after slot 4, so the mate bundle is on or off the base |
| 6 | **pre-search pair** | S2-4 + S2-10 (or S2-10 alone, per §3) | 5.6 | closes the pre-search's two known defects together |
| 7 | **TT replacement** (S2-29) | same-position slack and, if its record allows, `TT_WAYS` | 5.6 | touches every store; do it before any margin is retuned |
| 8 | S2-2 endgame qsearch (other reader) | PS-7; `FIX_LMR_EG` as an arm on the endgame venue | 5.6-11.2 | the endgame cluster |
| 9 | **hygiene bundle** | S2-8, S2-11, S2-16, S2-17, S2-32 (those that pass their counts) | 5.6 | one not-harmful gate for five small fixes |
| 10 | **PV block** | S2-7 + S2-18 | 5.6 | after slot 7, since TT cuts are what truncate the PV |
| 11 | PS-3, only if its count says the promoted move loses more often than the old best | — | 0-5.6 | root, critical moments |
| no games | S2-20, S2-21, the node counter, the root sort, the UCI-layer fixes, the speed items | — | bench identity only | result-identical for complete searches |
| 12+ | **C7 block, last** | the verify retune with its honest baseline (reduction capped at 2, no verify); S2-19 folded in; C4's re-test; then §5's step 1 or the store form; `RFP_THREAT_GATE` as an arm if `RFP_MARGIN` moves | ~22-28 | by rule: retune only when the fixes are in |

**Totals (estimate).** Slots 1-11: 11 to 13 runs, 62 to 73 h. C7 block: 4 to 5 runs, 22 to 28 h. Outside judges on,
say, four passes: 20 h. About 105 to 120 h of box time, which is seven to nine days at the box's usual night and weekday
availability.

**What can share a run.** Only items that act on the same narrow subject, with the attribution taken at fixed depth:
the mate bundle with S2-14; S2-4 with S2-10; S2-7 with S2-18; the hygiene bundle. Nothing else should be bundled: the
project's own record is that search changes are antagonistic, and a neutral bundle of unrelated parts cannot be read.

**A rule the owner may want to set.** In phase H so far a neutral fix is parked (S2-3, S2-9, S2-26), which is right for
changes of form. For the FIX-ONLY class it would leave provably correct one-line fixes off for good, because none of
them can reach +2σ alone. Gen 5 and gen 6.5 shipped such fixes as bundles on "not harmful". The hygiene bundle and the
mate bundle assume that rule applies again.

---

## 7. Where the register or the plan is wrong, overstated or stale

| entry | correction | status |
|---|---|---|
| S2-10, "nearly every reduced move is searched twice" | Overstated. The scout is fail-soft; when its lower bound is already ≥ beta the re-search is skipped (`search_engine.cpp:9260`). Only replies near the best one, and unreduced early quiets, are searched twice. Size unknown until counted. | REPORTED (by reading) |
| S2-4 / S2-10 line numbers | Drifted by about +69 lines since 11-18 (the knob and its counters were added): the scout is at `:9236`, the test at `:9260`, the flagged stores at `:9272-9287` and `:9303-9318`. | VERIFIED |
| S2-7, "one store at node entry fixes it" (§6b) | Insufficient. A TT hit or a draw inside `score_move` returns before the child node is ever entered (`search_v2.cpp:1026-1072`), so a reset at the child's entry never runs and the parent still copies the stale row. The reset has to come before each child is searched (`pv_length[cur_depth + 1] = 0` in the parent or at `score_move` entry), and likewise in `pre_minimizer`, the ply-1 node and the root. | VERIFIED |
| S2-11, "at least 4-5 plies below the null" (§6b) | The shortest case is 3 plies below the null (the opponent returns a piece to its square in two moves while the null-mover undoes its last move). The conclusion stands: rare, and it needs cooperation against interest. | REPORTED |
| S2-12 | Not a separate defect; see the table. | VERIFIED |
| S2-25 | The kink is real but minor. The table is also flat at 12 from index 14, and the target is an absolute ply, so deep in the tree the null search has zero depth. That is the item worth a form test, in Stack E. | VERIFIED |
| PS-3, class "D" | A heuristic whose value is unknown, not a plain defect. | VERIFIED mechanism |
| PS-6, "count after S2-3's count" | Already covered: #280's knob changes the razor's inputs, and the run read neutral. | REPORTED |
| Plan §0b, A6 (child futility at ply 1) | Fires at iteration depth 5 only. Not an item. | VERIFIED |
| Plan §0b, "LAZY-EVAL W1 is overdue" | Read on 11-19 in #274. | from OPTIMIZATION_LOG |
| Register §4, mate bundle row | Correct, and stronger than written: with the bundle off, mates are never stored in the TT at all, which ties it to S2-33. | VERIFIED |
| Register §9.5, "S2-10: fewer nodes (LMR back at open-window pre-search nodes)" | The direction is right; "LMR back" overstates it for the same reason as the first row. | REPORTED |

**Two observations outside my brief, for the bug-hunt reader:**
- An exact 0 is never stored in the TT (`cache_management.h:1195`). The refusal is what keeps abort values and
  repetition zeros out, but it also discards every genuine 0. If the eval returns exactly 0 for its drawn classes, drawn
  endgames are searched with no TT help. Count the share of refused stores with value 0 on the endgame set.
- The null-move depth form of S2-25 (above).
