# Search defect register (2026-11-18)

Every search defect found in AlphaUX's search so far: the ones fixed and shipped, the ones tested and parked, and the
findings of the 11-18 code sweep (six read-only audits, each claim then checked in the code). It doubles as a brief for
anyone hunting more: §1 says how to read this code without being fooled, §2 lists the defect shapes that keep recurring,
and §8 lists what nobody has audited yet.

## 1. Reading this code without being fooled

- **The production engine** is the UCI binary (`uci_main.cpp`) built from these sources with `V2_PRESET=shipped`
  (`uci_main.cpp:80`).
- **The interior search** at plies ≥ 2 is `sv2::negamax` / `sv2::qsearch` in `search_v2.cpp`, `#include`d at the end of
  `search_engine.cpp`. Two parts are still the older v1 code in `search_engine.cpp`:
  - ply 1, the `minimizer` function's `cur_depth == 1` branch, which calls `sv2::score_move<false>` for each reply;
  - the root (`alpha_beta`, `get_engine_move`), the pre-search (`reorder_legal_moves` → `pre_minimizer`, about 38% of
    nodes) and the time manager.
- **Score frames.** v1 is max/min with root-frame scores; sv2 is negamax with side-to-move scores. The TT and the q-cache
  hold root-frame scores.
- **A header value is not the runtime value.** `search_engine.h` gives a knob's header default. What actually runs is
  decided by the start-up read in `search_engine.cpp` (`env_int` / `env_flag`, ~2100-3420) and by the shipped preset block
  (~2615-2815, `if (Config::V2_SHIP_GEN >= N)`).
- **Dead code.** Large parts of `search_engine.cpp` are dead: v1 interior functions, and code inside `/* */` blocks. One
  claim in an earlier sweep cited commented-out code. Prove a line runs at shipped settings before calling it a defect.
- **Units.** A pawn is 1000; `MATE` = 9,999,999 (`search_v2.cpp:101`); a mate score carries its distance through
  `state_history.size()`.
- **Instruments.** STS / WAC results (fixed-depth or equal-node) do NOT screen search changes. The 11-18 calibration
  (`dev_notes/current/INSTRUMENT-MAP.md` §D) is behind that rule. Search changes are judged by games: self-play 10+0.1,
  SF18 at equal nodes, and Mediocre.

## 2. Defect shapes that recur (one example each)

1. **A sentinel escapes as a real score.** qsearch started `best` at "mated" and returned it when every capture was
   delta-pruned (C8, §3).
2. **A returned value violates the bound it claims.**
   - A qsearch node returned less than its own stand-pat (C8).
   - A fail-low returned without its futility-pruned moves' values (S2-3).
3. **A TT flag set against the wrong window.** A re-search's flag was computed against the entry window instead of the
   drifted one, so bounds were stored as EXACT. Fixed in the main search (gen 6.7); still present in the pre-search (S2-4).
4. **Asymmetry between the two halves.** A fix or rule present only for the maximizer or only for the minimizer (gen 7.5,
   +76 Elo).
5. **A shallower search overrules a deeper one.** A verify re-search replaced the deeper first score (gen 7.4, gen 7.6).
6. **Global state that sticks.** A value set in one situation is never reset: `DECAY_INTERVAL` (S2-5), PV lengths (S2-7).
7. **A gate that reads the wrong quantity.** The endgame qsearch skip reads the line's own reduced depth, not the
   iteration depth (S2-2).
8. **A helper that is wrong for one caller's question.** `see()` with no attacker returns the piece's full value, which
   inverts the check-extension filter (S2-1).
9. **A pruning rule missing a standard exemption.** Delta pruning of checking captures (S2-8); a null move against a
   losing-mate beta (S2-14).
10. **A cached value outlives the conditions that made it true.**
    - A lazy-eval estimate stored as a search bound (S2-9).
    - A repetition draw stored in a path-free cache (S2-16).
11. **An instrument defect, not an engine defect, but it voided results.**
    - fastchess drops undeclared `Knob_X` options, so matches ran gen against gen.
    - Engine stderr written to a 9P mount (`/mnt/*`) crippled the engine.
    - History tables bled across positions in multi-position diagnostics.

## 3. Found and resolved (shipped)

| ship | knob | defect | games |
|---|---|---|---|
| gen 5 (10-21) | `FIX_DRAW_EXIT` | iterative deepening quit on a 0 score with a repetition in view; it abandoned a won queen at depth 3 | +2.6 ± 29 (no harm) |
| gen 5 | `FIX_QS_CHECK` | a mate at the q-depth cap scored as a static eval | (bundled) |
| 10-21 | — | crash (empty move) when a node or time limit cut the first depth-3 pass | fixed unconditionally |
| gen 6.5 (11-03) | `FIX_MAX_NULL` | the maximizer's null move searched the wrong window (now (β−1, β)) | bundle +8.7 ± 18.7 |
| gen 6.5 | `NULL_NPM_GUARD 2` | null-move zugzwang guard counted pawns as pieces (K+7P was null-moved); now also refuses with no non-pawn material | (bundle) |
| gen 6.5 | `FIX_MATE_50` | a checkmate on the 100th half-move was scored as a 50-move draw | (bundle) |
| gen 6.5 | `FIX_HALFMOVE` | the halfmove clock was not reset on pawn moves (it tested the post-move board) | (bundle) |
| gen 6.5 | `FIX_REP_KEY` bit 1 | repetition was keyed on piece placement alone (no castling rights or ep) | (bundle) |
| gen 6.5 | `FIX_KILLER_SPLICE` | on a move-list cache hit, killers were swapped in instead of inserted after the captures | (bundle) |
| gen 6.5 | `FIX_MATE_GATE` | move-loop pruning ran while every move searched so far was mated | (bundle) |
| gen 6.7 (11-05) | `ENABLE_TT_FLAG_FIX` | a PVS re-search's TT flag was computed against the entry window, not the drifted window searched: bounds stored as EXACT (61k on 300 WAC positions) | +23.3 ± 13.3 pooled |
| gen 7.3 (11-12) | `SEE_PER_MOVE 3` | capture ordering and the qsearch filter used the square's exchange value, not the move's own (a cheaper attacker of ours also hits the square on 15-17% of captures) | +12.7 |
| gen 7.4 (11-15) | `SV2_VERIFY_DEEPER` | a near-miss re-search shallower than the move's first search replaced the deeper score | +38.4 |
| gen 7.5 (11-16) | `FIX_RESEARCH`, `FIX_MIN_SCOUT`, `FIX_MIN_RD1` | the two halves were asymmetric: a reduced move beating the bound was re-searched at full depth on one side only; the minimizer scouted at the maximizer's bound; a minimizer-only full window at rd 1 | **+76.4**; SF18 z +6.07 |
| gen 7.6 (11-17) | `SV2_VERIFY_PROMOTE` | a shallower verify that crossed the bound decided the move; now one full-depth search settles it | +18.1 |
| June (v1 era) | `ENABLE_TT_DEPTH_FIX` | the pre-pass searched depth−1 but stored depth in the TT | bench win |
| **C8, SHIPPED 11-19 (gen 7.7)** | `ENABLE_QSTANDPAT_SEED` | qsearch `best` started at "mated". (a) Every capture delta-pruned: the node returned a mate while not in check (`q_phantom`: 9.7% of qsearch nodes on WAC, 5.5% in game positions). (b) The stand-pat raised alpha and every capture scored below it: the node returned less than its own stand-pat (`q_short`: 0.10% WAC, 0.046% game12), and in about half of those a fake fail-low the parent cuts on (`q_short_flip`: 0.051% / 0.024%; #276, below the registered 0.1% bar). So (a) is about 100× more frequent and carries the defect. | **self-play +21.9 (+10.4 … +33.5) over 2,000 g**; SF18 +3.50 pp (z +2.26, 1,500 g per arm); Mediocre −0.50 pp (z −0.27, 300 g per arm, a null at 92%). Shipped as gen 7.7 (#278, bench 254 / 35,922,538 / EBF 3.670); the owner retired Mediocre as the third judge for SF1.1 the same day |

## 4. Found, tested, not shipped (and whether the verdict still stands)

| item | knob | result | still valid? |
|---|---|---|---|
| mate-score bundle (gen 6.6) | `FIX_TT_MATE`, `FIX_MDP`, `ENABLE_NULL_MATE_CLAMP` | below the SF18 bar | **doubtful**: measured while phantom mates (C8) were storable as real mates; re-test on top of C8. **Trigger fired 11-19 (C8 shipped in gen 7.7): due** |
| 50-move TT guard | `FIX_TT_R50` | +1.7, below the bar | yes |
| static cut not stored as a searched bound (C3) | `SV2_STATIC_CUT_STORE` | STS only | **no**: STS cannot screen search changes; needs games |
| malus only on searched moves (C5) | `SV2_MALUS_SEARCHED_ONLY` | STS only | **no**: games running tonight (#275) |
| history credit only where alpha rose (C4) | `SV2_HIST_ALLNODE_GATE` | games neutral (+6.9, +4.9) | yes, parked until the verify retune |
| threat-aware RFP | `RFP_THREAT_GATE` | games −5.8 / −30.0 (node cost) | the form failed; the idea is kept |
| keep the aspiration fail-high on a clock cut (C6) | `ASP_KEEP_FAILHIGH` | its liveness count was void (9P stderr) | re-count tonight |
| root clock exits adopted as complete (C9) | — | 0.94 per 1,000 moves | yes, below the bar |
| null move with an unproven mate (clamp alone) | `ENABLE_NULL_MATE_CLAMP` | only in bundles, v1 | **open**: never played alone on sv2 |
| light-endgame LMR bypass | `FIX_LMR_EG` | −19.8 ± 12.5 on gen 6.4, v1 | **open**: base changed since |
| quiet SEE prune | `ENABLE_SEE_PRUNE` | STS −141 | **void**: uses the same broken `see()` call as S2-1 |

## 5. New from the 11-18 sweep: verified in the code

Class D = a search decision changes; V = a wrong value whose size is read later. None of these is counted yet.

| # | defect | where | class |
|---|---|---|---|
| **S2-1** | **The check-extension SEE filter is inverted for safe checks.** `see(to, opponent, after)` returns the checker's full value when the opponent has NO attacker on the square, so an unattacked checker counts as "losing SEE > 300" and is not extended. Extended today: checkers the opponent attacks but cannot win. Never extended: checks from a distance, safe knight checks, pawn checks, castling with check. SEE also ignores that the side is in check (discovered / double checks). In place since the 06-21 ship (+43.9 vs extending every check). | `search_v2.cpp:1001`; `cpp_bitboard.h` `see_impl` incremental path (~1858-1876) | D, frequent |
| **S2-2** | **Endgame searches skip qsearch on reduced lines.** When the root's material is low (`phase_score` ≥ 96, about R+minor each or less), `use_q_precautions` is set. Every horizon whose line target `depth_limit` < 6 then returns a bare static eval: no captures, even in check. The test reads the line's own reduced target (LMR, null, verify), not the iteration depth. The null child at `depth_limit` 10 gets target 5, so its whole subtree is static. The header comment calls the gate "default-off"; it is on (shipped in the June collapse bundle). | `search_v2.cpp:456-458`; `search_engine.cpp:4344-4366`; `search_engine.h:3193-3200` | D, every endgame search |
| **S2-3** | **Futility-pruned moves' values are dropped from the fail-low bound.** A pruned move only updates `best_early_eval`, which is returned only if every move was pruned (unreachable: move 0 is never pruned). The node returns the best searched move's score as its upper bound, too low by up to the gap, and the parent stores it at full depth. SF raises `bestValue` to the futility value. Three audits found this independently. | `search_v2.cpp:818-824`, `913-914` | V → D via the TT |
| **S2-4** | **The pre-search stores bounds as EXACT.** `pre_minimizer` narrows beta as moves come in (`beta = min(beta, lowest_score)`) but flags each child's TT store against the original window. A child failing high against the narrowed beta is stored as EXACT, and EXACT entries cut at any window. Same bug as gen 6.7's flag fix, never ported to this v1 function. | `search_engine.cpp:9202-9213`, `9230-9241`, `9269` | D via the TT |
| **S2-5** | **The history decay interval sticks.** `DECAY_INTERVAL` (35,000 nodes) is raised to 125k / 150k / 200k when a search starts from an endgame. Nothing resets it, not even `ucinewgame`. fastchess reuses engine processes, so from game 2 on, openings and middlegames decay history about 6× less often than game 1. Multi-position diagnostics depend on the previous positions' material. All tuning since has lived in the stuck setting. | `search_engine.cpp:4351-4363`, `10209-10231` | D |
| **S2-6** | **A move-0 fail-high doesn't stop the root.** After move 0 fails high in an aspiration window, the root still searches move 1 before its first beta test. That wastes a full-depth search and can switch the best move. | `search_engine.cpp:5126` (no test), `5545` (first test) | D (time) |
| **S2-7** | **Stale PV tails feed move ordering.** `pv_length` is reset only once per root pass, not per node, so `updatePV` can copy a tail left by another subtree. The PV then feeds `moveFrequency` (a quadratic ordering bonus), also after cut iterations. | `search_engine.cpp:10438-10444`, `4622-4641`, `4968` / `4994` | D (ordering) |
| **S2-8** | **Per-move delta pruning skips checking captures.** Only promotions and en passant are exempt; SF also exempts checks. | `search_v2.cpp:1457` | D, low-medium |
| **S2-9** | **Lazy-eval estimates are stored as search values.** The lazy exit returns its estimate (up to 2.6 pawns off at p99). RFP returns it and the parent stores it in the TT at full depth. A q-root stand-pat returns it and `qsearch_entry` stores it in the q-cache as a lower bound. A later probe within the margin then cuts on the estimate. This contradicts LAZY-EVAL.md §1 ("never cached"); that holds only for the eval cache. | `eval_v2.cpp:5154-5164`; `search_v2.cpp:518`, `1436-1437`, `223-229` | V → D via caches |
| **S2-10** | **The pre-search scouts at the wrong bound.** `pre_minimizer` (a min node) scouts at (α, α+1) and re-searches at full depth whenever score < β, so nearly every reduced move is searched twice. This is the A2 shape fixed in gen 7.5 elsewhere. Values stay exact; it costs nodes. | `search_engine.cpp:9167`, `9191` | efficiency |

### 5b. Counts (queue #277, 11-18, count-only build `chess_engine_uci_cnt18c`, every gate equal; d10, seed off)

Registered bar: "material" = fires on ≥ 1% of its own opportunities on game12 (12 game positions) or eg60 (60 endgames).
Every item below passes it, except S2-24.

| # | measure | game12 | eg60 | WAC | prediction |
|---|---|---|---|---|---|
| S2-1 | checks the SEE filter drops (of checks the extension could take) | 90.9% | 91.0% | 89.9% | 40-80% (low) |
| S2-1 | of the drops, checkers no enemy piece attacks | 31.5% | 42.3% | 24.1% | 60-90% (high) |
| S2-2 | horizon nodes taking the static eval (in check: 99 / 3,450 / 2,765) | 0.24% | 5.45% | 0.81% | eg 20-60% (high) |
| S2-3 | no-cut exits whose futility-pruned value beats the returned bound | 20.2%, mean 379 mp | 26.4%, 434 mp | 20.8%, 478 mp | 30-80%, 300-1,500 mp |
| S2-4 | pre-search stores mislabelled EXACT | 3.4% | 4.5% | 4.6% | 0.1-3% (low) |
| S2-9 | RFP returns that are a lazy estimate (RFP returns are ~12% of WAC nodes) | 38.0% | 33.0% | 56.7% | 30-70% ✓ |
| S2-9 | q-cache stores that are a lazy stand-pat estimate | 5.3% | 4.8% | 7.3% | 5-30% ✓ |
| S2-6 | root passes whose move 0 failed high (move 1 still searched) | 5.7% | 9.2% | 24.9% | 5-20% ✓ |
| S2-5 | searches run with a raised decay interval on a non-endgame root | 5 / 12 | 3 / 58 | 254 / 300 | > 50% of WAC ✓ |
| S2-24 | searches ending on a mate-range score, seed off / on | 1 / 1 | 0 / 0 | 65 / 63 | — (no difference: phantoms rarely end ID) |

**Reading:**
- **S2-1:** only about 1 check in 10 is extended at all. About a third of all checks are safe checks that are never
  extended (the defect). The rest of the drops are checkers the opponent attacks and wins by SEE > 300, which includes
  discovered checks, where SEE ignores that the capture may be illegal. A fixed rule would extend roughly 4 to 5 times
  more checks than today, so the three-arm test (§6b) stands.
- **The seed** leaves these rates nearly unchanged, apart from S2-4: mislabels rise to 6.1-6.7%.

## 6. New from the 11-18 sweep: reported, not yet checked by me

- **S2-11** Repetition detection crosses a null move: the scan does not stop at the null, and pre-root counts still apply
  (SF stops at `pliesFromNull`). A false draw can give a false null cut. Two audits found this. `search_v2.cpp:521-590`,
  `search_engine.cpp:10253-10262`.
- **S2-12** qsearch's lazy stand-pat request has no mate-range guard; RFP's has one. `search_v2.cpp:1424` vs `:510`.
- **S2-13** The seed-off family beyond the two counted cases:
  - (a) a fail-low below the stand-pat at a null window;
  - (b) a phantom from two q-plies down arriving as a mate-sized value;
  - (c) an open-window short value stored as EXACT.

  C8 fixes all three.
- **S2-14** The null move has no guard for a losing-mate beta (RFP has one; SF uses `!is_loss(beta)`). `search_v2.cpp:523-524`.
- **S2-15** Losing checking captures are SEE-pruned at rd 1-3, although the comment says checks never are. `search_v2.cpp:1125-1134`.
- **S2-16** The q-cache stores path-dependent draws (repetition zeros) found below the q-root, then serves them on other
  paths. `search_v2.cpp:226-227`, `1338-1339`.
- **S2-17** qsearch has no 50-move test (rare). `search_v2.cpp:1338`.
- **S2-18** The PV-update rule differs by side: the minimizer writes the PV on fail-lows. This feeds S2-7. `search_v2.cpp:848`.
- **S2-19** D7's "deeper" rule uses the requested scout depth when the TT answered the scout from a deeper entry, so the
  verify still runs shallower than the evidence. `search_v2.cpp:1203-1280`.
- **S2-20** `pre_minimizer` never checks `time_up`: a partial min(…, 0) is stored with an EXACT / LOWER flag, and fake
  cutoff history is written. Mostly matters in node-limited play. `search_engine.cpp:9036-9321`, `8764`, `8888`.
- **S2-21** A partial qsearch value after time-up can come back through a null cut and be stored (at most one entry per
  search). `search_v2.cpp:582-588`.
- **S2-22** The ep square is set after every double push and enters every key (FIDE: only when an ep capture is legal).
  Repetitions are caught one ply late, and transpositions are split.
- **S2-23** History writes outside sv2 (pre-search, ply 1) are raw adds with no bound or malus. Low.
- **S2-24** Iterative deepening stops on any score ≥ 9,000,000; a phantom mate reaching an exact root score would end it
  early. Count it in seed-off arms.
- **S2-25** The null-move target table is non-monotone (`DEPTH_REDUCTION[9]` = `[10]` = 8, then −1 at depth ≥ 10), so the
  depth-10 null search is shallower than the depth-9 one.

**Latent** (off at runtime; fix before enabling):
- ProbCut's child bypasses `score_move` (no repetition test, TT probe or extension).
- A singular exclusion search with one legal move returns the mated sentinel.
- `see()` scores en passant one pawn low.
- `bench` state leaks into a later `go`.
- A `position` sent before `isready` keys repetitions with the knob unlatched.
- `Resign` plays `legal[0]`.
- `go depth 1/2` searches depth 3.

## 6b. A second reader (a Fable session, 11-18, read-only)

**The register held up:** every item it checked is real in the code. It confirmed S2-12, S2-14, S2-17, S2-18, S2-19,
S2-24 and S2-25 by reading. It did not read `pre_minimizer` (S2-4, S2-10, S2-20).

**Three structural causes under the §2 shapes:**
- **The parent writes the child's TT entry at full depth.** `score_move` cannot know whether the value came from a search,
  a static cut, a null cut or a lazy estimate (C3, S2-3, S2-9, S2-26, S2-27).
- **`depth_limit` is the line's own target,** not the remaining depth and not the iteration depth. A gate written as
  `depth_limit >= N` changes meaning inside reduced and null subtrees (S2-2, S2-25).
- **File-scope state with no owner** (S2-5, S2-7).

**Corrections to entries above:**

| # | change |
|---|---|
| S2-1 | The plain fix extends nearly every check, which is the arm that lost 43.9 in June. Test three arms: today, `CHECK_EXTENSION=0`, and fixed. If today beats off, name the accidental rule (defended contact checks, including every check only the king can take) and tune it. |
| S2-2 | Sharper scope (`NULLMOVE_EXTRA` 2 and the −1 at depth ≥ 10): the null child's target for iteration depths 7 to 12 is 4, 5, 6, 5, 6, 7. So in endgames every null subtree is static-leaf at iterations ≤ 8 and at exactly 10 (S2-25's kink). At ≥ 11 only nested reductions hit it. Iterations 3-5 have no qsearch at all. |
| S2-3 | The fix must add the margin. `best_early_eval` holds the child's bare eval; the bound the prune justifies is `early_score + fut_margin` (still below alpha). Keep it in a separate variable and fold it in only at the fail-low return, so `g_best_mated` and the exit-history test keep reading the searched best. |
| S2-7 | One store at node entry fixes it (`pv_length[cur_depth] = 0`). No free count exists in the match logs: the UCI layer replays the PV and cuts it at the first illegal move, so fastchess never sees one (its 500 PV warnings in #274 are all "PV continues after threefold repetition"). |
| S2-11 | Real but likely rare. It needs a tempo-losing triangle from one side and an out-and-back from the other, at least 4-5 plies below the null, with the side that does not want the draw cooperating. Count before fixing. |
| S2-15 | Not a defect by reference practice: SF also SEE-prunes losing checking captures. Only the comment is wrong (also: `extend` is false once the per-path cap of 3 is used). Coupled to S2-1, so don't size it separately. |
| S2-16 | Mechanism: a perpetual at the horizon. Cheapest fix: `addToQCache` skips an exact 0, as `addToSearchEvalCache` already does. Incomplete in principle: a 0 can shape a non-zero value, like the TT's rule. |
| S2-21 | One-line fix: test `time_up` after the null search. The ply-1 copy already does (`search_engine.cpp:6872`). |

**New candidates:**

| # | candidate | status |
|---|---|---|
| **S2-26** | **RFP returns the raw eval, and the TT then removes the margin.** `RFP_RETURN_BLEND` 0, so `search_v2.cpp:518` returns the eval, and the parent stores it as a full-depth lower bound. A later probe cuts for any beta up to the eval: reverse futility with margin 0. C3 treated the store as the problem; the value is. Returning `eval − margin·rd` (what RFP proved) also shrinks the RFP half of S2-9. `RFP_RETURN_BLEND` (a blend toward beta) was built in October and never played (SEARCH-AUDIT-PRUNING G8: "the cheapest arm in the report"). | verified: return value, never played |
| S2-27 | Null cuts get the same full-depth label: a value from a search several plies shallower, stored at rd (shape 5; most engines accept it). A one-bit origin tag (searched / static cut / null cut) would count how many TT cuts rest on each. | reported |
| S2-28 | `moveFrequency` enters ordering unscaled. The PV bonus is (d−1)²(L−i)² per iteration (≈ 144,000 at depth 20), added raw to capture and quiet scores (`move_gen.h:1302/1407/1573` via `mf_get`), against a history cap of 16,384. With S2-7 some of it lands on moves that were never in the PV. Our own term (no reference analogue): size it, don't remove it. | verified: added raw |
| S2-29 | Same-position TT replacement is strictly depth-preferred with no age (`cache_management.h:1218`). A deep entry whose bound no longer cuts blocks every shallower fresh result. SF also replaces when the new depth is within ~4 plies, the new flag is EXACT, or the entry is from an older search. | verified (sharpens §0b's "always replace, no aging") |
| S2-30 | The null move has no eval gate (`ENABLE_NULL_EVAL_GATE` 0), and above rd 6 the node has no static eval, so the null is tried even far below beta. Efficiency. v1-only record (PRUNING G1). | reported |
| S2-31 | The ply-1 malus list ignores `SV2_MALUS_SEARCHED_ONLY` (`search_engine.cpp:6935-6936`), so tonight's C5 arm covers plies ≥ 2 only. Small: pruning at ply 1 needs iteration depth ≤ 6. | verified |
| S2-32 | qsearch searches all four promotions and exempts them from delta pruning. | reported |

- **S2-26, what other engines return from reverse futility (read from source 11-19; % = share of the way from the eval to
  beta, our `RFP_RETURN_BLEND` scale):** the raw eval (0%): SF11, Ethereal, Weiss, Koivisto (Koivisto returns beta in a
  separate depth-1 case), and us today. The midpoint (50%): Berserk, Obsidian, PlentyChess, Caissa (51%). Further toward beta:
  SF18 67% (`(2 * beta + eval) / 3`), SF19 65% (`(661 * beta + 363 * eval) / 1024`), Reckless 69% (`lerp(eval, beta,
  0.6945)`). Stormphrax interpolates by a tuned factor (its value was not in the fetched file). So the pull toward beta is the
  modern direction, not universal, and its size is tuned per engine (50-69%). #282 tests the FORM once at 67. **Final-tuning
  candidates (owner 11-19: exact values wait for a tuning pass near the end, since the engine keeps moving the optimum):** the
  other values (50 / 65 / 69), returning what the prune proved (`eval − margin · rd`, a fixed amount off the eval rather than
  a fraction of the gap), and treating a lazy-estimate return (S2-9: 33-38% of our RFP returns) differently from a full eval.
  **S2-9 is NOT a tuning item (owner 11-19: "a more serious problem ... still part of the bug fixing"):** knob
  `LAZY_FAIL_HARD` (default 0) returns the bound a lazy estimate proved (beta at reverse futility and at qsearch's stand-pat
  cut, alpha − 1 as a futility-pruned move's bound under `SV2_FUT_BOUND`); games in #281, before S2-26's blend (#282).
  Lazy eval itself is not ours alone: SF11 has a global `LazyThreshold` (1400 + non-pawn material / 64) and returns the lazy
  value raw through RFP too; ours differs in being asked per site with the window's bound. NNUE engines evaluate fully.
- **S2-9, the form (measured 11-19 16:35-17:03 on a plain check build, every gate equal):** the first form, mode 1 (return
  beta), costs **+28.9% bench nodes** (249 / 46,312,304; registered +0.5 … +5%, a large miss). Ablated by site: reverse
  futility alone +24.3%, the stand-pat cut alone +20.6%, no change with lazy eval off (35,160,332 both), and +26.8% with the
  near-miss re-search off (`VERIFY_MARGIN=0`; its margin is 16 pawns, so it already re-searched nearly every reduced fail
  low). The cost is the lost fail-soft information itself: later probes cut on the stored size. **Mode 2** returns what the
  lazy exit proved, the estimate less the margin M it cleared its bound by (`g_lazy_margin`; never below beta): 249 /
  37,325,964 (**+3.9%**; registered +5 … +15%), reverse futility alone −0.6%, the stand-pat cut alone +2.9%; game12 d10 **9/12**
  at 2,785,346 (+15%) vs 7/12 (mode 1 also 9/12, at 3,221,293). #281 plays mode 2. **Mode 1 PARKED**, trigger: mode 2 passes
  and the question becomes whether a tighter bound helps further, or a node saver frees ~25% of the budget.
- **S2-9 mode 2, #281 (2026-11-19/20) — NEUTRAL (leaning negative), PARKED.** 2,000 games +570 −633 =797, 48.4%, **−10.9 Elo
  (95% −22.4 … +0.5)**, pentanomial [90, 248, 375, 209, 78]; registered −6 … +12 (a miss, low). **Not a node cost:** game depth
  11.7 (lfh) vs 11.6, nodes per move 360,716 vs 361,418, nps 1,590,156 vs 1,594,991. So the lean is per depth, and the
  "node-adding fix, pair with a node saver" park reason does not fit. A reading (not yet ablated): "estimate − M" is proven
  but biased low by M (1.5-3 pawns × site scale) on every lazy fail high, while the raw estimate is unproven but roughly
  unbiased; SF11 returns its lazy value raw. If so, the defect is the full-depth TT store of an unproven size, not the
  returned value. Outside judges skipped (≈ 5.1 h saved). #282 and #283 therefore run on gen 7.7. Re-test triggers: a form
  that keeps the raw return but stores a lazily cut node's bound at reduced depth (or flags it so a probe cuts only at the
  window it was proved for); or lazy eval's margins reshaped. Split by site first (reverse futility vs stand-pat) if it is
  reopened: the stand-pat half carries the +2.9% bench nodes.
- **S2-26 `RFP_RETURN_BLEND=67`, #282 (2026-11-20) — NEUTRAL, PARKED (a final-tuning candidate).** 2,000 games +577 −600
  =823, 49.4%, **−4.0 Elo (95% −15.3 … +7.3)**, pentanomial [79, 247, 368, 230, 76]; registered −5 … +20 on the gen 7.7 base
  (inside, at its low end). Game depth 11.7 / 11.7, nodes per move 372,522 vs 373,127, nps 1,645,865 vs 1,653,442; bench
  +0.34% (registered +1 … +6%, a miss low). The 11-20 04:14 Windows Update restart cut the run: the 900 games of chunks
  2821-2823 were kept, chunk 2824 (193 games overlapping the install) was set aside unread and its seed replayed whole, and
  speed was unchanged after the update (every chunk 1.69-1.72 M nps on both sides). Outside judges skipped (≈ 5.1 h saved).
  A reading (not ablated): the two forms that shrink a static cut's returned value both read below zero (S2-9 mode 2 −10.9,
  S2-26 −4.0), so our pruning margins may be tuned around the raw return (an absorbed tuning,
  [[a-correctness-fix-into-absorbed-tuning-is-not-free]]). Re-test triggers: the end-of-arc tuning pass with the other values
  listed above (50 / 65 / 69, `eval − margin · rd`, lazy and full evals apart); a return-value form paired with an
  `RFP_MARGIN` re-sweep; or S2-27 / S2-29 changing what a stored static-cut bound does in the TT.
- **S2-3, #280 (2026-11-19) — FINAL: NEUTRAL, PARKED (LOW).** 2,001 games +620 −611 =770, 50.2%, **+1.4 Elo (95% −10.7 …
  +13.5)**, pentanomial [95, 241, 325, 239, 100]; depth 11.7 / 11.7, nps 1,603,024 vs 1,606,204. Outside judges skipped (no
  self-play pass), ≈ 5.1 h saved against the old form. Mid-run read: self-play even, 50.3% after 1,580 of 2,000 games (gaming pause at 16:30). The registered
  prediction was −10 … +15, and the #277 count was below its prediction (20-26% of no-cut exits, ~0.4 pawn, against 30-80%,
  0.3-1.5 pawns). The games' own depth (1,587 games, 106k moves per arm): **11.674 vs 11.652**, time per move 0.226 s vs
  0.225 s, so the +22% game12 node cost does not appear in play. A plain null, not a masked per-depth gain: if neutral it is
  parked at LOW priority (not a front-line pairing partner), with triggers: futility pruning reshaped (its bound is eval +
  margin) or the root-razor work (PS-6, which S2-3's bounds feed).

- **Latent:** the correction-history update at a fail-low node learns from an upper bound; `moveGivesCheckFast` can report
  a false check on captures (its live callers pass quiets only).
- **For C7 (the verify retune):** since gen 7.5 no scout reduced by 3 or more is trusted on either side (a fail-high goes to
  full depth, a fail-low to depth − 2). So "cap the reduction at 2 and drop VERIFY" is the same search without the extra
  scouts: the honest baseline arm.

## 6c. The pre-search and the ply-1 entry (second Fable pass, 11-18)

Read in full: `reorder_legal_moves`, `pre_minimizer`, the merge and sorts, `root_tt_flag`, and the ply-1 `minimizer`
branch to its cutoff test. Not read: the ply-1 cutoff-history block (~7002-7078) and `minimizer`'s common exit.

**Known items, rescoped:**
- **S2-4: confirmed, but open-window calls only.** That means root move 0 of each pre-search pass, and the in-window
  re-search (~8880). In a null-window call beta cannot narrow without cutting, so those flags are right. Rare, but these
  are the PV nodes.
- **S2-10: confirmed, open-window calls only.** In a null window, (α, α+1) = (β−1, β), so the scout is right there and a
  re-search on score < β is a legitimate reduced fail-low. In open-window calls the scout-then-research pair removes LMR
  entirely.
- **S2-20: confirmed, narrower.** A returned 0 is refused by the TT's zero rule, so an unsound store needs a negative
  partial minimum above alpha. Second effect: with α ≥ 0 the first 0 makes a fake cutoff, and killer, counter, history and
  the move-cache order are written (~9272-9291).
- **S2-23: confirmed live** (~9277-9291, ~9316-9319). The continuation add is 4·rd² raw, where search v2 writes rd²
  through `hist_update`. The only commented block in `pre_minimizer` is ~9089-9129.

**New:**

| # | finding | class | status |
|---|---|---|---|
| **PS-3** | **After an aspiration fail-low, the front root move is picked by comparing upper bounds.** `descending_sort_wrapper` (~9777-9791) swaps the max `top_score` of the searched prefix to index 0. On a failed-low pass every `top_score` is a fail-soft upper bound, so the largest bound, not the best move, gets the next attempt's full-window search, seeds alpha, and takes the open-window pre-search (S2-10). S3 §7.2 covers `best_move` on such a pass, not the ordering. Count: flag a fail-low attempt, then on the next `alpha_beta` entry count front ≠ the completed move. Fix: no promotion when the failed pass's maximum did not beat its entry alpha. | D (ordering, time) | verified (mechanism) |
| PS-6 | S2-3 reaches the root razor: root `top_score`s are fail-soft bounds built from sv2 fail-low returns, which S2-3 makes too low by up to a futility margin (200-950 mp). That inflates `alpha − top_score`, and the razor's default action is break (threshold 100 mp from depth 8; ~5216, ~5252). Count: bucket `score_diff − razor_threshold` at each fire. | D, after S2-3's count | reported |
| PS-7 | S2-2 inside the pre-search: its target is D−1, so with `use_q_precautions` set every pre-search leaf is a static eval for D ≤ 6. Those scores order and razor the root at iterations 3-6. Count: split S2-2's counter pre-search vs main. | D, endgames | reported |
| PS-8 | Latent: `PRESEARCH_TAIL_MODE` 1-3 escape `TT_DEPTH_FIX` (mode 1 searches `tail_depth` but stores `depth`; modes 2-3 store `trusted_floor` with a real flag), at ~8888. | L | reported |
| PS-9 | Latent: `PLY1_NODE_PRUNE` returns a static eval that the root stores as a depth-D result and puts in `top_score` (~6807-6813, store ~5447): C3's shape at the root. Any earlier result for that knob carries it. | L | reported |
| PS-10 | Three root-level stores are safe after time-up only because a 0 is refused; `ENABLE_TT_STORE_DRAW=1` would arm them. | L | reported |

**Checked and clean:**
- **Reply lists stay paired with their root move.** The front swap stays inside the reached prefix, and appended pre-search
  entries keep their index.
- **Frames.** `negamax<true>` is in the root frame; the ply-1 `-score_move<false>(-β, -α, -β_orig, -α_orig)` is right.
- **Depth labels.** `TT_DEPTH_FIX` covers both live root-level stores.
- **Partial results never order the root:** on time-up, `reorder_legal_moves` returns the previous data.
- **The razor reads bounds, not partials.**
- **Ply-1 entry:** no RFP, null move or futility runs at shipped values. LMP and capture-SEE pruning reach ply 1 only at
  iteration depth ≤ 6 and ≤ 4.

## 6d. From the owner's UI games (11-19)

| id | finding | status |
| --- | --- | --- |
| S2-33 | **An iteration that explodes runs to the hard stop.** UI game 56 (`ui_games/ui_games.pgn`, 900+10 vs lichess AI level 7), 46.Kxg6 in a won pawn ending (FEN `8/6K1/6p1/6Pp/2k2P1P/2P5/8/8 w - - 1 46`, +24.5). | **reproduced 11-19** |

S2-33 in detail:
- **What happened:** depths 3-25 finished in 0.68 s (2.2M nodes). Depth 26 then ran 149 s, printed nothing and never
  finished. It was cut by the TM_MODE 1 hard stop, `min(5 × optimum, 0.8 × clock)`, which here was 5 × 29.9 s = 149.6 s
  (`uci_main.cpp`, the optimum / maximum block). The replay with the production binary and the game's clocks matched it to
  0.1 s.
- **Cost:** 150 s of the 224 s left. The next move, 47.f5, took 59 s for depth 22, its first mate score. A won ending
  reached 0:32 on the clock.
- **Why the 8 nodes/s:** the UI prints the last info line's nodes divided by the whole think, so its "8 nodes/s" is the
  depth-25 count over 150 s.
- **Two parts:**
  - (a) SEARCH: one iteration costs more than 200 times the one before it, at the depth where mate scores first appear. Look
    at aspiration windows and re-searches around mate-range scores, and at pruning switched off by mate-range bounds.
  - (b) TM: nothing stops an iteration that is far past the optimum. SF also has a maximum of about 5 × optimum, but its
    iterations do not jump 200 times in cost.
- **Same family:** 29.Rg5 in the same game took 145 s for one depth-18 iteration after the eval fell (+2.18 → +1.70).
  Optimum there was 39.5 s.
- **Count next:**
  - iterations whose nodes exceed 20 times the previous iteration's, per 1,000 moves, in the #275 games;
  - moves that hit the hard stop (the TMCNT line).

## 7. Records found wrong

- `search_engine.h:3193-3196` calls `ENABLE_QPREC_PHASE_GATE` "default-off, a documented negative"; it is on (S2-2).
- LAZY-EVAL.md §1 says a lazy value "is never cached"; it reaches the TT and the q-cache (S2-9).
- The S5 audit (§1.8, §1.10) still shows `FIX_RESEARCH` / `FIX_MIN_SCOUT` / `FIX_MIN_RD1` at 0; they have shipped since gen 7.5.
- Memory `search-correctness-asymmetries-2026-10` predates the gen 7.5 ship.
- KNOWLEDGE-MAP treats singular extension as resolved-neutral; that verdict was measured with live defects (MOVEGEN C1, S5 D2 / D4).
- CLAUDE.md's top baseline block (gen 6.5) is stale against gen 7.6.

## 8. Not yet audited (where to look next)

- The ply-1 node's own entry pruning: v1 `minimizer` `cur_depth == 1`, `search_engine.cpp` ~6640-7110 (RFP, null and
  futility at ply 1, and the A6 child-futility asymmetry).
- The root's carried ply-1 reply lists (`second_level_moves_list` / `SearchData`): do they still belong to their root move?
- TT replacement (direct-mapped, always-replace: the policy, not just its existence); q-cache aging (never aged).
- Move-scorer internals (`score_quiet`, `mf_get`, the threat-ordering terms) and history magnitudes and saturation.
- `lazy_request` internals and every caller's threshold.
- The time manager's rules (soft / hard limits, stability, score-drop) beyond what C6 / C9 cover.
- The Cython / UI path (one game per process: it plays the 35k decay regime of S2-5).
- Latent features before they are turned on: singular, ProbCut, IIR, improving, TT-move ordering.

## 9. What happens to each finding

1. Count it with a count-only counter on a fixed-depth bench. That is safe to run beside the owner's games.
2. If it fires, build a fix behind a knob (default 0 = byte-identical), with the result recorded in the knob's comment.
3. Register a prediction, then play games: self-play 10+0.1, SF18 at equal nodes, Stockfish 1.1 node-limited (the third
   judge since 11-19, replacing Mediocre; its node count from the #279 ladder).
4. Expect some findings to be "productive bugs". The engine was tuned around them (S2-1 and S2-5 especially), so fixing
   one may need its margins retuned with it.
5. **A node-increasing fix that reads flat or bad is parked, not rejected** (owner, 11-18). Its value can be masked until a
   node-saving change frees the budget, so park it with a trigger that names its partner. Expected node direction of each
   fix:
   - **more nodes:** S2-1 fixed (more extensions), S2-2 (qsearch at more horizons), S2-26 / S2-9 (smaller stored bounds,
     fewer TT cuts), S2-4 (fewer EXACT cuts);
   - **fewer nodes:** S2-6 (no move-1 search after a move-0 fail-high), S2-10 (LMR back at open-window pre-search nodes);
   - **unclear:** S2-3, S2-5.

## 10. Sweep 3 (2026-11-20): six read-only Fable agents, reports in `dev_notes/active/SWEEP3-*-2026-11-20.md`

Owner 11-20 (weekly-reset day): Fable agents for deep reads, design, bug finding and giants comparisons; "I don't mind spending
the time to see through all the items if they truly are things that we should be checking out". Nothing was run; every size
below is unmeasured unless marked. Claims I re-checked in the code are marked ✔.

| report | what it settles | headline |
|---|---|---|
| `SWEEP3-S2-1-CHECK-EXTENSION` | S2-1 | ✔ defect at `search_v2.cpp:1040`. Today's rule extends *contested* checks (attacked and adequately defended), incl. defended pawn checks (§5's "pawn checks never extended" is wrong). **§6b's "the plain fix extends nearly every check" is wrong:** by #277's counts the fix extends 31-47% of eligible checks (✔ arithmetic); June's −43.9 arm extended 100%, on v1. Every check is also exempt from LMR / LMP / futility, extended or not; SF11/18/19 and Reckless SEE-prune losing quiet checks (new, uncounted). Arms: knob `CHECK_EXT_FORM` 1 = SEE-correct, 2 = contact checks passing SEE, 3 = "king must move" (double, safe contact or knight checks), 4 = SF11's rule; plus `CHECK_EXTENSION=0` (no build). The agent leans to 3. ✔ The #277 count code is live in production (`search_v2.cpp:1039-1046`, an `attackersMask` per dropped check): guard it in the next build (result-identical). |
| `SWEEP3-S2-5-S2-2` | S2-5, S2-2 | ✔ Middlegame roots never reset `DECAY_INTERVAL` (`search_engine.cpp:4396-4412`), so after the first endgame-rooted search every later one runs at 125k-200k: 1-3 history halvings per move instead of ~10. That stuck regime is close to SF18/19 (age once per search); the designed 35k is not, so "restoring" it should lose. **The UI (one game per process) plays every opening at 35k**, a regime the tests almost never play. Arms `DECAY_FIX` 1 = re-decide per search (−25 … +5), 2 = reset on `ucinewgame`, 3 = middlegame roots take an explicit `DECAY_MG` 150000 (a deterministic null control). S2-2: one live site (`search_v2.cpp:459-470`); arms `QPREC_FIX` bits 1 in check / 2 null subtree / 4 iteration depth / 8 off; play 1, 3, 8; −3 … +8; judge partly on endgame starts. Separate runs, no 2×2. **Side finding ✔ (arithmetic, uncounted): `LMR_SHAPE 1` at `LMR_PRODUCT_DIV 64` is saturated.** SF11 divides the same product by 1,024; ours exceeds the `rem − 2` clamp at every remaining depth ≥ 3 and move ≥ 2, so every reduced scout starts at child remaining depth 1 before the history / improving / capture-chain adjustments. With `VERIFY_MARGIN` 16 pawns re-searching nearly every reduced fail-low, the effective reduction is the verify depth. Count: `_side_1120_fixed.sh` C. It belongs at the head of C7. |
| `SWEEP3-S2-33-EXPLODING-ITERATION` | S2-33 | H1 ✔ (mechanism): with `FIX_TT_MATE=0` no mate-range score is ever stored in the TT (`cache_management.h:1171-1177`, `:1195`), so a mate proof gets no transpositions; the game's first-mate iterations cost 21.6M / 42.9M / 172M nodes for mate in 9 / 10 / 12 (~2× per mate move). H2-H6: mate guards make mated nodes full width, re-search rules without a mate guard, the aspiration ladder (S3-1), no MDP. Ruled out: table bounds, S2-2, S2-5, the 32-bit counter. Decisive test: `_side_1120_fixed.sh` A / B. TM part: `TM_ITER_CAP=300` / `TM_ITER_FLOOR=100` (cut an iteration at 3× the scaled optimum unless the score dropped or the best move changed); 1-5 fires per 1,000 moves, −3 … +3. 29.Rg5 is a different mechanism (normal growth plus the score-drop scale). |
| `SWEEP3-TRIAGE` | every open item | 49 rows: GAMES NOW 2, COUNT FIRST 12, FIX-ONLY 11, PARK 17, **NOT REAL 6 (S2-12, S2-13, S2-15, S2-24, S2-27, A6)**. Games: the mate bundle + S2-14 (trigger fired 11-19), S2-29 TT replacement (count first), S2-4 + S2-10 (`PS_SCOUT_FIX`, −1 … −5% nodes), S2-7 + S2-18 (count first; the one-line S2-7 fix is insufficient), one hygiene bundle (S2-8, S2-11, S2-16, S2-17, S2-32). Bench identity only: S2-20, S2-21. The margin question: `RFP_MARGIN` was always set with the raw return, but the blend moved bench nodes 0.34%, so a fixed-depth 3×3 grid first and any game at the head of C7. Whole phase ≈ 105-120 h. Corrections: S2-10 overstated (a fail-soft scout ≥ beta skips the re-search), S2-11's shortest case is 3 plies, PS-3 is a heuristic, PS-6 covered by #280, A6 fires at iteration 5 only. |
| `SWEEP3-BUGHUNT-TM-PLY1` | TM, ply-1 entry, carried reply lists | No live class-D defect; ply 1 and the reply lists are sound; the TM cannot overrun the clock. New: **S3-1** a mate-range aspiration fail is widened ×2 from the old score three times before the full window, each pass re-running the pre-search (`ASPIRATION_RECENTER` excludes mates; SF11 re-centres), an S2-33 candidate; S3-2 the score-drop factor sees only the drop within this search, not since the previous move; S3-3 low clock: the hard limit floors at 10 ms and can cut the first pass (its suggested fix `TM_LOWCLOCK` is **out of date: revoked 11-06 by its replicate, +3.5**); S3-4 … S3-10 small or latent. S2-33(b) sharpened: the hard limit is 5× the unscaled optimum, up to 24× the scaled soft limit in "easy" positions (`TM_HARD_SCALE` sketch). |
| `SWEEP3-BUGHUNT-TT-LAZY-ORDERING` | TT, caches, lazy eval, move scoring | No new wrong-value defect in the TT, caches or lazy eval. New, ordering and sizing: **S3-20** quiet under-promotions score 5,000 / 10,000 / 15,000 (✔ `move_gen.h:1346`, `(promotion − 1) × 5000`), level with or above killers, and are never reduced or pruned; **S3-25** eval / q / move caches stay at 2^20 whatever `Hash` is (testable with existing knobs, no code); S3-21 the move-cache "best two quiets" re-rank swaps instead of inserting (the `FIX_KILLER_SPLICE` shape); S3-22 killers ahead of the remaining good captures after two cutoff moves; S3-24 same-piece captures score 0 in the good-capture tier; S3-23 a q-root q-cache hit rewritten with a recomputed flag; latent S3-26, S3-27. S2-28 sharpened: `moveFrequency` passes the capture tier from about iteration 17. |

**Run order for the rest of phase H (my merge, 11-20; ≈ 6-7 days of box time, so ≈ 11-27/28):**
1. Running / queued: #283 S2-6 → #284 S2-4.
2. One build (all new knobs default 0, byte-identical; the S2-1 count code guarded): `CHECK_EXT_FORM`, `DECAY_FIX` + `DECAY_MG`,
   `QPREC_FIX`, `PS_SCOUT_FIX`, `TM_ITER_CAP` / `TM_ITER_FLOOR`, the S2-14 guard, the hygiene fixes, and S3-1 if tonight's count
   supports it. Built inside a queue step while no timed work runs.
3. Games, in order: the mate bundle (+ S2-14; promoted by S2-33's H1 if `_side_1120_fixed.sh` A confirms it) → S2-1
   (`CHECK_EXTENSION=0`, forms 1 and 3) → S2-5 (`DECAY_FIX=3`, then 1) → S2-2 (`QPREC_FIX` 1 / 3 / 8) → S2-33(b) the TM cap →
   S2-29 (after its count) → S2-4 + S2-10 → S2-7 + S2-18 (after its count) → the hygiene bundle → C7 last (the verify retune,
   with the LMR saturation question and the margin grid at its head).
4. **Owner decision 11-20: hygiene fixes ship on "not harmful"** ("if there is a justifiable reason to say something is
   cleaner, then I suppose we should go with it"). Rule in `dev_notes/current/JUDGING-SEARCH-IDEAS.md` §4: the fix restores a
   statable invariant with one right form and no constant; self-play z > −1, then pooled outside z > −1, then the owner's yes;
   items that clear their own self-play are confirmed together in one bundle against production before the ship. Applied:
   - **S2-3 (#280, +1.4, z ≈ +0.2) is hygiene** (a fail-soft upper bound below the true value is unsound; SF raises
     `bestValue` to the futility value) → eligible, joins the confirmation bundle. (I had listed it as a form choice before
     the rule was written; it is not one: there is one right bound.)
   - S2-6 (#283) and S2-4 (#284) are hygiene (the root loop's own rule after move 0; EXACT only for an in-window result): read
     their z under this rule.
   - The mate bundle (+ S2-14) and the sweep-3 hygiene bundle (S2-8, S2-11, S2-16, S2-17, S2-32; S2-20 / S2-21 by bench
     identity) are hygiene.
   - S2-9 mode 2 and S2-26 stay parked: more than one defensible form / a constant.
   - Run order gains one item: the **hygiene confirmation bundle** (everything eligible, together vs gen 7.7, self-play +
     outside judges) before C7, so C7 tunes the clean base.
5. **Owner 11-20: targeted items get venue legs where they fire** ("endgame starts, smaller but longer runs to see
   behaviour"), registered per queue; standard self-play stays the not-harmful check:
   - S2-2 `QPREC_FIX`: an endgame-start leg is the primary read.
   - The mate bundle: endgame starts (moves and time to mate in won endings) and a smaller 60+0.6 run read for behaviour
     (longest think, hard-stop hits).
   - S2-33(b) the TM cap: a smaller long-TC run (60+0.6 or 180+1.8) read for the time-per-move distribution, cap fires and
     hard-stop hits.
   - S2-5 `DECAY_FIX`: a fastchess `restart=on` leg (a fresh process per game, as the UI plays).
   - Every venue's resolving power is counted before its null is read (pairs not decided by colour).
6. **End of phase H: a dedicated CCRL estimate (owner 11-20)** from a timed session against Stockfish 1.1 (≈ 2760 CCRL 40/15,
   to re-check): ~600 games at 60+0.6 plus ~150 at 180+1.8, a balanced book (not UHO), Mediocre at time odds as a second anchor;
   errors reported separately (statistics, anchor, TC / hardware). Plan in memory `later-test-vs-mediocre-for-an-absolute-rating`.
