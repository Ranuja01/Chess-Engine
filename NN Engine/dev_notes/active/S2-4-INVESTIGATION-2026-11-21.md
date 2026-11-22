# S2-4 investigation: `PS_TT_FLAG_FIX` neutral in self-play, negative against node-limited judges (2026-11-21)

A read-only audit. Read and Grep only; nothing was built or run. It covers `pre_minimizer`, every reader of its child
stores, the gen 6.7 main-search fix, and the node-limit and clock paths. **VERIFIED** means I read the code path end to end.
**HYPOTHESIS** means an inference about frequency or effect that needs a count. Line numbers are today's tree
(`search_engine.cpp` unless another file is named). The measurements come from `SEARCH-DEFECT-REGISTER-2026-11-18.md:259-271`.

## 0. Bottom line

- **The port is correct.** The flag now matches the window each child was searched with. The stored window arguments are
  dead fields. Replacement does not depend on the flag. The knob touches nothing else (§1).
- **The fix has little to correct.** A pre-search child entry sits at a ply-2 position. Its common readers either re-search
  when the stored value is below beta, or take a cut that a lower bound would also justify. A mislabel gives a wrong value
  in only two cases: after a two-step transposition deeper in the tree, or in an aspiration re-run (§2).
- **What the fix costs.** The pre-search's own reduced-depth probe accepts **EXACT only**. A false EXACT there was a *valid*
  free dismissal. The honest LOWERBOUND is ignored instead, and the S2-10 re-search rule turns each lost dismissal into a
  reduced search plus a full-depth one. The main search moved its version of this probe to bounds long ago; the pre-search
  did not (§3).
- **Nothing in the code is specific to fixed nodes.** Nodes are counted the same way in both arms, and no root rule reads
  the relabelled flag. The only mechanism that differs by regime is the clock manager's reallocation, which exists only in
  timed play (§4).
- **Most likely reading:** a small real cost plus noise on one seed per judge. The two 95% intervals overlap near −13 to
  −17 Elo (§4, §a).

## 1. Is the port correct? (Q1)

| # | claim | where | tag |
|---|---|---|---|
| 1 | The knob is read at start-up and used in exactly four ternaries; grep finds no other use | `search_engine.h:3381`; `:3333`; `:9272`, `:9276`, `:9303`, `:9307` | VERIFIED |
| 2 | Frame: `pre_minimizer` is the ply-1 min node. Its children run as `sv2::negamax<true>`, which works in the root frame (`root_lo<true>(a, b) = a`), so a child returning ≥ beta is a root-frame LOWERBOUND ("the root side gets at least this"). That is the meaning `use_tt_entry` and `tt_use<RootSide>` assume | `search_v2.cpp:172-199`, `:205-212`; `:10203-10228` | VERIFIED |
| 3 | Only beta narrows, and the one assignment is `beta = min(beta, lowest_score)`. alpha is never assigned, so alpha == alpha_orig and the UPPERBOUND test is identical in both arms. The knob changes exactly one case: score in [beta, beta_orig) becomes LOWERBOUND instead of EXACT, which is the case `g_ps_exact_mislabel` counts | `:9096-9097`, `:9344` (grep `beta =`), `:9286`, `:9317` | VERIFIED |
| 4 | Windows actually searched. The scout runs at (α, α+1) at reduced depth and is stored as UPPER or LOWER against that window, never EXACT, in both arms. The re-search and the non-LMR branch run at (α, β) with β as it stood for that move. The flag is computed before `:9344` narrows β again, so with the knob on the flag matches the searched window | `:9236-9253`; `:9266-9283`; `:9297-9314` | VERIFIED |
| 5 | Null-window calls cannot mislabel: with β_orig = α+1, β can only narrow by cutting. Only open-window calls are affected: root move 0 of each pass, and in-window re-searches | `:8829`, `:8942-8950`, `:9344-9347`; register `:300-302` | VERIFIED |
| 6 | The stored window arguments are dead. `TTEntry.alpha/beta` are only written. Their one reader, `use_tt_entry1`, has no caller (grep `use_tt_entry1?\(`), and the other mention is a commented block. Grep `->(alpha\|beta)` finds no hits | `cache_management.h:119-120`, `:1223-1224`, `:1252-1253`; `:10231-10282`; `:9158-9198` | VERIFIED |
| 7 | Replacement depends on depth alone: the slot is taken on a new key or when `entry.depth <= depth_used`. The flag never decides it, and the move-field rule does not read it | `cache_management.h:1207-1255` | VERIFIED |
| 8 | The other readers of the flag are off or not reached. Singular reads `tt_flag` but `ENABLE_SINGULAR` is false. The node-exit TT is off. The v1 interior's EXACT-only null probes are not reached under `SEARCH_ARM 1`. The ply-1 null probe sits behind `PLY1_NODE_PRUNE=0` | `search_v2.cpp:724-774`; `search_engine.h:2926`, `:3095`, `:3211`; `:6916`, `:7292`, `:8027` | VERIFIED |

**Verdict: no defect in the port.** Passing `alpha_orig, beta_orig` to the store is harmless, because those fields have no
reader.

## 2. Who can read a pre-search child entry

Depth arithmetic: the pre-search runs at P = D−1 (`ROOT_PRESEARCH_REDUCTION 1`, `search_engine.h:1404`; `:8553-8555`) and
stores its children at P−1 = **D−2** (`:9287`, `:9318`). The positions are ply 2 from the root, with the root side to move.
The TT is cleared per bench position (`:4289`) and kept across one side's moves in a game (`uci_main.cpp:246`, cleared on
`ucinewgame` at `:528`). In the table, rb is the ply-1 node's current β in the root frame.

| reader | depth it needs | reads a D−2 entry? | bug (EXACT) vs fix (LOWERBOUND) | tag |
|---|---|---|---|---|
| main ply-1 full probe (`search_v2.cpp:1060-1071`, reached through `minimizer` `:6998-7000`) | D−1 | no | — | VERIFIED |
| main ply-1 reduced probe (`search_v2.cpp:1258-1264`), late quiet replies only (`:1091-1092`) | reduced−1 ≤ D−2 | yes | s ≥ rb: both arms cut with the same score. s < rb: the bug takes s, and the minimizer's rule (`FIX_RESEARCH` 1, `:2820`) forces a full-depth re-search (`search_v2.cpp:1314-1319`, `:1342-1344`); the fix runs the reduced scout first and may accept its fail-high without the full search | VERIFIED |
| pre-search full probe (`:9149-9157`) | P−1 = D−2 at the same depth; the next iteration needs D−1 | only in an aspiration re-run, which re-enters `reorder_legal_moves` at the same depth (`:4529`, `:5030`) | the bug uses s at any window; the fix only when s ≥ β | VERIFIED; HYPOTHESIS that this is rare: it needs child 0's value to change between attempts |
| **pre-search reduced probe (`:9217-9233`), EXACT only** | reduced′−1 | yes (the previous iteration's entries) | bug: a hit. s ≥ β dismisses the reply with no search, which is valid because s is a lower bound; s < β leads to a full re-search. fix: no hit. The reduced scout runs at (α, α+1), then a full re-search follows whenever score < β (S2-10, `:9260`) | VERIFIED |
| full probes at ply ≥ 3 after a **two-step transposition** into a ply-2 position (Ra1-a2-a3 ≡ Ra1-a3, Ke1-d1-e2 ≡ Ke1-e2) | ≤ D−3 | yes | the bug takes s as exact at any window: **the one real wrong-value use**. The fix uses it only when s ≥ β | VERIFIED (code path); HYPOTHESIS (frequency: slider and king manoeuvres, so mostly quiet and endgame positions) |

**The pre-search's value for a root move the previous main pass scored is discarded.** `reorder_legal_moves` returns the
previous main data, and `descending_sort_wrapper` keeps the main entries for the first `count` moves. Only tail moves and
iteration 1 keep pre-search values (`:9001-9006`, `:9846-9881`; `ENABLE_ROOT_TABLE` false, `search_engine.h:1444`).
Root move 0's open-window pre-search call, the largest source of mislabels, therefore reaches the rest of the search only
through the TT and the histories it warms. Any accuracy the fix adds to that call's own value has no reader. VERIFIED.

## 3. Comparison with `ENABLE_TT_FLAG_FIX` (Q2)

- **The form is the same.** The flag is set against the drifted window, and the store passes the entry window into the
  dead fields: `search_v2.cpp:1364-1368` (and v1's `:6137-6143`, `:6603-6609`). VERIFIED.
- **Difference 1: where the mislabels live and who reads them.** Gen 6.7's mislabels were at every ply: 61k on 300 WAC
  positions (register `:71`). Any-window full probes read them across the whole tree, transpositions included, so the
  wrong-value case was common. S2-4's mislabels are at ply 2 only. Their wrong-value use needs a two-step transposition or
  an aspiration re-run (§2). VERIFIED (code); HYPOTHESIS (relative frequency).
- **Difference 2: the form of the probe.** The main search's reduced probe was converted from EXACT-only to a bound-aware
  `use_tt_entry` at the scout window. The old EXACT block is still visible, commented out, at `:6029-6034`; search v2 does
  the same at `search_v2.cpp:1262-1263`. `pre_minimizer` kept the EXACT-only form, with the bound-aware call commented out
  at **`:9231`**. In the main search an honest flag keeps every cut that a bound justifies. In the pre-search the honest flag
  ends the cut. VERIFIED.
- **Difference 3: S2-10 amplifies the cost.** In an open-window `pre_minimizer` call, a scout at α comes back below β almost
  every time, so each lost free dismissal becomes a reduced search plus a full-depth one (`:9236`, `:9260`; register
  `:303-305`). VERIFIED.
- **Difference 4: one side only.** S2-4 covers only the minimizer at ply 1; gen 6.7 covered both sides at every ply.
  VERIFIED.

## 4. Fixed nodes vs fixed time (Q3)

- **`go nodes`:** sets `NODE_LIMIT`. When no clock is sent, it also sets `TIME_LIMIT`/`MOVE_TIMES` to 1e9, and
  `TM_SOFT_LIMIT` stays −1, so `may_start_iteration` always passes and the engine searches until the limit
  (`uci_main.cpp:191-194`, `:238-241`; `:4458-4466`). The cut iteration is discarded (`TM_PARTIAL` 0,
  `ASP_KEEP_FAILHIGH` 0: `search_engine.h:3586`, `:3364`; `:4592-4626`), so the move played comes from the deepest
  completed iteration. VERIFIED. The queue sends `nodes=250000` with `tc=inf` (`selfplay/_queue_284.sh:162-163`). That
  fastchess then sends no `wtime` is a HYPOTHESIS; the engine log of a leg would confirm it.
- **Clock:** the search stops at the soft limit after a completed iteration. `TM_SCALE` scales the limit by the best move's
  node share, root stability and score drop (`:4459-4463`, `:4638-4655`; `uci_main.cpp:201-227`). VERIFIED.
- **Node accounting is identical in both arms.** A TT hit counts as one node (`:10207`, `:10216`, `:10225`, and `:9229`).
  The limit is checked in sv2 `negamax`, `minimizer` and qsearch (`search_v2.cpp:434-437`, `:6663-6667`,
  `search_v2.cpp:1389-1393`). It is not checked in `pre_minimizer` itself: its children check, and the root checks between
  moves (`:8836`, `:8961`). No count depends on the flag. VERIFIED.
- **No root rule reads anything the knob relabels.** `pre_minimizer` returns values, not flags. Root-level stores use
  `root_tt_flag`, which is honest (`:603-612`, `:8833`, `:8957`; `HONEST_ROOT_TT` true, `search_engine.h:3524`). A partial
  pre-search is never used (`:8836-8838`, `:8961-8964`). VERIFIED.
- **The step shape cannot carry it.** At a fixed budget, +1.8% nodes (bench) costs a completed iteration on about
  ln 1.018 / ln 3.67 ≈ 1.4% of moves; game12's −0.2% costs about 0%. The timed games read +0.3% nodes per move at equal
  depth. The arithmetic is VERIFIED; "game nodes behave like game12" is a HYPOTHESIS.
- **The budgets differ a little.** Timed play spent 352k nodes per move against 250k fixed (register `:263`), so the
  node-limited engine is about 0.26 ply shallower. That is not a mechanism.
- **The only mechanism that differs by regime:** `TM_SCALE` gives more clock to unstable roots in timed play and nothing
  does that at fixed nodes, so a loss of root stability would be partly paid back on the clock only. HYPOTHESIS. Check 3
  reads root stability directly. Time per move cannot test this, because the clock fixes the average.
- **The numbers.** Pooled 59.1% → 51.8% is about **−51 Elo** on the logistic, 95% ≈ −89 … −13. Self-play was −16.7 … +6.6.
  The intervals overlap at about −13 … −17; the two Elo scales are not identical, so this is indicative only. Each judge
  had one seed. The ctl and ptt legs share their openings (same `-srand`, `_queue_284.sh:134`, `:163`), while `cmpr`
  treats them as independent (`:241-243`). Since a changed tree diverges games early, they are close to independent anyway.
  HYPOTHESIS.

## 5. A productive bug? (Q4)

**For:** a false EXACT is a valid lower bound. At every common reader it either takes a cut that a lower bound would also
justify, or it triggers a real full-depth search (§2). So it rarely produces a wrong value, but it adds two things:

- (i) free dismissals at the pre-search's EXACT-only probe (`:9223-9230`).
- (ii) a **full-depth search of quiet replies** whose shallower bound has fallen below the current best, at the ply-1 node
  of root move 0 and of in-window root moves. Today these skip LMR (`search_v2.cpp:1262` → `:1314-1319`); the fix returns
  them to the normal reduced scout. In effect it is a "don't reduce" marker on the PV's reply node.

The mechanism is VERIFIED; its value is a HYPOTHESIS. Every tuning since the pre-search existed was done with both in
place. HYPOTHESIS.

**Against:** the footprint is small. game12 reads −0.2% nodes, the timed games +0.3% nodes per move at equal depth, and the
root's choice reads no flag (§4). A tuned-around mechanism worth −50 Elo would leave a larger structural trace. The data
is VERIFIED; the inference is a HYPOTHESIS.

## Not read

- The ply-1 cutoff-history block (`:7041-7150`) beyond its returns.
- The root razor's later branches (`:5385-5460`).
- `reduced_search_depth` in full (`:10425-`). "Reductions of 2 or more plies at ply 1 leave the pre-search entry in place"
  depends on it and is a HYPOTHESIS.
- The fastchess source.

## (a) Explanations, ranked by likelihood (all HYPOTHESIS)

1. **A small real cost plus noise (~55%).** The fix forfeits valid dismissals at the EXACT-only pre-search probe, and S2-10
   multiplies the cost (§3). Its accuracy gain is confined to two-step transpositions and aspiration re-runs, and it buys
   nothing for root move 0's discarded pre-search value (§2). Expected size: −1 … −6 Elo in both regimes. The −7.3 pp
   comes from one seed per judge, and the two reads are compatible near −13 Elo. The queue's own seeds 284112-284113 /
   284212-284213 are the cheap replication.
2. **The productive bug (ii) at the PV's reply node (~25%).** The bug forced full-depth searches of near-best quiet
   replies at ply 1 of root move 0. The fix reduces them, which is a real accuracy loss at the most important node.
   Nothing about it is specific to one regime; outside opponents may simply punish it more.
3. **Root decisions through tail values and the razor (~10%).** Dismissals via stale bounds versus full verification
   change the pre-search values of competitive *tail* moves. Those values are the root razor's reference, and its default
   action abandons every later root move (`:5381-5386`).
4. **Clock compensation only in timed play (~7%).** `TM_SCALE` hides a loss of root stability on the clock (§4).
5. **A port defect (~3%).** None found; Check 1 tests for one.

## (b) Code defects and corrected forms

- **The port: no defect** (§1). If wanted, the store at `:9287` / `:9318` could pass `alpha, beta` for consistency; that
  changes no behaviour, because the fields are dead.
- **The missing partner (the defect class behind the cost):** `pre_minimizer`'s reduced probe is still EXACT-only.
  `SWEEP3-TRIAGE-2026-11-20.md:210` keeps it unchanged for S2-10, but with S2-4 on that probe ignores every honest bound.
  A bound-aware form, mirroring the main search's `:6034`, behind a new default-0 knob:

```cpp
// pre_minimizer, inside `if (do_lmr)`, replacing :9217-9233. A bound decides the reduced scout exactly as the scout would.
TTEntry *entry = accessSearchEvalCache(zobrist, updated_state.castling_rights, updated_state.ep_square);
if (entry != nullptr)
{
    tt_probes++;
    if (entry->depth >= (reduced_depth - cur_depth))
    {
        if (Config::PS_TT_PROBE_FIX)   // the scout's window: (sa, sa + 1) once PS_SCOUT_FIX exists, (alpha, alpha + 1) today
            use_tt_entry(*entry, score, using_tt, alpha, alpha + 1, num_iterations, false, false,
                         updated_state.halfmove_clock, static_cast<int>(state_history.size()));
        else if (entry->flag == TTFlag::EXACT)
        {
            score = tt_score_from_entry(entry->score, static_cast<int>(state_history.size()));
            using_tt = true;
            increment_node_count_with_decay(num_iterations);
        }
    }
}
```

- **How it behaves.** A LOWERBOUND with s ≥ β is now dismissed free, as the false EXACT used to be. A bound between the
  scout window and β still re-searches at `:9260`. An UPPERBOUND re-searches.
- **Gate.** `PS_TT_PROBE_FIX=1` alone must change the bench.
- **Prediction (HYPOTHESIS).** With both knobs on, bench nodes fall below `PS_TT_FLAG_FIX=1`'s 36,566,334, and probably
  below 35,922,538, because the probe also starts using the honest scout bounds stored at `:9241-9253`, which it ignores
  today.
- **Next run.** This pair, or the planned S2-10 pair, both against gen 7.7.

## (c) Three cheap fixed-depth checks (registered before any run)

**The runs.** Use `build_uci/chess_engine_uci_s6pgo` with stdin
`setoption name Knobs value BENCH_PER_POS=1[,PS_TT_FLAG_FIX=1]`, `setoption name Debug value true`, then one bench,
then `quit`, and capture `2>&1`. Run one process per (arm, set):

| set | command | arms |
|---|---|---|
| A | `bench` (WAC, d10) | 0 / 1 |
| B | `bench 6` | 0 / 1 |
| C | `bench /mnt/e/chess_data/sv2_speed/game12.txt 10` | 0 / 1 |

That is six short single-threaded runs; the checks share them.

- `BENCH_PER_POS` and `Debug` only print (`uci_main.cpp:386`, `:412-413`, `:39`).
- `[sweep2] ps_exact_mislabel` is per search (reset at `:4368`).
- `[node_split] presearch=`, `[root_r1]`, `[tail_mode] razor_fires` and `[razor_audit]` are cumulative over the run
  (`:870`, `:889`, `:900-912`): read the last print.
- There is no fixed-node bench, because `bench` forces `NODE_LIMIT=0` (`uci_main.cpp:380`).

**Check 1: the knob acts only through the mislabelled stores (Q1, tested empirically).**

Prediction:
- The totals reproduce the #284 gates: A0 254 / 35,922,538; A1 252 / 36,566,334; C0 / C1 7/12 2,412,001 / 2,408,351.
- Every position whose arm-0 `ps_exact_mislabel` is 0 has the **same** node count in arm 1. Allow at most 1% of such
  positions, for move-gen-cache carry-over, since `clearSearchTables` keeps that cache (`:4259-4291`).
- At d10, at most 10% of positions have zero mislabels; at d6 (set B), at least 25%. With fewer than 20 such positions the
  check counts as vacuous.

Falsified if a zero-mislabel position differs beyond that allowance. That would mean a side path I missed: stop and
re-audit before any games.

**Check 2: where the node cost lives (explanation 1 vs 2 vs the transposition correction).**

Measure Δpresearch from the final `presearch=`, and Δmain = Δ(bench total − presearch).

Prediction:
- A (WAC): Δpresearch +2.5 … +6% of arm 0's pre-search nodes, carrying at least 60% of the +643,796; Δmain −1.0 … +0.8%.
- C (game12): Δpresearch > 0 and Δmain < 0, since the total falls by 3,650.

Reading:
- Δpresearch ≤ 0 on either set: explanation 1's cost is not the pre-search.
- Δmain ≤ −0.5% on both sets: the bug was buying full-depth ply-1 searches (explanation 2 is active).
- Δmain ≥ +0.5% on both sets: the fix is doing real correction work in the main tree (transpositions), which argues for
  pairing it rather than parking it.

**Check 3: does the knob move root decisions (explanations 3 and 4)?**

Read the arm-A final lines.

Prediction:
- `best_changed / iters` differs between arms by at most 1.0 pp.
- `razor_fires` differs by at most 3%.
- `winner_was_razored pct` differs by at most 1.0 pp.

Reading:
- All three inside: no root-decision channel, which supports explanation 1.
- `best_changed` up by at least 1.5 pp, together with a razor shift: explanations 3 and 4 gain weight. The next step is
  then the node-limited replication on the remaining seeds, before any pairing run.

## (d) Results of the three checks (run 2026-11-21 14:33-14:42, `selfplay/_side_1121.sh`; raw outputs `/mnt/e/chess_data/side_1121/S24_*.out`)

Per-position node counts come from the Debug `[search] nodes=` lines (one per position; their sum equals the bench total).
`BENCH_PER_POS` is read with `getenv`, not as a knob, so passing it through `Knobs` only echoed it; the Debug lines made it
unnecessary. Parser: the session scratchpad's `s24check.py`.

| | A WAC d10 | B WAC d6 | C game12 d10 |
|---|---|---|---|
| totals, arm 0 → arm 1 | 254 / 35,922,538 → 252 / 36,566,334 | 185 / 2,746,561 → 186 / 2,764,984 | 7/12 2,412,001 → 7/12 2,408,351 |
| relabelled stores (arm-0 `ps_exact_mislabel`) | 5,118 | 3,561 | 162 |
| positions with none | 24 (8.0%) | 38 (12.7%) | 1 (vacuous) |
| of those, nodes differ in arm 1 | **10 (41.7%)** | 0 | 0 |
| Δ pre-search nodes | +297,862 (+2.34%) | +7,654 (+0.55%) | +40,780 (+8.27%) |
| Δ main-search nodes | +345,934 (+1.49%) | +10,769 (+0.80%) | −44,430 (−2.32%) |
| pre-search share of Δ total | 46% | 42% | (total −3,650) |
| `best_changed / iters` | 27.84 → 27.89% (+0.05 pp) | 38.66 → 38.90% (+0.24 pp) | 25.61 → 30.49% (+4.88 pp) |
| `razor_fires` | 3,081 → 3,066 (−0.5%) | 1,492 → 1,495 (+0.2%) | 110 → 110 |
| `winner_was_razored` | 16.24 → 16.47% (+0.23 pp) | 20.55 → 20.66% (+0.11 pp) | 16.0 → 18.0% (+2.0 pp) |

**Check 1: FAILED as registered on A, passed on B.**
- The totals reproduce the #284 gates.
- At d10 the zero-mislabel share is 8% (≤ 10%: hit). At d6 it is 12.7% (predicted ≥ 25%: miss, though not vacuous).
- On A, 10 of the 24 zero-mislabel positions differ, against a 1% allowance. Their node deltas (WAC index, 0-based):
  - Within 5 nodes: 36 (+2), 86 (−1), 125 (−5), 235 (−1), 252 (+2), 254 (+4), 273 (−3).
  - Larger: 269 (+548), 277 (−922), 177 (−7,778, of which −7,894 is pre-search).
- On B, all 38 are identical.
- **The counter is complete (VERIFIED).** `pre_minimizer` narrows only beta (`beta = std::min(beta, lowest_score)`), so
  alpha == alpha_orig at every store. The UPPERBOUND test is therefore the same in both arms. The only relabel is "EXACT
  with score ≥ the narrowed beta" → LOWERBOUND, which is exactly what `ps_exact_mislabel` counts. `PS_TT_FLAG_FIX` has no
  other site (grep).
- So a position whose own search relabels nothing can differ only through state carried from earlier positions.
  `clearSearchTables` empties the TT and the q-cache. It keeps the eval cache and the move-gen cache, which the earlier
  positions filled differently in the two arms.
- The shape fits that carry-over: mostly 1-5 nodes, more of it at d10 than d6 (bigger trees, more cache churn), none at d6.
  This is HYPOTHESIS until the isolation run settles it. `selfplay/_iso_s24.sh` runs each of the ten, plus 24 and 27 (which
  matched), alone in a fresh process per arm. REGISTERED: every pair is equal. Result in §e.

**Check 2: neither registered reading triggered; the cost splits by corpus.**
- Δpre-search > 0 on every set (explanation 1's pre-search cost is real). It is below the predicted +2.5 … +6% on WAC
  (+2.34%) and carries 46% of the Δ there, not ≥ 60%.
- The main tree grows on WAC (+1.49%, outside the predicted −1.0 … +0.8%) and at d6 (+0.80%). It shrinks on game12 (−2.32%,
  the predicted sign).
- So "Δmain ≥ +0.5% on both sets" holds on A and B but not C, and "Δmain ≤ −0.5% on both" holds on C only.
- On the tactical suite the fix adds main-tree work as well as pre-search work. On the game positions the extra pre-search
  work (+8.3%) buys a smaller main tree.

**Check 3: inside on WAC (no root-decision channel there); not on game12 (information, 12 positions).**
- WAC: best_changed +0.05 pp, razor_fires −0.5%, winner_was_razored +0.23 pp, all inside the registered bounds. This
  supports explanation 1 on the tactical suite.
- game12 (not registered for check 3; n = 12 positions): best_changed +4.88 pp and winner_was_razored +2.0 pp. That is the
  direction of explanations 3 and 4, on the corpus closest to what the judges play.
- Too few positions to read alone. The registered next step for that direction stands: a node-limited replication on the
  remaining seeds before any pairing run.

**Where this leaves S2-4: still PARKED.** The port is correct by code and by counter. The node cost is real and splits by
corpus. On game positions the knob moves root decisions while saving main-tree nodes, which is the one place a
judge-specific effect could live. Its partner `PS_TT_PROBE_FIX` is built (s7): `PS_TT_FLAG_FIX=1,PS_TT_PROBE_FIX=1` reads
WAC 259 / 37,139,599 · game12 8/12 2,286,422. The pairing run waits behind §e and its turn in the run order.

## (e) Check 1 follow-up: the isolation run (2026-11-21 20:52-20:53, `selfplay/_iso_s24.sh`; outputs `/mnt/e/chess_data/side_1121/iso/`)

**REGISTERED "every pair equal": HIT, 12 of 12.** Each WAC position ran alone in a fresh s6pgo process per arm at d10. That
covers the ten that differed in-bench plus 24 and 27, which matched. With fresh caches, arm 0 and arm 1 search the
identical node count in every position. Examples (0-based index, nodes in both arms):

| position | 36 | 86 | 125 | 177 | 235 | 252 | 254 | 269 | 273 | 277 | 24 | 27 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| nodes | 67,794 | 65,806 | 43,165 | 254,836 | 103,200 | 10,566 | 35,645 | 48,350 | 19,941 | 74,183 | 9,201 | 28,522 |

**Reading:**
- The in-bench differences were carry-over through the eval and move-gen caches that the bench keeps between positions.
- **Check 1 passes on its intent:** `PS_TT_FLAG_FIX` acts only through the stores it relabels. No side path exists; the
  counter is complete (§d).
- The instrument lesson is now in `dev_notes/current/INSTRUMENT-MAP.md`: a bench position's node count depends on the
  positions before it.
- The S2-4 pairing run with `PS_TT_PROBE_FIX` is no longer blocked by check 1. It waits for its turn in the run order.
