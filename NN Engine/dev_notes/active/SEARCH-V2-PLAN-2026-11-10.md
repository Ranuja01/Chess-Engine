# Search v2: the plan (2026-11-10, for owner review)

The transition closed on 11-10: lazy eval shipped as gen 7.1, the margins re-sweep parked corrhist, and watch item W1
closed as level. This is the next arc. It turns the search into the structure strong engines share, built **beside** the
current one behind a switch, so every part of ours stays available and games decide what to keep.

## 0. Decisions already made (owner)

| date | decision | source |
|---|---|---|
| 11-05 | Build the traditional designs beside ours behind clean switches, never recode the old ones; games decide. Precedent: `EVAL_ARM`. | memory `search-v2-traditional-beside-ours-with-switches` |
| 11-05 | First instruments before the design is fixed: the `make_move` line profile with and without prefetch, a cache-miss map, `-fstack-usage`. | `owner-cpp-ideas.md` §13 |
| 11-07 | Lazy eval is infrastructure for search v2: re-measure it at each step; the window-aware request (site, bound, direction) is the contract to keep. | memory `lazy-eval-is-infrastructure-for-search-v2` |
| 11-10 | Corrhist waits for search v2's node-entry static eval and its consumers. | `EVAL-V2-PARKED-REGISTER.md`, search-phase row |
| 11-07 | Eval items wait for search v2: after it raises play depth by about 2 plies, a focused eval arc retunes at the new depth. | memory `retune-depth-follows-play-depth` |

## 1. Principles

- **Strength while distinctive.** Standard structure, our own ideas kept where they win: the move cache with
  promote-to-front, the root pre-search, protect-killers, the eval's own lazy exit.
- **Two levels.** `SEARCH_ARM=0` is today's search, frozen and byte-identical (fingerprint 247 / 39,852,210 must not move).
  `SEARCH_ARM=1` is the new search file. Inside it, each part with a traditional alternative is a knob, and the default
  is whichever wins in games.
- **A switch must not slow the path it guards**: select once per search (a function pointer or template at the entry),
  never a branch per node.
- **Stacks first, then 2×2s.** Search parts are co-designed and antagonistic (memory
  `search-changes-are-antagonistic-not-additive`): root LMR needs the pre-search ordering, singular needs the node's own
  TT store. Test coherent stacks, then take parts out of the winner.
- **Equal time, then games.** Fixed-depth benches made mirages of nearly every search change. Speed is judged on time to
  depth and real-play nps, strength only in games (ship rule: self-play ≥ +2σ, SF18 at equal nodes > −1σ, Mediocre not
  negative; the skeleton port's bar is "not harmful").
- **Shapes transfer from the references, magnitudes do not.** Reckless's pawn is 109 against our 1000; every margin is
  re-tuned on our scale.

## 2. Architecture

| part | v1 today (`SEARCH_ARM=0`) | v2 (`SEARCH_ARM=1`) |
|---|---|---|
| form | `maximizer()` / `minimizer()` / `pre_minimizer()`, an absolute Black-positive score | negamax, one `search<NodeType>()` with Root / PV / NonPV templates and `qsearch<NodeType>()`; the eval stays Black-positive and is converted to side-to-move once, at the call |
| per-ply data | parallel globals (`g_searchStack`, `g_excluded_move`, `g_evalStack`, `g_captureChain`, `killerMoves[ply]`) | one `Stack` struct per ply (F18): static eval, move, excluded move, killers, continuation-history pointer, cut-off count, in-check, tt-PV |
| per-position facts | recomputed (`is_check` several times a node, checkers in movegen) | computed once into the state (F16): checkers, king blockers, pinners, check squares |
| move type | 3-byte struct | 16-bit packed move inside v2, converted at the boundary (F17) |
| shared, reused as is | board, `update_state`, movegen, eval v1 / v2, TT, Zobrist, repetition, the book, the time manager | same code; v2 may get its own TT entry layout later (F8, 16 B) behind its own knob |

New code lives in `search_v2.cpp` / `search_v2.h`. The entry `get_engine_move()` dispatches once on `SEARCH_ARM`. The UCI
binary, Cython path and `bench` see no difference except the switch. Knobs stay in `namespace Config`, prefixed `SV2_`.

## 3. Build order, each stage with its gate

| stage | content | gate |
|---|---|---|
| **S0 instruments** (now) | per-iteration EBF and time to depth (`_ebf_per_iter.py`, any UCI engine); `make_move` line profile with `PREFETCH_TABLES` 15 vs 0; cache-miss map; `-fstack-usage` | baselines recorded here (§6), for gen 7.1 and SF11 |
| **S1 skeleton port** | negamax + templates + `Stack`, with **our** current features re-expressed one for one: PVS, aspiration, the pre-search, our move source (cached list with promote-to-front), our null move / LMR / futility / razoring / RFP / IIR / LMP / qsearch / lazy-eval sites, the same knob values | correctness: perft unchanged, mate and draw suites pass, symmetry of the score sign (mirrored bench); moves at fixed depth agree with `SEARCH_ARM=0` on most positions (registered before the run); **not harmful** at equal time, 2,000 g at 10+0.1 |
| **S2 speed of the structure** | the per-position facts (F16), 16-bit moves (F17), PV-only work removed from non-PV nodes, F21 (look-ahead prefetch) and F22 (hint per table) where S0's miss map points | real-play nps and time to depth vs S1; result-identical where claimed (bench byte-identical to S1) |
| **S3 move source and root** | the traditional move source beside ours: verified 16-bit TT move first, then a staged picker (good captures by SEE, killers, counter, quiets by history, bad captures) with Reckless's static quiet terms as an option; the root by previous-iteration scores beside the pre-search | a 2×2 (cache vs staged × pre-search vs previous score) in games at equal time; time to depth |
| **S4 node-entry static eval** | the static eval stored in the stack at every node (TT-stored when the TT has it); `improving`; the standard consumers read it: RFP, futility, null-move gate (eval ≥ beta), LMR adjustment, ProbCut; **corrhist on as an option** (the parked trigger); lazy eval's request moved to this one site where it pays | games per stack; W1 re-run (SF18 at equal nodes, lazy on vs off) |
| **S5 the standard pruning and extension set** | shapes from the references, our margins: an LMR table with history adjustment, LMP by improving, history pruning, SEE pruning of quiets and captures, singular extensions with multi-cut and negative extensions, the Reckless signals worth a test (|correction| as an uncertainty signal, fail-high lerp, line-based reduction signals) | each stack in games; 2×2s on the winners' parts |
| **S6 tune and ship** | SPSA on the margins of the winning stack (replicated, memory `spsa-tuning-needs-replication-not-convergence-stats`); time manager re-check; PGO build | the ship rule; then the focused eval arc at the new depth |

S1 is the expensive one. Its value is that every later stage is a knob on a modern structure rather than surgery on the
min/max split. S1 alone is expected to be about neutral; it is judged "not harmful" for that reason.

## 4. What each current feature becomes

| current feature | in v2 |
|---|---|
| move cache + `promoteMoveToFront` (91.9% first-move cut-offs) | ported in S1 as move source A (a snapshot copy, never a pointer); the staged picker + TT move is B (S3) |
| root pre-search (root razoring, root LMR substrate) | ported in S1; previous-score root ordering beside it (S3) |
| PVS, aspiration | ported; aspiration delta shape revisited in S5 |
| null move, IIR, RFP (600), futility, razoring, LMP, LMR product schedule | ported with today's values (S1), re-shaped in S4-S5 on the node-entry eval |
| protect-killers, threat ordering (`THREAT_ORDER_W`), killers / counter / history / move frequency | ported; history keying (from-to attacked, continuation at 1/2/4/6 plies) in S5 |
| SEE capture ordering, qsearch with checks | ported; qsearch SEE pruning in S5 |
| lazy eval (stand-pat, RFP, futility of a child) | ported with its sites (S1); moved to the node-entry eval in S4; W1 re-run at S4 and S5 |
| corrhist (`CORR2`, parked) | option in S4 |
| TT (depth-preferred, flag fix) | reused; F8's 16 B entry with a 16-bit move later, behind its own knob |
| the seven search correctness fixes (gen 6.5) and the TT flag fix (gen 6.7) | carried into the port by construction; each has a test case to re-run |

## 5. Risks

- **The port silently drops a productive bug.** Our record has several (memory
  `search-value-bugs-and-the-productive-bug-pattern`; FIX_LMR_EG lost −19.8). S1 compares moves at fixed depth with
  `SEARCH_ARM=0` and breaks down the disagreements by feature before games.
- **Equal-depth mirages.** Every stage is judged on time and games, never on nodes at fixed depth.
- **Antagonism.** Parts that win alone can lose together; stacks first.
- **Box time.** About 2,000 games (≈ 5.3 h) per read at 10+0.1. S3's 2×2 alone is 4 arms. Node-limited SF18 legs run in
  the owner's evening window.

## 6. Instrument baselines (S0)

### 6.1 Branching factor and time to depth (11-10, `_ebf_per_iter.py`)

120 game positions (`game_regret_set.csv`, 40 per phase, seed 1), one `ucinewgame` each, production gen 7.1 to depth 13
and SF11 (single thread, 16 MB hash) to depth 18. The figure is the two-ply ratio sqrt(nodes(d) / nodes(d−2)), since both
engines alternate odd and even iterations strongly. Output `E:\chess_data\ebf_2026-11-10\`.

| | gen 7.1 | SF11 |
|---|---|---|
| EBF, depths 8-13 | **1.92** (opening 1.91, middlegame 1.89, endgame 1.95) | **1.78** |
| EBF, SF11's last six depths (13-18) | — | 1.46 (endgame 1.37), still falling with depth |
| nodes to depth 13 | 1,890,577 | 142,085 (**13.3×** fewer) |
| time to depth 13 | 1,380 ms | 58 ms (**24×** faster) |
| nps at depth 13 | 1.37 M | 2.45 M (1.8×) |
| depth reached in our depth-13 time | 13 | about 18 (451 ms at 18) |

Node gap to SF11 by depth: 3.4× at depth 3, 8.4× at 6, 7.3× at 9, 10.6× at 12, 13.3× at 13. So the gap is set early
(a fat tree at shallow depth) and then widens because our EBF does not fall with depth the way SF11's does.

Reads:
- **The gap is mostly node efficiency.** In log terms 13.3× of the 24× is nodes and 1.8× is speed per node, about 80 / 20
  (memory `the-sf11-gap-is-two-thirds-node-efficiency` said two thirds; this real-play measure puts it higher). Our node
  counter also counts TT-hit bookkeeping, so the node part is somewhat overstated; the time column is exact.
- **Our EBF is flat at about 1.9 from depth 6 to 13; SF11's falls from 2.1 to 1.4.** A falling EBF is what depth-scaled
  reductions and pruning do (LMR growing with depth and move number, LMP, history pruning). That is S4-S5's target.
- **Our first reported iteration is depth 3** and already holds 766 nodes against SF11's 228: the pre-search and the
  shallow iterations are expensive. Worth a look in S3 (pre-search vs previous-score ordering).
- Our printed `bench` EBF (3.76) is a different quantity (memory `ebf-metric-is-not-comparable`); use this one.

### 6.2 Where the instructions and misses go (11-10, cachegrind)

Profiling copy `build_uci/chess_engine_uci_prof` (production flags + the gen 7.1 PGO profile + `-g -fno-omit-frame-pointer`;
bench 247 / 39,852,210). Workload: 9 game positions (3 per phase) to depth 10 through `_ebf_per_iter.py`. Simulated L1 32 KB
8-way, last level 128 MB 2-way. Output `E:\chess_data\s0_profile_2026-11-10\cg_annotate.txt`. About 5% of the instructions
are start-up and `ucinewgame` clearing (`clearSearchTables` 3.4%, static init 2.0%); they also own most last-level write
misses, and are excluded from the reads.

| group | share of instructions | biggest parts |
|---|---|---|
| eval v2 | **≈ 37%** | `placement_and_piece_eval_v2` 13.1, `ks_channels` 6.4, `build_side_attacks` 6.4, `placement_detect` 3.0, `ks_units` 1.8, `threats_mp` 1.5 |
| move generation | **≈ 18%** | `generateLegalMovesReordered` 8.1, `generatePseudoLegalMoves` 4.2, `generateLegalMovesPre` 5.3 (1.4 of it `std::vector` code) |
| search bookkeeping | ≈ 10% | maximizer / minimizer / `get_score_*` / `qSearch` |
| make / unmake | ≈ 7% | `make_move` 3.9, `update_state` 2.5, Zobrist 1.0, `hash_castling` 0.6 (F14) |
| SEE, attacks | ≈ 5% | `see_impl` 3.3, `attackersMask` 1.4 |

Reads:
- **Instruction-cache misses are about 7× the L1 data read misses** (362 M against 51 M). The eval alone takes 39% of
  them (`placement_and_piece_eval_v2` 69 M, `ks_channels` 27 M, `placement_detect` 17 M). The model has no micro-op cache,
  so the real cost is smaller, but code footprint is a lever on its own. It favours F11 (a frozen build drops v1's evaluator
  and every parked branch), v2's non-PV template (a smaller hot path), and keeping PGO.
- **Last-level data misses are few** (8.4 M read misses in 10.7 G instructions); the eval / TT cache probe owns 7% of them.
  Memory waits are not where this workload's time goes, which matches the downgrade of the prefetch items (§7).

### 6.3 Stack frames (11-10, `-fstack-usage`)

`E:\chess_data\s0_profile_2026-11-10\stack_top40.txt`. Per ply the search uses about 1.8 KB: `maximizer` 1,168 B +
`get_score_for_maximizer` 736 one ply, `minimizer` 1,232 + `get_score_for_minimizer` 416 the next. At a leaf the eval adds
`placement_and_piece_eval_v2` 2,848 + `ks_channels` 1,600 + `ks_units` 672. So near the leaves about 10 KB of the 32 KB L1
data cache holds stack. **v2 target: one search frame per ply well under 1 KB**, with move lists and per-ply data in the
`Stack` array rather than in frames.

### 6.4 Real-play profile and the prefetch question (11-10, perf)

perf on the profiling copy, 24 game positions to depth 12 (`E:\chess_data\s0_profile_2026-11-10\self_pf15.txt`,
`self_pf0.txt`, `annotate_make_move_pf*.txt`). Since the 11-10 WSL update, perf is called as
`/usr/lib/linux-tools/5.15.0-203-generic/perf record -e cpu-clock:u` (no call graphs at `perf_event_paranoid` 2; memory
`wsl-package-update-kills-detached-queues`), and it now resolves symbols itself.

Self time, prefetch on: `generateLegalMovesReordered` 11.2%, `placement_and_piece_eval_v2` 10.8, `accessSearchEvalCache`
6.9, `make_move` 6.6, `ks_channels` 6.2, `generatePseudoLegalMoves` 4.8, `build_side_attacks` 4.4, `see_impl` 4.0, `qSearch` 3.8.

**The 11-05 question: do `make_move`'s samples land on the prefetch hints?** Yes. With prefetch on, 53% of `make_move`'s
samples (about 3.5% of all) sit on the line after the TT hint in `prefetch_child_tables`. With prefetch off that cost moves
into the probes: `accessSearchEvalCache` 6.9 → 11.5%, `get_board_evaluation` 3.0 → 6.9, `qSearch` 3.8 → 7.1. The TT is
on 2 MB pages (checked: 208 of the engine's 218 MB in transparent huge pages), so it is not a guest TLB problem; WSL's
nested paging may still make each random probe a long walk (unproven). Time to depth 12, same 24 positions, identical
nodes, two alternating rounds:

| `PREFETCH_TABLES` | ms (mean) | vs none |
|---|---|---|
| 15 (TT, q-cache, eval cache, move-gen cache) | 641 | −11.1% |
| 14 (all but the TT) | 689 | −4.6% |
| 1 (the TT only) | 701 | −2.8% |
| 0 | 722 | — |

The four hints are worth more together than apart (the misses overlap), so the hot spot is the price of a hint that pays,
not a defect. Keep F1 as it is. Same-arm spread between rounds is about 5%, so only the 15-vs-0 and 15-vs-14 differences
are clear.

## 7. The 11-10 speed ideas (`owner-cpp-ideas.md` §14-15): review and placement

Each claim that could be checked against the code was checked (11-10); "verified" below means read in the source.

| item | verdict | where it goes |
|---|---|---|
| **15.1 deterministic yardstick** (cachegrind `Ir` + simulated misses) | **Adopt; the highest-value item.** It works on our `-march=native` build: the S0 run counted 10.7 G instructions with per-function misses. Limits: the model has no out-of-order overlap, no hardware prefetcher, a plain 2-way last level, so it cannot judge prefetch items (F20-F22) or branch-vs-cmov trade-offs. Use it to rank and reject result-identical changes; confirm with time. For search v2, instructions **per node** is a deterministic per-node cost even where the tree changes (S1, S2). | S0 now; every speed batch |
| **15.2 v1's bench wobble is a defect signature** | **Agree, and the check sharpened it.** `bench` runs at fixed depth 10 under the long-format limit with tables cleared per position, so a 1-3 node difference over 35 M cannot be a time abort (that moves thousands). Run the v1 bench once under Memcheck (`--track-origins=yes`). If clean, run it twice with ASLR off (`setarch -R`): if the wobble stops, the cause is address-dependent (ordering by a pointer value), which Memcheck cannot see. v2 benches do not wobble, but the code is shared. | one-off, next idle hour |
| **F24** gate the counter-move-history read with `g_cmh_dirty` | **Valid, exact** (verified: one 64-bit word per context row, bit = from-square, set by every write through `cmh_ref`; the decay already relies on the invariant). The prize is smaller than the guess: in the S0 cachegrind run the scorer's last-level misses are about 1% of the total. | small result-identical batch on today's search (it carries into S1's move source A) |
| **F26** presence map in front of `moveFrequency` | Valid, exact (verified: 2×64×64 ints = 32 KB, written only for PV moves). A small L1 saving. | same batch |
| **F25** rank / file attacks without PEXT | Valid premise (verified: separate rank, file and diagonal PEXT tables behind the `Row` proxy). Little here (PEXT is fast on Zen 3); its value is portability to Zen 1 / 2, where PEXT is microcoded. | release-readiness sweep |
| **History to 16 bits** | Right that it is not provable today (verified: five unbounded `historyHeuristics +=` sites). In v2, route every history write through the bounded update, which also makes the divisor a compile-time constant without F11. Changes results, so it is a v2 knob, not result-identical. | S5 |
| **F11** frozen-config build (X-macro, `dumpknobs`, compile errors as the audit) | A good release design; experiments keep the runtime knobs. Run its one-hour probe at the release sweep. | release sweep |
| **F20 / F21 / F22** prefetch window, look-ahead key, hint levels | Agree with the downgrade: removals before hiding. Keep T0 as the default; NTA only as a per-CPU option measured at the end. Time is the only judge (the yardstick cannot see prefetch). | after F24, only if misses remain; F22 at release |
| **F27** gives-check decided once | Valid and standard (SF's check squares + blockers); `attackersMask` is 1.4% of instructions in the S0 run. It is F16 in practice. | S2 |
| **F28** stop copying the board per child | Verified (`get_score_for_maximizer` takes `BoardState` by value). In v2 by construction: a preallocated state stack with a pointer bump. Not worth doing on today's search, which v2 replaces. | S1 by design |
| **F29** carry the piece in a 32-bit move | The problem is real (the piece is rediscovered by bitboard tests in several places). I would solve it the standard way instead: a 64-byte piece-on-square array in the state (SF, Ethereal), which serves every lookup (SEE, eval, history keys, not only moves in lists), and keep the 16-bit move for the TT and history. Widen list entries only if profiling shows decode cost. | S1 design decision |
| **F30** one 64-bit sort key per move | Valid; exact by construction (unique keys make any sort equal to the stable sort). Small. Applies to the sorted-list move source; the staged picker selects partially anyway. | S2 or any batch |
| **F31** material-signature table | Conditional, as written. Eval-side, ≤ 1%, and the eval cache already absorbs repeats. | eval arc |
| **Set-aside list** (TT probe before make, SSE2 both colours, eval attacks into movegen, incremental PST) | Agree with all four. | — |
| **15.4** newer compiler, `thread_local` | Worth one build each at the release sweep (we are on GCC 11: `/usr/include/c++/11` in the S0 run). Caveat: with `-Ofast` a different compiler or libm can change start-up floating-point tables (the LMR log table), so bench identity may break and has to be checked, not assumed. | release sweep |

## 7b. Transition items still open on 11-10 (owner: "let's at least give those remaining transition items a look")

Checked against the 10-19 transition list (`archive/handoffs/SESSION-HANDOFF-2026-10-19.md`), the 10-08 preplan and the
10-30 eval-items order:

| item | what it is | state on 11-10 | proposal |
|---|---|---|---|
| **Bench checkpoint #3** (10-19 list, the formal last step) | checkpoint #2's scope (POT brief §5f): static ladder; depth ladder vs SF11; critical-position move test vs SF11 / SF18; WAC / STS; strength vs SF18, Mediocre ≥ 200 g, plus a Stockfish 1.1 series point | **Partly in hand for gen 7.1:** static ladder (#169, gen 7's eval, which lazy eval does not change); depth ladder (§6.1); SF18 at equal nodes 56.9% over 2,250 g (#189 / #192 control arms); Mediocre 81.3% over 300 g at 10+0.1 (#189 control). **Missing:** the Stockfish 1.1 point, the record queue (`_queue_v2_record.sh`: STS300, STS full, WAC at LIGHTNING, speed) and the critical-position move test | run the missing legs (timed ones outside the owner's games) |
| **Batch 9** (banked 11-04) | (a) the child key once per node: `make_move_cache_key` + `hash_castling` run inside about 17 probe helpers, the prefetch and the repetition key; (b) the move-cache snapshot copy (`out = entry.moves`, a ~100-byte memmove per node); (c) no zeroing of what is overwritten (`KsChannels ch;` at two eval sites, the passer count array) | never built; result-identical; guess +1.5-2.5% nps. All three sit in code S1's port reuses | build with F24 + F26 as one batch, judged by the instruction-count yardstick, then real-play nps |
| **Tree blow-ups** (10-19 "v2's tree blows up on some positions") | positions where our tree explodes | **found again in §6.1's data:** position 54 (midgame) 21.9 M nodes / 12.1 s to depth 13 against SF11's 63 nodes (83 at depth 18); position 27 (endgame) 10.1 M against 1,870. SF11's tree collapses (a forced result is known); ours keeps growing about 2.7× a ply. Likely cause: no mate-distance pruning and no mate scores in the TT (the gen 6.6 bundle, not kept for its SF18 read). Not yet confirmed: the run did not save FENs (fixed in the probe 11-10; the sample can be regenerated) | confirm on the two FENs; then the mate handling is designed into v2 from S1 (standard, ply-adjusted TT mates + MDP), judged where it fires |
| **Owner-game triage re-run** (`_case_triage.py`, 17 cases; 10-18: EVAL 6, DEPTH 7, PRUNING 2, ok 2 on gen ~4) | which of your game positions fail by eval, depth or pruning | **re-run 11-10 on gen 7.1 (queue #193): ok 3, EVAL 6, DEPTH 7, PRUNING 1.** SF18's best moves are identical to 10-18, so every change is ours. g6c now solved (normal search h1h2 → e7f6, SF's move). g2a: the no-pruning search no longer finds f4f5 ⇒ now EVAL (static gap 15 win% pts). g3a stays the clean PRUNING case (no-pruning finds a4c3, normal plays e4g3). **Game 7 splits: g7a DEPTH** (neither search finds c3c4; our static at the end of SF's line within 9 pts) **and g7b EVAL** (gap 12). g1d's static gap closed most (ours at the end of SF's line +196 → +72 against SF −56). Logs `E:\chess_data\triage_*_1110.log`; the 10-18 inputs kept as `triage_*_1018.json` | the DEPTH (g1c g5b g5c g6a g7a g3b g2b) and PRUNING (g3a) cases are search v2's test set; test C proper (the plan moves' ordering index and reduction along SF's line at g7a / g7b) needs its probe, in S1+ |
| **POT-as-search** (10-19 list) | three parts: race + margins; the complexity scaler as an "unresolved" signal for LMR / time; pending-break protection (owner game 7) | never built. Two parked-register triggers point at "POT-as-search in the transition" (complexity scaler, middlegame POT) | owner's call: re-point all three to S4-S5 (they need the per-ply stack and reduction signals, and sit beside Reckless's line signals and the passer prune exemption), or build a first version now |
| SF11 pawn-ending rules (10-08 preplan, first item) | `NULL_NPM_GUARD`, `PE_PRUNE_GATE`, `PE_PASSER_EXT` | built 10-15; the guard shipped in gen 6.5; the other two did not fix the bias (the lever was the eval) | closed |

## 7c. POT reread: review of the analysis (`POT-REREAD-ANALYSIS-2026-11-10.md`, Fable 5.2, 11-10)

**Verified in the code or the record:**
- `PawnEntry` holds `lever`, `blocked`, `openFiles`, `halfOpen`; `pawn_hash_probe` and `passer_potential` exist.
- `ENABLE_PRUNE_SHADOW` / `ENABLE_SHADOW_EVENTS` and `_shadow_auc.py` exist.
- `transform_critical.csv` / `transform_trap.csv` exist in two sets (`classes_move/`, `classes_move_g2/`), so test S-2 can
  be cross-set.
- Test A's data is on disk: `E:\chess_data\deep\mg_cand_d12_s*of3.csv` and `mg_wdl_train_s*of3.csv` / `mg_wdl_val.csv`.
- The C64 pawn-ending cell raises the scale ceiling to 128/64 (`f += POT_V2_WIN_PAWNEND`, `fmax = 64 + …`). So a false edge
  in a dead-drawn pawn ending is amplified up to 2×, which makes E1 the missing guard on a shipped cell.

**Two corrections to my handoff brief that the analysis caught, and two more found while checking it:**
1. The material-scaled label matters little for the middlegame fit and the sibling move tests (agreed).
2. A feeder into an additive leg is an added score under another name, so M1 / M3 were in effect tested (agreed).
3. The boosted trees kept −7.26% (half their gain) at full size on SF's win model (POT brief §5o). "Did not survive" was
   the half-size check only.
4. `pot_owner_cases.txt` labels game 7 **EVAL** ("an unresolved, structurally pending break the KS term cannot see yet");
   only POT brief §5m re-read it as depth / pruning. I repeated the re-read as fact. Test C (with the triage re-run)
   decides.

**Agreed:**
- The core finding: magnitude is a second-moment quantity, and netting Black − White destroys it, so the magnitude half
  was never tested on its own terms.
- The split: magnitude steers trust and selectivity; direction becomes a narrow move class; neither adds to the eval.
- Tests A-D with their stop rules.
- The "unlikely" list.

**Additions:**
- **Test A:** replicate on the 5,988 held-out quiet WDL rows (`E:\chess_data\wdlcal`, d10). The trajectory starts are
  tactical.
- **Test B:** if no two-depth pass exists, run our d10 and d12 moves on `game_regret_set.csv` (fixed depth, safe at any
  time). The d7 regret ruler failed at ranking arms (#176). Depth gain by stratum within one arm is a different question,
  but read it with that in mind.
- **Test C:** today's LMR is keyed by **iteration** depth, not remaining depth (ENGINE-ORIENTATION). Near the leaves, a
  plan move can be reduced as if it had the whole depth ahead, which could make §4.2's reduction chain worse here than in
  SF. Record the reduction each plan move receives.
- **S-2:** the precedents both point at narrowness. `ENABLE_PASSER_PRUNE_EXEMPT` (+2.9% nodes) was flat; protect-killers
  (+1.16% nodes) won +12.4.

**Results so far (11-10 night):**
- **Test A** (`_pot_mg_fit.py MODE=magnitude`, registered in its docstring): structure summed over both sides predicts the
  size of our d12 error beyond the controls (score, phase, material, |static − d12|) by **1.22-1.25×** per fold, top fifth
  over bottom fifth; the shuffled null is 0.95-1.08. The bar was ≥ 1.25 in 4 of 5 folds ⇒ **registered MISS (0 / 5)**, by a
  hair and with a real, consistent effect. The bar is not moved after the fact; the owner decides whether the quiet-row
  replication runs anyway.
- **Test B** (`_pot_mg_fit.py MODE=depthgain`, registered before the d12 data finished; queue #193 d10 and d12 passes on
  `game_regret_set.csv`, 15,000 rows, mean regret 1.391 → 1.219, gain 0.172 win% pts): U = test A's β applied out of
  sample. Top fifth of U over bottom fifth, depth gain **1.13× on all rows** (+0.023 ± 0.066, 1 SE) against the 1.5× bar ⇒
  **registered MISS** (prediction 1.2-1.6×, too high). Opening 0.97, midgame 1.08, endgame 1.95 (+0.116 ± 0.119): every
  split within 1 SE. ☠️ **Underpowered:** the bar is a gap of ~+0.07, about 1 SE at this per-row variance (gain is mostly 0
  with rare large values), and the 2σ upper bound is ~1.9×. So the point estimate misses but the effect is not ruled out:
  UNRESOLVED, not refuted. A readable version needs ~4× the rows or a lower-variance target (critical rows only).
- **Reading A and B together:** structure carries a real but modest signal about where our score is wrong (A, ~1.24×),
  and no measurable signal about where two more plies fix it (B). The magnitude-as-selectivity form (extend or reduce on
  U) has no support yet. The trust form (wider aspiration windows or a less aggressive cut where U is high) is not what
  B measured; it stays a candidate for S5 only with its own test.
- **E1:** PARKED (tablebase study, register row).

### 7d. POT due diligence (owner 11-11: "lets give it the proper due diligence before writing it off or benching")

No POT item is closed or parked on tests A and B as they stand (A missed by a hair with a consistent effect; B was
underpowered). Four tests on today's engine come first. Each one's prediction and stop rule are written here BEFORE it
runs and are not moved afterwards. The search-side candidates (S-1 to S-5, analysis §5) are placed only after these.

| step | question | data and method | registered (11-11, before any run) |
|---|---|---|---|
| **A-rep** | does structural magnitude predict the size of our error on independent QUIET rows? | `wdlcal/cal_wdl.csv` (6,000 quiet rows, SF18 WDL labels). Fresh d10 pass of production (`_depth_residual_pass.py`) plus our static eval. β = test A's, fitted on all of test A's rows and FROZEN; the control model (same four controls) fitted on the quiet rows by 5 random folds; the same estimator (top fifth of U over bottom fifth of mean error, within control terciles). Shuffled-U null | **KEEP:** pooled ratio ≥ 1.25 AND bootstrap 95% lower bound > 1.10. **Prediction: 1.10-1.25** (quiet errors are smaller and less structural than the tactical trajectory starts) ⇒ a likely MISS on the bar with a lower bound above 1.0. **Stop:** lower bound ≤ 1.0 ⇒ the magnitude half has no quiet-row support |
| **B-powered** | is extra depth worth more where U is high? | `game_regret_set` (15,000) + `game_regret_set_v2` (11,940, after finishing its last 1,208 d12 rows) pooled ≈ 26,900 rows; the same U and estimator (`MODE=depthgain`, a `DIR` knob for both sets). Secondary: critical rows only, best-vs-second gap ≥ 10 win% pts in the stored multi-PV (lower variance) | **KEEP:** pooled ALL ratio ≥ 1.5 with top − bottom > 2 SE, OR critical-row ratio ≥ 1.5 with > 2 SE. **Prediction:** ALL 1.0-1.3 (MISS), critical 1.1-1.6 (open). **Stop:** both reads with 2σ upper bounds below 1.3 ⇒ magnitude-as-selectivity (S-3 extend / reduce on U, S-4 time on U) closes; the trust form (margins) needs its own test |
| **C** | are plan moves reduced or pruned as if they were ordinary late moves? | a probe on the triage cases' SF lines (g7a, g7b, g1c, g5b, g5c, g6a, g3a, g3b, g2b): for each SF move, its ordering index, reduction and whether it was pruned at the node where it is played, our normal search vs `NO_PRUNE` | registered 11-11 before any trace ran (`diagnostics/_line_trace.py` docstring; C++ `TraceRec`, `TRACE_TARGETS`, off = byte-identical). Unit: a visit, in the final iteration, to the board after SF's quiet move k ≥ 2. "Buried" = pruned (LMP / FUT / HIST / SEE) or reduced ≥ 2 plies. **KEEP** (supports S-2): in the missed cases (DEPTH + PRUNING) buried ≥ 50% of quiet visits, ≥ 1.5× the ok + EVAL cases, and in at least half of the missed cases. **STOP:** < 25% in the missed cases. **Prediction:** 35-60% missed, found lower but not 1.5×; most visits at k = 2-3 |
| **D** | do POT markers separate wrong prunes? | `ENABLE_PRUNE_SHADOW` + `_shadow_auc.py` with four new fields (U at the node; lever push; heavy piece onto a break file; break file at a king zone) on a quiet middlegame set, fixed depth | as analysis §5: **KEEP** a marker at AUC ≥ 0.60 on ≥ 200 wrong events that still separates within the existing leaders' strata; **Stop:** nothing above 0.55 (today, and again at S5) closes the move-class half. Made exact 11-11 before any pass (`diagnostics/_pot_shadow.py` docstring): the shadow record carries the node's board + move (log-only), markers computed offline with test A's features (U, lever, heavy_brk, brk_king); 1,500 quiet middlegame wdlcal rows at d10, `SHADOW_N=512`; KEEP = pooled ≥ 0.60 / ≤ 0.40 with ≥ 200 wrong AND stratum-weighted (i × rd × history sign) ≥ 0.57 / ≤ 0.43. **Prediction:** U 0.52-0.58 on LMR, lever / heavy_brk too rare or ≤ 0.57 ⇒ MISS, STOP not met |

Order: gen 7.2's ship queue first (#199; nothing else runs during a build), then A-rep and B-powered (no new code), then C
and D as their probes are written. All fixed-depth or node-limited: safe beside the owner's games.

**Results (queue #200, 11-11 13:02; `q200.log`):**
- **A-rep = KEEP (registered primary).** Production gen 7.2 at d10 on 5,940 quiet rows, test A's β frozen. Ratio **1.327**
  (bootstrap 95% 1.240 … 1.415), every fold ≥ 1.25 (1.264 1.304 1.414 1.252 1.367), shuffled-U null 0.964. Above the predicted
  1.10-1.25.
- The cross-check on the 11-06 arm-S dumps (`d10_v3s` / `static_v3s`, the pre-ship candidate; its static differs from gen 7.2
  on 173 rows and its d10 by 41.6 cp mean) reads **1.183 (1.097 … 1.282)**, null 1.001: a miss on that arm's bar by a hair.
- **Reading:** three reads on two independent row sets and three engines now agree: test A 1.22-1.25 on tactical trajectory
  starts, 1.33 and 1.18 on quiet rows, nulls ≈ 1.0. **Structural magnitude carries real information about WHERE our score
  is unreliable**, at about 1.2-1.35×. The size depends on the arm: an error-size ratio is conditional on which errors that
  engine makes. The magnitude half is not closed. Its use is the TRUST form: margins, aspiration and time where U is high
  (S-3, S-4), not an eval term.
- **B-powered = MISS, still UNRESOLVED (stop rule NOT met).** Both sets pooled, 26,919 rows: ALL ratio **1.31**, top − bottom
  +0.050 ± 0.050 (1 SE); opening 1.42, **midgame 0.74** (−0.063 ± 0.072), endgame 1.41. Critical rows (gap ≥ 10, n 1,403):
  top fifth +0.275, bottom −0.080, +0.356 ± 0.275 (1.3 SE). Set 2 alone (the named replication): ratio **1.53**, but
  +0.075 ± 0.074 (1 SE) ⇒ MISS on the SE half. The 2σ upper bounds (ALL ≈ 1.95×) are far above 1.3, so the stop rule is not
  met.
- **Reading:** the depth-gain signal points the right way on both sets and pooled, but the per-row target is too noisy to
  resolve it. +0.050 at 2 SE needs ≈ 4× the pooled rows (~110k SF18-labelled rows). The critical-row read needs ≈ 2.4× its
  rows. The midgame split is negative, the stratum where POT was expected to matter. **Selectivity on U (extend / reduce,
  S-4 time on U) stays unsupported but open.** A readable version needs a lower-variance target, not more of this one.

**Results, test C (11-11 13:20; `_line_trace.py`, test module `so_trace`; queue #201's own traces were VOID: the probe imported
the live module, which has no tracer, so 0 records; fixed, then re-run directly):**

| read | missed cases (DEPTH + PRUNING) | found cases (ok + EVAL) | ratio | registered |
|---|---|---|---|---|
| d10, final iteration | 11 / 20 = **55.0%** | 9 / 27 = 33.3% | 1.65× | **KEEP** (3 of 6 missed cases ≥ 50%: exactly the bar) |
| d12, final iteration | 13 / 23 = **56.5%** | 12 / 34 = 35.3% | 1.60× | **KEEP** (4 of 6) |
| d10, all iterations (secondary) | 36 / 78 = 46.2% | 20 / 92 = 21.7% | 2.1× | MISS (under 50%) |
| d12, all iterations (secondary) | 51 / 107 = 47.7% | 32 / 147 = 21.8% | 2.2× | MISS (under 50%) |

- **Ordering:** SF's quiet moves come late near the leaves: median index 5-7 at remaining depth 1, 3 at depth 2, 1-2 at depth
  6+. SF's tactical moves sit at index 0 at every depth.
- **Reach:** our search visits SF's line rarely, 0-152 records per case. Two cases (g5b at d12, g6b) barely touch it.
- **Reading:** by the registered rule, KEEP at both depths. Where we miss SF's move, SF's own quiet continuation is buried
  (pruned, or reduced ≥ 2 plies) in about 55% of final-iteration visits, against about 35% where we find it, and the
  contrast doubles across all iterations. The direction agrees at both depths and on both reads. The prediction (found
  lower but not 1.5×) was wrong.
- **Limits, stated with the KEEP:**
  - The primary reads rest on 20-23 visits from 6 missed cases; a single visit moves d10 across the 50% bar. A
    two-proportion test on the final-iteration counts is not significant (d12 p ≈ 0.18). The all-iteration contrast is
    large (z ≈ 4), but its visits cluster within cases.
  - The burial is consistent with the miss and does not prove that un-burying finds the move. The triage's no-pruning arm
    (LMR, LMP, futility, razoring and reverse futility all off) still misses 7 of the 8 missed cases at d10; only g3a
    recovers. So at equal depth, removing burial is not sufficient. The claim that survives: at equal time, the depth our
    reductions buy is spent away from these lines.
- **Consequence:** S-2 (reduce transformation moves less, exempt them from LMP) gains direct support as a WHERE-TO-SPEND
  item. Its own registered test stays the gate (transform sets + games, node cost under +3%).

## 8. Questions for the owner

1. **S1 scope** (open, refined 11-10). One for one cannot be exact. Today's two halves do not mirror each other (memory
   `search-correctness-asymmetries-2026-10`), and negamax has one code path, so for each asymmetry v2 has to pick one form:
   - **A1** LMR re-search: the maximizer accepts a reduced fail-high unverified, the minimizer verifies it at full depth
     (`FIX_RESEARCH`). Standard = verify.
   - **A2** the minimizer scouts at (alpha, alpha+1) instead of its own bound (`FIX_MIN_SCOUT`). Negamax does the correct
     thing by construction. On WAC the fix read 258 → 250 with −12.6% nodes: the margins may have been fitted to it.
   - **A3** a minimizer-only full-window branch at remaining depth 1 (`FIX_MIN_RD1`: 1 drops it, 2 adds it to the max side
     too). Productive on WAC both ways round (243 and 254 against 258), so a v2 knob.
   - A4 (`FIX_MAX_NULL`) is already fixed in production since gen 6.5.
   - Other quirks carry over as they are, each a later knob: LMR keyed by iteration depth (LMP by remaining depth), the
     light-endgame LMR bypass (`FIX_LMR_EG` lost −19.8).
   The A1 / A2 game numbers on record are void (the 10-20 matches never sent the knobs), so none has a games read.
   ~~Proposal (S1a): play A1 × A2 on today's search and port the winner.~~ **Withdrawn 11-10 on a record check:**
   `SEARCH-CORRECTNESS-ITEMS-2026-11-02.md` §3 already found that the bound, the reduction scale and VERIFY are ONE mechanism
   (VERIFY is the hidden second look that makes the reduction size work), to be replaced together as LMR v2, not fixed one
   at a time. **Revised proposal:** the negamax port carries a "root side" flag, so it reproduces today's asymmetries exactly.
   Then S1 needs no knob games first, and it can be held to **node-for-node identity at fixed depth against today's
   production search** (bench 247 / 39,852,210 through `SEARCH_ARM=1`): any difference is a port bug or a listed, intended
   change. LMR v2 (the symmetric bound, the reduction scale and VERIFY, together) is an S5 stack.
2. **Separate history tables for v2: yes** (owner 11-10: "we may be revisiting history formulas as well as how the tables
   are designed and shaped").
3. **SMP: later, like NNUE** (owner 11-10): an nps booster to apply later. Strength testing stays single-thread against
   single-thread engines, but before any public release (CCRL or similar) the engine must have a multi-thread option to
   stay competitive. The `Stack` design (one per thread) keeps Lazy SMP possible.
