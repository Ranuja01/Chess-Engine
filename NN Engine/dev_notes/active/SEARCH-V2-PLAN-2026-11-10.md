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

## 8. Questions for the owner

1. **S1 scope:** port every current feature one for one before anything new (the plan above), or start v2 with the
   traditional stack directly and port ours as options afterwards? The first keeps a clean "same search, new structure"
   read; the second gets to the interesting comparison sooner but loses that read.
2. **Separate history tables for v2**, or share ours? Sharing is less code; separate tables allow the re-keying in S5
   without touching v1.
3. **Lazy SMP** is out of scope (single-core goal), but the `Stack` design keeps it possible. Confirm.
