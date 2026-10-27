# SEARCH AUDIT — QSEARCH · TT · ROOT · TIME MANAGEMENT · DRAW/MATE (2026-10-22, read-only reference audit)

Scope: survey what 8 strong engines do in these five subsystems, put ours next to them gap by gap, re-examine the past
closures, and propose a staged, knob-gated build + test plan. Nothing was built or run (an overnight queue is live).
Tags: **[E]** = evidence with a citation (file:line for local code; URL + function for fetched code) · **[S]** = my inference.

Sources read:
- **Ours** (gen 5 tree, 2026-10-22): `search_engine.cpp`, `search_engine.h`, `cache_management.h`, `uci_main.cpp`.
- **Stockfish (local; paths relative to the parent folder `...\Programming\Chess Engine\`)**:
  - SF11 `stockfish_11/stockfish-11-win/src/`
  - SF15.1 `stockfish_15/stockfish_15.1_win_x64/src/`
  - SF18 `stockfish_1/src(SF18)/`
  - SF19 `stockfish_19/stockfish/src/`
- **Fetched, pinned commits**:
  - Ethereal `AndyGrant/Ethereal@0e47e9b` · Weiss `TerjeKir/weiss@c735b8f` · Berserk `jhonnold/berserk@3262851`
  - Obsidian `gab8192/Obsidian@720614a` · Stormphrax `Ciekce/Stormphrax@d1468d9` · Gyatso `GyatsoYT/GyatsoChess@75d38ad` (Nim)
  - URL form: `https://github.com/<repo>/blob/<commit>/src/<file>#L<n>`. Gyatso's files are under `Gyatso/src/`.
  - The fetched-engine line numbers come from extraction passes over the raw files at those commits. They are cited as `file:function Lnn`.
- **Context**:
  - `SEARCH-CORRECTNESS-PLAN-2026-10-19.md`, `UCI-BINARY-DESIGN-2026-10-19.md`
  - `research-2026-10-08/SEARCH-TRANSITION-PREPLAN.md` §1 items 7/18/21/22/25/30-32 and §6 items 11-13/16-17
  - The memory notes named in the brief.

---

## 0. BOTTOM LINE

1. ☠️ **The biggest finding is in TIME, not search.** We throw away every unfinished iteration [E `search_engine.cpp:3685-3694`, `:3734`]. In `movetime` mode (every st=1 test) a new iteration *always* starts unless the last one alone took the whole budget [E `uci_main.cpp:180-185`, `cpp:3594`]. So a large share of every move's time goes to a depth that is discarded [S, size unmeasured]. 6 of the 8 engines use at least part of the unfinished iteration; only Gyatso discards it all.
2. **The time manager has none of the consensus inputs.** The giants all stop between iterations on `elapsed > optimum × f(stability, score drop, best-move node share)`, with a mid-iteration hard stop. Ours gates on *the last iteration's duration* against a scaled preset table [E `cpp:3594`, `:3623`, `:3720`]. It uses no stability, score-drop or node-share input [E], and has no single-legal-move shortcut [E grep].
3. **Low time: start ID at depth 1 and never abort depth 1.** That is 8/8 practice [E]. Ours starts at 3, and the first pass cannot be interrupted safely [E `cpp:3564`, `:3507`, `:3883-3891`].
4. **qsearch is already close to the modern consensus on which moves it searches.** It searches captures (SEE ≥ 0), all promotions and ep, and no quiet checks, because the detector is dead [E `cpp:9397-9541`, archive handoff 07-28]. SF18/19, Ethereal, Weiss, Stormphrax and Gyatso search no quiet checks either. The quiet-check closure (−31…−64 Elo, v1 era) still holds.
5. **qsearch structure departs from consensus in five places.** Each is cheap to gate:
   - no TT probe or TT move in qsearch (only a score cache, stored at q-root only);
   - no "skip quiet evasions once not losing";
   - no move-count cap;
   - a `best = ±mate` init instead of the stand-pat seed, which all 8 use;
   - a hard q-depth cap of 10, which none use [E].
6. **TT: the record already knows it is direct-mapped with no aging, eval or ttPv. The new finding is the replacement rule.** A different position **always** evicts the slot, so depth-preferred applies only to the same key [E `cache_management.h:1063-1076`]. The consensus is 2-3-entry buckets with `depth − age` victim choice, mates stored with ply adjustment, and draws stored. We store neither mates nor 0 scores [E `:1046-1047`].
7. **Draw/mate gaps are all cheap and all consensus:**
   - mate-distance pruning (8/8; ours none);
   - randomised draw score (6/8);
   - mate beats the 50-move draw on the 100th ply (6/8);
   - 2-fold in-tree but 3-fold pre-root (5/8; our FIX_REP_KEY bit 2 tested −12 ± 42, unresolved, not refuted);
   - cuckoo upcoming-repetition (5/8).
8. **Root presearch + root razor:** no reference engine does either (0/8), but the fair equal-NODE test (−25 ± 19.6) still stands. The open questions are equal-TIME behaviour and two costs the record never priced:
   - the presearch re-runs on **every aspiration retry** [E `cpp:3995-3999`];
   - discarding the partial iteration wastes the presearch's investment too.
9. **Aspiration:** the DELTA closure holds as a one-knob result. The structural differences have never been tested: fail-high depth reduction (8/8), `beta = (α+β)/2` on fail-low (5/8), and centring on an *average* score. Centring on the mean of the last two iteration scores is a direct test of the odd/even oscillation that a 51% widen-event rate at a ½-pawn window suggests [S].
10. **Order:**
    - Stage 1: time manager (partial-iteration adoption, ID from depth 1, soft/hard split).
    - Stage 2: draw/mate cheap set.
    - Stage 3: qsearch TT + evasion pruning.
    - Stage 4: TT buckets/aging.
    - Stage 5: aspiration structure.
    Every item is a default-off knob, byte-identical at 0, with fire counts, judged at **equal actual average time** with pre-registered predictions.

---

## 1. FEATURE × ENGINE TABLES

Legend: ✔ = has it · ✘ = absent · numbers are the engine's own units (SF18 pawn ≈ 208; NNUE engines ≈ 100-200/pawn). OURS in millipawns (pawn = 1000).

### 1a. Quiescence
| feature | SF11 | SF15.1 | SF18/19 | Ethereal | Weiss | Berserk | Obsidian | Stormphrax | Gyatso | **OURS** |
|---|---|---|---|---|---|---|---|---|---|---|
| stand-pat = full eval | ✔ | ✔ | ✔ (+corrhist) | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ `cpp:8136` |
| TT score replaces stand-pat when bound allows | ✔ :1428 | ✔ (~7 Elo :1452) | ✔ :1578 | ✘ | ✔ | ✔ | ✔ | ✔ | ✘ | ✘ |
| fail-high blend toward beta | ✘ | ✘ | ✔ `(v+β)/2`; SF19 `(441v+583β)/1024` | ✘ | ✘ | end only | ✔ | ✔ 610/1024 | ✘ | ✘ |
| `best` seeded with stand-pat | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✘ (`±mate`; `ENABLE_QSTANDPAT_SEED=false` h:276) |
| quiet checks | q-ply 0 | q-ply 0 | ✘ | ✘ | ✘ | q-ply 0,−1 | q-ply 0 | ✘ | ✘ | ✘ de facto (detector dead) |
| promotions | queen | queen | all (via capture_stage) | all | all | queen | all | queen | all | all + ep unconditionally `cpp:9397-9409` |
| capture SEE threshold | ≥0 | ≥0 | ≥ −80 (SF19 −74) | ≥ max(1, α−eval−123) | ≥0 | ≥0 | ≥ −32 | ≥ −81 | ≥0 | ≥0 `cpp:9413-9414` |
| per-move futility (eval+margin+victim ≤ α) | ✔ +154 | ✔ +153 | ✔ +351 (SF19 +306) | node-level vs best case | ✔ +165 | ✔ +79 | ✔ +156 | ✔ +142 | ✘ | ✔ +1500 mp `cpp:8191-8217` |
| "futilityBase ≤ α && !see_ge(move,1)" prune | ✔ | ✔ | ✔ (`see_ge(α−fb)`) | — | ✔ | ✔ | ✔ | ✔ | ✘ | ✘ |
| move-count cap | ✘ | moveCount>2 | moveCount>2 | ✘ | ✘ | ✘ | 3 seen | 2 legal | ✘ | ✘ |
| evasions: prune quiets once not losing | evasionPrunable | after 2nd quiet evasion | skip non-captures | (no in-check qs) | ✔ | ✔ | ✔ | ✔ | ✘ (incomplete evasions) | ✘ all evasions searched `cpp:8053-8134` |
| TT probe + cutoff in qsearch | ✔ non-PV | ✔ | ✔ non-PV, depth ≥ DEPTH_QS | ✔ all nodes | ✔ non-PV | ✔ non-PV | ✔ non-PV | ✔ non-PV | ✘ | q-cache score only, every node `cpp:8043` |
| TT move tried first in qsearch | ✔ | ✔ | ✔ | if tactical | ✔ (even quiet) | in check only | ✔ (even quiet) | ✔ | ✘ | ✘ |
| q-store | every q-node | every | every, depth DEPTH_QS | every, d0 | every, LOWER/UPPER | every | every | every | ✘ | **q-root only** `cpp:9572-9593` |
| depth cap | ✘ (recaptures-only below q-depth −5) | same | ✘ | MAX_PLY | MAX_PLY | MAX_PLY | MAX_PLY | MAX_PLY | MAX_PLY | **MAX_QDEPTH=10** h:283 |
| in-check → mate score | ✔ | ✔ | ✔ | n/a | ✔ | ✔ | ✔ | ✔ | ✘ | ✔ (`FIX_QS_CHECK` at the cap) |

[E] SF11 `search.cpp:1355-1560` + `movepick.cpp:247-252` + `types.h:210-212`; SF15.1 `search.cpp:1421-1563`; SF18
`search.cpp:1496-1730`; SF19 `search.cpp:1662-1830`. Fetched engines: Ethereal `search.c:qsearch L812-927`; Weiss
`search.c:Quiescence L84-237`; Berserk `search.c:Quiesce L866-1020`; Obsidian `search.cpp:qsearch L483-634`; Stormphrax
`search.cpp:Searcher::qsearch L1450-1642`; Gyatso `search.nim:qSearch L90-153`.

### 1b. Transposition table
| feature | SF11/15.1 | SF18/19 | Ethereal | Weiss | Berserk | Obsidian | Stormphrax | Gyatso | **OURS** |
|---|---|---|---|---|---|---|---|---|---|
| entry | 10 B (key16, move, value, eval, genBound, depth) | 10 B | 10 B | 14 B | 10 B | 10 B | 10 B | 16 B | ~40 B (`key64, score, depth, flag, move, alpha, beta(dead), valid`) `cache_management.h:69-83` |
| bucket | 3 | 3 | 3 | 2 | 3 | 3 | 3 | 3 | **1** (`TT_WAYS=1` h:3325; N-way built, untested) |
| victim on miss | min(depth − 8·age) | min(depth − age) | min(depth − 4·age) | min(depth − age) | min(depth − age/2) | min(depth − 8·age) | min(depth − 2·age) | min(depth − 2·age) | **the one slot, always** |
| overwrite same key | EXACT ∨ d > old−4 (+2pv in 15.1) | EXACT ∨ d+2pv > old−4 ∨ aged | skip if not EXACT ∧ d < old−2 | skip if d+4 < old ∧ ¬EXACT ∧ age 0 | EXACT ∨ d+4 > old ∨ aged | same | same | always | d ≥ old `:1067` |
| generation/aging | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✘ |
| ttPv flag | ✔ | ✔ | ✘ | ✘ | ✔ | ✔ | ✔ | ✘ | ✘ |
| stored static eval | ✔ | ✔ (raw, pre-corrhist) | ✔ | ✔ raw | ✔ raw | ✔ raw | ✔ raw | ✔ | ✘ (separate 128 MB `evalCacheNew`, raw-zobrist key) |
| cutoff node types | non-PV | non-PV, `ttDepth > depth − (v ≤ β)`, cutNode-consistent if depth ≤ 5 | non-PV, all-node only on fail-low | non-PV | non-PV, `(cutnode ∨ v ≤ α)` | non-PV, fail-high +1 depth & cut-node only | non-PV, `(v ≤ α ∨ cutnode)` | **PV too** | **all nodes incl. PV**, child-probe `cpp:5060-5068`, `use_tt_entry :8659` |
| rule50 guard on cutoff | `rule50 < 90` | `< 96` | ✘ | ✘ | ✘ | `<90` + 50mr-bucketed key | `<90` + mate clamp by rule50 | ✘ | ✘ |
| mate scores stored, ply-adjusted | ✔ | ✔ (+GHI downgrade) | ✔ | ✔ | ✔ | ✔ (loss-branch bug) | ✔ (rule50-aware) | ✔ | **never stored** `:1046` |
| draw (0) scores stored | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✘ (`ENABLE_TT_STORE_DRAW=false` h:255) |
| "TT score more informative" eval substitution | ✔ (~4 Elo SF15 :738) | ✔ | ✘ | ✔ (`TTScoreIsMoreInformative`) | ✔ | ✔ | ✔ | ✘ | ✘ |
| useless-bound penalty | ✘ | SF19 `penalize(1)` :914-920 | ✘ | ✘ | ✘ | ✘ | ✘ | ✘ | ✘ |

[E] SF11 `tt.h:38-98`, `tt.cpp:36-56`, `:120-142`, `search.cpp:688-731`, `:1563-1584`; SF15.1 `tt.cpp:45`; SF18
`tt.cpp:35-123`, `:223-244`, `search.cpp:701-800`, `:1465-1471`, `:1763-1790`; SF19 `tt.cpp:109`, `:156-159`,
`search.cpp:874-920`. Fetched: Ethereal `transposition.c:tt_store L133-167`, `search.c L388-411`; Weiss `transposition.h
L49-101`, `search.c L302-311`; Berserk `transposition.c TTProbe L123-160`, `TTPut L162-180`, `search.c L440-442`;
Obsidian `tt.cpp L66-131`, `search.cpp L743-759`; Stormphrax `ttable.cpp L37-69`, `L160-219`, `search.cpp L679-706`;
Gyatso `tt.nim L111-155`, `search.nim L196-206`.

### 1c. Root / iterative deepening
| feature | SF11 | SF18/19 | Ethereal | Weiss | Berserk | Obsidian | Stormphrax | Gyatso | **OURS** |
|---|---|---|---|---|---|---|---|---|---|
| ID start depth | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | **3** `cpp:3564` |
| root pruning | ✘ | ✘ | ✘ | ✘ | SEE/history only, no LMP | ✘ (all pruning off at root) | ✘ | ✘ | **presearch + root razor (break)** `cpp:3999`, `:4258-4304` |
| root re-sort by last score | stable_sort each attempt :462/:506 | same :383/:424 | ✘ (normal picker) | ✔ each attempt | ✔ | ✔ incl. stopped | ✔ incl. stopped | ✘ | presearch sort `reorder_legal_moves :7135` |
| aspiration from depth | 1 (`previousScore` centre) | 1 (`averageScore` centre) | 4 | 1 | 5 | 4 | 3 | 3 | **5** h:3315 |
| initial δ | 21 + \|prev\|/256 | 5 + \|msq\|/9000 (SF19 /10193) | 10 | 9 + s²/16384 | 8 | 6 + avg²/13000 | 7 + \|asq\|·65/2²⁰ | 20 fixed | **500 mp** h:3314 |
| widening | δ += δ/4+5 | δ += δ/3 (SF19 +47δ/128) | ×1.5 | ×4/3 | ×1.25 | ×4/3 | ×(1+17/16) | ×1.5, cap 6 | ×2 failing side only, 3 tries, then full window h:3316-3317 |
| fail-low beta | (α+β)/2 | β = α | (α+β)/2 | (α+3β)/4 | (α+β)/2 | (α+β)/2 | (α+β)/2 | unchanged | unchanged `cpp:3668-3672` |
| fail-high depth reduction | ✔ failedHighCnt | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ ≤3 | ✔ ≤2 | ✘ |
| partial iteration used | ✔ (sorted rootMoves) | ✔ | fail-high only | last attempt | via sorted retries | ✔ | ✔ | ✘ | **✘** `cpp:3685-3694` |
| best move requires `> alpha` | ✔ (`moveCount==1 ∨ value>alpha`) | ✔ :1318 | ✔ | ✔ | ✔ | ✔ | ✔ | — | ✘ default (`ENABLE_ROOT_BEST_REQUIRES_ALPHA=false` h:1507) |
| MultiPV | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✘ | ✘ |

### 1d. Time management
| feature | SF11 | SF15.1 | SF18/19 | Ethereal | Weiss | Berserk | Obsidian | Stormphrax | Gyatso | **OURS** |
|---|---|---|---|---|---|---|---|---|---|---|
| optimum (sudden death) | move_importance skew-logistic, horizon 50 | `(0.012+(ply+3)^.45·.0039)·timeLeft`, ≤0.2·time | log-time-scaled `optScale·timeLeft`, ≤0.213·time | 2.5·(t+25inc)/50 | min(0.022·tl, 0.2t) | min(0.42t, 0.0575·total) | min(0.025, 0.214t/tl)·tl | (t/19 + 0.94inc)·0.68 | (t/20+0.9inc)·0.68 | **t/30 + 0.8·inc** `uci_main.cpp:190` |
| maximum | 7.3 MaxRatio | ≤0.8t, ≤7× opt | ≤0.825t, ~3-6.7× opt | 10·(…)/50 | min(5·opt, 0.8t) | min(0.92t, 5.9·alloc) | 0.8t | 0.65t | 0.58t | **min(4·budget, t/4) − 30 ms** `:191` |
| stop rule | between iterations, elapsed > opt × f | same | same | same (from depth 4) | same | same (from depth 5) | same (from depth 4) | same | same | **last iteration's DURATION ≤ MOVE_TIMES[d]** `cpp:3594`, `:3720` |
| mid-iteration hard stop | ✔ max | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ TIME_LIMIT `cpp:4203`, `:4514`, `:5477`, `:8006` |
| never abort depth 1 | — | — | SF18 `completedDepth ≥ 1` :1968 (SF19 drops it; rootMoves[0] always valid) | depth > 1 | ✔ | ✘ | ✘ | ✔ | ✘ (can send 0000) | ✘ (first pass; empty-move fallback `cpp:3883-3891`) |
| best-move stability factor | ✔ timeReduction | ✔ | ✔ (sigmoid in depth since last change) | ✔ pv_stability | `uncertain` veto | ✔ | ✔ | ✔ | ✔ | ✘ |
| best-move change count | ✔ | ✔ | ✔ | ✘ | ✘ | ✘ | ✘ | ✘ | ✘ | ✘ |
| score-drop (falling eval) | ✔ | ✔ | ✔ | ✔ vs 3 iters ago | ✘ | ✔ | ✔ | ✔ EMA | ✘ | ✘ |
| best-move node share | ✘ | ✘ | ✔ `nodesEffort` | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✘ |
| single legal move cap | ✔ stop | ✔ 500 ms | ✔ 502/500 ms | ✘ | ✔ 500 ms | ✔ 250 ms | ✔ 200 ms | ✔ 500 ms | ✘ | ✘ [E grep] |
| clock check interval | per node w/ counter | 1024 | 512 calls | 1024 | 2048 | 1000 | 4096 | 1024 | 2048 | 2048 (binary) / 200000 (Cython) h:3207 |
| move overhead | 30 (opt.) | 10 | 10 | 300 | 6 | 50 | 10 | 10 | 10 | 30 ms |
| low-time special | ✘ | ✘ | ✘ | ✘ | **pruning OFF until elapsed ≥ opt/64** | ✘ | ✘ | ✘ | ✘ | ✘ |

[E] SF11 `timeman.cpp:35-133`, `search.cpp:395-581`, `:1740-1743`; SF15.1 `timeman.cpp:37-103`, `search.cpp:462-492`;
SF18 `timeman.cpp:47-138`, `search.cpp:259-540`, `:1943-1972`; SF19 `search.cpp:560-605`, `:2105-2130`, timeman diff
vs SF18 (centiMTG→mtg, re-tuned constants). Fetched: Ethereal `timeman.c L59-130`; Weiss `time.c L27-77`, `search.c
L675-679`, `L749-759`; Berserk `uci.c:ParseGo L176-205`, `search.c L68-84`, `L332-365`; Obsidian `timeman.cpp L11-34`,
`search.cpp L668-672`, `L1446-1472`; Stormphrax `limit.cpp L34-118`, `L185-203`; Gyatso `timeman.nim L14-51`.

### 1e. Draws and mates
| feature | SF11 | SF18/19 | Ethereal | Weiss | Berserk | Obsidian | Stormphrax | Gyatso | **OURS** |
|---|---|---|---|---|---|---|---|---|---|
| in-tree repetition | 2-fold | 2-fold | 2-fold | 2-fold | 2-fold | 2-fold | 2-fold | 2-fold | 2-fold (`REPETITION_THRESHOLD=2` h:3218) |
| pre-root occurrence | needs 3-fold | needs 3-fold | needs 3-fold | **2-fold** | needs 3-fold | needs 3-fold | needs 3-fold | **2-fold** | **2-fold** (bit 2 of `FIX_REP_KEY` = SF rule, OFF) |
| key includes castling/ep | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✘ default (`FIX_REP_KEY` bit 1 OFF) |
| scan window | rule50 ∧ pliesFromNull | same | since zeroing | rule50 | min(fmr, nullply) | rule50 | rule50 | rule50 | whole game (unordered_map counts) `cpp:8765` |
| upcoming repetition (cuckoo) | `has_game_cycle` :600-608 | `upcoming_repetition` :630, qs :1505 | ✘ | ✔ (α<0) | ✔ (not qs) | ✔ (+qs) | ✔ (+qs) | ✘ | ✘ |
| 50-move: mate on ply 100 wins | ✔ `position.cpp:1116` | ✔ :1403 | ✘ (acknowledged bug) | ✔ | ✔ | ✔ | ✔ | ✘ | ✘ `cpp:4599` (`halfmove_clock >= 100`) + B1 in-search clock bug (FIX_HALFMOVE off) |
| draw score | ±1 by nodes :91-92 | `VALUE_DRAW−1 + (nodes&2)` :127 | ±1 | +1…+8 | −1…+2 (qs 0) | 0 | {2,1,0,−1} | −2…+2 | **flat 0** |
| mate-distance pruning | ✔ :660-667 | ✔ :680-690 | ✔ | ✔ (+qs) | ✔ | ✔ | ✔ | ✔ (incl. root) | **✘** |
| mate scoring frame | plies from root | same | same | same | same | same | same | same | **game ply** (`state_history.size()`) `cpp:8069-8073` |
| insufficient material in search | eval-level | eval-level | ✔ `boardIsDrawn` | eval | ✔ | ✘ | ✔ | ✘ | eval-level (draw scaling) [S, not re-read] |

---

## 2. PER FEATURE — CONSENSUS, DISAGREEMENTS, EVOLUTION, ELO

### 2.1 Quiescence
- **CONSENSUS (8/8 or 7/8)** [E §1a]:
  - full-eval stand-pat;
  - `best` seeded with the stand-pat;
  - captures filtered by SEE;
  - per-move futility with victim credit;
  - full evasions with a mate score when in check;
  - no q-depth cap beyond MAX_PLY;
  - a TT probe with non-PV cutoffs plus a store at q-depth.
- **DISAGREEMENTS**:
  - Quiet checks: SF11/15.1 at q-ply 0, Berserk 2 plies, Obsidian 1 ply; SF18/19, Ethereal, Weiss, Stormphrax and Gyatso have none.
  - Stand-pat fail-high blend: SF18/19, Obsidian and Stormphrax yes.
  - Move-count cap: SF15+ 2, Obsidian 3, Stormphrax 2, others none.
  - SEE threshold: from −81 to 0.
  - Ethereal never runs qsearch in check: the check extension keeps it in main search.
- **EVOLUTION** [E]:
  - SF11: checks at q-ply 0, recapture-only below q-depth −5, evasion pruning, TT eval substitution.
  - SF15.1 adds `moveCount > 2` futility/move-count pruning ("~5 Elo" `search.cpp:1510`), "SEE (~5 Elo)" `:1536`, cont-hist pruning ("~2 Elo" `:1550`), and quiet-evasion cut after the 2nd.
  - SF18: no quiet checks at all (`MovePicker(..., DEPTH_QS, ...)` = captures or evasions, `:1611-1614`), SEE −80, stand-pat blend `(v+β)/2` (`:1588-1590`), "skip non-captures" for evasions (`:1670`).
  - SF19: re-tuned blend `(441v+583β)/1024` (`:1748`), futility +306, SEE −74.
  - ⇒ **the direction of travel is fewer qsearch moves, not more.**
- **ELO**: the SF15.1 comments above [E]; SF15.1 "ttValue as better eval (~7 Elo)" in qsearch `:1452`. The magnitude of SF's quiet-check removal is unverified.

### 2.2 Transposition table
- **CONSENSUS**:
  - clusters of 3 × 10-byte entries in 32 bytes (6/8);
  - victim = min(depth − k·age), with a generation bumped per `go` (8/8);
  - move preserved on a same-key no-move store (7/8);
  - stored raw static eval (8/8);
  - mate scores ply-adjusted to/from the TT (8/8);
  - cutoffs at non-PV nodes only (7/8; Gyatso is the exception);
  - EXACT stores always overwrite.
- **DISAGREEMENTS**:
  - ttPv flag: SF/Berserk/Obsidian/Stormphrax yes, Ethereal/Weiss/Gyatso no.
  - rule50 guard: SF, Obsidian, Stormphrax.
  - Cut-node-consistent cutoffs: SF18 (depth ≤ 5), Obsidian, Stormphrax, Berserk, Ethereal.
  - SF19's `penalize` of a useless bound.
- **EVOLUTION**:
  - SF11 save rule `d > old−4 ∨ EXACT` (`tt.cpp:43-46`), then SF15.1 adds `+2·pv` (`tt.cpp:45`), then SF18 adds `∨ relative_age` (`tt.cpp:101-102`).
  - Victim age weight: SF11 8·age, SF18 1·age in generation units (`tt.cpp:237-238`).
  - TT cutoff: SF11 `depth ≥` (`search.cpp:703-706`), then SF18 `ttDepth > depth − (v ≤ β)` plus `cutNode == (v ≥ β) ∨ depth > 5`, plus a TT-move verification at depth ≥ 8 (`:760-800`).
- **ELO**: SF15.1 TT-hit history update "~1/~3 Elo" `:637-650` [E]. The bucket/aging Elo is not annotated anywhere I read ⇒ unverified.

### 2.3 Root / iterative deepening
- **CONSENSUS (8/8)**:
  - ID from depth 1;
  - no root pruning;
  - fail-high depth reduction in aspiration;
  - a root RootMove list in which non-improving moves get −∞ and a stable sort brings the new best to the front.
  - 6/8 use (part of) the unfinished iteration.
- **DISAGREEMENTS**:
  - Aspiration start depth: 1 (SF, Weiss) to 5 (Berserk).
  - Centre: SF11 `previousScore`, SF15.1+ `averageScore` (SF15.1 `(2v+avg)/3` `:1242`; SF18 `(v+avg)/2` `:1310`), Obsidian the move's last score with δ from the average.
  - δ: from scores alone (SF18 `meanSquaredScore`) or a constant.
- **EVOLUTION**:
  - SF11 δ = 21 + |prev|/256 (`:436`), δ += δ/4 + 5 (`:500`), fail-low β = (α+β)/2 (`:483`).
  - SF18 δ = 5 + |msq|/9000 centred on `averageScore` (`:355-358`), fail-low β = α (`:385-386`), fail-high α = max(β−δ, α) (`:393-394`), δ += δ/3 (`:401`), `adjustedDepth = rootDepth − failedHighCnt − 3·(searchAgainCounter+1)/4` (`:371-372`).
- **ELO**: unannotated in source ⇒ unverified.

### 2.4 Time management
- **CONSENSUS**:
  - optimum ≈ 2-3% of (time + horizon·inc) per move, maximum ≈ 3-7× optimum capped at 58-83% of the clock;
  - soft stop *between iterations* at `elapsed > optimum × factors`;
  - hard stop inside the iteration at the maximum;
  - factors: best-move stability (8/8 in some form), best-move node share (7/8 of the modern), score drop (6/8);
  - a short cap for a single legal move (6/8);
  - clock checked every 512-4096 nodes.
- **DISAGREEMENTS**:
  - Whether depth 1 may be aborted: SF18, Ethereal, Weiss and Stormphrax never abort it; SF19 drops the guard because rootMoves[0] is always a legal move; Gyatso aborts it and can emit `0000`.
  - Weiss turns all selective pruning OFF until `elapsed ≥ optimum/64` (`search.c L675-679`) — a low-time-specific policy no other engine has.
- **EVOLUTION**:
  - SF11: a skew-logistic `move_importance` (`timeman.cpp:45-70`), stop on `optimum × fallingEval × reduction × bestMoveInstability` (`search.cpp:539-557`).
  - SF15.1: closed-form `optScale/maxScale` (`timeman.cpp:81-99`) plus a `complexity` factor (`search.cpp:470-473`).
  - SF18: log-time-scaled constants, `originalTimeAdjust` (`timeman.cpp:105-120`), `nodesEffort` (`search.cpp:486-503`), sigmoid timeReduction (`:493-496`).
  - SF19: re-tuned constants, interpolated timeReduction/nodeEffort, and an early stop on `score ≥ mate_in(3)` (`search.cpp:571-600`).
- **ELO**: TM gains are large in fishtest history but I hold no local citation ⇒ unverified. They are not needed for the plan: the plan measures our own.

### 2.5 Draws and mates
- **CONSENSUS**:
  - mate-distance pruning (8/8);
  - 2-fold in-tree repetition (8/8);
  - 3-fold for pre-root occurrences (6/8);
  - repetition key = full position key incl. castling/ep (8/8);
  - scan limited to the rule50 window (8/8);
  - mate takes precedence on the 100th ply (6/8);
  - randomised ±small draw score (6/8; SF since ~SF10 as `value_draw`);
  - cuckoo upcoming-repetition (5/8; SF since SF10 era: SF11 `has_game_cycle` `search.cpp:600-608`).
- **ELO**: unannotated ⇒ unverified. These are correctness items: the bar is "not harmful" (the narrow-item rule).

---

## 3. OURS vs CONSENSUS — GAP BY GAP (+ which past closures still hold)

### 3.1 Time management — SPECIAL ATTENTION
How ours works today:
- **T1 — the soft gate is per-iteration, not cumulative.** `while (... elapsed <= MOVE_TIMES[depth_limit] ...)` with `t0 = Clock::now()` reset before each iteration [E `cpp:3594`, `:3623`, `:3720`]. ⇒ the decision to start depth d+1 depends only on how long depth d took. Elapsed total, stability and score never enter.
- **T2 — `movetime` mode starts an iteration whenever the last one took < the whole budget** [E `uci_main.cpp:180-185` sets `MOVE_TIMES.fill(s)`]. Then the root loop or qsearch hits `TIME_LIMIT` [E `cpp:4514`, `:8006`] and the iteration is **discarded** [E `:3685-3694`, `:3734`].
  - Every timed test since 10-19 (st=1 s) runs in this mode, e.g. queue #85-#88.
  - ⇒ [S] at an iteration growth factor g ≈ 2-3, the discarded tail is typically 30-60% of the move's wall time. Its only return is a warmer TT/move-gen cache for the next move.
  - This is unmeasured. The `[iter] cum_nodes` log (`ENABLE_ITER_LOG`, `cpp:3728-3729`) plus total nodes per move measures it **without any build**: discarded share = (total − cum_nodes at the last completed depth) / total.
- **T3 — clock mode (placeholder).** budget = rem/30 + 0.8·inc; hard = min(4·budget, rem/4) − 30 ms; `MOVE_TIMES[d] = budget × STANDARD.MOVE_TIMES[d] / 45` ≈ 0.11-0.14 × budget [E `uci_main.cpp:186-195`, `search_engine.h` STANDARD table].
  - ⇒ depth d+1 starts if depth d took ≤ ~0.13·budget. With g ≈ 2.5 the next iteration is *expected* to finish at ≈0.33-0.45·budget, but nothing caps it below 4·budget. One bad iteration therefore spends 4× the budget and is then thrown away [S].
  - No movestogo horizon logic beyond `rem/mtg`, no ply dependence, no stability/score/node factors, no single-move cap [E grep `size() == 1`: no root shortcut].
- **T4 — low time.**
  - ID starts at depth 3 [E `cpp:3564`].
  - The first pass cannot be interrupted safely: a UCI `stop` is applied only after the first completed iteration [E `cpp:3507-3508`; `uci_main.cpp:143-146`].
  - A node/time limit inside it leaves an empty move, now patched to "first ordered legal move, unsearched" [E `cpp:3883-3891`].
  - Consensus: depth 1 is a few hundred nodes and is always completed [E SF18 `search.cpp:1968`; Ethereal `tm_stop_early` `depth > 1`; Weiss `OutOfTime` never at depth 1; Stormphrax `rootDepth > 1`].
- **T5 — the "equal actual time" requirement.**
  - Today the binary's average time per move under the clock rule is whatever the gate produces. Comparing two TMs at the same nominal TC confounds *allocation* with *usage*.
  - `fastchess_report.py` already prints ACTUAL time/move per engine [E memory `uci-binary-and-fastchess`]. The plan in §5 uses it as a hard matching constraint.

Gap list (consensus → ours):
- (a) Soft stop on cumulative elapsed vs optimum × factors, between iterations → ✘.
- (b) Hard stop at a maximum → ✔ `TIME_LIMIT`.
- (c) Unfinished-iteration best move used → ✘.
- (d) ID from 1, depth 1 never aborted → ✘.
- (e) Stability / score-drop / node-share factors → ✘. We already track the root best move across iterations (`completed_move`) and have per-root-move node counts available cheaply [S]: `num_iterations` deltas around each root `minimizer` call at `cpp:4385-4403`.
- (f) Single-legal-move cap → ✘.
- (g) Clock-check interval → ✔ 2048 (binary).

### 3.2 Root
- **R1 — presearch + root razor (0/8 engines).**
  - The **equal-NODE closure holds**: root-LMR replacement −25.0 ± 19.6 / 1,668 g at 150k nodes, −11.7 ± 30 at 400k [E memory `presearch-economics...`].
  - Still untested [E same note]:
    - the giants' full bundle `ROOT_TABLE=1 ROOT_LMR=1 EXEMPT_BEST=1 PRESEARCH=0 RAZOR=0`;
    - equal-TIME (presearch-off pays ~12% NPS in its fill loop);
    - the provenance-tiered sort's games.
  - New costs found this audit:
    1. `reorder_legal_moves` runs inside `alpha_beta`, so **every aspiration retry re-runs the presearch** [E `cpp:3995-3999` comment "Charged on every aspiration widening too"]. The TT likely makes repeats cheap [S]. `g_presearch_nodes` split by attempt index would show it (counter-only build).
    2. Under T2 the presearch of the discarded iteration is wasted with it.
  - ⇒ **Verdict [S]: justified distinctive choice at equal nodes; liability-unknown at equal time.** Do not re-open the replacement question until (c)/(d) exist. Once partial iterations are adopted and ID starts at 1, the cost structure of the presearch changes (its per-iteration price is paid more often at low depth), so the equal-time re-test belongs *after* Stage 1, as a 2×2 (presearch × partial-adoption).
- **R2 — partial-iteration adoption needs the root to know which moves are PROVEN this pass.** We have it under `ENABLE_ROOT_TABLE` (`verified`, `top_score` sentinel = SF's −∞; `root_table_store(..., score > alpha_before_move, ...)` `cpp:4431`) and, without the table, implicitly: `best_move` changes only on `score > best_score` [E `cpp:4462`].
  - The rule SF uses [E SF18 `search.cpp:1318-1352`]: a root move becomes best only on `moveCount == 1 ∨ value > alpha`, and only if the search was not stopped while inside it [E `:1301-1302`].
  - Ours returns from the root loop *after* the move whose child was interrupted [E `cpp:4418-4427`] and *before* updating `best_move`. So `best_move` at a time-out is the best of the moves completed this iteration — exactly SF's semantics, **provided** `best_move` required `score > alpha`.
  - It does not by default [E `ENABLE_ROOT_BEST_REQUIRES_ALPHA=false` h:1507, the "STILL OPEN :3517" item] ⇒ **partial adoption must be gated together with BEST_REQUIRES_ALPHA=1**: a fail-low bound must never become the played move.
- **R3 — aspiration.** DELTA = 500 mp (½ pawn) from depth 5, widen ×2 on the failing side, 3 tries then the full window, β unchanged on fail-low, no fail-high depth reduction, centred on the last score.
  - Consensus δ ≈ 0.05-0.16 pawn, 3-10× narrower [E §1c].
  - The record:
    - DELTA=750 null; 4× width cuts fails 65% and loses 7 WAC [E SESSION-HANDOFF-2026-08-31:71-72];
    - "51%" counts widen EVENTS, not iterations [E same].
  - ⇒ the **DELTA closure holds as a 1-D knob.** Never tested: fail-high depth reduction (8/8), `β=(α+β)/2` (5/8), averaged centre (SF15+), start depth.
  - [S] Our root is always a maximizer with minimizers at odd plies, so an odd/even score oscillation between iterations is plausible. That is what averaged centring addresses; checkable from existing `info depth … score` lines in fastchess PGNs/logs without a build.
- **R4 — `best_move` on fail-low passes:** see R2. **R5 — MultiPV:** absent. It is not a strength item; it matters only for analysis/UI [S].

### 3.3 Quiescence
- **Q1 — moves.** Captures SEE ≥ 0, all promotions + ep unconditional, no quiet checks (the default simulate-path check test checks the mover's own king and so never fires [E `cpp:9506-9525` — `update_state` takes `turn` by value, `is_check(turn, …, opposingPieces)` tests the side that just moved; archive SESSION-HANDOFF-2026-07-28 "qsearch has NEVER searched a quiet check", counter 0/300]).
  - ⇒ **matches SF18/19, Ethereal, Weiss, Stormphrax, Gyatso.** The quiet-check closure (MASK+DEPTH0 games −31.5 ± 36.3, −53.6 ± 36.1, gate ≈ −64 / 154 g; v1, 07-28) **holds**, and modern SF agrees.
  - Clean-up only [S]: the dead simulate path costs a board copy + `is_check` per quiet move at every q-node. That is pure NPS waste, worth checking with the profiler.
- **Q2 — stand-pat seed.** All 8 seed `best` with the stand-pat; ours inits ±mate [E `cpp:8166-8172`]. Measured: fake-mate return on ~59% of searched q-nodes, short-of-standpat on 14% [E comment `cpp:8167-8171`].
  - The seed alone lost −0.13 depth and was called a productive bug (C1 bundle +6 / 790 g null) [E preplan §1 #30].
  - Interaction [S]: with seed OFF, a maximizing q-node whose captures are all delta-pruned returns `−9999999+moveNum`, a fake mate below alpha. That is a valid fail-low bound but a garbage magnitude. It cannot reach the TT (mates not stored) but does reach the q-cache? No: `addToQCache` skips mates [E `cache_management.h:956-957`]. So the damage is confined to fail-soft magnitudes in the parent (aspiration widening uses ×2 steps, not the score, so it is also insulated).
  - ⇒ **the closure holds for now** (no consumer reads the magnitude). It becomes a defect the moment we store mates, add MDP, or use fail-soft values in TM score-drop. **Gate Q2 together with any of those.**
- **Q3 — evasions.** All legal evasions searched, no pruning [E `cpp:8053-8134`]; 7/8 stop quiet evasions after a non-losing score. Expected effect [S]: q-node reduction in check-heavy lines; low risk.
- **Q4 — TT in qsearch.**
  - The q-cache (128 MB, key = TT key) is probed at every q-node but stored only at the q-root [E `cpp:8043` vs `:9592-9593`]. Its hit rate on interior q-nodes is therefore only via transposition to another q-root [S].
  - No TT move in qsearch.
  - Consensus: store every q-node with depth 0 and the best move; try the TT move first.
  - Cheapest step: store at every q-node into the existing q-cache (+ a move field). This is node-efficiency, so judge at equal time.
- **Q5 — depth cap 10.** None use one; SF11/15 limit q-depth > 5 to recaptures. With FIX_QS_CHECK the cap no longer misses mates, but **an in-check node at the cap still returns a static eval** [E `cpp:8012-8037`]. Consensus never stands pat in check.
  - [S] Options: (a) at the cap, if in check, search evasions (bounded by a second cap); (b) SF11 form — beyond q-depth k only recaptures on the last-move square.
- **Q6 — move-count cap (SF15+ 2, Obsidian 3, Stormphrax 2) and "futilityBase ≤ α ∧ !see_ge(1)":** absent. These are node-savers; prior record: node savings < 35% are bench-invisible ⇒ games at equal time only.
- **Q7 — stand-pat fail-high blend:** a modern NNUE-era item (SF18+, Obsidian, Stormphrax). Low priority [S]: its value is tied to fail-soft TT stores we do not do at q-nodes.
- **Closure "cheap stand-pat"** (mode 1 −235 STS, mode 2 −258 at equal time [E memory `lighteval-standpat-is-leaf`]) **holds**. All 8 use full eval at stand-pat; their saving comes from the TT-stored eval, which we already have via `evalCacheNew`.

### 3.4 Transposition table
- **TT1 — replacement.** Direct-mapped (`TT_WAYS=1`). On a *different* key the slot is **always** overwritten; depth-preferred only for the same key [E `cache_management.h:1058-1076`]. ⇒ deep entries are evicted by any shallow colliding store. No generation, so nothing distinguishes this move's entries from those of 20 moves ago.
  - The N-way set-associative path exists (victim = shallowest, no age) [E `:1080-1105`] but no games record found (grep of dev_notes: only the 07-06 design and the 07-15 map mention it) ⇒ **unmeasured, not closed.**
- **TT2 — cutoffs at PV nodes.** The child probe applies `use_tt_entry` at every node type incl. i==0 full-window children [E `cpp:5060-5068`, `:8659-8686`]. 7/8 forbid PV cutoffs: PV truncation, GHI exposure and fewer exact re-searches [S]. Gate as "non-PV only" (β−α == 1).
- **TT3 — mates and draws never stored** [E `:1046-1047`]. Consensus stores both, mates ply-adjusted. Ours scores mates by game ply (`state_history.size()`) [E `cpp:8069`], which is position-path-independent only up to the root offset ⇒ storing would need `value_to_tt(v, ply_from_root)`. Value [S]: mate-search speed and fewer re-searches in mating attacks; needs MDP first to be safe.
- **TT4 — no stored eval, no ttPv, no rule50 guard, child-probe architecture:** known [E memory `tt-and-cache-architecture`]. **The node-TT closure holds as an ordering lever** (TT-move live with presearch off changed 2,245 / 48.7M nodes [E memory `presearch-economics...`]). **It does not hold as plumbing:** a node-entry probe is the substrate for TT eval substitution (~4-7 Elo SF15.1 annotations [E]), qsearch TT, IIR, singular and ttPv.
- **TT5 — entry size ~40 B with two dead ints** [E `:69-83`]. Packing to 16 B would double effective capacity at 512 MB [S]. An NPS/hit-rate item, not Elo by itself.

### 3.5 Draws and mates
- **D1 — no mate-distance pruning** (8/8 have it) [E grep]. Needs the plies-from-root frame (`state_history.size() − root_size`), easy [S].
- **D2 — flat 0 draw score** (6/8 randomise ±1-2). In our millipawn scale ±1 mp is negligible to the eval, but it breaks the 3-fold blindness SF cites [E SF11 `search.cpp:91-92`].
- **D3 — 50-move: no mate exception and the in-search clock bug B1** (`FIX_HALFMOVE` built, off) [E plan §B, `cpp:3316-3329`]. 6/8 check mate first.
- **D4 — pre-root 2-fold.** 5/8 use SF's 3-fold-pre-root rule; Weiss and Gyatso match us. Bit 2 measured −12 ± 42 / 200 g (+ WAC 253 vs 257) ⇒ **unresolved, not refuted.** The tie-break is the narrow-item rule: judge where it fires (games with a pre-root repeat), not on general games.
- **D5 — repetition key without castling/ep** (bit 1 built, off; consensus 8/8). Still off, because the #89 bundle was −3 ± 31 and the move test was mean −0.020 ± 0.014 [E plan §E]. That is a correctness item that "showed no harm and no benefit"; the record's own rule for correctness items is "ship on not harmful".
- **D6 — no cuckoo upcoming-repetition** (5/8). Value [S]: earlier draw detection = cutoffs one ply sooner in shuffling lines. Low priority behind D1-D3.
- **D7 — `unordered_map` repetition counts** [E `cpp:3265`, `:3338`, `:8765`]: a hash insert per make/unmake and a lookup per node. The consensus is a linear scan of ≤ rule50/2 keys. An NPS item [S]; profile before touching.
- **B3 FIX_DRAW_EXIT, C1 FIX_QS_CHECK:** shipped in gen 5 [E commit 875326a]. Consistent with consensus: none of the 8 engines exit ID on a draw score, and all score mate in check.

---

## 4. IMPROVEMENT IDEAS (fit to our engine; all [S] unless cited)

1. **Partial-iteration adoption (PIA).** `completed_move = best_move` also when time-up, if (i) at least one root move was fully searched this iteration and beat alpha (`score > alpha_before_move`), and (ii) `ENABLE_ROOT_BEST_REQUIRES_ALPHA=1`. This is SF's rule.
   - Test: equal-time st=1 self-play; fire count = moves where the adopted move ≠ the last completed move; SF18 move test at those FENs.
2. **Elapsed-based soft stop (TM-1).** Replace the per-iteration gate with `elapsed_total + predicted_next ≤ soft`, where `predicted_next = last_iter_time × g_ema` (our own measured iteration growth, per phase). This is an original fit: our growth factor is unusual (presearch + EBF) and measurable from `[iter]` logs. The giants avoid the prediction by simply checking `elapsed > optimum × f` after each iteration and relying on the hard stop + PIA.
   - **Build both** as modes (A = giants' rule, B = predictive); judge at equal ACTUAL time.
3. **Stability / score-drop / node-share factors (TM-2).** Port Stormphrax's or SF18's multiplicative form, with *our* signals:
   - stability = consecutive iterations with an unchanged `completed_move`;
   - score drop vs the EMA of iteration scores (millipawns → pawn-scale conversion);
   - node share = nodes under the best root move / total.
   - **Original twist:** feed the presearch's *score spread* (top − second presearch score at the current depth) as a fourth factor. A wide gap means an obvious move, so stop early. No giant has a full-width shallow pass to read this from; we do, for free.
4. **ID from depth 1 + never abort depth 1 (TM-3).** Makes low-time moves real searches. Changes the fingerprint ⇒ judged with TM-1.
   - Measure the presearch cost at depths 1-3: the presearch requires depth ≥ 2 to generate second-level lists [E `cpp:7147-7153`], so depths 1-2 may need a presearch-off path. The no-presearch code path exists [E `:7175-7378`].
5. **Low-time policy (TM-4)** — owner request:
   - (a) emergency mode when `rem < k × (inc + overhead)` or `rem < X ms`: optimum = min(opt, rem/40), hard = rem/8, and skip the presearch below depth N;
   - (b) Weiss's idea — pruning-off for the first `optimum/64` [E Weiss `search.c L675-679`] — is for *very short* searches; for us it would mean "presearch off, root razor off" at depths ≤ 3, where both are noisy;
   - (c) a single-legal-move cap of 50-200 ms.
   - Test: games at 10+0.1 and 60+0.6 with a fixed starting-clock handicap to force low-time phases; count flag losses and time-forfeits.
6. **Aspiration structure (ASP-1..3)**, each its own knob:
   - fail-high depth reduction (cumulative, reset on fail-low; 8/8);
   - fail-low `β=(α+β)/2` (5/8);
   - centre on the mean of the last two iteration scores (original target: odd/even oscillation).
   - Pre-check without a build: compute iteration-to-iteration score deltas by parity from existing UCI logs.
7. **qsearch Stage (QS-1..4)**:
   - QS-1 stop quiet evasions after a non-losing score;
   - QS-2 store every q-node in the q-cache + q-move field, try the q-move first;
   - QS-3 an in-check-at-cap evasion pass (supersedes C1's mate-only test);
   - QS-4 SF15-form `moveCount > 2` cap for non-check non-recapture captures.
   - QS-1 and QS-4 interact with the stand-pat seed ⇒ 2×2 with `ENABLE_QSTANDPAT_SEED`.
8. **TT Stage (TT-1..4)**:
   - TT-1 generation byte + `depth − 8·age` victim in the existing `TT_WAYS` path (3-4-way);
   - TT-2 non-PV-only cutoffs;
   - TT-3 store mates ply-adjusted + MDP (D1) together;
   - TT-4 rule50 < 90 guard on cutoffs.
   - TT-1 is an equal-time item (hit rate); TT-2..4 are correctness items.
9. **Draw/mate cheap set (DM-1..4)**: MDP · randomised draw ±1 mp by node parity · mate-before-50 + FIX_HALFMOVE · FIX_REP_KEY bit 1 (+ bit 2 judged where it fires).
10. **Presearch cost accounting (no behaviour change):**
    - split `g_presearch_nodes` by aspiration attempt;
    - nodes spent in discarded iterations;
    - root-move node share.
    These are the inputs TM-2 needs anyway.

---

## 5. STAGED BUILD + TEST PLAN

Rules for every stage [E plan §D; memories]:
- each item = one knob, default 0 = byte-identical (Cython `258 / 47,689,413` gen-4 fingerprint and the gen-5 binary `bench`; `EvalArm 0` `250 / 35,310,778`);
- fire counter on the taken branch, read before any games (`FIX_COUNT_LOG` pattern);
- SF18 d16 move test at the fire points (exact-trigger FENs, not first divergence: the 10-21 lesson);
- predictions registered in this file before running;
- 2×2 for any pair touching the same mechanism;
- ≤ 3 processes per queue, `restart=on`, no side runs while a queue runs.

**Equal-actual-time protocol for time-manager items** (the owner's requirement):
- (i) every TM arm prints actual ms/move (`fastchess_report.py` already does);
- (ii) a TM arm is accepted for a verdict only if its mean actual time/move per game is within ±3% of the control's, over the same openings;
- (iii) achieve (ii) by a single `TM_SCALE` knob (multiplies optimum and maximum), calibrated on 40 games before the match: arm and control each tuned to the same target average;
- (iv) as a cross-check, run the arm at two TM_SCALE points bracketing the control's time and interpolate Elo at equal time;
- (v) report Elo **and** the time ratio. A TM change that "wins" by spending more is visible immediately.

**Stage 0 — measurement only (no behaviour change; counters/logs; byte-identical):**
- (a) discarded-iteration share at st=1 from `ENABLE_ITER_LOG` + total nodes;
- (b) presearch nodes per aspiration attempt;
- (c) iteration-score parity deltas;
- (d) root-move node share;
- (e) q-cache hit rate at interior q-nodes.
Registered predictions:
- (a) discarded share median 30-55% at st=1;
- (b) retries add < 15% to presearch nodes;
- (c) |Δ| larger on odd→even than even→odd by ≥ 1.3×;
- (e) interior hit rate < 10%.

**Stage 1 — time manager** (binary only; the Cython path keeps presets):
1. PIA (+ `ENABLE_ROOT_BEST_REQUIRES_ALPHA=1` bundled by necessity, stated as such).
   - Fire count; SF18 move test at the adopted-vs-completed divergences.
   - st=1, 400 g fresh seeds, equal time is automatic.
   - Prediction: +10…+40 Elo, adopted move better than worse at ≥ 1.5:1.
2. TM-1 (A: giants' rule; B: predictive) × PIA on — 2×2 at 60+0.6 with the equal-time protocol.
   - Prediction: A ≥ B ≥ placeholder at equal time; ≥ +15 Elo vs the placeholder.
3. TM-3 (ID from 1, never abort depth 1). Low-time venue: 10+0.1 and a 5 s + 0.05 handicap clock.
   - Primary metric: time-forfeits and empty-move fallback fires (must → 0); Elo secondary.
4. TM-2 factors one at a time on the Stage-1 winner, then jointly (multiplicative forms are tuned jointly — SPSA later).
5. TM-4 low-time policy, same low-time venue.

**Stage 2 — draw/mate cheap set:**
- MDP, draw randomisation, mate-before-50, FIX_HALFMOVE, FIX_REP_KEY bit 1. Each alone on the fire-count harness, then as a bundle at equal time.
- Ship on "not harmful": bundle ≥ −1σ and the move test at exact triggers not negative.
- FIX_REP_KEY bit 2 judged only on games where a pre-root repeat occurs (narrow-item rule).

**Stage 3 — qsearch:**
- QS-1, QS-3 (correctness-leaning) alone.
- QS-2 (q-store everywhere + q-move) for node-efficiency.
- QS-4 × `ENABLE_QSTANDPAT_SEED` as a 2×2.
- Equal time for all; STS at equal nodes as a mirage check only.

**Stage 4 — TT:** TT-1 (buckets + age) at equal time and two hash sizes; TT-2 / TT-4 correctness; TT-3 only after D1 (MDP) has shipped.

**Stage 5 — aspiration:** ASP-1/2/3 alone, then a 2×2 of the two best; equal time.

**Stage 6 — re-open the root architecture at equal TIME:** presearch vs the giants' full bundle (`ROOT_TABLE=1 ROOT_LMR=1 EXEMPT_BEST=1 PRESEARCH=0 RAZOR=0` + the tiered sort), on the Stage 1-5 base. The equal-node −25 ± 19.6 stays the prior.

Tag summary for the plan: everything in §4-§5 is **[S]** (design); the gaps it targets are **[E]** with the citations above.
