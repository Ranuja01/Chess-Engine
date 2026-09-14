# Diagnostics toolkit — CHECK HERE BEFORE WRITING A NEW PROBE

`diagnostics/` holds ~200 scripts. Nearly every question we ask has already been answered by one of them,
and rebuilding wastes time, fragments conventions, and produces weaker versions (a rebuilt probe usually
lacks the *control set* that makes the original trustworthy). **Search this file first.**

🧭 **This file says which probe EXISTS. [`INSTRUMENT-MAP.md`](INSTRUMENT-MAP.md) says what each probe can
RESOLVE and how it LIES — read it before designing any screen.** Picking the right tool is the second
question; the first is whether any tool we own can see the effect you are predicting. When you measure a
new null, noise floor, or failure mode, add the row THERE as well as here.

Run everything through the prompt-free wrapper:
`wsl.exe -e bash -lc "bash '<abs overnight_runner.sh>' pyrun diagnostics/<script>.py [ARGS] [KEY=VAL]"`

---

## Ranking convention — use WIN%, not centipawns
Rank and prioritise eval errors by **Lichess win% (k=0.00368208)**, the same logistic the fit scripts use.
Two pawns of error at +8 barely changes the expected result; two pawns at 0.0 flips the game. Raw-cp ranking
over-weights blowouts and hides the errors that actually cost points. `sf11_collapse_gap.py` and
`probe_fens.py --table` already rank/report this way.

## The reference ladder — ours / SF11 / SF15.1c / SF15.1n / SF18s / SF18-search
Carry ALL of them, not just SF18. The point is **SF's own progression**: whichever generation is closest to
truth for a situation is the source to read for that concept.
- **SF11** = pure classical (pre-NNUE). Closest-to-truth ⇒ read SF11's HCE; the concept exists and is hand-encodable.
- **SF15.1 classical vs NNUE (same binary)** = isolates what is genuinely un-encodable by hand from what we
  are simply missing. SF15.1 is the LAST classical-king-safety Stockfish.
- **SF18 static vs SF18 search** = separates eval holes from search-determined positions. If EVERY static
  disagrees with the search, it is search's job and not statically fixable — do not chase it.
⚠️ Absolute paths and the `STOCKFISH_PATH` judge binary: memory `sf-source-paths-and-pruning-shapes`.

---

## Eval tuning — static coarse → low-depth move-ordering sharpen (2026-08-10)
The eval-tuning method, in two phases. ⚠️ Phase-1 (static) is a **coarse region-finder only** — fitting to a
static target is ANTI-CORRELATED with Elo (proven again 2026-08-10; see `corpus-fit-is-anti-correlated-with-elo`
memory). ✅ Phase-2 (low-depth search move-ordering) is the Elo-aligned pass — now that WE search, target =
SF18's actual SEARCH-best move, on equal footing. **Rank/score by WIN% not cp** (§ Ranking convention above);
low depth is a PROXY for game depth, so GAMES decide.

| script | answers | status |
|---|---|---|
| `_central_regime.py` | central clamp: SATURATED (identifiable) vs LINEAR (collinear) per position, from `ev_breakdown` det_central+phase. Decides if a term is worth bounded re-shaping. | settled |
| `_asym_corpus.py` | builds the SIGN-GATED, ASYMMETRIC (SF11-anchor / preserve-our-edge), win%-weighted static corpus `diverse_corpus_asym.csv`. The asymmetric TARGET is baked into `target_total` so the existing worker descends it. | settled machinery, **objective FAILED validation** |
| `_ks_fit_eval.py` | static fit worker; now win%-IMPACT weighted (reads `weight` col; absent ⇒ 1.0 = byte-id for legacy callers). | settled |
| `joint_fit.py` | static coordinate descent; added `GRID_ONLY=` (focus a knob subset) + bounded OVD/CENTRAL knobs. Base modes via `FORCE=`. | settled |
| `_depth_timing.py` | times our FIXED-depth search at MAX_DEPTH (one process/depth). Measured D7=61ms/pos (~14k nodes — heavy pruning ⇒ low-depth searches are tiny ⇒ tuning on search is affordable). | settled |
| `_build_lowdepth_set.py` | stratified FEN set + SF18-best@d16 cached ONCE (top-1) → `lowdepth_tuneset.csv`; disjoint from `_mp_*`. | settled |
| `_lowdepth_tune.py` | driver+worker: coordinate-descend eval knobs to MAXIMISE SF18-best top-1 match at fixed depth. ⚠️ binary objective = NO gradient (descent can't see sub-flip gains). Seed from static-best. | settled, superseded by regret |
| `_build_regret_set.py` | multi-PV cache: SF18 top-K move evals @ fixed depth → `regret_set.csv` (`moves`="uci:cp;..."). One-time; enables graded regret. | settled |
| `_regret_tune.py` | focused: coordinate-descend to MINIMISE side-to-move win%-REGRET = `winpct(best)−winpct(our_move)` (multi-PV cache; POV-fixed). GRADED + move-based ⇒ gradient AND un-gameable by shrink. **The eval objective.** | settled |
| `_regret_tune_broad.py` | whole-eval regret descent, 4-core (`JOBS`), **held-out-gated** (rejects overfits), seeded-shuffle split. Proved eval CONSTANTS tapped out. ⚠️ shuffle the split or a multi-config game-dir index split fakes universal overfitting. | settled |
| `_build_game_regret_set.py` | mine ~15k GAME-representative FENs from the 108k selfplay games + SF18 multi-PV @d14 → `game_regret_set.csv` (disjoint from benches). | settled |
| `_ks_channel_decomp.py` | KS channel firing (king_safety / OvD / central) on over-read (ks_attack) vs control (positional) — deterministic, no SF. Found the over-read is Channel-1 proximity, not collinearity. | settled |
| `_ks_c1_decomp.py` | Channel-1 SUB-decomposition (attacker-weights / count / weak vs safe-checks / storm / open-files) on over vs control. Found proximity dominates, safe-check ~silent. | settled |
| `_ks_discrimination.py` | **unit DISCRIMINATION AUC** (attack vs quiet_neg on `ks_sts_corpus`) per config — the verify-at-each-step metric for KS detector upgrades (survives the dead curve; move-regret is move-neutral there). | 2026-08-12 |
| `_ks_footprint_regret.py` | **footprint D7 move-regret** (base vs cand, changed-move filter) on BOTH cross-sets (`game_regret_set` 15k + `_v2` 11,940). The deployment gate. ⚠️ do NOT DEFER it. **☠️ READ THE 09-07/09 REPAIR ROW BELOW BEFORE USING IT.** | 2026-09-09 |
| **☠️ HOW TO READ THE GATE (09-07/09 — it was being read wrong for months)** | **1. The MEAN's null is NOT zero** — two Elo-NEUTRAL arms read −0.1389 / −0.0619 at the same flip rate; arm-specific AND corpus-specific, cannot be divided out. **2. Use `win%` of changed moves** — natural ~50% null. Measured bands: **49.8-50.4** (`game_regret_set`, three neutral arms) and **50.7** (`_v2`). Measure it **per corpus AND per stratum**. **3. CROSS-SET REPLICATION IS MANDATORY** — two arms cleared the primary band by ~+1pp then died on v2 ⇒ **a single-corpus ~+1pp reading is NOISE and the real bar is ~2-2.5pp**. **4. `n_crit`=27** ⇒ criticality unmeasurable; always quote it. **5.** It is an **EVAL** screen at d7 — structurally **blind to anything gated at depth ≥6** (`ENABLE_IIR` changed 0 of 15,000). **6. 🐛 `/tmp` worker paths were UNQUALIFIED until 09-07** ⇒ every concurrent run before then silently shared files (now PID-keyed; same defect fixed in `_ks_drift_analysis`, `_ks_channel_collinearity`, `_move_change_arms`). 🧰 New outputs: `BY_PHASE` (corpus buckets) · `BY_PS` (the ENGINE's `phase_score`, straddling the 64→69 step) · `BY_CRIT` (best-vs-2nd) · `bett/wors` + `win%` on every row · `DUMP=` for changed-set OVERLAP (what decides whether two arms can bundle). | 2026-09-09 |
| `_ks_phase_split.py` | the footprint STRATIFIED by phase (opening/midgame/endgame) — localized the KS over-read to the OPENING (endgame KS helps). | 2026-08-12 |
| `_ks_drift_analysis.py` | worst move-flips + KS units base→cand (why a config drifts). | 2026-08-12 |
| `_ks_channel_collinearity.py [MODE=heat\|ks] [N=900]` | ablate each channel, take per-position contribution (`base − ablated`), correlate ⇒ **HIGH r (>0.5) = the channels DUPLICATE one signal, consolidation safe; LOW = distinct information, de-dup SHEDS signal.** **`MODE=heat` (new default, 09-09)** targets the `attackingLayer` consumers: measured **heat 433.3mp / central 215.9mp / OvD 22.7mp**, r = **+0.389 / +0.420 / +0.131** ⇒ **all under threshold ⇒ CONSOLIDATION REFUTED.** `MODE=ks` is the original king-channel set. 🐛 Its old `ch4` ablated with `IMBALANCE_SCALE=0`, which only affects OvD **MODE 0** and has been **INERT** since `OVD_BOUNDED_MODE=2` shipped — that channel contributed nothing to every earlier run; `OVD_CAP=0` is correct. | 2026-09-09 |
| **`ENABLE_ORACLE_EVAL=1 [ORACLE_CLASSICAL=1] ORACLE_ENGINE_PATH=<binary>`** | puts **SF's eval inside OUR search** (~6,200 evals/sec, **fixed-depth only**, NPS drops 14×). The instrument that priced the whole eval lane: SF15c = **+7pp / 3-4 plies**, SF11 = **90% of that**. ⚠️ Verify liveness by changed-rate (~58%) — a near-zero rate is the **silent-fallback** signature. SF binaries+source are in the SIBLING `Programming/Chess Engine/` dir; **SF11 HAS a linux build** (`stockfish_11_linux/.../Linux/stockfish_20011801_x64_bmi2`) — never infer a path from one tool's mapping. ⚠️ `ORACLE_SCALE` is for node-matching ONLY; carrying it into a fixed-depth run degrades the eval (read −0.0177 vs −0.6681 raw). | 2026-09-09 |
| **`SCALE_ATTACK_LAYER=<pct>`** | one-knob ablation of the ENTIRE `attackingLayer` heat map (`cpp_bitboard.cpp:9370-9381`, finalisation-time scale, early-out at 100). Measured 0/50/100/150 → win% 48.9 / 49.4 / base / 49.9 ⇒ **map is worth ~1.3pp (replicated) and its magnitude is AT the optimum.** Companions: `SCALE_CENTRAL=0` and `OVD_CAP=0` ablate two of its three consumers. | 2026-09-09 |
| `_ks_unit_trace.py` | live KS unit distribution (confirmed the dead quadratic: 100% of dangerous kings past the knee). | 2026-08-12 |
| `_build_variant_regret_set.py` | **whacky/variant corpus generator** (piece-swaps + 960-no-castle) → SF18 multi-PV; the structure-independent set, UN-USED. See [[whacky-variant-corpus-for-structure-independent-validation]]. | 2026-08-12 |

🧰 Run the tuners directly (they self-dispatch a WORKER subprocess per candidate): `WORKER` path sets
`PRESET=LONG_FORMAT MAX_DEPTH=<DEPTH> USE_OPENING_BOOK=0`. ⚠️ The `_move_match_arms.py` SF reference is
`movetime=0.3s` = TIME-BASED ⇒ non-deterministic + corrupted under CPU load; its small deltas are NOISE. The
deterministic instruments are the cached SF18 sets above.

---

## Pawn model (2026-08-05) — see [`PAWN_MODEL.md`](PAWN_MODEL.md) for the findings

🚨 **`pawn_marginal_real.py` is the ANCHOR — use it before believing any manufactured-position magnitude.**
The generator below was right about direction everywhere and wrong about MAGNITUDE by 2.6×; three headline
claims died when re-measured on real positions.

| script | answers |
|---|---|
| `pawn_marginal_real.py` | marginal pawn value on **REAL corpus positions**, ours vs SF18, by rank. The anchor. |
| `pawn_truth_generator.py` | manufactured positions with controlled rank × structure × obstruction contexts. `BASE_ABS_CP` conditions on a balanced baseline; **never** filter on the outcome (`MAX_ABS_CP`, default off). |
| `pawn_truth_analyze.py` | answers rank/structure/obstruction contrasts from a truth CSV, with CI; prints UNRESOLVED rather than a small number. |
| `pawn_truth_ours.py` | per-cell residual `SF18 − ours` on identical FENs ⇒ no double-counting of terms we already pay. |
| `pawn_gap_attribution.py` | splits a marginal-value gap by eval TERM (`ev_breakdown` on paired FENs). Run before reshaping anything. |
| `passer_detector_diff.py` | counts our passed-pawn predicate vs SF15.1's, by rank. Pure predicate compare, no engine calls. |
| `_pawn_clamp_headroom.py` | per-pawn clamp binding rate and survival of an added bonus (`ai.pawn_clamp_records()`). |
| `_endgame_passer_doublepay.py` | whether a flagged sub-threshold endgame passer is paid by both the inline bonus and `evaluate_passers`. |
| `_venue_power.py` | **what effect size a game venue can actually RESOLVE.** Run before spending games, and before writing any null into a doc. |
| `pawn_truth_casebook.py` | regenerates §4 of `PAWN_MODEL.md` from the truth CSVs, between markers, so the doc cannot drift. `WRITE=1` to inject. **Quarantines the pre-fix CSVs** — `pawn_truth.csv` and `pawn_truth_valid.csv` must never be cited. |
| `pawn_fit_shipped.py` | pawn/passer win%-descent **in the SHIPPED regime** (no `ENABLE_KS_CHECK_V2`). Use this rather than `ks_fit_wholesystem.py` when the result has to be directly applicable — earlier descents optimised inside a regime we do not ship. |
| 🚨 **`_eval_symmetry.py`** | **colour-swap and file-mirror INVARIANCE. Zero Stockfish, seconds.** `eval(mirror(b))` must equal `-eval(b)`; found a LIVE bug at **74.7% of positions**, worst 374 cp. Validated: symmetric positions return exactly 0, results are deterministic and order-independent. ⚠️ A side-to-move term is no excuse — `mirror()` swaps `turn` too. |
| **`pawn_marginal_real.py`** | now also `PIECE=P\|N\|B\|R\|ALL` (ALL prices every type on ONE sample and prints the scale-invariant exchange rate), `LADDER=1` (adds SF11 + SF15.1-classical columns — **the triangulation gate**), `TERMS=1` (per-term attribution of OUR marginal), `OUT=<csv>` (dump every removal with position features so follow-ups cost zero CPU). 🚨 **Never splice marginals from separately-sampled runs** — a ratio of medians from different position sets is not an exchange rate; that produced three retracted numbers. |
| **`_marginal_slice.py`** | slices the `OUT=` dump — error distribution, worst-decile concentration, and conditioning by phase/queens/pawn-count, **in both cp and win%**. Zero CPU, re-runnable. The cp-vs-win% comparison is the point: they disagree on which positions are broken. |
| **`static_vs_search_triage.py`** | now has `CORPUS=1 [CLASS=] [PHASE=] [N=] [DEPTH=]` — turns the per-FEN verdict into a SPLIT over a failure class (eval-reachable vs search-property), with a **scale-invariant sign test** beside the cp-distance test and a **neutral-fitted** scale factor (`KFIT`). ⚠️ Fitting the scale factor in-sample on collapses is conditioning on the dependent variable; it gave 0.31 against ~0.97 neutral. |
| **`_sibling_spread.py`** | **can a term change which move we play at all?** Evaluates EVERY legal child of real positions, then DELETES a term group and re-takes the argmax — deletion is the ceiling on what retuning could do. Reports sibling spread and flip rate gated by regret margin. **Carries `threats` as the control (+45 Elo shipped)**; the reading is comparative, never absolute. `QUIET_ONLY=1` for the piece-moves-only case. Zero Stockfish. |
| ★★ **`_d1_move_attribution.py`** | **which of OUR terms made us pick the wrong move?** Depth-1 move choice vs the three-way reference, scored in **win%** (k=0.00368208), with **NO filter** — each position carries a signed weight `our_err − sf11_err` against SF18-search, so tactical positions self-suppress instead of being excluded. 🚨 **SF is an ORACLE OVER MOVES only**; our breakdown attributes our OWN choice. Never places an SF term beside one of ours — term names are not 1-to-1 across engines (our king-zone attacker−defender was once TRIPLE-counted, which is why term-level "we vs SF" comparisons mislead). Reports the truth move's rank in our ordering. 🐛 **Known flaw:** the guard table is structurally empty (for a guard case our move IS the truth move) — it should attribute against SF11's pick; and the blame sums are outlier-dominated, so prefer median / frequency-as-top-offender. |
| ★★★ **`_move_change_arms.py`** | **the knob-arm form of the above — RUN THIS FIRST, BEFORE ANY BENCH SWEEP OR GAMES NIGHT.** `_sibling_spread` deletes a breakdown TERM, so it cannot be pointed at a mechanism that is gated OFF; this one runs one process per arm (knobs latch at init) and diffs the depth-1 argmax. Reports flip rate and the baseline REGRET each flip costs. `ARM=<KNOB=VAL[,…]> [N=400] [QUIET_ONLY=1]`. **Always pass a control arm** — `ENABLE_THREATS=0` is the +45 Elo shipped change and measures **13.2% / 9.2% at ≥10cp**. 2026-08-08: of six candidates only ONE cleared that bar; running it LAST instead of first cost a session. ⚠️ It shipped with a whitelist argv parser that silently dropped its own arm knobs and reported a confident 0.0% — the control caught it in one run. |
| `blend_corpora.py` | merges the schema-compatible fit corpora with the four hygiene checks that make a blend trustworthy: **stale-baseline detection (refuses unless FORCE=1)**, cross-source FEN dedup, split re-assignment by FEN hash, explicit tier weighting. |

🚨🚨 **CORPUS REBUILT AGAIN, LATER ON 2026-08-06: `diverse_corpus_wide.csv` 4,987 → 23,113 rows.** Bank
extended 4,987 → 24,656 and labelled **24,656/24,656 at d13** (depth matched to the existing labels on
purpose — the script defaults to d18, and mixing depths would put two truth standards in one target
column). Snapshots `*_pre0806b.csv`.
⚠️⚠️ **NO `val` from before this rebuild is comparable, INCLUDING the 4,987-row numbers.**
New shipped-default baseline: **ALL.train 228.347 / ALL.val 231.177** (previous corpus: 230.555 / 234.881
— *not* comparable, listed only to prevent accidental cross-corpus comparison).
- tiers: diverse 9134 · calm 8478 · target 3816 · working 1025 · crowded_safe 185 · sts_guard 187 ·
  passer_blowup 140 · passer_control 79 · passer_under_fire 69 — **diverse+calm is now 76%**, against a
  bank that was previously king-safety weighted. Composition decides the optimum; treat this as a
  deliberate shift toward general play, not a neutral enlargement.
- phases: opening 7722 · endgame 6109 · midgame 5571 · adveg 3711. splits: train 18439 / val 4674.
- ⚠️ `passer_blowup_guard` reads train 382 / val 748 on 140 rows — a high-variance guard that a descent
  can trip or satisfy on noise.
- ⚠️ `DIR target` shows capture 0% **by construction** (the tier is defined as positions where SF11 fires
  KS and we score zero). Not a finding.
- ✅ val/train = 0.988 ⇒ still NOT overfitting at 8.5× the data. More corpus buys reliability, not
  proxy→Elo conversion; that needs a different OBJECTIVE.

⚠️ Marginal value shows diminishing returns where the context already holds similar assets — a phalanx
pawn's marginal value is low partly because its partner already carries it. Read MEDIANS, not means.

---

## Per-FEN inspection
| script | what it gives |
|---|---|
| **`probe_fens.py <fens> [--sf-depth 22] [--table]`** | **THE canonical per-FEN probe.** ours + SF11 + SF15.1c + SF15.1n + SF18 static + SF18 search, White-POV pawns. `--table` = one row per FEN with **win% error last**. Input `label<TAB>fen`. |
| `dossier_overread.py` | side-by-side our total + term breakdown vs SF11, formatted for human eyeballing |
| `fen_term_dump.py` / `dump_eval.py` / `dissect_fen.py` | full per-term breakdown for specific FENs |
| `compare_terms.py` | SF11-static vs our static term table for FENs on argv |
| `breakdown_partition_check.py` | verifies the breakdown partition sums to `total` (⚠️ `pieces` already contains `material`; `pt_*`/`material` are SUB-VIEWS — do NOT sum them) |

## King safety — the 2026-08-18 instrument set (see [`KING_SAFETY_MODEL.md`](KING_SAFETY_MODEL.md))
⚠️ The old 82-position archetype bench (`_ks_bench_score.py`) is a WEAK instrument: at the default floor only
35% of its positions are live, three archetypes are structurally dead, three more are a single position each,
and it is NETTED between kings. Prefer the two below, and **always run bench arms with `KS_FLOOR=0`**.

| script | what it gives |
|---|---|
| **`_ks_auc.py [KEY=VAL] [HI= LO= PHASE= DUMP=]`** | **DISCRIMINATION.** AUC over `ks_sets/diverse_corpus_wide.csv` (23,113 rows carrying SF's per-term `target_ks`; midgame n≈5,500), SE≈0.007. ☠️ **Run with `KS_FLOOR=0` and ALWAYS pair with the volume control `KS_ATTACK_COUNT=2`** — the floor manufactures mass ties so volume alone inflates AUC (+0.107 floored vs +0.007 floor-free). That control caught a false "detection win". |
| **`_ks_calibration.py [KEY=VAL]`** | **MAGNITUDE.** ours/SF ratio per SF-|KS| band + Spearman. The dimension AUC is *structurally blind* to — a feeder defect that shrinks danger uniformly preserves order (AUC flat) while every number comes out too small. Found the 1-2 pawn hole and the overall 0.45 ratio. |
| `_ks_bench_liveness.py` | how much of the archetype bench is non-zero + per-archetype top-1 concentration. **Run before trusting any bench delta.** |
| `_ks_detect_dist.py [KEY=VAL]` | per-king unit means (attsq/weak/safe/attpc/units), DANGER vs QUIET vs STS_REGRESS, floor forced 0 |
| `_ks_dblpawn_coverage.py` | pure python-chess geometry probe: how much of our zone an SF-style mask would actually remove |

## Search internals (2026-08-18/19)
⚠️ **`NODES` excludes the q-tree** (`qnodes=` on the `[search]` line is separate) — judge q-tree levers on
`qnodes` + FIXED-TIME depth, never `NODES`. ⚠️ For node-reducers, fixed-time TACTICAL (`wac_timed_depth`)
alone MISLEADS (rewards discarding positional nodes) — cross-check `sts_timed_depth`; only a both-instruments
winner is real. Noise: mean DEPTH stable ±0.001 (rank on it); solves ±4; STS-score ±30-60 (repeat).
| script / knob | what it gives |
|---|---|
| `ENABLE_QUIET_PROBE=1` (gated, byte-id) | stderr `[qquiet]` (qsearch terminal quietness by reason: horizon/standpat/quiet/searched + tension) + `[qbug]` (short-of-standpat, fake-mate counts) + `[searchbug]` (null unproven-mate, TT-EXACT-mislabel). |
| `_qquiet_agg.py FILE=<err>` | aggregates the `[qquiet]` lines across a bench run |
| `_capg_orient_check.py CLS=<classified.csv>` | is a term's collapse over-read REAL (oriented by collapsing side) or a White-POV colour artifact |
| `_collapses_from_selfplay.py TAG=<gate/sprt tag>` | extract collapses post-hoc from gate/SPRT `game.jsonl` (they write no collapses.csv). ⚠️ A/B near-identical engines under-sample — vs_sf is stronger for collapse mining. |
| `wac_timed_depth` / `sts_timed_depth` (runner subs) | FIXED-TIME: mean depth + solves/score — THE instrument for node-reducing search changes |
| ⭐ **`depth_nps_bench.py --n 60 MAX_DEPTH=10 PRESET=LONG_FORMAT`** | **THE NODE JUDGE.** Median nodes/position on the game-representative quiet corpus; baseline **249,014**. ☠️ WAC's node column reversed SIGN against this on 4 of 6 configs (2026-09-03) — **a WAC-only node claim is UNVERIFIED.** |
| ⭐ **`depth_nps_bench.py --n 60 PRESET=LIGHTNING`** | **DEPTH reached at ~1s** (baseline median **12**, mean 12.2) — the cheapest honest proxy for what a change is worth in play. Use it whenever a node saving is claimed. |
| `_search_stability.py` | move-flip rate between two knob settings across N corpus FENs, one process per arm. 🔬 the SHIPPED engine flips its move on **20.8%** of quiet positions when only `ASPIRATION_DELTA` changes — this bounds what any single-position comparison can prove. ⚠️ `_move_change_arms.py` is a one-ply STATIC proxy and CANNOT see search behaviour. |
| **`_sts_reference.py --engine sf11\|sf15\|sf18 [--depth N \| --nodes N]`** | scores a REFERENCE engine on **our** `sts300.epd` with **our** scoring (reuses `sts_test.load_sts_epd`), so the SF gap can be re-measured after a change instead of quoting a stale anchor. ⚠️ `--depth` is NOT equal work (SF's d10 tree is far smaller and better ordered) — use `--nodes 249014` for the equal-cost reading. |
| ⭐ **`ENABLE_ORACLE_EVAL=1` + `ORACLE_CLASSICAL` + `ORACLE_SCALE` + env `ORACLE_ENGINE_PATH`** | **puts SF's static eval inside OUR search** (persistent UCI pipe, ~6,200 evals/sec, memoized by zobrist). Answers "what is an accurate eval worth in our engine?" — measured **SF11 +174 STS @d6, SF15c +219 @d8**. ⚠️ **FIXED-DEPTH ONLY**: NPS drops 14×, so any timed reading measures the pipe. ⚠️ `ORACLE_SCALE` re-expresses SF's value on our scale and exists ONLY for node-matching (our margins are absolute millipawns); carrying it into a fixed-depth run **degrades the eval badly** (−0.0177 at 200 vs −0.6681 raw). ⚠️ Quote `[oracle] fallback_pct=` with every result — in-check positions fall back to OUR eval. |
| ⭐ **`_ks_footprint_regret.py … CAND_KNOBS='K=V …' [CAND_NAME=] [DUMP=]`** | **THE deployment gate, now pointable at any arm pair** (override added 09-06; absent ⇒ the hardcoded KS arms run unchanged). "When it changes our move, is it better?" — SF18 win%-regret on the CHANGED-MOVE subset only. ★ This is the instrument that ranks evals; a raw **flip RATE cannot** (the SF15c oracle flips 63.5% of positions, a known non-improvement flips 40%, pure aspiration noise 20.8%). ⚠️ **Confirm the sign on the `_v2` cross-set** — the pawn taper read −0.3044 on set 1 and **+0.1675 on v2**. |
| **`_search_stability.py VS=1 [DUMP=<csv>] ARM='K=V'`** | **cross-arm** mode (added 09-06): diffs baseline vs each ARM at ONE delta and dumps the positions that flipped. The default mode instead diffs a config against ITSELF across aspiration widths. |
| **`_tail_term_stats.py A=<fens> B=<fens>`** | per-term **SE** (is a separation between two position sets real?) and **mean \|value\|** (what signed means hide). ★ Built because a KS "4.4× separation" between two 40-position tails was **−0.367 ± 0.517 = noise**, and because our per-piece placement components turned out to be **8-16× SF11's** — invisible in signed means since they cancel. |

## Game post-mortem (pasted PGN)
| script | what it gives |
|---|---|
| **`_pgn_walk.py SIDE=black DEPTH=12 PGN=<file> [KEY=VAL]`** | walks a **hand-pasted** game (UI/bot game, not a harness `games/<tag>/` dir — that's `annotate`), SF-evaluates every ply, and ranks plies by cp lost **on our own move** — the blunder list, not the eval curve. Dumps our static total + king_safety at the worst plies to start the ours/SF11/SF18 triangulation. |

## Collapse corpora
| script | what it gives |
|---|---|
| `collect_collapses.py` | pools every `selfplay/games/*/collapses.csv` into `ks_sets/collapse_dataset.csv`, tagged by family/seed |
| `classify_collapses.py` | tags each collapse ks_attack / ks_and_material / material / positional; supports cross-run vanish-attribution |
| **`collapse_term_attribution.py [CLASS=positional] [CONTROL=1]`** | **the strongest term tool.** Per-term excess over the average of SF11 **and** SF15.1, ranked, with a **quiet-position CONTROL set** so you can tell "this term is big" from "this term is big *here*" |
| **`sf11_collapse_gap.py --tags <family>`** | aggregate per-term gap over a family + worst-N FENs **ranked by win% error** |
| `_collapse_leverage.py [TAG=]` | **points forfeited** per collapse by phase / class / ply — leverage, not error size |
| `king_safety_probe.py` | worst offenders **plus a random sample** (avoids selecting on the error) |
| `bias_profile.py` | over-optimism bucketed by who is winning |

## Targeted questions
| script | question it answers |
|---|---|
| `eval_at_resolution.py` | is a deep-miseval an EVAL bug or a search artifact? |
| `detector_placement_proof.py` | is the `pieces`/placement gap detector-explainable? |
| `static_vs_search_triage.py` | statically fixable vs search's job |
| `blunder_probe.py` | classify a game blunder as EVAL vs SEARCH |
| `_sacrifice_loss_mine.py <games_dir> [PERSIST=4]` | mine `game.jsonl` for "ahead early then lost" — **no engine, zero CPU**. ⚠️ MUST use PERSIST; a per-ply snapshot overcounts 30× |
| `analyze_game.py` | per-move SF18 eval of a full game |
| ⭐ **`_draw_oracle.py [N=50] [CASES=a,b] [SEED=]`** | **does an endgame draw CLASSIFIER flag positions that are actually WON?** Mirrors every `is_practically_drawn` case in Python, generates random legal positions per material signature, checks each flagged one against the tablebase below. ★ Built 2026-09-13 and it immediately refuted 5 of v1's cases (`RB_vs_R` 28% · `R_vs_minor` 24-28% · `RN_vs_R` 22% · `wrongB_rookpawn` 10% false-positive), while `eq_only_minor` / `rookpawn_KPvK` / `B_vs_P` came back **0 for 62**. ⚠️ Random placement over-samples loose pieces and search resolves the tactical subset ⇒ rates are an **UPPER BOUND**. Caches every query (category + DTM) to the tracked fixture `ks_sets/tablebase_labels_draw.json`, so re-runs are free. Companions: `_draw_v2_verify.py` (rule fires on each case, negative controls untouched) · `_draw_short_mate_demo.py` (does search still play a mate when the rule scores the position 0?) · `_draw_deep_mate_finder.py` (constructs boxed-king geometry and asks the tablebase for deep wins). ★ **Revised same day:** `ARM=v1|v2` picks the classifier mirrored · `EDGE=1` forces one king onto a corner/corner-adjacent square, because **uniform sampling essentially never generates the rare boxed-king forced mates** that exist in KBvKB / KNvKN / KBvKN / KNNvK — a clean uniform run proves nothing about them · `SHORT=12` splits false positives by DTM: SHORT ones (search finds the mate itself) are reported and tolerated, **LONG ones are the gate**. Exit 1 = a LONG false positive exists. |
| ⭐ **Lichess tablebase API** (no script, no download) — `https://tablebase.lichess.ovh/standard?fen=<FEN, spaces as underscores>` | **GROUND TRUTH win/draw/loss for any position ≤7 pieces.** Returns `category` (`win`/`draw`/`loss`/`cursed-win`/`blessed-loss`), `dtz`, `dtm`, plus per-move rows. ★ The only true oracle we have for endgame CLASSIFIERS — it refuted two of `is_practically_drawn`'s cases on 2026-09-13 (K+R+B vs K+R read as a hard draw is a **win in 21**; K+R+N vs K+R a **win in 25**). ⚠️ We have **NO Syzygy data files on disk** — only SF's `syzygy/` *source* dirs; `python-chess` ships `chess.syzygy`/`chess.gaviota` but they need data we don't have. ☠️ Do NOT download the 5-man set: C: is at ~70 GB free and the repo already strains OneDrive sync. Be a good citizen — sample a few hundred positions, cache to a local file, don't sweep exhaustively. |

## Benches / tuning
| script | what it does |
|---|---|
| **`fit_bench_guarded.py`** | two-stage tuner: corpus MSE proposes, REAL benches dispose. ⚠️ **Always use this, never a raw corpus fit** — the corpus has repeatedly ranked candidates ANTI-correlated with move quality |
| `sf_bench_ceiling.py [SUITE=sts300.epd]` | reference ladder on an STS suite (native-ELF engines) |
| `sf_ceiling_win.py` | same for Windows-only .exe engines (SF1.1, SF17) via Windows python-chess |
| `_sts_theme_diff.py [tagA tagB]` | per-theme STS diff from the CSVs `sts_test.py` already wrote — **zero CPU**. ⚠️ per-theme deltas do NOT replicate; always check a second knob value |
| `refresh_bank_ours.py` | relabel `position_bank.csv` with current eval (2.7s) — **never fit against stale labels** |

---

## Subsystem maps (check these too — and check their DATE)
| doc | covers | ⚠️ |
|---|---|---|
| `passed-pawn-subsystem-map-2026-07-18.md` | all 22 passer eval channels, gates, double-counts | **has a 2026-08-01 delta header** — V3 shipped and changed which channels are live |
| `king-safety-subsystem-map-2026-07-23.md` | KS terms, modulators, clamp stack | pre-dates the `MOD_KS_REALIZ` ship |
| `search-architecture-map-2026-07-15.md` · `search-ordering-pruning-map-2026-07-14.md` | search structure | |
⚠️ **A subsystem map goes stale the moment a gate ships.** Before trusting one, check its date against the
baseline register and the commit log. When you ship a default, stamp the affected map in the SAME session —
a stale map is worse than none, because it reads as authoritative.

## Rendered readouts → [`../graphs/`](../graphs/README.md)
Self-contained HTML charts over measurements already taken, one file per readout
(`YYYY-MM-DD-<subject>.html`), plus a README carrying the authoring conventions. Use it when the finding
is a **shape** a table hides — the 2026-08-08 page shows a clamp sweep that is one flat smear inside its
own noise band, and a six-candidate triage in which exactly one clears the resolvability bar.
🚨 **Same epistemic status as reading source code:** admissible for FINDING a candidate, never for
EXPLAINING a measurement — a persuasive picture makes a wrong story more convincing. Ablate instead.
☠️ Never render the corpus objective; it is anti-correlated with Elo.
★ **Draw the noise band on every bench chart**, or it invites the over-reading it was built to prevent.

## Rules
1. **Search this file before writing a probe.** If something is close, extend it rather than fork it.
2. **Extend the canonical tool**, do not create a variant — `probe_fens.py` gained the SF15 columns and
   `--table`/win% rather than spawning a second probe.
3. **Carry a control set.** A term being large means nothing without a quiet-position baseline.
4. **Do not select positions by the error you are trying to explain** — filter on where points are lost, then
   sample across the range (see `king_safety_probe.py`, `_collapse_leverage.py`).
