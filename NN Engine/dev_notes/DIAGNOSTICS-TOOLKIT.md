# Diagnostics toolkit — CHECK HERE BEFORE WRITING A NEW PROBE

`diagnostics/` holds ~200 scripts. Nearly every question we ask has already been answered by one of them,
and rebuilding wastes time, fragments conventions, and produces weaker versions (a rebuilt probe usually
lacks the *control set* that makes the original trustworthy). **Search this file first.**

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
