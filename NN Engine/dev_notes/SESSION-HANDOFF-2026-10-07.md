# HANDOFF — 2026-10-07 (session 10-04 → 10-07)

This phase shipped one eval win (connected pawns), ran the **revival screen** over every eval term never fitted on the depth
target, gated five candidates through both instruments, re-anchored the external judge, and produced the first full
**bench snapshot** — which says the eval is close to done and the remaining gap is **endgame-structural (pawn endings first)**.

**Ship:** connected pawns (`PS_V2_CONN_MAG=21 PS_V2_SUPPORT=99 PS_V2_EG_RATIO=101`), closed 09-12 as "harmful at every magnitude"
on a bad instrument — the term was right, ~5× too big. SF18 +20 ± 15 · self-play +12.5 ± 9 ⇒ **≈ +14.5 ± 7.7**; generalises
on held-out K+pawns stress positions (paired −0.41pp, t −2.8). C3 §19e-19f.
**Gated, NOT shipped:** Kaufman depth re-fit (≈ +7 ± 8), mobility cells (SF 0), king protector C3-c (SF +16 / self-play −2 —
instruments split), king flank C3-b + PST depth re-fit as a pair (SF −3 ± 8 over 6 seeds / self-play +15 — split). All →
final joint retune. **Closed on fair tests:** threats (per-leg, joint −0.01%: tactical ⇒ search's), material taper, space,
long diagonal, reach, latent, placement bundle. Winnability ship knobs CONFIRMED on the depth target. C3 §20-20e.
**Judge re-anchored: SF18 @1000** (48.6%; @800 had drifted to 57-59%). **Mediocre v0.5 at ~1 s/move: v1 ≈ 15% (07-16) → v2 45%**
(≈ +250 Elo on the same anchor; v2 ≈ 2265-2340 CCRL-40/40-anchored ±100). STS300 @ equal nodes **1838** (09-22: 1689).
70 commits unpushed, no footers. **Nothing is running** (last job finished 10:14 on 10-07).

## ⚠️ READ THE HONESTY FIRST — errors that reached reports, chat or commits (all caught; none reached a ship)
- **A scorer closure** in `_conn_depth_fit.py` scored every arm with the BASELINE model (all read ±0.00%). Caught before any
  verdict; each arm now scores with its own model + a guard.
- **First revival-screen numbers were inflated** and I reported them before the controls: no global-SCALE nuisance (blocks won
  by stretching — "KAUF −5.0%"), the optimizer stopping at its start (exact −0.00% reads), and a FEN-hash val split leaking
  same-game rows ("PLACE −0.54%" → 0.00 by game). Fixed in the tool; INSTRUMENT-MAP 2026-10-04/07 #5.
- **The connected-pawn ship commit flipped EVAL-V2-CURRENT-CONFIG.md's line endings (870-line diff)** — I applied the handoff's
  "autocrlf=false" rule to an LF-stored file. Amended before push. ☠️ Check the STORED blob first (memory
  `git-line-endings-check-the-stored-blob`): CRLF blob ⇒ `git -c core.autocrlf=false add`; LF blob ⇒ plain add; a mis-add
  needs `git add --renormalize`. Read `git show --stat HEAD` after every commit.
- **The 10-04 handoff said the queen imbalance was "fixed by Kaufman"** — it only shrank (depth −5.7 → −4.7pp). Corrected in memory.
- **I raised "v2 memorises standard chess" as a red flag** (v1 2× better on 960/variant positions). Refuted the same hour: v1's
  edge is its static capture-gains on tactically unresolved random-walk positions (v1 without it: 725 vs v2 624).
- **Queue #28 died at exactly 30 minutes** after a VS Code reload (tracked background tasks are capped); 222 games lost.
  Relaunched detached. ☠️ Launch long queues with `selfplay/_launch_detached.sh` (memory `nohup-orphans-…` updated).
- **WSL `/tmp` is wiped when the distro idles** — the queue #20 dumps vanished; my guard caught it. Dumps live on `E:/` now.
- **The feature-pass closure refuses under the shipped connected knob** (flag 4) — queue #22 dropped three arms on tooling,
  not tables. Closures need `PS_V2_CONN_MAG=0` (memory `feature-pass-closure-refuses-under-shipped-connected`).
- I wrote "I'll take silence as a yes" for an overnight launch — retracted; never treat silence as approval.
- I briefly called queue #30 a failure because it finished in 45 s — false alarm (static evals are fast).

**Prediction scorecard this phase:** connected gate (+0…8) MISS high on both instruments · K+P stress (±0.3) MISS better ·
revival: MOB/PST/KPROT beat ranges, KAUF far beyond, PLACE missed (null), joint ALL HIT · threats per-leg (−0.1…−0.4) MISS low ·
winnability ship knobs MISS (no gain) · @1000 anchor (53-55%) MISS low · pair on SF (+8…+25) MISS · KPROT 4-seed HIT ·
endgame stretch (+5…15%) MISS (0) · search less biased than static HIT.
★ **The pattern:** fit size does not predict game size; a shared baseline correlates every arm of its seed; two fair
instruments can genuinely disagree; static tests on unquiet positions measure static tactics. Every correction came from a
control (scale nuisance, by-game split, second seed, second instrument, ablation of v1's capgains).

**Mode:** execution, interactive.

## ▶️ FIRST ACTIONS IN THE NEW CHAT, in order
1. **Confirm nothing is running** (nothing should be):
   `wsl.exe -e bash -lc "bash '/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine/selfplay/overnight_runner.sh' ps"`
   Expect `none running`. If anything shows, it is UNEXPECTED — report it to the owner before launching anything.
2. **Re-verify the fingerprints** before any build work (MAX_DEPTH=10 USE_OPENING_BOOK=0 PRESET=LONG_FORMAT):
   `… overnight_runner.sh' wac fp_v1` → **250 / 35,310,778 / 3.784**
   `wsl.exe -e bash -lc "V2_PRESET=shipped bash '…overnight_runner.sh' wac fp_v2"` → **254 / 50,622,239 / 4.029**
3. Then the **POT design discussion with the owner — endgame side first** (Next steps §0).

## Transfer documents, in reading order
- ★★ `dev_notes/SESSION-HANDOFF-2026-10-07.md` (this file) — the anchor.
- ★ Memory `eval-v2-rebuild-state.md` — top block = 10-07.
- ★★ `dev_notes/REFERENCE-BENCH-LADDER.md` → "2026-10-07 — BENCH SNAPSHOT": every dated number (static ladder × 5 corpora,
  STS eq-nodes history, gap strata, endgame types, judge re-anchor, Mediocre v1 → v2). Raw dumps `E:/chess_data/bench1007/`.
- `dev_notes/TEXEL-C3-DETECTORS-DESIGN-2026-09-27.md` (the C3 doc): §19e-19g connected ship + K+P stress + depth refresh ·
  §20-20e revival screen, threats per-leg, winnability re-fit, gates, self-play, confirmation, re-anchor · §21-21a bench +
  endgame inspection.
- `dev_notes/SESSION-HANDOFF-2026-10-06.md` (§8 = the 10-07 update) and `SESSION-HANDOFF-2026-10-04.md` (prior phase).
- POT: memory `ovd-is-the-owners-long-term-pressure-concept` (owner's concept + 10-05 decisions + design inputs),
  `dev_notes/POT-TYPE-DEFINITIONS-2026-09-30.md`, `POT-TRANSFORMATION-KNOWLEDGE-2026-09-30.md`.
- `dev_notes/OWNER-GAMES-ANALYSIS-2026-10-01.md` (incl. the 10-04 UI games: 14.Qd3 persistent to d12; the drawn K+P loss).
- `dev_notes/EVAL-V2-CURRENT-CONFIG.md` §1 — the shipped block (mirrored in the `V2_PRESET=shipped` block of
  search_engine.cpp and the runner's `V2=` line — keep all three in sync; fitted tables compiled in `ship_tables_v2.h`).
- `dev_notes/INSTRUMENT-MAP.md` (end section "2026-10-04/07"), `DIAGNOSTICS-TOOLKIT.md` (table "Added 2026-10-04/07"),
  `EVAL-V2-PARKED-REGISTER.md` ("REVIVAL ROUND 2026-10-05/07").

## Method memory — read before designing anything
★★ [[the-remaining-eval-gap-is-endgame-structural]] · ★★ [[static-discrimination-matters-even-when-search-fixes-the-verdict]] ·
★★ [[two-fair-instruments-can-genuinely-disagree]] · [[a-shared-baseline-correlates-every-arm-of-its-seed]] ·
★ [[the-sf18-gauntlet-anchor-drifted-too-weak]] (judge now **SF18 @1000**) · [[see-one-subsystem-through-before-switching]] ·
[[fit-data-depth-must-match-play-depth]] · [[final-retune-needs-a-giant-diverse-corpus]] (stress slices: train + held-out) ·
[[gate-new-candidates-on-fresh-seeds-not-ship-seeds]] · [[unattended-jobs-must-have-bounded-memory]] ·
[[keep-generated-output-out-of-vscode-indexing]] · [[git-line-endings-check-the-stored-blob]] ·
[[feature-pass-closure-refuses-under-shipped-connected]] · [[nohup-orphans-a-games-job-and-relaunch-double-cores-oom]] (UPDATED:
use the detached launcher) · [[later-test-vs-mediocre-for-an-absolute-rating]] · [[unique-where-better-never-self-nerf]] ·
[[baseline-fingerprints-register]] · [[dispatcher-prompt-free-wrapper]] · [[no-commit-footer]] · [[never-edit-the-runner-while-a-job-is-in-flight]].

## What this is
A non-negamax C++ HCE in `NN Engine/` (separate minimizer / maximizer / qSearch). The v2 eval (`eval_v2.cpp`) is Black-positive,
in millipawns, driven through Cython (`ChessAI.pyx`) under WSL; v1 is the frozen control. Roadmap: the strongest
single-threaded HCE → the owner's own NN taught by it (the eval carries over informationally, as the teacher).

## 🚨 BASELINES — reverify every build
| arm | WAC d10 |
|---|---|
| v1 (frozen control) | 250 / 35,310,778 / 3.784 |
| **v2 shipped** (`V2_PRESET=shipped`) | **254 / 50,622,239 / 4.029** |
Old configs by env override: the 10-03 ship = `PS_V2_CONN_MAG=0` (255 / 47,218,480); the 10-01 ship = the `OLD=` list in
`selfplay/_queue_bench_ladder.sh` (+ `PS_V2_CONN_MAG=0`). ☠️ `EVAL_ARM=1` alone is a skeleton. Quote NPS/time, never nodes alone.

## ▶️ THE MEASURED PICTURE (details: REFERENCE-BENCH-LADDER 2026-10-07 snapshot)
- **Static win% MSE vs SF18 search:** own-play v2 171 · SF11 151 · v1 189 · SF15.1c 192 · SF18s 62; diverse v2 124 · SF11 95 ·
  v1 239. K+P stress v2 904 BEATS SF11 1170 and v1 1210. Ships since 09-26 (≈ +45 Elo) left static MSE ~flat — depth fits.
- **Where the gap to SF11 lives:** we MATCH/BEAT SF11 in level positions (<1 pawn) and middlegame-leaning ones; the deficit is
  ENDGAMES (46% of rows → 114% of the excess) and decisive positions. Endgame types (static): pawn endings 1.57× SF11, rook+minor
  1.42×, queen 1.39×; pure rook / pure minor ≈ SF11. Side ahead UNDER-rated −5.7 win% pts (SF11 −3.4).
- **At depth (our d10 search):** bias −1.0 overall (search fixes most) — NOT a magnitude issue (eg-stretch fits 0); eg-leg
  re-pricing −3.4% (→ final retune); PERSISTS: **pawn endings −9.3**, pure minor −4.5, pure rook −4.4.
- **STS300 @ equal nodes:** v2 1838 · v1 1752 · (SF11 2374). **Mediocre v0.5 @ ~1 s:** v2 45% (v1 ≈ 15% in July).

## 🎯 LONG-TERM GOAL (owner, 2026-10-08): ~3000 CCRL, SINGLE CORE, hand-written eval
Today ≈ 2265-2340 (±100, Mediocre anchor). Gap ≈ 700 Elo, mostly SEARCH + SPEED (~450k NPS vs 1.5-3M in 3000-class HCEs).
Path = **"search v2"** with v2's method (reference audit → instruments first → one feature at a time → fair gates; SPSA for search
params). **The search switches to NEGAMAX** (owner: the min/max split came from translating by hand; "no one is looking at code
style — keep PLAY style unique while gaining strength"). Estimates: eval arc done ~2350-2450 · first transition ~2500 · speed
rebuild + reductions ~2700 · mature search + synergy ~3000 — months. Memory `long-term-goal-3000-single-core-hce-via-search-v2`.

## ▶️ ORDER RESHAPED 2026-10-08 (owner) — supersedes the plan below where they differ
Threats: real d10 −7.8%, self-play +32 ± 9, but SF18 @1000 **−5 ± 9 over 4 seeds** (2,000 paired games; queue #38 stopped early
at 4 seeds by owner decision). Dynamic terms interact with search (pruning reads the static eval; corr hist may absorb part
of what they add) ⇒ they cannot be priced before the search they interact with is settled. Order:
1. finish the known STRUCTURAL elements (KFL, passers — queue #39) · 2. POT design (mg = structural, design + fit now; eg
races/tempo/entry = design now, fit later) · 3. STRUCTURAL retune (split mg/eg legs; depth target + static component) ·
4. **SEARCH TRANSITION**: SF11 pawn-ending rules · corr hist · **threats + its search interaction** (qsearch / pruning /
side to move; first test: self-play at 250k nodes to see whether its gain fades with depth) · maybe NPS · 5. DYNAMIC-lane
retune on the new search (threats, mobility, space, rook files, eg-POT dynamic parts) · 6. rest of the search arc.

## ▶️ PLAN AGREED 2026-10-07 EVENING (owner) — supersedes the step order below where they differ
Evidence: REFERENCE-BENCH-LADDER §3a. The static endgame excess vs SF11 is threats-led (−75% counterfactual), then KFL/king,
initiative, placement, passers; pawn endings are a SEARCH gap (SF11 d10 −1.0 vs ours −8.5), not an eval one.
1. **Plug the known gaps** (threats, KFL, passers, placement): judged on STATIC/ordering accuracy + "no harm at depth/games";
   threats must also pay its NPS. SF11 = how far plugging can go.
2. **POT design — midgame AND endgame** (endgame POT = dynamic potential: races/square rule, king entry, reserve tempi, key
   squares; winnability stays the convertibility scale), only after 1; each feature must add measurably to OUR tuned eval,
   be cheap, and have its own owner. Built: `passer_potential` (probe-only) and lift form D (candidate).
3. **Giant retune, HYBRID:** Texel-style fit on SF18 labels (depth target + static component, eg weighted, K+8P stress
   train/held-out) for the bulk; GAMES gate every part on both instruments; SPSA only for a few scalar knobs. Then close the eval arc.
   ★ **Owner (later 10-07): fit EVERY term's mg and eg legs separately** (no averaging across phases; fixed-ratio knobs such as
   `MOB_V2_EG_PCT` / `PS_V2_EG_RATIO` become two free legs; mobility/king/pawn terms can grow into the endgame). KS keeps its own
   phase design (`KS_V2_EG_PCT`, KS-B legs) — nudged, not doubled. Clean = same code, numbers in tables, zero NPS cost.
   ☠️ **TWO LANES** (memory `root-delta-depth-proxy-is-biased-against-dynamic-terms`): STRUCTURAL terms (pawns, PST, Kaufman,
   passers…) on the depth target + static component; DYNAMIC terms (threats, mobility…) on the static target, each part
   CONFIRMED by a REAL d10 re-search (`MODE=dual VALOUT` → depth pass → `MODE=dualread` vs an identical-conditions ship re-run)
   before any game gate. Per-phase picks made on the val rows are optimistic — games are the independent check.
   🔧 **RETUNE PLUMBING AUDIT (10-08):** per-leg TABLE loaders exist for PST, Kaufman, C1 (mobility · pawn structure ·
   passers incl. king distance · placement's 9 sub-terms — missing cells keep LIVE θ, verified in the loader), KS-B, KFL, KPROT,
   PX. KNOB-ONLY (need a loader before the split-leg retune): **threats** (compiled per-victim tables, one PCT, bool legs),
   **space**, **rook files**, **connected pawns** (MAG + fixed `PS_V2_EG_RATIO`; computed BEFORE the C1 branch, so it survives
   any C1 table), **reach / longdiag / latent**. KS attack knobs + winnability stay scalars. Build these after the running games
   (never rebuild while a job runs), each byte-identical at its default.
4. **Search arc** opens with SF11's pawn-ending rules (no null move / no shallow pruning when the mover has only pawns; passed-
   pawn push extension — SF11 search.cpp:846/998/1079; ours: null-move guard counts pawns, `isUnsafeForNullMovePruning`).

## ▶️ NEXT STEPS (owner's order: finish the eval → POT → giant joint retune → search)
0. **POT design session with the owner — ENDGAME SIDE FIRST** (data-backed: pawn endings −9.3 at d10). Design inputs on record:
   pawn-ending conversion knowledge (outside / protected passers, king activity, opposition, the square rule — PX cell 50 is
   built at 0); "the leader can / cannot create a passer" as a WINNABILITY input (the shipped scale is inert at ≥ 2 leader
   pawns — why the 10-04 K+P draw read +0.8); drawishness in pure minor / pure rook endings. Owner's rule: winnability takes
   over from POT in the endgame; POT = feeders/modulators decided from STRUCTURE, never rescoring. Screen with
   `_eg_leg_inspect.py`-style endgame depth rows + the K+P stress set; gate on SF18 @1000 + self-play.
1. **POT middlegame potential** (owner's concept: cheap GM-like intuition of possible transformations while a subsystem is
   unresolved; recedes as it becomes kinetic). The 10-01 type screen was null as ADDED scores — test the FEEDER form.
2. **A sibling-ordering (discrimination) metric** beside win% MSE (owner principle 10-07), SF11 as the achievability control.
3. **The giant-corpus joint retune**: everything jointly incl. eg legs (−3.4%), Kaufman + `v2_piece_value` (never `values[]`),
   KFL+PST, KPROT, mobility, rook files, winnability +PASSED, PX, structure (isolated/backward separated cleanly); fit the
   DEPTH target AND a static/ordering component, endgames weighted; K+8P / variant / odds stress slices as TRAIN + a held-out
   check; gate every part.
4. **Search arc:** correction history first (eval→search bridge), better qsearch (to replace v1's capgains at speed),
   threat-aware ordering (threats live here now), the pruning/margin re-sweep on the final eval (static discrimination matters),
   the owner's games as the test set (14.Qd3, the LIGHTNING K+P loss). Then NPS (movegen 41% of node cost → pin-aware legal
   movegen); the owner will supply **Mediocre's source** for that phase. Then v2 vs v1 / Mediocre re-measures. Then the owner's NN.

## 🎯 THE REFRAMES (load-bearing)
- Fit on the DEPTH target (SF18 d14 vs our d10) — but the final retune ALSO needs static discrimination (pruning sees statics).
- The screen RANKS, games DECIDE: fit size ≠ game size. Two fair instruments; ship only on both (~2σ combined).
- Read POOLED seeds only (a shared baseline correlates arms). Re-anchor the judge after ships (now SF18 @1000).
- Static tests on UNQUIET positions measure static tactics — judge variants/odds with our d10 SEARCH vs SF18.
- Form before fit (multiplicative winnability, per-leg threats); a "wall" is usually a bad instrument or form (connected).

## ★★★ OWNER CALLS (this phase)
- **Finish one subsystem completely before switching** (eval first; walls are rarely final — reviving eval gave 100+ Elo).
- **Order:** POT designed and built BEFORE the giant retune (so the retune prices it); search after, entered via corr hist.
- **Stress positions** (K+8P etc.) for understanding vs memorisation — train slice AND held-out check, never the same rows.
- **Threats:** retried the Kaufman way at the owner's request → closed fairly; it belongs to search now.
- **Winnability treated as a known term now; the full POT design after the known terms** (POT = leftover recoverer).
- **Static discrimination matters** for in-search decisions even when search fixes the verdict.
- Stages use the eval's OWN phase (phase256: full MG / MG-leaning / EG-leaning / full EG). Always win% error, never raw cp.
- Ship rule unchanged: ~2σ combined across both fair instruments; re-anchor and confirm after each ship. Push only on say-so.
- Keep generated output out of VS Code / Pylance indexing (bulk data → `E:/`).

## ⚠️ WHAT IS ALREADY THERE
- **Knobs** (0 = byte-identical): connected `PS_V2_CONN_MAG/SUPPORT/EG_RATIO` (SHIPPED); `KPROT_V2`/`KFL_V2` + `_FILE` (silently
  OFF without a file); `THREAT_V2_PCT` + 5 leg flags; `SPACE_V2_MAG`, `LONGDIAG/REACH/LATENT_V2_PCT`, `ROOKFILE_V2_OPEN/SEMI`,
  `EVAL_V2_PAWN_MG`, `TEMPO_V2_*` (all measured ~null); `PST_V2_FILE` / `PST_V2_DUMP`; plus the 10-04 list (POT_V2_WIN*,
  KAUF_V2_*, KSB_V2*, MCL_V2_*, PX_V2*, C1_V2_*, WIN_V2*).
- **Candidate tables (not shipped):** `E:/chess_data/texel/revival/{kauf,mob_depth_c1,pst,kprot,kfl}_depth.txt`.
- **Depth passes on the CURRENT ship:** `diagnostics/ks_sets/fitC_{mg,eg}_ours1004_d10_s*of4.csv` (use `OURS=ours1004`).
- **Bench corpora:** `diagnostics/ks_sets/bench_{variant,odds,kp}.csv`, `kp_stress_*`; dumps `E:/chess_data/bench1007/`.
- **Tools:** DIAGNOSTICS-TOOLKIT "Added 2026-10-04/07" (`_revival_screen.py`, `_win_depth_fit.py`, `_conn_depth_fit.py`,
  `_kp_stress_check.py`, `_joint_depth_preview.py`, `_gap_strata.py`, `_endgame_types.py`, `_eg_leg_inspect.py`,
  `_bench_type_corpora.py`, `_reference_ceiling.py DUMP=`); queues `selfplay/_queue_*.sh` (#17-#31); launcher
  `selfplay/_launch_detached.sh <queue> <log> [--waits]`.
- **Gauntlet command (new judge):** `overnight_runner.sh gauntlet 500 1000 4 <tag> <seed> V2_PRESET=shipped <knobs>`.

## STATE
HEAD on NN-ENgine (this handoff's commit), **70+ commits unpushed**. Tree clean apart from the ChessUI submodule and old
untracked files. MEMORY.md ≈ 19.1 KB. **Nothing running.**
UI: `V2_PRESET=shipped PRESET=LIGHTNING python ChessUI/chess_ui_v2.py` (any preset; play only with nothing else running).

## DISCIPLINES
- ★★★★ A silent tool reads as a result: guard every queue (fingerprint, closure, symmetry, row counts) and REQUIRE the checked
  lines to be present and live (rows_live > 0).
- ★★★★ Two fair instruments (SF18 @1000 + self-play), fresh seeds + fresh baselines, ≥ 2 seeds pooled, register predictions
  first, never call a direction on partial data.
- ★★★★ Bounded memory unattended (sample RSS twice; CHUNK loops). Long queues DETACHED via the launcher; watch the LOG.
- ★★★ Ops: never edit a script a running queue uses; git add explicit paths + check the stored blob's line endings; ≤ 4 engine
  processes; owner plays 9pm-midnight (node-limited work only then); bulk output to `E:/`, never into the repo.
- ★ Commit by feature, no footer; push only on the owner's say-so.

## ★ LOAD-BEARING OWNER CONTRIBUTIONS (this phase)
- "Connected pawns / passers… Texel tune" + reading the pawn gap → the connected revival (+14.5).
- K+8P stress positions → the understanding-vs-memorisation test (connected generalises).
- "See one thing through" → eval-first order; "rethink threats like Kaufman" → the fair per-leg test that closed it.
- POT's middlegame concept restated (unresolved → transformation potential; winnability takes over in the endgame).
- "Better to be sure" → the re-anchor that showed the pair's SF read is genuinely ~0.
- The bench request + "break it down by types / use the eval's own phases / win% not cp" → the endgame finding.
- "Which endgames? under- or over-confident? inspect the eg pair" → the endgame inspection (pawn endings).
- "Static discrimination matters for in-search decisions" → the retune's static component + ordering metric.
- Mediocre as an absolute anchor (and the CCRL facts) → v1 ≈ 15% → v2 45%.
- VS Code / Pylance indexing → `python.analysis.exclude` + bulk data on `E:/`.
