# SESSION HANDOFF — 2026-10-10 (covers 10-07 → 10-10)

HANDOFF — this phase shipped **nothing**, on purpose, and changed how we measure. It found that the depth-target proxy was biased
against dynamic terms (threats read +12.6% by proxy, −7.8% at a REAL d10 re-search), then that even REAL d10 accuracy gains do not
become Elo (threats / PX / KFL: −3…−8% d10 error, all null-to-negative vs SF18). Neither cheap instrument resolves ±15-Elo items, so
GAMES decide, with a new owner ship rule (self-play + SF18 + a third opponent). Pawn endings turned out to be a SEARCH gap, not an
eval one. The retune is now prepared: plumbing started, a ~1M-position SF18 corpus built, the method and order agreed.
Pushed through ad463d9 on 10-09; **9 commits unpushed since** (HEAD a713e2d), no footers. **Nothing is running.**

## ⚠️ HONESTY FIRST — errors that reached reports or committed text
- I reported the 10-05 "threats CLOSED" as settled; it was the ROOT-Δ proxy (adds the root delta to a stored d10 score), biased
  against terms search resolves. Real re-search reversed it, and later PX and the path ladder too.
- I then over-read REAL d10 gains: "threats is the missing item" — SF18 said −5 ± 9 over 4 seeds. Third such miss after PX (−12) and
  KFL (−15). Memory `real-d10-accuracy-gains-did-not-become-elo`.
- First KFL static run was a silent no-op: knobs written after `pyrun <script>` are argv, not env (memory
  `pyrun-knobs-must-be-env-not-args`). Caught by byte-identical output.
- The path ladder was DEAD under any C1 table; a "KFL+PX+path" stack read byte-identical to KFL+PX. Fixed (3c22192).
- Queue #36 gated a stack (threats+mob-eg+space+rook files) that the combined re-search showed hurt middlegames; stopped 44 s in, relaunched
  as #37 with the clean pair.
- "Pawn endings 1.57× SF11" (10-07 bench) rested on a handful of val rows; on all rows we are BETTER (0.87×). Corrected in bench §3a.
- A blanket pawn-ending lift fitted great in-sample and failed the held-out K+P set; my "progress" guard mislabelled the owner's doubled
  g-pawns as a mobile majority.
- Corpus timing: I quoted "4 nights" (the pre-plan counted nights and multi-PV); real single-best labelling ≈ 200k/h (my first 420k/h
  read was the first minutes only). The labeller rewrote its whole file every 50 rows (O(n²)); now append-only.
- One labeller edit half-applied (second hunk landed, first didn't) — caught and fixed before any run.
- I read a 50-game Mediocre result as "negative"; the owner corrected it: ±54 Elo is inconclusive. Rule now needs ≥ 200 games/arm.
- Four parallel Fable research agents cost 11% of the weekly budget (memory `heavy-agent-calls-only-on-owner-say-so`).
- A watcher fired FAILED on harmless python-chess tracebacks from Mediocre's UCI id lines; re-armed with a narrower check.
**Prediction scorecard:** threats +5…+20 combined → +21 (borderline) but SF18 −5 (miss); pair ≥ threats → miss; regret instrument
expected to rank winners above nulls → 4/6 sign, inside its bar.
★ **The pattern:** removing a measurement BIAS (proxy → real re-search) did not make the measure PREDICTIVE; only games and controls
(harness null, identical-conditions re-run, byte-identity, held-out sets, a third opponent) settled anything.

Mode: execution, interactive.

## ▶️ FIRST ACTIONS IN THE NEW CHAT, in order
1. **Confirm nothing is running** (the last queue, #49, finished 10-10 04:57):
   `wsl.exe -e bash -lc "bash '/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine/selfplay/overnight_runner.sh' ps"`
   → expect `none running`. Anything else is UNEXPECTED: report it to the owner before launching anything. Detached queues do not
   appear in the UI; their logs are `E:\chess_data\q*.log` and `E:\chess_data\retune\q4*.log`.
2. **Re-verify the fingerprints** before any build work: `… overnight_runner.sh' wac fp_v1` → **250 / 35,310,778 / 3.784**;
   `wsl.exe -e bash -lc "V2_PRESET=shipped bash '…overnight_runner.sh' wac fp_v2"` → **254 / 50,622,239 / 4.029** (unchanged by this
   phase: every addition is byte-identical at 0; the 34,621-row static dump also matched the 10-05 dump exactly).
3. Then **step 0** of the next steps (assemble the corpus).

## Transfer documents, in reading order
1. **This file** ★★ and **`dev_notes/SESSION-HANDOFF-2026-10-07.md`** — its top sections "DECISIONS 2026-10-09" (ship rule, corpus
   inventory, retune method, stress-slice weighting, gate practice, POT↔retune nesting, dynamic terms, eval speed infra), "LONG-TERM
   GOAL", "ORDER RESHAPED 2026-10-08", "PLAN AGREED" + the RETUNE PLUMBING AUDIT. Those sections ARE the plan; they are not repeated here.
2. Memory `eval-v2-rebuild-state.md` (top block → this file).
3. **`dev_notes/research-2026-10-08/`** — four pre-plans (input, not decisions): `FINAL-RETUNE-PREPLAN.md` (plan, giants' tuners, 20 risks,
   data inventory) · `POT-DESIGN-PREPLAN.md` (+ the owner framing notes at its end: winnability ≠ SF winnable, complexity vs potential,
   per-phase halves, mg potential both ways) · `SEARCH-TRANSITION-PREPLAN.md` (33-item premise audit, negamax ladder, speed; + two VERIFIED
   defects at its end) · `EVAL-NUANCES-VS-GIANTS.md` (ranked top-10).
4. `dev_notes/TEXEL-C3-DETECTORS-DESIGN-2026-09-27.md` §20a (all 10-07…10-10 entries: dual fit, real re-search tables, games, instrument
   validation) and `dev_notes/REFERENCE-BENCH-LADDER.md` §3a (what makes up the endgame excess vs SF11).
5. `dev_notes/EVAL-V2-CURRENT-CONFIG.md` §1 (shipped block + the 10-09 winnability naming note), synced with `V2_PRESET=shipped` in
   search_engine.cpp and the runner's `V2=` line.
6. `dev_notes/DIAGNOSTICS-TOOLKIT.md` (10-07…10-09 rows) for the new tools below.

## Method memory — read before designing anything
★★★ [[real-d10-accuracy-gains-did-not-become-elo]] · ★★★ [[root-delta-depth-proxy-is-biased-against-dynamic-terms]] ·
★★ [[two-fair-instruments-can-genuinely-disagree]] (holds the owner's 10-09 ship rule) · ★★ [[long-term-goal-3000-single-core-hce-via-search-v2]] ·
[[the-remaining-eval-gap-is-endgame-structural]] (revised: pawn endings = search) · [[final-retune-needs-a-giant-diverse-corpus]] (corpus built) ·
[[ovd-is-the-owners-long-term-pressure-concept]] (10-09 structure + naming) · [[static-discrimination-matters-even-when-search-fixes-the-verdict]] ·
[[pyrun-knobs-must-be-env-not-args]] · [[long-background-tasks-must-be-purposeful-and-bounded]] · [[heavy-agent-calls-only-on-owner-say-so]] ·
[[later-test-vs-mediocre-for-an-absolute-rating]] (Mediocre has an `eval` command) · [[a-shared-baseline-correlates-every-arm-of-its-seed]] ·
[[unique-where-better-never-self-nerf]] · [[baseline-fingerprints-register]] · [[no-commit-footer]] · [[git-line-endings-check-the-stored-blob]].

## What this is
A non-negamax C++ HCE in `NN Engine/` (separate minimizer / maximizer / qSearch); the v2 eval (`eval_v2.cpp`) is Black-positive
millipawns, driven through Cython under WSL; v1 is the frozen control. Roadmap: strongest single-core HCE (**goal ~3000 CCRL**) via a
"search v2" that switches to **negamax** (owner 10-08: uniqueness lives in PLAY style, not code style), then the owner's own NN taught by it.

## 🚨 BASELINES (MAX_DEPTH=10 USE_OPENING_BOOK=0 PRESET=LONG_FORMAT)
| arm | WAC d10 |
|---|---|
| v1 (frozen control) | 250 / 35,310,778 / 3.784 |
| v2 shipped (`V2_PRESET=shipped`) | **254 / 50,622,239 / 4.029** |
Speed (pinned `wac_speed`, 10-07): ship best 415k NPS (spread ~12%). `EVAL_ARM=1` alone is a skeleton; quote NPS or time, never nodes alone.

## ▶️ THE MEASURED PICTURE (this phase; details in C3 §20a / bench §3a)
| arm vs ship | real d10 error | self-play @50k | SF18 @1000 | third opp. |
|---|---|---|---|---|
| threats (PCT 100 + hanging/king/pawn targets) | −7.8% | **+32 ± 9** | **−5 ± 9** (4 seeds) | — |
| threats + mobility eg-leg | −3.2% more | +29 ± 9 | +0 ± 13 | — |
| PX (passer cells + joint re-price) | −5.6% | +4.9 ± 9 | −12 ± 12 | — |
| KFL (king-to-pawn distance) | −2.8% (mg −9.6) | +10.6 ± 9 | −15 ± 12 | — |
| KFL + PST pair (10-06) | — | +15 | −3 ± 8 | Mediocre −42 ± 54 (50 g, inconclusive) |
- Instrument validation (d7 regret, 6k × 2 sets): tracks the SF18-game SIGN on 4/6 known arms, mostly inside its ~2-2.5 pp bar; inverts
  connected pawns. d10 MSE called all three nulls "gains". ⇒ both are HARM screens; games decide.
- Endgame excess vs SF11 (own-play rows): threats-led (−75% counterfactual), then king −18 %, initiative −16 %, placement −15 %,
  passers −12 %; ours better on scaling/imbalance/material. Pawn endings: SF11 d10 bias −1.0 vs ours −8.5 (ours d14 −5.8) = SEARCH.
- Mediocre v0.5 @ ~1 s/move: ship 43% (50 g, seed 111) and 45% (10-07) ≈ −35…−50 Elo.

## ▶️ NEXT STEPS (agreed order: corpus → structural retune → POT nested → search transition → search v2)
0. **Assemble the corpus** (`E:\chess_data\retune`, inventory in the 10-07 handoff "CORPUS BUILT"): dedupe; join the phase tag from
   `*_fens.csv`; tag QUIET vs UNQUIET (SF best move capture/check/promo, qsearch swing); band weights (emphasise ≈ −4…+4); hold out a split
   BY GAME; stress/variant slices as held-out understanding checks (train weight 0 / 0.25 / 0.5 — largest that doesn't hurt own-play).
   The ~3k slice-A rows without a d10 score are mate-scored labels (kept for the static fit, capped) — confirm.
1. **Structural retune** (hybrid): Texel-style gradient on the ~1M SF18 labels (static component, quiet rows) + depth target on the 137k
   rows (structural terms only); split mg/eg legs; L2 toward the ship; nested blocks; includes the winnability caps (start values in the
   handoff), CONN cells, Kaufman + `v2_piece_value` (never `values[]`), KFL/PST/KPROT/PX/passers/placement/mobility (static only).
   Gate per part or bundle: screening on self-play 50k; ship on the owner's rule (self-play ≥ +2σ, SF18 > −1σ, Mediocre ≥ 200 g not
   negative); a timed LIGHTNING confirmation at the end.
2. **POT nested:** design on the RESIDUAL of the retuned eval (complexity = magnitude, potential = direction, per phase; `passer_potential`,
   form D); fit on top with the rest frozen; light joint refit (known terms shifting a lot = POT taking over).
3. **Search transition:** SF11 pawn-ending rules + the two verified defects (null-move guard counts pawns; qsearch horizon before
   in-check) · corr hist · threats inside search (first: self-play @250k) · dynamic-lane plumbing + nested fit (threats, space, rook files,
   reach/longdiag/latent) · lazy eval with the pruning work.
4. **Search v2:** premise audit of every closed search item (all swept under v1) · negamax core with the verification ladder · speed
   (profile-first; pin-aware movegen; pawn hash + incremental material/PST — exact, ≤ ~11% since eval ≈ 11.5% of node cost) · node-efficiency
   instrument · Mediocre static-eval comparison. Then re-measure vs Mediocre / SF18; then the owner's NN.

## 🎯 THE REFRAMES (load-bearing)
- A real d10 re-search removes the proxy's dynamic-term BIAS but is still CALIBRATION, not move choice ⇒ screens harm, never ships.
- SF18 measures the DEFENSIVE half (holes a strong opponent exploits); self-play mostly the ATTACKING half against our own family; a
  third opponent tests whether the attack generalises.
- Gains don't add: KFL+PX, space+rook files cancelled when stacked ⇒ joint fits, then gate the stack.
- Endgame gaps split by CAUSE: structural → retune; search-caused → transition. No static eval (ours, SF11, SF15c) sees tempo/race/
  opposition cases; SF11 wins pawn endings by search (no null move / no shallow pruning with only pawns; passer-push extension).
- Our "winnability" = SF's `scale_factor()` only; SF12+'s `winnable()` = that + additive complexity (→ POT's magnitude half).

## ★★★ OWNER CALLS (this phase)
Winnability STANDALONE (name kept, naming note in the config doc) · POT = complexity + potential, per phase, mg potential pushes either way,
designed LAST on the residual and nested into the retune · ship rule with a third opponent (≥ 200 games/arm) · small gains may add up ⇒
the winnability caps go to the retune, not a standalone gate · fit mg/eg legs separately everywhere (KS keeps its own phase design) ·
corpus: diverse incl. "whacky" positions, understanding over memorisation but regular-play strength is the point · goal ~3000 single-core;
negamax in search v2; re-audit every closed search item against its premise · eval speed infra belongs to search v2 · heavy agent
fan-outs only on the owner's say-so (Opus 5.5, not Fable) · background-task max raised to 12 h with the 30-min default KEPT (the limit
exists to stop runaway token use on broken tasks) · push only on the owner's say-so.

## ⚠️ WHAT IS ALREADY THERE (built this phase; all byte-identical at 0)
- **Knobs:** `POT_V2_WIN_OCBX/_PC, ROOKE, QNOQ/_MINOR, LONEMINOR` (winnability material-class CAPS, SF15.1 forms; screened −1.2% static
  and depth, no harm) · `CONN_V2` + `CONN_V2_FILE` (102 connected-pawn cells, free mg/eg; table tracks the ship ≈ 0.7 mp) · the passer path
  ladder now works under C1 (`passer_path_kw`).
- **Probes:** `v2_feature_counts/theta` now **337 per side** (CONN cells 235-336; pawn_struct closure passes, max 4.5 mp) · `pawn_masks(...)["potential"]`
  = `passer_potential` (classic candidate passer; oracle 0/4000, mirror-clean).
- **Tools:** `_triangulate_sf11.py MODE=eg / egshare (DUMP=)` · `_eg_features.py` · `_pawn_ending_lift.py` (forms A-D) · `_pe_search_bias.py` ·
  `_static_vs_search_cases.py` (+ `ks_sets/pe_cases_fens.txt`, the 9 owner-reviewed positions) · `_passer_potential_oracle.py` ·
  `_revival_screen.py MODE=dual / dualread (REF=, VALOUT=)` · `_px_depth_fit.py` (by-game split, SCALE nuisance, ARMS=) ·
  `_build_game_regret_set.py` (MAX_GAMES, EG_QUOTA, EXCLUDE_EXTRA, SAMPLE_ONLY) · `_build_regret_set.py` (append-only, resumable).
- **Data:** `E:\chess_data\retune` (960k SF18 labels + 136,997 d10 rows on the current ship) · real-re-search arms
  `diagnostics/ks_sets/dual_val_*_d10_s*of4.csv` (val 4,952 rows; ship control `dual_val_shipctl`) · `E:\chess_data\texel\revival\`
  (dumps, `kfl_depth*.txt`, `mob_depth_c1*.txt`, `wc_*.csv`) · `E:\chess_data\texel\px_depth1008_*` (passer refit on the 10-04 base, d10 −2.9%).
- **Queues:** `selfplay/_queue_*.sh` #32-#49 (guarded: fingerprint → symmetry → work) · `selfplay/_launch_detached.sh` (`--waits` to chain).

## STATE
HEAD a713e2d on NN-ENgine (origin at ad463d9; 9 unpushed). Tree clean apart from the ChessUI submodule and old untracked files.
MEMORY.md ≈ 20.3 KB (near its useful size — prune before adding much). UI: `V2_PRESET=shipped PRESET=LONG_FORMAT python ChessUI/chess_ui_v2.py`.

## DISCIPLINES
★★★★ Games decide; fixed-depth instruments screen HARM only · real re-search against an IDENTICAL-CONDITIONS ship control (stored passes
don't reproduce across sessions: 146/300) · byte-identical-to-control = silent fallback · guard every queue (fingerprint, symmetry,
closure; require the checked lines) · register predictions first; never call a direction on partial data · ≥ 200 games for any third-opponent read.
★★★ Ops: ≤ 4 engines; never rebuild or edit a script a running queue uses (Stockfish-only labelling is safe to build beside); engine
knobs as ENV before `bash '<runner>'`; detached queues + one tracked wait sized ≈ 1.5× expected runtime with done/fail/stall exits;
bulk output on E:; git add explicit paths; commit by feature, no footer; owner games 9pm-midnight ⇒ node-limited/fixed-depth only.

## ★ LOAD-BEARING OWNER CONTRIBUTIONS (this phase)
"Could it be search not supporting it?" → the node-budget / stand-pat / self-exploitation hypotheses and threats moved to the transition ·
"keep what's clean in one phase, improve the other" → per-phase leg variants (mobility eg-only kept the eg gain, dropped the mg cost) and the
split-leg retune · "SF18 vs self-play: plugging our own weakness counts" → the third-opponent ship rule · "50 games isn't statistically
meaningful" → the ≥ 200-game rule · "use SF11's breakdown, be clear what each term contains" → the closure-checked term map and the
discovery that pawn endings are search · "winnability is its own thing; POT = potential, tempo, complexity" → the three-quantity split ·
"POT last, fit the gaps" → nested POT · "the guard exists for runaway tokens" → the bounded-task rule · "maximize the night / it's pausable"
→ the ~1M corpus in one evening.
