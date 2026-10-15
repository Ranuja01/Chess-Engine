# SESSION HANDOFF — 2026-10-15 (covers 10-10 → 10-15)

HANDOFF — this phase **shipped twice** after a week of nothing, then ran the structural retune to exhaustion and prepared POT.
- **Gen 2 (10-11):** the structural retune — one hybrid fit (static win% on quiet SF18 labels + the depth target on our d10
  search) of C1, KS-B, KFL, KPROT, PX, CONN, Kaufman, PST. Self-play +56.6 · SF18 +22 · Mediocre +10.
- **Gen 3 (10-12):** round 2 on the gen-2 base with explicit per-phase piece values (`V2_VAL_*`, PST means pinned). Self-play
  +22.8 · SF18 +9 · Mediocre +7.3 pts. **Total ≈ +25-30 Elo external** over the 10-04 engine (≈ +80 self-play).
- **Round 3 (R3A) not shipped:** self-play +21 but SF18 4-seed pooled ≈ −8 ± 10 — the halves read +3 / +4, so the split is a
  self-play-only gain ⇒ the structural retune is DONE (+22 → +9 → ~0). SPSA on the scalars: only `KS_V2_HALF` replicates.
- **POT is next:** the design brief (residual map, direction-vs-magnitude probe, endgame-signature appendix, data appendix) is
  ready, plus an endgame-enriched set and the owner's trajectory set. The owner's design discussion opens the next chat.
~45 commits since 10-10, none pushed (push only on say-so), no footers. **Nothing is running** (machine idle since 10-15 08:35).

## ⚠️ HONESTY FIRST — errors this phase (all caught before a ship; most by closure / byte-identity / ablation)
- The first game-map walk wrote its output only at the end and died at the 60-min task cap — an hour lost; now incremental,
  resumable, layout-first.
- My export wrote the C1 king-distance cells with `%.0f`; they are FRACTIONAL ⇒ endgame closure off by up to ~50 mp. Found by
  the δ = 0 floor test + per-term breakdown. Now full precision.
- Per-file isolated-pawn cells were fitted independently ⇒ file-mirror symmetry broken on 77% of positions. The fitter now ties
  a=h, b=g, c=f, d=e. (The memory rule already existed — I didn't apply it until the gate failed.)
- `_texel_kauf_fit` (imported) acts on `MODE=export` AND `MODE=closure` at import — my closure silently ran ITS check. Modes
  renamed `tables` / `verify`.
- The first depth / hybrid fits stopped after 1 iteration (column scale from the static rows only ⇒ σ ≈ 1e-9 on depth-only
  columns). Fixed before any result was used.
- A C1-loader loop bug (`while (!n_set …)`) and cap columns double-scaled by WIN_AWARE — both caught in review before running.
- Predictions missed: "weighting schemes within 5 cp" (17 cp) · "λ gain is cheap drift" (it grows with movement) · "the queen
  overshoot carries R3A's SF18 loss" (neither half did) · Mediocre "±25" (both arms read ~+50, noisy at 200 g).
- I treated one 1,000-game SF18 sample (R3A −14) as suspicious-real; fresh seeds showed it was mostly noise. Pooled reads only.
- Listed "flat" and "no weighting" as two arms — they are the same.
- PowerShell expands `$(…)` inside `wsl.exe -lc "…"` — use the Grep/Read tools for log reads, or single-quote-free commands.
- Watchers with the 12-h cap died before long queues ended; the queues themselves finished — read the LOG, not the watcher.
★ **The pattern:** every export / fitter defect surfaced in an engine-side check (δ = 0 floor, per-block closure, REALISED gain,
symmetry, byte-identity compiled == files) before any game; and games still overruled accuracy (R3A). Keep both layers.

Mode: execution, interactive.

## ▶️ FIRST ACTIONS, in order
1. **Confirm the run state:** `wsl.exe -e bash -lc "bash '/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine/selfplay/overnight_runner.sh' ps"`
   → expect `none running`. If anything IS running (the owner may have launched a queue since), identify it from the logs
   (`E:\chess_data\q*.log`, `E:\chess_data\retune\q*.log`, `E:\chess_data\traj\q67_s*.log`) and confirm it is healthy (log advancing,
   no ☠️ lines, RAM sane via `free -g`) BEFORE launching anything. Detached queues don't show in the UI.
2. **Re-verify the fingerprints:** v1 **250 / 35,310,778 / 3.784** (`… wac fp_v1`) · v2 shipped = gen 3 **259 / 47,869,841 / 3.927**
   (`V2_PRESET=shipped … wac fp_v2`) · `V2_SHIP_GEN=2` 256 / 47,473,313 · `V2_SHIP_GEN=1` 254 / 50,622,239.
3. **Then the POT design discussion with the owner** (next steps §0) — start from the brief's decision list.

## Transfer documents, in reading order
1. **This file** ★★, then `dev_notes/SESSION-HANDOFF-2026-10-13.md` (its top "UPDATE 2026-10-15" block + the ships table,
   method and owner-calls sections) and `SESSION-HANDOFF-2026-10-10.md` / `-10-07.md` for the plan's origin.
2. **`dev_notes/POT-DESIGN-BRIEF-2026-10-13.md`** ★★ — §1 residual map, §2 direction vs magnitude, §3 decisions, appendix A
   (SF endgame.cpp signatures vs SF11/SF15.1/SF18), appendix B (data ready). Plus `research-2026-10-08/POT-DESIGN-PREPLAN.md`
   (the Fable research catalogue of SF/Ethereal/Weiss winnable / scale / endgame recipes + the owner's framing notes at its end).
3. Memory `eval-v2-rebuild-state.md` (top block → 10-15) and `baseline-fingerprints-register.md`.
4. `dev_notes/EVAL-V2-CURRENT-CONFIG.md` §1 (gen 2 + gen 3 blocks) · `OPTIMIZATION_LOG.md` (10-11, 10-12 entries).
5. `dev_notes/DIAGNOSTICS-TOOLKIT.md` — the "Added 2026-10-10" table and the rows after it (all new tools + queues #50-#67).
6. Bench numbers: queue logs `E:\chess_data\q55_bench_gen2.log` (gen-2 ladder + gap strata) and `q62_eg_signatures.log`.

## Method memory — read before designing anything
★★★ [[static-only-fit-worsens-the-depth-target]] (new) · ★★★ [[real-d10-accuracy-gains-did-not-become-elo]] ·
★★ [[two-fair-instruments-can-genuinely-disagree]] (the ship rule) · ★★ [[spsa-tuning-needs-replication-not-convergence-stats]]
(10-15 result appended) · ★★ [[the-sf18-gauntlet-anchor-drifted-too-weak]] (re-anchored 10-14) · [[fit-data-depth-must-match-play-depth]]
· [[the-remaining-eval-gap-is-endgame-structural]] · [[ovd-is-the-owners-long-term-pressure-concept]] (POT) ·
[[final-retune-needs-a-giant-diverse-corpus]] · [[long-term-goal-3000-single-core-hce-via-search-v2]] ·
[[unique-where-better-never-self-nerf]] · [[baseline-fingerprints-register]] · [[pyrun-knobs-must-be-env-not-args]] ·
[[long-background-tasks-must-be-purposeful-and-bounded]] · [[heavy-agent-calls-only-on-owner-say-so]] · [[no-commit-footer]] ·
[[git-line-endings-check-the-stored-blob]] (stage, read `--stat`, then commit).

## What this is (unchanged)
A non-negamax C++ HCE in `NN Engine/` driven through Cython under WSL; v2 eval (`eval_v2.cpp`, Black-positive mp) is the ship,
v1 the frozen control. Roadmap: strongest single-core HCE (goal ~3000 CCRL) via "search v2" (negamax), then the owner's NN
taught by this HCE. Origin and v1 inventions: memory `owners-engine-origin-article`, CLAUDE.md files.

## 📜 HISTORY ARC (pointers)
v1 → v2 rebuild (09-10 → 10-01, term by term, gates) → ships 09-27 PST Fit A · 10-01 winnability · 10-03 Kaufman + joint KS ·
10-04 connected → 10-07/10 instruments + corpus (no ships) → **10-11 gen 2 · 10-12 gen 3** (this phase) → POT. Handoffs:
`SESSION-HANDOFF-2026-09-25 / -09-30 / -10-04 / -10-06 / -10-07 / -10-10 / -10-13 / -10-15.md`.

## ▶️ THE MEASURED PICTURE (where gen 3 stands)
- **External:** SF18 @1000 54.0% · @1500 37.3% · @2000 28.3% ⇒ use **SF18 @~1100** for POT gates. Mediocre v0.5 @1 s ≈ 45-52%
  per seed ⇒ ~2,300-2,370 CCRL ±100 (indicative).
- **Static vs SF11** (gen-2 bench, `q55`): gap shrank ~40%; ahead on own-play middlegames / pure endgames (home field), behind
  on endgame-leaning + decisive positions; off-distribution (odds, variants) SF11 still ahead. NNUEs ~2-2.5× more accurate.
- **Effective piece values** moved toward SF18-search ratios; remaining: pure-endgame rook (R/N 1.72 vs 1.93), queen slightly
  low, endgame pawns vs pieces low (0.41 vs 0.50).
- **Endgames** (POT brief §1 + appendix A): static under-confident in every type; search fixes most EXCEPT pawn endings (d10
  bias −14.6 / −6.5). We beat SF11's recipes in conversion endings (lone king, KQ vs KR/P, rook vs minor); SF11 still beats us
  on fortress / drawing patterns (KBPsK wrong bishop, KNPKB / KBPKN, KRPKR / KRPPKRP) and pawn endings (search + KPKP probe).
- **POT probe:** DIRECTION (passers, majorities, king activity) adds nothing on top of gen 3; MAGNITUDE (SF-winnable-style
  complexity) static EG −5.9% / depth EG −1.5% — biggest weight "pure pawn ending ⇒ more confident" (may be a search gap).
- **Winnability:** scale side at its optimum (SPSA no direction; the 4 caps fitted to ~0 — they push toward draws and our eval is
  already too drawish).

## ▶️ NEXT STEPS (order: POT nested → search transition → search v2 → NN; eval re-opens by light refits after each)
0. **POT design discussion (owner).** Decisions framed in the brief §3: (a) cause split — the two SF11 pawn-ending SEARCH rules
   first (option c), or POT magnitude without the pure-pawn-ending input; (b) SF's bounded complexity form
   `sign(eg)·max(C, −|eg|)` with our inputs; (c) middlegame potential still parked?; (d) which SF endgame recipes to port
   (fortress / drawing — appendix A); (e) how to use the TRAJECTORY set (fit POT(t) to `static(t+k) − static(t)` = transformation,
   and to `sf(t+k)` = outcome, then the depth target — the owner's idea).
1. *(optional, cheap)* Gate `KS_V2_HALF=818` (SPSA's only replicated knob): self-play @50k 2,000 g → SF18 @1100 2 × 500 → Mediocre.
2. **Build POT** as engine detectors (byte-identical at 0, symmetry-gated, closure-checked), assemble the EGX slice into the
   corpus (like GAUNT), fit nested on gen 3 with everything frozen (`_retune_fit.py` + new columns; `WIN_AWARE=1`), light joint
   refit, gate (self-play → SF18 @~1100 → Mediocre ≥ 200 g), **bench checkpoint #2** (`_reference_ceiling` + `_gap_strata` +
   `_endgame_types` + `_eg_signatures`).
3. **Search transition:** SF11 pawn-ending rules + the two verified defects (`research-2026-10-08/SEARCH-TRANSITION-PREPLAN.md`
   end) · corr hist · threats inside search · dynamic terms nested · margin re-sweep on the final eval · **bench checkpoint #3**.
   Early in the search arc: testing-throughput infra (native UCI binary instead of the Python wrapper, leaner per-process tables ⇒
   more concurrent games).
4. **Search v2:** premise re-audit of every closed search item · negamax core · speed (pin-aware movegen first, pawn hash,
   incremental material/PST) · then ~3000 single-core · then the owner's NN (corpus + labels reusable; HCE as teacher).
5. **Full retune after POT** (the eval is never "done"): include the owner's EXPLORE/EXPLOIT generator test — fit ± explore slice,
   read own-play held-out + games (own play) AND odds / variant / diverse benches (generality); both up ⇒ diversification wins.

## ⚠️ WHAT IS ALREADY THERE (this phase)
- **Engine:** `V2_SHIP_GEN` 1/2/3 (preset default 3; compiled `ship_tables_v3.h`, `ship_tables_v4.h`) · `V2_VAL_{N,B,R,Q}_{MG,EG}`
  piece-value knobs (0 = byte-identical) · loader fallbacks to compiled tables for C1 / KFL / KPROT / PX / CONN / KSB / KAUF / PST.
- **Tools (toolkit 10-10+):** `_corpus_assemble.py` · `_px_export.py IN=` (chunked, + `total`, + `winnab`) · `_retune_fit.py`
  (weights / lambda / hybrid; VAL, PIN, WIN_AWARE, CAPS, FEAT, CORPUS, D10_GLOB) · `_retune_export.py` (tables / verify, ONLY=,
  REALISED gain) · `_ship_tables_gen.py GEN=` · `_pot_probe.py` · `_eg_signatures.py` · `_build_trajectories.py` ·
  `_build_game_regret_set.py` (game column, EG_SIG_QUOTA).
- **Data (E:\chess_data):** `retune/corpus_v1.csv` (956,670) · `corpus_v2.csv` (+100k GAUNT) · `feat_ship / feat_gen2 / feat_gen3 /
  feat_gen3v2` exports (+ cap dumps) · d10 passes `A_ours1004 / 1011 / 1012` · `EGX_*` endgame set (60k + 58,929 d10) ·
  `traj/traj_s*of4.csv` (~39k trajectories) · bench dumps `bench1011/`, `bench1013/`.
- **Queues:** `selfplay/_queue_*.sh` #50-#67 (each guarded: fingerprint → symmetry → work), `_launch_detached.sh` (`--waits`).

## ★★★ OWNER CALLS (this phase)
Quick experimental understanding over a-priori choices · flat band weights, quiet-only static, stress weight 0.5 · ship gen 2 and
gen 3 (values arm for structure) · piece-value knobs OK'd · R3A not shipped ("sus") · corpus breadth for a general evaluator —
explore generator after POT · trajectory set for POT (transformation + outcome) · bench checkpoints after retune / POT / search
transition · concurrency 4 for timed runs OK (machine: 8 cores / 16 threads, 31 GB; RAM, not CPU, limits) · fill idle nights with
RUNS (cheap on context), avoid unattended BUILDING (eats context) · busy days ⇒ only short decisions · push only on say-so ·
the eval arc is never closed — it re-opens after each structural change.

## STATE
Branch NN-ENgine, HEAD 1eca812-ish + this file; origin at ad463d9 (~45+ commits unpushed). Tree clean apart from the ChessUI
submodule and old untracked files. UI: `V2_PRESET=shipped PRESET=LONG_FORMAT python ChessUI/chess_ui_v2.py` (plays gen 3).
MEMORY.md ≈ 21 KB — prune before adding much.

## DISCIPLINES (carry forward)
★★★★ Games decide (owner rule: self-play ≥ +2σ, SF18 > −1σ pooled over ≥ 2 seeds, Mediocre ≥ 200 g not negative); accuracy
screens harm only · every fit arm must carry the depth component · every exported arm passes: δ = 0 floor, closure + REALISED
gain ≥ ~80%, colour + file symmetry 0, compiled == files byte-identity, v1 unchanged · register predictions before each run ·
pooled reads only · re-anchor the judge after ships.
★★★ Ops: ≤ 4 engines (≈ 8 processes); never rebuild or edit a script a running queue uses; engine knobs as ENV; detached queues +
bounded watchers (read the LOG if a watcher dies); bulk output on E:; git add explicit paths, check `--stat`; commit by feature,
no footer; owner games 9pm-midnight + weekend afternoons ⇒ node-limited / fixed-depth only then; timed (Mediocre) runs in
daytime windows.
