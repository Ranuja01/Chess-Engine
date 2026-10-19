# SESSION HANDOFF — 2026-10-19 (covers 10-18 evening → 10-19): EVAL PHASE CLOSED, GEN 4 SHIPPED, SEARCH TRANSITION NEXT

- **Official ship: gen 4** = gen 3 + C64 (winnability class cells `POT_V2_WIN_PAWNEND=64 POT_V2_WIN_ONEMINOR=16`).
  `V2_PRESET=shipped` **258 / 47,689,413 / 3.926**; `V2_SHIP_GEN=3` = gen 3 (259 / 47,869,841); v1 250 / 35,310,778.
  `candidate == shipped` (pool empty). Pushed to `origin/NN-ENgine` (0b87008); master untouched (merge after a cleanup, weeks away).
- **POT for the eval: closed** (brief §5k-§5o). Complexity −0.41% (keep ≥ 2%), middlegame POT −0.46% / move z −0.77; hand-written
  pending-break features ≈ 0 weight. Owner reading: POT is not conceptually wrong — cheap static patterns cannot capture it at
  play depth (why the giants have none); its future is as a SEARCH signal and inside the NN. Never-measured preplan items (E1,
  M1-feeder, M2, M3) and every parked piece are in `EVAL-V2-PARKED-REGISTER.md` with re-read triggers.
- **Label defect found** (memory `sf-cp-labels-are-material-scaled-use-wdl`): SF18's cp is material-scaled; our fixed cp → win%
  formula is off by ~15 win% pts on average vs SF18's own WDL. A boosted-tree "−10%" (no leakage: by-tag −10.0%, unseen holdout
  −12.3%) vanished on WDL labels at small n. WDL calibration of the eval by class: −1.1% under both labels ⇒ NOT mis-scaled.
  ONEMINOR is static-miscalibrated on WDL (+14% where it fires) but positive in games by start type (pooled z ≈ +2.9) ⇒ kept;
  re-read at the final retune (fit on WDL).
- **Archive routine** done: 1,391 tags → E:\chess_data\archive (22.15 → 2.60 GB, 0 failures), C: 44 GB free; routine note in
  `NN Engine/CLAUDE.md`; `selfplay/GAMES-INDEX.csv`.
- Running: **queue #82** (armed, starts 23:30 10-19): SF18 WDL on the 22k trajectory train starts + `_pot_gbm_ceiling.py
  MODE=wdlfull` — the full-size version of the WDL check. Log `E:\chess_data\q82_wdl_full.log`.

## ⚠️ Honesty — this stretch
- I over-applied the POT time box twice: to all of POT (it covered the eg line), then "every POT piece missed" (preplan items
  E1 / M1-feeder / M2 / M3 were never tested). Owner caught both.
- The boosted-tree "systematic error" was reported as real before the label units were checked; the owner's challenge
  ("a simple model beat months of eval work?") led to the WDL test that withdrew it.
- I proposed dropping ONEMINOR on static WDL evidence; the existing games split by start type reversed it (games decide).
- Bugs caught before reading: the U flag held a value (U=1 read 302.9%); `_build_regret_set.py` ignores N with IN=.

## NEXT — the search transition (discuss the plan first; owner order: discuss at every step)
1. Infrastructure: native UCI binary / leaner tables (test throughput).
2. Speed, result-identical: KS-A ~30% of eval; 26% of late-eg eval unscoped; pawn-hash caching; pin-aware movegen.
3. Correctness: search mirror test (null-move CURDEPTH MINI 3 / MAXI 4; futility `cur_depth > 1` one side); null-move fix
   re-read; v2 tree blow-up positions (shallowest d8 vs v1 d10).
4. Features: corrhist (hybrid eval + search; its "last by necessity" trigger is now met) · threats in EVAL together with threats
   in SEARCH · triage DEPTH / PRUNING cases (`_case_triage.py`) · POT-as-search (race + margins, an "unresolved" signal for
   LMR / time, pending-break protection — game 7).
5. Margin re-sweep LAST at equal time; then bench checkpoint #3 → search v2. Final retune (WDL labels) after the transition.
