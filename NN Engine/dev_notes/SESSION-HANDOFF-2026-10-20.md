# SESSION HANDOFF — 2026-10-20 (covers 10-19 evening → 10-20): v2 IS THE ENGINE, BASELINES RECORDED, SEARCH TRANSITION UNDERWAY

- **v2 gen 4 is the official engine** (`V2_PRESET=shipped`, 258 / 47,689,413 / 3.926); v1 (`EVAL_ARM=0`, 250 / 35,310,778 / 3.784) is
  the frozen legacy control (still the Cython no-env default, only so fingerprint scripts keep working).
- **The baseline record = OPTIMIZATION_LOG 2026-10-20** (+ `BASELINE_PERF.md` §v2): equal-time games v2 vs v1 **+178** (st=1, 274 g)
  / **+199** (60+0.6, 300 g); v2 @ 1 s vs SF18 @ 5 ms **11.3%** (200 g), @ 20 ms 0/200; STS300 d10 **2115** (v1 1796; SF11 1983,
  SF15.1c 2134, SF18 2414 at d10 — but SF11 @ our 249k nodes 2374 ⇒ **−259 at equal nodes**, was −520 on 09-21); full STS 64.8%
  (SF11 65.5%); calibrated win% MSE v2 vs v1 −31…−43% (static + d10); SF11 static ~15% better on SF18's WDL, mostly endgames.
- **Search transition:** step 1 (native UCI binary + fastchess) DONE; step 3 (correctness) underway — five fixes as knobs (default
  off, byte-identical), first equal-time screens **A2 `FIX_MIN_SCOUT` +49 (+12 … +87)**, **A1 `FIX_RESEARCH` +31 (−8 … +71)**.
- Everything committed; push pending (owner says when). Nothing running.

## ⚠️ Honesty — this stretch
- The boosted-tree WDL check: I WITHDREW the "systematic error" claim on a 3.1k-row check that was underpowered; the full-size
  check (#82) restored half of it (−13.0% → −7.3% on WDL). Lesson: power a check before withdrawing a claim, as before making one.
- I first moved `Cython_Chess` to legacy/ (ChessAI.pyx imports it; my reference grep truncated at 3 matches) — the post-tidy
  fingerprint check caught it in minutes.
- A `sed -i` rewrote search_engine.cpp's CRLF line endings (19,229-line diff for 7 lines) — caught by `git show --stat` before push.
- Queue #83's first launch aborted at its bench guard (script never `cd`-ed into NN Engine); a stray `cat >>` with no input hung
  a command and created an empty file — both mine, both caught.
- fastchess stalls: I first guessed the cache clear; measurement showed RSS growth + process reuse ⇒ swapping (fix `restart=on`).
- Raw WDL MSE ranked v1 above v2 (a SCALE artefact — v1's best scale 0.63); only calibrated MSE is reported.

## STATE (where things are)
- Binary: `build_uci/chess_engine_uci` (+ `_fix`, same source) via `selfplay/build_uci.sh` (`OUT=`); verify with `bench` (== 258 /
  47,689,413). Options incl. `Knob_<NAME>`, `Debug`, `OwnBook`, `Preset`. Memory `uci-binary-and-fastchess`.
- fastchess v1.8.2 `E:\chess_data\engines\fastchess\`; openings `E:\chess_data\fastchess\openings_uho.epd`; report tool
  `selfplay/fastchess_report.py`; queue templates `selfplay/_queue_fc_*.sh` (ALWAYS `restart=on`, `timemargin=250`).
- Search fixes (`search_engine.h` FIX_*): B1 `FIX_HALFMOVE`, A4 `FIX_MAX_NULL`, A3 `FIX_MIN_RD1` (=1 drop / =2 both sides), A2
  `FIX_MIN_SCOUT`, A1 `FIX_RESEARCH`; B2 `FIX_REP_KEY` declared, NOT implemented (10 count sites, needs a unit test).
  WAC d10 per fix in `SEARCH-CORRECTNESS-PLAN-2026-10-19.md` §E; screens §F.
- UI: `ChessUI/chess_engine.py` `NNEngineUCI(preset | movetime | clock=(base, inc), verbose, debug, resign)` + `GameClock` +
  `Stockfish(movetime, clock)` — uncommitted in the friend's submodule.
- Data: `E:\chess_data\wdlcal\` (5,988 held-out rows with SF18 cp + WDL; static / d10 dumps for v1, v2, SF11) · `E:\chess_data\deep\`
  (mg trajectory starts, d12, WDL train/val) · `E:\chess_data\fastchess\` (all match PGNs + logs) · archives `E:\chess_data\archive\`.

## NEXT (owner to confirm; planned for the next window)
1. **Overnight:** A2 confirmation (≥ 400 g, fresh seeds) · then A2 × A1 2×2 · SF11 ladder (ours @ 1 s vs SF11 @ 10 / 30 / 100 ms,
   200 g each — the yardstick "playable with SF11") · SF18 loss autopsy on the 174 losses (SF18 d14 along each game → decisive
   moment → phase / tactical-positional / eval-vs-search).
2. Correctness continued: cheap three (B1, A4 + B2 once implemented) as a bundle; A3 at equal time both directions; the null-move
   fix re-read; v2 tree blow-up positions; owner-game triage DEPTH / PRUNING cases.
3. Speed programme (result-identical, batched, judged at ≥ ~30% NPS): move-gen cache size (also the memory growth), KS-A ~30% of
   eval, 26% unscoped late-eg eval, pawn-hash caching, pin-aware movegen.
4. Features: corrhist · threats in eval + search · time-manager upgrade (vs the placeholder rule, at equal average time) · POT-as-search.
5. Margins re-sweep LAST, then bench checkpoint #3 → search v2. Track: STS at EQUAL NODES vs SF11 (−259 today) and the SF11 ladder.
