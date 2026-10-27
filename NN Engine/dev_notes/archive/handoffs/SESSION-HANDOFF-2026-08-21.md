# SESSION HANDOFF 2026-08-21 — KS eval candidate NULL on diverse openings; a testing-methodology finding; SEE-captures live

## ⏱️ READ FIRST — the state in one screen
A ~33-hour session. The KS eval candidate that looked like a win was a **fixed-opening artifact**; diversifying the
game openings killed it. That diversification finding is the session's biggest result and **reframes the whole
game-testing (and possibly the D7-tuning) methodology**. Nothing shipped; the engine default is unchanged.

- **KS eval candidate = PRACTICAL NULL.** `ENABLE_KS_RING_GATE=1 KS_MIN_ATTACKERS=2 KS_FLOOR=0 CAPG_KS_DAMP=25 KING_SAFETY_MAG=4000`
  (endgame `ENABLE_KS_UNIFIED` OFF — the wave-tuner dropped it). Bench strong (D7 collapse-wave −0.74, STS-neutral,
  symmetry-clean, byte-id). Fixed-opening games **~+11.5 / 2000g** → but that was 6 REPLAYS of the same seeded
  openings (near-copies). On DIVERSE openings (UHO 1000-book, varied seed): **~−5 / 782g** (seg7 +0.9, seg8 −11).
  ⇒ the +11.5 did NOT generalize. Config stays GATED (ships nothing). Additive/eval-KS ≈ 0-for-10 now.
- **★★★ META-FINDING: fixed `openings.txt` (seed-0) gave OPTIMISTIC game reads for BOTH candidates.**
  KS: fixed +11.5 → diverse −5. SEE-captures: banked-fixed +32 → diverse ~+11. The small default opening set is
  NON-REPRESENTATIVE (likely sharp/tactical-skewed where our tweaks over-help). ⇒ **diversified openings (UHO +
  VARIED seed) are now MANDATORY for any game test.** Root cause pinned: `gate` sub hardcoded `--seed 0` on the
  small `openings.txt`; `schedule()` shuffles by seed, so a constant seed = the same games every segment.
- **SEE-captures (SEARCH candidate) = live positive lean, UNRESOLVED.** `ENABLE_TT_FLAG_FIX=1 ENABLE_NULL_MATE_CLAMP=1
  SEE_PRUNE_CAPTURES=1 SEE_PRUNE_CAPTURE_MARGIN=1000`. Diverse pool (seg1 +3.5, seg2 +18.3; 800g) **~+11 ± 24**, both
  segments positive (unlike KS). CI still includes 0. **seg3 (seed 3) RUNNING** to firm. This is the candidate with hope.

## 🏗️ WHAT'S RUNNING RIGHT NOW (survives compression as OS processes; resume from disk)
- SEE-captures diverse SPRT **seg3**, task `bxfo1816t` (conc4, UHO seed 3, 400g). Results → `selfplay/games/seecap_seg3/sprt.json`.
- Heartbeat sleep loop (last: `b76ngw4og`) — the hourly monitor; may need manual relaunch post-compression.
- **To resume:** read `OVERNIGHT-LOG-2026-08-20.md` (per-segment tally) + the running task output; pool W-L-D across
  seed segments; chain seg4 (seed 4), seg5… with `gate '<see-cfg>' seecap seecap_segN 400 5 4 openings_uho.txt N`;
  ONE core-loading job at a time; ~120 g/hr; a ~+11 effect needs a few thousand diverse games to clear 0.

## 🔬 THE THREE OPEN THREADS (in priority order)
1. **Rebuild the D7 regret set on DIVERSE openings.** `game_regret_set.csv` is sampled from self-play games that
   likely used the small fixed openings ⇒ the wave-tuner may have optimized for fixed-opening-derived positions
   (explains fixed-good/diverse-null). VERIFY which openings the current set's self-play used (`_build_game_regret_set.py`
   reads `selfplay/games/`), then rebuild from UHO-opening self-play and RE-RUN D7 tuning. **This could REOPEN closed
   eval lanes** — their D7 nulls may have been opening-biased. Highest-leverage methodological fix.
2. **Test the endgame-KS extension ON ITS OWN.** `ENABLE_KS_UNIFIED=1` (+ `KS_PHASE_FLOOR`) was turned OFF by the
   wave-tuner (it kept midgame ring-gate). The owner's endgame-non-zeroing idea is UNTESTED as an isolated candidate.
   The refactor code is banked (byte-id, symmetry-clean). Test it clean on diverse openings + the de-biased regret set.
3. **The eval forward roadmap** → `EVAL-MINDMAP-2026-08-20.md`: capgains (the #1 culprit + the real −6.75 game blowup;
   `CAPG_KS_DAMP`/`ENABLE_CAPG_REALIZ` scaffolding exists) > passers (blockade/realizability gate; phantom-knob hazard)
   > rook open-file (`ENABLE_ROOK_TENSION_COND` coded). All as FIRING-CONDITION decouples via the dormant
   `dynamic-conditional-eval.md` layer. Validate with the (rebuilt-diverse) D7 method + SAFE-set silence + games.

## 🧰 DURABLE WINS (stand regardless of candidate outcomes)
- **Gated KS refactor** (branch-independent endgame KS: `ENABLE_KS_UNIFIED`, `KS_PHASE_FLOOR`; byte-id default 250 /
  36,651,879, symmetry-clean). Ships nothing; available.
- **`_ks_wave_tune.py`** — the D7 curriculum/stakes tuner (collapse-first + general guardrail).
- **`gate` sub now takes `[openings_file] [seed]`** (overnight_runner.sh) — diversified game testing enabled.
- **`EVAL-MINDMAP-2026-08-20.md`** — full failure taxonomy (8 modes, 3 meta-roots) + forward roadmap.
- **`OVERNIGHT-LOG-2026-08-20.md`** — the full night's per-segment ledger.

## ⚙️ DISCIPLINE (unchanged)
Byte-id every build; colour-symmetry ship gate (`_eval_symmetry.py N=800`); STS+mirror on the balanced total; GAMES
decide on DIVERSE openings; ONE core-loading job at a time (JOBS/conc≤4 OOM); segment SPRTs; read outputs with Read
not shell-grep; knobs latch at init; no commits while owner away. ⚠️ **DON'T read mid-segment game numbers as signal**
— they swing ±40 and regressed to null repeatedly this session (I over-called +12→0 twice; only closed segments count).

## STATE
HEAD on branch NN-ENgine = shipped +20.8 bundle, unchanged. This session's code: the gated KS-unification refactor
(cpp_bitboard.cpp ring/taper + search_engine.{h,cpp} flags) + the `gate`-sub openings/seed passthrough + 3 new
diagnostics (`_ks_wave_tune.py`, `_build_p2_specset.py`, `_p2_specset.txt`) + 3 dev-notes (mindmap, overnight-log,
this handoff). All UNCOMMITTED, gated/byte-id at default. Memory outside repo. SEE-captures seg3 running.
