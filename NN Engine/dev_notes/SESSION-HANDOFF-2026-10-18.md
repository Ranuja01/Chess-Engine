# SESSION HANDOFF — 2026-10-18 (covers 10-15 → 10-18)

HANDOFF — the POT phase began: a C++ race, winnability class cells, the first narrow-item gates, two harness bugs fixed, a sharper
game venue, a "lazy ship" development base, and seven owner games that map our remaining weaknesses.
- **Official ship unchanged: gen 3** (`V2_PRESET=shipped`, 259 / 47,869,841 / 3.927). **Development base: `V2_PRESET=candidate`**
  = gen 3 + C64 (winnability cells `POT_V2_WIN_PAWNEND=64 POT_V2_WIN_ONEMINOR=16`), 258 / 47,689,413 / 3.926 — owner's "lazy ship":
  test new items on top of it, bundles carry it, ship officially only when a bundle passes; revoke on any sign of harm.
- **Pawn-ending drawishness is EVAL, not search** (SF11's pawn-ending search rules refuted, #69). Class cells fix it at depth
  (own held-out pawn rows d10 169.9 → 75.6, d12 141.3 → 61.1, bias fixed).
- **The POT race (POT proper) is PARKED at 0** — helps at d10, adds nothing at d12, orders moves ~blind; king-path routes did not
  help (z +0.63 vs needed +2). Kept for a search-transition test (race + tightened margins), per the owner's point that static
  accuracy feeds pruning.
- **Seven owner games** → recurring eval themes + a triage: 9 of 15 misses are SEARCH-side (depth / pruning), 6 EVAL.
~60 commits since 10-15, none pushed (push only on say-so), no footers. Nothing running at handoff.

## ⚠️ HONESTY FIRST — errors this phase
- Predictions missed: race "≈ −3" for the SF11 pawn-ending rules (got −7.0) · "static fit overshoots at depth" (it did not at
  d10) · set A targeted +20…+60 (venue unreadable) · set A timed "positive" (−12.7) · king-path +1…+3 (+0.63) · "≥ 85%
  discriminating pairs" (44%, and made with a wrong definition).
- The light-ending race gain (−5.8%) was CONFOUNDED by material class (class indicators alone −8.7%) — caught only because the
  owner asked about the size of misses.
- The knob-fit emulation first recovered the scale factor after its clamp (2,000 mp tail) — caught by MODE=verify before use.
- Depth pass silently dropped 113 decisive rows: amplified evals crossed RESIGN_THRESHOLD (engine returned no move).
- Random pawn-dominated starts were decided before move 1 (513/600 pairs) — a venue null mistaken for a candidate null at first.
- The OPENING BOOK played 5.3% of moves in FEN-start endgame venues and dropped promotion pieces (29 illegal-move voids).
- `_pair_discrimination.py` first counted two drawn games as informative.
- A numpy "single process" ran at ~700% CPU beside the owner's UI game (BLAS threads) — pin OMP/OPENBLAS/MKL_NUM_THREADS=1.
- I missed a milestone update (no watcher armed for one queue stage) and twice hit PowerShell `$`-quoting (write a script file).

## ▶️ FIRST ACTIONS
1. `overnight_runner.sh ps` → expect none running.
2. Fingerprints: v1 250 / 35,310,778 · shipped 259 / 47,869,841 · candidate 258 / 47,689,413.
3. Continue with the owner (see NEXT).

## Transfer documents, in order
1. This file. 2. `dev_notes/POT-DESIGN-BRIEF-2026-10-13.md` §5 → §5j (every decision and result, in order). 3.
`dev_notes/EVAL-V2-CURRENT-CONFIG.md` §0 (the candidate pool + evidence). 4. `NN Engine/CLAUDE.md` (the two presets, book off for FEN
venues). 5. `diagnostics/ks_sets/pot_owner_cases.txt` (owner game positions with SF / our readings). 6. Memory: eval-v2-rebuild-state
(top block), narrow-items-are-judged-where-they-fire, pot-may-not-work-use-stopping-points, no-side-runs-while-a-queue-saturates-the-box,
archive-old-selfplay-games, baseline-fingerprints-register.

## Rules / owner calls this phase
- **Narrow-item rule:** component checks + a TARGETED venue (critical-point starts) ≥ +2σ + standard / SF18 only > −1σ + Mediocre +
  timed when it costs cycles. **Bundling: yes** — keep evidence per item, test bundles. **Lazy ship** (candidate preset).
- **POT may not work** (owner's own idea): pre-registered keep/park per piece + time box; park, never delete. Aim (owner): close the
  gap to SF18's SEARCH and change deep-search decisions in mg AND eg ⇒ static accuracy against DEEPER truth (depth component at play
  depth 12+, trajectory outcomes), judged on CALIBRATION and DISCRIMINATION (sibling ordering / search move agreement).
- Static accuracy matters for the search arc (pruning / stand-pat / ordering) — the race's −22% nodes at d10 is evidence.
- Long runs overnight, interactive ones while the owner is awake; keep the owner updated at milestones (arm a watcher per stage).
- No side runs beside a queue (owner's live server); Python runs single-threaded math.

## STATE
- Built (all at 0 / byte-identical unless in `candidate`): SF11 pawn-ending search knobs (NULL_NPM_GUARD / PE_PRUNE_GATE /
  PE_PASSER_EXT); POT race + class cells + WIN_RACE feed + race weights (speed-optimised, closure exact); v2 profile scopes; book fix.
- Tools: `_pot_race_proto.py` (fit / moves / cases / split; KPATH), `_pot_knob_fit.py` (exact emulation), `_pot_race_closure.py`,
  `_pe_arms.py`, `_pot_speed.py`, `_eval_profile_v2.py`, `_eval_stability.py`, `_make_balanced_eg_starts.py`,
  `_make_critical_starts.py`, `_pair_discrimination.py`, `_case_triage.py`.
- Data: `E:\chess_data\pe\` (pawn rows, light multi-PV labels 6,000, knob-fit dump), `E:\chess_data\traj\traj_eg_s*of3.csv` (12,000
  endgame trajectories) + the 10-15 mid-game trajectories; venues `selfplay/eg_balanced_fens.txt`, `selfplay/eg_critical_fens.txt`.
- Profile: KS-A ~30% of v2 eval in every phase; 26% of late-endgame eval unscoped; v2's shallowest positions d8 vs v1 d10.

## NEXT (owner to choose order)
1. POT: complexity scaler test (last untested piece of the time-box pair) · DEEP-RESIDUAL set (our d12+ vs SF18, mg + eg, incl.
   rook endings with passers and closed middlegames) as the target for potential signals · fit on the trajectory sets.
2. More narrow items for the pool (bundle test when there are several); Mediocre ≥ 200 g for C64 (timed, quiet slot).
3. Search transition (now with evidence): triage DEPTH/PRUNING cases · race + margin re-sweep · v2 tree blow-up positions ·
   search mirror test (null-move depth MINI 3 / MAXI 4, futility `cur_depth > 1` one side) · null-move fix re-read.
4. Bench checkpoint #2 after a POT checkpoint with gains (scope in the brief §5f). Pending: archive old selfplay games (> 30 GB).
