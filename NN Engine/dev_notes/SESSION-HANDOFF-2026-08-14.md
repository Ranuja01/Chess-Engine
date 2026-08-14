# Session handoff — 2026-08-14: the criticality-split reframe, the additive-hurts-critical verdict, and the data build

> ## 🚨 READ THIS BLOCK FIRST  (UPDATED end-of-session — outcome CHANGED)
> The bundle **SHIPPED**. The +15 bundle (OvD+central+defaware1) was CONFIRMED **+20.8 Elo** (pooled 1871 lightning
> games, old sprt_bundle_ks 1362 g + two fresh batches; 95% CI [+6.9,+34.7] clears 0) and shipped as default —
> **new default fingerprint `250 / 36,831,767 / EBF 3.772`** (commit `7743cad`; arch `8f21a36`; tooling/docs `c85a0a0`).
> Pre-ship gates passed (symmetry-neutral, NPS tooling-clean). Pre-bundle baseline `250 / 35,426,396` recoverable via
> old knobs. This session ALSO produced the methodology + mechanism verdict + data pipeline below (all still current).
> ⚠️ Mid-session note preserved for context: the analysis/triangulation below was written BEFORE the ship + the
> WSL-concurrency crash/reboot; the ship and the 3-corpus triangulation verdict came after. See OPTIMIZATION_LOG
> 2026-08-14 entry for the consolidated outcome.

## ★★★★ THE CENTRAL FINDINGS (do not re-derive)
1. **CRITICALITY SPLIT (owner insight) — read SIGNAL, not the average.** Bucket move-regret by SF's best-vs-2nd
   win% gap: benign <3% (several good moves — a swap costs ~0, pure noise), minor 3-8%, moderate 8-20%,
   CRITICAL >20% (one right move — a miss is a real failure). The tool: `_ks_phase_split.py` now emits a
   CRITICALITY split + a NON-PAWN-MATERIAL map alongside the phase table. **Every "net-neutral" aggregate this
   project hid a real critical effect** — the average dilutes a loud critical signal against benign noise.
2. **AIRTIGHT PATTERN: incumbent KS HELPS critical; EVERY additive change HURTS critical.**
   - Incumbent KS (KS-on vs KS-off ruler): helps minor+moderate criticality on BOTH standard sets (extreme-32-crit
     band is noisy/opposite — too few positions). => KS is a REAL critical lever; critical positions are NOT
     purely search-bound.
   - Detectors-alone (no curve): minor +0.54, moderate +2.76, crit +8.6 => HURTS critical.
   - Detector+curve compound, endgame extension, accum: all HURT critical the same way.
   - **Mechanism:** additions make KS LOUDER; a louder crude signal on a PRECISE critical position pulls the move
     off the one right answer. This EXPLAINS additive-KS 0-for-9 at the mechanism level.
   - ★ **AUC ≠ critical-help:** the detectors improved discrimination AUC to 0.81 (past SF's 0.80) and STILL hurt
     critical move-regret. Discrimination-corpus wins do not transfer to critical moves.
3. **IMPROVEMENT DIRECTION PROVEN: not additive.** The only lane left is **SUBTRACTIVE / REDISTRIBUTIVE precision**
   — make KS *more discriminating at CONSTANT magnitude*, never louder. Consistent with the entire win history
   (de-king, MOD_KS_REALIZ, defaware1 were all subtractive/redistributive).
4. **KS is STRUCTURALLY MIDGAME-ONLY (fable-verified — `KS-PHASE-GATING-VERIFICATION-2026-08-13.md`).** The
   attack-unit KS block is inside `if(!isEndGame)` (cpp_bitboard.cpp:7203), i.e. phase_score≤64 (phase≥12, ~npm≥26).
   It CLIFFS to 0 at ps=65; `KS_PHASE_ZERO=104`'s designed smooth fade is DEAD CODE. There is NO king-danger model
   in the endgame (only attackingLayer proximity). The KS detectors are KS-LOCAL (feed nothing outside the KS block).
   `KING_SAFETY_MAG=3000`, `ENABLE_KS_REPLACE_LT=true` (so latent_threat is dead even in the midgame).

## ★★★ WHY OUR DATA COULDN'T SEE IT (the reason to build a corpus)
- The two standard regret sets (`game_regret_set` 15k, `game_regret_set_v2` 11,940, disjoint) are **compositionally
  IDENTICAL** (phase/material/criticality/eval-state all within ~2% — `_regret_set_profile.py`). Their disagreement
  on KS effects was **pure SAMPLING VARIANCE in the starved critical band** (only ~3.5% critical, ~28-32
  config-affected positions). Not different populations — too few critical positions.
- => The fix is **enrich the critical band from millions of positions**, phase/material-stratified, with a benign
  anchor as regression guard. Then critical-move-regret becomes a reliable judge.

## 🗂️ DATA SOURCES CATALOG (over-provisioned — STOP gathering, START using ONE)
- **Lichess puzzle DB** (LOCAL, ~1GB, ready): `.../Chess Coach/chess-coach/data/lichess/lichess_db_puzzle.csv`.
  Schema PuzzleId,FEN,Moves,Rating,...,Themes,... `_lichess_ks_filter.py` STREAMS it (never loads whole) → filters
  KS themes (kingsideAttack/exposedKing/mate*/sacrifice/...) + phase → applies Moves[0] → puzzle position. Got 7500
  (2500/phase incl. 2500 endgame-theme). ⚠️ theme "endgame" ≠ ENGINE endgame (puzzles keep pieces → mostly high
  material / engine-midgame). For engine-endgame, filter LOW non-pawn material.
- **Lichess 985M SF-evaluated positions** (HF `Lichess/chess-position-evaluations`): the SCALING source —
  PRE-EVALUATED (no labeling). ⚠️ stream/sample (never download whole); need MULTI-PV entries (many are single-line);
  NO theme tags (filter by position features). parquet-on-HF.
- **HF `Lichess/collections`** — has POSITIONAL sets (better than tactical puzzles for a static term; not assessed).
- Local **"Pre-evaluated chess position and its features"** set (`.../Chess Engine/data/...`) — likely ML feature
  vectors, may lack multi-PV move labels (unverified).
- **PGNs** (`Chess-Engine/PGNs`, AlphaZero-experiment games, "copied" → dedup first) + our selfplay — raw positions,
  need SF-labeling (less efficient than the pre-eval set).
- VALIDATED so far: KS HELPS critical on the small Lichess batch (−3.09 / 76 crit) — triangulates the standard sets.

## 🧰 NEW TOOLS (all this session)
- `_ks_phase_split.py`: now emits CRITICALITY split + NON-PAWN-MATERIAL map. Many experiment paths via env:
  KSMAG_TEST (KS on/off ruler), DETONLY (detectors alone), EGEXT (endgame extension), EGMAT (material taper),
  ACCUM/COORD_DIV/PROXCTL/NQ_TAX/PHASEZ (built earlier). ⚠️ base/cand hardcoded per path; PID-unique /tmp (concurrency-safe).
- `_regret_set_profile.py`: pure-CSV distribution compare of regret sets.
- `_lichess_ks_filter.py`: stream-filter Lichess puzzles → KS-themed subset.
- `_label_positions.py`: SF18 multi-PV label a position list → our regret format (fen,phase_bucket,best_uci,best_cp,moves).
- Runner subs (auto-approved): `unit_trace`, `ks_phase`, `regret_profile`, `lichess_filter`, `label`.
- Gated byte-id knobs built (all default-off): KS_ACCUM_MODE+, KS_COORD_GATE_MODE+, KS_EG_MAT_GATE+, KS_EXTEND_EG.
  All confirmed HARMFUL-or-neutral on critical — kept as reference points, not candidates.

## ⚠️ INSTRUMENT INTEGRITY (unresolved)
- **TT/cache contamination in `run_one`**: the move-search shares never-cleared file-scope caches across positions,
  so results are order-dependent (worst at low material — fable flagged it as the npm≤12 ruler-anomaly cause). The
  big signals (KS 40% move-change) dwarf it, but it muddies margins. FOLLOW-UP: a Cython cache-clear per position
  (needs a new exposed function + rebuild). Do with owner awake, then re-baseline.
- The criticality instrument GENERALIZES: toggle ANY eval term × (phase,material,criticality,eval-state) buckets =
  a per-term FAILURE ATLAS (which term helps/hurts which position cells). The guaranteed durable payoff.

## ▶️ NEXT ACTIONS (in order)
1. **Design the first REDISTRIBUTIVE KS experiment** (constant magnitude, sharper discrimination) — the only
   surviving direction. NOT additive. e.g. re-shape the unit-KS positive mix (demote proximity, raise weak/safe)
   at a HELD total magnitude; validate on the CRITICAL band of the enriched corpus. (Owner-in-loop: content design
   is where the agent's stories go wrong.)
2. **Finish the critical corpus** (overnight labeling running → `lichess_ks_labelled.csv`). Then validate every
   change on its CRITICAL bands, held-out train/val, never gradient-fit.
3. **Build the per-term FAILURE ATLAS** — ruler-criticality toggle for KS, threats, passers, capgains, imbalance on
   the standard sets. Tells us where the biggest regret lives across the WHOLE eval, not just KS.
4. **Games** only for an all-green-on-critical candidate, or the +15 SPRT fallback. No statistic predicts Elo.

## OVERNIGHT PLAN (2026-08-14, running/queued)
- RUNNING `bba5ry1hl`: SF18-label the 7500 Lichess KS corpus (~5h) → `lichess_ks_labelled.csv`.
- QUEUED (sequential phase-splits, no concurrency): the per-term failure atlas on the standard sets.
- HARD STOPS: no games (no all-green candidate), no commits, no gradient-fitting. Deterministic diagnostics only.

## DISCIPLINE (unchanged)
Byte-id every build. Validate on MOVES / CRITICAL band, both cross-sets. One phase-split at a time historically
(PID-fix now allows concurrency but verify). Runner form auto-approves; wrap python in wsl. Commit only when asked,
no footer. Games decide, run alone. Full results log: scratchpad `OVERNIGHT-RESULTS-LOG.md`.
