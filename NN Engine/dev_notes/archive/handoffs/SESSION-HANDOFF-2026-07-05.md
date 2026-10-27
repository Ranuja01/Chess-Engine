# ▶️ SESSION HANDOFF — read FIRST (2026-07-05)

Continuation of the holistic-eval campaign. Engine byte-id **245 / 39,146,294** (all new work is python/env-gated,
nothing rebuilt), nothing running, tree clean, branch NN-ENgine unpushed, NO commits. This is the index; detail in
`holistic-cotune-campaign-2026-07-05.md` (the living spec) + memory [[holistic-eval-pivot]] +
[[realizability-conditioning-architecture]].

## THE ARC (how we got here)
Isolated king-safety tuning does NOT convert to Elo (neutral/−everywhere) even consolidated + 4×-better-rep + faster.
→ Reframe (user): it failed from **ISOLATION**, not badness — eval is ONE vector; a term tuned against a frozen rest
is incoherent. → Grounded: **equal-depth vs SF11 = 7.5% ≈ −437 Elo PURE EVAL gap** + 35 collapses (24 EVAL holes by
triage). Eval is massively unmined; NNUE NOT next (close the classical gap first, off our OWN eval). → Built a
**holistic position-conditional co-tune pipeline** and ran instance #1 (king-attack cluster). GUIDING ARCHITECTURE:
every term = `value × realizability(cheap detectors)`, tuned per-term data-driven (capg generalized to the whole
eval). Fable rounds 1+2 source-verified.

## ✅ WHAT'S BUILT (all python, byte-id untouched)
- `diagnostics/build_cploss_corpus.py` → frozen 824-pos stratified corpus `selfplay/tune_data/cploss_corpus.csv`
  (collapse24/sts350/neutral200/game250; de-biased).
- `diagnostics/cploss_frozen.py` + sub `cploss_frozen <our_d=10> <judge_d=12> <lim=0> <shard=all> [KNOBS]` =
  **WDL-cploss compass** (winpct loss vs **SF18 judge**, fixed depth = deterministic, cached, per-stratum, shards).
- `diagnostics/surrogate_tune.py <spec> <tag> [--evals 80] [--ridge 0.5]` = pattern-search driver (Hooke-Jeeves +
  trust-region + ridge), BALANCED-stratum objective, train-fit → holdout-validate, prints node_ab config. Reuses
  spsa `_cfg_string`/`_clamp`. Logs `results/<tag>_log.csv` + `_state.json`.
- `selfplay/eval_cluster.json` = v1 KS-CONDITIONING spec (base = dedupe+enables; params = KING_SAFETY_MAG + the
  conditioners KS_DYN/KS_DYN_PIVOT/MOD_KS_BACKING/MOD_KS_CONTROL + SCALE_ATTACK_LAYER/SCALE_THREATS/IMBALANCE_SCALE).
- **JUDGE ROLES (locked, user):** SF18 = MOVE judge (cploss — drift-to-perfection, regret/argmax≠magnitude-target).
  SF11 = per-term EVAL breakdown (incremental_validity dual-screen + breakdown_gap, apples-to-apples handcrafted) +
  equal-depth venue.

## ⚠️ STAGE-1 RE-READ (2026-07-05, move-flip diagnostic) — collapse/neutral story RETIRED; game-stratum is the real signal
Built `diagnostics/moves_dump.py` + `move_flip_report.py` (+ `moves_dump` dispatcher sub) to decompose the cploss
delta per position. Findings (see [[compass-context-fragility]]):
- **The cploss compass is TT-context-fragile.** `run_one` does NOT clear the global TT/eval/history/move-gen caches
  between positions (only killers/counters/stacks) → a position's move depends on corpus context. The winner's
  per-stratum delta and even the BALANCED sign FLIP between all-context (+3.39 = winner worse) and holdout-context
  (−1.29 = better), entirely on the tiny collapse stratum.
- **RETIRE "collapse crushed ~3× / neutral regressed."** The clean holdout run DOES reproduce the handoff numbers
  (collapse 11.10→8.54, neutral 21.99→23.63, game 26.76→22.55, bal 23.87→22.59) — but across contexts collapse leans
  neutral-to-worse (all +15.81) and neutral leans better (all −2.79). Both are small-n + context artifacts.
- **The ONE robust signal: the winner improves GAME positions** (real game FENs; better in all 3 context views:
  all −1.95, holdout-subset −10.11, holdout −4.21; n≈122–242, ~37% of moves shift net-better). THAT is the genuine
  (modest) accomplishment and the honest reason to run the game gate. Compass PROPOSES → games DECIDE (doubly so now).
- Discipline: run the compass with a CONSISTENT context; never compare a per-shard number to an all-shard number
  (my invalid first pass); trust large-n strata only.

## ✅ STAGE-1 RESULT (as originally logged — read WITH the re-read above) — conditioning WORKS + narrow-limit CONFIRMED
WINNER: `KS_CONSOLIDATE=1 ENABLE_KS_REPLACE_LT=1 KS_ZONE2=1 ENABLE_THREATS=1 KING_SAFETY_MAG=3250 KS_DYN=512
KS_DYN_PIVOT=4 MOD_KS_BACKING=256 MOD_KS_CONTROL=32 SCALE_ATTACK_LAYER=100 SCALE_THREATS=100 IMBALANCE_SCALE=3`.
Holdout GENERALIZES; vs DEFAULT eval same holdout: balanced 23.83→**22.59** (−1.24 net). Per-stratum:
**collapse 11.10→8.54 (CRUSHED ~3× the eval holes)** · game 26.58→22.55 · sts flat · **neutral 21.99→23.63 (WORSE)**.
⇒ (a) conditioning CONVERTS on the compass where flat KS never did (dynamic conditioners drive it; MOD_KS_CONTROL
tuned to a small 32); (b) the neutral REGRESSION empirically CONFIRMS the make-room/over-valuation confound (user) —
narrow cluster has no capgains/placement to give way → hotter KS over-reads quiet positions. Validates mechanism AND
proves the broad cluster is needed. Compass = PROXY → games decide (−202 lesson).

## ▶️ NEXT ACTIONS (new context, priority order)
1. **node_ab gate the winner** vs base: `node_ab <mins> 250000 '<WINNER cfg>' '' 4 ks_gate` → transfer-ratio
   (gate-Elo / compass-gain). CAVEAT: KS is NPS-non-neutral + node_ab is depth-biased → treat as DIRECTIONAL; the
   FAIR venue is a **blitz** SPRT (`gate_blitz '<cfg>' kscond`). The flywheel + true balance only show at a TIME gate.
2. **Stage-2 flywheel** (the possibly-bigger half): re-tune the eval-coupled margins `RFP_MARGIN`/`RFP_MAX_DEPTH`/
   `ASPIRATION_DELTA`/`LMP_BASE` TO the cleaner eval (compass can't see "prune harder → deeper at equal time").
3. **BROAD "king-attack budget" co-tune** = the make-room fix (the neutral regression proves it's needed): add
   capgains (SCALE_CAPTURE_GAINS + capg-tension) + placement + OvD-conditioners (REALIZ_*) + mobility to the cluster;
   bigger 2–4k corpus; random restarts (jagged landscape). Then generalize the architecture to OvD → threats → rooks.

## ★ KEY INSIGHTS (user, carry)
1. Tune the CONDITIONING FUNCTION, not bare magnitudes (flat = best-on-average, over-fires fantasies). =[[realizability-conditioning-architecture]].
2. BALANCE/make-room: raising KS needs neighbors to give way or the TOTAL over-reads (KS looks wrong when fine);
   fixed-depth compass can't see over-value×pruning → gate the JOINT in games. (Neutral regression = this, live.)
3. More points = BREADTH (subsystems accumulate) + FLYWHEEL (search) + richer DETECTORS (build shelter_storm/
   safe-mobility — biggest classical juice) — NOT data (data = the NNUE limiter; SF11 +437 = classically reachable).
4. Single per-term numbers are ambiguous (my-king-safe vs their-king-unsafe; strength-vs-style on biased samples);
   the SHAPE/relative balance is the signal. Our KS is a differential — condition PER-KING (we do).

## FUN / FUTURE (prepared, not now)
- We HAVE SF1.x (whole 1.0→1.9.1 line, 2010 ja builds) + SF11 + SF18, several WITH SOURCE. **(a) SF1-full benchmark**
  (absolute low rung of the ladder SF1→SF11→SF18; natural not artificial-Elo-cap; build SF1.x from src for native
  Linux; 5-game UCI smoke first). **(b) Data-generation flywheel** = lightning games vs SF → mine collapses+wins →
  triage EVAL(eval corpus)/PRUNING+HORIZON(search targets) → feed tuning → repeat (machinery exists: vs_sf/
  label_collapses/triage). **(c) FABLE SOURCE-EVOLUTION ASK** — have Fable investigate SF1→SF11→SF18 source across
  eval/movegen/caching/search → concrete upgrades vs our engine → ADD to the Fable info bank (external-audit-2026-07-03
  + engine-vs-sf11-map-2026-07-04). Source-verify all. Detail in the campaign doc's FUTURE section.

## DISCIPLINE / GOTCHAS
byte-id 245/39,146,294 after any build. Dispatcher ONLY as `wsl.exe -e bash -lc "bash '<abs overnight_runner.sh>'
<sub> …"` (auto-approved; `=`-knobs after sub OK). ≤4 workers; never parallel heavy engine+games (OOM); recover with
`pkill -9 stockfish; pkill -9 -f <specific>` — NEVER blanket-kill python (API project) or `wsl --shutdown`. Bash tool
= Git Bash (no /mnt/c → use wsl.exe for FS ops, or the Read/Glob tools). ScheduleWakeup UNRELIABLE → use a background
`sleep` timer for periodic checks. Commit only when asked; no footer. Leave stray non-ours files. Compass PROPOSES →
node_ab GATES → SPRT SHIPS; no proxy ships alone (−202). Pointers: `holistic-cotune-campaign-2026-07-05.md`,
`ks-build-plan-2026-07-04.md`, memory [[holistic-eval-pivot]] [[realizability-conditioning-architecture]]
[[fable-audit-2026-07-04-fixed-nodes-pivot]].
