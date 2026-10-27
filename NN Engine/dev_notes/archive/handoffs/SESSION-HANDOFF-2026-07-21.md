# SESSION HANDOFF 2026-07-21 — Kaufman SHIPPED; collapse-classifier tooling; STS-artifact correction; investigation open

**NEW-CHAT ENTRY POINT. Read this first**, then `collapse-reduction-ledger.md` (2026-07-21 entries),
`position-bank-schema-2026-07-21.md`, `kaufman-imbalance-2026-07-20.md`. Method memory:
[[eval-collapse-diagnosis-method]], [[collapse-categorical-verification]], [[sts-wac-tag-artifact]].

## LATEST STATE (2026-07-21 cont.2) — read before the Kaufman section below
- **SHIPPED/COMMITTED: Kaufman default** (`6bd9d66`; ref WAC 41,586,391/240). Nothing else committed yet.
- **WORKING TREE (uncommitted, pending night games): `ENABLE_CAPG_PIN=true`** — THE win of this phase. capgains
  was counting ILLEGAL captures by absolutely-pinned pieces; the fix was implemented but gated off. Correct +
  **STS +28 (1527→1555) AND WAC +7 (240→247)** — improves both. **Working-tree byte-id ref: WAC 247 / 39,971,153.**
  Night: game-test Kaufman+CAPG_PIN, then commit if it holds.
- **KS recalibration = HELD (not shipping yet).** Bounded ~31% detection recovery (floor-relax+safe-check+
  attack-count); closes the KS under-read half (|ourKS| 0.44→0.76 vs SF11 1.13). But it **regresses STS on the
  pin-on build (1555→1467, −88)** where it was +22 on pin-off — the win%-fit had NO STS in its objective, so the
  config's STS was uncontrolled (+ STS is jagged). NEEDS re-tune on pin-on WITH an STS guard before shipping.
  **CORRECTION (do not over-dismiss KS):** the ~31% is a real KS-ACCURACY gain on the attack corpus (different
  metric/positions from STS); STS is a jagged PROXY, not the arbiter — GAMES decide. So game-test KS (does the
  detection gain cut collapses despite the STS caution?), don't kill it on STS. **NEW METHOD LEVER (user):** add
  the STS/WAC bench FENs as a VALIDATION TIER in `ks_fit.py` — the fit shifts move-choice on non-attack positions
  the wrong way, so guarding on bench FENs (not just bank win%) stops the STS regression at fit time.
  All coordination variants (product/gate/KS_OVERLOAD) = NO-GO (overshoot KS-active controls; DISCRIMINATION gap
  not range — units don't separate danger 12u from safe 10u, SF 15×; fit REJECTS more range).
- **CORRECTIONS (user caught):** "piece-activity over-read" = cumulative-snapshot ARTIFACT (`pieces` breakdown
  key includes material); OvD-reduction lead DEBUNKED (imbalance = 4% of over-reads; fit's IMBALANCE_SCALE↓ was
  SF18-target noise); Kaufman CLEAN. Remaining over-reads after CAPG_PIN are SF18=±99 MATES = search-bound.
- **SYSTEMATIC EVAL MAP vs SF11 (`eval_term_comparison.py`, mean |mag| pawns) — THE ROADMAP:** we OVER-read
  **Space/central ~9×** (0.46 vs 0.05) — the one clean over-read → fit `CENTER_*_MULT` down. We UNDER-read/LACK:
  **KingSafety** (0.44 vs 1.12), **Threats** (0.00 vs 0.34 — `ENABLE_THREATS` OFF), **Passed** (0.06 vs 0.33),
  **Mobility** (0.05 vs 0.23 — `ENABLE_MOBILITY` OFF), **Imbalance/OvD** (under, not over). Material/piece rows
  are boundary-fuzzy (SF folds PSQT into Material). Next levers: Space-down, enable+fit Threats/Mobility, Passers-up.
- **TOOLING (reusable):** `build_position_bank.py`(1852 labeled, 33,683-game archive)+`add_sf18_labels.py`;
  `ks_fit.py`+`_ks_fit_eval.py` (win%-space constrained fit, Lichess k=0.00368208; ADD STS guard next);
  `eval_term_comparison.py`, `dossier_overread.py`, `capg_pin_scan.py`, `refresh_bank_ours.py`.

## Headline: Kaufman shipped as default

## Headline: Kaufman shipped as default
- **Commits `2a8d127` (gated scaffolding, byte-id) + `6bd9d66` (flip ENABLE_KAUFMAN_IMBALANCE default true).**
- **NEW byte-id reference (Kaufman active): WAC 240/300 solved, node total 41,586,391** (was 243 / 39,914,378).
  Use this for all future byte-id checks. Kaufman on = shipped default; drive OFF with `ENABLE_KAUFMAN_IMBALANCE=0`.
- Evidence: 6-seed 200g vs SF@2400 = **+1.15% score, ks_attack collapse class down 6/6 seeds (-20%), total
  collapses -6.8%**. Scale-50 tested overnight = equal-within-noise on score but only 4/6 on the target class,
  so full scale (100) ships. Kaufman = FIRST game-positive eval lever of the campaign.

## Engine / conventions (unchanged)
Non-negamax C++ HCE in `NN Engine/` (min/max/pre_min; absolute eval **Black-positive**, single root flip;
values {0,1000,3250N,3450B,5000R,10000Q,12000K}; mate ±9,999,999). Real strength ~2000; SF@2400 = venue.
LITERAL runner path only: `wsl.exe -e bash -lc "bash '/mnt/c/.../NN Engine/selfplay/overnight_runner.sh' <sub>
[KEY=VAL]"`. Subs: build · wac <TAG> · sts <TAG> · gate · pyrun · gauntlet. **`wac`/`sts` REQUIRE a TAG as
arg1** (see artifact lesson below). Games: `pyrun selfplay/vs_sf.py --sf-elo 2400 --games 200 --concurrency 3
--our-config '<knobs>' --tag <t> --adjudicate-draw --seed <s>`. conc3=~4 cores/~76min per 200g.

## NEW TOOLING this session (validated, reusable)
- `diagnostics/collect_collapses.py` → unified collapse dataset from ALL selfplay/games/*/collapses.csv,
  tagged by config family + seed → `ks_sets/collapse_dataset.csv` (was 6891 collapses / 117 dirs).
- `diagnostics/classify_collapses.py` → per-position class via python-chess geometry (ks_attack / ks_and_material
  / material / positional), + `--vanish A B --seed N` cross-run attribution (deterministic games ⇒
  (family,seed,game,color) = SAME game across runs → "which collapses vanished/appeared"). Writes
  `collapse_dataset_classified.csv`.
- **KEY EMPIRICAL FINDING: the per-CLASS signal is STABLE across seeds where TOTAL is noise.** Kaufman cut
  ks_attack 6/6 seeds while total churned ±15/seed (even rose at one seed). ⇒ judge levers by per-class
  per-seed consistency, NOT total collapse count. This is our trustworthy overnight metric.

## Collapse profile AFTER Kaufman (the investigation entry point)
Over 6 seeds (ship config): **positional ~360 (DOMINANT), ks_attack ~106, material ~15.**
- KS still live (~18 ks_attack/seed) → KS redesign justified, BUT
- **positional is 3.4× the ks_attack class → the BIGGER collapse opportunity is likely POSITIONAL, not KS.**
- Extracted surviving KS-attack DECISION fens → `ks_sets/post_kauf_ks_collapses.txt` (115). CAVEAT: our-
  classifier heuristic, NOT SF18-validated (old dossier was 64% artifacts). SF18-validate before trusting.

## CRITICAL METHODOLOGY CORRECTION (banked)
- **`wac`/`sts` runner subs consume arg1 as the TAG.** Passing `sts KNOB=1 ...` puts the knob in the tag →
  env NOT applied. **Every "STS identical/held" claim earlier this session was this artifact.** Correct STS
  (proper tag): baseline 1606; Kaufman -79; KS_WEAK=3 -82; aim1/2/3 -103 — ALL THREE regress STS. GAME data
  was VALID (uses vs_sf.py --our-config, verified via stderr toggle-dump), so Kaufman's game win stands.
- **STS is DETERMINISTIC but JAGGED in eval-scale** (Kaufman scale 25/50/75/100 = 1575/1573/**1448**/1527 —
  75 anomalously worse than 100). ⇒ STS is a GUARD, not a tuning optimizer. Games decide scale.

## KS levers NOT ready (deferred to dissection)
- **aim (`ENABLE_KS_AIM`)**: -103 STS AND net-zero on its own ks_attack class in games (13 fixed / 13 created
  at seed 0) = counterproductive as built. Root hypothesis (user): DOUBLE-COUNTS with existing detection
  (attackingLayer / attacker counts / latent_threat heritage) → over-fires on non-threats. Fix = enumerate
  overlap, gate aim to only the MARGINAL latent lines nothing else sees. NOT a weight sweep.
- **KS_WEAK=3**: -82 STS (lowers effective floor, broad noise). Ungamed. Same "fire on non-threats" family.
- Coffin (`KS_INTERACT`) already DROPPED (structural, 2026-07-20).

## Kaufman scope truth (important for the pawn motivation)
Kaufman features = PURE piece-count products (census). It reprices pawns by the material MIX (imbalance
pricing — half of the original endgame-pawn motivation) but has ZERO knowledge of PASSEDNESS / rank /
advancement. The endgame passer-misranking is a SEPARATE axis (passer subsystem, mostly NO-GO). **Real fix
for the original motivation: extend the Kaufman-style ridge data-fit with PASSED-PAWN-BY-RANK features** →
fit our-own passer coefficients from the SF residual (heir to Kaufman's method). Queued.

## FIRST ACTIONS next window
1. **The investigation is OPEN** (user-steered): assemble past-tried FENs (the levers whose "fixes" caused
   bigger issues elsewhere — passer-v2, npedge, aim, coffin) + new post-Kaufman collapses; SF18-validate;
   understand WHAT is still broken and WHY past fixes regressed elsewhere (the CHURN/double-count theme —
   fixing one class exposes the next; over-additive terms double-count existing signal).
2. Weigh next priority: **positional-class dive** (dominant, ~360) vs KS double-count dissection (~106).
3. Deferred: aim double-count dissection; passer-by-rank data-fit.
