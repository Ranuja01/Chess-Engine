# SESSION HANDOFF 2026-07-23 — KS consolidation SHIPPED; KS over-read ROOT-CAUSED = per-king ASYMMETRY; PLAN approved

## ★★★ LATEST (2026-07-23 cont.) — READ FIRST. The plan is `.claude/plans/well-for-the-suppressor-wise-steele.md` (USE IT, don't write a new plan)
- **COMMITTED `6ac8855`:** KS consolidation (`evaluate_king_safety()` home + tunable shelter `KS_SHELTER_*`,
  gated `ENABLE_KS_V2`=false; dormant `MOD_KS_REALIZ`/`KS_REALIZ_FLOOR`) + **ev_breakdown CLEAN-PARTITION fix**
  (exposed `kaufman_imbalance` — was leaking up to ~6.8p; fields now sum to total → term diagnosis TRUSTWORTHY).
  On `4386414`(CAPG_PIN)+`f9ce2d2`(passer V3)+`6bd9d66`/`2a8d127`(Kaufman). Byte-id **247 / 39,971,153**.
- **ROOT CAUSE of the KS over-read = per-king ASYMMETRY** (not convertibility — that + breakthrough/zone/
  safe-check-dominant regime were all ruled out). Proven via **fable running the actual SF11 binary + tracing
  SF11/SF15/Ethereal**: SF11's "King safety −0.24" on FEN-1 (`8/1p6/4k3/1b2n3/p2q2P1/P3R2P/1P3Q2/3rN2K b`) is a
  NET — SF charges BLACK king −5.06 AND WHITE h1 king −5.30, ~cancel. **We charge black 33 units / white 4** →
  net +4.86 SIGN ERROR (SF18 −4.58). We UNDER-charge the attacker's OWN airy king. Two-sided ledger is SOLIDIFIED
  SF11→SF15 (SF15 −2.82/−3.00). See KS-map "BREAKTHROUGH" + the fable ground-truth in the map.
- **3 root causes (all `king_safety_danger`):** (1) safe-checks weighted 3-5 → a real queen safe-check reads like
  noise → `KS_FLOOR=13` HARD-ZEROS it; (2) transform is quadratic only below knee 12 then LINEAR → can't COMPOUND
  (SF proximity is EMERGENT: co-occurring signals summed then SQUARED `kd²/4096`; ours can't square a 30u swarm);
  (3) corner-king zone geometrically tiny (built on raw sq; SF clamps ring center → full 9). **Why past KS tuning
  was game-neutral:** the linear-above-knee clamp structurally destroys compounding = the local max.
- **MATERIAL stays OUT of KS** (correcting a session error: SF's `−6·score/8` is SHELTER not material; SF11/SF15/
  Ethereal all material-BLIND). Two-sided netting + the material term already skew the total → material-in-KS
  double-counts. NO blanket KS_DEFENDER/KS_OVERLOAD/central-penalty (banked failures).
- **APPROVED PLAN (asymmetry fix), gated default-off + byte-id, tuned multi-theme+multi-phase:** **A** per-type
  safe-check weights `KS_CHK_{Q,R,B,N}` sized so a lone Q/R check clears `KS_FLOOR` on its own (BRANCHLESS — floor
  untouched; targeting is the weights) + graded same-type bump `KS_CHK_MULTI` (our step beyond SF15's binary
  more_than_one); gate `ENABLE_KS_CHECK_V2`. **B** pure-quadratic transform = KNOB-ONLY (`KS_KNEE≥KS_CAP` ⇒ pure
  `u²/KS_DIVISOR`, existing mode, comment cpp:376) + RE-TUNE `KS_CAP`/`KS_DIVISOR` for a sane ceiling — coupled
  with A, ship/tune together. **C** clamped king-ring (`ENABLE_KS_ZONE_CLAMP`, cpp:561-566) — PHASE-SENSITIVE. **D**
  Ethereal zone-normalization (`KS_ZONE_NORM`, scale attacked-count by 9/popcount(zone)) — optional, overlaps C.
- **FIRST ACTIONS (single-core while user games; 4-core GAMES held for the overnight block, user signals):**
  R1 code A gated+byte-id → `ks_dump_one.py` FEN-1 white king un-floored (4→~16-20); R2 sweep B regime (knobs, no
  build) → FEN-1 sign fixed, OLD-3 blowup DOWN, OLD-4 (we beat SF11) HELD; R3 code C; R5 JOINT fit A+B(+C/D) with
  passer V3/Kaufman/OvD across all themes+phases + benches + move-match (cater to ALL, no theme/phase regress);
  R6 overnight games per-class verdict.
- **Acceptance:** old-3 (`3r4/ppp3Q1/nq2k2p/7N/3r2P1/2N1B2P/PP3P2/5bK1 w`, over-blow) DOWN toward SF18; **old-4
  (we beat SF11 — add to `ks_edge` guard) HELD**; this-window asymmetry set signs fixed; SF15 as 2nd reference.
- **Working tree (uncommitted, all byte-id 247):** diagnostic-only edits in `king_safety_danger` (a `breakthrough_sq`
  counter + `bkru`/`over` in the KSD dump; a `sc_dump` safe-check trace gated on env `KS_SAFECHK_TRACE`). Many new
  diagnostics + corpora (see below). Nothing committed since `6ac8855`.
- **SF ARCHIVE (all in the parent `Programming/Chess Engine/`):** SF11 (`stockfish_11/stockfish-11-win/src`),
  **SF15.1 = LAST classical-KS SF** (`stockfish_15_linux/stockfish_15.1_linux_x64/{src,stockfish-ubuntu-20.04-x86-64}`;
  classical via `setoption Use NNUE false`+`eval`; SF16/17/18 NNUE-only). SF11 linux binary for SF11-static:
  `stockfish_11_linux/.../Linux/stockfish_20011801_x64_bmi2` (the `.exe` in eval_vs_sf11.py is STALE + no Win-binfmt).
- **NEW DIAGNOSTICS (diagnostics/):** ks_dump_one (per-king KSD), ks_sf_trace + ks_sf11_term_survey (SF per-king),
  ks_regime_probe, ks_zone_analysis, ks_failure_hunt + ks_term_dossier (triangulation), ks_component_survey,
  breakdown_partition_check, mine_and_dissect, ks_build_fixable_tier, ks_v2_retier. **Corpus `ks_sets/
  diverse_corpus_ksplus.csv`** (603 rows: ks_fixable target / ks_edge our-SF11-edge guard / ks_real / families,
  phased). Method memory: [[sf11-sf18-triangulation-method]], [[ks-passer-joint-tune-discipline]].
- **RULED OUT this session (don't re-litigate):** convertibility/defense-gate rebuild, per-square breakthrough
  gating, zone-size restriction, safe-check-DOMINANT regime (made it WORSE — backwards separation), material-in-KS,
  MOD_KS_REALIZ/material-backing (raised phantom MSE), all KS coordination knobs (KS_ATT_PRODUCT/MIN_ATTACKERS/
  OVERLOAD — discrimination gap). The asymmetry + compounding is the FIRST mechanism-grounded, cross-validated fix.

---
## (older) passer V3 shipped (gated), CAPG_PIN committed, THEN = KS consolidation

Read this first, then `king-safety-subsystem-map-2026-07-23.md` (the KS map + CONSOLIDATION BLUEPRINT — the next
work), then `passer-doubled-hce-comparison-2026-07-22.md` (passer V3 story). Memory banner routes here.

## STATE (committed this session)
- **`f9ce2d2`** — passer centralization `evaluate_passers()` + composite-R + KS instrumentation, **gated
  `ENABLE_PASSER_V3=false`, byte-identical.** New byte-id ref (CAPG_PIN on): **WAC 247 / 39,971,153** (240 pin-off).
- **`4386414`** — CAPG_PIN shipped default-on (correctness: drops illegal pinned-piece captures; game-neutral,
  STS+28/WAC+7).
- Branch `NN-ENgine`. Working tree still has MANY pre-existing uncommitted files (dev-notes from prior sessions,
  modified scripts) — NOT this session's; leave them. This session's uncommitted-but-on-disk: the KS-map
  blueprint addition, `SESSION-HANDOFF-2026-07-23.md`, `_rename_base.py`, memory updates.

## WHAT SHIPPED: passer V3 (LAZY-ACCEPTED baseline, default-off)
Categorical GAMES win (3 seeds ×200g SF@2400): **positional collapse class −10% (59→53/seed, down all 3 seeds),
score +2.8% (won 2/3), STS +30.** One `evaluate_passers()` owns rank×composite-R (graded contest = our unique
lever: attackers−defenders vs SF's binary; + rear-file rook/queen; + king-proximity) + soft-gated king-race;
§6-A/rook-additive/aggregate-delta gated off; exports `priced_passer[]`. 9 knobs. Move-match +7, passers no
longer a top collapse driver (+0.07). **Treat V3 as the working baseline going forward** (test future items WITH
`ENABLE_PASSER_V3=1`); flip default-on + delete the scattered code + V2 once it holds across the next items.
Passer joint-tune (Stage-2 on `diverse_corpus`) + the set-capacity guard (`6k1/.../2ppp3/6B1`, per-side
saturation) remain as passer follow-ups.

## ✅ KING-SAFETY CONSOLIDATION — COMMITTED `6ac8855` (2026-07-23); realizability lever DISPROVEN; breakdown fixed
- **Committed `6ac8855`** (byte-id 247/39,971,153): `evaluate_king_safety()` one-home + tunable shelter
  (`KS_SHELTER_*`, `ENABLE_KS_V2` default-off) + dormant `MOD_KS_REALIZ`/`KS_REALIZ_FLOOR` + **ev_breakdown
  clean-partition fix** (exposed `kaufman_imbalance` — was leaking up to ~6.8p unattributed; residual now 0, so
  term-level diagnosis is TRUSTWORTHY). Reporting-only + gated ⇒ byte-id.
- **`MOD_KS_REALIZ` / material-backing DISPROVEN at scale** (`ks_realiz_sweep.py` on `diverse_corpus_ksv2.csv`):
  every config RAISED phantom MSE + broke guards. KS *is* the systematic driver (`ks_phantom_strict.py`: 41/45,
  ~59% of over-read) BUT phantom |KS|≈real |KS| and SF18 truth on high-KS ranges 0.00→+7.46 at the same |KS| ⇒
  the separator is TACTICAL convertibility (king escape / real safe checks), not material or magnitude ⇒ static
  eval historically can't (KS re-work game-neutral). Kept dormant; see KS-map "CORPUS VERDICT".
- **FEN dissections (clean breakdown):** FEN-1 `Q7/1p1k3p/...` (+20.6 vs SF +6.5) = ROOK-PLACEMENT over-credit
  (`pt_rooks +10.79` on open a/f-files) + `piece_value_boost +2.68` (material-domination) + `king_safety +3.24`
  — NOT doubled pawns. FEN-2 `r5k1/5p2/2P4p/...` (−2.55 vs SF +4.08, SIGN FLIP) = `king_safety −2.34`
  (det_ks_units_w=19) phantom, same class as FEN-3.
- **DONE this session:** SF11-static fixed (Linux binary `stockfish_11_linux/.../stockfish_20011801_x64_bmi2`
  via WSL — the `.exe` was stale + no Win-interop binfmt). Three-way triangulation working (`ks_failure_hunt.py`
  → ours/SF11/SF18, verdict FIXABLE / WE_BEAT_SF11). ROOT CAUSE confirmed systematic: **`KS_DEFENDER=0`**
  (defenders ignored in 100% of fixable-KS) + `KS_OPEN_FILE` on central kings + missing initiative offset; lone-
  attacker gate RULED OUT (queens present). See KS-map "ROOT-CAUSE CONFIRMED" + "SF11+ETHEREAL MODELS".
- **CORPUS BUILT: `ks_sets/diverse_corpus_ksplus.csv`** (603 rows) = `diverse_corpus_ksv2` + SF11/SF18-validated
  KS tiers: `ks_fixable` (36, tune target), `ks_edge` (4, WE-BEAT-SF11 guard = preserve our edge), plus existing
  `ks_real`/`ks_phantom`/families, phase-stratified, train/val split. Tools: `ks_failure_hunt.py`,
  `ks_component_survey.py` (KS_DEBUG_DUMP per-king), `ks_build_fixable_tier.py`, `ks_term_dossier.py`.
- **NEXT (code + multi-core, HELD while gaming):** implement the confirmed gated levers — (1) `KS_DEFENDER`
  netting, (2) `KS_OPEN_FILE` conditioned on king exposure [code], (3) NEW initiative/material offset, keep
  `KS_OVERLOAD` graded as uniqueness — default-off + byte-id, then JOINT-tune on `diverse_corpus_ksplus` WITH
  passer V3 + Kaufman + OvD across phases (extend `ks_fit_diverse.py` GRID), metric = root win%-accuracy vs SF18,
  guard `ks_edge`/`ks_real`/families, then 3-seed 200g SF@2400 per-class collapse profile. SOBER: KS re-work has
  been game-neutral before — the bet is the better instruments + joint method + the NEW initiative term.

## (historical) KING-SAFETY CONSOLIDATION — IMPLEMENTED gated + byte-id (2026-07-23), NEXT = the TUNING phase
Shipped gated (`ENABLE_KS_V2=false` default, WAC **247/39,971,153** gate-off AND identity-crossover; NOT committed).
Consolidation came out SMALLER than the blueprint after a byte-id finding (see the KS map's "IMPLEMENTED" section):
- **`evaluate_king_safety()`** = the ONE home (pure extraction of the inline unit-KS block; sole KS path now).
- **Tunable shelter in-place** (`ENABLE_KS_V2`): flat 185/75 → `KS_SHELTER_FULL/PARTIAL × KS_SHELTER_MAG/100`.
- **`MOD_KS_REALIZ`(+`KS_REALIZ_FLOOR`)** = whole-budget realizability gate with its OWN deeper floor → damps the
  phantom BELOW the 0.5× wall MOD_KS_BACKING saturates at (the real prize; live-verified, default-off byte-id).
- **★ BYTE-ID FINDING (re-scoped the plan):** `evaluate_kings_midgame`'s return is phase-BLENDED (41-69) AND feeds
  `square_values[r]` read in capture ordering (~7683) ⇒ relocating the shelter/`baseIncrement` king slice OUT to the
  post-loop home is NOT byte-identical. So the "de-king / re-home baseIncrement / O/D-decoupling" steps were DROPPED
  — unnecessary (unit-KS was already ONE post-loop term) AND contradict the scope correction (baseIncrement IS the
  placement lens we keep). The O/D-decoupling trap never had to be entered. NO OvD/placement surgery was done.
- **NEXT = TUNING, not structure:** build the KS corpus tier (phantom over-fires + real-attack guards) into the
  existing mixed set → joint-fit KS+OvD (passer V3 ON, +material), win%-space → RE-FIT `KING_SAFETY_MAG` up as
  `MOD_KS_REALIZ` takes over phantom control → per-class collapse profile in games (HOLD multi-core while gaming).

## ★ (SUPERSEDED by the IMPLEMENTED section above) KING-SAFETY CONSOLIDATION — original blueprint
Residual v3 positional collapses = **we OVER-READ our own attacks** on exposed kings. Diagnosed to code:
king-zone pressure is **TRIPLE-COUNTED** across 3 live, unconditioned layers — unit-KS (`king_safety_danger`,
×30 via `KING_SAFETY_MAG=3000`, `KS_MIN_ATTACKERS=0` ⇒ lone queen fires) + `attackingLayer` (king-centric, ×5
`ATTACK_OPEN_MULT` on open sq) + OvD imbalance re-spend (×`IMBALANCE_SCALE=3`). None asks "does it convert."
- **[E] FEN-3 differential** (`3r4/ppp3Q1/nq2k2p/7N/3r2P1/2N1B2P/PP3P2/5bK1 w`): our +10.75, `KING_SAFETY_MAG=0`
  → **+3.55 ≈ SF18 +3.04** — the ENTIRE over-read is the KS term. FEN-4 (`2n2b1r/1pB1k3/1p4Qp/1N6/4P3/Pn2P3/
  1q2BP1P/5K2 w`, correct, SF +8.69) survives on MATERIAL (KS=0 → +7.82), not KS.
- **USER'S KEY CONSTRAINT**: FEN-3 KS is DIRECTIONALLY CORRECT (SF sees +3 despite white down material — attack
  beats material). Goal = **blowup magnitude CONTROL toward SF's modest positive, NOT killing KS**; keep #4 fully
  credited, keep #3 positive. Bidirectional discipline like passers.
- **[E] STEP-0 PROBE**: `MOD_KS_BACKING` (material) DISCRIMINATES — damps #3 (−3.6) > #4 (−1.35), both toward SF;
  saturates too low (fix needs a STRONGER gate on the WHOLE budget). `MOD_KS_CONTROL` = WRONG direction
  (amplifies — circular, built from the same attackingLayer). `KS_MIN_ATTACKERS≥2` = no-op for multi-attacker.
- **⚠️ SCOPE (user chess insight — read the KS map's "SCOPE CORRECTION"): the 3 layers are TIME-HORIZON LENSES,
  not pure dupes.** attackingLayer = PLACEMENT/piece-direction (central attacks, x-rays — KEEP, do NOT de-king);
  OvD = LONG-TERM potential (KEEP, coordinate KS+OvD as a near/far pair — NOT the over-read); **KS = oncoming
  storm = the ONLY over-read** ([E]: FEN-3 KS=0 → +3.55 incl OvD +2.16 ≈ SF +3.04, so OvD/placement were correct).
- **PLAN (revised, SAFER): realizability-gate the KS (oncoming-storm) lens ONLY** — its credit should scale with
  whether the storm converts (step-0 `MOD_KS_BACKING` probe works, needs to be stronger/on the whole KS budget) —
  + co-tune KS with OvD (near/far). This LIKELY AVOIDS the O/D-decoupling trap (we're not ripping king credit out
  of placement/OvD). The full "de-king attackingLayer / purge OvD" teardown is SUPERSEDED. Optional: finish the
  `KS_CONSOLIDATE` shelter-dedup (185/75 vs KS_SHIELD) — but that DOES touch the O/D writes, so gate + byte-id
  carefully. REBALANCE: round-based (KS-only corpus Stage-1 → mixed joint Stage-2, win%-space). REOPEN gated
  items (MOD_KS_BACKING/KS_MIN_ATTACKERS/KS_AIM) on merit.
- **SOBER**: KS re-work has been GAME-NEUTRAL every time → expect a tunability/cleanliness win, verify byte-id +
  per-class collapse profile, don't bank Elo. Method: build KS corpus (phantom over-fires + real-attack guards,
  SF18-labelled) as the deterministic instrument FIRST.

## SECONDARY / HOUSEKEEPING
- **Doubled-pawn** (structural): condition our flat 125mp penalty on unsupported/immobile + eg-weighted, BOTH
  phases (SF `S(11,56)` !support eg-heavy; Ethereal `PawnStacked` half-if-unstack). Don't cut pawn value.
- **`ENABLE_KS_DEBUG` dump is in the DEAD `latent_threat` fn** → never fires. Relocate into the LIVE
  `king_safety_danger` to get per-king unit sub-components.
- The KS-map corrected an earlier error: the `+4 offset/×2 doubling/+75` fn is DEAD `latent_threat` (KS v1
  replaces it) — do NOT analyze it as live.

## DURABLE FACTS / GOTCHAS
- Runner (LITERAL, auto-approved): `wsl.exe -e bash -lc "bash '/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/
  Chess Engine/Chess-Engine/NN Engine/selfplay/overnight_runner.sh' <sub> …"`. Subs: build/wac<TAG>/sts<TAG>/
  pyrun/gauntlet. **`pyrun` passes args as sys.argv, NOT env** — a diagnostic must self-parse KEY=VAL→os.environ
  BEFORE importing ChessAI (passer_verify/dissect_fen/passer_movematch do; term_dump/movematch do NOT).
  wac/sts use `env "$@"` so KEY=VAL reaches env there.
- Games: `pyrun selfplay/vs_sf.py --sf-elo 2400 --games 200 --concurrency 3 --our-config '<knobs>' --tag <t>
  --seed <s> --adjudicate-draw`. Tag=dir name (NOT tag+seed) → use distinct tags `{cfg}_s{seed}`; `collect_
  collapses` parses seed from the `_s<N>` suffix. `vs_sf` writes collapses.csv in `w` (overwrite). Watch stale
  prior-session dirs in `selfplay/games/`.
- Collapse profile pipeline: `pyrun diagnostics/collect_collapses.py` → `classify_collapses.py` →
  `profile_collapses.py base v3 SEEDS=0,1,2`. Judge by per-CLASS matched-seed drop (not total).
- New env knob needs BOTH the `inline` in search_engine.h AND `env_flag/env_int` reg (search_engine.cpp ~1071)
  or env is ignored. commit only when asked; NO commit footer (project convention). Single-core when user games.
- Eval is absolute Black-positive millipawns; non-negamax. `ai.ev_breakdown(board)` dict exposes term fields
  incl. `det_ks_units_w/b`, `king_safety`, `central`, `imbalance_white/black`, `capture_gains`, `pt_*`.
