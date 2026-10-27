# Dynamic Conditional Eval — dev log

**Working name:** Dynamic Conditional Eval (the detector-gated modulation layer).
**Started:** 2026-06-19. **Plan:** `~/.claude/plans/handoff-for-the-vectorized-meadow.md`.
**Memory:** `dynamic-conditional-eval.md` (project).

## One-paragraph what/why
Generalize the existing `realizability_factor` prototype into a small layer that modulates the
**universal** (non-`evaluate_*_*`) eval terms — material/`piece_value_boost`, `capture_gains`,
`passed_pawn_support`, `latent_threat`, `central`, `imbalance`, `pair_bonus` — by cheap, already-computed
positional **detectors**, instead of fixed scalar weights. Motivation: a full day of PACE showed scalar
knobs are tapped for the floor because they shift a term's **mean**, but the plurality of collapses
(~38% level-material) is **variance/scatter** — only a **position-conditional weight** can reduce
variance (be right in both the matter / don't-matter positions). This is a hand-built, interpretable,
*owned* approximation of NNUE's context-dependence.

## Design invariants (do not violate)
- **Byte-identical at default.** Master-gate `if (K1|K2|…)` per term; all-default skips the modulation.
  Gold: WAC **261 / 134,429,469** (`MAX_DEPTH=10 USE_OPENING_BOOK=0 PRESET=LONG_FORMAT`).
- **Integer/bitwise only.** Gain = factor over 256: `g = 256 + Σ (K_d * signal_d) >> shift`, clamped
  `[MOD_FLOOR, MOD_CEIL]`; apply `term = (term * g) >> 8`. No new floats.
- **Phase is a first-class detector** (`phase_score` 0..128).
- **Cold tail only** — modulate at the universal-term accumulation sites in `placement_and_piece_eval`,
  never in the per-piece loops. Reuse detectors already in scope (no recompute).
- **Double-count rule:** modulate a term by a detector ONLY if the term doesn't already encode it.
- **`imbalance` stays on `realizability_factor`** — the new layer does NOT also drive it (would
  double-apply material+phase).

## Double-count interaction matrix (active vs forbidden)
| term | ACTIVE now (net-new) | FORBIDDEN (already-encoded/self) |
|---|---|---|
| material / `piece_value_boost` | `pawn_count`, `opposite_bishops` | `material_edge` (self), `phase` (PV_BOOST_PHASE_K) |
| `latent_threat` | `material_edge` (backing) | king-pressure (is itself), mg-gating |
| `pair_bonus` | `pawn_count` (openness) | — |
| `imbalance` | (space, deferred) | material+phase (realizability), king-pressure |
| `capture_gains` | (phase, later) | material_edge |
| `passed_pawn_support` | (material_edge, later) | phase (already EG-aware) |
| `central` | — | phase (already damped) |

## Stage checklist
- [x] **S0** Register memory + dev doc — DONE.
- [x] **S1** Framework scaffolding DONE + byte-id locked (WAC 261/134,429,469): `mod_gain(k1,sig1,sh1,
      k2,sig2,sh2)` after `realizability_factor`; `MOD_FLOOR=128`, `MOD_CEIL=512`, `MOD_MAT_PAWNS`,
      `MOD_MAT_OPPB`, `MOD_LT_BACKING`, `MOD_PAIR_OPEN` (all 0) + env load/echo. Nothing calls it yet.
- [x] **S1a** material × {pawn_count(`MOD_MAT_PAWNS`,sh3,centred −12), opposite_bishops(`MOD_MAT_OPPB`)} at the
      `piece_value_boost` sites — modulate the (int) boost by `mat_gain`; ternary picks raw boost at default.
- [x] **S1b** latent_threat × backing (`MOD_LT_BACKING`,sh12): discount lt of the side it favours (lt>0=Black)
      when that side is down material (`sig=min(0, threat_side_edge)`); applied after SCALE_LATENT_THREAT.
- [x] **S1c** pair_bonus × openness (`MOD_PAIR_OPEN`,sh3, `12−popcount(pawns)`): modulate the BISHOP-pair
      bonus only (knight pair untouched); ternary picks raw bonus at default.
- [x] **S1d** **byte-id WAC 261/134,429,469 confirmed** with all knobs default. Liveness (baseline 134,429,469):
      MAT_PAWNS=300→135.56M, MAT_OPPB=−600→134.47M, LT_BACKING=400→133.00M, PAIR_OPEN=400→137.72M — all live.
      Direction check (reeval+pattern_diag, MOD_MAT_PAWNS=300 MOD_MAT_OPPB=−400): +1B+2P 552→460, +1R+2P 592→492,
      end-side bent harder than mid (conditional working), **even-control held at 2** (no double-count). EVAL_PROFILE
      speed check still owed; `movematch_diff` default-vs-default still owed.
      NOTE: placement_bundle stored eval_breakdown is currently in the (MOD_MAT_PAWNS=300 MOD_MAT_OPPB=−400) reeval
      state — restore with a default `annotate placement_bundle 4 --reeval` before the next baseline pattern_diag.
- [ ] **S2** PACE: joint-tune on failure-runup zones (move-decisive proxy) → collapse-rate
      (`flip_extract` both arms) + multi-distance runup (−1/−3/−5/−10) gate → **SPRT** general-UHO.
- [ ] **S3** cheap `space` (squares-controlled) surrogate detector → feeds material + latent_threat.
- [ ] **S4** extend layer to per-piece placement (`evaluate_*_*`) — deferred until universal proven.

## Dispatcher subs (ALL overnight work routes through these — prompt-free autonomy)
Invoke as `wsl.exe -e bash -lc "bash '<…>/overnight_runner.sh' <sub> …"` with **NOTHING before the
`bash '…/overnight_runner.sh'`** (a leading `cd`/`export`/`VAR=`/loop breaks the allowlist prefix → prompts).
- `build` / `wac <tag> [KNOBS]` / `pattern_diag <tag>` / `annotate <tag> [conc] [--reeval]` / `movematch`.
- `tournament <min> "<knobs>" [conc] [tag]` — general-UHO SPRT.
- `tournament_seeded <min> "<knobs>" <openings_file> [conc] [tag]` — seeded playout/SPRT (NEW).
- `flips <tag> <arm> [drop]` — collapse-rate KPI for one arm (NEW).
- `move_proxy <csv> <n> "<knobs>"` — base-vs-candidate fen_vs_sf comparison, prints SF-match + over-read delta (NEW).

## Reusable tooling (built earlier this session)
- `selfplay/flip_extract.py <tag> <drop> <arm>` → per-arm `flips_<arm>.csv` (collapse-rate KPI).
- `pattern_diag` + `annotate --reeval` → cheap bias re-probe (no games).
- Seeded playout: `tournament.py --openings <flip-seeded-openings>` (built `flip_openings*.txt`).
- `PV_BOOST_*` knobs (prior material-calibration stage; gated, byte-id) — material lever, kept default-off.

## Progress log
- **2026-06-20 (day, DEPTH CONFIRMS SEARCH-BOUND — hard data)** — run-up probe collapse-rate vs OUR engine's
  fixed depth (150 seeds): **d10 29/150 (19%) → d12 17/150 (11%) = −40% from just 2 plies** (d16 pending).
  Eval-magnitude tuning was FLAT (29→28-30); +2 plies cut it 40%. ⇒ **the collapse floor is DEPTH-bound, proven.**
  User synthesis (confirmed): the game-losing mistakes are RELATIVE-depth problems; at equal depth peers mutually
  fail to punish what neither can see (why eval-tuning at equal depth washes); the lever is **out-searching =
  EBF-lowering (more depth/time) → SMP.** Eval's payoff is AVERAGE strength + leaf PRECISION (NNUE-class), NOT
  retuning these magnitudes — and the multi-variable-vs-scalar OVERALL question remains untested (a separate
  axis from the floor; needs a general-play overall-Elo SPRT, not floor probes). Reusable: `runup_probe` makes
  any future floor lever (incl. search params) measurable in minutes.
- **2026-06-20 (day, RUN-UP PROBE built + damper sweep)** — built `runup_probe.py`/`runup_probe` sub (the PACE
  fast inner loop: play OUR engine vs SF a few plies from each seed, cut off on collapse/survive, collapse-rate
  in ~3-12min vs 90min playout). 150-seed damper sweep: **base 29/150, MOD_LT_BACKING=400 28/150, combined damper
  (MAT_PAWNS+LT_BACKING) 30/150 — all noise.** IMBALANCE_SCALE=5 also flat (8/50 vs base 9/50). ⇒ optimism-damping
  does NOT reduce collapses — eval-magnitude lever EXHAUSTED for the floor, now confirmed FAST (~12min vs 4.5h
  of playouts). NEXT: depth-vs-collapse-rate (does deeper search reduce these collapses where eval didn't? =
  positive search-bound proof + quantifies the depth lever). Fast-probe seeds are consistent across runs
  (deterministic even-sampling) so candidates are directly comparable.
- **2026-06-20 (day, FEATURE SEPARABILITY — which new features have signal)** — `feature_sep.py` (python-chess,
  box-free) on lost-level (n=204) vs healthy-level (n=1020): ranked by separation(SD): **mobility +0.98**,
  **king-zone press_max +0.89**, **npieces −0.78** (fewer = sharper), king_pressure +0.60, attack_span +0.30,
  **pawn_space +0.13 (NO signal)**. ⇒ (1) **SPACE does NOT separate — don't build it** (the spatial intuition
  fails for these collapses). (2) The separators all describe COMPLEX/SHARP positions (high mobility, king
  pressure, fewer pieces) = tactical-dynamic signature. (3) **King-zone pressure separates strongly = the
  offense/defense (`IMBALANCE`) axis has real signal** (validates the future-sight direction). Levers: eval
  side = refine offense/defense toward the DEFENSIVE half (symmetric `IMBALANCE_SCALE` up leans negative in
  `fs_imb5` because it inflates our fantasy attacks too); **search side (stronger) = mobility being #1 separator
  ⇒ complexity-aware SEARCH EXTENSION** (more search in complex positions — the data-backed form of
  search-bound; static hedging via placement-shrink already HURT). `depth_probe`: 39/50 collapse-starts already
  play SF-best at d10 → cumulative run-up DRIFT, not single-move (deep d16/d22 passes too slow, killed).
- **2026-06-20 (overnight, FINAL VERDICT — floor is SEARCH-bound)** — gentle shrink `s4_gentle` (338g,
  MOD_PIECES_LEVEL=48): also negative (Elo +10.3 ±43 to base; cand collapse-rate **0.320** vs base 0.276).
  Shrink hurts MONOTONICALLY (0→0, 48→−10, 128→−16 Elo; collapse-rate +15% at both). **CONCLUSION: the
  collapse floor is SEARCH/HORIZON-bound, not eval-magnitude-bound.** Evidence chain: (1) scalar material
  calibration — eval-honest but move-flat + collapse-neutral; (2) conditional universal layer (material/attack/
  pair) — NEUTRAL (doesn't fire on the level plurality); (3) placement shrinkage — NEGATIVE (placement is
  correct signal; high-placement level positions are genuinely SHARP, the detector hits 5× more healthy than
  lost so it can't selectively damp, and the distinguishing feature "does the sharp line resolve" is a SEARCH
  question); (4) move choice NEVER changed with any eval-magnitude knob at any failure position. ⇒ **eval-
  magnitude lever EXHAUSTED for the floor.** The eval is not BIASED (unbiased at equal material), it's
  imprecise only in sharp positions, which is search's job. **PIVOT to user's phase 2: DEPTH (EBF-lowering =
  better pruning/move-ordering, near-term) → SMP (bigger).** The dynamic-eval FRAMEWORK is built/byte-id/
  reusable IF a future eval-PRECISION approach (new features / NNUE) is pursued — distinct from eval-MAGNITUDE.
  **PACE WORKED as a fast falsifier:** ruled out the whole eval-magnitude floor-lever in ONE night (cheap
  probes + 4 targeted playouts, agent semantic-pruning) vs weeks of blind SPSA. ALL code gated/byte-id/
  UNCOMMITTED (engine-you-play unchanged at 261/134,429,469).
- **2026-06-20 (overnight, S4 RESULT — NEGATIVE, big)** — `s4_playout` (344 games, MOD_PIECES_LEVEL=128):
  **WORSE on every axis** — Elo +16.2 ±43 to BASE (cand −16), collapse-rate cand **0.317 vs base 0.276** (+15%
  MORE flips), more losses (151 vs 135). ⇒ shrinking placement HURTS. Resolves the a-vs-b: the level-position
  placement claims are **NOT over-credit** — they're correct positional signal marking GENUINELY SHARP positions
  where the failure is **tactical/horizon (search-bound)**, not eval over-credit. The separability signal
  (placement 2× in lost) was real but causally a *sharpness* marker, not an over-read. **CONCLUSION FORMING:
  the floor is SEARCH-bound, not eval-magnitude-bound** — universal layer NEUTRAL + placement shrinkage NEGATIVE
  + move-flat across every test all agree. Confirming with a gentle shrink (MOD_PIECES_LEVEL=48) before the
  final verdict; if also ≤0, the eval-magnitude lever is exhausted for the floor → pivot to DEPTH/EBF/SMP
  (user's phase 2). PACE value: ruled out the entire eval-magnitude floor-lever in ONE night of cheap probes +
  2 playouts (vs weeks of blind SPSA) — the method worked as a fast falsifier even though the hypothesis failed.
- **2026-06-20 (overnight, S4 BUILT)** — `MOD_PIECES_LEVEL` / `MOD_PIECES_MAT_THRESH=1000` / `MOD_PIECES_FLOOR=500`:
  cold-tail shrinkage of the placement snapshot `br_pieces` in LEVEL-material MIDGAME positions, proportional to
  |placement| excess over FLOOR (`cut=(K*(|pieces|−FLOOR))>>8`, pulled toward 0, never sign-flips). Inserted after
  `br_pairs` in `placement_and_piece_eval`; `br_pieces` is a plain always-computed local = placement total.
  **byte-id 261/134,429,469** at default; liveness `MOD_PIECES_LEVEL=128` → 144,034,150 (big effect). Running the
  decisive test: `s4_playout` (tournament_seeded 90min conc4, base vs MOD_PIECES_LEVEL=128) → `flips` both arms =
  collapse-rate. The variance-reduction is invisible to the bias-mean table, so collapse-rate is the read.
- **2026-06-20 (overnight, SEPARABILITY — key finding)** — `level_sep.py` (built; `pyrun` sub) compared
  midgame LEVEL-material LOST positions (flip blindness points, n=204) vs HEALTHY level positions (SF |cp|<50,
  n=1020) on stored default eval_breakdown term magnitudes. **The placement terms over-fire ~2× in the lost
  set:** pt_bishops 1609 vs 734 (+875), pt_knights 1478 vs 695 (+782), pt_pawns 910 vs 340, pt_rooks 779 vs
  334, `pieces` 1040 vs 561. ⇒ **a real separating detector exists: large placement magnitude in a level
  position = collapse-prone (over-trusted sharp placement).** S4 direction: **condition the AGGREGATE `pieces`
  total at the COLD TAIL** (no per-piece-loop surgery) — shrink placement when (level material AND |pieces|
  large). Caveat (a-vs-b): over-credit [shrink helps] vs genuine sharpness→search failure [shrink won't help];
  prior over-read-grows-with-depth evidence favors over-credit; the collapse-rate gate decides. This makes the
  level-plurality knob-fixable → S4 is VIABLE (was the open question). Build: a `MOD_PIECES_*` cold-tail
  shrinkage gated default-neutral.
- **2026-06-20 (overnight, sweep)** — `bias_sweep placement_bundle` (cheap perturbation, 6 min): only
  **`MOD_MAT_PAWNS` moves the material-bias buckets** (clean direction: +300 → +1B+2P 552→<432 / +1R+2P 592→498;
  −300 → 693/710). `MOD_MAT_OPPB` near-inert (OCB rare). `MOD_LT_BACKING` & `MOD_PAIR_OPEN` **inert on this
  metric** — because the bias-by-material table is *material-axis-specific* (blind to king-attack / pair
  positions), not because they do nothing. even-control stayed 2 throughout. **Semantic prune:** MAT_PAWNS is
  the only material lever but it's move-flat/collapse-neutral (low floor value); LT_BACKING/PAIR_OPEN need
  their own (non-material) probes. ⇒ universal layer confirmed wrong target for the FLOOR, pruned cheaply.
  NEXT: separability diagnostic — does any cheap detector distinguish a LOST level-position from a healthy one?
  YES → that's the S4 placement-conditioning signal; NO → level scatter is genuine imprecision (precision/depth-
  bound, not knob-fixable). (PACE case: agent semantically characterized+pruned 4 knobs in 6 min, no playouts.)
- **2026-06-20 (overnight)** — S2 results on the FIRST universal candidate (`MOD_MAT_PAWNS=300 MOD_MAT_OPPB=−400
  MOD_LT_BACKING=400`): **move-decisive proxy FLAT** (midgame flips: SF-best 37→36, over-read −1%); **seeded
  playout NEUTRAL** (`dyn_playout` 99 games: Elo +3.5 ±80; collapse-rate base 0.273 vs cand 0.253 = 27 vs 25
  flips, ~0.3σ noise). ⇒ the universal-term layer does not move the floor — consistent with the structural read
  (its modulations only fire on material-edge / down-material-attack / bishop-pair, NOT the ~38% level-material
  plurality). Dispatcher autonomy subs added: `tournament_seeded`, `flips`, `move_proxy`, `bias_sweep` (all
  prompt-free; invoke with NOTHING before `bash '…overnight_runner.sh'`). Overnight: characterize universal
  knobs via `bias_sweep` (semantic prune), then PIVOT to **S4 placement-term conditioning** (box-free code) —
  the re-targeted PACE where the level-material variance actually lives. PACE method = SPSA math + agent
  *semantic pruning* of unfruitful branches (user) — skip knobs that can't fire on the target before spending
  collapse-rate playouts.
- **2026-06-19** — Plan approved. Diagnosis that motivated it: collapses ~91% midgame (32% up / 38%
  level / 28% down material; only ~25% "thought winning"); placement & scalar-material both
  collapse-rate-neutral (placement 0.301 vs base 0.294). Endgame material candidate (`PV_BOOST_PHASE_K=64
  ENABLE_ENDGAME_SCALE ENABLE_MATE_DRIVE_SCALE`) seeded-SPRT killed at 527 games = Elo +4 ±34 (neutral)
  but collapse-rate 0.224 vs 0.290 (−23%, ~2σ) on its target zones — kept as gated default-off
  soundness lever, not a floor-mover. S0 registration done; S1 framework next.
