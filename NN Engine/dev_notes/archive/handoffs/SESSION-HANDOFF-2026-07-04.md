# ▶️ SESSION HANDOFF — read this FIRST (2026-07-04)

Big multi-hour session. Everything below is banked; engine byte-id intact (WAC **245 / 39,146,294**, all new work
gated-off), nothing running, tree clean, branch NN-ENgine unpushed. This is the index; detailed docs linked inline.

## STRATEGIC STATE (the arc)
Started trying to fix the collapse (engine grabs material into lost sharp positions). The cheap conditional-damp
fix (`MOD_PIECES_CONTROL+DEFEND`) cut static-vs-SF sign-flips 65→50% held-out but **LOST ~202 Elo SPRT** → proved
static-accuracy proxies (STS/WAC/bench-flips) are NOT strength; only games are. That triggered a pivot (Fable
audit #2, source-verified):
- **Built + calibrated a fixed-node strength gate** (the ruler the proxies weren't): reproduced a known −202 as
  −194±58 AND unbiased 0.0 at truth-0, ~3.6× faster than lightning. → `node_ab` sub.
- **Resolved the compass paradox:** SF is a feature **LIBRARY** (borrow its outcome-fishtest-selected term
  DIRECTIONS/algorithms) — sound; SF as a magnitude **TARGET** — proven dead. Operationalized via **incremental
  validity** (Δ held-out outcome-loss from adding a feature to sigmoid(a·our_eval)).
- **Ranked the missing features by real outcome signal (by-game):** king-safety +0.0031, mobility +0.0022,
  rooks +0.0020, threats +0.0013. Static features (space/pawns/imbalance) ~0 = aggregate-calibrated (the global
  scalar outcome-Texel retune is TAPPED).
- **EVAL-FIRST (user):** the ~590-Elo EQUAL-DEPTH gap to SF11 is eval-dominated → make eval strong+fast+accurate
  FIRST, to power the search/EBF work later. Search is modern+complete but secondary (see the map doc).

## ★ KEY FINDINGS TO CARRY
1. **King-safety is architecturally BROKEN** (`engine-vs-sf11-map-2026-07-04.md`): pawn-shelter TRIPLE-counted live
   (evaluate_kings_midgame + setAttackingLayer + latent_threat), weak-squares double-counted, all FLAT/midgame-only/
   un-realizability-conditioned; the only realizability(KS_DYN)+taper site (`king_safety_danger`) is GATED OFF. This
   IS the collapse over-read, AND it CONFOUNDED the 23% KS screen (enabling KS added a 4th shelter copy).
2. **`latent_threat` POV error** (user): it's a KING-SAFETY term mis-named/tuned as "threats" (THREAT_* knobs, scans
   king zones). SF11's threats = piece-on-piece (our NEW `threats` term). → fold latent_threat INTO the KS
   consolidation; the real threats concept is now the separate `threats` fn.
3. **Threats granular architecture** (user): don't clone SF's monolithic threats() — split by TYPE (immediate=DONE
   +0.00063; pawn-push/restricted = separate fns; each its OWN realizability). GATE THE BUNDLE (pieces are small).
4. **Phase treatment differs per fn (data):** KS + mobility = midgame-only (KS collapses 10× in endgame — active
   king is GOOD); threats = ALL-PHASE (+0.0010 endgame signal). The granularity payoff.
5. **shelter_storm is the missing KS richness** — absent from our code AND our docs; need SF11 `pawns.cpp
   evaluate_shelter` (per-file shelter-pawn rank + storm distance + open/half-open by side + best-of-castle-squares).
   Reconstruct source-verified. Our KS severity is also under-scaled vs SF's kingDanger²/4096 + safe-check weights.

## TOOLING BUILT (the funnel — reuse, don't rebuild)
- **`node_ab <mins> <nodes> '<p1cfg>' '<p2cfg>' [conc] [tag]`** — fixed-node paired A/B gate (LONG_FORMAT so nodes
  bind first). CALIBRATED. The mid-funnel strength ruler. `NODE_LIMIT` engine knob (default 0=off, byte-id).
- **`diagnostics/incremental_validity.py <corpus> [our_term]`** — the OUTCOME COMPASS (by-game, 4-fold, sign-
  stable). Generalized to any term (argv[2]); measures Δ held-out outcome-loss vs the sf11 ceiling column.
- **`diagnostics/patch_our_cols.py <in> <out> <term> [KEY=VAL env...]`** — FAST corpus iteration (~5min): recompute
  only OUR cols with a given eval config, keep SF11 labels. Turns the build→measure loop into ~8 min.
- **`diagnostics/phase_screen.py <corpus>`** — incremental validity stratified midgame/endgame (per-fn phase-gate).
- **`diagnostics/breakdown_gap.py [corpus]`** — mass ours-vs-SF11 per-term corr/coverage/mean-gap + classify
  (MISSING/MALFORMED/MIS-SCALED). The gap-diagnosis tool.
- **`selfplay/tune_corpus.py`** — `--no-sf11` (fast/large, outcome-only), `game` id col, `ALL` tags, `--mirror`.
- **`selfplay/tune_fit.py`** — `--target result` outcome-Texel, `--fit-k`, by-GAME holdout, `--corr`.
- **`bench_gate`/`bench_split`** — static sign-flip bench (diagnostic only, NOT a ship gate — see the −202 lesson).
- Corpora: `tune_data/threats_corpus.csv` (SF11-labeled, by-game, 37k) + `_v3` (all-phase threats) + `ks_corpus`/
  `ks_v2`/`ks_v3` (KS configs). `sf11_bygame_corpus.csv` = the clean by-game SF11 corpus.

## ★ FAST SPSA/SPRT ACCELERATION (discussed, to build)
The `node_ab` fixed-node gate (~3.6× faster than lightning, calibrated) OR a new "super-lightning" preset (~d3-4)
could accelerate SPSA/SPRT dramatically. CAVEAT: depth-bias — shallow/fast is FAITHFUL for STRATEGIC eval
(placement/structure/mobility) but BIASED for TACTICAL (king-safety resolves with depth). So: CALIBRATE any fast
lane against a KNOWN-Elo change before trusting it; strategic-eval SPSA can use fast nodes, KS/tactical needs a
time/blitz gate. `spsa.py` supports `--lane eval --depth N` (fixed-depth, valid for EVAL knobs only) — wire the
fixed-node budget into the SPSA lane for cheap eval-knob tuning. NOT yet built into spsa.py; the node_ab gate is.

## PRIORITIZED PLAN (eval-first, outcome-data-ranked) — full detail in `engine-vs-sf11-map-2026-07-04.md`
1. **KS CONSOLIDATION** (the #1 eval lever, biggest + fixes the collapse root): lift/dedupe the scattered king
   scoring (latent_threat + evaluate_kings_midgame shield + setAttackingLayer king-credit) INTO `king_safety_danger`
   (realizability+taper), byte-id-verify (MOVE credit, don't duplicate) → re-screen for a CLEAN capture number
   (23% was confounded) → add per-file **shelter_storm** (SF pawns.cpp) → align **severity** to SF kingDanger²/
   safe-check weights → midgame-only → fit → node_ab + BLITZ → SPRT. Plan: `ks-build-plan-2026-07-04.md`.
2. **Un-gate + tune mobility** (+0.0022 real lever; per-piece non-linear table; retire cheap surrogates if it wins).
3. **Threat family**: bundle-gate immediate-threats (+0.00063, DONE) + build pawn-push/restricted granular fns.
   Spec: `threats-build-spec-2026-07-04.md`.
4. **Continuous endgame scale-factor** (replace binary `advanced_endgame` draw-detect; cheap correctness).
5. **(Later, search, secondary):** add a **TT best-move field** (unlocks singular-ext/IID/multi-cut — all blocked
   today by TTEntry having no move) → **ProbCut** (the one buildable missing mechanism, reuses RFP node-entry eval).

## OPEN ITEMS / BUGS
- **`isNearGameEnd` uninitialized** (cpp_bitboard.cpp, only set in phase_score>96 branch) = UB → init false; separate
  byte-id-checked fix. (Explains spurious advanced_endgame_fired on normal-endgame test positions.)
- Immediate-threats (`ENABLE_THREATS`) is banked gated-off → bundle-gate later, don't ship solo (too small).
- The stray non-ours files in the tree (enum_mapper, ENUM_MAPPER_*, _chesscom_gap_fens, old CE/, verified_tactics,
  ChessUI submodule) — LEAVE.
- Commit only when asked; no commit footer.

## FABLE (limited-time, via USER)
Two audits done (2026-07-03 RFP/levers, 2026-07-04 the compass resolution + KS/allocation), both source-verified.
Discipline: SF-as-library-not-target; source-verify every mechanism; escalate ONLY on a genuine design fork (e.g.
a non-obvious shelter_storm restructure) — NOT research/mapping (self-doable). Records: `fable-audit-2026-07-03/04.md`.

## POINTERS
Master map: `engine-vs-sf11-map-2026-07-04.md`. Plans: `ks-build-plan`, `threats-build-spec`, `outcome-texel-
campaign` (all 2026-07-04). Compass/pivot: `fable-audit-2026-07-04.md`. Collapse: `collapse-fix-2026-07-04-
overnight.md` (the −202 + gate calibration). Memory: [[fable-audit-2026-07-04-fixed-nodes-pivot]] (master session
record), [[collapse-eval-overread-fix]], [[outcome-texel-campaign]], [[ks-detection-rebuild]],
[[position-conditional-eval-program]], [[fast-selfplay-eval-depth-bias]].
