# SESSION HANDOFF — 2026-07-08

## ✅ 2026-07-10 pt.11 — CYCLE BROKEN: fantasy-vs-real separates at AUC 0.85; discriminator = PIECE-MATERIAL BACKING
Ran the decisive fantasy-vs-real fit (`diagnostics/fit_fantasy.py` on `corpus_ks.csv`; no new harvest — FANTASY
= collapse over-reads (our≥+150, SF≤+50), REAL = control positions we AND SF call winning (our≥+150, SF≥+150)).
**HOLDOUT AUC = 0.85** (38 fantasy / 45 real) ⇒ OUTCOME 1: our cheap detectors DO separate a fantasy advantage
from a real one — we are NOT missing the feature; we were fitting the wrong contrast (collapse-vs-control found
win-SHARED material-lead → l1a killed wins).
- **THE DISCRIMINATOR = `npedge` (NON-PAWN / piece material edge):** REAL wins `npedge` **+437cp** (up ~a piece);
  FANTASY `npedge` **+12** (no piece edge — pawn-only/positional). Secondary: `counter_pressure`/`def_us`/
  `off_opp`. So: **we over-credit a PAWN/positional lead when it is NOT backed by a PIECE-material edge.**
- **THE FIX (targeted, spares real wins):** a realizability damp on the positional-optimism cluster keyed on
  `f(low npedge = unbacked, counter_pressure)` — fires on fantasy (pawn-up-sharp, no piece), SPARES real
  (piece-up) wins. This is EXACTLY where l1a went wrong (it damped the pawn-material lead = shared with real
  wins). Screen it on the fantasy-vs-real contrast (must drop fantasy residual while holding BOTH control AND
  the real-win positions), then gauntlet for SCORE (not collapse-rate). Likely needs a small build (condition
  on npedge; `MOD_MAT_PAWNS` is pawn-count not piece-edge) — the offline screen + AUC say it'll separate.
- Tools added: `wdl_collapse.py` (61% losses), `fit_fantasy.py` (this), `screen_knob.py` (fast offline A/B).
byte-id 247 intact; nothing built/committed. Fable consult: `dev_notes/fable-question-fantasy-realizability-2026-07-10.md`.

## ⚠️ 2026-07-10 pt.10 — LANE 1 (broad optimism DAMP) is NET-NEGATIVE; offline→gauntlet link HOLDS; pivot toward LANE 2
Built the FAST offline screener (`diagnostics/screen_knob.py` — recompute our eval under any knob vs the CACHED
SF in `corpus_ks.csv`, ~15s, no games/SF; validated: default reproduces the over-read exactly). Screened the
EXISTING convertibility hooks (all default-off, built before but never shipped): `MOD_PIECES_CONTROL` (−71 MG,
saturates), `MOD_PIECES_DEFEND` (−31, the "greed-under-attack collapse cluster" hook, cleanest), `MOD_MAT_PAWNS/
OPPB` + `ENABLE_ENDGAME_SCALE` (endgame). Candidate **l1a** = `MOD_PIECES_CONTROL=512 MOD_PIECES_DEFEND=512
ENABLE_ENDGAME_SCALE=1 MOD_MAT_PAWNS=512 MOD_MAT_OPPB=512` passed the offline screen cleanly (MG −101, EG −68,
control +11). The screen also REJECTED an over-aggressive LEVEL variant (control drifted +26) before games.
- **GAUNTLET l1a (2 seeds vs banked baseline):** collapse rate **21.6%→12.2%** (halved!) BUT score
  **48.7%→37.1% = −11.6% (≈−80 Elo)**, seed-robust (s0 44.7→37.5, s1 56.2→36.6). ⇒ **the offline→gauntlet link
  HOLDS for the MECHANISM (residual down → collapses down) but broad optimism-DAMPING is NET-NEGATIVE for
  SCORE.**
- **KEY REFRAME:** our positional optimism is LOAD-BEARING — it drives active play; broadly damping it makes us
  passive and worse, while the collapses it removes were largely draws-we'd-draw-anyway (removing the flag
  gains nothing). Matches `[[eval-accuracy-payoff-is-pruning]]` (aggressive eval > calibrated-passive). ⇒
  **collapse-rate is NOT the objective; gauntlet SCORE is.** Lane 1 (DAMP positive terms) ≠ Lane 2 (ADD missing
  king-danger, which only subtracts WHEN our king is in danger, not in the safe 94%) — Lane 2 may be net-+
  exactly where Lane 1 is net-−. NOW isolating DEFEND-only (targeted, THRESH-gated) to see if a TARGETED damp
  differs from the broad bundle; if it too is negative, damping is out → pivot fully to Lane 2 (KS carve-out+add).
byte-id 247 intact (all knobs default-off, no build); nothing committed.
- **DEFEND-only gauntlet (targeted, THRESH-gated): +3 (neutral)** — confirms the broad CONTROL/MAT hooks caused
  l1a's −11%, not the targeted one. **Loss-vs-draw split (`wdl_collapse.py` on the triage CSVs): 61% of
  collapses are real LOSSES** (67% EVAL), 39% draws → the over-read IS costing games, worth fixing.
- **⭐ CIRCLING DIAGNOSIS (user-flagged, correct):** we were fitting the WRONG contrast. **"collapse vs
  control"** finds features (material-lead) that are SHARED with real WINS → damping them kills wins (l1a's
  −11% was predictable). The DECISIVE contrast is **FANTASY-wins (peaked→LOST) vs REAL-wins (peaked→WON)** —
  the detector must separate which advantage CONVERTS. SF11 does it (reads fantasy ~0) so a cheap detector
  EXISTS; our current ones are correlates. **Bounded test:** fit our detectors on fantasy-vs-real → AUC>0.7 +
  sane detector = build the TARGETED damp (spares real wins); AUC~0.5 = we're MISSING an SF11-style feature.
  Either result breaks the cycle. Full lesson: [[collapse-fix-load-bearing-optimism]].

## 🎯 2026-07-10 pt.9 — DIAGNOSIS NAILED + Fable 2-lane plan (verified). START HERE for the eval-collapse fix.
The collapse over-read (+368cp static, conditional, classical-solvable) is now precisely decomposed and
Fable-consulted (`dev_notes/fable-question-ks-missing-vs-tuning-2026-07-10.md` + its response). Tools:
`diagnostics/verify_triage_static.py --dump` (per-FEN detector+term corpus, any-knob A/B, offline residual
screen), `fit_convert.py`, `ks_gap.py`, `pawn_gap.py`, `test_pieces.py`; corpus `diagnostics/corpus_ks.csv`.

**TWO INDEPENDENT LANES (Test A confirmed independence: pieces over-read +193 in SF-danger subset ≈ +207 in
no-danger — fixing KS won't drain pieces):**
- **LANE 1 — material-lead over-valuation when UNCONVERTIBLE (the big lever, ~+280 of the +368).** Per-piece-
  type decomp: the `pieces` +177 is **pt_pawns +172** (dominant) + pt_knights +74; rooks/bishops UNDER-credited
  (Fable's "rooks/queens" guess was WRONG). Pawn lane characterized (`pawn_gap.py`): `corr(pt_pawns,pawn_lead)
  =+0.95` (MATERIAL-lead driven, NOT advancement +0.25); both phases (EG Δ+224 > MG Δ+148); vs SF11 corr 0.48
  slope~1 but our pawn +159 vs SF +27 = we TRACK SF's pawn structure but add a large CONDITIONAL offset (control
  pt_pawns ≈ −17). `piece_value_boost` (+49) and `capture_gains` (+58) are ALSO material-lead terms → the whole
  optimism cluster co-inflates on the SAME latent factor ("up material, won't convert"). ⇒ **ONE shared
  realizability damp** keyed on (material-lead × non-convertibility), centered f=1 off-class, one-sided clamp
  (damp-only ~[0.6,1.0]), sparse. EXISTING hooks: `MOD_PVBOOST_COMP` (pvb, damps material-lead by opp
  offense−our defense — cpp_bitboard.cpp:6871), `endgame_convertibility_scale`, `capg-tension`. (NOTE Fable's
  "pvb is total-crossing-+1500 derived, fixing pieces cascades it" is WRONG — pvb is an independent material-
  lead boost; it needs its own damp/the shared factor.)
- **LANE 2 — king-danger MISSING DETECTOR (sharp ~24% subset; small mean −35cp but the loss games).** Our
  king-danger eval is UNCORRELATED with SF11's (corr −0.04, 94% of SF-danger missed) = structural blindness,
  NOT under-weighting. Reconciled with `[[ks-detection-rebuild]]`: the rebuilt detector CAN read units but is
  PARKED; corr-0 is on the shipping (parked) build. Prior REPLACE −47 = latent_threat-removal regression (not
  KS badness); beside ~0 = double-count; ALL verdicts were LIGHTNING (anti-predictive) → **KS formally
  REOPENED** at the gauntlet. Fix = carve out overlap first (`KS_CONSOLIDATE` strips shelter from pieces +
  reduce latent_threat to its non-KS residue) THEN add the clean attack-unit danger term (separate lane:
  adding negative signal, can't be a multiplier).

**METHODOLOGY (Fable's gift — the "done" criterion, all offline before games):** each arm must move the
COLLAPSE-class static residual toward CONTROL's +8 while holding control within ±20cp — screen via
`verify_triage_static --dump` A/B, gauntlet only the survivors (≥2 seeds, collapse rate primary). "Done" =
class residual matches control (our_ks corr −0.04→>0.5; pt_pawns +172→~control; control totals ≤±20cp).
Minimal experiment = 3 arms (Lane1 / Lane2 / both); sub-additivity = the interdependence, measured.

**STATE:** byte-id 247 intact (king-danger breakdown instrumentation is breakdown-only). Baseline grown to 4
seeds (44.7/56.2/47.8/46.2 ≈ 48.7%, collapse ~24%). Nothing built for the lanes yet; nothing committed. NEXT =
Lane-1 offline screen: A/B a material-lead×unconvertibility damp against the over-read (no gauntlet yet).

## 🌙 2026-07-10 OVERNIGHT pt.8 — convertibility fit BORDERLINE; KS under-detection is the real finding; factor STAGED
Autonomous overnight (prompt-free). Built the contrastive corpus + fit pipeline:
- **Tooling (new):** `diagnostics/verify_triage_static.py --dump <csv>` (per-FEN detector corpus: kdu_us/opp,
  off/def edges, mobility, material/pawn/oppb, over_read; KEY=VAL knob A/B; EG/MG split). `diagnostics/
  fit_convert.py` (logistic P(collapse|detectors), ALL + MIDGAME-ONLY, holdout AUC + per-detector Δ).
  **Corpus at `diagnostics/corpus.csv`** (434 rows: 67 collapse / 367 control). ⚠️ intermediate files MUST be
  Windows paths — WSL /tmp is NOT shared across pyrun invocations (silently lost the first dump).
- **STEP 0 build (byte-id 247 preserved):** king_safety_score now also runs in breakdown mode
  (`ks_active || g_capture_eval_breakdown`, cpp_bitboard.cpp:6453) to populate the king-danger detectors
  without touching `total` (wac never sets the breakdown flag ⇒ byte-id safe).
- **FIT (Gate G2 borderline):** ALL holdout AUC 0.78, MIDGAME-ONLY holdout **AUC 0.85** (43 collapse / 299
  control) — the midgame collapses DO separate. BUT the separators are **activity + material magnitudes**
  (`pawn_lead`, `off_us`/`def_us`/`off_opp` all higher in collapse = a-pawn-up sharp/active positions), NOT a
  clean convertibility detector. `counter_pressure` fits wrong-signed (overfit alarm).
- **⭐ KEY FINDING — our KS massively UNDER-DETECTS:** `kdu_us` (our king-danger units) = ~+2 in BOTH collapse
  and control, where SF11 flags **−300..−485** of king danger in the sharp subset. So `king_safety_danger`
  computes ~nothing where SF sees a real attack ⇒ it can't be the convertibility detector (it IS a hole).
  This reframes: the midgame collapse may be fixable by **repairing king_safety_danger's under-detection**
  so it actually SUBTRACTS the danger SF sees — pulling the `pieces` over-optimism back down — rather than a
  separate `pieces` damping factor.
- **KS_ZONE2 test (existing knob, re-dump `corpus_zone2.csv` + refit): did NOT fix it** — `kdu_us`/`kdu_opp`
  stay ~+2/+3 in BOTH classes with the wide 2-ring zone. So the under-detection is STRUCTURAL (the attack-unit
  weights / `attack_bitmasks` mechanism itself), not the zone — a morning eval-repair, not a knob. Cleanest
  sane midgame signals that DO separate: `pawn_lead`(+), `off_us`(+), `our_overreach`=off_us−def_opp (Δ+219,
  +) — i.e. "a pawn up while over-extending our offense". Morning options: (a) repair king_safety_danger to
  match SF11's KS term (the root); (b) build the activity/material factor and let the gauntlet arbitrate
  (Fable-style, risk = damp-the-active-subset); (c) more/better data first.
- **DECISION:** factor build STAGED for the morning (borderline/diffuse signal + broken sound detector = do
  NOT autonomously build a blunt activity-damper = the −202 trap). Overnight pivoted to SAFE data-growth:
  baseline seeds (tighten the ±6% reference) + grow the collapse corpus + pieces sub-decomposition. byte-id
  247 intact; nothing committed.

## 🎯 2026-07-09 pt.7 — PHASE 0.5 VERIFY: static-hole CONFIRMED, SF11-tripwire FAVORABLE, culprit ≠ king-safety
Ran `diagnostics/verify_triage_static.py` (our STATIC via `ev_breakdown` + SF18 static + SF11 per-term) on the
66 triage collapse decision-FENs (g_base_s0+s1). Mover-POV cp. Results:
- **Outcome A CONFIRMED (static eval hole, NOT a d18 phantom):** EVAL class (n=47) mean **our_static_gap vs
  SF18 = +368cp** ≈ mean search_gap +273cp. Our STATIC over-reads ~3.7p on the collapse FENs (the 2026-07-06
  "static≈SF" held only at PEAK FENs, not these decision FENs). ⇒ eval-completeness lane validated; corrHist/
  OTV/singular correctly banked (they need a search−static gap that's absent).
- **SF11 tripwire FAVORABLE (the big result):** mean **SF11_total = +5cp** on EVAL — classical HCE SF11 ALSO
  reads these ~0 while we read +368. ⇒ the discrimination does NOT need NNUE; **cheap CLASSICAL detectors can
  capture it** = green light for the whole pre-NNUE conditioning program, we're inside the HCE-solvable regime.
- **Culprit ≠ king safety (redirect):** mean **ourKS_term = +2cp** (contributes ~nothing). KS shows as a real
  SUB-component only in the sharp/loss subset (SF11 KS −300..−485 we miss: games 96, 0-s1, 20-s1, 141-s1).
  **Per-term MEAN decomposition of the +368 (mover-POV, SF total≈0 ⇒ +ve = over-reader):** `pieces=+177` (the
  dominant ~half) · `capture_gains=+58` · `piece_value_boost=+49` · KS+2. `pieces` is the consistent top
  over-reader across ALL classes (EVAL +177 / HORIZON +265 / PRUNING +275). ⇒ target = **broad positional /
  material-edge over-valuation** (`pieces` placement term first, then capture_gains + piece_value_boost), NOT
  king safety. Connects to the archived `material-edge-overvaluation` finding + the realizability architecture
  ("biggest flat term = most headroom"). OPEN sub-question before conditioning `pieces`: is it INTRINSIC
  mis-scale (uniformly too high → recalibrate) or MISSING COMPENSATION (position equal despite our nominal edge
  via fortress/counterplay/non-convertibility we don't score → add a realizability/convertibility factor)? The
  drawn-endgame slice (Phase 1) is the isolable "unconvertible edge" case → do it first.
- **⭐ CONDITIONAL CONFIRMED (keystone, control comparison):** ran the same decomposition on n=183 CONTROL
  midgame FENs (general plies, same games). **CONTROL static_gap = +8cp (≈0, calibrated!), pieces=+31** vs
  **EVAL collapse static_gap +368, pieces +177** — a 46× gap jump / 5.7× pieces jump. The over-read is
  **CONDITIONAL on the collapse class, NOT an intrinsic mis-scale.** ⇒ **(1) global recalibration (Texel/SPSA
  on pieces scale) is REFUTED** — pieces is right on average; rescaling breaks the 94% to patch the 6% =
  EXACTLY why years of Texel/SPSA went nowhere (optimizing an already-right average). **(2) a CONDITIONAL
  realizability factor is the only path** = Fable's centered-f=1 + contrastive corpus, and the user's
  "magnitude=f(detectors)" thesis, now data-proven. **(3) UNIFIES endgame+middlegame:** both = "our real edge
  doesn't CONVERT" (fortress/KPK drawn endgame; middlegame counterplay/king-danger-to-us). ⇒ **ONE
  convertibility/compensation realizability factor** keyed by non-convertibility detectors (opponent
  counter-pressure, king-danger-to-us, fortress/blocked, insufficient-winning-material, opposite bishops),
  scaling down the aggregate optimism (pieces+capgains+pvb all inflate together in the class → a shared factor,
  not per-term). Endgame_convertibility_scale (Phase 1) = the endgame special case. **The contrastive corpus
  already exists: collapse FENs = unconvertible class, control sample = matched real class.** Tool:
  `diagnostics/verify_triage_static.py` (triage.csv args = collapse; a games-dir arg = CONTROL sample;
  KEY=VAL args set engine env for single-core knob A/B; splits EVAL/HORIZON/PRUNING into -EG/-MG).
- **ENDGAME-SCALE A/B (single-core knob test) — existing scale INSUFFICIENT:** EVAL-EG (n=17, 36% of collapses
  are endgames: 24/66) baseline static_gap +434 (pieces +260, pvb +70) → `ENABLE_ENDGAME_SCALE=1` only +372
  (−62cp, all from pvb; **pieces +260 UNTOUCHED**). Cause: `endgame_convertibility_scale`'s "winning passed
  pawn pulls s→1" clause makes a KPK/passed-pawn DRAW get scale≈1 = defeats the damp in the drawn-passer case.
  ⇒ Phase 1 ≠ "flip the scale"; the drawn endgames need real DRAW DETECTION (generalize KPK all-files +
  fortress/insufficient-winning patterns in `is_practically_drawn` → hard 0), and/or gate the scale's passer-
  pull off when a draw signature is present. EVAL-EG is ~pure `pieces` (+260) → PST/king-activity/piece-values
  not seeing the draw. `pieces` is the common over-reader across EG+MG.
- **CAVEAT:** "largest term" is a magnitude proxy, NOT the proven error term (pieces is our biggest term
  everywhere). **NEXT (single-core):** proper per-term our-vs-SF11 gap on the collapse corpus (`breakdown_gap.py`
  style — MISSING/MALFORMED/MIS-SCALED per term) to NAME the culprit term, then apply Fable's TERM-AGNOSTIC
  realizability method (contrastive corpus → classifier f(detectors) → gauntlet) to it. Phase 1 endgame-draw
  scaling is unaffected (still the parallel quick-win). byte-id 247 intact; nothing built; no commits.

## 🧭 2026-07-09 pt.6 — COLLAPSE TRIAGE (diagnose-first Phase 0A): EVAL-dominated → REDIRECT to eval-completeness
After singular banked neutral, ran `triage` on the 2 banked BASELINE gauntlet collapse corpora (deep re-search
@ depth 18 + SF classify EVAL/PRUNING/HORIZON). **Both seeds EVAL-dominated:** s0 = **26 EVAL / 2 PRUNING / 7
HORIZON**, s1 = **23 / 3 / 8** (~70% EVAL). The collapse is an EVAL-FUNCTION problem, not search/horizon.
**KEY CONSEQUENCE (updates the plan's corrHist queue):** these EVAL over-reads are shared by our OWN depth-18
search (it still plays the move, eval stays far above SF) → the search−static training signal is ~0 exactly
here → **corrHist CANNOT fix them, and OTV (re-search) can't either** (both need search to see through the
static error; it doesn't). So the value-CORRECTION machinery is the wrong tool → **redirect to eval-COMPLETENESS**
(fix the over-reading terms). The eval holes (worst-gap rows of `games/g_base_s0/triage.csv`) cluster:
1. **Drawn-endgame over-reads (~25-30% of EVAL, the cheapest/cleanest target):** our eval reads +5 to +8 in
   DEAD DRAWS — KPK (game 23: `8/8/8/5k2/4p3/4K3/8/8` = +5.2!), R-vs-R+P fortress (g85 +4.4), N+P-vs-N (g113
   +5.2), Q+N-vs-Q (g47 +7.9), down-the-exchange (g101 +4.6), opposite-bishops (g149 +7.8). `ENABLE_RP_KPK_DRAW=1`
   is ALREADY ON yet the KPK reads +5.2 → our draw/fortress/insufficient-material SCALING has real holes.
   Infra exists: `advanced_endgame_eval`, `ENABLE_RP_KPK_DRAW`, `ENABLE_ROOK_ENDGAME_CAP`.
2. **Middlegame king-safety / attack over-reads (the loss-collapses, harder dynamic class):** g0 peaked +29 then
   LOST; g96 peaked +9.3; g143 reads +4.6 vs SF +0.5 — attack/position valued +2..+9 while opponent counterplay
   / our own king weakness is under-weighted. Matches the prior corr-0.27-KS / 0.23-threats gap.
**STATUS:** corrHist plan (Phase 1) is now questionable (can't fix search-shared eval holes). 0B (OTV gauntlet
re-test) is predicted neutral by this (OTV only helps the ~22% HORIZON slice) — pending user call on whether to
run it as due-diligence vs pivot straight to eval-completeness (endgame-draw-scaling first = cheapest collapse
reducer). byte-id 247 intact, nothing built, no commits. Plan file: ~/.claude/plans/handoff-ebf-reduction-recursive-cascade.md.

## ⭐⭐⭐ 2026-07-09 pt.4 — SINGULAR campaign IN PROGRESS (minimal TT-move path). Plan: ~/.claude/plans/handoff-ebf-reduction-recursive-cascade.md
Goal = singular extensions (the phantom fix; 23% gauntlet collapse). Full cache audit done (complete map in
the plan + will get a memory when Stage 4 validates). Approved minimal path: TT-move field → node-local
populate → node-entry probe → singular (write-once direction-parameterized helper). Fable-refined:
**mandatory excluded-move key-XOR** (exclusion search must live in a parallel keyspace or it poisons N's
entry), trigger-rate log as the null-result confound detector, shared extension cap, phantom-rate as the
primary metric. **DONE + byte-id 247/41,479,610 verified (both singular off AND ENABLE_SINGULAR=true):**
- **Stage 1** — `TTEntry` gained a `Move move` field (kept dead alpha/beta to avoid the 20-site churn);
  `addToSearchEvalCache` gained a trailing `Move move=Move()` param + the SF move-rule (real-move-or-keychange
  updates it, independent of the depth gate; same-key+no-move preserves). Zero call-site edits. cache_management.h.
- **Stage 4** — the two cutoff sites (min ~3523 / max ~4095) now do a gated **move-only write** into the
  node's OWN TT entry (`if (ENABLE_SINGULAR){ e=accessSearchEvalCache(node); if(e) e->move=move; }`) instead
  of the old g_ttMoveTable write. Byte-id-safe (writes only entry.move; SF rule protects it from the parent's
  child-keyed store). g_ttMoveTable left DEAD (reads ENABLE_TT_MOVE-gated-off; cosmetic removal deferred).
- **Knobs** `ENABLE_SINGULAR=false`, `SINGULAR_MARGIN=2`, `SINGULAR_MIN_DEPTH=6` (search_engine.h ~893) +
  env-load + echo.
**⚡ KEY FINDING 2026-07-09 (verified, CORRECTS the plan): the exclusion-search key-XOR is NOT needed.** All
21 live `addToSearchEvalCache` calls key by the CHILD (`updated_state`); our engine NEVER stores a node's own
entry from within its own search (child-store architecture — parent stores child). So the singular exclusion
search (re-running N's move loop excluding ttMove) stores only N's *children* (depth-preferred keeps the
deeper main entries) and never poisons N's own entry. Fable's mandatory key-XOR was SF-architecture-based
(SF self-stores); ours differs. Guards the exclusion still needs (cheap): skip the Stage-4 move-populate and
the node-entry pruning shortcuts (RFP/null/ProbCut) while `g_excluded_move[cur_depth]` is set.

**RESUME POINT = node-entry probe (read-only, byte-id) + the singular extension** (direction-parameterized
`template<int Sign>` helper called from min & max; excludedMove param threaded through minimizer/maximizer;
the excluded reduced-depth null-window search cloning the ProbCut scaffold; the excluded-move key-XOR into
make_move_cache_key; singularBeta = ttValue − Sign·MARGIN·depth; extend under g_check_extensions; trigger
counters near g_otv_fires). Then gate: byte-id off → scoped mirror test → node_ab → gauntlet ≥2 seeds → SPRT,
phantom-rate first. Nothing committed. generatePawnKey still banked for correction-history (post-singular).

### SINGULAR CORE — ⭐ MAX SIDE BUILT + REACHED 2026-07-09 (byte-id 247 off; ENABLE_SINGULAR=true = 242/46.3M, fires)
**MAX side DONE** (5 inserts applied in `maximizer`: node-entry probe + `excluding` before the loop
(~3919), move-loop skip, gate+exclusion+extend before updateZobristHashForMove (~3954), `depth_limit +
singular_extra` into get_score_for_maximizer, move-populate `&& !excluding` guard (~4127)). Uses
`g_check_extensions < CHECK_EXTENSION(=3)` as the runaway cap. `wac ENABLE_SINGULAR=true` = 242/300,
46,314,854 nodes (+11.7% = extensions firing; −5 WAC = the max-ONLY asymmetry + WAC≠strength, correctness
TBD via the mirror test + games). byte-id 247 off intact. **NEXT: mirror to MIN** (minimizer, sign-flipped
per the spec below — ttFlag==UPPERBOUND, sb=ttScore+MARGIN*rem, exclusion `minimizer(cur_depth,cur_depth+rem/2,
sb,sb+1, t0, dummy_ints,dummy_moves,dummy_entry, …)`, singular if v>sb; ##VERIFY the min window vs the min
cutoff convention; add `excluding`+guard to the min move-populate at ~3527). Then: trigger-print (g_sing_*
near g_otv_fires) + g_sing_eligible increment + scoped mirror test → node_ab → gauntlet ≥2 seeds → SPRT.
Consider a dedicated singular-extension counter (not g_check_extensions) as a proper cap before games.

### SINGULAR CORE — ✅ MIN SIDE MIRRORED + DEDICATED CAP 2026-07-09 (byte-id 247 off; both sides fire 4.11%)
**MIN side DONE** — 5 inserts mirrored into `minimizer` sign-flipped (local `sing_dummy_ints/moves/entry`
added since minimizer has none; `excluding`+node-entry probe before the loop; move-loop skip; gate
`ttFlag==UPPERBOUND`, `sb=ttScore+MARGIN*rem`, exclusion `minimizer(cur_depth,cur_depth+rem/2, sb,sb+1, t0,
sing_dummy_*, state_history, position_count, zobrist, previousMove, num_iterations, last_move_was_capture,
false, is_in_null_search)`, singular if `v>sb`; `depth_limit+singular_extra` into get_score_for_minimizer;
move-populate `&& !excluding`). ##VERIFIED the min window vs the cutoff convention (a minimizer cutoff fires
on `lowest≤alpha` ⇒ stores UPPERBOUND; symmetric to max's LOWERBOUND — search_engine.cpp:2824/:107).
**DEDICATED EXTENSION CAP built** — `g_singular_extensions` + `SingularExtensionGuard` (RAII, wraps each
extended get_score call on BOTH sides) + knob `SINGULAR_MAX_EXT=8`; both singular gates now cap on
`g_singular_extensions < SINGULAR_MAX_EXT` (NOT the shared `g_check_extensions`, so a pure singular chain is
bounded independently of check extensions). Instrumentation: `++g_sing_eligible` per TT-move node (both
sides) + a `[singular] eligible/gatepass/fire/fire_per_eligible` stderr line near the otv print + a `SING:`
summary in the `wac` sub. **Measured:** byte-id 247/41,479,610 off (both the MIN mirror and the counter swap
are byte-identical disabled). `wac ENABLE_SINGULAR=true` = 246/300, 54.16M nodes (+31% vs off; EBF
3.68→3.73), **fire/elig = 4.11%** (eligible 62.2M / gatepass 4.65M / fire 2.55M) — well above the ~1-2%
TT-starvation floor, so a later null result reads as "doesn't help", not starvation. WAC rebalanced from −5
(max-only) to −1 (both), as predicted. Nothing committed.

### ☠️ SINGULAR VERDICT 2026-07-09: NEUTRAL — BANKED (default-off, byte-id 247 preserved), NOT the phantom fix
Full gate ran. **Result = a thoroughly-diagnosed NULL** (not a starvation artifact — the confound detector
clears it):
- **node_ab** (fixed-node 250k, 165g, conc3): singular +6.3 **±62.3** Elo = neutral, point estimate meaningless.
- **gauntlet ≥2 seeds vs SF18@400** (160g/arm, our 250k nodes), matched openings, per-seed delta =
  singular − banked baseline (base s0=44.7% / s1=56.2%):
  - `SINGULAR_MARGIN=2`: s0 55.7% (**+11.0**) / s1 45.6% (**−10.6**) → mean ≈ 0, **SIGN FLIPS with seed**.
  - `SINGULAR_MARGIN=400`: s0 52.2% (**+7.5**) / s1 48.4% (**−7.8**) → mean ≈ 0, **SIGN FLIPS again**.
  - The apparent deltas are **baseline seed-swing** (44.7↔56.2 = 11.5 pts on the baseline alone), not
    singular. Same false-positive shape as null_eval_gate — the ≥2-seed rule caught it a 2nd time.
- **Collapse rate NOT reduced** (base 21.6% avg → singular 23.7%@m2 / 23.4%s0+19.0%s1 @m400) — the DIRECT
  mechanism failed. Verifying OUR chosen move is singular does **not** cure the over-optimistic collapse.
- **Confound cleared:** fire/elig healthy at both margins (4.11%@m2, 1.96%@m400 — SF-like), so the null is
  "singular-as-built doesn't help", NOT "TT starvation stopped it firing". Margin lever characterized:
  m2=4.11%/+31% nodes, m100=3.16%, m400=1.96%/+11% nodes (m400 = the correct SF-like operating point; m2
  over-extended near-everything — MARGIN is millipawns, m2 = ~0.024p ≈ no margin).
- **No SPRT** — SPRT is the ship gate; 4 gauntlet arms + node_ab across 2 seeds × 2 margins are decisively
  neutral, nothing to ship.

**IMPLICATION for the phantom program:** the ~+3.9 phantom / ~22% collapse is NOT a move-singularity problem.
Collapses persist ~20-24% independent of singular + margin. The over-optimism lives in the backed-up VALUE
itself (or in refutations deeper than singular's rem/2 exclusion reaches), not in "was our move uniquely
best." ⇒ **pivot to the VALUE side: correction history** (was deferred behind singular; phantom-gate the
update; uses the banked `generatePawnKey`) is now the lead lever. Untried singular knobs (SINGULAR_MIN_DEPTH
lower, full-depth exclusion instead of rem/2, multicut) exist but there's no positive signal to justify the
compute — leave singular banked unless corr-history work resurfaces a reason. Code stays in (all env-gated,
byte-id 247 off); safe to leave or later strip.

### SINGULAR CORE — turnkey spec (worked out 2026-07-09; SUPERSEDES the paragraph above; key-XOR NOT needed)
byte-id 247 clean; globals pre-placed + compiled-unused: `Move g_excluded_move[MAX_PLY]` +
`long g_sing_eligible/gatepass/fire` (cache_management.h extern ~166 / cpp_bitboard.cpp def ~150; set/clear
balanced at the site so no per-search reset). **Do MAX side fully (build -> byte-id-off -> reached) THEN
mirror MIN.** 5 insert points per side (max fn ~3577 / min fn ~2829):
1. NODE-ENTRY PROBE (before the move loop; max ~3919 / min ~3318):
   `bool excluding = Config::ENABLE_SINGULAR && g_excluded_move[cur_depth].from_square != g_excluded_move[cur_depth].to_square;`
   if `ENABLE_SINGULAR && !excluding`: `TTEntry* nodeTT = accessSearchEvalCache(zobrist, current_state.castling_rights, current_state.ep_square);`
   grab ttMove/ttScore/ttDepth/ttFlag + haveTT. (read-only = byte-id)
2. MOVE-LOOP SKIP (loop top; max ~3921 / min ~3320): `if (excluding && move == g_excluded_move[cur_depth]) continue;`
3. GATE+EXCLUSION+EXTEND (right BEFORE updateZobristHashForMove, zobrist still = NODE hash; max ~3944 / min ~3341):
   `int rem = depth_limit - cur_depth; int singular_extra = 0;`
   MAX gate: `ENABLE_SINGULAR && !excluding && haveTT && !currently_in_check && move==ttMove &&
   ttMove.from_square!=ttMove.to_square && rem>=SINGULAR_MIN_DEPTH && ttDepth>=rem-3 && abs(ttScore)<9000000
   && ttFlag==TTFlag::LOWERBOUND` -> `++g_sing_gatepass; int sb = ttScore - Config::SINGULAR_MARGIN*rem;
   g_excluded_move[cur_depth]=ttMove; int v = maximizer(cur_depth, cur_depth+rem/2, sb-1, sb, t0,
   state_history, position_count, zobrist, previousMove, num_iterations, last_move_was_capture, false,
   is_in_null_search); g_excluded_move[cur_depth]=Move(); if (v < sb){ ++g_sing_fire; singular_extra=1; }`
   MIN mirror: `ttFlag==UPPERBOUND`; `sb = ttScore + MARGIN*rem`; exclusion `minimizer(cur_depth,
   cur_depth+rem/2, sb, sb+1, t0, dummy_ints, dummy_moves, dummy_entry, state_history, ...)` (min needs the 3
   dummies); singular if `v > sb`. ##VERIFY the min window/inequality vs minimizer cutoff convention
   (beta=min(beta,lowest); beta<=alpha) BEFORE trusting -- the sign-sensitive line.
4. APPLY EXTENSION: pass `depth_limit + singular_extra` to get_score_for_maximizer (max ~3961) /
   get_score_for_minimizer (min ~3040). Optionally 0 it if g_check_extensions already high.
5. MOVE-POPULATE GUARD: add `&& !excluding` to the Stage-4 move-populate at both cutoffs (min ~3527 / max ~4099).
v1 SIMPLIFICATION (conservative, not wrong): do NOT guard RFP/null/ProbCut during exclusion re-entry (early
high-return just misses a fire, never wrong); add later if fire-rate low. Trigger print near g_otv_fires
(~1765) under ENABLE_SINGULAR. GATE: byte-id 247 off -> `wac ENABLE_SINGULAR=true` changes -> scoped mirror
(positions+color-flips, identical min/max trigger+extend) -> node_ab -> gauntlet >=2 seeds (phantom-rate
first) -> SPRT.

## ⭐⭐ 2026-07-09 pt.3 — TIER-1 REORDER (Fable-adjudicated): TT-consolidation → singular FIRST; corrHist deferred
Tier-1 pawn program: **P1 pawn_key BUILT** (`generatePawnKey` in cache_management.h, from-scratch not
incremental — recompute is <16 XORs, sidesteps the ep/promo incremental-bug class the user has hit before;
byte-id 247 intact, corrHist will consume it later). **P2 pawn-structure cache ABANDONED** — the pawn eval
is NOT a pure function: pure-pawn `structural_bonus` is nonlinearly clamped with king-relative
`positional_bonus` (`min(225, structural+positional)`, cpp_bitboard.cpp:868) AND the pawn loop mutates ~8
downstream-consumed globals (offensive/defensive/central/attack_bitmasks/passer masks) → can't extract a
cacheable unit without a big risky refactor for modest payoff. Revisit only under a broader "make eval pure"
pass. **P3 correction history DEFERRED behind search-soundness** — Fable + our own bias_profile agree: our
bias is SEARCH-phantom (static well-calibrated ≈ SF; search invents +3.9), phantoms are self-sealing but
RARE (our OTV number = 0.4%), so corrHist (static→backed-up-search) is mildly contaminated not poisoned, and
a **phantom-gate** (skip update when |backed_up−static| > ~1.5p = OTV trigger reused as training filter)
makes it safe. BUT singular attacks the phantom *generator* (root cause) while corrHist only averages the
residue → **do singular first**; corrHist also *wants* eval16-in-TT + pawn_key, so it's a ~1-day experiment
after TT-consolidation. **NEXT LEVER = TT-consolidation (eval16 + move16 into the TT entry)** — serves the
speed goal (RFP/futility/improving near-free at TT hits = static-eval machinery cheaper w/o NPS) AND unblocks
**singular extensions** (needs TT-move) = the real phantom fix, AND sets up corrHist. Then singular, then
corrHist (phantom-gated, gauntlet-measured on the 23%→? collapse rate). **DESIGN CONSTRAINT (user):** static
eval stays standalone-usable (eval breakdowns / stand-pat / eval upgrades) → all search-based value
adjustments are a SEARCH-SIDE OVERLAY at the consumption points, NEVER baked into `placement_and_piece_eval`
(SF does this too — corrHist lives on `ss->staticEval`). Sources: Fable q (dev_notes/fable-question-*), our
ROADMAP:17 (OTV 0.4%), search-soundness-fable-q3:114 (singular). Nothing running, no commits, byte-id 247.


## ⭐ LATE-SESSION PIVOT (2026-07-08 pt.2): measurement crisis → external gauntlet BUILT + CALIBRATED
The pruning campaign hit a **measurement crisis**: every unshipped lever has a venue-dependent sign
(null-gate +14 lightning / −73 STS; ProbCut +17.5 node_ab / −83 lightning; passer −39 STS). Root cause =
**instrument resolution**: STS penalizes any prune at fixed depth by construction; self-play is blind to
shared-blind-spot changes. Fable consult (`dev_notes/fable-question-ebf-accuracy-frontier-2026-07-08.md`,
verified against code) → **pivot: pruning demoted to SECONDARY; primary = the gauntlet + speed/accuracy.**
Plan: `~/.claude/plans/handoff-ebf-reduction-recursive-cascade.md`.

**GAUNTLET BUILT + CALIBRATED (the durable win).** `overnight_runner.sh gauntlet <games> <sfnodes> [conc]
[tag] [KNOBS]` = OUR engine (`NODE_LIMIT=${OUR_NODES:-250000}`) vs **native-ELF SF18** at fixed `--sf-nodes`,
paired UHO, deterministic. **Anchor = sfnodes=400 ≈ 51% (40g) / 45.3% (150g).** Blind-spot-immune (NNUE eval
maximally alien) + our first ABSOLUTE axis. Infra: `--sf-nodes` in `selfplay/vs_sf.py`; `gauntlet` sub in
the runner. **WHY SF18 not SF1.1:** SF1.1/.exe = WSL binfmt-flaky (`Exec format error`, whole runs void);
SF18 native-ELF = 100% reliable. Memory: [[external-gauntlet-calibrated]]. ~23% collapse rate vs SF18 =
over-optimism still measurable at an external venue.

**null_eval_gate ADJUDICATED at the gauntlet (first customer):** baseline 45.3% vs null-gate **51.0%**
(150g each, same seed = paired) = **+5.7%, neutral-to-positive → NOT secretly bad.** Resolves the −73 STS as
the fixed-depth artifact (lightning +14 was truthful). **Ship candidate** (not yet ship-*proven*: +5.7%
±~11% spans 0 → needs the overnight accumulation to firm/flip default).

**ProbCut store-off DIAGNOSTIC built (Fable TT-pollution test).** New `g_no_tt_store` flag +
`ENABLE_PROBCUT_NO_TT_STORE` knob makes `addToSearchEvalCache` a no-op inside the ProbCut verification
subtree (both blocks). byte-id 247 preserved; reached-check OK (m1500 store-off 233/37.99M vs store-on
236/38.08M). **Lightning store-off vs base RUNNING** (`sprt_pc_storeoff`): if it → neutral/+ (vs store-on's
−83), TT pollution CONFIRMED → ProbCut becomes fixable/shippable; else apply MIN_DEPTH≥7-8, else bank.

**STORE-OFF RESULT: TT-pollution REFUTED.** ProbCut m1500 store-off lightning ~−58 (≈50g) vs store-on −83
— both clearly negative. Not a bookkeeping bug; the lightning collapse is the shallow-verification blunder
cost (fires near the root at shallow TC). Decision tree → next = MIN_DEPTH variant, else bank.

**OVERNIGHT RESULTS (2026-07-09) — all 4 jobs + seed de-risk DONE. Headline: the seed check caught a
false positive.**
- **null_eval_gate → NOT a ship (neutral, sign-unstable).** Gauntlet Δ = **+3.0% at seed0 (600g) but −3.2%
  at seed1 (300g)** — the lever's sign FLIPS across seeds. Reconciles the other within-noise positives
  (+14 lightning w/ LLR barely +, +11.3 node_ab ±40). Integrated ≈ **neutral**. The seed-0 +3.0% was noise;
  shipping it would have chased a false positive (cf [[compass-context-fragility]] / bench-flip lesson).
  Only concrete benefit = −2.3% nodes @ 0 tactical loss (efficiency-only; NOT a strength gain). LESSON: the
  gauntlet + seed-robustness check did their job — **always confirm a gauntlet lever at ≥2 seeds before
  shipping** (baseline itself swung 48.7%↔52.7% across seeds → only same-seed deltas are meaningful).
- **passer-danger → PARK.** Gauntlet +0.7% (600g) = neutral. Not hidden strength (kills the STS-masking
  hope); not harmful. Consistent w/ statScore subsumption.
- **ProbCut → BANK.** Store-off refuted TT-pollution (−58 vs −83). MIN_DEPTH=8 lightning = −25 ±40
  (−83→−58→−25 as it fires less, never neutral). Genuine post-soundness late-lever; +17.5 equal-node real
  but doesn't survive real TC. Diagnostic knobs (`ENABLE_PROBCUT_NO_TT_STORE`, gate `[conc]`, gauntlet
  `[seed]`) left in place.
- **META (confirms Fable):** all 3 disputed pruning levers → neutral/negative at the honest venue ⇒ pruning
  is TAPPED. **Next campaign = SPEED (pawn-hash NPS, guaranteed positive/venue-independent) + eval ACCURACY
  (threats).** Nothing shipped, no commits, byte-id 247 intact.

**OVERNIGHT BATCH — exact sequence (drive one at a time, Read result, decide next; all conc3 multi-core):**
1. `gauntlet 600 400 3 gnt_base_o` — shared baseline (SF18@400, seed0). ~80 min.
2. `gauntlet 600 400 3 gnt_ng_o ENABLE_NULL_EVAL_GATE=true` — SHIP confirmation. Same seed = paired vs #1.
   GATE: null-gate clearly ≥ base (600g CI ~±3.3% each) → propose flipping default (needs user commit +
   re-verify new byte-id). Neutral/below → hold.
3. `gauntlet 400 400 3 gnt_passer_o ENABLE_PASSER_DANGER=true` — Fable Q4, paired vs first 400 of #1.
   GATE: ≥ base → −39 STS was blind-spot-hidden real strength (keep/tune); < base → STS right, drop.
4. `gate 'ENABLE_PROBCUT=true PROBCUT_MARGIN=1500 PROBCUT_MIN_DEPTH=8' pc_md8 sprt_pc_md8 400 5` — ProbCut
   resurrection test (rem≥8 ⇒ verification child ≥4 ply). Want ~neutral at lightning (not −58/−83). If
   neutral, follow with `node_ab 45 250000 '' 'ENABLE_PROBCUT=true PROBCUT_MARGIN=1500 PROBCUT_MIN_DEPTH=8' 3
   pc_md8_nab` to check the +17.5 survives → resurrect; else BANK ProbCut for post-soundness.
Report all deltas by morning. Nothing ships without a user commit. byte-id 247 off after any build.
Passer `sts_full`/`_passer_match` single-core diagnostics = optional (the gauntlet passer test #3 supersedes).

## Shipped this session
- **statScore-LMR SHIPPED** (default-on): `ENABLE_STATSCORE_LMR=true`, `STATSCORE_OFFSET=512`,
  `STATSCORE_DIVISOR=1024` (search_engine.h). Continuous graded LMR reduce-less. **+23 lightning SPRT
  (accepted after ~900g at +23, we stopped it — LLR crawls at that effect size) / +33 node_ab.**
  Replaces the tiered `HISTORY_LMR_CAP` path (off = old tiered = byte-id 245). **NEW SHIPPED BENCH
  SIGNATURE = 247 / 41,479,610** (was 245/39,146,294; +2 WAC, +6% nodes = the reduce-less cost).
  Uncommitted (commit when asked, no footer). Memory: [[statscore-lmr-shipped]].

## Killed / resolved this session (don't re-litigate)
- **MALUS — DEAD (5th).** Q3 cutoff-calibration logger (`ENABLE_CUTCAL_LOG`) + count-refinement GREENLIT
  it on calibration (statScore monotone P(cut) 0.41→0.94; 0-bucket splits by tried-fail count
  0.48/0.36/0.29/0.25). Built the decoupled Q-table (`ENABLE_QCUT`, malus→statScore only, ordering clean,
  λ-dial). STILL regressed (STS 1550→1483→1439 monotone with λ, no EBF gain). **Falsifies the decouple
  hypothesis → root cause = context-blind global `[from][to]`. History reduces LESS only.** Code gated
  default-off. dev_notes/history-scoring-calibration-2026-07-07.md.
- **passer-danger** (`ENABLE_PASSER_DANGER`, cpp_bitboard.cpp `passer_danger`): built, STS +78, fires
  14.4% on collapse positions, but self-play node_ab −3.8 ±34 = **inconclusive in the WRONG venue**
  (self-play shares the blind spot). NOT shelved → needs the asymmetric `vs_sf1` test.

## ACTIVE CAMPAIGN: EBF-reduction / node-balancing (Lane 2b)
Plan approved 2026-07-08. Full plan: `~/.claude/plans/handoff-search-soundness-campaign-shiny-nygaard.md`.
Dossier: `dev_notes/ebf-node-balancing-campaign-2026-07-08.md`.
**Thesis:** claw back statScore's +6% nodes via POSITION-SPECIFIC prunes (history can't reduce-more).
**DECIDED sequencing: pruning-first + eval-speed parallel** (ProbCut is self-verifying = eval-independent,
so it delivers now; eval-accuracy held until the "eval wall" — when Phase-C margin sweeps stop improving).
**Shadow re-price baseline (shipped engine):** LMP 1.23% wrong-enter (near ceiling — DON'T tighten),
futility 0.27% (slack). Tool = `ENABLE_PRUNE_SHADOW` (already built; the wrong-prune pricer).

### Phase B progress (IN PROGRESS — where to resume)
- **Eval-scaled null-move R — BUILT + plumbed, byte-id off = 247.** Knobs `ENABLE_NULLMOVE_EVAL_R=false`,
  `NULLMOVE_R_DIV=1920`, `NULLMOVE_R_CAP=3`. Inserted after `reduced_depth -= NULLMOVE_EXTRA` (min
  search_engine.cpp ~3212 / max ~3700), sign-mirrored (min: `(alpha-ev)/DIV`, max: `(ev-beta)/DIV`).
  **⚠️ v1 LIMITATION:** guarded on `rfp_static_eval != NO_STATIC_EVAL` → fires only at rd∈[1,6] non-PV
  nodes (where RFP computed the eval); at those shallow nodes the null is already near-minimal so it
  backfired (WAC 239, nodes +0.2%). To be a real lever it needs the static eval at DEEPER null-move nodes
  (probe eval cache via `cur_hash`, or compute before the null flip at ~3205). PARKED tuning.
- **ProbCut — BUILT (2026-07-08), byte-id 247 off intact.** Inserted in both `minimizer` and `maximizer`
  after the OTV block / before the move loop, sign-mirrored per the spec (max pushes β up + calls
  `minimizer(…, dummy_ints, dummy_moves, dummy_entry, …)` cut on `v>=probcut_beta`; min pushes α down +
  calls `maximizer(…)` no dummies, cut on `v<=probcut_alpha`). Candidates = first ≤3 capture/promo
  (`promotion!=1`) with `see(to,turn,state)>=0`. Also added the missing `PROBCUT_DEPTH_REDUCTION` echo.
  **RESULT — venue split (the key finding):** WAC node curve `MARGIN=1000/-2.9%`, `1500/-8.2%`,
  `2200/+1.6%` (default 2200 too high — barely fires, net node *loss*). node_ab @250k (equal nodes) m1500 =
  **+17.5 ±59.8 for ProbCut** (positive, mechanism sound). **Lightning SPRT m1500 = ~−83 @34g** (stopped
  early, clearly negative). **Diagnosis:** ProbCut's shallow verification search inherits our search's
  phantom-PV miscalibration; at lightning `MIN_DEPTH=5`+reduction-3 leaves only a ~1-ply oracle → wrong
  cuts. Deep (equal-node) it's trustworthy → +17.5; shallow (lightning) it's blind → −83. **Same "wrong
  venue" as passer-danger.** ⇒ **ProbCut is a LATE lever: banked, default-off, plumbed.** Convert path =
  (1) empirical margin/MIN_DEPTH calibration (reuse the cutcal logger: P(full-depth confirms | reduced cut)
  by depth — likely need rem≥7-8 so child ≥3-4 ply), (2) Phase-D eval speed+accuracy (cheaper+trustworthy
  verification). Re-test at equal-node / slower-TC, NOT lightning, as the eval lane delivers.

### Phases A / C / D (not started)
- **A — drawer wins (node_ab-away, multi-core):** `ENABLE_SEE_PRUNE`(+SEE_PRUNE_MARGIN/MAX_DEPTH),
  `SEE_PRUNE_CAPTURES`, `ENABLE_NULL_EVAL_GATE` (already node_ab +11.3 ±40 this session — re-confirm+SPRT),
  `NULLMOVE_PROGRESSIVE`. All built, default-off, exact/eval-gate = eval-safe.
- **C — margin retune (sweeps):** widen `FUTILITY_MARGINS {200,450,650,950}` (has slack), tighten
  `RFP_MARGIN=1500`/`RFP_MAX_DEPTH=6`. NOT LMP (error ceiling).
- **D — eval feeders (parallel):** pawn-hash NPS (+8.6%=+11 prior) + eval32-TT → sf-source-evolution-bank
  doc; threats term → eval-lane-strategy doc.

## Gating discipline (every lever)
shadow wrong-prune price (single-core) → WAC+STS floor (single-core) → **node_ab paired A/B (multi-core —
FLAG the user before launching, cores shared with a parallel API project, conc ≤4, conc3 safer)** →
lightning SPRT + ONE slower-TC spot-check. **byte-id 247/41,479,610** off-path after every build. Compass
RETIRED (games decide). Mirror-A/A. "Cure priced above disease."

## Operational (CRITICAL for unattended runs)
- **ONLY prompt-free invocation** (now in NN Engine/CLAUDE.md + [[dispatcher-prompt-free-wrapper]]):
  command MUST start `wsl.exe -e bash -lc "bash '/mnt/c/.../NN Engine/selfplay/overnight_runner.sh' <sub>
  …"` (prefix match + `*`). A leading `R='…'`/`cd`/`export` wrapper PROMPTS (that's what stalled a night).
  Raw `pgrep`/`pkill`/`env python` PROMPT. Read results with the Read tool on the Windows-path file.
- Subs: `build` · `wac <tag> [KNOBS]` (SOLVED/NODES/EBF) · `sts <tag> [KNOBS]` (STS/3000) · `node_ab
  <mins> <nodes> '<p1>' '<p2>' [conc] [tag]` · `gate '<cfg>' <label> <tag> <maxg> <elo1>` (SPRT) ·
  `vs_sf1 <games> depth|time <val> [conc] [tag] [KNOBS]`. WSL python = /home/ranuja/anaconda3/bin/python.
- Diagnostics built this session (all gated default-off): `ENABLE_PRUNE_SHADOW` (LMP/futility wrong-prune),
  `ENABLE_CUTCAL_LOG` (P(cut|statScore) + tried-fail split), `ENABLE_QCUT` (dead malus), `see_selfcheck.cpp`
  (SEE fix verified 0.00%).

## State
byte-id 247/41,479,610 intact (null-R + ProbCut knobs gated off). Nothing running. NO commits. Branch
NN-ENgine unpushed. Next action = build ProbCut (single-core) per the spec above, then the multi-core
node_ab gauntlet (Phase A → B → C).
