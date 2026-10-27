# SESSION HANDOFF 2026-07-22

## ★ LATEST STATE (passer redesign in progress) — READ THIS FIRST, then the overnight section below
Full detail: `passer-doubled-hce-comparison-2026-07-22.md`; plan `.claude/plans/well-for-the-suppressor-wise-steele.md`.
- **WHY:** POSITIONAL is the dominant collapse class (~5× ks_attack). Root = passed-pawn subsystem, diagnosed to
  code: midgame rank table CLAMPED ±275 (cpp_bitboard.cpp:827/931) → passers UNDER-fire; rear doubled pawn
  mis-flagged passed + smeared per-piece credits → doubled OVER-credit. Audit VERDICT = **poor COORDINATION,
  not missing detectors**: the board-driven path gate (`passer_danger`, reads attack_bitmasks) + the
  consolidation (`ENABLE_PASSER_V2`) already EXIST but are DEFAULT-OFF; king-race is endgame-only. V2 failed
  before = INCOMPLETE consolidation (left clamp + smeared credits live) judged on a null self-play SPRT.
- **SECRET SAUCE (why SF's bigger passer table doesn't blow up):** bonus = rank_table × PER-PAWN board-driven
  path-safety gate (enemy pieces attacking/blocking the path + king race), and contributions are
  ADDITIVE-BOUNDED not compounding. Same gate boosts a clear-path passer AND dampens 3 passers a bishop stops.
- **DESIGN = fresh gate `ENABLE_PASSER_V3`** (strict SUPERSET of V2; delete V2 once V3 ships): (1) multiplicative
  R-gate on the rank bonus (R∈[0,256] = passer_danger realizability, made PER-PAWN), (2) remove the ±275 clamp,
  (3) disable the sign-smeared §6-A per-piece credits, (4) hook the king-race into the midgame blend (L6285-6308).
- **TESTING MECHANISM (the method — reusable):** incremental GATED rounds, each verified DETERMINISTICALLY on
  `ks_sets/passer_corpus.csv` (288 rows, phase-labeled, SF18 truth, tiers under_fire/blowup_guard/control +
  hand-built GUARDS incl. the KING-RACE PAIR `6k1/8/8/8/2ppp3/8/6B1/6K1 b` SF18 −0.29 vs
  `8/8/3k4/2ppp3/8/5B2/8/6K1 b` SF18 −5.17 — same 3 passers, king race flips 5 pawns; our eval is ~identical &
  backwards = the acceptance test) via `passer_verify.py` (per tier×phase over_read vs SF18 + priced-once
  breakdown split) BEFORE games; then joint multi-family fit; games judge by per-class collapse profile, NOT
  self-play SPRT. Builders: `build_passer_corpus.py`; harness `passer_verify.py`.
- **PROGRESS:** Round 0 (corpus+harness+baseline: under_fire −4.40, blowup_guard +4.19, control +0.40; value in
  pt_pawns, passed_supp ≈0) DONE. Round 1a (rear doubled-pawn no longer mis-flagged, getPPIncrement ~8157 under
  V3) DONE — verified #1 +5.80→+5.35, #3 −5.75→−5.49 (toward SF18), small (clamp is Round 2); **byte-id V3-off
  247/39,971,153 (exact)**. GOTCHA (banked): a new `inline` Config knob needs `env_flag()` reg
  (search_engine.cpp:1071) + toggle dump or env is ignored.
  **Round 2b (DONE):** multiplicative R-gate on the MIDGAME rank bonus. Extracted `passer_realizability_R(sq,
  white)` from `passer_danger` (pure refactor, byte-id kept); under V3 the midgame passed bonus is deferred out
  of the pawn loop into `g_passer_mid_deferred[]` and priced ONCE post-loop (beside passer_danger, full
  attack_bitmasks) as `bonus * mid_w/range * R/256`. **byte-id V3-off still 247/39,971,153.** Verify: under_fire
  midgame −4.19→−3.96, dossier #2 +4.17→+3.50 (toward SF −2.32), blowup_guard midgame held. Direction right,
  magnitude MODEST. **KEY FINDING: the midgame clamp is the SMALLER share — the dominant passer under-fire is
  adveg (−6.13) + endgame (−3.64), in the `isEndGame`/`evaluate_pawns_endgame`/`advanced_endgame_eval` path
  (uncapped already, so NOT a clamp problem = a magnitude/king-race issue) which Round 2b did not touch.**
  **Round 3a (DONE):** enabled all-phase king-race delta under V3 (6892) + bypassed AE legacy copy (4861).
  Corpus A/B DISPROVED "king-race is the dominant lever": AE→delta swap is value-neutral for adveg (already had
  king-race), and 4× `PASSER_KRACE_MAG` closed adveg only −6.13→−5.31 while blowup rose — weak, over-read-prone.
  **Round 3b (DONE):** endgame R-gate (mirror 2b in `evaluate_pawns_endgame`; `g_passer_end_deferred[]`; priced
  ×R in blend + isEndGame post-loops); `!ENABLE_PASSER_V3` added to R's king-term (5720/5736) so R=path-safety,
  delta owns king-race. **byte-id V3-off 247/39,971,153.** A/B verdict: **passer valuation is MULTI-CHANNEL** —
  no single knob (rank magnitude / king-race / path-safety R) closes under_fire without lifting the guard (each
  guard position stopped by a different mechanism: blockade / path-attack / rook-behind / king-can't-escort-two;
  R's path-safety misses the latter two — e.g. guard `8/2r4k/8/6PK/6P1` our +5.32 vs SF +0.04). Candidate **C**
  (`PASSER_KRACE_MG_PCT=0 PASSER_KRACE_MAG=100 SCALE_ENDGAME_RANK=150`) nets +BOTH tiers vs 2b baseline
  (under_fire −4.35→−4.18, blowup +4.16→+3.98, control flat) = machinery works two-sided.
  **NEXT = Round 3c: Stage-1 CONSTRAINED FIT** over {SCALE_ENDGAME_RANK, PASSER_KRACE_MAG/MG_PCT, PASSER_DANGER_D2,
  + fold rook-behind/king-escort into the gate} minimizing under_fire s.t. blowup_guard held (+control +move-match
  `diagnostics/suites/passers.csv`) on `passer_corpus.csv`, warm-start = candidate C; reuse `ks_fit_diverse.py`/
  `_ks_fit_eval.py`. Then Round 2c (§6-A disable) before Stage-2 joint fit on `diverse_corpus.csv` (guard every
  family), then games. Full V3 machinery now in place & gated default-off (bug-fix + mid R-gate + eg R-gate +
  king-race), all byte-id 247 — remaining work is CALIBRATION, not new wiring.
- **KS DECISION (settled): KEEP base KS, do NOT ship the count-gate.** Game-neutral within noise (matched-seed
  `profile_collapses.py`); KS was the dominant class, now controlled ~12/seed = a banked win; the residual
  ks_attack collapses are our-eval OVER-READING our own attacks (triage: 17/23 not-danger) = same family as the
  OvD c5 over-read → the fix is REDUCING over-reads (passer/OvD/space), not strengthening KS.
- **CAPG_PIN: game-neutral → RECOMMEND COMMIT** (correctness + STS+28/WAC+7; working-tree default true, NOT
  committed). **OvD realizability over-read** (c5) is a separate open eval lead.

---
# SESSION HANDOFF 2026-07-22 (overnight autonomous block complete)

**Read this first.** Then `collapse-reduction-ledger.md` (2026-07-22 entries), `overnight-plan-2026-07-22.md`,
`ks-sts-overfire-dissection-2026-07-21.md` + `ks-architecture-sf-ethereal-vs-ours-2026-07-21.md` (the KS
research this all rests on), and `game-analysis-2026-07-22.md` (the OvD eval-bug lead).

## HEADLINE RESULTS (games decided; STS did NOT)
1. **KS count-gate = GAME-NEUTRAL within noise (NOT a regression — earlier "NO-GO" was overstated).** Paired
   200g SF@2400 seeds 0/1. The ABSOLUTE MATCHED-SEED profile (`profile_collapses.py`, the fair metric — NOT the
   noisy vanish nets I first used) says: **ks_attack FLAT 11.5→12.0, positional 62.0→72.5, score −2.2%** — ALL
   within the 2-seed noise band (ks_attack class ~12/seed, score SE ~2.5%). Don't ship the gate, but don't
   discard KS either. **KEY: KS strengthening did NOT reduce the ks_attack collapse class**, and that class is
   small (~12/seed) vs POSITIONAL (~62/seed = ~5× bigger, the dominant mode). KS-in-ISOLATION was the wrong frame.
   - **TRIAGE (`ks_collapse_triage.py`) shows WHY:** of 23 base ks_attack collapse FENs, **17 = "not-danger"
     (SF18 ≈0..+2.5 but our DEEP eval reads +50..+80 for us)**, 5 eval-blind, 1 search-bound. The class is
     DOMINATED by our engine OVER-READING its own attacking chances (same phenomenon as the c5 OvD bug) — NOT a
     KS under-read, NOT search. So strengthening KS pushed the WRONG way. The fix is REDUCING the over-read.
   - **FRAMING (user): KS is a BANKED WIN, not a null.** It WAS the dominant collapse class (v1 cut KS-caused
     15→7; DEF=5 cut ks_attack 70→54); now it's ~12/seed = controlled, no longer dominant, benches improved.
     Further STRENGTHENING is neutral because KS is ALREADY handled (residual ks_attack = over-reads, a different
     fix). KS stays in the SET as a balanced term. Don't ship the count-gate (diminishing/wrong-direction); DON'T
     regard KS as a failure.
   - **ACCUMULATION PRINCIPLE (user, load-bearing):** vs SF, fixing one collapse class does NOT drop the TOTAL —
     SF just exploits the next weakness, so total stays ~flat within noise while the targeted CLASS drops. Judge
     by per-class categorical verdict. Fix enough classes → cumulative strength eventually clears the 200g noise
     floor. KS = one banked fix; POSITIONAL (~62/seed, ~5×) is now the dominant class = the next target.
   - **NEXT PHASE (user-directed) = BALANCED JOINT fit {KS + OvD + Space/positional}** on the diverse SF18 corpus,
     guard ALL families, target = reduce the attack/positional OVER-read toward SF18 (the triage's dominant
     failure mode), so no feature overfits in isolation and the 5 genuine eval-blind danger cases aren't sacrificed.
2. **CAPG_PIN = GAME-NEUTRAL (no regression) → RECOMMEND COMMIT.** BASE(pin-on) vs PIN_OFF, 2 seeds: score
   +0.75% pin-off (noise), collapses noisy/neutral, no stable per-class direction. It's a CORRECTNESS fix
   (drops illegal pinned-piece captures) + deterministic STS +28 / WAC +7. It HOLDS. **ACTION FOR USER: commit
   CAPG_PIN** (flip `ENABLE_CAPG_PIN` committed default true; working-tree-only now). NOT auto-committed.

## THE DISCIPLINE WIN
STS (a jagged move-choice guard) said the gate was +33 and shippable; GAMES + the per-class categorical
verdict + vanish attribution caught a real positional-collapse regression. We did NOT ship on STS. Reinforces:
**games are the arbiter; STS guards; judge by per-CLASS vanish attribution, not total or one seed's score.**

## OPEN LEAD (next session — the real eval bug found)
**OvD/imbalance over-reads UNREALIZABLE king attacks.** Concrete SF-grounded bug from the engine-vs-SF18 game:
28...c5 blunder, FEN `5q1k/7p/2ppRp2/p5p1/2P3P1/Q7/PP3PPK/3r4 b` — our total −3.21 vs SF11 −0.38;
`term_dump_fen.py` localizes it to **Imbalance/OvD −1.57 vs 0** (KS actually UNDER-reads there). This is an OvD
REALIZABILITY over-read, NOT KS. Distinct lever from KS. Fix = realizability-condition the OvD term (NOT a
blanket IMBALANCE_SCALE cut — already debunked). Build a dossier of similar positions first. See
`game-analysis-2026-07-22.md`.

## STATE
- COMMITTED: Kaufman (6bd9d66). Working tree: ENABLE_CAPG_PIN=true (recommend commit). KS scaffolding gated.
- Byte-id ref (working tree, Kaufman+CAPG_PIN): WAC 247 / 39,971,153.
- Game data: `selfplay/games/{base,gate,gatef13,pinoff}_s{0,1}/`. Classified collapse dataset +
  vanish-attribution used per-class throughout. Tools added this block: `analyze_game.py`, `blunder_probe.py`,
  `term_dump_fen.py`, `build_diverse_corpus.py`, `ks_fit_diverse.py`, `ks_structural_gate.py`,
  `ks_sf_feature_sep.py`, `ks_ours_units_auc.py`, `ks_band_scan/join.py`, `ks_overfire_vs_sf11.py`.
- Nothing running. Overnight block done.

## FIRST ACTIONS next session
1. If the user approves: commit CAPG_PIN (narrow: flip the default; keep byte-id note).
2. Open the OvD realizability over-read (the c5 dossier) — the one concrete eval bug with SF truth.
3. KS is PARKED (game NO-GO via these levers; a from-scratch multiplicative-modulator redesign is the only
   remaining KS path — future, out of the fit-only scope).
