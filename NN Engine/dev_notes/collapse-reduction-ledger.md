# Collapse-reduction ledger (running, across sessions)

**Purpose:** track the collapse PROFILE over time as we accumulate eval fixes, per [[collapse-categorical-
verification]]. The strategy: reduce a collapse class WITHOUT resurfacing prior fixes → the total trends down
→ Elo eventually follows (score lags below the ±15-Elo/200g noise floor until enough classes are fixed).

**Method / verdict rules:**
- Venue: paired 200g vs SF@2400, conc3, per seed. Report MULTIPLE seeds (seed variance ≈13% ≫ 3.5% SE — one
  seed's score is unreliable; the collapse-COUNT trend is the leading indicator).
- Two-sided verdict per fix: (a) target collapse class shrinks AND (b) prior-fix classes do NOT resurface.
- ⚠️ CLASSIFIER: `ks_collapse_attribute` uses SF11-static KS≤−1.5 = **contaminated for attack positions**
  (mislabels real attacks as "other"). **TODO next round: build an SF18-search-based collapse classifier** so
  per-class attribution is trustworthy. Until then: total-count + no-KS-resurface are the robust signals.

## Trajectory

| date | fix | venue | total collapses (base→fix) | class signal | ship? |
|---|---|---|---|---|---|
| 2026-07-17 | **KS v1** (`ENABLE_KS_REPLACE_LT` etc.) | 200g SF@2400 seed0 | 83 → 83 (**flat total**) | **KS-caused 15→7 (−53%)**, +8 other (next class exposed) | **SHIPPED** `73c7cad` |
| 2026-07-18 | **KS DEF=5** (`KS_SAFE_CHECK_DEF=5`, defensive-asymmetric safe-check) | **600g** SF@2400 seeds 0/1/2 | 258 → 253 total (FLAT) | **KS-ATTACK class 70→54 (−16, −23%)**, other +11 (next class exposed) — via OUR KS detector (non-contaminated); SF11-static classifier wrongly read "flat" | **SHIPPED `41c4123`** (categorical WIN, like KS v1: class↓ at flat total). Score neutral −0.7% (expected, early accumulation). |

## 2026-07-22 overnight — KS count-gate (FLOOR6+gate) GAME test = NET NEGATIVE (positional regression)
Config GATE = `KS_FLOOR=6 KS_SAFE_CHECK=8 KS_ATTACK_COUNT=2 KS_MIN_ATTACKERS=2` (STS 1588, deterministically
sound) vs BASE (Kaufman+CAPG_PIN default, KS floor-13). Paired 200g SF@2400, seeds 0 & 1.
- Score: BASE 39.2/41.0 (avg 40.1) vs GATE 36.5/39.2 (avg 37.9) — GATE −2.2% both seeds.
- Total collapses: BASE 78/77 vs GATE 95/85 — GATE +17/+8.
- **VANISH ATTRIBUTION (deterministic same-game diff, the trustworthy per-class metric):**
  | class | seed0 fixed−new | seed1 fixed−new | verdict |
  |---|---|---|---|
  | ks_attack (TARGET) | 11−7 = −4 | 5−5 = 0 | marginal benefit (real but small) |
  | positional | 28−48 = **+20** | 31−39 = **+8** | **consistent REGRESSION both seeds** |
- **MECHANISM (dissected, not discarded):** the gate DOES cut its target class (ks_attack net-down/flat), but
  it rides on `KS_FLOOR=6` which OVER-ACTIVATES KS on ordinary/positional positions → a positional-collapse
  regression that outweighs the target gain. STS (move-match on curated positions) did NOT predict this because
  game collapses (winning positions thrown away) are a different phenomenon than STS move-choice. **Lesson
  reinforced: STS is a jagged GUARD, games are the arbiter — a +33 STS did NOT transfer.**
- **NEXT (in progress):** DECOUPLE — GATE_F13 (`KS_SAFE_CHECK=8 KS_ATTACK_COUNT=2 KS_MIN_ATTACKERS=2`, floor
  STAYS 13) to test whether floor-6 is the sole culprit and a milder gate is neutral-or-positive on positional.
- Good element retained: the coordination gate cuts ks_attack; the floor-drop is what breaks positional.

### R2a DECOUPLE test: GATE_F13 (`KS_SAFE_CHECK=8 KS_ATTACK_COUNT=2 KS_MIN_ATTACKERS=2`, floor STAYS 13) seed 0
- Score 32.8% (WORST), 89 collapses. Vanish vs base_s0: **ks_attack 6−8 = net +2 (REGRESSED target!)**,
  positional 38−48 = **+10** (still regressed). Decoupling the floor did NOT help — the heavier weights
  (safe_check=8, attack_count=2) hurt on their own.
- **VERDICT (CORRECTED — earlier "NO-GO regression" was OVERSTATED):** I first read the noisy VANISH NETS as a
  positional regression. The ABSOLUTE MATCHED-SEED profile (`profile_collapses.py`, seeds 0,1, same base build)
  is the fair metric and says otherwise:
  | seeds 0,1 | ks_attack/seed | positional/seed | total |
  |---|---|---|---|
  | base | 11.5 | 62.0 | 77.5 |
  | gate | 12.0 | 72.5 | 90.0 |
  ks_attack is FLAT (11.5→12.0), positional +10.5, score −2.2%. **At 2 seeds these are ALL within the noise
  band** (ks_attack class ~12 → per-seed noise ~3.5; score SE ~2.5%). So the honest verdict is **NEUTRAL /
  INCONCLUSIVE**, NOT a regression. (The flashy "−38% ks_attack" from a 4-seed base avg was a CONFOUND — old
  base_s2/s3 carry ~27 ks_attack/seed, a different regime; ignore them.)
- **The real signal: KS strengthening did NOT reduce the ks_attack collapse class** (flat), AND that class is
  small (~12/seed) vs POSITIONAL (~62/seed, ~5× bigger = the dominant collapse mode). KS-in-ISOLATION was the
  wrong frame. Likely the ks_attack collapses are SEARCH-bound (tactical, depth gap), not static-KS — triage
  pending (`ks_collapse_triage.py`). **KEEP KS in the SET (do NOT discard); it is one balanced term.**
- **GO-FORWARD (user):** stop tuning KS alone. Next = a BALANCED JOINT fit over KS + OvD + Space/positional on
  the diverse SF18 corpus, guard-constrained across ALL families, so no feature overfits in isolation. The
  DOMINANT target is the POSITIONAL class (5× ks_attack). Space over-read (~9× vs SF11, CENTER_*_MULT) is the
  clean positional lever; OvD realizability over-read (c5) is another. Double-count map applies.
- Discipline: STS (+33) did NOT transfer to games → STS is a jagged GUARD, games+per-class-profile arbitrate.
  But use the ABSOLUTE matched-seed profile, not vanish nets, and remember 2 seeds is thin for a ~12-count class.

### TRIAGE: what the ks_attack collapse class actually IS (`ks_collapse_triage.py`, 23 base FENs)
Our deep eval (d13) vs SF18 on each ks_attack collapse decision-FEN: **not-danger 17 / EVAL-blind 5 / SEARCH-bound 1.**
- **17/23 "not-danger": SF18 sees ≈0..+2.5 but our DEEP eval reads +50..+80 for the collapsing side** (e.g.
  ourDeep +55 vs SF +2). ⇒ the class is DOMINATED by our engine OVER-READING its own attacking chances (the
  imagined attack evaporates = "collapse"). NOT a KS under-read, NOT search-bound (only 1/23).
- **This is the SAME over-read as the c5 OvD bug** (over-crediting unrealizable king attacks). ⇒ **strengthening
  KS pushes the WRONG way** (more danger-seeing when the failure is over-optimism about OUR attack). Explains why
  KS tuning didn't move the class.
- **UNIFIED CONCLUSION:** the dominant collapse driver (positional ~62/seed AND much of ks_attack) is our EVAL
  OVER-READ of attacks/initiative. FIX = REDUCE the over-read (OvD realizability + Space ~9×), tuned JOINTLY
  with KS (so the 5 genuine eval-blind danger cases aren't sacrificed). KS stays in the set as a BALANCED term.
- NEXT PHASE = balanced joint fit {KS + OvD + Space/positional} on `diverse_corpus.csv`, guard all families,
  target = reduce the attack/positional over-read toward SF18. Tools: `profile_collapses.py`, `ks_collapse_triage.py`.
- Remaining overnight slots → CAPG_PIN A/B (independent, pending-commit validation).

### R2b CAPG_PIN A/B: BASE (pin-ON) vs PIN_OFF (`ENABLE_CAPG_PIN=0`), seeds 0 & 1
- Score: pin-ON (base) 39.2/41.0 (avg 40.1) vs pin-OFF 40.2/41.5 (avg 40.85) — +0.75% pin-off = WITHIN NOISE.
- Total collapses: pin-ON 78/77 vs pin-OFF 88/73 — noisy/neutral.
- Vanish (pin-on→pin-off), per class: seed0 pin-off +4 ks_attack +8 positional (pin-ON better); seed1 pin-off
  −7 positional +1 ks_attack (pin-off better). **NO stable per-class direction → NEUTRAL.**
- **VERDICT: CAPG_PIN is game-NEUTRAL (no regression).** Combined with: it's a CORRECTNESS fix (drops illegal
  captures by absolutely-pinned pieces from capgains) + deterministic STS +28 / WAC +7 (improves BOTH benches).
  Per the "commit if it holds" criterion, it HOLDS → **RECOMMEND COMMIT** (flip `ENABLE_CAPG_PIN` committed
  default; currently working-tree-only). NOT auto-committed — left for the user.

## Per-fix score detail (context; score is seed-noisy, NOT the primary metric)
- **KS v1:** raw score/total FLAT; the win was categorical (KS class halved). "flat total ≠ failure."
- **DEF=5 (2 seeds):** score seed0 +10.3% (baseline ran low 37.5%), seed1 −3.4% (baseline normal 42.2%) →
  avg +3.4% but inside noise floor. **Collapses down BOTH seeds** = the reliable signal.

## Status of the other levers (honest)
- **pin / tempo / all-3: SINGLE-SEED (seed 0 only) ⇒ INCONCLUSIVE.** Seed-0 numbers (pin 40.5, tempo 39.2,
  all-3 40.8) looked mildly + / "drag DEF5 down" — but seed 0 was an unrepresentative low-baseline seed
  (baseline 37.5 vs true ~39.8), so those reads are unreliable. NOT confirmed either way. Deterministically:
  low-reach (7/64) + load-bearing-capgains cost (STS −34/−10) ⇒ low prior; but not game-falsified.
- **`KS_DEF_MAG`** (blunt defensive danger multiplier): deterministic NO-GO (STS 1606→1473, over-amplifies
  moderate danger → paranoia). Not game-tested; shelved.
- **BIG PICTURE:** SCORE is null/inconclusive at 600g (measurement floor; DEF5 Elo ≈ 0 ± big — the
  [[strategy-reset-2026-07-15]] problem). BUT the CATEGORICAL class metric (valid classifier) shows DEF5 IS
  working (KS-attack −23%). ⇒ the lesson is NOT "eval fixes don't work" — it's "judge by the target CLASS with
  an un-contaminated classifier, not total/score." Accumulate class-fixes; Elo follows once enough land.
  Method upgrade: `diagnostics/ks_class_reattribute.py` (our-KS-detector classifier on drop_fen, units>=13).

## pt_pawns / passed-pawn class — subsystem mapped (2026-07-18)
Full verified map: **`dev_notes/passed-pawn-subsystem-map-2026-07-18.md`** (Fable-generated, human-verified;
2 false-alarm "bugs" corrected — capgain-sign #22 + rook-dblcount #10 are both FIXED by default). Key: passed-
pawn value is scattered across ~20 live sites feeding many breakdown terms; the **raw mask is unconditioned and
drives 13 consumers**; **path-attack conditioning is effectively absent** (only `passer_danger`, default-OFF,
has it); a single advanced passer is credited in up to ~9 live channels. Redesign = ONE realizability verdict
(R∈[0,256]) feeding all channels via a `passer_weight[sq]` array; guard-rail = P3 keep-out. Don't ship on self-
play SPRT (was null); judge by real-opponent collapse profile.

## 2026-07-19 REDIRECT: the "pt_pawns/passer" class was mis-scoped -> it's over-positive ENDGAME pawn PLACEMENT
Full note: `dev_notes/pawn-placement-overread-redirect-2026-07-19.md`. Deterministic decomposition showed the
fen3/P2 over-read is ~0 from the passed-pawn boost and mostly general endgame pawn PLACEMENT (attacking-layer +
hardcoded chaining bonuses + eg rank table). Root: `pt_pawns` = whole pawn-evaluator output, NOT SF's `Passed`
term -> the original "our pt_pawns +5..9 vs SF Passed ~0" was apples-to-oranges. SF11 study: SF gives ~0 raw
advancement to a non-passed pawn (eg dominated by penalties). Two gated levers built (default-off, byte-id):
- **`ENABLE_PASSER_V2`** (passer-realizability consolidation: passer_danger blockade+path both-phase, D4 zeroed;
  passer_realizability_delta sole king-race; drop kings_endgame passer; `CAPG_PAWN_RANK_CLAMP=275`).
- **`ENABLE_NPEDGE_DAMP_EG`** (endgame extension of the unbacked-pawn-placement damp; guard: defender non-pawn
  material >= `NPEDGE_EG_PIECE_FLOOR`=3250 so pure pawn endgames spared). Firing screen: deflates fen3
  +5.58->+3.83, P2 +4.91->+2.41 at MAX90; P3 holds +9.72->+6.86. passers.csv 150-pos guard: eg-damp 74->76
  match + 582->552 eval-err (improves); passer-v2 holds. All deterministic gates passed.

### A/B battery SEED 0 (200g each vs SF@2400, paired openings) — RESULT: all 3 levers WORSE than base.
| tag | config | score | collapses |
|---|---|---|---|
| ab_base_s0 | shipped DEF-5 | **42.0%** | **75** |
| ab_damp_s0 | `ENABLE_NPEDGE_DAMP_EG=1` (MAX90) | 32.0% | 98 |
| ab_v2_s0 | `ENABLE_PASSER_V2=1` | 37.5% | 101 |
| ab_both_s0 | both | 37.0% | 96 |
All three down on score AND collapses (+21..26) on identical seed-0 openings. MAX90 damp worst (−10%).

### A/B 2-SEED verdict (adding seed 1 + gentle MAX40) — DIRECTION DOES NOT PAY OFF
| config | s0 | s1 | avg score | avg collapses |
|---|---|---|---|---|
| base (DEF-5) | 42.0 | 38.0 | **40.0%** | 82.5 |
| v2 | 37.5 | 40.0 | 38.75% (−1.25, noise) | 97 |
| damp MAX40 | 36.2 | 38.5 | 37.35% (−2.65) | 95.5 |
| damp MAX90 | 32.0 | — | 32.0% (clear NO-GO) | 98 |
- **No lever improves score or collapses.** v2/damp40 are within ~2-seed noise (±2.5%) of base but on the LOW
  side; MAX90 is a clear loss. Collapse count TRACKS score here (lower score => more peak-then-lose games), so
  it gives no independent lift. **Despite clean deterministic screens** (deflate fen3/P2, P3 holds, passers.csv
  improves), the gain did NOT transfer to play.
- **Interpretation (matches the pre-build worry):** the over-positive endgame pawn placement is **load-bearing
  for our shallower search** — SF can give ~0 advancement because deep search compensates; deflating it here
  removes a crutch our ~2000/EBF-3.2 search relies on, so net play is neutral-to-worse. Same shape as capgains
  (load-bearing for move-select) and mobility-fixed-node.
- **VERDICT: NO-GO at all tested magnitudes; both levers stay default-off.** The valuable output is the
  DIAGNOSTIC (pt_pawns != passer; over-positive eg placement; SF study), not a shipped lever.

### Bench sweep (WAC solved/300, STS/3000) — confirms the joint no-go
| config | WAC | STS | 2-seed game | collapses |
|---|---|---|---|---|
| base | ~240 | 1606 (53.5%) | 40.0% | 82.5 |
| v2 | 240 | **1606 (=base)** | 38.75% | 97 |
| damp MAX40 | 242 | **1606 (=base)** | 37.35% | 95.5 |
| damp MAX90 | ~237-240 | 1556 (−50) | 32.0% | 98 |
- **v2 + damp40 are BENCH-CLEAN** (WAC ≈ base, STS identical to base) — passed the bench gate, were
  game-tested, showed NO game/collapse benefit. **damp90** dents STS −50 AND games −8%.
- **All NO-GO for the BLANKET levers. Levers stay default-off; nothing committed beyond DEF-5.**

### HYPOTHESIS TEST (2026-07-19): "needs deeper search" is FALSIFIED — it's EVAL, not depth
`diagnostics/sf11_depth_test.py` — SF11 (classical HCE, apples-to-apples) searched at OUR leaf depth 12:
| pos | our eval | SF11@12 | SF11@24 | SF18@22 |
|---|---|---|---|---|
| fen3 | +5.58 | **+0.76** | -0.00 | -0.30 |
| P2 | +4.91 | **-1.15** (wrong sign for us) | -1.39 | -1.73 |
| P3 (genuine) | +9.72 | **+4.54** | +7.35 | +4.50 |
**SF11 at OUR depth already evaluates these correctly; we do not.** + the recorded ~590-Elo equal-depth gap
(`eval_vs_sf11.py`) => **our per-node EVAL is the bottleneck, NOT search depth.** So the earlier
"load-bearing / needs deep search / lane closed" reading was WRONG. Corrected conclusion:
- The over-read is an eval bug that is **FIXABLE at the depth we already reach** (not a depth wall).
- The blanket levers failed because a **global scaledown is not the right fix** (SF11 gets these right with
  NUANCED CONDITIONAL eval, not a blanket damp; our damp fixed fen3 but distorted P3 -> net ~0).
- **LANE RE-OPEN with a CONDITIONAL mechanism.** Validated next direction (user's idea): material-imbalance-
  conditional pawn scoring, SF11-quadratic-imbalance style — default array for balanced material, a specific
  predetermined array swapped in for a detected imbalance (e.g. 4P-vs-minor). Tune the arrays; screen on
  fen3/P2 (deflate) + P3 (hold) + passers.csv, then games. One thing at a time.

## 2026-07-20: KAUFMAN material-imbalance term — FIRST POSITIVE eval lever (2-seed; confirmation running)
Full note: `dev_notes/kaufman-imbalance-2026-07-20.md`. Public Kaufman model, coefficients FIT from our data
(`kaufman_fit.py`, ridge on SF11-residual; rediscovered correct chess: knight×pawn +, rook×rook −). Gated
`ENABLE_KAUFMAN_IMBALANCE` (default off, byte-id; skips flat pair bonuses when on). Deterministic: eval-gap
1.285→1.223, STS identical, WAC −3, fen3 +5.58→+4.90, **P2 +4.91→+1.74**, P3 holds.
| config | s0 | s1 | avg score | avg collapses |
|---|---|---|---|---|
| base | 42.0/75 | 38.0/90 | 40.0% | 82.5 |
| **Kaufman** | 42.2/78 | 43.8/77 | **43.0% (+3.0)** | **77.5 (−5)** |
**6-SEED (final):** base = 42.0/38.0/38.0/41.2/38.8/40.0 avg **39.67%** (collapses avg **87.67**);
Kaufman = 42.2/43.8/43.2/37.5/40.0/38.2 avg **40.82%** (collapses avg **81.67**). **+1.15% score, −6 collapses
(−6.8%); Kaufman wins 4/6 seeds** (lost s3,s5). Extension seeds softened score from 4-seed +1.9% → 6-seed
+1.15% ≈ **0.85 SE (NOT conclusive on score alone)**; collapse reduction held (−6.8%, lower 4/6). **Honest
verdict: WEAKLY POSITIVE** — score CI includes 0, but signals CONVERGE positive (score+, collapses−, eval-gap↓,
P2 deflate, no bench decay, rediscovers correct chess) and NOTHING points negative. First & only game-positive
eval lever this session. **Ship candidate (user's call) — lean ship on convergence, but score not proven.**
Validates the chain: eval-not-depth → SF-study → fit our own material term → win.

## KS eval-gap dossier (2026-07-20) — the DOMINANT eval lever, next lane
`diagnostics/ks_gap_dossier.py` → `ks_sets/ks_underread_vs_sf11.txt` (36/400 STS positions SF sees |KS|≥1p and
we read <half). KS is the biggest slice of our 1.28p eval gap (mean|KS diff|=0.63p). Under-read failure modes:
(1) **deadzone** (units 11-12 < KS_FLOOR → 0); (2) **difference-cancellation** (symmetric units 18v18 → our
danger_white−danger_black ≈ 0, but SF sees asymmetric danger); (3) **detector blind spots** (uW=uB=0 yet SF
sees ±1.5); (4) **magnitude** (when we fire +1.8, SF is +4.5 = 2-3× larger). Next-session KS work targets these.

## Next class to diagnose — DEF-5 baseline profile (2026-07-19, from ab_base_s0+s1 = current shipped engine)
Pooled **165 unique DEF-5 baseline collapses** -> `diagnostics/ks_sets/def5_baseline_collapses.txt`
(`diagnostics/pool_def5_collapses.py`). **The worst-swing collapses are overwhelmingly MIDDLEGAME
TACTICAL/ATTACKING positions** (queens+pieces on board), peaking at +4..+18 pawns — several at MATE scores
(e.g. peak 9,999,978 -> lost) — then collapsing to a loss. **NOT the endgame-pawn class.** This is
middlegame over-confidence / broken-refutation (we see attacks/mates that don't hold), tied to our soft depth
(EBF ~3.2, ~2.5-3x shallower than SF). Cross-ref older [[collapse-is-tactical-not-static]].
- **NEXT (user-steered): SF18-search triage** of the worst N — separate "eval over-read" (SF18-static also
  high, our magnitude too big) from "search-horizon" (SF18-static modest but SF18-SEARCH refutes our line).
  The latter is a SEARCH/refutation problem (extensions/verification/qsearch), NOT an eval lever. Pick ~3
  middle-layer FENs for the user before building anything. Update this ledger each round.

## 2026-07-21 — KAUFMAN SHIPPED + collapse-classifier tooling + STS-artifact correction
**Kaufman shipped default** (`2a8d127` scaffolding byte-id + `6bd9d66` flip). 6-seed 200g vs SF@2400:
+1.15% score, **ks_attack collapse CLASS down 6/6 seeds (-20%)**, total collapses -6.8%. Scale-50 tested
overnight = equal-within-noise on score/collapses but only 4/6 on the target class → scale-100 ships.
NEW byte-id ref: **WAC 41,586,391 / 240 solved** (was 39,914,378 / 243).

**New tooling (the categorical instrument the ledger needed):** `diagnostics/collect_collapses.py`
(unify all games/*/collapses.csv → `ks_sets/collapse_dataset.csv`) + `diagnostics/classify_collapses.py`
(per-position class + `--vanish A B --seed N` cross-run attribution). **KEY: per-CLASS signal is STABLE
across seeds where TOTAL churns/is-noise** (Kaufman: ks_attack -6/6 seeds while total rose at one seed).

**Profile AFTER Kaufman (6 seeds):** positional ~360 (DOMINANT), ks_attack ~106, material ~15. KS still live
(~18/seed) but **positional is 3.4× larger = likely the bigger opportunity.** Surviving KS-attack fens →
`ks_sets/post_kauf_ks_collapses.txt` (115, NEEDS SF18-validation before trusting — heuristic labels).

**STS-ARTIFACT (methodology):** `wac`/`sts` subs consume arg1 as TAG → knobs passed without a tag are NOT
applied. This session's "STS held" claims were that bug. Correct STS (proper tag): baseline 1606; Kaufman -79;
KS_WEAK=3 -82; aim1/2/3 -103 — ALL regress STS. Games VALID (vs_sf.py --our-config, stderr-verified) so
Kaufman's win stands. STS is DETERMINISTIC but JAGGED in eval-scale (Kaufman 25/50/75/100 = 1575/1573/**1448**
/1527) = a GUARD not an optimizer. **KS aim + KS_WEAK NOT ready** (regress STS; aim net-zero on its own class =
double-counts existing detection) → deferred to the double-count dissection.

## 2026-07-21 (cont.) — KS RECALIBRATION by data-fit: safe partial win + detection diagnosis
Built the durable **position bank** (`build_position_bank.py` → `ks_sets/position_bank.csv`, 1852 labeled from the
33,683-game archive; SF18 truth on 500 via `add_sf18_labels.py`; schema `dev_notes/position-bank-schema-2026-07-21.md`)
and the **constrained WIN%-space fit** (`ks_fit.py`+`_ks_fit_eval.py`; Lichess k=0.00368208; SF18-gated SF11-KS
target; per-tier controls + directional/magnitude verdict). Code: gated `KS_MIN_ATTACKERS` gate + `KS_ATT_PRODUCT`
(cpp_bitboard.cpp king_safety_danger; default 0 = byte-id, WAC 41,586,391/240).
- **GOLDEN metric (per user): SF11-static-KS CAPTURE on the SF11∩SF18-AGREE set** (SF11 agrees SF18 on 450/500;
  370 KS-danger agree positions). Aim at SF18, score by SF11-static capture where they agree, side with SF18 on
  the 50 disagreements (14 hard-excluded).
- **RESULT: safe win = 0%→~31% direction / 0%→13% magnitude capture on golden targets** via `KS_FLOOR 13→6` +
  modest weight; calm/working controls held (~89-109% capture on working = we match SF11 there). **Coordination
  `KS_ATT_PRODUCT`/gate = NO-GO** (over-fires: wrongsign 1→13, controls break) — the constrained fit + directional
  metric caught the blowup deterministically.
- **DETECTION TRACE (per-zone-square, KS_TRACE):** the remaining misses are NOT a weak-square-definition gap —
  `weak=0` is CORRECT (squares have defenders; SF finds 0 weak too). The real signal is **per-square OVERLOAD**:
  FEN4 g6 attacked by N+R+Q, defended by 1 pawn (3v1). Our weak-test discards pawn-defended squares; our additive
  weights under-count concentration; the failed product ignored defenders (→ over-fired). Candidate lever =
  **attackers − defenders per zone square** (discriminative: ~0 on defended kings).
- **CAUTION (user): OvD DOUBLE-COUNT.** Overload overlaps the offense-vs-defense imbalance term → the fit must
  include OvD/imbalance knobs and fit the WHOLE eval's win% jointly, not KS in isolation.

## 2026-07-21 (cont.2) — CAPG_PIN BUG FIXED (the real win) + over-read corrections
**KS lever fully characterized (bounded, not a big win):** safe ~31% dir / 13% mag capture via KS_FLOOR 13→6 +
KS_SAFE_CHECK→8 + KS_ATTACK_COUNT→2 (STS +22, WAC −4, symmetry 0, controls held). ALL coordination variants
(KS_ATT_PRODUCT / KS_MIN_ATTACKERS gate / KS_OVERLOAD attackers−defenders) = NO-GO: overshoot the KS-active
controls (our units don't SEPARATE dangerous 12u from safe 10u; SF separates 15×; it's a DISCRIMINATION gap, not
a range gap — the fit REJECTS more KS_CAP/KNEE/MAG range). Remaining KS capture needs SF-granularity per-piece
DETECTION (defenders as per-term GATES, not our blanket overload), a rebuild.
**CORRECTIONS (user caught me):** (1) my "piece-activity over-read +6.68" was an ARTIFACT — the `pieces`
breakdown key is a CUMULATIVE snapshot (includes material), so the per-term sum double-counted. Placement is
small. (2) OvD-reduction lead DEBUNKED by direct measurement: imbalance is only 4% of over-reads (+0.19 of
+5.12); the fit's IMBALANCE_SCALE↓/SCALE_ATTACK_LAYER↓ signal is target=SF18 NOISE, not real. (3) Kaufman is
CLEAN (0 over-read contribution). So carry ONLY the validated KS knobs; leave OvD/product/attack-layer.
**THE REAL WIN — CAPG_PIN (user's diagnosis):** `approximate_capture_gains` counted ILLEGAL captures by
absolutely-pinned pieces (FEN3 `5k2/pp2r2Q/2n2q1p/3p2p1/8/BN4P1/P4P1P/4R1K1 b`: Re7xh7 wins the Q +10, but Re7
is pinned by Ba3→Kf8). The fix (`ENABLE_CAPG_PIN`, cpp_bitboard.cpp:7574 slider_blockers off-ray drop) was
IMPLEMENTED but gated OFF. **Enabled default.** Correct chess law + **STS 1527→1555 (+28), WAC 240→247 (+7)** —
improves BOTH benches; FEN3 flips Black +5.37 → White +6.37 (SF18 +7.10). Rare (10/500 bank positions; 5 big
fixes / 3 small regressions = accidental-cancellation, fix elsewhere). **NEW byte-id ref: WAC 247 / 39,971,153.**
Over-reads that REMAIN after the pin fix are mostly SF18=±99 forced MATES (search-bound) — the capture_gains-
without-refutation class, not static-eval bugs. **Night build = Kaufman + CAPG_PIN + KS config (3 independent
validated pieces); game-test + RECOLLECT collapses for a fresh FEN set (user).**
- **KS-STS INTERACTION (caution):** the KS config was +22 STS on pin-OFF but **−88 on pin-ON** (1555→1467). The
  win%-fit had NO STS in its objective (optimised bank win% only), so the config's STS was uncontrolled; + STS is
  jagged. → **KS HELD; re-tune on pin-on WITH an STS guard before shipping.** CAPG_PIN ships alone (clean).
- **SYSTEMATIC EVAL MAP vs SF11 (`eval_term_comparison.py`, mean |mag| pawns) = the roadmap beyond KS:** clean
  OVER-read = **Space/central ~9×** (0.46 vs 0.05) → fit `CENTER_*_MULT` down. Clean UNDER-reads/missing:
  **KingSafety** 0.44 vs 1.12 (KS config lifts to 0.76), **Threats** 0/0.34 (`ENABLE_THREATS` off), **Passed**
  0.06/0.33, **Mobility** 0/0.23 (`ENABLE_MOBILITY` off), **Imbalance** under (reconfirms OvD-not-over). Material/
  piece rows boundary-fuzzy (SF folds PSQT into Material). Next levers: Space-down, enable+fit Threats/Mobility,
  Passers-up — all attackable with the win%-fit + position-bank framework.
