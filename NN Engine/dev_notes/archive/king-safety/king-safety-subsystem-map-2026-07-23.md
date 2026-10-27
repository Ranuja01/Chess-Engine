# King-Safety subsystem MAP (2026-07-23) — the reference for the KS over-read / consolidation work

Built like the passer inventory: a bill-of-materials for every place king-danger is credited, the double/
triple-count map, and the consolidation decision. Sources: fable KS-map agent + direct code verification +
the FEN-3/4 `KING_SAFETY_MAG=0` differential. **Confidence tags: [E]=empirically confirmed, [C]=code-verified,
[F]=fable-read (plausible, not independently re-verified).**

## Live-vs-dead (the framing correction — I had this wrong at first)
- **DEAD: `get_latent_threat_score`** (~cpp_bitboard.cpp:5338-5586, the `black_increment−white_increment` fn with
  the `+4` presence offset, `×2` sparse-defender doubling, `÷3`, `+75` central-file bonus). **KS v1
  (`ENABLE_KS_REPLACE_LT=true`, search_engine.h:677) SKIPS it** at ~6708. **[E]** proof: an `ENABLE_KS_DEBUG`
  dump placed in this function NEVER fired. ⇒ my earlier "+4 offset / ×2 doubling" analysis was of DEAD code.
- **LIVE: unit-KS** = `king_safety_danger` (5038-5259) → `king_safety_score` wrapper (5267-5279). **[E]** proof:
  `KING_SAFETY_MAG=0` zeroes exactly the +7.20 `king_safety` term.

## Layer inventory (LIVE king-danger contributions)
| # | Layer (symbol, line) | What it credits | Scale / key knobs | Phase |
|---|---|---|---|---|
| 1 | **unit-KS** `king_safety_danger`→`king_safety_score` (5038-5279); call 6733 | attack-units on each king zone → `ks_safety_table[units]` (units²/`KS_DIVISOR=4` to knee 12, then linear, cap 80), netted white−black | **×30** (`KING_SAFETY_MAG=3000`/100) [C]; att N2/B2/R3/**Q5**, `KS_ATTACK_COUNT=1`, `KS_WEAK=2`, `−KS_SHIELD=2`, `KS_OPEN_FILE=2`, safe-check `KS_SAFE_CHECK=3`/`_DEF=5`, `KS_NO_QUEEN=−6`; `KS_FLOOR=13` deadzone; **`KS_MIN_ATTACKERS=0` ⇒ a LONE QUEEN is never gated out** [C] | tapered: full ≤48, 0 ≥104 (`ks_phase_taper`) — the ONLY real KS taper [F] |
| 3 | **king mg shelter** `evaluate_kings_midgame` (~2875-3024) | flat shelter **185mp/shield-pawn**, **75mp** partial; + `attackingLayer`-derived `baseIncrement` (≤×4) shelter/exposure per ring square; writes O/D scores | 185/75 flat (killed only by `KS_CONSOLIDATE`, default false = LIVE); baseIncrement UNGATED [F] | mg branch only, hard switch |
| 4 | **`attackingLayer` + OvD** `setAttackingLayer` (~7993); OvD imbalance (~6794) | attack map is **king-centric**: +inc per king-2-ring sq, **×`ATTACK_OPEN_MULT=5`** on open sq near king; added per-piece into `total` AND into off/def scores; OvD then re-spends `(off−max(def,0))×IMBALANCE_SCALE=3` | ATTACK_OPEN_MULT=5, IMBALANCE_SCALE=3 [C]; UNCONDITIONED, UNTAPERED [F] | separate mg/eg tables hard switch; OvD all-phase |
| — | modulators (all default OFF): `MOD_KS_BACKING=0` (damp-only material backing), `MOD_KS_CONTROL=0` (scale by O/D control edge — note: circular, that signal is built from the same king-zone attackingLayer), `KS_LIGHT_MAG=0` | | | |
| — | KEEP piece-local: endgame king PST `whitePlacementLayer[KING]` (~4420, genuine centralization); static threats (piece-on-piece); passer king-race (passer subsystem) [F] | | | |

## THE OVER-READ IS TRIPLE-COUNTED (the prize) [E on the numbers, F on the mechanism]
A queen near an exposed central king credits the SAME facts in three LIVE, unconditioned places:
- **unit-KS** (layer 1) → FEN-3 `king_safety +7.20` (×30, lone queen fires since KS_MIN_ATTACKERS=0)
- **attackingLayer** (layer 4) → per-piece king-zone pressure into `total`, ×5 on open squares
- **OvD imbalance** (layer 4) → re-spends the O/D scores → FEN-3 `imbalance_white +2.16`
None asks "does the attack convert." This IS the fantasy-attack signature (matches memory's ks_attack triage:
"our own attacks over-read", and the c5/OvD lead).

## FEN-3 vs FEN-4 DIFFERENTIAL (empirical, `KING_SAFETY_MAG=0`) [E]
| | with KS | KS=0 | SF18 | ideal KS | read |
|---|---|---|---|---|---|
| #3 `3r4/ppp3Q1/nq2k2p/7N/3r2P1/2N1B2P/PP3P2/5bK1 w` (over-fire) | +10.75 | **+3.55** | +3.04 | ≈0 | KS is the ENTIRE over-read; killing it ≈ SF |
| #4 `2n2b1r/1pB1k3/1p4Qp/1N6/4P3/Pn2P3/1q2BP1P/5K2 w` (correct) | +10.52 | **+7.82** | +8.69 | ≈+0.9 | KS is a minor bonus; MATERIAL/imbalance carry it |
⇒ Our KS over-credits attacks broadly; the "correct" case survives only because material/pieces carry it, NOT
because KS is right. #3 lives ENTIRELY on phantom KS.

## CONSOLIDATION recommendation (fable + my double-check)
- **Yes to one phase-blended `evaluate_king_safety()`** — BUT the big lever is **layer 4 (attackingLayer ×5 +
  OvD re-spend), not unit-KS.** Merging only unit-KS + shelter (finishing the `KS_CONSOLIDATE` stub, which today
  kills only the 185/75 flat and leaves baseIncrement + O/D writes live) = the SAME incomplete-consolidation trap
  that sank PASSER_V2.
- **The real prize = TUNABILITY:** today you can't lower the ~3-pawn fantasy reading without touching
  KING_SAFETY_MAG + ATTACK_OPEN_MULT + IMBALANCE_SCALE in three places. Consolidate → one place → one
  **realizability gate on the WHOLE king-danger budget** (not one third).
- **RISK / TRAP:** the mg-king shelter baseIncrement feeds `whiteDefensiveScore`/`whiteOffensiveScore` used by
  OvD ELSEWHERE — you must DECOUPLE the O/D accumulators from `total` before lifting shelter (the clamp+smear
  that killed PASSER_V2).
- **SOBER CAVEAT:** KS re-work has been GAME-NEUTRAL every time. Expect cleanliness/tunability, NOT direct Elo.
  Verify byte-id gate-off + per-class collapse profile, exactly like PASSER_V3.

## THREE PATHS (decision pending)
- **A — full KS consolidation** (one home; de-king attackingLayer OR route OvD-king-zone through one realizability
  gate). Cleanest/biggest; riskiest (O/D decoupling); likely game-neutral + tunability upside.
- **B — cheap targeted gates NOW** (`KS_MIN_ATTACKERS≥2` to kill lone-queen fantasy + `MOD_KS_BACKING` on).
  30-min A/B, but only touches the unit-KS third.
- **C — go at the biggest channel** — the OvD/attackingLayer king-zone re-count (the 07-22 OvD lead),
  realizability-condition THAT.
- Lean: **B as a cheap probe** — if gating the KS third is game-neutral again, the money is in layer 4 (C) and we
  scope the consolidation (A) around layer 4, not unit-KS.

## STEP-0 PROBE (2026-07-23) [E] — the budget responds; material-backing is the right gate, control is circular
User's key refinement: **FEN-3's KS is DIRECTIONALLY CORRECT** — SF18 +3.04 sees white better DESPITE being down
material, i.e. the attack genuinely beats the material deficit. So the goal is **magnitude/blowup CONTROL toward
SF's modest positive, NOT killing KS** (keep #4 fully credited, keep #3 positive). Probe on the current binary
(dissect_fen, env works):
| config | FEN3 (SF +3.04) | FEN4 (SF +8.69) |
|---|---|---|
| base | +10.75 | +10.52 |
| KS_MIN_ATTACKERS=2 | +10.75 (NO-OP, multi-attacker) | +10.52 |
| **MOD_KS_BACKING (300/600)** | **+7.15 (−3.6)** | **+9.17 (−1.35)** |
| MOD_KS_CONTROL (300/600) | +17.95 (WORSE) | +13.22 |
- **`MOD_KS_BACKING` DISCRIMINATES** (contra my earlier worry that material can't tell 3 from 4): it damps the
  phantom (#3, −3.6) MORE than the real one (#4, −1.35) because #3's attacker is down MORE material, and it keys
  on the DEGREE (`min(0, threat_side_edge)`). #4 lands +9.17 ≈ SF. Saturates by 300 (can't fully fix #3, still
  +7.15) ⇒ need a STRONGER/FINER version acting on the WHOLE budget, not just unit-KS.
- **`MOD_KS_CONTROL` = WRONG DIRECTION** (amplifies) — confirms fable's circularity (built from the same king-
  zone attackingLayer). Exclude, or invert.
- **`KS_MIN_ATTACKERS≥2` = no-op for multi-attacker phantoms** (still useful for lone-queen ones).
- ⇒ The realizability gate should be **material-backing-based** (MOD_KS_BACKING-like), stronger, on the whole
  king-danger budget. This is the gate the consolidation should build. Budget responds ⇒ consolidation viable.

## ⚠️ SCOPE CORRECTION (user chess insight, 2026-07-23) — it's NOT "purge duplicates", it's a TIME-HORIZON spectrum
The three "layers" are NOT redundant copies of one fact — they are three time-horizon LENSES on king-ward
pressure, and (per the user) all three are legitimate:
- **attackingLayer = PLACEMENT / piece-direction** (central-square attacks, x-rays, pieces aligning toward an
  eventual attack). Matters even with NO attack yet; king-heavy ON PURPOSE. **KEEP IT — do NOT de-king it.**
- **OvD (offensive/defensive scores) = LONG-TERM attacking POTENTIAL** ("someday"); already part of the KS
  tuning; general pressure that's king-heavy but legitimate.
- **KS = the ONCOMING STORM** (near-term, closer to realizable).
- **[E] The FEN-3 data proves this**: `KING_SAFETY_MAG=0` → +3.55 (which INCLUDES OvD imbalance +2.16) ≈ SF18
  +3.04. So OvD + placement were paying the RIGHT amount; the ENTIRE over-read was the KS (storm) lens over-
  valuing a storm that doesn't arrive.
- **⇒ REVISED, SAFER SCOPE:** the realizability gate applies to the **KS (oncoming-storm) lens ONLY**. **attacking
  Layer stays placement; OvD stays long-term potential (coordinate KS+OvD as a near/far PAIR, don't purge OvD).**
  This LIKELY AVOIDS the O/D-decoupling trap (we're not ripping king credit out of placement/OvD). Much smaller,
  lower-risk than a triple-layer teardown — a targeted KS-realizability-gate + KS/OvD co-tune, fitting the
  game-neutral history. The "de-king attackingLayer / purge OvD king-zone" items below are SUPERSEDED — keep only
  the KS-lens gating + optional shelter-dedup.

## CONSOLIDATION BLUEPRINT (transfer doc) — what consolidates, what dupes, what survives, how to rebalance
Target: one gated `evaluate_king_safety()` (default-off, byte-id when off, phase-blended), like `evaluate_passers()`.

### SURVIVE (lift into the one home)
- **Unit-KS core** (`king_safety_danger`→`king_safety_score`, layer 1) — it's the right skeleton: it already has
  the ONLY real phase taper (`ks_phase_taper`) and the units components (attackers-by-type, `KS_WEAK`,
  `safe_checks`, `KS_SHIELD`, `KS_OPEN_FILE`, `KS_STORM`, `KS_NO_QUEEN`). Keep as the core; it becomes THE king-
  danger budget.
- The passer king-race, endgame king PST (centralization), static threats — **stay piece/subsystem-local** (not KS).

### PURGE the duplicates (fold their signal into the core, delete the separate credit)
1. **attackingLayer king-zone slice** (layer 4): the `×5 ATTACK_OPEN_MULT` open-square boost + the per-piece
   king-2-ring credit into `total` — DUPLICATES unit-KS's attacker/weak-square counting. **De-king the
   attackingLayer** (make it a plain space/activity map, no 2-ring king boost) so unit-KS solely owns king
   pressure.
2. **OvD imbalance king-zone component** (layer 4): `(offensive−max(defensive,0))×IMBALANCE_SCALE=3` re-spends
   the same king-zone pressure = the TRIPLE count. Route the king-zone O/D through the same realizability gate OR
   exclude the king-zone from OvD.
3. **mg-king shelter** (layer 3): the flat 185/75 per-shield-pawn + the `baseIncrement` shelter/exposure —
   DUPLICATES unit-KS `KS_SHIELD`/`KS_OPEN_FILE`. **Finish the `KS_CONSOLIDATE` stub** (today it kills only the
   flat 185/75 and leaves baseIncrement + the O/D writes live = incomplete, the PASSER_V2 trap).
4. **DEAD**: `get_latent_threat_score` (+4/×2/+75 increment fn) — already replaced by KS v1; delete with V2.

### ⚠️ THE TRAP (must handle first, same as PASSER_V2): the mg-king `baseIncrement` and attackingLayer credits
also WRITE `whiteOffensiveScore`/`whiteDefensiveScore`, which OvD consumes ELSEWHERE. So you must **decouple the
O/D accumulators from `total`** before removing the shelter/attackingLayer king credits, or OvD breaks globally
(the clamp+smear that sank PASSER_V2). Do this decoupling as its own gated, byte-id step.

### REBALANCE after dedup (round-based, like passers)
- Purging the triple-count DROPS the effective KS magnitude ~2-3× (it was triple-counted). So **`KING_SAFETY_MAG`
  (currently ×30) must be RE-FIT UP**, or the unit weights recalibrated, so a REAL attack (#4) lands at the right
  level. Stage-1: fit the consolidated KS ALONE on the KS corpus (phantom over-fires down, real attacks held) +
  benches. Stage-2: joint fit with passer(V3)+OvD+material+all, win%-space, no family regresses.
- The **ONE realizability gate** (material-backing-based — MOD_KS_BACKING-like but STRONGER; the step-0 probe
  showed it discriminates #3 from #4 but saturates too low) now acts on the WHOLE budget = the phantom control.

### RECONSIDER the formerly-default-off items (they may open up once single-counted)
With a clean single-counted, realizability-gated budget, each gated item can be judged on its OWN merit (does it
improve the KS-corpus fit) without the triple-count confound that made prior KS work game-neutral:
- **`MOD_KS_BACKING`** → becomes the core realizability gate (on the whole budget, not one third).
- **`KS_MIN_ATTACKERS≥2`** → cleaner lone-attacker phantom gate on a single budget.
- **`KS_AIM`/weak-square, `KS_OVERLOAD`** → these ADD attack detection; reconsider ONLY if the deduped baseline
  UNDER-fires somewhere (now safe to add — no double-count risk).
- **`KS_CONSOLIDATE`** → the vehicle; finish it.
- Expectation (sober): KS re-work has been GAME-NEUTRAL every time → this is primarily a TUNABILITY + cleanliness
  win (lower the fantasy reading in ONE place; gate the whole budget) — verify byte-id + per-class collapse
  profile, do NOT bank on Elo.

## ✅ IMPLEMENTED (2026-07-23) — consolidation is SMALLER than the blueprint (a byte-id finding forced a re-scope)
The consolidation shipped gated + byte-id-verified (`ENABLE_KS_V2=false` default, WAC 247/39,971,153 both gate-off
AND identity-crossover). What was built:
- **`evaluate_king_safety(wk,bk,phase,turn)`** (`cpp_bitboard.cpp` after `king_safety_score` ~5280) = the ONE home.
  A pure extraction of the former inline KS call-site block (unit-KS `king_safety_score` + MOD_KS_BACKING +
  MOD_KS_CONTROL), now the sole KS path (call site ~6733 just does `if (ks_active) total += evaluate_king_safety(...)`).
- **Tunable shelter IN-PLACE** (gated `ENABLE_KS_V2`): the flat 185/75 literals in `evaluate_kings_midgame`
  (2936/2939 white, 3003/3006 black) become `KS_SHELTER_FULL/PARTIAL × KS_SHELTER_MAG/100`. Identity at 185/75/100.
- **`MOD_KS_REALIZ` (+`KS_REALIZ_FLOOR`)** = the whole-budget realizability gate in the home: same material-backing
  signal as MOD_KS_BACKING but with its OWN floor so it can damp BELOW the shared `MOD_FLOOR=128` (0.5×) that
  MOD_KS_BACKING saturates at. **This is the real prize** — the step-0 probe's "+7.15 stuck" was exactly the 0.5×
  wall; MOD_KS_REALIZ can cut deeper. Default 0 = byte-id. (Independent of ENABLE_KS_V2; acts on netted `ks`.)
- **★ THE BYTE-ID FINDING that re-scoped the plan:** `evaluate_kings_midgame`'s return (`result_mid`) is
  **phase-BLENDED** (`mid_weight*result_mid/blend_range` in phase 41-69) AND feeds `square_values[r]=abs(blended)`
  which is read downstream in capture ordering (`~7683`). So relocating the shelter / `baseIncrement` king-`total`
  slice OUT of `evaluate_kings_midgame` into the post-loop home is **NOT associativity-safe** — it changes the
  blend AND square_values ⇒ not byte-identical. The Plan-agent's "already-separate-statements ⇒ free to move"
  argument missed this. ⇒ **Steps "de-king / re-home baseIncrement" (blueprint items 1 + the O/D-decoupling trap)
  are DROPPED**: they're unnecessary (the unit-KS budget was already ONE post-loop term) AND they contradict the
  SCOPE CORRECTION (attackingLayer/baseIncrement IS the placement lens we keep). The O/D-decoupling trap never had
  to be entered. Net: consolidation = wrapper + tunable shelter + whole-budget gate; NO O/D surgery.
- Knobs registered `search_engine.cpp:1020-1024, 1103-1104` + toggle-dump; inlines `search_engine.h` near
  KS_CONSOLIDATE / MOD_KS_CONTROL. `KS_CONSOLIDATE` kept as a legacy toggle (superseded by ENABLE_KS_V2 but
  harmless); dead `get_latent_threat_score` still delete-later (ship-time, with passer V2).
- **NEXT = the TUNING phase (not more structure):** build the KS corpus tier (phantom over-fires + real-attack
  guards) into the existing mixed set, then joint-fit KS+OvD (+ passer V3 ON, +material) in win%-space, RE-FIT
  KING_SAFETY_MAG up as MOD_KS_REALIZ takes over phantom control, judge by per-class collapse profile in games.

## ⛔ CORPUS VERDICT (2026-07-23) — MOD_KS_REALIZ / material-backing is DISPROVEN at scale
Ran the realizability gate against a re-tiered `diverse_corpus_ksv2.csv` (45 phantom-KS rows = KS fires & we
over-read SF18; 33 `ks_real` guards = KS fires & SF18 AGREES; + calm/working/crowded_safe/sts_guard families;
`ENABLE_KS_V2=1 ENABLE_PASSER_V3=1`). **Every `MOD_KS_REALIZ×KS_REALIZ_FLOOR` config RAISED phantom MSE and
broke guards** (`ks_realiz_sweep.py`): the phantom rows are NOT under-backed-attacker positions (val MSE flat —
the gate never fires on them), and where it does fire it damps the REAL guards. **The step-0 FEN-3 probe was a
single under-backed position; material-backing does NOT generalize.** ⇒ do NOT ship MOD_KS_BACKING/REALIZ; keep
them dormant (default 0, byte-id) as documented dead-ends.
- **BUT KS *is* the systematic driver** (`ks_phantom_strict.py`, direction-aware): on **41/45** phantom rows the
  KS term is large AND aligned with the signed over-read, accounting for **~59%** of it. (My earlier "signed-mean
  ~0 ⇒ KS not the driver" was an ARTIFACT of the over-read pointing at White on some rows / Black on others; per
  row KS lines up with it. Corrected.) Secondary driver = `pt_pawns` (mean_abs 1.16), tying to the deferred
  doubled-pawn/passer FEN-1/FEN-2 work.
- **The REAL obstacle = the phantom-vs-real DISCRIMINATOR, not the culprit.** Phantom |KS| (1.98) and real-attack
  guard |KS| (1.61) OVERLAP, and SF18 truth on high-KS positions ranges from 0.00 (phantom) to +7.46 (real
  crush) at the SAME |KS| — the separator is "does it convert" (king escape squares / real safe checks /
  defensive resources) = the TACTICAL signal search resolves and static eval historically can't (why KS re-work
  is GAME-NEUTRAL every time). A blunt |KS| cut collateral-damages the real attacks. Any future KS accuracy lever
  must find a NON-material, NON-magnitude convertibility signal — or accept this is search's job.
- **Consolidation KEPT regardless** (user-endorsed): the one home + tunable shelter is less code + easier feature
  isolation, byte-id 247, worth keeping/committing on its own merits independent of the (failed) gate.

## 🔬 SF11 + ETHEREAL king-safety MODELS (2026-07-23) — the convergent structure + our gaps + our uniqueness
Read both sources against the mined fixable-KS FENs (ours over-fires 2-5p where SF11's King-safety term ≈0).

**CONVERGENT STRUCTURE (SF11 `evaluate.cpp:370-461` AND Ethereal `evaluateKings`) — both do this ⇒ it's correct, adopt:**
- **`kingDanger` = SUM of weighted real signals**, then a **NON-LINEAR transform with a DEADZONE**: SF11
  `if (kingDanger>100) score -= kingDanger²/4096` (mg) `/16` (eg); Ethereal `-mg*max(0,mg)/720` (mg) `-max(0,eg)/20` (eg).
  Below the knee = ZERO. Quadratic ⇒ only large COORDINATED danger explodes; a few pieces "pointing" never reach it.
- **SAFE-CHECK filter** (near-identical in both): a check counts ONLY if its square is `safe` = NOT defended by us
  (`~attackedBy[Us]` | weak&attacked-twice-by-them). Defended check squares → `unsafeChecks` (small) or ignored.
  = the literal "pointing vs pressuring" distinction (user's position-3 note: defended squares aren't danger).
- **WEAK squares** = ring squares attacked by them & NOT defended-twice by us (only Q/K defends) — only UNDEFENDED
  pressure scores (SF `+185×`, Ethereal `SafetyWeakSquares S(42,41)`).
- **NO-QUEEN discount = LARGE in BOTH** (SF `−873×!queen`, Ethereal `SafetyNoEnemyQueens S(−237,−259)`). Attacks
  rarely convert without a queen. **OURS `KS_NO_QUEEN=−6` is TRIVIAL — the single biggest miscalibration.**
- **ATTACKER GATING**: Ethereal scores only if `kingAttackersCount > 1 − popcount(enemyQueens)` (≥2 attackers OR
  1+queen). **OURS `KS_MIN_ATTACKERS=0` = DISABLED** (lone attacker always fires — the FEN-2 lone-queen phantom).
- **DEFENDER proximity** subtracts (SF `−100×knight-defends-king`, `−4×kingFlankDefense`; Ethereal `KingDefenders[]`
  pawns/knights/bishops in king area).

**SF11-ONLY (consider): `−6×mg(score)/8`** — our OWN eval/initiative REDUCES our king's danger (if we're winning
elsewhere, the attack matters less). Directly fixes the "we're up material but panic about our king" SIGN FLIPS
(positions 1&4). Ethereal lacks this explicit term.

**ETHEREAL-ONLY (candidate for our uniqueness): attack-density NORMALIZED by king-area size**
`9.0 * kingAttacksCount / popcount(kingArea)` — scales raw attack count by how big the zone is.

**OUR GAPS (why we net toward the attacker):** (1) no-queen discount trivial (−6 vs −237/−873); (2) attacker
gating off; (3) no own-initiative offset; (4) safe-check/weak-square filtering looser (we count attackers pointing
regardless of whether the target square is defended); (5) deadzone+transform far weaker (`KS_FLOOR=13`+table vs
`>100`+quadratic).

**OUR UNIQUENESS (NOT a copy — the passer playbook): GRADED per-square CONTEST.** SF11/Ethereal use ~BINARY signals
(square weak? safe-check? yes/no). We have `num_attackers[]`/`num_supporters[]` per square — so instead of "weak
square (undefended) → +185", grade the ring by `clamp(attackers − defenders, 0, k)` per square, win%-calibrated.
Match their STRUCTURE (safe-checks + no-queen + defender/initiative subtractions + quadratic-deadzone), DIVERGE on
graded-vs-binary square scoring + Ethereal's density normalization. Same identity we gave passers (graded control +
win%-calibration), not a different skeleton.

## ✅ ROOT-CAUSE CONFIRMED (2026-07-23) — `KS_DEFENDER=0` is the systematic over-count (ours, not a missing SF feature)
Triangulation (`ks_failure_hunt.py`: ours/SF11-static/SF18-search, filter SF11-right-we-wrong) + per-king component
dump (`ks_component_survey.py`, KS_DEBUG_DUMP) across the fixable-KS set. **Every fixable-KS position has zone
defenders (`defpc` 1-4) that subtract ZERO** because `KS_DEFENDER=0` (search_engine.h:756) — we tally the attack,
blind to the defense. That's the #1 driver, confirmed systematic (not the position-1 fluke). Example dumps:
`attsq/weak/safe/defpc/openf/units`: Ke6 `6/4/1/4/3/33`→ourKS+4.86 vs SF11−0.24; Kf7 `7/2/1/4/2/28`→+3.96 vs −2.53.
- Secondary over-counts: `KS_OPEN_FILE=2` firing on CENTRAL/exposed kings (openf 2-3 → +4-6, meaningless for a
  centralized king); safe-check `×5` on thin attacks. Missing: an INITIATIVE/material offset (SF `−6·score/8`) so a
  winning side's king isn't feared. **RULED OUT: the lone-attacker gate `KS_MIN_ATTACKERS`** — the `attpc=1` cases
  have an enemy queen present, so the ≥2-or-1+queen gate never fires. Not the lever (avoids the easy-knob trap).
- **A/B proof the levers move it** (position 1): default units33/+4.86 → `KS_DEFENDER=3` units21/+2.70 →
  `+KS_OPEN_FILE=0` units15/+1.62 (toward SF11 −0.24 / SF18 −4.58). Confirmed our own knobs, mis-set.
- **WE-BEAT-SF11 tier EXISTS (guard our edge):** positions where OUR static agrees with SF18 more than SF11 —
  incl. a KS one `5rk1/5q2/Pp4p1/n1p1p1P1/2PpQBP1/P2P4/8/3B2K1 w` (ours +2.14, SF11 −0.10, SF18 +1.53). Tuning KS
  toward SF11 must NOT regress these (corpus-level uniqueness preservation, per user).
- **Confirmed lever list for the JOINT tune:** (1) `KS_DEFENDER` netting [the systematic one], (2) `KS_OPEN_FILE`
  conditioned on king exposure [code change], (3) NEW initiative/material offset, (4) keep `KS_OVERLOAD` graded
  contest as our uniqueness. Tune ALL jointly WITH passer V3 + Kaufman + OvD across fixable + we-beat-guard +
  real-attack-guard + calm + phase buckets; metric = root win%-accuracy vs SF18; games decide. This is ROOT-accuracy
  work → search/pruning payoff ([[eval-accuracy-payoff-is-pruning]]).

## ★★★ BREAKTHROUGH (2026-07-23, via fable + SF11/SF15 ground truth) — it's PER-KING ASYMMETRY, not convertibility
The whole "static KS over-read = we over-value a non-converting attack on the enemy king" framing was **WRONG**.
Fable ran the actual SF11 binary + traced the code on FEN-1 (`8/1p6/4k3/1b2n3/p2q2P1/P3R2P/1P3Q2/3rN2K b`):
- **SF11's King-safety −0.24 is a NET, not "SF is calm."** SF11 trace: `White −5.30 | Black −5.06 | net −0.24`.
  SF charges the BLACK king −5.06 (HUGE, fully includes the f5 safe queen-check we also detect) AND the WHITE
  bare-h1 king −5.30. They nearly cancel. SF was equally alarmed about BOTH kings.
- **OUR bug = per-king asymmetry.** `det_ks_units_w/b = 4/33`: we charge the swarmed black king ~right (33) but
  massively UNDER-charge the airy white king (4) → net "+4.86 White winning" = SIGN ERROR (SF18 −4.58 Black wins).
  We're NOT over-reading the attacked king; we IGNORE that our own attacking king is just as exposed.
- **SF15.1 CONFIRMS the mechanism is DURABLE:** classical KS `White −3.00 | Black −2.82 | net −0.19` — same
  two-sided ledger SF11→SF15 (magnitudes softened, NNUE-companion). SF15.1 is the LAST classical-KS SF (16/17/18
  NNUE-only). Two independent SF versions agree the SYMMETRIC two-sided charge is the correct design.
- **Why the asymmetry (user's proximity hypothesis, likely):** our danger is PROXIMITY/zone-occupancy driven
  (attacker-presence + per-attacked-square count + weak + open-file adders) → lights up for a queen in the king's
  FACE (black e6) but misses a real but DISTANT safe check (…Qd5+ at h1) which our safe-check term detects yet
  weights trivially (3-5 units). SF's danger = safe-checks (780-1080, DOMINANT) + weak-ring, computed identically
  for both kings, + quadratic COMPOUNDING (ours goes linear above the knee, can't express "5 threats = mate") +
  defender activity subtracting pre-transform.
- **This explains every earlier dead-end** (breakthrough-gating, zone-size, safe-check-dominant regime all failed
  to separate) — they analyzed the WRONG king. The target is the WHITE (attacker's own) king under-charge.
- **CROSS-ENGINE fable analysis (our code + SF11 + SF15 + Ethereal) IN FLIGHT** — for the concrete minimal fix
  (respecting banked no-blanket-defender/overload/material/central failures). SF15 verified runnable under WSL
  (`stockfish_15_linux/.../stockfish-ubuntu-20.04-x86-64`, `setoption Use NNUE false` + `eval`).

## Housekeeping
- The `ENABLE_KS_DEBUG` dump (search_engine.h + reg) is currently placed in the DEAD `latent_threat` fn → never
  fires. To use it, relocate the dump into the LIVE `king_safety_danger` (per-king unit sub-components).
