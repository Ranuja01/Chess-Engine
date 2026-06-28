# Collapse-elimination campaign — running log

Persistent track: find the singular MISEVALUATIONS that lose real games (self-play-invisible, worst-case
Elo), fix each, validate by **position-fix + no bench regression + ACPL-vs-SF not worse ⇒ ship** (NOT
self-play). Plan: `~/.claude/plans/handoff-lossless-speed-campaign-tranquil-rose.md`. Validation model +
backlog: [[external-play-gaps]]. **Document every angle tried here** so we never re-walk a dead end.

**Baseline (current, post-combo1):** WAC d10 **252 / 67,931,145**, STS d10 **1503 (50.1%)** (OMP-pinned,
book-off). ACPL on `away_standard` n=112: MEAN 122 / MEDIAN 46.5 / blunders 21 (SF 0.25s). Dispatcher:
`overnight_runner.sh {wac,sts,cploss_probe,fenvs,tournament} ...`.

## Collapse corpus (the real-loss positions)
`diagnostics/_tal_gap_fens.csv` — 3 chess.com tal-BOT (~2705) losses, engine=White, from won/equal:
- **benoni-29** `r3r3/1b2qpbk/p2p2pp/1ppP4/P1B1P1PP/2N2Pn1/1P1Q1B2/R3R1K1 w - - 0 29` — winning-capture dodge:
  engine played **c4b3** (Bb3); the win is **a4b5** (axb5, +pawn, hits e8-rook, keeps g3-knight trap). Gap-T.
- **french-28** `5r2/p1qbn1pk/4p1pp/1p1pPr2/2pP1NQP/P1P2P2/R1PB2P1/1R4K1 w - - 8 28` — a-pawn march a4→a3→a2
  undervalued; engine shuffled. Gap-P (passed-pawn danger).
- **benoni 44-57** — the lost R-vs-passers ending the above walked into.

## Parked fixes to bundle (Phase 1) — all gated default-off, built+validated on OLD baseline (258/97.5M)
- **Gap-T `VERIFY_MARGIN`** (default 6000; fix 16000): search_engine.h:719, consumed search_engine.cpp:1971/2275.
  Wider VERIFY re-search catches the buried axb5 line. Old result: benoni-29 fixed, STS recovered.
- **Gap-P P1 `PASSER_ENEMY_CREDIT_PCT`** (default 100; fix <100): cpp_bitboard.cpp:5317/5381. Trims the
  wrong-signed enemy blockade/path-control credit on advanced passers.
- **Gap-P C1 `ENABLE_PASSER_BLOCKADE_QUALITY` + `PASSER_CONTEST_PCT`** (off / 30): cpp_bitboard.cpp:7360.
  A file-contesting rook/queen gets only CONTEST_PCT% of PP_BLOCKADE_PEN (un-zeroes a rook-contested passer).
- **Gap-P C2 `ENABLE_PASSER_KRACE_MG` + `PASSER_KRACE_MG_PCT` + `PASSER_KRACE_MAG`** (off/100/100):
  cpp_bitboard.cpp:6391/4655 — lifts the king-race realizability into all phases. WAC-costly (SPRT tier).
- Other gated `ENABLE_*_FIX` toggles (ROOK_DBLCOUNT/KNIGHT_MOB/ROOK_ENDGAME_CAP/QPREC_PHASE_GATE/
  ROOK_RANKWIN) — inventory + individually screen for the bundle.

## Log (newest first)

### 2026-06-28 — King-safety Phase B: tuned div6 OVER-FIT (discard); swap@600 = the marginal real candidate
**CORRECTED after rigorous re-check (the +22.7 was an illusion).** Move-match diagnostic (full 15-theme, all
SAME-SESSION apples-to-apples): untuned **swap@600 = 7755 vs base 7715 (+40, balanced)** — WINS attack/king
(AT +86, Center +85, Open Files +38, King Activity +2), modest non-king losses (Knight Outposts −47,
Recapturing −38, Undermine −38, 7th Rank −33, Square Vacancy −32). **Tuning was the MISTAKE: `KS_DIVISOR=6`
(div6) was tuned on a 4-theme king-SUBSET (AT/Center/Knight-Outposts/Recapturing), scored +73 there — but the
FULL-suite move-match is 7517 = −198 vs base** (it helped the 4 tuned themes and TANKED the other 11 =
classic over-fit). Deterministic STS (same-session, base re-run = 1568 exact): swap@600 1548 (−20), div6 1469
(−99). **So both clean benches agree div6 is WORSE; swap@600 is marginal.** The overnight TOURNAMENT was run
on **div6 (the over-fit config, not swap@600)**: batch1 368g pre-outage = +22.7 (small-sample noise); batch2
259g POST-power-outage = −16; pooled 627g = **+6.7 (flat)**. The cross-outage sign-flip = TIMED-tournament
machine-state confound (LIGHTNING depth ∝ CPU clock/load; batch2 ran on a freshly-rebooted machine). **.so
VERIFIED intact (base WAC 252/70,150,573 + base STS 1568, both exact) — the outage reverted NOTHING.**
**VERDICT: div6 OVER-FIT → DISCARD. swap@600 (`ENABLE_KS_REPLACE_LT=1 KING_SAFETY_MAG=600`, default knobs) =
the real candidate but MARGINAL (+40 move-match = +0.27%, −20 STS ≈ neutral) — never play-tested. There was
never a confirmed big win; +23/+73 were noise + over-fit.** **METHODOLOGY LESSON (the div6 trap): NEVER tune
eval knobs on a narrow theme SUBSET — it over-fits, trading away the un-tuned themes. Tune against the FULL
suite with the non-target themes held as CONTROLS, and gate EVERY candidate on the FULL-suite move-match (div6
looked +73 on the subset but was −198 on the full).** **▶️ NEXT: re-tune FROM swap@600 with control sets
(full-suite objective, king-themes-up subject to controls-held) → gate on full move-match + a CLEAN
post-outage tournament. If swap@600 can't be pushed past marginal, fall back to the COMPLEMENT (keep
latent_threat + add only KS safe-checks = prior clean +13 STS, no regression) or park the KS lever.**

### 2026-06-27 — King-safety swap Phase A: NEUTRAL platform found (swap@MAG=600 ≈ benches)
Go-forward = replace flat `latent_threat` with the high-DOF attack-unit `king_safety_score`, then data-tune
(plan `~/.claude/plans/handoff-lossless-speed-campaign-tranquil-rose.md`). **Phase A (structural swap) DONE:**
new gate `ENABLE_KS_REPLACE_LT` (search_engine.h, default false) — when on, SKIP the latent_threat add and
route king danger through king_safety_score (no double-count; needs KING_SAFETY_MAG>0). cpp_bitboard.cpp:
latent_threat gate `&& !ENABLE_KS_REPLACE_LT`; KS gate `(ENABLE_KS_REPLACE_LT || KING_SAFETY_MAG!=0)`.
**Byte-id OFF = 252 / 70,150,573.** MAG sweep (swap on, DEFAULT KS knobs) STS: 50→1404, 100→1408, 200→1513,
400→1522, **600→1548 (−20, ≈neutral within noise)**, 650→1458, 800→1475; **WAC@600 = 252/300** (no tactical
regression). **⇒ the untuned rich king_safety_score MATCHES the evolved latent_threat on both benches at
KING_SAFETY_MAG≈600** — the encouraging floor (user's point: a flat hand-set term shouldn't beat a data-tuned
richer one with real board-condition signals; the extra DOF — safe-checks/shield/open-files/defender-balance
— is all still at default). The MAG-sensitivity (550/600/650 = 1503/1548/1458) shows the default SHAPE wants
tuning. **KEY (user): king_safety_score is ~5-24× CHEAPER than latent_threat → matching it at fixed depth ⇒
potential DOUBLE win: speed (more depth at equal time, +Elo even at neutral fixed-depth strength) + positional
(once tuned). Measuring depth-at-equal-time next.** Two-level conditioning frame (user): Level-1 = the formula
already conditions danger on one-pass board features (LIVE, tune its weights = Phase B); Level-2 = make the
weights themselves functions of GLOBAL detectors (offense/defense, space, mobility) — the [[detector-
conditioned-knobs]] vision, but EVIDENCE-GATED (placement L2 overfit; king-safety is a better candidate
because king danger genuinely IS detector-driven — cheap-proof-first AFTER L1 wins). Downstream (user): a
cheaper+stronger eval revives the shelved eval-speed-dependent SEARCH items (improving heuristic etc. died on
per-node eval cost — "eval work pays off twice", [[improving-heuristic-shelved]]). **All commits PUSHED to
origin/NN-ENgine (was 63 ahead).** Phase A swap UNCOMMITTED (gated default-off).
The strategic pivot (user): make the placement/PST value a FUNCTION of cheap board-state detectors (space,
material, pawn structure, mobility, king-pressure) and Texel-tune it, to kill the `pieces` variance that the
term-attribution exposed. **Tested fairly + cheaply offline BEFORE any C++ build** (the "measure first" gate):
- Tooling (all UNCOMMITTED): `diagnostics/detector_placement_proof.py` (global-gain proof over corpus.csv),
  `diagnostics/gen_midgame_corpus.py` (curate IMPORTANT-MIDGAME: phase<64 & |SF|<250, reuse SF labels, add
  per-piece-type `pt_*` via ev_breakdown + detectors → `tune_data/midgame_corpus.csv`, 5037 rows),
  `diagnostics/fit_conditioned_placement.py` (rich per-piece-type × 14-detector ridge fit, train/test).
- **RESULT (held-out gap MSE, curated midgame): per-piece FLAT scale +14.6%; per-piece CONDITIONED +10.4%
  (WORSE); conditioning BEYOND flat = −4.9% (OVERFITS).** Global-gain version on near-equal/important-midgame
  added only +3.3–3.5% beyond flat, and positional detectors (mobility/pawn-struct/king-pressure) added ~0.
  **⇒ the placement variance is NOT detector-explainable by cheap hand-detectors — conditioning generalizes
  WORSE than a plain per-piece scale. The scatter is the NNUE-shaped hole (a learned eval has low placement
  variance because it captures square×context interactions hand-detectors can't).** Disconfirms the
  detector-conditioned-PLACEMENT thesis ([[detector-conditioned-knobs]], [[dynamic-conditional-eval]]) for this
  use. **The one generalizable lever = per-piece-type FLAT placement recalibration (+14.6% static, ~Texel of
  `SCALE_PLACE_*`)** — but it's the scalar approach (MSE≠play, prior SCALE_PLACE washed; per-piece-type at these
  magnitudes is the only untested variant → would need a PLAY/tournament gate, not MSE). Method win: a ~30-line
  numpy held-out fit killed a multi-week C++ subproject in minutes. Real variance-killer remains NNUE-as-eval
  (shelved). pawn=1000 / our_total Black-positive; SF_static White-POV cp; convert our→White cp = −our_total/10.

### 2026-06-27 — term-attribution over collapse positions: it's `pieces` VARIANCE, not passers
**2nd batch (20 games → 5 collapses; triage 1 EVAL / 1 PRUNING / 3 HORIZON) + term-attribution over ALL 5
collapse decision FENs (`eval_breakdown --fen`).** Answer to "is there a recurring CONCEPT error (e.g. passers
undervalued)?": **NO clean concept — the recurring offender is the `pieces`/placement term as VARIANCE.**
Gaps (our_static − SF_static): +3.3 (pieces +1.13), −4.1 (ALL terms ≈0 = missing concept), +5.6 (pieces +2.93
+capture_gains), −2.3 (pieces −1.81), −10.6 (mate outlier, capture_gains). **`passed_pawn_support` = 0.00 in
ALL five** (passers are NOT the pattern — and the campaign already fixed the passer hole that DID recur in the
real games, Gap-P). `king_safety`/`latent_threat` ≈ 0 too. **⇒ (1) the dominant residual eval error is `pieces`
placement scatter — too-high in over-reads, too-low in under-reads (VARIANCE, not a directional bias) = the
known central ceiling that scalar damping WASHES → the parked Texel/conditional lever ([[material-edge-
overvaluation]], [[dynamic-conditional-eval]]). (2) ONE under-read (gap −4.1) had ALL our terms ≈0 while SF saw
+3.7 = a MISSING CONCEPT we don't model, most likely king-safety/attack (our structurally crudest area,
[[king-safety-design]] — though the built attack-unit term tested Elo-flat).** Honest campaign state: the
recurring real-game point-holes (Gap-T, Gap-P, KPvK) are PLUGGED; the remaining gap is placement-variance +
under-modeled king-safety, NOT a tidy point-hole. Next lever = the hard Texel/placement-calibration meta-lever
or continued real-PGN point-mining (diminishing returns), NOT more LIGHTNING vs-SF batches (re-confirm the same).

### 2026-06-27 — vs-SF batch mined + 3-way triaged → known scatter wall, NO new clean hole
**16-game LIGHTNING vs-SF(2400) batch → 7 collapse points; 3-way triage (`triage_collapses.py`):**
**5/7 HORIZON, 2/7 EVAL, 0 PRUNING.** The HORIZON ones: our DEEP move equals SF's best (g0 h3h2=sfBest,
g4 f1g1, g7 d8b6, g9 h7g8) — the game blunder was a shallow LIGHTNING-depth miss, depth fixes it (search lane).
The 2 EVAL ones are PURE OVER-READS where our deep move ALSO equals SF's best (g1 d8d7, g12 d3c4) — not move
errors, just over-valuation. **g12 (the big one): our_static +7.18 vs SF_static +0.69 = +6.49 gap, driven by
`pieces` +4.94** (placement/PST) — i.e. the KNOWN central eval ceiling ([[material-edge-overvaluation]],
[[eval-precision-term-attribution]]): the `pieces`/placement over-valuation that every scalar damping WASHED
(variance, not scalar-fixable; the parked Texel/conditional lever). **⇒ vs-SF-LIGHTNING re-surfaces SEARCH-depth
+ the known placement scatter, NOT new KPvK-style clean structural holes. Clean structural holes come from
DEEPER-time real games (chesscom/tal-BOT).** Harness + triage VALIDATED and working; refined triage to classify
EVAL directly when deepMv==sfBest (move correct ⇒ pure eval), low-prune probe only when we persist in a move SF
dislikes. **OPERATIONAL: WSL→SF interop drops on every idle instance-restart (binfmt WSLInterop unregistered);
robust recovery = `wsl.exe --shutdown` + the long continuous job (batch+triage) in ONE Bash call with an
interop-ready wait loop — the busy instance holds interop; a `nohup`/foreground keepalive does NOT survive.**
Uncommitted: `selfplay/vs_sf.py`, `diagnostics/triage_collapses.py`, `_kpk_oracle.py`, `_chesscom_gap_fens.py`.

### 2026-06-27 — tal-BOT corpus re-confirmed + vs-SF collapse-mining harness BUILT
**tal-BOT re-confirm (current default-on build):** shipped fixes HOLD — benoni-29 → a4b5, french-33 ev≈0
(Gap-P intact). benoni 44-57 ending is GENUINELY lost (our moves match SF: bn-54 g2f3=g2f3 our −9.45 / SF
−8.12; bn-57 g5g6=g5g6 our −10.3 / SF −5.7). Only blemish = mild over-pessimism in rook-vs-connected-passers
(bn-57 −10.3 vs −5.7) but move-neutral → low priority. No new critical hole in the tal corpus.
**vs-SF harness BUILT (`selfplay/vs_sf.py` + dispatcher sub `vs_sf <elo> <games> [preset] [win_thresh]`):**
our engine (reuses `EngineProc`) vs strength-capped SF (`UCI_LimitStrength`/`UCI_Elo`, python-chess), alternating
colors; records OUR eval trajectory and auto-flags COLLAPSES (our-POV peak ≥ win_threshold then not a win) →
dumps the peak→drop run-up window to `games/<tag>/collapses.csv` (the corpus seed). Standalone (does NOT touch
the committed self-play loop), lower-risk than retrofitting SF-as-player into tournament.py. Syntax/import-clean;
UNCOMMITTED. **WSL→SF interop was DOWN (binfmt WSLInterop unregistered → 'Exec format error'); restored via
`wsl.exe --shutdown` + keepalive re-pin** (the documented recovery).
**SMOKE-TEST (2 games, SF elo 2400, LIGHTNING) → harness WORKS** (flagged 2/2 collapses with run-up FENs).
**▶️ KEY METHODOLOGY FINDING — LIGHTNING vs-SF surfaces HORIZON blunders, NOT eval holes.** Diagnosed game-0
(peak +2.17 → lost): we played **Bc3** at the game depth (d11) rating it +2.17, but SF says Bc3 LOSES
(+1.12 → −2.98); at d16-17 OUR ENGINE AVOIDS Bc3 and plays Bd3 (SF +0.5, holds) ⇒ the blunder was a
DEPTH/HORIZON miss, not an eval hole (depth fixes the move). The residual eval over-read is only modest
(+1.96 vs SF +1.12 = the known broad over-optimism). **⇒ At LIGHTNING, our shallow search (d11) loses to
SF's tactical shots = SEARCH-lane (at peak), not new fixable EVAL holes. The real eval holes (chesscom
rook-pawn, tal-BOT) came from STANDARD-time games where the engine searched deep enough to SEE the tactics
but MIS-EVALUATED. To mine eval holes, run vs-SF at DEEPER time (STANDARD), OR auto-triage each collapse
eval-vs-horizon (re-search deep: move-changes=horizon-discard, eval-stays-wrong-vs-SF=eval-keep).** Harness
deliverable DONE + validated; the strategy lever = time control / triage. NEXT: deeper-time batch or triage filter.

### 2026-06-27 — ✅ Phase 2 fix #1 SHIPPED: rook-pawn KPvK draw (chesscom-2200 conversion loss)
**Source:** `selfplay/external/chesscom_2200_white.pgn` (NN-Engine=White vs chess.com 2200 bot, won a pawn
move 29 then drew). Extracted endgame FENs via `diagnostics/_chesscom_gap_fens.py` (replay + material tally).
**Diagnosis — EVAL hole, not horizon:** White was +1 the whole game but the extra pawn collapsed to a lone
**h-pawn (rook pawn)** → dead-drawn K+h-vs-K (moves 57-69). Our eval scores those KPvK draws at **~+4870..+4960**
(`piece_value_boost`/mate-drive on a 1-pawn lead, NO draw detection); deeper search does NOT shrink it
(d6 +4406 → d14 +5060 → d18 +4893 — every leaf reads the same wrong way). This caused the half-point loss:
at move 56 the engine, seeing the resulting KPvK as +4.7, **traded rooks (Rxf5) INTO the dead draw**. SF
confirms every critical position = **0.00** (move 56 SF keeps the rook `f8a8`; even move 47 = 0, so the whole
ending was already drawn — no lost win, just the consistent over-read).
**Oracle (`diagnostics/_kpk_oracle.py`):** full KPvK retrograde solve (KQ-vs-K promotion shortcut), rook-pawn
files only = 83,238 states (6,526 WIN / 76,712 DRAW). Strict chebyshev-opposition rule
`defender_dist <= min(pawn_dist, attacker_dist)` to the promotion corner = **0 false-draws** (never flags a won
position drawn), catches 22,904 draws incl. the chesscom positions. (tempo variant catches 28,190, also 0
false-draws — not worth a side-to-move dependency the existing rook-pawn cases don't have.)
**Fix (gated `ENABLE_RP_KPK_DRAW`, default-off):** new lone-rook-pawn KPvK case in `is_practically_drawn`
(cpp_bitboard.cpp, after the KvK check), mirroring the existing KBP/KN rook-pawn blocks (returns 0 before the
material boost). search_engine.h:~356 flag + search_engine.cpp env-parse/dump.
**Validation:** byte-id OFF = **252 / 70,150,573** (exact baseline). ON: chesscom KPvK → **ev 0** (was
+4870/+4936); move 56 → **f8h8 (keeps rooks), no longer Rxf5**; benoni-29 still **a4b5** (shipped fix holds).
**No-regression PASS: WAC flag-on 252 / 70,150,573 (byte-identical), STS flag-on 1568/3000 (identical)** —
the fix touches ONLY rook-pawn KPvK so it never appears in either suite. **SHIPPED default-on**
(search_engine.h `ENABLE_RP_KPK_DRAW=true`; rebuilt, default-on WAC = 252/70,150,573, KPvK ev=0 with no
flag). **Gate decision (user):** a self-play tournament is uninformative for a self-play-invisible fix — ship
on verifiable position-fix + no bench regression (the established [[external-play-gaps]] validation model).
Baseline unchanged: **WAC 252 / 70,150,573 / STS 1568**. Tooling UNCOMMITTED: `diagnostics/_chesscom_gap_fens.py`,
`diagnostics/_kpk_oracle.py` (reusable KPvK oracle). Fix itself uncommitted (default-on in working tree).

### 2026-06-27 — is_light detour CONCLUDED (dead); ▶️ Phase 2 = SF-opponent gap-mining (decided)
**is_light eval-speed track DEAD** (between the bundle ship and now): stand-pat is the LEAF eval for quiet
positions → cheapening craters positional STS (mode1 1266 / mode2-surrogate 1243 vs 1501); improving/null-move
already use cheap_eval (improving SPRT-null); futility-light bench-AMBIGUOUS (+10 standalone vs −107 in-sweep
= timed-depth is wall-clock/thread noisy). Lesson: BOTH proxies fail for small effects — hunt BIG worst-case
fixes, not micro speedups. Built gated/byte-id/UNCOMMITTED (KS_LIGHT_MAG + light queen/knight mobility + light
KS surrogate; superseded). [[lighteval-standpat-is-leaf]].
**▶️ PHASE 2 SOURCE DECISION:** primary = **SF-opponent games** (our engine vs strength-targeted SF
`UCI_LimitStrength`/`UCI_Elo` ~2200–2700 — exploits our eval holes differently than our own eval → finds
self-play-invisible gaps; needs a small vs-SF harness extending tournament.py). Immediate (no build) =
diagnose the existing real losses: `selfplay/external/chesscom_2200_white.pgn` (NN-Engine vs 2200 bot,
won-pawn-then-drew-rook conversion failure) + the 3 tal-BOT in `_tal_gap_fens`. Self-play corpus mining =
LOW yield (depth-bound). Loop: eval-or-horizon? (`fenvs`/`ourmove` at rising depth) → diagnose
(`eval_breakdown --fen`) → targeted fix (gated) → no-regression on plugged gaps + TIMED TOURNAMENT → ship
solo. Plan: `~/.claude/plans/handoff-lossless-speed-campaign-tranquil-rose.md`.

### 2026-06-27 — ✅✅ BUNDLE SHIPPED (defaults flipped, UNCOMMITTED for sign-off)
Flipped 5 knobs to default-on in search_engine.h: VERIFY_MARGIN=16000 (done earlier) + ENABLE_ROOK_DBLCOUNT_FIX
+ ENABLE_ROOK_DBLCOUNT_SYM_UP + ENABLE_QPREC_PHASE_GATE + PASSER_ENEMY_CREDIT_PCT=0 + ENABLE_PASSER_BLOCKADE_QUALITY.
Rebuilt; all 6 confirmed active via toggles dump. **NEW SHIPPED BASELINE: WAC 252 / 70,150,573 / STS 1568
(52.3%); wall 162s / 432k nps** (≈ original 167s/406k = speed maintained-to-slightly-faster; qprec's +nodes
offset by higher nps). Note: WAC 252 RECOVERED (Gap-T-alone was 245) and STS 1568 is the BEST yet — the full
bundle is bench-strong AND play-strong (+38.7 Elo), so the STS non-additivity scare was fully a fixed-depth
artifact. (Recover old byte-id 252/67,931,145 with the 6 knobs reset to their old defaults.)
- **PGN collapse positions FIXED (shipped build, d12, SF-free `ourmove` since WSL SF-exec was flaky):**
  benoni-29 → **a4b5** (the winning capture, was Bb3); french-33 ev **+0.93→−0.32**, french-36 **+0.33→−0.34**
  (the eval now SEES Black's a-pawn passer danger where it used to read itself winning and walk into the loss).
  ⇒ both real-loss classes (winning-capture miscalc + passer-danger under-read) addressed. Caveat: the deep
  benoni 44-57 endgame conversion not re-verified (dev doc flagged a possible SF-static realization ceiling
  there); the eval-side danger is fixed, deep-endgame technique is a separate (search) question.
- **Speed / accuracy maintained:** tactical WAC 252 (= original), positional STS 1568 (+65), nps ~maintained.
- **COMMITTED `4fe05fd`** ("Ship collapse-elimination bundle (+38.7 Elo) + campaign tooling"; engine +
  dev docs + tooling; unrelated junk left untracked). Then Phase 2: mine NEW collapse classes (gate on
  TIMED TOURNAMENT, not STS). New dispatcher sub `ourmove` (SF-free move check).
- **NEXT ACTIVE TRACK (separate plan): `is_light` v2** cheap-surrogate light eval to unlock improving/2-ply
  history — plan `~/.claude/plans/handoff-lossless-speed-campaign-tranquil-rose.md`.

### 2026-06-27 — ▶️▶️ RESULT: BUNDLE IS +38.7 ±27 ELO IN PLAY — "non-additivity" was a FIXED-DEPTH ARTIFACT
**875 games (lightning, 4w): base (shipped Gap-T) 44.5% / bundle 55.5% → bundle +38.7 ±27 Elo (SIGNIFICANT,
CI ~+12..+66).** The bundle = Gap-T + rookdblsym + qprec + Gap-P P1+C1. **This OVERTURNS the STS-based
"never bundle" conclusion** — the −77..−139 STS "destructive non-additivity" was a FIXED-DEPTH search-hole
ARTIFACT; in real timed games the same bundle is a LARGE win. by_color symmetric (base loses as both
colors) ⇒ genuine strength, not a color/adjudication artifact.
- **CRITICAL LESSON: STS (fixed depth d10) is a POOR predictor of PLAY strength for these eval/search
  changes — it mispredicted by ~+178 STS-equivalent.** I nearly PARKED Gap-P (the likely top contributor —
  it fixes the real-loss passer-danger class) on its −106 STS. **Do NOT gate collapse/eval fixes on STS;
  gate on the TIMED TOURNAMENT.** ([[fixed-depth-bench-ceiling]] is far more severe than assumed.) The
  user's "just try them all in a tournament / persist, don't pivot on benches" instinct was vindicated.
- **Open: which components drive +38.7?** Likely Gap-P (passer collapse) dominant. Attribution runs
  (Gap-P alone, etc.) would confirm but cost ~6h each. The bundle TOGETHER is the proven +38.7 win.
- **SHIP DECISION (pending user): flip defaults for the bundle** (rookdblsym `ENABLE_ROOK_DBLCOUNT_FIX=1
  ENABLE_ROOK_DBLCOUNT_SYM_UP=1`, qprec `ENABLE_QPREC_PHASE_GATE=1`, Gap-P `PASSER_ENEMY_CREDIT_PCT=0
  ENABLE_PASSER_BLOCKADE_QUALITY=1 PASSER_CONTEST_PCT=30` — Gap-T VM=16000 already default). Large
  multi-knob change resting on one (significant) tournament → recommend a confirm re-run +/or component
  attribution before committing, but per the ship-on-no-regression bar this is a big GAIN, not a regression.

### 2026-06-27 — bundle-in-PLAY test (settling "is non-additivity real in games or a fixed-depth artifact?")
The bench non-additivity (STS −77..−139) could be partly a fixed-depth search-hole artifact; a TIMED
tournament searches differently. Running ONE overnight tournament to definitively settle the recurring
"never bundle" question. **base = shipped Gap-T (VM=16000); cand = Gap-T + rookdblsym + qprec + Gap-P P1+C1**
(`ENABLE_ROOK_DBLCOUNT_FIX=1 ENABLE_ROOK_DBLCOUNT_SYM_UP=1 ENABLE_QPREC_PHASE_GATE=1 PASSER_ENEMY_CREDIT_PCT=0
ENABLE_PASSER_BLOCKADE_QUALITY=1 PASSER_CONTEST_PCT=30`), 4 workers, 360 min, tag `bundle_sprt`. Prior is
AGAINST (benches strongly negative); if flat/positive in play ⇒ non-additivity is a bench artifact and we
CAN bundle (big); if negative ⇒ "ship one at a time" confirmed in games too.

### 2026-06-26 — ✅ Gap-T SHIPPED (default VERIFY_MARGIN 6000→16000, UNCOMMITTED for sign-off)
SPRT base vs VERIFY_MARGIN=16000 (658 games, lightning, 4w): 49.5% base / 50.5% cand = **Elo +3.7 ± 31.2**
(slightly positive, no regression; self-play CAN'T confirm a self-play-invisible collapse fix — the thesis).
Meets the ship bar (collapse fixed + STS +62 + ACPL-neutral + no self-play regression). **Flipped the default
in search_engine.h:719 (6000→16000) + rebuilt + reconfirmed.** **NEW DEFAULT BASELINE: WAC 245 /
64,569,899 / STS 1565 (52.2%)** (old byte-id 252/67,931,145 recovers with `VERIFY_MARGIN=6000`). benoni-29
plays a4b5 by default now. **Not git-committed** (left for user sign-off). First collapse-campaign ship.
**rookdblsym (+34) / qprec (+9) are now SHELVED — they conflict DESTRUCTIVELY with the shipped Gap-T**
(gapt+rookdblsym 1426 = −139 vs the new 1565 baseline), i.e. Gap-T already captured more positional credit
than they offer, and stacking regresses. So they cannot be added on top. **Phase 2 real work = re-tune
Gap-P (the still-open french passer collapse, a DIFFERENT class) + mine NEW collapse classes from self-play
losses (`flip_extract`) — each shipped SOLO, one at a time, re-validated against the NEW baseline.**


### 2026-06-26 — Phase 1 parked-fix re-validation on CURRENT baseline (252/67,931,145, STS 1503)
Key lesson up front: **the parked fixes were validated on the OLD baseline (258/97.5M); combo1 changed the
landscape — re-validation was essential.** Position-fix checks via new `fenvs` dispatcher sub (fen_vs_sf on
explicit FENs, LONG_FORMAT d12). Benches OMP-pinned/book-off.

- **Gap-T `VERIFY_MARGIN=16000` = CLEAN WIN → SHIP candidate (SPRT-gating).**
  - benoni-29 FIXED: our move c4b3 (Bb3, eval +1005) → **a4b5 (axb5) = SF best**, eval +3011. The winning
    capture is found.
  - STS **1503 → 1565 (+62)** (recovers combo1's pruned positional credit, as the old doc predicted SEE+VM
    are complementary); WAC 252 → 245 (−7, fixed-depth artifact); ACPL median 46.5 → 44.5 (neutral-better,
    mean noisy). Net positional GAIN + a real collapse fixed.
  - It's a SEARCH default-change (self-play-VISIBLE) → SPRT before default-flip (old doc's own caveat:
    VM=6000 was self-play-tuned). **SPRT queued** (base vs VERIFY_MARGIN=16000, 4 workers).

- **Gap-P (passer danger) = REGRESSES STS on the combo1 baseline → PARK, needs re-tuning.**
  - P1+C1 (`PASSER_ENEMY_CREDIT_PCT=0 ENABLE_PASSER_BLOCKADE_QUALITY=1 PASSER_CONTEST_PCT=30`): STS **1397
    (−106)** (was +47 on the OLD baseline = the clean-ship-tier BROKE under combo1). C1 adds most of the loss.
  - P1 alone (`ENEMY_CREDIT_PCT=0`): STS 1472 (−31); `=50` worse at 1374 (−129, non-monotonic/erratic — same
    behavior as the conditioning knobs). Best Gap-P variant is still −31 STS.
  - **Bundle (Gap-T + Gap-P P1): STS 1470 (−33)** — Gap-T's +62 did NOT absorb Gap-P; the bundle ≈
    Gap-P-alone, i.e. Gap-P's passer change DOMINATES STS regardless of Gap-T. ⇒ they do NOT co-exist
    cleanly; Gap-P drags the bundle below baseline. Don't bundle Gap-P as-is.
  - TODO Phase 2: re-tune Gap-P on the combo1 baseline (which STS positions does removing enemy-blockade
    credit hurt? is the french-passer fix separable from the STS-costly part?) + check the french-28
    position-fix (slow grind — validate via eval-sign on F33/F36, not one move).

- **Parked `ENABLE_*_FIX` toggle screen (STS, individual, vs 1503):** rookdbl asym **−71** but
  **rookdblsym +34** (the rook double-count fix NEEDS its symmetry correction `ENABLE_ROOK_DBLCOUNT_SYM_UP`);
  **qprec (ENABLE_QPREC_PHASE_GATE) +9**; knightmob +2 / rookrankwin −1 (neutral); knightmobsym **−91**
  (knight-mob is the OPPOSITE of rook — its sym version hurts); rookendcap **−54**. ⇒ STS-positive
  correctness fixes to bundle with Gap-T: **rookdblsym (+34), qprec (+9)** (+ knightmob neutral). Testing
  bundle coexistence next (Gap-T + rookdblsym + qprec).

- **▶️ BUNDLE COEXISTENCE = DESTRUCTIVE (key meta-finding).** Individually-positive fixes combine to STS
  REGRESSIONS: gapt+rookdblsym **1426 (−77)** (vs gapt +62, rookdblsym +34 alone!); +qprec 1452 (−51);
  +knightmob 1486 (−17, non-monotonic); rookdblsym+qprec (no gapt) 1460 (−43). **The eval/search knobs are
  deeply NON-ADDITIVE — you cannot bundle individually-validated changes; interactions dominate and are
  mostly destructive.** This is now the THIRD instance (conditioning LT+realiz, Gap-P, these toggles) ⇒ a
  core property of this engine, and likely WHY every multi-change bundle has washed/regressed historically.
  (Caveat: at fixed depth some of this is search-hole artifact — but the no-regression bar fails either way,
  and the lesson "ship ONE change at a time, never bundle" holds.) ⇒ **Gap-T ALONE is the ideal config**
  (1565/+62, benoni-29 fixed, ACPL neutral). No bundle qualifies.
- **DECISION: SPRT Gap-T alone** (`VERIFY_MARGIN=16000`, search default-change, self-play-visible). Launched
  base vs VERIFY_MARGIN=16000, 4 workers, 300 min, tag `gapt_sprt`. If +Elo or flat-no-regression → flip the
  default 6000→16000 (the dev doc's owed SPRT). Other toggles (rookdblsym, qprec) are individually STS-clean
  but can't bundle → revisit each as a SOLO ship candidate later (own SPRT), not together.

- _Campaign opened; Phase 0 doc + `fenvs` sub + `_tal_gap_fens` corpus in place._
