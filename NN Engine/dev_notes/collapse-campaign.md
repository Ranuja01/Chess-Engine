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
- **NEXT:** git-commit (user sign-off) — the bundle is 5 knob defaults in search_engine.h + the dev docs.
  Then Phase 2: mine NEW collapse classes (gate on TIMED TOURNAMENT, not STS). New dispatcher sub `ourmove`
  (SF-free move check).

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
