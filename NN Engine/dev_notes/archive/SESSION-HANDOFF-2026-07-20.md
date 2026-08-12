# SESSION HANDOFF 2026-07-20 — Kaufman ship-candidate + KS aim detector; SF18-truth methodology

**NEW-CHAT ENTRY POINT. Read this first**, then `kaufman-imbalance-2026-07-20.md` (the shippable win),
`king-safety-redesign-investigation-2026-07-20.md` (KS aim detector + coffin autopsy),
`collapse-reduction-ledger.md` (running record). Method memory: [[eval-collapse-diagnosis-method]],
[[collapse-categorical-verification]].

## Where we are
Non-negamax C++ HCE engine in `NN Engine/` (separate minimizer/maximizer/pre_minimizer; absolute eval
**Black-positive**, single root flip by side_to_move; `values[]`={0,1000,3250N,3450B,5000R,10000Q,12000K}; mate
±9,999,999). Real strength ~2000; **SF@2400 = the sensitive game venue**. Shipped default is DEF-5-active
(**byte-id ref = WAC node total 39,914,378**). Committed: `73c7cad`+`0140f7b` (KS v1) + `41c4123` (DEF-5).

## THE THROUGHLINE THIS SESSION: our EVAL is the gap, and SF18-SEARCH is truth
- **Proven eval-not-depth** (`diagnostics/sf11_depth_test.py`): SF11 (classical HCE) at OUR leaf depth 12
  evaluates fen3/P2/P3 correctly while we read them badly wrong. So eval fixes DO pay off at our depth.
- **SF18-SEARCH is the only ground truth. SF11-STATIC is a diagnostic aid, contaminated on sharp positions**
  (it dropped 64% of a KS dossier as artifacts; 105/200 "control" positions were actually decisive). Rule:
  validate every eval claim with SF18-search, never SF11-static. Where SF18 agrees with us over SF11, we're
  BEATING SF11 — keep it.

## Two live workstreams
### 1. KAUFMAN quadratic material-imbalance — SHIP CANDIDATE (user's call), not committed
Public Kaufman-1999 model; **coefficients FIT FROM OUR DATA** (`diagnostics/kaufman_fit.py`, ridge on the
SF11-residual — NOT copied; rediscovered correct chess: knight×pawn +178, rook×rook −90). Gated
`ENABLE_KAUFMAN_IMBALANCE` (default off = byte-id; skips flat pair bonuses when on; β table in cpp_bitboard.cpp).
Deterministic all-pass (eval-gap 1.285→1.223, STS identical, WAC−3, P2 +4.91→+1.74, P3 holds). **6-seed games:
+1.15% score / −6.8% collapses, wins 4/6 seeds** — score ~0.85 SE (not conclusive alone) but signals CONVERGE
positive and nothing negative ⇒ **weakly positive, lean-ship on convergence. SHIP = flip default true + narrow
commit, on user OK.**

### 2. KING SAFETY redesign — aim detector BUILT (bench-clean), game-testing; coffin dropped
The KS eval gap is real but two-part: **detection** (7/13 SF18-genuine attacks read 0-3 units — our detector is
a zone-membership + line-of-sight test, blind to a slider whose blocker sits outside the ~12-sq zone) and
**signal-to-noise**. Built:
- **Front A `ENABLE_KS_AIM` (gated, byte-id, BENCH-CLEAN):** latent king-aim — catches enemy sliders aligned
  with the king through EXACTLY ONE blocker (the discovered line LOS misses). Reuses empty-board king rays +
  `betweenPieces`, iterates only 0-3 aligned sliders (no new per-piece loop). Weights `KS_AIM_BISHOP/ROOK/QUEEN`.
  aim 6/8/11: FIRENEW 0.24→**0.71** (3× detection), **STS 1606 + WAC 243 held**. **Game-testing now**
  (`games/ab_ksaim_s0`, seed-0 vs existing `ab_base_s0`).
- **Coffin (`KS_INTERACT`) DROPPED — structurally wrong** (not mis-tuned): our base is already SF-shaped
  (additive units → ONE quadratic); the coffin adds a SECOND multiplicative non-linearity that gets SQUARED by
  the transform = volatile over-read → STS 1606→1405. Redundant + unstable. Autopsy in the KS note.
- **Next (after the game):** SF-aligned ADDITIVE alternative — raise `KS_WEAK` (strict `KS_SF_WEAK`, currently 2
  vs SF's heavy weight) for the undefended-hole signal, amplified by the existing quadratic (stable).

## KEY LESSONS this session (hard-won)
1. **SF18-search is truth; SF11-static + old curated sets (control/dossier) are contaminated.** Validate with
   SF18. Where we beat SF11 (SF18-confirmed), keep it.
2. **Fit our own coefficients from data** (Kaufman ridge) — principled + our-values + auto-handles our scale.
3. **Deterministic wins don't always transfer to games** (pawn-placement lever NO-GO despite clean screens;
   the aim detector must still pass games). Judge by collapse-profile + score, categorically.
4. **Architecture: additive signal → ONE quadratic (SF-shared).** Don't bolt on a second multiplicative
   non-linearity. "Our own, not a copy" = our detectors/weights on a SOUND architecture (the aim term), NOT a
   novel-worse structure (the coffin). Flair is welcome; **accuracy is the judge** — drop what fails it.
5. **Compare LIKE terms** (pt_pawns = whole pawn-evaluator output ≠ SF's `Passed`; mis-scoped the "passer class").
6. **Bench venue:** WAC is fixed-node (over-penalizes eval accuracy); **STS positional is the truer gauge**;
   games decide.

## SHELVED / NO-GO this session (don't retry as-is)
- Pawn-placement blanket levers (`ENABLE_PASSER_V2`, `ENABLE_NPEDGE_DAMP_EG`): 2-seed NO-GO; the over-read is
  general endgame pawn PLACEMENT, not passer realizability; deflating it doesn't help at our depth.
- KS re-weight-only (floor down / safe-check up on the CURRENT detector): partial + wakes noise + WAC−13.
- The coffin (`KS_INTERACT`): structural, above.

## OPERATIONAL (unchanged)
Build/bench INLINE, LITERAL runner path only (a `$VAR` breaks the auto-approve prefix → prompts): `wsl.exe -e
bash -lc "bash '/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine/selfplay/
overnight_runner.sh' <sub> [KEY=VAL...]"`. Subs: build · wac <tag> [KNOBS] · sts [KNOBS] · gate · pyrun ·
gauntlet. Games: `pyrun selfplay/vs_sf.py --sf-elo 2400 --games 200 --concurrency 3 --our-config '<knobs>' --tag
<t> --adjudicate-draw --seed <s>`. conc3 = ~4 cores/~80min per 200g; deterministic screens ~1 core. Read task
`.output` + games CSVs with the Read/Grep TOOLS (not shell tail/grep). Commit only when asked; no footer; leave
stray non-ours files. USER GAMES on this box occasionally — HOLD 4-core game runs when the user is gaming.

## STATE / tools + corpora built
Uncommitted gated (all default-off/byte-id): `ENABLE_KAUFMAN_IMBALANCE`+`KAUFMAN_SCALE`, `ENABLE_KS_AIM`+
`KS_AIM_*`; plus older shelved (`ENABLE_PASSER_V2`, `ENABLE_NPEDGE_DAMP_EG`, capgains pin/tempo, `KS_DEF_MAG`).
Diagnostics built: `sf11_depth_test`, `kaufman_fit`, `eval_vs_sf11`(argv-env added), `ks_gap_dossier`,
`ks_sf18_revalidate`, `ks_genuine_units`, `ks_separation`(4-tier), `ks_tune`, `ks_sf18_control`, `v2_firing`,
`pool_def5_collapses`, `_passer_match`(argv-env). Corpora `ks_sets/`: `ks_underread_sf18.txt`(13 genuine),
`control_sf18safe.txt`(66 SF18-safe guard), `ks_underread_vs_sf11.txt`(36 raw), `def5_baseline_collapses.txt`
(165), plus `danger.txt`/`control_*` (KS v1). Game dirs: `ab_base_s0..s5` (DEF-5 baseline), `ab_kauf_s0..s5`
(Kaufman), `ab_ksaim_s0` (aim, running).

## FIRST ACTIONS next window
1. Read the aim-only game result (`games/ab_ksaim_s0/`) vs `ab_base_s0`: collapses + score. If promising →
   more seeds + gentler aim variant.
2. Kaufman SHIP decision (user) — recommend commit `ENABLE_KAUFMAN_IMBALANCE=1` default (weakly-positive,
   converging, gated/byte-id).
3. KS: try the SF-aligned ADDITIVE `KS_WEAK` boost (coffin alternative); joint with the aim term; bench + games.
4. Method: keep validating with SF18-search; judge levers by collapse-profile + STS + games, not SF11-static.
