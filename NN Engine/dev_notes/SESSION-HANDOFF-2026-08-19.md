# SESSION HANDOFF 2026-08-19 — (1) search lane resolved, (2) a COMPLETE structural KS fix (gated, bench-only)

## ⏱️ CURRENT STATE (read first)
Two lanes moved this session. Nothing is shipped; everything is gated + byte-identical at default + UNCOMMITTED.

1. **SEARCH lane — SEE-captures banked as a game-positive-leaning candidate.** C1 (`TT_FLAG_FIX+NULL_MATE_CLAMP`)
   RESOLVED to a confirmed practical NULL (pooled 296-282-212 / 790g = +6 Elo, CI [−15,+27]). Then screened +
   tuned `SEE_PRUNE_CAPTURES`: fixed-time gains on BOTH instruments, chose `SEE_PRUNE_CAPTURE_MARGIN=1000`
   (safest+best). Game gate **BANKED 62-48-44 / 154g** (=+32 Elo but CI spans 0; resumable by pooling). Full:
   [[search-value-bugs-and-the-productive-bug-pattern]], handoff `SESSION-HANDOFF-2026-08-18.md`.
2. **KS lane — the session's deep work — FIRST complete structural KS fix in the engine (bench-validated, ZERO games).**
   `ENABLE_KS_RING_GATE` written/built/byte-id/symmetry-clean; validated on all 4 bench gates. Details below. Full:
   [[ks-collapse-tail-is-endgame-gate-blindness]], roadmap `KING_SAFETY_MODEL.md §5b`.

## ⚖️ HONEST ASSESSMENT (calibrated, NOT hype — the owner asked for this explicitly)
- **What's proven:** a correctly-built, giant-aligned, four-bench-gate-validated structural KS component + a
  game-positive-leaning search candidate. Real code, real bench validation.
- **What's NOT proven:** ANY Elo. Zero games on the KS work; SEE-captures' 154g CI still spans 0. Bench-green ≠ Elo.
- **The prior is against KS converting:** additive/eval KS is **0-for-9 in games**; corpus-fit anti-correlated
  (−85.6); the onset floor-reduction that looked good on bench SPRT'd ~0. A clean KS bench result has repeatedly
  been NECESSARY-not-SUFFICIENT. Even if the KS fix converts, collapses are RARE tail events ⇒ likely a modest
  robustness gain (a few Elo), not a jump. **Treat every bench win as a reason to run games, never as a result.**
- **Stage:** KS ≈ 1/3 through the code (ring-gate done; P2 queen-via-checks + capgains pairing unbuilt), 0% through
  the proof (games). Search: a banked candidate awaiting a decisive SPRT.
- **Strategic call for next session:** the SEE-captures SPRT already leans positive (real game signal); KS carries a
  strong not-converting prior. Highest-EV next ship is arguably RESUMING THE SEE-CAPTURES SPRT, even though KS is
  the more satisfying, better-diagnosed lane. Weigh this; don't default to KS just for momentum.

---

# 🏰 THE KS ARC (the session's centrepiece)

## The diagnosis (verified, phase-mapped)
Owner reframed my "KS is a small-Elo lever" (that was the MEAN, regret ±0.1w%): COLLAPSES are the TAIL the mean is
blind to. On **92 real `ks_attack` collapses** (vs SF@2400, in `ks_sets/collapse_dataset_classified.csv`),
`king_safety` is the **#1 over-read term (+0.26pw, top-34%, sign-flips vs quiet control)**. Per-FEN: our KS reads
**0.00** where SF-classical reads −1..−5.5 (a queen/rook hunting an exposed king). Statically reachable.
- **P1 FEEDERS: FINE** (96% detection floor-free, AUC 0.91; giant-grade) + our DEFAWARE edge. Don't touch.
- **P2 TRANSFORM: mis-shaped** — queen weight INVERTED (`{N2,B2,R3,Q5}` queen-highest vs SF `{N81,B52,R44,Q10}`
  queen-lowest) ⇒ mild over-read (0.05-0.25 band ratio 2.24). Coupled: de-inverting deepens the 1-2 pawn hole
  unless queen danger routes through CHECKS (as the giants do). Our transform is ALREADY quadratic (units²/div).
- **P3 OUTPUT: two un-giant KLUDGES = the big defects.** (a) `isEndGame` cliff @ phase 65 (cpp_bitboard.cpp:7203)
  ⇒ endgame blindness / the collapses. (b) `KS_FLOOR` deadzone ⇒ the 1-2 pawn hole (75% of that band reads 0).

## Giant grounding (primary source, this session) — adopt the base, keep our edge
SF15.1 (`stockfish_15\...\src\evaluate.cpp`) and Ethereal (github, WebFetch'd) INDEPENDENTLY CONVERGE:
- NO phase/endgame gate — KS computed always, fades via attackers.
- Entry gate = attackers+queen: `kingAttackersCount > 1 − popcount(enemyQueens)`.
- Quadratic transform (SF d²/4096, Ethereal −mg²/720).
Two independent giants converging ⇒ this is the superior method; ADOPT the base (not "clone SF"). Our `isEndGame`
cliff + `KS_FLOOR` are the un-giant deviations. Where we stay uniquely ours = DEFAWARE (finer than their binary
attackedBy2). Ethereal's area-norm we tested (`KS_ZONE_NORM`) = marginal. Full: `KING_SAFETY_MODEL.md §5b`.

## The METHOD win (durable, generalizes) — how to break the KS cycle
Every prior KS attempt optimized the target and NEVER measured the SAFE set ⇒ over-fired ⇒ flat/worse in games.
The rule now: **make the collateral (SAFE) set a first-class output of the FIRST probe; gate on "stays SILENT on
safe positions", not "helps the target".** Then crank recall while watching precision; stop where precision breaks.
🧰 `_ks_eg_specificity.py FENS=<collapse+safe combined>` (over-val vs SF per group), `_ks_calibration.py PHASE=`
(ours/SF ratio per SF-|KS| band; skips safe positions), `_ks_specificity.py`, `_ks_hurt_characterize.py`.

## ✅ CODE TARGET #1 — DONE, gated, four-gate-validated: `ENABLE_KS_RING_GATE`
The `KS_MIN_ATTACKERS` entry gate (cpp_bitboard.cpp:5787) exists but NEVER BINDS — our `attackers_sq` counts the
BROAD zone, so even mild positions have ≥2. Fix: count the SF-exact king ring = the forward-inclusive zone
(`white/black_king_ks_zone[k]`) MINUS double-own-pawn-defended squares (`ring &= ~(l&r)`, per-colour pawn-diag wrap
masks). The pawn-fortress exclusion is what lets a forward-inclusive ring BIND. Wiring: search_engine.h (flag,
~1235), search_engine.cpp (env_flag register ~1449 + toggles echo ~2028), cpp_bitboard.cpp:5787 (hot path).
**VALIDATED at 3× magnitude** (`KS_MIN_ATTACKERS=2 KS_FLOOR=0 KS_EXTEND_EG=1 KING_SAFETY_MAG=9000`):
- recall −1.42 (held) · precision safe-eg **+0.057 (vs −0.069 over-fire WITHOUT the gate = DECOUPLES)** ·
  calibration Spearman **0.642 neutral** (tight ring1-only was 0.604 = lost forward-danger; SF-exact fixes it) ·
  **colour-symmetry CLEAN** (0 new violations; the 11/1241mp are PRE-EXISTING PST residual, not ours) · byte-id.
This is the giant DECOUPLING: quiet stays silent so magnitude can be strong on real attacks. FIRST time in-engine.

## WHAT'S LEFT on KS (next session, precisely scoped)
1. **P2 queen-via-checks** — the ring-gate can't touch the mild over-read (2.16, that's the queen weight). Lower the
   raw queen weight (`KS_ATT_QUEEN` 5→2 pulls mild 2.24→1.69 but deepens the hole) AND strengthen the safe-check
   terms (`ENABLE_KS_CHECK_V2`/`KS_CHK_QUEEN`, currently parked/entangled) so queen danger routes through checks —
   then the raw weight can drop without losing the hole. Validate on calibration bands.
2. **Add `CAPG_KS_DAMP` (the pairing)** — it READS `king_safety_danger` (cpp_bitboard.cpp:7628); on the fixed KS
   signal it damps capgains during real king danger (additive −0.082, precision-safe; GROWS on the stronger signal).
   It's the only piece that touches the owner's actual `capture_gains` game-loss. First-class part of the bundle.
3. **Bundle → GAMES.** Assemble `ENABLE_KS_RING_GATE=1 KS_MIN_ATTACKERS=2 KS_FLOOR=0 KS_EXTEND_EG=1 KING_SAFETY_MAG=~6000-9000
   CAPG_KS_DAMP=25` (+ P2), re-verify byte-id/symmetry, then SPRT. Validate on collapse-count + Elo, NOT corpus.

---

# 📜 HISTORY / CONTEXT (so the new chat knows past failures + successes)
- **KS is 0-for-12+** historically: additive KS 0-for-9 in games; KS_ONSET floor-reduction SPRT'd ~0; corpus-fit
  −85.6. CHANNEL LAW: no KS lever has intrinsic value, sign depends on live king-credit channels. See
  [[ks-twelve-attempt-history-and-the-channel-law]], `KING_SAFETY_MODEL.md` (REFUTATION RECORD — don't re-propose
  dbl-pawn prune, over-large zone, KS-local x-ray, suppressor lane).
- **Search is 0-for-13** (micro-opts); C1 the 4 value-bug fixes = null (correct but Elo-neutral = the
  productive-bug pattern). SEE-captures is the one game-positive-leaning search candidate.
- **The two owner GAME LOSSES were `capture_gains` (−6.75) and `passed_pawn_support` (−4.55), NOT KS** —
  [[live-game-losses-are-single-term-blowups]]. The aggregate-collapse attribution of those terms is a NULL
  (search-absorbed); they're rare TAIL events. This is why the capgains-damp pairing matters.
- **SHIPPED wins** (context for what DOES work): the +20.8 OvD+central+defaware bundle; +45 capped threats; +36.7
  material-fix bundle; the 7-fix symmetry bundle. All STRUCTURAL/subtractive, confirmed in ONE tournament.
- **Method laws:** corpus objective anti-correlated with Elo; STS misleads both ways; validate on MOVES not cp;
  rank by win% not cp; every eval-term error is bidirectional (no scale works). GAMES decide.

# 🔧 STATE / DISCIPLINE / TOOLS
- **Branch NN-ENgine = shipped +20.8 bundle.** All this session's work GATED + byte-identical + UNCOMMITTED (commit
  pending owner confirm). New gated flags today: `ENABLE_KS_RING_GATE` (+ this session's KS/search flags from -08-18).
- 🚨 BASELINE (byte-id reverify every build): **250 / 36,651,879 / EBF 3.751 / STS 1771 / WAC 250/300**. Colour-skewed
  suites → use `_mirror`. Symmetry baseline residual = 11/800 (1.4%, worst 1241mp `3q1rk1/...`) = PRE-EXISTING, not new.
- Build/bench LITERAL runner: `wsl.exe -e bash -lc "bash '/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine/selfplay/overnight_runner.sh' <sub> [K=V]"`. Read outputs with the Read tool, never shell-grep. Games ALONE, JOBS≤4, `wsl.exe --shutdown` before big game runs (OOM). Knobs latch at init (one process per setting).
- ⚠️ MEMORY.md at 20.4KB (approaching 24.4KB read cap) — needs a careful compaction (keep all history pointers).
- Instruments added this session: `_ks_specificity.py`, `_ks_eg_specificity.py`, `_ks_hurt_characterize.py` (+ the
  -08-18 search-internals + `_ks_auc.py`/`_ks_calibration.py`). Collapse tail = `ks_sets/collapse_dataset_classified.csv`.
