# SESSION HANDOFF 2026-08-17 — KS is a FEEDER (phase-1 detection) problem; the "when" is downstream

## TOP BLOCK — read this first
**State → reframe → the diagnosis → the ordered fix → what's parked.**

- **The reframe (the whole session's payoff):** the KS "when-to-fire" problem is **downstream of phase-1 DETECTION accuracy.** You cannot threshold a signal that is noisy at its own baseline. Our feeders **squeeze the danger separation from BOTH ends**, so quiet and real-attack distributions overlap and NO threshold separates them — which is exactly why 12 threshold-tuning attempts failed.
  - **Quiet OVER-detected** by: (ii) weak counted over the whole ~15-sq zone (SF: 9-sq ring only) + (iv) we never remove **double-pawn-defended squares** from the ring (SF11 `kingRing &= ~dblAttackByPawn` :247). Measured: our quiet baseline ≈ **5 units where SF nets ~0**.
  - **Real attacks UNDER-detected** by: (i) no x-ray (Q-behind-R battery reads as 1 attacker) + (iii) pinned defenders counted as full defenders (SF clips to the pin line + charges +98).
- **How SF actually separates them (BOTH halves):** (a) precise **pre-pruned feeders** keep the quiet positive-sum small, AND (b) **big signed suppressors** net quiet far below the fire threshold — `−873 no-queen ≈ 24 Elo (SF's single largest KS term)`, −100 knight-defender, −6·shelter, −4·flank-defense. Ours are single-digit (`KS_NO_QUEEN=6`, `KS_SHIELD=2`, `KS_DEFENDER=0` OFF). SF fires KS in quiet positions too — it just *nets it away*. Detection precision was only HALF the answer; the big suppressors are the other half.
- **SF architecture didn't change in 5 years** (SF11→SF15.1): same >100 threshold, same `d²/4096` transform, same −873 gate, same 11-term signed sum. They only sharpened DETECTOR PRECISION (SafeCheck saturation table, dbl-pawn ring filter, RookOnKingRing latent-aim as a small Score OUTSIDE kingDanger). Ethereal's flavour: a hard **entry gate** (refuse to compute below 2 attackers / 1 with queen) + `MAX(0,·)` penalty-only output = per-position zero, no hand-set threshold.

### THE ORDERED FIX (all SUBTRACTIVE = the only direction that has ever won here; SF-grounded; phase-1 FIRST)
1. **Prune double-pawn-defended squares from the scanned zone** (~2-line port of SF11 :247). Biggest single cut — the per-zone-square count is independently tagged the "~85% over-read source". SF *proves* those squares carry zero danger.
2. **Restrict weak / attack-count to the ring-1 core** (or enable `KS_ZONE_NORM`) — stop the forward-rank inflation.
3. **Pin-clip defenders (`KS_PIN_MODE` exists, off) + wire x-ray/battery** — fix the UNDER-read tail so real attacks stop reading quiet.
4. **THEN** scale suppressors to reference proportions (`KS_NQ_SUP` etc.) + apply the accum threshold (`KS_ACCUM_MODE`, built, off) — now over a CLEAN signal.
- **Success metric (cheap, deterministic):** on the archetype bench, the QUIET baseline should collapse from ~5 units toward 0 as each feeder fix lands, WITHOUT dropping DANGER detection. Then a threshold separates cleanly. Validate each: bench → STS(D10) → D7 criticality (full sample) → SPRT.

## What this is
Non-negamax C++ HCE engine in NN Engine/ (minimizer/maximizer/pre_minimizer; absolute Black-positive eval; millipawns; mate ±9,999,999; ~1900-2000). KS lives in `cpp_bitboard.cpp king_safety_danger` (~5323-5761), `king_safety_score` (~5769), enters eval at 7617 (midgame-only, `!isEndGame`).

## BASELINE (shipped default = the +20.8 bundle). Clean per-position-clear harness: **250 / 36,651,879 / EBF 3.751 / STS 1771**. WAC 250/300. All KS detector/accum knobs default byte-identical.

## Build/bench — LITERAL runner, single line: `wsl.exe -e bash -lc "bash '/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine/selfplay/overnight_runner.sh' <sub> [KEY=VAL]"`. Only the runner form auto-approves. Runner subs used this session: `pyrun <script.py> [K=V]` (analysis, sets STOCKFISH_PATH), `sts <tag> [K=V]` / `wac <tag> [K=V]` (D10 move-solving, write results/sts_results_<tag>.csv), `gate '<cfg>' <lbl> [tag] [maxg] [elo1] [conc]` (SPRT), `fast_ab <min> <depth> '<p1>' '<p2>' [conc]`, `vs_sf <elo> <games> ...` (collapse mining). JOBS≤4 (memory). Knobs latch at init ⇒ one process per setting. Clean harness on by default (`run_one` clears per position unless `DIAG_NO_CLEAR=1`); STS 1771 confirms it.

## NEW TOOLING this session (the KS evaluation machine — CHECK BEFORE WRITING ONE)
- **`_ks_archetype_mine.py` + `_ks_build_bench.py` → `ks_sets/ks_archetypes.csv`** — the ARCHETYPE BENCH: 82 positions, DANGER(A1-A5, KS should be HIGH) / COUNTER(B1-B5, KS should be LOW, incl. the empirical AWAY over-reads) / (boundary C1/C2 not yet built). Counters middlegame-filtered (endgames leaked in and faked over-production).
- **`_ks_bench_score.py [K=V]`** — the SCORER: per-archetype our-KS vs SF danger-to-subject-king; DANGER should detect HIGH, COUNTER should stay LOW. Baseline = 19% danger detection.
- **`_ks_detect_dist.py`** — phase-1 detection distribution (KSD dump: attsq/weak/safe/attpc/units) DANGER vs QUIET vs STS_REGRESS. THE tool that measured the 5-unit quiet baseline.
- **`_sts_regress.py <base_tag> <cand_tag>`** — diff two STS result CSVs → the D10 positional regressions (the collateral set) + `_sts_regress.fens`.
- **`_ks_regret_score.py [MAXN= JOBS= DEPTH=]`** — D7 move-regret with seeded 0.7 train/held split, NOW criticality-bucketed (HELDcrit + n_crit). ⚠️ NOISY below full sample — sign-flipped between MAXN 2500/6000; full (15k) is stable. Edit its CONFIGS list to add candidates.
- **`_sprt_collapse_mine.py` / `_sprt_counter.py` / `_sprt_term_profile.py`** — mine SPRT game.jsonl into helped/hurt decision sets + per-term SF triangulation.

## THE METHOD (reusable, this session's process — it worked, use it)
Diagnose a KS candidate cheaply BEFORE games: **archetype bench** (danger up / counters flat) → **STS(D10) + diff the regressions** (`_sts_regress`, then triangulate WHY via `_ks_detect_dist`) → **D7 criticality-bucketed regret at FULL sample** (the kill criterion) → SPRT. This caught a fluke (V2's n=30 dHcrit −0.97 reversed to +0.64 at n=60, then settled to a real −0.73 at n=146) BEFORE spending games. The bench's job is OVER-PRODUCTION control (counters); D7-critical is the Elo-bearing gain; both are needed (a static win ≠ a move win, proven repeatedly).

## CLOSED / PARKED LANES (do not re-run standalone)
- ☠️ **Safe-check MAGNITUDE (V2 = `ENABLE_KS_CHECK_V2` + big `KS_CHK_*` weights):** ENTANGLED. Full-sample D7 shows a REAL critical gain (−0.73, n_crit=146) but STS −181 (−77 best-gated). The critical gain and positional cost are the SAME signal — every gate (attacker-count +87, DEF_MAG +104, accum-threshold +83) recovers STS only by ALSO suppressing real detection (MIN_ATTACKERS=3 dropped A2 open/storm 60→31%, A5 weak 24→−11%). Gates don't stack (combining FAILED). PARKED — re-test on CLEAN feeders, don't tune it further now.
- ☠️ **KS_BATTERY** — wired this session (fable-reviewed footprint overlay, byte-id, symmetric) but **0/46 collapse coverage**. Correct, kept off. (It IS part of fix #3's x-ray/under-read side.)
- ☠️ Coordination product (`KS_COORD_GATE_MODE`) at default div=4 LOWERS detection (product < flat sum for 2-3 attackers); div=1-2 escalates but over-produces on counters (bench-confirmed). Needs the queen-weight de-inversion (queen highest→lowest) to discriminate — untested.
- ☠️ Material lane: SF11's "Material" term is NOT comparable to raw piece values (fit gives Q≈6.5 nonsense). Our own `material` term is impure (~0.8 pawns of PST/AST bundled). Material un-diagnosable via SF11; needs SF18-search or a known standard.
- ☠️ (from prior) additive KS 0-for-9; corpus-fit −85.6 Elo; endgame-KS-hurt was a contamination ghost.

## STATE. HEAD on NN-ENgine = shipped +20.8 bundle. Nothing running. This session's work is DOCS + tooling + gated knobs, UNCOMMITTED (no eval code written except the KS_BATTERY footprint overlay in cpp_bitboard.cpp ~5349, gated off/byte-id; and `KS_BATTERY` default flipped 3→0 in search_engine.h). The regret tool's CONFIGS list was edited. Memory files live outside the repo. No commits (owner away-rule + not asked).

## DISCIPLINES. Byte-identity every build + wac_speed peak. Clean harness (per-position clear) — STS 1771 not 1670. |balanced STS|<~150 unresolvable. D7 regret NOISY below full sample. Validate MOVE-quality (STS/D7) not just static (bench detection ≠ moves). Games decide, run ALONE, JOBS≤4. Read tool for outputs, never shell-grep. Commit only when asked, no footer. No unsupervised eval code / no load-bearing value changes.

## WEIGHT MY EXPLANATIONS. The measurements held; my stories needed the owner's pushback throughout, and the pushbacks were load-bearing:
- I spent the whole experiment budget tuning phase-2/3 MAGNITUDE (V2 safe-check weights, gates) on a dirty phase-1 — exactly the "how loud" trap — while the owner **repeatedly** pointed at the FEEDERS (weak-square / attacker-defender definitions) as the issue. I mis-applied the empirical diagnosis (which measured the UNDER-read) to dismiss the feeder route, and I took the easy knob path over the hard code path. The owner's "how does SF balance ALL of these" forced the fable phase-interaction analysis that produced the actual diagnosis.
- I over-called "capped" from noisy sub-sample D7 (the owner's "invest the hour" for full-sample resolved it — the critical gain was real).
- I asserted the attacker-gate "keeps the critical gain" without testing; the owner's "is that stable elsewhere?" was right — it gutted single-attacker real dangers.
- ★ The core reframe (feeders are upstream of "when"; the coupled unit must be built feeders-first) is the owner's, sharpened over many turns. It is the entire diagnosis.
