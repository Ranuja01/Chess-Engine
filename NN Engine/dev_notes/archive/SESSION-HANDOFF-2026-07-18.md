# SESSION HANDOFF 2026-07-18 — KS v1 shipped; DEF-5 categorical win; passer class mapped

**NEW-CHAT ENTRY POINT. Read this first**, then `capgains-ks-collapse-session-2026-07-17.md` (capgains+KS+DEF5
detail), `passed-pawn-subsystem-map-2026-07-18.md` (the next-class map), `collapse-reduction-ledger.md` (running
record). Method memory: [[eval-collapse-diagnosis-method]], [[collapse-categorical-verification]].

## Where we are
Strength work on a non-negamax C++ HCE engine in `NN Engine/` (separate minimizer/maximizer/pre_minimizer;
absolute eval **Black-positive**, single root flip by side_to_move; pawn=1000…queen=10000mp — NOTE actual
`values[]`={0,1000,3250,3450,5000,10000,12000}, memory's N3150/Q9000 was WRONG; mate ±9,999,999). Real strength
~2000 (SF@2400 ≈ the sensitive venue). **The collapse-diagnosis loop WORKS** and we're accumulating class-fixes.

## The 3 collapse-fix items (all KS-family so far + 1 passer, diagnosed)
1. **KS v1 — SHIPPED/COMMITTED** (`73c7cad` KS mechanism + `0140f7b` gated scaffolding). Replaces latent_threat
   with SF-aligned king_safety_danger (weak-square, safe-check, no-queen + KS_FLOOR deadzone). Game-gate
   POSITIVE: KS-caused collapses 15→7 (−53%) at flat total. **Shipped default is now KS-active: byte-id
   248 / 42,272,840** (no longer the old 247/41,479,610).
2. **KS DEF-5 (`KS_SAFE_CHECK_DEF=5`) — SHIPPED/COMMITTED `41c4123`** (was ship-candidate; user approved 2026-07-18). Defensive-asymmetric safe-check
   (boosts the SIDE-TO-MOVE king's safe-check weight only; symmetric global boost hurt WAC). Fixes the
   "castled king under real attack sits 1 unit below the KS_FLOOR deadzone" hole (P2b −0.00→−1.44 ≈ SF).
   **Game verdict (600g, seeds 0/1/2 vs SF@2400): score NEUTRAL (−0.7%), total collapses FLAT, BUT the
   CATEGORICAL KS-attack class 70→54 (−23%)** via our own un-contaminated detector (`ks_class_reattribute.py`;
   SF11-static classifier wrongly read "flat" — it's blind to attacks). Same "flat total ≠ failure" shape as KS
   v1. STS +16, byte-id/symmetry clean, WAC −5 (fixed-node artifact — both offense+defense boosts cost it).
   **RECOMMEND COMMIT** (KS-v1 precedent). So the 2 KS items work SEPARATELY: v1 = the base mechanism, DEF-5 =
   defensive-magnitude calibration on top.
3. **Passer / `pt_pawns` over-read — DIAGNOSED + subsystem MAPPED, NOT built.** The now-dominant "other" class
   (after KS/DEF-5): we over-value ADVANCED pawns (`pt_pawns` +5..+9 where SF's Passed ~0; SF18-search says
   ~0/worse). Root: **unconditional rank-DOMINANT scoring vs SF's conditioning-DOMINANT.** Full verified map in
   `passed-pawn-subsystem-map-2026-07-18.md`: value scattered across **~20 live sites**, **raw mask unconditioned
   drives 13 consumers**, **path-attack conditioning effectively ABSENT** (only `passer_danger`, default-OFF).
   FIX = ONE realizability verdict `R∈[0,256]` (advanceable × safe-path × king-race) computed once, feeding all
   channels via a `passer_weight[sq]` array — upgrade the legacy `getPPIncrement`, NOT a copy. **Guard-rail =
   P3 keep-out** (`1k2r3/p7/8/4PQ2/3PK3/P3P3/2q2P1P/5B2 w`: genuine runners we're CORRECT on — SF18 +4.84;
   must NOT deflate). Tricky FENs: fen3 `rn6/5p2/pBp1pk2/P4p2/1p5b/5B1P/1P2K3/3R4 b`, P1 (perpetual)
   `r4rk1/2R4p/2pN1p2/1p4p1/p2p3P/5bP1/8/4R1K1 b`, P2 `3r4/pp6/3k1p1p/3rp1b1/P1Rp2p1/3B4/2K3PP/4BR2 b`.

## SHELVED / no-go this session (don't retry): capgains **pin** (`ENABLE_CAPG_PIN`) + **tempo**
(`ENABLE_CAPG_TEMPO`) — correct but low-reach (7/64) and drag things in combination (load-bearing capgains
signal; STS −34/−10); single-seed inconclusive in games. `KS_DEF_MAG` (blunt danger multiplier) — STS
1606→1473, over-amplifies moderate danger → paranoia. All gated default-off in the tree.

## THE METHOD (the mission — loop per collapse class; [[eval-collapse-diagnosis-method]])
(1) localize at the REAL venue (collapses vs SF@2400). (2) eval-vs-search triage. (3) **apples-to-apples STATIC
vs SF11-STATIC** — ⚠️ **SF18 is SEARCH (depth), NOT static**; SF11-static is INVALID for attack positions (blind
to perpetuals/promotion/fortress — reads a mate-in-5 as 0.00). So: SF11-STATIC for the static gap, **SF18-SEARCH
as ground truth**. (4) THE MIDDLE LAYER — hand the user ~3 FENs (worst+random), our terms vs SF11 vs SF18, named
in chess terms; user adds chess insight. (5) component-dump the culprit. (6) definition-diff vs SF11 + Ethereal
source (agree=core). (7) deterministic firing/control-set + over-read screen BEFORE games. (8) **CATEGORICAL game
verdict — did the TARGET CLASS shrink** (with an UN-CONTAMINATED classifier — build an SF18-based one; SF11-static
mislabels attacks), AND did prior fixes NOT resurface. NEVER judge by raw total/score.

## KEY DISCIPLINES / lessons (hard-won)
- **Judge by the CATEGORICAL target-class, not total/score.** Score is seed-noisy (13% seed swing ≫ 3.5% SE);
  DEF-5 looked +10.3% on seed 0, −3.4% seed 1, −9% seed 2 = neutral. The CLASS metric is the leading indicator.
- **Deterministic suites ≠ game Elo, and pick the RIGHT suite:** WAC is tactical (fixed-node PENALIZES eval
  accuracy), **STS is positional** (use it for eval/KS changes). DEF-5 = WAC −5 but STS +16.
- **Deterministic suites CAN'T judge capgains-REDUCTION** (capgains is load-bearing for move-selection).
- **Self-play SPRT ≠ real-opponent Elo** — collapse-profile fixes can add real Elo where self-play sees nothing
  (self-play never exposes the shared holes). Judge collapses vs SF, not self-play SPRT.
- **Fix the MECHANISM conditionally, never blanket-reweight** (PST/reweight falsified 3×). SF's conditional logic
  is the model — copy the MECHANISM, adapt, don't copy values.
- **When mapping/auditing, VERIFY the specifics** (esp. bug claims + knob defaults). This session a Fable map
  flagged 2 "bugs" (capgain-sign, rook-dblcount) that were already FIXED by default.

## OPERATIONAL (auto-approve rules)
Build/bench INLINE only via: `wsl.exe -e bash -lc "bash '/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess
Engine/Chess-Engine/NN Engine/selfplay/overnight_runner.sh' <sub> [KEY=VAL...]"` (prefix match; SINGLE-LINE;
trailing knobs OK). Subs: build · wac <tag> [KNOBS] (SOLVED/NODES/EBF) · sts (STS score) · gate/SPRT · pyrun ·
gauntlet. Games A/B: `pyrun selfplay/vs_sf.py --sf-elo 2400 --games 200 --concurrency 3 --our-config '<knobs>'
--tag <t> --adjudicate-draw --seed <s>`. ⚠️ raw git-bash cd/grep/tail on TASK .output files PROMPT — read task
.output with the Read tool; run analysis via pyrun; bank docs via Edit/Write. conc3 MAX (conc4 OOMs keras);
~78-86 min/200g. Engine search is single-threaded (games use ~3-4 cores via conc3; deterministic wac/sts/pyrun
= ~1 core). vs_sf writes per-game `game_NNN.jsonl` incrementally (flushed) → a paused run's games are recoverable;
only the end-of-run collapses.csv/results.csv aggregate is lost on a hard stop. byte-id after every build.
Commit only when asked; no commit footer. Leave stray non-ours files (enum_mapper, old CE/, ChessUI).

## STATE
Shipped default is now DEF-5-active (rebuild + byte-id after next clean build). Committed: `73c7cad` + `0140f7b`
(KS v1) + `41c4123` (DEF-5 defensive safe-check, `KS_SAFE_CHECK_DEF=5` + neutral `KS_DEF_MAG=100` scaffolding).
UNCOMMITTED, all gated default-off: `ENABLE_CAPG_PIN`, `ENABLE_CAPG_TEMPO`, + `CAPG_DEBUG_DUMP` diagnostic
(the shelved capgains levers). Nothing running. New tooling (diagnostics/):
ks_class_reattribute (our-KS-detector collapse classifier — the un-contaminated one), other_class_profile,
ptpawns_dossier/classic, ks_material_3fen(_sf), ks_pin_phantom, ks_capg_dump, ks_shortfall, build_def5_remaining,
_capg_* position probes. Corpora: `ks_sets/other_collapses.txt`(64), `def5_remaining_collapses.txt`(253 = the
passer-class corpus). Game dirs: games/night_base(_s1/s2), night_def5(_s1/s2), night_all3, night_pin, night_tempo.

## FIRST ACTIONS next window
1. ~~Ship decision on DEF-5~~ **DONE — shipped `41c4123`** (user approved 2026-07-18). Narrow surgical commit
   (defensive-KS family only; capgains pin/tempo stayed unstaged/shelved).
2. **Start the passer redesign** from the verified map: design the single realizability verdict R (path-attack
   via `passer_danger`'s D2 machinery + king-race via `passer_realizability_delta` + support/blockade via
   `getPPIncrement`), feed all channels via `passer_weight[sq]`; gate default-off; screen deterministically
   (fen3/P2 should deflate, **P3 must hold**, control set clean, STS) BEFORE any games; then categorical game
   gate with an SF18-based classifier.
3. (Method upgrade) Build the **SF18-based collapse classifier** so the categorical verdict is precise.

## POINTERS
SF11 source `stockfish_11/stockfish-11-win/src/evaluate.cpp` (`passed()` ~648, `king()` 380-465, PassedRank
125, safe-check weights 84-87). Ethereal = 2nd reference (user: `github.com/AndyGrant/Ethereal`; passer eval =
`PassedPawn[canAdvance][safeAdvance][rank]` + king-distance — clean conditioning template). Memory banner
⭐⭐⭐⭐⭐ routes here.
