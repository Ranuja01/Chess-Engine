# POT design brief — what the retuned eval still gets wrong (2026-10-13)

**For the owner's POT design discussion.** Groundwork only: a residual map of ship gen 3 and fit-only probes of candidate POT
inputs (`diagnostics/_pot_probe.py`). Nothing here touches the engine. Context: `research-2026-10-08/POT-DESIGN-PREPLAN.md`
(+ your framing notes at its end), the 10-12 bench (`dev_notes` / memory `eval-v2-rebuild-state`), and the agreed order: POT is
designed on the RESIDUAL of the retuned eval, fitted on top with everything else frozen, then a light joint refit.

## 1. Where gen 3 is still wrong (held-out own-play games, endgame-leaning + endgame)

Win% MSE, and SIDE-AHEAD BIAS in brackets (win% points; negative = the side that is ahead is under-rated = too drawish).

| endgame type | static eval | our d10 search |
|---|---|---|
| pawn ending, pawns equal | 586 (**−23.8**) | 273 (**−14.6**) |
| pawn ending, a pawn up | 321 (**−15.2**) | 146 (**−6.5**) |
| minor only, equal / up | 129 (−12.9) / 163 (−6.4) | 39 (−2.0) / 106 (−2.9) |
| rook only, equal / up | 79 (−9.1) / 134 (−6.3) | 39 (−3.9) / 69 (−2.5) |
| rook + minor, equal / up | 93 (−7.7) / 93 (−4.7) | 38 (−2.5) / 37 (−0.9) |
| mixed / imbalanced, equal / up | 102 (−4.2) / 86 (−3.3) | 60 (+0.5) / 43 (+1.4) |
| queen ending, up | 162 (−4.5) | 121 (+1.3) |

- **Statically, every endgame type is under-confident.** Our search removes most of it — **except pawn endings**, where even the d10
  search stays badly under-confident (−14.6 / −6.5).
- Cross-check with the 10-12 bench (vs SF11 on the bench corpora): the same three types lead the gap — pawn, queen, rook+minor
  endings — and our effective endgame pawn is under-valued against pieces (pawn/knight 0.41 vs SF18 search 0.50).

## 2. Candidate POT inputs, each fitted ALONE on top of the frozen gen-3 eval

Endgame-weighted ((256 − phase)/256). Val change vs the ship, static (quiet) and depth (our d10), all positions and EG stages:

| candidate (form) | static all | static EG | depth all | depth EG |
|---|---|---|---|---|
| DIRECTION · passers (count, most advanced rank, protected, outside) | +0.0% | +0.1% | +0.3% | +0.4% |
| DIRECTION · majority wings | 0.0% | 0.0% | 0.0% | 0.0% |
| DIRECTION · king centralisation · king-to-passer race | 0.0% | 0.0% | 0.0% | 0.0% |
| **MAGNITUDE · complexity (SF12+ `winnable()` inputs) as sign(eval) × input** | **−3.4%** | **−5.9%** | **−1.05%** | **−1.5%** |
| direction + magnitude together | −3.7% | −6.3% | −0.8% | −1.1% |

Complexity weights (mp, endgame leg; positive = MORE confident): **pure pawn ending +580** · pawns on both flanks +132 ·
pawn count +20/pawn · king outflanking +20 · king infiltration −52 · passer count −32. Effect on the depth bias: pawn endings
equal −14.6 → −11.0, a pawn up −6.5 → −4.4; rook+minor −2.5 → −1.7; others ~unchanged.

**Reading:**
1. **POT's MAGNITUDE half (complexity: how decisive the existing edge is) carries the signal; the simple DIRECTION half
   (potential) adds nothing** — today's PX / passer / KFL terms already own passers, majorities and king activity.
   This matches your framing that complexity and potential are separate quantities, and says which one is missing now.
2. The magnitude gain is mostly STATIC (search already resolves much of the under-confidence) — but static discrimination
   matters for pruning / ordering (your 10-07 principle), so a static gain is not worthless.

## 3. Decisions for you

1. **Cause split for pawn endings.** The biggest single weight (+580, pure pawn ending) may be compensating a SEARCH gap: SF11
   wins pawn endings by search (no null move / no shallow pruning with only pawns, passed-pawn push extension), which the search
   transition is scheduled to add. Options: (a) build POT magnitude WITHOUT the pure-pawn-ending input now and let the search
   transition fix pawn endings, then re-measure; (b) include it now and expect to shrink it after the transition; (c) do the
   two SF11 pawn-ending search rules FIRST (small, verified), then design POT on that residual.
   *My recommendation: (c) or (a)* — the record's rule is "split endgame gaps by cause; search-caused → the transition", and (c)
   is small.
2. **Form.** SF12+ applies complexity as `sign(eg) × max(complexity, −|eg|)` on the endgame leg (it can only move toward a
   draw by up to the whole edge). The probe used an unbounded linear form. Proposal: adopt SF's bounded form (it cannot flip a
   sign), with our own inputs and weights fitted nested on top.
3. **Owner inventions to keep distinctive.** Your POT concept also has a middlegame half (potential while a subsystem is still
   unresolved, pushing either way) — parked since 10-01 as null at depth. The probe says nothing new about it; keep parked, or
   re-test once complexity is in?
4. **Winnability vs POT boundary.** Winnability (the multiplicative scale) stays standalone; complexity would be additive and
   live in POT. Round 3 (running) prices the four winnability caps — its result tells us how much of the endgame bias the
   scale side can absorb before POT's additive side starts.

## Appendix A — the endgame-knowledge package: Stockfish's endgame.cpp signatures in OUR games (2026-10-13)

`_eg_signatures.py` (queue #62): ~6,000 held-out own-play positions grouped by the material signatures Stockfish 11's
`endgame.cpp` has recipes for (the catalogue itself is in the 10-08 pre-plan §3a), scored by the static ladder against SF18 d14.
Win% MSE, bias in brackets (negative = side ahead under-rated); "share" = of all our endgame-leaning + endgame positions.

| signature | share | **v2 gen 3** | SF11 | SF15.1 cl. | SF18 NNUE | reading |
|---|---|---|---|---|---|---|
| rook ending (general) | 8.2% | **122** (−6.1) | 136 (−6.5) | 260 | 47 | ahead of SF11 |
| lone king vs mating material (KXK) | 4.6% | **57** (−3.7) | 392 (+17.5) | 392 | 39 | far ahead (SF11 has no conversion scale here) |
| KRPKB rook+pawns vs bishop | 3.9% | **153** (−6.1) | 160 | 377 | 77 | ≈ / ahead |
| KBPKB opposite bishops | 1.9% | 198 (−7.9) | 195 (−16.6) | 243 | 79 | level (we're less drawish than SF11) |
| KPsK pawns vs bare king | 1.8% | **196** (−11.8) | 226 (+2.0) | 388 | 32 | ahead, but under-confident |
| minor ending (general) | 1.7% | 122 (−4.8) | **99** | 158 | 41 | SF11 ahead |
| pawn ending (general) | 1.6% | 511 (**−20.2**) | **438** (−17.5) | 714 | 224 | SF11 ahead; worst bias (search) |
| KQKR(Ps) queen vs rook | 1.4% | **92** (−0.3) | 129 | 288 | 63 | ahead |
| KRPKR | 1.1% | 206 (−8.5) | **176** (−15.2) | 280 | 34 | SF11 ahead (its KRPKR scaling rules) |
| KNPKB / KBPKN | 1.0% each | 210 / 189 | **140 / 152** | 283 / 247 | 89 / 86 | SF11 ahead (KNPKB / KBPKN scaling) |
| KRPPKRP | 0.9% | 186 (−9.1) | **145** | 247 | 69 | SF11 ahead (its KRPPKRP rule) |
| KRKP rook vs pawns | 0.75% | 335 (**−17.5**) | 340 (−10.9) | 466 | **64** | both HCEs weak; NNUE / search |
| KBPKB same-colour bishops | 0.7% | 215 (−11.8) | 222 | 428 | 109 | level, under-confident |
| KBPsK (wrong-bishop fortress) | 0.4% | 208 (+0.4) | **102** (−7.5) | 435 | 36 | SF11's fortress recipe wins |
| KQKP queen vs pawns | 0.3% | **119** | 279 | 163 | 61 | ahead |
| KBNK · KNPK · KRKN · KRKB | ≤ 0.2% each | **ahead** | | | | ahead |
| KPKP | 0.15% | 316 (−17.9) | **253** | 433 | 30 | SF11 ahead (KPKP → bitbase probe) |

**Reading for the design discussion:**
1. **We already beat SF11's recipes where conversion matters** (lone king, queen vs rook/pawns, rook vs minor, general rook
   endings) — POT must not regress these.
2. **SF11's recipes still beat us on FORTRESS / DRAWING knowledge**: wrong-bishop fortress (KBPsK), minor vs minor with pawns
   (KNPKB / KBPKN), KRPKR / KRPPKRP drawing technique. These are SCALE-side (winnability) rules — candidates for porting or
   improving, each small, each measurable here.
3. **The largest misses are pawn endings, KPKP and rook vs pawns**, where both hand-written evals are weak and only search
   (or NNUE) gets it right ⇒ supports the cause split: pawn-ending search rules first (§3 decision 1, option c).
4. Under-confidence is the common thread (bias negative in almost every signature) — consistent with §2's finding that POT's
   magnitude half (complexity) is the missing piece.

## Appendix B — data ready for POT's fits (2026-10-15)
- **Endgame enrichment (EGX):** 60,000 positions with an SF endgame signature (≤ 4,000 each, 55k endgame), SF18 d14 labels +
  58,929 gen-3 d10 depth rows — `E:\chess_data\retune\EGX_fens.csv` (game ids), `EGX_sf18.csv`, `EGX_ours1012_d10_s*of4.csv`.
  Not yet assembled into a corpus (add as a slice like GAUNT, split by game).
- **Trajectory set (owner's idea):** ~39k starts (70% unresolved systems, 30% random control) rolled forward by SF18 @50k
  nodes/move, our static + SF18 d14 at +0/8/16/24 plies + resolution markers — `E:\chess_data\traj\traj_s*of4.csv` (156,478
  rows). Fit targets: transformation `static(t+k) − static(t)` and outcome `sf(t+k)`, then the depth target.
- **Judge:** gen 3 vs SF18 @1000 / 1500 / 2000 = 54.0 / 37.3 / 28.3% ⇒ gate POT at SF18 **@~1100** nodes.
- Winnability scalars: SPSA (2 cold-start runs) found no consistent direction for BASE / SP / OCB ⇒ the scale side is at its
  optimum; POT's additive (complexity) side is where the endgame under-confidence must be addressed.

## 5. DECISIONS (owner, 2026-10-15 discussion) — these supersede §3's options
1. **Pawn endings: search rules FIRST (option c), entering search is fine.** Evidence it is search is an INFERENCE: same 313
   pawn-only rows, our static 0.87× SF11's error but our d10 bias −8.5 vs SF11 −1.0 (d14 −5.8) + a VERIFIED defect: the null-move
   guard counts pawns as pieces (K+≥7P is null-moved; K+Q+N+B is refused). ⇒ guard on NON-PAWN MATERIAL (consistent with v2's
   phase, which already excludes pawns); test SF's `npm(us)==0` and a threshold form. v2-only knobs (v1 frozen). Then SF11's
   shallow-pruning gate and passed-push extension, each its own knob. Measure on the 313 rows (`_pe_search_bias.py`), then games.
2. **Complexity = the reliability of the other systems' verdict** (owner): unresolved ⇒ the regular terms read UNRELIABLE, not 0;
   ~0 in a kinetic position is a real verdict. It never adds score on its own (true in SF11 `evaluate.cpp:718-731` / Ethereal too:
   always × sign of the existing edge). Two legs, both fitted: (a) EDGE — SF-bounded confidence on the existing edge;
   (b) POTENTIAL — complexity SCALES how far POT's directional nudge may move the score. A measured, tuned NUDGE toward what
   search would find; no attempt to compute transformations.
3. **Endgame potential** (the probe's DIRECTION null did NOT test these): opposition (direct / distant, parity), key squares,
   king entry vs defender distance, breakthroughs, spare tempi. **Tempo within reason:** a rule fires only where the structure's
   verdict DIFFERS between White-to-move and Black-to-move (contested opposition, mutual zugzwang, one-move races) and scores that
   difference; where both agree it is exactly 0 ⇒ no side-to-move boost in non-critical positions. Gate: EVAL STABILITY
   (ply-to-ply static swing, depth n vs n+1 oscillation) must shrink where it fires.
4. **Exact theory → the draw section + winnability** (SF's tiers: exact / known-win + drive / technique gradient / graded scale).
   Fortress ports (KBPsK, KNPKB / KBPKN, KRPKR / KRPPKRP) → `DRAW_V2_CLASS` if the condition is STATIC and provable, else the
   scale. ☠️ Every binary draw rule that failed the tablebase on 09-13 was a RACE ⇒ races / entries / opposition belong to POT
   as nudges, never to a draw verdict. "No potential left ⇒ drawish": POT computes it, FEEDS winnability (one owner).
5. **Trajectory labels:** potential FOR and AGAINST the side to move, realised / not (feature-matched counter-examples) /
   partial; balanced on side to move (else the fit learns a disguised tempo bonus). ☠️ The existing set has NO pure endgames
   (starts mg / mgl / egl, middlegame markers) ⇒ build an ENDGAME trajectory slice from EGX with endgame unresolved markers.
   Tablebase (Lichess API, 7-man; no Syzygy on disk) only for a sample, and only with owner OK (external service).
6. **Middlegame potential: parked** — same concept, harder to define.
7. **Cheap in cycles** (owner): pawn-cache for pawn-only features, endgame legs skipped by one phase branch, reuse counts.
   ☠️ Node-limited self-play CANNOT charge eval cost ⇒ every arm also gets a same-session NPS read + a timed check before a ship.
   Search is unoptimised and may be hiding static-discrimination gains ⇒ another reason the search rules go first.
8. Order of WRITING doesn't matter (owner): write everything during queue #68, compile + test when the engine is free.

### 5a. RESULTS 2026-10-15
- **KS_V2_HALF=818** (queue #68): self-play −6.1 ± 17.9 ⇒ not shipped; POT's base stays gen 3.
- **Race-margin prototype** (`_pot_race_proto.py`, 7,097 EGX pawn endings, 5-fold game CV): static + race features −7.2%
  held-out win% MSE vs our static alone, 95% CI [−11.0, −3.3] (1,212 games), shuffled null +0.1…+0.3. Decisive races −9.4%,
  contested −7.1%, no-race rows +1.4% (silent where no race — as designed). Weights: promotion race with side to move +0.39,
  best goal margin +0.17, spare tempi ON CONTESTED RACES +0.15; stm alone, overload flag, contested count ≈ 0.
- **☠️ Pawn-ending SEARCH rules do NOT fix the d10 bias** (queue #69, d10 on 313 own + 1,972 EGX pawn-only rows vs SF18 d14):
  base −7.32 · npm null guard −7.34 (29% of rows changed) · legacy-OR-npm −7.33 · prune gate −7.34 (72% changed, +33% nodes)
  · passed-push ext −7.05 (MSE −2.1%, +17% nodes) · all three −6.97 (+52% nodes). Registered prediction ("all" ≈ −3) FAILED.
  ⇒ The §0a inference (SF11 d10 −1.0 vs ours −8.5 ⇒ our null move / pruning) is REFUTED for these rules; the lever is back on the
  EVAL side (POT race) — consistent with the prototype. Knobs stay at 0 (no ship); the extension may be re-read at equal TIME later.

### 5b. RESULTS 2026-10-16 (race prototype extended: breakthrough goal + minor-piece cover; light endings = pawns + minors)
- Overnight #71: winnability OFF on the pawn rows ⇒ d10 bias −7.3 → −4.8 but MSE +9% (the scale damps decisive AND drawn);
  bias by depth ours −11.7/−9.6/−7.3/−5.8/−4.2 (d6-14) vs SF11 −4.4/−1.7/+0.4/+2.2/+3.7 (same slope, ~7 pts behind).
  NULL_NPM_GUARD=1 self-play −1.4 ± 17.9 (not shipped). Eval symmetry on EGX endgames clean (0/8000, 0/7997).
- Light endings 29,291 / 2,885 games: static + race −5.8% … ☠️ **CONFOUNDED**: material-CLASS indicators (pawn-only, one minor,
  signed by the edge) alone give **−8.7%** (pure pawn endings are under-confident AS A CLASS — a winnability cell). With the
  class control in every arm, the race adds **−3.9%** (helped 41% / hurt 40% — near coin flip with pieces) and **−4.1% gated to
  pawn endings** (helped 54% / hurt 39%, gain 4.5 vs loss 4.2). Pawn-only fit (no class confound): −7.0%, CI [−10.8, −3.0].
- With-edge half carries the race gain (−5.3 of −5.9, pre-control) ⇒ the race FEEDS winnability first; against-edge (POT's own
  term) −1.0% alone. Breakthrough goal as built: −4.2% on light rows where it fires, +6.3% on pawn rows (formula too crude).
- MOVE test (5,974 SF18 multi-PV light endgames, 5 game folds): top-1 30.2 → 29.7%, pairs 74.2 → 74.6%; critical (283)
  58.0 → 60.8% / 82.1 → 83.6% (~1σ). The race term ALONE orders SF's move above a ≥10-pt loser 52.5% (≈ blind): it knows WHO
  wins the race, not WHICH MOVE does (straight-line king distances cannot see the opposition step). Owner game-1 positions not fixed.
- Magnitude: nudge median 1.9 / 99th 10 / max 24 win% pts; gains and losses the SAME size (2.9 vs 2.8) — net gain from frequency.
- **Proposed first C++ build:** (1) winnability confidence cell for pure pawn endings (+ one minor) — the −8.7% class effect;
  (2) POT race margin GATED to pure pawn endings (pawn-cache), feeding winnability + POT's own term; light-piece endings wait
  for a king-path flood fill + defender overload. Test at depth on the pawn rows, then games.

### 5c. BUILT + FITTED 2026-10-16 (commit b2c64bd + knob fit)
- C++: `pot_race()` (pure pawn endings) + `POT_V2_WIN_PAWNEND / _ONEMINOR` class cells (may RAISE f) + `POT_V2_WIN_RACE` feed +
  `POT_V2_RACE_W_*` additive term; all 0 ⇒ v1 / gen 3 / gen 2 fingerprints unchanged; closure C++ == prototype (2,872 rows,
  0 mismatches); symmetry 0/8000 + 0/7997 with every knob on; live 95.8% / 77.6%.
- STATIC fit by exact emulation (`_pot_knob_fit.py`, verify 3,000/3,000 exact), 27,511 EGX rows, 5-fold by game: pawn endings
  391.5 → 187.4 (−52%), one-minor 160.0 → 148.0 (−7.5%). Knobs: PAWNEND 64 · ONEMINOR 16 · WIN_RACE −12 · RACE_W BEST 137 ·
  PROMO 217 · TEMPI 202 · DECISIVE 51 (POT_V2_RACE=1).
- DEPTH (#72, d10, 2,313 pawn rows; OWN 313 = held out): base 169.9 / bias −7.72 → **A full 68.0 (−60%) / +0.09, nodes 0.78** ·
  C cells only 75.6 / −1.52 · B half 90.5 / −3.13 · D race only 115.1 / −5.51 · (our d14 108.1 / −4.80 · SF11 d10 114.2 / −1.80).
  Prediction "the static fit overshoots at depth" FAILED. ☠️ Run 1 lost 113 decisive rows: the amplified evals crossed
  RESIGN_THRESHOLD (−15,000 mp) ⇒ the engine returned no move; re-run with resignation disabled (numbers above). Watch resign in games.
- Next: games — targeted (K+P stress starts) + standard self-play @50k, then SF18 @1100, Mediocre.

### 5d. 2026-10-16 afternoon — pre-game checks, profile, speed plan
- #73: ONEMINOR cell on 2,000 one-minor endings at d10: bias −3.54 → +0.07, MSE 73.8 → 74.2 (flat) ⇒ safe. Dropping WIN_RACE
  (set S): own pawn rows 79.5 vs A 68.0 (+17%, predicted ≤ 3%) ⇒ KEEP WIN_RACE (set A goes to games). SPEED inside pure
  pawn endings: NPS 618k → 480k (−22%), time to d10 +3.6% (nodes −19%) ⇒ a TIMED check before any ship.
- v2 eval PROFILE (`_eval_profile_v2.py`, gen 3 + set A, 2,000-game positions): KS-A ~30% of eval in every phase (24% late
  endgame) · attack build 8-15% · C3 ~10% · pawn detector 5-8% · structure / placement ~7% each · PX 3-5% · winnability+race
  1-2% · ☠️ 26% of LATE-ENDGAME eval is outside every scope (draw class / KPK probe / mobility scoring? — scope it).
  Speed levers for search v2: a result-identical KS early exit when no danger is possible; pawn-hash caching; the unscoped block.
- RACE SPEED PLAN (result-identical ⇒ byte-identity + closure re-check only): (1) distance RINGS `RING[sq][d]` (64×8 masks): the
  nearest square of a set from a king = first d with RING & set ≠ 0, replacing per-square loops; (2) precomputed key-square masks
  `KEYSQ[colour][sq]`; (3) cache the pawn-only part (passer / candidate masks, key-square sets, promotion squares, spare-tempi
  counts) in the pawn hash — in pawn endings most nodes are KING moves, so the cache hits; (4) targets: one pawn-attack fill per side.
- Queued tonight: #74 games (targeted 1,200 from pawn-dominated starts + 2,000 UHO, then SF18 @1100 if self-play passes or
  targeted ≥ +2σ with standard > 0) → #75 TIMED targeted (600 games LIGHTNING, after midnight).
- Prototype: `KPATH=1` = real king routes (no own pawns, no enemy-pawn-attacked squares, not next to the enemy king) for the
  race distances — ready to test (pawn fit + move test) in a gap.

### 5e. FIRST GAMES 2026-10-16 night (set A vs gen 3, node-limited @50k) — judged under the NARROW-ITEM rule (owner, 10-16)
- #74 targeted on random pawn-dominated starts (kp_all): +1.2 ± 23.1 — ☠️ UNREADABLE: 513/600 opening pairs split 1-1 by
  colour (170 resigned on ply 1), only 87 discriminated (POT 51.1%). Prediction +20…+60 not testable there.
- #74 standard UHO 2,000 g: −3.3 ± 17.9 (≈ −0.4σ) ⇒ "not harmful" ✅. SF18 skipped (old trigger) — still OWED under the narrow rule.
- #75 targeted on BALANCED endgame starts (`selfplay/eg_balanced_fens.txt`, |SF18| ≤ 80 cp, 300 pawn + 300 one-minor):
  **+8.3 ± 23.4** (+150 −122 =899), 534/600 pairs discriminated (POT 51.2% on them). Prediction +10…+40: just under. NOT ≥ +2σ.
  ⇒ positive where it fires, NOT PROVEN. Next: a sharper venue (starts a few moves before critical points, owner's refinement),
  or ~2,400 more balanced games (±13), or a better item (KPATH).
- ☠️ ENGINE BUG (pre-existing, BOTH engines, 29 void games: POT 16 / gen 3 13): with an immediate promotion at the root the
  engine returns the move WITHOUT the promotion piece (`h7h8` in `8/7P/6K1/8/8/8/pk6/8 w`, `g7g8`, `f2f1`) ⇒ illegal move.
  0 voids in UHO matches (rarely reached). Can hit owner UI games in pawn endings. Symmetric here (same starts) ⇒ no bias in the
  A/B, but FIX IT before the next pawn-ending venue.
  ⇒ ROOT CAUSE (23:40): NOT the search — the OPENING BOOK path (`ChessAI.opening_book` never set the promotion piece). Fix in
  ChessAI.pyx (source; build after #75). ☠️ BIGGER: games from FEN starts consult the book for 30 plies and M11.2.bin holds
  endgame positions ⇒ in the balanced match 5.3% of moves (23% of games, plies 1-12) were BOOK moves, not either eval's ⇒ the
  +8.3 is DILUTED toward 0. Re-run with USE_OPENING_BOOK=0 (#75 v3, `q75v3_pot_timed.log`); every FEN-start venue: book OFF.
- #75 v3 node-limited, book OFF: **+9.0 ± 23.1** (+156 −125 =919), 0 voids (⇒ the voids were all the book), 552/600 pairs
  discriminate, POT 51.4%. Prediction (slightly above +8.3, 0 voids) held. Still < +2σ ⇒ not proven under the narrow rule.
  Timed run (600 LIGHTNING, book off) started 00:11.

### 5f. 2026-10-17 — C64 (class cells only) + direction (owner)
- #76 depth 12 (own rows): base 141.3 / −6.30 · A 63.9 / +0.67 · **C64 61.1 / −0.71** · C99 75.4 / +0.68 ⇒ the race's edge
  vanishes at d12 (search finds it) while its cost remains; C99 (static refit 99) overshoots at depth.
- Race speed work (result-identical): race NPS cost inside pure pawn endings −22% → −9%.
- C64 games (narrow rule): balanced node-limited **+11.9 ± 23.1**, balanced TIMED **+13.3 ± 32.7** (set A timed was −12.7) ⇒
  ≈ +12 at ~1.3σ combined (same starts). #77 DONE 10-18 02:05: standard UHO 2,000 g **−0.3 ± 17.9** ✅ · SF18 @1100 seeds
  273 / 274: gen 3 49.2 / 49.1% (anchor ≈ 50% ✓) vs C64 48.4 / 49.4% ⇒ pooled −0.25 pts (≈ −2 Elo, ~−0.1σ) ✅ not harmful.
  ⇒ C64: NOT HARMFUL everywhere, positive where it fires, targeted gain NOT yet ≥ +2σ ⇒ not passed yet (resolution, not harm).
  Next: sharper venue (a few moves before critical points) or ~3,600 more balanced games (combined ±11.5); Mediocre ≥ 200 g.
- Owner: POT is his own idea and may be low-yield ⇒ pre-registered keep/park per piece + time box (memory
  pot-may-not-work-use-stopping-points). Aim (owner): close the gap to SF18 SEARCH and change deep-search decisions in mg AND eg
  with potential signals ⇒ static accuracy against DEEPER truth: the depth component at PLAY depth (12+), trajectory outcomes
  (16-24 plies under SF18 play) as labels for potential, and search move agreement on critical positions as the decision check.
- **BENCH CHECKPOINT #2 (after a POT checkpoint with gains; owner 10-17):** static (_reference_ceiling, _gap_strata,
  _endgame_types, _eg_signatures vs SF11 / SF15.1c / SF18) · search (depth ladder d6-14 ours vs SF11 on the bench corpora;
  critical-position move test at play depth vs SF11 / SF18 choices; WAC/STS fingerprint) · strength (v1 vs v2 at equal nodes and
  equal time; SF18 at the current anchor; Mediocre ≥ 200 g for an approximate CCRL).

## 4. Next steps once you've decided
Build the chosen POT inputs as engine detectors (byte-identical at 0, symmetry-gated), fit them nested on the round-3 ship with
everything else frozen, light joint refit, then the usual gate (self-play → SF18 → Mediocre) and the bench checkpoint (#2 of 3).
