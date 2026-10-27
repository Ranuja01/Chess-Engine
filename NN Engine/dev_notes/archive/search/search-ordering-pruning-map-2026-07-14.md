# Search ordering + history + pruning MAP (2026-07-14)

Triggered by the "how do we lower EBF (2.6-3.7 vs SF 1.5-2) WITHOUT removing important nodes" question,
after the eval/conversion lane closed (imbalance NO-GO, simplification move-bias NO-GO). Fable's SF11
search audit + three code-audit passes. **Headline: we are NOT missing SF's machinery — it's mostly built;
the gaps are KEYING, GATING (default-off), and TUNING (handpicked constants).**

## The map

### Ordering (`move_gen.h` `generateLegalMovesReordered`, score at :680-748, stable_sort :764)
- Live default: MVV-LVA captures (SEE-demoted losers, tier `capture_base_value=200000`) + killers
  (10000/9000) + counter-move flag (8000) + **main history `[side][from][to]`** + **1-ply counter-move
  history `[side][prevF*64+prevT][from*64+to]`** + move-frequency. One combined int key.
- **Default-OFF (built but parked):** TT-move-first (`ENABLE_TT_MOVE=false` — partially substituted by the
  movegen-cache promoting the in-search cutoff move), 2-ply cont-hist in ordering (`ENABLE_CONT_HIST_2PLY`),
  capture history (`ENABLE_CAPTURE_HIST`), check-ordering (`ENABLE_CHECK_ORDER`, "too hot, recalibrate").
- Reuse-time hoisting: `buildMoveListFromReordered` (search_engine.cpp:5697) swaps killer/counter (and gated
  TT move) to front of the cached list; lazy quiet-tail re-sort (`ENABLE_LAZY_RESORT=true`).

### History stack (`cache_management.h` tables; updates in search_engine.cpp)
- Tables: killers[ply][2], counterMoves[64][64], **historyHeuristics[2][64][64]** (main), **counterMoveHeuristics
  [2][4096][4096]** (1-ply cont), **contHist2[2][4096][4096]** (2-ply cont), moveFrequency[2][64][64],
  captureHistory[2][64][64]. Continuation history EXISTS (2 plies).
- **KEY GAP: cont-hist is keyed by `from-square × to-square` of the prior move, NOT `piece-type × to-square`
  like SF.** Piece-based keying generalizes far better + fills faster. Prime structural fix.
- Bonus `b=((depth_limit-cur_depth)^2 * HISTORY_BONUS_SCALE)/100`. Malus EXISTS (`ENABLE_HISTORY_MALUS`,
  penalize quiets tried before cutoff). Gravity EXISTS (`ENABLE_HISTORY_SATURATION`, hist_update self-decay).
  Update sites: min :3198-3260, max :3669-3721/:4284-4336, qSearch :4756/:4922.
- Decay (handpicked/untuned, the user's point): `DECAY_FACTOR=1` (>>=1 halving), `DECAY_INTERVAL=35000`,
  a magic `*16` for cont tables, `moveFrequency >>=2`. All env-tunable, never tuned.

### LMR + pruning + re-search (`search_engine.cpp` get_score_for_min/maximizer 2235-2891)
- LMR reduction: `DEPTH_REDUCTION[depth_limit]` base − log2(moveCount)/phaseScale − LMR_EXTRA, then
  **statScore adjust = main + 1-ply + 2-ply cont-hist** (`ENABLE_STATSCORE_LMR`, shipped) → history-informed.
- **RE-SEARCH IS SAFE:** reduced scout that beats alpha re-searches at FULL `depth_limit`, full window
  (2525-2556 / 2860-2889). ⇒ **LMR is NOT where we lose good moves; we can push reductions harder here.**
- **LOSSY prunes (skip, NO re-search) = LMP / futility / RFP / SEE-prune:** all **flat-gated** (quadratic
  move-count / flat margin tables), NOT history- or improving-conditional. **This is where "pruning removes
  important nodes" happens.** Active defaults: `ENABLE_LMP/FUTILITY/RFP/RAZORING/NULLMOVE=true`,
  `ENABLE_SEE_PRUNE=false`.
- `improving` flag exists (eval vs 2-ply-ago) but **wired into LMR ONLY**, not LMP/futility (SF uses it there).

## The decomposition (answers "how does SF prune safely")
1. **LMR re-searched → push freely** (we have this; it's just untuned/timid).
2. **LMP/futility conditioned on history+improving → skip only genuinely-bad moves** (we have flat guards = the gap).
Our **first-move-cutoff ≈ 87%** (existing cutoff_histogram m0=1.92M/2.21M) = SF-class → the EBF gap is NOT
first-move ordering; it's timid LMR + flat lossy guards.

## Ranked program (search-structure lane; can't hurt the MEAN like eval features)
1. Diagnostic: which lossy prune removes important nodes (A/B toggles, below).
2. Make LMP/futility history + improving-conditional (improving already exists; wire it in).
3. Push LMR harder via SPSA (DEPTH_REDUCTION/LMR_EXTRA) — safe via full re-search.
4. Re-key cont-hist from×to → piece×to (structural quality).
5. Turn on + jointly tune parked signals (2-ply cont-hist, capture-hist, check-order) + SPSA decay/margins.
Venue: search levers need EQUAL-TIME gating (tournament/gauntlet), not fixed-depth.

## Diagnostic results (pruning A/B, WAC depth-10, baseline 247 / 41,479,610)
| config | SOLVED | NODES | EBF | read |
|--------|--------|-------|-----|------|
| baseline | 247 | 41,479,610 | 3.680 | — |
| LMP off | 241 (−6) | 78,946,318 (+90%) | 4.053 | good prune (saves ~half the tree) |
| futility off | 242 (−5) | 50,333,349 (+21%) | 3.755 | good prune |
| **RFP off** | **254 (+7)** | 62,413,351 (+50%) | 3.837 | **CULPRIT — recovers important nodes** |
| razoring off | 249 (+2) | 52,772,115 (+27%) | 4.166 | minor secondary culprit |
| null-move off | 241 (−6) | 67,467,675 (+63%) | 3.803 | good prune |

**VERDICT: RFP (reverse futility / static-null, `ENABLE_RFP`, shipped +73 Elo) is THE over-pruning culprit** —
the only prune whose removal recovers tactical solves (+7). It prunes a whole node when the STATIC EVAL says
"above beta, stop" — and our eval OVER-READS messy positions, so RFP fires on over-optimistic evals and prunes
the search that would find the refutation. **This is the mechanistic link between the eval over-read and search
over-pruning.** Razoring is a minor secondary (+2, also static-eval-gated). LMP/futility/null-move are all good
(lose solves when off, save huge nodes).

## NEXT: condition RFP (do NOT disable — it's +73 Elo; keep its savings, stop it trusting unreliable evals)
Candidates (offline-screen on WAC/STS first — does it recover the +7 without losing the node savings? — then
equal-TIME gauntlet since RFP is Elo-shipped):
1. Depth-scaled / larger `RFP_MARGIN` (blunt).
2. Gate RFP on `improving` (SF-standard; the flag exists, just not wired into RFP at :3306-3310).
3. Gate on an eval-reliability / tactical-tension detector (reuse the capg tension detector — don't
   static-prune in messy/high-tension nodes where the eval is untrustworthy).
Also separately: push LMR harder (safe full-depth re-search) via SPSA to lower EBF (3.68 → toward SF 1.5-2).

## PRUNE-VERIFICATION HARNESS RESULT — RFP conditioning = NO-GO (2026-07-14, definitive)
Built the mechanism harness (logger `ENABLE_PRUNE_LOG`/`PRUNE_LOG_STRIDE` byte-id-safe at both RFP sites;
`diagnostics/prune_verify.py` = per-fire verification search RFP-off with automatic non-negamax POV
calibration; `diagnostics/prune_discriminate.py` = AUC; `diagnostics/prune_collect.py` = arbitrary-corpus
collector). Labeled ~18k WAC fires + ~12.5k messy `overread_bench` fires (RFP fires 8.2M total on the messy
corpus → subsampled 2500/rd).
- **RFP prunes ~99.7% CORRECTLY on BOTH clean (WAC) and messy (over-read) corpora** — wrong-rate 0.3-0.4%,
  corpus-INDEPENDENT (rd1 ~0.8% → rd4-6 ~0%, i.e. only the small-margin near-leaf fires ever err).
- **The eval-reliability discriminator is INVERTED** (`eval_instab` AUC 0.18): wrong prunes are in STABLE
  near-window positions, NOT messy ones → the RFP→eval-over-read thesis is REFUTED at the mechanism level.
  Only predictor of wrongness = margin-proximity (= the `RFP_MARGIN` knob, a known tradeoff). At 0.3% base
  rate with no clean separator, no conditional gate has viable precision/recall. **NO-GO.**
- **The harness CORRECTED the black-box diagnostic**: RFP-off's "+7 WAC solves" was search-path CHURN
  (19 recovered / 12 lost), NOT recovered blunders. "RFP is the over-pruning culprit" was an artifact of
  outcome-only A/B testing. Verified offline (~2h) vs a multi-seed gauntlet that would've been ambiguous.
- **Implications:** (1) if RFP (the supposed worst offender) is well-behaved, the eval-trusting-prune family
  (razoring/futility/null-gate) is likely well-behaved too — conditioning them is probably also a non-lever
  (optionally confirm on ONE move-level prune, e.g. futility). (2) The EBF gap vs SF is therefore NOT
  over-pruning → it's UNDER-REDUCTION (LMR timidity). **Pivot to Phase 4: push LMR reductions** — safe
  because LMR re-search is full-depth (audit-confirmed), so pushing can only cost re-searches, never remove
  important nodes. SPSA `DEPTH_REDUCTION`/`LMR_EXTRA` at equal time. The harness + logger stay as reusable
  infra (default-off, uncommitted). Nothing committed.

## LMR-PUSH DATA → we are ORDERING-LIMITED; the lever is the ordering FOUNDATION (2026-07-14 pt.2)
Tested "just reduce harder" via `LMR_EXTRA` (default 0; higher = more reduction).
- **WAC fixed-depth (deterministic):** LMR_EXTRA 1/2/3 → nodes −5%/−12%/−15%, solves 239/241/243 (noisy, ~−2%),
  EBF barely moves (3.680→3.615). Looked like headroom (accuracy holds when reducing more).
- **Equal-budget gauntlet (fixed-node, the real test):** LMR_EXTRA=3 seed0 +1.2% / seed1 **−11%** (+20 collapses),
  pooled **−4.9%** vs baseline. LMR_EXTRA=2 seed0 −1.6%. **Flat-to-NEGATIVE, and it INCREASES collapses.**
- **Verdict:** the fixed-depth "headroom" was MISLEADING — it hid the real-game accuracy cost. Pushing reduction
  causes blunders → **our ordering cannot safely support harder reduction = we are ORDERING-LIMITED.** The
  single-knob "reduce harder" branch is CLOSED. (LMR_EXTRA left default 0, uncommitted.)

## The EBF strategy (why ordering is the lever) — the mechanism, for the next session
- **EBF = effective branching** = geo-mean of moves-searched-to-depth per node. A node has 30-35 moves; the way
  to sub-3 is to search ranks K+1..N (the 25-35 tail) at steadily-REDUCED depth + prune the deep tail — NOT the
  top few. Reduction should scale with rank (SF's log(moveCount)); we have a rank-scaled DEPTH_REDUCTION/
  statScore-LMR, ordering-compression lets us make it STEEPER safely.
- **What to improve is NOT FMC (87%, already fine = cut-node efficiency). It's the ORDERING TAIL:** compress the
  cutoff-index distribution so the best move almost never sits at a rank where pruning fires. Current histogram:
  95.5% of cutoffs in top-3, but 4.5% from rank>=3 and 2.3% from rank>=8 — those tail cutoffs are what a pushed
  prune would MISS.
- **Key split — ordering unlocks the LOSSY prunes, not LMR:** LMR re-searches → tolerant of a sloppy tail →
  that's why LMR_EXTRA held on WAC but was weak/negative at the gauntlet (re-search claws savings back; over-
  reduction still blunders). **LMP/futility SKIP (no re-search) → they're the big node-savers (LMP-off ~doubled
  the tree) but pinned to conservative flat thresholds by the sloppy tail.** Compress the tail → push LMP/futility
  to a new safe ceiling (harness-metered) = the real EBF lever.
- **Pre-search = latent future lever (SHELVED):** the root pre-search (`reorder_legal_moves`/`pre_minimizer`)
  spends nodes to buy ordering; the engine strongly depends on it (clean "no pre-search" A/B needs real rework,
  user-confirmed). But if the ordering foundation lands, the pre-search's ordering role shrinks → it becomes
  trimmable later = a downstream EBF win. Not now.

## NEXT (the substantive build): the ORDERING FOUNDATION — low-risk (ordering can't hurt the MEAN)
From the history audit, the exact gaps (in leverage order):
1. **Re-key continuation history `from×to` → `piece×to`** (`counterMoveHeuristics`/`contHist2`) — SF's keying;
   generalizes + fills far better; feeds ordering + LMR statScore + prune gates simultaneously. Highest leverage.
2. **Turn on + jointly tune the parked signals**: `ENABLE_CONT_HIST_2PLY`, `ENABLE_CAPTURE_HIST`,
   `ENABLE_CHECK_ORDER` (recalibrate CHECK_ORDER_BONUS), `ENABLE_TT_MOVE`.
3. **Fail-high sibling malus with gravity** into main + continuation history (our MALUS died context-blind;
   with piece-keyed contHist it may live) → makes ordering CONVERGE.
4. **SPSA the handpicked decay/margins** (`DECAY_INTERVAL`, `DECAY_FACTOR`, LMP/futility thresholds).
Measure by: cutoff-tail compression (histogram) + then re-push LMP/futility/LMR at the gauntlet. Ordering
changes only reorder the search → cannot hurt the mean → safe to iterate (no load-bearing-optimism trap).

## EXECUTION LOG (ordering-foundation build, 2026-07-14 pt.4) — plan `ROADMAP-2026-07-14-ordering.md`
- **Step 0 DONE:** roadmap banked.
- **Step 1 DONE — tail instrumented (`ENABLE_CUTOFF_CLASS`, byte-id 247 preserved).** Cutoff move-class × rank
  (search_engine.cpp: `g_cutoff_class_hist` + `cutoff_move_class`; NOTE the engine's `Move.promotion==1` is the
  "no-promo" sentinel, use `>1`). WAC result: **of the rank≥3 tail cutoffs, 57% are plain QUIET (60% at m8+)**,
  43% already-exempt (cap/killer/counter/promo). ⇒ **ordering→LMP cash-in is LIVE** (Fable's "tail may be all
  exempt" worry does not bite); the quiet tail is the target.
- **Step 2 (re-key from×to → piece×to) — SCOPED, next focused push.** ~25 inline `X.from_square*64+X.to_square`
  sites (reads: move_gen.h ordering + search_engine.cpp cutcal_statscore:408 + LMR statScore:412/589/633;
  writes: :3271-3295 / :3742-3766 / :4364+; decay: cache_management.h:1027/1037; decls: cpp_bitboard.cpp:143/149,
  cache_management.h:155/159). Board (`cs`/`current_state`) IS available at every hot site → re-derive piece via
  `piece_type_at(board,sq)`; only `cutcal_statscore` needs a board param added; 2-ply `p2` context is a minor
  re-derivation approximation. Keep tables [4096] (both keys fit; resize to [~448] as a follow-up cache win only
  if piece-key wins the A/B). **TRAP: re-run `ENABLE_STATSCORE_PROFILE` to re-derive STATSCORE_OFFSET/DIVISOR
  after re-key** (the +23 channel was fit to the old distribution). Gate `ENABLE_PIECE_CONTHIST` (off ⇒ byte-id).
  Banked-uncommitted so far: `ENABLE_CUTOFF_CLASS`(off), `ENABLE_PRUNE_LOG`(off), `ENABLE_SIMPL_BIAS`(off).
- **Step 2 IN PROGRESS (partial, byte-id 247 preserved — flag off ⇒ helpers return from×to = identical).**
  - **DONE:** knob `ENABLE_PIECE_CONTHIST`(off) + env-parse + toggle-dump; helpers `piece_type_at` /
    `cont_ent_key(move,st)` / `cont_ctx_key(prev,st)` in cache_management.h (after the contHist externs).
    Converted READ sites: LMR statScore + tier-0-rescue (search_engine.cpp ~:589/596/633/642, board `cs`);
    `cutcal_statscore` (:408, added a `const BoardState&` param; callers :3244/:3251 pass `current_state`).
    Build clean, `wac`=247/41,479,610.
  - **REMAINING (finish next push, keep flag OFF until ALL done — mixing keyed/unkeyed sites corrupts):**
    (1) **WRITE sites** search_engine.cpp ~:3271-3295 / :3742-3766 / :4364-4382 — mechanical: replace
    `previousMove.f*64+previousMove.t` → `cont_ctx_key(previousMove, current_state)`, `p2...` →
    `cont_ctx_key(p2, current_state)`, `move...` → `cont_ent_key(move, current_state)`, `q...` →
    `cont_ent_key(q, current_state)` (board `current_state` available). (2) **ORDERING reads** move_gen.h
    `score_move` lambda (:721,:745) — the raw masks (pawnsMask/knightsMask/... + `state`) are IN scope, so
    compute piece-key inline (or a raw-mask piece_type_at); `score_quiet` (:781, free fn, :795/:817) — ADD a
    board/masks param + update its caller `buildMoveListFromReordered`. (3) Also the 2nd score_move at :952.
    Decay sites (cache_management.h:1027/1037) need NO change (index-agnostic halving). Decls stay [4096]
    (both keys fit; resize to ~[448] as a follow-up cache win only if piece-key wins the A/B).
  - **DESIGN NOTE:** using RE-DERIVATION (Option A: look up piece from the board per read) for the A/B — the
    ordering hot-path pays ~6 bitboard tests/move but at FIXED NODES that doesn't distort the tail-mass gate
    or the gauntlet result (only wall-speed). If piece-key WINS, ship the performant version (Option B: stamp
    moved-piece in the Move struct at generation, read without a board lookup).
  - **THEN Step 2d:** flip `ENABLE_PIECE_CONTHIST=1`, re-run `ENABLE_STATSCORE_PROFILE` to re-derive
    `STATSCORE_OFFSET`/`DIVISOR` (the +23 channel was fit to the old distribution), gate = cutoff-tail-mass
    compression (via `ENABLE_CUTOFF_CLASS` histogram) on WAC + overread_bench.

## STEP 2 COMPLETE — re-key from×to → piece×to DONE + measured (2026-07-14 pt.5)
- **All sites converted** (byte-id 247/41,479,610 with `ENABLE_PIECE_CONTHIST=0` — clean-knob proof intact).
  Writes: search_engine.cpp min/max/root (cont_ctx_key/cont_ent_key). Reads: LMR + move_gen.h (score_move
  lambda, score_quiet w/ derived kingsMask, 2nd score_move) via mask-based `*_bb` helpers (cache_management.h)
  computed identically to the BoardState helpers so READ index == WRITE index for the same position.
- **statScore re-derived (the note's TRAP):** `ENABLE_STATSCORE_PROFILE` w/ piece-key ON → P50=0, spread→
  **STATSCORE_OFFSET=0 STATSCORE_DIVISOR=2048** (was 512/1024). The misfit mattered: misfit run = WAC 246 /
  43.79M nodes (−1 solve, +5.6% nodes = REGRESSION); re-derived = WAC 250 / 39.46M (see below). ALWAYS re-derive.
- **WAC (tactical) = clean WIN** [piece-key ON + 0/2048 vs baseline]: SOLVED **250 vs 247** (+3); NODES **39.46M
  vs 41.48M (−4.9%)**; EBF 3.655 vs 3.680; cutoff tail share m3+ **4.334% vs 4.461%**, m8+ **2.180% vs 2.263%**
  ⇒ **tail compresses + node-efficient = the Step-2 offline gate PASSED.**
- **STS300 (strategic move-match, FIXED-DEPTH) = REGRESSION, 2×2 (isolates cause):**
  |            | ss-LMR ON | ss-LMR OFF |
  | from×to    | 51.7% 1550| 50.4% 1512 |
  | piece×to   | 49.0% 1469| 47.2% 1415 |
  ⇒ the −2.7%/−3.2% drop is from the **ORDERING re-key itself** (consistent across both ss columns), NOT the
  statScore re-derivation (which helps both keyings ~+1.5%). Mechanism: piece×to collapses 4096→~448 contexts →
  denser/faster-filling (helps tactical convergence, −nodes) but LOSES move-pair specificity that aided
  strategic discrimination at fixed depth.
- **THE FORK:** re-key is tactically-positive / strategically-negative-at-fixed-depth / node-efficient. Fixed-
  depth STS is the metric we pre-committed to DISTRUST (LMR_EXTRA lesson); the −4.9% nodes is a depth gain only
  an EQUAL-BUDGET gauntlet can score. Options: (A) run re-key SOLO equal-budget gauntlet now (1-2 seeds) to
  resolve the tension before investing Steps 3-5 — cheap insurance, small plan deviation from "one bundled
  gauntlet"; (B) proceed to Steps 3-5 (malus/parked/cash-in) and let the ONE bundled gauntlet arbitrate (plan
  as written; risk = building on a foundation with a real strategic cost). NOTHING committed; flag default-off.

### STEP 2 failure-mode decomposition (STS300, fixed-depth) — the re-key harm is a REDUCTION interaction
| regime            | baseline | piece-key | deficit |
| default (LMP+LMR) | 51.7%    | 49.0%     | −2.7%   |
| LMP off           | 52.2%    | 51.2%     | −1.0%   |
| LMR off           | 58.1%    | 57.3%     | −0.8%   |
- Disabling EITHER reduction lever collapses the deficit ⇒ piece-key harm ≈ entirely an interaction with
  LMP/LMR: piece-key orders good STRATEGIC quiets LATER (specificity loss from 4096→448 context collapse),
  and LMP prunes / LMR reduces them. Pure-ordering residual is small. (Tactical WAC wins because it searches
  deeper per position + tactical patterns are dense → piece-key generalizes there.)
- ⇒ The crux is DEPTH-SENSITIVITY: does the denser/faster-filling piece-key table produce BETTER ordering as
  the search deepens (fills)? If the STS deficit SHRINKS at higher fixed depth → shallow artifact, carry to the
  equal-budget gauntlet. If it PERSISTS/grows → real strategic-ordering defect, re-key needs rework (blend
  keying / context-only / bonus-scale) before it's a foundation. NEXT: STS300 @ depth 12, base vs piece-key.

### STEP 2 depth-sensitivity (STS300 @ depth 12) — deficit is DEPTH-STABLE, not a shallow artifact
| depth | baseline | piece-key | deficit |
| d10   | 51.7%    | 49.0%     | −2.7%   |
| d12   | 51.7%    | 49.0%     | −2.7%   |
- Scores bit-identical d10→d12 (STS move-choice is depth-insensitive in this band) ⇒ the strategic deficit is a
  STABLE property of the piece-key ordering, NOT something the denser table's fill/generalization heals with
  depth. Extra depth does NOT overcome the loss.
- **VERDICT on the re-key as a foundation:** it is a TACTICAL↔STRATEGIC TRADE, not a clean win. It CONTRADICTS
  the campaign thesis ("better ordering → SAFER pruning") for strategic positions: piece-key orders good
  strategic quiets LATER → LMP/LMR (the exact levers Step 5 wants to push HARDER) prune/reduce them MORE. Note
  the main butterfly historyHeuristics[64][64] is UNCHANGED (still from×to specific); the loss is localized to
  the continuation tables (counterMoveHeuristics/contHist2) that we re-keyed — SF ADDS piece×to contHist to
  from×to history rather than re-keying the counter table onto it. ⇒ candidate fix = keep from×to specificity
  AND add piece×to as a SEPARATE additive term (SF-style coexistence), not a key REPLACEMENT. DECISION PENDING
  (user): (A) equal-budget gauntlet as the only true arbiter; (B) rework to additive-coexist keying; (C) drop.

### STEP 2 coexist REWORK (2026-07-14 pt.6) — additive coexistence REFUTED; piece×to is an inherent WAC↑/STS↓ trade
Rebuilt as SF-style coexistence: reverted counterMoveHeuristics/contHist2 to from×to (byte-id 247 preserved),
added a SEPARATE `pieceContHist[2][512][512]` term SUMMED into the quiet ordering (writes at all cutoff sites,
reads in the 3 move_gen score fns, own decay), gated `ENABLE_PIECE_CONTHIST` + weight knob `PIECE_CONTHIST_SHIFT`
(right-shift). statScore LEFT from×to (isolates the pure ordering effect; no re-derivation needed). byte-id
247/41,479,610 with flag off.
- **Weight sweep (flag on), WAC / STS vs baseline 247 / 51.7%:**
  | SHIFT | WAC | STS  |
  | 0     | 240 | 48.8 |
  | 2     | 244 | —    |
  | 4     | 240 | —    |
  | 6     | 249 | 47.7 |
  Full weight dominates/adds noise (WAC 240, STS 48.8). Heavy attenuation (÷64) recovers WAC (249, +2) but STS
  STILL −4.0% (47.7). EVERY piece×to config (replacement AND coexist, all weights) lands STS 47–49%, never
  recovers 51.7. ⇒ **Retaining from×to specificity does NOT separate the effects.** The piece×to continuation
  signal is INHERENTLY tactically-helpful / strategically-harmful in our ordering (it reorders strategic quiets
  into the LMP/LMR-reduced tail regardless of weight). Coexistence hypothesis REFUTED.
- **REMAINING FORK (post-restart decision):** (A) gauntlet the REPLACEMENT as-is (piece-key incl statScore,
  re-derived STATSCORE_OFFSET=0/DIVISOR=2048 — the strongest tactical version; WAC 250 / −4.9% nodes / STS
  −2.7%) at fixed-node equal budget — the ONLY remaining arbiter of whether the tactical+node gain nets out vs
  the strategic cost; if flat/negative → (C). (C) DROP the piece×to lever entirely; move to the next ordering
  lever (Step 3 sibling-malus + gravity, Step 4 parked signals) on the existing from×to ordering.
- **STATE for restart:** built .so is DEFAULT (flag off) = byte-id 247. Coexist code banked, all gated default-
  off. Nothing committed. The replacement version is recoverable from git-less memory: it was `cont_*_key`
  helpers selecting piece×to on the flag (now they're unconditional from×to + separate pieceContHist term) —
  to gauntlet the REPLACEMENT, either restore the selector helpers OR just gauntlet the coexist@shift0-ish;
  simpler: the replacement A/B is reproducible via ENABLE_PIECE_CONTHIST=1 in the PRE-coexist tree (see pt.5).

### STEP 3 cutcal offline gate (2026-07-14 pt.7) — GREEN: a malus-separable population EXISTS
`ENABLE_CUTCAL_LOG` on the default from×to engine (byte-id preserved). The [0,1024) statScore bucket = **78% of
ALL quiet fail-highs** all read history ~0 (current signal can't rank them). Split by tried-and-failed count:
  tf=0 P(cut)=0.48 · tf=1 0.36 · tf=2 0.29 · tf=3+ **0.25** — clean monotone.
⇒ a fail-high **malus + gravity** (push repeatedly-refuted quiets negative) injects calibration the non-negative
history CANNOT express (all these read ~0). It demotes LOW-P(cut) quiets into the LMP/LMR tail correctly — and
like Obsidian's threat-ordering it demotes BAD quiets, not good strategic ones ⇒ should DODGE the piece×to
strategic-loss trap. Step-3 malus is GREEN to build. (Complementary: threat-ordering handles tactically-LOSING
quiets; malus handles repeatedly-REFUTED quiets — different bad-quiet populations.)

### STEP 4 parked-signal scan (2026-07-14 pt.7) — CHECK_ORDER is the keeper; 2ply/capture NO-GO; TT_MOVE inert
Baseline WAC 247 / 41.48M / STS 51.7%. Single-signal WAC+STS (1-core):
| signal            | WAC | NODES     | STS   | verdict |
| ENABLE_CONT_HIST_2PLY | 241 | +4.5%  | 49.2% | NO-GO (worse everywhere; note our contHist2 is from×to now) |
| ENABLE_CAPTURE_HIST   | 242 | +1.4%  | 48.5% | NO-GO (STS −3.2%) |
| ENABLE_TT_MOVE        | 247 | =      | 51.7% | INERT — byte-identical ⇒ knob not wired OR TT move already ordered first (investigate low-pri) |
| **ENABLE_CHECK_ORDER**| 244 | **−2.7%** | **51.5%** | **KEEP** — m8+ deep tail 50008→35660 (−29%), STS held (−0.2%=noise), WAC −3=noise |
⇒ CHECK_ORDER promotes quiet checks earlier → compresses the DEEP tail + saves nodes while HOLDING STS = a
trap-dodger (demotes nothing strategic). Bundle candidate. The 2-ply/capture signals REORDER via learned tables
and hurt STS like piece×to did — consistent with the "learned reorder = strategic risk" pattern. TT_MOVE inert
= likely redundant with the existing hash-move ordering (moveGenCache path); not worth chasing now.

## CONSOLIDATED CROSS-ENGINE SYNTHESIS (2026-07-14 pt.8) — the convergent build order
Three strong engines (Obsidian NNUE, Caissa NNUE, Ethereal HCE-lineage) mined; briefs in
dev_notes/{obsidian,caissa,ethereal}-search-brief-2026-07-14.md. The convergence is striking and it EXPLAINS
our piece×to failure:
- **⭐ CONVERGENT LEVER #1 — threat-conditioned FROM×TO history.** Obsidian (static threat ordering term),
  Caissa (`[stm][fromAtk][toAtk][from×to]`), Ethereal (`[turn][atk(from)][atk(to)][from][to]`) ALL keep from×to
  AND add a threat-context dimension. ⇒ Our piece×to mistake was REPLACING from×to (discarding strategic
  specificity); the right move is ADD threat context ON TOP of from×to. Demotes tactically-bad quiets, preserves
  strategic ordering ⇒ dodges our trap BY CONSTRUCTION. This is THE foundation lever.
- **⭐ CONVERGENT LEVER #2 — history-GATED pruning (the direct antidote to our failure mechanism).** Ethereal
  exempts HIGH-history quiets from futility/LMP (`hist < {14296,6004}[improving]`), softens SEE-prune by
  `hist/128`, continuation-prunes on MIN(cmhist,fmhist); Caissa scales futility/SEE margins by statScore
  (`+statScore/383`, `−statScore/134`); Obsidian history-prunes `hist<−c·depth`. Principle: PRUNE ONLY WHAT
  ORDERING ALREADY CALLS BAD → good strategic quiets (high history) are PROTECTED from the LMP/LMR tail that ate
  them in the piece×to experiment. This is the cash-in that's SAFE by design.
- **LEVER #3 — fail-high malus + gravity** (our cutcal GREEN, tf-count P(cut) 0.48→0.25; all 3 engines use
  gravity `h += v − h·|v|/16384`). Demotes repeatedly-refuted quiets.
- **LEVER #4 — CHECK_ORDER** (our scan: −2.7% nodes, m8+ −29%, STS held). Already built + gated; just enable.
- **LEVER #5 — deep stat_bonus collapse** (Ethereal: bonus→32 past depth 13) so deep tactical cutoffs don't
  overwrite the strategic history the ordering relies on. Cheap history-hygiene.

### PROPOSED BUILD ORDER (all offline-gated; STS-must-not-drop is the ordering kill criterion)
1. **Threat-conditioned from×to history** (#1) — the foundation. Build the threat-split butterfly (add
   [atk(from)][atk(to)] dims to a from×to history, computed from our attack bitmasks — reuse LATENT_THREAT
   machinery). Gate ENABLE_THREAT_HIST. Offline: WAC↑ + STS-HELD + fixed-depth nodes↓. KILL if STS drops.
2. **CHECK_ORDER on** (#4) — free-ish, already-built, STS-held; fold into the bundle.
3. **Fail-high malus + gravity** (#3) — cutcal-validated; add on top of #1.
4. **History-gated pruning** (#2) — the SAFE cash-in: exempt high-(threat-)history quiets from futility/LMP +
   statScore-scaled margins. Gate via prune-verification harness wrong-rate ≤0.3%.
5. **Deep stat_bonus collapse** (#5) — history hygiene; cheap, offline node/STS check.
6. **[NIGHT 4-core] ONE bundled equal-budget gauntlet** of whatever held STS + optionally confirm-drop piece-key.
- **Piece-key = DROP** (superseded by threat-conditioned from×to, which is the correct form of the same intent).

## STEP-3 threat-conditioned history (2026-07-15 pt.9) — LEARNED threat-split = NO-GO (history under-fill)
Built ENABLE_THREAT_HIST: threatHist[2][2][2][64][64] = the from×to butterfly split by (from-attacked, to-attacked)
via a per-node opponent_threats() bitboard (all-enemy-attacks), computed identically at ordering reads (3 score
fns) + cutoff writes (helper threat_hist_on_cutoff). byte-id 247 preserved flag-off.
- **Additive-on-top (butterfly + threatHist@full):** WAC 239 / nodes +17% / STS 49.3% — double-counts history.
- **Replace (threatHist INSTEAD of butterfly, Ethereal-faithful, no specificity loss since it keeps from×to):**
  WAC 243 / **nodes +15%** / tail m3+ **+48%** / **STS 47.5%**. Still strongly negative.
- **ROOT CAUSE = history UNDER-FILL, not weighting.** Splitting from×to 4 ways makes each bucket ~4× sparser;
  at our budgets (fixed-depth-10, ~40M nodes/300 pos) the split table is STARVED → worse ordering → nodes
  explode. **Same lesson as piece×to seen from the other side: our history is under-filled at our node budgets,
  so ANY table fragmentation hurts** — denser-different (piece×to) OR sparser (threat-split) both fail. Ethereal/
  Caissa fill these at much higher node counts; we don't.
- ⇒ **Learned threat-split history = NO-GO (banked gated-off).** THE PIVOT: stop fragmenting history. Prefer
  levers that (a) DON'T touch the history table shape: **CHECK_ORDER** (validated −2.7% nodes/STS-held),
  **history-GATED pruning** (Ethereal — reads existing from×to history to gate futility/LMP; the big safe
  cash-in, no new table), **fail-high malus+gravity** (cutcal-green, modifies existing from×to in place); and
  (b) Obsidian's **STATIC threat ordering term** (fixed piece-value-scaled bonus/malus from the threat bits —
  needs NO table filling, so it dodges the under-fill trap — but needs LESSER-piece threats, a bigger build).
- STATE: built .so default flag-off = byte-id 247. threat-hist code banked gated-off (ENABLE_THREAT_HIST/
  THREAT_HIST_SHIFT). Nothing committed.

## OVERNIGHT Block 1 (2026-07-15 pt.10) — history-gated pruning results + malus WIN + bundle
Built ENABLE_LMP_HIST_EXEMPT / ENABLE_HIST_PRUNE (read existing from×to butterfly at LMP sites, min+max; byte-id
247 flag-off). Baseline 247/41.48M/STS 51.7%.
- **LMP history EXEMPTION = NO-GO.** EX2000 STS 50.1; EX8000 STS 50.1/nodes +11%; EX32000 = baseline (threshold
  above the history range → never fires). Exempting high-history quiets from our aggressive LMP (LMP_BASE=1)
  just searches junk + shifts the fixed-depth PV worse.
- **History PRUNING (`hist < -coef*rd`) = INERT then NO-GO.** All coefs 128..8192 byte-identical to baseline —
  because ENABLE_HISTORY_MALUS defaults OFF ⇒ butterfly history is NON-NEGATIVE ⇒ prune never fires. With malus
  ON: prune@512 hurts (WAC 243/STS 49.0), prune@2048 never fires (history barely negative). Drop history-prune.
- **★ MALUS ALONE = a genuine node-saver.** ENABLE_HISTORY_MALUS=1: WAC **247 held** / NODES **39.19M (−5.5%)** /
  STS 50.6% (−1.1%). (Contradicts the old "MALUS dead" note — that was the statScore-LMR malus; THIS is the
  ordering-history malus demoting tried quiets. Demotes bad quiets in place, no fragmentation.)
- **BUNDLE CANDIDATES for the overnight gauntlet:** CHECK_ORDER (−2.7% nodes, STS 51.5 held) + MALUS (−5.5%
  nodes, STS 50.6). Combined chk+malus offline测 pending. Gauntlet the strongest node-saver-that-holds-move-
  quality at equal budget (fixed-depth STS dips are the metric we distrust; the node savings = depth).
- All levers gated default-off; byte-id 247 default intact; nothing committed.
- **COMBINED chk+malus (ENABLE_CHECK_ORDER=1 ENABLE_HISTORY_MALUS=1):** WAC **250** (+3) / NODES **35.99M
  (−13.2%!)** / deep-tail m8+ **−43%** / STS 49.0% (−2.7%). The two node-savers STACK hard (−13% nodes) with
  tactics UP — classic tactical↑/fixed-depth-strategic↓ trade; equal-budget gauntlet is the arbiter (−13% nodes
  = real extra depth). Gauntlet queue (sequential conc3 after baseline anchor): chk+malus (aggressive, most
  nodes) → CHECK_ORDER solo (safe, STS-held) → malus solo (middle), each ≥2 seeds vs the baseline anchor.
- **OPERATIONAL NOTE (night):** dev-note banking must use the Edit/Write TOOLS, NOT `cat>>`/heredoc via bash
  (heredoc is not the auto-approved dispatcher prefix → prompts → blocks unattended). Runs = dispatcher form only.

## OVERNIGHT Block 2 — GAUNTLETS vs SF18@400, fixed-node 250k (2026-07-15 pt.11)
- **BASELINE ANCHOR (default engine, seed 0): 45.8%** (72 collapses/300, 58min conc2). Matches historical
  sb_off_s0=45.8% → venue calibrated/reproducible. Bundle must beat 45.8% at equal budget.
- Keras caps concurrency: ≤3 our-engine workers (conc4 OOMs). All gauntlets/Mediocre run conc3 max.
- Queue (sequential): chk+malus s0/s1 → CHECK_ORDER s0 → baseline s1 → malus s0 → [Mediocre tourney].
- **★ chk+malus (ENABLE_CHECK_ORDER=1 ENABLE_HISTORY_MALUS=1) seed 0: 53.0%** vs baseline 45.8% SAME SEED =
  **+7.2 pts paired** (72 collapses, unchanged from baseline → gain is from converting normal games, i.e. the
  −13%-node depth showing as general strength). PROMISING but seed-0 only; baseline swings by seed (s0 45.8 /
  s1 52.7 historically) = regression-to-mean risk (cf. mobility lever). MUST confirm seed 1+ paired before belief.
  Next: chk+malus s1 + baseline s1 (paired), then s2, then attribution (CHECK_ORDER vs malus solo).
- **chk+malus seed 1: 46.5%** (70 collapses). If baseline s1 ≈ historical 52.7 → seed-1 paired = **−6.2** =
  mirror of s0's +7.2 ⇒ REGRESSION-TO-MEAN (helps when baseline low, hurts when high; mean ≈ +0.5 ≈ neutral).
  Running baseline s1 on the current build to confirm the paired delta before any verdict.
- **VERDICT — chk+malus = REGRESSION-TO-MEAN, NO-GO.** baseline s1 = 52.7% (matches historical). Paired:
  s0 45.8→53.0 (+7.2), s1 52.7→46.5 (−6.2), MEAN 49.3→49.8 ≈ **+0.5 neutral**. Helps when baseline low, hurts
  when high = net zero. Vindicates the −2.7% STS drop. Consistent w/ session pattern: node-savers that dip STS
  don't convert. ⇒ PIVOT to CHECK_ORDER solo (held STS 51.5 → the seed-robust candidate). Test both seeds.
- **CHECK_ORDER solo seed 0: 54.5%** vs baseline 45.8 = **+8.7 paired**, AND **57 collapses vs 72** (fewer =
  real signal). BUT s0 is the low-baseline seed (chk+malus also shone here then regressed). Running CHECK_ORDER
  s1 vs baseline 52.7 = the decisive robustness test.
- **CHECK_ORDER solo verdict: SAME regression-to-mean.** s0 45.8→54.5 (+8.7), s1 52.7→47.0 (−5.7), MEAN
  49.3→50.8 = **+1.5 (within noise over 600g)**. Both levers pull both seeds toward ~50 = strongly opening-set-
  dependent, ~neutral on average. Venue reproducible (base s1 = 52.7 exact) → per-seed deltas are real, not
  noise; the SEED-dependence is the phenomenon. NEITHER node-saving ordering lever is a seed-robust ship.
  Getting a 3rd seed (s2) to firm the CHECK_ORDER mean, then Mediocre tourney. **SESSION-LEVEL CONCLUSION
  forming: node-saving ordering/pruning levers = regression-to-mean at our budget; the real strength levers are
  the research-surfaced CORRECTION HISTORY (eval-adjacent, non-optimism) + PAWN-HASH SPEED (regression-immune).**
- **CHECK_ORDER 3-seed:** s0 +8.7 / s1 −5.7 / s2 +3.1 (base s2=46.7, chk s2=49.8), **MEAN +2.0%**. Regression-to-
  mean signature (helps when baseline<50, hurts when >50); +2% is seed-DOMINATED (±7% spread) = suggestive not
  conclusive over 3 seeds. Best candidate but needs a larger seed campaign (next session) to resolve +2 vs noise.
  FINAL Block-2 verdict: node-saving ordering/pruning levers ~neutral-to-marginal at our budget; NOT a clear ship.
- **CHECK_ORDER 4-seed (final): s0 +8.7 / s1 −5.7 / s2 +3.1 / s3 −0.3 (base s3=47.8, chk s3=47.5), MEAN +1.45%**
  (drifted down from 3-seed +2.0). Within noise (95% CI ≈ ±5.6%; a real +1.5% needs ~10+ seeds). CLOSED: node-
  saving ordering lane mined out (~neutral). ⇒ NEXT LANE = **correction history** (highest ceiling, eval-adjacent
  non-optimism, reuses pawn-hash) + **pawn-hash eval cache** (regression-immune speed); both share the pawn-hash
  foundation. See dev_notes/hce-eval-mechanisms + cpp-speed-techniques briefs + [[cross-engine-search-research]].
- **OPS: keep dispatcher commands SINGLE-LINE** (embedded newlines in a pyrun -c break the auto-approve prefix →
  prompt). Read task-output files with the Read tool, never a gratuitous pyrun.
