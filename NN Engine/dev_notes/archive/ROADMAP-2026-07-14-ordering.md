# ROADMAP 2026-07-14 — the ORDERING FOUNDATION → cash-in prune-push (EBF lane)

Supersedes the search-lane portion of ROADMAP-2026-07-12 (the refutation-pipeline frame). Grounded in this
session's search work + a code-grounded Fable gut-check. Full technical map: `search-ordering-pruning-map-2026-07-14.md`.

## Where we are
- **Eval/conversion lane fully CLOSED** (mobility, damps, count-imbalance, simplification move-bias — all NO-GO;
  "make the eval/move-choice less optimistic" is dead from every angle).
- **Search lane, this session:** prune-verification methodology built ([[prune-verification-methodology]]);
  **RFP conditioning NO-GO** — the harness REFUTED the black-box "RFP is the over-pruning culprit" (RFP prunes
  ~99.7% correct on clean+messy corpora; the "+7 WAC solves" was search-path churn). **LMR push = ORDERING-
  LIMITED** (LMR_EXTRA=3 equal-budget gauntlet −4.9% + collapses — pushing reduction blunders).
- **EBF ~3.68 vs SF11 ~1.5-2** (SF11 is HANDCRAFTED-eval → <3 is a legit target). We are NOT missing SF's
  machinery — it's keyed/gated/untuned: continuation history keyed **from×to not piece×to**; TT-move / 2-ply-
  contHist / capture-hist / check-order parked default-OFF; decay/margins handpicked; LMR re-search is safe.

## The thesis (Fable, code-grounded)
Lower EBF toward <3 by compressing the **cutoff TAIL** (best move ~always in the top-K where prunes fire) so the
**lossy** prunes (LMP/futility) + steeper rank-scaled reduction can go harder SAFELY → more depth → fewer
collapses. FMC (87%) is fine (cut-node efficiency); the lever is the tail (4.5% of cutoffs from rank≥3, 2.3%
from rank≥8). LMR is re-search-tolerant/weak; the ordering unlocks the SKIP prunes.

## Why this is genuinely different from the flat 1a re-sort
The flat Lane-1a re-sort re-scored the quiet tail with `score_quiet` (move_gen.h:781) — the SAME context-blind
terms — so re-sorting a weak signal was a no-op BY CONSTRUCTION. The SIGNAL is thin three code-visible ways:
`historyHeuristics` global from×to (no piece/context — same blindness that killed MALUS); `counterMoveHeuristics`
[2][4096][4096] sparse + `DECAY_INTERVAL=35000`-halved ~7×/search → mostly returns 0; killer 10000 / counter
8000 FLAT bonuses dominate history by scale. Piece×to keying (~384 contexts vs 4096) fills ~10× denser. So we
improve the SIGNAL, not the re-sort.

## THE DECISIVE FRAME — bundle, not pieces
The foundation is expected **NEUTRAL SOLO**. Its own internal AND: signal-quality alone ≈ neutral (order barely
changes where prunes don't fire) × prune-push alone proven NEGATIVE (LMR_EXTRA −4.9%). **Elo lives only in the
PAIR** ⇒ gauntlet `{re-key + malus + tuned signals} × {pushed LMP/futility + steeper reduction}`, never a piece.

## Predictive chain (all but the last verifiable OFFLINE)
cutoff-tail-mass compression (signal works?) → prune-verification-harness wrong-rate ≤ ~0.3% at pushed
thresholds (push now safe?) → ONE bundled ≥4-5-seed equal-budget gauntlet (depth converts?). Fixed-depth
accuracy LIES (LMR_EXTRA taught this) — it is NOT a gate.

## Sequence + gates + code traps
0. Bank this roadmap.
1. **Instrument the tail first** — move-CLASS dim on `g_cutoff_histogram`. TRAP: LMP already exempts
   captures/checks/killers/counters (search_engine.cpp:2381-2384); if tail cutoffs are already-exempt classes,
   the LMP unlock is small → cash-in shifts to futility + rank-reduction. Know WHERE the cash-in is first.
2. **Re-key contHist from×to → piece×to** (`ENABLE_PIECE_CONTHIST`, default off ⇒ byte-id 247). Free cache win
   (134MB→KB). TRAP: statScore OFFSET/DIVISOR were derived from the OLD distribution — re-run
   `ENABLE_STATSCORE_PROFILE` + re-derive or the +23 channel degrades. Gate: tail-mass compresses on WAC+
   overread_bench. If NOT → suspect DELIVERY (lazy re-sort scope/top-K/pin) before the signal.
3. **Sibling malus + gravity** — gate FIRST via built `ENABLE_CUTCAL_LOG`/`cutcal_profile_dump`.
4. **Parked signals on + SPSA decay/margins.** Check tail-cutoff SCORE COMPOSITION (killer/counter flat may
   still dominate re-keyed history) not just AUC.
5. **Cash-in prune-push:** `improving`→LMP/futility, history/statScore→LMP threshold, steeper rank reduction.
   Gate: harness wrong-rate ≤0.3%.
6. **ONE bundled ≥4-5-seed equal-budget gauntlet.** Ship on seed-robust positive only.

## STOP RULE (pre-committed)
Tail compresses AND harness clean AND bundled gauntlet still flat ⇒ depth-via-EBF doesn't convert at our node
budgets ⇒ vs_sf11 equal-depth retake / ceiling conversation, NOT another ordering iteration.

## Realism / ceiling
EBF ~2.8-3.2 (not 1.5-2 — SF's needs singular/multicut + fishtest co-tuning on top), ~+0.5-1 ply, bundle Elo
+15-40 (near the 4-5-seed floor → bundle only, never pieces). Ship gate = equal-budget SCORE, not EBF.
Pre-NNUE ceiling ~3100-3300 CCRL stands; 2700→3000 doesn't need NNUE.
