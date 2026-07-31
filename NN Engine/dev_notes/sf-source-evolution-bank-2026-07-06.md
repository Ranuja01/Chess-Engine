# SF source-evolution BANK (Fable Q1 + follow-up, 2026-07-06) — the steal-list

Durable reference distilled from Fable's Q1 (SF1→SF11→SF18 subsystem evolution) + the NPS/systems follow-up.
**Source-verify status noted per item.** We hold all three trees on disk:
- **SF 1.1 (2008)** = `Chess Engine/stockfish_1/stockfish-11_ja/` (folder "11" = "1.1"; binary self-IDs "Stockfish 1.1 JA,
  2004-2008", period UCI options Aggressiveness/Cowardice/KingSafetyCurve). **This is what we played 2026-07-06.**
- **SF 11 (2020, last strong classical HCE)** = `Chess Engine/stockfish_11/stockfish-11-win/src/`.
- **SF 18 (modern NNUE)** = `Chess Engine/stockfish_18_linux/src/` (full source).
(Beware: `stockfish_1/src/` top-level is a *modern* NNUE tree, NOT SF1 — ignore it.)

## Why this maps onto our 2026-07-06 loss (the framing)
vs SF1.1 we decomposed the loss into two independent failures → Fable's answer has two lanes that match 1:1:
- **Equal depth / equal NODES (EBF identical ~3.83, ~46k nodes/move both) → we lose on MOVE QUALITY** = eval
  over-optimism ([[collapse-eval-overoptimism]]) → **LANE A (eval calibration)**.
- **Equal TIME → SF1.1 out-searches us** (higher NPS) → **LANE B (byte-identical systems/NPS wins)**.

---

## LANE A — EVAL CALIBRATION (attacks over-optimism = the root cause; do FIRST)

1. **Correction history (SF18) ★ top pick.** VERIFIED `stockfish_18_linux/src/history.h:209-251`
   (`UnifiedCorrectionHistory`, keyed by pawn_key / minor_piece_key / nonpawn white+black) + `search.cpp:80-96`
   (`correction_value()`; "Add correctionHistory value to raw staticEval"). corrected = raw + Σ·weights/131072.
   A running, hash-keyed corrector that learns our eval's SYSTEMATIC bias online. **Search-side, eval-agnostic, no
   eval-code change, no NNUE.** Direct attack on "eval reads +2 when lost." Non-negamax port: apply the correction to
   the static eval used by RFP/futility/improving on BOTH min and max sides (sign care). Our hooks: needs pawn_key +
   material-key in the incremental zobrist (see Lane B #1), and a small keyed int table.
2. **Initiative/complexity clamp (SF11) = our sharpness meta-gate (Q2.3).** VERIFIED `stockfish_11/.../evaluate.cpp:699-730`;
   clamp `u = sign(mg)*max(min(complexity+50,0),-abs(mg))` (L730), complexity from passers/pawn-count/both-flanks/
   outflanking/infiltration/npm (L718-728). Sign-PRESERVING, damp-toward-zero → caps eval blow-up, no discontinuity,
   cannot flip an assessment. Build from OUR existing detectors (g_capg_tension, passer count, npm) per Fable Q2.3.
3. **Make-room INSIDE the KS term (SF11).** VERIFIED `evaluate.cpp:455`: `kingDanger -= 6 * mg_value(score) / 8`
   (also `-873*!count<QUEEN>(Them)` L453; apply quadratic-mg/linear-eg `score -= make_score(kD*kD/4096, kD/16)` L461,
   gated kingDanger>100 L460). The danger term self-discounts when the rest of the eval already favors us → make-room
   without a tuner. Adopt into our KS function directly (independent of the co-tune ridge fix in Q2.2).
4. **Threats suite (SF11), corr-0.23 hole.** by-victim tables ThreatByMinor/ThreatByRook (evaluate.cpp:116-120),
   Hanging S(69,36), ThreatBySafePawn S(173,94), ThreatByPawnPush, Slider/KnightOnQueen. Sparse-firing, simple
   bitboards. TO-PORT (we have ENABLE_THREATS scaffolding).
5. **Mobility area exclusion recipe (SF11)** (evaluate.cpp:230): `~(low/blocked own pawns | own K,Q | king-blockers |
   enemy pawn attacks)` — the exact boundaries for our planned safe-mobility.
6. **Scale factors (SF11)** — OCB sf=22/64, cap `36+7·strongPawns`, rule50 fade (evaluate.cpp:743-761) = the graded
   version of our parked ENDGAME_SCALE. **Imbalance polynomial** (material.cpp quadratic) + **Space** (npm≥12222 gate,
   weight² — what our dead SPACE lacked) + **Tempo=28**.

**KS evolution note:** SF **1.1 already had** the attack-unit skeleton (attackers·weight + zone + kingDangerPST −
shelter → SafetyTable[100]). What SF1→SF11 ADDED is exactly our gap: safe-check taxonomy (Q780/R1080/B635/N790),
weak-square algebra, pinned-blockers, flank attack/defense, no-queen −873, mobility-diff-in-danger, quadratic
application. Threats / imbalance / space / initiative / mobility-area are ENTIRELY SF1→SF11 additions = our untracked
subsystems. **NNUE-deletion hint** (SF15.1 wholesale, not term-by-term): what SURVIVED into SF18 = material routing +
the optimism/complexity channel → sharpness-conditioning is load-bearing in every era (supports #2).

---

## LANE B — NPS / SYSTEMS (byte-IDENTICAL output; ~30-60% NPS est = real Elo; parallelizable)

Priority order (Fable), with our-engine hooks:
1. **Pawn hash + incremental pawnKey ★ biggest single NPS win.** SF: incremental pawnKey/materialKey in do_move
   (position.cpp), pawn-structure terms computed once per unique pawn config (Pawns::probe). Our eval recomputes every
   pawn term every leaf (profiled #1 cost ~20-28%). Add a pawn key to our incremental zobrist, split pawn-only terms
   out of `placement_and_piece_eval`, front with a direct-mapped table. Byte-identical.
2. **Kill heap traffic on the hot path.** `accessMoveGenCache` returns `std::vector<Move>` BY VALUE → alloc+memcpy+free
   per cache HIT; `buildMoveListFromReordered` another vector; qsearch a fresh `noisy_moves` per node. → return by
   const-ref / per-ply fixed scratch buffers indexed by cur_depth (we already did CaptureStack in eval — same pattern).
3. **TT consolidation (justified twice: capacity + one-line-per-probe).** SF: 10B entry {key16,move16,value16,eval16,
   genBound8,depth8}, 3/cluster in 32B, index `mulhi(key,clusterCount)`. Ours: ~32-40B entries w/ dead alpha/beta,
   direct-mapped, no move, no eval field, no aging + separate evalCacheNew/quiesceEvalCache/moveGenCache = ~4 cache
   misses/node. Pack + 3-way (TT_WAYS knob exists) + replacement `min(depth − 8·relative_age)` + generation bump/search
   + **merge static-eval into the TT entry** (the eval16 field makes our just-shipped RFP nearly free at TT hits).
4. **Prefetch in make_move.** We already compute the child zobrist before make_move → `__builtin_prefetch` the TT +
   eval-cache lines for the child key (SF: position.cpp:966). Two lines, byte-identical, few %.
5. **Mailbox array in BoardState.** SF board[64] updated in do_move → `piece_on` = 1 load. Ours answers "what's on sq"
   with 6-branch mask chains (piece_type_at, SEE get_value_at, zobrist) + rebuilds pieceTypeLookUp every eval. Maintain
   `pieceTypeLookUp` in BoardState/update_state.
6. **int16 history + gravity, no decay sweeps.** Our counterMove/contHist are `int[2][4096][4096]` = **134MB each**
   (DRAM miss/probe, 134MB decay sweeps). SF: int16, [piece][to]-indexed, self-limiting gravity `h += bonus −
   h·|bonus|/limit` (aging free). Our planned [piece][to] reindex + built hist_update gravity (saturation ON, decay OFF).
7. **Per-node blockers/checkers struct.** SF computes checkers/pinners/king-blockers once/node into StateInfo. We
   recompute slider_blockers/checkers up to 16×/node + relevant_pin_exists per LMR. Hoist into a per-node struct.
8. **Eval as set-algebra over cached attack bitboards** (opportunistic): `popcount(attacks & zone)·avgWeight` replaces
   per-square 2D `attackingLayer[x][y]` walks (we proved it with the cheap-bishop popcount surrogate; same transform
   for queen/rook attack-layer accumulation).
9. Minor: `more_than_one(b)=b&(b-1)`; huge pages for TT (after #3); partial insertion sort cutoff `−3000·depth`.

Items 1-7 = byte-identical-OUTPUT → fit the bench-signature discipline (byte-id 245 unaffected until deliberately changed).

---

## SEARCH evolution (Q1) — copy FORM freely, RE-TUNE margins
SF1 already had: fixed-R null-move + zugzwang verification, primitive LMR, 3 futility margins, razoring, IID, killers,
history. SF1→SF11 added the modern set. **Eval-COUPLED (re-tune to OUR eval noise, never copy the constant):** all
futility/razor/null-gate/SEE/LMP margins. **Eval-AGNOSTIC (copy the form):** singular+multicut structure, ProbCut
structure, statScore-LMR machinery, contHist plies {1,2,4,6}, TT design, MovePicker staging. **SF18 post-NNUE, fully
portable to HCE:** correction history (Lane A#1), IIR (depth−1 at no-TT-move nodes, replaces IID + weakens our root
pre-search), cutoffCnt/hindsight reductions, doDeeper/doShallower re-search bands, negative extensions in singular.
Steal-rank (Fable): (1) singular+multicut SF11 form (sBeta=ttValue−2·depth, half-depth excl, ttDepth≥d−3), gated on a
TT-move build; (2) ProbCut SF11 (beta+189−45·improving, ≤2+2·cutNode, qsearch-verify then d−4); (3) correction history;
(4) IIR; (5) dynamic null-move R + `min((eval−beta)/192,3)` (our flat NULLMOVE_EXTRA failed = wrong shape).

MoveGen: SF1 already had plain magics — speed came from **MovePicker economics** (lazy staging, partial-sort limit
−3000·depth, good/bad capture split `see_ge(m,−55·value/1024)`, TT prefetch), not slider tricks. = our 16-pass movegen
backlog's reference design.

TT: 10B×3/32B unchanged SF11→SF18 = a CONVERGED design (fifteen years of fishtest left it alone) → trust it.

---

## PRIORITY SEQUENCE (my read, for the user to ratify)
1. **Correction history (Lane A#1)** — the scalpel for over-optimism, eval-agnostic, lowest-risk. Needs pawnKey (Lane B#1).
2. **Pawn hash + pawnKey (Lane B#1)** — biggest NPS + prerequisite for #1. Byte-identical.
3. **Kill movegen-cache vector-by-value (Lane B#2)** — free Elo, byte-identical.
4. **Initiative clamp / make-room-in-KS (Lane A#2,#3)** — cap the blow-up at the source.
5. Then: TT consolidation, threats suite, the rest — as evidence/effort allows. Everything games-gated (compass is
   untrustworthy — [[compass-context-fragility]], the −33 Elo candidate).

**Discipline:** source-verify each specific constant before porting (done for A#1-3); Lane B changes must keep byte-id
(they're byte-identical-output); margins are RE-TUNED not copied; games decide (node_ab/SPRT), not the cploss compass.
