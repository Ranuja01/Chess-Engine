# Search architecture map (2026-07-15) — Fable full read of search_engine.{h,cpp} + cache_management.h

Grounding reference for the current byte-id-247 build. Read before TT/ordering/pruning work so we don't re-audit.

## Structure (NOT negamax)
- Explicit minimax split: `minimizer()` (search_engine.cpp:3060) + `maximizer()` (:3944), ~900-line hand-mirrored;
  per-child helpers `get_score_for_minimizer` (:2357) / `_maximizer` (:2700) hold TT probe + PVS scout + LMR/LMP/
  futility + re-search. EVERY pruning device exists TWICE, sign-mirrored by hand (silent min/max asymmetry risk).
- Eval sign: `get_board_evaluation` (:6309) = absolute `placement_and_piece_eval` + single global flip
  `if (Config::side_to_play) total=-total` → engine-perspective. maximizer = always engine side; root (engine to
  move) calls `minimizer` on each root child. Same flip in cheap_eval/eval_by_mode-mode1/qSearch mate (:5232).
- Root: `get_engine_move`(:1702)→`alpha_beta`(:2030), ID from depth 3, aspiration windows (ASPIRATION_DELTA=500,
  widen×2 up to 3×, then full; :1827) with completed_move/score guard; resign ≤ −15000; stop ≥ 9,000,000.
- Root pre-search: `reorder_legal_moves`(:4584)→`pre_minimizer`(:4802) = a THIRD simplified minimizer copy (LMR+TT,
  no null/futility/LMP) that builds the 2nd-level move lists `minimizer` uses at cur_depth==1. `ENABLE_ROOT_PRESEARCH=0` reuses prior iter's lists.
- Frame: cur_depth = plies from root, depth_limit = target, remaining = depth_limit−cur_depth. Reductions/extensions
  change the CHILD's depth_limit (LMR passes smaller reduced_depth; check-ext does depth_limit++), not decrement.
- Audited min/max asymmetries (documented parity, not colour bugs): min futility extra `cur_depth>1` gate (:2529);
  min PVS `i==0 || rd==1` vs max `i==0` (:2411/:2757); NULLMOVE_CURDEPTH_MINI 3(min)/4(max); min cur_depth==1
  special root-child branch (:3221) has NO RFP/null/IIR/ProbCut/singular (interior `else` :3439 gets full set).

## TT / moveGenCache / ordering
- **TTEntry** (cache_management.h:69): key/score/depth/flag/`Move move`(SINGULAR ONLY, not on live ordering path)/
  dead alpha,beta/valid. **NO hash move on the live path** (the TT-hash-move lever = still open). Direct-mapped
  (TT_WAYS=1), depth-preferred same-key replace. Child-keyed probe AFTER make_move inside get_score_* (:2391/2737).
- **Store `addToSearchEvalCache`(:941) refuses |score|≥9,000,000 AND score==0** → every mate AND every exact-DRAW
  subtree re-searched from scratch each visit; drops legit EXACT draws. No mate-distance adjustment.
- moveGenCache: `buildMoveListFromReordered`(:5906) replays a FROZEN sorted list on hit (+ killer/counter promote +
  ENABLE_LAZY_RESORT top-2 bubble); miss → `generateLegalMovesReordered`(move_gen.h:~745) scores+stable_sorts once
  (captures 200000+MVV-LVA, SEE for unclear; quiets HH + killer + 8000·counter + counterMoveHeuristics from×to +
  moveFrequency + promo). **Ordering FROZEN at first visit** except cutoff-promotion + top-2 lazy resort → history
  evolution can't reorder cached nodes (the "ordering unfreeze" lane).

## Pruning/reductions (shipped defaults, mirrored both sides)
RFP (rd[1,6], margin 1500mp/ply, +73) · null-move (cur_depth≥3/4, table reduction −NULLMOVE_EXTRA(2)−1@d≥10, no
verify) · futility (do_lmr-eligible, rd(1,4], margins {200,450,650,950}, full eval) · LMP (rd≤5, i≥1+rd²,
sentinel) · LMR (DEPTH_REDUCTION − log2(i−2)/scale, +continuous statScore-LMR delta clamp±2, reduce-more scaled
HISTORY_LMR_SCALE=2, pin-protected) · **VERIFY_MARGIN=16000** = nearly EVERY reduced fail-low re-verified at
depth−2 (big hidden cost/safety trade, engine-unique) · root razoring = a **`break`** (abandons ALL remaining root
moves on one stale-scored move) · check-ext=3 (SEE-filtered). qSearch(:5168): stand-pat, whole-node delta-prune
(DELTA_MARGIN=1500), noisy = promos+SEE≥0 captures + quiet-checks-at-every-qply (full board-copy per quiet =
the profiled cost), losing-SEE captures excluded entirely, MAX_QDEPTH=10.

## ⚠️ STRUCTURAL FINDINGS worth acting on (NEW this digest)
1. **alpha_beta root-child TT stores over-trust depth by +1** (:2134, :2250 store `depth_limit` for a child
   searched to remaining depth_limit−1) — the EXACT bug `ENABLE_TT_DEPTH_FIX` fixed in reorder_legal_moves but
   NOT here. A real (small) correctness issue on the root path.
2. **Live `std::cout` debug traps in the hot path** — e.g. minimizer :3705 `if (current_state.occupied ==
   7199354783056128661) std::cout ...`. Fires if that exact position occurs mid-search. Should be removed.
3. TT refuses score==0/mate (above) — re-search waste + dropped EXACT draws.
4. Root razoring `break` is aggressive (drops good root moves on a stale prev-iter bound under aspiration/fail-low).
5. History pollution: node-end `historyHeuristics += (depth_limit−cur_depth)` on best move of FAIL-LOW nodes
   (:3928, pre_minimizer :5162); `best_move` defaults {0,0,0} → degenerate/all-TT-hit node credits `[0][0]`.
6. Dead weight: `use_tt_entry`'s is_maximizing/use_extra_precautions args + TTEntry alpha/beta fields (~8B/entry).
7. Frozen movegen ordering (above) = the documented ordering-unfreeze lane, still open.
8. qSearch checks repetition but NOT the 50-move rule (:5171); losing-SEE captures never get a tactical re-proof.

Key files: search_engine.cpp/.h, cache_management.h (TT/moveGenCache/QCache), move_gen.h (generateLegalMovesReordered/score_quiet).
