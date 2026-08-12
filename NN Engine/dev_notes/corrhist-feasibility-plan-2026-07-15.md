# Correction History — feasibility + build plan (2026-07-15)

The highest-ceiling lever from the cross-engine research ([[cross-engine-search-research]]): an ONLINE per-
structure static-eval de-biaser. NOT a feature-add → dodges load-bearing-optimism (structurally can't add
optimism; it's self-correcting toward the search's own verdict). Convergent across Obsidian/Caissa/Weiss.
Full brief: dev_notes/hce-eval-mechanisms-2026-07-15.md.

## Feasibility (from 2026-07-15 scoping) = GREEN, medium build, NO pawn-content scar tissue
- **Pawn key EXISTS:** `generatePawnKey(pawnsMask, occW, occB)` cache_management.h:512-525 — XOR of pawn zobrist
  components only, ≤16 XORs, side-excluded, banked FOR THIS, currently UNCALLED (zero risk). Non-pawn key does
  not exist (small from-scratch add, defer to v2).
- **Static eval captured:** `get_board_evaluation` (search_engine.cpp:6235) = placement_and_piece_eval + single
  root sign flip (`if Config::side_to_play total=-total`, :6286). Node-entry full static eval already in local
  `rfp_static_eval` at every prune site (min + max). Per-ply `g_evalStack` exists but is windowed/surrogate — use
  `rfp_static_eval` (true full) instead. Returned best score = node return value.
- **One injection point:** apply correction to `rfp_static_eval` (RFP/null gates), `early_score` (futility),
  `static_eval` (qsearch standpat). All Config-gated already.
- **Eval-cache trap (known-avoidable):** `evalCacheNew` (raw full-zobrist eval cache) must stay RAW — apply the
  correction ONLY at consumer sites, NEVER inside get_board_evaluation (else poisons cache + double-counts).
- **Corrhist tables = new standalone arrays** (no existing structure-keyed cache).

## THE design care point (non-negamax): SIGN/FRAME
Our eval is absolute Black-positive with a SINGLE root flip by side_to_play → within a search the value frame is
consistent (no per-node negamax flip). Corrhist stores a correction in that frame, conditioned on stm.
delta = (searchBestScore − staticEval) in the frame; corrected = staticEval + corrhist[stm][key]. MUST verify the
exact frame/sign of rfp_static_eval and the min/max return values BEFORE wiring, so corrhist doesn't anti-correct.
This is the first build step (a read + a tiny logging check), not a guess.

## BUILD PLAN (each gated ENABLE_CORR_HIST default-off ⇒ byte-id 247)
1. **Verify frame/sign** (read + optional 1-run log): confirm rfp_static_eval & return-value frame; decide the
   corrhist update sign for min vs max nodes. Gate everything on this.
2. **Table + key:** `pawnCorrHist[2][CORR_SIZE]` (stm × pawnKey%CORR_SIZE, e.g. 16384). Call generatePawnKey at
   the node (cheap recompute; or stash per-ply). Start PAWN-ONLY (highest signal, simplest); non-pawn = v2.
3. **Apply (read):** at rfp_static_eval / early_score / static_eval: `corrected = raw + CORR_W * entry / CORR_DIV`
   (Weiss-ish weights; tune). Gated.
4. **Update (write):** at min/max node end, when NOT in check && best move not a capture && bound consistent:
   `bonus = clamp((bestScore − staticEval)*depth / CORR_K, ±CORR_CAP)`; gravity `entry += bonus − entry*|bonus|/
   CORR_GRAV`. Gated. (This is the sign-critical site.)
5. **Keep evalCacheNew raw** (do NOT correct inside get_board_evaluation).
6. **Offline GATE (the killer feature — gauntlet-free first pass):** measure mean |staticEval − deep-search
   score| on a fixed corpus (WAC/STS positions) with corrhist ON vs OFF — expect the ERROR to DROP. Build/borrow
   a small harness (log static eval + a deeper search score per position; compare). THEN WAC/STS (should hold/
   help; corrected eval feeds prune gates), THEN prune-verification harness wrong-rate at the gates, THEN equal-
   node gauntlet ≥4 seeds.
7. **Later (v2):** non-pawn corrhist + continuation-corrhist; then |correction| → widen/tighten RFP/futility
   margins (the eval-trust signal that raises the prune-safely ceiling).

## PHASE 0 RESULT (2026-07-15) — SIGNAL EXISTS (modest); gate PASSED
Built `ENABLE_CORRHIST_LOG` (byte-id 247 preserved) dumping `[CORRLOG] pawnKey maxbit staticEval bestScore rd` at
minimizer/maximizer node-ends (quiet best, not-in-check, non-mate). `diagnostics/corrhist_signal.py` = cross-
validated held-out residual reduction, per key variant, vs a random-key control (the shrinkage floor). WAC,
stride 1 = 587,336 rows (~100 samples/key; stride-32 was underpowered ⇒ ~=control, misleading).
- FULL SET: pawn×maxbit gross +5.99% vs control +2.87% ⇒ **NET +3.12%**. Best key = **pawn × maxbit** (keying by
  the maximizer/root-side bit matters; pawn-only NET only +1.2%). pawn×maxbit×depth ~same, more overfit.
- **QUIET |err|<300 (raw MAE 134): NET +3.29% SIGNAL. QUIET |err|<600 (raw MAE 227): NET +4.19% SIGNAL.** ⇒ the
  signal CONCENTRATES on quiet positions (where corrhist operates + prune gates fire). RFP-range rd≤5 full set
  NET +2.97% (diluted by tactical outliers with huge MAE that RFP won't fire on anyway).
- **⇒ The pessimistic "structurally blocked / eval-hole" prediction is REFUTED: pawn structure carries a real,
  control-beating, cross-validated systematic bias corrhist can learn.** BUT MODEST: quiet raw MAE ~130-230mp,
  net correctable slice ~4-10mp ⇒ expect a SMALL positive, gauntlet decides. GO to Phase 1 with tempered
  expectations. Use key = **pawn × maxbit**, lambda~16 shrinkage confirmed helpful.

## PHASE 1 RESULT (2026-07-15) — built, byte-id 247; corrhist ON = the load-bearing-optimism trade
Built: pawnCorrHist[2][16384] keyed [maxbit][pawnKey&mask]; integer-EMA update `e += (best-static-e)>>CORR_SHIFT`
at min/max node-ends; apply `rfp_static_eval += entry*CORR_W/CORR_DIV` at both RFP sites; evalCacheNew kept raw.
byte-id 247 flag-off. Baseline 247/41.48M/STS 51.7.
- CORR_W=192 (0.75x): WAC 241 / nodes +11.9% / STS 49.7. CORR_W=32 (0.125x): WAC **250 (+3)** / nodes +4.1% /
  STS 48.7 (−3.0). CORR_W=−192 (sign flip): WAC 234 (−13) / STS 50.9.
- **SIGN CONFIRMED CORRECT** (positive weight helps WAC, flip hurts it). But every positive weight = **tactical↑ /
  strategic↓ / nodes↑** — the SAME trade as piece-key/threat-hist/chk+malus (all gauntleted neutral). MECHANISM
  = load-bearing optimism: de-biasing pulls the systematic over-read DOWN → RFP fires less (nodes↑, never a
  pruning win) → the optimism that drives active play is damped (STS↓). A signal-validated de-biaser reproduces
  the eval-lane wall. Nodes↑ also costs depth at fixed budget.
- Gauntlet CORR_W=32 (best tactical config) s0+s1 vs baselines 45.8/52.7 = the ship gate (STS distrusted).

## FINAL VERDICT (2026-07-15) — corrhist = NO-GO (net −1.7%), eval-ADJACENT lane CLOSED
Ship-gate gauntlet CORR_W=32: **s0 48.3% (base 45.8, +2.5) / s1 46.8% (base 52.7, −5.9) → MEAN −1.7%**, seed-
dependent (+low-baseline / −high-baseline), collapses unchanged (72/64 vs 72/62). NET NEGATIVE.
- **The signal-gated method WORKED as designed**: Phase 0 cheaply PROVED real signal (refuted the structural-
  block prediction) → we built it correctly (sign confirmed, byte-id clean) → the gauntlet gave a definitive,
  well-understood verdict. No wasted iteration against an unknown.
- **The finding**: even a signal-validated, correctly-signed, NON-feature-add eval de-biaser (correction history,
  the highest-ceiling lever from ALL the cross-engine research) FAILS to convert — because our eval's optimism is
  LOAD-BEARING. De-biasing pulls the systematic over-read down → damps the optimism that drives active play →
  −Elo. This closes the eval-ADJACENT lane exactly as the eval-FEATURE lane closed. Load-bearing optimism is
  confirmed FUNDAMENTAL, not feature-specific.
- **SHELVED**: corrhist code banked gated-off (ENABLE_CORR_HIST + CORR_* + ENABLE_CORRHIST_LOG); byte-id 247
  default intact; nothing committed. generatePawnKey still available for the (separate) pawn-content SPEED cache.
- **STRATEGIC IMPLICATION**: lanes now closed = eval-feature, eval-adjacent (corrhist), ordering/pruning node-
  saving (regression-to-mean). The ONLY structurally-regression-immune lane left = **SPEED** (pawn-hash eval
  cache / prefetch / lazy movepicker / SMP): same eval + same moves, just faster → more depth in TIME-based play.
  CAVEAT: our fixed-NODE gauntlet can't see a speed win (same nodes = same result) → speed needs a TIME-based
  measurement (depth@movetime / time-control gauntlet). This is the "ceiling conversation": pre-NNUE search/eval
  levers are ~mined out; the remaining real lever is throughput→depth, measured in time not nodes.

## Sequencing vs the pawn-content cache (separate, later)
Content cache (speed) is the ENTANGLED lever: pawn eval sets globals attack_bitmasks/central_score/off-def/
material that king-safety/imbalance/passer read; a cache must re-emit those. Safe boundary = pawnKey →
{score, passed masks, pawn_rank_bonuses[64]} + re-emit globals. Do AFTER corrhist; both reuse the pawn key.
