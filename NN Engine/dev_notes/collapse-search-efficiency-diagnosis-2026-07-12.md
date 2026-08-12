# Collapse diagnosis COMPLETE + nightshift — it's search efficiency, not eval (2026-07-12)

## Bottom line
The ~22% gauntlet collapse is a **SEARCH-EFFICIENCY gap, not an eval-magnitude hole.** Our search is
~2.5-3x less depth-efficient than SF11 at surfacing the refutations that punish our "winning" moves. The
path forward is the **eval-for-depth** thesis (cheaper/faster/stronger eval → lower EBF / higher NPS → more
depth → fewer collapses), judged by **EBF/NPS/depth, NOT static accuracy**. Do not build another static
eval term for the collapse. byte-id 247 intact; nothing built/committed.

## Evidence chain
1. **3 static-damp attempts failed** (l1a −80 Elo; npedge −1.8% NO-GO; npedge+quiet-gate self-defeating) —
   the move-selection lens showed a static positional change scatters the tactical `sts` stratum
   (intra-subtree volatility). See `npedge-damp-build-2026-07-10.md`, `[[collapse-is-tactical-not-static]]`.
2. **Classified 197 losing collapse decisions** (`diagnose_collapse_moves.py`): 37% downstream (peak move
   was SF's own → loss later), **27% over-push into counterplay** (~40% into a capture/check refutation),
   25% other quiet, 13% minor.
3. **Over-push triage** (`triage_slice.py`, our deep+lowprune): 68% HORIZON (deeper avoids), 26% EVAL, 6%
   PRUNING. (Caveat: HORIZON = "more nodes avoids it", partly tautological.)
4. **⭐ SF11 test** (`sf11_overpush_test.py`) — the decisive one: on the 53 over-push FENs, SF11's classical
   STATIC eval AGREES with ours (SF11 static gap −95cp, ours −92cp; both PREFER our over-push in ~71%), but
   SF11's SEARCH avoids it (d6 66% / d10 83% / d14 94%). `other_error` slice = same (SF11 d6 78%). ⇒ a
   classical static term cannot fix this — you'd have to beat SF11's static eval, which SF11 itself doesn't
   (it uses search). The danger is not in the classical static eval; SEARCH corrects it.
5. **⭐ Our per-depth avoidance** (`our_depth_avoid.py`): OUR engine avoids the over-push d6 36% / d10 42% /
   d14 51% — vs SF11 66/83/94. Our d14 (51%) < SF11's d6 (66%) = ~2.5-3x depth-efficiency gap. But MONOTONIC
   ⇒ depth directly reduces collapses. "Plays SF18's exact move" also climbs 9/17/25% (d6/10/14).

## Nightshift results (2026-07-12, all prompt-free knob A/Bs; over-push d10 avoidance, baseline 42%)
- **Search-soundness levers = consistent NULL** (none closes the SF11 gap): OTV 42% (flat), CHECK_EXTENSION=5
  36%, ENABLE_SINGULAR 32%, pruning-relaxed (LMP/LMR/nullmove off) 40%. All ≤ baseline. ⇒ the gap is
  HOLISTIC search quality (SF11's ordering/eval-in-tree/reductions together), not a single knob. The
  search-technique lane is tapped for the collapse (consistent with prior "search tapped" findings). Also:
  pruning-relaxed not helping = our pruning is NOT cutting the refutation (matches triage PRUNING=6%); the
  refutation just isn't surfaced/valued by our search shallow.
- **Eval-speed via `wac` = wrong instrument + negative knobs:** ENABLE_CHEAP_QUEEN_MOBILITY 242 WAC
  (−5)/+0.5% nodes; ENABLE_CHEAP_KNIGHT_MOBILITY 239 (−8)/+7.6% nodes. Both approximations LOSE accuracy
  without saving nodes. **KEY: `wac` counts nodes-to-solve, NOT NPS/wall-clock — it cannot measure the
  "cheaper eval → faster" benefit.** The speed prong needs a fixed-TIME or NPS bench (which the dispatcher
  lacks). This is the main tooling gap to close before pursuing the eval-speed lever.
- **Depth-curve confirmation:** OUR over-push avoidance d6/d10/d14/**d18 = 36/42/51/68%** (SF18-exact-move
  9/17/25/30%) — monotonic and ACCELERATING (+17pts d14→d18); our d18 (68%) reaches SF11's d6 (66%).
  `other_error` generalization: our d10 45% vs SF11 d10 88% — same ~2x gap, not over-push-specific.
  ⇒ depth increasingly fixes the collapse; the thesis (buy depth → fewer collapses) is well-supported.

## The lever (user's thesis = campaign throughline)
Cheaper/faster/stronger eval → lower EBF / higher NPS → more DEPTH → fewer collapses. Works INDIRECTLY (via
depth), not by statically flagging the danger. **Judge by EBF/NPS/depth, reject changes that raise EBF even
if "more accurate" (npedge raised EBF +7%).** SPEED prong robust (eval-speed bundle shipped +11); EBF-via-
accuracy conditional. Connects [[eval-accuracy-payoff-is-pruning]], [[eval-speed-bundle-shipped]], EBF-campaign.

## Recommended next steps (morning)
1. **Build a fixed-TIME / NPS instrument** (the missing tool) — e.g. depth-reached at fixed movetime, or
   NPS bench — so eval-speed changes can be measured on the thesis's own metric. Without it we're flying
   blind on the speed prong.
2. **Profile the eval** to find genuine speedups that DON'T cost accuracy (unlike the cheap-mobility
   approximations, which trade accuracy). Target the hotspots (bishop colour-complex was ~33%; already
   cheapened and shipped — find the next).
3. **Validate the depth→collapse link at the gauntlet** by running our engine at higher OUR_NODES (needs a
   non-prompt-free env prefix → do it with the user present): if collapse rate drops as nodes rise, the
   whole thesis is confirmed at the real venue.
4. Optionally re-triage the 37% "downstream" slice (walk to first SF-divergence) to cover 100% of losses.
