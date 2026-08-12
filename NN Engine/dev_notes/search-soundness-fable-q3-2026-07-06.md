# Search-soundness BANK (Fable Q3, 2026-07-06) — the phantom-PV diagnosis + our-code reconciliation

Last Fable interaction (deadline). Source-verified against our code where load-bearing. Companion to
[[sf-source-evolution-bank]]. THE lane: our search over-values lines whose refutation was reduced into invisibility.

## The finding that triggered it (bias_profile.py, SF18-anchored)
At our over-optimistic peaks: our_static +0.16 = SF_static +0.15 = SF_search(d12) +0.10 (static eval WELL-CALIBRATED),
but our SEARCH returned +4.03 → **+3.9 pawns of PHANTOM the static never claimed.** = a phantom PV: a buried
refutation (fails low at reduced depth, is trusted, silently deletes the opponent's best reply, inflates to root).
Loses 95% to SF1.1(2008) at EQUAL depth+nodes = soundness, not knowledge/speed. 2026-06-01 ablation: LMR alone
caused 28/29 deep-misevals.

## Fable Q3 — how SF prevents phantoms (ranked for SOUNDNESS)
1. **Continuously-refreshed move ordering (the real #1).** SF builds a fresh MovePicker every node visit, re-scoring
   quiets from LIVE history → a refutation that fires anywhere migrates forward everywhere next visit → phantoms die
   young. Under aggressive LMR, ordering quality decides which moves suffer reduction error.
2. **Iterative deepening + TT move + aspiration fail-low = TEMPORAL verification.** Fail-lows aren't verified within
   an iteration — they're re-tried across iterations with improving ordering (TT/history seeds); a phantom that
   collapses produces a root fail-low the aspiration loop re-searches before committing.
3. Unconditional re-search on reduced fail-HIGHs (we have it; guards ONE direction only).
4. **Singular extensions** — the direct anti-phantom device: one move carrying implausible value → extend, don't
   trust. Needs the TT move. (Multicut = the pruning by-product, not the soundness part.)
5. **doDeeper** (SF18: reduced search returns > bestValue+50 → re-search deeper) — distrust surprising reduced results
   by MAGNITUDE, not window position.
6. Null-move verification (narrow; zugzwang only). 7. statScore/cutNode LMR gates (marginal).

**TT: aging = capacity hygiene, NOT the soundness safeguard.** An ageless TT doesn't manufacture phantoms (bounds
don't expire); it rots depth-preferred replacement (ancient deep entries crowd out current search → effective TT
shrinks → worse ordering seeds = a QUALITY TAX). Genuine soundness caveats: (a) EXACT-on-fail-soft bug (we found+fixed
ours); (b) graph-history path-dependence (rep/50-move scores invalid on other paths) — small, accepted, don't "fix" by
storing rep-draws eagerly. Our LMR ablation already convicted the right defendant (LMR/ordering, not TT).

**Min/max port:** everything ports (mechanisms assume "re-search when cheap ≠ expected", mirrors fine). Watch SIGN:
trigger inequalities flip in the minimizer; singular margin subtract-for-max/add-for-min; doDeeper margins flip;
killers/counters/contHist side-agnostic. DISCIPLINE: implement each as a direction-parameterized helper, instantiate
twice, + add a **mirror self-test** (search a position and its color-flip at fixed depth, assert IDENTICAL node counts)
to the bench signature — catches search sign-asymmetry the way the eval-mirror probe caught eval asymmetry.

**VERIFY_MARGIN = wrong axis.** Our phantoms (+3.9) are FAR-miss fail-lows (the sac scores ~3p below alpha, outside
any sane margin) + graduated re-search at depth−2 often can't see the point. Widening buys coverage sublinearly, pays
nodes linearly (NULLMOVE_PROGRESSIVE economics). Don't widen it.

## OUR-CODE RECONCILIATION (verified — Fable's "frozen ordering" is OVERSTATED; the real hole is narrow)
`buildMoveListFromReordered` (search_engine.cpp:4992-5053) on a cache HIT:
- Cutoff move IS promoted to front (`updateMoveCacheForBetaCutoff`, cache_management.h:869-921). ✓ not frozen.
- Quiet tail IS re-scored from CURRENT history + top-`PROMOTE_TOP_K`(2) bubbled — **but ONLY when
  `do_topk = last_cutoff_index > LAZY_RESORT_MIN_CUTOFF_IDX(2)`** (a prior DEEP cutoff flagged the node poorly-ordered).
- Full stable_sort re-sort is **OFF**: `RESORT_AFTER_REUSES=1000000` because **the periodic full re-sort REGRESSED STS**
  (search_engine.h:329). ⇒ Fable's headline fix (unfreeze ordering) = ALREADY TRIED, regressed.
- **THE HOLE:** at `last_cutoff_index == -1` nodes (never cut here = all-nodes / fail-low PV nodes = exactly where the
  phantom's buried refutation lives), the quiet tail gets **NO** history re-score → a history-positive refutation that
  hasn't cut HERE stays buried across every ID iteration = self-sealing, confirmed and localized.

## THE CAMPAIGN (prioritized, given the STS-regression history)
Ordering-unfreeze is the RISKY lever (full re-sort already regressed STS; extending re-score to lci=-1 nodes re-scores
many nodes → same risk). Lead instead with the NON-ordering levers:
1. **★ Optimism-triggered verification (Fable's bespoke, tailored to our +3.9 signature):** at a PV node where
   backed-up value exceeds the node's OWN static eval by > T (~1.5-2p), re-search the PV child with reductions OFF (or
   depth+1). = doDeeper generalized to our exact signature; rare trigger (cheap); our well-calibrated static (fresh
   finding) makes the trigger trustworthy. NO ordering risk. FIRST experiment.
2. **Reduce-less on contHist-positive replies** — the shelved reduce-less arm (adjudicated under the broken cploss
   compass, like KS) → node_ab re-trial.
3. **Singular extensions** (post-TT-move) — purpose-built for "one move carries implausible value."
4. (Risky/last) targeted ordering: re-score quiet tail at lci=-1 nodes too — but guard against the STS regression.

**EBF 2.2 + soundness are the SAME property** (info quality per node): SF allocates selectivity by refreshed evidence
(errors transient), we allocate by static rules over a stale order (errors persistent). Gate on games (node_ab/SPRT);
mirror self-test in the bench signature. This lane (per LMR ablation + SF1.1-at-equal-nodes) is plausibly worth more
than anything else queued. Ties [[engine-cpp-optimization]] (06-01 LMR ablation, VERIFY_MARGIN), [[collapse-eval-overoptimism]].

## Fable Q4 — OTV implementation design (the parts that decide it)
1. **Re-search method:** NOT depth+1 (SF `doDeeper` = 1-ply paranoia on a fail-HIGH re-search, reductions ACTIVE
   inside — wrong tool; a +3.9 sac-refutation needs 3-4 plies, so depth+1 just deepens the phantom line). NOT full
   reductions-off (2-4× nodes). RIGHT = **reductions OFF for the first `OTV_PLIES`(2) plies of the re-searched child
   subtree, normal below** — the refutation is almost always an early reply (subtree-ply 1-3) late-ordered+reduced;
   unreduce where the phantom is anchored, anything failing high hits the existing full-depth re-search chain. Trigger
   is rare → afford the conclusive version (a verification that can't see the refutation CERTIFIES the phantom = worse
   than none). Impl: path-scoped `g_verify_plies` + RAII (clone CheckExtensionGuard), `do_lmr &= (g_verify_plies==0)`.
2. **Placement = in-loop, VERIFY-BEFORE-ACCEPT** (before `best=score` AND before the cutoff return), never post-loop:
   a superseded suspicious child never propagates (skip); post-loop misses the cutoff path, and at scout nodes the
   CUTOFF is the propagation channel (phantom-high min-scout return distorts the parent). Structural relative = singular
   exclusion search. Staged: v1 = PV/full-window (beta-alpha>1) + root; v2 = cutoff-verify at scout nodes if v1 converts.
3. **Reference = static eval** (phantom's literal signature = static-vs-search divergence). Works for US specifically:
   `approximate_capture_gains` bakes resolved exchanges into static → trivially-won material shows NO static gap → low
   false-trigger rate → tight margin viable. T=**1750**, sweep {1250,1750,2500} node_ab, target **1-3% of PV accepts**.
   Sign: max `score>static+T` / min `score<static-T`. Depth-scale `T+150·remaining` ONLY if false-triggers cluster deep.
4. **Recursion:** FREEBIE — children finish before parents → a refuted phantom deflates at its birth node, ancestors
   never see the gap → refuted phantoms DON'T cascade. Risk = a CONFIRMED tactic re-triggering at every PV ancestor →
   guards: (1) verified-once=trusted (per-(node,move) flag, accept unconditionally after 1 verify = no infinite loop);
   (2) path budget `OTV_PATH_CAP`(2-3) via `g_verify_count` RAII; (3) suppress nested OTV in verify subtrees
   (`g_verify_plies>0` doubles as this). Measure trigger-rate + node-overhead on WAC+collapse BEFORE strength.
5. **HELD PREDICTION (Fable):** trigger-rate at T=1750 > ~5% of PV accepts ⇒ static less calibrated across-the-board
   than the peak measurement suggested ⇒ rethink margin/reference BEFORE spending games.
Note: **OTV is BESPOKE — SF has no analog** (closest = singular exclusion). = the "ours not SF-distillation" principle:
adopt universal soundness/speed physics, keep our eval + min/max identity, borrow SF mechanism FORMS but re-tune
constants. Full plan: `~/.claude/plans/handoff-collapse-fix-eventual-allen.md`.

## Fable Q5 — the roadmap after OTV shelved / reduce-less converts (+18.3 node_ab)
**1. Port CONTINUOUS statScore-LMR (generalizes the flat +18 reduce-less into a graded two-sided channel).** SF11 shape:
`statScore = mainHistory[us][from_to] + contHist(1,2,4 plies back)[movedPiece][to] − 4926;  r -= statScore/16384;
d = clamp(newDepth − r, 1, newDepth)`. (SF18 = same shape retuned `−statScore·850/8192`.) THREE PORT CAUTIONS:
(a) the −4926 is a RE-CENTERING offset → make statScore zero-mean; **compute OUR OWN offset = empirical median of our
combined history sum on a fixed-depth run, DON'T copy 4926**. (b) divisor is unit-dependent → pick so **P95(|statScore|/
divisor) ≈ 1.5 plies**, sweep ×½/×2 on node_ab; **turn ENABLE_HISTORY_SATURATION ON** (bounded units → stable divisor).
(c) clamp delta to ±2 plies initially. Continuous > flat +1 MODERATELY: graded credit + subsumes tier-0 reduce-more as
the negative half + YOU-SPECIFIC synergy: a refutation accumulates history immediately → its reduction shrinks next
visit → a continuous patch on the frozen-cache self-sealing-phantom loop. Expect an increment on +18.3, not a multiple.
Days of work INSIDE `history_lmr_delta` — replace the tiered return with the computed clamped delta.
**2. Direction balance:** reduce-LESS = soundness (protects refutations), reduce-MORE = EBF (prunes never-cut noise) —
both have converted (reduce-more shipped June, reduce-less now). SF = ONE symmetric zero-centered continuous channel
(offset auto-balances) + asymmetric STRUCTURAL channels on top (reduce-more: cutNode+2/ttCapture+1; reduce-less:
ttPv−2/singular−2). Port that architecture; keep our structural exemptions (checks/captures/killers unreduced).
**3. Verification NOT dead — singular has the right cost shape (OTV had the wrong one).** OTV = fresh reductions-off
full-depth re-search for a 0.4% event = cure priced above disease. SINGULAR flips every cost: exclusion at HALF depth
null window (cheap); interrogates the TT MOVE (no discovery cost, value already cached); productive both ways (multicut
= free cutoff / extension = anti-phantom depth); fires on high-prior nodes not anomalies. Promote singular/multicut to
THE priority verification, gated behind the TT-move field. SF11 v1 verbatim: `singularBeta = ttValue − 2·depth`, excl
search at `depth/2`, extend on fail, return singularBeta on multicut. Cautions: margin + multicut bound SIGN-MIRROR in
minimizer (color-flip A/A node-count test); put singular under a shared per-path extension cap (we're already ext-rich).
**4. RE-RANK (strength/effort):** (1) **continuous statScore-LMR** — builds on the only converting lever, zero prereqs,
days; + the sibling **re-score quiet tail from live history on move-gen-cache hits** (treats the CAUSE — stale ordering
— vs statScore's symptom; compound. CAUTION: the full tail re-sort already regressed STS, so this is the risky one).
(2) **TT-move → IIR → singular/multicut** — bigger prize but a prereq chain (TT entry rebuild first, wanted anyway for
eval16/aging/packing). NEXT CAMPAIGN, not this week. (3) **Pawn-hash + NPS lane** — run IN PARALLEL (byte-identical →
needs only bench + timed sanity, no node_ab/SPRT slot; near-guaranteed Elo — prior +8.6% NPS = +11; makes every future
search mechanism cheaper; zero interaction risk). **CLOSING (Fable):** lane-1 statScore arrives first — do NOT re-rank
the TT/singular campaign DOWN if statScore comes back big; phantom PVs are a STRUCTURAL hole, statScore is a statistical
PATCH over it; the SF1.1-at-equal-nodes datum says the hole is worth more than any single patch.
