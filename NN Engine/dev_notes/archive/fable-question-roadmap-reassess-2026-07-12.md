# Fable consult — ROADMAP REASSESSMENT (final consult): the 2026-07-06 search-soundness roadmap, 6 days of results, and what to do now (2026-07-12)

*You (Fable) are going away today, so this is a long-horizon ask: reassess the master roadmap
(`dev_notes/ROADMAP-2026-07-06.md`) given what we've learned, and hand us a durable, prioritized plan we
can run without you. Self-contained; repo pointers §7. Venue trust unchanged: external gauntlet vs
throttled native SF18 (≥3 seeds) = TRUTH; lightning self-play + cploss compass ANTI-PREDICT; node_ab/wac =
screens.*

## 0. The one-line ask
Given §2–§4, is the roadmap's **search-soundness-FIRST, eval-blocked-by-search** frame still right, what is
the **next highest-leverage lever** now that `statScore-LMR` shipped, and how should the **eval lane be
reframed** around the user's "eval-for-depth" thesis? Update the prioritized program.

## 1. The old roadmap in one paragraph (what we were told 2026-07-06)
Root cause = our search hallucinates **phantom PVs** (returns ~+3.9 over its own well-calibrated static
eval); **we lose to SF1.1(2008) at EQUAL NODES on MOVE QUALITY — not eval, not speed.** Therefore
**SEARCH-SOUNDNESS FIRST**; **eval is BLOCKED by search** (a ±0.5p term can't outvote a +3.9 phantom → "the
−33 KS is unadjudicable under this search, not failed" → dynamic eval terms keep dying). Lanes: L1
search-soundness (OTV shelved −17.8; **reduce-less/statScore-LMR = the positive levers**), L2
TT+ordering+negamax-lite+singular, L2b missing prune features (null-eval-gate, ProbCut, dynamic-null-R,
IIR), L3 eval-completeness (parallel, low-coupling only; **HELD until search sound**), L4 NPS. Gates:
node_ab → SPRT; byte-id after byte-identical changes.

## 2. What happened since (6 days) — the roadmap PREDICTED our main result
- ✅ **`statScore-LMR` SHIPPED** (L1's "NEXT"): +23 lightning SPRT / +33 node_ab, new byte-id 247/41,479,610
  (was 245). The continuous reduce-less/reduce-more channel the roadmap wanted. This is the one clean win.
- ⚠️ **We then spent ~a week on the EVAL lane — the "~22% collapse" — and it FAILED exactly as the roadmap
  predicted ("eval blocked by search").** Three static-eval attempts to fix the collapse over-read:
  l1a broad optimism-damp **−80 Elo**; a Fable-designed surgical `npedge` realizability damp (offline
  slices all clean) **−1.8% NO-GO**; a tension-gated variant self-defeating. The move-selection lens showed
  WHY: a static positional change **scatters the tactical stratum** (intra-subtree volatility reorders
  exchange sequences — the design rule you gave us: a term may be volatile only in the direction the
  tactics already move). **So: the roadmap was right — we should not have been in the eval lane. We
  re-learned "eval is blocked by search" at a cost.**
- ⭐ **NEW, SHARPER evidence for the search-first thesis (stronger than the original "+3.9 phantom"):**
  we diagnosed the actual losing collapse moves and compared to **SF11 (classical HCE)**:
  - **SF11's classical STATIC eval AGREES with ours** — on the "over-push" losing moves, SF11-static
    *prefers* our move (mean gap −95cp; our own static −92cp; both prefer it in ~71%). So it is NOT that
    our static eval is uniquely weak; **classical static eval as a class over-values these** (the material/
    space grabbed looks good statically). *(Method note the user insists on: compare TOTALS, not term
    names — SF11's "King safety"/"Threats" ≠ ours; only total-vs-total is valid.)*
  - **SF11's SEARCH fixes it shallow:** SF11 avoids the over-push at **d6 66% / d10 83% / d14 94%**.
  - **OUR engine at matched depth is ~2.5–3× less efficient:** we avoid at **d6 36% / d10 42% / d14 51% /
    d18 68%** — our **d18 (68%) barely reaches SF11's d6 (66%)**. But the curve is **monotonic and
    accelerating** ⇒ depth directly and increasingly fixes the collapse. Generalizes beyond the slice
    (`other_error`: our d10 45% vs SF11 88%).
  - ⇒ **The collapse is a SEARCH-EFFICIENCY / SELECTIVITY gap, not an eval-magnitude hole.** SF surfaces
    the refutation at d6 what we bury until d18. This is the roadmap's "move quality at equal nodes,"
    now measured against a *classical* engine (so it's not the NNUE boundary for the search part).
- **Search knobs we A/B'd on the over-push do NOT close it** (over-push d10 avoidance, baseline 42%):
  OTV 42%, CHECK_EXTENSION=5 36%, **ENABLE_SINGULAR 32%** (our singular, built last week, was also
  gauntlet-NEUTRAL → banked), pruning-relaxed 40%. All ≤ baseline ⇒ **the gap is HOLISTIC** (ordering +
  eval-in-tree + reductions together), not a single knob.

## 3. The reframes the new evidence forces (user-driven, please weigh in)
1. **Eval↔search coupling is making us CIRCLE.** Every offline lens (cploss, move-flip, SF11-static,
   depth-avoidance) isolates ONE facet of a coupled system, so each points at a different lane and we
   oscillate "it's search / no it's eval." The user's synthesis: **it's BOTH, coupled** — SF is better at
   eval AND search and they compound; a residue (~32% of over-pushes we still miss at d18) is genuine
   eval/judgment, the other ~68% is search *reaching* the refutation. How do we structure the program so
   we STOP attributing-by-facet and just make-a-change → measure END STRENGTH at fixed compute?
2. **⭐ The user's "eval-for-depth" thesis (candidate unifier):** make eval **stronger AND cheaper/faster**
   so it *powers* search — better leaf scoring (the 32% residue) + better ordering/pruning (reach
   refutations shallower, the 68%) + more raw depth (speed). **Judge eval changes by EBF/NPS/depth, NOT by
   static accuracy/cploss** (this retroactively explains npedge: it was an "accuracy" change that RAISED
   EBF +7% = shallower search). Does this correctly re-cast L3 (eval-completeness) and L4 (NPS) into a
   single "eval-serves-search" lane, and should eval-ACCURACY work stay HELD (per the roadmap) until search
   is sound — with the *only* sanctioned eval work being speed/EBF that buys depth?
3. **The collapse was a SYMPTOM, not a campaign.** Given the above, "fix the 22% collapse" was the wrong
   unit of work — it's the visible face of the search-efficiency gap. Should we retire "collapse" as a
   named target and just push search soundness + eval-for-depth, expecting the collapse rate to fall as a
   side effect (measured at the gauntlet)?

## 4. Questions (the durable plan we need)
1. **Re-commit to search-first?** Given the SF11 evidence (we reach at d18 what SF11 reaches at d6) is a
   sharper restatement of your original thesis, do you re-affirm search-soundness as THE lane, collapse as
   a symptom, and eval-accuracy HELD? Any part of the 2026-07-06 ordering you'd now change?
2. **The next lever, concretely.** `statScore-LMR` shipped; `OTV` shelved; `singular` built but NEUTRAL/
   banked (and it did NOT help over-push avoidance). The SF11 gap looks like **selectivity/ordering** (SF
   surfaces refutations shallow). Of the roadmap's remaining pieces — **L2 TT-rebuild → IIR → (singular
   redux?)**, the **ORDERING-CEILING reopen** (our WAC d10 FMC ~87%, you flagged ordering has room again),
   L2b **null-eval-gate / ProbCut / dynamic-null-R** — **what is the single highest-leverage next lever**
   to close a ~3× shallow-refutation-selectivity gap vs a *classical* engine? Does singular being neutral
   change its priority?
3. **What does SF11 do at d6 that we don't at d18?** You have SF sources. Mechanistically, is our 3×
   selectivity deficit most likely (a) move ordering (we don't try the refutation early), (b) extensions
   (SF extends the forcing line so "d6" is effectively deep on it), (c) reductions (we reduce the
   refutation into oblivion), or (d) eval-in-tree (we mis-score the refutation's consequence so it doesn't
   back up)? Which is the highest-EV to attack first, and how would we *measure* which one it is (we have
   `our_depth_avoid.py` for knob A/Bs and SF11 as a yardstick)?
4. **Tooling gap:** the eval-for-depth thesis needs a **fixed-TIME / NPS instrument** (depth-reached at
   fixed movetime); our gates are node_ab (fixed-node, hides speed) and wac (nodes-to-solve, hides speed).
   Is depth-at-fixed-movetime the right screen for eval-speed changes, and any pitfalls (TC-scaling, the
   roadmap's "reductions flip sign with depth" guardrail)?
5. **Pre-NNUE ceiling honesty:** SF18@400-nodes (our gauntlet opponent) uses NNUE. The SEARCH part of the
   gap is classical (SF11 proves it), but is any of the residual collapse a genuine NNUE boundary we should
   stop chasing? Where's the honest pre-NNUE ceiling for this engine, and does the vs_sf11 equal-depth
   re-take (roadmap's tripwire) belong now?
6. **Anything in the roadmap you'd now DELETE or DEMOTE** given 6 days of results (e.g. is the L2
   negamax-lite/TT-rebuild still worth its cost before simpler ordering/pruning wins; is correction-history
   still parked; is `pre_minimizer` A/B still the capstone)?

## 5. Concerns
- We've burned significant effort circling eval↔search; we need a program that is **robust to the coupling**
  (make-and-measure at the gauntlet, not facet-diagnosis) yet still **understandable** (not black-box
  gauntlet-thrashing, which is noisy: baseline swings ±6%/seed).
- Every "cheap" lever we try lately is NEUTRAL (singular, OTV, the eval damps). Is the engine at a genuine
  local optimum where only a **structural** change (TT-rebuild + real TT-move + IIR, or an eval-speed step
  that buys real depth) moves the needle — i.e., stop pre-screening small knobs?

## 6. What we want out
A **re-prioritized long-term roadmap** (supersedes 2026-07-06) with: the confirmed frame, the ordered next
3–5 levers with their gates, the eval lane's new definition (speed/EBF-for-depth vs accuracy-held), the
tooling to build first, and the honest pre-NNUE ceiling. Concrete enough to run without you.

## 7. Repo pointers
- Roadmap: `dev_notes/ROADMAP-2026-07-06.md` + Fable bank (`sf-source-evolution-bank-2026-07-06.md`,
  `search-soundness-fable-q3-2026-07-06.md`, `eval-lane-strategy-fable-q6-2026-07-06.md`,
  `tt-ordering-rebuild-fable-q7q8-2026-07-06.md`).
- This campaign: `dev_notes/collapse-search-efficiency-diagnosis-2026-07-12.md`,
  `dev_notes/npedge-damp-build-2026-07-10.md`. Memory `[[collapse-is-tactical-not-static]]` (search-
  efficiency conclusion), `[[eval-accuracy-payoff-is-pruning]]`, `[[statscore-lmr-shipped]]`,
  `[[singular-banked]]`, `[[external-gauntlet-calibrated]]`.
- Diagnostics (all built this campaign, prompt-free): `diagnose_collapse_moves.py` (classifies losing
  collapse moves), `triage_slice.py` (EVAL/HORIZON/PRUNING), `sf11_overpush_test.py` (SF11 static-vs-search
  + our-static totals), `our_depth_avoid.py` (our per-depth avoidance, knob-A/B-able), `tension_by_stratum.py`,
  `moves_dump.py`+`move_flip_report.py` (move-selection lens). Corpus `diagnostics/collapse_classified.csv`.
- Code: `search_engine.cpp/.h` (search + knobs), `cpp_bitboard.cpp` (eval), `cache_management.h` (TT).
  byte-id 247/41,479,610. SF11 binary wrapped in `diagnostics/eval_vs_sf11.py`.
