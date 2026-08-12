# Eval-lane strategy (Fable Q6, 2026-07-06) — eval is BLOCKED by search, not done

Resolves the calibrated-vs-incomplete tension. Companion to [[search-soundness-fable-q3-2026-07-06]] + [[holistic-eval-pivot]].

## The core reframe
"Well-calibrated on average" and "missing terms" describe DIFFERENT moments of the error: calibration = the MEAN at
sites (we're fine); incompleteness = the CONDITIONAL structure — blind spots on position CLASSES (our corr 0.27 KS /
0.23 threats vs SF, while averages look fine). So eval is NOT done — it's incomplete on DYNAMIC classes, and that
incompleteness is currently BLOCKED by the search phantom.

**The blocking mechanism (named + measured):** at sharp positions the phantom line carries +3.9 of fictitious value;
a correct new eval term worth ±0.5p at the leaves CANNOT outvote it → the term's signal is wasted (or adds margin
noise) in exactly the lines search mismanages. Blockage ∝ how much of a term's value sits in SHARP positions → our
dynamic terms keep dying; the one SUBTRACTION-shaped win (capg) survived. = why KS (−33) didn't convert.

## Clean measurement: the 3-rung ladder that brackets the search (all tools we own)
1. **Outcome incremental-validity (Δlogloss)** — ZERO search → pure representation value.
2. **Low-depth WDL-cploss (d2-d4)** — move-choice with reductions barely active (our LMR doesn't bite until depth
   ≥4-5) → almost no phantom exposure.
3. **node_ab** — deployment truth, full search.
DIAGNOSTIC PATTERN: wins rungs 1-2, loses rung 3 = **good-but-blocked-by-search** (confound classified, not suspected).
Loses rung 2 = genuinely not helping move-choice regardless of search. 4th check: recompute rung-3 per-position
contribution EXCLUDING phantom positions (|search−static| > ~1.5p) — if the term's sign FLIPS when phantoms are
excluded, the masking is caught red-handed.

## Eval-term ranking for a DYNAMIC-WEAK engine (+ trap flags)
1. **Threats suite** — best strength/effort + LEAST trap-prone: one-move patterns w/ concrete material stakes →
   search verifies/refutes within 1-2 plies → minimal phantom exposure. Sparse-firing (self-gated, no make-room
   pressure), small tabulated magnitudes, 1-2 days. **TOP — safe to build NOW.**
2. **Mobility-area** — a REDEFINITION of our existing 0.78-corr mobility term, not an addition → ZERO new budget →
   make-room trap CAN'T apply. Modest ceiling, near-zero risk. Safe now.
3. **Initiative/complexity** — sign-preserving damp = SUBTRACTION-shaped = the only eval family that has ever shipped
   for us. Barely search-coupled. Low risk, moderate win. Safe now.
4. **KS final-form** — highest ceiling BUT **will fail the way KS did if built now — because it IS KS** (value
   concentrated in the positions search mismanages; king attacks resolve over many plies = MAX phantom exposure).
   **SEQUENCE AFTER the search campaign.**
5. **Space** — died 3×; only with SF's gates (npm≥thresh, weight²); lowest priority, after everything.

## The −33 KS verdict, re-read: "unadjudicable under the search that tested it" (NOT "failed")
Reordered causes: (a) SEARCH MASKING — now LEADING (named mechanism + magnitude +3.9 vs a fractions-of-a-pawn term);
(b) conditioning mechanics (anti-make-room ridge, hot magnitude) — contributory; (c) KS worthless at our level — now
LEAST likely (incremental-validity was real +0.00095 ~4× its replaced term; SF11 proves the concept converts). ⇒
**DON'T re-litigate, DON'T delete KS.** The consolidated gated KS bundle = a FREE pre-built PROBE: re-run the identical
bundle on node_ab AFTER the search campaign → its delta-vs-today = a direct measurement of how much search was
suppressing the eval lane = **the cleanest coupling number we'll ever get.**

## Sequencing (both lanes, coupled in our favor)
- **MAIN:** search soundness (statScore-LMR + unfreeze ordering → TT-move → singular/multicut) — the phantom is
  measured, large, mechanically understood; nothing else has that combination.
- **PARALLEL eval (NOT blocked):** mobility-area, initiative, threats — screened by the 3-rung ladder, node_ab-gated.
  Safe concurrently (value least in phantom territory).
- **HELD for after the search campaign:** KS final-form, space + the KS-bundle coupling re-test.
- **BACKGROUND:** byte-identical NPS lane (competes with nobody for gate throughput).
- Sounder search → eval converts BETTER (removes override noise at sharp positions + margins/reductions key off
  trusted values so each term does more work/node). = the SAME coupling from the other side ("cleaner eval raises the
  pruning ceiling") — runs BOTH ways; search is the MEASURED/TRACTABLE end right now.

## THE number to re-take when the search campaign lands
**Re-run `vs_sf11` equal-depth.** Part of the −437 is SEARCH; the POST-campaign RESIDUAL is the TRUE eval gap, and it
(not the current −437) sizes any further eval-completeness investment. It's ALSO the honest input to the NNUE go/no-go
tripwire: residual COLLAPSES → classical eval was never as far behind as it looked; residual HOLDS → the remaining gap
is genuinely representational, now measured under a search that can finally express it.
