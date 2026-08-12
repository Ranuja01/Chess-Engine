# Fable FOLLOW-UP (last chance) — pipeline-frame acceptance + 3 sharp questions before we build (2026-07-12)

We accept the refutation-pipeline frame (stages 2 ordering + 4 leaf-visibility broken; AND-structure explains
the neutral knob table; retire "collapse" to a KPI; static-magnitude HELD, leaf-visibility ACTIVE; 3-arm pair
test with kill criterion; Lane 2 NPS always-on; Lane 3 TT-rebuild→TT-move→IIR→singular-redux). We'll run your
two conviction measurements (refutation-rank at cache-hit nodes; 2-4-ply-forward our-vs-SF11 static) BEFORE
building. Three questions where your judgment is uniquely useful and time-sensitive:

## 1. The stage-4 MEDICINE (biggest build risk). Your fix is `ENABLE_THREATS` (`get_static_threats_score`),
but as built it is: (a) **symmetric** — `threats_by(Black) − threats_by(White)`, so it credits OUR threats
too; in an over-push WE are the nominal aggressor, so it may partly REINFORCE the push, not damp it; (b)
contains a **currently-hanging bonus** (`nd==0 || na>nd`) = VOLATILE, violating your own volatility rule; (c)
per our standing method rule, **SF11's "Threats" ≠ ours** — your 2-4-ply test proving SF11-static registers
the refutation's damage does NOT prove OUR term registers it. So even if stage 4 is convicted, `ENABLE_THREATS`
may be the wrong medicine. Should we (i) activate as-is and trust the move-selection lens to catch a bad sign,
(ii) first RESHAPE it to opponent-side-only + standing-underdefended-only (drop our-side credit AND the
volatile hanging term) so it's a pure "opponent has a real threat against us" leaf signal, or (iii) build a
different minimal leaf-visibility term? What is the SMALLEST leaf term that makes a d6 leaf register "the
refutation's consequence" without the symmetry/volatility failure modes — and should we validate the fix by
re-running the 2-4-ply-forward static comparison with OUR-threats-ON (does it move OUR static toward SF11's),
not just SF11 vs baseline?

## 2. ORDERING unfreeze — standalone, or does it need the TT rebuild first? You scheduled TT-rebuild in Lane 3
but ordering-unfreeze (tail re-sort on cache hits) as Lane 1a — ahead of it. Can the per-iteration tail re-sort
be cleanly prototyped on the CURRENT frozen-move-gen-cache architecture (re-score the untried quiet tail from
live history, once per ID iteration, evidence-prefix pinned) as an honest A/B, or does a meaningful ordering
fix actually REQUIRE the verified TT-move (Lane 3) to seed the re-sort — making "Lane 1 before Lane 3" a
sequencing trap? If the tail re-sort is genuinely standalone, what is the tightest scope that avoids the
prior full-re-sort STS regression (you said quiet-tail-only + evidence-prefix-pinned + once-per-iteration —
confirm that's the whole guardrail)?

## 3. ATTRIBUTION + contingency in the 3-arm test.
(a) If the two conviction tests SPLIT — e.g. refutation-rank is already early (stage 2 NOT broken) but the
leaf-visibility test convicts stage 4 (or vice versa) — does the pair test collapse to the single convicted
arm, or do you still want the pair (in case the offline test under-reads a stage)?
(b) Threats-on changes the eval, which changes pruning AND ordering — so 1b is not eval-only; it perturbs the
same search 1a touches. How do we keep 1a and 1b ATTRIBUTABLE in the 3-arm gauntlet (1a / 1b / 1a+1b), and is
there a confound that would make a super-additive pair look real when it's just 1b's search side-effects?
(c) Kill criterion: if BOTH stages look clean offline AND the gauntlet pair is still neutral, what's the
honest next diagnosis — straight to the vs_sf11 equal-depth retake + ceiling discussion, or is there a stage
(1/3/5) we've under-weighted?

## 4. (optional) The "inject the refutation at index 0" ceiling test — worth the instrumented build to get the
ordering lane's MAX prize in effective plies before committing to 1a, or is the rank distribution alone enough
to size it?

Repo/context: `dev_notes/fable-question-roadmap-reassess-2026-07-12.md` (the reassessment you answered),
`dev_notes/collapse-search-efficiency-diagnosis-2026-07-12.md`, `diagnostics/our_depth_avoid.py` /
`sf11_overpush_test.py` (we'll extend the latter to N-ply-forward), `get_static_threats_score` (cpp_bitboard.cpp:5137).
