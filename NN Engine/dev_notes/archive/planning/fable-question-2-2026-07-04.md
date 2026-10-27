# Fable question #2 (draft to relay) — the COMPASS problem + optimal Elo-growth allocation — 2026-07-04

Frame: provenance, not a fence — challenge these. All "VERIFIED" items are gate/SPRT/source-checked. We have a
few days of Fable access left; this is the deepest strategic fork we can't self-resolve. Don't spend it on
anything we can read from our own source/data.

## VERIFIED FACTS (this session)
1. **We built + CALIBRATED a fixed-node paired self-play gate** (deterministic node budget → clock jitter removed).
   Anchored both ends: accurate (reproduced a known −202 Elo change as −194±58) AND unbiased (0.0±43 at a true-0
   change). ~3.6× faster than lightning. We finally have a cheap, trustworthy strength ruler.
2. **Outcome-Texel retune of the LIVE eval scalars is TAPPED.** Fresh 183k-position, by-GAME-holdout corpus,
   fitted K=2.60, joint fit of the live positional family: control result-loss 0.13201→0.13198 (0.02% = FLAT).
   Well-powered, not underpowered. Re-weighting the features we have does not improve outcome prediction.
3. **Gap-map (our per-term breakdown vs SF11's), by CORRELATION = "do we track the feature's variation":**
   material 0.61, space 0.60, pawns 0.58 (fine); **king safety 0.27, threats 0.23 (we barely track SF); mobility
   0.78 but ~6× under-scaled.** Our two DYNAMIC functions (KS, threats) are "malformed" — they compute values that
   don't track SF's; static features are fine (consistent with the flat retune). **STRATIFIED by position type
   (sanity-checked):** KS corr is WORST in the midgame decision tail (near-equal, 0.22) and we compute ~0 where SF
   sees the big attacks that decided games (midgame-decided: SF 0.109 vs ours 0.001); KS corr is 0.00 in endgames
   where both read ~0 (correct — kings safe → the tool shows NO false gap). So the KS gap is real, midgame-
   concentrated, and behaves sensibly — but it is still an SF-AGREEMENT measurement (the crux).
4. **THE RECURRING RESULT — SF-agreement ≠ strength.** A placement change that cut our static sign-flips-vs-SF
   65→50% (held-out) and held STS/WAC LOST ~200 Elo in games. KS changes that improved KS-vs-SF died at lightning
   ~5×. Our diagnostic compass (SF per-term agreement) has repeatedly not predicted game strength.
5. SF11 (classical, ~3450) is ~750 Elo above us WITH a hand-crafted eval → the representation ceiling is real and
   classical-reachable; not an NNUE requirement in principle. We have SF11's KS algorithm documented, so we CAN
   rebuild our KS function to track it.

## THE CRUX (what we can't self-answer)
Our strongest current lead — "fix the malformed KS/threats functions to track SF" — is derived from SF-agreement,
the compass that FACT 4 says doesn't predict strength. So:

1. **Is "low correlation with SF's KS" (fact 3) a REAL Elo lever, or a compass artifact?** We suspect a
   distinction: MAGNITUDE-agreement with SF (proven dead — scale confound + search covers tactical terms) vs
   REPRESENTATION-TRACKING (corr — whether our function responds to the same board features SF's does; KS=0.27 =
   we essentially don't). Are these genuinely different, such that fixing representation-tracking could convert to
   Elo where magnitude-matching didn't? Or is corr-with-SF just as unreliable a compass?
2. **Should we re-point the WHOLE diagnostic apparatus off SF-agreement and onto OUTCOME-predictiveness?** i.e.
   instead of "where do we differ from SF's breakdown", ask "which added/fixed feature most improves GAME-RESULT
   prediction on the by-game corpus" — using SF only as a hypothesis source, never a target. Is there a rigorous
   way to do feature-gap mapping against outcomes rather than against SF, given the fixed-node gate for final
   truth? (Fact 2 says the outcome objective is flat for reweighting — but is it flat for NEW/fixed features?)
3. **Effort allocation for Elo over the next few weeks** — rank: (a) rebuild the malformed KS/threats functions to
   SF's form, validated on the fixed-node gate + a longer-TC venue (KS pays off with depth); (b) the mobility
   mis-scale (tracks SF at 0.78 — cheap, but flat-mobility was dead in games → conditioned?); (c) start the NNUE
   pipeline now (the dynamic-eval representation gap is the canonical NNUE win, and our by-game corpus is the seed);
   (d) something we're not seeing. Where is the marginal Elo cheapest, and what's the sequence?

We'll self-do: the sharpness-stratified gap-map (does the KS gap widen in the decision tail?), diffing our
`king_safety_danger` against SF's documented algorithm, and any fixed-node A/Bs. We want your read on the COMPASS
and the ALLOCATION — the parts that need judgment beyond our source.
