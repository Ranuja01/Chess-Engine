# Fable question — classical-eval path to SF11 (draft for the user to relay) — 2026-07-04

Framing for Fable: **provenance, not a fence.** These are our findings; challenge them. Everything in
"VERIFIED FACTS" is source- or SPRT-checked, not asserted. We want your strategic read, and we will
source-verify every mechanism you propose before acting (we've caught mechanism errors before).

## CONTEXT (one paragraph)
Non-negamax C++ engine (~2700 Elo), separate minimizer/maximizer, absolute eval (Black-positive, single root
flip), pawn=1000…queen=10000. Yardstick = **Stockfish 11 (classical, pre-NNUE, ~3450 CCRL)**; neutral game
arbiter = SF18 (NNUE). We have a strong HCE + modern search (RFP just shipped +73 Elo, LMR/null/LMP/futility/
razoring/TT/aspiration/qsearch, 1-ply continuation history).

## VERIFIED FACTS (source- or SPRT-checked)
1. **Equal-DEPTH gap to SF11 ≈ 590 Elo** (search-speed removed → this is pure eval + move-ordering quality).
   SF11 is classical, so this gap is classical-eval-vs-classical-eval — NOT an NNUE requirement.
2. **Static-accuracy proxies are strength-blind, sometimes anti-predictive (SPRT-proven tonight):** a placement
   conditioning combo cut static sign-flips vs SF from 65%→50% on a held-out (by-game) split AND held STS/WAC —
   yet lost **~202 Elo** in a lightning SPRT (+47 −178 =25 / 250). STS, WAC, and static-vs-SF sign-flips do NOT
   predict game strength here.
3. The over-read bench we built compared **our STATIC eval to SF's 0.5s SEARCH score** (mis-specified — a
   static-vs-static column existed and wasn't used). "Match SF" therefore meant "drag static toward a deep-search
   verdict on sharp positions," which degrades the static eval's role as a search *guide*.
4. **Placement-magnitude conditioning has intrinsic midgame collateral** — it can't damp the ~6% sharp tail
   without damping the ~94% of ordinary midgame where placement is load-bearing (now ~5 independent failures:
   MOD_PIECES_LEVEL/CONTROL/DEFEND, KING_EG scalar, tonight's combo).
5. **Search knobs are SPSA-flat twice** (θ unmoved from defaults, plus-arm scores ~0.5) → search lane tapped.
6. **Throughput is the constraint:** ~95 min per 200-game lightning SPRT iteration; we cannot run fishtest-scale.
7. Per-term static-eval efforts have been dead ~5× (aggregate-blind to the tail, or collateral, or strength-blind
   proxy). King-safety specifically: our static KS detector fires on real attacks but is primitive vs SF11's
   attack-unit model; cranking it doesn't help and worsens the mean gap.

## THE STRATEGIC QUESTION
Given SF11 proves a **classical** eval can sit ~750 Elo above us, and given our repeated per-term failures look
like **method + throughput** problems (strength-blind proxies, single-term tuning, low game volume) rather than a
low classical ceiling — **what is the highest-leverage path to get "nearby" SF11 with a classical HCE?**

Specifically, rank/challenge these:
1. **Bottleneck attribution:** is our gap dominated by (a) a missing/underdeveloped **term set** (esp. king
   safety, threats, space, imbalance tables), (b) **tuning method** (per-term static fits vs joint game-outcome
   tuning), or (c) **throughput** (can't reach fishtest-scale SPRT volume)? Where's the marginal Elo cheapest?
2. **Import vs hand-roll:** should we port SF11's king-safety attack-unit model (and threat/space tables)
   wholesale and re-tune magnitudes, rather than hand-designing detectors that keep dying at lightning?
3. **Outcome-Texel:** is a joint game-outcome fit (fitted-K, mirrored tables, held-out) of the EXISTING term set
   the right lever, given per-term STATIC fits are dead and aggregate methods were "blind to the collapse tail"?
   Does joint outcome-fitting escape the single-term-deadness trap?
4. **The proxy problem (the real blocker):** we have NO cheap strength-predictive signal — only the SPRT works and
   it's slow. Is there a known cheap proxy that actually predicts classical-eval strength (self-play at very fast
   TC? a large curated SPRT-calibrated position set? ACPL vs SF at fixed nodes?), or is the answer simply "build
   more SPRT throughput first"?
5. **Collapse tail specifically:** given static conditioning keeps failing on the dynamic king-attack tail, is the
   right move a narrow **search-side** guard (e.g. don't grab material that shallow-SEE says loses while our king
   is under net attack), an improved **static KS model** (SF11-style), or is this genuinely the one place NNUE
   would pay first?

We will do the tactical triage ourselves (e.g. whether the specific losing move was a static-leaf over-read vs a
search/horizon miss). We want your read on the STRATEGY: term-set vs method vs throughput, and the cheapest route
to a classical eval that closes a meaningful fraction of the 590.
