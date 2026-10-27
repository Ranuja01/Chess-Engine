# Parked discussion — depth, positional eval, and the road past ~2700

> **⚠️ DEPTH-LABEL CONVENTION CHANGED 2026-06-03.** `MAX_DEPTH` is now **literal** (`MAX_DEPTH=10` searches to depth 10); a historical `MAX_DEPTH=11` ≡ today's `MAX_DEPTH=10`. The "d10"/depth labels in this file are unchanged.

Captured 2026-06. **As of 2026-06-04 this is now the ACTIVE strategic direction** (see the MEASURED note).

## MEASURED (2026-06-04) — no longer theory: our EBF ≈ 3.4
STANDARD self-play (5 games, 323 searched moves): **mean depth ~13.4 @ 21s/move, ~10.2M nodes/move → EBF ≈
3.4** (`10M^(1/13)`). Histogram: mode d12–13, tail to d16–17, rare d20–21 (opening ~d12.4, midgame ~d14.1).
So the "EBF dominates" point below is **confirmed for us**: at EBF 3.4 we get d13 in ~10M nodes; reaching d20
in the *same* node budget needs EBF ~2.2 — a ~7-ply gap that **no raw-speed win closes** (10% PGO ≈ +0.07
ply; EBF 3.4→2.5 ≈ +3.5 plies). ⟹ the lever for ~3000 at the current system level is **EBF (move ordering +
pruning) + eval** (eval feeds ordering → cutoffs → lower EBF); PGO is the finisher; NNUE/SMP the deliberate
last-step multipliers. **DIAGNOSTIC IN FLIGHT** to call ordering-vs-pruning: behavior-neutral `g_fh_total`/
`g_fh_first` first-move-cutoff counters + a `[search]` stderr line. **Fork: <~85% first-move-cutoff =
ORDERING-bound; ~90%+ = PRUNING-bound.** Details in `HANDOFF.md` / `OPTIMIZATION_LOG.md`.

## Why Stockfish reaches depth ~30 and we reach ~12–20 (both C++)

It is **not one secret** — it's a compounding stack, and the dominant factor is **search
selectivity, not raw speed.** Our own framing already half-said it: we prune *less*
(wide-but-shallow), so we win at a fixed shallow depth but lose in real time where SF reaches
~25+.

1. **Effective branching factor (EBF) — dominates.** Depth is exponential in EBF. SF runs
   EBF ~1.5–2 (reduces/prunes most moves hard); a less-tuned engine sits ~3–5. At EBF 2 you
   reach ~depth 25 in the nodes EBF 3 reaches ~16. SF's low EBF = decades-tuned LMR (a
   reduction *table* over depth×movecount×signals) + aggressive-but-sound null/futility/
   razoring/SEE/LMP + **rich move ordering** (history, *continuation history* = move-pair
   history, capture history) so first-move cutoffs hit ~90%+, which is what makes the
   aggressive reductions safe. Ours: LMR was buggy (now fixed), no continuation history,
   pruning less tuned → higher EBF → less depth. **This is most of the gap.**
2. **Raw nps — secondary (~2–5×).** We do ~650–700k nps; classical SF ~2–3M+, NNUE SF ~1M+.
   Only ~1–2 plies, not the whole gap. Our nps sinks: a **heavy, recomputed-every-node eval**
   (every feature added slows the node) and **`BoardState` copies into `state_history`** vs
   SF's in-place do/undo.

### Concrete, findable gaps in OUR code (highest-leverage first)
- **Aspiration windows — DONE, VALIDATED, SHIPPED & live by default** (`ASPIRATION_DELTA=500`,
  `HONEST_ROOT_TT=true`; was commented out at `get_engine_move`). Narrow window centred on the prev
  eval + incremental widening. The blocker that corrupted the prior attempt was NOT the windowing —
  it was 4 root/preliminary TT stores hardcoding `TTFlag::EXACT` (valid only under the infinite
  window), fixed by `HONEST_ROOT_TT`. Equal-clock LIGHTNING: +4 solves / +0.32 depth (median 9→10);
  STS +2.2pp; −18% nodes @d10. DELTA confirmed 500 at LIGHTNING (vs 800/1200). See `OPTIMIZATION_LOG.md`.

### Parked research — context-adaptive search control (the unifying idea; gated on self-play)
The aspiration DELTA turned out to be a **tactical↔positional knob**, not just speed: at LIGHTNING,
wider (1200) got +7 WAC and fewer window-fails but ~−3pp STS; narrower (500) led STS. (Caveat: the
deterministic d10 sweep had 500/800/1200 *tied* on STS, and the LIGHTNING STS drop was internally
inconsistent with its depth/WAC rise — so the split is partly noise. 500 kept as the positional-best,
efficiency-sweet-spot default.) This opens a broader idea: let a **volatility / context signal
modulate many knobs at once** (LMR reduction, futility/razoring margins, null-move, extensions,
aspiration delta) instead of fixed constants — how strong engines actually work. Three hard-won
guardrails before building any of it:
  1. **Local, not global.** Volatility is per-node — a quiet position hides one sharp line. Use cheap
     local signals (in-check, capture, eval-trend, q-search activity, history), not a global
     position score (which is also the "can't measure it cheaply" problem).
  2. **Safe direction only.** Our documented weakness is LMR **over-pruning quiet winning moves** (the
     exact thing VERIFY fixes). So "prune MORE when quiet" points *backwards* and risks re-introducing
     those misses; the LMR profiler showed dropped moves are indistinguishable from noise by local
     signals. Use volatility to prune **less / extend more in sharp nodes**, never more in quiet ones.
  3. **Lowest-risk concrete candidate = the "improving" heuristic** (is the static eval higher than 2
     plies ago? — one bit of state, near-free, standard). We almost certainly don't have it; it
     modulates LMR/futility in the safe direction. Check first.
  Adaptive-delta variant: carry the *initial* delta across iterations by recent fail rate (we now have
  the `[aspiration]` counter) — auto-widens in volatile positions without phase detection. All of this
  is subtle pruning tuning in an order-sensitive engine whose regressions fixed-depth WAC can't catch
  → **the real arbiter is self-play.**
- **Continuation history** (move-pair ordering) — biggest modern EBF lever after plain history.
- **Staged / lazy move generation** — captures first, generate quiets only if no cutoff (we
  score all moves up front; ~90% of nodes cut off early).
- **Lazy eval** — cheap terms first, bail before expensive terms when clearly out of window;
  directly attacks the heavy-eval nps drain.
- **Incremental eval / state** (the parked **E3** speed item) — do/undo, incremental accumulator.

### The part you can't copy
SF's pruning/reduction *constants* come from **Fishtest** (millions of games per patch over
~15 years). Adopt the *structure* (formula shapes, techniques), not the magic numbers — they'd
need re-tuning on our engine. Learn from the **Chess Programming Wiki** (clearer than SF source)
+ a readable engine (Ethereal/Berserk) for code.

## The core tension
"Add more positional eval features" makes each node smarter but **slower → less depth.**
"Reach SF-like depth" needs a **leaner, more selective** search. With a handcrafted eval you
trade along that frontier every time. Two escapes: keep eval lean + maximize selectivity/nps
(classical SF), or **NNUE** (a learned eval that is rich *and* fast — incremental, vectorized).
NNUE is the only thing that gives positional depth *and* search depth at once.

## Three distinct bets (compete for time; complementary)
- **(a) Cheap positional features** — improves per-node quality, costs a little depth. Needs an
  **STS baseline** to justify each term. The "timeless features" path (below).
- **(b) Depth / selectivity** — aspiration windows, continuation history, staged movegen,
  lazy/incremental eval. This is literally "be more like SF's search" and most directly closes
  the "we plateau at 12–20" gap. (= the parked speed track at scale.)
- **(c) NNUE** — the both-at-once; biggest project (data, integration, keep it fast). The repo's
  latent NN path. Separate discussion.

## Positional "timeless features" direction
Philosophy (validated by the existing attack-pressure/defensive-resilience term that took us
1600→2000): a **cheap static signal that front-loads what the search would otherwise need depth
to discover** (king-attack buildup is the marquee example). Candidate gaps, by value/cost:
**mobility / piece activity** (possibly under-weighted), **pawn-structure quality** (passed/
isolated/doubled/backward, chains, majorities), **space**, **outposts**, **good/bad bishop**
(extends existing color-complexity), **rook activity** (7th rank; have open files). None
trustable without the STS baseline + eventually self-play, and each will **churn** (order
sensitivity) — judge on net.

## Measurement we still lack
- **Positional baseline:** Strategic Test Suite (STS1–STS15, scored 0–10 per position; freely
  downloadable; CPW links it). Doubles as the **quiet-game validation** for the check extension.
- **Self-play games** (extension-on vs off, etc.) = the real strength arbiter; needs cache
  isolation (separate processes — caches are shared globals). The gold standard for "helps a
  real gamer." **BUILT** — `NN Engine/selfplay/` (process-per-side; SF-cp arbiter + PGN; tournament
  + post-hoc analyzer). See memory `[[selfplay-harness]]`.
- **NEW eval-track signal from self-play (2026-06-04):** with the Stockfish-cp arbiter on, the engine
  reads **systematically hotter than SF** (e.g. −23 vs −10 when lost, +0.8 vs 0.0 when ~equal) and
  notably **overestimates R+B-vs-connected-passers endings** (held ~+5 on one that drifted to −0.9 as
  it concretized; both engines shared the illusion). Smells like overvaluing an extra minor vs
  connected passers + king activity, and/or a small trade-eval lean. The post-hoc `annotate.py`
  `analysis.csv` (cp-loss / max engine-vs-SF divergence per game) is the tool to localize it — the
  first concrete eval-calibration lead the harness surfaced.
