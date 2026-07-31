# History-scoring calibration — the scale-sensitive-consumer lever (2026-07-07)

> **RESOLVED 2026-07-08 — MALUS DEAD (5th).** The measure-first gate ran: the Q3 cutoff-calibration logger
> greenlit malus (statScore monotone P(cut) 0.41→0.94; 0-bucket splits by tried-fail count 0.48/0.36/0.29/
> 0.25). But the decoupled-consumer Q-table (`ENABLE_QCUT`, built) then **STS-regressed** (1550→1483→1439
> monotone with λ, no EBF gain) → **falsifies the decouple hypothesis**: root cause is the context-blind
> global `[from][to]` signal itself, not ordering-pollution. **History reduces LESS only.** Decay tuning =
> lowest-value (scoring already well-calibrated). Node-clawback moved to POSITION-SPECIFIC prunes →
> `ebf-node-balancing-campaign-2026-07-08.md`. The candidate ranking below is retained for the record.


## Why this doc exists

Continuous **statScore-LMR** (`history_lmr_delta`, gated `ENABLE_STATSCORE_LMR`) is the first history
consumer whose behavior depends on the **absolute magnitude** of the history tables, not just their rank.
That changes the value of "clean up the history scoring" from housekeeping to a real lever — and it
reopens the gravity/malus/saturation question that failed four times, because the failure was
*consumer-specific* and there is now a *new* consumer with the opposite needs.

## The scale-invariance split (the core framing)

- **Move ordering** (`generateLegalMovesReordered`, `move_gen.h`) reads history for **rank only** →
  scale-INVARIANT. Multiply every history value by 10 and ordering is identical. This is why the
  hand-chosen, un-synced magnitudes never bit ordering.
- **statScore-LMR** compares `statScore` against **absolute thresholds** (`STATSCORE_OFFSET`,
  `STATSCORE_DIVISOR`) → scale-SENSITIVE. This is the first place the ad-hoc magnitudes matter, and why
  the offset/divisor had to be measured empirically (not ported from SF's −4926).

## statScore-LMR results (node_ab vs shipped default, fixed-node, conc3/4)

Unimodal in the divisor at `STATSCORE_OFFSET=512`, `HISTORY_LMR_SCALE=0`:

| divisor | Elo | games | read |
|---|---|---|---|
| 512  | +6.8 ±31  | 662 | too aggressive (touches every move; reduce-more on all zero-history quiets) |
| 768  | +27.1 ±36 | 489 | plateau |
| 1024 | ~+33 pooled (+44.9/350 then +24.6/481) | 831 | plateau |
| 1536 | +6.1 ±40  | 399 | too gentle, rolling off |
| 2048 | ~baseline | —   | roll-off |

Winner: **offset 512, divisor 768–1024 ≈ +30 Elo node_ab**, pending lightning SPRT (the real TC gate —
node_ab refunds the EBF cost, so it flatters reduce-less-style levers). `STATSCORE_KILLER_BONUS=1` added
nothing (+24 vs +27 base = noise) → **pure-continuous is the v1**. Mechanism confirmed: buried-refutation
reduction (`avg-reduction-when-dropped` in the LMR profile) fell 2.22→1.90 ply.

**At the winning `512/1024`:** integer-truncation means a zero-history quiet gets delta 0 (NO reduce-more),
`+1` only at statScore ≥ 1536, `+2` at ≥ 2560 (≈ top 2-3% of quiets by the measured distribution:
median 0, mean 546, P97.5 3072, non-negative). So the shipped-candidate behavior is **"search only the
top few percent most-proven late quiets one ply deeper; leave everything else at base LMR."**

## Exact current scoring (source of truth; anchor to symbols, not lines)

- Base bonus: `b = remaining_depth² × HISTORY_BONUS_SCALE / 100` (scale=100), at every quiet-cutoff site.
- **Unsaturated (default) update:** `historyHeuristics += b`, `counterMoveHeuristics += 4b`,
  `contHist2 += 4b`, `captureHistory += b`. Malus (off) mirrors with the same 4× on continuation.
- **Saturated path (`hist_update`, `ENABLE_HISTORY_SATURATION`) SILENTLY DROPS the 4×:**
  `counterMoveHeuristics` gets plain `b`, `contHist2` gets `b/CONT2_GRAVITY_DIV`. So saturation is NOT
  "the same feature, bounded" — it also **reweights main-vs-continuation**. Any saturation test that
  didn't account for this confounded bounding with reweighting.
- `hist_update` gravity: `h += delta − h·|delta|/MAX_HISTORY` (MAX_HISTORY=16384).
- **Decay:** all tables `>>= DECAY_FACTOR (=1)`, but **cadence differs 16×** — `historyHeuristics` /
  `captureHistory` decay every `DECAY_INTERVAL` (35k nodes, TC-overridden to 125k-200k);
  `counterMoveHeuristics` / `contHist2` every `DECAY_INTERVAL × 16`. `moveFrequency` is the lone odd one
  (`>>= 2`, per-ID-iteration). Killers/counterMoves are cleared per search, not decayed.
- Net: `statScore` at equal weights is dominated **~4:1** by continuation history AND its continuation
  side **persists 16× longer** — an entirely emergent, un-designed weighting.

## Prior gravity/malus/saturation record — 4 tests, never shipped, all regressed

1. **v1 isolation matrix (2026-06-10, uncommitted):** `iso_sat` (saturation alone) **−73**, `iso_mal`
   (malus on simple scale) **−99** (worst). Lightning 3-rep: **ΔSTS ≈ −34** (an earlier +17 was noise;
   STS has a ±30 floor unless `USE_OPENING_BOOK=0`). Diagnosis at the time: "malus has no consumer (we
   lack SF's history-driven pruning)."
2. **Committed dormant (`5385761`, 2026-06-11):** decomposed into `ENABLE_HISTORY_SATURATION` +
   `ENABLE_HISTORY_MALUS` + `MALUS_DIV`/`MAX_HISTORY`/`CONT2_GRAVITY_DIV`/`ENABLE_HISTORY_DECAY`, all
   default-off.
3. **Re-tested WITH the LMP consumer (2026-06-15/16 EBF sprint) → DROPPED.** Still net-negative at every
   `MALUS_DIV` (**STS d2 46.7 / d3 49.5 / d4 47.2 vs 51.5 control**). **This falsified the "no consumer"
   theory.** New root cause: *malus pollutes the global `[side][from][to]` history table → demotes good
   non-cutting quiets everywhere → a pruning-bound (then 92% FMC) + eval-bound search can't absorb it.*
   The aggressive lazy-resort reproduced the exact failure mode; the gentle "nudge, don't reorder" is
   what shipped (+59 Elo).
4. **SPSA re-bench (2026-06-21):** WAC 256, STS 50.7% (−1.5%), −7.3% nodes → "prunes content; net-neg."

Contrast — history-family levers that DID ship: reduce-*more* (`HISTORY_LMR_SCALE=2`, "the find," STS
+131), continuation-aware LMR (`ENABLE_CONT_HIST`), and statScore reduce-less (this campaign). **History
as a REDUCTION signal works; history-table MALUS for ORDERING is what fails.**

## The reframe (only askable now that statScore-LMR exists)

The malus failure is an **ordering-pollution** failure, measured on FMC/STS. But statScore-LMR is a
**different consumer**: scale-sensitive, reads absolute calibration, and specifically suffers because a
never-cut quiet and a never-tried quiet both read `0` — **exactly the gap malus fills.** So the signal
that poisoned ordering may be what the cutoff-calibration consumer wants.

**Central hypothesis to test:** *decouple the consumer* — a separate malus-bearing / saturated statistic
that feeds ONLY statScore-LMR, leaving the ordering history table malus-free. The four prior verdicts do
NOT bind this, because they all measured the ordering table.

Also changed since the failures: FMC moved 92% → ~87% (the 254M→39M pruning evolution), softening the
"can't absorb it" premise.

## Candidates (ranked) and the measurement gap

1. **Malus routed to a cutoff-calibration statistic used only by statScore-LMR** (decouple from ordering).
2. **Saturation as unit-stabilizer** — but isolate bounding from the 4× reweighting it silently applies.
3. **The 4× main-vs-continuation rebalance** — make it a designed ratio (target SF's), not emergent.
4. **Decay-cadence unification** — the 16× continuation spacing; node-budget-dependent → a self-play
   target, not a fixed-depth sweep.

**Measurement gap:** every prior test used STS/WAC (ordering + fixed-depth, ±30 noise) — which measures
ordering quality, NOT cutoff-calibration (how well a history value predicts "this move cuts"). Building a
calibration metric (label each decision `(history_value, did_it_cut)`, measure monotonicity/AUC) is the
prerequisite for tuning the scale-sensitive consumer without re-breaking ordering. This extends the
`shadow`/`statscore_profile` tooling already built.

## Coupling caveat (IMPORTANT)

`STATSCORE_OFFSET` / `STATSCORE_DIVISOR` are fit to the CURRENT history distribution. **Any change to the
history scoring (bonus formula, the 4×, decay cadence, saturation, malus) shifts that distribution and
invalidates the offset/divisor — they must be re-derived** (re-run the `ENABLE_STATSCORE_PROFILE` sweep +
node_ab). Sequencing: ship statScore-LMR on current units first, then treat history-scoring as its own
re-derivation campaign.

## Sequencing vs the roadmap

Not a direct gap-closer (the passer term is). It's compounding infrastructure that sharpens a tool two
consumers now depend on → **parallel/background lane, after statScore-LMR ships, alongside the passer
term** — not ahead of either.
