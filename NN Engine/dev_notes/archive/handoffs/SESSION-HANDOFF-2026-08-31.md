# SESSION HANDOFF 2026-08-31 - PROTECT_KILLERS shipped (+12.4/5276g, 2nd search win); the accuracy-map method; six lanes closed

## READ FIRST
- SHIPPED: `PROTECT_KILLERS=1 PROTECT_MAX_IDX=8` - **+12.4 +/-9.4 Elo / 5,276 games, ALL FIVE SEGMENTS
  POSITIVE.** New baseline **250 / 35,310,778 / EBF 3.784 / STS 1796**. Revert `PROTECT_KILLERS=0` ->
  exactly 249 / 34,362,161 / 3.767 / 1657.
- NEW METHOD: the **accuracy map** (labelled shadow events + AUC ranking). It is what found the ship. See 3.
- HARNESS FIX: `captureHistory` was never cleared by `clearSearchTables` => EVERY bench ever run was
  contaminated. All pre-08-28 fingerprints are void. See 2.
- Six lanes closed with MECHANISMS, not nulls. See 5. Three measurement laws. See 4.

## 1. THE SHIP
`guard_killer` exempts killer/counter-moves from LMR when move index <= `PROTECT_MAX_IDX`. The knobs existed
for weeks, commented "stop reducing the moves most likely to be the critical misses", NEVER enabled.

| axis | before | after |
|---|---|---|
| WAC | 249 | 250 |
| STS | 1657 | **1796 (+139)** |
| nodes | 34,362,161 | 35,310,778 (+2.76%) |
| LMR wrong-reductions | 1.696% | **0.723% (-57%, 5.4 sigma)** |

Games: seeds 71/113/157/199/241 -> +3.6 / +17.4 / +11.9 / +18.3 / +5.5; pooled **+12.4 +/-9.4**, CI [+3.0, +21.8].
Plateau: `PROTECT_MAX_IDX` 4/8/16/64 -> STS 1739/1796/1750/1755, all positive, spread inside the +/-150 floor
=> the MECHANISM carries it, not the constant.

**THE SHIP NEEDED A THIRD EDIT.** `env_flag("PROTECT_KILLERS", false)` passed a HARDCODED false rather than
`Config::PROTECT_KILLERS`, so flipping the header default alone would have been overwritten at init - the
dump would read 0, the fingerprint would not move, and it would look like a no-op.
=> **When shipping a gated knob, verification is INVERTED: prove the fingerprint MOVES.** Grep the `env_*`
registration for hardcoded fallbacks first.

## 2. HARNESS: the captureHistory clear-gap
`clearSearchTables` cleared eight learning tables but not `captureHistory`, which is live => every bench
carried capture ordering across unrelated FENs. Diagnostic-only (games spawn fresh subprocesses, no strength
changed) but it voids the bench record.
**A shared harness defect does NOT cancel in an A/B**: the two arms were biased by DIFFERENT amounts
(-2/-138 vs +6/-74); `LMR_SHAPE`'s fixed-depth case fell from +8 WAC/+92 STS to **+0/+28**.
Also fixed: `prune_collect.py` discarded EVERY position from any .epd corpus and printed "0 searched" with
exit 0 - a total failure that reads as a clean null.

## 3. THE ACCURACY MAP (the method that produced the ship)
`ENABLE_PRUNE_SHADOW` searches what a prune skipped and labels it wrong/right. New `ENABLE_SHADOW_EVENTS`
emits one [SHADOWEV] record per event with every cheap signal available AT THE DECISION POINT;
`diagnostics/_shadow_auc.py` ranks them by AUC against the wrong label.
- LMR: `-move_index` **0.886** · `-rd` 0.798 · **killer_or_cm 0.750** · statScore 0.718 · hist 0.707
- LMP: `-rd` 0.672 · PV-window 0.665 · hist 0.574
- RFP: verified by re-search, **~0.2% wrong (99.8% correct) - do NOT tune RFP_MARGIN.** Futility 0.18%.
**Guard cost scales with BREADTH, benefit with MARKER PRECISION.** Killers = ~3 moves/node => +1.16% nodes;
the same idea blanket (PROTECT_PV at idx 64) => **+151% nodes**. Guard NARROWLY on a high-AUC marker.
`staticPlacementScore` AUC 0.545/0.559 => cheap eval comps do NOT discriminate prunes.

## 4. MEASUREMENT LAWS
- **Node savings below ~35% are Elo-neutral.** -6.14% nodes = 0.11 ply at EBF 1.7 = a fifth of the 0.5-ply
  bar; `CONT_HIST_PIECE_KEY` gamed -0.6 +/-23.1 despite the best bench profile of the arc.
  My +8-9 prediction scaled SEE-captures' +18.4 - but that was a BUNDLE. **Never apportion a bundle's Elo.**
- **The games harness null is +4.6 Elo, not 0**; true candidate-effect sd ~16 Elo vs +/-25/night => one night
  cannot rank two mediocre arms. BUT the +4.6 is **unanswerable** (separating it from 0 needs ~90,000 games)
  => it must NOT be a ship gate, only a magnitude caveat.
- **Only completed segments are readable.** Seg 3 read +0.7 at n=949 and finished +11.9; seg 4 read +50.8 at
  n=62. I called the trend wrong three times inside one run.

## 5. LANES CLOSED
1. KS FLOOR/KNEE/CAP - a REDISCOVERY; the record says the dead quadratic is PROTECTING us (units ~85%
   proximity, counted ~4x). KNEE->40 already failed.
2. KS Stage-3 continuity - `KS_ONSET_MODE=1` games -12.5 +/-23.1; `ENABLE_KS_UNIFIED` +9.3 +/-23.1. Both
   unresolved-null. The margin arithmetic SURVIVES: the KS_FLOOR crossing is a **1260 mp step, 1.3-6.3x
   every futility margin**, undamped for backed attacks.
3. Root razoring on bounds - presearch OFF + root table: razor fires 53,051x (15x) and razors the WINNER
   **83%** of the time for 1.2% nodes. **Fail-low bounds cannot discriminate; root pruning needs SCORES.**
4. Aspiration - the 51% "unhealthy" fail rate is the TRADE-OFF, not a defect: 4x width cuts fails 65% and
   LOSES 7 WAC solves. The counters measure widen-EVENTS, so "51%" was never a per-iteration rate.
5. `LMR_PRODUCT_K` - algebraically identical to DIV (red ~ K^2/div). DIV<=64 is clamp-saturated (32 is
   byte-identical to 64).
6. `RootMove` prev_score ordering - already built (level-2 sort key), among the 14 closed root-table arms,
   and structurally redundant because our presearch re-scores every root move EVERY iteration.

## 6. QUEUE (nothing running; all bench work)
1. **Test the presearch premise's CONCLUSION.** Premise SUPPORTED: presearch-off gives better-calibrated
   history (P(cut)@>=4096 0.895->0.979) and fewer wrong prunes (LMP 2.1 sigma). Conclusion untested: from
   presearch-off (245 / 48.6M) crank back past baseline - **RFP first (0.2% wrong), futility second (0.18%),
   LMP last (0.96%)**. WARNING: `ENABLE_ROOT_PRESEARCH=0` ALONE scores 123/300 - always add
   `ENABLE_ROOT_RAZOR=0`.
2. **ENABLE_NODE_TT** - clean-harness STS +44 at -0.5% nodes, never shipped; the ONLY site writing
   TTEntry::move. Owner correction: we DO have persistent ordering (move-gen cache + LAZY_RESORT top-K
   promote, shipped) => the gap vs the giants is per-node VALUES, not ordering.
3. **The 58-position collapse corpus** (selfplay/games/vssf_2403/collapses.csv) - 29.0% of games; **71% are
   MIDDLEGAME decisions while already winning by 3+ pawns**, only 10% endgame.
4. **CONT_HIST_PIECE_KEY** - Elo-neutral but 268MB->4MB tables, -6.14% nodes, better m0. INTERACTS with the
   ship (eroded its STS gain +139->+41 in the 2x2) => re-measure on the new baseline.
5. **KS detector re-open** - attempt #24 got AUC 0.748->0.810 then failed with a phase SIGN-FLIP
   (+0.37/+0.27/-0.32). The prescribed fix (phase-conditioning) was never written.

## 7. Housekeeping
Committed this session: the ship, shadow-event instrumentation, CONT_HIST_PIECE_KEY (gated off),
CAPTURE_HIST_VICTIM (gated off, dominated), five new diagnostics.
KING_SAFETY_MODEL.md still stale. Repo lives inside OneDrive - 237k CPU-seconds of syncing over 15 days;
a file count in selfplay/games timed out after 2 minutes.
