# Session handoff — 2026-08-13: the signed-accumulator step-1 result, an integrity catch, and a reframe

> ## 🚨 READ THIS BLOCK FIRST
>
> **Headline: the signed-accumulator "when" object is BUILT + byte-id-clean, but on CLEAN data it produces NO
> robust move-regret gain — neutral on v2 (+0.007), a regression on main (+0.106), and every component flips sign
> across the two cross-sets (the classic KS inconsistency). The exciting "big opening fix / machinery robust"
> result from earlier in the night was a HARNESS-CONTAMINATION ARTIFACT (concurrent phase-splits sharing /tmp
> files); I caught it via a shared-position-count anomaly, fixed it (PID-unique paths), and re-ran everything
> clean, ALONE. Nothing shipped, nothing committed. Default build byte-identical (`250 / 35,426,396 / EBF 3.800`).
> My honest lean: fork 3 — the accum joins the 0-for-9 KS history; bank the diagnosis and SPRT the +15 bundle.**
>
> ### The three real deliverables of the night
> 1. **An integrity catch + fix.** `_ks_phase_split.py` wrote workers to fixed `/tmp/_ps_{base,cand}_{i}.csv`;
>    ANY two concurrent phase-splits clobbered each other. Tell: a shared-position count halving (11936→7732) +
>    nonsense deltas. Fixed to PID-unique paths. **All concurrent phase-split numbers from the night are void.**
> 2. **A clean, trustworthy (if negative) result** for the accum at the trace-derived magnitudes (below).
> 3. **Grounding + infrastructure**: two comprehensive KS maps (code inventory + failure trajectory), the design
>    spec, the accum implementation (gated, byte-id), the composition trace, the whacky cross-set (6000 rows),
>    and two auto-approved diagnostic runner subs (`unit_trace`, `ks_phase`).

## CLEAN RESULTS (each run ALONE, no concurrency) — base = the defaware bundle
| config | set | opening | midgame | endgame | ALL |
|---|---|---|---|---|---|
| step-1 accum `ATTCOUNT=0 NQ=10 THRESH=10 LIN=96` | v2   | +0.050 | +0.043 | −0.127 | **+0.007** |
| step-1 accum (same) | main | −0.016 | +0.177 | +0.110 | **+0.106** |
| proximity-removal alone `ATTCOUNT=0`, no accum | main | +0.106 | +0.180 | +0.010 | **+0.119** |
| proximity-removal alone | v2 | −0.113 | +0.103 | −0.449 | **−0.082** |

**Decomposition — NOTHING is robust (everything flips sign across sets):**
- proximity-removal alone: v2 **−0.082** (better) vs main **+0.119** (worse) — opposite signs.
- accum machinery effect (accum − proxctl): v2 **+0.089 (HURTS)** vs main −0.013 (negligible) — inconsistent.
- accum net: v2 +0.007 (neutral), main +0.106 (worse) — **neutral-to-worse on both; no improvement anywhere.**
(The contaminated data had faked a −0.143 machinery win + a −0.28 opening — both void.)

## THE DIAGNOSIS — two layers
**(a) The magnitudes are under-powered.** Accum danger = `(net − THRESH)·LIN/16 = (net−10)·6`. With `ATTCOUNT=0`
proximity→0, units drop ~4; the old table gives ≈`6·units − 36`. At units~14: accum 24 vs old 48 — roughly **half
to a third** the danger ⇒ the accum UNDER-READS genuine danger. A magnitude rebalance (keep some proximity / lower
THRESH / raise LIN) is one open fork. The object runs correctly; the starting constants remove too much.

**(b) The deeper, more important finding — the opening over-read may not be a MOVE-REGRET lever.** Even where the
accum clearly cuts opening KS danger, opening regret does NOT improve (neutral both sets). The accum CHANGES ~45%
of opening moves, but the changes WASH (regret-neutral), while the midgame changes net-HURT (under-reading real
danger). So: reducing opening KS danger reshuffles opening moves without net benefit, and costs real midgame/
endgame danger reads. This aligns with [[most-eval-error-is-move-neutral]]: the opening KS over-read is real in
the EVAL but is largely regret-neutral on MOVES. If so, the months-long "fix the opening over-fire" target is not
where deployment strength is won, and no magnitude tweak changes that.

## THE FORK FOR THE MORNING (owner's call — I did NOT autonomously tune after the contamination scare)
1. **Magnitude rebalance** — re-derive the accum constants so danger magnitude matches the old path (keep partial
   proximity, restore the map gain), then re-test the WHOLE object clean. Tests whether (a) alone explains it.
2. **Confront the regret-neutral-opening question** — if the opening over-read doesn't move regret, the accum's
   value (if any) is in the MIDGAME/ENDGAME danger *quality*, not the opening gate. Re-target: does the object
   improve midgame/endgame move-regret when it does NOT sacrifice danger magnitude? That's the real strength lever.
3. **Bank + step back** — the additive/curve/detector/coordination/accum lines have all now failed on clean
   move-regret; the channel law + [[most-eval-error-is-move-neutral]] may be telling us KS is not a move-regret
   lane at all, and the +15 OvD/central/defaware bundle is the deliverable to SPRT.

## WHAT IS DURABLE (not contaminated)
- `dev_notes/KS-CODE-INVENTORY-2026-08-13.md` + `KS-FAILURE-TRAJECTORY-2026-08-13.md` (the two fable maps).
- `dev_notes/KS-SIGNED-ACCUMULATOR-DESIGN-2026-08-13.md` (design + code-inventory corrections).
- The accum code (`cpp_bitboard.cpp` king_safety_danger, gated `KS_ACCUM_MODE`; byte-id at default) + knobs
  (`KS_ACCUM_MODE/NQ_SUP/WIN_SUP/ACCUM_THRESH/ACCUM_LIN/ACCUM_SQUARE/ACCUM_DIV`, `KS_COORD_GATE_MODE/DIVISOR`).
- Composition trace: on danger-kings proximity ~23% / discriminating(weak+safe) ~27% / 88% have a weak square.
- Runner subs `unit_trace`, `ks_phase` (auto-approved); PID-fixed `_ks_phase_split.py`.
- Whacky cross-set `ks_sets/variant_regret_set.csv` (6000 rows) — NOT yet used (accum wasn't worth a transfer test).

## WHAT WAS DISCARDED (contaminated by /tmp collision — DO NOT cite)
brwp73218, bpf8pmzfu, b6a7fads5, the step-2 weak-val runs (bbh01gzs8 + killed bt77hqvcn), and any coord-sweep
(bubuoa159) cell that overlapped a concurrent run. The clean re-runs (bx6t0uiqg, b6qjpwb3m, b4g7t3xwv, bka4svlik)
supersede them.

## DISCIPLINE NOTES
- One phase-split at a time historically; the PID fix now makes concurrency safe but VERIFY shared-count stays full.
- The night's lesson restated: I over-read a contaminated headline as a breakthrough; the shared-position-count
  anomaly + the control + the clean re-run are what caught it. Always re-run a surprising win ALONE before believing it.
