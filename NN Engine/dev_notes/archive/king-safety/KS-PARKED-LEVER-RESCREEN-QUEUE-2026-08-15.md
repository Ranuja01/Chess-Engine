# Parked-lever re-screen queue — nulls the contamination may have invalidated (2026-08-15)

**Why this exists.** The diagnostic-harness contamination ([[diagnostic-harness-history-contamination]], fixed
2026-08-14) silently changed the chosen move in in-process search diagnostics (STS/WAC/movematch/regret),
worst at low material. So any lever KILLED by a **pre-2026-08-14 search diagnostic that never reached games**
was killed by an instrument that was lying — that null is INVALID and the lever is re-openable. Items killed in
GAMES/SPRT/gauntlet or by STATIC eval (symmetry/per-term-gap/corpus) are NOT affected and are NOT here.

⚠️ **PROVENANCE / not independently re-verified.** This is a consolidation of a full read of
`OPTIMIZATION_LOG.md` + the SESSION-HANDOFF files; citations (date + knob) are included but each ledger entry
has NOT been independently re-read against the source. Treat as a re-screen worklist, not fact.

⚠️ **DISCIPLINE — these are INVALID NULLS to clear, NOT expected wins.** Given corpus-anti-correlation and the
0-for-9/0-for-13 histories, MOST will re-fail or be neutral on the clean instrument. The value is that we stop
carrying false conclusions — not that 13 hidden wins exist. This queue is LOWER PRIORITY than the KS lane
(`KS-CONTINUITY-RUNPLAN`); run it in leftover engine hours or a separate night. Re-screen on the CLEAN regret
ruler (fixed depth/nodes) first; only a clean-screen survivor earns a games slot.

## The 13 RE-SCREEN levers (ranked by contamination risk)

Risk = margin inside the ~±150 balanced-STS / +101-STS-contamination band AND/OR the contaminated mechanism
(history/ordering tables) directly implicated.

| # | knob | kill signal (instrument) | low-mat/quiet? | note |
|---|---|---|---|---|
| **KS** | | | | |
| 1 | `ENABLE_KS_CHECK_V2` | STS −166/WAC −14, default table, never games | partial (midgame KS) | **ALREADY covered = KS arm S1.** Corroborates S1 is a contamination-suspect re-open. |
| **Eval** | | | | |
| 2 | `ENABLE_CAPG_NET_SELECT`+`_PROMO_CREDIT` | balanced STS −65 | yes (endgame capgain) | inside the noise band |
| 3 | `ENABLE_CAPG_FILE_INVARIANT_TIEBREAK`+`_LVA_STATIC` | balanced STS −116/−153 | **high** | 08-09 handoff itself: "−116 very likely noise" |
| 4 | `ENABLE_WINNABILITY`/`_CLOSEDNESS`/`_ENDGAME_SCALE` | STS −50 (corpus-said-best), games night never run | yes | ⚠ corpus anti-correlated — re-screen on regret/games, not corpus |
| 5 | `PASSER_RFLOOR_R5/R6` | STS 1600 vs 1647 new baseline | yes (passer eg) | ⚠ old-baseline game win likely overlaps shipped de-king |
| 6 | phase-blend (`PHASE_BLEND_LO`/`_RANGE`) | STS −107 alone | med-high | ⚠ needs fix+refit of EG_SUPPORT/EG_LATENT; bare re-screen may still cost |
| **Search** | | | | |
| 7 | `ENABLE_CONT_HIST_2PLY` | STS −88/WAC −4 | **HIGH — history table IS the contaminated mechanism** | ★ top-priority re-screen |
| 8 | `ENABLE_IMPROVING` | STS −37/WAC −11 | **HIGH (history/quiet ordering)** | ★ well inside band; only prior game read is a weak old −5 |
| 9 | `RAZOR_FLOOR=800` | STS plateau instability 600→1495/1000→1481 | **HIGH (root-ordering sensitive)** | ★ chaotic root-ordering metric is exactly what contamination perturbs |
| 10 | `ENABLE_SEE_PRUNE` (quiets) | STS −141 | med | above band; re-test on fixed-nodes regret |
| 11 | `ENABLE_CORR_HIST` qsearch re-site | STS −49 (no-qcache) | med | low — real value is a re-keying job, not bare re-screen |
| 12 | `LMP_MAX_DEPTH=8` | nodes/solves NO-GO d10 AND d12 | med | low — d12 re-confirmation partly survives contamination |
| 13 | `ENABLE_LMR_REMDEPTH` | STS d10/d12 "no plateau" | med | low — partly a structural integer-division artifact |

**★ Highest-value re-screens: 7, 8, 9** — the contaminated mechanism was the history/ordering tables, and these
three key directly on history/root-ordering, so their STS kills are the most likely to be pure artifact. If any
of the 13 flips clean, it's these. (Rows 2, 3 also flagged high-confidence by margin.)

## UNCLEAR — need the raw ledger entry re-read before classifying
- `NULLMOVE_PROGRESSIVE` (OPT-LOG 2026-06-04 S5): parked on a fixed-depth node reading (+4.7%@d10 vs −18%@d12),
  tagged "overnight A/B → NULLMOVE_OVERNIGHT.md" — unclear whether that games A/B ever ran.
- RFP return-blend toward beta `(2*beta+eval)/3` (OPT-LOG 2026-07-26): bench NO-GO but "gated behind corrhist,"
  so the veto may be a dependency, not a contaminated-instrument kill.

## Correctly EXCLUDED (do not re-open)
- KS additive detectors (`KS_SQC/PIN/WEAK_VAL/FLANK`), `KS_ACCUM_MODE`, coord gate — **re-run on the CLEAN
  instrument 2026-08-14**, confirmed harmful/neutral on critical + 0-for-9 games ⇒ STAYS-DEAD.
- `EG_CLAMP_*`, `SEE_PRUNE_CAPTURES`, `ENABLE_HISTORY_MALUS`, `LMR_EXTRA`, `ENABLE_IIR`, `ENABLE_SIMPL_BIAS`,
  the 3-arm search bundle (SPRT −15), all pawn arms — killed/parked in GAMES/gauntlet ⇒ STAYS-DEAD-games.
- Colour-symmetry defects, per-term-gap, raw-corpus fits (−85.6 SPRT) ⇒ STAYS-DEAD-static/games.
- `ENABLE_STATIC_ORDER` — a parked *positive* (+8 STS, free on nodes), not a kill; worth a clean re-screen but
  not in this "invalidated null" queue.
