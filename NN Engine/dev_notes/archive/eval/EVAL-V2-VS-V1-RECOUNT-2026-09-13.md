# Eval v2 vs v1 — full recount, 2026-09-13 (end of day)

@author: Ranuja Pinnaduwage (maintained with Claude)

★ **What v2 has, what v1 has, and where every v1 concept is going — plus the current benches side by side.**
v1's live/dead status below was read from the `search_engine.h` DEFAULTS, not inferred from `EvalBreakdown` field
names (several fields are gated off, and several live terms hide inside `pieces`).

---

## 1. INVENTORY — concept by concept

| concept | v1 (shipped eval) | v2 (shipped config) | v2 plan |
|---|---|---|---|
| material | flat `values[]` + `EG_EXIST_*` piece tapers + `piece_value_boost` (`PV_BOOST_MAG=10000`) | ✅ flat values, pawn 1000 both phases | taper UNDECIDED; `PV_BOOST` ☠️ not ported (removal improved v1 6/6) |
| placement (PST) | ✅ PSTs + per-piece clamps (`MG_CLAMP_*`) | ✅ PSTs (rung 0), no clamps | — |
| phase | 3-way boolean `phase_score`, 25 reachable values | ✅ continuous `phase256`, limits LOCKED 61700/15800 | — |
| king safety | ✅ `KING_SAFETY_MAG=3000`, replaces latent threat; stm-asymmetric safe checks | ✅ SF-shaped zone + x-ray, Hill curve (rung 1, **+101 Elo**) | shelter → later |
| pawn structure | ✅ chain / wall / doubled live; isolated & backward penalties **0**; `PAWN_CLAMP` | ✅ doubled + isolated + backward, no clamp (rung 2a) | connected/support ☠️ parked |
| passers | ✅ V3: multiplicative realizability `R` + clamp chain | ✅ additive rank table, king distance, candidates (rung 2b) | — |
| draw detection | ✅ `is_practically_drawn`, 10 cases — **5 flag forced wins (10–28%)** | ✅ `draw_class`, 8 oracle-clean cases, **ON** | lone-pawn cases → KPK bitbase (slice 4) |
| heat map / attacking layer | ✅ live (`SCALE_ATTACK_LAYER=100`) | ☠️ deliberately not ported | decomposed into dedicated subsystems |
| mobility | full mobility **OFF**; ✅ cheap bishop colour complex + cheap rook mobility live | — | **slice 2 ▶️ NEXT** |
| outposts | **OFF** (`OUTPOST_*=0`) | — | slice 2 |
| rook files / 7th / connected | ✅ live (`ROOK_*`) | — | slice 2 |
| central | ✅ live (bounded) | — | slice 3 |
| space | **OFF** (`SPACE_MAG=0`) | — | slice 3 |
| threats | ✅ live (`SCALE_THREATS=75`) | — | slice 3 |
| Kaufman imbalance + pairs | ✅ live (Kaufman owns the pairs) | — | slice 3 |
| convertibility scale | built, **OFF** (reverted `ab070b7`, an unresolved null) | — | slice 4 (eg-leg form) |
| mate drive / king races | ✅ `advanced_endgame_eval` | — | slice 4 |
| winnability | **OFF** | — | slice 4 |
| capture gains | ✅ live | — | slice 5 (= the whole variant gap) |
| OvD | ✅ live | — | slice 5 |
| corrhist (search) | **OFF** | — | slice 5 |
| tempo | none | ☠️ parked at 0 | re-test after mobility + at margin re-sweep |
| pawn majority | **OFF** | — | not planned |
| colour symmetry | 11 defects fixed, ~1.4% residual | ✅ **exact 0.000** mirror and tempo swing | gate every component |

**Count:** v1 runs roughly **18 live term families** (plus several gated-off ones). v2 runs **5 subsystems** — material +
placement, king safety, pawn structure, passers, draw classifier.

---

## 2. BENCHES — current build, 2026-09-13

### STS (fixed depth) and WAC (d10)
| | STS | WAC solved | WAC nodes | EBF |
|---|---|---|---|---|
| **v1** (arm 0) | **1796** | 250 | 35,310,778 | 3.784 |
| v2 rung 0 (material + PST) | 1364 | — | — | — |
| v2 rung 1 (+ king safety) | 1480 | — | — | — |
| **v2 rung 2, shipped config** (+ pawns + draw classifier) | **1698** | 246 | 63,221,361 | 4.087 |
| (draw classifier OFF) | 1698 | 246 | 63,216,318 | 4.087 |

☠️ WAC does NOT discriminate strength (v1 beats SF on it) — it is a fingerprint here, not a ranking. STS floor ±150 arm-vs-arm.

### §I eval accuracy vs SF18 — 6 corpora, 2,500 rows each, baseline = v1 (negative = more accurate than v1)
| corpus | v1 MSE | v2 rung 0 | **v2 rung 2** | rung 2 + draw |
|---|---|---|---|---|
| self-play primary | 470.10 | −16.06% | **−14.95%** | −14.95% |
| self-play v2 | 440.80 | −12.61% | **−13.85%** | −13.85% |
| self-play x4 | 403.92 | +2.56% | **−3.86%** | −3.86% |
| UHO openings | 209.75 | +23.51% | **+12.50%** | +12.50% |
| variant / 960 | 272.49 | +138.33% | **+138.69%** | +138.69% |
| KS-critical (lichess) | 2163.24 | −7.92% | **−9.81%** | −9.81% |
| **mean / worst** | | +21.30% / +138.33% | **+18.12% / +138.69%** | same |

### Games
Rung 1 vs rung 0 **+101 Elo** · rung 2 vs rung 1 **+60.4 ±25.5**. ⚠️ **v2 vs v1 in games has never been run** — and is
not worth running yet at a 98-STS deficit.

### Not measured tonight
NPS for either arm (timed; the owner's evening games would distort it). Queue a `wac_speed` pair for a quiet window.

---

## 3. WHAT THE NUMBERS SAY

1. ★ **On standard chess, five v2 subsystems are MORE accurate than v1's ~18 live terms.** Excluding the variant set, rung 2 is
   **~6% more accurate than v1 on average** and better on 4 of 5 standard corpora. Only UHO (+12.5%) is worse.
2. ☠️ **Yet v1 is 98 STS points stronger.** Accuracy is not strength ([[corpus-fit-is-anti-correlated-with-elo]]). Two known
   mechanisms: v2 still lacks the middlegame channels that decide move choice (mobility, central, space, threats), and every
   search margin (`RFP_MARGIN`, `DELTA_MARGIN`, futility) is fitted to **v1's** eval spread, which costs v2 until the margin
   re-sweep. v2 also searches **~79% more nodes** at d10 — thinner eval, weaker pruning.
3. ☠️ **The variant/960 gap did not move at rung 2** (+138.3% → +138.7%). Recorded earlier as ONE term, **capture gains**
   (+176.89%) — so it closes at slice 5, not before. Expect this column to stay bad through slices 2–4.
4. ✅ **Rung 2 improved the three hardest-to-fix columns**: x4 (+2.6 → −3.9), UHO (+23.5 → +12.5), KS-critical (−7.9 → −9.8).
5. ✅ **The draw classifier is free on every instrument** — identical §I on all six corpora, identical STS and WAC solves,
   +0.008% nodes. As designed, it fires on essentially no positions in midgame-heavy corpora; its value is correctness in
   rare endings, not bench movement.
