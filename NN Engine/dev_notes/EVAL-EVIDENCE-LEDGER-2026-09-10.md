# Eval evidence ledger — 2026-09-10

**Every gated eval arm we own, re-measured against PAIRED nulls on three corpora.** This is the input to
any rebuild decision: what has evidence, what is carried on fitted constants, and what overlaps what.

⚠️ **Method** — all readings are `_paired_null.py`: the arm read against three neutral arms
(`ASPIRATION_DELTA=300/800`, `EVAL_NOISE_SIGMA=30`) **on the FENs where both changed the move**, per corpus.
The previous screen (09-08) compared each candidate's global win% to a null measured on a *different*
population — candidates flip 21-27% of moves, neutral arms 35-38% — which is why several verdicts moved.
See `INSTRUMENT-MAP.md` §F2/§F3.
📏 Corpora: `game_regret_set` (15,000) · `_v2` (11,940) · `_x4` (14,713, built 09-09). ⚠️ Not equivalent
draws: x4 is 5pp more opening / 6pp less endgame than primary.
★ **Rule, pre-registered before any number was seen: nothing counts unless the sign is consistent on ALL
THREE corpora.** 13 arms x 3 corpora will throw a 2σ cell by chance.

---

## ✅ SURVIVORS — sign-consistent across all three corpora
| arm | knob | primary | v2 | x4 | pooled | note |
|---|---|---|---|---|---|---|
| `thr_corner` | `THREAT_MINOR_ON_DEFENDED=1 THREAT_SAFE_PAWN_REQUIRE_SAFE=1` | +1.7 | +1.7 | +0.5 | **+1.3** | built 09-09 from SF11's source |
| `mobility` | `ENABLE_MOBILITY=1` | +1.6 | +0.8 | +1.1 | **+1.2** | ☠️ was **"50.0% NULL (2732/2729)"** on 09-08 |
| `thr_hanging` | `THREATS_STANDING_ONLY=0` | +0.9 | +0.8 | +0.5 | **+0.7** | ☠️ was **"49.9% NULL"** on 09-08 |

⚠️ All ~1σ. None clears the bar. Two of three were **recorded null against the wrong comparator and the
correction went UPWARD** — the concrete payoff of the 09-09 instrument finding.

## ❌ INCONSISTENT — null (positive on some corpora, negative on others)
| arm | knob | primary | v2 | x4 |
|---|---|---|---|---|
| `thr_minor_def` | `THREAT_MINOR_ON_DEFENDED=1` | +2.0 | −0.1 | +0.0 |
| `thr_safepawn` | `THREAT_SAFE_PAWN_REQUIRE_SAFE=1` | +1.9 | +0.0 | −2.2 |
| `thr_att2` | `THREAT_ATT2_PROTECT=1` | +1.7 | +0.9 | −0.8 |
| `ks_mob_edge` | `KS_MOB_EDGE=64` | +1.0 | −0.6 | +0.7 |
| `central_50` | `SCALE_CENTRAL=50` | +1.7 | −0.4 | +1.0 |
| `ovd_off` | `OVD_CAP=0` | +1.2 | −1.5 | +0.8 |
| `heat_150` | `SCALE_ATTACK_LAYER=150` | +0.7 | −0.6 | +0.3 |
| `ks_zone_off` | `KS_ZONE_ATTACK_PCT=0` | +1.9 | −0.0 | +0.6 |
| `pieceval_late` | `PIECEVAL_RECOMPUTE_LATE=1` | +1.4 | −0.9 | +0.0 |
★ The last two are the historical *"cleared primary then died on v2"* pair. **Still inconsistent under
paired nulls** ⇒ the correction does NOT rescue everything; it moved four arms, two up and two nowhere.

## 🛑 VETO — consistently negative (the direction this instrument reads reliably)
| arm | knob | primary | v2 | x4 |
|---|---|---|---|---|
| `central_bnd2` | `CENTRAL_BOUNDED_MODE=2` | −0.0 | −0.8 | −0.3 |

## ☠️ COMBINATIONS — the refutation
| arm | primary | v2 | x4 | pooled |
|---|---|---|---|---|
| `thr_3way` (corner+hanging) | +1.2 | −0.2 | −0.4 | ~+0.2 |
| **`bundle3` (corner+hanging+mobility)** | +1.3 | −0.1 | +0.2 | **+0.47** |

**The bundle is BELOW every component alone** and loses the sign-consistency each had.
| corpus | union of the 3 changed sets | `bundle3` changed | **cancelled** |
|---|---|---|---|
| primary | 8,427 | 6,326 | **25%** |
| v2 | 6,791 | 4,968 | **27%** |
| x4 | 8,302 | 6,088 | **27%** |
⇒ ~26% of each component's move changes are **restored to the base move** when combined, replicated 3/3.

## 🔗 OVERLAP MATRIX (changed-set intersection)
| pair | overlap |
|---|---|
| A ∩ B (inside `thr_corner`) | **62-64% of B** |
| `corner` ∩ `hanging` | **51-55% of hanging** |
| `corner` ∩ `mobility` | **49-51% of mobility** |
| `hanging` ∩ `mobility` | **60-62% of mobility** |
⇒ **Nothing here is disjoint.** That is the mechanism behind the cancellation, and it is a property of
~30 terms carrying ~2 signals — not of this particular trio.

## 🔥 SEPARATELY: the KS ablation (09-09)
`KING_SAFETY_MAG=0` (removes the dedicated attack-unit danger term, NOT the subsystem):
aggregate ~0 on all three. Crowded-board (≥26 pieces) **+2.2 / +2.9 / −1.8**, pooled **+0.83 ± 1.09** ⇒
null. The magnitude ladder (0/1500/3000/4500) was an **artifact** of the global-null comparison.
⚠️ The gate measures the MEAN; KS's signature is a TAIL (#1 over-read on 92 `ks_attack` collapses). "KS is
worthless" does **not** follow.

---

## ▶️ WHAT THIS MEANS FOR A REBUILD
1. **Term-at-a-time is closed by arithmetic** (no term is worth ~1/3 of a whole-eval upgrade) and
   **bundling — the only stated escape — is now refuted empirically.** There is no path to a candidate by
   assembling what exists.
2. ★ **A term can become load-bearing BY ACCIDENT.** Every constant was fitted while its neighbours were
   wrong, so removing a bad term measures worse than keeping it — `a-correctness-fix-into-absorbed-tuning-is-not-free`
   twice over. **Rebuilding is the only way to break that**: nothing inherits a constant fitted around it.
3. ✅ **A rebuild is the right SIZE for our instruments.** We cannot resolve 1pp; the SF15c oracle read a
   whole-eval difference at **+7pp on the first attempt**. Rebuilt-vs-current is that kind of object.
4. ⚠️ **MUST SURVIVE** — measured load-bearing, not assumed: the attackingLayer heat map (**1.3pp**,
   channels NOT collinear, magnitude at its optimum) · the shipped bundles `MOD_KS_REALIZ` (+36.7),
   threats (+45), OvD/central (+20.8), de-king (~+50).
5. **Best single leads to carry in**: `thr_corner` and `mobility` (~+1.2-1.3pp each, sign-consistent).
   ⚠️ They do not compose with each other in the CURRENT eval — untested in a clean one.
