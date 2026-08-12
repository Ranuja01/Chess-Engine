# Session handoff — 2026-08-04 (night B): the threats ship, and a corpus that changed the answer

**Read this top block first.** Covers the overnight block AFTER `SESSION-HANDOFF-2026-08-04.md` (which
documented the previous night). Prior context: `-08-03.md` (compensation source read), `-07-31.md`.

---

## ⚡ STATE — A NEW DEFAULT IS SHIPPED (uncommitted)
```
ENABLE_THREATS=1  SCALE_THREATS=75  THREATS_STANDING_ONLY=1  THREAT_PER_TARGET_CAP=800
```
| | WAC | nodes | EBF | STS d10 | STS d12 | NPS (med-3) |
|---|---|---|---|---|---|---|
| **NEW DEFAULT** | **250** | **35,791,173** | **3.804** | **1685** | **1674** | **446,218** |
| previous (`ENABLE_THREATS=0`) | 246 | 35,089,668 | 3.846 | 1746 | 1751 | 458,988 |

✅ Defaults reproduce the tested config **to the node**; ✅ revert verified exact. **Nothing is committed.**
🏆 **+45.0 ±40.6 Elo, +171 −121 =96 over 388 games, SPRT H1 accepted.**

## 🚨🚨 THE BENCH WAS WRONG IN BOTH REGIMES — AND THE ARM STILL WON
STS fell **1746→1685 (d10)** *and* **1751→1674 (d12)**. Earlier that day I used exactly this d12 signature to
kill `thr_so_75` and wrote that "the second regime resolved it". **That reasoning is now falsified**: a
d12-confirmed STS loss belongs to a +45 Elo arm. ⇒ **The second-regime check establishes REPRODUCIBILITY,
not correctness.** Several arms killed on STS alone this week are now actively suspect.
⚠️ **Do not "fix" the STS drop** — it is concentrated in the `sts_guard` tier, which is what the term trades.

## 📉 IT DOES NOT REDUCE COLLAPSES — the gain is ordinary play
Four 200-game `vs_sf` mines (pre-ship ×2, shipped ×2; the shipped pair used elo 2400 and **2401** so they
land in separate families):
| | score /200 | collapses | points forfeited |
|---|---|---|---|
| pre-ship | 89.0 · 90.4 | 69 · 58 | 61.5 · 48.0 |
| **shipped** | **95.6 · 103.0** | **66 · 61** | **58.0 · 49.5** |
| avg | 44.85% | 63.5 vs **63.5** | 54.75 vs **53.75** |
★★ **Score +4.8pp, collapse burden IDENTICAL across 800 games.** ⇒ ~50-60 points per 200g are still thrown
away from winning positions and **that lane is completely untouched by this ship.**
⚠️ This contradicts the corpus profile, which showed the gain in the worst-scored decile ⇒ **"corpus worst
decile" is NOT the same population as "games we collapse in".**

## 🎯 ABSOLUTE AXIS — real, but opponent-strength dependent
Fixed-node gauntlet vs SF18, 400 games per point (2 seeds), our side capped at 250,000 nodes:
| SF18 nodes | pre-ship | shipped | Δ |
|---|---|---|---|
| 400 | 55.5% | **64.35%** | **+8.85pp (~2.5σ)** |
| 600 | 47.75% | **47.75%** | 0.00 |
⇒ **parity moved ~540 → ~573 nodes.** ★★ **It converts advantages better; it does NOT close the gap where
SF genuinely outplays us.** 🚨 I read n600 alone first and wrote "worth nothing absolutely" — wrong.
**One node point is not the anchor; the curve is.** Seeds swing 5.5pp (at n600 they simply swapped).

## 🚨🚨🚨 THE BIG ONE — THE CORPUS DECIDED THE OPTIMUM
Rebuilt `diverse_corpus_wide` (2076 → **2713 rows**; general tiers **29% → 56%**) and re-ran the same grid:
| arm | old corpus | rebuilt |
|---|---|---|
| **75 / cap800 (shipped)** | 263.920 | **279.510 ← best** |
| 110 · 125 · 140 | 263.310 · **262.932 ← best** · 263.267 | 280.129 · 280.966 · 282.007 |
**The ranking inverted.** The "2-D interior optimum at scale 125" — bracketed on BOTH axes, smooth
curvature, best of ~40 arms — was an artifact of a corpus that under-weighted general play. **An SPRT on it
was queued for tonight and would have chased noise.** Owner's insistence on folding general positions in
BEFORE tuning is what caught it.
▶️ **Re-derive every optimum after a corpus change. Val is not comparable across corpora. CURVATURE IS NOT
PROOF.** ⇒ `s125_cap800` is DEAD; tomorrow's challenger set must be re-derived from scratch.

## 🐛 DATA LOSS — `build_position_bank.py` truncated the bank
It opened its output with `"w"` and never read the existing file, **despite a docstring promising
append-only accumulation**. Every run replaced the bank; I destroyed the accumulated `sf18` labels this way.
✅ **FIXED**: merges by FEN with **existing rows winning**, `.tmp` + `os.replace`, and reports
`existing + new = total (n already SF18-labeled)`.
✅ Recovered by re-mining: bank is now **4,987 rows, 2,525 SF18-labeled**, single-vintage against the shipped
engine. Only the SF18 compute had to be re-paid. `diverse_corpus_wide.csv` was untouched, which is the only
reason the day's results stayed reproducible.
☠️ **Snapshot before regenerating** — `diverse_corpus_wide_preship.csv` and
`collapse_dataset_classified_{s1,s2_preship}.csv` are saved.

## ✅ MECHANISM CONFIRMED — the cap bounds the STACK, not just SAFE_PAWN
`THREAT_SAFE_PAWN` promoted to a knob to deconfound. With the cap on, SAFE_PAWN 800 ≡ 1600 **byte-identical**
(the cap already clamps it). But **no uncapped SAFE_PAWN reduction reaches the capped accuracy** — even 400
uncapped reads 264.068 vs capped 263.920. ⇒ it also bounds minor+rook+king combinations. No wholesale
re-weighting of the threat constants needed.
✅ Also measured: **hanging ≈87% a subset of capture_gains** (14/16 co-occur, r=0.60) ⇒ `STANDING_ONLY`
belongs on whenever threats are on.

## ▶️ FIRST ACTIONS
1. **Decide the commit** — the ship plus a large diagnostics/tooling delta is uncommitted.
2. **Re-derive the challenger set on the rebuilt corpus.** Nothing from the old grids survives.
3. **Passers: constants are closed** (5 mechanisms dead, inputs all load-bearing on two corpora). The live
   lead is structural — blockade/contest ignore what the blockade COSTS the defender
   ([[passer-defect-is-blockade-cost-blindness]]); `ai.passer_records()` exists to verify a fix on w1.
4. **Threats: 6 structural gaps still unbuilt** (pawn targets, `ThreatByPawnPush`, `RestrictedPiece`, queen
   threats, `stronglyProtected`, pawn-defended pieces). The cap bounds a term still missing half its cases.
⚖️ **Only spend games on changes expected to exceed the ~20-40 Elo measurement floor.** Constant variants sit
under it (an A−B of +11 needed ~4000 games) — that is why `s125` vs shipped was never gamed.
