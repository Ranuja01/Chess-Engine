# SESSION HANDOFF 2026-09-05 — the EBF gap is measured; the LMR schedule is saturated

## READ FIRST
- ⭐ **THE GAP IS NOW MEASURED, NOT ARGUED.** SF11 reaches **depth 15** on the same 249,014-node budget
  where we reach **depth 10**; it needs 26,265 nodes to our 249,014 for depth 10 (**9.48×**). Venue-matched
  EBF: **ours 1.914 vs SF11 1.452**. ⇒ the equal-node strength gap is **tree size**, not per-node eval
  quality. 📄 [[the-sf11-gap-is-two-thirds-node-efficiency]] · 🧰 `diagnostics/_sf_node_bench.py`
- ☠️ **"Real EBF ≈1.7" IS RETIRED** — that came from WAC totals; venue-matched it is **1.914**. A sixth
  case of WAC misleading on nodes, this time flattering us.
- ☠️ **THE SHIPPED LMR SCHEDULE IS SATURATED.** The log-product is clamped to `rem-2` for every move ≥2 at
  every remaining depth ≥4, so in practice it is a flat *"reduce every late move to 2 plies remaining"* —
  not a log-product schedule at all. 📄 [[lmr-product-schedule-is-fully-saturated-against-the-rem-2-clamp]]
- 📊 **52-config margin sweep: the pruning lane is closed.** `depth@1s` moved for exactly one
  accuracy-preserving config (IIR). Combos were anti-additive without exception.
  📄 [[margin-sweep-52-configs-margins-are-pinned-and-combos-are-antagonistic]]
- ✅ **IIR games: +7.7 ±17.7 over 2,400** (seeds 71/113 at +8.4/+7.0). Sign stable, magnitude unresolvable.

## 1. THE TWO NEW INSTRUMENTS (both reproducible, both new this session)
| tool | what it answers |
|---|---|
| `diagnostics/_sf_node_bench.py --engine sf11 --n 60` | SF's nodes AND depth on the IDENTICAL 60 quiet-corpus FENs (same stratum filter, same seed 1234). Gives nodes-to-depth, the d10→d12 slope, and the depth SF reaches on OUR budget. |
| `diagnostics/_sts_reference.py --engine sf11 [--depth N \| --nodes N]` | SF scored on OUR sts300 with OUR scoring. ⚠️ `--depth` is NOT equal work; `--nodes 249014` is. |
⚠️ **Baselines are now SIX numbers**: WAC solves / WAC nodes / EBF / STS / **quiet nodes 249,014** /
**depth@1s 12**. Quiet d12 = **912,452** (that is where 1.914 comes from).

## 2. THE ANCHOR TABLE
| engine | STS @ d10 | STS @ 249,014 nodes | nodes to d10 | d10→d12 slope |
|---|---|---|---|---|
| SF18 | 2414 (80.5%) | 2605 (86.8%) | — | — |
| **SF11** (HCE ceiling) | **1983 (66.1%)** | **2374 (79.1%)** | **26,265** | **1.452** |
| **OURS** | **1796 (59.9%)** | 1796 (59.9%) | 249,014 | **1.914** |
⇒ equal DEPTH we are 6.2pp behind SF11; equal WORK, **19.2pp**. 🎯 **SF11 is the target, not SF18.**
📐 Time-to-depth decomposes as nodes × NPS: at d10 that is 9.48× × 5.6× ≈ **53× slower**; at d15 ≈ **144×**,
because the EBF term compounds while the NPS term does not.

## 3. WHAT THE MARGIN SWEEP CLOSED (52 configs, zero baseline drift)
- **`depth@1s` is immovable** across the entire pruning space — only `iir` (13) and two over-pruned SEE
  settings that cost 315-483 STS.
- **Futility and razoring are NON-MONOTONIC** ⇒ noise, not trade curves. Only **RFP** behaves properly
  (1000→−13.1%, monotone both directions).
- **Aggressive pruning works and costs exactly what the theory says**: `seep_d7` = −62.6% nodes for
  **−483 STS**. The magnitude is reachable; the accuracy is not. ⇒ margins are pinned by eval noise.
- **Every combo was anti-additive**, including `c_kitchen`, which lost IIR's ply.
- ☠️ **SIX DEAD KNOBS**: `NULLMOVE_PROGRESSIVE`, and `LMR_PRODUCT_K` at 2200/2600/2800/3000/3200.

## 4. ☠️ THE LMR LIVE RANGE — SWEPT, AND THE LANE IS CLOSED
`red_raw(d,m) = (K/100)² · ln(d)·ln(m) / DIV`. Shipped K=2480/DIV=64 gives **≈50.9 plies at d10,m10**
against a clamp of 8. 23-config sweep of the never-tested live range:
| K | quiet nodes | STS | d@1s |
|---|---|---|---|
| 2000 | 249,014 | 1796 | 12 (SATURATED, byte-identical) |
| 1600 | 249,110 | 1831 | 12 |
| 1200 | 240,514 | 1662 | 12 |
| 800 | 317,564 | 1785 | 12 |
| 600 | 399,807 | 1821 | 11 |
| 400 | 693,568 | **1865** | **10** |
✅ Saturation CONFIRMED (K≥2000 inert, K≤1600 live; DIV likewise — 128 inert, 4096 → 924,238).
☠️ **MY RE-SEARCH HYPOTHESIS IS REFUTED.** I predicted a graded schedule might net FEWER nodes because
maximal reduction drives re-searches. The response is **monotonic the other way** — less reduction ⇒ more
nodes, 240K→838K, no over-reduction penalty anywhere.
☠️ **NOTHING BEATS BASELINE ON `depth@1s`** (all ≤12) ⇒ **the flat-maximal `rem-2` reduction sits at or near
the OPTIMUM.** The saturation is an accident in the right place, not a defect. **LANE CLOSED.**
⚠️ **TRAP**: k400's **STS 1865 is the highest ever recorded** — and it is a FIXED-DEPTH ARTIFACT of paying
2.8× nodes while losing TWO plies at equal time. Do not chase it.
🔎 `LMR_SHAPE=0` = 234,591 / 1736 / 12 — FEWER fixed-depth nodes than the schedule that beat it by +20.7 Elo
in games. Fixed-depth nodes do not rank.

## 5. ⚠️ WHAT THIS DOES TO THE +20.7 ELO LMR_SHAPE SHIP
The ship stands — it was decided on 2,800 games. But its **stated mechanism does not**: the header and the
handoff credit "the classical Stockfish log(rem)×log(move) shape," and that shape is never expressed. What
actually shipped was a move from the old additive iteration-depth schedule to a flat maximal `rem-2`
reduction. Second time this ship's explanation has failed (cf. the blunder-avoidance story).

## 6. ▶️ QUEUE
1. **Read the LMR live-range sweep.** If a graded schedule nets fewer nodes or gains a ply, that is the
   first real EBF lever found.
2. ☠️ **`cutNode`/`allNode` — CLOSED 2026-09-05.** Derived via PVS parity (`ENABLE_CUTNODE_PROBE`, ~10 lines,
   byte-identity verified) rather than threading 42 call sites. **AUC vs wrong LMR reductions: `−cut_node`
   0.5678, `−all_node` 0.4792, against `−move_index` 0.8151** ⇒ no signal where it matters; the refactor is
   not justified. A real LMP signal exists (`−all_node` 0.6194) but at ~55% breadth it is the `PROTECT_PV`
   failure shape (+151% nodes at a stronger marker). `window_ORIG` (0.675-0.703) already outscores both.
   📄 [[cutnode-allnode-does-not-separate-wrong-lmr-reductions]]
3. **`lowPlyHistory`** — 📄 `dev_notes/LOWPLYHISTORY-PORT-SPEC-2026-09-04.md`; accuracy-map probe first.
   ⚠️ Needs a per-`go` reset on the GAME path (`clearSearchTables` is diagnostic-only) or it silently
   becomes a second global history.
4. **Eval accuracy → margin re-sweep**, gated COMBINED. This is where the 5 plies live.

## 7. STATE
Nothing committed. Uncommitted: `last_proven`/mode 5 (measured, a NULL — see §13 of the 09-03 handoff),
all gated knobs default-off, `_sf_node_bench.py`, `_sts_reference.py`, `_search_stability.py`.
Sweep scripts and results live in the session scratchpad (deliberately outside OneDrive — sync lag caused
stale reads during the games run).
