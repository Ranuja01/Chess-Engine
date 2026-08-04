# Session handoff — overnight block of 2026-08-03/04 (two clean kills, two new instruments)

**Read this top block first.** Prior: `SESSION-HANDOFF-2026-08-03.md` (the compensation source-read and the
R-floor kill), then `SESSION-HANDOFF-2026-07-31.md` (the ship), then `OPTIMIZATION_LOG.md`.

---

## STATE — the engine is untouched
Default build is still the shipped **`246 / 35,089,668 / EBF 3.846 / STS 1746`** (`c2d4259`, +36.7 Elo).
**Nothing was committed, no default was flipped, nothing is running.** Every result below is env-knob or
diagnostics-side. The night produced **no shippable candidate** — by design: both candidate lanes were
tested to destruction instead.

## ✅ NEW INSTRUMENTS (both durable, both now in the register)
1. **NPS median-of-3 = 458,988** (456,361 / 458,988 / 461,341; 1.1% spread; all three byte-identical at
   `246 / 35,089,668`). ~2% slower than the retired default's ~468k — noted, not actionable.
2. **d12 companion fingerprint: `261 / 104,930,403 / EBF 3.606 / STS 1751`.**
   ★★ **Two extra ply buy +15 WAC but only +5 STS.** Depth buys TACTICS and almost nothing POSITIONAL ⇒ our
   STS deficit is **eval-limited, not depth-limited** — independent instrument support for the eval lane,
   from a different direction than the 6-for-6 vs 0-for-13 game record.
   ⚠️ Corollary: **d10 STS is blind to a depth-sensitive change** — re-read such candidates at d12.
   ★ 35.1M → 104.9M nodes over 2 ply = **1.73×/ply**, reconfirming REAL EBF ≈ 1.7 (not the printed 3.8).
3. **ABSOLUTE ANCHOR (new venue result):** at `NODE_LIMIT=250,000`, 400 games per point (2 seeds each):
   **n400 = 55.5%** (seeds 55.5/55.5) and **n600 = 47.75%** (seeds 50.5/45.0)
   ⇒ **PARITY ≈540 SF18 nodes ≈ 460× their nodes for the same result.**
   This is our only machine-independent progress axis. ▶️ Future builds: seeds 0 and 1 at n600, pool,
   compare to 47.75% (±2.5pp). ⚠️ Seeds agreed exactly at n400 but differed 5.5pp at n600 — sampling, not
   structure; **do not assume any node point is "the stable one".**

## ☠️ KILL 1 — THREATS ARE NOT A MAGNITUDE PROBLEM
`fit_bench_guarded GRID=threats` on `diverse_corpus_wide.csv` (held-out split). Baseline val 269.088.
- **`THREATS_STANDING_ONLY` beats full threats at EVERY scale** (so_50 266.26 · so_75 267.47 · so_100 270.24
  vs 25→269.47 · 50→269.47 · 75→271.76 · 100→277.26).
- **Corpus loss grows MONOTONICALLY with magnitude** — the term gets *worse* the more of it you apply.
- Benches: `thr_so_50` −104 STS · `thr_25` −146 · **`thr_so_75` −25**.
- ★★ **`thr_so_75` looked like noise at d10 (−25 is inside the ~100 jaggedness band). The SECOND REGIME
  resolved it: at d12 it reads 1674 vs baseline 1751 = −77.** Both regimes negative, deeper one more so.
  ⇒ genuinely harmful; the strict `STS_TOL=0` guard was right and my "within noise" read was wrong.

⇒ **Scaling a term that is missing half its cases just amplifies the half it has.** Ours lacks pawn targets,
`ThreatByPawnPush`, `RestrictedPiece`, queen-threat terms, and it skips pawn-defended pieces where SF still
pays `ThreatByMinor`. **Re-enabling threats means porting FORM, not turning a dial.**

## ☠️ KILL 2 — PASSERS ARE NOT A CONSTANTS PROBLEM
`fit_bench_guarded GRID=passer` on `passer_fit.csv` (288 rows). Baseline val 539.272. Six arms beat baseline
on held-out corpus; **all four benched arms REJECT**:

| arm | corpus val | STS | WAC |
|---|---|---|---|
| `mag_up` (MAG_SCALE 115) | **533.99** best | 1611 (−135) | 246 |
| `contest_hard` (STOP 120 / PATH 55) | 534.70 | 1632 (−114) | 241 |
| `rcap_up_mag` (R_CAP 384 + MAG 110) | 535.33 | 1571 (−175) | 242 |
| `king_strong` (KING_FAR 24 / HELP 9) | 538.37 | 1709 (−37) | 243 |

plus `contest_soft` −207 and `contest_stoponly` −134 from the same session.
★★ **The corpus asks for MORE passer value; the bench refuses every way of giving it.** With the dead
R-floor that is **~20 arms in every available direction, all negative** ⇒ the channel sits at a sharp local
optimum and the real miss (w1 reading −0.01 where SF reads −1.26) **is not reachable by its constants.**

⇒ **Both channels converge on the same verdict from opposite directions: FORM, not MAGNITUDE.**
▶️ **Do not re-propose a constants grid for either without a new mechanism.** The dead `rfloor` grid is kept
in `fit_bench_guarded.py` as the record of why.

## ✅ THE LEVERAGE MAP REPLICATES — with error bars
Second independent 200-game `vs_sf 2400` sample on the same build (58.9 min):

| | sample 1 | sample 2 | |
|---|---|---|---|
| our score | 89.0/200 | 90.4/200 | ✅ |
| collapses / points | 69 / 61.5 | 58 / 48.0 | ≈1 Poisson sd |
| positional | 63% | **69%** | ✅ |
| king safety | 34% | **26%** | ⚠️ **8pp swing** |
| opening+early-mid | 69% | **67%** | ✅ |
| pts/collapse | ~0.9 flat | 0.80–0.91 flat | ✅ frequency, not severity |
| endgame | 8% | **9%** | ✅ verification set, NOT a target |
| ply 40–79 | target | **62% of lost points** | ✅ |

⇒ The map's shape is real. ⚠️ **But class shares carry ±6-8pp — never target one off a single sample**, and
the small buckets (`material` 5%, `ks_and_material` 3%) are 2-3 collapses and mean nothing individually.

## 🔬 EVAL-AT-RESOLUTION — clean, but scoped
`--eval` on the shipped build: **lost branches 4/4 AGREE** with SF (so a big negative is the search
*faithfully scoring a genuinely lost line it was wrongly forced into* — the misread is the FORCING);
**3/4 won lines are forced mates**, which a static eval cannot represent ⇒ tool's own verdict is
TACTICAL/search. ⚠️ **The frozen set is WAC-only**, i.e. exactly the regime where we already know depth pays
(+15 WAC / +5 STS). **It cannot speak to positional collapses (69% of lost points) — do not cite it as
"our eval is fine."** ▶️ `--generate` more bases from positional collapse FENs (`ks_class=positional`,
npm≥35, ply 40-79); **ADD bases, never regenerate**, or past runs stop being comparable.

## 🐛 TOOL FIXES / TRAPS FOUND TONIGHT
1. **A corpus path with a space becomes a degenerate tie.** `CORPUS=/mnt/.../Chess Engine/...` split on the
   space, resolved to nothing, and printed **9e9 for every candidate** — which reads like a ranking.
   ✅ `fit_bench_guarded.py` now takes a **bare basename** resolved under `ks_sets/` and **hard-exits if the
   corpus is missing**. ★ **A fit that returns an identical number for every arm is a BUG REPORT, not a result.**
2. ☠️ **`vs_sf` tags on ELO ONLY** (`--tag vssf_${elo}`) ⇒ a second run at the same elo **overwrites
   `collapses.csv` AND the PGNs**, so the usual "re-score from PGNs any time" escape does not apply, and
   `collect_collapses.py` then silently replaces that family's rows. ✅ New `_snapshot_classified.py
   SUFFIX=s1` preserves the pooled classified CSV first (sample 1 saved, 1.69 MB).
   ▶️ **Give a second sample its own tag.**
3. 🚨 **The gauntlet seed moves a 200-game point 5.5pp** (50.5% → 45.0%, same build/nodes/openings).
   ▶️ **≥2 seeds per node point, quote the pooled number**; a 400-game pooled point still carries ≈2.5pp se.

## ▶️ PROPOSED NEXT (nothing started — owner's call)
1. **Decide the commit.** Uncommitted and all diagnostics-side except one gated knob: `PIECEVAL_RECOMPUTE_LATE`
   (default false, byte-identity verified), the `fit_bench_guarded` corpus/argv fixes + three grids,
   `_snapshot_classified.py`, `DIAGNOSTICS-TOOLKIT.md`, the passer-map delta header, ~9 probes.
2. **The KS smooth-form redesign** — now the only live eval lane the evidence supports: one gate + a smooth
   polynomial, blend the OUTPUT, replacing the clamp stack. Both constant-sweep lanes are closed and KS is
   the remaining structural candidate.
3. **Extend the eval-resolution set with positional bases** (overnight job; re-pays SF's deep search).
4. **Threats/passers structural port + refit** if the KS work stalls — form first, constants second.

🚨 Unchanged and governing: `evaluate_passers` → `priced_passer[]` → capgains → the material accumulator →
**`MOD_KS_REALIZ`, which is shipped and load-bearing.** Any passer/material change has a live path to the
+36.7 ship. Per-class guard + games are mandatory.
