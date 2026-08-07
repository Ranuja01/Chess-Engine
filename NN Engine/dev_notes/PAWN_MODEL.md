# PAWN_MODEL — the pawn valuation system: mapping, evidence, and what would falsify it

**Canonical document.** Read this before changing anything in the pawn evaluation, and **update it if you
change the system** — a stale model doc is worse than none. Sibling of `OPTIMIZATION_LOG.md`: both hold
knowledge that does not expire. A falsified experiment stays falsified; a measured pawn value stays measured.

Sections 2-8 are timeless. Section 1b is explicitly dated state and must be refreshed when the engine changes.

---

## 1. Scope and provenance

Answers: *what is a pawn worth, and what makes one better or worse than another?*

| instrument | what it measures | trust |
|---|---|---|
| `pawn_marginal_real.py` | marginal pawn value on **real corpus positions**, ours vs SF18 search | **highest** — the anchor |
| `pawn_truth_generator.py` + `_analyze.py` | same, on **manufactured** positions with controlled contexts | direction only; **magnitudes proved wrong by 2.6×** |
| `pawn_truth_ours.py` | per-cell residual `SF18 − ours` on identical FENs (no double-counting) | high, inherits its input's trust |
| `pawn_gap_attribution.py` | which eval TERM carries a gap | high |
| `passer_detector_diff.py` | our passed-pawn predicate vs SF15.1's, counted | exact (pure predicate comparison) |
| `_ks_fit_eval.py` / `ks_fit_wholesystem.py` | **win%-error vs SF18 across corpora — the tuning objective** | the criterion |
| `ai.pawn_clamp_records()` | per-pawn clamp headroom | exact |

Units: engine is **millipawns**, `values[PAWN] = 1000`, so **1 cp = 10 engine units**.
Eval is **absolute Black-positive**; White-POV cp = `-ev / 10`.

---

## 1b. Current system at a glance — **DATED 2026-08-05**

- Two evaluators, `evaluate_pawns_midgame` / `evaluate_pawns_endgame`, phase-blended. Independent tables are
  therefore possible per phase (SF must encode `S(mg, eg)` pairs; we do not).
- **Midgame**: rank tables `default_/passed_midgame_pawn_rank_bonus`, structural `pawn_wall_file_bonus`
  (phalanx) and `pawn_chain_file_bonus` (support), both **file-indexed, no rank term**. Clamped
  `min(225, structural + positional)`.
- **Endgame**: `endgame_pawn_rank_bonus`; structural bonuses are **hardcoded literals** (`100` phalanx /
  `135` support / `115` defending / `50` latent) that **no knob reaches** — `SCALE_PAWN_WALL`/`_CHAIN` only
  rebuild the file tables the midgame path reads. Clamped `min(175, structural)`.
- **Passers**: `getPPIncrement` flags them; `evaluate_passers` is the sole payer under `ENABLE_PASSER_V3`
  (`mag × R/256` + king race). `R` is `clamp(0, 384)` then `min(R, PASSER_R_CAP=320)`.
- **Tunable as of 2026-08-06**: whole-table `SCALE_*` · per-rank `RANK_DEF/PSD/EG_R2..R7` · per-file
  `CHAIN_F_A..H`, `WALL_F_A..H` · `ISOLATED_PAWN_PEN`, `BACKWARD_PAWN_PEN` (default 0) · both caps
  `PAWN_CLAMP_MID/EG` · endgame structural `EG_PHALANX/SUPPORT/DEFEND/LATENT` · per-phase structural rank
  curves `STRUCT_R_MG/EG_R2..R7` · `STRUCT_OPPOSED_MG/EG_PCT` · `PASSER_R_MAX` ·
  `ENABLE_PASSER_DEFER_ON_FLAG` · the six `PP_*` · the passed-pawn support magnitudes
  `PPS_OWN_BLOCK/ENEMY_BLOCK/OWN_ATTACK/ENEMY_ATTACK` · passer core.
- **Gated off**: `ENABLE_PAWN_OBSTRUCTION_BLEND` (measured harmful — see §8).
- **Still hardcoded / NOT reachable**: SF's `WeakUnopposed`, `BlockedPawn[]`, phalanx-count modulation of
  the connected bonus, and the king-support term inside `advanced_endgame_eval`.
- **Corpus**: `diverse_corpus_wide.csv` rebuilt to **4,987 rows** (was 2,713) after labelling the bank to
  4,925/4,987. ⚠️ `val` is therefore NOT comparable with anything measured before 2026-08-06.
- Default fingerprint `250 / 35,791,173 / EBF 3.804`.

## 1c. Fitted results by iteration (DATED 2026-08-06)
| iteration | corpus | val | games |
|---|---|---|---|
| 1 — rank/file tables fitted | 2,713 | 279.51 → 265.53 | **−15** |
| 2 — + endgame surface, `opposed`, caps | 2,713 | → 262.10 | **~+4** |
| `ISOLATED`/`BACKWARD` alone | — | — | **+12.4 ±30.2** (700g) |
| 3 — shipped regime, + `PPS_*`, widened corpus | **4,987** | **234.88 → 213.03** | **~+18 @ 389g, unsettled** |
⚠️ Iteration 3's SPRT was stopped mid-flight for a context transfer and was volatile when stopped
(checks: −6, +6, +4, +2, +2, +8, +18). **+18 is a local upswing, not a converged estimate.**
★ Iteration 3 is the first fitted **in the shipped regime** (earlier ones optimised inside
`ENABLE_KS_CHECK_V2=1`, which is not the default, so their winners were never directly applicable).
★ **`PPS_OWN_ATTACK 60→35` was the single largest move of iteration 3** — the passed-pawn support
magnitudes had never been tuned at all.
★★ Consistent across all three: **pawn values want to come DOWN nearly everywhere, except rank 7 which
wants UP.** `PAWN_CLAMP_MID` has walked 225 → 175 → **140**.

---

## 2. Principles (each with its measured magnitude)

**P1 — ⚠️ REVISED 2026-08-06. The pawn LEVEL error is real but it is not the useful framing; the
piece/pawn RATIO is.**
Original claim: on 1,621 real-position removals, ours +58 cp of positional value per pawn vs SF18 −6 cp,
near-uniform across ranks. That measurement stands *against SF18* — a later one-sample run put our global
eval scale at k ≈ 1.0 versus SF18, so the two are on a comparable ruler.

But three things reframe it:
1. ☠️ **Absolute marginal values are NOT comparable across engines in general.** SF11 prices a pawn at 120
   and SF15.1-classical at 65 on the *same* positions — nearly 2× apart, purely internal scaling. Only the
   RATIO is scale-invariant. (I briefly retracted P1 entirely on these grounds, then partially reinstated
   it once ours-vs-SF18 was measured at k ≈ 1.0. It has moved twice; treat the *ratio* as load-bearing and
   the *level* as secondary.)
2. ✅ **The scale-invariant defect, one sample, full ladder:** rook is exact (ours **4.98** pawns vs SF18
   5.01 / SF11 5.02 / SF15.1c 5.09) while **minors sit ~7% below the classical consensus** (N ours 3.37 vs
   3.60 / 3.67 / 3.92; B 3.62 vs 3.78 / 3.82 / 4.01). We inflate pawns and rooks identically (1.53×) and
   minors less (N 1.32×, B 1.39×) — so minors are cheap *in pawn units*, which is the tradeable quantity.
3. ☠️ **It is NOT a uniform offset, so a flat correction is the wrong instrument.** In win% the error is
   two-sided and fat-tailed (p10 −7.4, p50 +5.6, p90 +21.2 for knights) with the worst decile carrying
   **31-36%** of all error. A uniform ratio correction would shift the median, leave the spread, and make
   the already-undervalued tail worse.

🚨 **All of the above is measured on an eval that violates colour antisymmetry in 74.7% of positions**
(see `eval-colour-asymmetry-is-live-and-widespread`), and `pawn_marginal_real.py` removes WHITE pieces
only. Re-measure after that fix before acting on these magnitudes.

**P2 — Our rank SHAPE is approximately right; the LEVEL is not.**
Real positions: ours spans 144→279 (**1.9×**) across ranks 2-7, SF18 spans 91→192 (**2.1×**). Corrections
should move the level, not re-slope the curve. *(Supersedes the retracted "flat curve" claim, §8.)*

**P3 — Our passed-pawn predicate is a strict SUBSET of SF's, missing 14% of its passers.**
19,310 pawns: `ours-only = 0` at every rank; SF-only 1.6% overall, rising 0.7%→8.7% from rank 2 to 6, and
**exactly 0% at rank 7** — a 7th-rank pawn cannot have a stopper, so both definitions always agree there.

**P4 — The 225 midgame clamp is correctly sized for structure, and is mostly spent on something else.**
Binds 35.6-36.8% of pawns, corpus-invariant over 7 sets. Mean raw **211 = 70 structural + 141 positional**
(placement + attacking layers); positional alone caps 21-23% of pawns. The endgame clamp does **not** bind
(11.5%, ~106 cp headroom) — raising it does nothing.

**P5 — Fitting beats hand-picking, and the values are far smaller than intuition suggests.**
First joint win%-descent over the pawn tables: pawn-only subset **val 279.51 → 265.53** in the shipped
regime, all nine guard tiers improving. It switched `ISOLATED_PAWN_PEN` on at **80** and
`BACKWARD_PAWN_PEN` at **120** — 2.5× below the hand-picked 200/100 they had previously been judged on.
It also wanted level DOWN, **chaining UP (120)**, and passed-table **R2-R5 = 75, R6-R7 = 125**.

**P6 — `PASSER_R_CAP` saturates at the internal ceiling, but does NOT want a higher one.**
The descent drives the cap to **384**, exactly the `clamp(R, 0, 384)` bound. ↩️ **Corrected 2026-08-06:** once
`PASSER_R_MAX` was made tunable and offered **512**, the descent **kept 384**. So the ceiling is reached but
not *pressing* — the earlier reading ("the architecture is blocking a passer valuation it wants to pay") was
overstated. Raising the cap alone remains inert; raising the ceiling buys nothing measured so far.

**P7 — `opposed` is a real and sizeable signal, and we were discarding it.**
`opposed` = an enemy pawn anywhere ahead on our OWN file (a strict SUBSET of "not passed" — a pawn contested
only on an adjacent file is not passed but IS unopposed). `getPPIncrement` detects it and throws it away via
an early `return 0`, collapsing "my file is blocked but I am otherwise healthy" into the same bucket as
"hopeless". Exposed as a modulator on the structural bonus, the joint descent drove **both phases to the
grid minimum, 60%** — i.e. an opposed pawn's structure is worth ~40% less. SF uses the same signal twice
(`Connected[r] * (2 + phalanx - opposed)`, and as the gate on `WeakUnopposed`).
⚠️ Corpus-objective evidence only; not yet game-validated.

**P8 — The per-pawn caps want to be LOWER, not higher.**
Made tunable and offered both directions with new structural terms competing for the headroom, the descent
chose **`PAWN_CLAMP_MID` 225 → 175** and **`PAWN_CLAMP_EG` 175 → 125**. This contradicts both the redesign
brief's "raise the clamp within reason" and the intuition that added terms need more room.
⚠️ Corpus-objective evidence only.

---

## 3. Mapping — where each concept lives

| concept | symbol | note |
|---|---|---|
| rank advancement | `default_/passed_midgame_pawn_rank_bonus`, `endgame_pawn_rank_bonus` | rebuilt at init by `rebuild_scaled_pawn_tables()` |
| phalanx (same-rank neighbour) | `left`/`right` → `pawn_wall_file_bonus` | SF's `phalanx` |
| support (diagonally behind) | `sw`/`se` → `pawn_chain_file_bonus` | SF's `support` |
| doubled | hardcoded 125 mid / 150 eg | file popcount |
| isolated / backward | `ISOLATED_PAWN_PEN` / `BACKWARD_PAWN_PEN` | **bypass both clamps** (`total +=` in a different function) |
| passer detection | `getPPIncrement` | occupancy-based; also computes `opposed` and **discards it** via an early `return 0` |
| passer valuation | `evaluate_passers` | sole payer for flagged passers |
| realizability | `passer_realizability_R` | contest / rear-file / king proximity |

**Missing versus SF**: `opposed` as a modulator, `WeakUnopposed`, `BlockedPawn[]`, phalanx/opposed
modulation of the connected bonus, and a rank term on chain/wall.

---

## 4. Case book

**Generated** by `diagnostics/pawn_truth_casebook.py` from the truth CSVs — regenerate with
`pyrun diagnostics/pawn_truth_casebook.py WRITE=1` rather than editing between the markers.

<!-- CASEBOOK:BEGIN -->
<!-- GENERATED by diagnostics/pawn_truth_casebook.py -- DO NOT EDIT BY HAND. Regenerate with:
     pyrun diagnostics/pawn_truth_casebook.py WRITE=1 -->

⚠️ **All figures below come from MANUFACTURED positions unless a source says otherwise.**
Manufactured magnitudes have been shown wrong by 2.6× against real positions (§8), so treat the
ORDERING as informative and the MAGNITUDES as provisional. A contrast marked **UNRESOLVED** is
not a small effect — it is an absence of evidence.


### Source: `pawn_truth_b080.csv` — 1029 samples

**Marginal value by rank** (cp; mean ±CI, median):

| rank | 3 | 5 | 7 |
|---|---|---|---|
| all | +93 ±16<br>med +64<br>n=475 | +108 ±19<br>med +81<br>n=417 | +500 ±76<br>med +411<br>n=137 |

**Q2 — does one rank of advancement change value, and by how much?**


**Q1/Q3 — weak pawn on rank r vs STRONG pawn on rank r−1**

- (no rank pair has both cells populated in this set)

**Q4a — obstruction ordering** (what is IN FRONT of the pawn):

- `blocked`: +115 ±35 (med +78, n=192)
- `contested`: +115 ±24 (med +76, n=186)
- `opposed`: +86 ±21 (med +70, n=194)
- `passed`: +258 ±46 (med +145, n=284)
- `piece_blocked`: +140 ±35 (med +95, n=173)

**Q4b — does structure matter more when the pawn is stuck?**

- `blocked`: -48 ±101 — **UNRESOLVED**
- `contested`: +63 ±59 — **strong** wins
- `opposed`: +16 ±47 — **UNRESOLVED**
- `passed`: -21 ±115 — **UNRESOLVED**
- `piece_blocked`: +39 ±99 — **UNRESOLVED**

**Q4c — file** (centre vs edge):

- file `c`: +154 ±24 (med +83, n=519)
- file `e`: +152 ±24 (med +94, n=510)


### Source: `pawn_truth_b200.csv` — 1425 samples

**Marginal value by rank** (cp; mean ±CI, median):

| rank | 3 | 5 | 7 |
|---|---|---|---|
| all | +94 ±12<br>med +66<br>n=668 | +94 ±14<br>med +81<br>n=575 | +456 ±57<br>med +382<br>n=182 |

**Q2 — does one rank of advancement change value, and by how much?**


**Q1/Q3 — weak pawn on rank r vs STRONG pawn on rank r−1**

- (no rank pair has both cells populated in this set)

**Q4a — obstruction ordering** (what is IN FRONT of the pawn):

- `blocked`: +109 ±23 (med +88, n=263)
- `contested`: +109 ±18 (med +72, n=264)
- `opposed`: +84 ±19 (med +65, n=255)
- `passed`: +239 ±36 (med +138, n=362)
- `piece_blocked`: +123 ±25 (med +80, n=281)

**Q4b — does structure matter more when the pawn is stuck?**

- `blocked`: -44 ±73 — **UNRESOLVED**
- `contested`: +22 ±52 — **UNRESOLVED**
- `opposed`: -16 ±48 — **UNRESOLVED**
- `passed`: +3 ±100 — **UNRESOLVED**
- `piece_blocked`: +57 ±76 — **UNRESOLVED**

**Q4c — file** (centre vs edge):

- file `c`: +149 ±17 (med +92, n=702)
- file `e`: +132 ±18 (med +84, n=723)


### Source: `pawn_truth_b400.csv` — 2015 samples

**Marginal value by rank** (cp; mean ±CI, median):

| rank | 3 | 5 | 7 |
|---|---|---|---|
| all | +69 ±10<br>med +46<br>n=920 | +92 ±12<br>med +68<br>n=842 | +381 ±42<br>med +307<br>n=253 |

**Q2 — does one rank of advancement change value, and by how much?**


**Q1/Q3 — weak pawn on rank r vs STRONG pawn on rank r−1**

- (no rank pair has both cells populated in this set)

**Q4a — obstruction ordering** (what is IN FRONT of the pawn):

- `blocked`: +88 ±16 (med +70, n=361)
- `contested`: +87 ±15 (med +58, n=376)
- `opposed`: +69 ±15 (med +56, n=362)
- `passed`: +199 ±26 (med +105, n=491)
- `piece_blocked`: +120 ±24 (med +68, n=425)

**Q4b — does structure matter more when the pawn is stuck?**

- `blocked`: -30 ±55 — **UNRESOLVED**
- `contested`: +19 ±36 — **UNRESOLVED**
- `opposed`: -11 ±43 — **UNRESOLVED**
- `passed`: +1 ±72 — **UNRESOLVED**
- `piece_blocked`: +64 ±77 — **UNRESOLVED**

**Q4c — file** (centre vs edge):

- file `c`: +137 ±13 (med +81, n=978)
- file `e`: +101 ±14 (med +61, n=1037)


### Source: `pawn_truth_blockade.csv` — 2817 samples

**Marginal value by rank** (cp; mean ±CI, median):

| rank | 5 | 6 | 7 |
|---|---|---|---|
| all | +99 ±10<br>med +80<br>n=1400 | +208 ±17<br>med +190<br>n=989 | +508 ±42<br>med +424<br>n=428 |

**Q2 — does one rank of advancement change value, and by how much?**

- rank 6 vs 5: +109 ±20 — **rank 6** wins
- rank 7 vs 6: +300 ±45 — **rank 7** wins

**Q1/Q3 — weak pawn on rank r vs STRONG pawn on rank r−1**

- weak@6 vs strong@5: +80 ±33 — **weak@6** wins
- weak@7 vs strong@6: +316 ±75 — **weak@7** wins

**Q4a — obstruction ordering** (what is IN FRONT of the pawn):

- `blocked`: +89 ±19 (med +65, n=557)
- `contested`: +205 ±17 (med +174, n=567)
- `opposed`: +71 ±19 (med +49, n=281)
- `passed`: +341 ±27 (med +254, n=789)
- `piece_blocked`: +173 ±23 (med +137, n=623)

**Q4b — does structure matter more when the pawn is stuck?**

- `blocked`: +66 ±44 — **strong** wins
- `contested`: -19 ±43 — **UNRESOLVED**
- `opposed`: +14 ±45 — **UNRESOLVED**
- `passed`: -17 ±66 — **UNRESOLVED**
- `piece_blocked`: +27 ±55 — **UNRESOLVED**

**Q4c — file** (centre vs edge):

- file `c`: +212 ±17 (med +136, n=1369)
- file `e`: +188 ±15 (med +149, n=1448)


### Source: `pawn_truth_files.csv` — 11039 samples

**Marginal value by rank** (cp; mean ±CI, median):

| rank | 3 | 4 | 5 | 6 |
|---|---|---|---|---|
| all | +80 ±7<br>med +52<br>n=3213 | +76 ±7<br>med +50<br>n=3053 | +98 ±5<br>med +78<br>n=2850 | +224 ±12<br>med +183<br>n=1923 |

**Q2 — does one rank of advancement change value, and by how much?**

- rank 4 vs 3: -4 ±10 — **UNRESOLVED**
- rank 5 vs 4: +22 ±9 — **rank 5** wins
- rank 6 vs 5: +126 ±14 — **rank 6** wins

**Q1/Q3 — weak pawn on rank r vs STRONG pawn on rank r−1**

- weak@4 vs strong@3: -10 ±18 — **UNRESOLVED**
- weak@5 vs strong@4: +10 ±16 — **UNRESOLVED**
- weak@6 vs strong@5: +94 ±24 — **weak@6** wins

**Q4a — obstruction ordering** (what is IN FRONT of the pawn):

- `blocked`: +95 ±8 (med +66, n=2440)
- `contested`: +147 ±9 (med +95, n=2452)
- `opposed`: +81 ±9 (med +54, n=1898)
- `passed`: +132 ±9 (med +86, n=2383)
- `piece_blocked`: +75 ±9 (med +56, n=1866)

**Q4b — does structure matter more when the pawn is stuck?**

- `blocked`: +17 ±19 — **UNRESOLVED**
- `contested`: -2 ±25 — **UNRESOLVED**
- `opposed`: +17 ±23 — **UNRESOLVED**
- `passed`: +0 ±22 — **UNRESOLVED**
- `piece_blocked`: +27 ±23 — **strong** wins

**Q4c — file** (centre vs edge):

- file `a`: +101 ±6 (med +65, n=2693)
- file `c`: +134 ±11 (med +81, n=2746)
- file `e`: +98 ±7 (med +82, n=2852)
- file `h`: +102 ±6 (med +62, n=2748)


### Source: `pawn_truth_low.csv` — 4169 samples

**Marginal value by rank** (cp; mean ±CI, median):

| rank | 2 | 3 | 4 |
|---|---|---|---|
| all | +145 ±13<br>med +100<br>n=942 | +108 ±12<br>med +74<br>n=1636 | +89 ±13<br>med +55<br>n=1591 |

**Q2 — does one rank of advancement change value, and by how much?**

- rank 3 vs 2: -37 ±18 — **rank 2** wins
- rank 4 vs 3: -18 ±17 — **rank 3** wins

**Q1/Q3 — weak pawn on rank r vs STRONG pawn on rank r−1**

- weak@3 vs strong@2: -22 ±37 — **UNRESOLVED**
- weak@4 vs strong@3: -37 ±30 — **strong@3** wins

**Q4a — obstruction ordering** (what is IN FRONT of the pawn):

- `blocked`: +148 ±16 (med +101, n=891)
- `contested`: +111 ±15 (med +68, n=899)
- `opposed`: +111 ±17 (med +78, n=893)
- `passed`: +93 ±16 (med +70, n=862)
- `piece_blocked`: +69 ±19 (med +50, n=624)

**Q4b — does structure matter more when the pawn is stuck?**

- `blocked`: -68 ±41 — **weak** wins
- `contested`: -4 ±40 — **UNRESOLVED**
- `opposed`: -7 ±44 — **UNRESOLVED**
- `passed`: +32 ±34 — **UNRESOLVED**
- `piece_blocked`: +11 ±52 — **UNRESOLVED**

**Q4c — file** (centre vs edge):

- file `c`: +125 ±13 (med +72, n=2072)
- file `e`: +93 ±8 (med +74, n=2097)


☠️ **Quarantined and excluded** (produced before the generator defects were fixed): `pawn_truth.csv`, `pawn_truth_valid.csv`. See §8.

<!-- CASEBOOK:END -->

---

## 5. Known limits — read before quoting any number here

- **Marginal, not total.** Every figure is the value of *adding one pawn*, so contexts that already contain
  similar assets show diminishing returns. A phalanx pawn's marginal value is low partly because its partner
  already carries the value — not because phalanxes are bad. **Our eval models no diminishing returns.**
- **Medians, not means.** Even with a balanced baseline, some samples are decisive; means carry a heavy tail.
- **Manufactured ≠ real.** The generator imposed obstruction contexts uniformly across ranks, sampling free
  rank-7 passers at a rate real games never produce. It was right about direction and wrong about magnitude
  by 2.6×.
- **The corpus proxy does not predict Elo.** On this project −7.3 corpus → ~0 Elo, and −5.2 → +45 Elo.
  Use it to RANK candidates, never to size an effect.
- **STS and WAC have misread pawn changes in both directions** (−196 ⇒ 0.0 Elo, +58 ⇒ ~0, −61 ⇒ +45).

---

## 6. Falsification register

| principle | what would overturn it |
|---|---|
| P1 (level too high by ~57 cp) | a real-position marginal measurement, n ≥ 1,000, showing \|gap\| < 20 cp |
| P2 (shape is fine) | real-position spans differing by more than ~1.5× between ours and SF18 |
| P3 (subset detector) | any position where we flag passed and SF does not (`ours-only > 0`) |
| P4 (clamp correctly sized) | a measured structural axis above ~40 cp, or a bind rate that moves with corpus |
| P5 (fitting beats hand-picking) | a hand-picked config beating the fitted one on held-out val with guards held |
| P6 (R pinned) | `PASSER_R_CAP > 384` changing the engine without also moving the internal clamp |

---

## 7. Provisional — manufactured-position only, NOT yet validated on real positions

Do not build on these without re-measuring:
- Obstruction ordering: passed +341 · contested +205 · piece_blocked +173 · pawn-blocked +89 · opposed +71.
- A weak pawn on the 6th beats a strong one on the 5th (+80 ±33); weak@7 beats strong@6 (+316 ±75).
- Centre-vs-edge file premium ~18 cp (symmetric: a≈h < c≈e).
- Strong-vs-weak structural premium +23 (r5) / +36 (r6).
- Blockaded rank-7 passer +227..+399 vs free +589..+701; **a pawn blocker (+89) hurts more than a piece
  blocker (+173)**.

---

## 8. Refutation record — kept permanently

- ☠️ **"Our rank curve is flat (1.5×) where SF's is steep (7.6×)."** Manufactured artifact. Real positions:
  **1.9× vs 2.1×**. Retracted 2026-08-05.
- ☠️ **"We underpay rank 7 by 83 cp."** Real positions say **overpaid by 88 cp**. Retracted same day.
- ☠️ **"`contested` pawns are starved by the binary selector."** They are our **best-calibrated** context
  (gap −70 vs −137..−152 elsewhere). The argument came from reading one rank table in isolation while ~150 cp
  per pawn is paid by other terms regardless.
- ☠️ **The graded obstruction blend** (`ENABLE_PAWN_OBSTRUCTION_BLEND`). Built, then measured **+7.33 val
  worse** and damaging to `contested`. Two independent instruments. ★ It was **not an SF port** — SF has no
  ordinary/passed table pair to interpolate between; it uses one boolean plus an unconditional `PassedRank[r]`.
  Violated *port FORMS, refit CONSTANTS* by porting a form that does not exist.
- ☠️ **"The linear rank table rewards early pushes."** Measured slopes: ours **−28** cp r2→r3, SF **−29**.
  Came from reading the table in isolation; other terms already offset it.
- ☠️ **"`ISOLATED`/`BACKWARD_PAWN_PEN` failed on four venues."** They were **inconclusive** — each venue's
  95% CI was ±35-67 Elo, combined −8 [−27, +11]. Resolving +10 Elo needs ~4,344 games; we ran 400.
  The descent later switched both ON.
- ☠️ **"Raise the per-pawn clamp"** (the original redesign brief's instinct) and **"index the table on
  file × rank"** (its premise). The clamp is correctly sized; file is the weakest of four factors.

★★★ **The pattern across all of these: a conclusion drawn from a code fragment, or from manufactured
positions, beat a measurement that was already available. Locate the enclosing branch, and validate the
instrument on real positions, before believing a magnitude.**
