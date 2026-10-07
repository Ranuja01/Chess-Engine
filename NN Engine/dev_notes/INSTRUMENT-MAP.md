# Instrument map — READ BEFORE DESIGNING ANY SCREEN

[`DIAGNOSTICS-TOOLKIT.md`](DIAGNOSTICS-TOOLKIT.md) answers *"which probe exists?"* — search it before
writing one. **This file answers the two questions that come first and are asked nowhere else:**

> **What effect size can this instrument actually RESOLVE?**
> **What does it read when NOTHING has changed, and how does it LIE?**

Those two columns are what the toolkit's ~200 entries mostly lack, and their absence has cost this project
repeatedly: the deployment gate was read against an assumed zero null for months, and the same gate was
pointed at a depth-6-gated knob it is structurally blind to.

---

## ☠️ THE FOUR LAWS OF READING AN INSTRUMENT

1. **A HARNESS'S NULL IS NOT ZERO UNTIL MEASURED.** Four separate instances: the games harness reads
   **+4.6 Elo** on a null; the regret gate's *mean* reads **−0.06 to −0.14** on Elo-neutral arms; a shared
   `captureHistory` defect did not cancel in an A/B; flip RATE does not rank evals at all.
   ★ **Run the arm you believe is NEUTRAL through the instrument, rate-matched, per corpus and per
   stratum.** One neutral point gives a number; two give the BAND, and the band is the bar.
2. **DETERMINISTIC ≠ POWERED.** A metric that is exactly reproducible and large *feels* authoritative while
   its sample cannot support it. The criticality split is deterministic, has the largest magnitude on its
   table, and pointed confidently backwards on the one change whose Elo sign we knew.
   ★ **Quote `n` beside every subgroup read.**
3. **BYTE-IDENTICAL-TO-CONTROL IS THE SIGNATURE OF A SILENT FALLBACK,** not proof of a clean gate.
   ★ **Verify a knob is LIVE by its changed-rate before trusting any result it produces.**
   ⚠️ Byte-identity guards *decisions*, not *cost*: instrumentation that changed no decision cost **30% of
   NPS**. There is no guard on cost.
4. **A SHARED FEEDER DOES NOT IMPLY A SHARED SIGNAL,** and an ABLATION measures the current build's
   DEPENDENCE, not the term's NECESSITY. An ablation number is a SIZING number: it says what a replacement
   must recover.

---

## § A — "Is this worth Elo?"  → games, and nothing else

| instrument | resolves | null | how it lies |
|---|---|---|---|
| **games, 1200** (one night) | **±25 Elo** | **+4.6**, not 0 | Per-night error (±25) EXCEEDS the sd of real candidate effects (**~16**), so one night can never rank two ordinary arms — only detect a large one. Under ~+30 in one night is **UNRESOLVED**, not positive. |
| **games, 4000** | **±10.7 Elo** | +4.6 | The `LMR_SHAPE` ship (+20.7) needed 2800 games to separate. |
| **games, 5276** | ±9.4 | +4.6 | `PROTECT_KILLERS` shipped at +12.4 ±9.4 here. |
| `_venue_power.py` | — | — | **Run it BEFORE spending games and before writing any null into a doc.** It says what the venue can resolve; most nulls in the record are unresolved, not refuted. |

☠️ **Across 57 pools / 88k games, no A/A control has ever been run.** Every historical error bar is
theoretical, not measured. The +4.6 may be selection (we test what we expect to help) or a genuine p1 bias —
**unseparated**.
☠️ **NEVER PEEK.** One 1200-game run read **−31 at 91 games** and **+1.4 at 1200**.
⚠️ `gate` hardcodes `--seed 0` → fixed openings inflate reads. **`openings_uho.txt` + a varied seed are
mandatory.** ⚠️ Fixed-time work is corrupted by machine suspend and by the owner's evening gaming; fixed
depth is safe then. ⚠️ Tournament Elo sign is **p1**.

---

## § B — "Does this eval change pick BETTER MOVES?"

### ⭐ The deployment gate — `_ks_footprint_regret.py`
SF18 win%-regret on the CHANGED-MOVE subset. **This is the instrument that ranks evals.**

| read | value |
|---|---|
| **statistic to use** | `win%` — the share of changed moves that are better. **NOT the mean.** |
| **null, `game_regret_set`** | **49.8 – 50.4** (three neutral arms) |
| **null, `game_regret_set_v2`** | **50.7** — 0.8pp higher. Reading v2 against a flat 50 makes the null look like a replication. |
| **null, per stratum** | drifts **47.5 – 52.0** ⇒ null-correct by stratum too |
| **single-corpus SE** (n≈5,000 changed) | 0.7pp ⇒ resolvable ≈1.5pp |
| ☠️ **real bar, with cross-set replication** | **~2 – 2.5pp** |
| **the whole prize** | a complete eval replacement (SF15c oracle) reads **57.0%** ⇒ **+7pp is ALL there is** |
| ☠️ `n_crit` | **27 / 43** ⇒ criticality **UNMEASURABLE**; quote it every time |

#### 📌 MEASURED PER-STRATUM NULL — `game_regret_set`, d7, `ASPIRATION_DELTA=300` (2026-09-09)
**Do not re-measure this; read candidates against THESE numbers, not against 50.** The aggregate band hides
a ±1.6pp spread across strata, and reading a stratum against a flat 50 misleads in BOTH directions — on the
09-09 threat arm it inflated ps1 (+2.5 apparent vs **+1.8** true) and understated ps2 (−4.7 apparent vs
**−6.3** true).

| stratum | n changed | **null win%** |
|---|---|---|
| aggregate | 5212 (34.7%) | **50.0** |
| opening | 1572 | **51.1** |
| midgame | 2397 | **49.4** |
| endgame | 1243 | **49.8** |
| **ps1 mid_far (≤53)** | 2971 | **50.7** |
| ps2 mid_EDGE (58-64) | 428 | **51.6** |
| ps3 end_EDGE (69-74) | 281 | **49.2** |
| ps4 end_far (≥80) | 1532 | **48.4** |
| cr1 benign | 4340 | 49.7 |
| cr2 minor | 669 | 52.6 |
| cr3 moderate | 180 | 47.4 |
| ☠️ cr4 CRITICAL | **23** | 60.9 — **unreadable, never adjudicate here** |

⚠️ Binomial SE on a stratum difference: ~1.3pp at ps1 (n≈2900 both arms), ~3.7pp at ps2 (n≈300-400). A
2pp reading at ps2 is noise; the same 2pp at ps1 is ~1.5σ. **Size the cell before believing the cell.**

#### 📌 MEASURED PER-STRATUM NULL — `game_regret_set_v2`, d7, `ASPIRATION_DELTA=300` (2026-09-09)
| stratum | n changed | **null win%** |
|---|---|---|
| aggregate | 4278 (35.8%) | **49.8** |
| opening | 1296 | **47.4** |
| midgame | 1987 | **51.6** |
| endgame | 995 | **49.2** |
| **ps1 mid_far (≤53)** | 2431 | **49.5** |
| ps2 mid_EDGE (58-64) | 356 | **53.4** |
| ps3 end_EDGE (69-74) | 262 | **53.1** |
| ps4 end_far (≥80) | 1229 | **48.6** |
| ☠️ cr4 CRITICAL | **19** | 47.4 — **unreadable** |

#### 📌 MEASURED NULL — `variant_regret_set` (whacky/960-no-castle), d7, `ASPIRATION_DELTA=300` (2026-09-09)
| stratum | n changed | null win% |
|---|---|---|
| aggregate | 1219 (20.3% of 6000) | **48.0** |
| `.opening` | **1200 — 98.4% of changed** | **47.6** |
| ps1 ≤53 | 1215 (99.7%) | 47.9 |
| `.midgame` | 19 | unusable |
| ps2 | **4** | unusable |
| ps3 / ps4 / `.endgame` | **absent** | — |

☠️ **This corpus is ~98% ONE STRATUM** — the generator's short walks bucket everything as `opening`/ps1
(caveat already recorded in [[whacky-variant-corpus-for-structure-independent-validation]], now confirmed by
measurement). ⇒ it can test the CROWDED-BOARD half of a finding and **nothing else**: no ps2, no endgame.
⚠️ It also cannot SEPARATE `.opening` from ps1 (1200 vs 1215 of the same 1219) where the standard corpora
can — so a positive reading confirms transfer but cannot localise further.
▶️ A re-gen with `WALK_MAX≈45` would buy midgame/endgame spread and make it a general third cross-set.
⚠️ Power: n≈1200 per side ⇒ difference SE ≈ **2.0pp**. A **directional transfer check**, not a significance
test.

☠️★★★★ **THE SAME STRATUM LABEL HAS A 3.7pp NULL SPREAD ACROSS CORPORA.** `.opening` reads
**51.1 (primary) · 47.4 (v2) · 47.6 (variant)** — primary is the outlier. Reading the variant set against a
flat 50 would score it 2.4pp negative before measuring anything at all. **This is the single strongest
argument for measuring the null per corpus; it is not a refinement, it is the difference between a result
and an artifact.**

☠️★★★★ **THE NULL IS ITSELF A MEASUREMENT WITH ERROR — do not treat these as exact.** At n≈4,000-5,000 a
null cell carries SE ≈ 0.75pp, so `arm − null` has SE ≈ **1.1pp**, not 0.7. The previously DOCUMENTED v2
aggregate null was **50.7**; this measurement reads **49.8** (1.2σ apart, consistent) ⇒ **the v2 aggregate
null is a BAND of 49.8-50.7**, and the primary band 49.8-50.4 likewise comes from three neutral arms, not
one. ★ Two neutral points minimum; one gives a number, two give the band.
☠️ **The two corpora's per-stratum nulls diverge sharply** — opening is **51.1 primary vs 47.4 on v2**
(~2σ, n≈1300-1600 each), ps1 is 50.7 vs 49.5, ps2 is 51.6 vs 53.4. A stratum label denotes a materially
DIFFERENT population in each set. ⇒ per-corpus AND per-stratum nulls are mandatory, never interchangeable.

**How it lies:**
- ☠️ **The MEAN's null is large, negative, and ARM-SPECIFIC.** Two Elo-neutral arms at the *same* flip rate
  read **−0.1389 and −0.0619** — a 2.2× spread that cannot be divided out. Mechanism: conditioning on "the
  move changed" selects positions where the base was marginal, so any perturbation regresses toward better.
  `reg_base` is recomputed per arm and always sits above the corpus mean (3.6456 → 3.7059 across arms).
- ☠️ **It is a d7 EVAL screen and is structurally BLIND to anything gated at depth ≥ 6.** `ENABLE_IIR`
  changed **0 of 15,000** positions through it.
- ☠️ **A single-corpus ~+1pp reading is NOISE.** Two arms cleared the primary band by ~+1pp and then died:
  `KS_ZONE_ATTACK_PCT=0` (51.5% → v2 **+0.0**, phase pattern reversed) and `PIECEVAL_RECOMPUTE_LATE=1`
  (51.1% → v2 **49.2%**, ~2σ negative).
- 🐛 **`/tmp` worker paths were UNQUALIFIED until 2026-09-07** — every concurrent run before then silently
  shared files. Same defect fixed in `_ks_drift_analysis`, `_ks_channel_collinearity`, `_move_change_arms`.
  **Re-run any surprising win from before that date, alone.**
- ⚠️ The aggregate ruler was validated against exactly **one** known-Elo-signed change (a dose-response,
  correct sign). "Correct on n=1" is a pass, not a proof. Its **criticality split failed the same check**,
  pointing −0.887 in favour of a change known to lose ~38 Elo, at `n_crit`=60.

★ **Prior to apply to every eval candidate:** what fraction of 7pp could this plausibly capture, and can
anything we own resolve that? For most single-term ideas the honest answer is "under a point, and no."

### The move-choice family

| instrument | resolves | control / null | how it lies |
|---|---|---|---|
| `_move_change_arms.py` | d1 argmax flip rate per arm; **run FIRST, before any bench sweep or games night** | ★ **always pass `ENABLE_THREATS=0`** — the +45 Elo shipped change, which measures **13.2% flips / 9.2% at ≥10cp** | One-ply STATIC proxy — **cannot see search behaviour**. It once shipped with an argv whitelist that silently dropped its own arm knobs and reported a confident **0.0%**; the control caught it in one run. |
| `_sibling_spread.py` | can a term change our move at all (deletion = the ceiling on retuning) | carries `threats` as control | Comparative only, never absolute. Cannot be pointed at a mechanism that is gated OFF. |
| `_search_stability.py` | move-flip rate between two settings | — | ☠️ **The shipped engine flips its move on 20.8% of quiet positions on `ASPIRATION_DELTA` alone.** This is the floor beneath any single-position comparison. |
| `_d1_move_attribution.py` | which of OUR terms picked the wrong move, in win% | signed weight vs SF11 | 🐛 guard table structurally empty; blame sums outlier-dominated — read medians / frequency-as-top-offender. SF is an oracle over MOVES only; never place an SF term beside one of ours. |

☠️ **Flip RATE alone cannot rank evals**: the SF15c oracle flips 63.5%, a known non-improvement flips 40%,
pure aspiration noise flips 20.8%.

---

## § C — "Is the eval more ACCURATE?"

| instrument | resolves | null / noise | how it lies |
|---|---|---|---|
| **STS300** (fixed depth, arm-vs-arm eval knobs) | **±150 balanced STS** | confirmed two ways: a near-inert arm read **−151**; an arm benching **−59** read **+1.4 ±23.1 over 1200 games** | ☠️ **Do not rank inside ±150.** `LMR_PRODUCT_K=400` tops BOTH fixed-depth suites and is **2 plies worse in play**. Per-theme deltas do **not** replicate — always check a second knob value. The suite is colour-skewed; mirror it. |
| **STS300** (run-to-run repeatability) | ±30–60 | book flake was ±30, **fixed** | 🐛 `sts_test` detected "booked" by grepping fd-captured stdout, which intermittently missed. **Prepend `USE_OPENING_BOOK=0` to every WAC/STS command.** |
| **WAC300 solves** | **±5–6 solves** | — | Fine for accuracy. ☠️ **Its NODE column is not** — see § D. |
| ⭐ **`ENABLE_ORACLE_EVAL`** | the whole lane: SF15c **+7pp / 3–4 plies**, +219 STS @d8; SF11 **+174 STS @d6** and ~90% of SF15c | changed-rate **~58%** = live | ⚠️ **FIXED-DEPTH ONLY** — NPS drops 14×, so any timed reading measures the pipe. ⚠️ **Quote `[oracle] fallback_pct=`** — in-check positions fall back to OUR eval. ⚠️ `ORACLE_SCALE` is for node-matching ONLY; carrying it into a fixed-depth run **degrades the eval badly** (−0.0177 at 200 vs −0.6681 raw). ⚠️ Verify the sign/scale warnings on `namespace Oracle` first. |
| `_ks_auc.py` | KS **discrimination**, SE ≈ **0.007** | ☠️ **run `KS_FLOOR=0` AND pair with the volume control `KS_ATTACK_COUNT=2`** | The floor manufactures mass ties, so volume alone inflates AUC (**+0.107 floored vs +0.007 floor-free**). That control caught a false "detection win". |
| `_ks_calibration.py` | KS **magnitude** (ours/SF ratio per band) | — | The dimension AUC is **structurally blind to**: a feeder defect that shrinks danger uniformly preserves order (AUC flat) while every number comes out too small. |
| `_eval_symmetry.py` | exact — `eval(mirror(b))` must equal `−eval(b)` | symmetric positions return **exactly 0** | Deterministic, order-independent, zero Stockfish, seconds. The one instrument here with no noise floor. |
| `probe_fens.py --table` | per-FEN, full reference ladder, **win% error last** | — | ★ Rank by **win% (k=0.00368208)**, never raw cp — they disagree on which positions are broken. |

☠️ **Corpus MSE is ANTI-correlated with Elo.** Ours reads 245.5 vs SF11's 95.3. Any "switch it off and the
fit improves" result is SUSPECT. Use `fit_bench_guarded.py` (corpus proposes, real benches dispose), never a
raw corpus fit. Never render the corpus objective as a chart.

---

## § D — "Is the tree smaller / the engine faster?"

| instrument | resolves | baseline | how it lies |
|---|---|---|---|
| ⭐ **`depth_nps_bench.py --n 60 MAX_DEPTH=10 PRESET=LONG_FORMAT`** | **THE node judge** — median nodes/position on the game-representative quiet corpus | **249,014** | The correct venue. Its verdict retro-explained a 4,000-game null that WAC could not. |
| ⭐ **`depth_nps_bench.py --n 60 PRESET=LIGHTNING`** | **depth reached at ~1s** | median **12**, mean 12.2 | The cheapest honest proxy for what a change is worth in play. Use it whenever a node saving is claimed. |
| **WAC node count** | ☠️ **NOTHING, for search changes** | 35,310,778 | **FOUR OF SIX configs REVERSED SIGN** against the quiet corpus. Mechanism: WAC's previously-scored root prefix is 7.4% of the root list vs **24.6%** on quiet positions, so anything touching root ordering operates on a ~3× smaller population. **A WAC-only node claim is UNVERIFIED.** The direction of the venue effect is not predictable — measure it, never reason about it. |
| **NPS** | ±14.6% across six runs of ONE byte-identical binary | 450,201 is the **LIGHTNING mean** (LONG_FORMAT ~386k) | ☠️ **Byte-identity does not imply speed identity**: adding cold instrumentation cost ~30% of NPS with zero decision changes, attributed to code layout under `-Ofast -flto`, not to execution. Treat any NPS delta on a build that added significant code as **unattributed**. The stderr redirect target is a second independent confound. |
| **printed EBF** | ☠️ **nothing comparable** | 3.784 | The formula is `pow(num_iterations, 1/depth_limit)` over a **cumulative** counter (aspiration attempts, presearch, qsearch, TT bookkeeping) divided by the loop's **exit** depth, not the nominal one. It is internally inconsistent — two rows of one table implied a 63% and a 2% cut for the same arithmetic. ✅ Keep it for byte-identity fingerprints and same-definition A/B; **never judge on it**. The marginal EBF **1.914** (quiet, 249,014) is the comparable figure. |

### ☠️ THE SPEED BAR — most of this class is closed by arithmetic
**±0.5 ply is Elo-neutral** (measured: bought +0.44/+0.49 ply and lost 5 WAC solves; spent 0.55–0.66 ply and
gained nothing). At the real EBF of ~1.7, a **−6.14%** node saving is `ln(1.0614)/ln(1.7)` = **0.11 ply** —
one fifth of the detectable threshold.
⇒ **A node saving must be ≈35% before it is worth measurable Elo. A speed change needs ≳35% NPS.**
⇒ Staged/lazy movegen (5–8%), movegen micro-fixes (1–3%), pawn-king cache (~3%) are all **far below the bar**.
⇒ **Stop screening node-savers for Elo below ~35%.** Judge them as engineering and ship or drop on that basis.
⇒ ★ The only search win in this project's history **INCREASED** node count (`LMR_SHAPE`, +20.7 with +8% more
nodes). **Accuracy, not volume, is the remaining route.**
🐛 **Never scale Elo from a bundle's headline** — a joint measurement cannot be apportioned to its parts.
⚠️ Node-savers → judge at **FIXED TIME**. Node-increasers → **EQUAL WORK** (step-shaped; use multiple budgets).
⚠️ `NODES` excludes the q-tree (`qnodes=` is separate) — judge q-tree levers on `qnodes` + fixed-time depth.

---

## § E — "Is this term real, distinct, and LIVE?"

| instrument | resolves | threshold | how it lies |
|---|---|---|---|
| `_ks_channel_collinearity.py MODE=heat` | do two terms carry ONE signal or two? | **r > 0.5 = duplicate ⇒ consolidation safe. Lower = distinct ⇒ de-dup SHEDS signal.** | Measured heat 433.3mp / central 215.9mp / OvD 22.7mp at r = **+0.389 / +0.420 / +0.131** ⇒ all distinct. 🐛 Its old `ch4` ablated with a knob that had been **INERT** since `OVD_BOUNDED_MODE=2` shipped — that channel contributed nothing to every earlier run. |
| `SCALE_ATTACK_LAYER` / `SCALE_CENTRAL` / `OVD_CAP` | whole-term ablation, **no build needed** | heat map 0/50/100/150 → **48.9 / 49.4 / base / 49.9** ⇒ worth **~1.3pp**, magnitude AT the optimum | ⚠️ Measures **DEPENDENCE of the current build, not NECESSITY**. It sizes a replacement; it does not say the term is required. |
| `knob-audit` agent | is the knob declared, env-wired, reached at defaults, unfenced? | — | **ECHOED ≠ WIRED** (`toggles` dump advertises dead knobs) and **a FEATURE FLAG IS NOT A FEATURE**. Run before spending any measurement on an arm. |
| `_tail_term_stats.py` | mean \|value\| and **SE** for two FEN sets | — | Built because a KS "4.4× separation" between two 40-position tails was **−0.367 ± 0.517 = noise**. A signed mean hides a term that is −6 half the time and +6 the other half. |
| `breakdown_partition_check.py` | does the breakdown sum to total? | — | ⚠️ `pieces` already contains `material`; `pt_*`/`material` are **SUB-VIEWS** — do not sum them. `pt_*` opens with `total -= values[X]`, so it is **material-INCLUSIVE** — comparing it against SF11's placement-only rows produced a "8–16×" claim that was withdrawn. |

★ **A stratum label means what its BUILDER meant.** The corpus `phase_bucket` column is a **piece count**,
not our `phase_score`. Find the WRITE SITE before reading any field — four instances of getting this wrong.
★ **A CLAMP IS A FREE UPPER BOUND** — bound the mechanism before measuring it.
★ **CRANK UNTIL IT BREAKS**: RFP 1500→400 collapses OURS −207 STS while SF11 gains **+78**. A change that is
null at its natural magnitude may be readable at 4×; quote **eval + re-cranked margins TOGETHER**.

---

## § F — ☠️ WHAT WE CANNOT MEASURE (the honest gaps)

| question | why it is out of reach |
|---|---|
| **Does this help in CRITICAL positions?** | `n_crit` = **27 / 43** on both corpora. No result we have — the heat map's 1.3pp included — speaks to the positions that decide games. **A criticality-enriched corpus is the prerequisite for any such claim.** |
| **Is a single eval term worth anything?** | The cross-set bar is ~2–2.5pp and the whole prize is 7pp. A term would have to be worth **a third of an entire eval replacement**. **Term-at-a-time screening is closed by arithmetic.** Bundling disjoint terms is the only route. |
| **What is our real EBF?** | Unknown. The printed metric is not the literature quantity, and the ~3.0–3.2 estimate rested on a d=10 assumption that is falsified. Needs per-iteration node logging. |
| **Warm-state failures** | ☠️ **No instrument we own can see them.** A position solved correctly at every cold setting (45.6s / 24.0s / 36.5s) was blundered in-game with 20+ seconds, because history had rewarded the same from/to pair earlier in the game. Benches are key-checked, not cold — but they are *incoherently* warm, which is a different regime from real play. |
| **Is the games harness biased or are we selecting?** | Unseparated. No A/A control has ever been run. |
| **Speed below ~35% NPS, node savings below ~35%** | Structurally invisible — below the ±0.5-ply measurement floor. |
| **A term whose REGIME is a few % of positions** (added 2026-09-17) | ☠️ **The CHANGED-MOVE SUBSET is the sample, not the corpus.** Space's only real effect lives on `centre_locked` = **3.2%** of positions; enlarging that class by classifying two MORE corpora (**+13,500 positions**) grew it 1,652 → 1,974 rows and the changed-move count **607 → 644**. A win% on ~600 changed moves carries ≈ **±2pp** of standard error by itself — the size of the whole effect. ⇒ Compute the expected changed-move count (≈ corpus × flip rate) and its SE **before** building the read. Rare regimes need PURPOSE-BUILT corpora, not more of the same material. |
| **The MAGNITUDE of an SPRT result** (added 2026-09-17) | ☠️ **An SPRT DECIDES; it does not measure.** One arm pair produced **+60.7 / +44.5 / +11.4** Elo across three statistically sound runs, because a sequential test stops on a favourable swing and the estimate is biased away from the bound it crossed. Pooling all 1,178 games of the same pairing gave **≈ +31**. ⇒ Quote the pooled tally across seeds; bracket a size with TWO different bounds (here: H1 at ≥10 and H0 at ≥50 ⇒ 10 < true < 50). |
| **Overlap of a SYMMETRIC predicate** (added 2026-09-17) | ☠️ **The collinearity gate's White−Black differencing is structurally blind to any predicate whose two sides have equal cardinality by construction.** `blocked` is the case: `blocked[White]` and `blocked[Black]` are the two halves of the same white/black RAM pairs (`eval_v2.cpp:682`), so the popcounts are ALWAYS equal and the difference is identically 0 — it reads as a dead column, which looks like a bug and is actually a definition. ⇒ "Does space or mobility re-express the RAMMED centre?" is unmeasurable under this convention, and `centre_locked` is exactly the class where space showed its only effect. Needs a different reduction (W+B total, or signing by side to move). ★ Distinguish from a genuine small-sample zero: `lever` also read zero-variance at N=25 and is NOT an identity (one pawn attacked by two gives 1 vs 2) — it varies fine at N=2500. **Check a zero-variance column against its DEFINITION before deleting it.** |
| **Whether a term is STRUCTURE or just the CENSUS** (added 2026-09-17) | ⚠️ Not out of reach, but invisible until you add the control, and three columns were affected. Differencing a detector count W−B carries the MATERIAL difference along with it: against the raw pawn-count difference, `ps_halfopen` reads **−0.93**, `ps_pattacks` **+0.91**, `ps_passed` **+0.83** (replicated −0.92 / +0.88 / +0.70 on the KS corpus). Their mutual VIF of 7.8–18.6 is therefore ambiguous between "shared structure signal" and "both restating rung 0". ⇒ **Any collinearity read over count columns needs a census CONTROL column** (`ps_npawns`), or subsystem overlap and material overlap are indistinguishable. |
| **ANYTHING IN THE ENDGAME-CONVERSION FAMILY** (added 2026-09-19) | ☠️☠️ **§I IS DEMONSTRABLY BLIND TO IT.** The slice-1 draw classifier — a term of exactly this kind — read **IDENTICAL on all six §I corpora**, not merely small. And [[eval-payoff-is-opening-midgame-not-endgame]] measured a WHOLE better eval (SF15-classical substituted in) buying only **−0.32/−0.72** in deep endgames against −1.0/−1.2 in opening/midgame, with the standing instruction to *"deprioritise endgame-only eval work unless the BY_PS split says otherwise"*. ⇒ **Slice 4 cannot be run like slices 2–3: no §I magnitude ladders.** The working instruments are the **TABLEBASE oracle** (`_draw_oracle.py`, Lichess 7-piece, DTM-weighted, `EDGE=1` corner bias — built because uniform sampling never generates boxed-king mates) and **games**, under the owner's gate for self-play-invisible work: position-proof + no bench regression. ⚠️ Also note the d7 regret endgame strata have their OWN nulls (ps4 48.4/48.6, ps3 49.2/53.1 — NOT 50), and `phase_bucket` there is a PIECE COUNT, not our phase. ★★ **QUANTIFIED 2026-09-19, and it is worse than "insensitive": the corpora DO NOT CONTAIN THE POSITIONS.** A scan of all 47 corpora in `ks_sets/` for pawnless K+R vs K+minor found **`diverse_corpus_wide.csv` (n=23,113) holds exactly 1**, the richest corpus anywhere holds **9 / 14,713**, and most hold **zero**. ⇒ this is not a resolution problem that more positions of the usual kind would fix — endgame-conversion work needs **purpose-built corpora**, generated per material signature (`_draw_oracle.py`'s `SIGS` + `random_position` already do this; `ks_sets/t2b_corpus.csv` is the first one). |
| **A GATE RUN ON A CLASS ITS CORPUS DOES NOT CONTAIN** (added 2026-09-19) | ☠️☠️ **A PASS IS NOT EVIDENCE UNTIL THE TERM IS PROVED TO FIRE.** Tier-2b's colour-symmetry gate read a clean **0/4000** — on a corpus containing **one** in-class position. The pass meant "never fired", not "symmetric". ★ The hazard was already ON THE RECORD and went unapplied: the slice-1 KPK entry states *"`_eval_symmetry.py` was NOT run for this: its sample contains essentially no KPvK, so it would be a vacuous pass"* — same tool, same corpus, five days earlier. ⚠️ **Worse, it was also UNREADABLE:** `_eval_symmetry.py` reads `ev_breakdown(b)["total"]`, NOT the eval's return value, so a term that returns early without publishing its breakdown hands the gate the PREVIOUS position's score. ⇒ **Two checks before believing any detector gate: (1) a differential proving the term changes the eval on the target class, (2) that the term publishes a breakdown on every path it returns from.** Both defects were live here simultaneously; each alone would have made the pass meaningless. Cost to detect: one corpus scan and three FENs. |
| **An EG-LEG-ONLY scale, in v2 as it stands** (added 2026-09-19) | ☠️ **Not expressible — an architectural gap, not a tuning one.** All five references scale the ENDGAME LEG ONLY inside the blend (5/5, both lineages), but v2 holds no whole-eval `(mg,eg)` pair: every term blends per-side at its own site with `c.phase256` into a single `total` int. Recovering the EG leg needs `Σ eg_i`, which v2 does not keep; multiplying `total` is equivalent only if `mg_i == eg_i` for every term — false for placement/passers/pawn-structure (KS is unphased, space MG-only). ⇒ Needs a parallel **`eg_total` accumulator**, shipped byte-identical on its own first. ★ This also explains an old null: v1's form (whole `total` × a boolean `isEndGame` cliff) matches NONE of the five, so its −3.3 STS reading measured a shape nobody uses. ⚠️ Exception: pawnless 4–5-piece cases are deep-endgame by construction (mg ≈ eg), so tier-2b-style scales do NOT need the change. |
| **Whether two terms COEXIST well at their chosen magnitudes** (added 2026-09-17) | ☠️ **The collinearity gate cannot see this.** VIF/`r` measure co-movement of detector COUNTS; they say nothing about the relative HEIGHT of the scored curves. Threats × KS: every leg VIF ≤ 1.30 on a general 10k sample AND on `lichess_ks_labelled` — genuinely disjoint detectors — yet threats taxed KS-critical accuracy at every magnitude, because our KS saturates at `KS_V2_MAX` = 4.0 pawns while SF's/Ethereal's kingDanger is an unbounded quadratic that overtakes threats ~2:1 in severe attacks. ⇒ A clean gate means "not the same signal", NOT "safe together". Reading curve balance needs a SOURCE comparison or a 2×2, not the gate. |

---

## § F2 — ☠️★★★★ THE GATE IS A VETO, NOT A SELECTOR (2026-09-09)

An instrument with SE ≈ 1pp **cannot choose** among candidates whose true effects are 0-1pp. That is
arithmetic, not carelessness. But the same instrument rejects reliably: a −2pp reading is real information,
a +1.3pp reading is almost none. **We were using it as a selector.**

### The artifact this produced — three arms, one shape
Every arm was run on the primary corpus, looked good, and was *therefore* promoted to the cross-set. So
**every v2 reading we hold is conditioned on the primary having looked good.**

| arm | primary | v2 |
|---|---|---|
| `KS_ZONE_ATTACK_PCT=0` | +1.1 to +1.6 | +0.0 |
| `PIECEVAL_RECOMPUTE_LATE=1` | +0.7 to +1.2 | −1.5 |
| `THREAT_MINOR_ON_DEFENDED=1` (09-09) | +1.3 (ps1 +1.8) | **−0.5** (ps1 **+0.2**) |

If all three have a TRUE effect of **zero**, conditioning on "we noticed it" predicts a primary reading of
about +1σ ≈ +1pp — observed +1.1 / +0.7-1.2 / +1.3 — while v2, unconditioned, should read ~0; observed
0.0 / −1.5 / −0.5, mean ≈ −0.7 against SE-of-mean ≈ 0.6. **It fits.**
🐛 **This row was first recorded as −1.4, computed against the DOCUMENTED v2 null (50.7) instead of a
MEASURED one (49.8)** — the exact error this section exists to prevent, committed within the hour by its
own author. Arm A went **FLAT on replication (ps1 +1.8 → +0.2)**, not reversed. The verdict is unchanged
(null, not a candidate) but "reversed" and "flat" are different claims and only one was true.
★ **A correction rule is not self-applying. Re-read your own numbers against it before publishing them.**
⇒ ☠️ **"Cleared one corpus then reversed" is what a ZERO-effect knob looks like after you select on the
first corpus.** It has twice been recorded as evidence that the two corpora want different evals. That is
the more interesting explanation, not the more likely one. Real composition effects exist — but a
SEQUENTIAL FILTER cannot distinguish them from selection, because the signature is identical.
▶️ **Run BOTH corpora before looking at either, and pool.** Free, removes the selection step entirely, and
gives one reading at SE ≈ 0.7pp instead of two 1pp readings conditioned on each other.
⚠️ It rescues nothing: pooled, the 09-09 arm reads ~50.3% — still null. What changes is that we stop
narrating three nulls as a live mechanism.

### ⇒ THE THREE-WAY TRIAGE (replaces "it failed")
| reading | status | why |
|---|---|---|
| clearly negative, multi-σ | **VETOED — dead** | a veto is robust to the winner's curse |
| null **at CRANKED magnitude** | **REFUTED — dead** | ★ cranking separates INVISIBLE from INERT. Null at 4× is the mechanism not mattering, and no bundle rescues it (`KS_MOB_EDGE`) |
| null at natural magnitude, never cranked | **REOPENABLE as bundle material** | never tested at a resolvable size; this is MOST of the ~85-attempt record |

### ⇒ CONSEQUENCES FOR HOW WE WORK
1. **Kill on the gate; never promote on it.** Nothing advances for reading positive.
2. **Choose BUNDLES, not knobs.** If nothing single is worth 2pp, no protocol can choose a single knob — the
   object is not resolvable. Four disjoint 0.5pp changes are one 2pp object that IS. Attribute by
   leave-one-out only AFTER a bundle wins.
   ⚠️ Bundling manufactures no effect: if the parts are truly zero the bundle is zero. The win is finding
   that out in ONE run instead of eleven — and a null bundle is a far stronger closure than eleven null
   singles, because it closes the set at a resolvable size.
   ⚠️ **Check disjointness FIRST** — `Hanging` was ~87% a subset of capture gains and was never independent.
   The `DUMP=` changed-set overlap is the test.
3. **Keep the candidate count SMALL.** The winner's curse scales with how many things you screen. Thirty
   swept knobs guarantee a fake winner near +1.5pp; four you can each explain mechanistically barely bias at
   all. ★ That is the real argument for reading a reference engine over sweeping — fewer, better-motivated
   candidates, not smarter ones.
4. **Never quote the SCREENING number as the estimate.** A set used to select is inflated by construction.
5. **GAMES choose. The proxy's job is to stop us wasting games**, not to pick the winner.

### ⚠️ AND THE AGGREGATE READ THE RULE DOES NOT EXCUSE
Individually a null is uninformative. **A long RUN of them is not.** Outposts, whole-board mobility, piece
mobility, Hanging, the att2 clause, the mobility→kingDanger wiring — ~15 SF ports, all null to negative,
while what carries value here is OURS (heat map 1.3pp; `MOD_KS_REALIZ` inside the +36.7 bundle). Same logic
as THREE UNRESOLVABLE SIGNALS THE SAME WAY ARE A RESULT. ⇒ **SF-porting has a poor track record in this
engine**, and the reframe does not rescue it: a term tuned inside SF's search and eval does not
automatically fit outside them. ★ Prefer fixing what OUR term does wrong over adding a foreign one beside
it — and note that KS's failure is DIRECTIONAL (additive 0-for-11, only subtractive wins), which better
resolution does not change.

## § F3 — ☠️☠️★★★★ win% IS POPULATION-DEPENDENT — a GLOBAL null is the wrong comparator (2026-09-09)

The record says *"win% needs no PER-ARM band — unlike the mean, whose null is arm-specific."* **That is too
strong.** It was verified on two neutral arms flipping 35.3% and 33.2% — a 2pp range. It does NOT hold
across populations with different flip rates, and **every candidate we screen has a different flip rate
from every neutral arm we own.**

### The confound, visible in six arms' own output
| arm | flip% | `reg_base` |
|---|---|---|
| noise100 (neutral) | 37.9% | 3.633 |
| asp300 (neutral) | 34.7% | 3.652 |
| noise30 (neutral) | 34.8% | 3.617 |
| threats coverage | 26.7% | 3.938 |
| KS off | 25.9% | 4.029 |
| KS 1500 / 4500 | 21.6 / 21.7% | 4.073 |

**Lower flip rate ⇒ systematically HIGHER base regret.** A knob that flips fewer moves flips only where the
base's top two were closest — where the base is most likely wrong and ANY perturbation regresses toward
better. Neutral arms cluster at 35-38%; candidates cluster at 21-27%. **They are different populations.**
☠️ **You cannot fix this by dialling a neutral arm down.** `EVAL_NOISE_SIGMA` SATURATES (30 → 34.8%,
100 → 37.9%): a large pool of near-tied moves flips under any perturbation. Two attempts failed.

### 🧰 THE FIX: `_paired_null.py` — match the POPULATION, not the rate. Zero CPU.
Read the arm against the null **on the FENs where BOTH changed the move**. Same positions, same selection.
Needs only the `DUMP=` files, so it re-reads history for free.
▶️ `pyrun diagnostics/_paired_null.py [MINPC=26] ARM=<dump> NULLS=<a>,<b>,<c>`
⚠️ The intersection is biased toward EASY-TO-FLIP positions — closer to matched than a global null, not
perfect. ★ **Pass SEVERAL nulls: the neutral arms disagree by ~1pp globally and by 2.3pp on identical
positions.** One paired null gives a number; three give the band.

### 📏 What it measured
☠️ **On the paired population every arm — candidates AND all three neutrals — reads 52-56%, not 50%.** The
"natural" win rate where both arms found the position marginal is **~2.5pp above the global null** — the
same size as the effects being reported.
| reading | global null | **paired (3 nulls)** |
|---|---|---|
| KS ablation, aggregate | +0.1 / +0.4 / +1.4 across the ladder | **+0.6, spread 1.7 ⇒ NULL** |
| KS ablation, ≥26 pieces | +2.2 … +3.4 | **+2.4, spread 1.5, all three positive** |
⇒ **The KS magnitude ladder was an ARTIFACT** — every magnitude "beat" the base because each was read
against an easier population. The crowded-board result SURVIVES, barely.
⚠️ Each pairing is only 0.6-1.2σ and the nulls disagree by 2.3pp on the same positions ⇒ **this corpus
cannot settle a ~2.5pp stratum effect either way.** That is the case for the 4× corpus, stated as a number.

★ **RE-READ ANY PAST win% RESULT THROUGH `_paired_null.py` BEFORE CITING IT.** The global-null comparison
has been the screening method for months, and candidates systematically flip fewer moves than neutral arms.
☠️★★★★ **THE CORRECTION IS NOT ONE-DIRECTIONAL — do not re-read only the flattering results.** It DEFLATED
the KS aggregate (+0.1/+0.4/+1.4 → +0.6) and **INFLATED** the threat-coverage arm on the primary corpus
(+1.3 → **+2.0**). Whether pairing helps or hurts depends on how the arm's changed set sits relative to the
null's, which is not predictable from the flip rate alone. ⇒ **a past NULL may be hiding a result just as
easily as a past WIN may be hiding an artifact.**
📌 Worked example — threat coverage, `THREAT_MINOR_ON_DEFENDED=1`: paired **+2.0** on primary (3 nulls,
spread 1.4) and **+0.0** on v2 (one null, so a point without a band ⇒ read as ±1.4). Pooled ~+1.0. The
cross-set failure SURVIVES the better comparison, so the verdict is unchanged — but only the numbers, not
the conclusion, were safe to assume.

## § G — ▶️ PROTOCOL — before you spend anything

1. **`record-check` first.** Has this been tried? Was it **RESOLVED**, or merely **UNREADABLE**? ~85 eval
   attempts are mostly unresolved nulls, not refutations — "we tried that" is usually the wrong reading.
2. **`knob-audit`.** Is the knob live at defaults? A byte-identical result is a silent fallback until proven
   otherwise.
3. **Look up the resolution here**, then **count the resolvable effect BEFORE calling a null.** If the
   instrument cannot see the effect you predict, the run is wasted whatever it returns.
4. **Measure the null** — the neutral arm, rate-matched, **per corpus AND per stratum**.
5. **Register the prediction before the run.** Direction and magnitude, written down.
6. **Run the leave-one-out BEFORE flipping a default.**
7. **Cross-set or it didn't happen.** One corpus at +1pp is noise.
8. **Three unresolvable signals pointing the same way ARE a result.** Conversely, one large deterministic
   number in a sparse cell is not.
9. **Play games after any search change** — the 09-03 root-sort blunder shows ONLY under timed d20.
10. **Run a surprising win AGAIN, alone** (shared-tmp contamination).

### Operational gotchas that silently corrupt a run
- ☠️ **`sts`/`wac` take `<tag>` FIRST, knobs after.** Reversing it is the tag artifact — and the harness
  discards the guard's output (`2>/dev/null`), so it fails silently. **A guard whose output the harness
  discards is not a guard.**
- ☠️ **THE JUDGE IS THE TARGET COLUMN — never change it incrementally.** Every number here is scored
  against **SF18 multi-PV @ d14**: the null bands, the +7pp prize, SF11's 0.08 gap, the heat map's 1.3pp.
  **SF19 (on disk since 2026-09-09, ~+44 Elo) must NOT relabel or extend an existing corpus** — mixed
  labels put two truth standards in one column and destroy comparability silently, with no error and no
  visible wrongness. The 08-06 rebuild avoided exactly this by labelling at d13 to match, rather than the
  script's d18 default. Any corpus enlargement uses **SF18 @ d14**. A move to SF19 as judge is a deliberate
  re-baselining of the entire program in one pass, never a swap. SF19 IS safe as a reference-ladder rung
  (`_sts_reference.py` scores a reference on OUR suite; it relabels nothing) and is NOT a better roadmap
  target — there has been no classical eval since SF15.1, so it prices the net ceiling, not the
  hand-writable one.
- ☠️ **Canonical byte-identity command: `MAX_DEPTH=10 USE_OPENING_BOOK=0 PRESET=LONG_FORMAT`.** Without
  `LONG_FORMAT` the default `STANDARD` has a 45s limit, so hard positions time-abort and node counts become
  **machine-load dependent** (256,518,458 idle vs 260,131,069 under load — pure timing, looked like a leak).
- ☠️ **Knobs latch at init ⇒ ONE PROCESS PER SETTING.**
- ☠️ **Pre-2026-08-14 positional/regret numbers are confounded** by history contamination across FENs in one
  process. Fixed by `clearSearchTables()` (diagnostic-only, `DIAG_NO_CLEAR=1` opts out). Magnitude:
  STS 1670 → **1771**. Games/SPRT, static-eval diagnostics and byte-id fingerprints are **not** affected.
- ☠️ **conc ≤ 4 · never `nohup` · never rebuild while a job runs · never `| tail` a long run.** RAM, not
  cores, is the binding constraint — cap ~2 engine-loading runs.
- ⚠️ Timed benches are non-deterministic and load-sensitive. Run arms **back-to-back on an idle box**.

---

## Maintenance
★ **When you measure a new null, a new noise floor, or a new way an instrument lies, add the row HERE** —
not only to the memory that records the finding. A resolution number that lives in one memory file is
reachable only by whoever remembers its name.
⚠️ **THE HEADER IS NOT THE RECORD** — this applies to this document too. Every number above carries the
measurement it came from; if you cannot find the measurement, treat the number as unverified.

---

## §H — ☠️ STS IS CHAOTICALLY SENSITIVE ON A COARSE EVAL (measured 2026-09-11, eval v2 rung 0.5)

**Question it answers:** can STS tune a constant inside an early v2 rung? **No.**

Measured on v2 rung 0.5 (material + PST only), sweeping `EVAL_V2_PAWN_MG`, all runs fixed-depth d10:

| PAWN_MG | 999 | **1000** | 1001 | 995 | 1010 | 1200 | 900 | 800 | 700 | 650 | 600 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| STS | 1362 | **1364** | **1433** | 1410 | 1407 | 1429 | 1412 | 1432 | 1472 | 1441 | 1353 |

★ **STS is EXACTLY deterministic here** — 1000 re-ran twice at 1364/1364, so the usual ±30-60 repeatability
band does NOT apply and every difference above is reproducible.
☠️ **And a 0.1% eval change (1000 → 1001) moves it 69 points.** 1001 = +69, 999 = −2, 995 = +46,
1010 = +43. Large, **non-monotone**, and not sign-symmetric.

⚠️ The first hypothesis — symmetric tie-breaking in a coarse eval — was FALSIFIED: it predicted 999 would
jump like 1001, and 999 sat on the baseline. What the data supports is weaker and messier: **chaotic
sensitivity of a fixed-depth search to a coarse eval**, where a sub-millipawn input change reshuffles move
choices unpredictably.

### ▶️ THE RULE
**Reproducible ≠ attributable.** A deterministic instrument can still be unusable, because determinism only
means the same input gives the same output — not that the output tracks the thing you varied.
- ✅ STS is fine for **rung-level** questions ("is this feature worth having?"), where the expected move is
  hundreds of points — v1's whole eval is +432 over v2 rung 0.
- ☠️ STS is USELESS for **constant-tuning inside an early rung**: local scatter from a ±0.5% input change
  (~70 pts) is comparable to the entire trend across a 2x sweep (~110 pts).
- ⇒ Tune constants on the **d7 regret gate** (thousands of positions vs a measured null band, so tie
  reshuffling averages out), then confirm the rung with games.

### ⚠️ AND IT SHRINKS AS THE LADDER GROWS
The floor is a property of eval COARSENESS, so it is worst at rung 0 and falls as terms are added and exact
ties become rare. ⇒ **Early rungs are the HARDEST to measure, not the easiest** — the opposite of the
intuition that a thin eval gives clean readings. Re-measure this floor at each checkpoint rather than
assuming the rung-0 number holds.

### §H2 — ☠️ A THIN EVAL WIDENS THE REGRET GATE'S NULL BAND TOO (measured 2026-09-11)

§H showed STS going chaotic on a coarse eval. The regret gate degrades the same way, independently:

| base | neutral arms | band | width |
|---|---|---|---|
| v1 (shipped eval) | asp300 / asp800 / noise30 | 50.1 - 51.6 | **1.5pp** |
| **v2 rung 0.5** (material + PST) | asp300 **47.9** / noise30 **51.8** | 47.9 - 51.8 | **3.9pp** |

⇒ **The null band is ~2.6x wider on the thin eval.** Both of our cheap instruments lose resolution at the
bottom of the ladder, for the same underlying reason: a coarse eval leaves many near-ties, so any
perturbation reshuffles move choice more violently.

★ **Early rungs are the HARDEST to measure, not the easiest.** The intuition that a barebones eval gives
clean readings is exactly backwards, and it is now confirmed on two independent instruments.

☠️ **WORKED EXAMPLE — why this is not academic.** The rung-0.5 pawn sweep read
`pmg650/700/800 = 49.3 / 50.5 / 51.3`. Against the **borrowed v1 band** (50.1-51.6), 800 looked like it was
at the top and nearly clearing — a shippable-looking signal. Against its **own measured band** (47.9-51.8)
it is interior, and in fact BELOW the noise30 null. Same numbers, opposite conclusion.
⇒ `A HARNESS'S NULL IS NOT ZERO UNTIL MEASURED` extends to: **it is not the SAME null when the BASE changes.**
Re-measure the band whenever the base arm changes, not only when the corpus does.

### ▶️ CONSEQUENCES FOR THE LADDER
1. Do not attempt fine constant-tuning at rungs 0-2. Nothing cheap can resolve it.
2. Quote every early-rung result against a band measured **on that rung's own base**.
3. ⇒ Reinforces the checkpoint-games structure: accumulate 3-4 rungs, then decide with games, because the
   cheap instruments are at their weakest exactly where the ladder starts.
4. Re-measure both floors (STS scatter, gate band) at each checkpoint — they should SHRINK as the eval
   gains resolution, and that shrinkage is itself a progress signal.

---

## ★★★ §I — EVAL ACCURACY vs SF18: the instrument that CAN resolve early rungs

🧰 `diagnostics/_eval_accuracy_arms.py` · **question:** how close is our STATIC eval to SF18's assessment?
**No search, no move choice, no null band needed, deterministic.**

Measured 2026-09-11 on `ks_sets/lichess_ks_labelled.csv` (n=2609 after dropping mates/terminals):

| arm | win%-MSE | vs ref |
|---|---|---|
| v2 rung 0.5, no KS | 1908.38 | — |
| + pawn_mg 1001 (**null**) | 1908.53 | **+0.01%** |
| + pawn_mg 995 (**null**) | 1907.53 | **−0.04%** |
| + KS-A (product) | 1785.61 | **−6.43%** |

### ★★ RESOLUTION: noise floor ≈ 0.05%; the KS effect is ~130x it
☠️ **The SAME 1-millipawn change that moved STS by 69 points moves this by 0.01%.**
⇒ This instrument is ~1000x less sensitive to irrelevant perturbation than a fixed-depth move bench,
because it never routes through search tie-breaking (§H) or the changed-move population (§F2).

### ★★ §I2 — ☠️ SLICING §I BY POSITION CLASS RAISES ITS RESOLUTION, NOT ITS INDEPENDENCE (2026-09-16/17)
🧰 `_position_class.py` writes pawn-structure class corpora (`centre_tension` · `centre_locked` · `centre_open` ·
`centre_cleared` · `other`, plus an orthogonal `pin_dense` tag) that §I and the regret gate consume unchanged.
**It works as a MAGNIFIER:** `MOB_V2_PIN` read **−4.75% on `pin_dense` vs −0.51% globally (9×)**; `MOB_V2_EXCL_LOWRANK`
−2.39% on `centre_tension` vs −0.54% on `centre_cleared`; space's ONLY real effect was `centre_locked` −0.04..−0.13.
☠️ **But a per-class §I win is NOT a second instrument.** §I is corpus fit; slicing changes the POPULATION, not the
instrument, so §I-by-class and §I-global are ONE vote however much louder the sliced one sounds. Both magnified effects
above then read NULL on the d7 regret gate **on their own classes** (pin −0.6pp vs a class-measured 50.0; exlow −1.4pp vs
50.4). I wrote "confirmed on this instrument" after the §I result and had to withdraw it.
### ▶️ THE ARITHMETIC THAT DECIDES WHETHER A CLASS READ CAN RESOLVE ANYTHING
A class-local effect reaches the aggregate only in proportion to the class's SHARE: `centre_locked` is **3.2%** of
positions, so 0.032 × 0.13 ≈ **0.004 = a tenth of §I's own floor**. ⇒ Compute `share × effect` BEFORE running the class
read, and compare it to 0.05%.
⚠️ **Class neutrals do NOT transfer.** Measured bars on the same base: `pin_dense` 50.0 · `centre_tension` 50.4 ·
`centre_locked` 49.4 — against whole-corpus 49.9 / 51.1. The bar is a property of the CORPUS.

### ▶️ WHY IT MATTERS FOR THE LADDER
§H and §H2 concluded that NOTHING we owned could tune a constant at early rungs — STS goes chaotic
(~70 pts from a 0.1% change) and the gate's null band widens to 3.9pp on a thin eval. **§I reopens that.**
Constants and shapes CAN be resolved at rung 0-2, on accuracy, at a floor of ~0.05%.

| question | instrument |
|---|---|
| is the eval TRUER? | ★ §I accuracy — resolves at ~0.05% |
| does it change the MOVE? | d7 regret gate — ⚠️ blind to KS, needs a matched null band |
| does anything else break? | STS — regression guard only, ~70 pt floor at early rungs |
| is it ELO? | ☠️ games. Nothing above predicts Elo |

### ⚠️ BOUNDS — do not over-read
- Closeness to SF18 is **not Elo**; `corpus-fit-is-anti-correlated-with-elo` still applies. §I RANKS arms,
  it does not promote one.
- SF18 SEARCHES, so no static eval reaches 0. Only DIFFERENCES between arms on one corpus are meaningful
  (see `_reference_ceiling.py` for the achievable floor).
- Both nulls used were TINY perturbations (1mp, 5mp). They are the right control for demonstrating the
  contrast with STS, but a larger irrelevant change would test the floor more harshly.
- ☠️ Units are load-bearing: `ev()` is ABSOLUTE BLACK-POSITIVE MILLIPAWNS, `best_cp` is WHITE-POV
  CENTIPAWNS. A sign or scale slip yields a plausible loss that means nothing.

### 📌 The day's lesson that produced it
The KS shape question was answered "better", then "worse", then "better" by three different instruments
within an hour — 1-D STS sweep (+114), 8-config STS screen (worse in 3 of 4 cells), then §I (consistently
better, 40x the floor). ★ The move-choice instruments disagreed with each other; the accuracy instrument
agreed with itself and had a floor 1000x lower. **When instruments disagree, prefer the one that measures
the quantity you actually changed** — we changed the EVAL, so measure the EVAL, not a move three plies of
search downstream of it.

## ☠️ 2026-09-26/27 — new ways instruments misled (or nearly did)
- **Self-play overstates an eval gain ~3×.** Texel fit A read +111 vs v2 in self-play and +38 vs SF18 (paired, same
  openings and colours, `gauntlet`, 1,000 games). Validate every eval ship EXTERNALLY.
- **A held-out loss gain is not Elo below some size.** C1 improved held-out loss 0.5-0.76% and was Elo-null
  (−5.5 ± 16); fit A's 2.75% was +38 external.
- **Silent tool failures, caught by their signatures:**
  - `_eval_accuracy_arms.py` printed "no rows" (schema mismatch);
  - `_ks_footprint_regret.py` arms did not inherit BASE_KNOBS (three different arms returned byte-identical rows;
    the neutral flipped 55%). Fixed `ad14423`;
  - `vs_sf.py` ignored FEN starts. Fixed;
  - `engine_server.py` swallowed table-load errors. Fixed `c561d42`: check `[c1]` / ☠ lines in game logs.
- **Threshold on a field that already contains the threshold:** `ks_counts` units are post-onset. The recall study
  compared them to the onset again, and the headline was wrong by 7× (`c3e7fc9`).
- **"King danger" labels are confounded with already losing.** Always report KS results on NEAR-EQUAL positions.
- **Lopsided starts saturate colour-swapped pairs** (queen odds 2% informative pairs). Use `_variant_report.py`'s pair
  view; read heavy odds only against a fixed stronger opponent.
- **`sprt.py --max-minutes` overrides `--max-games`.** The real cap is the time budget.

## ☠️ 2026-09-28/30 — new ways instruments misled (or nearly did)
1. **50k SPRT + 50k replication can both pass a bundle that is flat at play depth.** Fit K1p read +34 self-play and +41 vs
   SF18 at 50k nodes but −11 at 250k. ⇒ Gate every fitted change vs SF18 at ≥ 250k nodes, and **each part of a joint fit
   separately** (K1p's KS alone was +30; its PST/C3 parts cancelled it).
2. **A story before the ablation.** I blamed "tactical KS is depth-fragile" for a night; the KS-only arm disproved it.
   Run the per-part ablation BEFORE explaining a bundle's failure.
3. **Pairing against the ship-selection seeds biases every candidate negative** (the baseline was selected-high there) —
   use fresh seeds + a fresh baseline (memory `gate-new-candidates-on-fresh-seeds-not-ship-seeds`).
4. **Classes defined by the baseline game's own outcome regress to the mean** (length, sharp/quiet): "K1p loses quiet
   games −14.6pp" was this artefact. Read every such split against a near-null CONTROL arm on the same games.
5. **A tool's degenerate classifier** (sharp = any ≥300 cp swing put 995/1000 games in one class) — read class COUNTS first.
6. **Over-strong L2 can pin a fit to ~0 and still "converge"** (Fit W first run: the unfitted SF prior beat it) — compare
   against the prior; grid λ.
7. **Better held-out prediction, worse games, from a DISCONTINUOUS form.** Additive `sign(T)·C` winnability amplified 98%
   of near-level endgames to ±½ pawn; the training rows (quiet, decided) are only 2.5% near-level, so the fit barely paid
   for it while search lives there. ⇒ check a term's behaviour around T = 0 (`_win_amplify_check.py`); prefer continuous
   (multiplicative) forms for leader-relative terms.
8. **The move-regret split by criticality is unreadable on our sets** (cr3/cr4 hold 2-27 changed moves per arm) — a
   critical-position question needs a purpose-built critical corpus.

## 2026-10-01/04 — new ways instruments misled (and the fixes)
1. ★ **The STATIC residual is mostly search-resolvable.** SF18 − our static: side-to-move +2.41pp → +0.80 against our
   d10 SEARCH; mean |gap| 8.39 → 5.63. Use the DEPTH residual (`_depth_residual_pass.py`) to look for missing eval
   knowledge; POT's middlegame was null on it, the queen imbalance persisted on it.
2. ★★ **The external judge drifted too weak.** SF18 @400 nodes was anchored at ~50% and v2 reached ~75%; three material
   terms read negative there but positive in self-play, and Kaufman flipped −29.6 → +18.9 on SF18 @800 (54%). Re-anchor
   after every ship (the 10-03 ship moved the @800 baseline to ~56%); gate on BOTH the calibrated judge and self-play.
3. **A class defined by the baseline game's CONTENT still regresses** (persistent queen-imbalance games: Kaufman −15.5pp
   — and the unrelated C3 arm −14.9pp). Always read a game-class split against a control arm.
4. **Hand-picked cases overstate patterns:** 8 triangulation cases said "threats" (6/8); the aggregate over 111 said
   threats is a wash (20 helps / 18 hurts) and structural KS is the consistent miss (48/17, +24 cp).
5. **A fit that frees per-FILE cells breaks the file-mirror gate** unless mirrored files are tied (pawn fit: 2,435 / 3,170
   violations) — the guard caught it; the fitter now ties a=h, b=g, c=f, d=e.
6. **Partial self-play reads swing:** the 10-03 confirmation read 46.4% at 460 games and finished +13.6 at 2,000 — the
   "never call a direction on partial data" rule, again.

## 2026-10-04/07 — new ways instruments misled (and the fixes)
1. ★★ **Judge re-anchored: SF18 @1000** (48.6% on the 10-04 ship; @800 had drifted to 57-59%, @1200 = 42.5%). Re-sweep when
   the baseline leaves ~45-55%.
2. ★★ **A SHARED BASELINE correlates its arms:** seed 78's baseline read low (54.3%) and EVERY arm of that seed read +2…+5pp;
   seed 79 separated them. Read pooled seeds only, never one seed of a shared-baseline batch.
3. ★★ **Fit size does NOT predict game size:** mobility (fit −1.44%) → 0 in games; king protector (−0.91%) → +16 on SF18 but −2
   in self-play; Kaufman depth re-fit (−3.16%) → +7 combined. The screen ranks; games decide.
4. ★ **Two fair instruments can genuinely DISAGREE** (not noise): KPROT SF +16 (2,000) vs self-play −2; KFL+PST pair SF −3 ± 8
   (3,000, three SF levels) vs self-play +15. The rule ships only on both — a style-specific gain is not a strength gain.
5. ★ **Revival-screen artefacts, all caught before a verdict:** no global-SCALE nuisance ⇒ blocks win by stretching; the
   optimizer stops at its start once SCALE dominates (exact −0.00% reads) ⇒ start at the baseline nuisances + tight
   tolerances; a FEN-hash val split leaks same-game rows ⇒ split by GAME.
6. ★ **The static ladder on UNQUIET positions measures static TACTICS:** v1 "beat" v2 2× on random-walk variants purely via
   capture-gains (v1 without it: 725 vs v2 624). Judge variants/odds understanding with our d10 SEARCH vs SF18.
7. **Static accuracy ≠ Elo, again:** the 10-03/10-04 ships (≈ +30 Elo) left static MSE slightly WORSE (depth-target fits);
   STS at equal nodes rose 1747 → 1838 with them.
8. **Tooling:** feature-pass closures refuse under the shipped connected knob (add PS_V2_CONN_MAG=0); WSL /tmp is wiped
   when the distro idles (dumps → E:); tracked background tasks die at 30 min after a reload (launch detached).
