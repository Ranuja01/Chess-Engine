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

---

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
