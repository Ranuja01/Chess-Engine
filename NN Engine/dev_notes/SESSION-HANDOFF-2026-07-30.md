# Session handoff — 2026-07-29 → 07-30 (overnight campaign + the gravcap ship)

---

# 📌 LATER SESSION 2026-07-30 — two lanes closed, one soundness fix shipped

**Read this block first; everything below it is the earlier gravcap session.**
Baseline unchanged and re-verified after every build: **`254 / 35,982,407 / EBF 3.820 / STS 1629`**.

| commit | what | state |
|---|---|---|
| `d0d3ac1` | `[prune_pair]` fire counters + log/handoff | byte-identical |
| `f8b3d11` | `ENABLE_LMR_REMDEPTH` + `LMR_REMDEPTH_SCALE` | default-off, **lane CLOSED** |
| `4115315` | `ENABLE_CORRHIST_QSEARCH` + `DISABLE_QCACHE` | default-off, **lane CLOSED** |
| `daf3adf` | `[qcache_hygiene]` counters + gated fix | byte-identical |
| **`19c1c21`** | **`QCACHE_SOUND_STORE` default ON** | ✅ **SHIPPED (correctness)** |

## 🚨 THE METHOD RESULT THAT OUTLIVES BOTH LANES
**For a depth-keyed mechanism, the BENCH DEPTH is part of the config.** `ENABLE_LMR_REMDEPTH` reads as a
wash at d10 (**+0.9% nodes**) and as **−11.1% nodes** at d12 — because `DEPTH_REDUCTION[D] = D − 1` for all
D ≤ 9 and `rem` ≤ 9 at `MAX_DEPTH=10`, so **the bench sits entirely in the table's flat region while real
timed play sits past it.** Counters prove they are different features (`avg_ply_delta` −0.355 at d10 vs
−0.0015 at d12). ⚠️ "Run STS on the EXACT config" was **obeyed and still produced a false verdict.**
★ Owner's catch — *"benches at D10 may not represent a timed scenario where depth generally exceeds that."*
▶️ **ACTION: re-examine every depth-keyed knob swept only at d10** — `LMP_MAX_DEPTH`,
`HIST_PRUNE_MAX_DEPTH`, the RFP depth cap, `DEPTH_REDUCTION` itself. **Part of the search 0-for-13 record
may be this artifact**, which makes it the highest-value item on the board.

📐 **Second method result — price trades in PLY-EQUIVALENTS.** Same baseline at two depths gives STS
1629@d10 → 1737@d12 ⇒ **~54 STS/ply**. So a −110 STS arm surrenders ~2 plies to buy an 11.1% node cut worth
~0.08 ply: **off by 25×**, no tuning rescues it. ⚠️ Triage only — gravcap (−29 STS, +33 Elo) is the
standing counterexample that no statistic predicts Elo.

## ✅ SHIPPED — the q-cache stored values the search never produced
`qSearch` returns a bare `0` on **timeout abort, node-limit abort, and repetition draw**;
`get_q_search_eval` cached all three as evaluations. The cache has **no age field and is never cleared**, so
one move's timeout unwind answered questions for the rest of the game. ★ The probe/store logic is
textbook-correct — **the unsoundness is upstream, in what qsearch hands it.**
🚨 `[qcache_hygiene]` = **0 at fixed depth, 144 timed** ⇒ no bench we own can reach those paths, so the fix
is **byte-identical by construction** and shipped on correctness, not on a measurement.
☠️ **The main TT was NEVER affected** (I claimed it was, wrongly): the unguarded overload is inside a
`/* */` block; the live one refuses `score == 0`, and the abort value *is* 0.
🪤 **`ENABLE_TT_STORE_DRAW=1` removes that protection** and starts caching aborts as real bounds — warned in
the header now.
⚠️ **Do not oversell it:** ~30 bad entries per game in an **8.4M-entry** cache. **Unlikely to explain the
lightning/standard blunders** (~1200 cp swings = systematic, not a stale leaf); warm state in the
non-position-keyed history/killer tables remains the better hypothesis.
▶️ **One untested number could change that verdict:** `draw_stores` read 0 only because WAC has no
repetition history. **Measure it in a real game** — the counter is already in the build.

## ☠️ RETRACTED — "the q-cache masks eval work" (the lead below is DEAD; investigated and killed)
A **SEARCH** change amplifies as much as any eval change (`LMP_MAX_DEPTH=8`: **−170 cached, +7 uncached**),
and **all three tested changes beat the uncached control including a known-bad one** (control 1497 < LMP8
1504 < corrhist 1578 < KS 1602) ⇒ **`DISABLE_QCACHE` is a PATHOLOGICAL BASELINE and not a valid measurement
regime.** ★ Same lesson as the depth artifact: **the REGIME is part of the config**; the measurements were
right, the cross-regime inference was wrong. ✅ **Discriminator to reuse: measure a change of the OPPOSITE
KIND in both regimes.**

## ☠️ SECOND RETRACTION — "unsound but profitable" was ALSO wrong; the q-cache is FINE
| config | solves | nodes | q-cache hits |
|---|---|---|---|
| full cache | **254** | 35,982,407 | 1,457,574 (**99.5% bound**) |
| `QCACHE_EXACT_ONLY=1` | 246 | 38,963,635 | 9,568 |
| `DISABLE_QCACHE=1` | 246 | 38,451,470 | 0 |

⚠️ **The bound reuse is CORRECT and STANDARD (same as SF).** ☠️ `QCACHE_EXACT_ONLY` was
**near-tautological**: PVS runs most searches on a **null window** (`beta = alpha+1`) where **EXACT is
impossible**, so the 99.5%/0.5% split is what theory predicts and I disabled the only path that can hit.
☠️ **"A cache can't change fixed-depth results" is false** — every TT does. **254 → 246 is ordinary**, and
the 132-STS cost is **not a puzzle**, just what removing a working TT costs.
✅ **REAL gaps vs SF: no depth field, no age/generation, direct-mapped always-overwrite.** The missing
generation is the same root cause as the poisoning bug — entries outlive their search with no way to prefer
fresh ones. **That is the only genuinely actionable item here.**
(`TIME_LIMIT=600 s/move` ⇒ nothing truncates; my earlier truncation worry was also wrong.)

## ~~★★ THE BEST UNCLAIMED LEAD — the q-cache MASKS eval work~~ (SUPERSEDED)
Corrhist RFP-only is **flat with the cache on** (1630 vs 1629) and **+81 STS with it off** — while using
**more** nodes (WAC no-qcache: control 246 / 38.45M vs corrhist 247 / 38.72M), which **falsifies** the
confound that the time-truncated no-cache regime merely rewards node savings.
⇒ Cached qsearch values override corrected evals downstream, so this would suppress **any** eval-side
improvement routed through qsearch — a candidate explanation for a whole class of "eval change measured
null" results in the ledger.
⚠️ Still unexplained and worth its own look: removing a **bound-checked** cache costs **132 STS**
(1629→1497) for only ~7% more nodes. That is a large quality swing for a supposedly sound lookup, and both
observations come from the same knob, so understand it before trusting either.
⚠️ `DISABLE_QCACHE` is **diagnostic only** — never shippable.

## ☠️ Closed this session
- **`ENABLE_LMR_REMDEPTH`** — no plateau exists (150→200 swings nodes 27 pts with nothing between; integer
  division yields only 1 or 2 plies at the dominant rem). d12 best arm is −11.1% nodes for **−110 STS**.
  The horizon-truncation defect has now measured null/negative **four** ways.
- **`ENABLE_CORR_HIST` re-wire** — the "mis-wired" defect was real and **fixing it made things worse**:
  live (91k stand-pat flips) but **−49 STS** matched, still negative with the cache bypassed. SF keys
  corrhist four ways with divisor 131072; ours is one coarse pawn table ⇒ a **structural** correction
  mis-prices **tactical** leaves. ☠️ `improving` is inert by construction (pawn-key cancels over 2 plies) —
  argued, **not measured**.
- ▶️ Delta pruning's unsound UPPERBOUND is **dead code by default** (`!ENABLE_QDELTA_PERMOVE`, which ships
  `true`). Nothing to chase.

## ▶️ NEXT — both of my proposed leads were investigated and CLOSED this session
1. ☠️ **d10 depth-artifact audit — DONE, 0-for-2.** Artifact real (RFP 415×) but rescued nothing.
2. ☠️ **q-cache masking — DONE, RETRACTED.** Replaced by "unsound but profitable, don't fix it."
3. ▶️ **`draw_stores` in a real game** — still open, still cheap, the one number that could revise the
   shipped q-cache fix upward. **Lowest-effort item on the board.**
4. ▶️ **Aggression × guard Pair B** — now the default lane again, but note the record: today's search work
   went **0-for-6** (remdepth, corrhist-qsearch, RFP cap, LMP cap, + two retracted leads), every arm
   trading a few % nodes for 25-170 STS, **off by 15-25× on ply-pricing.**
   ⚖️ **The standing asymmetry is the real signal: eval is 5-for-5 (+83.4 Elo), search is 2-for-15.**
   Search-side pruning tuning is priced badly here *because the eval cannot support more aggression* —
   which the `RFP_MAX_DEPTH` result independently confirmed (SF's cap growth tracks eval trust we lack).
   ⇒ **Strongly consider returning to the EVAL lane** ([[ordering-retest-queue-under-gravcap]] PST/AST
   tiebreaker, passer V3 re-judge) rather than more search knobs.

---

# 🏆 OUTCOME — `gravcap` SHIPPED, +33.0 Elo (2026-07-30)

**Everything below this section is the road that led here; this is the current state.**

## Committed on `NN-ENgine`
| commit | what | fingerprint |
|---|---|---|
| **`6e26ffd`** | gated root-table + pruning infrastructure, all knobs default-off | **byte-identical** 249 / 38,840,709 / EBF 3.934 / STS 1662 |
| **`5a8655e`** | gravcap shipped as default | 🚨 **NEW BASELINE 254 / 35,982,407 / EBF 3.820 / STS 1629** |

🚨 **The old `249 / 38,840,709 / EBF 3.934 / STS 1662` is RETIRED and now MISLEADING** — it reproduces only
with gravity + capture-history forced OFF. Check every future build against **254 / 35,982,407 / 3.820 /
1629**. NPS unchanged (~468k, median-of-3), so the −7.4% nodes is not a speed artifact.

## The win
```
ENABLE_HISTORY_SATURATION=1 ENABLE_HISTORY_MALUS=1 ENABLE_CAPTURE_HIST=1 \
STATSCORE_OFFSET=0 STATSCORE_DIVISOR=683
```
**+506 −392 =305 over 1203 games = 54.74% ⇒ +33.0 Elo, 95% CI [+16.1, +50.1]** (`gate`, LIGHTNING, conc 4,
tag `sprt_gravcap`). First search-lane win since qdelta.

★ **Malus is a PRECONDITION, not a feature.** Without it history accumulates only bonuses, saturates, and
stops discriminating — which is why every history CONSUMER previously measured null or inert (capture
history "marginal", `ENABLE_HIST_PRUNE` literally inert, malus alone +10.9 ±36.6). ⚠️ The five values are
**one atomic unit**: the statScore constants are re-derived from the post-gravity distribution, and keeping
the old 512/1024 under gravity costs **−90 STS**.

## ☠️ How it was nearly thrown away — the durable lesson
| stage | verdict | reality |
|---|---|---|
| WAC / STS | +2 solves, −29 STS ⇒ "flat" | blind to it |
| cploss tail screen, 1 run/side | "**+58% blunders — cancel the SPRT**" | **noise** |
| baseline replicated ×4 | `rate>10%` = 5.3 / 4.9 / 6.7 / 6.5% | instrument can't resolve <2pp |
| 1203 games | **+33.0 ±17 Elo** | ✅ |

**Every bench either missed this or argued against it. Only games found it.** ⚠️ Also: the SPRT's LLR sat at
+2.06/2.944 while the point estimate was already +33 — at `elo1=5` the LLR crawls regardless of true
strength. **Score the PGNs; set `elo1` 20-30 for a large expected effect.**

## ✅ PHASE 1 PAIR A — RESULT (2026-07-30)
Fire counters added (`[prune_pair]`: `lmp_fires` / `lmp_exempt_saves` / `hist_prune_fires`) **before** the
2×2, which is what made it interpretable. Baseline `lmp_fires = 19,970,133`.

★ **`ENABLE_HIST_PRUNE` is no longer inert.** It prunes on *very negative* butterfly history; bonus-only
history never goes negative, so the condition was **unsatisfiable** — the mechanical reason for the old
null. With malus: 4,794 fires at stock `COEF=512`, 78,744 at 128, 262,696 at 64/d6.

★★ **The guard DOES rescue over-aggression** — coef64/d6 alone **247 / +1.0% nodes**; + guard@1000
**250 / −1.2%**; guard + coef64/**d8** **251 / −3.3%** (best node result of the sweep). Aggression can be
pushed FURTHER once something catches what it breaks. ⚠️ **Interaction, not addition:** each half's
individually-best value combined (guard@1000 + coef128) was **worse than either alone** (247 / +2.9%).

☠️ **But the threshold surface is CHAOTIC — no value is selectable.** `LMP_HIST_EXEMPT` 4000/1500/1000/250
⇒ nodes **−0.8% / +4.3% / −2.8% / +6.3%**, i.e. 1500 is worse than BOTH neighbours, while
`lmp_exempt_saves` rises monotonically. The mechanism scales smoothly; the outcome does not.
⚠️ **Not measurement noise** — fixed-depth node counts are deterministic and will reproduce exactly. Only
the **DIRECTION** survives; no point on the curve does.

## ☠️ CORRECTED — the pre-search-off penalty
The earlier "+24.5%" compared gravity-on-p-off against gravity-**OFF**-p-on (mixed baselines). Like-for-like,
same config with the pre-search on vs off: **54.7% (pre-gravity) → 30.7% (gravcap) → 27.8% (+guards).**
Removal still costs **8.5M nodes AND 10-14 solves** (that solve loss is *outside* the noise band).
⇒ **Not reachable by accumulating small ordering wins:** ~24 points came from ONE structural change
(malus); a whole guard sweep bought ~3.

## ▶️ NEXT LANE (plan approved): aggression × guard PAIRS
Guards here have only ever been tested at **baseline** aggression, where they cannot pay — a guard that
prevents prunes which were fine just adds nodes and measures null by construction. Meanwhile the one
aggression test without a guard (`LMR_EXTRA=2`) cost −55 Elo. **Both diagonals tested; never the corner.**
All guards already exist default-off. **Pair A: `ENABLE_HIST_PRUNE` × `ENABLE_LMP_HIST_EXEMPT`** — gate on
**fire counters** before spending games. Then the structural fix: `reduced_search_depth` keys on ITERATION
depth, not the node's REMAINING depth (measured damage: wrong-reduction 1% at L1-L5 → 6.2% at L8).

---

# ★★★ EVENING RESULT — ROOT LANE CLOSED, GRAVITY IS THE CANDIDATE (2026-07-29)

**Read this first; the root-table section below is now history.** Base **249 / 38,840,709 / STS 1662**,
byte-identity re-verified after all eight builds. Nothing committed.

## 🚨 DO THIS FIRST, BEFORE LAUNCHING THE SPRT — the NPS gap
**Node counts are only half of fixed-time performance and no NPS measurement was taken on the candidate.**
Gravity adds work at every beta cutoff (malus across history, counter-move, cont-hist-2ply, capture
history). **If it costs ~11% NPS it exactly cancels the 11% node saving and the SPRT confirms nothing.**

1. `depth_nps_bench` on **baseline** → confirm the clean reference (~460-477k).
2. Same on the **candidate** → confirm gravity does not eat the saving.
3. Launch the SPRT **only if NPS holds.**

⚠️ **Measure IDLE.** NPS noise is ±5% and load-sensitive; a +22 STS under gaming load once became −58 idle.
Never take this reading while anything else is running.

## ★ THE SPRT CANDIDATE — recalibrated, not the shipped-constant variant
```
ENABLE_HISTORY_SATURATION=1 ENABLE_HISTORY_MALUS=1 \
ENABLE_HIST_PRUNE=1 ENABLE_CAPTURE_HIST=1 ENABLE_CONT_HIST_2PLY=1 \
STATSCORE_OFFSET=0 STATSCORE_DIVISOR=683
```
**251 solves / 34,547,294 nodes (−11.1%) / STS 1633.** ⇒ **SPRT vs defaults, `gate`, conc 4, full night.**
A node-saving change must be judged at **fixed TIME** — that is where 11% converts to depth, and it is why
the bench understates it (qdelta precedent: WAC +1 for +53.8 Elo).

| arm | solves | nodes | vs base | STS |
|---|---|---|---|---|
| baseline | 249 | 38.84M | — | 1662 |
| gravity only, shipped constants | 245 | 35.22M | −9.3% | — |
| gravity only, recalibrated 0/683 | 242 | 36.44M | −6.2% | — |
| **gravity + all three, recalibrated** | **251** | **34.55M** | **−11.1%** | **1633** ✅ |
| gravity + all three, shipped constants | 249 | 33.78M | −13.0% | **1572 (−90)** ☠️ |
| gravity + capture_hist, recalibrated | 254 | 35.98M | −7.4% | 1629 |
| gravity + cont_hist_2ply only | 240 | 35.31M | −9.1% | — |
| gravity + decay OFF | 243 | 38.44M | −1.0% | — |

⚠️ **NEAR MISS — read it before trusting any WAC/node pair.** I called recalibration "falsified" on WAC and
nodes, and was one command from gating the shipped-constant arm at **−90 STS**. Recalibration is worth
**+61 STS** for +2.3% nodes: the constants govern how `STATSCORE_LMR` grades **QUIET** moves, which STS
measures and WAC cannot. **Run STS on the exact config being gated, never on a near-variant.**

## ★★★ THE PRE-SEARCH QUESTION IS RESOLVED — and both levers are required
| config | solves | nodes | p-off penalty |
|---|---|---|---|
| p-off, no table, no gravity | 57 / 112 | — | collapse |
| **p-off + gravity, NO table** | **111** | 53.94M | collapse |
| p-off + table, no gravity | 238 | 60.09M | **+54.7%** |
| **p-off + table + gravity** | **243** | **48.34M** | **+24.5%** |

- **The TABLE makes pre-search-off VIABLE** (111 → 243 solves) — gravity alone still collapses.
- **GRAVITY makes it AFFORDABLE** (+54.7% → +24.5%) — the table alone moved the penalty 1.3 points.

⇒ **The pre-search sells INTERIOR HEURISTIC POPULATION** (its subtree = 38.7% of all beta cutoffs) **and**
silently props up a root table too sparse to stand alone. ⇒ **KEEP `ENABLE_ROOT_TABLE`: it earns no Elo
alone but is a structural PRECONDITION for the single-search endpoint.** Remaining gap: **+24.5%**.

## Candidate set under gravity — everything measured, one winner

| config | solves | nodes | STS | verdict |
|---|---|---|---|---|
| baseline | 249 | 38.84M | 1662 | — |
| **gravity (recal 0/683)** | **251** | **34.55M (−11.1%)** | **1633** | ✅ **GATE THIS** |
| gravity + root table | 245 | 40.16M (+3.4%) | — | costs nodes |
| gravity + subset (no alpha seed) | 241 | 36.48M | — | costs nodes |
| gravity + subset + `ROOT_PRESEARCH_REDUCTION=3` | 226 | **27.66M (−28.8%)** | **1553 (−109)** | ❌ both accuracy axes |

⇒ **No reduced-pre-search variant survives.** The 28.8% node cut is real but costs −23 solves AND −109 STS;
half a ply cannot buy that back. **Gravity alone, pre-search untouched, is the only clean candidate.**

## ★★ WHY the root table costs nodes (non-obvious, and it explains the whole lane)
Storage is free — the cost is the **soundness fix riding along with it.** Razoring under the table requires
`last_real != UNPROVEN && age <= MAX_AGE && last_real_depth >= depth_limit - deficit`, and **`last_real` is
written only by `root_table_store`, i.e. only by the MAIN search.** Pre-search scores never populate it, so
every move scored only by the pre-search is treated as "no evidence", becomes unrazorable, and gets a
**full-depth search**. Razoring saves ~31% of nodes, so the table silently switches much of it off.

★ That is the **measured-vs-fabricated line drawn wrong a THIRD time**: a pre-search score is *measured*,
merely **shallow**. ▶️ **Untried fix:** populate `last_real`/`last_real_depth` from pre-search results at
their true shallow depth and let `ROOT_RAZOR_MAX_DEPTH_DEFICIT` grade them — replacing binary trust with the
depth-graded judgement the field exists for.

## ☠️ Three of the plan's own hypotheses died tonight
1. **§E — the pre-search does NOT sell root ordering.** A persistent table at 93% evidence coverage left
   pre-search-off at **+54.7%** vs +56% with no table. Surviving candidates: interior TT warming and
   history population (its subtree = 38.7% of all beta cutoffs).
2. **statScore recalibration — FALSIFIED.** Alone it is worse on both axes; against the full feature set it
   buys +2 solves for +2.3% nodes. **The gain is the FEATURES malus unblocks, not the constants.**
3. **`ENABLE_HISTORY_DECAY=0` — RETRACTED.** Costs 3.2M nodes. Keep decay ON alongside saturation.

⚠️ **Six of my mechanism STORIES were falsified in one evening while every MEASUREMENT held**
(root-LMR-under-table, L2-sort-fixes-LMR, razor-to-LMR-saves-nodes, stale-reduction-saves-nodes,
decay-is-redundant, statScore-recalibration). Weight narrative in these notes accordingly.

## ☠️ Root lane: CLOSED after 14 arms
Nothing reached baseline nodes (best **+7.1%**; best solves 254 at +13.1%). `ROOT_STALE_TO_LMR` — the only
branch that could remove work — collapsed to **170/300** because reducing a stale move makes it fail low ⇒
stays unproven ⇒ ages ⇒ reduced again: **staleness feeds itself.**
✅ **KEEP `ENABLE_ROOT_TABLE` as a CORRECTNESS fix** (all 8 `alpha_beta` exits now leave a valid table;
the full-length invariant is structural, not conventional; `ROOT_RAZOR_CONTINUE`'s 92/300 collapse is gone)
— **but do not gate it for Elo.**

---

# 🔨 EVENING BUILD — `ENABLE_ROOT_TABLE` IS BUILT (2026-07-29, uncommitted)

**Everything below this section is still valid as history; this is the current state.**
Full detail in `presearch-replacement-plan-2026-07-28.md` § BUILD LEDGER.

**Baseline byte-identity re-confirmed after every build: 249 / 38,840,709 / EBF 3.934.**

## What exists now
- **`ENABLE_ROOT_TABLE`** — root table sized to the **full** move list before the loop, updated **in place**,
  never cleared, never pushed. Because the table is complete before the first move, **all eight of
  `alpha_beta`'s exits now leave a valid table instead of a stump — the full-length invariant is structural
  rather than a convention six sites uphold.** Fixes the correctness hazard that broke 3× in one session.
- **`ROOT_RAZOR_MAX_AGE`** (default 1) — razoring consults `last_real`/`age`, i.e. *the most recent measured
  score and how stale it is*, replacing the positional `synthetic_from` cutoff. This is the "razor only on a
  reasonably recent score, otherwise skip" design; skip-instead-of-break is now sound.
- **`ROOT_LMR_EXEMPT_BEST`** — built, and **measured INERT** (see below). Keep; retest inside the bundle.
- `RootScore` gained `prev_score` / `last_real` / `verified` / `age`; two-level `std::stable_sort`;
  `[root_table]` counters.

## Measured (fixed depth; base 249 / 38,840,709)
| arm | solves | nodes | evidence coverage |
|---|---|---|---|
| `ROOT_LMR_EXEMPT_BEST=1`, no table | 243 | 37,756,371 | ☠️ **bit-identical to off — INERT** |
| table, store-only-verified | 252 | +17.0% | 4.4% |
| table + `last_real` split | 243 | +9.8% | **92.8%** (was ~34%) |

## ★ Three corrections banked tonight
1. **The exemption's guarded set is EMPTY BY CONSTRUCTION** — we re-sort by score each iteration, so the
   ex-best sits at index 0, below `ROOT_LMR_MIN_IDX`. It only becomes live under store-only-verified, where
   a fail-low ex-best gets a sentinel and sinks into reduction range. **SF needs both rules because each
   creates the other's failure mode.**
2. **"SF has 100% coverage" was SLOT coverage.** SF's *score* coverage is ~3% as well — fail-soft PVS proves
   a value only for alpha-raisers. Slot coverage was the real defect; it is now 100%.
3. **The pruning line is MEASURED vs FABRICATED, not proven vs unproven.** A fail-low is a real fail-soft
   value; only fill values are fabricated. Porting SF's store-only-verified literally starved the razor
   (+17% nodes). **SF can discard fail-lows because SF never razors at the root; we do.**

## ▶️ NEXT
`ROOT_RAZOR_MAX_AGE` is the live axis (how stale a score may be and still be razorable). A mechanism sweep
over age ∈ {0,1,2,4} + pre-search-off + root-LMR-inside-the-bundle was launched on the 1-core window.
**Shortlist rule: evidence high AND nodes ≤ baseline.** Neither arm meets it yet — nodes are still above
base, so **nothing has earned games.** ⚠️ Both solve deltas (+3, −6) are inside the **no-sign-information
band** (<15 solves); do not read them as verdicts.

---

# ★★★ MORNING ADDENDUM — the architecture answer (supersedes several conclusions below)

## 1. The pre-search is SELF-FINANCING (measured)
It costs ~41% of nodes and returns ~the same in main-search savings via the TT/ordering it warms. **Every
removal variant RELOCATES work rather than eliminating it.** One statement that fits every arm in the arc.

## 2. Subset with NO alpha seeding is ACCURACY-NEUTRAL
| alpha seeding under `ENABLE_PRESEARCH_SUBSET=1` | WAC | nodes | STS |
|---|---|---|---|
| seed exactly | 244 | +5.5% | 1617 |
| margin 300 | 247 | −0.1% | 1626 |
| **no seeding** | **248** | +0.8% | **1663** (base 1662) |

⇒ The −5 solves / −45 STS was **alpha poisoned by stale PVS fail-low bounds**, NOT the skipping. We CAN
reuse known data without losing strength — it just saves no nodes.

## 3. ☠️ It is NOT IID (Fable audit, quoted code) — my framing was wrong
SF11 IID: `if (depth >= 7 && !ttMove) { search(..., depth - 7, ...); }` — **conditional, −7 plies, keeps only
the MOVE.** Ours: unconditional, every iteration, per-move loop at **depth−1**, keeps scores. The SF lineage
is a retreat: IID (~1 Elo) → depth-cut (SF15.1) → **IIR (SF16, ~9 Elo)** → SF18. **No strong engine has a
full-width root pre-pass** — not 4 SF versions, not Ethereal. ⚠️ Our `ENABLE_IIR` is INTERIOR-node only and
does NOT substitute; it is complementary (it prices cold first-visits, which matters MORE if the pre-search
shrinks).

★ **In SF the PREVIOUS ITERATION *is* the pre-search** — iterative deepening supplies the shallow pass,
persisted once in `RootMoves` and reused, never re-derived.

## 4. ★ SF's root bookkeeping — the design we should copy (verified in SF11 source)
```cpp
// search.h — two-level sort key
bool operator<(const RootMove& m) const {
    return m.score != score ? m.score < score : m.previousScore < previousScore; }
// search.cpp L407-410 — reset each iteration
for (RootMove& rm : rootMoves) rm.previousScore = rm.score;
// search.cpp L1248-1269 — store ONLY verified scores
if (moveCount == 1 || value > alpha) { rm.score = value; ... }
else rm.score = -VALUE_INFINITE;   // "unproven this iteration", NOT a measured value
```
- **Depths are never mixed.** Level 1 = this iteration's scores (all same depth); level 2 = previous
  iteration's (all same depth). A shallow score can never outrank a deep one because they are never compared.
- **Reduced-depth scores are NEVER stored.** Beat alpha ⇒ PVS re-searches at full depth ⇒ stored score is
  honest. Fail low ⇒ `-VALUE_INFINITE` + stable sort keeps its prior position.
- ⇒ **LMR-instead-of-razoring is strictly better than what we do:** a razored move can never revive; a
  reduced move gets a fresh cheap chance every iteration.
- **We violate this**: pre-search scores and main-search scores share one untyped `top_score` field and are
  sorted together.

## 5. ★ WE DISCARD DATA EVERY ITERATION
`alpha_beta` does `previous_search_data.scores.clear()` then re-pushes ONLY the moves it searched (~14 of
~35). A move scored at iteration 5 and razored at iteration 6 **loses its score permanently**. `moves_list`
survives (ordering carries), `scores` do not. SF's `RootMoves` is never cleared — it is UPDATED.

## 6. Coverage hole starts at ITERATION 1
`RAZOR_BASE_FIRST=750 / RAZOR_FLOOR_FIRST=200` ⇒ threshold ≈ **1000mp at depth 3** — looser than the ~100-300
used later, but it still fires. Beta cutoffs cannot (full window until `ASPIRATION_MIN_DEPTH=5`). So razored
moves are never scored, sort last, stay last, get razored again — self-reinforcing from the first iteration.

⚠️ **DEPENDENCY ORDER (adopting the sort alone would be HARMFUL):**
**coverage (LMR replaces razor-break) → `-VALUE_INFINITE` + `previousScore` two-level stable sort →
pre-search removal.** Without coverage, permanently-razored moves sit at `-inf` on BOTH keys with the
pre-search's scores no longer consulted.

## 7. Why they can and we can't — the efficiency statement
SF's reduced scout yields a **binary** answer ("did it beat alpha?") for a fraction of a search; ours yields
a **precise score**, recomputed every iteration and then discarded. **We are over-computing**, not missing
information.

## 8. ⚠️ INVARIANT FRAGILITY (violated 3× in one session)
`scores` must be full-length AND every entry must carry a non-empty legal reply list — enforced by
convention across `reorder_legal_moves`, `descending_sort_wrapper` and four push sites, by nothing in the
type system. Violations tonight: both draw returns (silent), `ROOT_RAZOR_CONTINUE`'s missing push (abort),
and my own razor-continue patch reusing a moved-from `entry` (**81/300**). Each produced a different,
non-obvious symptom. **SF has no such hazard: `RootMoves` is fixed-size and UPDATED, never pushed.**
⇒ the persistent-table change is a **correctness** item, not just a node optimisation.

---

# Overnight sections (earlier; some superseded by the addendum above)

**Nothing shipped. Nothing committed.** The night's output is a comprehensive negative on the root
pre-search lane, one solid mechanical finding about root razoring, and — most importantly — **the discovery
that neither bench regime can resolve effects of the size we have been chasing.**

---

## 🚨🚨 THE INSTRUMENT RESULT (read this first; it invalidates a lot of today's earlier work)

### Fixed TIME is not reproducible
| run | STS /3000 |
|---|---|
| base, early in the night | **1688** |
| **base, ~1h later — same binary, same idle machine** | **1563** |

**A 125-point drift on an unchanged configuration**, larger than every effect measured against it
(tail C4 −58, root LMR −92). ⇒ **Every fixed-time comparison from tonight and yesterday is unusable**,
including the "base wins on both axes" claim I made mid-session and the earlier "+22 STS trade".

### Fixed DEPTH is reproducible but CHAOTIC
| run | STS |
|---|---|
| base | **1662** |
| base repeat | **1662** (bit-identical) |

Deterministic to the point — but the response to a knob is **not smooth**. `ROOT_LMR_MIN_IDX` STS:

| MIN_IDX | 3 | 5 | 6 | 7 | **8** | **9** | 10 | 12 |
|---|---|---|---|---|---|---|---|---|
| STS | 1518 | 1506 | 1581 | 1569 | **1632** | **1636** | 1532 | 1527 |

100+ point cliffs between adjacent values. **Determinism ≠ smoothness.** A single good point is as likely
to be a chaotic spike as a real optimum — the same trap `CHUNK=4` set earlier.

⇒ **Neither regime settles small effects. Only self-play games are load-symmetric** (both arms share binary
and machine simultaneously). Corollary: **byte-identity guarantees identical DECISIONS, not identical
depth-reached-per-second**, so at fixed time any code addition perturbs the comparison.

---

## ☠️ THE LANE IS CLOSED — every candidate measured worse

All at **fixed depth** (the reproducible regime). Base = 249 WAC / 38,840,709 nodes / STS 1662.

| arm | WAC solves | WAC nodes | STS |
|---|---|---|---|
| `ENABLE_ROOT_RAZOR=0` | 249 | **+31%** | — |
| `ENABLE_ROOT_LMR=1` (MIN_IDX 3) | 246 | **−5.6%** | 1518 (−144) |
| root LMR MIN_IDX **9** (best found) | 243 | −2.8% | **1636 (−26)** |
| razor off + root LMR | 239 | +9.2% | — |
| `PRESEARCH_TAIL_MODE=2 CHUNK=4` | 249 | +9.4% | 1564 (−98) |
| `ENABLE_PRESEARCH_SUBSET=1` | 244 | +5.5% | — |

**Best candidate of the night is root LMR at MIN_IDX 9: −6 WAC, −26 STS, −2.8% nodes.** At EBF 1.72 a 2.8%
node saving is **~0.06 ply** — it cannot buy back 26 STS points. Expected to lose.

---

## 🔍 SOLID FINDING: root razoring is UNSOUND BUT PROFITABLE

Byte-identical audit. After a razoring iteration, does the NEXT iteration pick a move that had been razored
away?

| venue | iterations after a razor | winner had been razored | rate | avg depth past razor |
|---|---|---|---|---|
| WAC | 3,472 | 581 | **16.7%** | 8.9 moves |
| quiet corpus | 577 | 103 | **17.9%** | 4.4 moves |

**Consistent across venues** — the only measurement all session that did not flip with venue. **Ten times our
1.58% LMR wrong-reduction rate**, and a **LOWER BOUND** (a move razored at every depth never gets to prove
itself). The discarded winner sits 4–9 moves past the cut.

**Yet deleting the razor is clearly worse: +31% nodes at fixed depth.** ⇒ **its pruning value exceeds its
error cost.** Root LMR does not rescue it either (razor-off + LMR is the worst arm tested).
⚠️ Threshold tuning (`RAZOR_BASE` 200/450/600, `RAZOR_FLOOR` 200) was swept at fixed time and is
**UNREADABLE** — the arms span less than the base's own drift. **Re-sweep at fixed depth if pursued.**

---

## 🔑 STRUCTURAL CONTEXT (unchanged, still the best lead)

`alpha_beta`'s root loop applies **no LMR and no LMP** — it razors and `break`s, so ~23 of ~35 root moves are
never touched by the main search at any iteration.

- **SF11 → SF18, invariant across 7 versions:** shallow-depth pruning is gated `!rootNode` in both
  (SF11 L997, SF18 L1051); singular likewise. **Stockfish never prunes or abandons a root move.** SF11
  reduces at root with `moveCount > 1 + rootNode + (rootNode && bestValue < alpha)` and
  `(!rootNode || best_move_count(move) == 0)` — recently-best root moves are never reduced.
- **Ethereal:** *"No `break` or `continue` that abandons remaining root moves is present."*

⚠️ `break` vs `continue` is a **non-distinction for us**: the root list is score-sorted and the razor tests
`alpha - scores[i].top_score > razor_threshold`, so once it fires at *i* it fires for all *j > i*. **That is
why `ROOT_RAZOR_CONTINUE=1` is byte-identical.**

⚠️ **My root-LMR implementation is a simple index-based reduction, NOT SF's.** It does **not** implement the
`best_move_count` exemption (recently-best moves are still reduced). That is the most likely reason it costs
STS, and it is the obvious next thing to try if this lane is ever reopened.

---

## STATE

- **HEAD `0f4edc8`**, branch `NN-ENgine`, **nothing committed.**
- **Byte-id with all gates off: WAC 249 / 38,840,709 / EBF 3.934 / STS 1662** — verified after every build.
- Working tree adds (all default-off): `ENABLE_ROOT_LMR` + `ROOT_LMR_MIN_IDX/BASE/DIV`, `ENABLE_ROOT_RAZOR`,
  `PRESEARCH_TAIL_MODE` {0/1/2/3} + `PRESEARCH_CHUNK` + `PRESEARCH_TAIL_REDUCTION`,
  `ENABLE_PRESEARCH_SUBSET`, `ENABLE_ROOT_SORT_SPLIT` + `sortSearchDataRange`, the latent-UB fix, the
  const-ref, and cold counters (`draw_empty_ply1`, `tail_mode`, `razor_audit`, `root_lmr`).
- ★★ **RESULT IN — SPRT `ENABLE_ROOT_LMR=1 ROOT_LMR_MIN_IDX=9` vs base, tag `sprt_rootlmr9`:
  819 games, W 303 / L 292 / D 224 = 50.67% ⇒ +4.7 Elo, 95% CI [−15.6, +25.0] (±20.3).**
  Scored directly from the PGNs (`[White]` header per game + `[Result]`), so the `tournament` sign trap
  does not apply. **LEVEL, not negative** — and this is the second-tightest interval in the whole ledger.
  Launched with a deliberately poor prior after the fixed-depth bench said **−6 solves / −26 STS**.
  ⇒ **The bench was wrong about the sign.** Root reductions had never been tested with games in this
  engine; the one time we did, the bench-based pessimism did not survive.
- Harness fixes landed in `overnight_runner.sh`: `wac`/`sts`/`wac_timed` now put caller knobs **after** the
  defaults so `PRESET`/`MAX_DEPTH` overrides actually apply (they were silently discarded before).
- `depth_nps_bench.py` now takes a fixed-seed random sample (`--seed`, default 1234); the old head slice was
  **14% biased toward cheap positions**.

## NEXT

1. ✅ **SPRT read: +4.7 ±20.3 over 819 games. Root LMR does NOT close** — it is the only search knob in
   two sessions to come back non-negative on games. It is not a ship either (the CI straddles zero and a
   +5 candidate needs ~4000 games to resolve), but it is the **first live lead the search lane has had.**
2. ★ **Add SF's `best_move_count` exemption and re-run** — `(!rootNode || best_move_count(move) == 0)`,
   i.e. **never reduce a root move that was best in a recent iteration.** Our root LMR omits it entirely.
   This is the single highest-value follow-up: it is the guard SF considered necessary, it is cheap, and
   we are measuring level *without* it. Re-judge with `gate`, not a bench.
   ⚠️ **Do NOT tune `ROOT_LMR_MIN_IDX` on the fixed-depth STS sweep** — that curve has 100-point cliffs
   between adjacent values (1518/1506/1581/1569/1632/1636/1532/1527 for 3–12) and it just mispredicted
   the sign. If the exemption version needs a second value, spend the games.
3. **Do not trust bench deltas below ~125 STS points at fixed time, or single knob values at fixed depth.**
   Plateau-check everything; repeat the baseline in the same batch.
4. **Eval remains the better bet** — 5 ships / +83.4 Elo and ~18pp of classical headroom, versus a search
   lane now 0-for-many across two sessions.

## ☠️ RETRACTED THIS SESSION (cumulative)

| claimed | reality |
|---|---|
| counters cost 30% via a hot-path global | mechanism falsified; ~28%, cause unproven |
| NPS gap was WSL boot / power plan / GPU / OneDrive | all four wrong — instrumentation + stderr target |
| const-ref would buy speed | null |
| tail skip costs nodes, lane dead (WAC) | reversed on quiet corpus |
| closed on fixed-depth STS (−98) | sign reversed at fixed time (+22) |
| warming is load-bearing | score is 73%, warming 27% |
| tied fill scores are a bug | "fix" measured worse; earlier numbers stand |
| tail C4 is a tactical↔positional trade (+22 STS) | gaming-load artifact; −58 on an idle machine |
| base wins on both axes at fixed time | **unproven — 125-point baseline drift** |
| root LMR is a loser (−6 solves, −26 STS) | **819 games: +4.7 ±20.3 — LEVEL. The bench had the SIGN wrong.** |

**Ten retractions, all caught by measurement.** The durable lesson is in the instrument section above,
and the last row is its sharpest instance: **a deterministic, bit-reproducible bench predicted the wrong
sign of a game result.** Fixed-depth WAC/STS reproducibility is not accuracy — it measures one arbitrary
slice of a chaotic surface. **Games decide; benches only triage.**
