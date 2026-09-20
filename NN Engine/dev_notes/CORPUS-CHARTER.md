# CORPUS CHARTER — what each position set is FOR, and what may never be done to it
★ written 2026-09-20, recovering the agreed method from the record rather than re-deriving it

The owner's rule, stated 2026-09-20: *"be clear on how things are done, and if different methods are done,
split the sets so one type doesn't contaminate another."* The record already agrees — this writes it down.

---

## ★★★★ THE OBJECTIVE — ANSWERED BY THE OWNER 2026-09-20: IT IS STAGED
The record held two unreconciled proposals (d7 regret; pinned-scale corpus fit). ⇒ **Neither alone.**

**Corpus:** *"an overwhelmingly large set of positions that are representative of all sorts of occurrences,
balanced across all types of phases and situations."*

| stage | objective | purpose |
|---|---|---|
| **1** | **ABSOLUTE** tuning | get to a reasonable point — keep the 09-18 constraints: **pin the global scale**, per-subsystem scale constraints, **acceptance by GAMES** |
| **2** | **CRITICALITY** | high-impact scenarios, read on **live differentiating sets** per epoch |
| **3** | **D7 REGRET** | *"to ensure move choice is actually improving"* |

★★ **THE ARGUMENT IS THE SF15-vs-SF11 CASE.** SF15-classical has a WORSE static MSE vs SF18-search than
SF11, yet is believed the stronger eval. ⇒ *"it's not just the absolute MSE, but also in how many critical
positions it changes the values considerably, rather than how many it changes towards the correct answer in
general."* ⇒ **a candidate with the same or WORSE MSE may still be better**, and a worse MSE must not
auto-veto. That is exactly why MSE-only acceptance failed 5/5: it optimises the average position, and games
are not decided there.

☠️ **THE REAL BLOCKER IS A DEFINITION, NOT COMPUTE.** *"We'd want to define in eval what a critical scenario
is."* Nothing in the record defines it; `n_crit` is 27/43 and §F already calls a criticality-enriched corpus
the PREREQUISITE for any claim about deciding positions. **Stage 2 cannot start until "critical" is defined.**

★ Owner explicitly opened this to research beyond chess: *"best ways to tune in scenarios like this … in
data engineering in general where we have this sort of evaluation and search synergy … maybe even some RL
type algorithm."* ★★ And: *"that optimization strategy may also be transferrable to search parameters when
that time comes"* ⇒ **the tuning METHOD is a deliverable, not scaffolding.**
→ [[the-tuning-objective-is-staged-and-criticality-weighted]]

---

## ☠️☠️ OBJECTIVES THAT ARE CLOSED — a bigger corpus does NOT reopen them
| objective | scale it failed at | verdict |
|---|---|---|
| static win%-MSE vs SF18-search | 23,113 rows, 63 knobs, val −41 | **games −85.6 ±74.3** |
| SF11-total MSE | 37k | railed to a global shrink; lost to a plain global scale |
| asymmetric SF11/SF18 | 22,740 | −192 STS, move-match neutral |
| **outcome-Texel logloss** (`result_white`) | **183,574 rows / 16,659 games, by-game holdout** | **FLAT — control 0.13201 → 0.13198 (0.02%)**, train moved 15× more |

★★ **`val/train ≈ 0.99` at 8.5× the data ⇒ this was NEVER an under-fitting problem.** The mechanism:
*"corpus loss optimises SCALAR eval fit; Elo comes from MOVE ORDERING; the two are DECOUPLED — no
corpus/target/weighting fixes it."* ⇒ **more positions, more variety, and more augmentation are all the
wrong axis.** ⚠️ Do NOT let "but this corpus is bigger/cleaner/more varied" be the argument for a retry.
⚠️ One genuine reason to expect a different result: **v2 removes both structural causes** — it is
non-degenerate (40-column collinearity gate) and colour-clean, which v1 was not.

---

## THE SETS, BY PURPOSE — never pooled, never concatenated

### ▶️ CURRENT REGRET-SET INVENTORY (2026-09-20) — one judge, mutually disjoint, VERIFIED
**Built 2026-09-20 from the NESTED (v2-era) layout, which had never been sampled:**

| file | rows | phase mix (eg / mg / op) |
|---|---|---|
| `game_regret_set_v2era.csv` | **10,000** | 2,133 / 4,626 / 3,241 |
| `game_regret_set_v2era_b.csv` | **12,000** | 2,615 / 5,602 / 3,783 |
| `game_regret_set_v2era_c.csv` | **11,174** | 2,226 / 5,477 / 3,471 |

✅ **33,174 positions, verified disjoint from EVERY OTHER regret set on disk** (not just from each other).
SF18 multi-PV top-8 @ d14, `PLY_STRIDE=7`, columns `fen / phase_bucket / best_uci / best_cp / moves`.
★ Three independent sets ⇒ tune / holdout / replication without building anything further.
⚠️ The 3rd returned 11,174 of 14,000 requested ⇒ the nested pool is **essentially mined out at stride 7**.
More needs new GAMES, or a different stride (which is a different METHOD ⇒ a separate set, per rule 1).

### ☠️☠️ A DELETED SET, AND THE LESSON THAT COST IT
`game_regret_set_0920.csv` (8,000 rows) was built before the exclusion glob was fixed and turned out to be
**100% contained in `game_regret_set_x4.csv`** — an hour of SF18 labelling for ZERO new positions. Deleted.
☠️ **And the verification that was supposed to catch it did not, because its SCOPE was an assumption.**
The first disjointness check compared the sets *I knew about* and reported "45,000, all disjoint". Measuring
against **every `game_regret_set*.csv` on disk** immediately exposed the 8,000-row duplication.
★ ⇒ **VERIFY AGAINST THE FILESYSTEM, NOT AGAINST YOUR LIST.** A measurement scoped by an assumption
inherits that assumption, and reads as confirmation. The correct check is
`distinct(union of glob(*)) == sum(len(each))`.
⚠️ Pre-existing overlaps that are BY DESIGN, so they do not indicate a defect: `game_regret_set_v2_s0..s3`
are 3,000-row shards of `game_regret_set_v2.csv`.
⚠️ **The phase mixes DIFFER** (opening 3,241 vs 1,913) — independent evidence that the nested tags are a
genuinely different game population, not more of the same. ⇒ do not pool the three into one file; that
would make the mix ratio an invisible knob (hygiene rule 1).

### 1. PLAY-DISTRIBUTION — the ONLY sets a weight fit may see
- ★ **`ks_sets/game_regret_set*.csv` — "THE set to use."** Real self-play positions, `PLY_STRIDE=7`,
  **SF18 multi-PV top-8 @ depth 14** cached. Builder `_build_game_regret_set.py` (resumable; dedupes and
  excludes the move-match validation sets and existing regret/tune sets ⇒ leak-free by construction).
- `selfplay/tune_data/*.csv` — per-term + outcome corpora from stored games.
- ⚠️ **`selfplay/tune_data/v2_corpus_0920.csv` (46,240 rows, 6,252 v2-era games)** carries SF11 per-term
  columns and `result_white`. ☠️ Its outcome labels target objective (d) above, which is **already measured
  flat** — so treat it as a **TRIANGULATION substrate** (ours-vs-SF11 per term on v2's own distribution),
  **not** as a tuning set.

### 2. SYNTHETIC / CLASS-TARGETED — instruments only, NEVER weights
Random legal positions by material signature (`_draw_oracle.py EMIT=` / `EMIT_EPD=`), the tier-2b and
headroom suites, 960/variant sets.
☠️ **Explicitly prohibited from the tuning corpus:** *"Generalization/diagnostic + targeted signal ONLY …
do NOT blend them into one pot (the mix ratio becomes an invisible knob) … Do NOT static-fit on it"*, and
*"tuning on it chases 960 quirks and destroys the independence."*
★ Their legitimate use is exactly what they did on 09-19: build a class the real corpora do not contain
(1 in 23,113 for tier-2b), so a detector gate is not vacuous.
⚠️ `variant_regret_set.csv` is 6,000 rows but **97% opening** — needs `WALK_MAX≈45` regen before it is a
usable validator.

### 3. REFERENCE-LABELLED — ground truth for triangulation, not for fitting
SF11 per-term columns (`sf11_*`), Lichess tablebase labels (`tablebase_labels_draw.json`), SF18 static.

---

## ☠️ HYGIENE RULES — each one has a scar behind it
1. **Keep differently-generated sets SEPARATE.** Pooling makes the mix ratio an invisible knob.
   ⚠️ Specific: *"`diverse_corpus_wide` is d13 single-PV — do NOT pool with the d14 multi-PV regret sets."*
2. **ONE JUDGE PER TARGET COLUMN.** *"THE JUDGE IS THE TARGET COLUMN — never change it incrementally.
   SF19 must NOT relabel or extend an existing corpus. Any corpus enlargement uses SF18 @ d14."*
3. **Corpus composition decides the optimum** ⇒ re-derive every optimum after ANY corpus change, and
   **snapshot before regenerating**. Absolute val numbers are NOT comparable across corpora.
   ★ Corollary applied 09-20: extend into a NEW file; never grow an existing set in place.
4. **Batch corpus changes** — every change invalidates all prior optima, so dribbling costs a re-baseline
   each time.
5. **Seeded-shuffled split BY GAME, not by position.** An index split put different game POPULATIONS in
   tune vs held. *"Clean overfitting is usually a split artifact — check it."*
6. **Measure the null per corpus AND per stratum.** The regret mean's null is NOT zero and is arm-specific;
   read **win% of CHANGED moves** (null band 49.8-50.4 primary / 50.7 v2), real bar **~2-2.5pp**.
7. **Cross-set replication before anything is folded in.**
8. **Absent ≠ zero.** A gated-off term must write BLANK, never 0, or a fitter learns that a never-computed
   term is worth nothing. ★ Live example: `v2_rookfile` is 0-filled of 46,240 rows in the 09-20 corpus —
   correct, and it is the #1 item on the revisit shortlist.
9. ⚠️ **Any v1-era tool assumes v1's breakdown partition** and will crash or mislabel under `EVAL_ARM=1`.
   Three instances on 2026-09-19/20 alone. ☠️ Worst kind: v1 `phase_score` (0 = opening, 128 = endgame) and
   v2 `v2_phase256` (256 = opening, 0 = endgame) are **opposite and half-scaled** — aliasing them inverts
   every phase-conditioned reading and looks entirely plausible.

## ☠️☠️ SAMPLING PROVENANCE — TWO GAME LAYOUTS, AND THE SAMPLER ONLY SEES ONE (found 2026-09-20)
`selfplay/games/` contains **two directory shapes**:

| layout | path | matched by `_build_game_regret_set.py`'s default `GAMES_GLOB='*/*.jsonl'` |
|---|---|---|
| **flat** (older tags) | `games/<tag>/game_000.jsonl` | ✅ yes |
| **nested** (newer tags) | `games/<tag>/game_000/game.jsonl` | ☠️ **NO** |

★★ Every **v2-era** tag — the showdowns, `margin_ab_rfp1000`, the slice-2/3 runs — uses the NESTED layout,
so it has **never been in the regret sampler's pool**. The standing regret sets were therefore drawn
entirely from flat-layout games ⇒ **the tuning distribution is skewed toward PRE-v2 play.**
⇒ Use `GAMES_GLOB='*/*/game.jsonl'` to reach them. ⚠️ Any claim about the regret set being "the natural
distribution the engine faces" must say WHICH engine — as built, it is v1-era's.

### ⚠️ AND THE SAMPLER IS DETERMINISTIC — RE-RUNNING YIELDS NOTHING
`stride_f = max(1, len(files) // 2500); sel_files = files[::stride_f]` spreads over ~2,500 games by taking
every N-th file. The selection is **identical across runs**, so a second build with the same glob samples
the SAME files, and after exclusion returns **0 positions** — which looks like "the pool is exhausted" and
is not. The levers are `GAMES_GLOB` (different tags ⇒ smaller list ⇒ `stride_f` drops to 1 ⇒ every file
selected) and `SHARD=i/n` (disjoint partitions of one selection, for parallel labelling).
★ Guard added the same day: the exclusion list now GLOBS `ks_sets/game_regret_set*.csv` instead of naming
one file, so sets built this way are disjoint by construction and the pool size is printed.

## ON AUGMENTATION
Mirrors/flips are **exact symmetries of the target**, so they *"add no INFORMATION to a fit — they only
impose constraints"*, and `val/train = 0.988` says regularisation was never the bottleneck
(`diagnostics/_eval_symmetry.py` docstring). ⚠️ Recorded in TOOL SOURCE, not in memory/dev_notes — a
record search over docs alone will miss it. ★ The stronger form of the same idea is used instead: make the
fit *structurally incapable* of asymmetry (one table per piece, mirror Black at load).

## ★ WHAT WOULD BE GENUINELY NEW AT TUNING NIGHT
1. v2's eval — non-degenerate and colour-clean; neither prior structural cause exists.
2. **Global scale PINNED** + per-subsystem scale constraints (never done in any prior fit).
3. **Acceptance by GAMES only** — all five prior failures passed their own objective.
4. Fit run **LAST**, after the remaining slices, else invalidated by *value = constant × mechanism*.
5. Fit set = real-play distribution; wacky/960 and a criticality-enriched set **held out as validators**.
⚠️ Criticality enrichment is still UNBUILT (`n_crit` 27/43), and it is the stated prerequisite for any
claim about deciding positions.
