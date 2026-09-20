# CORPUS CHARTER — what each position set is FOR, and what may never be done to it
★ written 2026-09-20, recovering the agreed method from the record rather than re-deriving it

The owner's rule, stated 2026-09-20: *"be clear on how things are done, and if different methods are done,
split the sets so one type doesn't contaminate another."* The record already agrees — this writes it down.

---

## ☠️ THE OBJECTIVE QUESTION IS OPEN. DO NOT IMPROVISE IT.
The record contains **TWO** answers, both recorded as settled, never reconciled:

| | proposal | status |
|---|---|---|
| **A** | **d7 low-depth win%-regret vs SF18 multi-PV.** *"Static corpus is the cheap coarse region-finder BUT it's anti-correlated with Elo ⇒ do NOT fit constants on it; use it only to seed a sane region. D7 low-depth regret is the live instrument."* (08-11) | BUILT (`_regret_tune.py`, `_regret_tune_broad.py`), used for VALIDATION, **never used for a joint constant fit** |
| **B** | **Corpus-shape fit with the GLOBAL SCALE PINNED** + per-subsystem scale constraints + acceptance by games (09-18 tuning-night conditions) | PROPOSED ONLY — no data generated, no fit run |

⇒ **Ask the owner which one tuning night uses.** ★ It does not block data collection: **both need SF18 d14
labels on real-play positions**, and the record says *"generate the DATA early, run the FIT last — data is
reusable, doubles as NNUE training data."*

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
