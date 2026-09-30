# HANDOFF — 2026-09-30

This phase shipped the second Texel eval win, built three new king detectors plus POT's endgame half (winnability), and
learned how fitted changes fail at playing depth. The re-priced king safety shipped at **+30 Elo vs SF18 at 250k nodes**,
about +22 in self-play. Everything else built this phase stays at weight 0 until it passes the depth gate.

- **Fit K (Texel: PSTs + non-linear KS + C3 detectors, nested arms).** The bundle won self-play (+34) and was +41 vs SF18
  at 50k nodes, but flat at 250k (−11). A per-part ablation found that the KS part alone transfers (+30 [+3, +59]); the
  re-fitted PSTs and the C3 detectors cancelled it. **Shipped: the KS knobs only.**
- **Fit K2 (KS structure: attacker weights, coordination, x-ray, defence-aware, gate): null.** The KS block is done for now.
- **C3 detectors (shelter/storm, pawnless flank / king–pawn distance, KingProtector):** built, verified, at 0. Their
  d6-fitted values read −22 [−49, +7] on fresh seeds.
- **POT (Potential) — "OvD reworked" (the owner's name, 09-29).** The endgame half (winnability) has a strong static
  signal, but both fitted versions lost ~−23..−25 Elo at depth: one fitted to d6 outcomes, one to SF18 labels with a cap.
  The cause was diagnosed: the additive `sign(T)·C` form is discontinuous at a level score. The middlegame transformation
  features showed no signal vs SF18 search.
- **Baseline snapshot:** accuracy, STS and WAC for v1, v2-before and v2-now, with the references. The v1 collapse revisit
  shows v2 fixes a large share of v1's classic failures.
- **101 commits unpushed** on `NN-ENgine`, no footers. **Nothing is running.**

## ⚠️ READ THE HONESTY FIRST — errors that reached reports or committed text this phase

- **"KS is depth-fragile" (09-28) was WRONG.** I explained K1p's flat 250k result as tactical terms priced for shallow
  play, and told the owner so. The KS-only arm then read +30 at 250k. The drag was the positional parts fitted with it.
  Memory `fit-data-depth-must-match-play-depth` was rewritten. ★ Lesson: run the per-part ablation BEFORE telling a story.
- **"K1p loses QUIET games (−14.6pp)" was regression to the mean.** The classes were defined by the baseline game's own
  outcome. A near-null control arm shows the same −10.7pp. Retracted (C3 doc §9b).
- **My first "sharp" classifier was degenerate:** 995 of 1,000 games fell into one class. Caught before reading it.
- **I paired new candidates against the ship-selection seeds (36/37).** The shipped config was chosen partly on those
  games, so every candidate reads biased negative. Caught when every seed-36 arm read negative. Memory
  `gate-new-candidates-on-fresh-seeds-not-ship-seeds`. All later gates use fresh seeds with a fresh baseline.
- **Fit W's first regularisation was ~10× the effect.** It pinned the weights near 0, and the UNFITTED SF prior beat it.
  Caught by comparing against the prior.
- **Winnability "better held-out, worse games": the form was discontinuous.** I ran two fits and two gates before
  measuring the form around T = 0 (`_win_amplify_check.py`: 98% of near-level endgames amplified to ±½ pawn).
- **Smaller defects:**
  - The runner `label` sub is a different labeller; my call errored harmlessly.
  - The runner `ps` sub missed `vs_sf.py`. Fixed in `d0bca9e`.
  - `engine_server.py` still filters `[c3]` load confirmations. Load errors (☠) do get through, so a failed load is
    visible. Adding `[c3]` to the forwarded prefixes was promised and is NOT done.
  - A fitter unpack bug and a dense 1.15 GB matrix were both caught in the smoke run.
  - I wrote "he" for the owner in a memory; fixed to "they".
- **Registered predictions:**
  - KS channel screen: 1/4.
  - Move-class screen: 0/4 outright.
  - K2: 1/5.
  - Winnability SF gate: predicted positive, read negative.
  - The KS-only +30 surprised me: I had predicted the bundle, not KS, would carry it.
- ★ **The pattern this phase:** a fitted change can win every static and 50k-node instrument and still lose at depth. And
  a class defined by an outcome, or a baseline selected by its outcome, manufactures effects. Every catch came from a
  CONTROL: a per-part arm, a null arm, a fresh seed, or the unfitted prior.

**Mode:** execution, interactive.

## ▶️ FIRST ACTIONS IN THE NEW CHAT, in this order

1. **Confirm nothing is running or orphaned.** Nothing should be. The last job, `gauntlet_fitWsf_s41`, finished.
   - Run once: `wsl.exe -e bash -lc "bash '/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine/selfplay/overnight_runner.sh' ps"`
   - Expect `none running`. If anything is listed, inspect it before launching anything; the owner did not start it.
2. **Re-verify the fingerprints** (the build under the working tree is the current one):
   - `… overnight_runner.sh' wac fp_v1` → expect **250 / 35,310,778 / 3.784**;
   - `… overnight_runner.sh' wac fp_v2 V2_PRESET=shipped` → expect **252 / 49,094,807 / 4.012**.
3. **Then start step 1 of NEXT STEPS** (the winnability re-form). It needs no owner input to begin the research. The POT
   middlegame re-think needs the owner present.

## Transfer documents, in reading order

1. memory `eval-v2-rebuild-state.md` — ★ the entry point; its TOP block is this handoff.
2. `dev_notes/TEXEL-C3-DETECTORS-DESIGN-2026-09-27.md` — ★★ the current work. Read **§15 (state and decisions) first**, then:
   - §8, 8a–8c: the POT design, from the OvD redesign;
   - §9, 9a, 9b: Fit K, the per-part ablation and the ship;
   - §10, 10a: Fit K2 (null);
   - §11: the baseline snapshot, then POT's mg pilots;
   - §12: winnability (build, fits, gates, discontinuity diagnosis);
   - §13: the POT name and lineage;
   - §14: the SF-label round and the owner's no-overlap condition.
3. `dev_notes/EVAL-V2-CURRENT-CONFIG.md` §1 — the shipped block. Mirrored in three places that must stay in sync: the
   `V2_PRESET=shipped` block in `search_engine.cpp`, the runner's `V2=` line, and this doc.
4. `dev_notes/INSTRUMENT-MAP.md` — the end section "2026-09-28/30 — new ways instruments misled" (8 items).
   `dev_notes/DIAGNOSTICS-TOOLKIT.md` — the table "Added 2026-09-28/30".
5. Still load-bearing:
   - `dev_notes/SESSION-HANDOFF-2026-09-25.md` and the 09-27 handoff (below, via memory);
   - `dev_notes/TEXEL-FIT-C-DESIGN-2026-09-27.md` (Fit C term inventory, C1 null);
   - `dev_notes/EVAL-V2-INVENTORY-2026-09-25.md` (the audit that reopened the eval; Fit A);
   - `EVAL-V2-PARKED-REGISTER.md` and `CORPUS-CHARTER.md`.

## Method memory — read before designing anything

- [[fit-data-depth-must-match-play-depth]] — ★★★ gauntlet EACH PART of a joint fit at ≥ 250k nodes (and my retracted
  reading).
- [[gate-new-candidates-on-fresh-seeds-not-ship-seeds]] — ★★★ fresh seeds + fresh baseline; next unused seed: **42**.
- [[ovd-is-the-owners-long-term-pressure-concept]] — ★★★ now **POT**: status, owner decisions, the **no-overlap
  condition**, and the lineage requirement for canonical docs.
- [[owners-engine-origin-article]] — the owner's origin article is readable via the Medium RSS feed; the heat map is
  OvD's root. Read it once for POT's intent.
- [[unique-where-better-never-self-nerf]] — the owner's design rule, including the 09-28 scope note on "never loosen": it
  binds hand-tuning; a fit-moved threshold is flagged, a pinned control is kept, and games decide.
- [[texel-pst-fit-is-the-biggest-eval-win]] · [[ks-recall-is-structural-and-firing-does-not-discriminate]] ·
  [[the-fifteen-nulls-headline-mixed-three-classes]]
- [[sprt-point-estimates-inflate-at-the-bound-they-stop-on]] · [[the-tuning-objective-is-staged-and-criticality-weighted]]
- [[eval-headroom-is-failures-that-persist-as-depth-rises]] · [[a-detector-gate-passes-vacuously-unless-the-term-is-proved-to-fire]]
- [[baseline-fingerprints-register]] · [[wsl-concurrency-crashes-the-box]] (the 09-29 restart event) ·
  [[a-handoff-loses-the-task-timer-relaunch-a-watcher]] · [[dispatcher-prompt-free-wrapper]] · [[no-commit-footer]]

## What this is

- A non-negamax C++ HCE in `NN Engine/`:
  - separate minimizer / maximizer / qSearch functions;
  - the v2 eval (`eval_v2.cpp`) is Black-positive, in millipawns, driven through Cython (`ChessAI.pyx`, a remnant of
    the owner's original Cython engine) under WSL.
- Roadmap: the strongest single-threaded HCE (target 3000+ CCRL), then the owner's own NN taught by it. That completes
  the owner's 2017 AlphaZero-inspired ambition; the original engine was a policy-filter + shallow-search + judgement-eval
  hybrid.
- ☠️ The eval is not "replaced" by the NN. Search and movegen carry over mechanically; the eval carries over
  informationally, as the teacher.

## 🚨 BASELINES — reverify every build (MAX_DEPTH=10 USE_OPENING_BOOK=0 PRESET=LONG_FORMAT)

| arm | WAC d10 |
|---|---|
| v1 (frozen control) | 250 / 35,310,778 / 3.784 |
| **v2 shipped (`V2_PRESET=shipped`)** | **252 / 49,094,807 / 4.012** (was 249 / 53,405,821 / 3.973 before the 09-29 KS ship) |

- ☠️ `EVAL_ARM=1` alone is a skeleton. Quote NPS or time, never nodes alone.

## ▶️ THE MEASURED PICTURE

**Shipped 09-29: Fit K KS knobs** (the preset now carries WEAK 64 · ADJ 50 · NO_QUEEN 402 · CHK Q/R/B/N 260/193/141/189 ·
HALF 646 · ONSET 450 unchanged, plus balance channels ADJ_INST −12 · UNSAFE 19 · FLANK_ATT 11 · FLANK_ATT2 −1 ·
KNIGHT_DEF 15 · CONTEST_EXCESS 14 · CONTEST_SQ 30 · CONTEST_SQ_Q 19):

| check | result |
|---|---|
| vs SF18 @250k, 1,000 paired | **+30 [+3, +59]** |
| SPRT @50k | H1, +20.5 ± 19.0 / 1,768 |
| replication, 2,000 fixed | +23.8 ± 17.9 (pooled ≈ +22) |
| variant starts | +12.9 [−0.7, +26.5]; queen families +, queenless lean − |
| odds vs SF18 | neutral (−0.95 ± 3.5) |
| game types (vs a control arm) | sharp games +4.8pp; largest where king danger arises; none hurt |
| v1 collapse positions | largest regret cut on KS-attack collapses |
| firing | danger magnitude now tracks outcome (r ≈ −0.05 vs ≈ 0 before) |

**Per-part ablation @250k, 1,000 paired each:**

| arm | result |
|---|---|
| K1p full | −11 |
| K1p, KS change halved | −12 |
| PST + C3, shipped KS | −11 |
| C3 only | −8, later −22 on fresh seeds |
| **KS only** | **+30** |

**Baseline snapshot (C3 doc §11):**

| evaluator | accuracy (own-play / diverse) | STS d10 / @249k | WAC |
|---|---|---|---|
| v2 now | 168.08 / 124.18 | 1891 / 1734 | 252 |
| v2 before the KS ship | 170.17 / 126.93 | 1888 / 1760 | 249 |
| v1 | 188.85 / 238.77 | 1796 / 1752 | 250 |

- References, same rows: SF18 static 61.9 / 68.9 · SF15 NNUE 72.8 / 61.6 · SF11 151.4 / 95.3 · SF15 classical 192.4 / 139.2.
- v2 is past SF15 classical on both corpora.

**POT (OvD reworked):**
- **Winnability static signal:** low-complexity endgame edges are over-rated by ≈ 8 points of score.
- **Fit W (d6 outcomes):** endgame held-out −1.4%, but **−25 [−51, +3]** on fresh seeds.
- **Fit W-SF (SF18 labels, ½-pawn cap):** val MSE vs SF −8.4%, but **−23 [−49, +5]** on fresh seeds.
- **Discontinuity:** W-SF amplifies 98% of near-level endgames to ±500 mp.
- **mg features vs SF18 search:** lever / tension / majority / projected inputs all null. The only signal (−4.3σ): we
  over-rate the leader's mg edge when passers are on the board. That belongs to the passer terms.

**K2 KS structure:** no arm beats the base control on both holdouts.

## ▶️ NEXT STEPS (the order agreed with the owner, 09-30)

1. **Rebuild POT winnability's FORM from the references.** Use an Opus `engine-contrast` agent to extract the exact forms
   (inputs, phase leg, sign / continuity handling, scale-factor interplay) of:
   - SF11 `initiative`;
   - SF12+ / SF15.1 `winnable` (which folds in the scale factor);
   - Ethereal `evaluateComplexity`;
   - Weiss `ScaleFactor`.

   Then:
   - Design a CONTINUOUS form. The likely candidate is a multiplicative eg scale factor, E_eg = T·f(C) with f ≈ 0.5–1.2.
     Note that v2 ships in non-pair mode; `EVAL_V2_PAIR` exists (bound 3 mp) if a true eg leg is needed.
   - Check it with `_win_amplify_check.py` BEFORE fitting.
   - Fit on `ks_sets/fitC_eg_sf18.csv` (and compare with outcomes).
   - Run closure, symmetry and fingerprints.
   - Gate: SF18 @250k, 1,000 paired, on **fresh seeds 42 + 43** with a fresh baseline.
2. **POT middlegame — re-think the ALGORITHM with the owner** (owner: "it's never been seen before, so consider what we
   have, what's good, what's not"). The concept stands; the tested features are null.
   - Owner's framing: opening/middlegame transformations that carry a position into a strong middlegame and a winnable
     endgame; piece and king placement matter for how they affect transforming the position, not KS; the variant
     corpora supply the unusual structures.
   - Under the no-overlap condition: an owner per feature, a collinearity check (`_pot_mg_screen.py` style) before code,
     incremental value only, few features.
   - The mg SF18 labels (`fitC_mg_sf18.csv`, incl. 4,944 variant rows) are ready. The variant rows need an engine pass
     for our eval.
3. **Passer re-tune** — Texel, the same method as PST / KS, starting from the lead above. Gate each part at depth on
   fresh seeds.

**Later:**
- the C3 detectors re-priced on SF18 labels (or folded into POT);
- the `[c3]` log prefix in `engine_server.py`;
- the RFP / margin re-sweep at the end of the eval work (owner's call);
- the search block retuned for v2 at equal time (node-TT, TT-move, IIR, singular, ProbCut, qsearch, margins);
- the §I-only rejections;
- the final joint retune;
- the NPS rewrite decision (movegen is 41% of node cost);
- then the owner's NN;
- a CRITICAL-position corpus, to answer "does it upgrade critical positions?" (the regret set has 2–27 critical changed
  moves per arm: unreadable).

## 🎯 THE REFRAMES (load-bearing)

- **The gate protocol, in order:**
  1. closure + symmetry + byte-identical fingerprints;
  2. SF18 gauntlet @250k, 1,000 paired, on FRESH seeds with a fresh baseline (`_gauntlet_pair.py`);
  3. if it's a bundle, a **per-part** ablation;
  4. SPRT @50k + a 2,000-game fixed replication;
  5. variants (pair view) + odds vs SF18;
  6. the owner ships.

  The owner wants "not overly cautious — a good balance". Fit A's path is the bar.
- **Re-pricing broad, existing terms transfers** (Fit A PSTs, KS). **New narrow corrections priced from d6 outcomes do
  not**, so far: C3, Fit W, the K1p positional parts. New terms need depth-independent labels, sane magnitudes and
  continuous forms.
- **A leader-relative term must be continuous at T = 0.** `sign(T)·C` manufactures edges from noise exactly where search
  lives.
- **Outcome-defined classes and outcome-selected baselines regress to the mean.** Always read them against a control
  arm or on fresh seeds.
- **An SPRT decides; a fixed-length pool counts magnitude.** Self-play inflates versus external play (≈ 3× for Fit A, ≈ 1×
  for KS).

## ★★★ OWNER CALLS (this phase)

- **Skip the move test for built KS detectors;** keep it for new concepts. Fit + games decide.
- **"Never loosen thresholds"** binds hand-tuning. A fit-moved threshold is flagged, an onset-pinned control is kept, and
  games decide. K1p vs K1p-pinned were equal, so the pinned form shipped.
- **OvD reworked is named POT (Potential).** Always credit OvD's lineage in canonical docs.
- **No forced overlap** ("we may end up with a mess like v1 again"): one owner per concept, collinearity before code.
- **POT mg: re-think the algorithm.** POT eg: learn from SF11 / SF15 / Ethereal / Weiss. Then passers.
- **Fit on the variant corpora too:** unusual structures teach transformations, not memorised patterns.
- **Ship bar:** balanced; SPRT + replication + external.
- **Use Opus agents for research. The GPU has 8 GB.**

## ⚠️ WHAT IS ALREADY THERE

**Knobs** (all byte-identical at their defaults unless noted):
- `PST_V2_TAPERED` (2 = shipped) / `PST_V2_FILE`.
- `C1_V2_FIT`.
- KS modes `KS_V2_ATT_XRAY` / `PIN_DEF` / `GATE` / `DEFAWARE`.
- KS balance weights, now **shipped non-zero**.
- **`KS_V2_W_N/B/R/Q`** (attacker weights 31/31/47/78).
- **C3:** `KSB_V2` (+ `KSB_V2_CASTLE`), `KFL_V2`, `KPROT_V2`, each with a `*_FILE`.
- **POT winnability:** `WIN_V2` + `WIN_V2_{PASSED,PAWNS,OUTFLANK,INFILT,FLANKS,PAWN_END,UNWIN,BASE}` + `WIN_V2_CAP`.
- The `ev_breakdown` keys `v2_shelter` / `v2_kflank` / `v2_kprot` / `v2_winnab` are published only when their term is on.

**Probes:**
- `v2_feature_counts` / `v2_feature_theta`: **184 per side** (C1 0–105; C3 cells 106–183).
- `ks_counts`: 35 keys (per-type safe checks, per-type attackers / x-ray / contested share, `w_att_x`).
- `win_inputs`.

**Tools:** the toolkit table "Added 2026-09-28/30":
- the Texel pipeline: `_texel_extract` → `_texel_engine_pass` → `_texel_feature_pass` → `_texel_ks_pass` →
  `_texel_k_fit` / `_texel_k2_fit`;
- POT: `_texel_win_pass` / `_texel_win_fit` / `_texel_win_sf_fit`;
- oracles: `_c3_oracle`, `_win_oracle`;
- gating: `_gauntlet_pair`, `_gauntlet_gametype_split`, `_odds_pair_from_log`;
- labelling: `_build_regret_set IN=`;
- screens: `_pot_mg_screen`, `_ovd_*_proto`, `_ks_fire_compare`, `_win_amplify_check`;
- `_watch_run` (the timer after a handoff).

**Data** (`E:/chess_data/texel/`):
- `fitC_stage1.csv.gz` (1,844,613 quiet rows from 30k d6 games);
- `fitC_features.npz`;
- `fitC_ks.npz` (old KS) / `fitC_ks2.npz` (new KS, per-type);
- `fitC_pass/zero_0_of_1.csv`;
- `fitC_win.npz`;
- the Fit A / K / W outputs (`pst_fitA*`, `*_fitK1p*`, `ks_fitK2_*`, `win_fitW*`);
- `v2_variant_stage1.csv.gz`.

**`diagnostics/ks_sets/`:**
- `fitC_eg_sf18.csv` (19,779 eg rows, SF18 d14);
- `fitC_mg_sf18.csv` (14,842 mg rows incl. 4,944 variant) + their samples;
- `collapse_regret_set.csv` (v1's 276 collapses, SF18-labelled).

**Gauntlet baselines:**
- shipped on seeds 38–41: `gauntlet_ship2_s38/39`, `gauntlet_ship3_s40/41`;
- Fit A config on 36/37 (`gauntlet_fitA[_s37]`, `gauntlet_shipped_0928`);
- KS-only (= today's ship) on 36/37 (`gauntlet_K1p_ks[_s37]`, selection seeds: do not pair new candidates against them).

## STATE

- HEAD `d0bca9e` on `NN-ENgine`: **101 commits unpushed**, no footers. The tree is clean apart from the ChessUI
  submodule and old untracked files.
- **Nothing running.**
- MEMORY.md is ~17.4 KB (at the limit; prune before adding).
- To play in the UI: `V2_PRESET=shipped PRESET=LONG_FORMAT python ChessUI/chess_ui_v2.py`.

## DISCIPLINES

- ★★★★ **A silent tool reads as a result.** Look for the signatures: identical rows, "no rows", a byte-identical control,
  an implausible rate. Prove a knob fires: log the knob, check the games diverge from the baseline, and run a closure
  check.
- ★★★★ **A CONTROL for every split and every gate:** a null arm, a per-part arm, fresh seeds, the unfitted prior.
- ★★★★ **Validate at play depth (≥ 250k) and externally (SF18).** A 50k SPRT is necessary, not sufficient.
- ★★★★ **Pilot for feasibility; get power for verdicts. Register predictions before looking.** Ablate before explaining.
- ★★★ **Operations:**
  - one engine-loading job at a time (≤ 4 games) after the 09-29 WSL restart; two jobs of 2 games each ran fine before
    it, cause unknown;
  - never rebuild under a job that runs from the working tree;
  - each WSL job as its own auto-approved runner call;
  - no nohup;
  - write files with Write;
  - move anything worth keeping out of the scratchpad (it dies with the session).
- ★★★ The owner games ~9pm–midnight: node-limited work is fine then.
- ★ Commit by feature with no footer; push only on the owner's say-so.
- ★ Neutral pronouns for the owner in all written text.

## ★ LOAD-BEARING OWNER CONTRIBUTIONS (this phase)

- **"Just go straight to games"** → skipped the KS move test; fit + games decided.
- **"Not overly cautious, a good balance"** → the ship bar; the pinned arm was a fallback, not a veto.
- **"Why the change vs earlier ships?"** → clarified that the pipeline = Fit A's path; only the pinned control was new.
- **"KS is low-fire but should decide critical moments"** → the game-type split (sharp games +4.8pp) and the
  critical-corpus need.
- **"Maybe it's not tuning all of KS"** → the depth reframe: position-level info that search cannot see is what transfers.
- **"The collapse corpus"** → the v1-collapse revisit: v2 fixes a large share; KS best on KS collapses.
- **"Baselines + MSE vs SF18"** → the snapshot table, and the before/after for every future ship.
- **The OvD history and the Medium article** → POT's naming, its lineage requirement, and the origin record.
- **"Don't force overlap"** → the no-overlap condition.
- **"The mg algorithm may just not be optimal"** and **"learn from SF / Ethereal / Weiss for winnability"** → the next steps.
- **Whacky corpora for transformation understanding** → the variant mg labels.
