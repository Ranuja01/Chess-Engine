# SESSION HANDOFF 2026-09-09 — the heat map decomposed, three theses refuted, the gate repaired

## READ FIRST
📄 **[[eval-lane-state-2026-09-09]]** is the single source of truth for where the eval program stands.
This note is the run record behind it. Prior: `-09-07.md` (§4e/§4f = the 09-08 builds and screen),
`-09-06.md` (the eval lane priced), `-09-05.md`, `EVAL-TERM-REVIEW-TABLE-2026-09-08.md` (the term/duplication
table with line numbers — ⚠️ its clusters 1-2 verdicts are superseded, see §3).

## 1. ⭐ THE HEAT MAP IS LOAD-BEARING, CORRECTLY WEIGHTED, AND ITS CHANNELS ARE NOT COLLINEAR
`SCALE_ATTACK_LAYER` (`cpp_bitboard.cpp:9370-9381`) scales both heat planes at finalisation ⇒ a ONE-KNOB
ablation of the whole map. Win%, null band **49.8-50.4** (three neutral arms: `ASPIRATION_DELTA`
200/300/800 → 49.8 / 49.9 / 50.4):
| scale | win% | verdict |
|---|---|---|
| 0 | 48.9 (v2 49.4 vs its 50.7 null) | **−1.3pp, REPLICATED** |
| 50 | 49.4 | −0.5 to −1.0 |
| **100 (shipped)** | — | baseline |
| 150 | 49.9 | flat |
⇒ **monotone to 100, then flat — the magnitude sits AT the optimum.** A real plateau check, and rare here.
⇒ ☠️ **no cycles trade**: you cannot shrink the map to save the ~96 read sites without paying in accuracy.
⚠️ This measures the CURRENT build's DEPENDENCE, not the map's NECESSITY (owner's correction) — everything
downstream is fitted around it and OvD loses its only feeder. Treat 1.3pp as the SIZING BAR any replacement
must recover.

## 2. THE DECOMPOSITION (both corpora)
| arm | changed | primary | v2 |
|---|---|---|---|
| OvD off (`OVD_CAP=0`) | 15-20% | 50.4 (flat) | 51.3 (flat) | ✅ free, replicated |
| central off (`SCALE_CENTRAL=0`) | 30-34% | −0.2 to −0.8 | −1.6 (2σ) | ambiguous |
| both off | 31-34% | −0.6 to −1.2 | −1.2 | composes ADDITIVELY |
| ALL heat off | 39-44% | −0.9 to −1.5 | −1.3 | |
**Collinearity** (`_ks_channel_collinearity.py MODE=heat`, 744 positions, retargeted 09-09):
mean|contribution| heat **433.3mp** · central **215.9mp** · OvD **22.7mp**;
r(heat,central) **+0.389** · r(heat,OvD) **+0.420** · r(central,OvD) **+0.131**.
⇒ ☠️ **ALL under the 0.5 duplication threshold ⇒ the channels carry DISTINCT information ⇒ DE-DUP WOULD SHED
SIGNAL.** The consolidation thesis is refuted by its own designated test.
⇒ **OvD is free because it is TINY, not because it is redundant.** `central` is NOT the free re-sum.
🐛 Fixed in the tool: its `ch4` ablation used `IMBALANCE_SCALE=0`, which only affects OvD **MODE 0** and has
been INERT since `OVD_BOUNDED_MODE=2` shipped — that channel silently contributed nothing to every earlier
run. `OVD_CAP=0` is the correct ablation.

## 3. ☠️ SUPERSEDES `EVAL-TERM-REVIEW-TABLE-2026-09-08` CLUSTERS 1-2
That table called `central` a "REAL — pure re-sum" and OvD "REAL at the feeder". **Both verdicts are wrong**:
central is 215.9mp at r=0.39 and its removal costs up to −1.6pp; OvD is independent (r=0.13 with central).
The rest of the table (line-numbered term inventory, dead code, never-fitted literals) stands.

## 4. 🐛 TWO REAL BUGS: SHIPPED, MEASURED NEGATIVE, REVERTED
1. `get_latent_rook_activity_score` scanned **DIAGONAL** rays for a ROOK (`:2285`; the same function is
   correct at `:2237/:2242`) → `ENABLE_ROOK_LATENT_RAY_FIX`.
2. The rook MIDGAME loop never writes `square_values` (`:7534`) though the endgame loop does (`:7921`), so a
   midgame rook reads as **value 0** and is picked as the CHEAPEST attacker in the capgains gather (`:9031`)
   → `ENABLE_CAPG_ROOK_SQVAL`.
**Leave-one-out on one binary:** baseline **250** / ray-only **248** / sqval-only **247** / both **247**.
Regret: ray 48.7% (n=1410), sqval 49.2% (n=2056) — each ~1σ negative.
⇒ **FOUR aligned weak negatives ⇒ REVERTED** (fingerprint returned to 250 / 35,310,778 / 3.784 exactly).
📄 [[a-correctness-fix-into-absorbed-tuning-is-not-free]] — the constants absorbed the defects; re-fit first.
★ **Run the leave-one-out BEFORE flipping a default, not after.**

## 5. STATE
Baseline **250 / 35,310,778 / EBF 3.784** (verified after every build; five builds this session).
Six knobs in the tree, **all default-off**: `ENABLE_ORACLE_EVAL`/`ORACLE_CLASSICAL`/`ORACLE_SCALE`,
`THREAT_ATT2_PROTECT`, `KS_MOB_EDGE`, `ENABLE_ROOK_LATENT_RAY_FIX`, `ENABLE_CAPG_ROOK_SQVAL`.
⚠️ **NOTHING COMMITTED THIS SESSION** — the knobs, ~8 diagnostic changes (including the `_ks_footprint_regret`
`/tmp` collision fix that made concurrent runs silently wrong), and several dev_notes are uncommitted.
⚠️ Untracked and cited: `SEARCH-SWEEP-2026-08-25`, `SESSION-HANDOFF-2026-08-24/-27`,
`PASSER-REARCHITECTURE-SPEC-2026-08-22`, `SEARCH-INFRA-FLAVORS-2026-08-23` — **commit before any archive
pass**, or `git mv` cannot preserve their history.
