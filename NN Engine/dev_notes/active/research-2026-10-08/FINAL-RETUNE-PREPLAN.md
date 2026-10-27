# FINAL RETUNE — RESEARCH PRE-PLAN (written 2026-10-09, read-only research)

**Status: INPUT FOR DISCUSSION with the owner, not a decision.** Nothing was run, built or edited for this document; an
overnight game gate (queue #42, PX and KFL as separate arms vs the ship, SF18 @1000 seeds 108/109 + self-play seed 110) was
using all four engine slots while it was written. Every claim is tagged **EVIDENCE** (file:line / log / URL) or
**SPECULATION** (my inference). Where the record already settled a point, the record is cited rather than re-derived.

The retune this plans is step 3 of the owner's 10-08 order (`SESSION-HANDOFF-2026-10-07.md:122-130`): finish the known
STRUCTURAL elements → POT design → **STRUCTURAL retune (split mg/eg legs; depth target + static component)** → search
transition → DYNAMIC-lane retune on the new search → rest of the search arc. So this document plans the STRUCTURAL lane in
full and prepares the DYNAMIC lane's plumbing and corpus so it can run later without re-deriving anything.

---

## 0. Executive summary

- **What exists (EVIDENCE):** a working depth-target fitter with STM + SCALE nuisances and a by-game split
  (`diagnostics/_revival_screen.py:65-97`), a dual depth/static mode (`:256-338`), a real-re-search reader (`:341-396`),
  per-leg table loaders for PST / Kaufman / C1 (mobility, structure, passers, placement) / KS-B / KFL / KPROT / PX
  (`eval_v2.cpp:355, :2100, :3922, :3986, :4021-4044`), ~34k depth-labelled rows (our d10 vs SF18 d14) on the current ship,
  1.84M + 3.69M quiet result-labelled rows from own games, 33k SF18 multi-PV d14 regret rows, and bench/stress sets
  (variants 3,000 · odds 480 · K+P 600 · K+P stress 600). Inventory in §5.
- **What is missing (EVIDENCE):** loaders for the knob-only terms (threats, space, rook files, connected, reach / longdiag /
  latent; handoff `:149-154`), per-piece `v2_piece_value` knobs (`eval_v2.cpp:199-216` has only a global piece-side percent
  and a pawn mg), the path-ladder-under-C1 fix (`eval_v2.cpp:1565-1580` `continue`s before the ladder at `:1601`), a
  sibling-ordering metric, a quiet filter for the static component, a criticality definition, and a corpus roughly
  5-10× today's depth-labelled rows.
- **Proposed shape (SPECULATION, built on the owner's stated calls):** data first (reusable; ~3 nights of SF18 labelling +
  1 night of our d10 passes), then nested joint fits with the shipped values as the prior (L2 toward the ship, not toward
  zero), objective = depth-target win% MSE + static win% MSE on quiet rows, each normalised by its own baseline, endgame
  rows up-weighted, K+8P / variant / odds stress slices as a bounded TRAIN slice and as disjoint HELD-OUT checks; then
  per-part game gates (SF18 @1000 + self-play, ≥ 2 pooled seeds, ~7.5 h per arm) in the order the evidence ranks them,
  a stacking check (joint vs the sum of shipped parts), SPSA only for ~5 scalars (two cold-start runs), and the margin
  re-sweep handed to the search arc.
- **Budget (SPECULATION from measured rates in §2.9):** ~4 nights of data, minutes of fitting, ~8-12 nights of gating +
  2-3 nights of confirmation/stack — about three weeks of overnight slots if nothing else runs, longer interleaved with
  the POT work.

---

## 1. Where we stand — the inputs this plan takes as given

### 1.1 The ship and the instruments (EVIDENCE)
| item | value | source |
|---|---|---|
| shipped v2 fingerprint | WAC d10 **254 / 50,622,239 / 4.029** (`V2_PRESET=shipped`) | handoff `:91-97`; q39-q41 logs reproduce `Solved 254/300 NODES: 50622239` |
| external judge | **SF18 @1000 nodes** vs ours @250k = 48.6% (@800 drifted to 57-59%) | REFERENCE-BENCH-LADDER `:375`; memory `the-sf18-gauntlet-anchor-drifted-too-weak` |
| second instrument | self-play 2,000 games @50k nodes, fresh seeds | C3 §20c `:1090-1098` |
| ship rule | ≈ 2σ combined across BOTH instruments; re-anchor after each ship | handoff `:192`; memory `two-fair-instruments-can-genuinely-disagree` |
| depth target base | `ks_sets/fitC_{mg,eg}_ours1004_d10_s*of4.csv` — 14,697 mg + 19,281 eg rows (our d10 search of the 10-04 ship) | C3 §19g `:997-999`; `wc -l` 3,670-3,684 / 4,815-4,834 per shard |
| labels | `fitC_mg_sf18.csv` 14,842 rows (incl. 4,944 variant) · `fitC_eg_sf18.csv` 19,779 rows, SF18 d14 | SESSION-HANDOFF-09-30 `:284-285` |
| feature matrix | `E:/chess_data/texel/px_labelled.npz`: 34,621 fens × 235 Black−White columns, phase, flags, θ_mg/θ_eg | npz header read |
| win% map | `winpct(cp) = 100/(1+exp(−0.00368208·cp))` clamped ±1500 cp, K frozen | `_revival_screen.py:29-31`; MEMORY.md |

### 1.2 Items the record has already routed to the final retune (EVIDENCE)
| item | why it waits | source |
|---|---|---|
| Kaufman cells depth re-fit | +7 ± 8 combined; owner rule: fit jointly with `v2_piece_value`, never `values[]` | C3 §20c; memory `final-retune-needs-a-giant-diverse-corpus:37-41` |
| KFL + PST depth re-fit (pair) | SF18 −3 ± 8 (3,000 g) vs self-play +15 — instruments split | C3 §20e `:1112-1117` |
| KPROT (king protector, C3-c) | SF +16 (2,000) vs self-play −2 — split; also reads DYNAMIC on real re-search (eg −1.3 / mg +3.5) | C3 §20d; §20a `:1045-1047` |
| mobility cells (fit −1.44%) | 0 in games; eg-leg-only −3.2% on real re-search but 0 in games | C3 §20b; §20a `:1045-1051` |
| eg legs of all v2 columns | −3.35% on the endgame rows, bias unchanged | C3 §21a `:1150-1152` |
| pawn structure (STRUCT 66-76) | +5.7 n.s.; passer re-price alone HURTS (+4.06% re-search, −22 Elo) | C3 §19c, §20a `:1058-1060` |
| PX cells + joint passer re-price | **−5.56% real re-search on the ship base**; gating now (queue #42) | C3 §20a `:1063-1067`; `E:/chess_data/q42_structural_gate.log` |
| path ladder | −1.25% on the ship base; **DEAD under any C1 table** | C3 §20a `:1065-1066`; `eval_v2.cpp:1565-1580` |
| rook files (α 0.77), winnability +PASSED (−0.33%), KAUF values (−0.67%), taper/space/longdiag/reach/latent (null on the proxy, dynamic ⇒ unread) | below the 0.5% bar or dynamic | C3 §20 table `:1008-1020`; §20a `:1068-1073` |
| threats | self-play +32 ± 9 / SF18 −5 ± 9 ⇒ search transition first | C3 §20a `:1053-1057` |
| weak-unopposed mg, `PST_V2_KING_EG_ONLY`, tempo (= STM nuisance) | parked to "the joint retune" | EVAL-V2-CURRENT-CONFIG `:128-134`; PARKED-REGISTER `:186` |

### 1.3 The owner's calls that shape the method (EVIDENCE — quoted from the record)
1. **Fit every term's mg and eg legs separately**; fixed-ratio knobs (`MOB_V2_EG_PCT`, `PS_V2_EG_RATIO`) become two free legs;
   **KS keeps its own phase design** (`KS_V2_EG_PCT`, KS-B legs) — nudged, not doubled (handoff `:142-144`).
2. **Hybrid**: Texel-style bulk fit on SF18 labels (depth target + static component, eg weighted, K+8P stress as train AND
   held-out), GAMES gate every part on both instruments, **SPSA only for a few scalars** (handoff `:140-142`).
3. **Never tune `values[]`** (shared with v1 and search: SEE, ordering, null move); re-price through `v2_piece_value`
   jointly with Kaufman — "each would absorb the other's correction" (memory `final-retune-…:37-41`).
4. **Static discrimination matters** even where search fixes the verdict (pruning / stand-pat / ordering read statics);
   add a sibling-ordering metric; the final fit has a static component (memory `static-discrimination-matters-…`).
5. **Two lanes**: STRUCTURAL terms on depth + static; DYNAMIC terms on the static target, each CONFIRMED by a real d10
   re-search vs an identical-conditions ship re-run before any game gate (handoff `:145-148`; memory
   `root-delta-depth-proxy-is-biased-against-dynamic-terms`).
6. **Stress positions for understanding vs memorisation** — train slice AND held-out, never the same rows; judged by our
   d10 SEARCH vs SF18 d14, not static, not d7 regret (memory `final-retune-…:25-35`).
7. **The objective is STAGED and criticality-weighted**: absolute → criticality → d7 regret; a worse average MSE must not
   auto-veto; "critical" is undefined and is the blocker for stage 2 (memory `the-tuning-objective-is-staged-…`;
   CORPUS-CHARTER `:9-36`).
8. **Pin the global scale; acceptance by GAMES; fit LAST** (after POT is built, else "value = constant × mechanism"
   invalidates it) — memory `corpus-fit-is-anti-correlated-with-elo:62-76`.
9. **Fit set = real-play distribution; wacky/960 held out as validators** (CORPUS-CHARTER `:168-175`) — later refined by
   the owner (10-04) to BOTH roles on disjoint rows, weighted so stress rows test rather than dominate.

---

## 2. The plan, step by step

### 2.0 Pre-conditions
- P1 **POT endgame side designed and built** (handoff `:159-166`): the retune must price it. If POT's endgame design slips,
  the retune can still proceed on everything else, but a later POT ship re-opens the joint fit (law 8). **Decision for the
  owner:** retune after POT (the record's order) or run the structural retune now and accept one more joint refit later.
- P2 **Queue #42's verdict read and (if positive) PX / KFL shipped first**, then a fresh depth pass of the new ship
  (~1.6 h, §2.9) before any fit: every fit's base is the CURRENT ship's d10 (C3 §19g).
- P3 **Nothing running** when loaders are built (never rebuild while a job runs — handoff `:154`).

### 2.1 Plumbing — what to build before the fit (all byte-identical at defaults)
Audit basis: handoff `:149-154` (RETUNE PLUMBING AUDIT 10-08) plus my own reading of `eval_v2.cpp`.

| term | today's form (EVIDENCE) | leg structure | loader | to build |
|---|---|---|---|---|
| PST | `PST_V2_FILE` 768 ints (6 × 64 mg, 64 eg) | 2 legs | ✓ `eval_v2.cpp:355` | nothing; the fitter ties files a=h… (`_revival_screen.py:189-205`) |
| Kaufman census cells | `KAUF_V2_FILE` lines `O/T a b mp`, 36 cells, untapered | **1 leg** | ✓ `:2100-2132` | ⚠️ the loader accepts only `O`/`T` (`:2122-2123`); the fitter's **`L` value-correction cells** (`_texel_kauf_fit.py:39-40`) have no engine home — they must land in `v2_piece_value` (next row). **Open:** give Kaufman cells two legs? SF's imbalance is untapered; the record fitted it untapered (`_joint_depth_preview.py:10`). Proposal: keep 1 leg (one-owner, fewer params) |
| piece values | `v2_piece_value(t, phase256)` — pawn mg knob + ONE global `EVAL_V2_PIECE_MG_PCT` (`:199-216`); `values[]` shared | 2 legs needed | ✗ | **new per-type knobs `V2_VAL_{N,B,R,Q}_{MG,EG}`** (mp deltas on `values[]`, default 0 ⇒ identity short-circuit kept), mirrored in `v2_piece_value_legs` (`:220-229`); pawn stays the unit; fitted jointly with the Kaufman cells (owner rule) |
| C1: mobility 0-65 · structure 66-76 · passers 77-96 · placement 97-105 | `C1_V2_FILE` lines `k leg start fitted`, missing cells keep live θ | 2 legs | ✓ `:3922-3979` | nothing for the cells. ⚠️ `v2_c1_init` warns that `PS_V2_MAG` / `PASSER_V2_MAG` are folded in (`:3926-3927`) — fine, both are 100 |
| **path ladder under C1** | `passer_value_mp` `g_c1_fit` branch `continue`s at `:1580`, the ladder lives at `:1598-1638` in the constant branch ⇒ byte-identical to "no ladder" under any C1 table (C3 §20a `:1065-1066`) | — | — | **fix = compute the ladder in the C1 branch too** (hoist `:1601-1638` above the `continue`, add `kw`-converted values to `fmg/feg`), or **retire the scalar ladder** and let the PX STOP cells (0-23 "blocked by type + free + path", `_px_depth_fit.py:27`) own path safety. One-owner rule says pick one — **owner decision** (§6 Q6). Either way, the knob must stay byte-identical at `PASSER_V2_PATH_PCT=0` |
| KS-B shelter/storm | compiled ship cells or `KSB_V2_FILE` | 2 legs | ✓ `:4028-4037` | nothing |
| KFL, KPROT, PX | `c3_load_table` | 2 legs | ✓ `:3986-4016`, `:4038-4043` | nothing |
| **connected pawns** | `PS_V2_CONN_MAG/SUPPORT/EG_RATIO` scalars over `PS_CONN_RANK_MP[r]` × phalanx/opposed modifier, eg = `v·(r−2)/4·EG_RATIO/100` (`:1299-1322`); computed BEFORE the C1 branch; sets extractor flag 4 (`:3731-3733`) ⇒ feature-pass closure refuses under the ship | fixed ratio ⇒ **two free legs wanted** | ✗ | add columns to `v2_features` (per-rank connected count split phalanx/opposed, supporter count) + a `CONN_V2_FILE` table (per rank mg/eg + support mg/eg); remove flag 4 for connected once decomposed; exact closure vs the shipped knobs at the start values (memory `feature-pass-closure-refuses-under-shipped-connected`) |
| **threats** | `THREAT_V2_PCT` × constexpr per-victim tables (`TH_MINOR/ROOK_{MG,EG}[5]`, king, hanging, safe-pawn, push, restrict — `:1832-1840`), 5 bool legs | 2 legs per cell (30 cells) | ✗ | `THREAT_V2_FILE` (same `k leg start fitted` contract) — **dynamic lane**, so the loader is needed for the LATER dynamic retune and for static dumps now; build it with the batch so no second rebuild is needed |
| **space** | `SPACE_V2_MAG` × raw × phase256/256 — **eg leg is identically 0 by construction** (`:2265-2271`) | mg only | ✗ | `SPACE_V2_MAG` is already one free mg leg; an eg leg would be a FORM change (SF/Ethereal space is mg-only). Proposal: no loader; keep as a scalar in the dynamic lane; **owner call** whether an eg leg is wanted at all (§6 Q5) |
| **rook files** | `ROOKFILE_V2_OPEN/SEMI` mg + fixed `ROOKFILE_*_EG_PCT` constants (`:2298-2299`) | fixed ratio ⇒ 4 free cells | ✗ | two more knobs `ROOKFILE_V2_OPEN_EG/SEMI_EG` (or a 4-cell file) — dynamic lane |
| reach / longdiag / latent | `*_V2_PCT` × constexpr (mg, eg) pairs (`:2561, :2566, :2585`) | fixed ratio | ✗ | 2 cells each via placement's C1 table (extend `V2F_*` columns) — all read null on the proxy; dynamic lane |
| mobility eg share | `MOB_V2_EG_PCT=125` (`:1815-1818`) | folded into the C1 mobility eg legs | ✓ | nothing: a C1 mobility table makes the eg leg free per cell |
| KS phase | `KS_V2_EG_PCT=66` scalar on the Hill-curve output (`:4213-4225`); KS-B cells 2 legs | owner: keep | ✓ | nothing; `KS_V2_EG_PCT` → SPSA scalar list |
| winnability | `POT_V2_WIN_BASE/SP/OCB` scalars, multiplicative eg scale | scalars | ✓ (knobs) | `_win_depth_fit.py` exists; +PASSED needs engine closure before any gate (C3 §20a `:1069-1071`) |
| tempo | = the STM nuisance (~+8 cp) | — | — | not fitted; the nuisance stays un-shipped (handoff `:1021-1022`) |

**Build discipline (EVIDENCE: every loader so far followed it):** each file-fed term is OFF without its file and reports
`[c3]`/`[c1]`-style lines; a missing cell keeps the LIVE value (C1) or 0 (C3 blocks); byte-identity at defaults is checked
against `254 / 50,622,239 / 4.029`; symmetry `_eval_symmetry.py` colour 0/4000 + file 0/3170; exact closure of the
feature pass against the engine (`_conn_depth_fit.py MODE=closure` pattern, max 1 mp). Add a **`V2_RETUNE_FILES=` single
env listing** (SPECULATION: convenience) so a gate arm is one env line rather than seven paths — the runner's `V2=` line,
`search_engine.cpp`'s preset block and `EVAL-V2-CURRENT-CONFIG.md §1` must stay in sync (handoff `:70-71`).

### 2.2 Corpus assembly
Principles already on record (CORPUS-CHARTER `:115-136`): one judge per target column (SF18 d14; never relabel with SF19),
keep differently-generated sets separate (never pool d13 single-PV `diverse_corpus_wide` with d14 multi-PV), split BY GAME,
snapshot before regenerating, absent ≠ zero, measure the null per stratum.

**Target size (SPECULATION, reasoned):** the joint structural fit has roughly PST 384 + v2 columns 235 × 2 + Kaufman 36 +
values 8 + connected ~12 ≈ **900 parameters**. Today's 32,708 depth rows give ~36 rows/parameter; the nested fits so far
worked because each block was fitted ALONE against a frozen rest. A joint fit with the ship as prior tolerates fewer rows
than a from-scratch fit, but the C1 lesson (collinear blocks trading value, TEXEL-FIT-C `:93-96`) argues for **≥ 200k
depth-labelled rows (≈ 220 rows/parameter)**, and the owner's "gigantic" corpus. Search-score labels are far less noisy
than game results (a d14 score is near-deterministic for the target), so this is already generous by Texel standards
(§3). Proposed composition, all labelled SF18 d14 single-best `best_cp` (the existing contract of `_build_regret_set.py`):

| slice | rows (target) | role | source / how |
|---|---|---|---|
| **A. own-play standard** — positions from v2-era self-play (UHO starts), `PLY_STRIDE=7`, both game layouts | **120k** (eg ≥ 50%: sample later plies / phase < 128 deliberately) | TRAIN core (weight 1) | `_build_game_regret_set.py` with `GAMES_GLOB='*/*/game.jsonl'` (CORPUS-CHARTER `:138-159` — the nested layout was never sampled, the sampler is deterministic so vary `GAMES_GLOB`/`SHARD`); the existing 32.7k rows stay as they are |
| B. variant / 960-shuffle (castling disabled) | 20k, `WALK_MAX≈45` for phase spread | TRAIN slice (weight ≤ 0.5) + a disjoint 5k HELD-OUT set | `_build_variant_regret_set.py` (memory `whacky-variant-corpus-…`); `openings_variant.txt` 2,376 starts; `v2_variant_stage1.csv.gz` 239k result rows exist for the static side |
| C. odds starts | 5k | TRAIN slice (weight ≤ 0.5) + held-out `bench_odds` (480) | `openings_odds*.txt` 481 + 121 starts; `bench_odds.csv` exists |
| D. **K+8P stress** (dense 7-8 pawns, no pieces; pure 3-6) + per-subsystem strip-downs (K+pieces no pawns; K+R+P; K+minor+P) | 5k TRAIN (weight ≤ 0.3, capped at ~3% of total weight) + **SEED 41's 600 stay HELD-OUT, never trained** | understanding vs memorisation | `gen_kp_fens.py SEED≠41` (`diagnostics/gen_kp_fens.py:18-25`); `kp_stress_sf18.csv` 600 labelled (SEED 41) |
| E. endgame-type enrichment (pawn endings, pure minor, pure rook, rook+minor, queen) | 20k | TRAIN (eg weight) + per-type bias readout | sample by `_endgame_types.classify` from own games; `bench_kp.csv` 600 + `pawn_ending_labels.csv` exist |
| F. **existing SF18 d14 multi-PV regret sets** | 33,174 (v2era a/b/c) + 15k + 14.7k + 11.9k + 6k older | HELD-OUT (second instrument for ordering) — see §2.4; the three v2era files are disjoint and independent | `ks_sets/game_regret_set_v2era*.csv`; CORPUS-CHARTER `:59-72` |
| G. criticality-enriched | (needs the definition — §6 Q3) | stage-2 weighting | regret sets carry `moves` (top-8 multi-PV with cp) ⇒ best-vs-2nd gap is computable today |

Every TRAIN row additionally needs (i) our d10 search of the current ship (`_depth_residual_pass.py`, 4 shards, CHUNK loop
— `:16-18`) for the depth target, (ii) a static dump of the ship (`_revival_screen.py MODE=dump`), (iii) the feature
export (`_px_export.py` → npz). (i) is the expensive one and must be **repeated after every ship** (the base moves).
**Quiet flag for the static component:** `_triangulate_sf11.py` defines quiet as "no winning capture for the mover"
(`:214, :364`); the result-label corpora used a qsearch-style filter at extraction (`fitC_stage1` "quiet rows"). The static
component should use the same quiet definition on every row (stored as a column), because the static ladder on unquiet
positions measures static tactics (INSTRUMENT-MAP `:635-636`).

### 2.3 Labels and bases
- Judge: SF18 d14, `_build_regret_set.py IN= K=4 SF_DEPTH=14` (C3 §14 `:609`). Mates/`|best_cp| ≥ 50000` dropped;
  draw-classifier / tier-2b rows (`flags & 3`) excluded from the linear model (`_revival_screen.py:52-55`).
- Depth base: our d10 search, side-to-move → White cp, 4 shards (`_depth_residual_pass.py`). Store per ship tag
  (`ours1004` → `ours10xx`), never overwrite — `dualread` needs identical-conditions re-runs (the stored pass reproduced on
  only 146/300 rows across sessions, memory `root-delta-…`).
- Static base: ship totals (Black-positive mp) → White cp /10 (`_revival_screen.py:269`).
- After each ship: re-pass depth (1.6 h / 34k rows) and re-dump static (minutes) before the next fit.

### 2.4 The objective
Current fitter model (EVIDENCE `_revival_screen.py:65-97`): `ours' = base·(1+s) − X·δ/10 + stm·STM`, loss = mean
(winpct(ours') − winpct(SF))² on train, L2 `λ·|δ|²/1e4`, L-BFGS-B with tight tolerances, nuisances started at the
baseline's fitted values. The dual mode (`:307-314`) sums the depth and static losses **each normalised by its own train
baseline**, equal weight, with separate nuisances per target. Proposal for the final fit (SPECULATION unless cited):

```
L(δ) = w_d · Σ_i ω_i · (wp(d10_i·(1+s_d) − X_i·δ/10 + stm_i·t_d) − wp(SF_i))² / L_d0
     + w_s · Σ_{i quiet} ω_i · (wp(static_i·(1+s_s) − X_i·δ/10 + stm_i·t_s) − wp(SF_i))² / L_s0
     + λ · Σ_k (δ_k / scale_k)²            (L2 toward the SHIP, per-block scale)
     + λ_S · smoothness(PST, mobility curves)   (Fit A's neighbour penalty, _texel_pst_fit.py:17-19)
```
- **Weights `w_d : w_s`** — start **1 : 1 normalised** (the existing dual default); report both columns and the endgame
  sub-columns for every candidate (the dual table format `:320-338`). Owner's principle: a static gain is worth having only
  if the depth column does not get worse (`:261`). **§6 Q1.**
- **Row weights ω_i**: endgame rows (phase256 < 128) × (1+β), β ≈ 1 to start (owner: "endgames weighted"; the eg excess is
  114% of the gap to SF11, BENCH-LADDER `:359-360`); stress slices capped (§2.2); equal weight per GAME (Fit A's rule) so
  long games do not dominate. Decided positions are NOT down-weighted for SF labels (unlike result labels) — the ±1500 cp
  clamp already flattens them; **§6 Q2**.
- **Nuisances** `s` (global SCALE) and `t` (STM) are fitted and **never shipped** — this is the "pin the global scale" rule
  implemented as a free, discarded parameter (`_revival_screen.py:66-69`). Note the owner's tempo knobs are the shipped
  form of `t`; the record says tempo is a threshold switch, not eval (CURRENT-CONFIG `:185`) — leave it to the search arc.
- **Prior = the ship, not zero.** Every δ is a CHANGE from the shipped value (the C1/C3 file contract `start fitted`), so
  L2 pulls toward what games already accepted. Per-block scale: δ in mp divided by the block's typical magnitude so a 10 mp
  PST change and a 10 mp Kaufman change cost the same relative amount (SPECULATION: cheap to add, prevents the optimiser
  spending all its freedom in the largest block).
- **Identifiability constraints kept from Fit A**: PST per-(piece, leg) MEAN pinned (a PST mean is material — Fit A
  `:13-15`), file mirror ties a=h…, colour symmetry by construction (Black−White columns).
- **Split legs**: X carries every column twice (× phase/256 and × (256−phase)/256) — already the `legs()` design
  (`_revival_screen.py:104-105`). Kaufman stays 1 leg (§2.1).
- **Bootstrap stability** (Fit A: 5 refits over resampled GAMES; a cell whose mean change < 2× its bootstrap sd returns to
  its start, `_texel_pst_fit.py:20-22`) — apply to the joint fit; it is the cheapest guard against the C1 "trade between
  collinear blocks" artefact.
- **Nested order (owner 09-27, C3 §8b: "fit them NESTED, never only jointly")**: (1) blocks built at 0 or known-mispriced
  alone against the frozen rest, (2) all structural blocks jointly from those starts, dynamic blocks FROZEN at the ship,
  (3) report per-block deltas AND the joint — the per-part gates use the per-block δ applied alone on the ship.
- **Validation**: by-GAME 15% split (the 10-05 fix) + whole-run holdout (a game tag the fit never sees, Fit A's `val_run`),
  + held-out stress sets read by **our d10 SEARCH vs SF18** (win% |gap| + W/D/L agreement, `_kp_stress_check.py`), not
  static. Register predicted val ranges BEFORE each fit (the scorecard habit, handoff `:39-42`).
- **Discrimination metric (to build; owner principle 10-07)**: `_sibling_order.py` — for each regret-set row, static-eval
  the children of SF's top-8 moves (`moves` column) and score (a) Spearman vs SF's multi-PV cp order, (b) top-1 agreement,
  (c) the same for SF11 static as the achievability control (`_term_separability.py SF11=1`, `:61-71`). 33k rows × 8
  children ≈ 265k static evals ≈ minutes. Report it beside the static MSE for every candidate; do not fit on it
  (SPECULATION: a ranking loss could be added later; first see whether it moves at all).

### 2.5 Two lanes and the confirmation step
- **STRUCTURAL lane (this retune):** PST · Kaufman + piece values · pawn structure · passers (+ PX, path ownership per
  §2.1) · connected · KS-B/KFL (king-pawn structure) · winnability knobs (+PASSED) · eg legs of all of these. Objective =
  depth + static (§2.4). Dynamic blocks FROZEN at the ship.
- **DYNAMIC lane (after the search transition, 10-08 order step 5):** threats · mobility · space · rook files ·
  reach/longdiag/latent · KPROT · eg-POT dynamic parts. Objective = static target (what pruning sees); each part
  CONFIRMED by a real d10 re-search (`MODE=dual VALOUT=` → `_depth_residual_pass.py` 4 shards ≈ 16 min for ~5k val rows →
  `MODE=dualread ARM= SHIP=` against an identical-conditions ship re-run) before any game gate (handoff `:145-148`).
- **Why mobility is frozen in lane 1 (EVIDENCE):** C1's joint refit "mostly traded value between collinear blocks
  (mobility ↔ PST, passer ↔ pawn PST)" (TEXEL-FIT-C `:96`); mobility is v2's largest term (≈ +162 Elo). Freezing it removes
  the biggest trading partner; the cross-lane stack check (§2.6) catches what freezing hides.
- **Classification check (SPECULATION, cheap):** KPROT and KFL both sit on the structural/dynamic border (KPROT eg −1.3 /
  mg +3.5; KFL's mg signal overlaps threats' legs — C3 §20a `:1046, :1064`). Run the real re-search on each structural
  part too (it costs ~16 min per arm) — the proxy is only trusted for structural terms, and "structural" is a hypothesis
  per term until the re-search agrees with the proxy.

### 2.6 Gating order, stacking check, confirmation
Rules already on record: fresh seeds + fresh baselines (memory `gate-new-candidates-on-fresh-seeds-not-ship-seeds`);
≥ 2 seeds per batch, read pooled only (`a-shared-baseline-correlates-every-arm-of-its-seed`); both instruments
(`two-fair-instruments-can-genuinely-disagree`); register predictions first; never call a direction on partial data;
re-anchor after each ship; winner's curse — re-measure before quoting.

**Per-part gate (one arm):** SF18 @1000: 2 seeds × 500 paired games (ship + arm share the seed's baseline) + self-play
2,000 @50k = ~7.5 h (§2.9). Batching several arms on one seed's baseline saves the baseline games but couples the arms —
acceptable only with ≥ 2 seeds and pooled reads.

**Order (evidence-ranked; SPECULATION on the ranking itself):**
1. PX + passer joint re-price, KFL — **already in queue #42**; read its verdict first.
2. Kaufman cells + `v2_piece_value` jointly (the queen-imbalance lead persists at depth −4.7pp,
   memory `v2-overvalues-queen-vs-minor-compensation`).
3. PST depth re-fit (+12 ± 7.6 combined, borderline, C3 §20c).
4. eg legs of the structural columns (−3.35%, never gamed).
5. pawn structure (+ the two correctness fixes folded INTO the refit: isolated-also-backward double charge, backward
   needs neighbours — C3 §19a `:944-948`; law `a-correctness-fix-into-absorbed-tuning-is-not-free`).
6. connected cells (the shipped knobs as the start).
7. KS-B / KS attack nudge (KS keeps phase design), winnability +PASSED.
8. **The JOINT structural fit as one arm** vs the ship-with-all-passed-parts: the **stacking check**. Parts can cancel
   (`bundling-is-refuted-components-cancel-26-percent`; K1p: `fit-data-depth-must-match-play-depth`); KFL already costs
   PX's endgame (C3 §20a `:1064`). Known pairs to 2×2 explicitly: KFL × PX, KFL × threats (later), space × rook files
   (later, mg), Kaufman × piece values (fitted together by rule — gate together).
9. **Confirmation** of the shipped bundle on fresh seeds + anchor re-check (+~20 Elo moves the anchor ~5pp).

**Ship granularity (§6 Q10):** incremental (each passing part ships, base re-passed, next fit starts from the new ship —
the record's practice) vs one bundle at the end (cleaner joint values, one anchor shift). Incremental costs a 1.6 h depth
pass per ship and makes later fits conditional on earlier ships; bundle risks a single cancellation hiding a good part.
The record favours incremental with a final joint confirmation.

### 2.7 SPSA for scalars only
Candidates (SPECULATION, from the knob audit): `KS_V2_EG_PCT`, `KS_V2_MAX`/`KS_V2_HALF`/`KS_V2_ONSET` (the Hill curve is
non-linear — not Texel-fittable without a form change), `POT_V2_WIN_BASE/SP/OCB` (multiplicative — ditto; the depth fit
of these read +0.04%, so SPSA is the right instrument), possibly `PASSER_V2_PATH_PCT` if the ladder is kept as a scalar.
Protocol on record (memory `spsa-tuning-needs-replication-not-convergence-stats`): **two cold-start runs with
`spsa.py --seed N≠0`**, `--a 2.0 --c 0.45`, fixed d6 (~1,680 games/h), ~220 iterations × 60 paired games ≈ 13k games ≈
8 h per run; ship only knobs that agree in direction and size across both runs; ratify with `sprt.py` at `NODE_LIMIT=50000`
with BOTH configs (the runner's `gate` sub is v1). Vector ≤ 6-9 knobs.

### 2.8 Hand-off to the search arc
A retuned eval with larger spreads re-couples the absolute-mp pruning thresholds (`RFP_MARGIN`, `FUTILITY_MARGIN_SCALE`,
`QDELTA_PERMOVE_MARGIN`, `ASPIRATION_DELTA`, `VERIFY_MARGIN` — CURRENT-CONFIG `:185, :287-288`; Fit A note "re-sweep them
after shipping"). Memory `eval-accuracy-payoff-is-pruning` argues the eval bundle should be gated **with** the margin
re-sweep or it is judged at half pay-out. The 10-08 order puts margins in the search arc; the compromise on record is:
gate the eval parts alone (eval-only verdicts), then open the search arc with the margin re-sweep on the final eval.
**§6 Q13.**

### 2.9 Compute and time (EVIDENCE for rates, SPECULATION for totals)
| step | measured rate | source | estimate |
|---|---|---|---|
| our d10 depth pass | ~4,950 rows in 14-16 min with 4 shards ≈ **5.5 rows/s** | `E:/chess_data/q39_known_elements.log` (16:40 → 16:56 incl. fingerprint) | 34k rows ≈ 1.7 h · 200k rows ≈ 10 h (one night) |
| SF18 d14 labelling, K=4 | eg sample 02:00 → `fitC_eg_sf18` 04:52 for 19,779 rows ≈ **7k rows/h**; mg 14,842 rows in ~1.3 h ≈ 11k/h | file mtimes in `ks_sets/`; concurrency assumed 4 | 170k new rows ≈ 2-3 nights |
| static dump | 34k rows in ~45 s | handoff `:38` ("static evals are fast") | negligible |
| feature export + fit | Fit A engine pass 21k pos/s per process; L-BFGS fits minutes | INVENTORY §8 `:331` | < 1 h per fit round |
| SF18 @1000 gauntlet | 1,000 games (ship + 1 arm × 500) ≈ **2.5 h** (q38 seed 104: 10:47 → 13:21) | `q38_threats_sf18.log:1,1013,2024` | 2 seeds ≈ 5 h per arm |
| self-play 2,000 @50k | ≈ **2.2 h** (q37 01:50 → 04:03) | `q37_threats_gate.log:5042,7049` | 2.2 h per arm |
| SPSA run | 220 it × 60 games at d6 ≈ 13k games ≈ 8 h | RETUNE-PLAN `:312, :343`; memory (1,680 g/h) | 2 runs = 2 nights |
| **total** | | | data ~4 nights · fits hours · gates 8 parts × 7.5 h ≈ 8 nights · stack + confirm + re-anchor ≈ 3 nights · SPSA 2 nights ⇒ **~17 overnight slots** |

Constraints: ≤ 4 engine processes (WSL), bounded memory (CHUNK loops; `run_one` leaks ~1.2 MB/FEN), detached launches via
`_launch_detached.sh`, owner plays 9pm-midnight (fixed-depth / node-limited only), outputs to `E:/`, never edit a runner
while a job runs (handoff `:214-221`).

---

## 3. How the giants tuned, and nuances we may be missing

Sources were fetched directly (Ethereal `tuner.c`/`tuner.h` + Grant's `Tuning.pdf`; Weiss `src/tuner/tuner.{c,h}` +
`evaluate.c`; fishtest wiki + `spsa_workflow.py`; zamar/spsa README; CPW pages). talkchess.com refused direct fetches, so
talkchess items come from search snippets and are marked **(snippet)**. Everything else in 3.1-3.4 is **EVIDENCE** (URL
given); 3.5's applications to our plan are **SPECULATION**.

### 3.1 Ethereal — Andrew Grant, "Evaluation & Tuning in Chess Engines" (2020) + `src/tuner.c`
Paper: https://github.com/AndyGrant/Ethereal/blob/master/Tuning.pdf · code: https://github.com/AndyGrant/Ethereal/blob/master/src/tuner.c
- **Linear trick (paper §1.2, §4.1-4.2):** `E = L · (Cw − Cb)`; every weight application in `evaluateBoard()` is traced into
  per-colour coefficient vectors and the eval is rebuilt from `L, Cw, Cb, ρ, ξ` with no board. Only non-zero tuples are
  stored (`TTuple {index, wcoeff, bcoeff}`, "Skip Vectors") — "orders of magnitude in memory … an order of magnitude in
  speed". **We already do this** (`v2_features` + θ, `eval_v2.cpp:3702-3708`; our closure check = Grant's "the linear model
  must reproduce the eval"). Our dense int16 `diff` (34,621 × 235 = 16 MB) is fine; at 200k × 470 use int16 or sparse.
- **Phase (§1.3, `initTunerEntry`):** `phase = 4Q + 2R + B + N` (0-24), `mixed = (mg·phase + eg·(24−phase)·sfactor)/24`;
  every term has separate MG and EG weights ("Phases treated separately despite correlation", §3.3) — exactly our `legs()`.
- **Scale factor ξ (§1.3, §3.8):** `entry->sfactor = T.factor / SCALE_NORMAL` (OCB, unwinnable material) multiplies ONLY the
  EG leg and its gradient (`gradient[i][EG] += egBase·(wcoeff−bcoeff)·entry->sfactor`); treated as a constant ("likely
  negligible" that it depends on E). **We do not do this** — our winnability scale is applied to the total after the linear
  part and the fitter ignores it (active on 8.4% of rows, C3 §20a `:1068`). See 3.5.
- **Safety is non-linear (§1.4, §3.5):** safety terms are summed linearly per colour into `S`, then
  `f_mg(x) = −x·max(0,x)/720`, `f_eg(x) = −max(0,x)/20`, with the exact gradient through the quadratic; complexity is clamped
  and its gradient gated (§3.4, §3.7); `TuneNormal/TuneSafety/TuneComplexity` toggle the classes (tuner.h). Our KS Hill
  curve is the analogue: scalars to SPSA, KS-B cells linear inside the curve's input — consistent with "KS keeps its own
  phase design" and with Grant's separate treatment.
- **K (§2.3, §4.4):** `σ = 1/(1+e^{−K·E/400})`; `computeOptimalK()` grid-searches K minimising the MSE of the STATIC eval,
  then K is frozen. Ours is frozen too (0.00368/cp on SF labels; Fit A fitted K on the start tables, then froze it).
- **Loss (§2.4):** `1/N Σ (R_i − σ(E_i))²`, R ∈ {1, ½, 0} from `[1.0]/[0.5]/[0.0]` in the FEN file.
- **Optimizer (§4.6, tuner.h):** AdaGrad per weight; `LRRATE 0.10`, `LRDROPRATE 1.00` (no drop), `LRSTEPRATE 250`,
  `MAXEPOCHS 100000`, **`BATCHSIZE 16384`**, **`NPOSITIONS 42,487,498`**, **`NTERMS 904`**, 64 threads. The paper used
  full batch; the shipped header is mini-batch. Our L-BFGS-B is fine at our size.
- **Dataset (§2.2; (snippet) https://talkchess.com/forum3/viewtopic.php?f=7&t=75350):** earlier ≈7.8M = Zurichess 1.6M +
  Laser random-tree 1.3M + Ethereal self-play 4.9M; current recipe: 1M self-play games at 1s+.01s … 4s+.04s with aggressive
  adjudication; drop games < 10 moves; sample 10 positions per game; run a **depth-12 search and apply the whole PV**,
  saving the PV leaf as the "resolved" quiet position with the ORIGINAL game result. Mixed Standard + FRC gave "minor
  gains for Standard". Mate-score positions dropped.
- **Subsets vs all (§4.1; (snippet) https://talkchess.com/forum3/viewtopic.php?f=7&t=74877):** parameters are DELTAS from
  the shipped weights, so "only a subset … at any given time" can be tuned (= our `start fitted` file contract). Reported
  gains: all linear terms ≈ +10 Elo; King Safety alone ≈ +3.4; Linear+Safety+Complexity jointly ≈ +2.3 — retuned
  repeatedly, each pushed as its own OpenBench SPRT. **The paper does NOT discuss piece values, collinearity or
  regularisation (confirmed absent).** (snippet, 2017 "Texel Tuning - Success!", https://talkchess.com/viewtopic.php?p=736294):
  per-term learning rates scaled by firing frequency per phase — the AdaGrad precursor. (snippet
  https://talkchess.com/viewtopic.php?t=75012): 99% of time was FEN parsing; caching binary entries cut a run "from 5 hours
  to less than one second".

### 3.2 Weiss — `src/tuner/tuner.c` (Terje Kirstihagen)
https://github.com/TerjeKir/weiss/blob/master/src/tuner/tuner.c
- Header: "Gradient Decent Tuning for Chess Engines as described by Andrew Grant". `DATASET lichess-big3-resolved.book`,
  **`NPOSITIONS 7,153,652`, `NTERMS 554`**, labels purely game result. ("resolved" presumably = PV leaf as Ethereal's —
  SPECULATION.)
- **K hardcoded 2.25** (a Newton-step `ComputeOptimalK` exists but is `__attribute__((unused))`).
- **Adam, full batch**: β₁ 0.9, β₂ 0.999, ε 1e-8, `LRRATE 0.1`, `MAXEPOCHS 10000`; coefficients stored as the single
  difference `T.term[WHITE] − T.term[BLACK]`; MG/EG legs separate, `eval = (mg·phase + eg·(MidGame−phase)·scale)/MidGame`,
  `scale = T.scale/128` on the EG gradient; tempo added as a constant.
- **King danger is NOT tuned**: `danger = attackPower·CountModifier[min(7,count)]/128` is passed as a fixed base offset, and
  `InitTunerEntries` ABORTS if `|seval − coeffEval| > 1` — a built-in linearity check (ours is the Python closure gate).

### 3.3 Stockfish classical era — SPSA on fishtest
- Wiki (https://github.com/official-stockfish/fishtest/wiki/Creating-my-first-test): `θ += ck·rk·(wins − losses)`,
  `ck = c0/(1+k)^γ`, `ak = a0/(A+1+k)^α`, `rk = ak/ck²`; the user gives per-parameter `start,min,max,c_end,r_end`; the game
  count is fixed by the server; "tuning many values at once (like a PSQT table) generates random change"; if values barely
  move after a few thousand games, ck is too low; `nodestime=600` when the knob does not change NPS.
- Code (https://github.com/official-stockfish/fishtest/blob/master/server/fishtest/spsa_workflow.py): defaults `A 0.1`,
  `alpha 0.602`, `gamma 0.101`; `c = c_end·N^γ`, `a = r_end·c_end²·(A+N)^α`; update `θ = clip(θ + R·c·result·flip)`.
- Classical-era values (zamar/spsa README, https://github.com/zamar/spsa): **r_end 0.002, c_end 4 cp**; insensitive knobs
  need a larger c. CPW (https://chessprogramming.org/Stockfish%27s_Tuning_Method): "30000-100000 super-fast games", "7-35
  variables at the same time"; Kiiski: the method "doesn't converge and it needs to be stopped at a 'suitable moment'".
  α 0.602 / γ 0.101 are Spall's minimal convergent values (https://www.chessprogramming.org/SPSA).
- **Verification**: FAQ (https://github.com/official-stockfish/fishtest/wiki/Fishtest-faq): a good-looking tune is then
  scheduled as a two-stage SPRT. No evidence SF ever shipped regression-fitted classical weights (the PSQT-approximation
  thread https://talkchess.com/viewtopic.php?p=730523 was third-party) — SPECULATION: SF stayed SPSA-only for classical
  eval. Our 09-22 runs (147-220 iterations × 60 games ≈ 9-13k games, 6-9 knobs) sit at the LOW end of SF's 30-100k; hence
  the two-run replication rule.

### 3.4 CPW "Texel's Tuning Method" (Peter Österlund) — https://www.chessprogramming.org/Texel%27s_Tuning_Method
- 64,000 games at ~1s+0.08s between current/previous versions → ~8.8M FENs; book and mate-score positions excluded; label
  0/½/1; score = white-relative **qsearch** score; `E = 1/N Σ (R − σ(K·q))²`; "Compute the K that minimizes E. K is never
  changed again" (K = 1.13 originally). Optimizer: coordinate local search (+1, else −2, repeat), ~400 params, ≈ 6 h to a
  local minimum on 16 cores. Claimed advantage: "the need for different evaluation terms to be 'orthogonal' disappears".
- Pitfalls listed: correlation ≠ causation; the engine cannot learn what it cannot play out (KBNK); non-independent samples
  from one game (⇒ our by-GAME split); absurd fitted values (Q on b7/g7 −128 cp, K on b8 +200 cp — the Texel confound Fit A
  noted for rook mg); Österlund DROPPED his early filter of positions where qsearch disagreed with the game search: "the
  q-search function has to deal with them all the time in real games". Texel gained ~100 Elo (1.03), ~150 over 2.5 months
  (snippet https://talkchess.com/viewtopic.php?p=556319). (snippet) the method "only fixates K, not the value of any
  evaluation weight (such as the nominal value of a pawn)".
- **Zurichess quiet-labeled.epd** (snippet https://talkchess.com/forum3/viewtopic.php?t=61427;
  https://github.com/KierenP/ChessTrainingSets): 75k games from a 2-move book → 20 positions/game (1.5M) → remove positions
  where qsearch finds a winning capture → **each remaining position replayed with Stockfish to produce the label** → 725k.
  The most-used public Texel set already labels with a STRONGER ENGINE's play, not the generating games' results — the
  ancestor of our SF18-label design.

### 3.5 Published nuances (EVIDENCE) and what they mean for us (SPECULATION)
- **Search-score vs result labels / blends.** Ethereal resolves positions with d12 PVs but keeps the WDL label, and the paper
  names re-playing from resolved positions as the ideal; Zurichess re-played with SF. Gedas' texel-tuner
  (https://github.com/lynx-chess/texel-tuner) accepts fractional labels ("0.6 for 60%"), `preferred_k = 0` to auto-fit K,
  `enable_qsearch` / `filter_in_check` filters, `TAPERED` pairs. Leorik (snippet
  https://talkchess.com/viewtopic.php?t=79049&start=420) tunes on ~50M self-play WDL positions and found more data
  SATURATES. ⇒ Our depth-target label (SF18 d14 win%) is the fractional-label extreme; the saturation note supports a
  ~200k-row target rather than millions.
- **Scale freedom / shrinkage** ("Tapered Evaluation and MSE", snippet https://talkchess.com/forum3/viewtopic.php?t=76265):
  without a fixed K the tuner inflates or shrinks everything (queen = 100 pawns); fit K first and freeze it; MG(30)/EG(570)
  can beat MG(300)/EG(300) in MSE; anchoring one value (mg pawn = 100) helps but "is unfortunately not sufficient"; never
  anchor both legs. ⇒ Our frozen K + discarded SCALE nuisance + mean-pinned PST + the pawn as the unit covers this, but the
  "two free legs" design must watch for leg-swapping (a term's mg and eg trading places); the per-block prior toward the
  ship (§2.4) is the guard — report mg/eg spreads per block.
- **Regularisation: none in Ethereal or Weiss (confirmed).** L2-as-Gaussian-prior appears only generically on CPW
  (https://www.chessprogramming.org/Automated_Tuning), which also notes mini-batches can "exploit rare data for the worse".
  ⇒ Their substitute for regularisation is 7-42M rows plus deltas from a shipped start. With ~200k rows we keep the
  L2-to-ship prior and the bootstrap stability filter — a genuine difference from the giants, not a copy.
- **Zero-count terms:** sparse tuples + per-parameter adaptive rates; rare terms simply get few gradient hits. ⇒ our
  "fires %" column stays mandatory (a block that never fires is vacuous, not null).
- **Verification is universally by games** — Ethereal pushed each tune as its own SPRT; SF requires SPRT after SPSA.
  Identical to our rule; the giants' edge is throughput (OpenBench / fishtest), not principle.
- **Nuances worth copying (cheap):** (1) carry the shipped winnability scale into the eg coefficient (`coeff_eg·(1−p)·f_i`)
  so eg legs are not mis-priced on scaled rows; (2) Weiss's abort-on-closure-error at load — an in-engine
  `|total − Σ count·θ| ≤ 1 mp` assert at table load would catch a stale export before a gate; (3) PV-leaf resolution as the
  quiet filter for the STATIC component (stricter than "no winning capture for the mover"), while keeping Österlund's
  warning in mind for the DEPTH component (search meets unquiet positions constantly); (4) cache binary feature entries
  (we do — npz); (5) sample ~10 positions per game, drop games < 10 moves.
- **Nuances NOT to copy:** tuning ~900 weights from scratch on 34k rows (Ethereal's 42M makes that safe; ours does not); a
  free K per fit (our K is the pin); tuning PSQT-sized vectors by SPSA (fishtest's own warning).

## 4. Risks and failure modes from our own record — with mitigations

| # | risk (EVIDENCE) | mitigation in this plan |
|---|---|---|
| 1 | **Corpus fit anti-correlated with Elo** — 5/5 v1 fits bench-negative, best fit −85.6 Elo; mechanism = global shrink toward SF's conservative labels (memory `corpus-fit-is-anti-correlated-with-elo`) | SCALE + STM nuisances fitted and discarded (the pin); L2 toward the SHIP; acceptance by games only; v2 is non-degenerate + colour-clean (the structural causes are gone — Fit A, +38 vs SF18, is the counter-example `texel-pst-fit-is-the-biggest-eval-win`) |
| 2 | **Joint parts cancel at depth** (K1p: bundle +34 self-play, parts −11/+30 at 250k; `fit-data-depth-must-match-play-depth`; C1 −5.5) | nested fits; per-part gates at play depth; the joint-vs-stack check (§2.6 step 8); bootstrap stability filter |
| 3 | **Root-Δ proxy biased against dynamic terms** (threats +12.6% proxy vs −7.8% real; `root-delta-…`) | two lanes; real re-search on every part (16 min each) |
| 4 | **Two fair instruments disagree** (KPROT, KFL+PST, threats) | ship only on both; splits park for the joint fit, where they may pay jointly |
| 5 | **Shared baseline correlates arms; anchor drift; selection seeds bias** | ≥ 2 seeds pooled; re-anchor after each ship; fresh seeds + fresh baselines |
| 6 | **Fit size ≠ game size** (MOB −1.44% → 0; KPROT −0.91% → +27 SF) | the screen RANKS; register predictions; games decide |
| 7 | **Fitter artefacts**: no SCALE nuisance ⇒ stretch wins; optimiser stopping at start (exact −0.00%); FEN-hash split leaks same-game rows; scorer closure bound to the baseline | all four fixed in `_revival_screen.py` and `_px_depth_fit.py` (10-05/08); keep the guards; a baseline-identical arm reads as a BUG, not a null |
| 8 | **File-mirror symmetry broken by per-file cells** (pawn fit 2,435/3,170) | tie a=h… in every fitter; `_eval_symmetry.py` colour + file gate on every table |
| 9 | **Closure refuses under the shipped connected knob** (flag 4) | decompose connected into columns (§2.1) so closures run on the real ship |
| 10 | **Correctness fixes into absorbed tuning are not free** (two real bugs measured negative) | fold fixes INTO the refit (pawn structure, §2.6 step 5), never before it |
| 11 | **Memorisation of own-play structures** (owner's worry) | K+8P / variant / odds held-out slices judged by our d10 search vs SF18; train slices weight-capped |
| 12 | **Val-row picks are optimistic** (handoff `:148`) | games are the independent check; by-game split; whole-run holdout |
| 13 | **Corpus composition decides the optimum** (threat scale inverted 125 ↔ 75 on a re-mix; memory `corpus-composition-decides-the-optimum`) | fix the composition BEFORE fitting; snapshot; re-derive after any change; never grow a set in place |
| 14 | **Static target on unquiet rows measures static tactics** (v1 capgains "wins" variants) | quiet filter for the static component; variants judged by search |
| 15 | **value = constant × mechanism** — anything added after the fit invalidates it | POT built first (P1); dynamic lane fits after the search transition on the then-current ship |
| 16 | **Pruning margins tuned on the old eval** (Fit A note) | margin re-sweep opens the search arc (§2.8); quote eval + margins together |
| 17 | **Ops**: 30-min task cap after a VS Code reload; `/tmp` wiped; `run_one` leak; OneDrive/VS Code indexing | detached launcher; `E:/` outputs; CHUNK loops; `python.analysis.exclude` |
| 18 | **Phase aliasing** (v1 `phase_score` 0=opening/128=eg vs v2 `phase256` 256=opening/0=eg) | every fitter reads `phase` from the v2 npz; never a v1-era tool under `EVAL_ARM=1` |
| 19 | **Winner's curse / SPRT bound inflation** | confirmation runs re-measure; quote pooled tallies, never the stopping estimate |
| 20 | **Criticality unmeasurable on current corpora** (`n_crit` 27/43; `criticality-split-reads-signal-not-average`) | stage 2 needs the definition + an enriched set (§6 Q3); stage 1 proceeds without it |

---

## 5. Data and corpus inventory (what exists; what to generate)

### 5.1 `E:/chess_data/` (EVIDENCE: `ls`/`du`/npz headers, 2026-10-09)
| path | size / rows | what | use |
|---|---|---|---|
| `texel/v2_stage1.csv.gz` | 3,692,732 rows (48k own v2 games), `game_id, split, ply, fen, result_white, search_white_mp, pieces, phase_hint` | Fit A result-labelled quiet rows | static-side result component (optional); NNUE data later |
| `texel/fitC_stage1.csv.gz` | 1,844,613 rows (30k d6 self-play games) | Fit C/K data | same; the depth-target samples were drawn from here |
| `texel/v2_variant_stage1.csv.gz` | 238,978 rows | variant-game result rows | static held-out / optional train slice |
| `texel/fitC_features.npz` | 1,844,613 × 184 int16 + θ | C1 feature matrix on fitC | legacy layout (184/side) — re-export under 235 if reused |
| `texel/c1_std.npz` | 3,692,732 × 106 | C1 matrix on stage1 | legacy |
| `texel/fitC_win.npz` | 1.84M × 7 winnability inputs | POT winnability | — |
| **`texel/px_labelled.npz`** | **34,621 × 235** int16, phase, flags, θ_mg/θ_eg | the depth-target feature matrix (current layout) | the fit's X today |
| `texel/revival/*.txt` | kauf/mob/pst/kprot/kfl depth tables (+ `*_egonly`) | candidate tables, not shipped | starts for the nested fits |
| `texel/px_depth1008_{c1,px}.txt`, `px_depth_*` | passer/PX fitted tables (10-04, 10-08) | queue #42 arms | — |
| `texel/kauf_full.txt` | 36 shipped Kaufman cells | compiled in `ship_tables_v2.h` | start |
| `texel/pass*/`, `smoke/`, `fitC_pass/` | engine passes for Fit A/A2/C (zero/full totals) | | legacy |
| `bench1007/` | `bench_{variant,odds,kp}.csv` (3,000 / 480 / 600 rows, every evaluator), `diverse_corpus_wide.csv`, `playdist_ceiling.csv`, `eg_sf11_terms_quiet*.csv`, `pawn_ending_labels*.csv` | 10-07 bench dumps | held-out static ladders; endgame triangulation |
| `engines/fitA/` | 1.2 MB | an engine snapshot | — |
| `q32…q43_*.log` | queue logs | timings used in §2.9 | — |

### 5.2 `diagnostics/ks_sets/` (EVIDENCE: `wc -l`, headers)
| file(s) | rows | schema | role |
|---|---|---|---|
| `fitC_{mg,eg}_sf18.csv` | 14,842 / 19,779 | `fen, phase_bucket, best_uci, best_cp, moves` (SF18 d14) | TRAIN labels today |
| `fitC_{mg,eg}_sample.csv` | 15,000 / 20,000 | `fen, row, game_id` | by-game split keys |
| `fitC_{mg,eg}_ours1004_d10_s*of4.csv` (+ `ours1003`, `ours`) | 14,697 / 19,281 | `fen, ours_cp_white, depth, nodes` | depth base per ship |
| `dual_val_*_d10_s*of4.csv`, `dual_val_sf18.csv` | ~4,950 val rows per arm, 25+ arms | real re-search passes (threats, mob, space, …, PX, KFL, path, stack) | the `dualread` evidence; reusable as identical-conditions controls ONLY within their session |
| `game_regret_set_v2era{,_b,_c}.csv` | 10,000 / 12,000 / 11,174 (disjoint) | multi-PV top-8 d14 | sibling-ordering metric; held-out depth rows (need our d10) |
| `game_regret_set{,_x4,_v2,_uho}.csv` | 15,000 / 14,713 / 11,940 / 6,000 | same | pre-v2-era distribution — held-out only |
| `variant_regret_set.csv` | 6,000 (97% opening — needs `WALK_MAX≈45` regen) | same | held-out after regen |
| `bench_{variant,odds,kp}.csv` | 3,000 / 480 / 600 | `fen, target_total, split, type` | static held-out |
| `kp_stress_fens.txt` / `kp_stress_sf18.csv` / `kp_stress_{on,off}_d10_*` | 600 (SEED 41) | `fen, best_uci, best_cp, pawns` | **HELD-OUT stress — never train** |
| `diverse_corpus_wide.csv` | 23,113 | d13 single-PV | DIFFERENT JUDGE — never pool |
| `lichess_ks{,_labelled}.csv`, `collapse_*`, `pawn_truth*`, `passer_*`, `t2b_corpus`, `tablebase_labels_draw.json` | various | class-targeted / reference-labelled | instruments only, never weights (CHARTER §2-3) |

### 5.3 Game archive and openings (EVIDENCE)
- `selfplay/games/`: 1,479 tags; ~108k self-play games total per the toolkit (`DIAGNOSTICS-TOOLKIT.md:54`); two layouts
  (flat / nested) — the nested v2-era layout has **never been sampled** by the regret builder (CHARTER `:138-150`).
  ⚠️ A plain file count timed out at 2 min in the past (OneDrive sync; user CLAUDE.md) — count by tag, not by file.
- Openings: `openings_uho.txt` 1,002 · `openings_uho_ext.txt` 14,785 · `openings_variant.txt` 2,377 · `openings_odds.txt`
  481 · `openings_odds_np.txt` 121.
- `selfplay/tune_data/*.csv` (June-July corpora, v1-era, SF11 per-term columns) — triangulation substrate only.

### 5.4 To generate (SPECULATION on sizes; method on record)
1. ~120k own-play standard rows from the NESTED v2-era games (eg ≥ 50%), SF18 d14 labels, our d10 pass, static dump,
   feature export — 2 labelling nights + 1 depth-pass night.
2. ~20k variant rows at `WALK_MAX≈45` + 5k held-out; ~5k odds rows.
3. ~5k K+8P / strip-down stress rows (new SEEDs) for TRAIN; SEED 41 stays held-out.
4. ~20k endgame-type-enriched rows.
5. Our d10 pass over the 33k v2era regret rows (≈ 1.7 h) so they serve as a second held-out depth set.
6. Quiet flag column for every row (static component).
7. A criticality column (best-vs-2nd multi-PV gap) where multi-PV exists — stage 2 preparation.

---

## 6. Open questions for the owner

1. **Objective weights:** depth : static = 1 : 1 normalised (the dual default) or depth-heavier? And the endgame weight β
   (I propose ×2 on phase256 < 128 rows to start, read the eg sub-columns either way).
2. **Decided positions:** down-weight (Fit A's ×0.25) or keep (SF labels are not noisy the way results are)?
3. **"Critical" — the stage-2 definition.** Candidate: the mover's win% gap between SF18's best and 2nd-best move
   (multi-PV, already in the regret sets; bands benign <3% / critical >20% per the 08-14 insight). Alternative: rows where
   our d10 and SF18 disagree by > X win%. Stage 1 can run without it; stage 2 cannot.
4. **What is frozen in the structural lane:** mobility (largest term, dynamic) frozen at the ship — agreed? KPROT/KFL
   classification settled by the real re-search per part?
5. **Space and rook files:** space is mg-only by form (eg leg ≡ 0); do you want a free eg leg (a form change) or keep the
   SF/Ethereal form? Rook files: two extra eg knobs or a 4-cell file?
6. **Path ladder ownership:** fix it under C1 (hoist above the `continue`) or retire the scalar ladder and let the PX STOP
   cells own path safety (one-owner rule)? Queue #42 was run WITHOUT the ladder, so its PX verdict is ladder-free.
7. **Labels:** pure SF18 d14 search (today) or a small game-result component on the static side (Fit A's label won +38)?
   If both, they are separate columns and separate objectives, never a relabel.
8. **Corpus size / sources:** ~200k depth-labelled rows (~4 nights) acceptable? Own games only (plus variants/odds/stress),
   or add external strong-engine / human games (a different distribution — CHARTER rule 1 says separate set, and 960 needs
   castling verified first)?
9. **Gate budget:** per-part gating of ~8 parts ≈ 8 nights + 3 confirmation nights. Alternatively gate in tiers: parts with
   a real-re-search read ≥ 1% alone, the rest only inside the joint arm.
10. **Ship granularity:** incremental (part by part, re-pass the base each time) or one joint bundle at the end?
11. **Replication of the fit itself:** two independent fits (different val split seeds / bootstrap samples) must agree in
    direction before a part is gated — adopt as a rule?
12. **SPSA scalar list** (§2.7) and whether `KS_V2_MAX/HALF/ONSET` are in scope for the eval arc at all (the KS rung is
    +101 Elo; the 09-22 SPSA found no KS knob that replicated).
13. **Margins:** gate eval parts alone and leave the margin re-sweep to the search arc (10-08 order), or gate the final
    bundle together with a margin re-sweep (memory `eval-accuracy-payoff-is-pruning`)?
14. **Sibling-ordering metric:** build it before the fits (as a reported column) — yes/no, and should SF11's achievability
    control be mandatory on every read?
15. **Threats' static-only gain** (SF11's static endgame excess is threats-led, −75% counterfactual, BENCH-LADDER `:369-370`):
    the dynamic lane prices it after the search transition — or do you want a static-component-only read now as a
    pre-registered prediction for that lane?

---

## 7. Sources read (all read-only)
Memory: `MEMORY.md`, `final-retune-needs-a-giant-diverse-corpus`, `fit-data-depth-must-match-play-depth`,
`root-delta-depth-proxy-is-biased-against-dynamic-terms`, `corpus-fit-is-anti-correlated-with-elo`,
`the-tuning-objective-is-staged-and-criticality-weighted`, `static-discrimination-matters-even-when-search-fixes-the-verdict`,
`two-fair-instruments-can-genuinely-disagree`, `a-shared-baseline-correlates-every-arm-of-its-seed`,
`texel-pst-fit-is-the-biggest-eval-win`, `spsa-tuning-needs-replication-not-convergence-stats`,
`the-remaining-eval-gap-is-endgame-structural`, `v2-overvalues-queen-vs-minor-compensation`,
`the-sf18-gauntlet-anchor-drifted-too-weak`, `a-correctness-fix-into-absorbed-tuning-is-not-free`,
`bundling-is-refuted-components-cancel-26-percent`, `eval-accuracy-payoff-is-pruning`, `sf11-texel-scale-invariance`,
`gate-new-candidates-on-fresh-seeds-not-ship-seeds`, `criticality-split-reads-signal-not-average`,
`whacky-variant-corpus-for-structure-independent-validation`, `corpus-composition-decides-the-optimum`.
Dev notes: `SESSION-HANDOFF-2026-10-07.md`, `SESSION-HANDOFF-2026-09-30.md:262-300`, `TEXEL-C3-DETECTORS-DESIGN-2026-09-27.md`
§14, §19-21a, `EVAL-V2-CURRENT-CONFIG.md:1-288`, `EVAL-V2-PARKED-REGISTER.md`, `REFERENCE-BENCH-LADDER.md:340-385`,
`INSTRUMENT-MAP.md` (sections + `:607-641`), `DIAGNOSTICS-TOOLKIT.md` (tool rows), `CORPUS-CHARTER.md`,
`EVAL-V2-INVENTORY-2026-09-25.md:324-370`, `TEXEL-FIT-C-DESIGN-2026-09-27.md:69-100`, `EVAL-V2-RETUNE-PLAN-2026-09-22.md`.
Code: `eval_v2.cpp` (`:195-229`, `:1296-1335`, `:1540-1660`, `:1790-1821`, `:1826-1885`, `:2090-2132`, `:2228-2306`,
`:2550-2592`, `:3700-4044`, `:4196-4225`), `diagnostics/_revival_screen.py`, `_px_depth_fit.py`, `_joint_depth_preview.py`,
`_eg_leg_inspect.py`, `_depth_residual_pass.py`, `_texel_pst_fit.py:1-60`, `_texel_kauf_fit.py:1-40`,
`_texel_engine_pass.py:1-30`, `gen_kp_fens.py` (grep), `_triangulate_sf11.py` / `_term_separability.py` (grep),
`selfplay/overnight_runner.sh:1387-1423`.
Data: `E:/chess_data/` listings, npz headers, csv headers and row counts; queue logs q37-q43.
External (fetched by the reference-research pass; URLs in §3): Ethereal `src/tuner.c`/`tuner.h` + Grant's `Tuning.pdf`;
Weiss `src/tuner/tuner.{c,h}` + `evaluate.c`; fishtest wiki (Creating-my-first-test, FAQ) + `spsa_workflow.py`; zamar/spsa
README; CPW "Texel's Tuning Method", "Stockfish's Tuning Method", "SPSA", "Automated Tuning"; lynx-chess/texel-tuner;
KierenP/ChessTrainingSets; talkchess threads via search snippets only (direct fetch refused).
