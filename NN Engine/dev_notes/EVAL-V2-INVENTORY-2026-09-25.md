# v2 inventory — what is missing, what is parked, what is built and unused, and how to measure each

@author: Ranuja Pinnaduwage (maintained with Claude)

Written 2026-09-25 from four read-only audits of the dev notes, memory and code. I spot-checked every load-bearing quote
against its file. Abbreviations:

| short | file |
|---|---|
| REG | `EVAL-V2-PARKED-REGISTER.md` |
| GAP | `EVAL-V2-GAP-AUDIT-2026-09-21.md` |
| CFG | `EVAL-V2-CURRENT-CONFIG.md` |
| S2 | `EVAL-V2-SLICE2-MOBILITY-DESIGN.md` |
| S3 | `EVAL-V2-SLICE3-DESIGN.md` |
| LOG | `EVAL-V2-REBUILD-LOG.md` |
| RET | `EVAL-V2-RETUNE-PLAN-2026-09-22.md` |
| IM | `INSTRUMENT-MAP.md` |

## 0. THE CORRECTION THIS DOCUMENT EXISTS FOR
I closed the eval lane on "fifteen consecutive move-null concepts". That headline mixes three different classes, and only
one of them is a refutation. It needs these corrections:

1. **Never built.** About 13 concepts, including four that all four references carry:
   - king shelter / pawn storm;
   - the endgame scale factor;
   - a tapered (mg/eg) PST;
   - king-to-pawn distance / pawnless flank.
2. **Built but unreadable.** About 10 items. They were closed by an instrument that cannot resolve them, or tested alone
   below the resolution floor.
3. **Rejected on a real negative.** About 10 items. Only this class is closed.
4. **The §I accuracy instrument's GAIN side predicts nothing** (LOG:3263-3269: harm agrees with moves 4/4, gain disagrees
   4/4). Proof in practice: `MOB_V2_EG_PCT` sits in S2:226's §I-REJECTED row, and it then SHIPPED at 125 on games
   (~+10 Elo). ⇒ **Every §I-only rejection is unread, not closed.**
5. **Only one gain-seeking bundle was ever gamed** (09-16: pin + exlow + bishop pair + space, H1 +60.7). Every other parked
   item was judged alone. Our own arithmetic says no single term except material can clear the ~2-2.5pp bar
   ("term-at-a-time is closed by arithmetic").
6. **A textbook Texel fit has never been run on v1 or v2** (§3). The law "corpus fit is anti-correlated with Elo" was
   measured on **SF-label distillation over subsets of v1's knobs**, not on the recipe hobby engines used.
7. **The passer ladder was rejected on `PASSER_V2_MAG=60`.** The shipped value is now 100.
8. **Corrhist:** my handoff said "resolved negative on v1". The register says UNREADABLE (REG:97): July harness, q-cache
   masking, STS only. The 08-24 closure was an OFFLINE signal analysis (position-local residual; sample starvation). It
   was **never gamed**. Treat it as unread.

## 1. MISSING OR PARKED, AND WHY

### 1A. Never built
| concept | refs | why not built | status of the reason |
|---|---|---|---|
| **KS shelter / storm** | 4/4 | "needs pawn structure", "eval has no castling rights" | **both STALE** (REG:27). Live blocker: who owns storm vs OvD (GAP K1). SF double-wires it; Ethereal's table fails our mirror gate |
| **Endgame scale factor** | 4/4 | needs the (mg,eg) pair | pair built 09-21 (§2A). ⚠️ The endgame-headroom screen found only 5 failures in 315 at d12 ⇒ probably teacher value more than Elo |
| **Tapered PST** (per piece, mg + eg; endgame king centralisation) | 4/4 | needs the pair; the tables are `constexpr`, not file-loadable | the king-only subset (`PST_V2_KING_EG_ONLY`) was tried at 3 mp, which is not a test of the concept. **Never attempted** |
| **King far from pawns / pawnless flank** | 3/4 | "never tried, never considered" (REG:118) | no code exists |
| **KingProtector** (minor-to-king distance) | 3/4 | "overlaps the KS zone" (`eval_v2.cpp` comment) | REG:156-159: the overlap is "partial, not empty" |
| **KS defender count** · queen threats · doubled-isolated · square rule · closedness→piece values | 2-3/4 | GAP tier 3, not reached | no knobs exist. v1's `pawn_closedness()` is built and unused |
| **Lazy-eval band** | 2/4 | "lane recorded, not started" | gates threats' cost question and winnability |
| **Winnability** | 3/5 | v1's version was structurally faulty (CFG:378) | v2 needs the eg leg (§2A) |
| **OvD** (owner's concept) · **capture gains** | 0/5 (ours) | slice-5 gate | capgains: clean v1 ablation is move-null (+0.4pp) but a real critical-band accuracy (~6.8pp) ⇒ teacher value |
| central · heat map as an addition | 0/5 | on-rule / one-owner charter | stay closed |

### 1B. Built, parked, UNREADABLE (each gated to 0)
| knob | what closed it | why that isn't a verdict |
|---|---|---|
| `PS_V2_WEAKUNOPP_MG/EG` (4/4 unanimous) | d7 regret flat, 49.3 vs a 49.8 null | flat and never negative. Its trigger, "the joint retune", fired, and the retune ran WITHOUT it |
| `EVAL_V2_PAWN_MG` 550-600 (material taper) | regret −0.6/−0.8pp (σ≈0.88); STS inside ±150 | "UNDECIDED, not refuted" (CFG:130). Never gamed. Must be co-swept with `RFP_MARGIN` |
| `PS_V2_REAR_DOUBLED` (P4) | harm check passes; fires on 3.4% | below regret's resolution by construction. The owner's call |
| `PST_V2_KING_EG_ONLY` | 3 mp median | unmeasurable alone. Frozen out of the retune |
| `KS_V2_EG_PCT` 40 / 20 | inside the neutral band, **one neutral arm only** | unresolved. A revisit must condition on ATTACKER PRESENCE, not `phase256` |
| `WEAKQ_V2_PCT=25` (shipped) | "inert on §I" | §I gain side is void; never ablated |
| `THREAT_V2_PCT` | regret null on both corpora (50.1/50.1 · 49.7/50.5) | a move-null on a resolving instrument ⇒ closest to refuted of this group. Its triggers (lazy eval, queen legs) never fired |
| space · bishop pair | rode the 09-16 bundle; leave-one-out cost 0.00 / +0.11 | effectively retired |

### 1C. Rejected on a resolved negative (stay closed)
- **Rook files:** two disjoint regret sets, never positive.
- **`PASSER_V2_PATH_PCT`:** 7 arms, WAC −4, STS −89. ⚠️ Measured on the old passer magnitude (60, now 100); one cheap
  re-screen at 100 is the only thing worth doing.
- **`KS_V2_EG_PCT=70`:** WAC −9.
- **Kaufman:** monotone §I harm, play-distribution +24.8 worse.
- **Connected pawns**, both shapes.
- **Long diagonal.**
- **`MOB_V2_SAFE`.**
- **Tempo:** mechanism is margin coupling.
- **Tier-2b:** tablebase ground truth, zero headroom.

⚠️ The **§I-only rejections** of `MOB_V2_TABLE` 1-3, `MOB_V2_EXCL_QUEEN` and `PASSER_V2_MG_PCT` sit in the same row that
wrongly rejected `MOB_V2_EG_PCT` ⇒ they are UNREAD.

## 2. BUILT BUT UNUSED

### 2A. Eval
- ★ **`EVAL_V2_PAIR=1`**, the mg/eg accumulator.
  - Validated: bound 3.00 mp vs a 16 mp bound, symmetry 0/800, fingerprint `246 / 50,631,624`.
  - **Never gamed and never used** for its purpose (tapered PST, scale factor, KS phase legs, winnability).
  - This is the gate to the largest unbuilt cluster.
- The UNREADABLE knobs in 1B.
- `KS_V2_ATT_PROFILE`, `KS_V2_CHK_PROFILE`, `KS_V2_CHK_COUNT`, `KS_V2_PAWN_ATT`: built; the header says "NEITHER IS
  TESTED". No measurement was found (check `KING_SAFETY_MODEL.md` before calling them never-measured).
- `EVAL_V2_RUNG`: effectively inert now that every term has its own knob.

### 2B. Search — ☠️ every verdict below is v1-only, and most are fixed-depth bench reads
No search-lane memory mentions `EVAL_ARM`/v2. Only `RFP_MARGIN`, `FUTILITY_MARGIN_SCALE` and `QDELTA_PERMOVE_MARGIN` were
ever swept on v2.

| feature | default | v1 verdict | why it is not closed |
|---|---|---|---|
| **Node's own TT store** (`ENABLE_NODE_TT`) | OFF | "unresolved"; STS +38 (inside ±150) | ★ Standard in every reference. We store in the PARENT's frame, so no node has its own best move |
| **TT-move ordering** (`ENABLE_TT_MOVE`) | OFF | "live and harmful": **−3 WAC** | −3 is inside WAC's ±5-6 resolution, and WAC is anti-correlated with strength. Standard engines order the hash move first; ours uses a moveGenCache cutoff promotion instead. Unread |
| **Singular extensions** | OFF | "mirage": +9 WAC at fixed depth, −1 at equal nodes, STS −56 | bench-only, v1, and a 1.56% fire rate (SF's gate is far more selective). Never gamed |
| **IIR** | OFF | best-profiled candidate: −16% quiet nodes, +1 ply | **games never run** |
| **ProbCut** | OFF | "banked" (07-09 gauntlet, record not located) | margin judged ~1.5× too conservative; re-test never done |
| **Corrhist** | OFF | offline signal closure | never gamed (see §0.8) |
| Quiet SEE pruning · 2-ply cont-hist · NMP variants · LMR variants · qsearch options (`QSTANDPAT_SEED`, `QSEE_RESORT`, `QCHECK_FULL`) · OTV · improving | OFF | mixed, mostly benches; qsearch options "never games-tested" | all at `EVAL_ARM=0` |
| **Interior razoring** | ABSENT | — | only root-list razoring exists |

☠️ **v1-leak hazard:** `RFP_EVAL_MODE=1`, `FUTILITY_EVAL_MODE=1` and `ENABLE_IMPROVING` call v1's `cheap_eval` even under
`EVAL_ARM=1`. Mode 2 silently degenerates to full v2. All default 0, so shipped v2 is clean, but any v2 search sweep must
avoid them.

**Standard-feature checklist (code-verified):**
- PRESENT: TT (parent-frame), aspiration, NMP, RFP, futility, LMP, LMR table, losing-capture SEE prune, history, capture
  history, 1-ply cont-hist, killers, countermoves, check extension, qsearch SEE, per-move delta pruning.
- PRESENT-BUT-OFF: node TT, TT move, singular, IIR, ProbCut, quiet SEE prune, corrhist, improving.
- ABSENT: interior razoring.

### 2C. Infrastructure
- **No pawn / KS / material cache**, though v2's layer split was designed for one (speed only).
- **No lazy-eval band.**
- **The replacement layer is draw-only**: it has no scaling channel.

## 3. TEXEL: NEVER RUN, AND IT IS THE HOBBY-ENGINE TOOL
Hobby engines at 3000+ tune the eval with **Texel**. The recipe:
- the logistic of the qsearch-resolved static eval, fitted against **own-game RESULTS**;
- on QUIET positions;
- over the **whole vector, including tapered PSTs**;
- with a fitted K;
- on ~1M positions.

It costs hours on one machine and needs no games during the fit.

**Our record:**
| fits | labels | what was fitted | quiet | games result |
|---|---|---|---|---|
| 5 (A, D1-4, E, F) | SF11 or SF18 values | v1 knob subsets | no | the only game read is E's **−85.6 ±74 over 116 games**, 08-07, i.e. in the harness contamination era |
| B (07-04) | game results, 183k rows | 8 v1 term scales, PSTs pinned | not implemented | objective **FLAT (0.02%)**, never gamed |
| C (07-16) | game results, 1,705 rows | scales only | — | — |

- **No fit ever included PST cells.** Making the `constexpr` PSTs loadable was scoped as "the big build" and never done.
- **v2 has never had any corpus fit.**
- The record itself concedes that v2 removes both structural causes of the old failures: degeneracy and colour asymmetry
  (CORPUS-CHARTER:52-53).

⇒ "Corpus fit is anti-correlated with Elo" is established for **SF-distillation of v1 knob subsets**. It is untested for
Texel on v2.

**Data we already have:** KS runs 1 and 2 alone produced ~26,000 v2-vs-v2 games at d6. Earlier SPSA runs add more.
That is a Texel-scale corpus of our own games, already on disk.

## 4. HOW TO GAUGE EACH ASPECT (full detail: IM; summary of the instrument audit)
Two universal rules:
- **A null is not zero until measured**, per corpus and per stratum.
- **Proxies can only veto; GAMES decide.**

| instrument | what it answers | blind to | resolution | correct role |
|---|---|---|---|---|
| **Static eval MSE vs SF18 search (§I)** | how close the static value is to SF18 d14 | whether moves change, search interaction, rare regimes | floor ~0.05%. **Off-distribution by default**: v2 beats v1 on `diverse_corpus_wide` but loses on our own play | **VETO only** (harm 4/4 right, gain 0/4). Never an objective: minimising it shrinks the eval (−85.6 Elo), and it trades away the critical band where we beat SF11 |
| **SF11-static control** | could a hand-written eval have found this move | — | ceiling 41.8% | mandatory control on any static-vs-search comparison; a bound, not a target |
| **WAC bench** (solves / nodes / EBF) | byte-identity to a baseline; gross tactical loss | strength (anti-correlated across 7 engines) | ±5-6 solves; printed EBF is not a real EBF | fingerprint + tactical veto |
| **STS300, fixed depth** | did something break at rung scale | ordering and search terms (mobility read flat, was +162 Elo) | ±150 | regression guard; never rank inside ±150 |
| **STS300, equal nodes** | cross-engine eval + search at equal work | NPS | own-arm null unmeasured (assume ≥±150) | reference ladder / attribution. Measure the node budget ON the suite |
| **d7 regret** (two corpora) | does a changed d7 move get better by SF18 | anything gated at depth ≥6; criticality; rare classes | read **win%** vs a measured paired null; bar ~2-2.5pp; whole-eval prize only +7pp | **VETO** on multi-σ negatives. Winner's curse seen 3×. Use `_paired_null.py` |
| **Move-match** | does the move flip | quality | aspiration noise alone flips 20.8% | liveness screen only |
| **Fire-rate probe** (`_eval_knob_delta.py`) | does a knob move the eval, and how often | direction, value | exact | admission screen for SPSA; necessary, never sufficient |
| **Symmetry** (`_eval_symmetry.py`) | `eval(mirror) = −eval` | nothing, IF the term fires in the corpus | exact | hard ship gate; prove the term fires first |
| **Fixed-d6 games** | eval-arm difference at ~1,680 g/hr | pruning, speed | — | SPSA candidate generator only |
| **`NODE_LIMIT=50000` games** | eval Elo at ~800 g/hr | NPS | ±10 at ~6,000 games | **decider for eval** |
| **Timed LIGHTNING SPRT** | Elo in the real regime | — | ±25 at 1,200; the harness null is +4.6, not 0 | **decider for search / speed**. An SPRT decides; pool games for magnitude |
| **SPSA drift z + replication** | is a tuned knob's drift real | — | no single run has cleared \|z\|=2 | two seeds, then an SPRT |
| **NPS / time benches** | speed | — | ±14.6% run to run | a change needs ≳35% nodes/time to matter; quote TIME, not nodes |

**Which instrument decides which question:**

| change | screen | veto | decider |
|---|---|---|---|
| eval term or small-term bundle | fire-rate + symmetry | regret (both sets, paired null), §I harm, STS −150 | NODE_LIMIT SPRT, **bundled**, gain-seeking |
| eval magnitudes | probe | regret, STS | SPSA ×2 seeds → NODE_LIMIT SPRT |
| **whole-vector / PST fit (Texel)** | held-out result-loss | symmetry, STS, WAC, §I harm | NODE_LIMIT SPRT vs shipped |
| search margin / feature | quiet-node + depth@1s benches at 2-3 budgets | WAC tactics, STS | **timed** SPRT (never fixed depth, never d7 regret: blind at d≥6) |
| speed | back-to-back `depth_nps_bench` on an idle box | — | fixed-time games only if ≥35% |

## 5. WHAT COULD HELP, AND THE CHEAPEST FAIR TEST FOR EACH
Ordered by expected value per hour. A NODE_LIMIT SPRT runs ~800 games/hr, so a ~15-Elo bundle resolves in ~2-4 h.

1. **Bundle A: built, small, each below its floor alone (gain-seeking SPRT at `NODE_LIMIT=50000`).**
   - Members: `PS_V2_WEAKUNOPP`, `PS_V2_REAR_DOUBLED`, `PST_V2_KING_EG_ONLY`.
   - Dilution control: a `WEAKQ_V2_PCT=0` ablation.
   - Cost: hours; no build.
2. **`EVAL_V2_PAIR=1` non-inferiority SPRT.** It is a prerequisite, so ask only "not worse". Cost: hours; no build.
3. **Texel feasibility on v2.** In order:
   - (a) extract quiet positions and results from the games already on disk;
   - (b) fit a small linear subset first (material + PST as they stand) with fitted K and a by-game holdout;
   - (c) game the result against shipped.

   If it gains, it becomes the tool for everything below and makes the post-search retune cheap. If it loses, we have
   finally measured the law on the right recipe. Cost: a few days of building (file-loadable tables, a feature
   extractor).
4. **Tapered PST**, on `EVAL_V2_PAIR=1`:
   - first arm: reference tables (SF11 / Ethereal / Weiss) converted by positional scale, then gamed;
   - second arm: Texel-fitted, if 3 works.
5. **KS coverage bundle**: shelter/storm + pawnless flank + KingProtector + defender count.
   - Needs a build and the storm-vs-OvD ownership decision first.
   - Screen with the two-depth move test before building (memory `eval-headroom-is-failures-that-persist-as-depth-rises`).
6. **Material taper co-swept with `RFP_MARGIN`**: 2×2 NODE_LIMIT games.
7. **Re-screen** the passer ladder at `PASSER_V2_MAG=100`, and the §I-only rejections (MOB table / EXCL_QUEEN) in an SPSA
   admission probe. Cheap, low prior.
8. **Search (next block, all at EQUAL TIME on v2):**
   - node-TT → TT-move → IIR → singular (cheaper gate) → ProbCut;
   - qsearch;
   - margins;
   - add interior razoring.

   Each of these was closed only on v1 benches.

## 6. RECOMMENDED ORDER
- **Eval block:** 1 → 2 → 3, then 4 and 5 depending on 3's outcome, then 6-7.
- **Then the search block:** 8.
- **Then** the final joint retune, which is cheap if Texel works.

This closes eval with the tool that makes returning to it later cheap, rather than leaving it with nothing but games.

## 7. NIGHT OF 2026-09-25 — PRE-FLIGHT AND REGISTERED PREDICTIONS
**Pre-flight (shipped base `V2_PRESET=shipped`):**
- **Bundle A** = `PS_V2_WEAKUNOPP_EG=127` (MG 0: every shipped v2 pawn term is endgame-weighted) +
  `PS_V2_REAR_DOUBLED=2` + `PST_V2_KING_EG_ONLY=1`.
  - Fire rate 60.8% of 4,000 positions, median 70 mp (WeakUnopposed alone: 46.4%).
  - Colour symmetry 0/800; file-mirror 21/651 at 5 mp (the known queen-PST residue).
  - Signed mean +35 mp ⇒ corpus skew, not colour (symmetry is clean).
  - WAC **255/300** (+1), 50,578,535 nodes (−4.5%). No veto.
- **`EVAL_V2_PAIR=1`:**
  - Bound on the shipped base: max **4 mp**, mean 1.2 mp over the 41.2% of positions that moved; signed mean −0.1. The
    bound holds.
  - WAC **244/300 (−10)**, 50,415,978 nodes. ⚠️ A ≤4 mp perturbation cannot cause a genuine tactical loss ⇒ this is
    d10 search chaos, and it means **WAC's floor for tiny perturbations is at least about ±10, not ±5-6**. It is not
    treated as a veto; the SPRT decides.

**SPRT 1** `sprt_bundleA`: Bundle A vs shipped, `NODE_LIMIT=50000`, LONG_FORMAT, conc 4, UHO seed 31, elo0 0 / elo1 10,
cap 6,000 games.
- Prediction, registered before any games: **H0 is more likely than H1** (~60/40). Every member was individually flat,
  and the bundle's evidence is only the 09-16 precedent.
- If H1: leave-one-out plus a `WEAKQ_V2_PCT=0` dilution control; pool games for magnitude.

**SPRT 2 (after SPRT 1)** `sprt_pair1`: `EVAL_V2_PAIR=1` vs shipped, same settings, elo0 −10 / elo1 0 (non-inferiority).
- Prediction: H1 ("not worse"), since the mean delta is 0.5 mp over all positions.

**Texel stage 1 done (09-25 night)** — `diagnostics/_texel_extract.py` → `E:/chess_data/texel/v2_stage1.csv.gz` (outside
OneDrive).
- **3,692,732 positions from 48,475 v2 games** (spsaeval3, spsarun2, spsaks1, spsaks2). 10% of games are held out by
  game hash.
- Dropped: 727k book positions · 437k in-check · 939k where the move played was a capture or promotion · 39k with no
  next move.
- Results: W 41.8% / D 19.2% / L 38.9%.
- **Label sanity:** the stored d6 search score predicts the result with fitted **K = 0.000594 / mp** (1 pawn ≈ 64%
  expected score).
  - Held-out MSE 0.1054 vs a constant baseline of 0.1922 (ratio 0.549).
  - MSE by |score|: 0.158 (<500 mp) · 0.155 (500-1500) · 0.109 (1500-3000) · 0.029 (≥3000; 34% of rows).
  - ⇒ The labels are informative. A third of the rows are decided positions, so consider down-weighting them in the
    fit.
- Next (after the SPRTs free the machine): stage 2 confirms quietness with the engine (static eval == qsearch), then the
  first small fit (material + PST) with fitted K and the by-game holdout, judged only by a NODE_LIMIT SPRT.

**Planned gate (owner's idea, 09-25): VARIANT-START ROBUSTNESS.** Trigger: the first Texel fit that wins on UHO.
- Replay candidate vs shipped from non-standard starts:
  - **symmetric** (both sides N→B · knights + queens only · shuffled 960-style back ranks WITH NO CASTLING RIGHTS):
    balanced, every game informative. The point is an unfamiliar configuration that avoids fitting known structure,
    not 960 castling;
  - **mixed** (one side's knights → bishops): tests imbalance pricing;
  - **odds** (knight / rook): tests conversion and draw logic only; results are mostly foregone.
- **Why:** several v2 terms are implicitly calibrated to the standard piece set, so a fit can overfit to standard chess:
  - PST means (+20 per knight, +38 per queen) re-price material;
  - bishop pair;
  - phase is computed from material;
  - mobility baselines.

  A candidate that wins on UHO but loses on variants learned the distribution, not chess.
- **Prerequisite (unchecked):** FEN-start support in `tournament.py` openings. No 960 castling support is needed:
  castling rights are dropped.
- ⚠️ Colour-reversed pairing per start (as the tournament already does) keeps the asymmetric starts fair.
- ★ **King-safety payoff (owner):** shuffled back ranks with no castling leave kings exposed on their start file ⇒ KS
  fires far more often.
  - The KS-shape lane closed 09-25 because its knobs fire on only 5-12% of positions. Variant data could make KS
    parameters identifiable (in a fit or SPSA), and it tests KS CONCEPTS apart from castled-king patterns.
  - It also shows where shelter/storm / pawnless flank (unbuilt) would matter.
  - ⚠️ Permanently uncastled kings overstate exposure ⇒ KS takes variant data as a SHARE only; the standard SPRT has
    the final say.
  - First cheap measurement: fire-rate probe (KS on vs off) on positions from a few hundred shuffled-start games vs
    the standard ~29%.
- **Training use (proposed):**
  - **Fit A** = standard games only.
  - **Fit B** = A + ~20% symmetric and mixed variant positions (not odds). King-shelter parameters are fitted on
    standard positions only. The starts are split, so the gate uses starts the fit never saw.
  - Compare on a standard SPRT, then on the variant gate.
  - Rationale beyond robustness: variants DECORRELATE features that always co-vary in standard games (piece counts ×
    phase × development; home squares × opening). That attacks the degeneracy the record says blocked every fit.
- **Later:** training data only for classes a fit shows are under-determined.

**SPRT 1 RESULT (09-26 ~01:20): Bundle A — H1 ACCEPTED.**
`+1208 -1087 =918 of 3213 (51.9%) elo ~ +13.1 +/- 14.1 LLR +3.018` (elo0 0 / elo1 10, NODE_LIMIT=50000, seed 31,
907 g/hr). My registered prediction (H0 ~60/40) was WRONG — the 09-16 pattern again: individually-flat members pay
as a group. ⚠️ Point estimate is inflated at the stopping bound ⇒ quote "≈ +10, bracketed"; pool with any follow-up.
Ship decision is the OWNER's. Follow-ups: leave-one-out + `WEAKQ_V2_PCT=0` dilution control (§5 item 1), and a
magnitude bracket if wanted.
**SPRT 2 launched** `sprt_pair1`: `EVAL_V2_PAIR=1` vs shipped (NOT vs shipped+A, so each result stays clean),
elo0 −10 / elo1 0, seed 32.

**SPRT 2 RESULT (09-26): `EVAL_V2_PAIR=1` — H1 ACCEPTED (non-inferiority, elo0 −10 / elo1 0).**
`+409 -348 =283 of 1040 (52.9%) elo ~ +20.4 +/- 24.8 LLR +3.017` (seed 32). Reads as NOT WORSE — not a gain claim (a
≤4 mp per-position change cannot plausibly be worth +20; ±25 wide). ⇒ the (mg,eg) accumulator is cleared to BUILD ON
(tapered PST, scale factor, KS phase legs). The pre-flight WAC −10 is confirmed as d10 search chaos, not tactics.
Machine free as of this entry. Owner decisions pending: ship Bundle A? ship pair mode as the base for the PST work?

**SPRT 3 launched (09-26, overnight):** `sprt_baseA_pair` = shipped + Bundle A + `EVAL_V2_PAIR=1` vs shipped, elo0 0 / elo1 10, seed 33 (fresh openings). Purpose: REPLICATE Bundle A before any ship, confirm A composes with pair mode (the PST build base), and POOL with SPRT 1 for magnitude. Registered prediction: H1 (A replicates; pair adds ~0).

**SPRT 3 RESULT (09-26 ~11:40): INCONCLUSIVE (time budget).** `+3537 -3372 =2619 of 9528 (50.9%) elo ~ +6.0 +/- 8.2 LLR +1.108`. Registered prediction (H1) NOT met: the true effect sits between the bounds. POOLED with SPRT 1 (12,741 games): see the doc line below. ⚠️ Correction: `sprt.py` ignores `--max-games` when `--max-minutes` is set (sprt.py:84) — the "cap 6,000" stated for SPRTs 1-3 was wrong; the real cap was 600 min.
Pooled SPRT 1 + SPRT 3: **12,741 games, 51.12%, elo +7.8 +/- 5.1 (95%)** ⇒ Bundle A is a small REAL gain (~+3 to +13). SPRT 3 alone +6.0 +/- 5.9 on fresh openings. Confound: SPRT 3 also carried `EVAL_V2_PAIR=1` (non-inferior, not pinned at 0).
