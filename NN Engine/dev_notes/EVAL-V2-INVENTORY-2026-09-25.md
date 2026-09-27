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

## 8. TEXEL FIT A — THE FIRST TEXTBOOK TEXEL FIT EVER RUN HERE (09-26)
**Engine** (`142a775`): `PST_V2_TAPERED` gives v2 its own (mg,eg) tables.
- The default tables reproduce shipped byte-for-byte (255 / 50,578,535).
- `PST_V2_FILE` loads fitted tables; `PST_V2_DUMP` writes the active ones; `PST_V2_ZERO` supplies the fixed part.
- Loading the dumped file round-trips byte-identically. Symmetry is 0/800 on a scrambled table with negative cells.

**Data:**
- `_texel_engine_pass.py`: 3.69M positions, ~21k positions/s per process.
- The linear model reproduces the engine: |full − (fixed + Xθ₀)| median **0.00 mp**, p99 2.4 mp, 0 inert rows.

**Fit** (`_texel_pst_fit.py`):
- 384 tied half-board cells, per-(piece,leg) mean pinned, L2 + smoothness on the change.
- Games weighted equally; decided rows ×0.25.
- K = 0.000631 / mp, frozen.
- Holdouts: val_hash 268k rows, val_run = all of `spsaks2` (1.0M rows).

| λ₂ | val_hash | val_run | max \|Δ\| |
|---|---|---|---|
| 1e-10 | −2.09% | −2.33% | 329 mp |
| 3e-11 | −2.40% | −2.67% | 507 mp |
| 1e-11 | −2.54% | −2.83% | 660 mp |
| **3e-12 (chosen)** | −2.59% | −2.89% | 849 mp |

- With 5 bootstraps of 1.5M rows: **FINAL val_hash −2.38%, val_run −2.75%**; 220/368 cells stable.
- Compare: the only prior result-label fit (v1, 8 term scales) was −0.02%.

**Shapes:**
- King mg: corner high (b1 +761), centre low (d3 −468). King eg: an active king is rewarded. Both match all references.
- Knight rim strongly negative.
- Queen eg punishes a passive queen (ranks 1-2 up to −800).
- Pawn mg pays advancement (rank 7 up to +647): v2's passer term is eg-only, so the PST absorbs the mg passer value.
- ⚠️ Rook mg spread 1,227 mp (ranks 1-2 −680, ranks 6-8 +550). Probably part real (activity, 7th rank), part the
  Texel confound: advanced pieces correlate with already winning.
- Spreads are 400-1,270 mp vs today's 25-60 mp, but comparable to SF11's PSTs converted at PAWN scale.

**Gates:**
- Colour 0/800 AND file mirror **0/651**: the tied fit removed the old queen residue, so v2 is fully mirror-symmetric.
- WAC 249 (−6; inside the ±10 perturbation band), 53,405,821 nodes (+5.6%).

**SPRT 4** `sprt_fitA`: fit A vs shipped (incl. Bundle A), NODE_LIMIT=50000, seed 34, elo0 0 / elo1 10, 600 min cap.
- Registered prediction: **H1**, but with a real chance of H0 because of pruning coupling (the margins were tuned
  on a 25-60 mp PST).
- If H0: test the conservative λ₂=1e-10 table (`pst_fitA_cons.txt`) and/or re-sweep `RFP_MARGIN` with fit A before
  concluding.

**Conservative candidate** `pst_fitA_cons.txt` (λ₂=1e-10, 5 bootstraps): val_hash **−2.05%**, val_run **−2.30%**, 282/368 stable; spreads 56-527 mp (vs 400-1,270 for fit A). Keeps ~85% of fit A's held-out gain at ~1/3 the magnitude ⇒ the fallback arm if SPRT 4 reads H0 on pruning coupling.

## 9. OVERNIGHT PLAN 2026-09-26 → 27 (owner out ~18:00 → 02:00-04:00; agreed)
**Built today, before leaving:**
- `cc53b53`: tournaments accept FEN starts. `gen_variant_starts.py` → `selfplay/openings_variant.txt` holds 2,376
  starts, 88 in each of 27 families:
  - symmetric arrays: std-shuffle, all N→B, all B→N, N→R, B→R, no queen, Q→N/B/R, NQ-only, BQ-only, RQ-only,
    double queen, minors only;
  - near-equal mixed material;
  - K+pawns endgames with R / N / B / Q / N-vs-B / RN-vs-RB.
- `_texel_phase_grid.py` (Fit A2): engine passes for EVAL_V2_MG_LIMIT {50000, 61700, 72000} × EG_LIMIT {10000, 15800,
  22000}. A liveness check confirms the phase actually changed.

**Decision tree (autonomous; NOTHING ships without the owner):**
1. SPRT 4 (fit A vs shipped) verdict →
   - **H1:** SPRT 5 = fit A replication on fresh openings (seed 35), then pool.
   - **H0 / inconclusive:** SPRT 5 = conservative `pst_fitA_cons.txt` vs shipped; if that also fails, a 2×2 of
     fit A × `RFP_MARGIN` {1000, 1300}.
2. A 2-game smoke test of the variant book at d4 before any variant run (FEN starts through the real harness).
3. **Variant gate:** the best PST candidate vs shipped on `openings_variant.txt`, a fixed ~2,000 games at
   NODE_LIMIT=50000 (a measurement, not an SPRT). Read overall and per family.
4. **Fit A2:** refit the PST per phase setting (`_texel_pst_fit.py PASS=pass_mgX_egY LAMBDAS=3e-12 BOOT=1`); rank
   the settings by val_run. Candidates only; games wait for the owner.
5. If the machine is free: d6 variant-start self-play for Fit B data (shipped vs shipped, ~3-5k games).

**Added by the owner before leaving (09-26 ~16:00):**
- **ODDS book** `selfplay/openings_odds.txt` (`gen_odds_starts.py`): 480 starts. Standard position with White
  handicapped by pawn / knight / bishop / exchange / rook / queen / two minors / rook-for-three-pawns, 60 each, short
  random walks.
  - Why: odds show CONVERTING an edge and DEFENDING a deficit. SF at a short time beats our LIGHTNING even a knight
    down.
  - Report with `diagnostics/_variant_report.py TAG=<tag> BOOK=<book>`: per family, plus odds split by role.
- **MORNING REPORT** (owner's ask): how much v2 improved on every instrument, vs the references and v1.
  - Arms:

    | arm | knobs |
    |---|---|
    | v1 | engine default |
    | v2-prev | `V2_PRESET=shipped PS_V2_WEAKUNOPP_EG=0 PS_V2_REAR_DOUBLED=0 PST_V2_KING_EG_ONLY=0` |
    | v2-shipped | `V2_PRESET=shipped` (Bundle A) |
    | v2+fitA | shipped + `PST_V2_TAPERED=1 PST_V2_FILE=/mnt/e/chess_data/texel/pst_fitA.txt` |
    | v2+fitA_cons | shipped + the conservative table |

  - Instruments (commands per the 09-26 lookup):
    1. §I win%-MSE vs SF18 d14, on BOTH `playdist_ceiling.csv` (on-distribution, trustworthy ranking) and
       `diverse_corpus_wide`. Reference rows (SF11, SF15.1c/n, SF18 static) come from `_reference_ceiling.py`, run
       once; our arms from `_eval_accuracy_arms.py`.
    2. d7 regret on both sets, base = v2-shipped, arms fitA / fitA_cons / v2-prev (i.e. Bundle A reversed), with
       neutrals `ASPIRATION_DELTA=300` + `EVAL_NOISE_SIGMA=30`; win% + paired null.
    3. STS300 at d10 and at `NODE_LIMIT=249014 MAX_DEPTH=64` (equal-nodes command inferred, flagged).
    4. WAC fingerprints.
    5. Odds: v2-shipped (and fitA if it passes) vs SF18 at fixed nodes on the odds book; v1 on the same for contrast.
  - Scheduling: after the SPRT branch; benches are deterministic, games in the remaining time.
  - ⚠️ §I can VETO but never PROMOTE; games decide.
- ⚠️ **Owner's caution on lopsided starts:** with colour-swapped pairs, a start where the side with the edge always
  wins gives a 1-1 pair ⇒ ~50% however much an arm improved.
  - `_variant_report.py` now prints the PAIR (pentanomial) split and the share of INFORMATIVE pairs (≠ 1.0) per family.
    Baseline: Bundle A on UHO had **66% informative** (1,059 / 1,606).
  - Rules:
    - In engine-vs-engine matches, read only families near that baseline; heavy odds (queen, rook, two minors) are
      expected to saturate.
    - Odds are read mainly AGAINST A FIXED SF18 opponent at fixed nodes, where the handicap offsets a strength gap:
      each arm plays the same starts, and conversion / defence rates are compared.
    - For tuning data, decided positions are already down-weighted by the fitter.

**SPRT 4 RESULT (09-26 ~16:30): Texel fit A — H1 ACCEPTED.** `+185 -101 =70 of 356 (61.8%) elo ~ +83.6 +/- 42.4
LLR +3.036` (vs shipped incl. Bundle A, NODE_LIMIT=50000, seed 34). Pairs: 178, [14 16 66 36 46], 63% informative,
net +84 half-points. The largest single eval gain in the record (retune ≈ +10, Bundle A ≈ +8, mobility area ≈ +31),
but ⚠️ it stopped at 356 games ⇒ the estimate is inflated at the bound (precedent: +60.7 → +31 pooled). Registered
prediction H1 ✓.
**SPRT 5 = FIXED-LENGTH replication** (not an SPRT, so there is no stopping-bias): `fitA_rep`, 2,000 games, fresh
openings seed 35, same configs. Pool with SPRT 4 for magnitude. Ship decision = OWNER, after replication.

**FIT A2 (phase limits) — FLAT, CLOSED (09-26 ~18:00).** 9 settings of EVAL_V2_MG_LIMIT {50000, 61700, 72000} ×
EG_LIMIT {10000, 15800, 22000}.
- Each setting has its own engine pass. Liveness: the phase changed on 56-74% of rows.
- Each setting then got a PST refit (λ₂=3e-12, 1 bootstrap).
- Fitted val_run spans **0.137965 (61700/10000) … 0.138241 (72000/22000)**. The current setting (61700/15800) reads
  0.138014, only 0.035% behind the best: under the ~0.05% resolution floor.
- ⇒ The phase definition is not a lever on this data; no candidate goes to games. The owner's
  "material as an input feature" is answered for the phase channel.
- Aside (unrelated bug, not fixed): the invalid-limits fallback in search_engine.cpp restores 40000/10000, not the real
  defaults 61700/15800.

**SPRT 5 / REPLICATION RESULT (09-26 ~19:00): fit A CONFIRMED on fresh openings.** `fitA_rep`, fixed 2,000 games,
seed 35: `+1144 -498 =358 (66.1%) elo +116.4 +/- 14.6` · as White +577 −243 =180, as Black +567 −255 =178. Pairs
[63 76 332 210 319], 67% informative (UHO baseline 66%). **POOLED with SPRT 4: 2,356 games, 65.5%, elo +111.3 +/- 13.4.**
Registered prediction (H1) ✓, and the magnitude came in LARGER than the stopped SPRT estimate.
⚠️ **The open risk is SELF-PLAY EXPLOITATION:** fit A was trained on v2-vs-v2 results and tested against v2, so part of
the gain may be steering into positions v2 specifically misjudges. ⇒ EXTERNAL CHECK before any ship: the calibrated
gauntlet (ours NODE_LIMIT=250000 vs SF18 --sf-nodes 400; the v1 anchor was ~51%), 500 games per arm, SAME seed 36:
`gauntlet_shipped` then `gauntlet_fitA`. The transferable gain = the difference between the two arms' scores.

**EXTERNAL CHECK (09-26 ~20:15): fit A's gain TRANSFERS, at ~1/3 of the self-play size.**
Calibrated gauntlet, ours NODE_LIMIT=250000 vs SF18 --sf-nodes 400, 500 games per arm, SAME seed 36 (identical openings
and colours):
- `gauntlet_shipped`: 66.6% (+120 vs SF). `gauntlet_fitA`: 71.5% (+160 vs SF).
- **Paired difference: +4.90pp ± 4.75 (95%), z = 2.02 ⇒ ≈ +40 Elo [+1, +83].** 279 games changed result (147 better, 132
  worse, net favours fit A by half-point weight).
- ⇒ Self-play +111 vs external +40: the expected self-play inflation (~2.8×). The gain is real against an engine fit A
  never trained on, but only just significant at n = 500.
- ▶️ Tightening with a second seed (37), 500 per arm, then pooling. Registered expectation: the pooled gap stays
  positive at ~+25..+50.
- Note for the record: shipped v2 at 66.6% vs the August v1 anchor of ~51% at the same setting.
**Seed 37 replicates (09-26 ~22:15):** shipped 65.7% · fit A 70.1% · paired +4.40pp ± 4.94.
**POOLED external (1,000 paired games, 2 seeds): shipped 66.15% (+116 vs SF18@400n) → fit A 70.80% (+154);
difference +4.65pp ± 3.43, z = 2.66 ⇒ ≈ +37.5 Elo [+9.6, +67.4].** Registered expectation (+25..+50) ✓.
⇒ Fit A is REAL and TRANSFERS: ≈ +111 in self-play, ≈ +38 against an engine it never trained on. Ship decision = owner.

## 10. MORNING REPORT DATA (09-26 night)
**Eval accuracy vs SF18 d14 search** (val win%-MSE, 3,000 rows each, same rows and split for all rows; lower is better):
- References from `_reference_ceiling.py`; our arms from the new `_accuracy_arm_grid.py`.
- Why the new tool: `_eval_accuracy_arms.py` reads `best_cp` and silently reported "no rows" on these `target_total`
  corpora. It is now wrapped via `_reference_ceiling.py REFS=0`.

| evaluator | playdist_ceiling (own play) | diverse_corpus_wide |
|---|---|---|
| SF18 static | 61.93 | 68.85 |
| SF15.1 NNUE | 72.76 | 61.61 |
| SF11 classical | 151.41 | 95.26 |
| **v2 + fit A** | **170.17** | **126.93** |
| v2 + fit A cons | 176.47 | 132.14 |
| v2 shipped (Bundle A) | 186.88 | 148.05 |
| v2-prev | 188.19 | 149.31 |
| v1 | 188.85 | 238.77 |
| SF15.1 classical | 192.35 | 139.20 |

⇒ Fit A: −8.9% own-play and −14.3% diverse vs shipped. On own-play it closes ~half the gap to SF11, and it is now well
ahead of SF15.1 classical on both corpora. Bundle A: −0.7% / −0.8%, as expected of small defect fixes.
⚠️ §I vetoes, never promotes: this corroborates the games, it does not replace them.
**d7 regret** (primary set) running: base shipped; arms fitA, fitAcons, v2prev, null_asp300, null_noise30; JOBS=2. One
invocation (DUMP keeps only the last arm ⇒ no paired null this time; read win% vs the two neutrals).
**STS300 and WAC:**

| arm | STS d10 | STS @249,014 nodes | WAC d10 |
|---|---|---|---|
| v1 | 1796 (register) | **1752** (first measured) | 250 / 35,310,778 |
| v2 shipped | 1803 | 1687 | 255 / 50,578,535 |
| v2 + fit A | **1888** | **1760** | 249 / 53,405,821 |
| v2 + fit A cons | — | — | 249 / 48,801,449 |

Fit A vs shipped: +85 at d10, +73 at equal nodes. Both are inside the ±150 floor, but same-signed with the games.
Equal nodes does not credit v2's ~41% NPS advantage over v1.

☠️ **d7 regret run 1 DISCARDED.** fitA, fitAcons and v2prev returned byte-identical rows (8,229 changed, 46.9%), and the
neutrals changed 55% of moves instead of ~33%.
- Cause: `_ks_footprint_regret.py` candidate arms did NOT inherit `BASE_KNOBS`, so every arm ran v1 + its knobs, while
  only the base pass ran shipped v2.
- Earlier ladders repeated the full v2 block in every arm (their neutrals changed ~32%), so past results are
  unaffected.
- **Fixed:** arms now inherit BASE_KNOBS and override them; backwards-compatible. Re-running.
- The silent-fallback signature (identical rows across arms) caught it before it could be read as a result.

**VARIANT GATE (09-27 ~01:30): fit A vs shipped on `openings_variant.txt`.** Fixed 2,000 games, NODE_LIMIT=50000, seed
40, conc 2.
- **Overall 52.6%, +17.9 Elo [+4.6, +31.3].** Pairs 1000 [98 127 470 184 121]; 53% informative (UHO 66%: the endgame
  families are drawish); net +103.
- ⇒ Per the owner's gate reading ("wins on UHO AND holds/wins on variants ⇒ robust"): ROBUST, not structure
  memorisation. But the gain on unfamiliar starts (+18) is ~1/6 of the standard-opening self-play gain (+111).
- Families (60-90 games each; every CI crosses 0 except where noted):
  - Rook-heavy positive: BtoR +96 [+26, +176] · RQ_only +91 [+23, +167] · NtoR +55 · QtoN +78 [+4, +160].
  - Bishop-heavy / pure-960 negative-leaning: array_std (standard set shuffled) −50 · BQ_only −30 · all_NtoB −24 ·
    QtoB −10.
  - Hypothesis only: the fitted castled-corner king and standard bishop cells don't transfer to shuffled back ranks.
    Revisit with Fit B (variant data in the fit).
  - K+P endgame families: 21-43% informative ⇒ mostly draws, non-discriminating.

**d7 REGRET, primary set (15,000 positions), corrected run.** Base = shipped; every arm inherits it.

| arm | changed | win% | Δ regret | critical band (>20%) |
|---|---|---|---|---|
| null asp300 | 32.8% | 50.0 | −0.031 | 60.0% (n=36) |
| null noise30 | 34.8% | 50.6 | −0.013 | 60.0% (n=31) |
| v2prev (Bundle A off) | 31.1% | **49.3** | +0.140 | 68.2% (n=23) |
| **fit A** | 45.1% | **52.1** | **−0.233** | 66.7% (n=43) |
| fit A cons | 42.1% | **52.5** | **−0.245** | 65.7% (n=36) |

- Fit A and fit A cons read +1.5..+2.1pp over the null band. That is just under the ~2-2.5pp bar, but the changed-move
  regret falls ~7%.
- Removing Bundle A reads −0.7..−1.3pp; consistent with its games.
- Cross-set `_v2` running.

**ODDS, engine vs engine (09-27): fit A vs shipped on `openings_odds.txt`.** 960 games, seed 41.
- **Overall 56.1%, +42.9 [+21.7, +64.5].**
- Owner's saturation warning CONFIRMED: queen odds 2% informative pairs, two minors 7%, rook 25%. The discriminating
  families are pawn (62%) and exchange (52%).
- By role (all families): fit A **converts 88.8%** of its edges vs shipped's 76.5%, and **defends 23.5%** of its
  deficits vs shipped's 11.2%. Better at BOTH.

  | family | fit A converts / shipped converts | fit A holds / shipped holds |
  |---|---|---|
  | pawn | 77.5% / 50.8% | 49.2% / 22.5% |
  | exchange | 76.7% / 57.5% | — |

- ⇒ Directly relevant to the owner's observation (SF beats our lightning a knight down): a large part of the
  material-handling weakness was the placement eval.

☠️ **Harness fix:** `vs_sf.py` still hard-coded the standard start. With a FEN-start book it would have silently played
standard-position games (each odds line parses as an empty move list). It now uses `opening_start_fen()`. Verified:
game 0 of `oddsSF_shipped` starts from a rook-for-pawns FEN.
**ODDS vs SF18** (ours NODE_LIMIT=250000, SF18 --sf-nodes 400, 480 games, seed 42): `oddsSF_shipped` running, then
`oddsSF_fitA` on the same seed.

**d7 REGRET cross-set (`game_regret_set_v2.csv`, 11,940 positions), same arms and base:**

| arm | changed | win% | Δ regret |
|---|---|---|---|
| null asp300 | 33.9% | 50.7 | +0.163 |
| null noise30 | 36.2% | 49.6 | +0.091 |
| v2prev | 32.3% | 49.1 | +0.149 |
| **fit A** | 46.0% | **52.5** | **−0.183** |
| **fit A cons** | 41.7% | **53.6** | **−0.231** |

- ⇒ **REPLICATES.** Fit A reads 52.1 / 52.5 (≈ +2pp over nulls on both sets); fit A cons 52.5 / 53.6 (≈ +2.8pp).
  Regret falls on both sets while the nulls rise.
- Bundle A removal reads −1pp on both sets.
- ★ The conservative table reads slightly BETTER per changed move on both sets, but games tested only fit A. A direct
  fitA-vs-cons game test is worth running before the ship choice.

**Fit B data generation** (09-27 ~03:00): `fitB_variant_d6`, shipped vs shipped at d6 on `openings_variant.txt`,
4,000 games, conc 2, seed 50. Extract later with `_texel_extract.py TAGS=fitB_variant_d6` (both configs carry
V2_PRESET=shipped).

**ODDS vs SF18@400 nodes, shipped** (`oddsSF_shipped`, 480 games, seed 42; `diagnostics/_odds_vs_sf_report.py`
rebuilds each game's start from the schedule, because vs_sf writes no per-opening summary):
- Overall 68.1%: **converting 92.7%, defending 43.5%**. Queen down: still holds 30.6%. Knight down: 43.3%.
- ⚠️ SF18 at 400 nodes is too weak to reproduce the owner's scenario (we score ~66% against it in normal games). The
  fit-A-vs-shipped comparison on the same seed is still valid (`oddsSF_fitA` running).
- ▶️ To reproduce "SF beats our lightning a knight down": SF18 at ~3,000 nodes (≈ the UI's 5 ms), knight + pawn odds
  only.

**ODDS vs SF18@400, fit A vs shipped, PAIRED on identical starts and colours (seed 42, 480 games each):**
- **all +7.3pp ± 3.7 (z 3.90)**: shipped 68.1% → fit A 75.4%.
- **Defending a deficit: +12.3pp (z 3.77)**, 43.5% → 55.8%.
- Converting an edge: +2.3pp (z 1.29), 92.7% → 95.0%; near the ceiling.
- Largest per-family moves (n ≈ 30 each): rook down, holds 42% → 72%; pawn down 41% → 55%; queen down 31% → 42%;
  exchange converting 80% → 94%.
- ⇒ Against an outside engine, fit A's material-handling gain is mostly in HOLDING worse positions.
- ▶️ Owner-scenario run: SF18 at 3,000 nodes (≈ the UI's 5 ms), knight + pawn odds only
  (`selfplay/openings_odds_np.txt`, 120 starts), both arms, same seed.

**OWNER'S SCENARIO REPRODUCED — knight + pawn odds vs SF18 @ 3,000 nodes** (≈ the UI's 5 ms), ours at 250k nodes,
240 games per arm, seed 43, paired:

| | shipped | fit A |
|---|---|---|
| a knight UP, converting | 72.5% | **80.8%** |
| a pawn up | 29.2% | 32.5% |
| defending (SF has the edge) | 4.2% | 8.3% |
| overall | 27.5% | 32.5% |

- **Paired: +5.0pp ± 4.8 (z 2.06).**
- ⇒ SF at this setting still scores ~1 in 5 a knight down against fit A. The eval gain narrows the gap but does not
  close it; the rest is search / speed, the next roadmap block.
- The Fit B data is ready: `fitB_variant_d6` (4,000 games) → `E:/chess_data/texel/v2_variant_stage1.csv.gz`,
  **238,978 positions** (W 39.7 / D 28.4 / L 31.9). That is ~6% of the standard set, so Fit B needs weighting or more
  variant games to reach the ~20% share.

**fit A vs fit A cons, head to head** (1,000 games, NODE_LIMIT=50000, seed 44): `+387 -373 =240 (50.7%) ≈ +5 ± 18`
⇒ LEVEL. The two tables are equivalent in games; regret mildly favours cons.

## 11. MORNING SUMMARY (09-27) — nothing shipped overnight; owner decisions below
1. **Texel fit A is the largest eval gain on record.**
   - Self-play: **+111 ± 13** (2,356 games, replicated on fresh openings).
   - Against SF18: **+38 [+10, +67]** (1,000 paired games, 2 seeds).
   - Variant starts: **+18 [+5, +31]** (2,000 games).
   - Odds engine-vs-engine: **+43 [+22, +65]**.
   - Odds vs SF18@400: **+7.3pp (z 3.9)**. Knight/pawn odds vs SF18@3k: **+5.0pp (z 2.1)**.
   - Accuracy vs SF18 search: **−8.9% own-play, −14.3% diverse** (now ahead of SF15.1 classical; ~half-way to SF11).
   - d7 regret: **+2pp over nulls on BOTH sets**. STS: +85 at d10, +73 at equal nodes. WAC 249 (inside noise).
   - Symmetry: colour 0/800 and file mirror 0/651.
2. **Choice of table:** fit A (λ₂ 3e-12) vs cons (1e-10) are level in games (+5 ± 18). Cons has ~1/3 the magnitude,
   more stable cells (282 vs 220) and slightly better regret; fit A is the one with the full validation battery.
3. **Owner decisions:**
   - (a) ship fit A or cons (with PST_V2_TAPERED=1 into the V2_PRESET block);
   - (b) re-sweep RFP_MARGIN after the ship: the eval scale changed ~10-20× for PSTs;
   - (c) Fit B design: the variant data is ~6% of the standard set (weight ×3, or more variant games);
   - (d) apply Texel to the next tables (mobility, KS) before the search block, or go to search now.
4. **Closed:** Fit A2 phase limits (flat). **Fixed:** regret ladder arms inheriting BASE_KNOBS · vs_sf FEN starts ·
   accuracy tool schema.
