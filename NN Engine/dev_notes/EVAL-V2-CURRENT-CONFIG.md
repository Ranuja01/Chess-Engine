# Eval v2 — CURRENT CONFIGURATION AND DECISION REGISTER

@author: Ranuja Pinnaduwage (maintained with Claude)

★ **One page, one question: what is v2 right now, and what did we decide about every piece of it.**
Update at the END OF EVERY RUNG. The design docs carry reasoning, `EVAL-V2-REBUILD-LOG.md` carries history;
this carries only the standing state, so a disagreement about "what are we going with" is settled here.

Last updated: **2026-09-13**, after rung 2 passed games and slice 1's tempo component was built and parked.

---

## 1. THE SHIPPED CONFIGURATION

```
EVAL_ARM=1
  # rung 1 -- king safety (KS-A)
  KS_V2_ZONE_SF=1  KS_V2_XRAY=1  KS_V2_COORD=256
  KS_V2_WEAK=57    KS_V2_ADJ=61  KS_V2_NO_QUEEN=321
  KS_V2_CHK_Q=126  KS_V2_CHK_R=122  KS_V2_CHK_B=80  KS_V2_CHK_N=152
  KS_V2_MAX=4000   KS_V2_HALF=600   KS_V2_ONSET=450
  # rung 2 -- pawns
  PS_V2_MAG=100          # structure: doubled + isolated + backward
  PASSER_V2_MAG=60       # passers, endgame leg only
```
Everything else is at its default, and the defaults encode the decisions below.

| measured | value |
|---|---|
| rung 1 vs rung 0 | **+101 Elo** (four venues, 1,882 games pooled) |
| rung 2 vs rung 1 | **+60.4 ±25.5 Elo** (987 games, SPRT H1 accepted) |
| STS | rung 0 1364 · rung 1 1480 · **rung 2 1698** · v1 1796 |
| arm 0 control | **250 / 35,310,778 / EBF 3.784** — byte-identical, re-verified 09-13 |

---

## 2. DECISION REGISTER — every component, and why

### ✅ SHIPPED
| component | form | note |
|---|---|---|
| material | flat `values[]`, pawn 1000 both phases | rung 0 |
| placement | `whitePlacementLayer` PSTs | rung 0. ★ prices FILE/CENTRE, not advancement (flat ~25mp from rank 3) |
| **king safety** | SF-shaped zone + x-ray, Hill curve `MAX·u²/(u²+HALF²)`, onset subtracted | rung 1. No phase gate, deliberately |
| **pawn structure** | doubled (86/263) · isolated (file table, **mg leg 0**) · backward (rank table, **mg leg 0**) | rung 2a |
| **passers** | `PassedRank[]` **unconditional + additive**, rank-gated at the 4th, king distance enemy:ours ≈ 2.4:1, candidate ×50% | rung 2b. ☠️ **no `R`, no multiplier, no clamp chain** |
| phase | continuous, `MG_LIMIT=61700 / EG_LIMIT=15800` | ★ LOCKED to SF's shape scaled to our pieces (92.3%/23.7% of starting npm) |

### ☠️ PARKED — built, switched off, with a named re-test trigger
| component | knob | why parked | re-test when |
|---|---|---|---|
| **connected / support** | `PS_V2_CONN_MAG=0` | harmful at EVERY magnitude and BOTH shapes; support alone was ~71% of the damage. v1's `pawn_chain_file_bonus` IS the support term, so it double-prices what the PST and structure already carry | inside 2b's connected×passed test, or once central exists |
| passer midgame leg | `PASSER_V2_MG_PCT=0` | halves the worst case at equal mean. Our converted table is mg-heavy by construction (SF's eg pawn is dearer, ours is flat) | if material ever tapers |
| **material taper** | `EVAL_V2_PAWN_MG=1000`, `EVAL_V2_PIECE_MG_PCT=100` | ⚠️ **UNDECIDED, not refuted** — §I says −6.59% (130× its floor, a large win); WAC and both STS forms read **inside their noise floors**. Two implementations (pawn-side and piece-side) trend the same, so the unit-of-account hypothesis is refuted and the ratio itself is the open question | a checkpoint bundle, or a 4-night run |
| weak-unopposed | absent | 4.27× lift to `passed` — fires mostly on pawns 2b rewards | after passers are tuned |
| weak-lever | absent | 3.59% firing, pure endgame | later |
| pawn wall (as a separate term) | merged | it IS phalanx; now an input to connected, not its own score | n/a |
| **draw classifier — lone-pawn cases** | `DRAW_V2_KPK=0` | ☠️ v1's chebyshev-opposition test **ignores whose move it is** — measured **6.2% FALSE POSITIVES** on KPvK (flags forced WINS as dead draws). Adding the missing tempo term cuts it to **0.6%** over 320 samples / 4 seeds — 10× better and still not zero, so it fails the asymmetric gate. ⚠️ The June claim *"validated against a full KPvK retrograde oracle"* does NOT hold for the rule as shipped | when an **exact KPK bitbase** exists (zero FP by construction; SF ships one at `stockfish_11/src/bitbase.cpp`, ~24 KB, built at init; we own the retrograde tooling in `_kpk_oracle.py`). Then fold into `DRAW_V2_CLASS` |
| **tempo** | `TEMPO_V2_MG=0`, `TEMPO_V2_EG=0` | ☠️ **PARKED ON MEASUREMENT 2026-09-13, FINAL.** Built, and its correctness proved by an EXACT identity (below). ⚠️ **Correction to an earlier entry: the response is NOT "monotone downward" — the full ladder is NON-MONOTONIC and unordered.** STS off 1698 → **25/14: 1656 (−42, inert)** · 50/28: 1511 (−187) · 100/55: 1504 (−194) · 200/110: 1631 (−67) · 800/440: 1312. The worst points sit in the MIDDLE. A term with real evaluative content gives a smooth curve with one peak; this is threshold crossing. ☠️ Also: **200 mp was the wrong magnitude** — v2's ENTIRE midgame positional spread is 5-35 mp (rim-vs-central knight = 30), so the pawn-denominated conversion made tempo ~8× the largest positional term we own ([[convert-reference-constants-by-positional-scale-not-by-the-pawn]]). Correcting that makes it **inert, not good**. Corroborated by the node screen: tempo has **zero positional variance**, so 100% of its **+10.82%** node cost at ref is margin/parity interaction by construction, and the response is NON-monotonic (+10.8% at 1x, +2.8% at 4x) — the step-shaped signature of a constant crossing fitted absolute thresholds (`RFP_MARGIN` 1500/ply, `DELTA_MARGIN` 1500, `OTV_MARGIN` 1750). Two instruments agree ⇒ **spend no games** | at the checkpoint margin re-sweep — margin coupling is its ONLY demonstrated channel, so that is the one condition under which the measurement could change. ⚠️ **NOT at every rung**: the rung-gradient test (1364→1336 / 1480→1378 / 1698→1631, i.e. −28/−102/−67) is **non-monotonic and entirely inside the ±150 floor**, so "tempo is waiting for a richer eval" is UNSUPPORTED. Worth one re-test after slice 2 lands **mobility** — the biggest missing quiet-move responder — and not before |

### ☠️ DELIBERATELY NOT PORTED
| v1 component | why |
|---|---|
| `PV_BOOST_MAG=10000` | removing it improved v1 on **6/6 corpora** (−6.13% mean) |
| heat map / attacking layer | a 5-in-1 no giant has; each job gets a dedicated subsystem |
| `PAWN_CLAMP_MID/EG` | measured to be MASKING HARM (chain 14×, struct 23×, both signs inverted) |
| `STRUCT_OPPOSED_*_PCT` | post-hoc multiplier; both references put `opposed` INSIDE the connected formula |
| `pawn_chain_file_bonus` (file-keyed) | confirmed central double-count, harmful 6/6 |
| `passer_realizability_R` + its 4-stage clamp chain | the multiplicative form: 4/4 references are additive, and v1's own comment records a passer scoring LESS than a non-passer |
| `PASSER_ENEMY_CREDIT_PCT` | ☠️ re-enabling is an **UN-FIX** — its zeroing shipped +38.7 Elo |
| `pawns_simd_initializer`, `get_latent_threat_score`, legacy `!ENABLE_*` arms | dead code |
| `pawn_majority`, `latent_threat` | dead at defaults / measured 0 of 5,000 |

### 🔨 BUILT, GATED OFF, AWAITING THE OWNER'S CALL
| component | knob | state |
|---|---|---|
| **binary draw classifier** | `DRAW_V2_CLASS=false` | ★ **CURRENT (end of 2026-09-13): members = KvK / KBvK / KNvK / KBvKB / KNvKN + KBvKN / KNNvK / wrong-coloured-bishop rook pawn in SF's FORTRESS form. ALL FOUR GATES PASS** — arm 0 byte-identical `250 / 35,310,778 / 3.784` · rule fires on every new case with both negative controls untouched · tempo identity exactly 0.000 with the rule ON · **STS 1698** unchanged — after fixing a breakdown-contract bug (it published nothing when firing, which crashed `eval_symmetry.py` with `KeyError: 'total'`; it now publishes `total = 0` with only `EB_TOTAL`). Tablebase evidence: every false positive in 1,592 minor-piece positions was a **mate in 1**, and search was shown to play one with the rule scoring 0. ⏳ **Awaits the owner's sign-off on the DTM-weighted gate** before it is turned on. Lone-pawn cases stay OFF in `DRAW_V2_KPK` pending an exact KPK bitbase. Earlier history of this row: ✅ **Built and gated 2026-09-13.** Carries ONLY the cases that measured **0 false positives in 382 samples across 5 seeds**: KvK / KBvK / KNvK / KBvKB / KNvKN. ⚠️ **Qualified 2026-09-13: that sample was UNIFORM, and KBvKB / KNvKN contain rare boxed-king forced mates that uniform placement essentially never generates — so "0 in 382" is true and is NOT proof of clean.** Re-checked under the corner-biased sampler with a DTM-weighted gate (`EVAL-V2-SLICE1-DRAW-DESIGN.md` §2d); every reference draws these cases anyway, and search finds any mate inside its horizon. ☠️ Five of v1's ten cases flag forced WINS (`RB_vs_R` 28% · `R_vs_minor` 24-28% · `RN_vs_R` 22% · `wrongB_rookpawn` 10%) and are deliberately excluded — they are a MAGNITUDE and belong in the convertibility scale. Gates all pass: arm 0 byte-identical `250 / 35,310,778 / EBF 3.784` · knob provably executes (KBvKB asym 5→0, KNvKN asym 10→0, every won position untouched) · **STS 1698 unchanged** · tempo identity gate still exactly 0.000. ⭐ Per the owner's June gate this ships on **oracle proof + no bench regression**, not a games SPRT — it fires on ~0% of midgame positions by construction |

### ⏳ NOT YET BUILT — the remaining ladder
Slice plan in §5 below. Components: mobility + per-piece placement · central + space ·
threats · Kaufman + pairs · rook files · **draw classifier (9 cases, live in v1)** · mate drive ·
convertibility scale · corrhist · winnability · capgains · OvD.
⚠️ **Convertibility scale — know its real history before building it (found 2026-09-13):** v1's
`endgame_convertibility_scale` shipped default-on in `046a17f` and was **reverted the same day in `ab070b7`**
("−3.3 STS / −3 WAC at d10"); `STRENGTH_BACKLOG.md` wrongly said "shipped" for three months. That revert is an
**unresolved null** (WAC −3 inside ±5–6 and non-discriminating; STS inside ±150), so it neither vindicates nor
condemns v2's version — which differs in FORM anyway (EG leg only inside the blend, not v1's whole total behind
a boolean). ★ Keep its one real lesson: the scale is BROAD, so measure v2's on its own.
⚠️ **tempo is NO LONGER on this list** — built 2026-09-13 and parked on measurement (see the PARKED table
and `EVAL-V2-SLICE1-TEMPO-DESIGN.md`). Slice 1 therefore loses a component rather than being reordered.
☠️ **Contempt: absent everywhere, and unmeasurable in self-play** — it needs a different opponent.

---

## 3. VERIFICATION IS PROPORTIONATE (owner, 2026-09-13)

★ **Ground-truth CLASSIFICATIONS can be verified by proof/oracle; MAGNITUDE judgements need measurement.**

| kind of change | how it earns its place |
|---|---|
| categorical / ground truth (draw detection, insufficient material, the square rule) | ⭐ **oracle proof + firing rate + cost.** If it is provably sound, rare, and free, there is no case to answer — no games attribution required. ⚠️ The guarantee is ASYMMETRIC: prove **no false positives** (a won position flagged drawn is catastrophic; a missed draw only forfeits an opportunity). Precedent: v1's KPK rule is already "validated against a full KPvK retrograde oracle" |
| magnitude / ordering (every positional term) | §I + STS attribution per component, then games per slice |
| speed only (pawn hash, caching) | byte-identity cached vs uncached; ⚠️ `half-a-ply-is-elo-neutral` says do not expect Elo |

---

## 4. THE UNIT SPLIT
★ **The RUNG is the design and attribution unit. The SLICE is only the games unit.**
Every component gets its own four scans, design, knob, gates and §I/STS attribution regardless of which
slice it is games-tested in. Rung 2 ran exactly this way: 2a and 2b were attributed separately (+99 and
+119 STS) and only the games were bundled.
⚠️ **Known limit:** §I attribution does NOT reliably predict which member carries the Elo — it read KS at
−1.44% where games said +101. If a slice fails, the per-component §I numbers are weaker evidence than they
look; the mitigation is a leave-one-out games run on the most suspect member, paid only when needed.
⚠️ This limit exists in EITHER ordering. It is a cost of bundling, not of any particular sequence.

---

## 5. THE SLICE PLAN -- games budget, not design order

### ★★ CURRENT: REVISED PLAN (agreed with the owner, end of 2026-09-13) -- supersedes the original table below
⚠️ **Naming:** RUNGS 0-2 (material+PST, king safety +101, pawns +60.4) are DONE. On 09-13 the *remaining* ladder was
regrouped into SLICES, numbered from 1 again -- so "slice 1" is NOT "rung 1".

| slice | contents | games |
|---|---|---|
| **1** | ✅ **draw classifier** (`DRAW_V2_CLASS`, built + gated, awaits gate sign-off) · ☠️ tempo PARKED | none -- classifications ship on oracle proof + no bench regression |
| **2** ▶️ **NEXT** | **mobility + per-piece placement** (outposts, bishop colour complex, long diagonal, minor-behind-pawn, trapped rook, queen weak) **+ rook files** (they ARE per-piece placement) | **ALONE** -- likely the last term big enough to read solo |
| **3** | central + space + threats **+ Kaufman/pairs** | bundle, tested for REGRESSION |
| **4** ★ NEW | **ENDGAME CONVERSION**: exact KPK bitbase · corner-drive value for KR vs minor (SF tier 2b) · endgame-leg scale incl. SF's generic pawnless rule and Ethereal's lone-minor rule · mate drive · fifty-move plumbing · **+ winnability** | bundle, tested for REGRESSION |
| **5** | ☠️ **corrhist + capture gains + OvD** -- LAST BY NECESSITY (residual correctors) | bundle |

**Why it moved:**
- Tempo parked; the draw classifier ships through the oracle, not games; the leftover endgame work turned out to be one
  coherent unit. Mate drive + convertibility scale moved from slice 1 to slice 4; Kaufman to 3; rook files to 2.
- ★ **Winnability moved from the correctors (old slice 4) into endgame conversion.** It was MIS-CATEGORISED as a residual
  corrector: it is STRUCTURAL (passers, pawn flanks, infiltration, npm), not learned from search residual, and it is the
  same family as scale factors -- SF11 applies `score += initiative(score)` immediately before `scale_factor(eg_value(score))`.
  Designing it WITH the endgame scale is the point: both reshape the eg leg and share detectors (OCB, pawn count), so the
  overlap check has to see them together or they double-count.
- Endgame conversion goes AFTER the middlegame channels: it fires almost only in endgames, so order does not bias either
  measurement, while mobility/central/space get harder to read alone as the eval grows.
- The KPK bitbase can FLOAT -- oracle-gated like the classifier, build it in any idle gap.
- Slice 5 is not "too weak" -- strength is not an ordering criterion (bundles are tested for regression), and capture
  gains alone is the entire v1-v2 variant gap (+176.89%).

**Tempo triggers (parked, NOT permanent):** (1) after slice 2 mobility lands -- an STS ladder at POSITIONALLY-scaled
magnitudes ~25/50/100 mp (mobility is the biggest missing quiet-move responder, the untested half of the owner's richness
hypothesis); (2) at the checkpoint margin re-sweep. Low prior: SF itself deleted tempo.

---

### (original 2026-09-13 table, SUPERSEDED by the block above -- kept as history)

★ **The RUNG is the design/attribution unit; the SLICE is only the GAMES unit.** Every component below
still gets its own four scans, design doc, knob, gates and §I/STS attribution. Only games are bundled.

**Why bundling at all:** measured venue power -- one night (~1,200 games at ~116 games/hr, conc 4)
resolves **~+20 Elo and nothing smaller**; +10 needs ~4 nights; +5 needs ~16. And the ladder is shrinking:
KS **+101** -> pawns **+60.4**.

| slice | components | rationale |
|---|---|---|
| **1** | ~~tempo~~ - draw classifier - mate drive - Kaufman + pairs - rook files | order-INVARIANT terms that cannot cancel (a categorical classifier overlaps with nothing). Thickens the eval before the big measurements. ☠️ **tempo REMOVED 2026-09-13, parked on measurement** — and the rationale that put it here was partly wrong: a side-to-move constant is order-invariant WITHIN a node but is **not margin-invariant**, and with zero positional variance its entire measurable effect was margin interaction. ★ **Generalise: "cannot cancel with the other components" is NOT the same as "has no confound of its own."** Check each remaining slice-1 member against the absolute margins too, not just against each other |
| **2** | mobility + per-piece placement (outposts, bishop colour complex, bishop long diagonal, minor-behind-pawn, trapped rook, queen weak) | largest remaining missing channel -- test ALONE while that is still possible |
| **3** | central + space + threats | shared attack maps, coherent unit |
| **4** | ☠️ corrhist - winnability - capgains - OvD | **LAST BY NECESSITY** -- see below |

### ★ GAMES POLICY FOR SMALL TERMS (agreed 2026-09-13, with the owner's refinement)

**The measured asymmetry that drives it:** our instruments detect HARM far better than small GAIN. Tempo at
50 and 100 mp read **−187 / −194** STS, comfortably outside the ±150 floor; nothing beneficially sized has
ever resolved on its own.

1. **Bundle small, reference-backed terms and test the bundle for REGRESSION, not for gain.** A bundle
   reading ≥ −10 Elo passes and ships. One night with a real decision, instead of ~16 nights chasing +5.
2. ★ **Owner's refinement: when a term every reference carries fails, rethink OUR IMPLEMENTATION first** —
   its scale, wiring, or detector — before concluding against the concept. Leave-one-out *locates* the
   failure; it is not the verdict. Live precedent: tempo's magnitude was pawn-converted, which made it ~8×
   v2's entire positional spread ([[convert-reference-constants-by-positional-scale-not-by-the-pawn]]).
3. **The bundle is a games-budget decision; the per-component design doc + this register is the
   attribution of record.** ★ v1 did not degenerate because it bundled — it degenerated because it bundled
   **with no record** of what each of ~30 terms was for, so nobody could tell which to cut.
4. ☠️ **Not for novel or ours-alone MAGNITUDES.** Those test alone or do not ship.
5. **Checkpoint ablation is the audit**, asking "can we delete this?" — far more answerable than "did it help?"

⚠️ **Cost accepted knowingly:** we will ship terms whose individual Elo we never learn.

**The four tracks:** correctness-gated classifications → **oracle proof + no bench regression, no games**
(owner's June rule) · **mobility tests ALONE** in slice 2, probably the last term big enough to read solo ·
**everything smaller bundles** under rule 1 · **rethink-then-leave-one-out only on failure.**

### ★ CLASSIFICATIONS ARE GOVERNED BY ORACLES, NOT BY CONSENSUS (agreed 2026-09-13)
The adoption rule ([[adopt-reference-methods-only-if-universally-superior]]) exists for MAGNITUDES, where we
cannot verify correctness and consensus is the best evidence available. For a ground-truth CLASSIFICATION,
"4 of 5 engines do X" is only a proxy — **a tablebase IS correctness.** ⇒ An ours-alone classifier that
passes the oracle belongs in v2 with no reference support needed.
⚠️ But ours-alone classifiers carry the **highest prior risk**: on 2026-09-13 every mechanism that failed the
oracle was a v1 race test. The oracle is the bar, with no reasoning our way past it.
☠️ An ours-alone MAGNITUDE (e.g. the convertibility scale's passer pull-back) gets **no** oracle exemption.

### ☠️ WHY THE RESIDUAL-CORRECTORS MUST BE LAST (the owner proposed running them first; this is why not)
Owner's argument: low-value slices get harder to pass as the eval grows, so run them while they are a
larger fraction. **Correct for REDUNDANCY-limited terms, and it backfires for RESIDUAL-CORRECTORS.**
`corrhist` corrects the residual between static eval and what search found. On a four-term eval that
residual is enormous, so corrhist would look EXCELLENT -- and that value evaporates as real terms land.
Same for winnability and capgains.
★ It is the same law pointed the other way: *"a feature measured where it is redundant looks worthless"*
<=> *"a residual-corrector measured where the residual is huge looks essential."*
=> Reversing the order does not escape the measurement bias, it INVERTS it.
✅ **What survives from the owner's proposal: the SMALL ORDER-INVARIANT terms (slice 1) genuinely can go
early**, and doing so means the big rungs are measured against a fuller eval, closer to what ships.

⚠️ **The bundling-attribution cost exists in EITHER ordering** -- it is a cost of bundling, not of a
sequence. §I does not reliably predict which member carries the Elo (it read KS at -1.44%; games said
+101). Mitigation when a slice fails: a leave-one-out games run on the most suspect member.

---

## 6. ☠️ DRAW DETECTION ALREADY HAS HISTORY -- CONSULT IT BEFORE DESIGNING

⚠️ **Found 2026-09-13 by auditing `HANDOFF.md`, AFTER I had discussed draw detection as new work.**
A record-check would have found it first. See memory [[endgame-draw-detection]].

| already done (v1) | |
|---|---|
| `is_practically_drawn` | **9 cases, LIVE and unconditional** in the endgame branch; returns 0 outright |
| KPvK rook-pawn case | SHIPPED 2026-06-27, from a real external loss (engine traded INTO a dead draw reading **+4870**) |
| R+N-vs-R, KRKN, KRKB | SHIPPED 2026-06-08; the `+1N` endgame over-read went **461 -> 142** |
| 🧰 `diagnostics/_kpk_oracle.py` | **83,238 states, 0 false-draws.** The oracle tooling EXISTS |
| ⭐ the standing GATE (owner, June) | *"a self-play tournament is uninformative for a self-play-invisible fix -> ship on verifiable position-fix + no bench regression"* |

★ ★ **That gate IS today's "verification is proportionate" rule -- the owner set it three months ago.**
=> For v2's draw classifier: re-derive (do NOT port -- v1's reads globals and returns via control flow),
verify each case against the oracle for **NO FALSE POSITIVES**, measure firing rate and NPS, and ship on
that. No games attribution required or expected.

### ⚠️ A NAMED UNFINISHED FOLLOW-UP, still open
From the 2026-06-08 entry: *"a **graded drawishness scale** (oppo-bishops / R-vs-2-minors /
R+N-vs-R-WITH-pawn -- the cases the binary detector cannot express), which would also subsume the
mate-drive knob."*
=> That is **`endgame_convertibility_scale`**, which was BUILT (3 concepts: <=1 minor with no pawns /
opposite-coloured bishops / a winning passer pulling back toward 1) and left at `ENABLE_ENDGAME_SCALE=false`.
★ It belongs in slice 1 alongside the classifier, and the June note says it may SUBSUME the mate drive --
so build them together and test whether the mate-drive knob is still needed.
