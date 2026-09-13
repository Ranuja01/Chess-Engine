# Eval v2 — CURRENT CONFIGURATION AND DECISION REGISTER

@author: Ranuja Pinnaduwage (maintained with Claude)

★ **One page, one question: what is v2 right now, and what did we decide about every piece of it.**
Update at the END OF EVERY RUNG. The design docs carry reasoning, `EVAL-V2-REBUILD-LOG.md` carries history;
this carries only the standing state, so a disagreement about "what are we going with" is settled here.

Last updated: **2026-09-13**, after rung 2 passed games.

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

### ⏳ NOT YET BUILT — the remaining ladder
See `EVAL-V2-REBUILD-LOG.md` for the slice plan. Components: mobility + per-piece placement · central + space ·
threats · Kaufman + pairs · rook files · **tempo (absent from BOTH v1 and v2)** · **draw classifier (9 cases,
live in v1)** · mate drive · convertibility scale · corrhist · winnability · capgains · OvD.
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
