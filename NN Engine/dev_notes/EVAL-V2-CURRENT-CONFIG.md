# Eval v2 — CURRENT CONFIGURATION AND DECISION REGISTER

@author: Ranuja Pinnaduwage (maintained with Claude)

★ **One page, one question: what is v2 right now, and what did we decide about every piece of it.**
Update at the END OF EVERY RUNG. The design docs carry reasoning, `EVAL-V2-REBUILD-LOG.md` carries history;
this carries only the standing state, so a disagreement about "what are we going with" is settled here.

Last updated: **2026-09-19**. Since the 09-13 revision: placement bundle E games-confirmed (≈ +13) · mobility AREA
shipped (≈ +31) · the collinearity gate completed to 40 columns across all five subsystems · the margin re-sweep run
under `EVAL_ARM=1` for the first time and **`RFP_MARGIN=1000` shipped** (−17.0% nodes at identical solves, free) ·
the **first v1-vs-v2 showdown ever played** (1500 games, **elo −9.5 ±20.7 ⇒ indistinguishable**) · v2 placed on the
**reference-accuracy ladder** (245.46 → **155.72**, closing 59.7% of the gap to SF11's 95.26) · **slice 3 CLOSED with
Kaufman parked, five concepts for five parks**. ⚠️ Fingerprint rows below are ordered newest-first; the CURRENT one
is the `RFP_MARGIN=1000` row.

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
  # slice 1 -- draw classifier (enabled 2026-09-13 on the owner’s conditional sign-off)
  DRAW_V2_CLASS=1        # hard 0 only where the stronger side lacks mating material; revert = remove this line
  # slice 2 -- mobility (passed games 2026-09-14: +205 -66 =48 / 319 pooled, ~ +162 Elo; owner sign-off)
  MOB_V2_MAG=600         # knight mg table range in mp, SF11 shape; revert = remove this line
  DRAW_V2_KPK_EXACT=1    # exact KPK bitbase (165,676 states, 0 false draws / 0 false wins); owner sign-off 2026-09-14; revert = remove this line
  # slice 2 -- placement bundle E (regression SPRT 1,200 games pooled: +499 -463 =238, ~ +10 Elo, LLR ~ +1.9, inconclusive/positive; owner sign-off 2026-09-15)
  OUTPOST_V2_PCT=100     # SF11 outpost form
  BADB_V2_PCT=100  BADB_V2_FORM=1    # SF15.1 bad bishop (won the form ladder 6/6)
  TRAPROOK_V2_PCT=10     # SF11 step form, file-symmetrised
  WEAKQ_V2_PCT=25        # inert on §I; keep-or-drop at the checkpoint
  BEHIND_V2_PCT=25  BEHIND_V2_FORM=1 # Weiss minor-behind-pawn form; revert bundle = remove these five lines
  # slice 2 residue -- MOBILITY AREA refinements, shipped 2026-09-17 on the owner's sign-off (≈ +31 Elo, 1,178 games,
  # two seeds: H1 accepted at the ≥10 bound, H0 at the ≥50 bound ⇒ formally bracketed 10 < true < 50).
  # ☠️ Both were parked as §I-ONLY nulls when measured ALONE; they became measurable only as a BUNDLE (owner's idea).
  # ★ Neither adds a concept: they refine the AREA of mobility, the largest term v2 owns (≈ +162 Elo).
  MOB_V2_PIN=1           # SF's pin-line restriction: king-blocker squares leave the area; our pinned pieces count only
                         # along their pin line (mobility only -- KS attack maps unchanged)
  MOB_V2_EXCL_LOWRANK=1  # own pawns still on ranks 2-3 leave the area (SF11/SF15.1; Weiss rank-2 only)
  # ── SEARCH margin, re-swept FOR v2 and shipped 2026-09-18 on the owner's sign-off ────────────────
  # ☠️ THIS IS A SEARCH KNOB, NOT AN EVAL KNOB, AND IT IS GLOBAL. It lives HERE (the v2 env block) and must
  # NEVER become a `search_engine.h` default: the default is shared with arm 0, so changing it would alter the
  # FROZEN v1 control and invalidate its `250 / 35,310,778 / EBF 3.784` fingerprint. v1 keeps 1500.
  # Evidence: -17.0% WAC nodes and -17% quiet-node median at IDENTICAL 250 solves; regression SPRT
  # (elo0=-10/elo1=0) H1 ACCEPTED at +591 -531 =353 / 1475, LLR +3.079 ⇒ the node saving is FREE.
  # ⚠️ Do NOT quote that run's +14.1 point estimate as the magnitude -- it stopped ON the upper bound, where the
  # estimate is biased upward. What is established is the BOUND: it does not cost ~10 Elo.
  # Plateau checked: 1250 -> 253 solves / -10.2% nodes · 800 -> 248 solves / -19.8% ⇒ usable plateau 1000-1250,
  # and 1000 is its node-cheapest point at no solve cost. Revert = remove this line.
  RFP_MARGIN=1000
```
Everything else is at its default, and the defaults encode the decisions below.

### ⚠️ KNOBS ADDED 2026-09-20/21 — ALL DEFAULT OFF, ALL VERIFIED BYTE-IDENTICAL AT THEIR DEFAULT
| knob | default | state |
|---|---|---|
| `EVAL_V2_PAIR` | **0** | ★ The `(mg,eg)` accumulator. 0 = each term blends at its own site (the shipped path). 1 = legs accumulate and interpolate ONCE. ☠️ **Mode 1 CANNOT be byte-identical** (~16 truncations/position); the BOUND is the test — measured max **3.00 mp** vs a 16 mp bound, 0 violations / 4,000 positions, symmetry 0/800. Mode 1 fingerprint `246 / 50,631,624 / EBF 4.031`. ⚠️ **A real behaviour change — do not flip the default without its own measurement.** Prerequisite for the endgame scale factor, a tapered PST, KS phase legs, and the eg-leg winnability/OvD direction |
| `PS_V2_WEAKUNOPP_MG` / `_EG` | **0 / 0** | SF `WeakUnopposed`, the only 4/4-unanimous term v2 lacked. Built inside `pawn_structure_mp` (its firing set is 100% contained in `isolated\|backward`, so a second owner would be two names for one signal). **Measured MOVE-NULL** — 49.3 / 49.3 / 50.1 vs a 49.8 neutral, no dose ordering. Trigger to revisit: the joint retune |
| `ROOKFILE_V2_OPEN` / `_SEMI` | **0 / 0** | ☠️ **REJECTED 09-20**, now on measurement rather than an unread STS row: every magnitude to SF11's pawn conversion (367/164) at or below its null, sign replicated on a disjoint set, never positive |
| `PST_V2_KING_EG_ONLY` | **0** | ☠️ **A REAL DEFECT, AND NEGLIGIBLE.** The sixth placement table is commented "Kings - Endgame" and is CENTRALISING; v1 reads it only in `evaluate_kings_endgame`, v2's census reads all six tables with no phase gate ⇒ **v2 pays king centralisation in the opening/midgame where v1 pays none** (wrong-signed, v2-only). 1 = the king's placement contributes eg-only. ✅ byte-identical at 0, symmetry 0/800. 📏 Fire rate 18.5% but **median 3 mp, max 18 mp, mean 0.7 mp over all positions** — two orders of magnitude under the regret floor, because both kings usually sit on similarly-valued cells so the DIFFERENCE is tiny. ▶️ **Fold into the retune; do not ship standalone** — the right end state is a shelter-shaped midgame king table (A3 proper), not this 5-line subset. ★ Needs no `EVAL_V2_PAIR`: one term can carry its own phase at its own site |
| `KS_V2_EG_PCT` | **100** | ☠️☠️ **REJECTED 2026-09-21.** KS-A's endgame leg (K2/A8), chosen because SUBTRACTIVE is the only shape that has ever won in KS (additive 0-for-11). ✅ byte-identical at 100, symmetry 0/800. 📏 median 65 mp, max 1123 — properly resolvable. ☠️ d7 regret: `70` **+0.0949 worse**, `40`/`20` inside the historical neutral band ⇒ move-NULL; **WAC 241/300 (−9 solves, the session's hardest veto)**; **nodes +6.4%**; STS **1796 vs 1854**. ★★ **THE DIAGNOSIS IS THE VALUE: `phase256` is the WRONG CONDITIONER for king safety.** Our phase is material-based, so "endgame" conflates quiet K+P endings with sparse MATING attacks — the −9 solves are lost mates. ⇒ any revisit must condition on **attacker presence** (queens/rooks), not phase; v1 already carries that shape as `KS_EG_MAT_GATE`/`_HI`/`_FLOOR` — record-check before rebuilding. ⚠️ Only ONE neutral arm was run (the tool wants two), so the move channel is an unresolved null, not a harm; the rejection rests on the WAC veto and the node cost |
| `PS_V2_REAR_DOUBLED` | **0** | ✅ **A DEFECT FIX THAT PASSES ITS HARM CHECK — the one item of the 09-21 bundle worth shipping, on CORRECTNESS.** v2 flagged `passed` on a clear ENEMY span alone, never asking whether one of OUR OWN pawns is ahead on the file ⇒ two stacked own pawns both flagged and **both paid in full** ([eval_v2.cpp](../eval_v2.cpp) `if (!st){ pb \|= m; continue; }`). Only the front pawn can promote. 1 = demote rear to candidate · 2 = Ethereal, no credit. ✅ byte-identical at 0, symmetry 0/800. 📏 Fire rate **3.4%**, median 78 mp, max 386. ✅ Harm check (mode 2): **WAC 252/300 (+2)**, nodes **−0.6%**, STS 1829 (−25, inside floor). ☠️ **Deliberately NOT benefit-measured**: at 3.4% fire the move-change rate is ~1-2%, below the d7 gate's resolution BY CONSTRUCTION ⇒ this is a correctness decision, not a measurement one, and it is the owner's call. ⚠️ Mode 1 built but untested |
| `PASSER_V2_PATH_PCT` | **0** | ☠️☠️ **REJECTED ON MEASUREMENT 2026-09-21 — 7 arms, 3 channels, null-to-negative in all.** Keep at 0. d7 regret vs a neutral arm (−0.1522): weight-increasing arms +0.037 / −0.071 / +0.031; **mass-compensated arms +0.217 / +0.053 / +0.175 WORSE** ⇒ at constant passer mass the ladder is worse than flat, so the shape carries no information and the first ladder's mild win% gain was the extra weight. WAC `246 / 50,795,598 / 4.032` (−4 solves, **+2.7% nodes**); STS **1765 vs 1854**. ☠️ **This closes P3 (rook-behind-passer) too** — both halves of it live inside this ladder in SF and shipped in these arms. ⚠️ Untested alternative: our `unsafe` uses the shared `KS_V2_XRAY` map, strictly larger than SF's plain `attackedBy`, so every `k` may sit one rung low. Full record in the gap audit. **The code stays in, gated at 0, because the build is verified and the next question (`KS_V2_XRAY=0`) reuses it.** ★★ **The PASSER PATH-SAFETY LADDER (gap audit P1+P2), built 2026-09-21.** SF11 `evaluate.cpp:626-635`: `k = 35/20/9/0` by how far the enemy's attacks reach up the forward span, `+5` for a defended stop square or our own R/Q behind, all × the `w = 5r−13` v2 already shipped WITHOUT its ladder. ☠️ **This is also the only home for rook-behind-passer (P3)** — SF spends R/Q-behind twice inside it (an ENEMY R/Q behind keeps the span maximally unsafe; our OWN is the +5), so a standalone P3 bonus would be a third invented form. ✅ Gates: byte-identical at 0 (`250 / 49,440,513 / EBF 4.031`), colour symmetry **0/800** at 100. 📏 Fire rate on the play distribution (4,000 pos): **25.3%**, median **93 mp**, p90 **662 mp**, max **2,069 mp**, **signed mean −4.9 mp** ⇒ it reshapes without shifting the mean, the profile the twelve nulls lacked. ⚠️ Carries its own magnitude because `PASSER_V2_MAG=60` was tuned against a ladder-LESS passer (the ladder more than triples a rank-7 passer, faithfully to SF's own 3.6× ratio) ⇒ expect a `MAG` re-sweep. ⚠️ Needs the shared attack build; inert with KS/mobility/space/threats all off, and the toggles dump says so. ⚠️ Uses the shared x-ray map (`KS_V2_XRAY`), so `unsafe` is strictly larger than SF's plain `attackedBy` — one rung more pessimistic by construction |
| `TEMPO_V2_MG` / `_EG` | **0 / 0** | Recommend PERMANENT closure — on the re-swept margin its node effect FLIPPED SIGN (+1.67/+2.50% vs September's −5.06/−5.10%) ⇒ threshold coupling, not evaluation |

| measured | value |
|---|---|
| rung 1 vs rung 0 | **+101 Elo** (four venues, 1,882 games pooled) |
| rung 2 vs rung 1 | **+60.4 ±25.5 Elo** (987 games, SPRT H1 accepted) |
| **slice 2 mobility vs rung 2 + draw** | **≈ +162 Elo** (319 games pooled over 2 segments, seg 2 +164.0 ±45.7, SPRT H1 accepted, 2026-09-14) |
| **slice 2 placement bundle E vs mobility ship** | **≈ +13 Elo — GAMES-CONFIRMED 2026-09-16** (segment 3 alone ACCEPTED H1: +611 −548 =296 / 1,455, +15.1 ±21.0, LLR +3.039; pooled over 3 segments +1,110 −1,011 =534 / 2,655). Earlier reading, superseded: ≈ +10 Elo (±~23) — regression SPRT elo0 −10 / elo1 0, 1,200 games pooled over 2 segments (seg 1 killed by a Windows Update restart), LLR ≈ +1.9, INCONCLUSIVE with a positive lean, never negative; regret clean (+0.1 / −1.5pp); §I −1.5%+ better 6/6; collinearity VIF ≤ 1.30. Shipped on owner sign-off 2026-09-15 |
| ☠️ **SHIPPED v2 FINGERPRINT FROM 2026-09-15 (placement E in)** | **SUPERSEDED — PRE-AREA config: WAC d10 `250 / 61,352,373 / EBF 4.114`** (measured 2026-09-15 on the unchanged `.so`) — use THIS for byte-identity; the mobility+KPK row below is now the PRE-PLACEMENT config |
| ☠️ **SHIPPED v2 FINGERPRINT FROM 2026-09-14 (mobility in)** | **SUPERSEDED — PRE-PLACEMENT config: WAC d10 `250 / 60,036,572 / EBF 4.043`** (+39 nodes vs mobility-only, same solves — the bitbase fires in a few lines). Use THIS for byte-identity. Mobility-only, pre-KPK, for the record: **WAC d10 `250 / 60,036,533 / EBF 4.043` · STS 1588.** The old `246 / 63,221,361` / 1698 is the PRE-MOBILITY config — a byte-identity check against it will now fail on a correct build |
| **slice 2 residue: mobility area (pin + exlow) vs SHIP+E** | **≈ +31 Elo** — 1,178 games over two seeds: seed 17 H1 ACCEPTED at elo0 0/elo1 10 (+310 −222 =159 of 691, LLR +2.980), seed 18 H0 accepted at elo0 30/elo1 50 (+202 −186 =99 of 487) ⇒ **formally bracketed 10 < true < 50**; pooled +512 −408 =258 (54.4%). Shipped 2026-09-17 on owner sign-off. ⚠️ Quote the POOLED tally, never a single run's elo — the same pairing read +60.7 / +44.5 / +11.4 across three sound SPRTs ([[sprt-point-estimates-inflate-at-the-bound-they-stop-on]]) |
| ☠️ **SHIPPED v2 FINGERPRINT FROM 2026-09-18 (`RFP_MARGIN=1000` in)** | ☠️ **CURRENT: WAC d10 `250 / 49,440,513 / EBF 4.031`** — use THIS for byte-identity, and ONLY with `RFP_MARGIN=1000` present in the env. ★ Same 250 solves as the previous shipped config in **10.1M FEWER nodes (−17.0%)**, reproduced byte-identically across two independent runs (the 09-17 sweep cell and the 09-18 ship check) ⇒ deterministic. ⚠️ **This is a SEARCH-knob change, so the eval is byte-identical to the row below** — running v2 WITHOUT `RFP_MARGIN=1000` correctly returns `250 / 59,549,832 / EBF 4.080`, and that is NOT a failed build. v1 is untouched at `250 / 35,310,778 / EBF 3.784` |
| ☠️ **SHIPPED v2 FINGERPRINT FROM 2026-09-17 (mobility area in)** | **PRE-MARGIN config: WAC d10 `250 / 59,549,832 / EBF 4.080`** — the eval-only fingerprint, still the byte-identity reference for any EVAL change. ★ The SAME 250 solves as the previous shipped config in **1.8M FEWER nodes (−2.9%)**, which is the [[a-truer-eval-buys-pruning-headroom-the-crank-result]] pattern and independent corroboration from a different instrument class. The 09-16 row `250 / 61,352,373 / 4.114` is now the PRE-AREA config |
| STS | rung 0 1364 · rung 1 1480 · **rung 2 1698** · v1 1796 |
| arm 0 control | **250 / 35,310,778 / EBF 3.784** — byte-identical, re-verified 09-13 |
| v2 WAC d10 (shipped config) | **246 / 63,221,361 / EBF 4.087** (draw classifier ON; OFF = 246 / 63,216,318 — +0.008% nodes, same solves) |
| §I accuracy vs v1 (09-13) | rung 2 **better on 4/6 corpora** (−15.0 / −13.9 / −3.9 / KS −9.8%), worse on UHO +12.5% and **variant +138.7%**; draw classifier identical on all six. Full table: `EVAL-V2-VS-V1-RECOUNT-2026-09-13.md` |

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
| **draw classifier** | binary, 8 cases; hard 0 only where the stronger side lacks mating material | slice 1, `DRAW_V2_CLASS=1`. Oracle-gated, no games (owner's June rule) |
| **exact KPK bitbase** | SF11 retrograde, built once through a thread-safe static, no runtime tables | slice 1, `DRAW_V2_KPK_EXACT=1`. 165,676 states, 0 false draws / 0 false wins |
| **mobility** | SF11 per-piece concave tables, area = ~(enemy pawn attacks \| own blocked pawns \| own king), `MOB_V2_MAG=600` | slice 2. **≈ +162 Elo** (319 games pooled) — the largest single gain of the rebuild. ☠️ STS read it −110 at this very setting; §I and the regret gate carried it |
| **placement bundle E** | outpost SF11 @100 · bad bishop **SF15.1** @100 · trapped rook @10 (file-symmetrised) · weak queen @25 · minor-behind-pawn **Weiss** @25 | slice 2. **≈ +13 Elo**, 2,655 games over 3 segments (seg 3 H1 accepted). ⚠️ The first 1,200 games were INCONCLUSIVE — a max-games stop, not a null |
| **mobility area** | `MOB_V2_PIN=1` (SF pin-line restriction, mobility only) · `MOB_V2_EXCL_LOWRANK=1` (own rank-2/3 pawns leave the area) | slice 2 residue, shipped 09-17. **≈ +31 Elo**, 1,178 games / 2 seeds, bracketed 10 < true < 50. ☠️ Both were §I-only NULLS alone — they became measurable only as a BUNDLE (owner's proposal). ★ Neither adds a concept; both refine mobility's AREA |

### ☠️ PARKED — built, switched off, with a named re-test trigger
| component | knob | why parked | re-test when |
|---|---|---|---|
| **threats** (slice 3) | `THREAT_V2_PCT=0` + 5 leg flags + `THREAT_V2_GATE` | ★ **Best-verified term of the slice** (oracle 0 mismatches over BOTH defence gates × all 7 legs × both occupancies, fire 38.6→82.3→94.8% · symmetry 0/800 · collinearity 27 terms no flags). §I liked it strongly (`th100` **−9.71%** on the variant corpus) but EVERY magnitude taxes `lichess_ks_labelled` in proportion (+0.35 → +3.90) ⇒ nothing clears both columns; regret **NULL on both corpora** (primary 50.1 vs bar 50.1 · `_v2` 49.7 vs 50.5) on a 35-36% footprint. ☠️ The "it double-counts KS" story was REFUTED twice (general 10k sample AND `lichess_ks_labelled` itself). ★ The SHAPE mismatch is real and explains ~half the tax: SF/Ethereal KS is an unbounded quadratic that overtakes threats ~2:1 in severe attacks, while ours saturates at `KS_V2_MAX`=4.0p and threats at th100 reaches 4.2p. `th100+ksmax8000` cut the tax +3.90 → +2.13 — but the ceiling-alone CONTROL is itself +0.54, and acting would reopen a **+101 Elo** rung on accuracy evidence alone | (a) a **lazy-eval** lane (it is the slice's most expensive term, so a strength-neutral term's COST is the live question) · (b) if `KS_V2_MAX` is ever re-opened for its own reasons · (c) the unbuilt SF legs (`Knight/SliderOnQueen`, `WeakQueenProtection`) |
| **space** (slice 3) | `SPACE_V2_MAG=0` + REGION/SAFE/WEIGHT/BEHIND/GATE_PCT | Built + verified (oracle 0 mismatches on BOTH form families incl. every axis flipped; symmetry 0/800). SF's quadratic-weight form is HARMFUL at every magnitude; Ethereal's linear form **at scale parity** is INERT (all arms inside the ±0.05 floor). ★ Only real effect is class-local on `centre_locked` (−0.04..−0.13) WITH a mechanism — locked centres are where mobility collapses, so safe-square room carries what mobility cannot. ⚠️ That class is **3.2% of positions and cannot be thickened from our pools** (+13,500 positions bought 322 class rows and 37 changed moves); its regret read was +1.5pp on 467 changed moves = unresolvable | **a purpose-built closed-centre corpus** (King's Indian / French / Closed Sicilian structures, mined or generated) — NOT another pass over these pools |
| **bishop pair** (slice 3) | `BPAIR_V2_MAG=0`, `BPAIR_V2_FORM` | 5/5 references pay it, and it is **ALREADY OWNED** by v2's PST + mobility on three agreeing instruments: §I every magnitude helps general corpora and hurts KS-critical (nothing clears) · per-class §I **flat in all 5 structure classes** (−0.05..−0.20, no hidden regime) · regret NULL both corpora (−0.4 / −0.1pp). ☠️ "5/5 references have it" argues for a CANDIDATE, never for a second owner | when **Kaufman** is built (it should OWN the pair, as SF's matrix does), or if PST/mobility are ever re-scaled |
| mobility FORM alternatives | `MOB_V2_TABLE=0`, `MOB_V2_EG_PCT=100`, `MOB_V2_SAFE=0`, `MOB_V2_EXCL_QUEEN=0` | The form bake-off concluded **NO CHANGE**: SF11's table shape beat SF15.1 (+0.35 mean) / Weiss (+1.86) / Ethereal (+3.15) — their tables assume their own PSTs. Endgame share and magnitude 800/1000 buy general accuracy and cost KS-critical. OURS-FIRST `MOB_V2_SAFE` (v1's lower-value-attacker test) fails the worst-case rule. ☠️ `MOB_V2_EXCL_QUEEN`'s single 09-14 §I point REVERSED on re-measurement | only if the §I/worst trade changes — e.g. after a KS re-tune, since the worst column IS the KS-critical corpus |
| **connected / support** | `PS_V2_CONN_MAG=0` | harmful at EVERY magnitude and BOTH shapes; support alone was ~71% of the damage. v1's `pawn_chain_file_bonus` IS the support term, so it double-prices what the PST and structure already carry | inside 2b's connected×passed test, or once central exists |
| passer midgame leg | `PASSER_V2_MG_PCT=0` | halves the worst case at equal mean. Our converted table is mg-heavy by construction (SF's eg pawn is dearer, ours is flat) | if material ever tapers |
| **material taper** | `EVAL_V2_PAWN_MG=1000`, `EVAL_V2_PIECE_MG_PCT=100` | ⚠️ **UNDECIDED, not refuted** — §I says −6.59% (130× its floor, a large win); WAC and both STS forms read **inside their noise floors**. Two implementations (pawn-side and piece-side) trend the same, so the unit-of-account hypothesis is refuted and the ratio itself is the open question | a checkpoint bundle, or a 4-night run |
| weak-unopposed | absent | 4.27× lift to `passed` — fires mostly on pawns 2b rewards | after passers are tuned |
| weak-lever | absent | 3.59% firing, pure endgame | later |
| pawn wall (as a separate term) | merged | it IS phalanx; now an input to connected, not its own score | n/a |
| **draw classifier — lone-pawn cases** ⭐ **TRIGGER MET 2026-09-14: exact KPK bitbase BUILT (`DRAW_V2_KPK_EXACT`, default off) — `_kpk_oracle.py --all-files --engine` 165,676 states, 0 false draws / 0 false wins; eval-level verify ALL AS EXPECTED both arms; arm 0 + v2 byte-identical. WAC d10 rule ON 246 / 63,221,296 (−65 nodes, same solves — it fires). ✅ SHIPPED in §1 as `DRAW_V2_KPK_EXACT=1` on the owner's sign-off 2026-09-14 (classifications ship on oracle proof; not deferred to the endgame slice). ☠️ The old `_kpk_oracle.py` was broken (move-counter keying, 7.7% wins) — the June "validated" claim below is UNVERIFIED.** | `DRAW_V2_KPK=0` | ☠️ v1's chebyshev-opposition test **ignores whose move it is** — measured **6.2% FALSE POSITIVES** on KPvK (flags forced WINS as dead draws). Adding the missing tempo term cuts it to **0.6%** over 320 samples / 4 seeds — 10× better and still not zero, so it fails the asymmetric gate. ⚠️ The June claim *"validated against a full KPvK retrograde oracle"* does NOT hold for the rule as shipped | when an **exact KPK bitbase** exists (zero FP by construction; SF ships one at `stockfish_11/src/bitbase.cpp`, ~24 KB, built at init; we own the retrograde tooling in `_kpk_oracle.py`). Then fold into `DRAW_V2_CLASS` |
| **tempo** ⏳ **RE-TEST TRIGGER FIRED 09-14 (mobility passed games): STS 50/28 on the mobility-600 base = 1695 vs 1588 (+107), where the same tempo read −187 without mobility — a sign flip, inside the floor; second point 25/14 = 1704 (+116, was −42 without mobility) ⇒ ladder complete with 100/55 = 1709 (+121): three consistent positive points (were −42/−187/−194 without mobility) but FLAT across 4× magnitude ⇒ direction credible; ☠️ WAC node screen: 25/14 −5.06% and 100/55 −5.10% — IDENTICAL across 4× ⇒ tempo is a BINARY switch on search thresholds, not an eval signal ⇒ stays PARKED, folds into the margin re-sweep checkpoint; no tempo games** | `TEMPO_V2_MG=0`, `TEMPO_V2_EG=0` | ☠️ **PARKED ON MEASUREMENT 2026-09-13, FINAL.** Built, and its correctness proved by an EXACT identity (below). ⚠️ **Correction to an earlier entry: the response is NOT "monotone downward" — the full ladder is NON-MONOTONIC and unordered.** STS off 1698 → **25/14: 1656 (−42, inert)** · 50/28: 1511 (−187) · 100/55: 1504 (−194) · 200/110: 1631 (−67) · 800/440: 1312. The worst points sit in the MIDDLE. A term with real evaluative content gives a smooth curve with one peak; this is threshold crossing. ☠️ Also: **200 mp was the wrong magnitude** — v2's ENTIRE midgame positional spread is 5-35 mp (rim-vs-central knight = 30), so the pawn-denominated conversion made tempo ~8× the largest positional term we own ([[convert-reference-constants-by-positional-scale-not-by-the-pawn]]). Correcting that makes it **inert, not good**. Corroborated by the node screen: tempo has **zero positional variance**, so 100% of its **+10.82%** node cost at ref is margin/parity interaction by construction, and the response is NON-monotonic (+10.8% at 1x, +2.8% at 4x) — the step-shaped signature of a constant crossing fitted absolute thresholds. ☠️ **CORRECTED 2026-09-17 — the threshold list here was wrong for TWO OF THREE.** It read "`RFP_MARGIN` 1500/ply, `DELTA_MARGIN` 1500, `OTV_MARGIN` 1750". Verified in source: **`DELTA_MARGIN` is DEAD at defaults** (`search_engine.cpp:7673`/`:7685` gated on `!ENABLE_QDELTA_PERMOVE`, which ships TRUE; `:7717` uses it only as a fallback when `QDELTA_PERMOVE_MARGIN == 0`, and that ships 1500) — and the `[toggles]` dump PRINTS it at `:2512` while never printing the live knob, which is exactly how it entered the list. **`OTV_MARGIN` is behind `ENABLE_OTV = false`.** The step-shape READING stands; only its channel list changes. Live eval-denominated thresholds: `RFP_MARGIN` · `FUTILITY_MARGIN_SCALE` · `QDELTA_PERMOVE_MARGIN` · `ASPIRATION_DELTA` · `VERIFY_MARGIN` (★ root razoring is NOT one — it keys on PRE-SEARCH SCORES, not the static eval, despite being in mp). ⇒ **The checkpoint re-test must run against THAT list.** Two instruments agree ⇒ **spend no games** | at the checkpoint margin re-sweep — margin coupling is its ONLY demonstrated channel, so that is the one condition under which the measurement could change. ⚠️ **NOT at every rung**: the rung-gradient test (1364→1336 / 1480→1378 / 1698→1631, i.e. −28/−102/−67) is **non-monotonic and entirely inside the ±150 floor**, so "tempo is waiting for a richer eval" is UNSUPPORTED. Worth one re-test after slice 2 lands **mobility** — the biggest missing quiet-move responder — and not before |

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
| **binary draw classifier** | `DRAW_V2_CLASS=1` in the shipped config (code default false) | ✅ **SHIPPED IN THE v2 CONFIG 2026-09-13** on the owner’s conditional sign-off (cleanly revertible + the giants’ practice + tests show no harm: STS 1698 unchanged, WAC solves identical at +0.008% nodes, §I identical on all six corpora). ★ **CURRENT (end of 2026-09-13): members = KvK / KBvK / KNvK / KBvKB / KNvKN + KBvKN / KNNvK / wrong-coloured-bishop rook pawn in SF's FORTRESS form. ALL FOUR GATES PASS** — arm 0 byte-identical `250 / 35,310,778 / 3.784` · rule fires on every new case with both negative controls untouched · tempo identity exactly 0.000 with the rule ON · **STS 1698** unchanged — after fixing a breakdown-contract bug (it published nothing when firing, which crashed `eval_symmetry.py` with `KeyError: 'total'`; it now publishes `total = 0` with only `EB_TOTAL`). Tablebase evidence: every false positive in 1,592 minor-piece positions was a **mate in 1**, and search was shown to play one with the rule scoring 0. ⏳ **Awaits the owner's sign-off on the DTM-weighted gate** before it is turned on. Lone-pawn cases stay OFF in `DRAW_V2_KPK` pending an exact KPK bitbase. Earlier history of this row: ✅ **Built and gated 2026-09-13.** Carries ONLY the cases that measured **0 false positives in 382 samples across 5 seeds**: KvK / KBvK / KNvK / KBvKB / KNvKN. ⚠️ **Qualified 2026-09-13: that sample was UNIFORM, and KBvKB / KNvKN contain rare boxed-king forced mates that uniform placement essentially never generates — so "0 in 382" is true and is NOT proof of clean.** Re-checked under the corner-biased sampler with a DTM-weighted gate (`EVAL-V2-SLICE1-DRAW-DESIGN.md` §2d); every reference draws these cases anyway, and search finds any mate inside its horizon. ☠️ Five of v1's ten cases flag forced WINS (`RB_vs_R` 28% · `R_vs_minor` 24-28% · `RN_vs_R` 22% · `wrongB_rookpawn` 10%) and are deliberately excluded — they are a MAGNITUDE and belong in the convertibility scale. Gates all pass: arm 0 byte-identical `250 / 35,310,778 / EBF 3.784` · knob provably executes (KBvKB asym 5→0, KNvKN asym 10→0, every won position untouched) · **STS 1698 unchanged** · tempo identity gate still exactly 0.000. ⭐ Per the owner's June gate this ships on **oracle proof + no bench regression**, not a games SPRT — it fires on ~0% of midgame positions by construction |

### ⏳ NOT YET BUILT — the remaining ladder
Slice plan in §5 below. ★ **UPDATED 2026-09-17 — slices 1-3 are essentially done.** Remaining: **Kaufman + pairs** (the last
named slice-3 item; it should OWN the bishop pair) · rook files (off; §I liked them 6/6, STS monotone harmful ⇒ needs a
move-level read) · mate drive · convertibility scale · corrhist · winnability · capgains · OvD.
☠️ **No longer on this list — built and dispositioned:** mobility + per-piece placement (SHIPPED) · draw classifier + exact
KPK (SHIPPED) · mobility area (SHIPPED) · **central (NOT BUILT — 0/5 references have a standalone central term; they price
centrality once via PST + mobility, and our error is SMALLEST in contested-centre positions)** · space (built, PARKED) ·
threats (built, PARKED) · bishop pair (built, PARKED — already owned).
⚠️ **Read [[v2-positional-signal-is-nearer-saturation-than-its-term-count]] before adding the next concept:** slice 3 added
FOUR concepts the giants carry and none produced move-level information on top of v2's existing terms, while the only +Elo
refined an existing term's AREA. That is a hypothesis from a pattern of four, not a law — but it changes the prior on
"add another term" versus "sharpen a term we own".
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
| **2** ▶️ **IN PROGRESS (09-14)**: mobility core + rook files BUILT, all gates pass, default OFF (`MOB_V2_MAG`, `ROOKFILE_V2_OPEN/SEMI`); ladder + crank 09-14: STS flat inside floor at every magnitude (cannot resolve); §I optimum ~1000 (−7.56%); ★ d7 regret @600 vs same-session neutral **+4.9pp primary / +3.1pp `_v2`, REPLICATED** ⇒ two instruments agree ⇒ **games candidate `sprt_ab` shipped vs +MOB_V2_MAG=600, awaiting OWNER decision**; rook files stay OFF (STS monotone harmful); ★★★ **MOBILITY 600 PASSED GAMES: +205 −66 =48 / 319 pooled, ≈ +162 Elo (seg 2 +164.0 ±45.7, H1 accepted) — ✅ SHIPPED in §1 (owner sign-off 2026-09-14)**; placement sub-terms (outpost, reachable outpost, minor behind pawn, bad bishop, long diagonal, trapped rook, weak queen) **BUILT + GATED, default 0** — oracle 0 mismatches both x-ray paths, byte-identity held, file-mirror clean after symmetrising SF's trapped-rook side test; §I ladder queued behind the games; placement sub-terms not built. `EVAL-V2-SLICE2-MOBILITY-DESIGN.md` | **mobility + per-piece placement** (outposts, bishop colour complex, long diagonal, minor-behind-pawn, trapped rook, queen weak) **+ rook files** (they ARE per-piece placement) | **ALONE** -- likely the last term big enough to read solo |
| **3** | central + space + threats **+ Kaufman/pairs** | bundle, tested for REGRESSION |
| **4** ★ NEW | **ENDGAME CONVERSION**: ~~exact KPK bitbase~~ (✅ SHIPPED in slice 1 on 09-14 — `DRAW_V2_KPK_EXACT`, 165,676 states verified) · corner-drive value for KR vs minor (SF tier 2b) · endgame-leg scale incl. SF's generic pawnless rule and Ethereal's lone-minor rule · mate drive · fifty-move plumbing · **+ winnability** | bundle, tested for REGRESSION |
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

★★★ **COLLINEARITY GATE + ONE-OWNER RULE (owner charter restated 2026-09-14: v2 exists to clean up v1 — keep what is good, learn from or improve on the giants, implement each subsystem cleaner and more efficiently, and NEVER recreate v1's collinearity).** ⚠️ §I additivity (placement bundle 97% additive) is NOT non-collinearity: terms can add on accuracy and still re-express one signal — v1's "~30 terms / ~2 signals". Standing gate for every slice from slice 2 on: (1) per-position term contributions across the corpora (from the existing probes) → correlation matrix + VIF; flag any pair |r| ≳ 0.7 or a term largely explained by the others; (2) **ONE OWNER PER CONCEPT**: a flagged pair keeps the concept in the subsystem that measures better and REDEFINES the other to be disjoint — never both "because the giants do" (SF pays several concepts twice by design); (3) in form ladders, ties on §I + regret go to the form least correlated with existing terms. Slice-2 suspects to check first: trapped rook vs mobility's negative floor · bad bishop vs bishop mobility · outpost vs knight mobility. ✅ **FIRST RUN 09-14 (`_v2_term_collinearity.py`, 10,000 positions): PASS** — every placement term VIF 1.00-1.30; largest cross-correlations bad bishop × bishop mobility +0.38, reachable outpost × knight mobility +0.31, trapped rook × rook-mobility table −0.25 ⇒ the trapped-rook double-pay concern is NOT supported by data. ★★ **EXTENDED 2026-09-17 — KING SAFETY IS NOW COVERED (the hole open since slice 2).** Probes added: `space_counts`, `threats_counts` (7 legs) and **`ks_counts`** (6 channels per king: attacker count · weighted attacker sum · weak zone squares · king-adjacent attacks · safe checks · scored units). The gate now runs **27 terms**, and takes `SETS=` so it can be pointed at ONE corpus. Results: **NO cross-subsystem flag** on a general 10,000-position sample AND on `lichess_ks_labelled` itself (5,000) — every threats leg VIF ≤ 1.30, `th_king` **1.10**, largest threats×KS correlation `th_restricted × ks_natt` **+0.32**. ⇒ **Threats does NOT re-express king safety, so reopening the KS rung has no evidence behind it.** ★ Overlap is a property of a POPULATION — that is why the KS-critical corpus was re-run separately rather than trusting the general sample to generalise. ☠️ Intra-subsystem pairs are EXEMPT from flagging (`MOB`, and `KS_SET` added after the first KS-aware run flagged `ks_natt × ks_watt r=+0.93` — an attacker count against its own weighted sum over the identical piece set, i.e. ONE detector reported twice): **a gate that flags its own redundant columns trains you to ignore it.**
☠️☠️ **AND THE LIMIT THAT COST US MOST THIS SLICE: VIF/`r` measure co-movement of detector COUNTS, not the relative HEIGHT of the scored curves.** A clean gate means two terms do not measure the same thing — it does NOT mean they coexist well at their chosen magnitudes. Threats' detectors are provably disjoint from KS's, yet threats taxed KS-critical accuracy at every magnitude because our KS **saturates** at `KS_V2_MAX` = 4.0 pawns while SF's and Ethereal's kingDanger is an **unbounded quadratic** that overtakes threats ~2:1 in severe attacks (ours: ratio 0 → 0.85, never > 1). ⇒ **Curve BALANCE needs a source comparison or a 2×2, never the gate.** (2×2 measured: `th100+ksmax8000` cut the tax +3.90 → +2.13, but the ceiling-alone control is itself +0.54 and acting would reopen a +101 Elo rung on accuracy evidence alone ⇒ not done.)
★★★ **PAWN STRUCTURE COVERED 2026-09-17 — the gate is now COMPLETE across all five scoring subsystems (40 columns), and clean on both populations.** ☠️ It required **no probe and no rebuild**: `pawn_entry_probe` / `pawn_masks` have exported every Layer A mask since 09-12 for the rung-2 oracle, so only the column set was missing. Added 12 predicate popcounts + **`ps_npawns`** (raw pawn-count difference) as a control. **Results: NO cross-subsystem flag** — general 10,000 positions, strongest cross pair `mob_table_mg × ks_natt` **−0.41**; `lichess_ks_labelled` 5,000, strongest `th_restricted × ks_natt` **+0.33**. ⇒ **Pawn structure does not re-express mobility, placement, threats or king safety, and vice versa.**
★★ **AND THE PREDICTION IT REFUTED — the third and strongest instance of "sharing an INPUT is not sharing a SIGNAL".** Two pairs were registered in the tool beforehand as the ones we *knew* were wired to a shared map: `ps_pattacks × mob_*` (mobility's area SUBTRACTS enemy pawn attacks) and `ps_halfopen × traprook_units` (`trap_rook_units` literally TAKES `halfOpen` as a parameter, `eval_v2.cpp:1630`). Measured: **−0.01 to +0.07** and **−0.07**. ⇒ One term consuming another's output as an INPUT predicts essentially NOTHING about count co-movement. Three instances now (KS↔threats VIF 1.10 · mobility area ↔ pawn attacks 0.01 · trapped rook ↔ halfOpen 0.07). ☠️ Corollary, and it cuts BOTH ways: a shared-input story is not evidence of double-counting — and a clean gate is still not evidence of safe coexistence (see the curve-height limit above).
⚠️ Still NOT covered: **`PawnEntry.attacks2`** (double pawn attacks, built at `eval_v2.cpp:657`, not among the probe's 27 exported slots — so "does threats' `stronglyProtected` re-express the double-attack map?" is unmeasurable without a probe change + rebuild) · **`blocked`** (identically 0 under the White−Black convention — needs a different reduction) · sibling-move-difference overlap.
★★ **v2-vs-v1 SHOWDOWN FAIRNESS RULE (owner, 2026-09-14): v2 must not be outshone by search parameters tuned to v1.** Every eval-denominated search threshold (RFP / futility / delta / razor / OTV margins, aspiration window) is absolute millipawns fitted to v1's spread. Before any v2-vs-v1 games: (1) measure NPS for both arms in a quiet window (`wac_speed`) — eval cost, on its own; (2) re-sweep those margins for v2; (3) play the showdown TWICE — v2 on v1's margins and v2 on its re-swept margins — so the difference quantifies the tuning handicap instead of hiding it. ⚠️ v2's 70% extra WAC nodes at d10 is SEARCH SHAPE (pruning/ordering), not NPS; v2's NPS is unmeasured this phase. A dedicated v2 search (search + movegen + caching, alongside the UCI pure-C++ reorg) is the program AFTER the eval slices, unless the re-swept showdown shows v2 is still badly node-bound.
★★★ **STEP (2) MEASURED 2026-09-17 — the first margin sweep ever run under `EVAL_ARM=1`, and the answer DE-RISKS the showdown.** Instruments: quiet-node median (`depth_nps_bench --n 60 MAX_DEPTH=10 LONG_FORMAT`, v1 baseline reproduced EXACTLY at 249,014) for cost + WAC solves as the tactical VETO; both fixed depth ⇒ deterministic ⇒ valid while the machine is in use. ☠️ NOT STS (±150 floor swallowed 25/25 cells of the 09-05 sigma×RFP sweep) and NOT WAC nodes alone (reversed sign in 4 of 6 search configs — both corpora were run and agree in sign here).
· **`RFP_MARGIN` is THE lever**, and the only one: v2 recovers **−17.0% nodes at IDENTICAL solves (250) at RFP=1000**; the break is between 800 (248) and 1000; 400 costs 9 solves. · `FUTILITY_MARGIN_SCALE=70` adds a free **−2.0%** (⚠️ monotone for v2, unlike v1 where it closed twice as non-monotonic). · `QDELTA_PERMOVE_MARGIN` is a **NON-LEVER** (≤3.2%, non-monotonic, both directions cost solves) — its first sweep under either eval. · `ASPIRATION_DELTA` and `VERIFY_MARGIN` remain **OPEN** for v2.
☠️ **The free recovery totals ~19% IF the two compose (2×2 required, never assumed) — and that is well below the ~35% bar at which node savings become Elo-visible.** ⇒ **The handicap is real, now measured, and SMALL: a few Elo, not tens.** Still play the showdown TWICE as the rule requires, but expect a NARROW spread and do not hold the showdown hostage to this lane. ★ Corollary: **"v2 searches ~90% more nodes" is MARGIN-CONDITIONAL, not a property of v2** — the v2/v1 node ratio slides 1.96 (RFP 400) → 1.51 (RFP 6000), so any quoted node/EBF gap is partly a statement about v1's margin choice. ★ Mechanism confirmed: v2's ordering is BETTER (first-move cutoff share 90.2% vs v1's 86.5%) and its excess is almost all in nodes that cut off IMMEDIATELY (m0 +74% vs m8+ +1.6%) — i.e. under-pruning, and tightening RFP removes exactly those. ⚠️ **What fixed depth CANNOT do is pick the winner**: a margin change is a node-saver ⇒ judged at FIXED TIME. The open decision is a timed SPRT of v2 @1500 vs @1000 (with @1250 as the alternative), which needs a quiet window.
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

★★★ **SLICE 5 DESIGN PRINCIPLE (owner, 2026-09-19) — capgains and OvD are OUR INVENTIONS, not reference terms, and
must clear a HIGHER bar than anything in slices 1-4.** Owner's framing: *"those are things the giants don't have and
are therefore not tried and tested. That doesn't mean they may not have a place, but we want to be really careful on
how and when and why to use them... we need to really consider the noise aspects, collinearity, real need, latency."*

**CAPTURE GAINS — original purpose: a horizon-effect guard reaching beyond qsearch.** ⚠️ Owner's point: with a proper
qsearch plus corrhist, that purpose may already be served. Evidence now on file, all pointing the same way:
- **0/4 references implement it** (SF11, SF15.1, Ethereal, Weiss).
- ☠️ It is the **root of v1's ENTIRE colour-symmetry residual** (`find_and_pop_last_viable_capture` selects by
  position in a SQUARE-ORDERED stack, so mirroring maps max→min). v2's clean 0/4000 is partly just its absence.
- ★ **Its absence costs nothing measurable**: v2 is LEVEL with v1 in 1500 games while lacking v1's largest single
  signal (v1 sibling spread ~2,290 mp vs v2's 36).
- ⚠️ It is EXPENSIVE (simulates capture sequences) in an eval with NO lazy exit.
- ☠️ **CORRECTION to my own repeated claim:** I called capgains "the biggest identified accuracy chunk", citing the
  **+138.69% variant-corpus column**. That citation does NOT support the claim — the record already WITHDREW "the
  variant column is the target" as substantially an INSTRUMENT ARTIFACT (`SCALE_CAPTURE_GAINS=0` alone reads
  +176.89% on that column). ⇒ **There is currently NO trustworthy measurement showing v2 needs capture gains.**
⇒ Its slice-5 entry condition is a demonstrated HORIZON failure v2 actually suffers, not a port of v1's term.

**OvD — original purpose: long-term pressure.** ⚠️ Owner's diagnosis: *"in v1 it sort of became a KS hybrid which
likely led to the noise we saw."* Mechanism is concrete — v1's `ovd_imbalance` consumes the same
offensive/defensive accumulators king safety uses (cf. `MOD_KS_CONTROL`, "the attacker's board-control edge... the
imbalance-term signal"). ⇒ ★ **This is exactly what the 40-column collinearity gate exists to catch, and it must run
BEFORE any OvD magnitude ladder, not after** — an OvD channel against the six `ks_*` channels. The one-owner rule was
written for this failure. If OvD re-expresses KS, it is REDEFINED to be disjoint or it does not ship.

**⇒ SLICE-5 GATE, stricter than slices 1-4 (no reference consensus to lean on):** (1) state the mechanism it guards
and show v2 ACTUALLY suffers that failure; (2) collinearity gate FIRST, against KS and mobility especially;
(3) cost/latency measured, since v2 has no lazy exit; (4) symmetry gate on the term specifically at N>=4000, with any
selection rule COLOUR-BLIND BY CONSTRUCTION; (5) games, because §I cannot see corrhist at all and has proven a poor
prioritiser throughout. ⚠️ And corrhist is instrumentally SEPARATE from the other two — search-side, games-only.

★★ **REGISTER ADDITIONS 2026-09-19** (appended; the tables above predate these)

| component | state | note |
|---|---|---|
| **Kaufman / polynomial material imbalance** | ☠️ **PARKED 09-18.** Built, gated off (`KAUF_V2_MAG=0`) | SF11's census quadratic, tables verbatim from source. THREE parameterisations all fail: SF's cells monotonically harmful over MAG 250-2000 (best +1.00 mean / +2.03 worst); a **zero-free-parameter DERIVED per-piece value-ratio rescale** also harmful (+2.05 / +3.19) ⇒ **the BASIS hypothesis is REFUTED**; only v1's FITTED cells improve the mean (−1.91) and only on their own self-play distribution (−8.65% `game_regret_set` but +2.38% UHO, +2.45% variant) ⇒ overfitting. Sign and scale both ruled out first. ⇒ **No principled parameterisation helps v2.** `KAUF_V2_FORM=1` carries v1's cells as a ☠️ DIAGNOSTIC only. ★ Pair ownership replicated twice: inside the term beats outside |
| **tier-2b technique value** (`TIER2_V2_MAG`) | ☠️ **PARKED 09-19 — MOVE-NULL.** Built, gated off (`TIER2_V2_MAG=0`). A purpose-built 200-position tablebase WIN-PRESERVATION suite, biased **33-to-9 in the term's favour**, found the base arm at **33/33** on the exact decision class the term exists to fix ⇒ **zero headroom**. Flat across MAG 0-300 (191-193/200 at d6), McNemar d10 p=0.453 / 1.000, and it costs **+8-9% nodes**. ⚠️ The BISHOP half is 100/100 for every arm at every depth — effective n is 100. ⚠️ **Do NOT run games**: that suite carries ground truth on every move and has more power per position than self-play at 0.06% class frequency. ★ The ACCURACY case SURVIVES and matters to the NNUE teacher (our eval is AUC 0.482 inside the class, over-reading drawn positions by **+2.1 pawns**). ⇒ slice 4's first concept joins slice 3's five: **six consecutive move-null concepts.** Original build note follows. | Pawnless K+R vs K+B / K+R vs K+N: discard the material lead, return SF15.1's `push_to_edge` (+`push_away` from the knight). A REPLACEMENT at `draw_class`'s short-circuit, not a term in a sum. MAG = PERCENT of SF's own scale. ✅ byte-identity EXACT (re-verified after the 09-19 eval edit: `250 / 49,440,513 / EBF 4.031`, m0 2,225,417). ☠️ Motivation is v2's **OVER-READ** (a rook up reads ~+1550 mp in a normally-drawn ending), NOT v1's 22-28% false draws. ⚠️ SINGLE-LINEAGE (SF only; Ethereal and Weiss leave it to search). ☠️ **The first symmetry pass (0/4000) was VACUOUS *and* UNREADABLE** — 1 in-class position in 23,113, and the early return published no breakdown (which is what `_eval_symmetry.py` reads). Redone on `ks_sets/t2b_corpus.csv`: **fires, moving the eval 790–1374 mp; symmetry 0/1324**. Tablebase truth: **74% drawn, sign right 33/33, AUC 0.663 KB / 0.721 KN, 85% of wins beyond 12 plies** ⇒ justified twice over. ⚠️ **MAG=100 reaches 985 mp for KRvKN**, not the ~423 the earlier note implies — that figure covers `push_to_edge` alone |
| **winnability** | ☠️ **DO NOT REBUILD AS PORTED** | Three structural faults, only the third about constants: (a) applied to the blended `total` instead of the `(mg,eg)` PAIR, so SF's consumer coupling — the corrected `eg` feeding the scale's strong-side pick and its OCB passer term — **never existed**; (b) **no lazy exit**, so ours fires everywhere while SF's layer only acts in the near-balanced band; (c) a near-monotone scale on a summed total **cannot reorder siblings** (0.5% meaningful move change vs a 9.2% control; same law that parked tempo). ★ Naming: SF1.1 none → SF11 `initiative()` → SF15.1 `winnable()`; Ethereal `evaluateComplexity`; Weiss none. ★★ **Ethereal's has NO king inputs** ⇒ the two-lineage CORE is pawn-structural (pawn count, both flanks, pure-pawn ending); outflanking/infiltration are SF-ONLY |
| **general EG-leg scale** | ⛔ **BLOCKED on architecture** | All 5 references scale the EG leg inside the blend; v2 has no `(mg,eg)` pair, so it is not expressible from `total`. Needs a parallel **`eg_total` accumulator**, shipped byte-identical on its own first. ★ Exception: pawnless 4-5-piece cases are deep-endgame by construction (mg ≈ eg), so tier-2b-style scales need no such change |

| **three instrument-integrity fixes** | ✅ **FIXED 09-19**, all byte-identical or python-only | (1) `eval_v2.cpp` — tier-2b's early return published **no breakdown**, so `g_eval_breakdown` kept the PREVIOUS position's values and every static tool read a chimera; now publishes the total-only contract `draw_class` already used. (2) `eval_breakdown.py` — `KeyError: 'pieces'` on any total-only publication, so it had been **dead for every draw-classified position under `EVAL_ARM=1` since 09-13**; now distinguishes a REPLACEMENT (`terms_available == 1`) from an ordinary v2 position (8/45 own terms published). (3) `_draw_oracle.py` — `tb_lookup` returned a cached dict as final, so a decisive entry stored `{"c":"win","m":None}` could never acquire a DTM; now self-heals, as the legacy-string branch already did. ★ New `FENS=<csv>` class-ground-truth mode on the same tool, plus `ks_sets/t2b_corpus.csv` (1,324 in-class, generated by reusing its own `SIGS`/`random_position`) |

⚠️ **SLICE PLAN STATUS:** 1 ✅ · 2 ✅ · **3 CLOSED 09-18 — five concepts, five parks** · **4 OPEN, first concept
PARKED 09-19** (tier-2b move-null on a powered tablebase suite; scale pair, `eg_total`, rule-50 plumbing, KPvK
won-case magnitude outstanding) · 5 last, under the stricter invented-term gate above.
☠️★★ **SIX CONSECUTIVE CONCEPTS HAVE NOW DIED MOVE-NULL** (central · pair · space · threats · Kaufman · tier-2b),
every one of them reading POSITIVE on static accuracy first. ⇒ before building the next one, run the move test
FIRST: the win-preservation suite (`_draw_oracle.py EMIT_EPD=`) would have retired tier-2b before any C++ existed.
