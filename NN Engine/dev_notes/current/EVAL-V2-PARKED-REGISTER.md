# EVAL v2 — PARKED / DEFERRED / SKIPPED REGISTER  ★ built 2026-09-20

Decision aid for the "unpark and revisit" stretch. Every parked eval item, classified by **why** it is
parked — because the distinction that matters is not *"was it tried?"* but *"was the trial READABLE?"*

**Classes:** `REJECTED` a real result exists · `UNREADABLE` measured, but the instrument was later
invalidated or the knob was dead · `NEVER-MEASURED` deferred/blocked/lapsed · `ON-RULE` failed the
>=3-of-5 reference universality bar.

⚠️ **INSTRUMENT TRUST.** §I = static accuracy vs SF18 — a STATIC PROXY. The record's own verdict: it "read
KS at −1.44% where games said +101", and **six consecutive concepts died MOVE-NULL while reading positive on
§I first**. ⇒ a §I-only verdict, positive *or* negative, is not a move-level result. ★ Asymmetry worth
keeping: instruments detect HARM better than small GAIN, so a §I-harm verdict is stronger than a §I-gain one.
STS alone is untrusted (±150 floor). Pre-2026-08-14 game harness = contamination era. TB (tablebase
win-preservation) and GAMES are trusted.

---

## ★★★ THE PATTERN: TRIGGERS THAT FIRED AND WERE NEVER CASHED

The shelter finding was not a one-off. **Multiple items are parked behind conditions that have since been
met**, and nobody went back:

| item | its recorded trigger | status |
|---|---|---|
| **tempo** | "re-test at the checkpoint margin re-sweep" | ☠️ **FIRED** — `RFP_MARGIN=1000` shipped 09-18. No re-test on record |
| **KS-B shelter** | "needs pawn structure (rung 2)" + "eval has no castling rights" | ☠️ **BOTH STALE** — rung 2 shipped; v2 reads castling rights for trapped rook |
| **corrhist** | "LAST BY NECESSITY — a residual corrector measured where the residual is huge looks essential" | ☠️ **CLEARED** — slices 1-4 are now dispositioned |
| **heat map as an ADDITION** | "if it still pays once dedicated KS / mobility exist" | ⚠️ **MET** for KS + mobility (central/space never built) |
| **weak queen** `WEAKQ_V2_PCT=25` | "inert on §I; keep-or-drop at the checkpoint" | ⚠️ checkpoint passed, never ablated |
| **KS-A frontier maps** | "cash the maps at checkpoints" | ⚠️ a checkpoint (v1-vs-v2) happened, not cashed |
| **weak-unopposed** | "revisit after passers are tuned" | ⚠️ passers shipped 09-13 |

⇒ **Add a standing step to every park: name the trigger AND who checks it.** A trigger nobody re-reads is a
silent permanent rejection.

---

## ▶️ EXECUTION ORDER AGREED 2026-09-20 (the "parked items slice")
Owner's framing: *a parked items slice before finishing things off* — precursor to corrhist, ending where
the search lane begins. Ordered by **signal per hour**, which is NOT the same as the evidence ranking below:

~~**1. Rook files**~~ ✅ **DONE 09-20 — REJECTED** → **2. `PROFILE_EVAL` on v2** (build flag) → **3. Tempo re-test** (one ladder)
→ ~~**4. `WeakUnopposed`**~~ ✅ **DONE 09-20 — BUILT, MOVE-NULL** (flat across 4 arms, no dose response; it is
a CONDITIONING of pawns v2 already penalises — `P(weak | weak_unopp) = 100%` — not the coverage addition
this register called it; knobs kept, trigger = the joint retune)
→ **5. Material taper** (games) → **6. Shelter KS-B** (needs 2 calls)
→ **7. Corrhist** (search-side; the transition item).

☠️ **Three design calls were made the same day** — see `EVAL-V2-REBUILD-LOG.md` 2026-09-20:
- **Tuning objective is STAGED** (absolute → criticality → d7 regret), judged on how much it moves CRITICAL
  positions rather than average MSE. **Blocked on DEFINING "critical" in eval terms.**
- **OvD: the CONCEPT gets a fair shot**, not the v1 implementation. ⚠️ Storm-ownership collision with KS-B
  is still unresolved and blocks item 6.
- **Corrhist is the bridge to search** — and its **attribution method must be built BEFORE** the search lane
  opens, or improvements cannot be credited.

## ★★ RANKED SHORTLIST

| # | item | why it ranks | what would be new |
|---|---|---|---|
| ~~1~~ | ~~**Rook files**~~ ☠️ **DONE 2026-09-20 — REJECTED, stays at 0** | was: built · 5/5 references · §I better 6/6 · closed on STS alone inside the floor · never regret-tested | ✅ **ANSWERED.** Ladder 50/25 · 150/67 · **367/164 (SF11's pawn conversion, 3.7× above anything previously tested)** vs a same-session neutral on the shipped base: **every arm at or below its null**, monotone with dose (−0.4 / −0.7 / −1.6pp), sign replicated on a disjoint 12,000-row set (−0.4pp) — **never positive**. Fires on 53.8%, colour-clean 0/800. Full record: `EVAL-V2-REBUILD-LOG.md` 2026-09-20 |
| **2** | **KS-B shelter, core only** | never built · **4/4 universal** · both blockers stale · a pure COVERAGE addition in the one subsystem whose gap clears the resolution bar | settle OvD-storm ownership; re-derive WEAK:ADJ; **screen before C++** |
| **3** | **Material taper** `EVAL_V2_PAWN_MG≈550` | the **largest §I effect in the whole rebuild**, explicitly recorded UNDECIDED; the only never-gamed large item. ⚠️ margins were just re-swept for a FLAT pawn, so a taper re-couples them | games with `RFP_MARGIN` co-swept — quote eval + margins together |
| **4** | **Corrhist** (search-side) | v1's closure was **STS-only, July harness, a masking q-cache, one coarse keying** — and every search sweep ran at `EVAL_ARM=0`. Ordering blocker now cleared | SF-style 4-way keying under `EVAL_ARM=1`, games-only |
| **5** | **Tempo re-test** | cheap; trigger fired with no re-run | one env ladder on the shipped config. Low prior (SF deleted tempo); if the node-step signature repeats, close for good |
| 6 | **Heat map as an ADDITION** | its retirement note names a re-entry test whose condition is now met; v1 sized it at 1.3pp on a TRUSTED instrument (regret, post-08-14), channels NOT collinear (max r=0.42) | ⚠️ diagnostic only — as a ship item it violates the one-owner charter |
| 7 | **Connected pawns, RANK-FLAT form** | the exclusivity hypothesis is recorded **UNTESTED**, and the untried form is named in the knob comment. Its "once central exists" trigger is DEAD (central never built) | the rank-flat shape only |
| 8 | **Capgains horizon diagnostic** | not a port — its entry condition is a demonstration **nobody has run** | a v2 horizon-failure corpus; design colour-BLIND |
| 9 | threats · space · OvD-v2 · winnability core · weak-unopposed | each has an unfired named trigger (lazy-eval lane / closed-centre corpus / collinearity gate / `eg_total` / passers) | only once the trigger is actually built — no re-run of the same reads |

---

## ☠️ GENUINELY DEAD — DO NOT REOPEN

- **tier-2b** (33/33) and the **scale pair** (35/35) — tablebase ground truth, zero headroom.
- **KPvK won-case · rule-50 · mate drive · convertibility scale** *as Elo work* — the endgame family has no
  eval-shaped headroom (5 failures / 315, four of them only-move). Architecture / teacher value only.
- **central** (0/5, and our error is *smallest* in contested-centre positions) · **knight pair + openness
  scaling** (0/5) · **rook on 7th** (2/5) · **shelter→danger feedback** (2/4) · **castling MAX** (SF-only).
- **latent_threat** — changed **0 moves in 5,000 positions**; KS-A is its successor.
- **piece_value_boost** — harmful on 6/6 corpora, and it reads capgain-mutated piece values.
- **Kaufman** verbatim + derived cells — three parameterisations harmful across five magnitudes.
  ⚠️ §I-only, but harm-side, so the direction is the trustworthy one. Not worth a games night.
- **Flat bishop pair · `MOB_V2_SAFE` · long diagonal** — measured §I losses with no mechanism story to
  overturn them.

---

## ⚠️ ITEMS WHOSE CLOSURE IS WEAKER THAN IT LOOKS

Recorded here so nobody quotes them as settled:
- ~~**Rook files** — STS alone, entirely inside the floor.~~ ☠️ **SETTLED 2026-09-20: now genuinely closed**
  on two disjoint move-level sets (see the shortlist row). ★ The item that motivated this whole section
  turned out to be a real null — which is a result, not a wasted day: it was the cheapest one to settle, and
  it was blocking a slot in every future regression bundle.
- **Corrhist** — STS + July harness + q-cache masking + `EVAL_ARM=0`.
- **Convertibility / endgame scale** — "−3.3 STS / −3 WAC", June, contamination era, inside the floor.
- **Winnability (v1)** — the first descent "tried it on every pass and selected none. Reported as a clean
  null. It was an artifact" (dead knob).
- **pawn_majority** — contamination-era harness; 0/5 as a standalone anyway.
- **Space (Ethereal linear form)** — INERT inside the ±0.05 §I floor, i.e. unreadable rather than refuted.
  Its named prerequisite (a purpose-built closed-centre corpus) was never built.
- **`LATENT_V2_PCT`** — one unattributed "our LATENT null"; instrument not stated. 0/5 references.
- **Bishop pair** — §I-rejected, but the MOVE read was an unresolved null both corpora.

---

---

# ★★ COMPANION: UNIVERSAL REFERENCE TERMS v2 HAS NEVER BUILT  (contrast, 2026-09-20)

Four engines (SF11 · SF15.1 · Ethereal @0e47e9b · Weiss @c735b8f). ⚠️ **SF11 and SF15.1 are ONE LINEAGE**, so
a "3/4" carrying both SFs is really **two independent designs** — noted per row.

| # | term | support | where it lives in the references | new input needed? | our record |
|---|---|---|---|---|---|
| **1** | **King far from pawns / pawnless flank** | **3/4** (2 designs) | SF `PawnlessFlank S(17,95)` + `−S(0, 16·minPawnDist)` (pawns.cpp); Ethereal `KingPawnFileProximity[8]`, S(36,46)…S(−12,−75) | **none** — king square + pawn bitboard | ☠️ **NEVER TRIED, NEVER CONSIDERED.** No knob, nothing in `eval_v2.cpp`, and **absent from the KS rung's own 14-component audit** |
| **2** | **Minor-piece distance to own king** (`KingProtector` / `KnightInSiberia`) | **3/4** (2 shapes) | SF `S(7,8)`×Chebyshev; Ethereal `KnightInSiberia[4]` S(−9,−6)…S(−47,−19), dead-band ≥4, knights only | none | ⚠️ **DEFERRED ON A HYPOTHESIS THAT IS FALSE IN v2** — "overlaps KS zone defence". **v2's KS-A carries NO defender count at all** (SF's knight-defender is DEFER #11), so the slot it was said to overlap is EMPTY |
| **3** | **Weak pawn on a half-open file** (`WeakUnopposed`) | ★ **4/4 — the only unanimous one** | SF `S(13,27)` on isolated/backward when `!opposed`; Ethereal `PawnBackwards[open][rank]`; Weiss `PawnOpen S(-10,-15)` + `PawnBackOpen S(-28,-12)` | **none — `opposed[]` and `halfOpen[]` already exist in v2** | ⚠️ **DROPPED PRE-BUILD ON A CO-OCCURRENCE ARGUMENT, NEVER MEASURED** ("fires mostly on pawns the passer term will reward, 4.27×"). ☠️ **SF deliberately pays BOTH** `WeakUnopposed` and the passer on the same pawn |
| **4** | **Rook / queen behind a passer** ☠️ **NOT A SMALL BUILD — see the 09-21 line read** | **3/4** (2 designs) | SF: own R/Q behind ⇒ `k += 5`, a rung on the PATH-SAFETY LADDER (`unsafeSquares` / `blockSq` empty / stop-square defended ⇒ k = 35/20/9/0), then × `w = 5r−13`. **Max increment 85, not ~110** (w = 17 at the 7th rank). Enemy behind ⇒ the whole span stays unsafe, so k collapses. Weiss `PassedRookBack S(21,46)` flat, own ROOK only | ☠️ **v2 HAS NO LADDER FOR IT TO JOIN** — `passer_value_mp` has the `w` multiplier but no `unsafeSquares`, no `blockSq`-empty test, no stop-square-defended test. Porting "SF's form" means building the ladder FIRST; attaching the +5 standalone is a THIRD form, neither SF's nor Weiss's | ☠️ **SCHEDULED FOR "RUNG 6", NEVER BUILT.** Rung 6 never reads the passed mask; zero hits for `rook.?behind` in `eval_v2.cpp`. v1's `ROOK_PASSER_OWN/ENEMY` is dead |

### ★★★ THE PATTERN REPEATS — three of four were REASONED away, not measured
- #3 dropped on a **co-occurrence argument**, #2 on an **overlap hypothesis** (false in v2), #4 on
  **scheduling** (the rung shipped without it). Only #1 was never thought of at all.
- ⇒ **#4 is an EIGHTH lapsed trigger**, alongside the seven above. The failure mode is systemic, not
  incidental: items die by reasoning in a design doc and are never revisited as measurements.

### ▶️ CHEAPEST FIRST
☠️ **#3 WeakUnopposed — BUILT AND MEASURED 2026-09-20: MOVE-NULL.** Flat at 49.3 / 49.3 / 50.1 against a
49.8 null across both shapes and a 2× magnitude range, no dose ordering. ★ And the pre-build overlap read
explains it: `P(weak | weak_unopp) = 100%` ⇒ it is a **conditioning of pawns v2 already penalises**, not the
"pure coverage addition" this document called it. ⇒ the **4/4-unanimous argument is now empirically spent**
on the one term where it was strongest. Knobs kept at 0; trigger = the joint retune.
Original framing, kept for the record: **#3 WeakUnopposed** is the standout: **4/4 universal**, inputs
**already computed** in v2's `PawnEntry`, never measured, and the argument that killed it is contradicted by
SF's own design. Then **#1** (needs
nothing but king square + pawns, and was never even considered). **#2** needs its overlap claim re-checked
against v2 rather than v1. **#4** is a multiplier path in SF, not an additive one — port the shape, not the
constant.

### ☠️ SINGLE-LINEAGE — skip per the >=3-of-5 rule (recorded so they are not re-derived)
King-flank attack/defence (2/4) · pinned-pieces-in-kingDanger (2/4 — ⚠️ though `MOB_V2_PIN` has since built
the pin machinery) · shelter castling-MAX (2/4) · Weiss `KingLineDanger` (1/4) · Ethereal `KingDefenders[12]`
(1/4) · closedness-conditioned piece values (1/4; v1 built it, dead code) · `PassedFile` (2/4) ·
Weiss `PassedSquare` (1/4; v2 has exact KPK) · `WeakLever` (2/4) · `RookOnClosedFile` (1/4, **no record hit
at all**) · `PawnDoubled2` (1/4, no record hit) · rook/bishop-on-king-ring · `BishopXRayPawns` ·
`RookOnQueenFile` · Ethereal `RookOnSeventh` · `ThreatOverloadedPieces` · `WeakQueenProtection`.

### ✅ RESOLVED BY LINE READ 2026-09-21 (engine-contrast agent, sources read directly)
- ~~Ethereal's lack of a rook-behind-passer~~ **CONFIRMED** — `evaluatePassed` read in full, no rook/queen
  reference. #4 stands at **3/4**.
- ☠️ **CORRECTION — "exclude shelter pawns attacked by enemy pawns" is 2/4, NOT 3/4.** SF15.1
  (`pawns.cpp:236`) and Weiss do; **SF11 (`pawns.cpp:191`) and Ethereal do NOT** — the SF lineage splits
  internally on it. Any "3/4" claim elsewhere in the record inherits this error.
- ☠️ **CORRECTION — #2's premise is only half true.** "v2's KS-A carries NO defender count" is right as a
  COUNT, but defence already enters KS-A through the `weak` set (`eval_v2.cpp:529`) and the safe-check set
  (`:543`). The KingProtector overlap is **partial, not empty** — a minor beside our king already lowers
  KS-A units. ⚠️ SF pays `KingProtector` anyway AND carries a separate knight-defender term v2 lacks.
- ☠️ **CORRECTION — the "rung 6 collision" (rook-behind × rook-on-open-file) is overstated.** Own rook
  behind our OWN passer is **mutually exclusive** with rook-on-file by construction (our pawn is on the
  file, so it is not own-semi-open). Only the ENEMY-passer case can co-fire.
- ⚠️ **SYMMETRY HAZARD for any shelter port:** Ethereal's `KingShelter` second index is the **ABSOLUTE
  file**, an asymmetric table — a verbatim port fails `_eval_symmetry.py`'s file-mirror check exactly as
  `PawnIsolated` did. SF's `map_to_queenside` tables are symmetric by construction.
- ★ **SF DOUBLE-WIRES SHELTER:** the score seeds `king()` directly AND feeds `kingDanger` as `−6·mg/8`
  (`evaluate.cpp:384`, `:455`). "Build shelter" is really "choose one channel or both" — a design decision,
  not a port.
- Still open: SF15.1's `WeakUnopposed` line was not read directly (only `WeakLever` was) — moot, the term
  measured null on 2026-09-20.
- ⚠️ Ethereal/Weiss were read at GitHub `master`, NOT the register's pinned commits (@0e47e9b / @c735b8f);
  constants may differ from the pins.

---


## ★ REVIVAL ROUND 2026-10-05/07 (C3 doc §20-21a) — every item MEASURED on the depth target + games
| item | outcome | where it goes |
|---|---|---|
| connected pawns (rank-flat form #7 above, and the shipped form) | ✅ SHIPPED 10-04 (CONN_MAG 21 / SUPPORT 99 / EG_RATIO 101; +14.5 ± 7.7) | done |
| threats (all 6 legs, Kaufman-style per-leg fit) | ☠️ CLOSED on a fair test (joint −0.01%; tactical ⇒ search's) | search phase only (threat-aware ordering) |
| material taper · space · long diagonal · reach · latent · placement bundle | null on the depth target (all fire) | final retune at 0 |
| rook files · winnability +PASSED | small (−0.21% / −0.33%) | final retune |
| Kaufman depth re-fit · mobility cells · king protector (C3-c) · KFL+PST pair | gated, NOT shipped (combined 0.9σ / SF 0 / instruments split / instruments split) | ✅ **done: all fitted jointly in the 10-11 structural retune (gen 2: KFL and KPROT switched on, Kaufman cells, C1 mobility, PST) and live in every gen since.** Not open; the 10-29 "re-screen" proposal was stale (found 11-02) |
| heat map | v1-only — never built for v2 | unbuilt |
| tempo | = the STM nuisance (~+8 cp) — not separable here | search phase / final retune |

## ★ POT PHASE (2026-10-13 → 10-19) — parked items and their RE-READ TRIGGERS (POT brief §5-§5o)

| item | state | evidence | re-read trigger |
|---|---|---|---|
| **Minor-ending over-confidence / ONEMINOR cell** (owner 10-19: "keep in mind, may return one day") | SHIPPED in C64 (games +); static WDL says mis-calibrated | WDL: minors-only need scale 1.91 vs 2.82 global (we over-state edges ~1.5×); ONEMINOR=16 raises WDL MSE +14% where it fires; games by start type: one-minor starts pooled z ≈ +2.9 | **the final retune** (fit on SF18 WDL labels, not fixed-formula cp) — and any SF11 minor-ending port (KNPKB / KBPKN) |
| POT race (straight-line), king-path race | PARKED at 0 (code + closure kept) | d10 −60% pawn rows with cells, d12 nothing beyond cells; move test z +1.57 / +0.63 | **search transition**: race + tightened RFP / futility / razoring margins at equal TIME (−22% pawn-ending nodes at d10) |
| Complexity scaler | PARKED (prototype) | d12 −0.41% vs ≥ 2% keep | if a search-side use of "how unresolved" is designed (LMR / time) |
| Middlegame POT (levers by who acts, storm, half-open, majority, space) | PARKED (prototype `_pot_mg_fit.py`) | A −0.46% vs control, t+16/24 ≈ −0.5%, move z −0.77; break features ≈ 0 weight | POT-as-search in the transition (game 7: a pending break pruned / not reached) |
| Preplan E1 (leader cannot create a passer → winnability), M2 (closed-file openability correction, −5.0σ target), M1 in FEEDER form on `transform_critical` / `transform_trap`, M3 (majority → passer mg feeder) | **NEVER-MEASURED** | — | narrow items, judged WHERE THEY FIRE (critical venue) on WDL labels; any time an eval slot opens |
| Learned correction (boosted trees, `_pot_gbm_ceiling.py`) | diagnostic only (owner: a model ⇒ NNUE, not trees) | −10% on fixed-formula cp label (no leakage) → gone on SF18 WDL at small n | queue #82 full-size WDL check; the NN phase (trajectory sets as data) |
| ☠️ Fixed-formula cp → win% labels | INSTRUMENT DEFECT (memory sf-cp-labels-are-material-scaled-use-wdl) | mean 15–16 win% pts from SF18's own WDL | every cross-class accuracy claim, the retune objective |

## ★ SEARCH PHASE (2026-10-27) — parked items and their RE-READ TRIGGERS

| item | state | evidence | re-read trigger |
|---|---|---|---|
| **Corrhist** (`CORR2`, SF-consensus form, 4-way keying; owner 10-27: "it could be a tuning problem if we are just using the tunings of other engines") | PARKED, default off, code kept | 10+0.1 vs gen 6.2 TM, 1,000 g each: full weights −8.7 ± 18.4 · half −2.4 ± 19.6 · quarter +7.6 ± 20.0, re-run +3.5 ± 19.2 (#110), pooled ≈ +5.5 ± 14 over 2,000 g (OPTIMIZATION_LOG 10-27). Smaller corrections score better; none is shown to beat off. Our pruning margins were tuned without it; the references tuned theirs with it on | **the full retune**. In the 11-07 close-out order that is the **margins re-sweep** (step 5, after lazy eval): corrhist is swept inside it, not as a separate item, and not in the eval-table retune (owner asked 11-07). `CORR2_WP` / `CORR2_WN` (and `CORR2_LIMIT`) tuned jointly with the RFP / futility / razoring margins, starting from the quarter weights `CORR2_WP=88,CORR2_WN=100` (#110 kept them positive) |
| **Threats in eval** (`THREAT_V2_*`; re-opened 10-28 by the owner: "genuine space to grow … compared to SF11 static") | OFF; **ship candidate (gen 6.4?) at mg 50 / eg 100** — #114 on gen 6.3: 10+0.1 +38.4 ± 19.7 (mg 50), +31.0 ± 19.1 (mg 0), 60+0.6 +61.4 ± 42.8 (mg 50) | Closed 10-05/07 (per-leg fit, joint −0.01%, "tactical ⇒ search's"), on the cp label since found material-scaled. On SF18 WDL (10-27): static net 0 once scaled, but split by stage (egl −11.8 ± 5.4, mg +30.9 ± 16.4 vs gen 6.2; SF11 leads us egl −58, eg −84); games +7.3 ± 19.9 alone, ≈ 0 on top of ordering (#111). Knobs `THREAT_V2_MG_PCT` / `THREAT_V2_EG_PCT` (−1 = PCT) | **on gen 6.3:** (1) mg/eg pairs on static + d10 WDL per stage vs SF11; stop if none lowers egl without raising mg. (2) per-leg on WDL. (3) best pair 1,000 g at 10+0.1 and 200 g at 60+0.6, then SF18 if positive. Games ≤ 0 or SF18 < −1σ ⇒ parks to the full retune. #115 crank: with RFP_MARGIN 600 mg 50 +73.0 ± 19.3, mg 0 +39.1, RFP 600 alone −27.5 ⇒ candidate C = mg 50 / eg 100 + RFP 600. ▶️ #116: fresh seed, WAC @250k nodes, SF18 @1100 on 2 seeds, 60+0.6, Mediocre, RFP 800 / 400 plateau |
| **Threat ordering** (`THREAT_ORDER_W`) | ✅ SHIPPED gen 6.3 (10-28), W 100 | +20.9 / +21.9 vs gen 6.2 at 10+0.1 (#111, 2 seeds); 60+0.6 +41.9 ± 45.6; SF18 @1100 53.8% vs 50.4%; Mediocre +113 vs gen 6.2's +119 on the same openings (paired −6 ± 37, #112); W50 −15.3, W200 −11.5 vs W100 (#113) | — (re-check W in the full retune) |

## ★ TRANSITION ARC, EVAL LEG (2026-10-30 →) — parked items and their RE-READ TRIGGERS

| item | state | evidence | re-read trigger |
|---|---|---|---|
| **Endgame caps** at SF15.1's values (`POT_V2_WIN_OCBX −42 / OCBX_PC 3, ROOKE −28, QNOQ −27 / QNOQ_MINOR 3, LONEMINOR −64`) | PARKED 10-31, knobs kept at 0 | static eg WDL −5.8% (scaling swap −12.7% → −2.4%); games vs gen 6.4: +13.9 ± 18.9 (#119), −2.8 ± 20.1 (#122), + RFP 500 +7.0 ± 19.1, 60+0.6 −8.7 ± 42.9 (#124); **transition starts +12.9 ± 14.5 (#131), +5.2 ± 13.9 (#132) ⇒ pooled +9.0 ± 10.0 (z 1.77)**; capped-ending starts (`capped.epd`) +9.7 ± 13.4, −1.0 ± 14.0 ⇒ pooled +4.3 ± 9.7 (#134); the two unselected replicates pool to +2.1 ± 9.9, so a few Elo at most where they fire; **owner 11-02: PARKED** (not the narrow-item pool) | **the final retune** (fitted jointly with the scale, on SF18 WDL) · or the scale moved onto the eg leg (§3 item 5). They overlap the damp threshold (both shrink drawish edges): if the damp ships, re-judge the caps on top of it, not on gen 6.4 |
| **Damp-only drawishness threshold** (`WIN_V2=1, WIN_V2_DAMP=1, WIN_V2_PHASE_MAX=0, WIN_V2_SKIP_PAWNEND=1`, SF11 ×4.69) | PARKED 10-31, default off, code kept | static eg WDL −2.9%, symmetry clean; games vs gen 6.4: alone −5.9 ± 19.9 (#124); + RFP 500 +15.3, +20.2 (#124 / #125) but RFP 500 alone +13.6, +10.4 (#125 / #126); **head to head damp + 500 vs 500 −20.9 ± 17.9 (#126)**; transition starts −0.4 ± 13.9 (#131), so null where it fires too | **the final retune** (weights fitted on SF18 WDL with the scale and the caps, not SF11's ×4.69) |
| **Passer prune exemption** `ENABLE_PASSER_PRUNE_EXEMPT=1` (span-checked true passers, landing rank ≥ 5, out of LMP / LMR) | default 0, code kept | UI game 2 move 11: finds Kf1 at d13-15 where shipped never does; eg_critical 928 g +4.5 ± 10.6 (#124); bench +2.9% nodes | **search arc**: judge with the selectivity rework, on a venue built from positions where a passer run decides (UI blunder set) |
| **FIX_LMR_EG** (move-number LMR in light endgames) | FAILED, default 0 | −19.8 ± 12.5 on eg_critical (#120): the light-endgame bypass is a productive bug | **search arc**: test reducing LESS there, not more |
| **Pinned defenders** `KS_V2_PIN_DEF=1` (a pinned own defender attacks only along its pin ray, in KS's private defender map) | PARKED 11-01, default 0, code kept (rebuild only when a non-pawn defender is pinned: result-identical, no bench cost) | games vs gen 6.4, UHO 10+0.1: +2.8 ± 19.0 (#129 `pin`); with `KS_V2_BLOCKERS=0` +0.7 ± 18.7 (#129 `pin_b0`), so BLOCKERS −20 vs 0 is also flat. Both under the +8 replicate bar (owner 11-01); depth-fit refresh and BLOCKERS refit not built | **the final retune** (fit KS with PIN_DEF on, on a refreshed depth-fit base) · or a KS venue built from positions where a pinned defender decides |
| **K+B+N vs K technique** (owner 11-03: "sure", park it) | NEVER BUILT — found 11-03 in a UI test | UI test at 60+0.6, `8/8/8/4k3/8/8/8/1NB1K3 w - - 0 1` (dark bishop) vs SF18 defending: we drove the king to **a8, the wrong (light) corner**, boxed it there for ~35 moves, no mate by move 50 (then, the draw claimable, the engine stopped caring and dropped the bishop). The ordinary eval only pulls the weak king to an edge / any corner. SF handles the ending with a dedicated value: known win + kings close + push to the corner of the BISHOP'S colour (`stockfish_11/src/endgame.cpp` KBNK). Same shape as our tier-2b K+R vs K+minor replacement (`tier2b_value_mp`, `TIER2_V2_MAG`) | **any eval slot** — a narrow item, judged where it fires: 50-100 K+B+N vs K starts (both bishop colours, random squares) vs an SF defender at 60+0.6, **pass = mates within 50 moves** (today 0 / 1); elsewhere only byte-identical (the replacement fires in this ending alone). Rare in games, so little Elo; it also matters as NN-teacher labels |
| **Shelter → danger coupling** `KS_V2_SHELTER_K=12` (SF11's 6/8 in our units: the king-shelter deficit feeds the attack-danger units, a few % of them) | PARKED 11-02, default 0 = byte-identical, code kept | games vs gen 6.4, UHO 10+0.1: −6.3 ± 19.9 (#132 `shelter`, seed 1323, Ptnml 90-82-166-80-82), inside its registered −10 … +15, under the +8 replicate bar | **the final retune** (K fitted jointly with the KS units, on a refreshed depth-fit base) · or a KS venue where a broken shelter decides |

## ★ THE FOCUSED EVAL ARC AFTER SEARCH V2 (owner 2026-11-07)

Owner: "the room left to grow statically isn't that much, so it will likely require another focused targeted arc rather than things here and there ... with stronger search, we can better understand how our eval translates into it, to adjust how the eval operates." So these wait for that arc and are not reopened in the transition. **Trigger for all of them:** search v2 has shipped and play depth at 10+0.1 has risen by about 2 plies or more → re-pass the fit sets at the new play depth, retune on it, then take these on the new base, judged in games.

| item | state | evidence | what to do in the arc |
|---|---|---|---|
| **Scale on the eg leg + pawn ramp to 4 pawns** (`EVAL_V2_PAIR=1`, transition item 5) | NEVER RUN, parked 11-07 | low prior: the endgame caps and the damp-only threshold both gained static accuracy (−5.8%, −2.9%) and no Elo | SF's exact object (scale on the eg leg, gated by material class), small code |
| **Queen vs minor compensation** | live misread, no fix built | [[v2-overvalues-queen-vs-minor-compensation]] | build the move test first, then the term |
| **KR vs K+minor** | live misread, no fix built | reads about +1.5 pawns on an ending that is about 74% drawn ([[kr-vs-kminor-is-74pct-drawn-and-v2-reads-it-as-plus-1-point-5-pawns]]) | note: "stay closed" in the 10-30 list as a tablebase question; reopen only as a static correction |
| **How eval translates into search** (new question for the arc) | not measured | owner's reason for waiting: 09-06 crank; threats plus `RFP_MARGIN` 600 about +66 while threats alone added nothing | per term: what changes in pruning, node counts and moves under search v2's margins, not only static error |

Also on that base: the other items above whose trigger names the search arc or the final retune (POT race, shelter coupling, caps, corrhist).

**Reference game (owner 11-07): look at this first.** Gen 7 at BLITZ vs Stockfish at 0.05 s:
[`../archive/games/owner-game-2026-11-07-blitz-vs-sf-q-vs-rn.md`](../archive/games/owner-game-2026-11-07-blitz-vs-sf-q-vs-rn.md).
With queen against rook + knight, 23.Qb7 let e5 go for the queenside pawns, which left all pawns on one wing, a drawn-looking ending
that White had been better in before. The owner wants the engine to "calculate that sort of transformation at move 23 better." It
bears on queen vs minor compensation and on scale on the eg leg. The file lists the checks to run first.

## PROVENANCE
Built by a full sweep of `dev_notes/` + memory on 2026-09-20. Companion docs:
`EVAL-V2-CURRENT-CONFIG.md` (shipped config + decision register) · `EVAL-V2-REBUILD-LOG.md` (the narrative) ·
`our_eval_reference.md` §v2 (the ours↔SF term map) · `INSTRUMENT-MAP.md` §F (what we cannot measure).
