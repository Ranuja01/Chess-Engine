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
| Kaufman depth re-fit · mobility cells · king protector (C3-c) · KFL+PST pair | gated, NOT shipped (combined 0.9σ / SF 0 / instruments split / instruments split) | final joint retune |
| heat map | v1-only — never built for v2 | unbuilt |
| tempo | = the STM nuisance (~+8 cp) — not separable here | search phase / final retune |

## PROVENANCE
Built by a full sweep of `dev_notes/` + memory on 2026-09-20. Companion docs:
`EVAL-V2-CURRENT-CONFIG.md` (shipped config + decision register) · `EVAL-V2-REBUILD-LOG.md` (the narrative) ·
`our_eval_reference.md` §v2 (the ours↔SF term map) · `INSTRUMENT-MAP.md` §F (what we cannot measure).
