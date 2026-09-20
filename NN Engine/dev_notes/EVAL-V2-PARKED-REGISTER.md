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

## ★★ RANKED SHORTLIST

| # | item | why it ranks | what would be new |
|---|---|---|---|
| **1** | **Rook files** `ROOKFILE_V2_OPEN/SEMI=0` | **Built · 5/5 references · §I better on 6/6 corpora · closed on STS ALONE with every point inside the ±150 floor · never had a regret or games read.** The recorded 2×2 (mobility × rook files) pre-dates the mobility ship, so it was never re-read on the shipped base. v2 currently scores **no rook-on-open-file bonus at all** | a regret read on the shipped base, then ride the next regression bundle |
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
- **Rook files** — STS alone, entirely inside the floor. The strongest "closed" item that is not actually closed.
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
| **4** | **Rook / queen behind a passer** | **3/4** (2 designs) | SF: own R/Q behind ⇒ `k += 5` (a MULTIPLIER, up to ~110 at rank 7); enemy behind ⇒ whole span unsafe. Weiss `PassedRookBack S(21,46)` flat | none — passed mask + file fill exist | ☠️ **SCHEDULED FOR "RUNG 6", NEVER BUILT.** Rung 6 never reads the passed mask; zero hits for `rook.?behind` in `eval_v2.cpp`. v1's `ROOK_PASSER_OWN/ENEMY` is dead |

### ★★★ THE PATTERN REPEATS — three of four were REASONED away, not measured
- #3 dropped on a **co-occurrence argument**, #2 on an **overlap hypothesis** (false in v2), #4 on
  **scheduling** (the rung shipped without it). Only #1 was never thought of at all.
- ⇒ **#4 is an EIGHTH lapsed trigger**, alongside the seven above. The failure mode is systemic, not
  incidental: items die by reasoning in a design doc and are never revisited as measurements.

### ▶️ CHEAPEST FIRST
**#3 WeakUnopposed** is the standout: **4/4 universal**, inputs **already computed** in v2's `PawnEntry`,
never measured, and the argument that killed it is contradicted by SF's own design. Then **#1** (needs
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

### ⚠️ UNRESOLVED IN THE CONTRAST
- Ethereal's lack of a rook-behind-passer term came from a **fetch summary, not a line read** — confirm in
  `evaluatePassed` before quoting #4 as 3/4 rather than 4/4.
- SF15.1's `WeakUnopposed` line was not read directly (only `WeakLever` was).

---

## PROVENANCE
Built by a full sweep of `dev_notes/` + memory on 2026-09-20. Companion docs:
`EVAL-V2-CURRENT-CONFIG.md` (shipped config + decision register) · `EVAL-V2-REBUILD-LOG.md` (the narrative) ·
`our_eval_reference.md` §v2 (the ours↔SF term map) · `INSTRUMENT-MAP.md` §F (what we cannot measure).
