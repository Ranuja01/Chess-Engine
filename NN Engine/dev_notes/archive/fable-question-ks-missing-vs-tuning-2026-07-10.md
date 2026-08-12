# Fable consult — is our eval MISSING king-danger detectors, or MIS-TUNING realizability, or the interdependence trap? (2026-07-10)

*Self-contained. You (Fable) have repo access — pointers in §7 — but everything needed is below. This follows
the earlier consult `fable-question-ks-realizability-2026-07-09.md`; the diagnose-first data below UPDATES it.*

## 0. Engine + venue trust (unchanged, brief)
Custom C++ **HCE** engine, pre-NNUE by choice, non-negamax, absolute Black-positive eval, millipawns. Refs
with per-term eval: **SF11** (classical HCE, prints labelled terms) and **SF18** (NNUE). **Venue trust:**
external **gauntlet** (our engine vs throttled native SF18, ≥2 seeds) = TRUTH; **lightning self-play** and a
retired **cploss compass** ANTI-PREDICT (this matters in §5). Fixed-node A/B (node_ab) = screen only.

## 1. The problem, quantified (what we're sure of)
Root cause of a ~22-24% "collapse" rate (our eval peaks ≥+2p then draws/loses vs SF): a **CONDITIONAL static
eval over-read**. Measured (`diagnostics/verify_triage_static.py`, mover-POV cp, on the collapse decision-FENs
vs a control sample of general midgame FENs):
- **Static, not search:** our STATIC eval over-reads +368cp on collapse FENs (≈ our depth-18 search's +273);
  a phantom-search fix (singular extensions) was NEUTRAL — the eval itself is wrong.
- **Conditional, not intrinsic:** CONTROL (183 general midgame FENs) static gap = **+8cp** (calibrated!),
  vs COLLAPSE **+368cp**. We are right on average and wrong only in this class ⇒ **global Texel/SPSA
  recalibration is refuted** (optimising an already-right average — which matches our years of stalled KS
  tuning going nowhere).
- **Classical-solvable:** SF11 (classical, no NNUE) ALSO reads the collapse FENs ~0 (+5cp) ⇒ the missing
  knowledge is cheap/classical, NOT an NNUE boundary.
- **Decomposition of the +368** (per-term mean, mover-POV; since SF≈0, +ve = over-reader): `pieces` **+177**
  (dominant, broad), `capture_gains` +58, `piece_value_boost` +49; **king-safety terms +2** (≈nothing).

## 2. ⭐ The king-danger finding: MISSING DETECTOR, not tuning (`diagnostics/ks_gap.py`)
We compared OUR king-danger eval (active term = `latent_threat`; a rebuilt attack-unit `king_safety` term
exists but is parked — see §4) against **SF11's "King safety" term**, both mover-POV net cp (−ve = our king in
danger), per position:
- **COLLAPSE (n=67):** mean our_ks **+2cp** vs SF11 **−35cp**. **corr(our, SF11) = −0.04**, OLS slope −0.01.
  Of the 16 positions where SF sees real danger (≤−100cp), **we miss 15 (94%)**.
- **COLLAPSE-MIDGAME (n=43):** corr −0.03; SF-danger 12, **we miss 11 (92%)**; worst subset SF11 KS = −300..−485
  where we read ~+2.
- **CONTROL (n=367):** corr +0.15 (still weak); we miss 80% of SF's danger positions too.
**Interpretation:** near-ZERO correlation ⇒ our king-danger eval does NOT track SF's at all — this is a
**structural blindness / missing detector**, not an under-weighting (tuning would show high corr + slope<1).
The blindness is GENERAL but the SF-danger positions are CONCENTRATED in the collapse class.

**Honest scope:** KS-blindness explains the SHARP subset (~24% of collapses, modest −35cp mean). The BIGGER
lever is the broad `pieces` +177 over-read across ALL collapses, which is NOT king-safety. So we likely have
≥2 missing/blind pieces: (a) king-danger, (b) whatever makes `pieces` over-value our position in this class.

## 3. The failure-mode signature (fits the collapse story)
The collapse midgame profile that DID separate from control (logistic, holdout AUC 0.85-0.87): "a **pawn up**
while **over-extending our offense** (off_us high, our_overreach = our-attack − opp-defense high)". So: we're
nominally up material and pressing, SF sees our king exposed / the attack unconvertible, and our eval credits
the material+activity while missing the danger. `pieces` + `capture_gains` + `piece_value_boost` all inflate
together in the class (a shared "we're winning" over-optimism), while king-danger reads ~0.

## 4. Prior KS work + the INTERDEPENDENCE trap (the user's key caveat — please weigh heavily)
We have ALREADY rebuilt the KS *detector* (2026-06-30, memory `[[ks-detection-rebuild]]`): wiring `KS_ZONE2`
(wider 2-ring zone), `KS_WEAK` (undefended zone squares), `KS_STORM`, and a per-king **dynamic magnitude**
`KS_DYN` (realness = att_cnt·(open_files+weak_squares), scale each king up/down) moved under-fire kings from
0→9-36 units, "term moves toward SF11's sign." **So the detector CAN be made to read correct danger.** It was
**PARKED** for these reasons, and this is the crux of the user's question:
- **REPLACE latent_threat with KS = −47 Elo.** Removing latent_threat regressed on the ks=0 positions (its
  king-pressure signal was load-bearing elsewhere).
- **Gentle KS BESIDE latent_threat = ~0 Elo** (redundant / double-counts — the two king terms overlap).
- Every realizability conditioner (`MOD_KS_CONTROL`/`MOD_KS_BACKING`/`KS_DYN`) only DAMPED the gentle KS.
- Also known: **pawn-shelter was triple-counted** across `pieces` and the king terms; `latent_threat` is a
  misnamed KS-ish term. The eval terms OVERLAP heavily.
- **The user's framing:** "rebalancing KS required rebalancing other items too; without them the 'correct' KS
  numbers were actually not correct — the entire eval is interdependent." (Matches memory
  `[[holistic-eval-pivot]]`: eval is ONE vector; isolated-KS dies from ISOLATION not badness.)
- **BIG CAVEAT that reopens all of it:** every one of those verdicts was on **lightning self-play** (which we
  now know ANTI-PREDICTS) with a fast-self-play depth bias. They were NEVER tested at the gauntlet, and never
  against this specific measured collapse class.

## 5. The capg precedent (the user's optimism, and our one eval win)
Our ONLY shipped eval win is **capg tension-conditioning**: `capture_gains` was over-reading; instead of
retuning its scalar we made it `value × f(tactical-tension detector)` — a conditioned SUBTRACTION, centered
(f=1 at neutral so the 94% is untouched), clamped, sparse. The user's intuition: if the KS/pieces under- and
over-valuation is a **massive, systematic, pattern-based** error (as the corr-0 / 94%-miss data suggests),
that's GOOD — it's the same shape as capg (a conditioned fix on a systematic pattern), not diffuse noise.

## 6. Questions
1. **Missing detector vs tuning vs interdependence vs other?** Given corr(our_ks, SF11_ks) ≈ 0 and 94% of
   SF's danger missed, is this best read as (a) a genuinely MISSING detector (our king-danger eval doesn't
   compute the thing SF computes), (b) a realizability/tuning problem (we compute it but our conditioning
   zeroes it), (c) the interdependence trap (latent_threat/pieces/shelter overlap so the danger is "priced"
   elsewhere and double-fixing breaks balance), or (d) something else? The rebuilt-detector history (§4) says
   we CAN read the units but adding them REGRESSED — how do you reconcile that with the corr-0 blindness?
2. **How to add king-danger WITHOUT the interdependence blowup** (and without straight-copying SF)? If
   latent_threat already carries overlapping king-pressure and `pieces` triple-counts shelter, what's the
   principled way to introduce a REAL king-danger subtraction that nets correctly — e.g. first REMOVE the
   overlapping/misplaced pieces of king-pressure from `pieces`/`latent_threat`, THEN add the clean detector?
   How do we know when the "rebalance the whole neighbourhood" is done vs. still half-done (the exact trap
   that made prior 'correct' KS numbers wrong)?
3. **The `pieces` +177 (the bigger lever):** king-danger is only ~24% of the collapses. The broad `pieces`
   over-read is conditional on the same "pawn-up-but-unconvertible / over-extended" class. Is that ALSO a
   missing detector (a convertibility/compensation term), the SAME interdependence (pieces over-values
   because king-danger isn't subtracted), or a separate realizability tuning? How would you tease these apart
   with the data we have (SF11 per-term is available)?
4. **Venue:** all prior KS verdicts were lightning (anti-predictive). Should we now trust the gauntlet +
   this collapse-class measurement to REOPEN KS, and what's the minimal gauntlet-gated experiment that would
   distinguish your hypotheses in §1 without a month of tuning?
5. **The capg analogy (§5):** does the corr-0 / systematic-94%-miss pattern support a capg-style conditioned
   fix (centered, clamped, sparse, on a cheap detector), and if so, is the right target the king-danger
   SUBTRACTION, the `pieces` over-optimism DAMP, or a single shared "unconvertible/exposed" realizability
   that scales the whole positional-optimism cluster? Given eval-interdependence, is one shared factor SAFER
   than several per-term fixes?

## 7. Repo pointers (verify / go deeper)
- Data + tools: `diagnostics/verify_triage_static.py` (--dump corpus + any-knob A/B), `diagnostics/
  fit_convert.py` (separation fit), `diagnostics/ks_gap.py` (the corr-0 / 94%-miss analysis),
  `diagnostics/corpus_ks.csv` (per-FEN our-vs-SF11 KS), triage CSVs in `selfplay/games/g_base_s0|s1/triage.csv`.
- Handoff `dev_notes/SESSION-HANDOFF-2026-07-08.md` pt.6/7/8 (the full diagnose-first chain). Memory:
  `[[ks-detection-rebuild]]` (the rebuilt-but-parked KS + REPLACE −47 + interdependence),
  `[[holistic-eval-pivot]]` (eval=one-vector), `[[realizability-conditioning-architecture]]` (value×f(detectors)),
  `[[capg-tension-conditioning]]` (the shipped precedent), `[[external-gauntlet-calibrated]]` (the venue),
  `[[sf11-texel-scale-invariance]]` (why we don't fit SF magnitude), `[[singular-banked]]` (search fix failed).
- KS code: `king_safety_danger`/`king_safety_score` + `get_latent_threat_score` in `cpp_bitboard.cpp`; knobs
  (`KING_SAFETY_MAG`, `ENABLE_KS_REPLACE_LT`, `KS_ZONE2/WEAK/STORM/DYN`, `MOD_KS_BACKING/CONTROL`) in
  `search_engine.h`.
