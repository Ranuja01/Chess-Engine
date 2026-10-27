# KS collapse / detector-quality investigation (2026-08-15→16)

**Frame (owner-driven).** The aggregate instruments (pvb per-term, regret-mean) are BLIND to KS's real cost,
which lives in the TAIL: game COLLAPSES (peaked-winning-then-lost). KS collapses are dominated by UNDER-read
(we miss real king danger), which the average dilutes. The right instrument is the collapse pipeline
(`vs_sf 2400` → collapses.csv), not regret-mean. Detector-DEFINITION quality is the lane with the only wins
(latent_threat weak-def→SF-like, OvD); magnitude tuning is tapped ([[static-eval-top-culprits-are-search-absorbed]]).

## Fresh collapse gather (shipped bundle, vs SF@2400, lightning)
150 games, our score 47.3%, **48 collapses (32% rate)** → `selfplay/games/vssf_2400/collapses.csv`.
Owner note: collapse rate is DOWN over the weeks — the detector-definition lane is working, slowly.

## 5-FEN triangulation (ours-static / SF11-classic / SF15.1c / SF18-search / our-D10) + per-term breakdown
`probe_fens.py` (needs `STOCKFISH_PATH` set to native SF18). Two distinct collapse drivers, confirmed by breakdown:

| FEN | ours | our-D10 | SF11 | SF18@22 | driver (from per-term) |
|---|---|---|---|---|---|
| g0 (W, uncastled e1) | +4.99 | +4.16 | −1.23 | −2.90 | **KS under-read**: SF11 King-safety **−4.25** vs our **−0.63** |
| g20 (W, a1 + Rb8/Rb6 battery) | +0.99 | +2.82 | −0.21 | −4.33 | **KS under-read**: SF11 KS **−2.32** vs our **~0** |
| g68 (W, up Q but mated) | +15.31 | +10.25 | +2.08 | 0.00 | **capture_gains +8.25** on a poisoned (mate-in-2) f1 knight; SF11 KS only −0.62 |
| g49 (B, king stormed) | −6.63 | −4.66 | +4.56 | +2.86 | **capgains hallucinates ~4.5 material** (SF11 Material +0.19) + KS under-read (SF11 KS +4.89) |
| g40 (W) | +5.11 | +1.01 | n/a | −6.66 | **IN CHECK** (black g2 pawn attacks h1) → SF static undefined; set aside |

Read: our static AND our D10 search agree with each other and are wildly wrong; **SF11/SF15 CLASSICAL (hand-coded
HCE) already see the danger** → statically fixable detector gap, not NNUE-only, not search's job.

## KS component dump (`KS_DEBUG_DUMP`, floor forced to 0 to see below-floor cases)
| FEN | attsq | weak | safe | attpc | openf | units | our KS |
|---|---|---|---|---|---|---|---|
| g0 (W) | 5 | 1 | **0** | 3 | **0** | **13** (=floor) | −0.63 |
| g20 (W) | 2 | 1 | 0 | **1** (battery rear rook invisible) | 1 | **4** | ~0 |
| g49 (B) | 5 | **0** | 0 | 3 | 2 | **12** (just below floor 13!) | ~0 |

**Root cause (unifying):** units cluster at 4–13, right at `KS_FLOOR=13`, because we sum attackers **flatly** — a
real 3-piece attack (g0, g49) reaches only ~12–13 units and gets floored to 0. SF's `attackersCount × weight`
**product** is super-linear → 3 coordinating attackers explode past threshold. Secondary confirmed gaps:
**battery-blindness** (`attpc` undercounts — rear rook of a battery invisible; `KS_BATTERY` unwired),
**safe-checks = 0 everywhere**, **weak-squares under-detected**, **corner-zone shrunk** (`attsq=2` on a1).

## Coordination product tested IN ACTION — null (the key negative)
`KS_COORD_GATE_MODE=1 KS_COORD_DIVISOR=4` (the mechanism that fixes both over- and under-read in diagnosis):
- **fast_ab 2128 paired D7 games: −8.8 ±17.3 Elo** (null-to-slightly-negative).
- **vs_sf collapse rate: 46/150 vs 48/150 baseline — UNCHANGED.**
- **depth_probe (collapse FENs): d10 = d16 = 8/20 match SF-best** (depth-INVARIANT) → collapses are **eval-bound
  at the leaf**, NOT search-depth-bound. More depth won't fix them; the reached positions are eval'd wrong.

⇒ Coordination alone **helps some positions and breaks others, netting ~0** — the recurring KS pattern. NOT
"static-KS is dead" (an over-conclusion I made and the owner corrected). The detectors ARE the lever; they're
double-edged, and were historically rejected because the breakages (new collapses / low validation) outweighed
the fixes.

## The insight: "properly tuned" = "properly INTEGRATED"
Every detector INDIVIDUALLY has failed for us (no-queens games-null, coordination null, raw safe-checks
STS-regressed). **SF runs them all together and doesn't regress** — the control is the shared machinery:
**typed+saturated safe-checks + a live QUADRATIC curve (a lone signal → ~0, a coordinated pile → large) +
attacker-count gate.** Our floor sits us on the LINEAR segment (quadratic band dead), so a flat detector just
adds and over-fires. The only candidate worth testing in action is the **integrated bundle**, not any single knob.

## Detector status (owner's question: missing / degenerate / off)
- **Coordination product** `KS_COORD_GATE_MODE` — BUILT, off, tunable (`KS_COORD_DIVISOR`).
- **Typed safe-checks** `ENABLE_KS_CHECK_V2` (+`KS_CHK_Q/R/B/N`) — BUILT, off.
- **Corner-zone clamp** `ENABLE_KS_ZONE_CLAMP` — BUILT, off.
- **Weak-value** `KS_WEAK_VAL_MODE`, `ENABLE_KS_WEAK_ATT2` — BUILT, off.
- **No-queen** `KS_NO_QUEEN`/`KS_NQ_SUP` — live/built (games-null tonight).
- **Battery** `KS_BATTERY` — **MISSING (declared but UNWIRED, cpp_bitboard.cpp:5293)** → needs code.
- **Open-file predicate** — WRONG (own-pawns-only, :5511-5519) → needs code.
Not "degenerate individually" — they're gated levers; the DEGENERACY is at the system level (collinear with the
live eval + floor), which is why individually they're marginal.

## THE METHOD (owner's staged program — the plan for the next window)
For a candidate detector/bundle, in order; at every failure, triangulate the regressions vs SF to learn control:
1. **Static** — does it move the 5 collapse FENs' static eval toward SF?
2. **D10** — does our search eval + move-selection also move right (not just static)?
3. **Tune** (may precede) — best candidate via tuning + static-tuning + **D7 regret with cross-validation**
   (seeded-shuffle split, [[low-depth-regret-tuning-method]]).
4. **If worse → triangulate the worsened positions vs SF11/SF15 static — COLLECTIVELY, not one-by-one.** Gather
   the ENTIRE regressed set and study it together; patching individual positions is whack-a-mole (every local fix
   moves the failure elsewhere → net-null forever). Reverse-engineer the SYSTEMATIC BALANCE SF uses to hold them ALL.
5. **SPRT** the candidate; if worse, again analyze how we fixed one set but regressed another while SF holds both.

## ★ THE CORE QUESTION (owner's framing — what this whole effort is really about)
We take inspiration from SF/Ethereal, implement their detectors, and things sometimes get WORSE. They have ALL
these detectors AND balanced them to hold in ~any position. **What are WE getting wrong such that we can only fix
the positions we look at and regress others?** That balance is a GLOBAL property, not per-position — findable only
by studying the regressions COLLECTIVELY + triangulated. Every past KS attempt died exactly here (fixed target,
regressed tail, net-null). Cracking the *balance* — not any single detector — is the goal.

## Immediate next step (for the new window)
Build + test the **integrated candidate**: `KS_COORD_GATE_MODE=1` + `ENABLE_KS_CHECK_V2=1` + lower/smooth
`KS_FLOOR` (so amplified real attacks aren't cut — units were 12–13) + `KS_MIN_ATTACKERS` gate (so calm stays ~0),
as ONE config. Run in action (fast_ab + vs_sf collapse rate). If it fails, triangulate its regressions vs SF.
Stage the **battery wiring** + **open-file predicate** fixes (code, gated, byte-id default) for owner review +
mirror-gate. ⚠️ Overnight/unattended: allowlisted runner subs ONLY (non-allowlisted prompts are auto-DENIED);
`probe_fens`/classifier are fine only when owner is present (SF11 hang risk).

## 2026-08-16 (cont.) — KS_BATTERY WIRED, then found 0-coverage; collapse driver is MATERIAL not KS
**Battery feeder wired (unit 1a).** `KS_BATTERY` was a phantom knob (declared default 3, registered, UNWIRED).
Wired it as a Stage-1 feeder per a fable 3-phase review: a local `(piece, xray_zone_mask)` overlay recovering a
rear R/Q (file/rank) or Q/B (diagonal) battery piece the occupancy-blocked `attack_bitmasks` misses, threaded
into the zone-loop `am` AND the defaware footprint test (fable proved that adding it to `attackers_sq` ALONE
nets 0 under the shipped `KS_DEFAWARE_MODE=1`: foot=0 -> skipped in daware while legacy_att still counts it).
Default flipped 3->0 so the shipped engine stays byte-identical (resolves the phantom-knob hazard). Gates: static
byte-id at default ✓; colour mirror 0 new violations ✓ (`_eval_symmetry.py` identical 11/1.4% at 0 vs 1); recovers
a REAL battery on g49 (White Rf1+Qf2 open f-file -> Black Kg8, `king_safety +1.08`, toward SF). Code: cpp_bitboard.cpp
~5349 (pass), ~5366 (overlay), ~5471 (defaware footprint); knob search_engine.h:1335.
**But coverage = 0/46.** `_batt_coverage.py` (engine-only, KS term at KS_BATTERY 0 vs 1) over the collapse set:
**identical on all 46 FENs — the battery fires on ZERO real collapse decision positions** (g49 was a hand-picked
illustration, not representative; verified the env mechanism works via `_bv0/_bv1` on g49 = 0.00->1.08). Clean-x-ray
major batteries into the king are essentially absent from our collapses. ⇒ battery is CORRECT but not worth a
bundle slot; kept wired+off (hazard resolved), not pursued standalone.
**Collective term triangulation (the real finding).** `_collapse_term_gaps.py` decomposes each collapse's gap vs
SF11 into KS/Material/capgains and tallies the dominant driver (42 fens; 4 in-check skipped): **MATERIAL 26/42 (62%),
KS 12/42 (29%), capgains 4/42 (10%)**; mean|gap| MAT 1.29 / KS 1.06 / CAPG 0.39; KS under-read >1pawn only 8/42.
⇒ **the collapse tail is driven more by MATERIAL reading than by KS.** CAVEATS: vs SF11 STATIC not SF18; static
gap != Elo ([[corpus-fit-is-anti-correlated-with-elo]]); material magnitude is the tapped/anti-correlated lane;
much of the "material gap" is likely our deliberate piece values on imbalanced positions and may be search-absorbed
(as capgains was, [[static-eval-top-culprits-are-search-absorbed]]). A MEASUREMENT of where the divergence sits,
not a prescription to tune material. Next decision (owner): pivot toward material realism, stay on the 12-position
KS-driver subset, or D7-test whether the material gap is even move-changing.

## 2026-08-16 (cont.2) — FULL KS-detector sweep: coverage+direction vs SF11 on the collapse set
`_ks_detector_sweep.py` (driver+subprocess-per-arm; each detector knob vs shipped default; metric = does our
king_safety term move TOWARD SF11's King-safety term on the 42 non-check collapse FENs). mean|SF11 KS|=1.16;
21/42 have |SF KS|>0.5. Columns: moved / mean|d| / towardSF / awaySF / gapReduction(sum |base-sf|-|arm-sf|).

| arm | moved | toward | away | gapRed |
|---|---|---|---|---|
| flank_contest (KS_FLANK_MODE=2) | 22 | 15 | 7 | **+8.26** |
| zone_clamp (ENABLE_KS_ZONE_CLAMP) | 17 | 12 | 5 | +3.62 |
| flank_breadth (MODE=1) | 31 | 16 | 15 | +3.24 (noisy) |
| zone2_wide (KS_ZONE2) | 13 | 8 | 5 | +1.81 |
| weak_val (KS_WEAK_VAL_MODE) | 9 | 6 | 3 | +1.14 |
| pin_mode (KS_PIN_MODE) | 2 | 2 | 0 | +0.42 |
| aim / interact / weak_att2 | 2-3 | ~ | ~ | ~0 |
| coord_gate | 7 | 1 | 6 | **−3.76** (hurts) |
| check_v2 | 5 | 2 | 3 | −1.61 (hurts) |
| sf_weak / min_attackers2 / **battery** | 0 | 0 | 0 | 0 (inert on this set) |

**Integration OVERSHOOTS (the key result).** Individually-toward-SF detectors, stacked additively, blow past SF:
`flank2+zone` 12/19 away −6.23; `+weak_val` 10/23 −13.41; `integrated_all` (flank2+zone+wv+pin+zone2) **8/26 away
−19.67**. **Magnitude conservation recovers it:** `int_all+accum` (KS_ACCUM_MODE, thresh 13) 11/20 −5.92 (still over);
`+KS_ACCUM_THRESH=28` **12/5 +5.34** (net toward); `+KS_ACCUM_LIN=40` 13/4 +1.52; `int_all + KING_SAFETY_MAG=1500`
**20/14 +2.94**. ⇒ quantitative confirmation of "properly integrated = redistributive at CONSTANT magnitude":
detectors on + total scaled to SF's level pulls toward SF; naive additive stacking overshoots. NOTE: on this static
metric the single detector `flank_contest` (+8.26) still beats the conserved integration (+5.34) — static screen
only; GAMES decide. Both flank_contest and int_all+acc_t28 pass colour symmetry (11/1.4%, no new violations).
GAME VALIDATION (fast_ab D7, conc 2, vs baseline):
- **flank_contest (KS_FLANK_MODE=2): −11.6 ±32.7 Elo** (48.3%, 599 games; W +130−110=60 / B +96−136=67). Neutral-
  to-slightly-negative, CI straddles 0 (noise-floor unresolvable). The static-best single KS detector does NOT
  clearly help general play — the recurring "helps static KS metric, ~neutral in games" pattern. NOT a verdict;
  collapse-rate + collateral still pending.
- **conserved integration (int_all+acc_t28): −5.0 ±32.0 Elo** (49.3%, 625 games). Also noise-floor-neutral, but
  marginally better than the single detector (−5.0 vs −11.6) — consistent with "integration > single". Neither KS
  candidate clearly moves general D7 Elo (both CIs straddle 0). Matches the whole KS history (additive 0-for-9,
  coord −8.8): KS detectors help the static metric, stay ~neutral in games.
- vs_sf collapse-rate, conserved integration, 200g: **58/200 = 29.0% collapses, our score 52.8%** vs SF@2400.
  (old baseline ref 48/150=32%/47.3% — unmatched). Matched 200g baseline running (b2ndpzjgh) for clean compare;
  29% vs 32% is within binomial noise (±3%) so not yet resolvable.
- **Collateral triangulation** (`_ks_collateral.py`, per-FEN base/flank/integ/SF KS + toward/away): flank 7 away,
  integ 5 away of 42. The AWAY positions are all the SAME shape — **false-positive OVER-reads**: SF reads the king
  safe (~0 danger), our BASELINE already over-reads (+0.7..+1.4), and the detectors amplify it further (clearest:
  the 3 "both overshoot" cases where SF=0). So KS errors run BOTH ways — under-reads (the collapses) AND over-reads
  (this collateral); the detectors help the first, worsen the second. The accum-threshold integration SUPPRESSES
  several of flank's false positives (5 vs 7; stays 0 where flank overshot) but CAN'T fix the cases where baseline
  already over-reads. ⇒ method answer forming: **SF holds both directions because it gates danger on accurate
  safe/DEFENDED reading (won't fire on a king it knows is defended); our detectors fire on proximity/breadth without
  fully discounting defense.** The next lever is a better safe/defended discriminator, not more detectors.
## 2026-08-17 — SPRT + game-collapse counter-analysis: KS is calibratable but NOT the collapse discriminator
**SPRT** (`gate`, conserved integration `int_all+acc_t28` vs base, conc 4): stopped ~1136 games at **~−20 Elo**,
LLR −1.27 trending to reject. A games-resolved "not an improvement" (was ±32 noise in fast_ab).
**Mined the SPRT games** (`_sprt_collapse_mine.py`, own-eval peak≥+2.0 then not-win): **HURT 191** (integration blew
a win baseline holds) vs **HELPED 165** (integration held a win baseline blows), net **+26** — double-edged and
net-negative, but close (the −20 Elo mechanism, quantified). FENs in `_sprt_hurt.csv` / `_sprt_helped.csv`.
**Counter-analysis** (`_sprt_counter.py`, our KS base vs integ vs SF11 on both sets):
| set | mean\|SF KS\| | \|our\| base | \|our\| integ | (\|integ\|−\|SF\|) |
|---|---|---|---|---|
| HURT (n=164) | 1.97 | 0.45 | 1.90 | **−0.08** (matched) |
| HELPED (n=151) | 1.50 | 0.37 | 1.21 | −0.29 |
**Findings (measurements):** (1) the integration WORKS as KS detection — baseline under-reads hard (~0.4 vs SF
~2.0), integration lifts us to ~SF's magnitude on both sets; the under-read is fixable and this fixes it. (2) But
KS calibration does NOT distinguish blew-it from held-it — both sets ~equally SF-calibrated. So **KS is not the
collapse discriminator**: these are positions we were winning, correctly saw the king danger (matching SF), and
still blew ⇒ decided by conversion/material/tactics, not KS. The −20 Elo is move-selection VARIANCE the KS change
injects (~half the moved positions overshoot SF), not a KS mis-read. (3) Confirms the earlier decomposition at the
GAME level: material 60% / KS 31% of the collapse gap — **the collapse Elo is in the material/conversion lane, not
KS.** ⇒ KS detection lane: correct-but-insufficient; park the integration (−20 Elo, gated off). Redirect to material.
REMAINING: matched collapse-rate (lower priority now); material-lane diagnostics + supervised proposal; handoff.

### Phase C COMPLETED — discriminator IS King safety, in the SHARP tail (corrects the "KS not discriminator" read)
The KS-only counter-analysis (mean magnitude) was misleading; the ALL-TERMS decomposition (`_sprt_term_profile.py`,
integration eval, hurt vs helped vs SF11) finds the axis that separates blew-it from held-it:
| term | SF\|hurt\| | SF\|help\| | dSF | dGap |
|---|---|---|---|---|
| **King safety** | 1.97 | 1.50 | **+0.47** | **−0.43** (biggest) |
| Material | 1.43 | 1.63 | −0.20 | +0.12 |
| Passed | 0.32 | 0.27 | +0.06 | +0.08 |
**Read:** the blown positions are the SHARPER king positions (SF KS 1.97 vs 1.50) and our KS is most wrong there
(dGap −0.43; per-position |disagree| 1.70 hurt vs 1.37 help). ⇒ the integration fixed the MODERATE king positions
(helped: KS matches SF) but NOT the sharp tail (hurt: still off ~0.4 + scatter ±1.7). It raised AVERAGE KS volume
without accuracy in the high-danger tail. Secondary axis: material over-read (+0.12) co-occurring in those same
positions ("quiet king + greedy material"). ⇒ **the balance lever = KS ACCURACY in the sharp-king tail (SF-danger
≈2), not KS volume; + a material co-read.** The −20-Elo integration is louder-everywhere-but-still-inaccurate-where-
it-matters. CAVEAT: signed gap direction muddied by mixed king colours; the robust facts are dGap-KS-biggest and
dSF-KS-positive (hurt = sharper). ⚠️ material term is IMPURE (bundles PST/AST/mobility per owner; value-fit residual
1.08 confirms) — material conclusions need a clean piece-value isolation first.

## Tooling added this session
`_ks_pattern_classify.py` (KS failure-pattern classifier, engine-free), `_ks_phase_split.py` modes
`CAPG_SCALE`/`COORD_DAW`(reverted), `depth_probe` compatible CSV needs `fen_start` column (built
`selfplay/games/vssf_2400/dp_fens.csv`). `probe_fens.py` needs `STOCKFISH_PATH` env set.
