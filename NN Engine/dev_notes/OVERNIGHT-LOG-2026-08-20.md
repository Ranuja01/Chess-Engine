# OVERNIGHT LOG — 2026-08-20 (autonomous game block, conc4, hourly heartbeat)

## ★ PIVOT 2026-08-20 (t+19.5h): DIVERSIFIED OPENINGS — seg1-6 were near-copies (INVALID for generalization)
ROOT CAUSE: the `gate` sub hardcoded `--seed 0` on the small default `openings.txt`, so ALL of seg1-6 replayed
the SAME shuffled opening subset (schedule() shuffles by seed; seed const => same games, only time-noise differs).
So seg1-6 (2000g, elo ~+11.5) is ~ONE segment of independent info, not 2000 games — near-copies, "slightly invalid."
FIX: `gate` sub now takes [openings_file] [seed] (7th/8th args; default openings.txt/0 = legacy). Use
`openings_uho.txt` (1000 diverse lines) + a DIFFERENT seed per segment => each samples a fresh ~200 openings.
seg6 STOPPED (redundant). **REAL VERDICT = seg1 (fixed anchor) + seg7 onward (diversified UHO, varied seeds).**
No gaming window for weeks => keep running day+night, accumulate to a tight CI. seg1-6 = preliminary only.

### DIVERSIFIED VERDICT tally (seg7+ on openings_uho.txt, varied seeds) — THE real result
| segment | tag | seed | W | L | D | games | notes |
|---|---|---|---|---|---|---|---|
| (anchor) | seg1 | 0 | 147 | 149 | 104 | 400 | fixed openings.txt; elo ~-2 |
| 7 | super_seg7 | 7 | 158 | 157 | 85 | 400 | UHO diversified; elo +0.9 = ~NEUTRAL |
| 8 | super_seg8 | 8 | 145 | 157 | 84 | ~400 | UHO diversified; elo ~-11 |

★★ KS VERDICT (diverse openings): seg7 +0.9, seg8 -11 => DIVERSE POOL ~782g elo ~-5 ± 28 = PRACTICAL NULL
(slightly neg). +11.5 fixed-opening did NOT generalize = opening-artifact. Additive/eval-KS 0-for-9 prior HOLDS.
Diversification did its job (caught the artifact). Config stays GATED (correct, Elo-null). DURABLE session wins:
clean gated KS refactor (byte-id/symmetry) + EVAL-MINDMAP + validated diversified-openings method. PIVOTED to
SEE-captures (seecap_seg1, UHO seed1).

★ seg7 (first DIVERSE segment) finished ~0 (+0.9), NOT the fixed-opening +11.5. I over-read seg7's t+21h mid-
value (+12) as "reproduces" -- it settled at 0 (regression-to-mean AGAIN). Diverse evidence so far = ~0, one
noisy segment (±40): does NOT confirm +11.5, doesn't refute. Joint (fixed 2000g + seg7 400g = 2400g): +954 -887
=559 elo ~+9.7 -- but dominated by the DISCOUNTED fixed near-copies; DIVERSE-ONLY = ~0. REVISED: confidence
LOWERED; true value unresolved ~0..+11, first diverse point leans LOW. Need seg8/9 diverse.

## Plan (owner-set 2026-08-20: eval candidate is the priority; search lane can wait)
**Super-candidate KS SPRT vs default — the WHOLE block, deep.** Segmented ~400g/conc4, pooled, run to a FIRM
verdict: decisive SPRT LLR (±2.944), OR a large pooled sample with a CI tight enough to call (win / practical-null).
Goal is CONFIDENCE, not breadth — a confirmed eval result validates the structural-KS + wave-tune approach and
opens the next round of eval ideas. **SEE-captures DEFERRED to another night** (search lane isn't going anywhere).
No commits while owner away.

## Configs
- **Super-candidate** (p1): `ENABLE_KS_RING_GATE=1 KS_MIN_ATTACKERS=2 KS_FLOOR=0 CAPG_KS_DAMP=25 KING_SAFETY_MAG=4000`
  (ring-gate + floor-removed + capgains-damp + magnitude 4000; endgame `ENABLE_KS_UNIFIED` OFF — the wave method
  dropped it). Bench: symmetry CLEAN, STS −44 balanced (neutral), D7 general 2.6078 vs off 2.7003, ≥20 collapse
  wave −0.74. Structural (not additive) — the only KS kind that has ever won.
- **SEE-captures** (p1, secondary): `ENABLE_TT_FLAG_FIX=1 ENABLE_NULL_MATE_CLAMP=1 SEE_PRUNE_CAPTURES=1 SEE_PRUNE_CAPTURE_MARGIN=1000`
  — banked 62-48-44/154g (+32 Elo, CI spans 0). Pool onto that.

## Runner
`gate '<p1cfg>' <label> <tag> <maxg> <elo1> <conc>` — SPRT elo0=0 elo1=5, p2=base=default. Segment = one gate call
(fresh process = OOM mitigation at conc4). Pool W-L-D across segments manually → pooled Elo + CI.

## Monitoring loop (on each poke)
- Segment completion → Read final W-L-D/LLR, add to pooled tally below, decide: SPRT decisive (LLR crossed ±2.944)
  or pooled ≥1200g → verdict → switch to SEE-captures; else launch next segment.
- Heartbeat (sleep 3600) fires → Read current segment interim output, check crash/stall, log, relaunch heartbeat.
- OOM watch: conc4 is the documented reboot risk; segmenting resets memory per process. If a segment stalls with
  no progress or WSL is unresponsive, stop and wait for owner.

## Pooled tally — SUPER-CANDIDATE
| segment | tag | W (p1) | L | D | games | notes |
|---|---|---|---|---|---|---|
| 1 | super_seg1 | 147 | 149 | 104 | 400 | non-decision (cap); elo ~-2 |
| 2 | super_seg2 | 165 | 143 | 92 | 400 | non-decision (cap); elo +19.1 ±40 |
| 3 | super_seg3 | 158 | 138 | 104 | 400 | non-decision (cap); elo +17.4 ±40 |
| 4 | super_seg4 | 162 | 145 | 93 | 400 | non-decision (cap); elo ~+15 |
| 5 | super_seg5 | 164 | 155 | 81 | 400 | non-decision (cap); elo +7.8 ±40 |
| 6 | super_seg6 | — | — | — | running | pooling to 2400g |

POOLED (2000g): +796 -730 =474 → elo ~+11.5 ± 14. Per-seg: -2,+19,+17,+15,+8 (more spread than "+15-19"; seg1
& seg5 lower). CI lb ~-2.5, not yet clearing 0. True value ~+8-13. Need ~3200g for CI to clear 0 if holds ~+11.

POOLED (1600g): +632 -575 =393 → elo ~+12.4 ± 15.5. Per-seg: -2, +19, +17, +15 (seg1 the outlier; last 1200g
all +15-19). Estimate pinned +12-13 over 800g+. CI lb ~-3.5. Pooling seg5->2000g toward CI-clears-0.

POOLED (1200g): +470 -430 =300 → elo ~+11.6 ± 18. Per-seg: -2, +19, +17 (last 800g consistently +17-19).
PERSISTENT SMALL POSITIVE — best structural/additive-KS game result of the whole arc. CI lower bound ~-6 (not
formally clearing 0). D7 bench PREDICTED this (collapse-wave -0.74) = method may be vindicated as a predictor.
⚠️ A ~+12 effect is near the edge of resolvability (needs ~2000g+ for CI to separate); may settle as
"likely small positive, not conclusive." Pooling seg4->1600g. STAY CALIBRATED (over-called clean-null at 400g).

## Pooled tally — SEE-CAPTURES (banked 62-48-44 / 154g)
(pending — starts after super-candidate verdict)

## Events
- launch: super_seg1 started clean (SPRT bounds ±2.944). Heartbeat armed.
- t+1h heartbeat: super_seg1 healthy, no OOM. game ~116/400: +44 -38 =35 (117g), elo ~+18, LLR +0.106.
  Rough start (-100s early) recovered to slightly-positive. ~116 games/hr at conc4. Heartbeat relaunched.
- t+2h heartbeat: healthy. game ~230/400: +87 -80 =63 (230g), elo ~+11, LLR +0.106. Stabilized at a small
  persistent POSITIVE (~+11-16). Notable vs additive-KS 0-for-9 prior. LLR climbs slowly (small effect won't
  cross ±2.944 fast) → will hit 400g cap, pool more. ~115 g/hr. Heartbeat relaunched.
- t+3h heartbeat: healthy. game ~353/400: +129 -126 =97 (352g), elo ~+3, LLR +0.009. ★ Early +11-16 REGRESSED
  toward ~0 as sample grew (the "trust the doubled sample" warning realized). At 352g ~NEUTRAL (~+3). Heading to
  a practical null (consistent w/ additive-KS prior + small-structural expectation). Will pool seg2 for firmer CI.
- seg1 DONE (400g cap, non-decision): +147 -149 =104, elo ~-2, LLR -0.095. seg2 launched (pool to 800g).
- t+4h heartbeat: healthy, no OOM. seg2 ~59g (+20 -21 =18). POOLED 459g: +167 -170 =122, elo ~-2. Practical null
  holding firm. Heartbeat relaunched.
- t+5h heartbeat: healthy. seg2 ~174g (+67 -61 =46, +12 this seg). POOLED 574g: +214 -210 =150, elo ~+2.
  Oscillating around 0 = firm practical null. Heartbeat relaunched.
- t+6h heartbeat: healthy. seg2 ~288g (+18 this seg). POOLED 688g: +262 -249 =177, elo ~+6. NOT a clean null:
  estimate wandering +18->-2->+2->+6, CI ~±24 still includes 0 = small UNRESOLVED positive. Pooling DEEPER
  (seg3->1200g, seg4->1600g) to resolve real-small-gain vs noise. Heartbeat relaunched.
- t+7h heartbeat: healthy. seg2 ~395/400 ending +17 (seg1 was -2 = ~±30 batch noise). POOLED 794g: +308 -291
  =195, elo ~+7-8. Mild positive held for hrs, CI ~±22 still includes 0. Launching seg3->1200g on seg2 cap.
- seg2 DONE: +165 -143 =92 (elo +19.1 ±40, inconclusive). POOLED 800g elo ~+8.7. seg3 launched->1200g.
- t+8h heartbeat: healthy, no OOM. seg3 ~99g (+21 this seg). POOLED 899g: +354 -328 =217, elo ~+10 ± 22.
  Mild positive FIRMING (+8.7->+10), consistently positive ~500g. CI still includes 0. Heartbeat relaunched.
- CONC VERIFY (owner flagged low games/hr): both seg headers show concurrency=4 (confirmed). Deficit vs
  expected 150-170/hr is SF DRAW-ADJUDICATION (per-thread SF confirm @0.5s; log full of adjudicated draws) =
  4 workers + 4 SF adjudicators contending on 4 cores. It IS 4-core; adjudication overhead, not lost cores.
  Can't run top/pgrep unattended to confirm OS-level. Not changing adjudication mid-run (segment consistency).
- t+9h heartbeat: healthy. seg3 ~216g cooled +54->+44 (hot streak partial-regress as cautioned). POOLED 1012g:
  +406 -359 =247, elo ~+16 ± 20. Mild-moderate positive held 1000g+; CI lower bound ~-4 (edge of clearing 0).
  seg4->1600g next to settle. Heartbeat relaunched.
- t+10h heartbeat: healthy. seg3 ~307g cooled +44->+22. POOLED 1107g: +438 -399 =270, elo ~+12 ± 19. Estimate
  easing +16->+12 as seg3 hot streak regresses. Settling ~+10-12 = real-looking small positive but may never
  cleanly clear 0 at these N (a +12 needs ~2000g+ to separate; sits below the ~20-Elo reliably-resolvable line).
  seg3->1200g soon, then seg4. Heartbeat relaunched.
- seg3 DONE: +158 -138 =104 (elo +17.4 ±40). POOLED 1200g: +470 -430 =300, elo ~+11.6 ± 18. seg4 launched->1600g.
- t+11h heartbeat: healthy, no OOM. seg4 ~46g (neutral start). POOLED 1244g: +488 -447 =309, elo ~+11.5 ± 18.
  Holding the small positive. Heartbeat relaunched.
- t+12h heartbeat: healthy. seg4 ~154g (+16 this seg, consistent w/ seg2/3). POOLED 1354g: +534 -486 =334,
  elo ~+12.3 ± 17. Firming: every post-seg1 batch +16-19; seg1 (-2) is the outlier. CI lower bound ~-5.
- OWNER (t+12.5h): away/remote, no gaming-window contention -> keep running into the day. RE-STATED 4-CORE RULE
  (do NOT run 2x 4-core at once). Compliant: only seg4 + sleep-heartbeat; segments strictly sequential.
- t+13h heartbeat: healthy. seg4 ~270g (+22 this seg). POOLED 1470g: +583 -526 =361, elo ~+13.5 ± 16.5. CI
  lower bound ~-3, closing on 0. Plan: keep pooling to ~3000-3200g for CI to clear 0. Then SEE-captures (pending
  owner ordering). Heartbeat relaunched.
- seg4 DONE: +162 -145 =93 (elo ~+15). POOLED 1600g: +632 -575 =393, elo ~+12.4 ± 15.5. seg5 launched->2000g.
- t+15h heartbeat: healthy. seg5 ~97g (slight -4 start, early noise). POOLED 1697g: +669 -613 =415, elo ~+11.5
  ± 15. Holding +11-12. Heartbeat relaunched.
- t+16h heartbeat: healthy. seg5 ~212g running +36 (hot, may cool). POOLED 1812g: +725 -646 =441, elo ~+14.5
  ± 15. CI lower bound ~-0.5 = at the edge of clearing 0. Heartbeat relaunched.
- t+17h heartbeat: healthy. seg5 ~333g settled +18 (cooled from +36, in line w/ others). POOLED 1933g: +772
  -698 =463, elo ~+12.8 ± 14.3. Per-seg: -2,+19,+17,+15,+18. CI lb ~-1.5. seg6->2400g next for CI to clear 0.
- seg5 DONE: +164 -155 =81 (elo +7.8). POOLED 2000g: +796 -730 =474, elo ~+11.5 ± 14. ⚠️ OWNER FLAG: per-seg
  values DECLINING (seg2-5: +19,+17,+15,+8) = watch for regression. Counter: pooled plateaued ~+12 (not sliding
  to 0), seg1 was LOWEST at start (not inflated-start pattern), decline within ±40 batch noise (4% monotone-by-
  chance). seg6/seg7 = the tell. seg6 launched->2400g.
- t+18h heartbeat: healthy. seg6 ~45g (too early, ~0-+8, noise). POOLED 2045g: +815 -748 =482, elo ~+11.4 ± 14.
  Need seg6 ~150g+ to read slide-vs-stabilize. Heartbeat relaunched.
- t+19h heartbeat: ⚠️ seg6 ~167g running NEGATIVE (-4). Per-seg slide continues+crosses 0: +19,+17,+15,+8,-4.
  POOLED 2167g: +860 -796 =511, elo ~+10.5 ± 13.5 (eased). Regression concern now REAL (5 batches monotone-down,
  latest negative = beyond noise-coincidence). Counter: seg6 mid-run (±20 noise), pooled still +10.5 holding.
  REVISED read: effect likely SMALLER than +12 peak, ~+6-10, possibly regressing to 0. seg6 finish = the tell.
- ROOT-CAUSE FOUND (owner): seg1-6 all same-seed same-openings = near-copies. The "declining trend" was likely
  time-noise on fixed positions + partial load confound, NOT a clean property. PIVOTED to diversified (see top).
- seg6 STOPPED. seg7 (UHO seed7) launched, verified loading, transition OOM-safe.
- t+20h heartbeat: healthy, no OOM. seg7 ~77g running +41 (too early, ±40 noise). ENCOURAGING: effect shows
  POSITIVE on DIFFERENT openings too (not just fixed-set artifact). Read magnitude at ~150g+. Heartbeat relaunched.
- t+21h heartbeat: healthy. seg7 ~195g settled +41->+12 = MATCHES fixed-opening ~+11.5 on a DIFFERENT opening
  set (reproduces => real, not fixed-position artifact). Diversified primary: seg7 +12/195g. seg8 (seed8) next.
- t+22h heartbeat: healthy. seg7 ~312g eased +12->+7. JOINT (all, 2312g): +921 -849 =542, elo ~+9.9. Diverse
  alone (seg7 +7) a touch softer than fixed (+11.5) but positive+within-noise. seg7 caps soon -> seg8 seed8.
- 3 RESEARCH AGENTS done -> EVAL-MINDMAP-2026-08-20.md written (failure taxonomy, ring-gate thread, toolkit,
  forward roadmap: dynamic-conditional-eval framework already scaffolded; capgains>passers>rook-file next).
- seg7 DONE diverse: +0.9 = ~NEUTRAL (over-read t+21h +12 as "reproduces"; settled 0). Confidence LOWERED.
- t+24h heartbeat: healthy. seg8 ~129g also ~+3 (near neutral). DIVERSE-ONLY pool (seg7+seg8, 529g): ~+1.3.
  ⇒ fixed-opening +11.5 looks like an OPENING-SPECIFIC ARTIFACT, not a general gain. Plan: finish seg8 (800g
  diverse); if still ~0-2, call KS a PRACTICAL NULL on diverse openings + pivot night to SEE-captures.
- t+25h heartbeat: ★ seg8 SWUNG NEGATIVE: ~-23 at 255g. DIVERSE pool (seg7 +0.9, seg8 -23; 655g): ~-8.5 = NULL-
  TO-NEGATIVE. VERDICT SHAPING: KS candidate does NOT convert on diverse openings; +11.5 was opening-artifact.
  Additive/eval-KS 0-for-9 prior REASSERTS (diversification caught what fixed openings hid). Finish seg8 for a
  clean 800g record, then PIVOT to SEE-captures. Durable wins stand: clean refactor + mind-map + method.
- seg8 DONE: ~+145 -157 =84 elo ~-11. KS DIVERSE POOL 782g ~-5 ±28 = PRACTICAL NULL. Pivoted -> SEE-captures.
- t+27h heartbeat: healthy. seecap_seg1 (SEE-captures, UHO seed1) ~82g running +30 (NOT reading it - ±40 noise;
  consistent w/ banked +32/154g lean, unresolved). Judge on closed diverse segments. Heartbeat relaunched.
- t+28h heartbeat: healthy. seecap_seg1 ~208g swung +30->-17 (same early-pos-then-regress as KS; glad I didn't
  read +30). One partial segment (±25), not a verdict. THEME: on diverse openings neither candidate reproduces
  its earlier positive. Let it close + run more seeds. Heartbeat relaunched.
- t+29h heartbeat: healthy. seecap_seg1 ~318g recovered -17->-2 = ~NEUTRAL (not the banked +32). Both candidates
  ~0 on diverse openings. Caps soon -> seecap_seg2 seed2. Heartbeat relaunched.
- seecap_seg1 DONE (diverse): +159 -155 =86, elo +3.5 ±40 = NEUTRAL (banked fixed = +32). seecap_seg2 (seed2) launched.
- t+31h heartbeat: healthy. seecap_seg2 ~153g running HOT +64 (past noise but 1 unfinished seg; NOT calling it).
  SEE diverse: seg1 +3.5, seg2 +64(153g) = big swing; pool ~553g ~+20 (dominated by unfinished seg2). SEE looks
  MORE alive than KS (KS was 0-to-neg) but unresolved ~0..+20. Wait for seg2 close + more seeds.
- t+32h heartbeat: healthy. seecap_seg2 ~271g cooled +64->+35 (still solidly +). SEE diverse pool (seg1 +3.5,
  seg2 +35; 671g) ~+16. POSITIVE LEAN, clearly better than KS (~-5), but CI ~±30 includes 0 + seg2 cooling +
  wide spread => not confirmed. seg2 caps ~1h -> seed3.
- seecap_seg2 DONE: +169 -148 =83, elo +18.3. SEE DIVERSE POOL (seg1 +3.5, seg2 +18.3; 800g): +328 -303 =169
  elo ~+11 ±24. BOTH diverse segments POSITIVE. vs KS diverse (~-5, bracketed 0). SEE = the live candidate;
  positive lean, CI still includes 0 (lb ~-13). seg3 (seed3) launched to firm.

## ★★★ META-FINDING (both candidates, 2026-08-21): FIXED OPENINGS gave OPTIMISTIC game reads
KS: fixed +11.5 -> diverse ~-5.  SEE-captures: fixed(banked) +32 -> diverse +3.5.  BOTH candidates' fixed-opening
positive VANISHED on the diverse UHO book. Hypothesis (2 cases, suggestive not proven): the small default
`openings.txt` (seed-0 fixed set) is NON-REPRESENTATIVE, likely skewed toward sharp/tactical lines where eval+
search tweaks over-help -> systematic optimism for candidate configs. ⇒ (1) DIVERSIFIED openings (UHO+varied
seed) are now MANDATORY for any game test; (2) RE-EXAMINE past "banked positive" reads that used fixed openings
(they may be inflated); (3) the +20.8 / +45 / +36.7 SHIPPED bundles used tournament A/B — check whether those
also used the fixed set (if so, re-confirm on diverse before trusting the magnitudes). Strongest methodological
takeaway of the night.
