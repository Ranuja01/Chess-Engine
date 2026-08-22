# KS clean-instrument arm sweep — overnight results (2026-08-15)

**What ran.** Autonomous overnight block. Seven `ks_phase` regret-ruler runs (D7, JOBS=4, cached SF18
labels — no live SF, no hang risk) on `game_regret_set.csv` (15k FENs), each an arm vs the shipped bundle,
read on the phase / non-pawn-material / criticality / queen×material splits. Then one `fast_ab` D7 games
confirmation on the best arm. All on the allowlisted runner. No code changed, no commits.

**⚠️ HEADLINE REVISED (later same session) — KS is NOT tapped; the earlier call tested only MAGNITUDE.**
The queenless over-read is a **COORDINATION-DETECTION gap**, and SF's attacker-coordination product
(`KS_COORD_GATE_MODE`, `COORD_DIV=4`) fixes it **3–4× better than any magnitude knob** — see §"Detection lane"
below. The original "tapped" conclusion held only for magnitude/phase/additive-detector knobs; the corrective
*detection* lane (the real SF11/15 deficit) was un-tested when that call was made. Coordination detection is
the validated mechanism; it over-applies to the opening and needs **phase-conditioning** (a code change) — a
live path, not a dead end.

**Original (magnitude-only) conclusion, kept as record:** on the clean instrument the KS single-knob /
restructure MAGNITUDE lane is tapped — NQ_SUPP/EGEXT/ACCUM/DETONLY all marginal-or-worse, NQ_SUPP=20
games-null. True, but it is NOT the whole KS story (detection ≠ magnitude).

---

## Instrument note (a correction I made mid-run)
Run 3 (NQ_SUPP=20) showed 1119 "changed" positions in the 2Q band, where the no-queen suppressor is
statically inert. I initially called this OMP-oversubscription non-determinism and re-ran pinned (3b). **3b
was BYTE-IDENTICAL to 3 → the ruler is DETERMINISTIC; my non-determinism call was wrong.** The real cause:
at D7 the search descends into lines where a queen is traded, so `KS_NO_QUEEN` fires in those queenless
*subtrees* even from a 2Q root. The "2Q inert control" is only inert for a *static* eval; under search it
legitimately leaks. All runs are valid. (Pinning the thread pools changed nothing, so it's optional here.)

## Run 1 — KSMAG_TEST (KS on vs off): KS is net-good, over-read is localized
- Instrument sees KS: decisive for the D7 move in **4190/15000 (28%)** — regret screening is informative.
- KS **helps overall** (ALL −0.25) and in **every phase**; **no endgame hurt** (contamination ghost stays dead).
- KS helps **most where it matters**: CRITICAL −2.30, moderate −0.96.
- Over-read (KS hurts) is confined to: material **npm 13-27** (late-mid +0.48), and by queen×material to
  **queenless** positions — 0Q 13-27 +0.37, 0Q 28+ +0.19, 1Q 13-27 +0.55. **2Q (both queens) KS always
  helps** (−0.35 high-mat, n=3180). ⇒ the #1 confirmed defect (queenless mid-high over-read) reproduces
  cleanly.

## Run 2 — prune-transmission (KS-free prune margins): the over-read splits in two
- **20-27 late-mid over-read is Stage-3 (prune-transmitted):** flips +0.48 → **−0.73** when KS leaves the
  prune margins. → a continuity target.
- **28+ high-material queenless over-read is LEAF:** got *worse* under KS-free margins (0Q 28+ +0.19 →
  **+0.84**). → a no-queen-gate target.
- Guardrails: KS-in-margins is **net-positive** (ALL benefit shrank −0.25 → −0.04 without it → do NOT remove
  KS from pruning). `EVAL_MODE=2` changes the search globally (changed-set 4190→4478), so this is suggestive,
  not a clean isolation. CRITICAL swing on n=25 = noise.

## Arm sweep — no arm improves regret

| arm | knobs | net Δ (ALL) | read |
|---|---|---|---|
| **NQ_SUPP=20** | KS_NO_QUEEN 6→20 | **−0.09** | best; helps the mid-high queenless over-read bands, BUT over-suppresses low-material queenless (0-6 +1.34, 0Q 0-12 +0.62) where KS is genuinely good. Flat suppressor too blunt — the over-read is material-localized, wants material-conditioning. |
| EGEXT=1 | KS_EXTEND_EG | −0.04 | marginal; **hurts endgame +0.29** (additive-KS-in-EG harmful, as before); does NOT fix the 20-27 band (+0.11). |
| ACCUM=1 | threshold + demoted proximity + net no-queen | **+0.05** | net WORSE; helps only 20-27 (−0.15) & 0Q 13-27 (−0.18); proximity removal (KS_ATTACK_COUNT=0) hurts broadly (0-6 +0.80). |
| DETONLY | SQC+pins+weak-val+flank all on | **+0.32** | decisively WORST; over-fires opening (+0.61) and hurts CRITICAL (+2.92). Empirically refutes "all-feeders-on" on the clean ruler (matches/exceeds the 2026-08-12 detector-stack over-fire). |

**The channel law reproduces on the CLEAN instrument:** no KS lever has a clean intrinsic sign. The best
(NQ_SUPP) is marginal and mixed; restructure (ACCUM) and detector-stack (DETONLY) are net-negative.

## fast_ab games confirmation (best arm, D7 A/B)
Arm: `KS_NO_QUEEN=20` vs shipped default. Purpose: confirm the regret-tapped verdict with the ARBITER
(games), so the conclusion rests on games, not regret alone.

**RESULT: NULL.** A (default) vs B (KS_NO_QUEEN=20): **+633 −625 =525 of 1783 games (50.2%), Elo +1.6 ±18.9**
for the default ⇒ NQ_SUPP=20 = **−1.6 ±18.9 Elo, CI includes 0** (1783 games, 120 min, D7, conc 2). No
signal. (Fast-depth compresses ~3× and RANKS only, so the real-TC CI is wider — but the point estimate is
~0, so there is nothing to chase even before compression.)

⇒ **AIRTIGHT: regret marginal (−0.09) AND games null (−1.6 ±18.9). Single-knob no-queen KS is tapped.**

---

## Detection lane — coordination is the validated mechanism (the real finding)
Prompted by the owner ("SF11/15 has advanced detection for over/under-fire; analyze where we failed"), dumped
the 35 worst KS over-fire positions in the queenless 13-27 band. Pattern: **queenless R+minor endgames where
KS pulls us off SF's best move onto a passive king-safety move** (KS-off plays SF-best; KS-on switches to f2f3
/ g2g3 / Kb3). Hypothesis: our **flat proximity + flat attacker-sum** reads pieces *near* an (often active)
endgame king as danger, missing SF's coordination + safe-check gating.

Tested the ablation — SF's attacker-coordination **product** (`KS_COORD_GATE_MODE`, count×weight, super-linear)
replacing our flat sum, vs the shipped default (defaware on):

| band | flat NQ_SUPP=20 | **COORD_DIV=4 (product)** | COORD_DIV=6 (gentler) |
|---|---|---|---|
| 20-27 late-mid | −0.17 | **−0.67** | +0.03 |
| 0Q 13-27 | −0.14 | **−0.45** | −0.20 |
| 1Q 13-27 | −0.63 | **−1.10** | +0.49 |
| 0Q 28+ | −0.06 | **−0.38** | −0.30 |
| ALL (net) | −0.09 | +0.05 | +0.08 |
| opening (collateral) | — | **+0.21 (n=2415)** | +0.19 |

**Findings:** (1) the coordination product fixes the queenless over-read **3–4× better than magnitude** —
confirming the defect is a *detection* gap, not a magnitude gap. (2) `div=4` is the right strength; `div=6`
loses the target fix while keeping the collateral ⇒ the divisor does NOT separate them. (3) The net is still
~0 because the product **over-applies to the OPENING** (+0.21 on 2415 pos) — where a real attack is *forming*
with few coordinated attackers, so the product under-reads it. This collateral is **phase-based, not
divisor-fixable.**

⇒ **The validated next build:** a **phase/material-conditioned coordination gate** — use the product in the
late-middlegame/endgame (where proximity-over-read lives) and the flat sum in the opening (where attacks are
forming). This is a small gated code change and is the single most-justified KS experiment now (supersedes the
material-conditioned no-queen as #1, though that remains a candidate for the leaf 28+ band). Also worth pairing
with typed safe-checks (`ENABLE_KS_CHECK_V2`) which corrects a separate detection inversion.

## What this means for the KS lane
1. **The shipped KS is good** — net-beneficial, especially on critical positions. Don't weaken it.
2. **The queenless mid-high over-read is real but not single-knob-fixable.** Flat no-queen suppression, the
   accumulator restructure, endgame extension, and the full detector stack are all marginal-or-worse. The
   only lever with the *right sign* in the target bands (NQ_SUPP) over-corrects low material.
3. **The one un-tried surgical fix** the data actually motivates: a **material-conditioned no-queen
   suppressor** — fire the no-queen tax ONLY at mid-high material (npm ≥ ~13), leaving low-material queenless
   KS (which helps) untouched. No current knob does exactly this (EGMAT tapers the wrong direction); it needs
   a small gated code change. This is the single most-justified next KS experiment.
4. **All-feeders-on is refuted** (DETONLY +0.32) — do not pursue turning the detector stack on.
5. **Broader read:** the KS lane is at or near its single-knob ceiling on the clean instrument. Further KS
   gains need either the surgical material-conditioned no-queen (worth one build + games test) or accepting
   KS is done and moving the 3-stage-audit method to the next subsystem (threats / passers / OvD).

## Addendum — parked-lever re-screen (leftover engine hours)
Spot-checked the top contamination-suspect parked levers (from KS-PARKED-LEVER-RESCREEN-QUEUE) on the CLEAN
STS (baseline 1771, confirmed same-build):
- `ENABLE_CONT_HIST_2PLY` → **1645 (−126)** — worse than its contaminated −88.
- `ENABLE_IMPROVING` → **1661 (−110)** — worse than its contaminated −37.

Both **confirmed dead** on the clean instrument. Meta-finding: **the contamination MASKED badness, it did not
hide wins** — it made genuinely-bad levers look borderline (which is why they were "parked as marginal"). So the
contamination-reopens hypothesis is NEGATIVE for these two; the remaining queue items are lower-suspect and
likely similar, but un-checked. (STS ±~100 is within the resolvable band, but −110/−126 clears it as clearly
negative.)

## REDIRECT (autonomous, post-KS) — the biggest static culprits are SEARCH-ABSORBED
Owner flagged circling risk on KS. So instead of a 13th KS lever, ran the "where is the lever at all" scan:
- **pvb per-term culprit** (clean STS, n_quiet=723): `king_safety` mean-delta **−0.04** (mildly HELPS the best
  move) — **KS is NOT our biggest static weakness.** The dominant culprit is **`capture_gains` +0.353**
  (with `material` +0.04); worst cases = we keep material where SF sacrifices.
- **capture_gains is MOVE-NEUTRAL in D7 search:** damping `SCALE_CAPTURE_GAINS` to 70% AND to 1% each changed
  **0 / 15000** D7 moves (knob verified live, search_engine.h:693; applied cpp_bitboard.cpp:7495/7821). Mechanism:
  D7 leaves are quiescent (qsearch resolved captures), so `approximate_capture_gains ≈ 0` there — nothing to scale.

**Synthesis:** our two biggest static-eval issues are BOTH absorbed by the search — KS levers net ~0 in
searched regret, and the #1 static culprit (capture_gains) doesn't change a single searched move. This is
[[most-eval-error-is-move-neutral]] at the subsystem level, and it explains the recurring wall: **static-eval
knob tuning has little play-headroom because the D7 search fixes the static errors.** The pvb static ranking
does NOT predict searched-move leverage (its #1 is search-neutral).

**Recommendation:** stop grinding static-eval magnitude knobs (KS, capgains). The remaining headroom is in
SEARCH quality or a stronger eval (NNUE), not eval tuning. Caveat: tested the top culprits (KS thoroughly,
capgains); pvb shows capgains is the ONLY significant static over-read and it's search-neutral, so the static
over-read headroom is genuinely thin — but this is not an exhaustive per-term games proof.

## Caveats
- Regret is a SCREEN; the fast_ab result is the games check. Magnitudes on tiny-n cells (CRITICAL n=6-30) are
  noise — ignore.
- Prune-transmission (run 2) is suggestive, not a clean isolation.
- The fine per-pattern classifier map (battery / open-file / corner-king) was NOT run tonight — it needs the
  non-allowlisted classifier + `sf11_collapse_gap` (live SF11 = hang risk), deferred to an attended session.
