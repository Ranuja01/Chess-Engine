# Eval v2 — rebuild log

**What this file is.** The running record of the ground-up eval rebuild. `OPTIMIZATION_LOG.md` remains the
record of the SHIPPED engine and is not touched by this lane; if v2 eventually wins it merges there as a
single entry with this log as its provenance.

★ **The valuable half of this log is the inverse of an optimization log: what was deliberately LEFT OUT,
and why.** An optimization log answers "why is this here?". After three years the question that actually
bites is "why did we put this back?" — and the reason v1 accumulated ~30 names over ~2 signals is that no
one ever wrote down the second answer. The **Register of omissions** below is therefore not an appendix.

---

## Rules this lane runs under

1. **v1 is frozen and byte-identical.** It is the control arm in the same binary, selected by
   `Config::EVAL_ARM=0`. Any change that moves `250 / 35,310,778 / EBF 3.784 / STS 1796` is a defect in the
   scaffolding, not a result.
2. **One feature per rung, a priori constants first.** Measure untuned → tune the feature's own constants →
   (☠️ red flag) retune neighbours. A feature that only pays after its neighbours are refitted is the
   fitted-around-neighbours signature we are rebuilding to escape, and is recorded as such, never as a win.
3. **Each rung is read against the previous rung** — candidate-vs-candidate, which is null-independent.
4. **Record regret AND NPS at every rung.** The curve is two-dimensional from the start; v1's costs are
   unattributable and that is exactly how a term buying 0.3pp for 8% NPS survives unnoticed.
5. **Colour symmetry is a ship gate at every rung**, not at the end. The historic failure mode is a
   sign/colour defect that still produces entirely plausible numbers.
6. **Detector and transformation are tested separately.** "Does it fire on the right positions" is a cheap
   classification question with no search and no null band; "does firing produce the right ordering" is the
   regret question. v1 fused them and could answer neither.

---

## 2026-09-11 — Step zero: scaffolding ✅ ACCEPTED (all 7 criteria passed)

**Built.** `eval_v2.cpp` / `eval_v2.h` — rung 0 (material + piece-square tables), a shared compute-once
`V2Context`, the shadow-arm delta recorder, and bit-tagged breakdown publication.

**Seam changes to existing files** (no eval arithmetic touched):
- `cpp_bitboard.cpp` — `placement_and_piece_eval` became a thin arm dispatcher; the existing 1442-line body
  was renamed `placement_and_piece_eval_v1` with its text otherwise unchanged.
- `search_engine.h` / `.cpp` — `Config::EVAL_ARM` (0=v1, 1=v2, 2=shadow) and `EVAL_V2_RUNG`, registered in
  the once-guard, echoed at the FRONT of the `[toggles]` dump, with a loud range check.
- `search_engine.cpp` — `eval_cache_key()` and its three call sites.
- `cpp_bitboard.h` — `arm` / `terms_valid` appended to `EvalBreakdown`, plus the `EB_*` bit enum.
- `ChessAI.pyx` — bit-filtered `ev_breakdown` dict, one-shot arm banners on both static doors.
- `setupAI.py` — `eval_v2.cpp` added to sources.

**Three decisions inside step zero that are load-bearing, with their reasons:**

1. ★ **Dispatch at the top of `placement_and_piece_eval`, not at `get_board_evaluation`.** The oracle
   intercepts at the latter, so it was the obvious site. But ~172 files reach the eval through
   `ChessAI.ev` / `ev_breakdown` rather than through search. Gating at the search seam would have left
   every one of them measuring v1 while reporting "no change" — a silent false pass on the exact tools that
   judge the rebuild.
2. ★ **`EVAL_ARM=2` (SHADOW) exists as a purity test, not a convenience.** v2 is required to write no
   global; the compiler cannot enforce that because `cpp_bitboard.h` exposes all of v1's. Shadow runs both
   evals in the same node and returns v1's value, so any write v1 reads moves v1's own result and the bench
   falls off 35,310,778. That divergence IS the test — never "fix" it by reordering the two calls.
3. ☠️ **`eval_cache_key` shipped BEFORE any rung reads castling rights, deliberately.** `evalCacheNew` is
   keyed on the zobrist alone and never cleared, and `generateZobristHash` hashes pieces + side-to-move
   only — no castling, no en passant. Sound for v1, which reads neither. The moment v2 reads the castling
   rights plumbed in on 09-09, two genuinely different positions collide on a full-key match and return
   each other's eval: no crash, no warning, position-dependent wrong numbers. Fixing it later would mean
   every measurement taken in between is suspect.

### Acceptance record

| step | result |
|---|---|
| `pre0` control (pre-change `.so`) | 250 / 35,310,778 · NODES_STABLE yes · MAX_NPS **369,901** (median 364,404, spread 3.2%) |
| build | ✅ clean, `eval_v2.o` in the link line |
| **arm 0 byte-identity** | ✅ **250 / 35,310,778 / EBF 3.784 / STS 1796** — all four exact |
| arm 2 purity (first attempt) | ☠️ **FAILED** 249 / 35,328,099, reproduced exactly on re-run |
| value-level purity | ✅ **0 divergent / 1,500 positions** — v2 writes no global v1 reads |
| **arm 2 purity (after fix)** | ✅ **250 / 35,310,778 / EBF 3.784**, cutoff histogram identical to arm 0 bucket-for-bucket ⇒ v2 runs in every one of 35M nodes and the search is bit-for-bit baseline |
| arm 1 reachable | ✅ **238 / 66,189,654 / EBF 4.068** — clearly different, knob took effect |
| colour symmetry, arm 1 | ✅ **0 violations / 800** at TOL=0 |
| file mirror, arm 1 | 21/651 at 5mp — ⚠️ inherited, see below |
| **NPS, arm 0** | ✅ MAX **380,422** vs pre-change control **369,901** (median 366,370 vs 364,404), NODES_STABLE yes ⇒ **no regression**. ⚠️ Read as neutral, NOT as a speedup: +2.8% sits inside the run-to-run spread (3.2% then, 6.3% now) and no mechanism makes v1 faster by adding a TU |
| `[toggles]` echo | ✅ `EVAL_ARM=` leads the dump on both arms; no `☠️ invalid` line |

★ **STEP ZERO ACCEPTED — all seven criteria pass.** v1 is frozen and byte-identical on four fingerprints;
v2 is reachable, colour-symmetric, and provably cannot perturb v1 even when run in all 35M nodes.

### Reading arm 1's first numbers
238/300 from material + PST alone, against the full eval's 250, is a high floor — but ⚠️ **WAC is a
TACTICAL suite and therefore weakly eval-sensitive**; STS is the honest read and is not yet taken. The
node count is the more informative half: **+87% nodes and EBF 3.784 → 4.068**. That is the pruning-headroom
coupling arriving exactly where the plan predicted it — `RFP_MARGIN`, `FUTILITY_MARGINS` and the razor
constants are absolute millipawns fitted to v1's spread, and a thin eval crosses them far less often.
⇒ Confirms the rule: **hold margins fixed across rungs**, re-sweep only at checkpoints, and never read an
early rung's node count as an efficiency result.

### Two further corrections from the ship-gate run
1. ☠️ **Appending `arm` / `terms_valid` to `EvalBreakdown` broke the `TERMS=1` table.** The symmetry
   harness treated them as eval terms and summed |x + y| over them — `terms_available` is a BIT MASK and
   read 2.4e15, swamping the table. Both added to `SKIP_TERMS` in `diagnostics/_eval_symmetry.py`.
   ★ General lesson: adding a field to a widely-read struct silently enrols it in every consumer's
   iteration. ~172 files read this one.
2. ⚠️ **My own acceptance criterion was unimplementable as written.** I required `TERMS=1` to "show three
   named rows at 0" as a guard against a vacuous symmetry pass — but the table only prints terms with
   NON-ZERO asymmetry (`if d:`), so a clean pass necessarily shows nothing. The guard's *intent* is still
   satisfied: colour swap on a material+PST eval passes only if `whitePlacementLayer` and
   `blackPlacementLayer` are exact mirrors, which is real content, not a tautology. Criterion restated
   rather than claimed as passed.

### ⚠️ Inherited PST asymmetry — a rebuild decision, not a defect
The file-mirror violations (21/651, all 5mp) come from the **queen PST being file-asymmetric by default**:
`QUEEN_PST_FILE_SYM_MODE = 0` leaves the tables untouched, and the code comment states the choice "is a
tuning question rather than a correctness one." v2 rung 0 inherits it because it reads the shared
placement layers.
⇒ ★ Rung 0 currently uses **v1's PSTs verbatim**. That is convenient for bring-up and is exactly the kind
of inheritance the rebuild exists to avoid carrying by accident. Decide on purpose, at the PST rung,
whether v2 owns its own tables.

### ☠️ The step-zero defect, and what it taught

The shadow arm failed while v2 was provably pure. Both were true, and reconciling them found a
**pre-existing hazard in v1 that the new arm merely exposed**:

★ **The eval's global side effects are load-bearing, and an eval-cache HIT skips them.**
`placement_and_piece_eval_v1` republishes the file-scope board state (`pawns`, `occupied`,
`pieceTypeLookUp`, `attack_bitmasks`, `square_values`, pressure/support, `central_score`) on every call;
`get_board_evaluation` returns early on a cache hit and never runs it, so downstream readers see the
PREVIOUS position's globals. ⇒ Anything that perturbs the cache hit/miss pattern changes search behaviour
**deterministically, with every eval VALUE identical.** My arm-aware cache key did exactly that.

⚠️ **Two of my own errors, recorded because they are the reusable part:**
1. I concluded the key change was "definitively not the cause" by reasoning that a full-key compare plus a
   pure eval means a hit always returns the correct value. Every step true; conclusion false — it reasons
   only about the RETURN VALUE. **When a function has side effects, "value unchanged" is not "behaviour
   unchanged."**
2. The purity test itself was badly designed: `wac` measures a SEARCH, and arm 2 runs two evals, so the
   test conflated "v2 is impure" with "arm 2 is slow" — the two hypotheses it existed to separate. A test
   that can fail for more than one reason is not a test. Replaced by
   `diagnostics/_eval_arm_purity.py`, which compares VALUES with no search and no clock.

**Fixes:** arm 2 now shares v1's plain-zobrist key (an arm that must reproduce v1's *behaviour* must
reproduce v1's *key*, not merely its value); the arm-2 banner no longer claims `ev` returned v2 when it
returned v1; the purity script now states that a clean value result does not license inferring a bench
divergence is a speed effect.
⇒ Memory: [[eval-global-side-effects-are-skipped-by-a-cache-hit]]. ✅ This also converts v2's zero-global
design from tidiness into a structural requirement.

**Acceptance still owed:** arm 1 reachability → colour symmetry both arms incl. `TERMS=1` → NPS both arms.

⚠️ **Open item:** the `build` sub's `touch` list in `selfplay/overnight_runner.sh` still needs
`eval_v2.cpp eval_v2.h` added. It was applied and immediately reverted because the ablation screen was in
flight and bash reads scripts by byte offset — the edit lengthened a line ahead of the executing position.
Re-apply once the screen finishes. (`build_profile` carries the same list.)

---

## Register of omissions — what v2 deliberately does NOT contain

| omitted | why | revisit as |
|---|---|---|
| **Heat map / attacking layer** | A 5-in-1 doing king attack, king defence, central control, mobility/space and xrays through one per-piece sweep. No strong engine has one; they all have a dedicated subsystem per job. Our own measurement supports this rather than opposing it: the three consumer channels read as NOT collinear (max r=0.42), which is what one mechanism doing three different jobs looks like. ⚠️ It measured **1.3pp** and was load-bearing — this is a decomposition, not a dismissal | A later rung, ADDED on top of dedicated KS/central/mobility/space. If it still pays, it carried something they miss |
| **Capture gains** | Owner's invention; no giant computes a static SEE pre-booking of pending exchanges, and it is expensive. ★ Standing puzzle: with capgains working we should rarely be sitting in the non-quiet positions where it is load-bearing — which points at qsearch leaving tension unresolved, i.e. a SEARCH defect wearing an eval costume | Building without it converts the puzzle into a result. If v2 reaches parity absent capgains, it was compensating for something else |
| **OvD** | Drifted into being a second king-safety term and fought the first — the clearest single instance of the ~30-names/~2-signals problem | ★ The concept it reached for is real and distinct: LONG-TERM prophylactic pressure (an oncoming storm making castling to that side bad) vs KS's IMMEDIATE pressure. Late rung, designed to be harmonious with KS |
| **Central pawn-chain bonus** | ⚠️ Candidate double-count of central scoring. SF11's connected-pawn bonus is keyed on **rank alone** — `Connected[] = {0,7,8,12,29,48,86}` — with structural modifiers only (phalanx, opposed, support count) and no file term anywhere | Decide by changed-set overlap against central scoring before building it |
| **`get_latent_threat_score`** | Already dead in v1: unreachable since `ENABLE_KS_REPLACE_LT` shipped on. ✅ Confirmed empirically 2026-09-11 — the `ab_latent` screen arm (`SCALE_LATENT_THREAT=0`) changed **0 moves in 5,000 positions** | Not at all. KS replaced it |
| **Floodfill mobility** | Expensive, and mobility's value is in the exclusion set rather than the reach | Cheap mobility feeding detectors, rung TBD |
| v1's 44-field breakdown taxonomy | Forcing v2 to fill it would pre-commit the new eval to v1's decomposition — the thing being escaped | Per-rung, publish what is genuinely computed and set the matching `EB_*` bit |

---

## Carried forward as priors, not as code

`EVAL-V2-RUNG-PRIORS.md` holds the reference constants each rung starts from, with the phase-dependent unit
conversion (SF pawn = 128 mg / 213 eg vs our flat 1000 ⇒ **×7.81 mg, ×4.69 eg, never one factor**).
Three findings there are **shape** rather than magnitude differences, which matters because ~15 SF ports
have read null here and every one of them ported a NUMBER into our existing shape:
1. SF's king-attack weights are **inverted** relative to ours — knight 81 / bishop 52 / rook 44 / queen 10,
   against our knight 2 / bishop 2 / rook 3 / queen 5.
2. SF mobility is an **exclusion set** (drop our blocked and back-rank pawns, our king and queen squares,
   our pinned blockers, and every square enemy pawns control), and its table is strongly NEGATIVE at low
   mobility — a trapped knight is −484 millipawns. Ours has no exclusion set.
3. SF space is **quadratic in piece count**, restricted to the four centre files, and has a **zero endgame
   leg**.

---

## Measurement policy for the ladder (settled 2026-09-11)

**Cheap instruments may REJECT, never PROMOTE. Games decide. Games run at CHECKPOINTS, not per rung.**

### Why not games per rung — the same arithmetic that closed term-at-a-time screening
Games harness null is **+4.6 Elo**; resolution is **±25 at 1,200 games, ±10.7 at 4,000**; the true spread
between real candidates is only **~16 Elo**. ⇒ A single rung is very likely **unresolvable by games**, so
playing them per rung spends nights to learn nothing AND stretches the project into months.
⇒ Accumulate 3-4 rungs, then play the checkpoint against the PREVIOUS checkpoint. A bundle is resolvable
where one rung is not. ★ And v2-vs-v1 is a whole-eval difference — comfortably inside games' resolution,
which is the ultimate test anyway.

### Cost, and why the node budget is also a SENSITIVITY lever
At ~380k NPS, a game costs about `2 x moves x nodes_per_move / NPS`:

| nodes/move | ~sec/game | 4,000 games @ conc 4 |
|---|---|---|
| 200k | ~84 | ~23 h |
| **50k** | **~21** | **~6 h = one night** |

★ Low budgets are not merely cheaper: with less search to compensate, **eval error shows up more**, so they
are a MORE sensitive instrument for eval work. ⚠️ Risk is that a low-budget result may not transfer to deep
play ⇒ `node-limit-equal-work-is-step-shaped`: always **two budgets**. Signs agree = it transfers; signs
flip = the change is search-depth-dependent, which is itself the finding.

### The bigger saving is SPRT, not volume
`gate` stops as soon as the evidence is conclusive either way, so a checkpoint carrying a real effect can
resolve in well under 1,000 games. Fixed-N is only for when we want a MEASUREMENT rather than a DECISION.
☠️ `gate` hardcodes `--seed 0` — `openings_uho.txt` + a VARIED seed is mandatory or the read is inflated.

⇒ **Whole-ladder game budget: ~4 checkpoints x 1-2 nights = 4-8 nights**, on nights the owner is not gaming.
⚠️ These are ESTIMATES, not measurements — the runtime estimate for the overnight screen was wrong by 3.5x
(8 min/arm predicted, 28 measured). **Calibrate with a pilot before committing a night.**

### The one exception: rungs whose signature the cheap instruments provably cannot see
★ **King safety (rung 1) goes to games regardless of cost.** Evidence: `MOD_KS_REALIZ=0` ablates a term that
SHIPPED at +36.7 Elo and reads **−1.1pp against a 1.3pp spread — invisible** — because KS's signature is a
TAIL and the d7 gate measures a MEAN. Screening KS on the gate would produce a confident WRONG answer
rather than no answer, which is worse than spending the night.

### ⚠️ What the regret gate does and does not lack
It is NOT an unrealistic-position problem: `game_regret_set` is sampled from ~2,500 real self-play games at
`PLY_STRIDE=7`, spanning genuine opening/midgame/endgame positions. What it lacks is **compounding** — it
scores one move in isolation and never sees decisions accumulate across a game, which is exactly how a thin
eval fails (drift, not blunder). It is also d7, i.e. shallow. ⇒ Use it for move-quality screening; never
expect it to price positional drift.

---

## 2026-09-11 — Rung 1 (KS-A): first result, and the first SHAPE evidence

**Built:** 6 of SF's 14 `kingDanger` components (zone with b..g file clamp · attacker count x weight ·
weak squares in ring · king-adjacent attacks · safe checks by type traced through our own queen ·
no-queen suppressor). 7 deferred with named reasons; 1 EXCLUDED because we already refuted it
(mobility -> kingDanger, null across 4x including cranked).

### Gates (all before any score was read)
| check | result |
|---|---|
| arm 0 byte-identity | ✅ 250 / 35,310,778 / EBF 3.784 |
| `KS_V2_MAX=0` = rung 0.5 identity | ✅ 238 / 66,189,654, twice, across a rebuild |
| colour symmetry, KS live | ✅ **0 violations / 800 at TOL=0** |
| file mirror | ✅ unchanged from pre-KS (21/651 @ 5mp) ⇒ KS-A adds ZERO file asymmetry |

### Results
| variant | STS | vs rung 0.5 (1364) |
|---|---|---|
| **ours (queen-high) + PRODUCT** | **1512** | **+148** |
| knight-high + product | 1466 | +102 |
| ours + legacy SUM | 1398 | +34 |

⇒ **KS-A recovers 148 of the 432-point gap to v1 — ~34% — from ONE subsystem, at a priori constants,
untuned.** On the escalation ladder that is the best available outcome: it pays before tuning, so the
feature carries signal and tuning is upside rather than rescue.

### ★★ THE FINDING: product vs sum is worth +114, and may explain 0-for-11
Same subsystem, same weights, same constants, only the attacker-term SHAPE changed:
**sum = +34 (inside the ~70 chaotic floor) · product = +148.** The sum captures ~23% of what the product does.

⇒ **Candidate explanation for the entire additive-KS failure record.** v1 SUMS attacker weights, so a lone
piece bearing on the king zone already scores full value — the shape cannot express coordination. Every
additive KS improvement attempted for months was layered onto that shape, where no amount of tuning could
have surfaced. ★ This is the rebuild's founding premise (our failures are SHAPE, not magnitude) receiving
its first direct evidence.

⚠️ **Bounds, stated:** STS only, one corpus, one constant setting, and corpus fit is anti-correlated with
Elo in our record. A hypothesis with support, not a proof. Games decide — and KS is the named rung where
the cheap instruments are known blind (`MOD_KS_REALIZ` ablates a +36.7 Elo term and reads invisible on the
d7 gate), so it goes to node-limited games rather than further screening.

### Predictions scorecard (registered before any run)
1. `> 100 STS` — ✅ **+148**
2. gate reads KS-A null — ⏳ not yet run
3. knight-high within noise of ours — ✅ **−46, inside the floor**
4. some constant reads flat over a 2x sweep — ⏳ not yet swept

### ☠️ Error caught before it could mislead
The a priori weights mixed TWO UNIT SCALES: v1's raw `{N2,B2,R3,Q5}` live where `KS_FLOOR=13`/`KS_CAP=80`
(units 0..80), while every other constant here is SF-scale (`WEAK 185`, checks 635-1080, units 0..2000+).
Left as-is, profile 0's attacker channel would have contributed ~14 units against ~780 from one safe check —
**effectively switched off** — and profile 1 would have "won" for reasons unrelated to shape, confounding
the one comparison the rung exists to make. Fixed by rescaling ours to SF magnitude preserving the ratio
exactly (2:2:3:5 x 187/12 = 31:31:47:78), so both profiles are compared at equal total weight.
★ Found by predicting each channel's magnitude before running, not by a test. Second time today that habit
caught a defect that would have produced a plausible, wrong number (the first was the eval-cache side
effects). ⇒ **Predict the magnitude of every new channel before measuring it.**

---

## 2026-09-11 (evening) — Rung 1 KS-A: the full record

⚠️ **Numbers here are v2-lane preliminaries. `OPTIMIZATION_LOG.md` stays canonical and gets an entry ONLY
if v2 beats v1 in games.** Nothing below has been near a game.

### Final configuration
```
KS_V2_MAX=4000  KS_V2_HALF=600  KS_V2_ONSET=450  KS_V2_COORD=256
KS_V2_WEAK=57   KS_V2_ADJ=61    KS_V2_NO_QUEEN=321
KS_V2_CHK_Q=126 KS_V2_CHK_R=122 KS_V2_CHK_B=80  KS_V2_CHK_N=152
KS_V2_ZONE_SF=1 KS_V2_XRAY=1    KS_V2_PAWN_ATT=0
KS_V2_ATT_PROFILE=0  KS_V2_CHK_PROFILE=0  KS_V2_CHK_COUNT=0
```
**Cross-corpus result: mean −1.44% eval error over 6 corpora, worst +0.48%** (variant set only).

### The three stages (owner's feeders → transformation → output model)
| stage | what was settled | evidence |
|---|---|---|
| **FEEDERS** | SF-shaped king zone (both axes clamped, no forward extension, minus double-pawn-defended) + slider X-RAY through queens/own rooks | audited vs SF11:239-247 and :268-271; adopted together — they are COMPLEMENTARY (x-ray alone regresses `_v2` +1.85%, the zone turns it −0.48%) |
| **TRANSFORMATION** | coordination PRODUCT (2.5%); Ethereal channel balance; both orderings measured NEUTRAL | see below |
| **OUTPUT** | bounded curve `MAX·u²/(u²+HALF²)`, quadratic at small u, saturating; ONSET threshold | ONSET swept on all 6 corpora |

### ★★ Findings that outlive this rung
1. **ORDERINGS DO NOT MATTER; BALANCE DOES.** Attacker ordering 0.13%, safe-check ordering 0.08% — both at
   the 0.05% floor. Channel balance was worth 4.83%, coordination 2.5%, the weak-square channel 2.63%.
   ★ SF (rook-high), Ethereal (knight-high) and Weiss (queen-high) all DISAGREE on ordering — three tuned
   engines diverging is evidence the parameter carries no signal. Their disagreement WAS the measurement.
2. ☠️ **MATCHING A REFERENCE'S TERM ≠ BEING RIGHT.** Tuning to minimise |our_KS − SF11_KS| produced
   `MAX=8000 ONSET=200`: best KS-term fit (0.566, misses 78→39) and **+9.70% worst-case eval error**,
   +6.4%/+9.7% on two general corpora. SF11's KS magnitude is calibrated to sit inside SF11's eval,
   balanced by shelter/flank/blocker channels we do not have.
3. ★ **Our channel set matches ETHEREAL's, not SF's.** Ethereal = attacker weight + attack counts + weak
   squares + no-queen + typed safe checks + shelter. Of SF's 8 extra components, 4 are SF-ONLY (unsafe
   checks, blockers-for-king, king flank attack/defence, knight-defender) ⇒ skip per the universality rule;
   shelter is UNIVERSAL (all three) ⇒ rung 2.
   ⇒ Importing SF's MAGNITUDES into an Ethereal-shaped formula was the root cause of the safe-check channel
   measuring as HARMFUL (SF's checks are 7-12x Ethereal's relative to weak squares).
4. ☠️ **The ONSET threshold is not optional.** Without it KS-A was a REGRESSION: +5.7% to +20% eval error
   on general corpora while helping 4-11% on a KS-selected one — over-firing in quiet positions. Both
   references suppress small danger to EXACTLY zero (SF `if (kingDanger > 100)`, Ethereal
   `SafetyAdjustment −74` + `MAX(0,·)`). I had removed it because v1's floor was mis-ordered against its
   knee — discarding a mechanism because our implementation of it was broken.
5. ⚠️ **We under-read king danger vs SF11 in ~20% of positions** (78/400 where SF11 sees >1 pawn and we read
   <half; worst SF11 −7.62 vs ours −1.14). Diagnosed as a MISSING-CHANNEL problem (shelter), not
   mis-tuning — raising MAX to close it costs +9.7% elsewhere. ★ Also structural: our curve SATURATES at
   MAX while SF/Ethereal stay quadratic, so we cannot exceed 4 pawns by construction.
   ☠️ **SUPERSEDED 09-12 — "quadratic longer before saturating" was TRIED AND REFUTED** (see the 09-12
   entry): moving the ceiling out is dominated by lowering the onset at equal worst-case cost. The
   saturation half of this finding is WITHDRAWN; the missing-channel half (shelter) still stands.

### Gates (all passing at the final config)
arm 0 byte-identical 250 / 35,310,778 / EBF 3.784 · `KS_V2_MAX=0` reproduces rung 0.5 exactly ·
colour symmetry 0/800 at TOL=0 · KS-A adds zero file asymmetry.

### ☠️ Open landmines
- `KS_V2_CHK_PROFILE=1` hardcodes **1046/1046/523/672** — stale SF-scale values, now ~7x oversized.
  **Profiles must express RATIOS against the live values, not absolutes.** Three scale errors in one day
  all had this shape.
- `KS_V2_CHK_COUNT=0` runs Ethereal's magnitudes in SF11's boolean form; Ethereal's assume per-square
  `popcount`. Built, untested.
- `KS_V2_PAWN_ATT=1` measured unsafe (+0.75% worst). Left off.
- `diagnostics/eval_vs_sf11.py` CRASHES on a v2 arm (`KeyError: 'capture_gains'`) because `ev_breakdown`
  now omits unpublished terms. ★ That is the honest-breakdown design working as intended; the tool needs a
  `.get(t, None)` and an "absent" marker. The KS-FIT line prints before the crash, so it is still usable.

---

## 2026-09-12 (00:24-00:26) — The KS shape is SETTLED: one dial, no shape freedom

⚠️ Two batteries of `_eval_accuracy_multi.py`, 8 + 7 arms x 6 corpora x 2000 rows, ~1 min each. The
anchor `g450_h600` reproduced the 09-11 record digit-for-digit on all six columns, so both runs are valid.
Scripts: `scratchpad/ks_shape_battery.sh`, `ks_ratio_battery.sh`.

### ☠️ REFUTED — "quadratic longer before saturating"
Finding 5 above named our saturating curve as a structural defect vs SF's unbounded `kingDanger^2/4096`
and marked the fix untried. It is now tried, by raising MAX and HALF together at constant `MAX/HALF^2`
(small-`u` slope identical, ceiling moved out 2.25x and 4x):

| arm | mean% | worst% |
|---|---|---|
| `g450_h600` (MAX 4000, HALF 600) | −1.44 | **+0.48** |
| `q_m9k_h900` (ceiling 2.25x out) | −1.46 | +0.68 |
| `q_m16k_h1200` (ceiling 4x out) | −1.41 | +0.79 |
| `both350` (ceiling UNCHANGED, onset 450→350) | **−2.02** | +0.71 |

★ The decisive pair is `q_m9k_h900` vs `both350`: **at the same worst-case cost (+0.68 vs +0.71),
lowering the onset buys 0.56pp more mean than raising the ceiling.** Both ceiling arms are off the
Pareto frontier. ⇒ The saturation ceiling is NOT the binding constraint; the ONSET is. Finding 5's
structural half is withdrawn; its missing-channel half (shelter) stands untested.

### ★★ The output stage has ONE free parameter
ONSET, HALF and MAX all trace the SAME monotone mean-vs-worst frontier — `HALF=400` behaves as a lower
onset (−1.77/+0.90), `HALF=900` as a higher one (−0.96/+0.24). Nothing reaches a point the onset cannot.

| frontier point | mean% | worst% |
|---|---|---|
| ours (no SF feeders), on650 | −0.70 | +0.05 |
| x-ray only, on550 | −1.09 | +0.18 |
| both feeders, on500 | −1.17 | +0.38 |
| **both, on450** (recorded config) | −1.44 | +0.48 |
| both, on350 | −2.02 | +0.71 |

⇒ "Which KS config" is not a shape question. It is one scalar — *how often does KS fire* — trading mean
accuracy against a variant-corpus regression, and no instrument we have prices that trade. **Games do.**

### ☠️ Matching SF's CHANNEL RATIO is a monotone DEGRADATION
SF11 weights weak squares to king-adjacency `185:69 = 2.68`; our tuned values are `57:61 = 0.93`. Swept at
matched total channel weight (`WEAK+ADJ = 118` held constant throughout):

| WEAK:ADJ | 0.37 | 0.93 (ours) | 1.57 | 2.68 (SF) | 5.6 |
|---|---|---|---|---|---|
| mean% | **−1.53** | −1.44 | −1.39 | −1.33 | −1.29 |
| worst% | +0.47 | +0.48 | +0.48 | +0.48 | +0.49 |

☠️ **Perfectly monotone, and pointing AWAY from SF.** Moving to SF's ratio costs 0.11pp; the best point is
the one furthest from it. Worst% is flat to ±0.02 across a 15x range. ⇒ Third instance of
`matching-a-reference-term-is-not-being-right`, now at the RATIO level and not just the magnitude level.
★ Mechanically coherent: SF prices proximity through shelter / flank / blockers-for-king channels we do
not have, so SF can afford a light adjacency weight. Ours carries that load, so it wants MORE. **Our ratio
is not a defect — it is the correct compensation for our channel set.**
⚠️ Correction to the 09-11 entry: "channel balance was worth 4.83%" measured the CHECK-channel weight
(the 7-12x oversized import), not WEAK:ADJ. Those are different balances; only the first was ever large.

### `KS_V2_CHK_COUNT=1` — tested, and it does not move the frontier
Ethereal's per-square `popcount` safe-check form: **−1.66% / +0.60%**. Best mean of any arm outside
`both350`, and it *looks* like a win on a raw ranking. But interpolating the onset frontier at worst
+0.60 gives mean ≈ −1.74 — so it is ON or just INSIDE the frontier, buying mean at a price the onset dial
sells more cheaply. ⇒ Not a regression, not an improvement: another point on the same dial.
★ This is what the frontier framing is for. A "best mean%" ranking would have shipped it as a win.

### Predictions scorecard (registered before each run)
✅ `half400` +0.90 worst called exactly · ✅ `half900` · ✅ `both350` · ✅ ratio span < 0.3pp (0.24pp)
❌ `q_m9k_h900` and `q_m16k_h1200` — I predicted the ceiling arms would WIN; they lost on both columns.
❌ `ratio_sf_2p68` — I predicted SF's ratio would beat ours; it is 0.11pp worse, monotonically.
★ Both misses are in the same direction: **I expected the reference's shape to be right and it was not,
twice in one hour.** The adopt-only-if-universal rule keeps being the thing that saves this, not the
instinct to converge.

### ▶️ What the cheap instruments can still say: NOTHING
Every knob in the transformation and output stages now lands on one monotone frontier. The remaining
untested items are a different stage or a different rung: the variant-corpus regression has never been
LOCALIZED (which positions?), and shelter (rung 2) is the diagnosed missing channel. ⇒ Further accuracy
screening at rung 1 has no decision left to inform. **Go to games.**

---

## 2026-09-12 (00:38) — v1 vs v2 HEAD-TO-HEAD, the first one ever run

⚠️ Until now every v2 number was measured **within v2** (KS-A vs v2-noKS). The claim "v2's three subsystems
match v1's thirty" was assembled from two separate runs on different baselines and **was never one table**.
It is now. Arm #1 = the shipped v1 eval, so every % reads "vs the engine we actually have".
`KING_SAFETY_MAG=0` (`search_engine.h:1559`, master percent scale) is the clean v1 KS ablation.

```
  arm                    grs      grs_v2   grs_x4   grs_uho  variant  ks_lab    mean%   WORST%
  v1_full                463.85   417.11   378.21   213.95   272.27   2145.52   (base)  (base)
  v1_noKS                -9.71%  -13.23%   -2.07%    1.39%   -6.41%    0.62%   -4.90%   1.39%
  v2_r0_matPST          -13.64%  -11.09%    8.28%   21.05%  140.77%   -8.27%   22.85% 140.77%
  v2_r1_on450           -15.10%  -11.51%    4.97%   19.28%  141.93%  -10.50%   21.51% 141.93%
  v2_r1_on350           -15.19%  -11.22%    3.58%   18.37%  142.47%  -12.23%   20.96% 142.47%
```

### ★★ The "matches thirty" claim, corrected
v2 rung 1 is **21.5% WORSE than v1 overall**, and it is **entirely one corpus**. Excluding `variant`:

| arm | mean% vs v1, five standard-structure corpora |
|---|---|
| v2 rung 0 (material + PST) | **−0.73%** |
| v2 rung 1 (+ KS-A, onset 450) | **−2.57%** |
| v2 rung 1, onset 350 | −3.34% |

⇒ The precise claim: **on standard structure, TWO terms already edge out THIRTY and three beat them by
2.6%. On the structure-independent variant/960 set v2 is 2.4x worse** (655.54 vs 272.27).
★ That is diagnostic, not merely bad: v2's positional knowledge is almost entirely PST, which encodes
STANDARD-CHESS PLACEMENT PRIORS and collapses when pieces start elsewhere, while v1's pawn-structure,
mobility and threat terms are position-RELATIVE and generalize. ⇒ The variant column stops being a
nuisance worst%-guard and becomes **the specific quantity rungs 2-4 (pawns/passers, central+space,
mobility) have to move**. It is the sharpest target the ladder has been given so far.

### ☠️ v1's king safety reads NET-NEGATIVE on this instrument
`KING_SAFETY_MAG=0` improves v1 by **4.90% mean** — −9.71% and −13.23% on the two primary corpora, −6.41%
on variant — while reading **invisible on the KS-labelled set (+0.62%)**, the one corpus where it should
prove itself. I predicted removing it would hurt: **wrong sign.**

⚠️ **Do NOT read this as "v1's KS is harmful."** The same instrument scored v2's KS at −1.44%; if it is
good enough to credit v2's it is good enough to condemn v1's, and we already know it is good enough for
NEITHER — `MOD_KS_REALIZ` ablates a **+36.7 Elo** term and reads invisible here, and
`corpus-fit-is-anti-correlated-with-elo` says any "switch off" winner is suspect by default. ★ The correct
reading is the instrument's own limit: **§I does not price king safety in Elo.** This is the strongest
evidence yet for the standing decision to send rung 1 to GAMES rather than to more screening.
⇒ ★ It also retro-explains the 0-for-11 additive KS record: every one of those attempts was layered onto
a term that this instrument says was already pulling the wrong way inside v1.

### Predictions scorecard
❌ `v1_noKS` — predicted +0.5..+4% (hurt), actual −4.90% (helped). Sign wrong.
❌ `v2_r1` overall — predicted −3..+3%, actual +21.5%.
✅ `v2_r1` on standard corpora — −2.57%, inside the predicted parity band.
✅ `on350` slightly better than `on450`.
★ The overall miss is mine twice over: I wrote the "matches thirty" claim AND predicted parity from it,
without ever having put the two arms in one table. **A claim assembled from two runs on different
baselines is a hypothesis, not a measurement.**

---

## 2026-09-12 (00:47-05:47) — RUNG 1 PASSES IN GAMES: +95 to +103 Elo

`node_ab`, LONG_FORMAT, conc 4, `openings_uho.txt` (1000 openings), SF arbiter + draw adjudication.
Both arms `EVAL_ARM=1` — this is v2-with-KS vs v2-WITHOUT-KS, one subsystem apart.
☠️ Note for reuse: EVERY `sprt` sub in the runner hardcodes `--p2-config ""`, which would have made the
baseline **v1** and answered a different question. `node_ab` is the only vetted sub that sets both sides.

| segment | onset | nodes | games | score | Elo |
|---|---|---|---|---|---|
| 1 | 450 | 100k | 1223 | 64.4% | **+102.9 ±22.9** |
| 2 | 450 | 300k | 501 | 63.4% | **+95.2 ±35.7** |
| 3 | 350 | 300k | 497 | 63.4% | **+95.3 ±35.9** |

✅ **Rung 1 is a PASS.** Far outside the harness null (+4.6) and the true candidate spread (~16), and
stable across a 3x node budget ⇒ no equal-work step-shape artifact. **Rung 2 opens.**

### ★★★ The result that outlives the rung: accuracy% does NOT convert linearly to Elo
Segments 2 and 3 are the SAME arm at two ends of the onset frontier — the only free parameter in the KS
output stage, spanning **−1.44% to −2.02%** eval accuracy (a 40% difference in the metric the whole 09-12
screening session mapped). In games they are **+95.2 vs +95.3 Elo: one tenth of an Elo apart.**

| change | Δ accuracy | Δ Elo |
|---|---|---|
| add the KS channel at all | −1.44pp | **~+100** |
| slide the onset along the frontier | −0.58pp further | **+0.1 ±50** |

Linear conversion would have priced the second row at ~40 Elo. It is zero.
⇒ ★ **ADDING THE CHANNEL PAYS; TUNING WHERE IT FIRES DOES NOT — *at this rung, at this resolution*.** This is
`every-eval-term-error-is-bidirectional` (*add SIGNAL, don't correct the MEAN*) arriving from an
independent direction: moving the onset only changes how OFTEN KS fires, which shifts a mean; adding KS
adds signal.

☠️ **CORRECTED SAME DAY (owner's catch) — this was an UNRESOLVED measurement reported as a NULL.** The CI
is **±50 Elo**; a 40-Elo frontier difference would have been invisible. What is established is a BOUND —
*the onset frontier is worth less than ~50 Elo at rung 1 on a three-term eval* — not a zero. Writing
"+0.1" and then reasoning from 0 is precisely
`the-eval-failure-record-is-mostly-unresolved-nulls-not-refutations`, the failure that steered two months
off 07-25's "eval->EBF disconfirmed".
⚠️ And our own law points the other way here: **a feature measured where it is redundant looks worthless.**
The frontier is currently measured where a huge MISSING-SIGNAL gap dominates (v2 is still 21.5% behind v1
on §I). Once the eval is dense and we are fighting for the last 20 Elo, a 20-Elo tuning gain is decisive
and is today below the floor.

### ▶️ The rule that actually follows: DEFER frontier tuning, do not ABANDON it
- **Channel gains are unbounded; tuning gains are bounded by the frontier span** ⇒ spend early effort on
  channels, where the ceiling is open.
- **§I costs minutes, games cost hours** ⇒ keep MAPPING the frontier every rung and RECORD it. That is
  nearly free and builds the map. Do not spend the GAMES budget adjudicating it.
- **Cash the accumulated maps at checkpoints**, when channels run out and differences clear the floor.
- ★ Accuracy also has value INDEPENDENT of Elo in this project: the roadmap is HCE -> the owner's own NN
  trained on its self-play, and a more accurate eval yields better-labelled training positions even at
  equal playing strength. The frontier is partly a DATA-QUALITY instrument, and that value survives even
  if the Elo bound stays flat.

### ⚠️ What this does NOT license
- **Node-limited charges KS nothing for its cost.** A pass here is necessary, not sufficient; v2 needs a
  TIME-based gate before it could ship. (Held at both 100k and 300k, but that is decision quality only.)
- **The baseline is v2-noKS, NOT v1.** v2 rung 1 remains 21.5% worse than v1 on §I overall and 2.4x worse
  on the variant/960 set. This says "rung 1 works", not "v2 is good".
- **Both arms are thin evals** where KS carries an outsized share. Expect this +100 to shrink sharply by
  rung 6.
- ☠️ **Do NOT now conclude v1's king safety is bad.** §I read v1's KS as net-NEGATIVE (−4.90% when
  ablated) and read v2's KS at only −1.44%, which games price at +100 Elo. §I understated KS by roughly
  two orders of magnitude in Elo terms. That is the third independent demonstration that **§I does not
  price king safety** (`MOD_KS_REALIZ` = +36.7 Elo reads invisible; the d7 gate is blind to KS).
  ⇒ The v1-KS-is-harmful reading is an INSTRUMENT artifact and must not be acted on.

### Predictions scorecard
✅ Owner predicted "this will pass handily" — +100 Elo.
❌ I framed the run's purpose as "establishing what 0.8-1.4% eval accuracy is worth in Elo". The run
   refuted the premise that such a conversion factor exists. ★ The calibration failed in the most useful
   possible way: it bounded the frontier's Elo value at ~0 instead of measuring a slope.

### 2026-09-12 (10:19-11:34) — TIME GATE: +99.4 ±63.6. No cost appeared.
New runner sub **`time_ab <minutes> '<p1cfg>' '<p2cfg>' [conc] [tag]`** — LIGHTNING (~1s hard cap), both
sides configurable. It fills a real gap: every other timed sub (`gate`, `gate_blitz`, `tournament`,
`tournament_seeded`) pins ONE side to `""` = the default eval, so none could compare two v2 arms.

| control | games | score | Elo |
|---|---|---|---|
| 100k nodes, on450 | 1223 | 64.4% | +102.9 ±22.9 |
| 300k nodes, on450 | 501 | 63.4% | +95.2 ±35.7 |
| 300k nodes, on350 | 497 | 63.4% | +95.3 ±35.9 |
| **~1s TIME, on450** | 158 | **63.9%** | **+99.4 ±63.6** |

★★ The score rate holds inside a **1.0pp band across a 3x node range AND across the node/time boundary**.
Four agreements beat any single CI; pooling the three on450 runs gives **~+101 Elo over 1,882 games**.
⇒ **Rung 1 passes under literal playing strength, not just equal work.** KS-A's cost does not eat the gain
(`half-a-ply-is-elo-neutral` — one subsystem of three inside an eval that is ~35% of node cost).

### ☠️ SPREAD COUPLING IS COMMON-MODE IN A WITHIN-v2 A/B — I expected the bill one comparison too early
I predicted +60..+95, charging KS for mis-sized margins (`RFP_MARGIN`, `FUTILITY_MARGINS`, razor constants
are absolute millipawns fitted to **v1's** spread). Actual +99.4. The reasoning error: **both arms run the
same margins, so both are equally mis-served and the mismatch CANCELS.** A timed v2-vs-v2 gate cannot see
spread coupling at all. ⇒ The margin bill is real but falls due at the **v2-vs-v1** comparison, which is
exactly where the plan put it ("hold margins fixed across rungs, re-sweep at checkpoints, quote eval +
margins TOGETHER"). ★ Generalises: **a confound shared by both arms of an A/B is invisible to that A/B** —
ask which comparison a cost actually shows up in before budgeting a measurement for it.

---

## 2026-09-12 (12:09) — The variant/960 gap is ONE TERM: capture gains

13 v1 ablations x 6 corpora, arm #1 = full v1. Read the VARIANT column.

| ablation | variant% | mean% | note |
|---|---|---|---|
| `SCALE_CAPTURE_GAINS=0` | **+176.89%** | +41.86% | ☠️ the whole gap |
| `ENABLE_KAUFMAN_IMBALANCE=0` | −4.50% | +7.22% | helps general, HURTS variant |
| `ENABLE_PASSER_V3=0` | +0.66% | **+5.91%** | real signal, but NOT on variant |
| `ENABLE_THREATS=0` | +2.65% | +1.98% | small, positive everywhere |
| `SCALE_PAWN_WALL=0` | +0.75% | +0.15% | ~null |
| `STRUCT_OPPOSED_*_PCT=0` | +0.26% | −0.06% | ~null |
| `SCALE_PAWN_CHAIN=0` | −0.24% | −0.16% | ~null |
| `KS_ZONE_ATTACK_PCT=0` | −2.83% | −2.75% | switch-off winner, 5/6 corpora |
| `PV_BOOST_MAG=0` | −4.19% | **−6.13%** | switch-off winner, **6/6 corpora** |

Strip capgains and v1's variant MSE goes **272.27 -> ~754**, while v2 rung 1 sits at **~659** ⇒
**v1-MINUS-CAPGAINS IS WORSE THAN v2 ON VARIANT.** The entire 2.4x gap is the one term v2 deliberately omits.
★ This locates the owner's standing capgains mystery ("we should rarely be in positions where it is
load-bearing"): **960 / piece-replacement positions**, where pieces start somewhere unfamiliar and static
positional knowledge does not transfer. Capgains is a tactical safety net covering for that.

### ☠️ WITHDRAWN: "the variant column is the target for rungs 2-4"
I wrote that this morning off the head-to-head, calling the gap a POSITIONAL-KNOWLEDGE problem. It is not.
§I scores a **static** eval against a **search-based** oracle, and capgains' whole job is to statically
pre-book material that search finds unaided ⇒ textbook `static-eval-top-culprits-are-search-absorbed`:
the term that most flatters a static eval against a searching reference is exactly the one search makes
redundant in play. **The variant gap is substantially an instrument artifact and must not aim the ladder.**
⇒ Rungs 2-4 are judged on the general corpora and on games; the variant column is DISCOUNTED as a known
capgains artifact until someone tests whether it survives search.

### ★ It strengthens the corrhist plan rather than reopening capgains
The plan's position (owner's): capgains is computed, expensive and no giant has one; **corrhist is the
learned replacement for the same job** — capgains pre-books tactics statically, corrhist learns the
residual between static eval and what search actually found. Today's number **quantifies that residual**
(~2.8x the variant error) and **locates it** (unfamiliar structure). ⚠️ And NPS was never the argument:
eval is ~35% of node cost. ⇒ Capgains stays PARKED-BUT-LIVE for a later rung, not closed.

---

## 2026-09-12 (12:58) — RUNG 2 PREWORK: the pawn clamp is HIDING HARM, and the chain bonus is a double-count

⚠️ Found only because the owner pushed back on a survey I had built by **grepping names I guessed**.
`getPPIncrement` contains no pawn-ish word and was missed; so were the two scoring entry points.
★ **METHOD FIX, adopt for every rung: enumerate a subsystem's consumers by DATA DEPENDENCY (who reads the
bitboard), never by name.** That sweep found **24 functions** touching pawn state, not the ~8 I listed.

### The dependency shape: passers are GLOBAL, not pawn-local
**Every piece evaluator takes `white_passed_pawns`/`black_passed_pawns` as parameters** (N/B/R/Q/K x
mid/end = 10), plus `boost_pieces_for_supporting_passed_pawns`, `passer_danger`, `evaluate_passers`,
`position_complexity`, `endgame_convertibility_scale`, `approximate_capture_gains` and the bishop/rook
latent-activity helpers. ⇒ `getPPIncrement` must run BEFORE every piece evaluator: a hard sequencing
constraint on v2's structure.

### `getPPIncrement` is SEVEN jobs fused, with SIDE EFFECTS
detect passed · detect SF candidates (`ENABLE_PASSER_DETECT_SF`) · detect rear-doubled (`ENABLE_PASSER_V3`)
· dock per stopper (`PP_OPP_PAWN_PEN`) · blockade quality (`PP_BLOCKADE_PEN`, `PASSER_CONTEST_PCT`,
`PP_UNBLOCKED`) · diagonal support (`PP_DIAG_SUPPORT`, `PP_FILE_CLEAR`) · ☠️ **publishes the passed-pawn
and candidate masks as a SIDE EFFECT.**
☠️ That last one **kills the naive pawn-hash plan**: caching the score skips the mask publication — the same
family as `eval-global-side-effects-are-skipped-by-a-cache-hit`. ⇒ **In v2 the detector must RETURN the
masks and the scorer must CONSUME them.** Structural split, not conventional.
★ **OWNER ORIGINAL TO KEEP DELIBERATELY: graded passed-ness.** `ppIncrement` is a CONTINUOUS obstruction
score (base, docked per stopper) feeding `passer_table_weight` as an interpolation input. **No giant does
this** — SF's `passed` is a boolean, priced afterwards by rank and blockers. Ours distinguishes one distant
stopper from three near ones; the `:893` comment records a pawn at 75 (below the 100 threshold) worth about
as much as one above it. Do not flatten this to SF's boolean.

### ☠️☠️ THE PAWN CLAMP HIDES HARM — and it explains every null structural ablation
`evaluate_pawns_midgame` ends: `total -= std::min(PAWN_CLAMP_MID, structural_bonus + positional_bonus)`.
**Structural and positional/heat-map bonuses share ONE clamp** (225 mg / 175 eg) and compete for headroom.
`clamp4x` (900/700) and `clamp_open` (100000) are **numerically IDENTICAL** ⇒ the clamp is fully open by 4x
and therefore genuinely BINDS at the default. Opening it costs **+3.20%** (constants were fitted under it).
Each ablation vs ITS OWN baseline:

| term | delta @ default clamp | delta @ open clamp | sign consistency |
|---|---|---|---|
| **chain** | −0.16% | **−2.29%** | **6/6 corpora** |
| **struct (opposed)** | −0.06% | **−1.38%** | **6/6 corpora** |
| wall | +0.15% | +0.25% | 3/3 split = neutral |

⇒ The clamp masked **14x (chain) and 23x (struct)** — not hidden VALUE but **hidden HARM**. The layer does
not "produce zero"; it produces harm plus damage-control that cancel. **`PAWN_CLAMP_*` is load-bearing
BECAUSE the terms under it are wrong** — accidental-load-bearing in its purest form.

### ★★ The central double-count is CONFIRMED, and it is the chain bonus
`pawn_chain_file_bonus[x]` is **FILE-keyed** (`cpp_bitboard.cpp:414`, `:1053`), and the SAME loop also calls
`update_global_central_scores`. So the chain bonus prices centrality a SECOND time — and once the clamp
stops hiding it, that double-count is **the most harmful pawn term in v1, 6/6 corpora**.
⇒ SF's rank-only connected bonus is VINDICATED; Ethereal's file weighting is the outlier we do NOT follow.
This closes the open question the plan listed as "decide by measurement, not argument".

### ▶️ Rung 2 decisions that follow
- ❌ **Do not port the file-keyed chain bonus.** Build SF's rank-keyed connected term
  (`{0,7,8,12,29,48,86} * (2 + phalanx - opposed) + 21 * popcount(support)`); price centrality ONCE.
- ❌ **Do not port `STRUCT_OPPOSED_*_PCT`** — SF's `opposed` lives INSIDE the connected formula, not as a
  post-hoc percentage on an already-computed bonus.
- ⚠️ `SCALE_PAWN_WALL` is a coin flip (3/3): build only if it earns its own place.
- ☠️ **v2 inherits NO pawn clamp.** Bounded-by-construction, per the plan — this is the concrete case why.
- ⚠️ **Pawns PRODUCE the heat map, OvD and central scores** in v1 (the attack-mask loop at `:1030-1053`).
  v2 dropped the heat map ⇒ rung 2 must decide what replaces the pawn→central/space contribution.

⚠️ Caveat: §I, and switch-off winners are suspect (`corpus-fit-is-anti-correlated-with-elo`). But 6/6 sign
consistency on two independent terms WITH a mechanism identified in the source is as strong as §I gets
short of games.

---

## ☠️★★★★ MANDATORY PRE-RUNG SCAN — run ALL FOUR before designing ANY rung (added 2026-09-12)

**Owner's standing instruction:** *"not just for pawns but for literally everything. I may forget some
things, but you with the ability to perform blanket scans should not be missing them."*

The rung-2 survey was built by **grepping names I guessed at**. That method is structurally blind to
anything named differently — `getPPIncrement` contains no pawn-ish word — and it found ~8 of 24 functions,
missed a side effect that invalidated the caching plan, and missed a shared clamp that inverted the
conclusion. **Guessing names is not a scan.** These four are exhaustive and name-independent:

| # | scan | what it catches | what it caught here |
|---|---|---|---|
| 1 | **Consumers by DATA DEPENDENCY** — every function reading the subsystem's bitboards/globals, found by the DATA's name, never the function's | functions named nothing like the subsystem | 24 pawn-touching functions, not 8; every piece evaluator takes the passer masks |
| 2 | **Full knob inventory INSIDE those functions** — `grep -oE 'Config::[A-Z_0-9]+'` over each span, with defaults, flagged live / zeroed / gated-off | knobs that exist but are dead, and knobs of the wrong TYPE | `STRUCT_R_MG_PCT`/`_EG_PCT` are rank-indexed ARRAYS, not scalars; ~12 pawn knobs sit at 0 |
| 3 | **Producers and SIDE EFFECTS** — what each function WRITES: globals, by-reference params, shared accumulators | anything that a cache would silently skip | `getPPIncrement` publishes the passed/candidate masks ⇒ killed the naive pawn-hash design |
| 4 | **Clamps and SHARED BUDGETS** — every `min`/`max`/`clamp` the output passes through, and WHAT ELSE shares that budget | terms whose ablation reads null because a neighbour expands into the freed headroom | `PAWN_CLAMP_MID/EG` masked chain 14x and struct 23x — and INVERTED the sign of both |

⇒ Scan 4 is the one that changes conclusions most cheaply, and it generalises past pawns: **any term
sharing a clamp with another term cannot be ablated honestly.** Before trusting ANY null ablation, ask what
budget the term shares. ★ Related: `CLAMP STACKS MAKE KNOBS STEP-SHAPED` and
`count-the-resolvable-effect-before-calling-a-null`.
⚠️ **Retroactive debt:** rung 1 (KS) was designed before this protocol existed. Scans 3 and 4 have not been
run against the KS path.

### 2026-09-12 (13:02) — Scan 4 run BLANKET across the whole eval: the masking is LOCALIZED, my generalisation was wrong
Having found `PAWN_CLAMP` inverting two terms, I claimed the per-piece `total` clamps contaminate **every**
ablation we have ever run (incl. the 47-arm veto screen). ❌ **Tested and false.**

```
  arm                grs      grs_v2   grs_x4   grs_uho  variant  ks_lab    mean%
  clamps_open        2.61%    5.64%    3.77%    1.82%    7.15%    2.66%    3.94%
  egclamp_open       0.00%    0.00%    0.00%    0.00%    0.00%    0.00%    0.00%
  ab_bishcomplex_op  1.66%    5.11%    2.83%    1.81%    9.45%    3.26%    4.02%
  ab_rookmob_op      2.84%    5.42%    3.51%    1.99%    7.65%    2.89%    4.05%
  ab_threats_op      6.05%    8.96%    6.49%    3.81%    6.20%   -0.28%    5.21%
```
Deltas vs their OWN baseline: bishop-complex −0.10% -> +0.08%, rook-mobility +0.08% -> +0.11%, threats
+1.98% -> +1.27%. **No amplification anywhere except pawns** (chain 14x, struct 23x).

**Mechanism, stated correctly:** `PAWN_CLAMP` bounds a **SUM OF TWO COMPETING QUANTITIES**
(`structural_bonus + positional_bonus`) at **225mp**, which both routinely exceed. The piece clamps bound a
whole piece `total` at **3750-4000mp**, far above typical scores, so they almost never bind. Of the +3.94%
from opening everything, **+3.20% is the pawn clamp alone**; all piece clamps together ~0.7%.

✅ **`EG_CLAMP_KNIGHT/BISHOP/ROOK/QUEEN = 0` are INERT** — 0.00% on all six corpora. ⚠️ Verified all four are
`env_int`-wired first, because byte-identical-to-control is also the silent-fallback signature. My worry
that `= 0` was deleting endgame piece evaluation is REFUTED.

⇒ ★ **CORRECTED RULE: a clamp on a SUM OF COMPETING QUANTITIES masks; a clamp on a total far above typical
magnitude does not — and which one you have is measurable in ONE arm** (open it; if the eval does not move,
it never bound). Scan 4 stands, but its output is "which clamps BIND", not "every clamp invalidates".
✅ The 47-arm veto screen's nulls are NOT broadly invalidated. Invalidated only: ablations of terms feeding
`structural_bonus`/`positional_bonus` inside `evaluate_pawns_*` (chain, wall, struct, pawn PST/heat-map).

---

## 2026-09-12 — CROSS-RUNG NOTES from the per-piece scan (not rung 2; recorded so they are not lost)

Scan 2 over each per-piece evaluator. v1's per-piece functions each mix **placement + mobility +
king-safety** content; the giants instead run ONE per-piece pass (SF `pieces<Us, Pt>()`) computing mobility,
placement AND the king-attack counts together off the shared attack maps.

| piece | what v1 actually has | v2 disposition |
|---|---|---|
| **king** | ☠️ `KS_SHELTER_FULL=185`, `KS_SHELTER_PARTIAL=75`, `KS_SHELTER_MAG=100` live **inside `evaluate_kings_midgame`** — i.e. shelter sits in KING PLACEMENT, not in `king_safety_danger` | ★ **owner: shelter belongs in KS.** Move to rung 2d. ⇒ King placement then reduces to **PST + phase-dependent king activity/mobility**, nothing more. ✅ We inherit 185/75 as a tuned prior rather than starting cold |
| **bishop** | `ENABLE_CHEAP_BISHOP_COMPLEX=true` is a **3-in-1**: `CHEAP_BISHOP_BLOCK=30` (own pawns on the bishop's colour = bad bishop), `CHEAP_BISHOP_MOB=6` (diagonal squares), `CHEAP_BISHOP_KING` (king zone), `CHEAP_BISHOP_FWD`. ⚠️ The FLOODFILL survives as `BISHOP_MOB_SECONDARY=5` ("per second-order diagonal square reachable after simulation") + `BISHOP_MOB_PAWN_ATTACK=15` but is **UNREACHABLE at defaults** (`search_engine.h:1144`) — the cheap path replaced it | decompose to THREE owners: colour complex to piece placement (rung 4), diagonal count to mobility (rung 4), king zone to **KS (already built)**. ★ owner: bishops are "mostly a unique set of mobility rules" — agreed, that is rung 4. Floodfill is a rung-4 CANDIDATE, not dead |
| **knight** | `CHEAP_KNIGHT_MOB` only; `OUTPOST_KNIGHT = 0` | rung 4: mobility + outposts (outposts need rung 2's `PawnEntry`) |
| **rook** | **19 knobs**, richest by far: `ROOK_OPEN_BASE=250`, `ROOK_SEMI=125`, `ROOK_7TH=150`, `ROOK_CONNECTED=150`, `ROOK_MINOR_BLOCK=15`, `ROOK_ROOK_BLOCK`, `ROOK_PASSER_OWN/ENEMY`, `ROOK_OWN_PAWN_BASE/RAMP`, `ROOK_ENEMY_PAWN_PEN`, `ROOK_ENEMY_RANKWIN_MODE`, `CHEAP_ROOK_MOB/FWD` | rung 6, implemented inside rung 4's pass. Needs `PawnEntry.openFiles/halfOpen` from rung 2 |
| **queen** | `CHEAP_QUEEN_MOB_MG`, `QUEEN_MOB_SAFE_MG` | ★ owner: consolidate into the rung-4 mobility pass |
| **pairs** | ✅ `BISHOP_PAIR_BONUS=300` / `KNIGHT_PAIR_BONUS=200` are **inside `if (!ENABLE_KAUFMAN_IMBALANCE)`**, which is `true` by default. The comment is explicit: *"Kaufman imbalance (below) owns bishop-pair + knight-redundancy"* | ✅ **Kaufman owns pairs** — answers the owner's question. Rung 5, no separate pair term |

### ▶️ RUNG LIST AMENDMENT

The agreed order had "mobility" (4) and "rook files" (6) but **no home for outposts, bishop colour complex,
bishop long diagonal, minor-behind-pawn, trapped rook or queen weakness.**
⇒ **Rung 4 widens to "the per-piece pass: mobility + piece placement"**, built as ONE loop over the shared
attack maps (the giants' structure, and the owner's own attack-map amortisation argument one level up).
★ **Rungs are an ATTRIBUTION unit, not a code-structure unit** — implement in one pass, gate each term
behind its own `EVAL_V2_RUNG` value, and we get the giants' efficiency WITH per-term attribution.

Amended order: 1 KS ✅ · 2 pawns + passers + shelter · 3 central + space · **4 per-piece pass (mobility +
placement: outposts, minor-behind-pawn, bishop colour complex, bishop long diagonal, trapped rook, queen
weak)** · 5 Kaufman (owns pairs) · 6 rook files (in rung 4's pass, gated separately) · 7 corrhist ·
8 threats · 9 winnability · 10 capgains (parked-but-live) · 11 OvD.

---

## 2026-09-12 -- PROTOCOL CLARIFICATION: when retuning an earlier rung is legitimate (owner)

Owner: *"if we do end up needing to retune previous rungs as we go or after everything is done, that's not
something to be ashamed of, so these values could change slightly by then."*

This resolves an apparent contradiction in the plan. The escalation table marks **"retune the NEIGHBOURS"**
as a ☠️ RED FLAG, while the tuning cadence says **"feature-local per rung, FULL RETUNE at checkpoints"**.
Both are right; they answer different questions.

| case | what is being asked | verdict |
|---|---|---|
| ☠️ **Rescue** -- retune neighbours so ONE new term starts paying | "can I make this term look good?" | RED FLAG. It means the term has no independent value and is being propped up. This IS the fitted-around-neighbours signature the rebuild exists to escape |
| ✅ **Checkpoint retune** -- all constants move together because the eval's COMPOSITION changed | "given this eval now has N terms, what are the right constants?" | LEGITIMATE and PLANNED |
| ✅ **Revision** -- a later rung reveals an EARLIER rung's constant was wrong | "was the earlier value an artifact of what was missing then?" | LEGITIMATE. ★ This is the ladder working, not a failure |

★ The pawn taper is exactly the third case: `EVAL_V2_PAWN_MG` measured NULL at rung 0.5 and is the largest
single result of the rebuild at rung 2. Its earlier value was an artifact of having no pawn structure for
it to act on -- which is precisely why it was PARKED with a re-test trigger rather than refuted.

⇒ **Immediate consequence: the rung-1 KS constants were tuned against a FLAT 1000 pawn.** If a taper ships,
`KS_V2_MAX=4000` stops being "4 pawns" in the midgame and becomes ~7.3 midgame pawns. Under this
clarification, re-checking KS after the taper is **expected**, not an admission that rung 1 was wrong.
⚠️ The discipline that still applies: a revision must be driven by a STRUCTURAL reason (the unit of account
changed), never by "the number looks better if I move it".

---

## 2026-09-12 -- STS OVERTURNS THE TAPER, AND THE PAWN RUNG IS A BIG WIN

Owner: *"WAC at d10 is a bit weird -- we were beating SF with it despite being definitively inferior.
They crushed in STS though, which is probably what we should be measuring with."*
=> Re-measured the whole pawn rung on STS. Recorded as
[[wac-at-d10-does-not-discriminate-strength]]; WAC remains valid ONLY as a byte-identity fingerprint.

```
v1 (arm 0)              1796/3000
rung 1 only             1480/3000
2a+2b, NO taper         1698/3000     +218 over rung 1
2a+2b + pawn taper 550  1614/3000     -84
2a+2b + pawn taper 700  1522/3000     -176
```

### ★ ★ ★ The pawn rung is worth +218 STS -- 3x the ~70-point chaotic floor
And **v2 with FOUR subsystems (material, PST, KS, pawns) is now 98 points from v1's ~30 terms**, down from
a 432-point gap at rung 0. => **The rebuild has closed 77% of the STS gap to v1.**

### ☠️ ☠️ The taper is a MONOTONE REGRESSION on STS, and §I said the opposite
§I called the pawn taper the single biggest win of the rebuild (-6.59% mean, worst -2.05%, negative on all
six corpora). STS calls it -84 / -176. **Two instruments, opposite signs, both large.**
=> `corpus-fit-is-anti-correlated-with-elo` with a SECOND instrument confirming rather than a suspicion.
⚠️ §I is a STATIC accuracy measure against SF18; STS is a MOVE-CHOICE measure. When they disagree this
hard, the change is altering the eval's SCALE rather than its ordering -- which is exactly what a material
change does.

### ★ The mechanism survived even though my evidence for it did not
I first flagged the taper using a WAC drop (245/300), which the owner then showed is worthless -- WAC at
d10 does not discriminate strength. **But the HYPOTHESIS was right**: tapering the pawn inflates every
positional constant relative to material in the midgame (`KS_V2_MAX=4000` stops being "4 pawns" and becomes
~7.3 midgame pawns), and STS confirms the direction on better evidence.
=> ⚠️ Worth separating: *the hypothesis was sound, the evidence I cited for it was not, and I should have
said so when I quoted it.*

### => THE DECISIVE TEST: taper the PIECES, not the pawn
Same material RATIO shift, but the pawn stays the UNIT OF ACCOUNT so every positional constant keeps its
meaning. New knob `EVAL_V2_PIECE_MG_PCT` (percent of a piece's value in the full midgame).
Calibration: in PAWN units SF's knight falls 6.10 (mg) -> 4.01 (eg), so preserving our shipped endgame
values and adding only the relative shift gives **152%**.
★ Prediction: if the taper's harm is the unit-of-account side effect rather than the material ratio
itself, the piece-side version should hold or improve STS where the pawn-side version lost 84-176 points.

---

## 2026-09-12 -- ☠️ I RANKED INSIDE THE NOISE BAND TWICE IN ONE HOUR

Owner: *"even STS is just one bench -- there are many error testers, real game tests, single eval tests
with MSE and more."* Checking INSTRUMENT-MAP resolutions AFTER making claims, which is the wrong order:

| reading | delta | floor | verdict |
|---|---|---|---|
| WAC 250 -> 245 | -5 | **+/-5-6** | ☠️ inside noise -- NOT EVIDENCE |
| STS 1698 -> 1614 (pawn taper 550) | -84 | **+/-150** | ☠️ inside noise -- NOT EVIDENCE |
| STS 1698 -> 1522 (pawn taper 700) | -176 | +/-150 | marginal, weak |
| STS 1698 -> 1629 / 1553 / 1568 (piece taper 120/152/180) | -69 / -145 / -130 | +/-150 | ☠️ inside noise, and NON-MONOTONE at the tail |
| **STS 1480 -> 1698 (pawn layer)** | **+218** | +/-150 | ✅ RESOLVABLE -- real |

★ ★ **I called the taper a "monotone regression" on two readings that were both inside their floors,
did it with WAC, was corrected, then did the identical thing with STS within the hour.** The map states
"do not rank inside +/-150" explicitly.

### The corrected evidential state
- **Pawn layer**: §I -0.97% (floor 0.05%) AND STS +218 (floor 150) => **two independent instruments, same
  sign, both resolvable. CORROBORATED.**
- **Taper (either implementation)**: §I -6.59% is resolvable at 130x its floor and says LARGE WIN; every
  move-level instrument is INSIDE its floor. => **UNDECIDED, not refuted.** Against it stands only the
  prior `corpus-fit-is-anti-correlated-with-elo`, which is a prior and not a measurement.

### ☠️ The unit-of-account hypothesis is REFUTED
I predicted the PIECE-side taper would hold or improve where the pawn-side lost, because it keeps the pawn
at 1000 as the unit of account. It trends down by comparable amounts (-69 at 120% vs -84 for pawn-side 550,
comparable ratio shifts). Two independent implementations of the same shift behave the same
=> the harm, if any, is the MATERIAL RATIO itself and not the side effect on positional constants.
⚠️ Fifth consecutive magnitude/shape prediction wrong, every one in the same direction: expecting a
principled correction to pay.

### ★ VENUE POWER -- run before spending games, as §A requires
```
+40 Elo vs SF18   273 games      KP paired  131
+20 Elo          1,087           KP paired  521
+10 Elo          4,344           KP paired 2,083
 +5 Elo         17,370           KP paired 8,331
```
**One night (~1,200 games) resolves ~+20 Elo and nothing smaller.** => If the taper is a +/-10 Elo effect,
no single night settles it either, and the decision would have to rest on principle rather than
measurement. ⇒ **Spend tonight on the CORROBORATED change (the pawn layer), not on a taper arm that three
instruments already cannot separate.**

---

## 2026-09-12 -- PROTOCOL: WHEN TO BUNDLE RUNGS (owner), and the boundary of the bundling refutation

Owner: *"pawns like KS are a huge part of the eval, which is mostly bare bones -- surely it still adds up.
If things start getting small though we may have to start bundling rungs eventually."*

### The first half is supported by our own numbers
| rung | STS | gain | games |
|---|---|---|---|
| 0 (material + PST) | 1364 | -- | -- |
| 1 (+ KS) | 1480 | **+116** | **+101 Elo** |
| 2 (+ pawns) | 1698 | **+218** | pending |

The pawn layer's STS gain is ~1.9x KS's, and KS was worth +101 Elo.
⚠️ Not extrapolated to a number -- no statistic of ours predicts Elo and STS->Elo is not linear -- but it
is a strong prior that this is a LARGE effect, comfortably above the ~20 Elo one night resolves. In a
barebones eval each subsystem is a large fraction of everything the engine knows, so the EARLY rungs should
keep clearing the bar. The question is what happens when they stop.

### ☠️ The bundling refutation has a BOUNDARY, and it is important
`bundling-is-refuted-components-cancel-26-percent` was measured **ON v1**, where ~30 terms overlapped
50-64% and three individually sign-consistent components combined to **+0.47pp -- BELOW every one of them
alone** while cancelling ~26% of each other's move changes.
★ **That refutation is about DEGENERACY, not about bundling as such.** v2 is built to make bundling safe
in the way v1 could not be: terms are constructed disjoint, and the firing-set overlap matrix is checked
BEFORE any constant is chosen. The refutation does not automatically transfer -- but it supplies the
PRECONDITION.

### ★ ★ THE RULE (venue power turns this into a number)
```
+40 Elo vs SF18   273 games      KP paired  131
+20 Elo          1,087           KP paired  521
+10 Elo          4,344           KP paired 2,083
```
One night (~1,200 games) resolves **~+20 Elo and nothing smaller**.

> **Test a rung ALONE if its expected effect is >= ~20 Elo. Below that, BUNDLE until the bundle clears
> ~20 Elo -- and before bundling, check pairwise overlap / contribution correlation between the members.**

⚠️ **The honest cost: a bundle that passes tells you the BUNDLE works, not which member did it.** That is
acceptable when the alternative is an UNRESOLVED result -- which is how most of the "nulls" in our record
were actually created (`the-eval-failure-record-is-mostly-unresolved-nulls-not-refutations`: of "KS
0-for-11" only ~4 were RESOLVED).
★ This supersedes the fixed "games at checkpoints every 3-4 rungs" cadence with a MEASURED trigger: bundle
when the expected effect falls below what the venue can resolve, not on a fixed count.

### ⇒ Applied right now
- **Pawn layer: test ALONE tonight.** Expected large (STS +218, ~1.9x KS's +116), two instruments agree.
- **Taper: do NOT spend a night on it.** Three move-level instruments read it inside their floors; if it is
  a +/-10 Elo effect the venue cannot resolve it in one night, and the result would be another unresolved
  null in the record. Carry it as an open question into a later checkpoint bundle.

---

## 2026-09-13 (07:26) -- ★ ★ RUNG 2 PASSES IN GAMES: +60.4 +/-25.5 Elo

`sprt_ab` (new), LIGHTNING, conc 4, `openings_uho.txt` **seed 7** (not the default 0 -- that is a known
read-inflater). Both arms `EVAL_ARM=1`, differing ONLY in the pawn layer.
```
A = rung-1 KS + PS_V2_MAG=100 PASSER_V2_MAG=60
B = rung-1 KS
+495 -325 =167 of 987  (58.6%)   elo ~ +60.4 +/-25.5   LLR +2.926
DECISION: H1 accepted -- P1 is stronger
```
✅ Control re-verified after the run: arm 0 = **250 / 35,310,778 / EBF 3.784**, byte-identical. v1 is
untouched despite a full day of eval work.

### ★ ★ THREE INSTRUMENTS AGREED, AND THEY WERE RIGHT
| instrument | reading | floor | verdict |
|---|---|---|---|
| §I eval accuracy | -0.97% | 0.05% | helps |
| STS300 | **+218** | 150 | helps |
| **games** | **+60.4 +/-25.5** | ~20 | **helps** |

=> The "corroboration" rule that chose this candidate over the taper was CORRECT: when two independent
instruments agree, trust it; when they conflict, do not spend a night. The taper had §I strongly positive
and every move-level instrument inside its floor -- had we run that instead, the night would most likely
have produced another unresolved null.

### ⚠️ Early SPRT readings were badly misleading, exactly as warned
| games | elo | |
|---|---|---|
| 125 | ~+6 | looked flat; I said "not behaving like a large effect" |
| 239 | ~+28 | drifting |
| **987** | **+60.4** | true |
★ At 125 games SE is ~+/-90 Elo -- the reading carried no information. **Do not report an SPRT point
estimate before the LLR is a meaningful fraction of its bound.**

### The ladder so far
| rung | content | games |
|---|---|---|
| 1 | king safety | **+101 Elo** |
| 2 | pawn structure + passers | **+60.4 Elo** |

⚠️ **The rungs are shrinking (101 -> 60), exactly as the owner anticipated.** +60 still clears the ~20 Elo
the venue resolves, so rung 3 can still be tested alone -- but the bundling trigger is now visibly
approaching rather than hypothetical.

### STS attribution, for rung-3 planning
rung1 1480 -> 2a only 1579 (+99) -> +2b@25 1626 (+146) -> +2b@60 **1698 (+218)**.
⚠️ Only the full config cleared STS's +/-150 floor; the components individually did not. The bundling
rule already fired in miniature at rung 2.

☠️ **`OPTIMIZATION_LOG.md` still NOT touched.** This is rung 2 vs rung 1 -- a v2-internal comparison.
v2 earns a canonical entry only by beating **v1** in games, and it is still ~98 STS points behind.

---

## 2026-09-13 -- SLICE 1 COMPONENT 1 (TEMPO): built, PROVED correct, and PARKED on measurement

Full design + result: `EVAL-V2-SLICE1-TEMPO-DESIGN.md`. Register: `EVAL-V2-CURRENT-CONFIG.md` PARKED table.

**Record-check first (the discipline that failed on draw detection).** Tempo had NEVER been built or
measured -- proposed 3x, skipped once at `hce-eval-mechanisms-2026-07-15.md:72` ("exactly the refuted
lane"). That skip inherits from the 07-02 bundle thesis, which was itself REVERSED as the 07-25 regime
error. So it closed nothing: PROPOSED ONLY / CLOSED ON A PROXY.

### 🐛 Our own reference fact sheet was wrong, and it would have flipped the adoption verdict
`sf-evolution-fact-sheets-2026-07-05.md:15` records SF 1.0 as having "NO tempo". It has one --
`stockfish_1/stockfish-1.1_ja/src/value.h:95` `TempoValueMidgame = Value(50)`, applied at
`position.cpp:909`. The fact sheet surveyed **evaluate.cpp**; SF1 keeps tempo in **position.cpp**, on the
incremental mg/eg accumulators.
☠️ Textbook case of our own rule: **enumerate by DATA DEPENDENCY, not by file or name.** The count decides
the verdict -- 3/5 (a real split) versus the truth, **4/5**, the sole abstainer being SF15.1, which
removed it from eval AND search together (`search.cpp:1461` is a bare `-(ss-1)->staticEval`) in the
generation where NNUE, natively side-to-move-aware, became the real eval.

### ★★ Do not copy a reference constant whose phase profile comes from a unit we do not share
SF11 (28), Ethereal (20) and Weiss (18) each write ONE FLAT constant -- but their pawn is dearer in the
endgame (128->213, 82->144, 104->204), so their tempo is silently **~1.7-2.0x more pawns in the midgame**.
They never chose that taper; the unit gave it to them. **Our pawn is FLAT at 1000 in both phases**, so
transcribing the flat form yields a flat-in-pawns tempo that NONE of them has. SF1.1, the only engine that
chose the profile deliberately, went steeper (3.1) and smaller (123 mg).
=> We took SF1s SHAPE with the others MAGNITUDES: mg 200 / eg 110, phase-blended on `phase256`.
⚠️ This is the SECOND time this exact trap appeared -- `search_engine.h:808` records the passer table
inheriting the same phase relationship we do not have.

### ⭐ The gate was an EXACT IDENTITY, which we have never had for a magnitude term
v2 was 100% side-to-move-blind, so with tempo the `eval_symmetry.py` TEMPO swing must equal exactly `2t`
on EVERY position. 600 FENs, full phase spread:

| config | swing mean / median / max | required |
|---|---|---|
| v1 arm 0 (harness proof) | +1.339 / +0.053 / 21.213 | large, ragged ✅ |
| v2 tempo OFF | **+0.000 / +0.000 / 0.000** | exactly 0 ✅ **blindness PROVED** |
| 200/200 constant | +0.400 / +0.400 / 0.400, zero spread | exactly 0.400 ✅ |
| 110/110 constant | +0.220 / +0.220 / 0.220, zero spread | exactly 0.220 ✅ |
| 200/0 mg leg | +0.206 / +0.210 / **0.400** | ceiling exact ✅ |
| 0/110 eg leg | +0.106 / +0.102 / **0.220** | ceiling exact ✅ |

★ The legs ADD exactly: 0.206 + 0.106 = 0.312 = the combined runs mean. Sign, magnitude and phase curve
all verified as identities.
☠️ Note the trap this avoided: `_eval_symmetry.py` -- the gate the protocol runs on every rung -- mirrors
`turn` too, so a BACKWARDS-signed tempo passes it clean. The mirror gate cannot see this term at all.

### The verdict: two instruments agree, so no games were spent
| arm | STS | nodes (WAC, fixed depth) |
|---|---|---|
| off | **1698** (= the recorded rung-2 value, a clean control) | 63,216,318 |
| ref 200/110 | 1631 (-67, INSIDE the ±150 floor) | 70,058,283 (**+10.82%**) |
| crank 4x | **1312 (-386**, far outside the floor) | 64,992,326 (+2.81%) |

☠️ WAC solves were 246/252/240 -- floor ±5-6, and the metric does not discriminate strength. Not read.

1. STS is **monotone downward** with no local maximum => optimum at or below 0. Same shape as connected
   pawns at rung 2. Only the crank point resolves, and it is clearly harmful.
2. ★★ **A term with ZERO POSITIONAL VARIANCE has no channel but margin/parity interaction.** Unlike a real
   term, none of tempos node movement can be "a truer eval prunes better" -- 100% of it is confound by
   construction. And it is NON-monotonic (+10.8% at 1x, +2.8% at 4x): the step-shaped signature of a
   constant crossing fitted absolute thresholds (`RFP_MARGIN` 1500/ply, `DELTA_MARGIN` 1500, `OTV` 1750).

=> **PARKED at 0.** Re-test trigger: the checkpoint margin re-sweep, the only condition under which the
measurement could change.

### ⚠️ What the slice plan got wrong, and the generalisation
`EVAL-V2-CURRENT-CONFIG.md` §5 billed slice 1 as "order-INVARIANT terms that cannot cancel". Tempo IS
order-invariant within a node and genuinely cannot cancel with its slice-mates -- **and that was never the
relevant risk.**
★ **"Cannot cancel with its slice-mates" is NOT "has no confound of its own."** Every remaining slice-1
member must also be checked against the absolute margins, not only against each other.

⚠️ And on my own pre-registered criteria: the ">10% node movement" threshold was set WITHOUT calibrating
against [[node-savings-below-35-percent-are-elo-neutral]], so the number was arbitrary. Its purpose was met
anyway -- by the non-monotonicity and the zero-variance argument, neither of which depended on the
threshold. ★ Pre-register the MECHANISM a criterion tests for, not only a number.

---

## 2026-09-13 -- SLICE 1 DRAW CLASSIFIER: a shipped v1 defect found, v2 classifier built and fully gated

Full record: `EVAL-V2-SLICE1-DRAW-DESIGN.md`. Register: `EVAL-V2-CURRENT-CONFIG.md`.

**The v1 defect.** `is_practically_drawn` is LIVE and unconditional (`cpp_bitboard.cpp:7941`) and returns 0 for the
WHOLE eval. Checked against the Lichess 7-piece tablebase with a new oracle (`diagnostics/_draw_oracle.py`), FIVE
of its ten cases flag FORCED WINS: R+B vs R 28%, R+N vs R 22%, bare R vs bare minor 24-28%, wrong-coloured-bishop
rook pawn 10%, and the rook-pawn KPvK rule 6.2% -- the one case that carried a documented oracle validation.
Shown LIVE in the shipped engine: K+R+B vs K+R (tablebase win in 21) evaluates to exactly 0. v1 is NOT fixed:
it is the frozen control.

**Why it was missed.** The June cases were validated on MEAN BIAS against an NNUE trust gate, which cannot see
false positives. And the cases were written as dynamic RACES (defender_dist <= min(...)), which ignore whose move
it is; SF writes the same ideas as STATIC FORTRESSES already reached.

**What the references do (from source, SF 1.1 / 11 / 15.1, Ethereal, Weiss).** Four tiers, unchanged in SF for 14
years: an exact KPK bitbase; value functions (known-win drives, and a TECHNIQUE GRADIENT for KRKB/KRKN -- a small,
corner-seeking value replacing material, not a draw); scaling functions; and a four-line generic pawnless rule
(material.cpp:198) that zeroes KmmKm and scales R+minor vs R to 14/64. v1 ENUMERATED what SF GENERALISED. Nobody
returns a hard 0 for R+B vs R.

**v2 draw_class, behind DRAW_V2_CLASS (default off).** Members: KvK, KBvK, KNvK, KBvKB, KNvKN, plus reference-derived
KBvKN (0/400), KNNvK (0/400) and SF fortress wrong-bishop (0/286). Lone-pawn cases split into DRAW_V2_KPK (off): a
tempo term cuts KPvK 6.2% to 0.6%, still not zero -- the real fix is an exact KPK bitbase. All four gates pass.

**The gate itself had to change.** Zero false positives is uncertifiable for minor-piece draws: rare boxed-king
mates exist (constructed and TB-confirmed: 6nk/8/6K1/4N3/8/8/8/8 w, Nf7# in 1) that no random sampler finds. But
every forced win in 1,592 biased minor-piece positions was DTM 1, search was shown to play such a mate with the rule
scoring 0, and null move is disabled below 7 pieces. PROPOSED: tolerate SHORT false positives (search finds them),
forbid LONG ones. Awaits owner sign-off.

**My errors, recorded so they are not repeated.** (1) Shipped rookpawn_KPvK into the classifier on 0-for-62 after
writing "unrefuted, not verified" -- a third seed found 6.2%. (2) Listed wrongB_rookpawn as provable -- 10%.
(3) Recommended migrating R-vs-minor cases into the convertibility SCALE; SF shows a flat scale is the wrong
instrument -- they need a shaped gradient. (4) Published an empty breakdown when the rule fires, which crashed the
symmetry gate with KeyError: total. (5) Crossed the PowerShell-to-WSL boundary with a dollar sign three times in
one day, once corrupting a run into ON-twice. (6) Claimed "v1 has no draw extras" from a name-scoped read; the
blanket scan found the real extras live in the scale/drive space.

**Also found.** STRENGTH_BACKLOG called the convertibility scale shipped for three months; it was reverted the same
day (ab070b7) on WAC -3 / STS -3.3, both inside noise floors -- an unresolved null, not a refutation.

## 2026-09-14 -- SLICE 2 PREWORK: v2's positional spread, measured on the move-ordering quantity

Checklist step 4 of `SESSION-HANDOFF-2026-09-13.md` §3: measure the scale BEFORE choosing any mobility magnitude.
The 09-13 "5-35 mp" came from five hand positions; this replaces it with the quantity that actually ORDERS moves.
🧰 `diagnostics/_v2_positional_spread.py` -- static eval of every legal QUIET child (no capture / promotion / check /
castling) of 370 `game_regret_set.csv` positions, from the root mover's POV. Siblings share material, so the spread is
purely non-material. One process per arm; knobs echoed.

| median, mp | rung 0 (PST only) | + pawns/passers | shipped (+KS) | v1 |
|---|---|---|---|---|
| sibling std | **11** | 19 | 36 | 2,290 |
| sibling range | 45 | 101 | 194 (p90 1,088) | 8,697 |
| \|child - sibling mean\| N / B / R / Q | 11.4 / 8.0 / **2.9** / 10.6 | 11.7 / 9.4 / 4.8 / 11.9 | 15 / 13 / 9 / 21 | 1,144 / 1,347 / 1,337 / 2,554 |
| hand: knight a3 vs e5 · rook d1(open) vs a1 | 30 · 0 | 30 · 0 | 30 · **0** | 554 · 304 |

**Readings.**
1. ★ **Placement orders piece moves by ~10 mp, and ROOK moves by 2.9 mp** -- and rook moves are the LARGEST quiet-move
   class (3,051 of 10,408 children). v2 has almost nothing to choose between rook moves with. That is precisely where
   mobility and rook files land.
2. The shipped spread's tail is NOT placement: pawns/passers widen the pawn and king rows in the endgame, and KS widens
   the queen row (p90 338). A mobility magnitude must not be sized against a spread KS is carrying.
3. v1's spread is ~200x v2's and is not a positional scale at all -- capture gains prices hanging pieces statically, so
   a quiet move that hangs a piece moves v1 by pawns.
4. ⚠️ **This is a SCALE, not a TARGET.** v2's PST cells are already 3-10x smaller than SF11's (`pt-star-is-material-inclusive`
   memory), so sizing mobility to v2's spread would bake in a compression we may not want. The defensible anchor is
   the references' own **mobility : PST-placement ratio**, applied to v2's measured PST spread -- a ratio within one
   engine carries no unit of account.
5. ⚠️ The SF18 column (multi-PV SEARCH scores on the listed moves, median range 950 nominal mp) is not comparable to a
   static spread -- it contains tactics -- and is recorded only as context.

**Record-check (same day), in one line:** nothing in slice 2 has a RESOLVED verdict either way. Whole-board mobility
is the best standing lead (+1.2pp, sign-consistent, ~1σ, 09-10 paired nulls); SF per-piece mobility's -175 STS was a
three-part bundle (area + floor + disabling the cheap surrogates) and closes nothing; outposts' "harmful" was one
point on a superseded comparator; long diagonal / minor-behind-pawn / trapped rook / queen weakness were never tried.
☠️ Mobility's STS sign once FLIPPED on cheap-rook-mobility presence ⇒ **2x2 mobility x rook files before reading either.**

## 2026-09-14 -- SLICE 2: mobility core + rook files BUILT, all gates pass, instruments CONFLICT at the reference point

Design + full tables: `EVAL-V2-SLICE2-MOBILITY-DESIGN.md`.

**Built** (all default off): `MOB_V2_MAG` -- SF11 MobilityBonus SHAPE, each leg converted by its own pawn, scaled so the
knight mg range = MAG mp; accumulated inside `build_side_attacks` (one attack pass shared with KS; x-ray follows
`KS_V2_XRAY`); area = not enemy-pawn-attacked, not own blocked pawn, not own king · `MOB_V2_EXCL_QUEEN/LOWRANK`
(where the references split) · `ROOKFILE_V2_OPEN/SEMI` from `PawnEntry` file masks · `mobility_probe` +
🧰 `_mobility_detector_oracle.py`.

**Gates, all ✅:** arm 0 `250 / 35,310,778 / 3.784` · v2 shipped, slice 2 off `246 / 63,221,361 / 4.087` · oracle 0
mismatches over 6,008 side-positions on both code paths (99.2% non-mirror) · colour swap 0/800 (file-mirror 21 @ 5 mp
identical with slice 2 off, pre-existing) · knob executes (knight rim-vs-centre 30 → 59, open-file rook 0 → −69) ·
tempo swing exactly 0.000 on 600 FENs.

**2x2 at mobility 115 / rook files 50-25:**

| arm | STS (1698) | §I mean | §I worst |
|---|---|---|---|
| mobility | 1618 (−80) | −1.39% | +0.35% KS-critical |
| rook files | 1608 (−90) | −0.32% | −0.06% (6/6) |
| both | 1659 (−39) | −1.70% | +0.29% KS-critical |

§I is additive (−1.71 predicted from the parts, −1.70 measured) and mobility's standard-corpus gain (~−2% on each of
four) is ~2x the pawn layer's. STS reads every cell negative INSIDE its ±150 floor. ⇒ **the instruments conflict; per
the corroboration rule no games are spent on this point.** Magnitude ladder (mobility 30/60/300, rook files 20-10 /
100-50, §I across all) launched 00:01 as `s2_ladder.sh`.
⚠️ My registered magnitude prediction (+60 to +150 STS at the best rung) is under strain -- a sixth consecutive
optimistic miss if it fails.

## 2026-09-14 (overnight) -- EXACT KPK BITBASE built and gated; the KPK ORACLE ITSELF WAS BROKEN

**Built** (default off, `DRAW_V2_KPK_EXACT`): SF11 `bitbase.cpp`'s retrograde classification in `eval_v2.cpp`, built once
through a thread-safe function-local static. ☠️ It deliberately uses NO runtime attack table (local shift arithmetic) --
a lazy static reading `BB_KING_ATTACKS` before `initialize_attack_tables()` would latch a garbage bitbase for the life of
the process. Hooked into `draw_class` ahead of the old lone-pawn heuristic; only a DRAWN K+P vs K returns 0. Probe
`kpk_probe` / `ChessAI.kpk_win`.

**Gate: exhaustive, not sampled.** `_kpk_oracle.py --all-files --engine` solves every legal KPvK state with the pawn on
files a-d (**165,676 states**, rook under-promotion included) and compares the engine on each:
**0 false draws, 0 false wins.** Oracle WIN 111,282 (67%) / DRAW 54,394. Arm 0 and v2 shipped both re-verified
byte-identical after the build.

☠️☠️ **The first run FAILED with 98,533 "false wins" -- and the defect was in the ORACLE, not the bitbase.** The tool
keyed states by `board.fen()`, which includes the halfmove/fullmove counters: every generated state is `... 0 1`, but a
king move's child is `... 1 1`, so child lookups missed and WIN never propagated except through pawn pushes. The
oracle labelled only **7.7%** of KPvK won. Tablebase ground truth on a disputed row (`8/8/8/8/8/8/P7/K1k5`) said WIN with
either side to move, matching the bitbase. Fixed by keying on placement + side to move.
★ **Consequence for the record:** a win-starved oracle cannot see a FALSE DRAW -- a rule's wrongly-flagged "draw" is
simply agreed with. So **every "0 false-draws" result this tool produced before 2026-09-14 is UNVERIFIED**, including
the June rook-pawn rule's "validated: 83,238 states, 0 false-draws". That is very likely WHY a "validated" rule measured
**6.2% false draws** against the real tablebase on 09-13. The mystery in the draw arc has a mechanism now.
★ **Transferable:** an oracle is an instrument. Before trusting its null ("0 failures"), check its POSITIVE rate against
ground truth -- a 7.7% win rate for KPvK was visible in its own header line.

⚠️ **My own prior was also wrong, and the tablebase caught it:** I wrote `4k3/8/4K3/4P3/8/8/8/8 w` into the verify plan
as a draw (Black "has the opposition"); it is a **WIN in 21** -- a king on the 6th ahead of its pawn wins regardless.
Every verify row is now tablebase-sourced or an exact colour mirror of one.

**Eval-level verify — ✅ `_draw_v2_verify.py`, both arms.** Rule ON (`DRAW_V2_CLASS=1 DRAW_V2_KPK_EXACT=1`): ALL AS
EXPECTED — the three tablebase draws (distant opposition W-to-move, its colour mirror, h-file rook pawn) read exactly
0; the three wins read −1104 / +1104 (exact colour mirror) / −1345. Rule OFF: the same draw rows read −1104 / +1104 /
−1274, so the RULE is what zeroes them.
🐛 The tool's old "midgame" control read **exactly 0 in both arms** under the shipped rung-2 knobs — a coincidental
PST/KS/pawn cancellation, so it failed while proving nothing. Replaced with a pawn-up middlegame (−1050), non-zero by
construction. ★ **A control must be non-zero BY CONSTRUCTION, not by luck.**
⚠️ `_eval_symmetry.py` was NOT run for this: its sample contains essentially no KPvK, so it would be a vacuous pass.
The verify rows' exact mirrors are the symmetry evidence.

**WAC d10 with the rule ON: 246 / 63,221,296 / 4.087** vs OFF 246 / 63,221,361 — **−65 nodes, same solves.** Not
byte-identical, and correctly so: the bitbase FIRED inside a handful of search lines that reached K+P vs K, and it moved
no solve. (The draw classifier's own enable read +0.008% nodes the same way on 09-13.)

▶️ Remaining before `DRAW_V2_KPK_EXACT` can fold into `DRAW_V2_CLASS`: STS (cannot fire on a midgame suite — expected
unchanged, cheap to confirm) · **the owner's sign-off**. Every correctness gate is done.
✅ **SHIPPED later the same day** as `DRAW_V2_KPK_EXACT=1` in the v2 config, on the owner's sign-off ("low latency,
adds accuracy, the giants do it"). Not deferred to the endgame slice: classifications ship on oracle proof.
☠️ **New shipped v2 fingerprint (mobility 600 + KPK exact): WAC d10 `250 / 60,036,572 / EBF 4.043`** — +39 nodes vs
mobility-only, same solves. Every later byte-identity check uses this.

## 2026-09-14 (overnight) -- SLICE 2 MOBILITY: STS cannot see it, §I and the regret gate both can, and they AGREE

Full tables: `EVAL-V2-SLICE2-MOBILITY-DESIGN.md` §3.2-§3.4.

| `MOB_V2_MAG` | 3 | 10 | 30 | 60 | 115 | 300 | 600 | 1000 | 2000 |
|---|---|---|---|---|---|---|---|---|---|
| STS (1698) | −2 | −37 | −56 | −81 | −80 | −64 | −110 | −76 | — |
| §I mean / worst | — | — | −0.38/+0.09 | −0.75/+0.18 | −1.39/+0.35 | −3.34/+0.92 | −5.75/+1.91 | **−7.56**/+3.31 | −5.93/+7.17 |

1. ☠️ **STS does not price mobility's magnitude**: flat and unordered from 10 to 1000 mp, all inside ±150. Two
   explanations I recorded along the way were withdrawn by later points — a perturbation floor (3 mp reads 0) and a step
   (1000 undoes 600).
2. ★ **§I optimum ≈ 1000 mp of knight range — near SF11's PAWN conversion (742), ~33× v2's knight PST spread.** The
   positional-scale sizing rule (which was right for tempo) did not hold for mobility on accuracy; v2's PSTs are already
   3-10× smaller than SF11's, so measuring against them measured an under-scaled denominator. Corpus fit — a candidate,
   not a verdict.
3. ★★★ **d7 regret gate at 600, vs a same-session neutral (`ASPIRATION_DELTA=300`) on BOTH corpora: +4.9pp primary
   (53.3 vs 48.4), +3.1pp on `_v2` (53.9 vs 50.8).** Replicated, both over the ~2-2.5pp bar; gains in opening/midgame,
   ~neutral in the endgame. n_crit 46 / 31. ⚠️ The v2 neutrals (48.4 / 50.8) differ from each other and from v1's band.
4. **Rook files:** §I gains on 6/6 corpora but small; STS monotone harmful (−72 / −90 / −147 at 20 / 50 / 100). Kept
   OFF.
5. **Registered predictions scorecard:** (1) negative mob × rook-file interaction — refuted on §I (additive), unreadable
   on STS. (2) STS interior peak 60-115 — refuted. (3) +60-150 STS — refuted, the sixth straight optimistic magnitude
   miss. (4) §I < STS signal — refuted, the opposite. (5) nodes fall 5-15% — direction right, −2.4% at 115.

▶️ **Corroboration rule satisfied (two agree, one cannot resolve) ⇒ a games candidate: `sprt_ab` shipped v2 vs
`+MOB_V2_MAG=600`. OWNER'S DECISION — not launched.**

## 2026-09-14 (day) -- ★★★ SLICE 2 MOBILITY PASSES IN GAMES: ≈ +162 Elo pooled (319 games)

Owner chose mobility ALONE (criteria: resolvable solo · contested magnitude only games settle · others sized against
it). `sprt_ab`, LIGHTNING, conc 4, `openings_uho.txt`, elo1 5. A = shipped v2 + `MOB_V2_MAG=600`, B = shipped v2.
Run in two SEGMENTS (paused to build/gate the placement sub-terms), pooled by adding tallies; the shipped v2 WAC
fingerprint `246 / 63,221,361` was re-verified on every build between segments, so both segments are the same arms.
```
seg 1  seed 11   +6   -2  =4  of  12
seg 2  seed 12   +199 -64 =44 of 307  (72.0%)  elo +164.0 +/-45.7  DECISION: H1 accepted
POOLED           +205 -66 =48 of 319  (71.8%)  elo ~ +162
```
★ **Largest single gain of the rebuild** (rung 1 KS +101, rung 2 pawns +60.4) — and in TIMED games, so the extra
per-piece work in the attack loop is already paid for.
★★ **The corroboration rule was right against the instrument that disagreed.** STS read mobility −110 at this very
setting (flat −37..−110 at every magnitude); §I (−5.75%) and the d7 regret gate (+4.9 / +3.1pp over same-session
neutrals, replicated) both said yes. Games sided with the two that could resolve it. ⇒ STS is not a veto on a
move-ordering term whose signal it cannot price.
★ **Sizing lesson confirmed in games:** the design's positional-spread ladder (30-300) would have missed it; the passing
point sits at ~20× v2's knight PST spread, near SF's pawn conversion.
⚠️ SPRT magnitude is ±45 — confidence, not precision. Registered prediction (+60-150 STS) was wrong on its instrument
but the underlying claim "the largest positional term, big enough to read alone" held.
✅ **SHIPPED in the v2 config (`MOB_V2_MAG=600`) on the owner's sign-off.** ☠️ **New shipped v2 fingerprint: WAC d10
`250 / 60,036,533 / EBF 4.043`, STS 1588** — every later byte-identity check uses these, not the pre-mobility
`246 / 63,221,361`. Next: placement bundle on top (§I: bundle B −2.11% mean, better on 6/6; regret gate running); rook
files stay off; tempo re-test trigger fired (first point +107 on STS, inside the floor, ladder queued).

## 2026-09-15 -- SLICE 2 PLACEMENT BUNDLE E: regression SPRT inconclusive with a positive lean; SHIPPED on owner sign-off

Path: single-term §I ladder → bundles (97% additive) → regret (B leaned −2.0 on `_v2`, diffuse under a dilution-controlled
leave-one-out; D clean) → collinearity gate (VIF ≤ 1.30) → per-term FORM ladder vs D (only SF15.1 bad bishop @100 won, 6/6;
our LATENT null) → **bundle E = D + `BADB_V2_FORM=1 BADB_V2_PCT=100`**, collinearity PASS (VIF 1.25) → E regret CLEAN
(primary 49.9 vs neutral 49.8, `_v2` 49.2 vs 50.7) → regression SPRT (elo0 −10, elo1 0, LIGHTNING, conc 4, `openings_uho.txt`).
```
seg 1  seed 13   +103 -83  =36  of  222   LLR +0.804   (killed 03:29 by a Windows Update planned restart)
seg 2  seed 14   +396 -380 =202 of  978   elo +5.7 +/-25.6   LLR +1.091   DECISION: inconclusive (max games)
POOLED           +499 -463 =238 of 1200   (51.5%)  elo ~ +10 (+/- ~23)   LLR ~ +1.9 (segment sum; bound +2.94)
```
Same `.so` both segments (written 09-14 23:53, before launch). Pooled LLR never negative. Segment 1's +31 was small-sample.
⚠️ NOT a formal H1: the evidence is "not harmful, probably mildly positive". Shipped on the owner's sign-off because the regression
bar's purpose (catch harm) is met.
★ Sequencing rule agreed with the owner: every later SPRT runs on the CURRENT shipped base (mobility form bake-off on mobility+E),
so what is gamed is what ships; re-run the collinearity gate and E's §I check if mobility's form changes; one cumulative SPRT at the
end of slice 2 (final slice-2 v2 vs the mobility-only ship) guards against drift from chained regression passes.
☠️ **New shipped v2 fingerprint: WAC d10 `250 / 61,352,373 / EBF 4.114`** (vs mobility+KPK `250 / 60,036,572 / 4.043`: same
solves, +2.2% nodes). Every later byte-identity check uses THIS. Next: mobility form bake-off on this base (neutrals must be
re-measured on it).

## 2026-09-16 -- ★★ PLACEMENT BUNDLE E IS GAMES-CONFIRMED: segment 3 accepts H1 (+15.1 ±21.0)

Continuation of the same A-vs-B pairing (A = SHIP+E, B = SHIP), same bounds, new seed. Poolable because the `.so` fingerprint
was UNCHANGED across the 09-15 rebuild that added the mobility form knobs (`250 / 61,352,373` before and after; all new knobs
default-off and byte-identical).
```
seg 1  seed 13   +103  -83  =36  of  222   LLR +0.804   (killed by a Windows Update restart)
seg 2  seed 14   +396 -380 =202  of  978   elo  +5.7 +/-25.6   LLR +1.091   inconclusive (max games)
seg 3  seed 15   +611 -548 =296  of 1455   elo +15.1 +/-21.0   LLR +3.039   DECISION: H1 ACCEPTED
POOLED          +1110 -1011 =534 of 2655   (51.9%)  elo ~ +13
```
★ **The 09-15 "inconclusive" was a MAX-GAMES STOP, not evidence of nothing.** The same effect resolved once the sample was
~2.2x larger. ⇒ Do not read "hit max_games without crossing a bound" as a null; read it as "not enough games for THIS effect size".
⚠️ The bound is still the regression bar (elo0 −10 / elo1 0): this proves E does not cost ~10 Elo and is consistent with a real
gain of ~+13; it was not designed to prove a gain. Bundle E's five terms (outpost SF11 @100 · bad bishop SF15.1 @100 · trapped
rook @10 · weak queen @25 · minor-behind-pawn Weiss @25) stay shipped, now on games evidence rather than provisionally.
★ Method note for small terms: 1,200 games gave ±23 and could not resolve; 1,455 more gave ±21 on the segment and a formal
accept. The lever that worked was MORE GAMES ON THE SAME PAIRING, not a new instrument.

## 2026-09-16 -- ★★★ THE PARKED SHELF IS WORTH +60 ELO TOGETHER (owner's proposal); four of my verdicts overturned

Owner: *"have you tried putting those terms together? Marginal increases in things like outposts likely only make a difference
in a few games... Bundling things when testing might help."* I had measured each parked term ALONE, below its instrument's
resolution, and had ruled: pin "null on its own class", `exlow` "no measured case, WITHDRAWN from the bundle", bishop pair
"already owned by PST + mobility", space "globally inert".
Cheap check first -- §I additivity (6 corpora, negative = better):
```
all4 (pin+exlow+bpair40+space560lin)  mean -1.70  worst -1.05   better on ALL SIX corpora
mobpair (pin+exlow)                   mean -1.59  worst -0.92   pin -0.51 + exlow -1.08 = EXACTLY additive
pairspace (bpair+space)               mean -0.12  worst +0.10   ~null, as each was alone
```
Then the clearance SPRT, bound FLIPPED to a gain test (elo0 0 / elo1 +10) so it could RETIRE the shelf rather than absorb it:
```
A = SHIP+E + pin + exlow + bpair40 + space560lin   B = SHIP+E   seed 16, LIGHTNING, conc 4
+248 -160 =101 of 509  (58.6%)  elo +60.7 +/-35.5  LLR +3.008   DECISION: H1 ACCEPTED
```
★★ **The lesson, which is about instruments and not about chess: "individually unresolvable" is not "individually worthless".**
Resolving ±5 Elo alone needs ~25,000 games; the same four terms as a group resolved in 509. I had already applied that
arithmetic to placement (bundle E) and failed to apply it to the parked shelf -- and my `exlow` ruling ("a bundle bar would
absorb it silently") inverted the right conclusion: the bundle is how such a term becomes MEASURABLE, provided the bound asks
for a GAIN rather than for harmlessness.
☠️ Also refuted: my "pin and exlow INTERFERE" claim, inferred from ONE sub-bar `_v2` regret reading. On §I they are exactly
additive. A sub-resolution reading is not evidence of an interaction.
⚠️ Bounds on the claim: the +60.7 is "clearly above 10", not a point estimate (an SPRT at elo1=10 crosses early when the truth
is far above it). Attribution is NOT established -- §I suggests pin + exlow carry ~94% of it, but games cannot attribute at
this effect size, so leave-one-out on §I with a dilution control is indicative only.
Proposed to the owner: ship all four AS TESTED; a subset would ship a configuration no game ever saw.

## 2026-09-17 -- THE BUNDLE IS TWO TERMS AND ~+31 ELO, NOT FOUR TERMS AND +60

§I leave-one-out against the full bundle (positive = removing it HURTS): `exlow` **+1.09 / +1.65** · `pin` +0.52 / +0.71 ·
bishop pair +0.11 / +0.34 · space **0.00 / +0.02**. The two members with their own measured nulls are also the two that cost
nothing to remove -- exactly what the dilution control exists to reveal. Resolved by games rather than by argument:
```
s3_mobpair_confirm  seed 17  elo0 0  / elo1 10   +310 -222 =159 of 691   elo +44.5 +/-30.4  LLR +2.980  H1 ACCEPTED
s3_mobpair_bracket  seed 18  elo0 30 / elo1 50   +202 -186  =99 of 487   elo +11.4 +/-36.3  LLR -2.839  H0 ACCEPTED
POOLED                                           +512 -408 =258 of 1178  (54.4%)  elo ~ +31
```
⇒ pin + `exlow` alone carry it; the truth is bracketed at 10 < true < 50 and pools to ~+31.
☠️☠️ **AN SPRT'S POINT ESTIMATE INFLATES AT THE BOUND IT STOPS ON.** One pairing produced +60.7, +44.5 and +11.4 across three
runs, all statistically sound as DECISIONS. Never quote a single run's elo as the magnitude; pool the tallies across seeds.
★ What actually earned the Elo: both surviving terms REFINE MOBILITY'S AREA -- the definition of the largest term v2 owns
(≈ +162). Neither adds a new concept. Sharpening what the biggest term counts beat adding two concepts the references carry.
Proposed: ship `MOB_V2_PIN=1 MOB_V2_EXCL_LOWRANK=1`; retire the pair and space from the candidate list (built, default-off,
triggers intact).

✅ **SHIPPED 2026-09-17 on the owner's sign-off.** New shipped fingerprint **WAC d10 `250 / 59,549,832 / EBF 4.080`** — the
same 250 solves as the pre-area config in **1.8M fewer nodes (−2.9%)**. Bishop pair and space retired from the candidate list
(built, default-off, triggers intact: Kaufman should own the pair; space wants a purpose-built closed-centre corpus).
⚠️ **Framing correction for the record (owner asked, and it matters):** the mobility FORM BAKE-OFF concluded **NO CHANGE** —
SF11's table shape beat SF15.1 / Ethereal / Weiss, and the eg-share and magnitude arms all failed their worst column. `pin`
and `exlow` came from that bake-off's candidate list but were PARKED as §I-only nulls; they are a SEPARATE later result,
produced by the owner's bundling proposal, not by the bake-off.
★ Cumulative v2 ship record: rung 1 KS +101 · rung 2 pawns +60.4 · mobility ≈ +162 · placement E ≈ +13 · mobility area ≈ +31.

## 2026-09-17 -- THREATS: BEST-VERIFIED TERM OF THE SLICE, MOVE-NULL, PARKED; and the KS double-count story REFUTED

Built SF's threat FORM family (7 legs, 2 defence gates, `THREAT_V2_PCT` as the SF-pawn-conversion anchor), riding the existing
attack maps -- no second pass. Gates, in the order slice 3 demanded:
```
oracle   SF gate/core legs            2,005 pos  0 mismatches  fired 38.6%
oracle   Ethereal gate/ALL legs       2,005 pos  0 mismatches  fired 82.3%
oracle   SF gate/ALL legs/XRAY=0      2,005 pos  0 mismatches  fired 94.8%
symmetry full proposed stack          colour swap 0/800; file mirror at the pre-existing 21 @ 5 mp
byte-id  threats OFF                  shipped 250/59,549,832 · v1 250/35,310,778  EXACT
collin.  27 terms, 10,000 pos         NO FLAGS; th_restricted VIF 1.10-1.27, th_king 1.10
§I       11 arms                      NOTHING clears both columns; monotone trade both ways
regret   th10, both corpora           primary 50.1 vs bar 50.1 (0.0pp) · _v2 49.7 vs 50.5 (-0.8pp)
```
⇒ **PARKED, built and default-off.** A term that changes 35-36% of our moves and improves none of them is not
under-measured. §I liked it strongly (`th100` **-9.71%** on the variant corpus, the largest single-corpus gain on this base)
and taxed `lichess_ks_labelled` in proportion (+0.35 at th10 -> +3.90 at th100).
★★ **The KS question the owner raised ("have we infringed on KS's domain?") now has a MEASURED answer: not measurably.**
Built `ks_probe` / `ChessAI.ks_counts` (6 channels per king) and extended the collinearity gate to 27 terms -- closing the
hole open since slice 2. No cross-subsystem flag, on a general 10k sample AND on `lichess_ks_labelled` itself (5k, re-run via
the new `SETS=`, because overlap is a property of a POPULATION). `th_king` -- the most obvious overlap candidate -- reads 1.10.
⇒ Reopening the KS rung has NO evidence behind it; KS stays as shipped at +101 Elo.
★★★ **What the gate STRUCTURALLY could not see, found by the owner's idea of comparing how the giants BALANCE the two:** in
SF/Ethereal king danger is an unbounded quadratic and threats is linear, so KS overtakes threats ~2:1 in severe attacks
(SF 0.5->2+, Ethereal 0.3->2.2). Ours SATURATES at `KS_V2_MAX=4000` = 4.0 pawns while threats at th100 reaches 4.2 ⇒ ratio
0->0.85, never >1. Tested as a 2x2: `th100+ksmax8000` cut the tax **+3.90 -> +2.13** while keeping the general gains.
☠️ NOT ACTED ON: +2.13 is still ~40x the §I floor, the `ksmax6000`-alone CONTROL is itself +0.54 on that column, and acting
would reopen a +101 Elo rung on accuracy evidence alone. `KS_V2_MAX` stays 4000.
★ **Method: VIF measures co-movement of detector COUNTS, not the relative HEIGHT of the scored curves.** A clean gate means
two terms do not measure the same thing -- NOT that they coexist well at their chosen magnitudes. I had conflated the two.
☠️ Three mechanism stories of mine were refuted this week: pin/exlow "interference" (exactly additive), space "helps where the
centre is contested" (the opposite), threats "double-counts KS" (refuted twice). ⇒ [[the-wiring-thesis-was-tested-and-is-unsupported]]
generalises to MY accounts, not just inherited ones.

---

## 2026-09-17 (later) — COLLINEARITY GATE COMPLETE: pawn structure covered, and no probe was needed

**The gate now spans all five scoring subsystems at 40 columns, and it is CLEAN on two populations.** Pawn structure was
the last coverage hole (flagged since slice 2, named as next-step #1 in `SESSION-HANDOFF-2026-09-17.md`).

☠️ **The handoff was wrong about the WORK, not the goal.** It budgeted ~an hour of C++ copying the `ks_probe` pattern.
A record-check first found `pawn_entry_probe` (`eval_v2.cpp:2005`) and `ChessAI.pawn_masks` (`ChessAI.pyx:297`) had
exported every Layer A mask since **2026-09-12** for the rung-2 detector oracle. Only the COLUMN SET in
`_v2_term_collinearity.py` was missing ⇒ a ~30-line tool edit, **no C++ change, no rebuild, zero fingerprint risk,
knob-free columns.** ★ A record-check before writing a tool paid for itself outright. Four documents still said
"the last hole" after the tool already covered it — **the header is not the record, in the pessimistic direction too.**

| run | positions | terms | strongest CROSS-subsystem pair | flags |
|---|---|---|---|---|
| four general corpora | 10,000 | 40 | `mob_table_mg x ks_natt` **-0.41** | **none** |
| `lichess_ks_labelled` | 5,000 | 40 | `th_restricted x ks_natt` **+0.33** | **none** |

⇒ **Pawn structure does not re-express mobility, placement, threats or king safety, and vice versa.** Overlap is a
property of a POPULATION, so the KS-critical corpus was re-run separately rather than trusted to generalise.

★★ **THE REGISTERED PREDICTION IT REFUTED — the third and strongest instance of "sharing an INPUT is not sharing a
SIGNAL".** Two pairs were written into the tool beforehand as the ones we *knew* were wired to a shared map:
`ps_pattacks x mob_*` (mobility's area SUBTRACTS enemy pawn attacks) and `ps_halfopen x traprook_units`
(`trap_rook_units` literally TAKES `halfOpen` as a parameter, `eval_v2.cpp:1630`). The tool's own comment said "a high
|r| here is EXPECTED". Measured: **-0.01 to +0.07** and **-0.07**. One term consuming another's output as an input
predicts essentially nothing about count co-movement. Three instances now: KS<->threats VIF 1.10 (09-17) ·
mobility area <-> pawn attacks 0.01 · trapped rook <-> halfOpen 0.07. ☠️ It cuts BOTH ways: a shared-input story is not
evidence of double-counting, and a clean gate is still not evidence of safe coexistence (the curve-height limit stands).

**Two new instrument gaps, both recorded in `INSTRUMENT-MAP.md` §F:**
1. ☠️ **A SYMMETRIC predicate is invisible to W-B differencing.** `blocked` is omitted because `blocked[White]` and
   `blocked[Black]` are the two halves of the same RAM pairs (`eval_v2.cpp:682`) ⇒ popcounts always equal, difference
   identically 0. So "does space or mobility re-express the RAMMED centre?" is unmeasurable here — and `centre_locked`
   is precisely the class where space showed its only effect. ★ `lever` ALSO read zero-variance at N=25 but is not an
   identity (one pawn attacked by two gives 1 vs 2) and varies fine at N=2500 ⇒ **check a zero-variance column against
   its DEFINITION before deleting it.** One right call, after two wrong ones.
2. ⚠️ **A differenced count carries the CENSUS.** Against the raw pawn-count difference: `ps_halfopen` **-0.93**,
   `ps_pattacks` **+0.91**, `ps_passed` **+0.83** (replicated -0.92 / +0.88 / +0.70 on the KS corpus). Their mutual VIF
   of 7.8-18.6 was ambiguous between "shared structure signal" and "both restating rung 0" until a `ps_npawns` control
   column was added; the tool now prints a `material share` block. **Any collinearity read over count columns needs a
   census control, or subsystem overlap and material overlap cannot be told apart.**

⚠️ Still uncovered: `PawnEntry.attacks2` (double pawn attacks, `eval_v2.cpp:657`, not among the probe's 27 slots) ⇒
"does threats' `stronglyProtected` re-express the double-attack map?" needs a probe change + rebuild; do it only if a
run leaves that pair open. Nothing built, nothing committed; `.so` untouched, so the shipped fingerprint is unchanged.

---

## 2026-09-17 (checkpoint, part 1) — THE v2 MARGIN RE-SWEEP: RFP measured for the first time under EVAL_ARM=1

**Why this was next:** the owner's showdown fairness rule (`EVAL-V2-CURRENT-CONFIG.md` §5) — every eval-denominated
pruning threshold is absolute millipawns FITTED TO v1, so a v1-vs-v2 showdown on v1's margins measures v1's TUNING as
much as v2's eval. ★ **Every prior margin sweep in the record ran `EVAL_ARM=0`. This is the first arm-1 measurement.**

**Fingerprints reverified first, both EXACT:** v1 `250 / 35,310,778 / EBF 3.784` · v2 `250 / 59,549,832 / EBF 4.080`.
The node judge also reproduced its documented baseline exactly (v1 @ RFP=1500 = **249,014** quiet-node median), which is
what licenses reading anything else off it.

**Instrument choice, and why not the obvious ones.** Quiet-node median at fixed depth (`depth_nps_bench --n 60
MAX_DEPTH=10 LONG_FORMAT`) for cost, WAC solves for tactical safety. ☠️ NOT STS: its ±150 floor swallowed 25 of 25
cells of the 09-05 sigma x RFP sweep, whose own author wrote "do not quote these numbers". ☠️ NOT WAC nodes ALONE:
four of six search configs reversed sign against the quiet corpus — here both corpora were run and agreed in sign.
★ Fixed depth throughout ⇒ deterministic, so the whole sweep is valid while the machine is in use.

| RFP_MARGIN | v1 quiet nodes | v2 quiet nodes | v2 WAC solves | v2 WAC nodes | vs shipped | v2 EBF |
|---|---|---|---|---|---|---|
| 400  | 163,437 | 319,679 | **241** ✗ | 41,767,917 | -29.9% | 3.963 |
| 600  | 173,377 | 340,512 | - | - | - | - |
| 800  | 197,224 | 399,288 | **248** ✗ | 47,783,648 | -19.8% | 4.007 |
| **1000** | 216,442 | 393,392 | **250 =** | **49,440,513** | **-17.0%** | 4.031 |
| 1250 | 228,358 | 403,758 | 253 | 53,476,787 | -10.2% | 4.048 |
| **1500 (shipped)** | **249,014** | 474,852 | **250** | **59,549,832** | - | **4.080** |
| 2200 | 286,017 | 467,323 | 252 | 65,385,526 | +9.8% | 4.100 |
| 6000 | 348,130 | 525,397 | - | - | - | - |

**Liveness PASSED on both arms** (v1 2.13x, v2 1.64x across the range) — the record's rule is to prove an extreme moves
the node count before believing any sweep.

### ★★ THE RESULT IS A DE-RISKING, NOT A WIN
**v2 can have 17.0% of its nodes back at IDENTICAL tactical safety** (RFP 1000: 250 solves, same as shipped). The
handicap is therefore REAL and now measured. ☠️ **But 17% sits well below the documented ~35% bar at which node savings
become Elo-visible** ([[node-savings-below-35-percent-are-elo-neutral-dont-scale-from-a-bundle]]), and the 1250 point is
-10.2%. ⇒ **v2's RFP handicap is worth LITTLE — a few Elo, not tens.** The showdown was being held hostage to a fairness
correction that measures small. Play it twice as the rule requires, but expect a NARROW spread between the two runs.
⇒ The break point is between 800 and 1000: solves intact at 1000 (250), wobbling at 800 (-2), broken at 400 (-9).
⇒ Remaining genuine trade, unresolvable at fixed depth: **1000** (250 solves / 49.44M) vs **1250** (253 / 53.48M).
Both go to the timed SPRT against 1500. ☠️ A margin change is a NODE-SAVER ⇒ judged at FIXED TIME; the fixed-depth
curves can bound cost and VETO unsafe margins, never pick the winner.

### ★ MECHANISM CONFIRMED (a prediction that held)
The shipped fingerprints' cutoff histograms showed v2's +68.6% nodes were NOT an ordering failure — v2's first-move
cutoff share is **90.2%** vs v1's **86.5%** — and that the excess was almost entirely in nodes cutting off IMMEDIATELY
(m0 +74%, while m3-7 +2.4% and m8+ +1.6%). A node entered and then cut off on its first move is exactly what RFP exists
to remove BEFORE entry, so the signature predicted under-pruning. Tightening RFP to 400 cut m0 2,747,142 -> 1,802,870,
landing near v1's 1,582,788 (+74% -> +13.9%). ⇒ The m0 excess WAS under-pruning. The 9 lost solves say some of those
nodes were doing real tactical work, which is why the optimum is INTERIOR rather than at the tightest margin.

### ☠️ TWO CORRECTIONS TO MY OWN CLAIMS THIS SESSION
1. **RETRACTED: "v2's RFP response is non-monotone."** I read two inversions off the quiet-corpus MEDIANS (800->1000
   -1.5%, 1500->2200 -1.6%) and flagged that a median over 60 positions can invert while the mean does not. WAC's node
   count is a deterministic SUM and is monotone increasing straight through 2200 ⇒ **median-selection artifact.**
   ★ A deterministic statistic is not automatically a trustworthy SHAPE; a median can move for a reason the mean cannot.
2. **WITHDRAWN: "1250 scores better (253 vs 250 solves)."** Solves across 1250/1500/2200 read 253/250/252 — exact
   (WAC is deterministic) but not MEANINGFUL, since which positions resolve is near-arbitrary with respect to strength
   and WAC does not discriminate strength at all (v1 beats SF11 on it). The correct reading: **solves are FLAT from 1000
   up while nodes fall monotonically as the margin tightens ⇒ tighten until solves BREAK.**

### ⚠️ BOUNDS ON THIS RESULT
Only `RFP_MARGIN` was swept, deliberately (it is the one margin with a monotone response on both sides; futility and
razor measured non-monotonic twice, and root razoring keys on PRE-SEARCH SCORES, not the static eval). The other live
eval-denominated levers — `QDELTA_PERMOVE_MARGIN` (never swept under EITHER eval), `FUTILITY_MARGIN_SCALE`,
`ASPIRATION_DELTA`, `VERIFY_MARGIN` — are untouched for v2, so COMBINED recovery could exceed 17%. Each gets its own
pass, then a 2x2 before any bundling (search changes are antagonistic, not additive). The ~35% Elo-visibility bar is
itself v1-era work, so applying it to v2 is an extrapolation, not a measurement.

### ☠️ DEAD KNOB FOUND, AND A MECHANISM STORY CORRECTED BECAUSE OF IT
`DELTA_MARGIN` is **dead at defaults via two independent mechanisms** across its three consumers (verified in source):
`search_engine.cpp:7673`/`:7685` are gated on `!ENABLE_QDELTA_PERMOVE`, which SHIPS TRUE; `:7717` reaches it only as a
fallback when `QDELTA_PERMOVE_MARGIN == 0`, and that ships 1500. The `[toggles]` dump PRINTS it (`:2512`) and never
prints the live knob. ⇒ **The parked tempo item's stated channel list was wrong for two of three thresholds** — it named
`RFP_MARGIN`, `DELTA_MARGIN` and `OTV_MARGIN`, and the latter is behind `ENABLE_OTV = false`. Tempo's step-shaped node
READING stands (it is an instrument observation); only its channel list changes, and the checkpoint re-test must run
against the LIVE list. Corrected in `EVAL-V2-CURRENT-CONFIG.md` and in memory.
★ **A knob named in a mechanism story must be proven LIVE before the story is trusted; the toggles dump is not proof.**

### ☠️ OPS FAILURE THIS SESSION
I edited `overnight_runner.sh` WHILE the sweep was executing out of that file, changing the length of a comment block
inside the running branch — bash reads scripts by byte offset, so that can corrupt a job mid-loop. Killed the run,
confirmed no orphaned python child (relaunching over one doubles the core load), verified the file parses, relaunched.
Cost ~1 minute. ★ The relaunch reproduced v1 @ RFP=400 = **163,437** byte-identically, which independently confirmed the
fixed-depth determinism and stable corpus seed the sweep depends on. ⇒ "Docs are safe to edit while jobs run" does NOT
extend to the dispatcher, which is simultaneously a document and a running program.

---

## 2026-09-17 (checkpoint, part 2) — the rest of the live margin family under EVAL_ARM=1

Continuing part 1, one knob at a time with the others at shipped values (☠️ never bundled: search changes are
antagonistic, not additive, so a 2x2 comes BEFORE any combination). All fixed depth ⇒ deterministic, machine-use safe.

### `QDELTA_PERMOVE_MARGIN` — NON-LEVER for v2 (and its first-ever sweep under EITHER eval)
| value | solves | WAC nodes | vs shipped | EBF |
|---|---|---|---|---|
| 750 | **248** ✗ | 57,672,096 | -3.2% | 4.081 |
| **1500 (shipped)** | **250** | **59,549,832** | - | 4.080 |
| 2500 | **249** ✗ | 59,161,690 | -0.7% | 4.044 |

Both directions COST solves, the node response is <= 3.2%, and it is NON-MONOTONIC (2500 sits BELOW 1500, which is
backwards for a loosened prune margin). ⇒ The documented non-lever signature. ★ The shipped 1500 was "a single value
picked when the feature shipped, never swept" ([[toggles-dump-advertises-dead-knobs]]) — it now survives its first test,
and the open item closes. ⚠️ This is the knob that MATTERS in qsearch; `DELTA_MARGIN` is the dead one the toggles dump
advertises (part 1).

### `FUTILITY_MARGIN_SCALE` — WEAK lever for v2, monotone, and the v1 verdict does NOT carry over cleanly
| value | solves | WAC nodes | vs shipped | EBF |
|---|---|---|---|---|
| 70 | **250 =** | 58,372,220 | **-2.0%** | 4.064 |
| **100 (shipped)** | **250** | **59,549,832** | - | 4.080 |
| 130 | 253 | 61,550,846 | +3.4% | 4.079 |

Node response is MONOTONE here (unlike v1, where 08-25 and 09-05 both measured it non-monotonic and closed it as "not a
lever") — but the whole span is only ±3.4%. ⇒ 70 is a free -2.0% at identical solves; nothing here is worth games on its
own. ★ Same discipline as part 1: the +3 solves at 130 is NOT a strength claim (WAC does not discriminate strength).

### ▶️ THE COMPLETE FIXED-DEPTH MARGIN PICTURE FOR v2
| knob | range | node response | solve behaviour | verdict |
|---|---|---|---|---|
| **`RFP_MARGIN`** | 400-6000 | **-29.9% .. +9.8%** | breaks below 1000 (248 @ 800, 241 @ 400) | ★ **THE lever: -17.0% FREE at 1000** |
| `QDELTA_PERMOVE_MARGIN` | 750-2500 | <= 3.2%, non-monotonic | both sides cost solves | non-lever |
| `FUTILITY_MARGIN_SCALE` | 70-130 | ±3.4%, monotone | flat to +3 | weak; -2.0% free at 70 |
| `ASPIRATION_DELTA` | - | not swept for v2 | - | ⚠️ OPEN (tactical<->positional trade knob; alone it flips 20.8% of quiet moves) |
| `VERIFY_MARGIN` | - | not swept for v2 | - | ⚠️ OPEN |
| ~~`DELTA_MARGIN`~~ | - | - | - | ☠️ DEAD at defaults (part 1) |
| ~~`RAZOR_*`~~ | - | - | - | ☠️ NOT eval-denominated: keys on PRE-SEARCH scores |

★★ **CONCLUSION, and it de-risks the showdown.** The free, tactically-safe recovery is **RFP 1000 (-17.0%)** plus
**futility 70 (-2.0%)** — call it ~19% IF they compose, which must be checked with a 2x2 and must not be assumed.
☠️ **Even 19% is well below the ~35% bar at which node savings become Elo-visible**
([[node-savings-below-35-percent-are-elo-neutral-dont-scale-from-a-bundle]]). ⇒ **v2's margin handicap is REAL,
now MEASURED for the first time, and SMALL — a few Elo, not tens.** The owner's fairness rule was right to demand the
re-sweep, and the answer is that the correction it protects against is minor. Play the showdown twice as the rule
requires, but expect a NARROW spread; do not hold the showdown hostage to this lane.

▶️ **What still needs a QUIET WINDOW (timed, cannot be done at fixed depth):** a margin change is a NODE-SAVER ⇒ judged
at FIXED TIME. The decision SPRT is v2 @ RFP=1500 vs v2 @ RFP=1000 (and the 1250 alternative), plus the NPS pair, then
the showdown twice. The fixed-depth work above can bound cost and VETO unsafe margins; it can never pick the winner.

---

## 2026-09-17 (checkpoint, part 3) — v1-vs-v2 SHOWDOWN: interpretation REGISTERED IN ADVANCE

☠️ **Written while run A is still playing, deliberately.** My prediction record this phase is 2 right / 6 wrong,
and the failure mode is explaining a number after seeing it. So the thresholds below are committed BEFORE the
result, and the owner's framing is recorded as agreed: **this is an INFORMATION CHECKPOINT, not a hard stop.**

**The matchup.** p1 = shipped v2 (KS + pawns/passers + draw/KPK + mobility + placement E + mobility area),
p2 = `EVAL_ARM=0` (frozen v1). Same binary, one knob apart. LIGHTNING, conc 4, `openings_uho.txt`, seed 7
(NOT 0 — a known read-inflater), elo0=0 / elo1=5, max 1500. ★ **First v1-vs-v2 games measurement ever taken.**

**Why a deficit would NOT be a verdict on the rebuild:** v2 is competing TWO SLICES SHORT. v1 additionally
carries capture gains, its own threats, central, Kaufman imbalance, OvD, winnability, rook files and king
shelter. ★ Capture gains is the big one: v1's sibling spread is ~2,290 mp against v2's 36 mp, almost entirely
from that term, and it is a SLICE-5 item in v2.

⚠️ **What must NOT be used as the excuse: the margins.** Measured earlier today (part 1), the maximum
`RFP_MARGIN` recovery for v2 is **-17.0% nodes at identical solves**, which is BELOW the ~35% Elo-visibility
bar. ⇒ v1-fitted search parameters are worth SINGLE-DIGIT Elo here, not tens. The EBF gap (4.080 vs 3.784) is
a SYMPTOM of a thinner eval crossing absolute margins less often, not the cause of a strength gap. **Run B
(`RFP_MARGIN=1000`, same matchup) measures the handicap directly as A-minus-B; quote THAT, never an argument.**

**REGISTERED THRESHOLDS (pooled tally, not the SPRT point estimate — [[sprt-point-estimates-inflate-at-the-bound-they-stop-on]]):**
| outcome | reading | consequence |
|---|---|---|
| v2 **>= -20** | competitive two slices early | strong; slices 4-5 should pass v1 outright |
| v2 **-20 .. -80** | ~one slice of missing content | on track; continue the ladder as planned |
| v2 **-80 .. -150** | more missing than the ladder accounts for | re-examine whether a PARKED term (threats/space) is load-bearing in GAMES despite being move-null on §I |
| v2 **< -150** | structural, not missing terms | stop adding slices; audit v2 for a defect no static instrument caught |
★ At 109 games the running estimate was -42 with a 2-sigma band of roughly -107..+23, so nothing below is
decidable yet. ⚠️ The early trajectory (-800 -> -226 -> -79 -> -42) is REGRESSION TO THE MEAN from a 4-loss
opening streak, NOT v2 improving; the estimate converges from wherever the first few games put it.
⇒ Letting it run to the full 1500 rather than stopping at a bound: the MAGNITUDE matters more than the verdict,
and a max-games stop gives the better tally to pool with run B and with later seeds.

---

## 2026-09-17 (checkpoint, part 4) — ☠️ SCOPE CORRECTION to parts 1-3: "the handicap is small" covers ONE FAMILY

Owner's push-back, and it is correct: *"that's just one search item... certain search parameters might just not be
jelling with v2 as well."* Parts 1-3 concluded "v2's margin handicap is real but SMALL (a few Elo, not tens)".
⚠️ **That claim is bounded to the EVAL-DENOMINATED MARGIN FAMILY and must not be read as bounding the search.**

**What parts 1-2 actually measured under `EVAL_ARM=1`:** `RFP_MARGIN` (the lever, -17.0% nodes free) ·
`FUTILITY_MARGIN_SCALE` (weak, -2.0% free) · `QDELTA_PERMOVE_MARGIN` (non-lever). These share a mechanism: a
millipawn eval compared against a millipawn threshold, so v2's different eval SCALE is the whole coupling.

**What it did NOT measure, and where the owner's hypothesis lives:**
- `ASPIRATION_DELTA`, `VERIFY_MARGIN` — eval-denominated and STILL UNSWEPT for v2.
- ☠️ **Everything NOT denominated in millipawns**: LMR shape/product, LMP, root/late razoring, null-move R,
  presearch ordering and chunking, history / continuation-history / capture-history scaling, IIR, TT policy.
  These key on DEPTH and on MOVE ORDERING, not on eval magnitude — but ordering quality is DRIVEN by the eval, so
  a knob fitted to v1's cutoff distribution can mis-serve v2 while being invisible to a margin sweep.
  ★ Concrete evidence they are not inert: v2's cutoff PROFILE differs measurably from v1's (first-move cutoff share
  **90.2% vs 86.5%**, and the node excess is concentrated at m0: +74% vs +1.6% at m8+). A different cutoff profile is
  exactly the input those knobs consume.
⇒ **Corrected statement: the eval-denominated MARGIN family is worth single-digit Elo to v2. The broader
search-parameter interaction is UNMEASURED for v2 and remains an open lane.**

⚠️ **Prior, and why it is only a prior:** for v1 the pruning-knob space was closed hard — ~35 arms + 52 configs,
nothing survived, and "of 52 configs only three values of depth@1s differed from 12 ⇒ the whole pruning-knob space
cannot buy a ply". ☠️ **Every one of those runs was `EVAL_ARM=0`.** A closed lane on v1's eval is not a closed lane
on v2's, for precisely the reason this whole checkpoint exists.

▶️ **Owner's sequencing, recorded as agreed: finish the v2 EVAL first, then a v2 SEARCH programme** (search +
caching + movegen, alongside the UCI pure-C++ reorganisation). ★★ The strategic argument for that order is already
in the record and is strong: **search and movegen carry into NNUE 1:1; the eval does not**
([[pre-nnue-strength-roadmap]]). So search work is DURABLE investment against the owner's NN endgame, while eval
work is the thing that gets replaced. ⇒ Do not let a broad search sweep pre-empt finishing the eval slices, but do
not record the search lane as closed for v2 either.
▶️ Cheapest honest screen when that lane opens: the node judge (quiet median) + depth@1s per arm under
`EVAL_ARM=1`, on the knobs whose INPUT is the cutoff profile, run as 2x2s rather than one-at-a-time
([[search-changes-are-antagonistic-not-additive]], and [[coordinate-descent-cannot-find-gated-mechanisms]] —
a knob whose effect is conditional on another is invisible to a one-at-a-time sweep).

---

## 2026-09-18 — ★★★★ THE v1-vs-v2 SHOWDOWN, RUN A: v2 IS LEVEL WITH v1

**The first v1-vs-v2 games measurement ever taken.** p1 = shipped v2 (KS + pawns/passers + draw/KPK + mobility +
placement E + mobility area), p2 = `EVAL_ARM=0` (frozen v1). Same binary, one knob apart. LIGHTNING, conc 4,
`openings_uho.txt`, **seed 7** (not 0 -- a known read-inflater), elo0=0 / elo1=5, max 1500.

```
+652 -693 =155 of 1500   score 48.6%   elo -9.5 +/- 20.7   LLR -0.832
DECISION: inconclusive -- hit max_games without crossing a bound  (INTENDED: the tally, not the verdict)
```
⇒ **95% CI approximately [-30.2, +11.2]. v2 and v1 are statistically INDISTINGUISHABLE.**

**Against the thresholds registered BEFORE the run (part 3):** the point estimate **-9.5 lands in the top bracket**
("v2 >= -20: competitive two slices early ⇒ strong; slices 4-5 should pass v1 outright"). ⚠️ Honest caveat: the CI's
lower bound (-30.2) reaches into the second bracket ("-20..-80: ~one slice of missing content"), so the top-bracket
reading is the point estimate's, not the interval's. ⚠️ ONE SEED. The pooling rule exists because a single run is a
single draw; a second seed is needed before this is quoted as settled.

### ★ WHY LEVEL IS THE INTERESTING RESULT -- what v2 achieves it WITHOUT
- **Two whole slices unbuilt**: capture gains, corrhist, OvD (slice 5) · endgame conversion + winnability (slice 4) ·
  Kaufman + pairs (slice 3's last item). v1 additionally carries its own threats, central, rook files, king shelter.
  ★ Capture gains is the striking absence: v1's sibling spread is ~2,290 mp against v2's 36 mp, almost all of it that
  one term. **v2 reaches parity without the single largest signal v1 owns.**
- **Search margins fitted to v1's eval scale** (measured worth: ~17% of nodes, below the ~35% Elo-visibility bar).
- **No joint retune, ever.** Every v2 constant came from reference shape + a one-at-a-time ladder.
- ★ And it does it while being **colour-clean (0/4000)** where v1 is not (0.9%, worst 2,233 mp), and **provably
  non-collinear** on the 40-column gate where v1 is ~30 terms / ~2 signals.
⇒ Same strength, structurally cleaner ⇒ the HEADROOM differs in kind. v1's additions cancelled ~26%; v2's have
measured 97% additive (placement bundle) and exactly additive (pin + exlow).

☠️ **What this does NOT license.** "Level with v1" is a milestone, not the target -- the roadmap target is SF11 /
SF15-classical. On static accuracy we are at ~2.6x SF11's win%-MSE error (v1, 08-07: ours 245.46 val vs SF11 95.26,
against an irreducible floor of ~69 because SF18 SEARCHES). ⚠️ **v2 has never been placed on that ladder** -- the
cheapest high-value measurement outstanding, and `_reference_ceiling.py` already accepts `CORPUS=`.
⇒ NEXT: margin A/B launched immediately (v2 @ RFP 1000 vs shipped, **regression framing** elo0=-10/elo1=0 -- the
question is whether the 17% node saving is FREE, not whether it gains); then a second showdown seed to pool.

---

## 2026-09-18 — MARGIN A/B: `RFP_MARGIN=1000` IS FREE (H1 accepted) ⇒ 17% of v2's nodes back at no cost

Paired within-v2 A/B, the properly-powered replacement for the A-minus-B design I had proposed (two showdowns
differenced carry ~±34 Elo at 2σ — arithmetic that cannot see a single-digit effect).
p1 = shipped v2 + `RFP_MARGIN=1000`, p2 = shipped v2. LIGHTNING, conc 4, `openings_uho.txt`, seed 11.
★ **REGRESSION framing** (elo0=-10 / elo1=0): the question is not "does tightening GAIN Elo" — the fixed-depth work
predicted below the visibility bar — but **"is the 17% node saving FREE?"**

```
+591 -531 =353 of 1475   score 52.0%   elo +14.1 +/- 20.8   LLR +3.079
DECISION: H1 accepted -- P1 is >= elo1 (i.e. NOT a regression)
```

☠️ **DO NOT QUOTE +14.1 AS THE MAGNITUDE.** This run STOPPED ON THE UPPER BOUND, which is precisely the condition
under which the point estimate is biased upward ([[sprt-point-estimates-inflate-at-the-bound-they-stop-on]] — the
same lesson that turned +60.7/+44.5/+11.4 into a pooled ≈ +31). What is established is the BOUND: `RFP_MARGIN=1000`
does not cost ~10 Elo, and formally reads >= 0. Bracketing the true size needs a second bound or pooled seeds.

⚠️ **My prediction survives, but only because the bar was low.** I predicted ~0 on the grounds that a 17% node
saving sits below the ~35% Elo-visibility bar. H1's bar was >= 0, so accepting it is consistent with a true value
anywhere from 0 upward — the test asked "is it free?", NOT "is it > 0?". ⇒ The 35%-bar extrapolation is neither
confirmed nor refuted here. Do not cite this run as evidence that node savings below 35% DO pay.

### ▶️ RECOMMENDED SHIP — with one implementation constraint that matters
Fixed depth: **250 solves (identical to shipped) at -17.0% WAC nodes**; quiet-node median -17%. Games: not a
regression. Plateau check ([[swept-knob-needs-plateau-check]]): the neighbours are healthy — 1250 gives 253 solves
at -10.2% nodes, 800 gives 248 (-2) at -19.8% ⇒ the usable plateau is **1000-1250**, and 1000 is its node-cheapest
point at NO solve cost.
☠️ **`RFP_MARGIN` IS A GLOBAL SEARCH KNOB, SHARED WITH v1.** Changing its DEFAULT would change the frozen v1
control arm and invalidate the `250 / 35,310,778 / EBF 3.784` fingerprint. ⇒ It must be added to the **v2 env
config block** (`EVAL-V2-CURRENT-CONFIG.md` §1), NOT to `search_engine.h` defaults. Every v2 measurement from here
carries it; v1 keeps 1500.
⚠️ Consequence for the record: shipping it moves v2's fingerprint (nodes fall ~17%), so a NEW register line is
required and byte-identity against `250 / 59,549,832 / EBF 4.080` will correctly fail.

---

## 2026-09-18 — ★★★★ v2 ON THE REFERENCE-ACCURACY LADDER: 245.46 -> 155.72, **59.7% of the v1->SF11 gap closed**

`_reference_ceiling.py N=3000` under the shipped v2 config — the FIRST time v2 has been placed on this ladder
(every prior number was v1, 2026-08-07). Static eval vs **SF18-SEARCH** labels, win%-squared error, lower better.

| static evaluator | train | val |
|---|---|---|
| SF15.1 NNUE | 67.60 | **61.61** |
| SF18 static | 64.84 | 68.85 |
| **SF11 classical** | 101.68 | **95.26** |
| SF15.1 classical | 141.44 | 139.20 |
| **OURS — v2 (shipped)** | **167.92** | **155.72** |
| ours — v1 (08-07) | 254.38 | 245.46 |

★★ **The run VALIDATES ITSELF: every reference row reproduced the August values EXACTLY** (SF11 101.68/95.26,
SF15.1c 141.44/139.20, SF18-static 64.84/68.85, SF15.1-NNUE 67.60/61.61). Same corpus, same 3,000-row HEAD slice,
same loss — only our arm changed. ⇒ a clean like-for-like against the recorded ladder, not a re-baselining.

**Arithmetic:** v1 -> v2 val error **-36.6%**. Gap to SF11 closed: (245.46-155.72)/(245.46-95.26) = **59.7%**.
Against the irreducible floor (68.85 — SF18 SEARCHES, so no static eval reaches 0): v1 carried **6.7x** SF11's
above-floor error, v2 carries **3.3x**. Roughly halved. v2 now sits only **16.5 points above SF15.1-classical**.
★ Note again that SF15.1c (139.20) is WORSE than SF11 (95.26) here — **"SF11/15 levels" is not one target**, and
SF11's 95.26 is the hand-reachable one.

### ☠️☠️ THE FINDING IS THE JUXTAPOSITION, NOT THE NUMBER
The same shipped v2 measured **elo -9.5 +/- 20.7 vs v1** in 1500 games the same day. ⇒ **v2 is 36.6% more accurate
than v1 and only LEVEL with it in games.** A third of our static error vanished without moving move choice. That is
[[most-eval-error-is-move-neutral]] at full scale, and it is the cleanest instance we have ever measured.
⇒ ☠️ **Do NOT present this as evidence the rebuild is winning on strength.** It is not. It is also a fresh warning
against reading §I as a strength proxy ([[corpus-fit-is-anti-correlated-with-elo]]).
★★ **Where it DOES land hard: the owner's TEACHER framing.** If the eval's destiny is to label positions for their
NN, static accuracy is the DIRECT metric rather than a proxy ([[the-hce-is-the-nnue-teacher-so-eval-carries-informationally]]).
By that measure v2 is already a far better teacher than v1 — **-36.6% error AND colour-clean 0/4000** where v1
mislabels 0.9% of positions by up to 2,233 mp — while being no worse a player. ⇒ The rebuild's payoff to date is
concentrated in TEACHING QUALITY and STRUCTURE, not in playing strength.

⚠️ Bounds: static accuracy only, one corpus (`diverse_corpus_wide`, d13 single-PV — do NOT pool with the d14
multi-PV regret sets), and the 3,000-row head slice is not a random sample (the recorded caveat since 08-07; the
full-corpus re-run remains proposed-only). Kaufman/corrhist/OvD/winnability are still unbuilt and no joint retune
has ever been run, so this is not v2's ceiling.

---

## 2026-09-18 — PER-CLASS ACCURACY LADDER: predictions REGISTERED BEFORE THE RUN

Owner's question: *where* did the 36.6% accuracy gain come from, and where is the remaining work? The ladder run
gives a single number — `_reference_ceiling.py` has NO stratification — so this is the per-class version.

**Setup (composition, no new tool).** `_position_class.py SETS=ks_sets/diverse_corpus_wide.csv OUT=ks_sets/classes_wide`
classified all **23,113** rows: `centre_open` 6,328 (27.4%) · `centre_cleared` 3,682 (15.9%) · `centre_tension`
1,776 (7.7%) · `centre_locked` **699 (3.0%)** · `other` 10,628 (46.0%) · `pin_dense` 3,966 (17.2%).
✅ **Schema verified BEFORE running** — the class files carry `fen, target_total, split, phase_bucket, tier`, so
`_reference_ceiling.py CORPUS=` will work. ☠️ This check was not optional: the tool's `try/except ... continue`
swallows a missing column and prints `nan`, the silent-fallback signature. The regret-set class corpora would have
failed this way (they carry `best_cp` in centipawns, no `target_total`).

**REGISTERED PREDICTIONS (low confidence, stated because my last class-level directional call was exactly backwards
— I predicted `centre_tension` would hold our largest error and it was the SMALLEST):**
1. Our gap to SF11 will be **LARGEST on `centre_open`** — most piece activity and tactics, the regime where the
   unbuilt capture gains and the parked threats should hurt most.
2. **SMALLEST on `centre_cleared`** — few pawns, more endgame-like, where v2's material + exact KPK + passers are
   its strongest owned terms.
3. `centre_locked` (699 rows, 3.0%) is **too thin to resolve** — it is the class where space showed its only effect,
   and it was already flagged as unresolvable at 1,974 rows. Expect to report it as unreadable, not as a finding.
⚠️ Guard against the slicing trap: six classes plus a tag is seven reads, so an extreme by chance is likely. Only a
gap that is LARGE and has a mechanism gets treated as a finding.

---

## 2026-09-18 — CONSOLIDATED: where v2's ACCURACY came from vs where its ELO came from (they disagree)

Owner's question: which terms account for the gain, and where is the remaining work? Everything below was already
recorded — this is a consolidation of the scattered rung ladders, not new measurement. Full per-term table with
file:line citations was built this session; the load-bearing summary:

### ☠️ THE TWO RANKINGS INVERT
| by §I accuracy (on its own base) | by measured ELO (games) |
|---|---|
| material taper **-6.59** (PARKED, undecided) | mobility **+162** |
| mobility -5.75 | KS **+101** |
| threats **-3.80** (PARKED, move-null) | pawns **+60.4** |
| placement bundle -2.11 | mobility area **+31** |
| mobility area pair -1.59 | placement E **+13** |
| **KS -1.44** | — |
| pawns -0.97 · passers -0.72 · structure -0.66 | — |
| rook files -0.64 (OFF: STS harmful) · bishop pair -0.11 · space ~0.00 | — |
☠️ **Those §I figures sit on FIVE DIFFERENT BASES and must never be summed.**
★★ **KS reads -1.44% on accuracy and delivered +101 Elo; mobility's STS read -110 at the very setting that won
+162.** ⇒ **§I is our best accuracy instrument and a POOR PRIORITISER.** Anything chosen by §I magnitude alone
would have picked the material taper and threats — both parked — over KS.

### ⇒ ANSWER TO "IS KS OFF?": NO. KS is accuracy-modest and Elo-huge.
Our accuracy deficit is not in king safety. The largest identified chunk is **capture gains**: the entire
**+138.69%** variant-corpus column vs v1 is a recorded capgains artifact (`SCALE_CAPTURE_GAINS=0` alone on v1 reads
+176.89% on that column), and capgains is unbuilt in v2 (slice 5).

### ☠️ WHAT IS NOT RECORDED (the honest gaps in the breakdown the owner asked for)
1. **No PER-PHASE §I split exists for ANY term.** The only phase breakdowns belong to the d7 REGRET gate
   (e.g. mobility +6.7 / +4.8 / +2.7pp opening/mid/end). Per-phase accuracy is NEW WORK.
2. **Per-class §I exists for only FOUR arms** (`pin`, `exlow`, bishop pair, space). Nothing for KS, pawns, mobility
   core, placement or threats.
3. **No §I number at all for:** tempo (deliberately skipped), exact KPK (WAC nodes + oracle only), central and
   Kaufman (never built).
4. The draw classifier's §I is a **measured zero** (identical on all six corpora), not an absence.
5. ⚠️ The "WORST column" is a DIFFERENT CORPUS by rung — variant/960 at rung 1, `lichess_ks_labelled` for every
   slice-2/3 term ⇒ worst-column figures are not comparable across rungs.

### ▶️ Still owed an accuracy number (all unbuilt): Kaufman + pairs · capture gains · corrhist · OvD · winnability
### · convertibility scale · mate drive · king shelter.
★ Recorded per-class BASE MSE (ours, regret-set corpora — ☠️ a DIFFERENT corpus from the 155.72 ladder, do not
mix): `pin_dense` **413.74** (our worst) · `centre_open` 346.95 · `centre_locked` 297.83 · `centre_tension` 285.98 ·
`centre_cleared` 283.59. ⇒ our error concentrates on PINNED and OPEN positions, i.e. the tactical regimes where
capture gains and threats would live.

---

## 2026-09-18 — ★★★ PER-CLASS REFERENCE LADDER: our accuracy deficit concentrates in PINNED/TACTICAL geometry

`_reference_ceiling.py CORPUS=ks_sets/classes_wide/<class>.csv N=3000` under shipped v2 — composition of two
existing tools, no new code. Schema verified first (the class files carry `target_total`/`split`, so the tool's
silent-`nan` path was avoided). ✅ **Both predictions registered before the run HELD, on both measures.**

| class | share | OURS v2 | SF11 | **gap** | SF15.1c | floor (SF18 static) | ours above floor | SF11 above floor |
|---|---|---|---|---|---|---|---|---|
| **`pin_dense`** | 17.2% | **168.64** | 93.63 | **75.01** | **133.50 (beats us)** | 68.99 | 99.65 | 24.64 |
| whole corpus | 100% | 155.72 | 95.26 | 60.46 | 139.20 | 68.85 | 86.87 | 26.41 |
| `centre_open` | 27.4% | 125.50 | 70.72 | 54.78 | 158.82 | 65.53 | 59.97 | **5.19** |
| `centre_cleared` | 15.9% | 115.39 | 74.30 | 41.09 | 164.54 | 62.00 | 53.39 | 12.30 |

### ★★ FINDINGS
1. **`pin_dense` is our worst regime** (gap 75.01) and the ONLY class where SF15.1-classical beats us. 1.8x the
   cleared-centre gap.
2. ★ **SF11 nearly SATURATES the achievable accuracy** — only **5.19** above the floor on open centres, 12.30 on
   cleared, 24.64 on pinned. ⇒ The us-to-SF11 gap is genuine CLOSABLE headroom, not the irreducible static-vs-search
   residue. This is the encouraging answer to "can we reach the giants on eval": it is work, not a wall.
3. ☠️ **SF15.1-classical is a BAD reference on this metric** — worse than SF11 in every class measured.
   ☠️ **CORRECTION (same day): I first wrote "worse than US in three of four". That was an ARITHMETIC ERROR.**
   Full count over 6 classes + the whole corpus: **we beat SF15.1c in only TWO** (`centre_open` 125.50 vs 158.82,
   `centre_cleared` 115.39 vs 164.54) and **LOSE the whole corpus** (155.72 vs 139.20) plus `pin_dense` (168.64 vs
   133.50), `other` (180.07 vs 130.44), `centre_tension` (200.72 vs 131.34), `centre_locked` (213.00 vs 120.27).
   ⇒ **We are NOT "beating SF15 in almost all cases."** Still: stop writing "SF11/15 levels" as ONE target —
   SF11's 95.26 is THE hand-reachable one, and SF15.1c is simply a weak static evaluator.
3b. ★★ **The more informative statistic is the SPREAD across classes, not the win count:**
   SF15.1c ranges 120-165 (spread **44**) · SF11 71-131 (spread **60**) · **OURS 115-213 (spread 98)**.
   ⇒ **Our error is by far the most STRUCTURE-SENSITIVE.** We are not uniformly worse; we are UNEVENLY worse —
   a better argument that specific regimes are underserved than any single class ranking, and it does not depend on
   the class ordering, which is corpus-dependent (see the tension/pin_dense swap below).
4. ★ The whole-corpus gap (60.46) EXCEEDS both centre classes, which is how `pin_dense` was predicted before it was
   run: a weighted average above its parts means the unmeasured classes must be worse. `other` (46%),
   `centre_tension` and `centre_locked` remain unmeasured.

### ▶️ THE BUILD PRIORITY THIS IMPLIES (measurement-derived, not story-derived)
Our error concentrates in PINNED, TACTICAL geometry — exactly where the UNBUILT/PARKED terms act: **capture gains**
(slice 5; already the largest identified chunk via the +138.69% variant column), **threats** (parked), **OvD**.
★ **Specific testable lead:** threats' strongest single leg is `hanging` (**-2.38%** §I @25), and pinned pieces are
frequently the inadequately-defended ones. Threats was parked for being move-NULL **globally**, and its per-class §I
was **NEVER RUN** ⇒ it has never been tested in the one regime where its mechanism most plausibly applies.
⚠️ ☠️ **But a per-class §I win is NOT a second instrument** ([[a-per-class-corpus-win-is-not-a-second-instrument]]) —
it is a louder reading of the same one. A threats-on-`pin_dense` §I gain would require a MOVE-LEVEL read on that
class against a neutral measured ON that class, exactly as pin's 9x class reading still came back null at d7.
⚠️ Also: `MOB_V2_PIN` already shipped and already read -4.75% §I on pin-dense (9x its global effect) — and pin_dense
is STILL our worst class. So the regime is not fixed by pin-line mobility alone.

---

## 2026-09-18 — ★★★ KS PRECISION/RECALL vs SF11: 13% capture GLOBALLY, 96% correct WHEN IT FIRES
### (baseline info, deliberately parked for the TUNING-PREP stage on the owner's call)

`_ks_fit_eval.py CORPUS=<...>` — existing tool, no new code. `target_ks` is **SF11's** own per-term king-safety
value (`_ks_auc.py:11`, and the output column is literally labelled `|SF11tgt|`).

| population | n | miss (silent) | wrongsign | correct | `\|ourKS\|` | `\|SF11tgt\|` | capture |
|---|---|---|---|---|---|---|---|
| whole corpus | 8,699 | 6,152 (71%) | 105 (1.2%) | 2,442 (28%) | 0.29 | 2.15 | **13%** |
| `centre_open` | 2,767 | 1,911 (69%) | 42 (1.5%) | 814 (29%) | 0.30 | 2.39 | **13%** |
| `centre_locked` | 259 | 202 (78%) | **0** | 57 (22%) | 0.18 | 1.94 | 10% |

☠️☠️ **READ `correctdir` CORRECTLY — IT IS NOT AN ACCURACY RATE.** I first reported it loosely and the owner
correctly challenged "are we directionally wrong 78% of the time?" **NO.** The complement of `correctdir` is
almost entirely MISSES (silent), not errors: 71% silent + 1.2% wrong. **Of the positions where we DO fire, we are
right 2,442/2,547 = 96%** (locked: 57/57 = 100%). ⇒ Our KS is **HIGH PRECISION, LOW RECALL.**
★ **And that is WHY it earns +101 Elo.** [[every-eval-term-error-is-bidirectional]]: a term that fires more often
helps half and hurts half. A term that stays quiet unless confident adds signal without that penalty. It also
explains the 0-for-11 additive-KS record — those attempts bought RECALL at the cost of PRECISION.

☠️ **The 87% uncaptured is NOT a demonstrated opportunity.** Three independent records say otherwise:
(1) matching SF11's KS term was TRIED and gave **+9.70% worst-case error** ([[matching-a-reference-term-is-not-being-right]]);
(2) additive KS is **0-for-11**, only subtractive wins ([[ks-twelve-attempt-history-and-the-channel-law]]);
(3) SF's kingDanger is an unbounded quadratic while ours saturates at `KS_V2_MAX=4000` — a deliberate difference,
and threats taxed KS-critical accuracy exactly when it pushed into that space.
⇒ A *differently shaped* higher-recall KS is not refuted, but it is a NEW hypothesis needing its own design, with
an 0-for-11 prior. Do not read "13%" as "multiply by 8".

### ★★ WHY THIS MATTERS FOR THE TUNING STAGE (the owner's framing — revisit it there)
A retune must not be free to rescale a term whose conservatism is load-bearing. The recorded tuning condition is
"PIN THE GLOBAL SCALE" ([[corpus-fit-is-anti-correlated-with-elo]]); this result argues for **per-subsystem scale
constraints too** — an optimiser handed our KS at 13% of SF11's magnitude and an SF-derived objective will push it
up, which is the shrink/inflate failure mode in the other direction. ⇒ Carry this table INTO the fit design.

### ⚠️ THE BLIND SPOT: this read is possible for KS ONLY
`target_ks` is the **only** per-term reference column in the corpus. There is no `target_mobility` or
`target_pawns`, so the same precision/recall profile **cannot be computed for any other subsystem today**.
▶️ Extendable: SF11 emits a per-term trace, and whoever built this corpus clearly did that for KS.
★ **Mobility is the most meaningful extension** — far more than KS — because the term-correspondence problem barely
applies: both engines count safe squares for pieces, so the definitions genuinely align. Pawn structure is middling
(shared predicate definitions, different decomposition). Mobility is also our largest Elo contributor (+162) and a
complete blind spot on this axis. ⇒ **Scope this as tuning-prep, not now.**
⚠️ Reference set for the remaining items should include the NON-SF engines (owner's note): Ethereal and Weiss for
Kaufman-style imbalance (Ethereal conditions knight/rook imbalance on `pawn_closedness`), corrhist and winnability.
☠️ Ethereal/Weiss must be FETCHED — only the Stockfish trees are local ([[reference-engine-sources]]).

---

## 2026-09-18 — ☠️★★★★ SLICE 3 CLOSES: KAUFMAN PARKS, AND IT IS FIVE CONCEPTS FOR FIVE PARKS

### Kaufman: three parameterisations of the census form, all fail
Base MSEs (unchanged across all three runs, which self-verifies the build's default path):
350.12 / 333.58 / 337.41 / 209.45 / 623.27 / 1937.37. NEGATIVE % = better.

| parameterisation | best mean% | its WORST% | verdict |
|---|---|---|---|
| **SF11 cells VERBATIM** (`FORM 0`), MAG 250-2000 | **+1.00** (at 250) | +2.03 | ☠️ monotonically harmful; no local optimum, no corpus likes it |
| **DERIVED per-piece value-ratio rescale** (`FORM 2`, zero free parameters), MAG 500-3000 | **+2.05** (at 500) | +3.19 | ☠️ harmful, and **barely different from raw SF** (+2.05 vs +2.53) |
| **v1's FITTED cells** (`FORM 1`, diagnostic), MAG 250-1500 | **-1.91** (at 500) | **+2.45** | improves the MEAN but never clears the WORST column |

**Ruled out before concluding anything:** SIGN (v1 and ours both do `total -= white_pov_sum`) and SCALE (a
hand-computed pair+minor-swap census reads +0.29 pawns at FORM 0 MAG=1000 -- a sane size, and 250 is a quarter of it).

☠️ **THE BASIS HYPOTHESIS IS REFUTED.** It predicted that SF's cells fail because they are corrections layered on
piece values ~2x steeper than ours (SF11 mg knight 6.10 pawns vs our 3.25). The derived differential rescale that the
hypothesis IMPLIES (piece x piece -> 0.27, piece x pawn -> 0.52, pawn x pawn -> 1.00) moved the result by **0.5pp**
and stayed harmful. ⇒ The failure is NOT calibration against a different value basis.
★★ What actually separates v1's working cells from SF's failing ones is **RELATIVE STRUCTURE** -- and specifically the
cells where v1 CONTRADICTS SF's sign (B x own-pawn **-106 vs +104**, N x enemy-pawn **-104 vs +63**). A derivation
cannot recover that; only a fit found it.
☠️ **And it cannot be claimed as insight, because v1's cells were fitted on OUR SELF-PLAY distribution and their gain
is confined to it:** `game_regret_set` **-8.65%** / `_v2` -5.69% / `_x4` -3.50%, but UHO **+2.38%**, variant
**+2.45%**, KS-critical **+1.53%**. That is the overfitting signature, not a validated form.
⇒ **No PRINCIPLED parameterisation of the census form helps v2. The only one that improves the mean is overfitted.**
★ Replicated twice: the bishop pair belongs INSIDE the term when it is on (7.04 vs 7.71 no-pair; -1.01 vs -0.47).

### ★★★★ SLICE 3'S RECORD IS NOW COMPLETE, AND THE PATTERN IS THE FINDING
| concept | reference support | outcome |
|---|---|---|
| central control | **0/5** | NOT BUILT -- no reference has a standalone central term |
| bishop pair | 5/5 | ALREADY OWNED by PST + mobility (flat in all 5 classes, regret null both corpora) |
| space | 3/5 | built + oracle-verified; globally INERT; class-local only on `centre_locked` (3.0-3.2%) |
| threats | 4/5 | best-verified term of the slice; §I liked it (-9.71% variant) but move-NULL on 35-36% of moves |
| **Kaufman + pairs** | 2/4 | ☠️ **PARKED 09-18** -- no principled cell set helps; the working one is overfitted |
⇒ **FIVE reference concepts, FIVE parks.** ★ The only thing slice 3 produced was ≈ **+31 Elo** from `MOB_V2_PIN` +
`MOB_V2_EXCL_LOWRANK` -- which add NO concept; they refine the AREA of mobility, the largest term v2 already owned.

★★★ **This is the test [[v2-positional-signal-is-nearer-saturation-than-its-term-count]] was waiting for.** It was
logged as "a hypothesis from a pattern of four, not a law". It is now five for five, with the fifth being the most
thoroughly parameterised attempt of the set (three cell sets x 5 magnitudes x 2 ownership settings). ⇒ Promote it
from hypothesis to **working conclusion**: on top of KS + pawns + passers + mobility + placement, a NEW CONCEPT adds
no move-level information, while refining the DEFINITION of an existing owner still pays.
⚠️ Bound it honestly: this is about POSITIONAL/census concepts. The unbuilt slices are different in KIND -- capture
gains is a tactical/ordering signal (v1's sibling spread 2,290 mp vs v2's 36), corrhist is a learned correction,
winnability is a whole-eval scale. None is "another positional term", so none is covered by this conclusion.

### PREDICTION SCORECARD (registered before each run, per the standing rule)
Kaufman SF ladder: **1 of 3** (magnitude curve WRONG, worst-column identity WRONG, pair-ownership RIGHT).
Basis test: **1 of 1 RIGHT** (v1's cells beat SF's at matched magnitude, and went negative).
Derived form: **0 of 1** (predicted it would clear; it barely moved). ⇒ Session running total ≈ **4 right / 11 wrong
or unreadable**. ★ The persistent failure mode is unchanged and worth restating: **I predict improvement where
measurement finds none.** Registering in advance is the only thing that has made this legible.

---

## 2026-09-19 — SHOWDOWN, SECOND SEED: v2 IS LEVEL WITH v1 ON TWO SEEDS / 3,000 GAMES

`sprt_ab`, p1 = **currently shipped** v2 (including `RFP_MARGIN=1000`), p2 = `EVAL_ARM=0`, seed **23**, LIGHTNING,
conc 4, `openings_uho.txt`, elo0=0 / elo1=5, max 1500.
```
+662 -676 =162 of 1500   score 49.5%   elo -3.2 +/- 20.7   LLR -0.400   (max-games stop, as intended)
```

| run | config | games | elo |
|---|---|---|---|
| seed 7 (09-18) | v2 PRE-`RFP_MARGIN` | 1500 | **-9.5 +/- 20.7** |
| seed 23 (09-19) | v2 SHIPPED, incl. `RFP_MARGIN=1000` | 1500 | **-3.2 +/- 20.7** |
| naive pool | ⚠️ mixed configs | **3000** | **-6.4 +/- 14.6** |

⇒ **v2 and v1 are INDISTINGUISHABLE on two independent seeds.** The point estimates straddle zero within 7 Elo.
⚠️ The pool is NOT strictly valid (the arms differ by one search knob), but `RFP_MARGIN=1000` measured **>= 0** in its
own paired A/B, so pooling is if anything CONSERVATIVE for the shipped config. Quote it as "two seeds, -9.5 and -3.2,
both inside noise", with the pooled figure as a secondary.
★ The single-seed worry is resolved: the 09-18 headline was NOT masking a real deficit.
★ Volatility note, one more instance: this run read **+3** at game 1,228 and finished at **-3.2** — ~6 Elo of drift in
270 games. The running estimate is not the result ([[sprt-point-estimates-inflate-at-the-bound-they-stop-on]]).

### ★★ WHAT LEVEL MEANS HERE, restated because it is the number the rebuild is judged on
v2 matches v1 while: **two entire slices unbuilt** (capture gains · corrhist · OvD · endgame conversion ·
winnability) · **never jointly retuned** (every constant from reference shape + a one-at-a-time ladder) · **five
slice-3 concepts parked** as adding nothing. And it does so while being **provably non-collinear** across all five
scoring subsystems (40-column gate, clean on two populations) and **colour-clean at 0/4000** against v1's 0.9% /
worst 2,233 mp. ⇒ Same strength, structurally cleaner, and 36.6% more accurate statically (155.72 vs 245.46 on the
SF18-search ladder, closing 59.7% of the gap to SF11's 95.26).
☠️ Still NOT a claim that the rebuild has won on strength — it has not. The payoff to date is in ACCURACY,
CORRECTNESS and STRUCTURE; playing strength is a wash. Slices 4-5 plus the retune are the test of whether that
converts ([[v2-is-level-with-v1-two-slices-early]] · [[the-hce-is-the-nnue-teacher-so-eval-carries-informationally]]).

---

## 2026-09-19 — SLICE 4 OPENED: tier-2b technique value BUILT (gated off), and the slice's shape is unlike 1-3

### ☠️ THE INSTRUMENT PROBLEM IS THE HEADLINE, NOT THE TERM
**§I is demonstrably BLIND to this family.** The slice-1 draw classifier -- a term of exactly this kind -- read
**IDENTICAL on all six §I corpora**. Not small: identical. And [[eval-payoff-is-opening-midgame-not-endgame]]
records a WHOLE better eval buying only -0.32/-0.72 in deep endgames vs -1.0/-1.2 in opening/midgame, with the
standing instruction to *"deprioritise endgame-only eval work unless the BY_PS split says otherwise"*.
⇒ **Slice 4 cannot be run like slices 2-3.** No §I magnitude ladders. The instruments are the TABLEBASE oracle and
games, under the owner's standing gate for self-play-invisible work: **position-proof + no bench regression**.
⇒ Also: nothing in this whole family has EVER produced a resolved Elo result in either direction. Every prior
measurement sits inside a floor, inside a bundle, or on an instrument since invalidated.

### ★★★★ AND A STRUCTURAL BLOCKER FOR THE GENERAL SCALE
All five references scale the **EG LEG ONLY, inside the blend** (5/5, both lineages). **v2 holds no whole-eval
(mg,eg) pair** -- every term blends per-side at its own site with `c.phase256` into a single `total` int.
Recovering an EG-leg scale from `total` needs `SUM eg_i`, which v2 does not keep; multiplying `total` is equivalent
only if `mg_i == eg_i` for every term, which is false for placement/passers/pawn-structure (KS is unphased, space
is MG-only). ⇒ **The consensus scale requires an ARCHITECTURAL change (a parallel `eg_total` accumulator), not a
term design.** Each term already computes its EG half before blending, so the marginal cost is one add per term.
☠️ And this explains the old null: v1's form (whole `total` x a boolean `isEndGame` cliff) **matches none of the
five**. The -3.3 STS reading measured a shape nobody uses.
★ Exception worth keeping: the tier-2b SCALE cases below are pawnless 5-piece endings, so `phase256` is deep
endgame and mg ~ eg by construction -- **those do NOT need the architecture change.**

### ▶️ BUILT THIS SESSION: `tier2b_value_mp` (`eval_v2.cpp`, after `draw_class`), knob `TIER2_V2_MAG` (0 = off)
Pawnless **K+R vs K+B** and **K+R vs K+N**: DISCARD the material lead, return SF15.1's `push_to_edge` formula
(corner 90 -> centre 28 in SF units) plus `push_away` from the knight for the KN case. A REPLACEMENT, not a term
added to a sum -- the same shape as `draw_class`. MAG is a PERCENT of SF's own scale (100 == SF's magnitude in our
mp: ~423 corner, ~131 centre); material-anchored, so the PAWN is the unit.
☠️ **CORRECTED MOTIVATION (I had this wrong earlier and said so):** v2's defect here is an **OVER-READ**, not a
false draw. v2 dropped v1's rules, so it returns the ORDINARY eval -- reading a rook up as ~+1550 mp in an ending
normally DRAWN with correct defence. Over-reads are how an engine trades INTO a dead ending believing it is
winning (the recorded +4870 loss). The 22-28% tablebase FALSE POSITIVES are **v1's** problem, not v2's.
⚠️ **SINGLE-LINEAGE (SF only).** Ethereal and Weiss leave these endings entirely to search. ⇒ CANDIDATE, not a
consensus adoption -- unlike the EG-leg scale and the two-tier split, which are 5/5.

**Gates:** ⚠️ ~~colour symmetry 0/4000 with the term ON at MAG=100~~ -- **THAT PASS WAS VACUOUS; see the 09-19
entry below, which replaces it.** ✅ Byte-identity at default EXACT: `250 / 49,440,513 / EBF 4.031`, cutoff
histogram identical (m0 2,225,417) ⇒ the term is correctly absent when `TIER2_V2_MAG=0`.

---

## 2026-09-19 — TIER-2B NON-VACUITY: THE PASS WAS EMPTY, AND TWO REAL DEFECTS WERE BEHIND IT

### ☠️☠️ THE 0/4000 SYMMETRY PASS WAS VACUOUS -- QUANTIFIED
`diverse_corpus_wide.csv` (n=23,113) contains **exactly ONE** pawnless K+R-vs-K+minor position. A scan of **all 47
corpora in `ks_sets/`** found no corpus with more than **9** (`game_regret_set_x4.csv`, 9 / 14,713); most have zero.
⇒ the run could not have exercised the term. ★ The hazard was already on the record and I did not apply it: the
slice-1 KPK entry says *"`_eval_symmetry.py` was NOT run for this: its sample contains essentially no KPvK, so it
would be a vacuous pass"* (2026-09-14). **Same tool, same corpus, same trap, five days later.**
★★ This also sharpens the "§I is blind to the endgame family" finding from a claim into a measurement: the
instruments are not merely insensitive here, **the corpora do not contain the positions at all.**

### ☠️ DEFECT 1 (eval, FIXED): tier-2b's early return published NO breakdown
`draw_class` publishes `total`/`arm`/`terms_valid = 1<<EB_TOTAL` on its early return, with a comment recording
exactly why. **Tier-2b's early return, added directly beneath it, published nothing at all** -- so `g_eval_breakdown`
kept the PREVIOUS position's values. Search was never affected (it uses the return value), but every static
instrument reads the breakdown: `_eval_symmetry.py` takes `ev_breakdown(b)["total"]`, **not** the returned score. Had
the term fired, the gate would have compared one position's score against another's. ⇒ the pass was not just empty,
it was **unreadable**. Fixed by publishing the same total-only contract as `draw_class`.

### ☠️ DEFECT 2 (tool, FIXED, PRE-EXISTING): `eval_breakdown.py` dies on any total-only publication
`_reconstruction` sums `ADDITIVE_TERMS` unguarded and raised `KeyError: 'pieces'`, killing the whole run on the first
such position. ⚠️ **Not introduced by tier-2b** -- verified directly on a draw-classified FEN (`8/8/8/4k3/8/8/4KB2/8`):
it crashes identically, so **every draw-classified position has crashed this tool under `EVAL_ARM=1` since the
classifier shipped on 2026-09-13**. Now reports the score and carries on, distinguishing two cases the first version
of the guard wrongly merged: a REPLACEMENT eval (`terms_available == 1`, i.e. only `EB_TOTAL`) versus an ORDINARY v2
position (v2 publishes 8/45 of its own terms; 7 of v1's 10 additive names are simply absent).

### ✅ NON-VACUITY, PROVED BY DIFFERENTIAL -- the term fires and moves the eval by ~0.8-1.4 PAWNS
Built `ks_sets/t2b_corpus.csv` (1,324 positions, 100% in-class) by REUSING `_draw_oracle.py`'s own `random_position`
and `SIGS` (which already carried `R_vs_minor` / `R_vs_minor_N`) -- no new generator. ⚠️ `ks_sets/*.csv` is
**.gitignored**, so the corpus is NOT tracked; it is reproduced from the tool that owns the machinery:
`pyrun diagnostics/_draw_oracle.py EMIT=ks_sets/t2b_corpus.csv CASES=R_vs_minor,R_vs_minor_N N=150`
(deterministic on `SEED`; 124 already-labelled positions first, then uniform and `EDGE=1` samples, both colours
strong). ★ The symmetry result below was re-verified on a SECOND, independently drawn 1,324-position sample from
this recipe -- overlap with the first was only the 124 cached FENs, and it read 0/1324 again.
Engine totals, `MAG=0` -> `MAG=100`:

| FEN | units | shipped (MAG=0) | term live (MAG=100) | delta |
|---|---|---|---|---|
| `1K6/8/5R2/3k4/8/8/5b2/8 w` (min) | 28 | −1505 | **−131** | 1374 |
| `r7/8/5k2/8/8/8/7B/K7 w` (KB max) | 90 | +1565 | **+422** | 1143 |
| `7K/8/r7/8/8/5k2/8/7N w` (KN max) | 210 | +1775 | **+985** | 790 |

★ The C++ reproduced an independent Python model of the formula EXACTLY on all three (registered before the run).
✅ **Colour symmetry re-run on the in-class corpus: 0 / 1324 violations, file mirror 0 / 1324** -- the same verdict as
before, but now non-vacuous. (Antisymmetry is in fact structural here: `edge_dist` and the king-knight Chebyshev
distance are both mirror-invariant, and only `rook_white` flips.)
⚠️ `units` reaches **210** for KRvKN (= 985 mp), not the ~90 / 423 mp the design note's "corner ~423, centre ~131"
implies -- that range describes `push_to_edge` ALONE; `push_away` adds up to +120 more. Faithful to SF15.1, but the
note under-states the shipped magnitude by more than 2x. ⇒ a KRvKN corner reads **+0.99 pawns**.

### ★★ DISCRIMINATION, MEASURED FREE FROM THE EXISTING TABLEBASE CACHE
`ks_sets/tablebase_labels_draw.json` already held **124** labelled in-class positions from the slice-1 sweep -- zero
network queries needed. Does `units` separate WON from DRAWN?

| class | won | drawn | units won (m/sd) | units drawn (m/sd) | AUC |
|---|---|---|---|---|---|
| K+R vs K+N | 16 | 46 | 154.0 / 34.1 | 120.8 / 40.3 | **0.721** |
| K+R vs K+B | 17 | 45 | 75.5 / 13.4 | 67.9 / 15.1 | **0.663** |

★ **All 33 decisive positions are wins for the ROOK side. 33/33 -- the sign is never wrong.**
★ Decisive rate **25.8% / 27.4%**, independently reproducing the 22-28% the record already carries for v1's
`R_vs_minor` rule -- so ~74% of the class is drawn while **v2 currently reads it as +1.5 to +1.8 pawns**. That
over-read is the defect, and it is horizon-independent: it does not depend on the won minority at all.
☠️ **Do NOT quote a pooled AUC** (it computes to 0.623, LOWER than either class). The KN branch adds up to +120 units,
so pooling mixes two different scales -- a Simpson-style artifact of this term's own construction.
### ★★ AND THE DTM ANSWER -- WHICH REVERSES THE CAUTION I WROTE AN HOUR EARLIER
I wrote *"until DTM is known the won-case gradient is unjustified; the drawn-majority over-read is what justifies
the term."* **Measured, that is wrong.** Extending `_draw_oracle.py` with a `FENS=` class-labelling mode (below)
and refreshing the 33 decisive entries:

| class | n | decisive | drawn | ≤12 plies | **>12 plies** | no dtm |
|---|---|---|---|---|---|---|
| K+R vs K+B | 62 | 27% | 73% | 4 | **13** | 0 |
| K+R vs K+N | 62 | 26% | 74% | 1 | **15** | 0 |

★★ **28 of 33 wins (85%) lie BEYOND the ~12-ply horizon.** Those are exactly the positions search cannot solve for
itself, so the won-case gradient is aimed at the right population after all. ⇒ tier-2b now has **two** independent
justifications, not one: the drawn-majority over-read (74%, horizon-independent) **and** a won minority that is
85% out of reach of search.
☠️ **A third defect, in the oracle (FIXED):** `tb_lookup` returned any cached dict as final, so a legacy entry
stored as `{"c": "win", "m": None}` could NEVER acquire a DTM -- every one of these 33 was stuck that way, which is
why the question looked unanswerable. The legacy-STRING branch already self-healed; the legacy-DICT case did not.
Decisive-without-DTM now falls through and re-queries once. Cost to settle the whole question: **33 queries.**

---

## 2026-09-19 (later) — THE FIRST MOVE-LEVEL READ ON TIER-2B: A NULL, WITH ZERO HEADROOM
> ⚠️ **READ THE n=200 VERDICT AT THE END OF THIS ENTRY FIRST.** The n=40 pilot sections below reported a
> NEGATIVE and then a "sweet spot"; **both were noise** and are kept only as a record of how the reading
> moved. The powered result is: no significant difference at any magnitude, and the base arm makes ZERO
> errors on the decision class the term exists to fix.

### ☠️ CORRECTION TO A NUMBER QUOTED ALL DAY: THE OVER-READ IS +2.1 PAWNS, NOT +1.55
Measured directly (`ev_breakdown` over the tablebase-labelled cache, 0 new queries), strong side's perspective:

| class | n | won % | eval WON | eval DRAWN | **AUC of our eval as a won/drawn classifier** |
|---|---|---|---|---|---|
| K+R vs K+minor (pawnless) | 124 | 26.6% | +2122 | **+2128** | **0.482** |
| K+R vs K+minor **+ P** | 166 | 24.1% | +847 | +645 | 0.583 |

Material accounts for 1550-1750; positional terms add ~400-570 on top. ⇒ quote **+2.1 pawns**.
★★ **Our eval carries ZERO information inside the pawnless class (AUC 0.482).** ⇒ tier-2b's push-to-edge
gradient (AUC 0.663/0.721) is **strictly more informative than what we ship**. The term's SHAPE is good.
☠️ **But the wider family is over-read too** (+645 mp on drawn K+R vs K+minor+P), which is what makes a
narrow fix structurally unsafe -- see the cliff below.

### ☠️☠️ THE CLIFF -- why damping only the pawnless corner moves the WRONG moves
A DRAWN position reads **+645** with the defender's pawn on the board and **+2128** once it is captured. So
capturing the last pawn RAISES our eval by ~1,480 mp in a position that was dead drawn either way. Tier-2b
at `MAG=100` damps only the far side of that step, turning a +1,480 incentive into roughly a −300 penalty.
⇒ **The boundary is discontinuous in BOTH arms.** A magnitude cannot remove a step; only a consistent
treatment of the whole family can. ★ Same shape as [[phase-is-a-3-way-boolean-and-everything-cliffs-at-one-material-step]].
⚠️ SF has the same structure (`Endgame<KRKB>` pawnless, ordinary eval one pawn earlier) and tolerates it on
search strength we do not have -- the recurring lesson about porting a reference term without its machinery.

### ▶️ THE INSTRUMENT (new, and it is not games)
`_draw_oracle.py EMIT_EPD=` builds a **WIN-PRESERVATION suite**: tablebase-WON positions, strong side to
move, emitted as a WAC-style EPD whose `bm` is the SET of win-preserving moves. ★ `tactical_test.py`'s
`load_epd` already parses `bm` as a set and scores `solved = chosen ∈ set` -- which IS "did the move keep the
win?", so **no new scoring code**; run it through `wac_suite <epd> <tag> KNOB=V`, twice. Fixed depth ⇒
deterministic ⇒ **valid while the machine is in use**.
☠️ NON-VACUITY IS BUILT IN: a position where EVERY legal move preserves the win cannot be failed. The very
first position sampled had 15/15 winning moves. Only positions with at least one preserving AND one throwing
move are emitted. ★ The suite also carries its OWN NULL: `c0` records preserving/total, so the expected score
of a RANDOM legal mover is computable -- read both arms against THAT, not against zero.

### ☠️ DESIGN 1 -- "don't drop your own pawn" (K+R+P vs K+minor): MEASURED NOTHING
24 positions, random-mover null 67.3%. **24/24 for BOTH arms at depths 10, 4; 23/24 both at 6 and 2.** Not a
power problem -- a MECHANISM problem, and my hypothesis was simply wrong. I claimed the over-read makes the
strong side indifferent to its own pawn. It does not: dropping the pawn is a 1000 mp swing that BOTH arms
price identically. The over-read changes the DESTINATION's value, not the RELATIVE COST of reaching it.
★ At depth 2 the eval is nearly the whole signal and the arms STILL agree ⇒ structural, not a depth artifact.

### ☠️ DESIGN 2 -- "the free capture" (K+R vs K+minor+P): SIGNAL, AND IT RUNS AGAINST TIER-2B
The over-read can only flip a choice when reachable by a MATERIAL-GAINING move, so gain and misvaluation
compound. 40 positions (only 24% of generated were TB wins), **0 unspoilable**, random-mover null **11.0%**.

| depth | A (`MAG=0`) | B (`MAG=100`) | nodes A → B |
|---|---|---|---|
| 6 | **39/40** | 37/40 | 97k → 107k (+10%) |
| 10 | **39/40** | 38/40 | 1.59M → 1.83M (+15%) |

★ **The mechanism works and still loses.** Arm B NEVER liquidates wrongly (arm A did once: `.005`, played
`Rxe5` into a draw at d6 -- the predicted blunder, observed). But arm B drops positions where the under-read
makes it DECLINE a winning capture: `.020` `8/8/1K5n/2p5/8/k7/6R1/8 w`, where **Kxc5 wins** and arm B plays a
rook move instead, at BOTH depths.
☠️ **And the suite is biased 10-to-2 IN TIER-2B'S FAVOUR**: of the 12 positions offering a liquidation,
entering the class is WRONG in 10 and CORRECT in 2. Arm A already declines all 10 correctly (search finds the
right move for other reasons), so the over-read has no headroom to cost anything -- while the under-read
costs real points on the 2.
⚠️ **NOT SIGNIFICANT**: n=40, gaps of 1-3. What raises confidence above the counts is the mechanism (named
positions, named moves, replicated across depths) plus the adverse suite bias. Enlargement pending.

### ★★ THE SALVAGE HYPOTHESIS -- level, not shape
Mean `units` on the class ≈ 99.7 ⇒ at `MAG=100` the term returns ≈ 468 mp against an observed level of
+2128 mp. **`MAG ≈ 455` is LEVEL-MATCHING**: the class keeps its current average value (no cliff) while the
AUC-0.72 ordering is added on top. Testable with **no code change** -- it is the same knob.
⚠️ Known risk at that setting: KRvKN reaches 210 units ⇒ **4,486 mp (4.5 pawns)**, far above the 1,750 material
difference, so the tails over-value cornered-king positions and may invite material sacrifice. If the ladder
shows cliff-harm at low MAG and tail-harm at high MAG with no good cell, the form itself is wrong for us and
the alternatives are (a) ADDITIVE gradient on top of the ordinary eval -- no cliff, adds information, does NOT
fix the over-read; or (b) damping the WHOLE family consistently -- fixes the over-read, no cliff, bigger change.
★ Note (a) and (b) serve DIFFERENT goals: (b) is an ACCURACY fix that matters to the NNUE TEACHER even if
move-neutral ([[the-hce-is-the-nnue-teacher-so-eval-carries-informationally]]); (a) is a MOVE fix.

### ☠️ THE MAGNITUDE LADDER REFUTES THE SALVAGE HYPOTHESIS (40-position pilot, d6)
| `TIER2_V2_MAG` | 0 | 50 | 100 | 200 | 300 | 455 | 600 |
|---|---|---|---|---|---|---|---|
| solved / 40 | **39** | 37 | 37 | *40* | 38 | 38 | 38 |
| nodes (k) | 97 | 108 | 107 | 107 | 100 | 95 | 92 |

☠️ **NON-MONOTONE** ⇒ by the project's own rule, *a knob whose response is not monotonic is not a lever*, and
at n=40 with 1-3 position swings the `MAG=200` 40/40 is almost certainly chance. **Do NOT quote it as a sweet
spot.** The level-matching prediction fails on both counts: the response is not monotone, and `MAG=455` (38)
does not beat `MAG=0` (39).
★ The ONE monotone reading is NODES, falling 108k → 92k as MAG rises ⇒ the +10-15% node cost reported at
`MAG=100` is specific to LOW magnitudes, not to the term. A stronger replacement signal prunes better.
⚠️ n=40 is too small for the solve column to decide anything. A 200-position suite is the prerequisite for
any verdict here, not a refinement of one.

### ★★★★ THE VERDICT (200 positions, 802 tablebase queries, random-mover null 13.4%)
**Neither the n=40 harm NOR the n=40 "sweet spot" replicated. Both were noise.**

| `TIER2_V2_MAG` | 0 | 100 | 150 | 200 | 250 | 300 |
|---|---|---|---|---|---|---|
| d6 solved / 200 | 191 | 192 | 192 | **193** | 191 | 193 |
| d10 solved / 200 | **197** | 194 | -- | **197** | -- | -- |
| d10 nodes | 7.48M | 8.08M (+8.0%) | -- | 8.17M (+9.3%) | -- | -- |

McNemar exact on the paired discordants, d10: `MAG=100` **p = 0.453**, `MAG=200` **p = 1.000**. ⇒ **no
significant difference at any magnitude tested.**

☠️☠️ **AND THE DECISIVE ROW -- THERE IS NO HEADROOM AT ALL:**

| verdict | n | A (`MAG=0`) | B (`MAG=100`) |
|---|---|---|---|
| **`liq_wrong`** -- entering the class THROWS the win | **33** | **33 / 33** | **33 / 33** |
| `liq_correct` -- entering the class is RIGHT | 9 | 9 / 9 | 8 / 9 |
| `none` -- no liquidation legal at all | 158 | 155 | 153 |

★★ **The base engine commits the over-read blunder ZERO times in 33 opportunities.** The suite was built
for that error, is biased 33-to-9 toward it, and arm A is perfect on it. Search already resolves it, exactly
as the `ENABLE_SIMPL_BIAS` record warned ("that best move was, on average, correct").
★★ **The tell that the remainder is noise:** 158 of 200 positions have NO liquidation available, so tier-2b
cannot legitimately affect them -- yet it flips 4-6 of them in BOTH directions. That is leaf-eval
perturbation deep in the tree, i.e. scatter.
★★ **And the effective n is 100, not 200:** the BISHOP half is **100/100 for every arm at every depth**.
All variation lives in K+R vs K+**N**+P -- the case carrying the extra `push_away` term and the 210-unit tail.

### ⇒ RECOMMENDATION: PARK TIER-2B. Slice 4's first concept joins slice 3's five.
The measurable effects are: **no move change that survives power**, and **+8-9% nodes** on this class.
⚠️ **Do NOT run games for it.** This suite has far more power per position than self-play (ground truth on
every move, no game-length noise) and it found zero headroom; games at 0.06% class frequency could not
resolve what a purpose-built, favourably-biased 200-position tablebase suite could not.
★ **What survives, and it is not nothing:** the ACCURACY case is untouched and was independently strengthened
today -- our eval has **AUC 0.482** inside the class (zero information) and over-reads drawn positions by
**+2.1 pawns**, while tier-2b's gradient carries AUC 0.663/0.721. Under the owner's teacher framing that is a
real defect in the TRAINING LABELS even though it is move-neutral in play
([[the-hce-is-the-nnue-teacher-so-eval-carries-informationally]]). ⇒ park the knob at 0, keep the finding.
### ★★★ INDEPENDENT CONFIRMATION FROM A DEPTH LADDER — the residual is HORIZON, not eval
Base arm on the same 200-position suite: **191/200 (d6) → 197 (d10) → 199 (d14)**, 100.8M nodes at d14.
⇒ the failures collapse toward zero as depth rises, so they were the SEARCH's depth deficit, not an eval
defect. An eval term cannot be paid for fixing what deeper search fixes for free.
★ Even the single d14 survivor is not tier-2b-shaped: `8/p7/8/5R2/4n1K1/8/7k/8 w` (plays `Ra5`, wants
`Re5`/`Rf4`/`Kf3`/`Kf4`, reads +2312) is a pure technique error in a position with **no liquidation legal at
all** (verdict `none`).
★★★ **GENERAL RULE, now built into the instrument:** the honest measure of eval headroom is **not** "base-arm
failures at depth D" but **"failures that PERSIST as D rises."** Reading a single depth counts the search's
horizon against the eval and will motivate terms that buy nothing at the depths we actually play. ⇒ every
future headroom screen runs at TWO depths.

★★ **This is the 6th consecutive concept to die as MOVE-NULL while reading positive on static accuracy**
(central, pair, space, threats, Kaufman, now tier-2b) ⇒ [[most-eval-error-is-move-neutral]] is now the single
best-supported law in this project, and [[v2-positional-signal-is-nearer-saturation-than-its-term-count]]
extends past positional/census concepts into the endgame-conversion family after all.

---

## ★★★★ 2026-09-19 (evening) — THE HEADROOM SCREEN: SLICE 4 HAS NO MOVE-LEVEL TARGETS

**The question that should have been asked before ANY of the last six concepts:** not *"is this term right?"*
but *"does the base engine actually make an error here that an eval could fix?"* The win-preservation suite
answers it directly, with tablebase ground truth, no C++, and no games. ★ Read at TWO depths, so the SEARCH's
horizon deficit is not charged to the EVAL.

**Batch A — base arm only, `TIER2_V2_MAG=0`, 140 positions:**

| class | n | random-mover null | d6 | **d12** |
|---|---|---|---|---|
| **K+R+B vs K+R** (the SCALE PAIR) | 35 | **9.9%** | 35/35 | **35 / 35** |
| **K+R+N vs K+R** (the SCALE PAIR) | 35 | **8.1%** | 35/35 | **35 / 35** |
| K+Q vs K+R | 35 | 69.1% | 35/35 | 35/35 |
| K+R+P vs K+R | 35 | 40.8% | 33/35 | **34 / 35** |

☠️☠️ **THE SCALE PAIR -- THE NEXT SLICE-4 ITEM -- HAS ZERO HEADROOM AND SHOULD NOT BE BUILT.** 35/35 at both
depths, on positions where only ~9% of legal moves preserve the win, i.e. genuinely sharp ones. That is the
exact class SF prices with a ~14/64 SCALE and where v1's draw rules measured **22-28% tablebase false
positives**. ★ **We need neither the scale nor the rules**: our search already converts these perfectly.
★ Only K+R+P vs K+R shows anything at all -- **1 persistent failure in 35 (~3%)**, the commonest real ending.
That is the only candidate target the screen found, and 1/35 is far too thin to build on without enlargement.

### ☠️ AND A CORRECTION TO MY OWN RULE, FOUND BY PUSHING THAT ONE POSITION DEEPER
`1r6/8/8/2R4P/8/2K5/8/3k4 w` (wants `Kd3`/`Kd4`/`Re5`/`Rf5`): **fails d12, fails d16, solves d20.** But the
COST is the point -- d20 took **82.7M nodes on one position** (≈ 3 minutes at our ~450k NPS) against ≈ **440k
nodes/position at d12**. At a realistic ~5 s (~2.25M nodes ⇒ roughly d14-15 here) it is **unsolvable in play**.
☠️ **"Deeper search fixes it" is NOT "search fixes it in a game."** Treating an arbitrary deep control as the
ceiling declares real errors to be non-problems. ⇒ **the ceiling must be the GAME-REACHABLE depth, measured
in NODES against the time control's budget** -- print nodes beside every depth ladder.
★ The batch-A conclusion SURVIVES this correction, because the scale pair is 35/35 at **d12**, which IS
game-reachable: those classes genuinely have no headroom. Only the `RP_vs_R` position, needing d20, stays a
real target.

**Batch B — 175 positions, same protocol:**

| class | n | random-mover null | d6 | **d12** |
|---|---|---|---|---|
| **K+P+P vs K+P** (pure pawn) | 35 | 54.8% | 33/35 | **35 / 35** |
| K+Q+P vs K+Q | 35 | 26.5% | 33/35 | **35 / 35** |
| K+B+P vs K+B | 35 | 35.2% | 32/35 | 33 / 35 |
| K+N+P vs K+N | 35 | 46.3% | 31/35 | 34 / 35 |
| K+R+P+P vs K+R | 35 | 59.9% | 34/35 | 34 / 35 |

☠️ **My registered prediction was WRONG**: I expected pawn endings to hold the headroom. **Pure pawn endings
are 35/35** — consistent in hindsight, since the exact KPK bitbase and the passer work already ship there.
The residue sits in MINOR-PIECE-PLUS-PAWN endings instead.

### ★★★★ THE WHOLE-FAMILY RESULT: 5 PERSISTENT FAILURES IN 315 POSITIONS (1.6%), AND NONE OF IT EVAL-SHAPED
Pushing all five deeper: **2 of 4 batch-B survivors solve at d16, ALL 4 at d20** (134M nodes for four
positions ≈ 33M each ≈ 75 s at ~450k NPS); the batch-A survivor also solves at d20 (82.7M nodes).
⇒ every residual failure is HORIZON, at depths **not reachable in play** — so by the corrected rule they are
genuine practical errors, not dismissable.
★★ **But they are not EVAL errors.** Their `c0` (winning moves / legal moves) is the tell:

| position | winning / legal | shape |
|---|---|---|
| `BP_vs_B.013` `8/8/7B/8/1b4P1/7K/8/4k3 w` | **1 / 12** | ONLY-MOVE |
| `NP_vs_N.023` `8/1p5K/8/1k6/1n6/8/8/5N2 b` | **1 / 14** | ONLY-MOVE |
| `BP_vs_B.034` `8/7B/1K2P3/8/8/8/4b3/6k1 w` | 2 / 14 | near-only-move |
| `RP_vs_R.007` `1r6/8/8/2R4P/8/2K5/8/3k4 w` | 4 / 14 | narrow |
| `RPP_vs_R.012` `5r2/R7/8/8/2P5/5k2/K4P2/8 w` | **13 / 18** | ★ the only eval-shaped one |

**An eval term does not find an only-move; a search does.** Four of five residuals are precision-under-horizon.
⇒ **the endgame-conversion family offers essentially no eval-shaped headroom**, and the one exception is a
single position. ★ This is a SEARCH/time-management finding, not an eval one: the same conclusion as
[[the-sf11-gap-is-two-thirds-node-efficiency]] arriving from a completely different direction.

⇒ **Combined with tier-2b's own null (33/33 on its target class) and its depth ladder (191→197→199), the
endgame-conversion family shows no move-level headroom worth a term.** ⚠️ Scope: this measures WIN
PRESERVATION from tablebase-won positions. It does NOT measure evaluation of drawn positions for TRADE
decisions made earlier in a game, nor anything the NNUE teacher needs -- those remain live.

### ▶️ PREDICTION SCORECARD (registered before each run)
| prediction | outcome |
|---|---|
| `push_to_edge` separates won/drawn weakly, AUC 0.55-0.70 | ✅ **right** -- 0.663 / 0.721, one class marginally above |
| engine totals at MAG=100 are exactly −131 / +985 / +422 | ✅ **right**, all three exact |
| (unregistered, and wrong) my first guard called a NORMAL v2 position a "replacement eval" | ☠️ caught by reading the `MAG=0` output, fixed |
| (asserted, not registered) "until DTM is known the won-case gradient is unjustified" | ☠️ **WRONG** -- 85% of the wins are beyond the horizon. I asserted a conclusion where I should have measured; it cost 33 queries to find out |
| "the over-read makes the strong side indifferent to losing its own pawn" (design 1) | ☠️ **WRONG, mechanism error** -- 24/24 both arms at 4 depths. A pawn is 1000 mp to BOTH arms |
| the win-preservation suite would show arm A throwing wins | ☠️ **WRONG** -- arm A scores 39/40; headroom was only ever 1-3 positions |
| (registered) tier-2b would help on liquidation decisions | ☠️ **WRONG, and backwards** -- it helps where A needed no help (10/10) and hurts where taking is right (1 of 2) |
| tier-2b's static case (AUC, 85%-beyond-horizon, 74% drawn) implied it was ready for games | ☠️ **PREMATURE.** Every static reading was positive and the FIRST move-level reading was negative. ★ Static accuracy and move value are different quantities -- the project's own law, which I restated this morning and then failed to apply to my own conclusion |
| (registered) "harm falls monotonically toward MAG≈455; MAG≈450 ≥ MAG=0" | ☠️ **WRONG on both counts** -- non-monotone, and 455 (38) < 0 (39). The level-matching arithmetic was sound and the response did not follow it |
| (registered) "arm B scores 2-6 lower at n=200, losses concentrating on `liq_correct`" | ☠️ **WRONG** -- d6 B was HIGHER, d10 lower by 3, neither significant. The concentration claim was right in kind (1 of 9) but far too small to matter |
| (implied by the n=40 pilot) tier-2b actively HARMS move choice | ☠️ **WITHDRAWN** -- did not replicate at n=200. I reported a negative off 1-3 position swings at n=40 and had to retract it, which is the same error as reporting a positive off the same |
| (registered) "most classes near-perfect; any headroom will be in PAWN and ROOK-AND-PAWN endings" | ⚠️ **HALF RIGHT.** "Most classes near-perfect" ✅ (7 of 9 classes ≥34/35 at d12). "Pawn endings" ☠️ **WRONG** -- `PP_vs_P` is 35/35; the residue is in MINOR+PAWN endings. The rook-and-pawn half was right (1/35) |

☠️☠️ **THE SESSION'S REAL LESSON, and it cost a whole afternoon to learn twice in one day.** This morning a
gate PASSED vacuously because the term could not fire; this afternoon a gate FAILED spuriously because n was
too small. Both times the printed number looked authoritative. ★ The n=40 pilot was RIGHT to run -- it
established the mechanism, the yield and the null cheaply -- but its SOLVE COLUMN should never have been
read as a result. **Pilot for feasibility; power for verdicts.** The tell was available before any
interpretation: a suite whose arms differ by 1-3 positions cannot resolve an effect of 1-3 positions.

☠️ **The pattern in that last row is the session's lesson, and it is the OPPOSITE of my usual failure.** My habit
is predicting improvement where none exists; here I talked myself OUT of a term's justification without measuring,
because the datum happened to be missing from a cache. ★ "The record does not contain it" is not "it cannot be
known" -- the missing DTM was 33 queries and a one-line cache bug away the whole time.

### ▶️ NEXT STEPS, in order
1. ✅ **DONE 2026-09-19** -- non-vacuity proved by differential (0.8-1.4 pawns), symmetry re-run non-vacuously
   0/1324, and two defects found and fixed. See the 09-19 entry above.
2. ✅ **DONE 2026-09-19.** `_draw_oracle.py` extended with a `FENS=<csv>` CLASS-GROUND-TRUTH mode (labels an
   explicit FEN list regardless of whether a draw rule fires -- the FP mode structurally could not do this) plus
   a `tb_lookup` self-heal for decisive-without-DTM. Result: AUC 0.663 KB / 0.721 KN, sign right 33/33, decisive
   26-27%, and **85% of the wins beyond the horizon**. ⇒ the term is justified twice over. **Remaining before it
   can ship: games.** Static evidence is now as strong as this family's instruments can make it.
3. **Tier-2b scale pair**: K+R+B vs K+R and K+R+N vs K+R at SF's ~14/64 (a SCALE, a different mechanism).
4. **`eg_total` accumulator** -- ship byte-identical on its own, then the general graded scale.
5. **Rule-50**: blocked on plumbing AND a zobrist hazard (the eval cache key excludes `rule50`, so entries would
   collide across counters).
6. **Winnability: do NOT rebuild as ported.** Three structural faults, only the third about constants:
   (a) applied to the blended `total` instead of the (mg,eg) PAIR, so SF's consumer coupling -- the corrected `eg`
   feeding the scale's strong-side pick and its OCB passer term -- **never existed**; (b) **no lazy exit**, so ours
   fires everywhere while SF's layer only ever acts in the near-balanced band; (c) a near-monotone scale on a
   summed total cannot reorder siblings (0.5% meaningful move change; same law that parked tempo).
   ★ Naming across engines: SF1.1 none -> SF11 `initiative()` -> SF15.1 `winnable()`; Ethereal `evaluateComplexity`;
   Weiss none. ★★ **Ethereal's has NO king inputs** -- the two-lineage CORE is pawn-structural (pawn count, both
   flanks, pure-pawn ending); outflanking/infiltration are SF-ONLY. A rework should keep the core and drop the
   single-source king terms.
7. **KPvK WON-case magnitude** (SF tier-1 `VALUE_KNOWN_WIN + PawnValueEg + rank`) -- we classify draws exactly but
   give won KPvK no shaped value. Tail item for this rung or the final sweep.

### ★ OWNER'S FRAMING, recorded because it predicted a measured result
*"King mobility is inherently maximized when safety, passed pawn stopping/supporting and other endgame things
become relevant"* -- the same argument as their centrality theory, which the five-engine contrast VALIDATED (0/5
references carry a standalone central term). ✅ Confirmed for the passer half: **v2 already owns king-shepherding**
via `PASSER_V2_KING_THEM`/`PASSER_V2_KING_US` (eg-leg, SF's form, enemy king's distance outweighing ours).
⚠️ Refinement so it is not over-applied: the test is NOT "is the quantity emergent?" -- almost everything is. It is
**"does an EXISTING TERM already order moves by it?"** Mobility is itself a term for an emergent quantity and was
worth +162 Elo, because nothing else ordered by piece freedom; central was worth 0 because two terms already did.
⚠️ UNVERIFIED and worth one check before treating as a gap: whether ANY reference carries a standalone king-mobility
term, and whether v2 owns v1's `boost_pieces_for_supporting_passed_pawns` (pieces supporting a passer) -- `BEHIND_V2`
is minor-behind-ANY-pawn in Weiss's form, which is a different concept.
