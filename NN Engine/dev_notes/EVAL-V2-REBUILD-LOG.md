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
