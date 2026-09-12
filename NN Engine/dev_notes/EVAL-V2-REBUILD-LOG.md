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
   ▶️ Revisit shape: "quadratic longer before saturating" is untried.

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
