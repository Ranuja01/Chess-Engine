# SESSION HANDOFF 2026-09-07 — the phase question answered, the cliff cleared, the regret gate repaired

## READ FIRST
- ⭐ **THE EVAL LANE HAS ITS FIRST CALIBRATED, REPLICATED PHASE ANSWER.** SF15-classical's eval inside our
  search, d7, both full regret corpora, each stratum read against its OWN Elo-neutral null band:
  **eval accuracy pays most at HIGH MATERIAL and least in DEEP ENDGAMES, and is UNAFFECTED by our phase
  cliff.** 📄 [[eval-payoff-is-opening-midgame-not-endgame]]
- ☠️ **THE PHASE-CLIFF HYPOTHESIS IS DEMOTED.** Every mapped discontinuity fires at `phase_score` 64→69;
  eval value RISES across it by **+0.35 / +0.33** in the two corpora. 📄 [[phase-is-a-3-way-boolean-and-everything-cliffs-at-one-material-step]]
- ☠️☠️ **THE EVAL DEPLOYMENT GATE WAS BROKEN IN TWO WAYS AND IS NOW FIXED.** Its null is large, negative
  and arm-specific; and its worker temp paths collided so ANY concurrent invocation was silently wrong.
  📄 [[move-regret-tool-null-is-large-negative-and-arm-specific]]
- ☠️ **NO CANDIDATE FOUND.** The result says WHERE eval accuracy is worth having, not WHAT is wrong.

## 1. THE MEASUREMENT
`_ks_footprint_regret.py`, `ENABLE_ORACLE_EVAL=1 ORACLE_CLASSICAL=1` (SF15.1 classical, native linux
binary), DEPTH=7, full sets (15,000 / 11,940). Null = `ASPIRATION_DELTA` 300 and 800, a lane already closed
as Elo-neutral, giving a BAND per stratum. Bucketed on the ENGINE's own `phase_score`:

| bucket | primary | v2 | verdict |
|---|---|---|---|
| ps1 mid_far (≤53) | **−1.07** | **−1.20** | robust, largest |
| ps2 mid_EDGE (58-64) | −0.82 | −0.69 | robust |
| *— `isEndGame` boundary —* | | | |
| ps3 end_EDGE (69-74) | −1.17 | −1.02 | robust |
| ps4 end_far (≥80) | −0.33 | −0.70 | **direction only — magnitude differs 2×** |

- **Cliff:** ps2→ps3 rises +0.35 / +0.33. No accuracy cost at the boundary. ⚠️ BOUNDED, not proven zero:
  the mid_EDGE null band is ~0.57 wide, so any cliff effect is **< ~0.6 where the local effect is ~1.0**.
- **Deep endgame:** weakest or joint-weakest cell in both sets, but −0.33 vs −0.70 is unresolved.
- Corpus `phase_bucket` rows (piece-count based) for reference: opening −1.06/−1.21, midgame −0.96/−1.00,
  endgame −0.32/−0.72.

## 2. ⚠️ THREE INSTRUMENT LAWS EARNED (these outlast the chess result)
1. **THE NULL IS NOT ZERO.** Two Elo-NEUTRAL arms read −0.1389 and −0.0619 at the same flip rate; the null
   also has strong PHASE STRUCTURE (−0.23 opening vs ~0 endgame). Against zero, the opening looks strongest
   in every arm ever run. **Nothing below ~0.2 is resolvable.**
2. **THE BAND IS PINNED BY MORE NEUTRAL ARMS, NOT MORE POSITIONS.** The mid_EDGE null (−0.45 for asp300)
   REPLICATED across both corpora; its 0.57 spread comes from ARM CHOICE, not sampling. Adding corpus does
   not narrow it.
3. **A TIGHT NULL BAND DOES NOT IMPLY A STABLE ESTIMATE.** I called ps4 "the most solid number" off a ±0.01
   null; its raw value then differed 0.25 across corpora. The null measures instrument bias; cross-set
   variance is a separate error source. **Quote both, or neither.**
★ Also: **design strata so the cells carrying the question are the BIG ones.** The two edge cells (n≈300-700)
were the smallest here while the cells I was not asking about held 2,900-4,500 — which is why the cliff
verdict is bounded rather than settled.

## 3. 🔧 TOOL CHANGES (`diagnostics/_ks_footprint_regret.py`, all committed to the working tree only)
- **`/tmp` COLLISION FIXED.** Worker paths were `/tmp/_fp_<tag>_<i>.csv` — identical across concurrent
  invocations. Two arms launched together clobbered each other and returned BYTE-IDENTICAL numbers for
  different knobs, with a truncated position count. Now PID-qualified. ⚠️ This defect was present for every
  parallel use in the tool's history.
- **`BY_PHASE`** breakdown on the corpus `phase_bucket` column.
- **`BY_PS`** breakdown on the ENGINE's `phase_score`, computed from the FEN, bucketed around the real
  64→69 step. ☠️ Needed because `phase_bucket` is a raw PIECE COUNT (`_build_game_regret_set.py:58-60`)
  which places our mid/end boundary INSIDE its "midgame" — it structurally cannot see the cliff.
- **Read-out rewritten** to state the null and demand a rate-matched neutral arm.

## 4. ▶️ WHERE THIS LEAVES THE LANE
- ✅ Deprioritise **deep-endgame eval work** — least available gain, and the two structural endgame suspects
  (`advanced_endgame_eval` firing unconditionally via an always-true `isNearGameEnd`; KS cliffing to zero)
  are not costing measurable accuracy.
- ✅ The **KS endgame-extension knobs** (`KS_EXTEND_EG`, `ENABLE_KS_UNIFIED`) were queued to smooth the
  largest cliff. The cliff result removes the motivation; do not spend the night on them.
- ✅ **ANSWERED — and it validates HCE-first numerically.** SF11 (purely hand-written) reaches **~90% of
  SF15-classical's gain**; the aggregate gap is **0.08 in BOTH corpora**, and the ENTIRE shortfall is one
  stratum — **high material (ps≤53), 0.20/0.18**. Every other cell flips sign between sets.
  ⇒ **no net is needed to capture nine-tenths of the available eval gain**; a net's marginal value is
  localised to high material. 📄 [[sf11-hand-written-captures-90pct-of-sf15c-gain]]
  ★★★★ **A CANDIDATE-VS-CANDIDATE GAP IS NULL-INDEPENDENT** (same corpus, same stratum, near-identical flip
  rates ⇒ the bias subtracts out) — more trustworthy than either arm's absolute value. **Prefer questions
  posed as a difference between two candidates.**
  🐛 I wrongly called this "blocked, needs a native SF11 build" — SF11's linux binary AND its full source
  were on disk all along at `Programming/Chess Engine/stockfish_11_linux/`. I inferred the only path from
  `_sts_reference`'s mapping (which points at the Windows exe) instead of looking. **One tool's path mapping
  is not the inventory.**
- ▶️ **NEXT: the bounded read.** SF11's `src/evaluate.cpp` is on disk and the target is now one stratum —
  *what does SF11 do at HIGH MATERIAL that we do not?* Previous "read SF11" attempts failed for lack of a
  target; this one is aimed by a measurement. ⚠️ Do not expect a single missing term.
- ⚠️ Still no candidate, and [[eval-disagreement-mining-no-missing-term-but-our-components-are-10x]] says it
  will not be one missing term (SF15c is two-sided: 185 better / 159 worse).

## 4b. ☠️ `ENABLE_PIECE_MOBILITY` TESTED AND CLOSED — but the MECHANISM is NOT closed
The SF11 contrast raised three candidates at high material (below). #3 turned out to be **already built and
dormant**: `MobilityBonus_{Knight,Bishop,Rook,Queen}` at `cpp_bitboard.cpp:93-96` are SF-shaped concave
tables **with the negative floor intact** (−50/−40/−40/−20) behind `ENABLE_PIECE_MOBILITY=false`
(`search_engine.h:1862`). Proposed for gating 2026-07-15 (`overnight-batch-2026-07-15.md:123-125`) and
**never run** — no result recorded anywhere.
Measured 09-07: **STS d8 1501 vs 1676 baseline (−175)**; regret null-corrected **aggregate +0.05 (AT the
null)**, ps1 −0.13, ps2 +0.25, ps3 −0.51, **ps4 +0.19 against a ±0.02 band (real harm)**. Two-sided by
stratum, netting to nothing.
☠️ **CLOSED: "SF's mobility scheme wholesale, replacing ours."**
⚠️ **NOT closed: "does charging for immobility help."** The knob is a THREE-PART BUNDLE — it adds the floor,
changes the mobility AREA definition, and **force-disables the cheap per-piece surrogates** the eval has been
tuned around (`search_engine.cpp:1985`). A −175 is at least as consistent with removing tuned terms as with
the floor being wrong. ★ [[node-savings-below-35-percent-are-elo-neutral-dont-scale-from-a-bundle]] — never
apportion a bundle to one part, in either direction.
▶️ To test the mechanism: add the floor ALONGSIDE the surrogates, behind its own default-off knob.
⚠️ Also note the record's "mobility RULED OUT" was a verdict on the REWARD-ONLY flat-per-square scheme that
is actually live — the penalty mechanism was never measured. **Third proxy-vs-mechanism mismatch this
session**, after the pawn taper and the `phase_bucket` strata.

## 4c. ▶️ THE THREE HIGH-MATERIAL CANDIDATES (from SF11's source, ranked on evidence)
1. **Space** (`stockfish-11-linux/src/evaluate.cpp:661-691`) — ⭐ **the standout.** Pure-mg
   (`make_score(x,0)`), weighted `(pieceCount−1)²/16`, and **gated on `non_pawn_material >= SpaceThreshold
   =12222`** — SF only computes it while ~74% of piece material remains, i.e. almost exactly our ps1
   stratum. ~140-210 units/side there. Counts safe squares on files c-f ranks 2-4, then counts AGAIN those
   ≤3 ranks behind an own pawn and unattacked. **We have nothing of this shape**: `SPACE_MAG=0` and the
   dormant knob (`cpp_bitboard.cpp:8510-8528`) lacks the safe-square area, the behind-pawn doubling and the
   piece-count weight. Needs a build.
2. **kingDanger accumulation SHAPE** (`evaluate.cpp:446-461`) — SF's leading term is a PRODUCT
   (`attackersCount × attackersWeight`, queen weighted LOWEST at 10) mapped **quadratically from a threshold
   of 100, uncapped in mg**. Ours is a flat additive sum (queen weighted HIGHEST), hard deadzone at
   `KS_FLOOR=13`, linear map, capped at 80, then two multiplicative damps. Dormant knobs exist
   (`KS_COORD_GATE_MODE`, `KS_ADJACENCY`, `KS_PIN_MODE`, `KS_ONSET_MODE`, all 0).
   ⚠️ KS is 0-for-11 additive — but this is a RE-SHAPE, and the record says only re-shape/subtractive wins.
3. **Mobility negative floor** — see 4b; needs an additive-floor build to isolate.
Runners-up, all pure/near-pure mg and fully absent: FlankAttacks `S(8,0)` + `3k²/8` into danger;
RestrictedPiece `S(7,7)`; ThreatByPawnPush `S(48,39)`; TrappedRook `S(52,10)`; WeakQueen/SliderOnQueen.
⚠️ Our `Hanging` counterpart exists but is fenced off by `THREATS_STANDING_ONLY=true`.

## 4d. 🌙 THE OVERNIGHT SCREEN (09-08) — 5 built-but-disabled terms RE-MEASURED, NOTHING CLEARED
Method: `_ks_footprint_regret` win% on the full `game_regret_set` (15,000) at d7, `DUMP=` on every arm.
**Null band from two Elo-neutral arms: `ASPIRATION_DELTA=300` → 49.9%, `=800` → 50.4% ⇒ band 0.5pp wide.**
(Compare the MEAN's band, 0.56 *units* wide in edge strata — win% is far better calibrated.)
| arm | changed | win% | verdict |
|---|---|---|---|
| `THREATS_STANDING_ONLY=0` (unfence SF-style Hanging) | 5378 (35.9%) | **49.9%** | NULL |
| `OUTPOST_KNIGHT=150 OUTPOST_BISHOP=80` | 3360 (22.4%) | **47.9%** | **HARMFUL** (−2.0 to −2.5pp, ~2.3σ) |
| `ENABLE_MOBILITY=1` (whole-board) | 6033 (40.2%) | **50.0%** | NULL (2732/2729 — exact) |
| `CENTRAL_BOUNDED_MODE=2` (dynamic knee by phase) | 3741 (24.9%) | **50.1%** | NULL |
| `SCALE_CENTRAL=50` (subtractive; the recorded ~9× over-read) | 4256 (28.4%) | **50.5%** | NULL (+0.6pp, 0.8σ) |
| `ENABLE_IIR=1` | **0** | — | ☠️ **REGIME MISMATCH, not a null** (below) |
⇒ **Five terms converted from "built but never measured" to MEASURED.** Four null, one harmful.
⇒ This is exactly what the 7pp ruler predicts: single terms sit under the ~1.5pp resolution floor
([[win-pct-is-the-honest-statistic-and-a-whole-eval-is-worth-7pp]]).
⚠️ The outpost magnitudes (150/80, from SF's `S(30,21)`) were a GUESS — one point does not close the
concept, cf. `EG_EXIST` where a single setting said nothing and the ladder said everything. But the sign is
negative, so it does not justify a ladder ahead of untested work.
☠️ **`ENABLE_IIR` changed 0 of 15,000 — a LIVENESS failure, not a result.** Knob name and wiring verified
correct (`search_engine.cpp:2021`); the cause is `IIR_MIN_DEPTH=6` with the harness at **DEPTH=7**, so IIR
has scope at the top two plies only. Its real results (quiet −16.0% nodes, STS 1818, +1 median ply) are all
at d10+.
★★★★ **BOUND ON THE INSTRUMENT: the d7 regret gate is STRUCTURALLY BLIND to any technique gated at depth
≥ 6.** It is an EVAL screen; do not put depth-gated SEARCH items through it. ✅ It announced itself as a
hard zero rather than as a plausible null — the changed-rate liveness check works.

## 4e. 🔨 09-08 — TWO BUILDS, THE WIRING THESIS TESTED AND REFUTED
Both knobs default-off and **fingerprint-verified byte-identical (250 / 35,310,778 / EBF 3.784) after each
rebuild** before any arm ran.
**(1) `THREAT_ATT2_PROTECT`** (`search_engine.h`, wired `search_engine.cpp`, used `cpp_bitboard.cpp`
`threats_by`) — SF's second `stronglyProtected` clause, `attackedBy2[Them] & ~attackedBy2[Us]`, i.e. skip a
target the enemy defends twice while we attack once. Our `threats_by` had the pawn clause only. `na`/`nd`
were already counted at the site. **Result: win% 50.4% = the top of the null band. NULL.**
**(2) `KS_MOB_EDGE`** — SF's `mobility → kingDanger` edge (`evaluate.cpp:452`) at the same site: adds
`(K * (attacked-squares[Them] − attacked-squares[Us])) >> 6` to a king's attack units. New file-scope
`g_mob_white/g_mob_black`, filled by a 64-square scan over `attack_bitmasks` after the piece loops (KS is
the first consumer). **Result: 16 → 49.9%, 64 → 50.6%. NULL across a 4× range INCLUDING CRANKED**, while
live (23-31% of moves changed). Only consistent structure: mild NEGATIVE in endgames at both magnitudes.
☠️ ⇒ **THE WIRING THESIS IS UNSUPPORTED.** The structural facts stand (13 SF term→term edges vs our parallel
sum; four king-pressure channels off one heat table; shelter computed four times) — but the INFERENCE that
wiring one such edge correctly buys measurable accuracy is refuted.
📄 [[the-wiring-thesis-was-tested-and-is-unsupported]]
🔧 Also fixed in passing: the stale comment at `cpp_bitboard.cpp:7746` claiming KS is "Default-off
(KING_SAFETY_MAG=0, ENABLE_KS_REPLACE_LT=false)" — it ships at 3000/true.

## 4f. 📉 THE 09-08 LEDGER — 11 ITEMS, NOTHING ABOVE THE FLOOR
`THREATS_STANDING_ONLY=0` 49.9 · outposts(150/80) 47.9 **harmful** · `ENABLE_MOBILITY` 50.0 ·
`CENTRAL_BOUNDED_MODE=2` 50.1 · `SCALE_CENTRAL=50` 50.5 · `ENABLE_IIR` 0 changed (**regime mismatch** —
`IIR_MIN_DEPTH=6` vs a d7 harness) · `KS_ZONE_ATTACK_PCT=0` **51.5 → v2 50.7 = null, phase pattern REVERSED**
· `PIECEVAL_RECOMPUTE_LATE=1` **51.1 → v2 49.2 = NEGATIVE** · `THREAT_ATT2_PROTECT` 50.4 ·
`KS_MOB_EDGE` 16/64 → 49.9/50.6.
☠️★★★★ **TWO ARMS CLEARED THE PRIMARY BAND BY ~+1pp AND BOTH DIED ON CROSS-SET** ⇒ **a single-corpus
reading near +1pp is NOISE**; the real bar with replication required is **~2-2.5pp**, so a single term must
be worth ~1/3 of a whole eval upgrade. Nothing will be.
⚠️ **The BUNDLING premise is weaker than it was sold**: it needs components that are positive-but-invisible,
and several here measure at or BELOW zero. Summing genuine zeros gives zero.
▶️ What remains standing is the DEGENERACY explanation (~30 terms / ~2 signals ⇒ absorption), which implies
only WHOLESALE change moves anything: de-duplication at scale (four king-pressure channels, four shelter
computations) or a net. ⚠️ Treat it with the scepticism the wiring thesis just failed to earn.

## 5. STATE
Baseline re-verified at session start: **EBF 3.784** (canonical). Nothing built, shipped or committed.
Uncommitted in the working tree: the oracle knobs (`ENABLE_ORACLE_EVAL`/`ORACLE_CLASSICAL`/`ORACLE_SCALE`),
`EVAL_NOISE_SIGMA` (parked), `_search_stability` VS mode, `_tail_term_stats.py`, and the three
`_ks_footprint_regret.py` changes above.
⚠️ Owner authorised knob-gated builds for testing; none was needed. Any future one must be **default-off and
fingerprint-verified byte-identical** before arms are run on it.
⚠️ RAM, not cores, is the binding constraint on this box — cap ~2 engine-loading runs concurrently.
