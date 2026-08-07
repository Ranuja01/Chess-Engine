# Optimization log (search/cache roadmap)

Baseline (pre-everything): eval **−56**, **3,144,112** positions, ~**16.7 s**, move 29. See [BASELINE_PERF.md](BASELINE_PERF.md). Methodology: one item at a time, rebuild + 3× `main.py`, diff at equal depth (per-depth PV/score = correctness invariant; nodes/time = effect).

> **⚠️ DEPTH-LABEL CONVENTION CHANGED 2026-06-03.** `MAX_DEPTH` is now **literal** — `MAX_DEPTH=10` searches to depth 10. Older commands/notes in this file used the off-by-one convention where the cap was `+1` (the iterative loop used `depth_limit + 1 < MAX_ITERATIVE_DEPTH`), so **a historical `MAX_DEPTH=11` ≡ today's `MAX_DEPTH=10`** ("d10"), `=12`≡`=11`, etc. When re-running any banked command below, subtract one from its `MAX_DEPTH`. New commands use the literal value.

## 🔧 2026-08-07 — COLOUR-SYMMETRY CLEANUP, partial: 74.5% → 57.4%. Fingerprint MOVED.

🚨 **New default fingerprint `35,138,590 / EBF 3.839`** (was `250 / 35,791,173 / EBF 3.804`). Byte-identity
was broken deliberately — a correct symmetry fix MUST change output. WAC still 250/300.
🚨 **The corpus baseline `228.347 / 231.177` is now STALE** — `our_total_base` on all 23,113 rows refers to
the old eval. Run `refresh_bank_ours.py` and re-derive before any fit.

### ✅ FIXED — king-race tempo polarity, the same bug in two places
`kingCanCatch` correctly flips polarity by colour; the tempo term beside it did not.
- `advanced_endgame_eval` inline passer block (**dead at default** — runs only when `ENABLE_PASSER_V3=0`)
- **`passer_king_race_one`, the V3 successor — LIVE.** The bug was copied forward verbatim.

In both, the White branch computed `diff = (turn) ? ... : ...` where `kingCanCatch` two lines up used
`(!turn)`. The catcher on that branch is Black, so the tempo credit had the wrong polarity and White's
passer was docked a penalty its Black mirror escaped. Hand-verified on b5/kings e1-e8: White's
`passedBonus` came out `3·ppInc/16` vs Black's `4·ppInc/16` — **exactly the 4/3 ratio measured**
(1.32 / 1.31 / 1.33 at b5 / c6 / d7).
✅ **Result: single pawn on a bare board is now PERFECTLY symmetric on all 48 squares** (was 30/46 broken,
worst +830 mp). Corpus-wide violations 74.5% → 57.4%.

### ↩️ TRIED AND REVERTED — capture-gains evasion polarity
`find_and_pop_last_viable_capture(opp_captures, …, current_turn)` at two sites reads as wrong: the stack
belongs to the other side, `find_last_viable_capture` beside it uses `!current_turn`, and both helpers
define `captureColour` identically. With the wrong polarity `isValid` fails for every entry and — since
the helper pops unconditionally — it DRAINS the opponent's whole stack. Changing it to `!current_turn`
**made asymmetry worse** (capture_gains 43 pos/583 mp → 50/688; total 82,264 → 92,450). Reverted; the
measurement is recorded in a code comment so it is not re-attempted on the same reasoning.

### ☠️ RETRACTED — `imbalance_white`/`imbalance_black` were never a defect
Reported as the largest remaining source (87 positions, 1722 mp mean). They are **signed** fields in the
Black-positive frame, so they negate-and-swap; the test checked plain swap and flagged all 87.
**`kaufman_imbalance` — the term that actually reaches `total` — is perfectly antisymmetric on every one.**
★★★ Three invariant families live in this breakdown and mixing them manufactures phantom bugs:
signed contributions **negate**; side-labelled magnitudes (`det_*_pieceval`, mobility) **plain-swap**;
side-labelled signed values (`imbalance_*`) **negate AND swap**. `_eval_symmetry.py` now encodes all three.

### ▶️ STILL OPEN — with a four-piece repro in hand
| source | positions | mean | note |
|---|---|---|---|
| `pieces` (`pt_pawns` 302, `pt_rooks` 233, `pt_knights` 223) | 616 | 69 | broadest |
| `advanced_endgame_total` / `ae_input` | 352 | 107 | |
| `material` / `capture_gains` / `det_*_pieceval` | 43 | 603 | all three fire on the SAME positions |
| `piece_value_boost` | 36 | 303 | **amplifier, not a source** |

🎯 **MINIMAL REPRO: `4k3/8/8/8/8/3N4/P7/4K3 w` — off by 35 mp, isolated to `pt_knights`.**
Own-side pawn + knight is asymmetric **24 out of 24** sampled placements. Clean in isolation: single piece
(N/B/R/Q all 0), single pawn (0/48), pawn vs enemy pawn (0/25), pawn + own rook (0/19), knight vs enemy
knight (0/29). Persists with the knight far from the pawn (+20 h3, +30 b3, +35 d3) ⇒ **not proximity**;
the knight evaluator reads some pawn-derived state asymmetrically. ✅ Already ruled out: the
`BB_PAWN_ATTACKS[colour][r]` guards at L3470/L3565 are correctly mirrored.
Also open: 2 white pawns is asymmetric 2/21 (worst +135 at `4k3/8/8/8/8/8/P6P/4K3`) — a separate
pawn-pawn thread.

⚠️ **`piece_value_boost` amplifies whatever remains.** Its ±1500 trigger is written symmetrically and is
NOT itself buggy, but it is a hard step: two positions differing by 61 mp landed either side of it and
came out 830 mp apart — a 13× magnification. Worth softening to a ramp regardless of colour.

## 🚨 2026-08-06 evening — TWO LATENT DEFECTS FOUND; pawn lane closed; 21 knobs + 8.5× corpus banked

**No games run. Engine byte-identical at defaults throughout (`250 / 35,791,173 / EBF 3.804`, verified 4×).**

### ☠️ BLOCKER: the eval is not colour-symmetric
`eval(mirror(b))` must equal `-eval(b)`. It fails in **1,121 / 1,500 = 74.7%** of positions, median 51 mp,
worst **3,738 mp (374 cp)**. Reproduces on four pieces: `4k3/8/8/8/8/8/4P3/4K3 w` vs its mirror, off by 27 mp.
✅ Instrument validated — bare kings / kings facing / startpos all return exactly 0; deterministic across
repeats; no order dependence (ruled out global-state contamination, which matters because our static eval
shares C++ globals). ⚠️ Side-to-move terms are not an excuse: `mirror()` swaps `turn` as well.
**Multiple sources**: `passed_pawn_support` owns the whole KPK case; `passer_king_race_one` (the eval's only
`turn` consumer) is ~7 of 27 mp; but in the worst real position ~2,300 mp of the asymmetry lies OUTSIDE
`passed_pawn_support`. ✅ Ruled out: `boost_pieces_for_supporting_passed_pawns` (its `y > 2` / `y < 5` gates
ARE correct rank mirrors). LATENT, not new — byte-identity held all day and cannot introduce this.
☠️ **Blocks the joint retune**: fitting ~60 constants against a function wrong in 3/4 of its domain lets the
optimiser absorb the bug. A correct fix MUST break the current fingerprint. 🧰 `_eval_symmetry.py`.

### 🐛 The phase blend never completes — a ~20% step
`blend_range = 30` implies a 40→70 ramp, but the blend only runs inside `!isEndGame` (phase ≤ 64), so the
endgame weight tops out at **24/30 = 80%** and then jumps to 100%. Every blended piece type has a
discontinuity of ~20% of `(result_end − result_mid)` at the boundary. Now knobbed (`PHASE_BLEND_LO/RANGE`);
RANGE=24 closes it.

### 📊 Where the pawn lane actually ended
- ☠️ **Sibling invariance REFUTED.** Deleting the pawn terms changes the top move **7.1%** of the time
  (≥25 cp regret) vs the `threats` control's **5.1%** — pawns reorder MORE than the +45 Elo term. The
  mechanism holds only among piece moves (`QUIET_ONLY` collapses spread 49.4 → 6.7 cp); the candidate set
  is not piece moves.
- **Exchange rate, one sample, full ladder**: rook exact (ours 4.98 vs 5.01/5.02/5.09) but **minors ~7%
  below the classical consensus** (N 3.37 vs 3.60/3.67/3.92). Global scale k ≈ 1.0 ⇒ a RATIO defect.
- **`pieces` is the top contributor in 56% of material collapses**, and its excess flips sign against a
  quiet control (+0.27 → −0.23).
- 🚨 **Ranking by cp vs win% INVERTS the conditioning** and the worst-decile sets overlap only 33-53%.
  Retracted a "6× worse when losing / 2× worse in endgames" conclusion on that basis. Error is fat-tailed
  and two-sided (worst decile = 31-36% of win% error) ⇒ a uniform ratio correction was withdrawn.

### ✅ Banked
Corpus **2,713 → 23,113 rows** (bank 24,656, fully SF18-labelled at **d13**, depth matched deliberately —
the script defaults to d18 and mixing depths would put two truth standards in one target column).
New shipped-default baseline **train 228.347 / val 231.177**; snapshots `*_pre0806b.csv`.
**21 new knobs, all gated, all byte-identical off**: `EG_EXIST_KNIGHT/BISHOP/ROOK/QUEEN`,
`MG_CLAMP_KNIGHT/BISHOP_A/BISHOP_B`, `EG_CLAMP_*` (declared, unwired), `ENABLE_WINNABILITY` + 11 `WINNAB_*`,
`ENABLE_CLOSEDNESS` + `CLOSED_N/R/B_PCT[9]`, `PHASE_BLEND_LO/RANGE`. NPS peak **454,491** vs register
445,330 ⇒ no cost. 🐛 `endgame_convertibility_scale` was found **built, wired and never run**
(`ENABLE_ENDGAME_SCALE=0`) — add it to the grid.

## ⚖️ Pawn iteration 3 — shipped regime, widened corpus, `PPS_*` exposed ⇒ **elo ~+18 at 389g, stopped** (2026-08-06)

First descent run **in the SHIPPED regime** (earlier ones optimised inside `ENABLE_KS_CHECK_V2=1`, which we
do not ship, so their winners were never directly applicable), on the corpus widened to **4,987 rows**, and
with the passed-pawn SUPPORT magnitudes exposed for the first time.

    corpus (new baseline)  shipped default  train 230.555 / val 234.881
    fitted                                  train 212.702 / val 213.028   (-21.85 val, all 9 guards held)
    SPRT vs base           +161 -141 =89 @ 389 games, LLR +0.556, elo ~+18  (stopped for context transfer)

★ **`PPS_OWN_ATTACK 60→35` was the single largest move of the round** — those magnitudes (`y*75`, `y*100`,
`y*60`, `y*50` in `boost_pieces_for_supporting_passed_pawns`) had never been tuned at all.
★★ Consistent across all three descents: **pawn values want to come DOWN nearly everywhere, except rank 7
which wants UP.** `PAWN_CLAMP_MID` has now walked **225 → 175 → 140**.
⚠️ The SPRT was volatile and unsettled when stopped: check trajectory **−6, +6, +4, +2, +2, +8, +18**.
Treat +18 as a local upswing, not a converged estimate.
⇒ **Four pawn arms now: −15, ~+4, +12.4 ±30.2, ~+18(unsettled).** None resolvable; none negative except the
first.
☠️ **The "sibling invariance" explanation was tested that evening and REFUTED** (`_sibling_spread.py`, 800
positions × 2 seeds, zero SF): deleting the pawn terms changes the top move **14.0%/16.6%** of the time
(**7.1%** at ≥25 cp regret) against the `threats` control's **10.0%/11.5%** (5.1%) — **pawns reorder more
than the term that won +45 Elo.** The mechanism holds only among piece moves (`QUIET_ONLY=1` spread
collapses 49.4 → 6.7 cp); the candidate set is not piece moves. ⇒ Leading explanation is now simply that
**the effect is ~10-20 Elo and under our measurement floor.** See
`pawn-scoring-may-be-unable-to-change-our-move` (kept as a refutation record).

🚨 **CORPUS CHANGED**: `diverse_corpus_wide` 2,713 → 4,987 rows (bank labelled to 4,925/4,987; ~4 pos/sec at
d13). Snapshots `*_pre0806.csv`. **No `val` measured before 2026-08-06 is comparable to one after.**
⚡ **SPEED, re-established properly**: pinned **peak 445,330 NPS** vs the register's 446,218 ⇒ today's
additions cost nothing measurable. (Median 431k, spread 5.9% — peak-of-N is the right statistic; a
median-of-3 would have shown a phantom slowdown.) 🐛 One genuinely unconditional cost was introduced and
then removed: the `opposed` mask was computed per pawn per eval even at the default. **Byte-identity cannot
detect added work — only added output change.**

## ⚖️ `ISOLATED`/`BACKWARD` ALONE, at last: **+12.4 ±30.2 Elo over 700 games** (2026-08-06)

First time these were ever tested in isolation. `ISOLATED_PAWN_PEN=120 BACKWARD_PAWN_PEN=120` (the value the
joint descent chose — note **2.5× below the hand-picked 200/100** they were previously judged on).

    W 280  L 255  D 165   score 51.8%   elo +12.4  margin ±30.2   LLR −0.013
    decision: inconclusive — hit max_games without crossing a bound

⚖️ **This is exactly what `_venue_power.py` predicted before the run**: resolving +10 Elo needs ~4,344
games; 700 buys ±30. So the honest statement is **"not ≥25 Elo, point estimate +12.4"** — NOT "failed".
★ It was also the **only** pawn arm of the session that never decayed: readings across ten checks were
+7, −3, +8, +8, +11, +17, +17, +20, +14, +9 — a flat non-negative band, versus `pawn2` which spiked to +44
and fell to +4, and iteration 1 which fell to −15.
⇒ The owner's thesis (isolated/backward measure *true structural weaknesses* and should be worth something)
is **not refuted and is mildly supported** — it simply sits under this project's game-resolution floor.
▶️ To resolve it would take ~4,000+ games. Worth doing only if it is bundled with other small positives, or
if a cheaper high-power venue is built.

## ☠️ Pawn iteration 2 — endgame surface + `opposed` + tunable caps ⇒ **elo ~+4, still nothing** (2026-08-06)

Made tunable for the first time: the ENDGAME structural literals (`EG_PHALANX/SUPPORT/DEFEND/LATENT` —
previously hardcoded, reachable by no knob, **never fitted once**), both per-pawn caps, per-phase structural
rank curves, `opposed` (the signal `getPPIncrement` computes then discards), `PASSER_R_MAX`, and the six
`PP_*` constants. Joint descent over ~105 knobs.

| metric | base | iter 1 | iter 2 |
|---|---|---|---|
| corpus val (shipped regime) | 279.51 | 265.53 | **262.10** |
| joint-run ALL.val | 300.37 | 256.71 | **252.32** |
| **SPRT vs base** | — | **−15** (232g) | **~+4** (324g, LLR −0.771, cut) |

☠️ **Three corpus improvements in a row (−7.3, −13.98, −17.4) and no Elo from any of them.** The pawn
subsystem is **0-for-3 in games**. The proxy improved monotonically while the game result did not move.
★★★ Matches `most-eval-error-is-move-neutral`: we keep making the eval agree with SF18 without changing the
moves we play.

**What the descent chose, and what it overturns:**
- ☠️ **Both caps want to go DOWN** (`PAWN_CLAMP_MID 225→175`, `PAWN_CLAMP_EG 175→125`) *even with new terms
  competing for the headroom* — against both the redesign brief and the "new terms need room" intuition.
- ✅ **`opposed` pays**: both phases pinned to the grid minimum 60% ⇒ an opposed pawn's structure is worth
  ~40% less. ⚠️ Corpus evidence only.
- ↩️ **`PASSER_R_MAX` declined the extra headroom** (offered 512, kept 384) ⇒ the earlier "the architecture
  blocks a passer valuation it wants to pay" reading was **overstated**.

## ☠️ Fitted pawn model: −13.98 corpus val, ALL nine guards improved ⇒ **elo ~−15 in games** (2026-08-05)

First time the pawn tables were ever FITTED rather than hand-picked (new per-rank `RANK_*_R2..R7` and
per-file `CHAIN_F_*`/`WALL_F_*` knobs; whole-table `SCALE_*` could only rescale a chosen shape). Joint
win%-descent also switched `ISOLATED_PAWN_PEN` on at **80** and `BACKWARD_PAWN_PEN` at **120** — 2.5× below
the hand-picked 200/100 they had previously been judged on.

| metric | base | fitted |
|---|---|---|
| corpus val (shipped regime) | 279.51 | **265.53** (−13.98) |
| all 9 guard tiers | — | **every one improved** |
| WAC / nodes / STS | 250 / 35,791,173 / 1685 | 245 / 34,724,789 / **1668** |
| **SPRT vs base** | — | **+84 −94 =55, LLR −1.731, elo ~−15** (stopped at 232g, trending to H0) |

☠️ **A −14 corpus gain with every guard improving produced no Elo.** Third proxy failure of the day, after
`+58 STS ⇒ ~0 Elo` (pawn structure) and the obstruction blend. ★★★ Consistent with
`most-eval-error-is-move-neutral`: 52% of our eval error is large but does NOT change the move we play, so
making the eval more SF-accurate moves the corpus number without moving the game result. **Pawn terms look
especially prone to this** — they shift standing evaluations more than they flip candidate moves.
⚠️ Not a clean kill of any single idea: the arm changed ~40 knobs at once. `ISOLATED`/`BACKWARD` in
particular remain untested individually.
⚠️ Caveat: ~4 minutes of the run overlapped a duplicate SPRT (see below), so a handful of games are suspect;
both arms were slowed equally so the bias is largely symmetric.

## 🐛 The `ps` sub reported a live SPRT as dead, and a duplicate was launched on top of it (2026-08-05)

`overnight_runner.sh ps` grepped only `tactical_test|sts_test|movematch|tournament.py|setupAI` — it never
matched **`sprt.py`** or **`pyrun`** jobs. A running SPRT showed "none running", was declared crashed, and a
second SPRT was started against it: seven workers competing on time-controlled games.
✅ Pattern fixed to include `sprt\.py|spsa\.py|gauntlet|annotate\.py|diagnostics/`.
★ Same blind spot had earlier caused two long `pawn_truth_generator` runs to be declared dead while merely
slow. **A negative from a monitoring tool is a claim about the tool until confirmed with `pgrep -af`.**

## ☠️ `ENABLE_IMPROVING` + `ENABLE_CONT_HIST_2PLY` re-tested — both FAIL, theory falsified (2026-07-30)

Both were queued as **revived by the malus ship**, on the triage rule *"does the feature CONSUME history?"*
Re-tested against gravcap (baseline `254 / 35,982,407 / STS 1629`), on the stripped build:

| arm | WAC | nodes | **STS** |
|---|---|---|---|
| `ENABLE_IMPROVING` | 243 (−11) | −0.6% | **1592 (−37)** |
| `ENABLE_CONT_HIST_2PLY` | 250 (−4) | +4.7% | **1541 (−88)** |

☠️ **Both fail on both metrics.** `CONT_HIST_2PLY` had **both** stated preconditions satisfied —
`CONT2_GRAVITY_DIV = 4` (the b/4 down-weight its header demanded) and the bonus/malus rework — and still
lost 88 STS. `ENABLE_IMPROVING` is worse than its banked −5 Elo despite the statScore recalibration that
was the reason to revisit it.
⇒ **The "consumes history ⇒ malus revived it" triage rule DOES NOT WORK.** It was stated in advance that a
positive `CONT_HIST_2PLY` would generalise the theory to `ENABLE_PIECE_CONTHIST` and `ENABLE_THREAT_HIST`;
it was negative, so **those remain speculative, not warranted.**
⚠️ Only `CONT_HIST_2PLY`'s WAC improved vs the earlier 240-under-gravcap note (→250); nodes and STS both
got worse, so that is not a revival.
⇒ Search is **0-for-9** for the session. ★ Both this year's search wins (qdelta, gravcap) came from a
**specific mechanism insight**, never from working through the knob list — which is now the ninth
data point for that pattern.

## ☠️ Ordering quality does NOT substitute for the pre-search (2026-07-30)

The pre-search sells **INTERIOR HEURISTIC POPULATION** (38.7% of all cutoffs) — owner: *"the ordering
gained there is somehow loadbearing; we haven't figured out how to replace it."* Since the pre-search-off
penalty fell **54.7% → 30.7%** when gravcap shipped, the hypothesis was that **history QUALITY substitutes
for pre-search QUANTITY**, which would give a program: stack ordering wins ⇒ trim the pre-search ⇒ the
LARGE node cut the corrected pricing says we need (halving ≈ 1.3 ply, vs 0.23 for `LMR_EXTRA=2`).

★ **Method worth reusing: the pre-search-off penalty is a far better instrument for ordering quality than
STS** — deterministic, node-based, and it moves in large legible steps.

**2×2, all arms `ENABLE_ROOT_TABLE=1` so p-on/p-off are like-for-like:**
| config | p-on | p-off | Δ solves | **node penalty** |
|---|---|---|---|---|
| control | 248 / 39,513,075 | 244 / 47,028,063 | −4 | **19.0%** |
| + `ENABLE_STATIC_ORDER` | 244 / 39,177,402 | 247 / 47,117,611 | +3 | **20.3%** |

☠️ **NO on the exact metric.** Static order does not cut the pre-search's node contribution (19.0 → 20.3,
slightly worse) ⇒ **gravcap's 54.7→30.7 was specific to history MALUS / discrimination, not to ordering
quality in general.** The "stack ordering until the pre-search is cheap" program is **not currently
justified.** ⚠️ The +3 vs −4 solve swing sits inside the ±3-4 band that failed to survive STS or pricing
every time this session — do not build on it.

📋 **UPDATED FIGURE: the pre-search is LESS load-bearing than the notes said** — fresh like-for-like control
is **19.0% nodes / −4 solves** vs the banked 30.7% / 10-14 solves. Measure future work against 19.0%.
🪤 **`ENABLE_ROOT_TABLE` defaults to FALSE**, and the table is what makes p-off viable (111→243 solves); a
p-off run without it returns ~113 solves. I made exactly that mistake and briefly computed a 50.5%
"penalty". ★ **A number that misses the banked figure by a wide margin is a SETUP MISMATCH, not a
discovery** — re-running the control in the same batch is what kept this interpretable.

## ⚖️ THE EXCHANGE RATE — why the search lane keeps losing (2026-07-30, owner's reframing)

★★ **Owner's observation: at d10 SF18 solves LESS tactically than us but is 24pp better positionally**
(STS ours 54.9% vs SF18 79.3%; our WAC is 254/300 = **84.7%, near ceiling**). ⇒ **Every search arm this
session was spending the resource we are POOR in (positional) to buy the one we are RICH in
(tactical/nodes).** That, not bad luck, is why 0-for-7 all priced 5-10× against.
▶️ **WEIGHT STS FAR ABOVE WAC when judging anything.** WAC deltas of ±3-6 are near-meaningless at ceiling;
STS deltas of 25+ are where strength actually lives. (I led with solves all session because the runner
prints them first, and mis-called `ENABLE_STATIC_ORDER` because of it — see below.)

☠️ **PRICING CORRECTION — I had been using the WRONG branching factor.** Ply-equivalents must use the
**real ~1.7/ply** ([ebf-metric-is-not-comparable]), not the **3.82** the bench prints, which that note
explicitly flags as not comparable. Corrected: `ply = ln(1 + saving) / ln(1.7)`, and STS ≈ 54/ply.

| node saving | ply | STS-equivalent |
|---|---|---|
| 3.32% (`RFP_MAX_DEPTH=8`) | 0.06 | ~3 |
| 12.9% (`LMR_EXTRA=2`) | 0.23 | **~12** |
| 50% (halving the tree) | 1.30 | ~70 |

⇒ Node savings are worth **~2.5× more** than I claimed; the "off by 15-25×" figures quoted earlier are
really **7-10×**. Every conclusion holds, the margin was overstated. ★ **Even HALVING the tree buys ~1.3
ply.** Node-saving is a weak lever for us at any magnitude we can reach.

★ **But pruning is BLOCKED, not dead** (owner: *"they do perform hyper aggressive pruning compared to us"*).
SF prunes far harder, uses **61.6× fewer nodes at d10**, and still judges 24pp better. Our pruning is too
**timid**, not too aggressive — and every attempt to fix that costs STS **because the eval cannot support
it**. Causal chain: **weak eval → pruning untrustworthy → 61× more nodes → which buy no judgment.**
⇒ SF's aggression is a **consequence** of a trustworthy eval, not a portable design choice. Eval and
pruning aggression are a pair, and **eval must go first.**

## `ENABLE_STATIC_ORDER` + aggression — ordering is mildly POSITIVE, aggression stays negative (2026-07-30)

Testing the owner's thesis that **move-ordering improvements should enable stronger pruning** (LMP/LMR key
on move INDEX, so what makes late-move pruning safe is that the tail is genuinely bad — an ordering
property, not an eval one). ★ The feature was **already built and gated off**, scoped exactly right:
`STATIC_ORDER_HIST_MAX = 0` = apply only where history is SILENT, i.e. in the **tail**, which is where
pruning safety is decided (`fmc-headroom-ordering-not-bottleneck`: FMC measures the HEAD).

| arm | WAC | nodes | **STS** |
|---|---|---|---|
| baseline | 254 | 35,982,407 | **1629** |
| `ENABLE_STATIC_ORDER=1` | 251 | **−1.6%** | **1637 (+8)** |
| `LMR_EXTRA=2` | 248 | **−12.9%** | **1567 (−62)** |
| corner (both) | 244 | −15.1% | — |

✅ **Static order alone is the only NON-NEGATIVE arm of the session: +8 STS at negative node cost.**
⚠️ I first dismissed it because it lost 3 WAC solves — **the wrong metric**, per the reframing above.
Small and not clearly gateable alone, but real and free.
☠️ **`LMR_EXTRA=2`'s −55 Elo was NOT stale.** Under gravcap it costs **−62 STS** against ~12 STS of node
value, and −62 tracks the original −55 Elo almost exactly ⇒ **gravcap's ordering gain did not buy the right
to reduce harder.**
☠️ The corner is **purely additive** (−3 + −6 = −9 WAC observed −10; −1.6% + −12.9% = −14.5% observed
−15.1%) ⇒ **no interaction**; ordering did not rescue aggression.
▶️ **The thesis is not fully tested though:** `LMR_EXTRA` is a **blunt GLOBAL** reduction increase, applied
even to moves history ranks confidently. The ordering-buys-pruning argument properly applies to
**INDEX-KEYED** pruning (LMP), where better tail ordering directly changes what sits at index 12. **Untested
— backlog.**

## Aggression × guard **Pair B — THESIS FALSIFIED, LANE CLOSED** (2026-07-30)

`[lmr_guards]` counters added first (`pv_saves` / `killer_saves` / `capchain_skips` / `capchain_less`),
each counting the branch that **acts** — a move the reduction would otherwise have taken and the guard
rescued. Refactor byte-identical: **254 / 35,982,407 / EBF 3.820**, all counters 0 with guards off.

The lane's premise was that guards had only ever been measured at **baseline** aggression, where they
cannot pay (null by construction), while the one aggression lever tried without a guard (`LMR_EXTRA=2`)
cost **−55 Elo** — *"both diagonals tested, never the corner."*

| arm | solves | nodes | vs base | guard fires |
|---|---|---|---|---|
| baseline | **254** | 35,982,407 | — | — |
| aggression `LMR_EXTRA=2` | 248 | 31,341,096 | **−12.9%** | — |
| capchain guard alone | 253 | 52,474,625 | **+45.8%** | `capchain_less` 3,833,402 |
| **corner A** agg + capchain | **244** | 43,643,465 | +21.3% | `capchain_less` 3,399,344 |
| **corner B** agg + `PROTECT_KILLERS` | **240** | 35,163,456 | −2.3% | `killer_saves` 600,169 |

☠️ **Both corners are worse than either diagonal.** Corner A (244) sits below aggression alone (248) *and*
guard alone (253); corner B (240) is worse still and returns nearly all the node saving
(−12.9% → −2.3%) while losing **14 solves**.
✅ **Conclusive rather than another ambiguous null because the counters FIRED** (3.8M / 600k acts) — the
"guard over an empty set" failure mode is ruled out, and the counters were built *before* the 2×2.
⚠️ Same **interaction-not-addition** signature as Pair A, now with two independent guard mechanisms.
☠️ The capchain guard alone costs **+45.8% nodes for −1 solve** — effectively disabling LMR inside capture
chains at enormous expense for nothing.
⚠️ **Unexplained, and deliberately not given a mechanism:** reducing *less* should search more thoroughly,
so at fixed depth solves should hold or rise; corner A instead falls below both baseline and aggression.
Outside the ±3 band, and not a clock effect (`TIME_LIMIT = 600 s/move`).
⇒ **Do not reopen by tuning guard thresholds** — Pair A already showed that surface is chaotic, and Pair B
shows the corner itself is the wrong place to stand.

## The q-cache is unsound-but-profitable — and the "masking" lead is **RETRACTED** (2026-07-30)

Knobs `QCACHE_EXACT_ONLY` + `[qcache_hits]` counters, default-off, byte-identical
(**254 / 35,982,407**). All comparisons fixed-depth and deterministic: `LONG_FORMAT` has
`TIME_LIMIT = 600 s/move` and the whole 300-position d10 run takes ~77 s, so **nothing truncates** — an
earlier worry of mine that the no-cache regime was time-truncated was simply wrong.

### Measurements
| config | solves | nodes | q-cache hits |
|---|---|---|---|
| full cache | **254** | 35,982,407 | 1,457,574 (**99.5% bound**) |
| `QCACHE_EXACT_ONLY=1` | **246** | 38,963,635 | 9,568 |
| `DISABLE_QCACHE=1` | **246** | 38,451,470 | 0 |

### ☠️ Do NOT read these as "unsound but profitable" — that reading was wrong
⚠️ **The bound reuse is CORRECT and STANDARD, the same as SF.** `LOWERBOUND` means "true >= score", so
returning it on `score >= beta` is a valid cutoff; `UPPERBOUND` on `score <= alpha` is a valid fail-low.
☠️ **`QCACHE_EXACT_ONLY` is near-tautological.** PVS runs most searches on a **null window**
(`beta = alpha + 1`), and no integer lies strictly between `alpha` and `alpha+1`, so **EXACT is impossible
on a null-window search** — it can only arise from the minority of full-window calls. The 99.5% / 0.5%
split is therefore **exactly what theory predicts**, and disabling bound reuse zeroed the cache because it
disabled **the only path that can produce hits**.
☠️ **"A real cache cannot change fixed-depth results" is also false** — every TT does, since a hit prunes a
subtree and returns a stored bound instead of a fresh fail-soft value (this is why engines are not
bit-reproducible across TT sizes). **254 → 246 on removal is ordinary**, and the 132-STS cost is **not a
puzzle**: it is what removing a working TT costs.

### ✅ Where the q-cache genuinely differs from SF
| | ours | SF |
|---|---|---|
| depth field | **none** | stores depth, requires `entry.depth >= needed` (qsearch at `DEPTH_QS`) |
| replacement | direct-mapped, always overwrite | 3-way bucket, depth + generation priority |
| aging | **none, never cleared** | generation counter refreshed per search |
| abort/draw stores | stored them (fixed, `19c1c21`) | refused |

★ The missing **age/generation** field is the most interesting gap — same root cause as the poisoning bug:
entries outlive the search that created them with no way to prefer fresh ones. The missing depth field is
defensible, since every `get_q_search_eval` call enters at `qDepth = 0`.

### ☠️ RETRACTED — "the q-cache masks eval work"
Measuring a **SEARCH** change in both regimes shows it is amplified as much as any eval change:

| change | type | cached | uncached | shift |
|---|---|---|---|---|
| corrhist RFP-only | eval | +1 | +81 | +80 |
| KS bundle | eval | −58 | +105 | +163 |
| **`LMP_MAX_DEPTH=8`** | **search** | **−170** | **+7** | **+177** |

Uncached absolutes — control **1497**, LMP8 **1504**, corrhist **1578**, KS **1602** — show **all three
changes beating the uncached control, including one that is unambiguously bad.** That is a **pathological
baseline**, not three good changes. ⇒ **`DISABLE_QCACHE` is not a valid measurement regime; deltas measured
in it do not transfer.** It stays a mechanism probe only.
★ **Same lesson as the depth artifact in a new costume: THE REGIME IS PART OF THE CONFIG.** There the bench
DEPTH was wrong, here the CACHE STATE was; in both the individual measurements were correct and
reproducible while the **cross-regime inference** was the error.
✅ **Method to reuse: before believing any cross-regime delta, measure a change of the OPPOSITE KIND in both
regimes.** If it moves too, the effect is generic.

## The d10 depth-artifact audit — **artifact CONFIRMED, 0-for-2 on rescuing knobs** (2026-07-30)

Re-tested the depth-capped knobs at d10 AND d12. Bases: d10 `254 / 35,982,407 / STS 1629`,
d12 `268 / 106,879,693 / STS 1737`.

| knob | d10 Δnodes | d12 Δnodes | ratio | d12 verdict |
|---|---|---|---|---|
| `RFP_MAX_DEPTH=8` | **−0.008%** (2,895 nodes) | **−3.32%** | **~415×** | ☠️ −4 solves, **STS 1712 (−25)** |
| `LMP_MAX_DEPTH=8` | +2.8%, −8 solves | +5.35%, −5 solves | ~2× | ☠️ bad at BOTH depths |

☠️ **The strong hypothesis is FALSE.** The audit was opened on the expectation that it might reopen failed
search work in bulk. It did not. The artifact is **real and large on `RFP_MAX_DEPTH`** — a d10 sweep of that
knob moved 2,895 nodes out of 36M and would have reported "null" for any value — but **measuring it properly
at d12 still says no**, and `LMP_MAX_DEPTH`'s d10 read was **directionally correct**.
⇒ **Do NOT assume a d10 null on a capped-depth knob is wrong; check the specific knob.**
📐 Ply-priced: −25 STS ≈ 0.46 ply bought with 3.32% nodes ≈ 0.025 ply ⇒ **off by ~18×**.
★ **Untested rule that fits n=2:** **node-level** prunes (RFP removes a whole subtree near the root) are
depth-sensitive; **move-level** prunes (LMP skips late quiets) are not. Predicts `HIST_PRUNE_MAX_DEPTH`
behaves like LMP. Would give a rule for which knobs ever need a d12 check.

🏛️ **✅ KEEP `RFP_MAX_DEPTH=6` — and a portability prediction confirmed.** Solves fall **monotonically**
with the cap at d12: **268 (6) → 264 (8) → 259 (11)**. SF's 6→8→11→14 is monotone across versions, which by
the portability heuristic means it **tracks a capability**; `sf-pruning-schedules-comparison.md` guessed
that capability is **eval trust via NNUE + corrhist**. RFP at high remaining depth prunes on a static eval
far from the leaf — and we have neither NNUE nor working corrhist. ⇒ **Our monotone solve loss is evidence
FOR an inference previously flagged "not stated in the source,"** and our value is correctly matched to our
eval quality rather than merely stale.

## Q-cache stored values the search never produced — **SHIPPED on correctness** (2026-07-30, later session)

Commits `daf3adf` (instrumentation, default-off) then `19c1c21` (`QCACHE_SOUND_STORE` default **on**).
Byte-identical throughout: **254 / 35,982,407 / EBF 3.820**.

`qSearch` returns a bare `0` on three paths that are not evaluations — **timeout abort, node-limit abort,
and repetition draw** (the last a property of the PATH, not the position). `get_q_search_eval` cached all
three, tagging them EXACT/LOWER/UPPER by comparison against the window. `quiesceEvalCache` has **no
generation or age field and is never cleared**, so a `0` written during one move's timeout unwind was
served as a real score for the rest of the game.

★ **The probe/store logic is textbook-correct** (EXACT always usable, LOWERBOUND only on a beta cutoff,
UPPERBOUND only on an alpha cutoff). The unsoundness is **entirely upstream, in what qsearch hands it** —
reading the cache code alone would never find it.

🚨 **Why no bench could see it, and why it shipped anyway.** `[qcache_hygiene]` reads **0 at fixed depth**
and **144 timed** (`PRESET=LIGHTNING MAX_DEPTH=30`, 300 positions). The blocked paths are unreachable at
fixed depth ⇒ the fix **cannot regress a bench by construction**, so it went in on correctness rather than
on a measurement. ⚠️ Timed WAC cannot judge it either — fixed-TIME drifts up to 125 solves on an unchanged
config; the 252 vs 250 observed here is noise.

☠️ **The main TT was never affected** — an earlier claim in this session that it shared the hole was wrong.
The unguarded `addToSearchEvalCache` overload sits inside a `/* ... */` block (closed at
`cache_management.h` L905); the live overload refuses `score == 0` unless `ENABLE_TT_STORE_DRAW`, and the
fabricated abort value **is** exactly 0. 🪤 **Therefore `ENABLE_TT_STORE_DRAW=1` removes that protection**
and starts caching aborts as real bounds — now warned about in the knob's header comment.

⚠️ **Realistic value: small.** ~0.5 poisoned stores per timed search ⇒ ~30 bad entries per game in a
**8.4M-entry** (`CACHE_SIZE = 1 << 23`) direct-mapped cache, many overwritten before being read. **It is
very unlikely to explain the lightning/standard game blunders** — those were ~1200 cp swings, the shape of
a systematic assessment error, not a rare stale leaf. The better-fitting hypothesis remains warm state in
the non-position-keyed history/killer/countermove tables. One amplifier argues against fully dismissing it:
the poisoned positions are exactly those being searched when the clock expired, so they are unusually
likely to recur on the next move.
▶️ **The one untested number that could change this verdict:** `draw_stores` read 0 only because WAC has no
repetition history. Games repeat constantly. Measure it in a real game — the counter is already in the build.
▶️ Delta pruning's unsound UPPERBOUND (`return static_eval` for a claim that only earns "true ≤ alpha") is
**dead code by default** — gated `!ENABLE_QDELTA_PERMOVE`, which ships `true`. Nothing to chase.

## `ENABLE_LMR_REMDEPTH` — remaining-depth LMR — **CLOSED, no gateable arm** (2026-07-30)

Commit `f8b3d11`, default-off, byte-identical. `reduced_search_depth` indexes `DEPTH_REDUCTION` with the
ITERATION depth, so one reduction constant hits every node and near the horizon truncates the child into
qsearch. The knob re-derives the base from the node's own remaining depth (`LMR_REMDEPTH_SCALE` = the
aggression dial), never letting the reduction consume the child's last ply.

🚨 **The durable lesson is about measurement: for a depth-keyed mechanism, the BENCH DEPTH is part of the
config.** `DEPTH_REDUCTION[D] = D − 1` for every D ≤ 9, and with root LMR off `rem` never exceeds 9 at
`MAX_DEPTH=10` ⇒ **the d10 bench sits entirely in the table's flat region while real timed play sits past
it.** The counters prove the two regimes are different features: `avg_ply_delta` is **−0.355** at d10
(a blunt 3:1 aggression shift) but **−0.0015** at d12 (balanced — the intended shape).
⚠️ The standing rule "run STS on the EXACT config being gated" was **obeyed and still gave a false verdict**,
because the config was held fixed while the depth was not. ★ Owner's catch.

| depth | SCALE | solves | nodes | vs base | STS |
|---|---|---|---|---|---|
| d10 | 100 | 248 | 40.50M | +12.6% | — |
| d10 | 200/250 | 256 | 36.32M | +0.9% (reads as a WASH) | 1595 (−34) |
| d10 | 300 | 249 | 32.49M | −9.7% | 1501 (−128) |
| d12 | 150 | — | 123.50M | **+15.6%** | — |
| d12 | 200 | 265 | 95.01M | **−11.1%** | **1627 (−110)** |
| d12 | 300 | — | 92.84M | −13.1% | — |

☠️ **No plateau exists** — 150→200 swings nodes 27 points with nothing between, because at the dominant
rem 4-9 the integer division can only yield 1 or 2 plies (which is also why 200 and 250 are byte-identical
at d10). 📐 **Priced in ply-equivalents:** baseline STS 1629@d10 → 1737@d12 ⇒ **~54 STS/ply**, so −110 STS
≈ 2 plies bought with an 11.1% node cut ≈ 0.08 ply — **off by ~25×**. ⚠️ Triage only; gravcap (−29 STS,
+33 Elo) is the standing counterexample.
▶️ **Re-examine every depth-keyed knob swept only at d10** (`LMP_MAX_DEPTH`, `HIST_PRUNE_MAX_DEPTH`, RFP
cap, `DEPTH_REDUCTION`) — part of the search 0-for-13 record may be this artifact.

## `ENABLE_CORR_HIST` re-wire — **CLOSED, the defect fix made it worse** (2026-07-30)

Commit `4115315`, both knobs default-off, byte-identical. Corrhist reached only `rfp_static_eval` (RFP +
the null-move eval gate) where SF applies the correction once at the `staticEval` assignment so every
consumer inherits it. Audit of the rest:
- ☠️ **`improving` is inert by construction** — it compares eval vs eval 2 plies apart, and a **pawn-key**
  correction rarely changes in 2 plies ⇒ it cancels in the difference. (Argued, **not measured**.)
- ✅ **qsearch stand-pat + horizon** were the real gap: bound comparisons, where an offset does not cancel.

`[corrhist_q] seen=8,085,408 flips=91,525 (1.13%)` ⇒ genuinely live, not inert. **And still harmful:**
matched no-qcache batch gives control **1497**, RFP-only **1578**, +qsearch **1529** ⇒ the extension costs
**−49 STS** against RFP-only *with the cache bypassed*, so it is not the cache artifact it first looked
like. ★ **Why SF can and we can't:** SF keys corrhist four ways (pawn + minor + non-pawn×2 + continuation)
with divisor 131072 — a fine, small correction; ours is ONE coarse pawn table, and a **structural**
correction mis-prices **tactical** leaves. Port FORMS, refit CONSTANTS — biting on the **KEYING** this time.

★★ **The incidental find is bigger than the lane: the q-cache MASKS eval work.** Corrhist RFP-only is flat
with the cache on (1630 vs 1629) and **+81 STS with it off** — while using **more** nodes (WAC no-qcache:
control 246 / 38,451,470 vs corrhist 247 / 38,720,267), which **falsifies** the obvious confound that the
time-truncated no-cache regime merely rewards node savings. ⇒ Cached qsearch values override corrected
evals downstream, which would suppress **any** eval-side improvement routed through qsearch.
⚠️ Still unexplained: removing a bound-checked q-cache costs **132 STS** (1629→1497) for only ~7% more
nodes — a large quality swing for a supposedly sound lookup. `DISABLE_QCACHE` is **diagnostic only**.

## `gravcap` — history gravity + capture history + recalibrated statScore — **SHIPPED, +33.0 Elo** (2026-07-30)

Commits `6e26ffd` (gated infrastructure, byte-identical) then `5a8655e` (defaults flipped).
**+506 −392 =305 over 1203 games = 54.74% ⇒ +33.0 Elo, 95% CI [+16.1, +50.1]** (`gate`, LIGHTNING, conc 4,
tag `sprt_gravcap`). First search-lane win since the per-move qsearch futility fix.

```
ENABLE_HISTORY_SATURATION  false -> true      # saturating h += delta - h*|delta|/MAX_HISTORY
ENABLE_HISTORY_MALUS       false -> true      # penalize quiets/captures tried-and-failed before the cutoff
ENABLE_CAPTURE_HIST        false -> true
STATSCORE_OFFSET           512   -> 0         # re-derived from the POST-gravity distribution
STATSCORE_DIVISOR          1024  -> 683
```

★ **Malus is a PRECONDITION, not a feature.** Without it history accumulates only bonuses, saturates and
stops discriminating — which is why every history CONSUMER had measured null or inert: capture history read
"marginal" (+0.7 STS/+1 WAC), `ENABLE_HIST_PRUNE` was literally inert (it prunes on *negative* history,
which a bonus-only table never produces), and malus alone scored +10.9 ±36.6. The whole machinery was built
and wired at every cutoff site, gated off, for weeks — this was a measurement failure, not a missing feature.

⚠️ The five values are **one atomic unit**: keeping the old 512/1024 constants under gravity costs **−90 STS**,
because they grade QUIET moves and gravity changes the scale of every table statScore reads.

**BASELINE CHANGE — old fingerprint RETIRED:**

| | pre-gravcap | **post-gravcap (current)** |
|---|---|---|
| WAC | 249/300 | **254/300** |
| nodes | 38,840,709 | **35,982,407** (−7.4%) |
| EBF | 3.934 | **3.820** |
| STS | 1662/3000 | **1629/3000** |

NPS unchanged (467k → 468k, median-of-3), so the node cut is not a speed artifact; fixed-time depth
+0.10–0.16 ply. The pre-gravcap engine remains reachable and **verified reproducible** via
`ENABLE_HISTORY_SATURATION=0 ENABLE_HISTORY_MALUS=0 ENABLE_CAPTURE_HIST=0 STATSCORE_OFFSET=512 STATSCORE_DIVISOR=1024`.

### ☠️ Measurement post-mortem — every bench missed it or argued against it
| stage | said | truth |
|---|---|---|
| WAC / STS | +2 solves, −29 STS ⇒ "flat" | blind to it |
| cploss tail screen, **1 run per side** | "+58% blunders — **cancel the SPRT**" | **noise** |
| same screen, baseline replicated ×4 | `rate>10%` = 5.3 / 4.9 / 6.7 / 6.5% | cannot resolve <2pp |
| 1203 games | **+33.0 ±17 Elo** | ✅ |

**Only games found it**, and it was nearly discarded twice. Two durable rules came out of this:
**replicate the control in the same batch before believing any delta**, and **`elo1=5` makes SPRT LLR crawl
regardless of true strength** — this run sat at LLR +2.06/2.944 while the point estimate was already +33, so
score the PGNs directly and use `elo1` 20-30 when a large effect is expected.

## King-safety swap Phase A + detector-placement DISCONFIRMED (2026-06-27)
After KPvK shipped, the campaign turned to the residual gap = `pieces`/placement VARIANCE (collapse term-
attribution: passers=0.00, so NOT a passer hole; one under-modeled-king-safety under-read). **Detector-
conditioned PLACEMENT tested cheaply offline (held-out fit, curated 5037-pos midgame corpus, per-piece-type ×
14 detectors, ridge) → DISCONFIRMED: conditioning OVERFITS (−4.9% beyond a per-piece flat scale); the variance
is NNUE-territory.** Only generalizable gain = per-piece-type FLAT placement recal (+14.6% static, but MSE≠play
→ play-gate; Phase C). Tooling: `diagnostics/detector_placement_proof.py` / `gen_midgame_corpus.py` /
`fit_conditioned_placement.py`. **Go-forward (user, the pre-NNUE HCE beef-up): REPLACE flat latent_threat with
the high-DOF attack-unit `king_safety_score`, data-tune it.** **Phase A SHIPPED gated (`cff38eb`, pushed):**
`ENABLE_KS_REPLACE_LT` (default false → byte-id 252/70,150,573) routes king danger through king_safety_score
instead of latent_threat (no double-count). **Neutral platform @`KING_SAFETY_MAG=600` (default knobs): STS 1548
(−20) / WAC 252 / +0.12 ply at equal time** — the untuned rich term matches the evolved flat one (the floor).
**Phase B (next): data-tune the KS shape knobs via PACE move-match + SPRT** (Texel static-fit is diagnosis/init
only — it degrades play; control-constrained static-gap safely inits the shape). DOUBLE-WIN thesis: cheaper term
→ speed + tuned positional; downstream revives eval-speed-dependent search. Plan/log:
`~/.claude/plans/handoff-lossless-speed-campaign-tranquil-rose.md`, `dev_notes/collapse-campaign.md`.
**Phase B RESULT (2026-06-28): tuned `div6` (KS_DIVISOR=6) OVER-FIT — tuned on a 4-theme king-SUBSET (+73), but
−198 on the FULL 15-theme move-match (STS −99) = discard. Untuned `swap@600` is the real candidate but MARGINAL
(+40 full move-match, −20 STS ≈ neutral), never play-tested. The +22.7-Elo overnight tournament was on div6 and
was small-sample noise (batch2 post-power-outage −16, pooled +6.7 flat; cross-outage TIMED-tournament confound;
.so verified intact WAC 252/STS 1568 exact). LESSON: never tune eval knobs on a narrow theme subset (over-fits);
tune the FULL suite with controls, gate on full move-match. NEXT: re-tune from swap@600 with control sets, else
the COMPLEMENT (latent_threat + only KS safe-checks = clean +13 STS) or park.**

## Phase-2 collapse fix #1 (2026-06-27) — rook-pawn KPvK draw SHIPPED (chesscom-2200 conversion loss)
First Phase-2 worst-case hole, mined from `selfplay/external/chesscom_2200_white.pgn` (NN-Engine=White vs a
chess.com 2200 bot: won a pawn move 29, drew). **EVAL hole (not horizon): a drawn lone rook-pawn KPvK read
~+4870** (`piece_value_boost`/mate-drive on a 1-pawn lead, no draw detection) — d6→d18 all +4.4..+5.0, so
search can't fix it. Caused the half-point loss: at move 56 the engine, seeing the resulting KPvK as +4.7,
**traded rooks (Rxf5) INTO the dead draw**. SF confirms all critical positions = 0.00 (it keeps the rook).
**Fix:** new lone-rook-pawn KPvK case in `is_practically_drawn` (returns 0), gated `ENABLE_RP_KPK_DRAW`
(search_engine.h), mirroring the existing KBP/KN rook-pawn blocks; rule = `defender_dist <= min(pawn_dist,
attacker_dist)` to the promotion corner. **Validated against a full KPvK retrograde oracle (`diagnostics/
_kpk_oracle.py`, 83,238 states): 0 false-draws (never flags a won position drawn).** Position-fix: KPvK
+4870→0, move-56 Rxf5→f8h8 (keeps rooks), benoni-29 still a4b5. **No-regression: WAC 252/70,150,573 (byte-id)
+ STS 1568 (identical)** — touches only rook-pawn KPvK, never in the suites. **SHIPPED default-on** (gate:
self-play-invisible → tournament uninformative; ship on position-fix + no-regression per [[external-play-gaps]]).
Baseline unchanged WAC 252/70,150,573/STS 1568. Uncommitted (default-on in tree + `_kpk_oracle.py`/
`_chesscom_gap_fens.py` tooling). [[endgame-draw-detection]] (the is_practically_drawn family this extends).

## `is_light` v2 (2026-06-27) — stand-pat lever DEAD; FUTILITY-light = clean win (testing)
Goal: a cheaper STANDING eval to free per-node budget. **Stand-pat is the WRONG target — it's the LEAF eval
for quiet positions** (qsearch returns it; it backs up to pick moves), so cheapening `QSTANDPAT_EVAL_MODE`
craters positional play: mode1 (material+PST) STS@time **1266** (−235, +0.64 WAC ply), mode2 (new
surrogate-light) STS@time **1243** (−258). Stand-pat must stay FULL. improving + null-move ALREADY use
cheap_eval (improving SPRT-null → speed was never its problem). **FUTILITY (`FUTILITY_EVAL_MODE`, default
0=full; a fail-high prune that only `return early_score` = semi-load-bearing) is at least SAFE (no crater)
but its effect is bench-AMBIGUOUS: `FUTILITY_EVAL_MODE=1` standalone = WAC 253(+10)/STS@time 1511(+10), yet
in the `evalmode_sweep` (OMP=1) = STS@time 1367 (−107) — a SIGN FLIP (timed benches are wall-clock/thread
noisy). **KEY: BOTH proxies fail for SMALL effects (fixed STS mispredicts; timed-depth wall-clock-noisy) →
only a TOURNAMENT resolves a small change.** ⇒ the is_light track yields little: stand-pat DEAD, safe sites
already-cheap or too-small-to-confirm. **Built gated/byte-id 252/70,150,573 UNCOMMITTED:** `KS_LIGHT_MAG` +
light-only cheap queen/knight mobility + light king-safety surrogate (superseded). Higher-ROI next:
collapse-mining Phase 2 (+38.7) or king-safety graduation. [[lighteval-standpat-is-leaf]].

## COLLAPSE-ELIMINATION campaign (2026-06-26/27) — bundle SHIPPED, +38.7 Elo; "STS mispredicts PLAY"
Worst-case-Elo hole-plugging from real chess.com tal-BOT losses (self-play-invisible). Re-validated the
parked fixes on the post-combo1 baseline (252/67,931,145/STS 1503) and shipped a 6-knob bundle as default:
**VERIFY_MARGIN 6000→16000** (Gap-T, fixes the benoni-29 winning-capture: now plays a4b5) +
**ENABLE_ROOK_DBLCOUNT_FIX + ENABLE_ROOK_DBLCOUNT_SYM_UP** + **ENABLE_QPREC_PHASE_GATE** +
**PASSER_ENEMY_CREDIT_PCT=0 + ENABLE_PASSER_BLOCKADE_QUALITY** (Gap-P, fixes the french a-pawn passer-danger
under-read: F33 +0.93→−0.32). **Tournament bundle vs shipped-Gap-T baseline = +38.7 ±27 Elo (875 games,
SIGNIFICANT).** NEW BASELINE **WAC 252 / 70,150,573 / STS 1568 (52.3%)**; speed maintained (~162s / 432k
nps); old byte-id (252/67,931,145) recovers with the 6 knobs reset to old defaults.
- **▶️ KEY METHODOLOGY FINDING — fixed-depth STS MISPREDICTS PLAY for eval/search changes.** On STS the
  bundle looked DESTRUCTIVE (gapt+rookdblsym −77..−139, "non-additive"); in TIMED games it is +38.7. The
  "never bundle / non-additivity" scare was a FIXED-DEPTH SEARCH-HOLE ARTIFACT. Nearly parked Gap-P (the top
  contributor) on its −106 STS. ⇒ **gate collapse/eval/search changes on the TIMED TOURNAMENT (or
  depth-at-equal-time), NOT fixed-depth STS.** ([[fixed-depth-bench-ceiling]] far more severe than assumed.)
  Running log: `dev_notes/collapse-campaign.md`.

## SEARCH/EBF campaign (2026-06-21) — SEE_EXTEND_MARGIN=300 SHIPPED, +43.9 Elo
Single-thread search-efficiency campaign (proxies predict strength → tuning converges). Baseline going in:
WAC 261 / 134,429,469 / STS 52.2%.
- **SHIPPED & committed (`5c194ae`): controlled check extensions, `SEE_EXTEND_MARGIN` default DISABLED→300.**
  The check-extension extended ALL checks (huge node adder); SEE-filtering extends only checks sac'ing ≤300.
  Pooled SPRT (2 runs, n=907, LIGHTNING/UHO) = **+43.9 Elo [+22.8,+65.3]**. Fixed-depth dip (−3 WAC / −4.7%
  STS) is the expected pruning artifact; depth-at-equal-time converts it to Elo. **NEW GOLD: WAC 258/300,
  97,507,126 nodes (−27.5%), EBF 4.20→3.83, STS 1426/3000 (47.5%)**; env-off `SEE_EXTEND_MARGIN=1000000`
  recovers old gold byte-exact. (Frontier: SEE=0 −31.5%/−9WAC; 300 is the sweet spot. Stacks on it — VERIFY,
  CHECK_EXT caps — all worse.)
- **KILLED by the measure-first gate (no code shipped): lazy/"light" eval** (skip expensive eval terms at
  qsearch stand-pat/futility). Eval-cache miss 79.4% = addressable, BUT the light-eval gap is ~85%
  `capture_gains` (median ~1.5 pawns) → the costliest tail term can't be skipped safely → low skip-rate.
- **SHELVED (gated default-off, byte-id): SEE cache + qsearch SEE re-sort.** Phase-A instrumentation measured
  199M see()/6.24-per-qnode and qsearch ALREADY well-ordered (qfmc 77.2%, qcut 0.31). `ENABLE_SEE_CACHE`
  (per-position [side][square], gen-validated) byte-identical but only 8.7% hit (see calls mostly distinct).
  `ENABLE_QSEE_RESORT` (noisy list by SEE-desc) −7 WAC fixed-depth / timed-depth NEUTRAL.
- **PARKED candidate: improving-cheap.** Revived the shelved improving heuristic with a standalone
  `cheap_eval()` (material+PST, sign-only trend) — net-+ on WAC timed-depth (+3 solves/+0.106 ply, beats
  full-eval improving). Knobs `IMPROVING_CHEAP`/`IMPROVING_REDUCTION`/`IMPROVING_DELTA_MARGIN` exposed
  (byte-id). Default-off, needs SPRT (deeper≠stronger caution).
- **BACKLOG: incremental-SEE rewrite (~3% search, byte-id).** `see_impl` recomputes the full attacker set each
  exchange iter; knight/king/pawn attackers never change, only sliding x-rays. Precompute non-sliders once /
  incremental x-ray → ~1.5–2.5×. Do the rewrite NOT the cache (cache+rewrite ≈ rewrite alone). Profiler re-run
  (post bishop/rook surrogates): PAWNS hottest eval term 20–28%; ATTACK_LAYER already cheap (cached).
- **▶️ NEXT:** PACE joint-tune (improving knobs + expose hardcoded formulas: `depth²` history bonus,
  `DEPTH_REDUCTION` table, decay) on reliable proxies → overnight SPRT. Memory [[search-ebf-campaign]],
  [[improving-heuristic-shelved]]; dev log `dev_notes/search-ebf-campaign.md` (§Session 2026-06-21).

## Eval-tuning system (2026-06-16) — tooling, not a shipped gain
Built the `/eval-tune` Claude skill + its instruments to tune eval by **move-match** (the proxy that tracks Elo) with a rare SPRT gate. Committed: `1113c9c` (`SCALE_*` per-term knobs, `tune_corpus.py`, `tune_fit.py`), `36cc20c` (`diagnostics/movematch.py` + dispatcher `movematch`/`movematch_diff`), skill at `.claude/skills/eval-tune/SKILL.md`. **Findings:** eval-match (matching SF static) does NOT predict strength — every fitted candidate degraded STS, and `SCALE_LATENT_THREAT=144` regressed King-Activity move-match 426→376 (caught cheaply by the proxy); scalar term-scaling is too coarse (bulk of the eval scatter is in PST). **NEXT = finer PST/term-internal knobs** to give the loop headroom. Memory [[agentic-eval-tuning-system]].

## Post-speed-track strength arc (2026-06-05 → 06-11) — narrative; details in HANDOFF.md + memory

The table below is the **speed track** (done: ~16.7s → ~9.2s, EBF ≈ 3.4, pruning-bound). Everything after is the **strength track**:
- **SHIPPED (committed):** color-symmetry fix (mirror residual 0.000); endgame draw-detection (`is_practically_drawn`: R+N-vs-R/KRKN/KRKB); continuation-aware LMR `ENABLE_CONT_HIST`@`THRESH=2000` (−6.2% nodes); reduce-more history-LMR (`HISTORY_LMR_MORE_CAP=1`). Harness: concurrency 6.5×, timed mode, UHO book, paired/parallel annotate.
- **REVERTED:** `ENABLE_ENDGAME_SCALE` (scale-on STS bench = −3.3 STS suite regression; the FEN spot-checks under-sampled it).
- **SHELVED (gated default-off, dormant knobs, all byte-identical):** the 3 ordering features (2-ply CH / capture-hist / check-order); the gravity/malus family (decomposed into `ENABLE_HISTORY_SATURATION` + `ENABLE_HISTORY_MALUS` + `MALUS_DIV`/`MAX_HISTORY`/`CONT2_GRAVITY_DIV`/`ENABLE_HISTORY_DECAY`) — isolation matrix proved both halves hurt independently; the **improving heuristic** (`ENABLE_IMPROVING`/`IMPROVING_EVAL_WINDOW`) — lightning showed −0.18 ply (eval cost > node savings). Conclusion: **move-ordering/reduction lane is closed** (pruning-bound + eval-bound).
- **TOOLING:** found d10 **STS had a ±30 noise floor from flaky opening-book-hit detection** → bench with `USE_OPENING_BOOK=0` (deterministic). New no-book baseline: **WAC 262 / 267,284,369 nodes / STS 1487**.
- **NEXT = EVAL** (the double bottleneck: imprecise + slow). First step: per-term profiling across game phases (this file's BASELINE_PERF.md sibling is the perf anchor).

## Eval-speed track (2026-06-11 → 06-15) — SHIPPED

Per-term `__rdtsc` eval profiler built (`PROFILE_EVAL=1`); hottest terms made lazy/cheaper, each gated (byte-identity- or self-play-gated) then default-on:
- **cheap-bishop** (`ENABLE_CHEAP_BISHOP_COMPLEX`, `dd8486e`): popcount surrogate for the colour-complex flood-fill (~33% of midgame eval). ~25% cheaper midgame, STS +33.
- **rook mobility surrogate** (`ENABLE_CHEAP_ROOK_MOBILITY`, `535513f`): popcount surrogate for the per-rook mobility sub-block (drops the 5-way lower-value-attacker test + nested 2nd-order scan). +7.8% nps, WAC +2, STS +18; profiler ROOKS 34%→12% endgame.
- **attack-layer caches** (`ENABLE_ATTACK_LAYER_CACHE` + `_MIDGAME`, `535513f`): lossless content-keyed caches of `setAttackingLayer`'s king-danger layer (endgame = pure fn of the two king squares; midgame = two independent white/black half-caches keyed over the king 2-ring `king_ring2`). EG ≈ free on WAC (pays in endgame play); MG +0.8% nps.
- **Bundle total: +8.6% nps / −3.8% wall @d10; self-play +11.2 ±20.7 Elo (1487 games). New baseline WAC 262 / 260,960,881 / STS 52.3%.**
- **REJECTED (gated default-off):** queen mobility surrogate (STS −66 — the don't-count-enemy-defended-squares filter is load-bearing for the queen) and knight mobility surrogate (WAC −3 / +11% node bloat — term already cheap, eval change only disrupts pruning). **Lesson: the cheap-mobility trick does NOT generalize past the rook** — it wins only when the dropped detail isn't load-bearing for that piece AND the term is expensive enough to beat the search disruption.
- **NEXT = EBF sprint** (LMP + SEE-pruning, both ABSENT in the main search; cutoff move-index histogram diagnostic first; re-test improving now eval is cheaper). See STRENGTH_BACKLOG.

## EBF sprint (2026-06-15/16) — LMP + lazy-hybrid quiet re-sort — **SHIPPED default-on, +59.2 Elo**

EBF ~4.65 @d10 (vs ~2 optimal); the shipped cheaper eval makes eval-cost-gated pruning affordable. Cutoff move-index histogram (`g_cutoff_histogram`, `be5a1b1`): **96% of cutoffs at move 0/1 ⇒ pruning-bound, not ordering-bound** (ordering only reaches the √B≈6 minimal-tree floor; the gap to ~2 is pruning the all-node width, which the histogram — counting only cut-nodes — never sees).
- **LMP** (`ENABLE_LMP`/`LMP_BASE`/`LMP_MAX_DEPTH`/`LMP_SCALE`, gated-off, `8eaf80e`): late-move pruning in `get_score_for_{min,max}imizer` reusing the `do_lmr` eligibility (never prunes captures/checks/promos/killers/in-check). Alone @B3/MAXD3: −47% nodes / WAC **255 (−7)** / STS 51.5% — powerful but over-prunes at fixed depth (the lost solves are history-invisible positional quiets).
- **Gravity/malus × LMP — DROPPED.** Re-tested WITH the LMP consumer it was supposedly waiting for; still net-negative on STS at every `MALUS_DIV` (d2 46.7 / d3 49.5 / d4 47.2 vs 51.5 control; WAC noise; B2 push-harder 255 < control 260). **Root cause (falsifies the prior "no consumer" theory): malus pollutes the global `[side][from][to]` history table → demotes good non-cutting quiets everywhere → shallow/eval-bound search can't absorb it.**
- **Lazy-hybrid cached-quiet re-sort** (`ENABLE_LAZY_RESORT` + `PROMOTE_TOP_K`/`RESORT_AFTER_REUSES`/`LAZY_RESORT_MIN_CUTOFF_IDX`, gated-off, byte-identical 262/260,960,881): on a move-gen cache hit, re-rank the stale quiet tail by LIVE `score_quiet` (standalone helper in `move_gen.h`), pinning captures+killers/counter; data in `MoveEntry.last_cutoff_index`/`reuse_count`, cutoff index recorded in `updateMoveCacheForBetaCutoff`. **AGGRESSIVE (full resort, K=4, idx>1): WAC 261 at ~0 nps cost BUT STS 47.0% — the full re-sort over-concentrates survival on history = malus's failure mode.** **🔒 GENTLE (no full resort, K=2, idx>2): WAC 261 / STS 52.2% (held) / 134.4M nodes (−48.5% vs baseline) — baseline strength on BOTH benches at half the nodes.** B2-push (256) & K=3 (251) worse ⇒ B3/K=2/idx>2 is the operating point; the periodic full re-sort is the harmful part (shelved, knob retained via high `RESORT_AFTER_REUSES`).
- **Timed-depth confirmation** (new `wac_timed_depth`/`sts_timed_depth` dispatcher subcommands; lightning, uncapped depth, mean depth excl mate-found): at EQUAL TIME the gentle config is **+0.53 ply (WAC) / +0.57 ply (STS) deeper AND +3 WAC / +2.3% STS stronger** — the fixed-depth −1 inverts to a gain (the d10-ceiling nuance). Fixed a latent `tactical_test.py` bug en route (eval/nodes CSV columns were blank — Python-print stdout buffer not flushed before the fd-capture read; mate sentinel = 9999995).
- **SHIPPED default-on (2026-06-16):** overnight tournament base vs the gentle config = **+59.2 ±19.2 Elo** (1729 games, LIGHTNING, SF18-adjudicated); `summary.py` confirmed the mechanism in-play = **+0.50 ply at equal time** (11.62 vs 11.12, −12.7% nodes). Defaults flipped in `search_engine.h` (`ENABLE_LMP=true` B3/MAXD3, `ENABLE_LAZY_RESORT=true`, `PROMOTE_TOP_K=2`, `LAZY_RESORT_MIN_CUTOFF_IDX=2`, `RESORT_AFTER_REUSES=1000000`). **NEW DEFAULT BASELINE: WAC 261 / 134,429,469 / STS 52.2% / EBF 4.20**; env-off (`ENABLE_LMP=0 ENABLE_LAZY_RESORT=0`) recovers 262 / 260,960,881 byte-exact.
- **ANALYTICS (`paired_analysis.py` + new `selfplay/loss_patterns.py`):** fast better at every imbalance bucket, wins from behind (53.5% disfavored) → not opening luck; losses are NOT bad starts (87% equalish) but **~70% collapse-from-okayish + ~25% squandered-winning in long games** (median 89 plies) = the eval/horizon weakness, shared with base. ⇒ **eval/horizon precision is the #1 remaining lever.**
- **Passer-exemption (`ENABLE_PASSER_PRUNE_EXEMPT`, gated default-off): FAILED its gate** — ADV=5 rank-proxy over-fires (825k fires, +12% nodes, STS −3.8%); needs the true-passed-pawn-mask refinement.
- **COLLAPSE DIAGNOSTIC DONE (2026-06-16; `flip_extract.py` + `fen_vs_sf.py`):** of 573 losses, 401 flips, **90% MIDGAME = eval OVER-OPTIMISM** (we over-read vs SF and the error GROWS with depth — deeper search steers into positions the biased eval over-rewards → fantasy advantage → collapse); passer/endgame ~10% = horizon. **⇒ EVAL PRECISION is THE ceiling, not search.**
- **▶️ NEXT:** (B) passer TRUE-passed-mask exemption (concrete-ready; the ADV=5 rank-proxy over-fired) → (C) **FEN-based eval-tuning system** (Texel inner loop = SF-distill corpus + control set; SPRT outer gate = no-regression Elo; "FEN-tuning for speed, SPRT for truth"; user's agentic-orchestrator vision, likely a Claude skill) → then SEE-pruning / futility / decay-knob unification / SPSA. Memory: [[collapse-eval-overoptimism]], [[lazy-hybrid-lmp-ordering-safety]], [[fixed-depth-bench-ceiling]].

| Item | What | Result | Verdict |
| --- | --- | --- | --- |
| Bug 1 | TT/eval deque dedup (correctness-neutral) | no effect on this position (no eviction at this size) | kept |
| Bug 2 | move-cache miss store | **node drop** (better ordering on collision cutoffs) | kept |
| 1 | score memoization in move sort | byte-identical; timing within noise | kept (clean) |
| 2 | best-move-first via `moveGenCache` (Option A) | zero effect — slot clobbered by subtree at completion | **reverted** |
| 3a | TT → direct-mapped array (vector, depth-preferred) | **−11% time, −8% nodes** vs baseline (with Bug2+Item1), eval −56→−54, move 29, bounded 512 MB | **kept** |
| 3b | TT hash move (`storeTTBestMove` + read-side promote) | **+40% nodes**, corrupted deep eval | **FAILED, reverted** |
| E1 | PEXT sliding-attack tables (`unordered_map`→`SlidingRow` flat array via `_pext_u64`) | **byte-identical** to 3a (eval −54, 2,890,864 nodes, move 29, all cache stats + PV/score tables); **~15 s → ~11.16 s (−26%)**, eval throughput ~190k/s → ~259k/s (+36%) | **kept** |
| E2a | `pawn_rank_bonuses` `unordered_map<uint8_t,int>` → zero-init `std::array<int,64>` + 2 consumers' params by `const&` (kills 1 alloc + 2 map copies/eval) | **byte-identical** to E1; **~11.16 s → ~9.97 s (−11%)**, eval throughput ~259k/s → ~290k/s (+12%) | **kept** |
| E2b | the two `std::vector<CaptureInfo>` in `approximate_capture_gains`(+`_gains1`) → fixed `CaptureStack` (32-cap stack) via `CaptureStack&` helper params; kills 2 vector allocs/eval | **byte-identical** (confirmed by a forced-depth-11 run: same 2,890,864 nodes + every cache stat identical + d10/d11 tables identical to E2a). Same workload, **9.97 → 9.22 s**, eval throughput **290k → 313k/s (+~8%)** | **kept** |

**Current good state = E2b:** byte-identical search to 3a. Forced-depth-11 measurement: eval −54, 2,890,864 positions, **9.22 s**, 313k evals/s, move 29. (In normal time-bounded mode the freed time lets it reach depth 12 — see note.)

**⚠️ Measurement method (validated on E2b):** the engine sits on a **depth-step boundary** — a small per-node speedup flips whether the next ply runs, so time-bounded `Time Taken`/`Positions Analyzed` totals swing by a whole ply and are NOT comparable across builds. The clean method (used for E2b): **force a fixed search depth** so both builds run the identical tree, then compare `Time Taken` / `Average Static Analysis Speed` on equal node counts. Correctness gate unchanged (per-depth PV + score tables + cache stats must match). NB: the full time-bounded run's `Average Static Analysis Speed` is *diluted* by the extra deep ply — don't use it as the speedup figure; use the fixed-depth run.

## CORRECTNESS — SearchData grouped-scores refactor (2026-06-04, byte-identical + crash-killing)
Self-play found a warm-cache corruption crash family (`bad_array_new_length` / `"pseudo-legal"` throw). Root:
`SearchData`'s parallel arrays desynced — `top_level` pushed unconditionally by `alpha_beta`, the second-level
arrays conditionally by `minimizer` (skipped on draw/time-up early-exits), worsened by the PVS pop. Fix =
group the three drifting fields into `RootScore{top_score, second_moves, second_scores}` so `SearchData` is
`{vector<Move> moves_list; vector<RootScore> scores}`; `alpha_beta` is the sole writer (one push per searched
move, PVS pop deleted); `minimizer` writes one `out_entry`. Desync is structurally impossible. **Diagnostics
first proved aspiration was a non-cause** (`[INV]` 12/12 with DELTA=500, still 10/12 with DELTA=0; signature
always `34 34 33 33`). **Validated:** WAC `MAX_DEPTH=10` nodes byte-identical to the digit (254,973,405) +
259/300 + same 41 fails; speed neutral (two runs 574s/667s straddle old 609 = jitter, not regression — the
changed ops are root-only, the deep `dummy_entry` is now *fewer* allocs); replay loop 0 `[INV]`/`[BADMOVE]`/
aborts (was 12/12). `search_engine.{h,cpp}` only. Full writeup: `CRASH_INVESTIGATION_PLAYBOOK.md`.

## SPEED micro-opt sequence (2026-06-04) — on the clean grouped-scores base
Measured one-at-a-time, main.py forced-depth best-of-3 + WAC node-identity gate (254,973,405).
| Item | What | Result | Verdict |
| --- | --- | --- | --- |
| S1 | `-fno-semantic-interposition` (setupAI.py; .so internal calls skip the PLT) | byte-identical (nodes+cache stats); **~2%** (best 14.06→13.70s) | **kept, commit `5acfb7f`** |
| S2 | strip `-fwrapv` (add `-fno-wrapv`) | byte-identical, flat-to-slightly-worse | **reverted** (no gain + latent UB risk) |
| S3 | drop dead `use_tt_entry1` + TTEntry `alpha`/`beta` fields | analysis only: TT is **DRAM-bound** (512 MB ≫ L3, zobrist-random) so entry size barely matters, AND 24 B straddles 64 B cache lines → likely *neutral-to-worse* | **skipped** |
| S4 | move-list scratch buffers (kill per-node malloc in `generateLegalMovesReordered`) | ~440k malloc/free over a 5M-node search ≈ **~0.3%** | **skipped** |
| S5 | null-move progressive reduction `NULLMOVE_PROGRESSIVE` (env, default-off) | **depth-dependent**: −18% nodes @ genuine d12, **+4.7% @ d10** (check-extension inflation makes `depth_limit>=12` fire on shallow forcing lines). Refined to gate on `depth_limit - g_check_extensions ≥ 12/14`. | **parked** — overnight A/B → `NULLMOVE_OVERNIGHT.md` |

## STRATEGIC — depth is EBF-bound, not nps-bound (2026-06-04, the pivot)
STANDARD self-play (5 games): **~d13 @ 21s/move**, ~10.2M nodes/move → **effective branching factor ≈ 3.4**
(`10M^(1/13)`). 2010–2018 traditional engines hit ~d18–20 at EBF ~2.2–2.5. Depth ∝ log(nodes)/log(EBF), so
**EBF dominates; raw nps is logarithmic** (10% PGO ≈ +0.07 ply; EBF 3.4→2.5 ≈ +3.5 plies at the *same* node
budget). ⟹ the lever for ~3000 at the current system level is **search efficiency (move ordering + pruning)
+ eval** (coupled: eval → PV stability → first-move cutoffs → lower EBF), NOT speed micro-opts. **PGO deferred
to the finisher** (profiles final hot path; cheap to redo). NNUE/SMP = deliberate "easy multipliers", LAST.

Standard-game depth histogram (323 searched moves): mode d12–13, tail to d16–17, rare d20–21; opening ~d12.4,
midgame ~d14.1. **DIAGNOSTIC DONE → PRUNING-BOUND.** Behavior-neutral `g_fh_total`/`g_fh_first` cutoff
counters at the 3 main-search cutoff sites + `[search]` stderr line; byte-identical (WAC d10 still
254,973,405). **Measured `first_move_cutoff = 91.9%`** (10,782,520/11,729,997) — squarely in strong-engine
range (90–95%) → **move ordering is already good; the EBF gap is PRUNING-bound, not ordering-bound.** ⟹ do
NOT chase move ordering; the depth lever is **more-aggressive pruning (LMR/null-move/futility) + smarter
verification** (the engine prunes conservatively for tactical safety — `VERIFY_MARGIN` exists for that; the
play is "prune harder + verify smarter" without losing deep tactics, validated on STS + self-play). The
queued null-move depth-adaptive overnight is step 1.

## 3b failure — root cause analysis
`storeTTBestMove` created "move-only" TT entries at **near-horizon nodes** (whose score write is gated off by `if (cur_depth < depth_limit - 1)` in `get_score_*`). Those entries — the *bulk of the tree* — never upgraded to real entries, so they **flooded/thrashed the direct-mapped TT** (evicting deep score entries) and **promoted noisy depth-1 moves** to the front → ordering worsened → +40% nodes → time-budget overrun → corrupted deep eval.

**Process lessons (to not repeat):**
1. Reason about **volume/frequency across the whole tree**, not just per-call correctness. (A 2-min "how often does this branch fire?" estimate would have flagged the flooding; the gating `if` was in code I'd read.)
2. **Default to the minimal mechanism** (attach-only, not a new entry type) — especially for a marginal/uncertain benefit (the engine already has cut-node hash moves via `moveGenCache`).
3. For an **ordering change the gate is "node count must not increase"** — a rise is an abort signal.
4. **Re-challenge the plan at implementation time** — a detailed plan is a hypothesis, not a license to stop thinking.

## E1 — what worked (PEXT sliding attacks)
Replaced `std::vector<std::unordered_map<uint64_t,uint64_t>>` for `BB_DIAG/FILE/RANK_ATTACKS` with `std::vector<SlidingRow>`, where `SlidingRow{uint64_t mask; std::vector<uint64_t> data; operator[](occ)=data[_pext_u64(occ,mask)]}`. The `operator[]` does the pext internally, so all ~75 two-index call sites (`[sq][mask&occ]` and `[sq][0]`) stayed **unchanged** — only the type, the builder (`attack_table`), and the externs in **both** `cpp_bitboard.h` *and* `move_gen.h` changed (move_gen.h independently re-declares them — easy to miss). Correctness proven by a `#ifdef PEXT_SELFCHECK` block asserting `BB_X_ATTACKS[sq][subset]==sliding_attacks(...)` for all sq×subset, then confirmed byte-identical `main.py` output. The hash + pointer-chase + cache-miss per attack query was the dominant eval cost; removing it bought the eval-throughput jump.

## Next
**Re-assess Item 4** (q-search O(1) quiet-check) — E1 made `is_check` cheaper, so re-judge whether it's still worth it. Then E3 (incremental eval state), then the strength track. **Use the fixed-depth measurement method** (see note above) from here on. Same per-item correctness gate.

Process notes:
- E2a's ~11% beat the "modest" guess — `unordered_map` ctor+copy+hash is genuinely costly at 2.9M evals/search. Per-eval heap traffic is worth hunting.
- E2b measurement lesson: once the engine straddles a depth-step boundary, time-bounded totals are useless for diffing (a faster build searches an extra ply → looks 2× slower). Force a fixed depth for clean equal-workload timing. Also: don't read the speedup off a full run's `Average Static Analysis Speed` — the extra deep ply dilutes it (showed 3.7% vs the true ~8% on equal work).

---

## STRENGTH TRACK (behavioral — tactical harness, NOT byte-identity)

| Item | What | Result | Verdict |
| --- | --- | --- | --- |
| 7 | q-cache bound flag (`QTTEntry` + window-gated probe/store) | WAC@d10 238→**240** (+7/−5 churn) | **kept** (correct soundness fix; churn = order-sensitivity tell) |
| eval-vs-search | `ChessAI.ev()` static wrapper + `eval_at_resolution.py` (frozen SF resolution set) | static eval AGREES with Stockfish wherever comparison is meaningful (WAC.204≈equal, .018 White+10); misevals are forcing/mating-sacrifice lines | **EVAL EXONERATED → search is the bug** |
| ablation | `Config::ENABLE_*` env toggles at all 9 live pruning sites + `ablation_sweep.py` (subprocess/combo, position×combo matrix) | see below | **LMR is the culprit** |
| LMR exemptions | `PROTECT_KILLERS`, `PROTECT_PV` env toggles on `do_lmr` | deep12: KM **+0** (dead — already ordering-boosted); PV **+7 but 4.3×** (full-window nodes ≈ half the tree); the PV move is already protected via the live PV ordering bonus | **both DROPPED as gameplay levers** |
| **VERIFY_MARGIN** (margin-gated verification re-search) + **VERIFY_RESEARCH_REDUCTION** (graduated re-search depth) | re-search a *reduced* move that fails low within `VERIFY_MARGIN` of alpha, at `depth_limit − VERIFY_RESEARCH_REDUCTION` | **WAC @d10: 240 → 257 (r1: VM=6000, RR=1) = +17 (+22/−5) at 1.76×**; r2 (RR=2) 249 (+9, +15/−6) at 1.40×; r0 (full re-search) recovers most on deep12 but at 2.9×. The recovery needs ~full-depth verification (RR>2 collapses it; wider margin is non-monotonic = order-sensitivity noise). | **SHIPPED as default** (r2: VM=6000, RR=2 — the real-play-better config; r1 won bigger at d10 but its 1.76× cost plies in time-limited play) |
| **REPETITION_THRESHOLD 3→2** (in-search 2-fold repetition draw) | score the *first* repetition on the search path as a draw, not the true threefold | fixes won-games-walking-into-perpetual-draws: engine valued h8=Q at +14.5 when ...Rb1+ is a forced-draw perpetual (eval flat d10–22 = a *detection* gap, not depth). WAC@d10 255 vs 249 control (+6, no regression) | **SHIPPED (default 2)** |
| **timeout best-move fix** (`get_engine_move`) | adopt an iteration's move/score only if `alpha_beta` returned without `time_up` | the actual cause of a *lost game*: d12 correctly found Rg8 but d13 timed out mid-`reorder_legal_moves` and the abort fell back to `moves_list[0]` (stale list head = h8=Q), overwriting the last completed best. Position-general (any iteration that changes its best move, then the next times out before re-sorting) | **SHIPPED** |
| **CHECK_EXTENSION** (per-path-capped check extension) | a move that gives check searches its child +1 ply (bump local `depth_limit`); file-scope counter + RAII guard cap it per root-to-leaf path; gated so default-off is byte-identical | cap-sweep plateaus ~3–4. Equal-clock real play: **+9 @d10, +20 @LIGHTNING, +7 @BLITZ — and FREE on the clock** (mates found early hit the iterative-deepening score cutoff → *less* total time, confirming the cutoff-pays-for-itself effect). All 13 d10 gains are depth-gated forcing wins (cross-checked: each solves at d16). Fixes the perpetual *reliably* (no d12 coin-flip). | **SHIPPED (default cap 3)** |
| SEE-gated extension (`SEE_EXTEND_MARGIN`) | extend a check only if its checker is SEE-sound (opponent can't win it by > margin) — skip spite checks | **net LOSS**: @d10 256/255 (−8/−9) at margin 0/1000 for −30% time (782→550s); @LIGHTNING 226/223 (−5/−8) with **no** time gain (equal clock). **Cut a sac-mate** (WAC.266 PASS→fail — SEE can't tell a mating sacrifice from spite). The saved time is on *already-solved* positions, not reclaimable on a per-position budget; the extension is already free-on-clock in real play. | **PARKED** (revisit under self-play — cross-position clock-banking is the only real upside, unmeasurable on WAC). Smart king-pressure version can't cheaply reuse `get_latent_threat_score` (needs the full eval's `attack_bitmasks[]`). Knob kept, default disabled. |
| **STS positional harness + baseline** (`diagnostics/sts_test.py`) | scores positional move-CHOICE via the EPD `c9` (UCI) + `c8` (parallel points) ops → `{uci:score}`; reuses `tactical_test.run_one`; total %/max + per-theme + CSV. Deterministic `sts300_d10` reference (300-pos stratified sampler `awk 'NR%5==1'`, 20/theme, fixed depth = reproducible) | **baseline = 1387/2970 (46.7%)** [36% perfect / 40% zero / 24% partial], a faithful proxy of the full-1500 LIGHTNING run (47.4%). Per-theme = the expected fingerprint: **weak abstract/long-horizon** — AKPC 28%, a/b/c-pawn-advance 30%, King-Activity 38%, Open-Files 40%, Pawn-Play-Center 42%, Undermine 43%; **strong concrete/material** — Bishop-vs-Knight 64%, Recapturing 60%, Simplification 58%, 7th-Rank 53%. | **LOCKED** as the per-change positional reference (the dual gate alongside WAC). Engine untouched (diagnostic only). Repro (literal-depth convention): `MAX_DEPTH=10 PRESET=LONG_FORMAT python diagnostics/sts_test.py sts300.epd sts300_d10`. |
| **Aspiration windows + honest root TT** (`ASPIRATION_DELTA/MIN_DEPTH/MAX_WIDENINGS/WIDEN_PCT`, `HONEST_ROOT_TT`) | **Diagnosis** of why the prior attempt corrupted: 4 root/preliminary TT stores (`search_engine.cpp` alpha_beta + reorder_legal_moves) hardcoded `TTFlag::EXACT`, which is sound *only* under the infinite root window (move 0's full-window score is genuinely exact). Narrowing the root window (aspiration) makes the PV move's score routinely a fail-low/high **bound**, stamped EXACT at full depth → `use_tt_entry` hands it back as the true value → wrong score/move on the principal line. The deep machinery already flags honestly; only these 4 sites assumed infinity. **Fix**: `root_tt_flag(score,alpha,beta)` computes the honest flag at the 4 sites (gated by `HONEST_ROOT_TT`; off = hardcoded EXACT = byte-identical) + an incremental-widening aspiration loop in `get_engine_move` centred on the previous iteration's eval, asymmetric widen (×`WIDEN_PCT`), full-window fallback after `MAX_WIDENINGS`; root-only. | **VALIDATED keeper.** STS@d10 **46.7→48.9% (+2.2pp)** (honest +1.3, aspiration +0.9). **Broken vs fixed A/B** (DELTA=500, HONEST 0 vs 1): STS **47.3 vs 48.9 = +45 pts** *proves the corruption* — while **WAC was blind** (260 vs 259). LIGHTNING equal-clock (matched `asp_light_base`): **+4 solves (236→240), +0.32 mean depth (median 9→10), deeper 99 / shallower 33, time-neutral**. Efficiency @d10: **−18% nodes / −22% time** (holds on STS positions too: −21.6%). −5 WAC@d10 churn is depth-pinned (not the play regime). | **SHIPPED & live by default (`ASPIRATION_DELTA=500`, `HONEST_ROOT_TT=true`), DELTA confirmed (2026-06-03).** Sweep @d10 {250,500,800,1200}: **D=250 too narrow** (STS 46.4% < baseline = frequent-fail backfire ⇒ wants wider-than-textbook); **500–1200 = noise plateau** (d10 STS ~tied 48.9/49.3/48.3). LIGHTNING confirm of 500/800/1200 surfaced a **tactical↔positional axis**: wider (1200) = +7 WAC, fewer window-fails, but ~−3pp STS; **D=500 = best STS + efficiency sweet spot → kept** (positional is the weighted-heavier, weaker axis; the LIGHTNING STS split is partly noise — d10 had them tied). **Build-gotcha banked:** the incremental Cython build didn't recompile on a header-only change, so the flipped inline Config defaults reverted at link time (env-explicit runs were fine; no-env reproduced the OFF baseline) → `--force` rebuild fixed it; verify via the `[toggles]` echo. **Methodology proof case:** first change gated on BOTH suites — the +45 STS / WAC-blind split proves the positional third rung is *necessary* for eval-soundness changes. The DELTA-is-a-tactical/positional-knob finding → parked "context-adaptive search control" idea in `DEPTH_AND_POSITIONAL_NOTES.md`. |

**UPDATE (2026-06) — strength stack shipped, methodology refined.** The VERIFY → threshold-2 → timeout-fix → check-extension arc is all default-on and validated **in real play** (LIGHTNING/BLITZ at equal clock), not just at the d10 isolation control. Methodology upgrade learned this cycle: a fixed-depth solve = per-node accuracy, the fast-preset WAC = tactical real-play, but **neither is *positional* strength** — a regular game is ~90% quiet, which WAC cannot measure. The check extension is validated *tactically only*; the quiet-game arbiter (**STS** positional suite + self-play) is still owed before full confidence. Open items: SEE-gating the extension (`SEE_EXTEND_MARGIN` — cut the ~73% spite-check latency that lands on already-solved positions), the STS baseline, and the depth/selectivity track (aspiration windows are commented out; continuation history; staged movegen; lazy/incremental eval) — all in `DEPTH_AND_POSITIONAL_NOTES.md`. Note **WAC.213 still fails even at d16** (beyond the extension's reach — eval-resolution or far deeper).

## CORRECTNESS / INFRA (2026-06-03)

| Item | What | Result | Verdict |
| --- | --- | --- | --- |
| **Memory-corruption crash fix** (`sortSearchDataByScore`) | self-play surfaced a **deterministic** segfault / `std::bad_array_new_length` on a warm-cache, check-heavy, exposed-king position at fixed depth ≥ effective-d10. ASan (minimal build: drop `-flto`, add `-fsanitize=address`, run with `LD_PRELOAD=$(g++ -print-file-name=libasan.so)`) pinned it: the function set the reorder count `n` from `top_level_preliminary_scores` alone, then reordered all 4 parallel `SearchData` arrays as length-n — but `descending_sort_wrapper`'s append branch grows three of them (not `moves_list`) so they desync → `vec[indices[i]]` reads past the shorter array. **Fix:** `n = std::min({all four sizes})`. | **Blast radius proven NARROW.** Post-fix WAC (`MAX_DEPTH=10`) is **byte-identical** to the pre-fix shipped `asp_d500` baseline: 259/300, same 41 fails, **`nodes` column identical to the digit on every position** (only `time` jitters); STS300 identical to `asp_sts_d500` move/eval. `descending_sort_wrapper` runs on every ID iteration 4→10 of all ~600 cold searches (thousands of calls) with ZERO behavioral change ⇒ the desync **never fires in cold single-position search**; it required the warm-cache/accumulated-game-state regime (only ever reproduced via warm replay, never cold-from-FEN). So **all banked cold benchmarks are uncorrupted**; the bug lived **only in real full-game play** (crash = visible tip; possible silent in-game move-corruption removed by the fix). **CORRECTION:** NOT a contributor to the fixed-depth order-sensitivity churn (the byte-identical cold nodes disprove it). | **SHIPPED** (pure upside, zero benchmark cost). |
| **`MAX_DEPTH` literal-depth fix** (`get_engine_move` ID loop) | off-by-one: the loop guard `depth_limit + 1 < MAX_ITERATIVE_DEPTH` made `MAX_DEPTH=N` search to depth `N−1`. Changed to `depth_limit < MAX_ITERATIVE_DEPTH` **and reordered the guard before the `MOVE_TIMES[depth_limit]` read** (the array is `std::array<double,64>` with default cap 64 = zero headroom, so the naive operator flip would read `MOVE_TIMES[64]` OOB at the boundary). | **Pure relabel, proven:** new `MAX_DEPTH=10` reproduces old `MAX_DEPTH=11` exactly (259/300, identical 41 fails, identical nodes). `MAX_DEPTH` now means literal depth. See the convention banner at the top of this file. | **SHIPPED.** |

**Ablation result (deep12 = 43 fundamental fails, fixed depth 10):** baseline **3/43**; all-pruning-off **32/43** (+29); **`no_LMR` alone 31/43 (+28)**; the other four single-offs each +0…+3; `only_LMR` (LMR the only mechanism on) **4/43**; every `only_X` with LMR OFF = 32/43. ⇒ **LMR single-handedly drops ~28 of the 29 recoverable fails**; futility/razoring/null-move/q-delta are nearly tactically neutral. Mechanism: the reduced LMR scout undervalues the quiet winning follow-up and the re-search gate at `search_engine.cpp:900` (`score>alpha && score<beta`) never fires on the fail-low → the move dies at its reduced value (no extensions exist to re-deepen forcing lines).

**Splitter:** all-off solves 32/43, not 43 → ~32 are LMR-pruning errors reachable at d10; **~8 stay failing even all-off → depth-bound** (incl. **WAC.213**, a deep mating sacrifice — needs a *check extension*, not less pruning). **WAC.018** (trivial R+P) also never flips — anomaly, likely root move-ordering/conversion, flag for `line_probe`.

**Cost:** can't just disable LMR — `no_LMR` on deep12 was **853 s vs 99.7 s baseline (~8.6×)**; all-off ~67×. The fix must be **sound LMR** (keep speed, stop dropping wins): (1) soften the reduction magnitude in `reduced_search_depth` (cheap, direct), (2) add a check/forcing **extension** for the depth-bound mate cluster. Measure both on solves (deep12 + full wac.epd) AND fixed-depth speed; find the knee.

**Toggle scaffolding (kept, default-ON, byte-identical):** `Config::ENABLE_LMR/FUTILITY/RAZORING/NULLMOVE/QDELTA` (search_engine.h) guard the 9 live sites; `initialize_engine` overrides them from env (`ENABLE_LMR=0 …`) once/process, echoing `[toggles] …` to stderr. Diagnostic tooling — leave in.

## EVAL-SPEED TRACK (2026-06-11) — profiler-localized + adversarial-rescan batch

Driven by the phase-resolved per-term eval profiler (`diagnostics/eval_profile.py`, compile-gated `PROFILE_EVAL=1` `__rdtsc__`) and a cross-model full-code rescan (memory `adversarial-rescan-2026-06-11`; SEE confirmed empirically by `diagnostics/see_selfcheck.cpp`). Deterministic baseline: **WAC d10 262 / 267,284,369 nodes / STS 1487**, `USE_OPENING_BOOK=0`. Full order of operations in `HANDOFF.md` (Phase A byte-identical → B correctness → C re-tests → D bigger levers). **This is the EASY/early batch — the bigger speedups (ATTACK_LAYER ~18%-late-end memo, CH 134MB→12KB reindex, qsearch/stand-pat redesign, lazy/16×-hoisted movegen, TT-entry shrink) are later phases.**

| Item | What | Result | Verdict |
| --- | --- | --- | --- |
| **Bishop colour-complex surrogate** (`ENABLE_CHEAP_BISHOP_COMPLEX`, commit `dd8486e`, gated default-off) | `get_bishop_colour_complex_score` was ~33% of the midgame eval (a per-bishop flood-fill for a bounded ±0.3p signal). Replace with a 4-popcount surrogate (own pawns on bishop colour = bad-bishop, vs mobility/enemy-half/king-zone scope; K_* knobs). Also `depth_map int→uint8_t` (byte-identical). | **midgame eval ~25% cheaper** (BISHOP_COLOUR 33%→2%), **STS 1487→1520 (+33)**, WAC 262→259, nodes −7.3%, **lightning +0.089 ply**. K_* sweep = suite-churn → defaults locked. | **SHIPPED gated** (depth_map permanent/byte-identical). Default-on flip owed on a self-play A/B. |
| **A1 — `has_any_legal_move` leaf early-exit** (commit `3354ee9`, PERMANENT) | `is_checkmate`/`is_stalemate` ran a full `generateLegalMoves` into local vectors at every fresh leaf just to test `size()==0` (profiler-invisible — sits outside the eval). Replace with a generate-then-first-`is_safe` early-exit; boolean identical to `…size()!=0`. | **byte-identical** (WAC d10 267,284,369 / 262 unchanged) **and ~7% faster wall-clock** (10m27s→9m43s @ fixed d10). | **SHIPPED, permanent.** |
| **A2a — eval-cache 0-sentinel fix** (PERMANENT, in batch) | `accessCacheNew` returned 0 on both miss and a valid-0 hit; callers used `!= 0` → every is_practically_drawn / dead-equal eval recomputed each visit. Change to `bool accessCacheNew(key, int& out)` via the existing `valid` flag; update the 2 callers. | byte-identical node count (same value served); speedup is in the drawish-endgame regime (not WAC-visible). Build/verify pending. | implemented; permanent. |
| **A2b — TT draw caching** (gated, planned) | `addToSearchEvalCache` refuses `score==0` → draws never TT-cached. Removal lets the TT cut on draws (node↓ in the endgame regime). | repetition draws are PATH-DEPENDENT → verify the repetition-return doesn't reach the TT store before enabling. | gated `ENABLE_TT_DRAW_CACHE` default-off; hazard-check first. |
| **A3 — qsearch quiet-check cost** (gated, planned) | `buildNoisyMoveList` does a full board-copy + `is_check` per quiet move at every q-ply. (1) include quiet checks only at `qDepth==0`; (2) replace simulate+`is_check` with the `ENABLE_CHECK_ORDER` direct-check mask. | behavioral (changes searched check-set → move choice + nodes); expected to show at lightning. | gated `ENABLE_QCHECK_DEPTH0`/`_MASK` default-off; full bench. |

**Correctness fixes in the same rescan (strength, not pure speed — Phase B/C):** SEE 2 bugs (one-sided attacker refresh + stale-`square_values` LVA pick; 2.16% of captures wrong — `see_selfcheck.cpp` is the fix gate); endgame colour asymmetries (signed `pawn_rank_bonuses` in capture-gains, knight 10/15, black-rook double-count — target the late-endgame scatter); dead `get_relevant_pin` (fix-and-measure, else delete the 8-signature plumbing = speed); improving sign-bug (invalidated the −0.18 shelving verdict → re-test). See `adversarial-rescan-2026-06-11` memory + `HANDOFF.md`.

## RESCAN CORRECTNESS ARC — RESOLVED (2026-06-14); new control baseline + EBF

The Phase B/C correctness items above are now all resolved (shipped / kept-off-with-data / deleted / skipped). **New d10 control baseline: WAC 260/300, 249,966,786 nodes, STS 51.7% (1550/3000)** (`MAX_DEPTH=10 USE_OPENING_BOOK=0 PRESET=LONG_FORMAT`; supersedes the 262/267,284,369 above). **EBF mean 4.64 @ d10 (~3.4 @ STANDARD d13), first-move-cutoff 92.1%** — vs CPW/SF-optimal ~2, so ~2× high = the #1 depth lever (see `BASELINE_PERF.md`).

| Item | What | Result | Verdict |
| --- | --- | --- | --- |
| **capgain** (#1a) | sign of `pawn_rank_bonuses` in `approximate_capture_gains`' black branch (baseline erased Black's credit for capturing advanced White pawns) | mirror-verified correct (−2.06/+2.06 vs unchanged White +2.06); self-play **neutral** (−9.3 ±33.7); causal probe: fires 17.3% but net-neutral ⇒ strength not eval-correctness-bound | **SHIPPED default-on** (`621e251`) |
| **TT-depth** (#3) | `reorder_legal_moves` pre-pass searched `depth_limit-1` but stored `depth_limit` (+1 TT over-trust) | **WAC 259→260, −2.1% nodes, STS +33** | **SHIPPED default-on** (`ENABLE_TT_DEPTH_FIX`, `c596f44`) |
| qprec phase-gate (#8) | restore intended `use_q_precautions` phase-gating (a stray unconditional `=true` overrode it) | tested **WORSE** (−3 solves, +9.1% nodes) — the accidental always-skip-shallow-qsearch is better | **kept OFF** (`ENABLE_QPREC_PHASE_GATE`, documented negative) |
| null-move 3-vs-4 (#8) | min null-moves at `cur_depth>=3`, max at `>=4` | **NOT a bug — parity artifact** (min at odd cur_depths, max at even; MAXI=3 byte-identical, MINI=4 worse) | knobs kept, defaults 3/4 |
| **dead-pin** (#2) | `get_relevant_pin` ANDed disjoint masks → inert; measure-then-delete | revive play-negative (full +9.5%n/−67 STS; mobility-only sub-test +4.9%n/−54 STS); accuracy-positive where it fires (~0.3p→SF) but **redundant with the live LMR pin-handling `relevant_pin_exists`** (verified correct, left untouched) | **DELETED, byte-identical** (`c113875`) |
| symup (#1b/#1c) | knight/rook endgame asymmetries symmetrized UP | accuracy-NEUTRAL (~0.05p) | knobs kept default-off |
| A2b TT draw-cache (#4) | cache `score==0` | repetition/50-move draws path-dependent → unsafe to cache by zobrist; only stalemate (rare) safe ⇒ ~no benefit | **SKIPPED** (poor ROI) |

**Meta:** the eval-CORRECTNESS lane is exhausted as a strength source (capgain/symup/pin all neutral-or-redundant; even a 17%-firing mirror-correct fix moved Elo ~0). **PIVOT → eval-SPEED** (compounds: cheaper eval → affordable speculatively-prune→verify → lower EBF → more depth) **+ ordering/NPS/FMC**. EBF ~3.4–4.6 vs ~2 optimal is the exponential depth lever; ordering is near-tapped (92% FMC) so the lever is **pruning/reduction aggressiveness**, which is gated on eval-speed.

## BYTE-ID NPS LANE — MINED OUT (~+12% cumulative, all committed 2026-07-13)

Lane 2 (make the SAME byte-identical eval/movegen faster → depth → fewer collapses; regression-immune since decisions unchanged). All shipped, each gated by `wac byteid_check` = **247/41,479,610** + median-of-3 `depth_nps_bench.py` (bench is ±5% noisy → medians):

| Item | What | Result | Verdict |
|---|---|---|---|
| **movegen no-copy probe** (`7192005`) | `is_checkmate/is_stalemate` copied a `vector<Move>` by value just to test `.size()!=0`; replaced with `moveGenCacheHasMoves` (no-copy) | **+6.7% NPS**, byte-id 247 | SHIPPED |
| **statScore-LMR** (`bd19bd5`) | committed the previously-uncommitted shipped continuous statScore-LMR (defined byte-id 247) | +23 Elo lightning SPRT / +33 node_ab | SHIPPED (the one real strength ship this session) |
| scaffolding (`1082835`) | committed the gated default-off eval+search infra (threats/mobility/npedge/singular/probcut/otv/pawnKey) | byte-identical | SHIPPED |
| **PAWNS tables + mailbox scatter** (`de80343`) | getPPIncrement span-loop → `passed_span` tables (existed unused); latent_support loop → `[is_white][sq]` table; `initializePieceValues` → branchless scatter | **~+3% NPS median**, byte-id 247 | SHIPPED |
| **movegen blockers/checkers hoist** (`3adcee3`) | `processMaskPairs` recomputed king/`slider_blockers`/`attackersMask` 2×/gen-miss node; hoisted to once/gen | **+2.2% NPS median**, byte-id 247 | SHIPPED |
| TT-prefetch in make_move | `ENABLE_TT_PREFETCH` + `prefetchSearchEvalCache` | +0.26% (noise) | BANKED default-off, uncommitted |
| pieceTypeLookUp incremental-mailbox | measured the rebuild via new `PROF_INIT_PIECE_VALUES` term | only 2.2% of eval → not worth incremental surgery | STOPPED at measure-gate (scatter shipped instead) |

**Killed pre-build (verify-first discipline):** latent_threat lazy-skip (presence term always fires + inner loops already no-op → not byte-id AND no savings); CAPTURE_GAINS skip (circular: `g_capg_tension` is the call's output); x-ray recompute (SCOUTED IRREDUCIBLE — already one composite magic lookup); SEE-cache (shelved, 8.7% hit); pawn-hash (impure: reads king via attackingLayer + non-pawn blockers + phase → not byte-id). Tools built: `diagnostics/depth_nps_bench.py`, `eval_profile_corpus.py` (+ `overnight_runner.sh build_profile`/`pyrun` subs).

## LANE 3 (TT-rebuild/search-structure) — STRUCK; IIR the one survivor, tested NO-GO

Two scouts + a code-grounded Fable re-consult DEFLATED Lane 3: verified TT-move is pre-captured (movegen cache already key-verifies + `promoteMoveToFront`s the cutoff move); eval-in-TT is minor (eval already cached in `evalCacheNew`); singular is already built+tested-NEUTRAL; negamax-fold's payoff evaporated with them. The one salvageable lever — **cache-miss-keyed IIR** (reduce depth 1 at movegen-cache-miss/first-visit nodes; `ENABLE_IIR`/`IIR_MIN_DEPTH` in min+max, byte-id 247 default-off): WAC sanity encouraging (−6.2% nodes, solves 248 vs 247) but **paired seed-1 gauntlet (LIGHTNING, vs SF18@400) = 49.8% IIR vs 52.7% baseline = −2.9%.** BANKED default-off. ⇒ **search-structure lane confirmed mined out.**

## MILESTONE + the real eval hole (2026-07-13/14)

**AlphaUX beat the 2705 chess.com Tal bot for the FIRST TIME EVER** (win by mate; build = current committed engine). SF11-grounded analysis of the game's slips (`selfplay/games/manual/first-tal-win-2026-07-13.pgn`; memory [[first-tal-bot-win]]) decomposed the ONE real eval hole cleanly (eval/pruning/depth): **25.Qxc5 = EVAL over-read, NOT pruning/depth** — our engine evals the clean Qxe4 endgame accurately (+5.10 ≈ SF +4.78) but over-reads the messy Qxc5 grab (+5.84 vs SF +3.52), so it backs the grab up as "better." **KEY: our eval is TRUSTWORTHY when simplified, OVER-OPTIMISTIC in messy positions** (where opponent counterplay/king-attack lives). Same as the systematic over-read in `overread_bench.csv` (2,172 sign-flip positions) — **already-tried-to-fix-and-FAILED via eval-magnitude damping** (static eval HELD permanently). **NEXT-CANDIDATE (un-tried mechanism): a GENTLE, conditional simplification/safety MOVE-bias when clearly ahead** (steers into the accurate-eval regime; a move-selection bias, NOT an eval-magnitude fix). DOUBLE-EDGED — Game 06-05 (draw) is the opposite failure (over-simplified into a draw), so it must be gated on "clearly ahead AND still winning after". Gauntlet-gated; treat as a fresh careful campaign.

**Meta:** the strength picture after this session — search-structure levers exhausted (IIR NO-GO), byte-id NPS mined (~12% banked, modest ~+0.15 ply), eval-magnitude-damp dead. The remaining honest levers: (a) the simplification MOVE-bias (new mechanism, double-edged), (b) the LMR/pruning-margin SPSA tuning tail (expensive), (c) the ceiling (~3100-3300 classical; 2700→3000 doesn't need NNUE). The one clean strength ship this session was statScore-LMR (+23, now committed).

## Imbalance term NO-GO + simplification MOVE-bias NO-GO (2026-07-14, overnight)
Pursued the "eval accurate-when-simplified / over-optimistic-when-messy" hole from two angles, both offline-first.
- **Diagnostics (single-core):** (1) STS themed baseline — the count-keyed SF-Imbalance term's target themes are ALREADY our strongest (Bishop-vs-Knight 65%, Recapturing 68% vs 51.6% avg); (2) collapse N-vs-B filter (`diagnostics/nvb_filter.py`) — only 4-11% of collapse slips are minor-trade decisions. ⇒ **count-keyed material-Imbalance term = NO-GO** (targets our strength, barely touches collapses; cancelled before building). Term-attribution (`diagnostics/slip_term_attrib.py`) had pinned the over-read to `pieces`/PST (77%), and the STS weak spot is **Offer-of-Simplification 42%** (a when-to-convert hole) → pivoted to the simplification MOVE-bias.
- **Built + gauntletted (d) the simplification move-bias** (the line-305 next-candidate): gated `ENABLE_SIMPL_BIAS`/`SIMPL_AHEAD_THRESH`/`SIMPL_MARGIN` (search_engine.h), root-only exact-tie/near-tie argmax repoint in `alpha_beta` toward a non-pawn-trading move when clearly ahead (SIMPL_AHEAD_THRESH<best<mate). Pure argmax repoint (never touches best_score/alpha/tree) ⇒ **byte-id 247/41,479,610 holds OFF and ON** (mate-guarded; nodes identical). **Gauntlet A/B (fixed-node vs SF18@400, near-deterministic venue, 2 seeds × margins 0/300/750): pooled OFF 49.3% vs bias 48.4% = −0.9%, and collapses consistently UP (+1 s0, +4 s1) — the OPPOSITE of intent. NO-GO.** Thesis disproven: overriding the engine's best move toward a simplifying trade loses because that best move was, on average, correct (load-bearing optimism again). Knob left in code default-OFF, NOT committed.
- **Verdict:** the eval/conversion lane resists a MOVE-selection fix too, not just eval-magnitude damps. The (c) SF-initiative aggregate term + the existing `realizability_factor` (REALIZ_MAT_K) damp remain untried at the gauntlet, but they are eval-magnitude changes with 2-3 uncalibrated params → must be OFFLINE-CALIBRATED + design-reviewed first (do NOT blind-gauntlet). Recommend tackling (c)/realizability offline-first together. Results detail: scratchpad `simplbias_gauntlet_results.md`.

## SEARCH LANE: prune-verification methodology + RFP NO-GO + ordering-limited (2026-07-14 pt.3)
Eval/conversion lane fully closed → pivoted to SEARCH ("lower EBF 3.68 → SF's ~1.5-2 without removing important
nodes"). Full map: `dev_notes/search-ordering-pruning-map-2026-07-14.md`. Fable SF11-search audit + 3 code-audits:
**we're NOT missing SF's machinery — it's built but keyed/gated/untuned** (contHist keyed from×to not piece×to;
TT-move/2-ply-contHist/capture-hist/check-order parked default-off; decay/margins handpicked). LMR re-search
is full-depth = SAFE (not the bug).
- **METHODOLOGY WIN — prune-verification harness.** User raised the core problem: a black-box gauntlet NO-GO
  conflates "genuinely low value" with "mis-integrated." Fix: judge each prune by MECHANISM — verify each fire
  against a labeled corpus of its own mistakes. Built `ENABLE_PRUNE_LOG` logger (byte-id-safe) +
  `diagnostics/prune_verify.py` (per-fire verification search, auto non-negamax POV calibration) +
  `prune_discriminate.py` (AUC) + `prune_collect.py`. See memory [[prune-verification-methodology]].
- **RFP conditioning = DEFINITIVE NO-GO (harness-verified).** The black-box pruning-A/B had fingered RFP
  (reverse-futility) as the over-pruning culprit (RFP-off +7 WAC solves). The harness REFUTED it: RFP prunes
  ~99.7% CORRECTLY on BOTH clean (WAC ~18k fires) and messy (`overread_bench` ~12.5k fires) corpora; the
  eval-reliability discriminator is INVERTED (eval_instab AUC 0.18 — wrong prunes are in STABLE not messy
  positions). The +7 was search-path CHURN, not blunders. **The harness caught a false lead before we built a
  gate against a non-problem** — the whole point of the methodology.
- **LMR push = ordering-limited.** `LMR_EXTRA` (reduce harder): WAC fixed-depth looked like headroom (−15% nodes,
  accuracy held) but the equal-budget gauntlet was flat-to-NEGATIVE (LMR_EXTRA=3 pooled −4.9% over 2 seeds,
  +collapses). **Pushing reduction causes blunders → our ordering can't safely support harder reduction.**
- **CONCLUSION / NEXT:** the path to <3 EBF is the ORDERING FOUNDATION (compress the cutoff TAIL so the best move
  isn't at ranks where pruning fires → unlock harder LMP/futility + steeper tail reduction). Highest-leverage
  first step = re-key continuation history from×to → **piece×to** (SF keying), then turn-on/tune the parked
  ordering signals + fail-high sibling malus/gravity + SPSA decay/margins. **Ordering changes can't hurt the MEAN
  (only reorder the search) = the first low-risk lane all session** (no load-bearing-optimism trap). Pre-search
  is a latent downstream lever (droppable if the foundation lands; needs rework, shelved). byte-id 247 preserved;
  nothing committed; banked-uncommitted knobs: SIMPL_BIAS(off), ENABLE_PRUNE_LOG(off) + the diagnostics scripts.

## ★ EVAL ARC MEASURED: +83 Elo (2026-07-25) — first tournament in weeks, and it is NOT flat

**`tournament 300 "<former defaults>" 4 evalarc_2026_07_25`** — 620 games, LIGHTNING, equal clock, paired UHO,
SF18-arbitrated, conc 4, OMP=1 both sides. p1 = current shipped defaults; p2 = the pre-upgrade set
(`KS_ZONE_ATTACK_PCT=100 KAUFMAN_SCALE=0 ENABLE_CAPG_PIN=0 ENABLE_KS_REPLACE_LT=0 KS_SAFE_CHECK_DEF=3`).

| | result |
|---|---|
| **base vs fast** | **+334 −188 =98 of 620 = 61.8% ⇒ Elo +83.4 ±32.1** |
| as White / as Black | +169−96=45 (62.9%) / +165−92=53 (62.4%) — symmetric, no colour artifact |

⇒ **the cumulative eval arc (KS v1 + DEF5 + Kaufman + CAPG_PIN + de-king) is worth ~+83 Elo**, CI entirely above
zero. The revert config was **verified non-inert FIRST** (WAC 243/41,326,736 vs shipped 248/44,038,704), so this
cannot be a silently-ignored-knob null ([[env-knob-name-verify]]). Commits: `8ca657d` (gated scaffolding, byte-id
247/39,971,153) + `e02219e` (ship de-king, byte-id 248/44,038,704).

### SF BENCH CEILING (first run of `diagnostics/sf_bench_ceiling.py`) — the missing frame
Full 1500-position STS, same c8/c9 scoring, **depth 10 for every engine** (factors out SF's 5× NPS):

| ours (pre) | ours (de-king) | SF11 | **SF15.1 classical** | SF15.1 NNUE | SF18 |
|---|---|---|---|---|---|
| 51.8% | **54.9%** | 65.5% | **73.3%** | 76.0% | 79.3% |

⇒ **NNUE is worth only 2.7pp over SF15-classical, while SF11→SF15 classical is 7.8pp.** ~18pp of CLASSICAL
headroom remains before NNUE is the relevant question — strong justification for continuing HCE work. It also
frames de-king: +3.1pp against a 10.6pp gap to SF11 ≈ 29% of the distance to SF11 closed by one change.

### EBF is INVARIANT to depth-local pruning — and the eval→EBF channel is DISCONFIRMED

| lever | nodes | EBF | WAC |
|---|---|---|---|
| base | 44.04M | **3.667** | 248 |
| `ENABLE_IMPROVING=1` | 37.77M (−14%) | 3.640 | 241 |
| `RFP_MARGIN` 1000 / 2200 | 38.95M / 45.14M | 3.640 / 3.695 | 237 / 246 |
| `LMR_EXTRA` 1 / 2 / 3 | 34.31M / 34.38M / 32.58M (−26%) | 3.626 / 3.617 / **3.604** | 239 / 238 / 231 |

**A 26% node cut moves EBF by 0.063.** EBF = nodes(d)/nodes(d−1), so a prune firing ~uniformly across depths
leaves the RATIO untouched — only savings that COMPOUND per-ply move it. Independently reconfirms the banked
`collapse-campaign` result ("maxing ALL pruning only gets EBF 3.8→3.6"). Also: `RFP_MARGIN=1500` is already
optimal (1000 and 2200 both worse) ⇒ **"free on/off" does NOT imply slack at the margin.**

Same binary, same search, only `KS_ZONE_ATTACK_PCT` flipped:

| | old eval (100) | new eval (50) |
|---|---|---|
| EBF | 3.656 | 3.667 |
| first-move cutoff | 86.40% | 86.33% |
| nodes | 39.97M | **44.04M (+10%)** |
| WAC at `LMR_EXTRA=1` | 242 (−5) | 239 (−9) |

The +83-Elo eval arc produced **no ordering gain, no EBF gain, and no gain in reduction safety.** Corollary worth
keeping: part of our old "efficiency" was FAKE — an over-optimistic eval buys beta cutoffs that shouldn't happen,
so a *more* accurate eval legitimately searches MORE nodes. **EBF is not purely a virtue metric.**

**★ ROOT-CAUSE HYPOTHESIS (untested):** `reduced_search_depth` returns an ABSOLUTE target depth
(`base = DEPTH_REDUCTION[depth_limit]` − `log2(move)/scale` − `LMR_EXTRA`) and **never reads `cur_depth`**;
null-move does the same. Reduction is therefore front-loaded near the root and vanishes deeper, whereas
SF/Ethereal reduce off REMAINING depth at every node (SF11 `reduction()`: `Reductions[d]*Reductions[mn]`,
`Reductions[i] = 24.8*log(i)` — a function of remaining depth AND move number). This predicts all three symptoms:
EBF cannot compound, `LMR_EXTRA` blunders (it eats the shallow region where lines resolve), and the 2026-07-14
"ordering-limited" verdict may be a misdiagnosis of a reduction-SHAPE problem. Revised theory: **SF's low EBF is
eval-enabled because SF's machinery CONVERTS eval accuracy into pruning** (corrhist, improving, per-move futility
crediting the victim, depth-relative reduction, singular); our search barely reads the eval, so accuracy has no
path to become depth.

### Closed / corrected this session
- **Root razoring is NOT SF razoring**: a root-only stale-score `break` (`max(300·0.75^(d−4), 100)`), saturated at
  its floor by depth ~8. Five magic numbers are now knobs (`RAZOR_BASE_FIRST/FLOOR_FIRST/BASE/FLOOR/DECAY_PCT`).
- **`ROOT_RAZOR_CONTINUE` is PROVABLY INERT** — the root list is sorted score-descending, so the razor condition
  is monotone in the move index and `continue` skips exactly the set `break` does. Byte-identical, as predicted.
- **`RAZOR_FLOOR=800` WITHDRAWN.** It scored WAC 249 / STS 1685 (base 248/1647) but the plateau check killed it:
  600→1495, 1000→1481, 1400→1502 vs base 1647. An isolated spike in a DETERMINISTIC-but-chaotic metric (root
  razoring perturbs which root moves are searched at all). Committing it would have fit the suite, not the engine.
  **Always check the neighbourhood before shipping a swept knob.**
- **Safe-check table (`ENABLE_KS_CHECK_V2`) is DEAD, unconfounded**: alone, at its default table (14/14/7/9) with
  `KS_FLOOR=13` INTACT → **WAC 234 (−14) / STS 1481 (−166)**. The earlier −166 was never a floor-change artifact.
  It was a corpus-fit selection, and **the corpus win%-MSE fit is now 5-for-5 bench-negative** (it also rejected
  de-king 3×). Principle is sound (SF15 really does weight checks per-type); our port mis-scales SF's
  quadratic-formula weights onto our linear knee-12 budget ⇒ park as **"principle sound, port wrong."**
- **`ENABLE_CONT_HIST` correction**: the *tier path* is NOT dead code — it is the reachable else-branch of
  `ENABLE_STATSCORE_LMR` and our only alternative LMR-history shape. Only `ENABLE_CONT_HIST`/`_2PLY`/
  `CONT_HIST_LMR_THRESH` are dead under the shipped default. Delete-candidate WITHDRAWN.
- **`ENABLE_PRUNE_SHADOW` verified REAL** (genuine shadow re-searches + enter%/cut% wrong-prune tallies,
  deterministic 1-in-`SHADOW_N` sampler, nesting-suppressed) — but instrumented at **LMP and futility only, NOT
  LMR**, and it is a live meter, not the offline record-once/sweep-many dataset previously assumed.
- **`ENABLE_CAPG_REALIZ` is the IDENTITY function** (`REALIZ_MAT_K=0`, `REALIZ_PHASE_K=0`, `REALIZ_FLOOR=256`
  ⇒ 256/256). It has never actually run. Worse, its material-backing signal is the one `6ac8855` already
  disproved for KS ("does not separate phantom from real king attacks at scale") ⇒ it needs a DIFFERENT signal,
  not merely opened knobs.
- **Open eval item: passer V3 + `PASSER_RFLOOR_R5/R6`** (one feature + its fix). Categorical game WIN on the OLD
  baseline (3 seeds, +2.8%, positional −10% all seeds) but bench-negative on the NEW one (WAC 245 vs 248, STS
  1600 vs 1647) ⇒ likely OVERLAPS de-king (both cut the same over-read). Needs re-judging on the current baseline.

## ★ SHIPPED: per-move qsearch futility = +53.8 Elo (2026-07-26) — `029f619` on `2487145`

Replaces the node-level qsearch delta prune with SF's per-move form, which **credits the captured piece**:

| | test | failure mode |
|---|---|---|
| old | `static_eval < alpha − DELTA_MARGIN` → **return** | skips EVERY capture at the node, incl. a hanging queen |
| new | `static_eval + margin + victim <= alpha` → **continue** | a queen needs a ~10000-larger gap than a pawn ⇒ big captures protected by construction |

**Game gate: 456 games** (LIGHTNING equal clock, paired UHO, SF18-arbitrated, conc 4, OMP=1 both sides) —
base `+149 −219 =88` = **42.3%** ⇒ **candidate +53.8 ±37.5 Elo** (CI +16..+91), colour-symmetric (58.8% W /
56.6% B). **Benches: WAC 248→249, STS 1647→1662, nodes 44,038,704→38,840,709 (−12%)** — better on all three axes.
Margin **plateau**: 1500 and 2200 both give STS 1662; 400/800 prune too hard (243/247); 3000 degrades (1597).
**NEW BYTE-ID REFERENCE: WAC 249 / 38,840,709.** `ENABLE_QDELTA_PERMOVE=0` reproduces 248/44,038,704 exactly.

**★ It was UNREACHABLE for two sessions.** The guard tested `move.promotion == 0`, but `move_gen.h` pushes **1**
for every NON-promotion (`:277,336,389`, comment "Else the move is not a promotion") and 2..5 for real ones.
Instrumented: `moves=3,912,755 caps=3,848,782 promo=3,912,755` — promo == moves exactly. **Use `promotion <= 1`.**
Two wrong diagnoses were published (dead mechanism, then wrong margin) before a fire counter settled it in ONE run.
**Tell: identical node counts across a swept margin = UNREACHABLE code, not a mistuned threshold** (same
fingerprint as `ENABLE_CONT_HIST`). `g_qdelta_permove_fires/_seen` are now permanent.

**★ CALIBRATION — fixed-depth benches UNDER-measure soundness fixes.** WAC moved **+1 solve (inside noise)** for a
change worth **~+54 Elo**, because a suite of forcing tactical positions rarely punishes "skipped every capture at
this node". The pre-run forecast was "probably flat in games". Weight games heavily for qsearch/soundness work.

**Capgains at stand-pat remains REQUIRED.** With the sound prune in place the `QSTANDPAT_EVAL_MODE=2` penalty
**halves (−13 → −6 WAC)** — confirming capgains was partly masking the broken selection (fable's audit vindicated)
— but the residual −6 WAC / **STS 1581 (−66)** shows it also does genuine **valuation** work at quiescent leaves.
**Selection (which captures to search) and valuation (what the node is worth) are separate jobs.** The
"cheaper eval" upside (capgains ≈13% of eval cost) stays closed.

### Same session: the LMR lane measured cleanly and paid NOTHING (contrast)
Built real LMR wrong-reduction measurement (`ENABLE_PRUNE_SHADOW` extended to the LMR drop site, node-type × level)
and fixed `lmr_profile_event`, which applied one maximizer-oriented drop/cutoff test to both sides — min-node
cutoffs were counted as drops, making the by-level rate alternate ~95%/~8% as a pure artifact.
Baseline wrong-reduction **1.58%** (LMP 1.25%, futility 0.21%); **MAX nodes degrade with depth (L6 4.12%,
L8 6.25%), MIN nodes flat ~1%**. Two fixes, both NO-GO: `LMR_REM_FLOOR_PCT` (inert where it mattered — at
remaining depth 1 the percentage floors to zero) and `LMR_MIN_REM=4`, which removes the **entire L6+L8 population
(~35% of ALL wrong reductions)** for WAC +1 / STS −8. ⇒ **our benches are insensitive to wrong-prune rates at this
scale** (same shape as the July RFP null). **A correctly-diagnosed defect is not automatically an exploitable one.**

### Cross-engine schedule study (`dev_notes/sf-pruning-schedules-comparison.md`, SF11/15/16/17/18 + Ethereal)
Portability heuristic: **core expression invariant across versions ⇒ load-bearing ⇒ port** (the qsearch
victim-credit above); **monotone trend ⇒ tracks a CAPABILITY ⇒ check we have it** (RFP depth cap `<6`→`<8`→`<11`→
`<14`; **ours is 6 = SF11's**, and the growth plausibly tracks eval trust via NNUE+corrhist — inference, not
stated in the source); **oscillating detail ⇒ don't chase**. ⚠️ **SF's razoring ≠ ours** — theirs is node-level
"eval hopeless ⇒ drop to qsearch", ours is a root-only stale-score `break`; we may lack SF-style razoring entirely.
Two cheap unported items: **RFP should return a value pulled toward beta** (SF16+ `(2*beta+eval)/3`; we return raw
eval) and **`improving` belongs in the futility MARGIN** (SF11-era `217*(d − improving)`; ours only feeds LMR
reduction). **Port FORMS, refit CONSTANTS** — SF15+ runs ~356–385 internal per displayed pawn while `PieceValue`
in the same file uses 126–208, which is how the safe-check table died (WAC −14 / STS −166).

## ⚖️ MEASUREMENT CRISIS + two NO-GO games (2026-07-26) — nothing shipped, but the selector question is now open

**Overnight tournaments (both LIGHTNING equal clock, paired UHO, SF18-arbitrated, conc 4):**

| arm | games | score | Elo |
|---|---|---|---|
| `ENABLE_HISTORY_MALUS` | 478 | 51.6% | **+10.9 ±36.6 — n.s.** |
| `SEE_PRUNE_CAPTURES` (m=0) | 475 | 51.4% | **+9.5 ±36.7 — n.s.** |

Neither ships. Both were selected on **equal-time STS gains of +107 and +93** — the largest we have ever recorded
— and both converted to ≈ nothing. Contrast the qsearch prune shipped the day before: **+1 WAC / +15 STS ⇒ +53.8
Elo.** Across three measured cases the bench↔games relation is near-INVERTED.

**The ruler fix was real but insufficient.** Judging node-cutting changes at fixed depth IS wrong (it hands the
savings back): malus read **STS −30 at d10 vs +107 at equal time**, capture-prune **−47 → +93**. Correcting that
removed an ARTEFACT — it did not make STS predictive of Elo. ⇒ **do not ship on STS at any ruler.**

**New instrument evaluated: `cploss_frozen`** (win%-loss of OUR move vs SF18 over a frozen stratified corpus;
lower better; corpus path positional, SF judge cache config-independent, `--shard train|holdout` built in).
⚠️ **Its own default is `PRESET=LONG_FORMAT MAX_DEPTH=<d>` = fixed depth, which INVERTED the sign on the known
+53.8 Elo qdelta ship** (32.18 vs 31.69). Re-run with `PRESET=LIGHTNING MAX_DEPTH=64` it gets that case right
(30.49 vs 33.06). Against three known game outcomes it is **1-of-3**:

| config | timed cploss vs default 30.49 | games |
|---|---|---|
| qdelta OFF | 33.06 (+2.57 worse ⇒ ≈ −54 Elo) | −53.8 Elo ✅ |
| malus | 32.09 (+1.60 ⇒ ≈ −34 Elo) | +10.9 n.s. ❌ |
| capture-prune | 34.15 (+3.66 ⇒ ≈ −77 Elo) | +9.5 n.s. ❌ |

Not a resolution failure: −77 Elo ⇒ ~39% score, observed 51.4% over 475 games. **Hypothesis (untested): the
corpus is FAILURE-BIASED** — built from collapse positions, i.e. sharp spots where extra pruning is most
dangerous — so it over-penalises aggressive pruners, while flattering qdelta (a soundness fix, which helps most
exactly there). ⇒ good detector of "did this fix our failures", poor detector of "is this better overall".

**NEXT (critical path): mine a WIDE cploss corpus** from the ~2,000 recent games (`evalarc_2026_07_25` 620,
`qdelta_pm1500` 456, `malus_night` 478, `spc_night` 475) plus the 33,683-game archive. In self-play **both sides
are our engine**, so blunders by either colour are ours. Keep `stratum` separating broad-sample from
mined-blunder; sample the DECISION POINT (before the blunder); mix in SF-opponent/deeper positions since
LIGHTNING blunders are partly clock artifacts. Then re-score all four known configs — if the broad stratum calls
malus/capture-prune flat while the collapse stratum keeps penalising them, the bias hypothesis is confirmed.

**Other results this session:** RFP return-value blend toward beta (SF16+ form) = **NO-GO at every value**
(worse on both benches, +7…+27% nodes) — it and `RFP_MAX_DEPTH=14` are both **gated behind corrhist**.
`ENABLE_SEE_PRUNE` (quiets) NO-GO (STS −141, no node saving). **`ENABLE_HIST_PRUNE` is INERT** — it fires on
`history < −COEF·rd` but history never goes negative without malus, which is off; that is the **third**
unreachable-by-construction feature found in two days (after `ENABLE_CONT_HIST` and `ENABLE_QDELTA_PERMOVE`).
**`ENABLE_CORR_HIST` already EXISTS** (`search_engine.cpp:3660`) but is applied at the **RFP site only** — SF
applies the correction where `staticEval` is assigned, so everything downstream inherits it. Corrhist is
therefore a RE-SITING job, not a build-from-scratch, and it now gates a cluster of other items.

### ★★ The selector hunt (2026-07-26 pt.2): NO metric we have predicts Elo

Built a **wide tiered corpus** (`diagnostics/build_cploss_wide.py`): **77,432 games** (126 annotated
`tournament.pgn` self-play + 36,708 vs-SF `game_*.jsonl`) → 5.1M decision points → 4.29M unique → 9,000 sampled,
3,000 each of **general / blunder / stable**. Tiers come from the ENGINE'S OWN eval trace (the PGNs are annotated
`{ base +0.49/d10 }`), deliberately NOT from cploss — classifying by the metric that later scores it would inflate
that tier by selection and hand every new config a regression-to-mean gain.

Then scored the three configs whose game results we know, on BOTH corpora:

| config | collapse (n=813) | wide (n=1500) | **GAMES** |
|---|---|---|---|
| default (qdelta ON) | 30.49 | 23.18 | — |
| qdelta OFF | 33.06 worse ✅ | **22.01 better ❌** | **−53.8 Elo** |
| malus | 32.09 worse ❌ | 21.84 better ✅ | +10.9 ±36.6 n.s. |
| capture-prune | 34.15 worse ❌ | 22.44 better ✅ | +9.5 ±36.7 n.s. |

**Each corpus calls a DIFFERENT subset right; neither predicts Elo.** Gaps are 2–12× the ~0.3 noise floor (timed
scoring is nondeterministic — the same config measured 23.46 and 23.18), so these are real measurements: the
instrument is **precise but not accurate for Elo**. Composition explains the split — the collapse corpus is
tactical (rewards the soundness fix, punishes pruners), the wide corpus is quiet-dominated (rewards pruners
buying depth, punishes a soundness fix that does nothing there).

**★ LEADING HYPOTHESIS (untested): cploss and STS are MEAN metrics; Elo is TAIL-driven.** A game is decided by a
few sharp moments, not average move quality. The qsearch prune stops RARE catastrophes — it barely moves any mean
(WAC +1, cploss mean slightly worse) yet is worth +53.8 Elo; the pruners shave a little off many moves and buy
depth (good mean, ~0 Elo). This explains all four known cases AND the earlier STS failure (+107/+93 STS ⇒ ~0 Elo).
**Test: blunder RATE (loss > 5/10/20 win%) and p95 instead of the mean** — needs a per-position `--dump`, which
`cploss_frozen` does not yet emit. Cheapest decisive check = default vs qdelta-OFF only.

**Per-stratum (what the tiering bought):** both pruners improve `general` (24.07→22.07 malus) and `stable`
(22.20→20.40/20.11) while doing **nothing or worse in `blunder`** (23.28→23.12 / +0.31) — depth pays off in quiet
play, not where tactics dominate, which is exactly why the collapse corpus condemned them. Caveat: the blunder
tier marks where we ONCE erred, not where we err NOW (re-searched fresh we mostly handle them), so the three tiers
span only 1.9 points.

**Tooling bugs found, both silent:** stratum names were hardcoded (`["collapse","sts","neutral","game"]`) so a
corpus with other tier names printed NO breakdown — computed and dropped; and `--limit` takes a PREFIX, so
tier-grouped rows would have made a limited run read one stratum only (rows are now interleaved).
**Operational:** never edit a script while a job is executing it (bash reads incrementally; shifting offsets
produced a phantom syntax error mid-run), and don't pipe long runs through `grep`/`tail` (buffering hides all
progress until exit).

### ★★★★ RESOLVED (2026-07-26 pt.3): fixed NODES + the TAIL. The MEAN is blind.

**Ruler solved.** `PRESET=LONG_FORMAT NODE_LIMIT=250000 MAX_DEPTH=64` is **deterministic AND fair** — two runs of
one config gave byte-identical output (23.82 / n=1477 / identical tail buckets). A leaner search reaches DEEPER
inside a fixed node budget, so node-savers still convert their savings; and there is no clock variance.

| ruler | deterministic | fair to node-savers |
|---|---|---|
| fixed DEPTH | ✅ | ❌ savings handed back |
| fixed TIME | ❌ **σ≈1.0** (one config: 23.46 / 23.18 / 22.30) | ✅ |
| **fixed NODES** | ✅ exact | ✅ |

`node_ab` already used `NODE_LIMIT` for this reason; we never applied it to cploss. **⇒ every fixed-TIME cploss
comparison from earlier today is VOID** — σ≈1.0 swamped the 0.7–1.3 gaps, and the qdelta comparison flipped SIGN
between runs. The "wide corpus gets qdelta backwards" claim is retracted; it was noise, not a corpus property.

**★★ And the MEAN carries ZERO signal for a +53.8 Elo change.** Default vs qdelta-OFF, deterministic:

| | mean | rate>5% | rate>10% | **rate>20%** | **p99** |
|---|---|---|---|---|---|
| default | **23.82** | 14.6% | 5.1% | **0.5%** | **17.0** |
| qdelta OFF | **23.82** | 14.1% | 5.1% | **0.8%** | **18.4** |

qdelta-OFF is slightly BETTER on small errors and clearly WORSE on catastrophes; the two cancel exactly.
**⇒ Elo is TAIL-driven while every metric we owned is MEAN-like.** Explains the whole week: STS +107/+93 ⇒ ~0 Elo,
WAC +1 solve for a +54 Elo change, and cploss means flipping sign between corpora (average quality really does
differ by corpus; Elo tracked the tail throughout). **SELECT ON `rate>20%` AND p99, NOT the mean.**
⚠️ Thin: that bucket is ~7 vs 12 positions at n=1500. The `>5%`/`>10%` buckets carry mass and show NOTHING — the
signal lives only where we have least data. **Run the full 9,000** (~45 vs ~70) before trusting a verdict.
`cploss_frozen` now emits tail stats + an optional `--dump` of per-position losses so a finished run can be
re-analysed at other thresholds without re-searching.

**Idea backlog captured** in memory `ordering-and-reduction-idea-backlog`: static-table tiebreaker for history-0
quiets (also usable as REDUCTION CONFIDENCE where history is thin — which is where our wrong-reductions cluster),
ProbCut re-test (invariant SF11→18 + Ethereal; our 2.2-pawn margin vs SF's ~1.5 ⇒ likely too conservative),
RankCut alpha-reset and Caissa depth-decay (1 knob each, opposite directions), SF11's `ttHitAverage` as precedent
for a running reliability statistic feeding LMR. **MultiCut rejected** — absent from every reference engine.
⚠️ Correction to this log's earlier claim: "ordering changes can't hurt the mean" is **false** — LMP prunes and
LMR reduces by move INDEX, so a bad tiebreaker can get good moves pruned.

---

## 🏆 SHIPPED 2026-08-01 — the material-count fix bundle (+36.7 Elo, 503 games)

**New default fingerprint: `246 / 35,089,668 / EBF 3.846 / STS 1746`** (was `254 / 35,982,407 / 3.820 / 1629`).
Now default-on: `ENABLE_MATERIAL_COUNT_FIX`, `ENABLE_PASSER_V3`, `MOD_KS_REALIZ=128`.
Verified on the shipped build: WAC and nodes reproduce the env-knob measurement **to the node**.
⚠️ **WAC solves FELL 254 → 246 while STS rose 1629 → 1746 — that is the intended trade**, not a regression
(WAC is at ceiling 84.7%; STS carries the ~19pp positional gap). Do not "fix" it later.

### The bug
`cpp_bitboard.cpp` phase-blend path calls **both** `evaluate_X_midgame` and `_endgame` for the same square.
The returned scores are blended; the `whitePieceVal/blackPieceVal +=` **side effect is not** ⇒ pawns, rooks,
queens and kings counted **twice** whenever `phase_score > 40`. 277 of 600 banked positions carried a wrong
`material`; per-side accumulators inflated ~3.8 pawns (max 30). Found by triangulating a real-game collapse
(drew from **+7414**; search depth ROSE 16→22 through it, so not a clock problem).

### ★★ The lesson: a fix's value can lie entirely in what it UNBLOCKS
The fix alone is worth **+5 STS — indistinguishable from noise**, and was reported as "not measurably
helping". The owner overrode that (*"it reopens items that were formerly utilizing the double counting"*).
`MOD_KS_REALIZ=128` then measured **−196 STS without the fix and +88 with it — a 279-point swing**, because
the knob damps king-safety using the material edge, and a doubled edge damped the wrong side. It had been
banked "DISPROVEN" purely because it was judged against corrupted input.
⇒ **Testing a bug fix in isolation can understate it to zero.**

### ⚠️ …but the swing is a BENCH phenomenon; games are milder
| arm (vs defaults) | STS | games |
|---|---|---|
| A = fix + V3 + `MOD_KS_REALIZ=128` | 1746 | **+36.7, SPRT H1 accepted (503g)** |
| B = fix + V3 | 1658 | +25.4 [−7,+58] (439g) |
| B + `RFP_MAX_DEPTH=8` | 1689 | +19.5 ±35.8 (500g) |
| lever alone, **no fix** | **1467 (−196)** | **0.0 ±35.8 (500g)** |

The lever alone is **neutral in games, not harmful** — the dramatic bench swing does **not** reproduce.
A − B = +11, unresolvable at feasible sample sizes (needs ~4000 games). **Only the bundle has a verdict.**
🚨 **Forward consequence: large negative STS overstates real harm.** We have killed many candidates on
negative bench readings alone (search is 0-for-13 and 0-for-9); a −196 arm being game-neutral means some of
those deserve re-testing.

### Falsified this arc (measurements all held; the explanations did not)
- **"Eval unblocks pruning"** — `RFP_MAX_DEPTH=8` is +31 STS on B but **−167 on A**, and **+19.5 Elo in
  games, below B's +25.4** ⇒ deeper reverse-futility is worth nothing here. Pruning lane parked.
- **`RFP_MAX_DEPTH` saturates at ≥8 at d10** (8/10/12 all exactly 1689) ⇒ a d10 plateau check cannot tell
  "8 is optimal" from "the bench went blind". Real games reach d16-22.
- **"A third of our losses are thrown-away won positions"** — a **30× artifact** of per-ply snapshot mining
  (a side being mated is briefly "up a queen" mid-combination). With a persistence filter: **~1%**.
- **`MOD_KS_BACKING` ≡ `MOD_KS_REALIZ`** at default floors — verified in code (`mod_gain` clamps both to
  `[128,256]`, and `sig ≤ 0` makes backing's ceiling unreachable) and empirically byte-identical. **Never
  set both**; they double-damp (−95 STS).
- **Per-theme STS structure is noise** — the theme pattern at `MOD_KS_REALIZ=128` reverses at 256.
- **Equal-DEPTH engine ladders are biased by pruning** — SF1.1 spends **12.8× SF11's nodes** to reach
  "depth 10" and thereby out-scores it. We spend 61.6× SF18's ⇒ our measured gaps are **floors**.

### Reference ladder (true 1:1, `sts300.epd` @d10)
ours **58.2%** · SF11 64.3% · SF15.1-classical 72.1% · SF15.1-NNUE 76.0% · SF18 77.5%.
⇒ gap to SF18 is **19.3pp, not the 24pp long quoted**, and **13.9pp of it is CLASSICAL** (NNUE worth only
3.9pp; everything after SF15.1 worth 1.5pp). **The hand-reachable term is by far the largest.**
