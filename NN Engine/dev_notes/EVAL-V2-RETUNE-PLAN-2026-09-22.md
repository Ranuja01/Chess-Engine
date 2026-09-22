# EVAL v2 — THE JOINT RETUNE: PROTOCOL AND KNOB SELECTION  ★ 2026-09-22

@author: Ranuja Pinnaduwage (maintained with Claude)

★ **One page, one question: how do we retune v2 without repeating the retune that has already failed five
times?**

---

## 1. WHY THIS IS NOT A CORPUS FIT

☠️☠️ **`[[corpus-fit-is-anti-correlated-with-elo]]`: 5 of 5 raw corpus fits were bench-NEGATIVE, and the
best corpus gain ever achieved scored −85.6 Elo in games.** More data and cross-validation do not fix it,
because the OBJECTIVE is wrong: minimising error against a labelled corpus optimises average agreement,
and strength lives in the positions where the choice is close. `[[most-eval-error-is-move-neutral]]` is the
same finding from the other side.

⇒ **The night is NOT spent on a corpus fit.** It is spent on `selfplay/spsa.py`, whose objective is
**paired self-play game results**: each iteration perturbs the whole knob vector at once and plays
`theta+` vs `theta−` as ONE tournament, so the measurement is directly the strength DIFFERENCE. The
corpus-fit failure mode does not apply because there is no corpus.

## 2. WHY THIS RETUNE HAS A CHANCE WHERE v1's DID NOT

1. ★ **v2 is not degenerate and v1 was.**
   `dev_notes/collinearity-why-the-eval-cannot-be-tuned.md` explains v1's flattening: ~30 terms carrying
   ~2 independent signals, so every fit had a flat, ill-conditioned objective. **v2 enforces a
   collinearity gate and a one-owner rule, and that gate is COMPLETE and CLEAN — 40 columns across all
   five subsystems** (`[[a-differenced-detector-count-carries-the-census]]`, 09-17). A joint tune over
   v2's magnitudes is therefore a well-posed problem in a way v1's never was.
2. ~~★★ There is a MEASURED target: SF1.1 beats v2 by +137 STS at equal nodes... its advantage is its
   hand-tuned CONSTANTS ⇒ calibration is worth +110 to +175 STS points.~~
   ☠️ **CORRECTED 2026-09-22, and this premise does not survive intact.** (a) The equal-work gap to SF1.1
   is **302, not 137** — v2's equal-nodes STS is **1689**, not 1854; the 1854 was a DEPTH-10 number
   wrongly entered in the equal-nodes column (our d10 costs far more than 249k nodes on sts300, which is
   not a quiet corpus). (b) "Its advantage is its hand-tuned constants" was an **unsupported inference**:
   the 2x2 at equal nodes shows **v2's eval is worth +364 and SF1.1's named subsystems only +214**, so if
   anything OUR eval carries more — the gap sits in the gutted baselines (1777 vs 1325), which is
   search-plus-residual-eval and is NOT cleanly attributable. See `REFERENCE-BENCH-LADDER.md`.
   ⇒ The oracle's **+104 to +174** for SF11's eval in our search still stands and is the only surviving
   leg of the original target. **The direction of this run's rationale holds; its sizing was wrong.**
3. ★ **Fifteen consecutive move-null term ADDITIONS** (2026-09) say the remaining eval headroom is not in
   coverage. Together with (2), calibration is the only eval lever left.

## 3. THE KNOB VECTOR — `selfplay/spsa_eval_v2.json`

⚠️ **This section describes the FIRST vector (six knobs). It was widened to NINE on probe evidence — see
"VECTOR WIDENED 6 → 9" below, which supersedes the count here.** The six rows remain accurate as the
rationale for those six knobs.

SPSA degrades into a random walk if the vector is too wide, so this began as SIX knobs: the
per-subsystem MAGNITUDES, which are the numbers that have never been jointly asked about. Every v2
subsystem was shipped at a magnitude chosen in ISOLATION — usually off a §I frontier or a conservative
pick — and the question "is mobility too large relative to king safety, is the passer too small relative
to pawn structure" has literally never been posed.

| knob | init | range | why it is IN, and how its range was set |
|---|---|---|---|
| `MOB_V2_MAG` | 600 | 300–900 | v2's LARGEST term (≈ +162 Elo). The magnitude came from an SF11 table conversion; the mobility AREA was later tuned but no record shows the MAGNITUDE itself swept |
| `PASSER_V2_MAG` | 60 | **0–150** | ★★ **Never gamed against any alternative.** 60 was picked off a §I frontier the design doc itself calls FLAT, and picked *for a material taper that never shipped* — the original recommendation was 20–30. Widest relative range by right |
| `PS_V2_MAG` | 100 | **30–200** | Pawn structure. Rung 2 as a whole measured ≈ +60 Elo but the structure-vs-passer split inside it was never isolated — **and the SF1.1 ablation ranks pawn structure the 2nd most load-bearing subsystem (−78)**, so it gets room. Widened from 40–180 on that evidence |
| `KS_V2_MAX` | 4000 | 2500–5500 | The single largest number in the eval. Most-tuned subsystem (rung 1 ≈ +101 Elo) ⇒ a narrower *relative* range, but it must be free to move |
| `KS_V2_HALF` | 600 | 350–900 | The saturation half-point — it shapes the curve rather than scaling it, and was only ever set jointly with `MAX` |
| `OUTPOST_V2_PCT` | 100 | 40–180 | Largest component of the placement bundle (bundle ≈ +13 Elo; components individually below every floor) |

### ★★ WHAT THE SF1.1 ABLATION SAYS ABOUT THIS VECTOR (2026-09-22)
Zeroing each subsystem in a hand-tuned reference eval of OUR term set (baseline 2104, sts300 d10;
positive control Mobility×2 = +23 ⇒ the knobs are wired):
**King Safety −82 · Pawn Structure −78 · Mobility −57 · Passed Pawns −34.**
- ✅ All four load-bearing subsystems are represented in the vector (`KS_V2_MAX`/`KS_V2_HALF`, `PS_V2_MAG`,
  `MOB_V2_MAG`, `PASSER_V2_MAG`).
- ★ **Importance and tuning-quality pull in opposite directions, and the range should reflect both**:
  importance sets how far a knob CAN move the result, prior tuning sets how likely the current value is
  already right. KS is important AND the most-tuned ⇒ moderate range. Pawn structure is important and its
  magnitude was never isolated ⇒ widened. Passers are the LEAST load-bearing but never gamed ⇒ its wide
  range is cheap insurance, not a high-expectation bet.
- ⚠️ This ranks SF1.1's balance, not ours. It says where a hand-tuned HCE keeps its strength; it does not
  say our constants are wrong in the same proportions.

### ★★ VECTOR WIDENED 6 → 9 ON PROBE EVIDENCE (2026-09-22, the owner asked "surely more is being tuned?")
The first vector was six knobs, chosen by judgement. Rather than guess again, each candidate was measured
with `diagnostics/_eval_knob_delta.py` (~90 s each) and admitted only if it actually moves the eval:
| candidate | perturbation | fire rate | median | max | verdict |
|---|---|---|---|---|---|
| `KS_V2_ONSET` | 450→250 | 31.1% | **214 mp** | 849 | ★★ **ADMITTED — the largest median of ANY knob probed this session**, 3x the KS eg taper |
| `MOB_V2_EG_PCT` | 100→55 | 76.8% | 55 mp | 490 | ✅ ADMITTED |
| `BADB_V2_PCT` | 100→55 | 72.2% | 47 mp | 516 | ✅ ADMITTED |
| `TRAPROOK_V2_PCT` / `WEAKQ_V2_PCT` / `BEHIND_V2_PCT` | — | — | — | — | ☠️ **EXCLUDED BY INFERENCE**: `BADB` at 100% weight gives a 47 mp median, so knobs at 10-25% weight land near 5-12 mp — pure dilution of the shared gradient |

☠️ **`KS_V2_ONSET` was frozen on a bad basis and this corrects it.** The rung-1 ONSET frontier was called
"MEASURED flat, +0.1 ± 50 Elo" — but a ±50 Elo instrument cannot call anything flat, and the knob turns out
to carry more eval movement than any other in the config. **A knob frozen by an underpowered instrument is
not frozen on evidence.**
★ The general rule this establishes: **adding knobs to an SPSA vector is not free** — every extra knob acts
as noise for the others, so a knob must EARN its dimension by demonstrated eval movement. Probe first.

### ☠️ FROZEN, EACH FOR A RECORDED REASON — do not silently add these
- ~~**`KS_V2_ONSET=450`** — the rung-1 ONSET frontier was MEASURED at +0.1 ± 50 Elo, i.e. FLAT.~~
  ☠️ **SUPERSEDED TWICE, and the round trip is the lesson.** (1) Un-frozen on 2026-09-22: a ±50 Elo
  instrument cannot call anything flat, and the fire-rate probe showed it moves the eval by a 214 mp
  median — more than any other knob measured. (2) Then admitted to the vector and **found to carry NO
  gradient** (52% step reversals = pure random walk), despite the most visually convincing trajectory in
  the run. ⇒ **Its original "flat" verdict was reached by a bad instrument and happens to be RIGHT.**
  Stays at 450. ★ Large eval LEVERAGE is not evidence that moving a knob HELPS.
- **`RFP_MARGIN=1000`** — properly swept and shipped, and it is a SEARCH knob ⇒ **wrong lane**: a
  fixed-depth game cannot score a pruning win (it just searches fewer nodes to the same depth).
- **`PASSER_V2_MG_PCT=0`** — measured harmful (it roughly DOUBLES the worst case at equal mean); its
  revisit trigger is "if material ever tapers", and the taper is closed.
- **`TRAPROOK_V2_PCT` / `WEAKQ_V2_PCT` / `BEHIND_V2_PCT`** — small, and `WEAKQ` is recorded inert on §I.
- **`PS_V2_REAR_DOUBLED` / `PST_V2_KING_EG_ONLY` / `KS_V2_EG_PCT`** — the 09-21 bundle: unratified
  correctness fix / 3 mp / rejected. ⚠️ **P4 is deliberately left OFF in the base**: turning it on would
  presume a ship decision the owner has not made, and at a 3.4% fire rate it cannot have moved the optimum.
  If it ships later, this tune remains valid for the config it was measured on.

## 4. THE RUN

☠️ **The runner's `spsa` sub does NOT forward `--base` or `--depth`**, so invoking it that way would tune
against v2's *header defaults* — `PASSER_V2_MAG=0`, `MOB_V2_MAG=0`, `EVAL_ARM=0` — i.e. not v2 at all, and
the result would be silently meaningless. **Invoke through `pyrun` instead** and pass the full shipped v2
config as `--base`.

- Lane **eval** ⇒ `LONG_FORMAT` fixed depth (`--depth 6`, the tool's intended eval lane). Fixed-depth work
  is also the lane the record marks SAFE during the owner's gaming window; only TIMED work corrupts.
- ✅ Openings are **hardcoded** to `selfplay/openings_uho.txt` and the seed is `1000 + iter`, so
  `[[fixed-openings-inflate-game-reads]]`'s mandatory conditions (UHO + varied seed) are already met.
- `--concurrency 4` per `[[wsl-concurrency-crashes-the-box]]`.
- `--resume` writes `theta` + `iter` every step ⇒ the run is killable and resumable; cores can be handed
  back at any time without losing progress.

## 5. HOW THE RESULT MAY AND MAY NOT BE USED

☠️☠️ **SPSA OUTPUT IS A CANDIDATE, NEVER A SHIP.** The tool's own docstring: *"a candidate GENERATOR; the
winner is ratified by a separate lightning SPRT, never shipped on SPSA alone."*
1. **Read the trajectory, not just the endpoint.** `<tag>_log.csv` records every iteration's knob vector
   and score. A converging trajectory is a result; a wandering one is a random walk that happened to stop
   somewhere. ⚠️ Judge this BEFORE looking at whether the final numbers are pleasing.
2. **Ratify on a different regime than it trained on** — the transfer check the v1 cluster spec demanded
   ("lean on holdout + transfer-ratio"). SPSA trains at fixed d6; ratify with an SPRT at a realistic
   depth/time. A gain that does not transfer is an artefact of the training regime.
3. ⚠️ **`[[sprt-point-estimates-inflate-at-the-bound-they-stop-on]]`** — an SPRT DECIDES; pool games to
   COUNT magnitude. Do not quote the stopping estimate.
4. ⚠️ **Re-run the byte-identity, colour-symmetry and WAC fingerprint gates on whatever ships.** A retune
   moves every constant, and `[[a-correctness-fix-into-absorbed-tuning-is-not-free]]` cuts both ways.

## 6. WHAT WOULD FALSIFY THIS PLAN
- The trajectory wanders with no convergence ⇒ the vector is too wide or 60 games/iteration is too noisy
  for these knobs; retry with 3 knobs or more games per iteration.
- The endpoint wins at d6 and fails to transfer at depth/time ⇒ the d6 lane is not a valid proxy for eval
  magnitudes, which would be a genuinely new finding and should be recorded as one.
- The endpoint is ≈ the incumbent ⇒ v2's magnitudes were already near-optimal, the +137 SF1.1 gap is NOT
  calibration, and the eval-side story reduces to search. That is a real possible outcome and it is
  informative: it would move the whole remaining programme to search selectivity.

---

# ▶️ RUN 1 RESULT — `spsaeval3`, 147 iterations x 60 paired games at fixed d6 (~8,800 games)

## ☠️☠️ HOW TO JUDGE AN SPSA TRAJECTORY — AND THE STATISTIC THAT MATTERS
The endpoint is meaningless on its own: a driftless random walk ends *somewhere*, and that somewhere will
look like an optimum. Two statistics were computed per knob, and **only one of them is valid**:

- ☠️ **`above%` (fraction of iterations spent above the start) IS A TRAP.** By the **arcsine law**, a
  driftless random walk spends MOST of its time on ONE side of its origin — 0% or 100% is the single most
  LIKELY outcome, not a rare one. A knob that "left its start and never came back" is exactly what noise
  produces. This statistic fooled the first read completely.
- ✅ **REVERSAL RATE of consecutive steps is the right test, and it is exact here.** SPSA's step is
  `ak·scale·g·δⱼ`. With a true gradient the perturbation sign δ CANCELS (a positive gradient gives a
  positive step whether δ was + or −), so steps are consistently signed ⇒ LOW reversals. With no gradient,
  `g` is noise uncorrelated with δ, so the step sign is random ⇒ **50%**. With 145 steps, σ = 4.2%.

| knob | init → final | net | reversals | verdict |
|---|---|---|---|---|
| **MOB_V2_EG_PCT** | 100 → **125** | +25.3% | **30%** | ✅ **4.8σ** |
| **OUTPOST_V2_PCT** | 100 → **92** | −8.1% | **30%** | ✅ **4.8σ** |
| **PASSER_V2_MAG** | 60 → **100** | **+66.1%** | **31%** | ✅ **4.5σ** |
| **BADB_V2_PCT** | 100 → **96** | −3.5% | **36%** | ✅ **3.3σ** |
| KS_V2_MAX | 4000 → 4365 | +9.1% | 41% | 2.1σ — marginal, NOT taken |
| PS_V2_MAG | 100 → 99 | −1.2% | 42% | 1.9σ — marginal, NOT taken |
| KS_V2_HALF | 600 → 659 | +9.8% | 45% | ns |
| MOB_V2_MAG | 600 → 626 | +4.3% | 51% | ns |
| **KS_V2_ONSET** | 450 → **328** | **−27.1%** | **52%** | ☠️ **NONE — pure random walk** |

## ☠️☠️ THE KS_V2_ONSET LESSON — THE MOST INSTRUCTIVE RESULT OF THE RUN
`KS_V2_ONSET` was **rescued from the frozen list** hours earlier because its fire-rate probe showed the
largest median eval movement of any knob measured this session (214 mp). It then produced the most
visually convincing trajectory in the run: a −27% decline that **never once returned above its starting
value in 146 iterations**. It was called "the standout find" at the time.
**Its step-sign reversal rate is 52%. There is no gradient. It is noise.**
★ Both halves of that are worth keeping:
1. **A knob having large eval LEVERAGE says nothing about whether moving it HELPS.** The probe correctly
   said ONSET can move the eval a lot; it cannot say which direction is better. Fire rate is a
   NECESSARY-condition screen, never a promoter — the same rule that governs §I.
2. **Eyeballing a trajectory is not analysis.** The shape that convinced me is the shape the arcsine law
   predicts for pure noise.

## ✅ THE CANDIDATE — FOUR KNOBS, NOT NINE
☠️ **The raw endpoint was NOT taken.** Five of its nine knobs sit at noise-driven offsets from values that
were ALREADY tuned (MOB_V2_MAG games-validated, KS_V2_MAX/HALF from rung 1's +101 Elo). Shipping those
displacements would import random error into a tuned config. Only knobs clearing 3σ were taken:

    PASSER_V2_MAG=100  MOB_V2_EG_PCT=125  OUTPOST_V2_PCT=92  BADB_V2_PCT=96

★★ **The two large movers are exactly the knobs the record flagged as NEVER PROPERLY TUNED**:
`PASSER_V2_MAG` ("never gamed against any alternative"; 60 was picked off a frontier the design doc itself
calls FLAT, and picked for a material taper that never shipped) and `MOB_V2_EG_PCT` (never swept). The
knobs that DID have real tuning history — mobility magnitude, KS max/half — produced no signal.
⇒ **The tuner moved what was untuned and left alone what was tuned.** That coherence is the strongest
evidence the run carried signal rather than noise, and it is independent of any single knob's statistic.

## VERIFICATION NOTES
- ✅ Harness bias checked: mean `y` = 0.4918 over 147 iterations (8,820 games, 1.5σ from 0.5, ns), with
  first-50 = 0.4960 and last-50 = 0.4961 — no drift. ⚠️ Mean `y` can only detect harness bias; it says
  NOTHING about convergence, since SPSA compares symmetric perturbations and `y ≈ 0.5` either way.
- ⚠️ The schedule needed `--a 2.0 --c 0.45`; the tool's defaults (`0.15`/`0.30`) are calibrated for SEARCH
  knobs and left the eval vector frozen at ~1.5 units/step on a scale of 300. **Caught after 4 iterations
  by inspecting the log — not after 9 hours.**
- ☠️ The runner's `gate` sub hardcodes `--p2-config ""`, so the baseline would run with NO env knobs, i.e.
  `EVAL_ARM=0` = **v1**. Ratifying a v2 candidate through `gate` would silently compare it to the wrong
  engine. `sprt.py` must be called directly with BOTH configs.

## ▶️ RATIFICATION — INCONCLUSIVE, AND THE BENCH IS NEGATIVE

| instrument | result |
|---|---|
| **SPRT vs shipped v2** (LIGHTNING equal-time, UHO, seed 7) | **+121 −104 =71 of 296 (52.9%), elo ≈ +20.0 ±46.5, LLR +0.283 — INCONCLUSIVE on the time cap** |
| **STS @ d10** | **1761 vs 1854 shipped = −93** (inside the ±150 floor, but negative) |

⇒ **DO NOT SHIP.** A positive point estimate deep inside its own error bar, and a negative bench reading.
Neither instrument supports the candidate; together they support waiting.

★ The SPRT stopped on an EXTERNAL time cap rather than a bound, so the estimate is UNBIASED
(`sprt-point-estimates-inflate-at-the-bound-they-stop-on` applies only to bound-stopped runs). ⚠️ Note the
drift: **+57 elo at 93 games → +20 at 296.** Regression toward the mean; the interim number was an early
streak, which is why interim SPRT reads must never be quoted as results.
⚠️ Reaching a bound needs ~2,000 games; 145 minutes bought 296 at **122 games/hr (117 s/game)**. That rate
is slow for LIGHTNING and the likely cause is the per-thread SF adjudication confirm at 0.5 s per check —
**worth measuring before the next ratification, since it roughly halves every SPRT we run.**

☠️ **The plan's target was also oversized.** §2 justified this run partly on "SF1.1 beats v2 by +137 at
equal nodes". The corrected equal-work gap is **302** (see `REFERENCE-BENCH-LADDER.md` — v2's equal-nodes
STS is 1689, not 1854). The argument's direction survives; its magnitude was wrong.

## WHAT THE RUN ACTUALLY ESTABLISHED
1. ✅ **A method that works.** Reversal-rate on the step signs separates gradient from random walk exactly,
   and it caught a knob (`KS_V2_ONSET`) whose trajectory was visually convincing and statistically empty.
2. ✅ **Two knobs with real gradients**, both flagged in the record as never-properly-tuned
   (`PASSER_V2_MAG` +66%, `MOB_V2_EG_PCT` +25%) — and no gradient on any knob that HAD been tuned. That
   coherence is the run's strongest evidence, and it is independent of the ratification failing.
3. ☠️ **But a real gradient at d6 did not convert into a measurable game gain**, and cost 93 STS points.
   The transfer from fixed-d6 to equal-time is the suspect: it is exactly the regime change the SPRT was
   chosen to test, and it did not survive it.
▶️ **NEXT, in order:** (a) a second SPRT segment at a different seed, POOLED, to get the magnitude — the
candidate is not refuted either; (b) fix the adjudication cost first so the segment buys more games;
(c) re-run SPSA at d8 if the d6 transfer is the fault, before concluding the knobs are wrong.
