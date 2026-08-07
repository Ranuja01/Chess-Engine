# Session handoff — 2026-08-08: the bench was the blocker, not the bug

**Read this top block first.** Prior: `SESSION-HANDOFF-2026-08-07.md` (the corpus-fit null + the first
two symmetry fixes). Canonical pawn reference: [`PAWN_MODEL.md`](PAWN_MODEL.md).

---

## 🚨 STATE

    DEFAULT (no knobs):  243 / 35,138,590 / EBF 3.839 / STS 1755
    WAS RECORDED AS:     250 / 35,138,590 / EBF 3.839 / STS 1755

⚠️ **The WAC solve count in the register was WRONG — it is 243, not 250.** Verified twice on the same
binary, byte-identical node counts both times. The 250 was the *previous* row's value carried across
when the row was updated; the two WAC fields were never re-read together. ⇒ the 2026-08-07
colour-symmetry fix was **+70 STS AND −7 WAC**, not +70 STS alone.
★ Lesson: **re-read every field of a fingerprint when you edit any field of it.** A stale digit turns
"reproduced exactly" into "regressed by 7" silently.

HEAD `e23f88a` on `NN-ENgine`. **Nothing running. Nothing committed** (commits prompt while the owner is
away, so the whole session is uncommitted by design — see §COMMIT).

## ★★★ THE HEADLINE — the instrument was wrong, and it inverted the answer

Continuing the symmetry cleanup found a third site immediately:
🐛 **`evaluate_knights_endgame` pays 10 per free mobility square for White and 15 for Black.** The
midgame twin is symmetric (20/20, 5/5); only the endgame diverges. The `4k3/8/8/8/8/3N4/P7/4K3 w`
repro was off by exactly 35 mp = the d3 knight's **7** free squares × the **5** mp gap.

Both repair directions were *already built and gated* in 2026-06 and forgotten:
`ENABLE_KNIGHT_MOB_SYM_UP` (raise White to 15) and `ENABLE_KNIGHT_MOB_FIX` (lower Black to 10).
They are **byte-identical on the symmetry test** (both 57.4% → 47.1% at N=800), so symmetry cannot pick
the value. STS had to choose — and STS could not:

| arm | STS orig | STS **mirror** | **balanced (sum)** | colour gap |
|---|---|---|---|---|
| baseline (10/15) | **1755** | 1651 | **3406** | +104 |
| `SYM_UP` (15/15) | 1712 | 1552 | **3264 (−142)** | +160 |
| `MOB_FIX` (10/10) | 1677 | **1728** | **3405 (−1)** | −51 |

☠️☠️ **`sts300.epd` is 177 white-to-move vs 123 black-to-move.** It cannot arbitrate a colour-symmetry
fix — it asks White's questions more often, so the score mixes "is this good" with "which colour does
this favour". Read orig-only, `MOB_FIX` is the *worst* arm (−78) and `SYM_UP` the better one (−43).
Colour-balanced, that **reverses**: `MOB_FIX` is exactly neutral and `SYM_UP` is the loser.
★★★ **A demonstrated local defect is not a demonstrated improvement — but neither is a bench verdict
from a bench that is unbalanced in the very axis under test.** This is the same shape as
`rank-by-winpct-not-cp`: the metric, not the data, flipped the conclusion.

## ✅ THE CANDIDATE — `ENABLE_KNIGHT_MOB_FIX=1`, free on every axis

| | baseline | `MOB_FIX` |
|---|---|---|
| colour-swap violations (N=800) | 57.4% | **47.1%** |
| STS balanced (orig+mirror) | 3406 | **3405** |
| bench colour gap | +104 | **−51** |
| WAC solves | 243 | **248** |
| nodes | 35,138,590 | 35,158,199 (+0.06%) |

Neutral on balanced positional play, **+5 WAC**, −10.3pp asymmetry. It also recovers most of the 7 WAC
solves the previous symmetry fix cost.
▶️ **NOT shipped — flipping a default is the owner's call.** To take it: `ENABLE_KNIGHT_MOB_FIX = true`
in `search_engine.h`, rebuild, re-record the fingerprint (it WILL move, on purpose).
⚠️ **Historical trap, do not be scared off by it:** this knob was in the 2026-06 three-fix bundle that
measured **−32.2 Elo**. That bundle was never isolated, and its villain was pinned on
`ENABLE_CAPGAIN_PAWN_FIX` — which was later shipped *alone* and measured ~neutral (−9.3 ±33.7). So the
−32 is **unattributed**, not evidence against this knob.

## 🐛 FOURTH SITE — localised, NOT fixed: the capture-gains rank bonus fires for one colour only

`material` in the breakdown is **not a piece count** — it is `blackPieceVal − whitePieceVal`, and those
globals are mutated inside `approximate_capture_gains`. That is why `material`, `capture_gains`,
`piece_value_boost`, `ae_input` and `advanced_endgame_total` all break on the **same 16 of 400**
positions with a ~680 mp mean.

🎯 **6-piece repro: `8/6k1/1Rp5/8/8/4p3/5P2/4K3 w`.** The only rank-bonus-eligible capture is Rb6xc6.

| `CAPG_PAWN_RANK_CLAMP` | asym | base `capture_gains` | mirror `capture_gains` |
|---|---|---|---|
| 200 | 200 | −1200 | **+1000** |
| 275 (default) | 275 | −1275 | **+1000** |
| 400 | 322 | −1322 | **+1000** |

★ The mirror side is pinned at **exactly ±1000 — the bare pawn value — at every clamp setting**, while
the base side gets the full clamped bonus. The bonus is applied in one orientation and contributes
nothing in the other. True uncapped magnitude **322 mp**; the default clamp of 275 is binding.
⚠️ Turn-gated (flip side-to-move → clean), so the capture *sequence* is part of the mechanism.
✅ Ablation: `CAPG_PAWN_RANK_CLAMP=0` drops accumulator violations **16 → 8**, so this is **half** of it.
A second independent root remains: `8/6k1/1Rp5/3p4/8/4p3/5P2/4K3 w`, asym exactly 1000, also turn-gated.

☠️ **Do NOT accept a code-reading story for this.** The two table sites (`evaluate_pawns_endgame` white
3204 / black 3316) read as correctly antisymmetric, and the neighbouring evasion-polarity "bug" in the
same function looks provably wrong yet measured WORSE when fixed. ▶️ Next step is a debug print of
`prb` in both orientations (a `CAPG_DEBUG_DUMP` hook already exists in the function), not more reading.

## 🎯 TWO MORE MINIMAL REPROS — four pieces each, handed over unfixed

`pieces` is the broadest remaining source (151 of 178 violating positions, mean 76 mp). Its two named
sub-views shrink to four-piece repros, and **neither is turn-gated** — so unlike the capture-gains
defect these are pure static-evaluator asymmetries:

| repro | material | term | asym | turn-gated |
|---|---|---|---|---|
| `8/1R6/5k2/1p6/8/6K1/8/8 b` | W Rb7 Kg3 · B Kf6 pb5 | `pt_rooks` | 105 mp | no |
| `8/8/p4k2/1p6/8/8/8/5K2 w` | W Kf1 · B Kf6 pa6 pb5 | `pt_pawns` | 85 mp | no |

★ The second closes last session's loose end ("2 white pawns asymmetric 2/21"): **single pawns are
symmetric, two pawns are not**, so it is a pawn↔pawn interaction, not a per-square table error. Both
are endgame (`ae_fired=True`), so `advanced_endgame_total` inherits the error in each case.
▶️ Get these with `pyrun diagnostics/_asym_minimize.py ASYM_TERM=pt_rooks SCAN_N=300` — the tool now
scans the corpus for its own worst case per term, since the terms do not break on the same positions.

## 🐛✅☠️ FIFTH SITE — FOUND AND FIXED, AND IT COSTS BENCH: the pawn-support wrap guards

The `pt_pawns` repro fell to code reading *because the repro anchored it*. In
`evaluate_pawns_endgame`'s BLACK branch the two file-wrap guards are **swapped**:

```
ne = (sq << 9) & ~BB_FILE_H   // up-RIGHT wraps onto file A -> needs ~BB_FILE_A
nw = (sq << 7) & ~BB_FILE_A   // up-LEFT  wraps onto file H -> needs ~BB_FILE_H
```

Each guard both fails to filter its real wrap AND deletes a legitimate supporter on the guarded file.
★ **Not a judgement call** — the MIDGAME twin (~1077) and the sibling near the pawn-shield code (~9047)
both use the correct pairing. This one site is the outlier, so the codebase defines its own intent.
★ The arithmetic closes exactly: b5 supported by a6 loses `EG_SUPPORT` (135) and then wrongly collects
`EG_LATENT` (50, gated on the support being ABSENT) = **−85 mp**, the measured value.

Gated as **`ENABLE_PAWN_SUPPORT_WRAP_FIX`** (default off; byte-identity with the gate off verified
EXACT at 243 / 35,138,590 / EBF 3.839).

| | baseline | `MOB_FIX` | **both fixes** |
|---|---|---|---|
| colour-swap violations | 57.4% | 47.1% | **27.5%** |
| **file-mirror violations** | 18.4% | 18.4% | **7.2%** |
| STS balanced (orig+mirror) | 3406 | 3405 | **3259 (−147)** |
| WAC | 243 | 248 | 244 |
| nodes | 35.14M | 35.16M | **37.11M (+5.6%)** |

✅ Two INDEPENDENT invariants improve together, and the file-mirror halving is mechanistically exactly
what a wrap-guard repair should do — that is strong corroboration the fix is real.
☠️ **And it still costs 147 balanced STS and 5.6% nodes.** Fourth instance of *a demonstrated local
defect is not a demonstrated improvement* (obstruction blend, capture-gains polarity, phase blend, now
this). ⚠️ −147 on a 6000-point suite is near the noise floor, but the discipline is explicit: do NOT
explain a negative away — that is exactly the error that produced the −86 Elo SPRT.
▶️ **The mechanism is known and actionable:** `EG_SUPPORT` (135) and `EG_LATENT` (50) were FITTED while
half of Black's diagonal supports were invisible. Restoring detection without refitting those two
magnitudes hands Black credit it never had during tuning. **The right treatment is fix + refit those
two constants**, not fix alone. That is a 2-knob sweep, not a descent.

### Fire rate — the wrap fix is NOT a rare edge case
`_eval_dump_simple.py` twice (one process per setting — knobs latch at init), 1500 corpus rows:

    positions 1500   changed 252 = 16.8%   median 75 mp   p90 135   max 803   mean 72.5

★ One position in six, ~7.5 cp median. So −147 balanced STS is **proportionate to the footprint**, not
disproportionate — which argues the bench loss is real signal rather than jaggedness. ⚠️ Note the bug
lives in the BLACK branch only, so the fix shifts Black's support credit systematically; a refit of
`EG_SUPPORT`/`EG_LATENT` moves BOTH colours, so it can rebalance the magnitude but cannot perfectly
undo a one-sided change. Expect the refit to recover most, not all, of the 147.

## 🚨 THE TACTICAL SUITE IS SKEWED TOO — worse than the positional one
**`wac.epd` is 190 white-to-move vs 110 black-to-move (63/37)**, a WORSE skew than `sts300`'s 59/41.
Every WAC number in this document (243 / 248 / 244) was read on that skewed suite, for colour fixes —
I applied the balancing lesson to the positional bench and not to the tactical one.
✅ `make_mirror_suite.py` now handles the **`bm`/SAN** form as well as `c9`/UCI (parse SAN on the
original board → mirror both squares → re-emit SAN on the mirrored board, asserting legality), and
`suites/wac_mirror.epd` is built and verified: census inverts `(190,110) → (110,190)`.
🧰 New runner sub **`wac_suite <suite.epd> <tag> [KEY=VAL...]`**.
▶️ Balanced tactical readings were in flight at handoff — re-read them before quoting any WAC delta.

## 🎯 RETARGETING — the SF11 filter is sized (full 23k, labels banked)

`label_sf11_static.py` labelled **23,084 / 23,113** rows with SF11-classical AND SF18-static.
`sf11_filter_sizing.py` (agreement = within 10 win% points of the SF18-SEARCH target):

| bucket | rows | share |
|---|---|---|
| SF11 agrees AND we agree | 12,831 | 55.5% |
| **SF11 agrees, we DON'T** ← the learnable set | 6,434 | 27.8% |
| we agree, SF11 does NOT (**GUARD**) | 1,112 | 4.8% |
| neither agrees (tactical) | 2,736 | 11.8% |

Mean win%² error vs SF18-search on the FULL corpus: **SF11 102.8 · OURS 229.7** — corroborates the 3k
head-of-file ceiling (95.3 / 245.5), so that result survives its sampling caveat.
★ **Our edge over SF11 is only 4.8%** ⇒ the risk of retargeting away our positional advantage is small
and explicitly protectable. ⚠️ Hold those 1,112 rows OUT as a no-regression guard — never train on them
(selected for us already being right ⇒ self-confirming + regression to the mean).

## ☠️🐛 SIXTH SITE — a SHIPPED BUNDLE recreated the bug it fixed, inverted

`evaluate_rooks_endgame`'s "rook behind an enemy pawn" term has TWO knobs, built in 2026-06 as
**mutually exclusive** repairs of one colour asymmetry:

| knob | effect | code |
|---|---|---|
| `ENABLE_ROOK_DBLCOUNT_FIX` | symmetrize DOWN — delete Black's extra | `+= FIX ? 0 : (att/8)*35` |
| `ENABLE_ROOK_DBLCOUNT_SYM_UP` | symmetrize UP — give White the match | `+= SYM_UP ? (7-att/8)*35 : 0` |

☠️ **BOTH shipped `true`** in the collapse bundle ⇒ White gained the term, Black lost it ⇒ the pair
**reproduced the exact asymmetry each was written to remove, sign-flipped** (originally Black-favoured,
since the bundle White-favoured). 🎯 `8/1R6/5k2/1p6/8/6K1/8/8 b` = **105 mp**, and it has been live in
every game since.

Both single-knob settings are symmetric, so symmetry could not choose. Balanced STS did:

| setting | orig | mirror | balanced | gap |
|---|---|---|---|---|
| both on (broken) | 1669 | 1590 | 3259 | +79 |
| **SYM_UP only** ✅ SHIPPED | 1673 | 1591 | **3264 (+5, free)** | +82 |
| FIX only | 1573 | 1573 | 3146 (−113) | **0** |

⚠️★★ **The DOWN arm scored a PERFECT zero colour gap and was 113 points WORSE.** Minimising colour bias
is NOT the objective — choose on the balanced TOTAL, never the gap.
★★★ Reusable: **two candidate repairs for one defect are ALTERNATIVES, not a bundle.** Shipping both was
exactly as broken as shipping neither. And since this asymmetry was introduced *by* a game-validated
ship (+45 Elo) and survived months undetected, **the mirror test belongs in the pre-ship gate**, not
just the debugging toolkit. See [[two-alternative-fixes-both-shipped-recreated-the-bug]].

## ⚠️ COLOUR vs PHASE — do not conflate them (owner correction)
The axis that MUST be symmetric is **colour**. A white/black difference *inside one function* is
presumptively a bug. A **midgame/endgame** difference is a legitimate design choice — the two phases are
separate evaluators so they can price things differently, and sweeping for phase divergence would be
mostly false positives.
⇒ The knight defect was `W=10 / B=15` **within the endgame function** (colour). Its midgame twin (20/20)
was only a WITNESS to intended convention — it does NOT argue the endgame should use 20, and the value
question was settled by balanced STS, not by the twin.
⇒ The pawn wrap guards need no twin at all: `<<9` shifts up-right so its wrap lands on file A, provable
from bitboard geometry alone.
⇒ **The systematic sweep to run is WHITE-branch vs BLACK-branch of the same function**, not midgame vs
endgame. ★ Observation worth noting but not over-reading: all three fixes so far are in ENDGAME
evaluators and every midgame twin was clean — likelier that the endgame branches got less scrutiny than
that phase divergence is itself wrong.

## 🧰 TOOLING (new this session)
| tool | what |
|---|---|
| `make_mirror_suite.py` | builds `suites/sts300_mirror.epd` — mirrors board AND each UCI move; asserts every mirrored move is legal; prints the side-to-move census as proof (`(177,123) → (123,177)`) |
| `overnight_runner.sh sts_suite` | `sts` with the suite as an argument, so the mirror twin runs in the identical regime |
| `_asym_pieceval.py` | finds accumulator-antisymmetry positions; prints engine accumulator vs material counted independently from the FEN, plus flag co-occurrence |
| `_asym_minimize.py` | greedy shrink to a minimal repro. ⚠️ selector env var is **`ASYM_TERM`** — calling it `TERM` silently minimised against the shell's `xterm-256color` |

## ▶️ FIRST ACTIONS
1. **Decide `ENABLE_KNIGHT_MOB_FIX`.** Free on every axis measured; needs an owner call to ship.
2. **Finish the capture-gains root** with the debug print, then the second root.
3. **Commit** — nothing since `e23f88a`.
4. Re-run `_reference_ceiling.py` on the full 23k (the 3k was head-of-file, not random).

## ⚠️ STANDING
- Use **`orig + mirror`** for anything colour-related; **`orig − mirror`** is a colour-bias measurement
  in move-choice units. The gap is eval **and** search (non-negamax, separate minimizer/maximizer), so
  some residual is structural — `MOB_FIX` reaches −51, not 0.
- The balanced suite is 6000 points; scale the STS jaggedness allowance with it. Only the ~0 for
  `MOB_FIX` is solid; the −142 for `SYM_UP` is nearer the noise floor.
- Corpus fitting stays PARKED — [[corpus-fit-is-anti-correlated-with-elo]]. Hunt defects.
