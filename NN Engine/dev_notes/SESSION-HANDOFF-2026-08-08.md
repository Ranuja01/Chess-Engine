# Session handoff — 2026-08-08: colour asymmetry 74.5% → 0.2%, and the bundle is FREE

> ## 🚨 READ THIS BLOCK FIRST — everything below it is the working log, in discovery order
>
> **EIGHT colour defects found and fixed.** Colour-swap violations **74.5% → 0.2%** (2 of 800),
> file-mirror worst **1519 → 24 mp**. The 7-fix bundle is **SHIPPED** (`a87e2f5`); the 8th
> (`ENABLE_CAPG_LVA_STATIC`) was measuring at handoff.
>
>     SHIPPED DEFAULT (7-fix):  250 / 34,426,396 / EBF 3.723 / STS 1777
>     MIRROR:                   250 / 33,879,603 / EBF 3.785 / STS 1630
>     BALANCED: tactical 500 (was 485)   positional 3407 (was 3406)   nodes −2.0%
>
> ★ **The bundle is FREE positionally and +15 tactical.** ✅ Shipped defaults reproduce every env-knob
> measurement to the NODE on all four suites.
>
> ### ★★★ THE ONE LESSON: a PARTIAL fix set has no meaningful cost
> As fixes landed the bundle read **−147 → −142 → −107 → −48 → −172 → +1**. Every intermediate number
> dissolved, including a −124 written up as a real cost ONE FIX before it vanished. Defects interact, so
> a half-fixed eval sits on no meaningful line between broken and correct.
> ⇒ **Never ship or spend games on a partial sweep. Hold defaults OFF, gate everything, take ONE bundle
> A/B at the end.** Per-fix numbers are for CHOOSING A DIRECTION only.
> ★ **Symmetry constrains SHAPE, not MAGNITUDE** — the last fix's two directions were both perfectly
> symmetric and 132 balanced points apart. And an arm with a PERFECT zero colour gap was 113 points
> WORSE ⇒ choose on the balanced TOTAL, never the gap.
>
> ### The eight, by mechanism (5 of 8 found by READING code)
> | # | defect | mechanism |
> |---|---|---|
> | 1-2 | king-race tempo polarity (2 sites, 08-07) | wrong colour predicate |
> | 3 | knight endgame mobility 10 vs 15 | per-colour constant |
> | 4 | pawn diagonal-support guards | swapped file-wrap masks (`<<9` wraps to A, `<<7` to H) |
> | 5 | rook `DBLCOUNT` pair | two MUTUALLY EXCLUSIVE fixes both shipped ⇒ bug returned inverted |
> | 6 | rook own-pawn rank window | `>4` should be `>2` (parked knob from June, re-tested) |
> | 7 | rook enemy-pawn rank window | `<5` should be `<3` |
> | 8 | capgain sort tie-break | `std::sort` unstable + NO tie-break ⇒ ties inherit square order |
> | 9 | KS modulators | `>>` on a SIGNED value rounds toward −inf; `/256` truncates toward zero |
> | 10 | capgain attacker choice | ranked by EVAL MAGNITUDE `square_values[]`, not piece TYPE |
>
> ⚠️ **Two were introduced BY game-validated ships** (`MOD_KS_REALIZ` with +36.7 Elo, rook `DBLCOUNT`
> with +45 Elo) ⇒ **winning Elo is no protection.** The mirror test is now a SHIP GATE in
> `NN Engine/CLAUDE.md`, with all six recurring shapes written out.
>
> ### ▶️ NEXT
> 1. Ship `ENABLE_CAPG_LVA_STATIC` if its 4-suite read is clean (1.4% → 0.2%).
> 2. **2 of 800 positions remain** — same capgain family, ~1100 mp. A residual, not a class.
> 3. **New-content SPRT** (`WINNABILITY`/`CLOSEDNESS`/`ENDGAME_SCALE`) — NEVER game-tested, and now it
>    would run against a clean eval instead of a moving one.
> 4. **Retargeted retune** — SF11-static labels banked (23,084 rows), filter sized: TRAIN 83.4% ·
>    GUARD 4.8% (hold OUT, never train) · discard 11.8%.

---

# Working log — the bench was the blocker, not the bug

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

## ✅ SEVENTH SITE — `ENABLE_ROOK_RANKWIN_FIX`, parked since June, is the biggest single win

The midgame rook penalises its OWN pawn on its file over a rank window. White fires for ranks 0-4
(`(att>>3) < 5`); the mirror of that is Black ranks 3-7, i.e. `> 2` — but the code says `> 4`, so
**Black silently skips ranks 3 and 4**. The magnitudes already mirror (`3-r` vs `r-4` agree under
`r ↔ 7-r`), so it is purely the window. Provable from the two lines.

`ENABLE_ROOK_RANKWIN_FIX=1` (no code change, the knob exists):
**violations 23.8% → 14.6%**, `pt_rooks` 7659/122 positions → **1809/36**.

☠️★★★ It was **parked in 2026-06** with: *"eval_symmetry couldn't confirm a residual drop (motif too
rare + loop break-coupling)"*. The defect was correctly suspected two months ago and abandoned because
the instrument of the day could not resolve it. ⇒ **A parked null from a weak instrument is not a
null.** Same family as [[coordinate-descent-cannot-find-gated-mechanisms]] and "a negative from a
monitoring tool is a claim about the TOOL".

### Four-fix bundle (all gated, defaults OFF)
| config | STS orig | mirror | balanced | gap |
|---|---|---|---|---|
| baseline | 1755 | 1651 | 3406 | +104 |
| 3 fixes | 1673 | 1591 | 3264 (−142) | +82 |
| **4 fixes** | 1641 | 1658 | **3299 (−107)** | **−17** |
★ The bundle cost SHRINKS as the eval gets more correct (−147 → −142 → −107) and the colour gap
collapses to ~0. Suggestive that part of the wrap fix's cost was interaction with the remaining bugs.
⚠️ Three data points against ~140 jaggedness on a 6000-point suite — suggestive, not established.

## ☠️🐛 EIGHTH SITE — capture-gains picks a DIFFERENT capture in each orientation (ROOT FOUND, NOT FIXED)
Traced with the new `CAPG_TRACE=1`, after three separate code-readings produced three wrong stories:

    BASE   (w) CAPG w from=41 to=42 ptFrom=4 fired=1 prb=275 vg=1275   Rb6xc6, rook takes pawn
               CAPG b from=20 to=13 ptFrom=1 fired=0 prb=0   vg=0
    MIRROR (b) CAPG b from=53 to=44 ptFrom=1 fired=0 prb=0   vg=1000   f7xe6, pawn takes pawn
               -- the mirrored ROOK capture never happens at all

`find_and_pop_last_viable_capture` selects by position in a stack built in **square order**, and square
order is not mirror-invariant: base white chooses from {41 rook, 13 pawn} → picks 41 (the max); mirror
black chooses from {17 rook, 53 pawn} → picks 53 (the max). But mirroring maps 41→17 and 13→53, so
**the max becomes the min** and the two orientations simulate different capture sequences.
★ Explains everything that was unexplained: the turn-gating, the mirror sitting at exactly the bare
pawn value at EVERY clamp setting, and why the earlier "evasion polarity" fix measured WORSE.
▶️ **Not a constant flip** — the selection RULE must become colour-blind (highest `value_gained`, or
SEE-best, with an invariant tie-break). Own gate, own measurement. Note this is *approximate*
capture-gains, so "highest index wins" was never principled, merely unexamined.

### ✅ FIXED — and the real cause was NARROWER than "selection by square index"
↩️ **Correction to the diagnosis above.** Selection is NOT by square index: the stacks are
`std::sort`ed on `value_gained`, which IS mirror-invariant. The defect is that **`std::sort` is not
stable and there is no tie-break**, so equal-valued captures keep insertion order — `ctz` ascending,
i.e. square order, which reverses under a mirror. Confirmed from the pending stacks:

    BASE   white  {13 P->20 val=1000, 41 R->42 val=1000}   back() = 41, the ROOK capture
    MIRROR black  {17 R->18 val=1000, 53 P->44 val=1000}   back() = 53, the PAWN capture

Both captures take an undefended pawn, so both are worth exactly 1000 — a TIE. Only a non-pawn
capturing a pawn enters the rank-bonus branch, so the mirror silently loses it.

**`ENABLE_CAPG_INVARIANT_ORDER`** (default off) adds the tie-break: `value_gained`, then **least
valuable attacker** (MVV-LVA — colour-blind and chess-sensible), then **own-perspective square**
(`sq` for white, `sq ^ 56` for black) so no tie can ever fall back on raw square order again.
Applied to BOTH capture-gains functions. Drain behaviour untouched.

| | 4 fixes | **+ capgain order** |
|---|---|---|
| colour-swap violations | 117 (14.6%) | **102 (12.8%)** |
| **total asym mass** | 30,912 | **6,475 (−79%)** |
| worst violation | 4,724 mp | **1,241 mp** |
| file-mirror worst | 1,519 mp | **194 mp** |

★ `advanced_endgame_total`, `ae_input`, `ae_matedrive` all **vanish** — they were purely downstream of
capgain via `material` (= `blackPieceVal − whitePieceVal`, mutated inside the capture loop).
`piece_value_boost` 6,949 → 458. ★ Position count barely moves but MASS drops 79%: this removed the
LARGE violations, not the numerous ones.

### 📈 BUNDLE COST RECOVERS MONOTONICALLY AS THE SWEEP COMPLETES
| fixes | balanced STS | vs baseline |
|---|---|---|
| 2 | 3259 | **−147** |
| 3 | 3264 | −142 |
| 4 | 3299 | −107 |
| **5** | **3358** | **−48** |
★★★ At −48 on a 6000-point suite against ~140 jaggedness, the correctness bundle is **effectively
NEUTRAL** — while cutting violations 74.5% → 12.8% and asymmetry mass by 79%.
⇒ **This vindicates holding the defaults OFF and refusing to ship mid-sweep.** Shipping at 3 fixes would
have banked a −142 that was mostly an artifact of the REMAINING bugs, and a night of games would have
been spent explaining a cost that was already dissolving. Per-fix costs really were conditional.
↩️ **Retires the "unexplained diffuse cost"** framing: most of the −142 was interaction with unfixed
defects; what remains is inside noise.

### ☠️ EVASION POLARITY IS NOT A SYMMETRY FIX — old warning RETIRED
`ENABLE_CAPG_EVADE_POLARITY_FIX` flips the evasion pops to `!current_turn` (the stack belongs to the
other side; with the wrong polarity `isValid` fails for every entry and the unconditional pop drains
the WHOLE opponent stack).
✅ **Knob verified live**: 58/1500 positions change, max **5768 mp**.
☠️ **Zero effect on colour symmetry** — 102 violations, 6475 mass, every term IDENTICAL to the digit.
The change is antisymmetric: it moves the eval but affects both orientations equally.
↩️ **The in-code "measured worse (82,264 → 92,450)" warning is RETIRED.** With the tie-break defect in
the same function now fixed, the polarity is asymmetry-neutral, so that old reading was measuring the
tie-break bug, not the polarity. ⇒ Whether the polarity is correct is a **capture-simulation** question
needing its own bench/games justification — do NOT bundle it with the symmetry work.
★ Method note: this null is trustworthy ONLY because the knob was proven to move the engine first.

### ▶️ RESIDUAL capgain root #2 — still open, cause UNKNOWN
`material`/`capture_gains` keep **2,549 over 11 positions**. 🎯 Repro
`3q1rk1/8/5n1b/2n5/2b5/3P4/8/2R3K1 w` (10 pieces): `capture_gains` base **−2175**, mirror **exactly 0**
— one orientation credits nothing at all. Load-bearing: Rc1, Pd3, Bc4 (two white attackers of the same
bishop; the mirror has the matching pair).
▶️ Prime suspect is the **evasion block**: `find_and_pop_last_viable_capture(opp_captures, …,
current_turn)` uses the wrong polarity, so `isValid` fails for every entry and the helper — which pops
unconditionally — **drains the whole opponent stack**. That would zero one side's gains exactly like
this, and it is turn-order dependent hence asymmetric.
⚠️★ The in-code note says fixing that polarity measured WORSE (82,264 → 92,450 asymmetry). **Re-test
it**: that measurement predates every fix tonight, including the tie-break defect in the SAME function
that was corrupting the measurement. Same lesson as the rank-window knob — a null from a compromised
instrument is not a null.

### Original design sketch (superseded by the above, kept for the drain warning)
⚠️ **Two coupled behaviours, not one.** `find_and_pop_last_viable_capture` selects `captures.back()`
**and pops every entry it passes, valid or not** — the scan is destructive. The in-code comment records
that something downstream depends on that draining (the "evasion polarity" change altered it and
measured WORSE). So a selection change is also a drain change unless deliberately kept separate.

Invariant key to select on, in order:
1. `value_gained` DESC — the heuristic's own notion of "best", and colour-blind.
2. attacker value ASC — standard MVV-LVA, also colour-blind.
3. **own-perspective square** ASC — the tie-break MUST be colour-relative or the bug returns. Use
   `captureColour ? sq : (sq ^ 56)` (rank flip) so both colours index the same geometry. A raw-square
   tie-break is exactly the current defect.

▶️ Gate it (`ENABLE_CAPG_INVARIANT_ORDER`), keep the drain behaviour byte-identical in the first cut,
and measure separately — it moves which sequence capture-gains simulates in many positions, so folding
it into the four-fix bundle would make the eventual games result unattributable.
✅ Verify with `_capg_trace_pair.py`: every BASE line must have a rank-flipped MIRROR twin with the same
`fired` and the same `|prb|`.

## 📊 WHERE THE SWEEP STANDS — 74.5% → 14.6%, and the tail is NOT rounding
| tolerance | violations |
|---|---|
| exact | 117 (14.6%) |
| >20 mp | **113 (14.1%)** |
| >100 mp | 18 (2.2%) |
★★ **Only 4 of 117 are sub-20 mp.** The remaining tail is NOT integer-rounding noise from the
`>>1`/`>>2`/`/100` scalings — it is ~95 positions in a tight **20-100 mp band, median 30**, which is the
signature of a SPECIFIC TERM (the rank-window defect looked exactly like this and was 86 positions).
⇒ Expect at least one, plausibly two, more moderate systematic defects. Exact zero is still coherent —
rounding is not yet the binding constraint.

| remaining source | positions | mean | status |
|---|---|---|---|
| `material`/`capture_gains` | 30 | 659 | root found (above), unfixed |
| `piece_value_boost` | 20 | 347 | almost certainly downstream of the same accumulators |
| `king_safety` | 67 | 33 | unexamined |
| `pt_rooks` residual | 36 | 50 | unexamined |
| `advanced_endgame_total` | 7 | 1246 | unexamined; rare but huge |
| `det_ks_units_w<->b` | **278** | 10 | ⭐ breaks on 2.4× more positions than `total` does ⇒ asymmetric almost everywhere and usually CANCELS downstream. Prime suspect for the 30 mp band whenever cancellation is imperfect. **Look here next.** |

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
