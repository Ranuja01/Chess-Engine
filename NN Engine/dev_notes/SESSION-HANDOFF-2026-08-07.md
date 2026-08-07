# Session handoff — 2026-08-07: a 2-line correctness fix beat a 63-knob descent by ~150 STS

**Read this top block first.** Prior: `SESSION-HANDOFF-2026-08-06.md`. Canonical pawn reference:
[`PAWN_MODEL.md`](PAWN_MODEL.md) (⚠️ P1 revised — see §RETRACTIONS).

---

## 🚨 STATE — the fingerprint MOVED, and the corpus baseline with it

    DEFAULT (no knobs):  250 / 35,138,590 / EBF 3.839 / WAC 250/300 / STS 1755
    WAS:                 250 / 35,791,173 / EBF 3.804 /            / STS 1685

Byte-identity was broken **on purpose** by the colour-symmetry fix. ⚠️ **STS 1755 is the number to beat.**
Corpus fit baseline on the 23,113-row corpus: **train 228.099 / val 230.949**.
HEAD `e23f88a` on `NN-ENgine`. Nothing running.

## ★★★ THE HEADLINE — correctness beat tuning, by a mile and in the opposite direction

| what | STS | games |
|---|---|---|
| **colour-symmetry fix (2 lines)** | **1685 → 1755 (+70)** | untested, shipped in the build |
| 63-knob joint descent, best config | 1755 → 1709 (−46) | **−85.6 Elo, H0 at 116 games** |
| the same descent's blend "fix" | 1755 → 1632 (−123) | dropped |

☠️☠️ **The largest corpus improvement this project has ever produced (−41 val) measured −85.6 Elo**
(`+32 −60 =24` of 116, LLR −3.144, tag `sprt_noblend`). Five previous raw-corpus fits were bench-negative;
this one is *games*-negative at the biggest proxy gain we have ever achieved. The relationship is not
weak — it is **inverted** once you optimise hard.
★ Mechanism: SF18-search is more conservative than our eval, so the cheapest way to cut win%-error is to
**shrink everything** — the winners were `PV_BOOST_MAG=0`, `IMBALANCE_SCALE=0`, `EG_EXIST_KNIGHT/BISHOP=0`,
`SCALE_PAWN_RANK` 100→30. Guards did NOT prevent it: all nine were enforced and eight tiers improved.
⇒ **Hunt correctness defects. Do not fit constants.** If you must fit, route through
`fit_bench_guarded.py` (benches dispose), never a raw corpus winner.

## 🎯 HOW GOOD IS GOOD — the achievable floor, measured for the first time
`_reference_ceiling.py`, every reference as a STATIC eval vs the SF18-SEARCH target, identical loss:

| evaluator | val |
|---|---|
| SF15.1 NNUE | 61.6 |
| SF18 static | 68.9 |
| **SF11 classical** | **95.3** ← the hand-reachable floor |
| SF15.1 classical | 139.2 |
| OURS | 245.5 |

⇒ A `val` of 231 was never "231 from perfect". **The hand-reachable floor is ~95.**
★★★ **SF11 BEATS SF15.1-classical (95 vs 139)** — newer is not better for hand-written eval; read SF11.
📏 Same-binary classical→NNUE (139 → 62) sizes what is genuinely un-encodable by hand.
⚠️ 3,000 head-of-file rows, not a random sample. Re-run on the full 23k before quoting exact values.

## ▶️ FIRST ACTIONS
1. **Continue the colour-symmetry cleanup — it is the highest-value lane by measurement.** 57.4% of
   positions still violate. 🎯 Repro in hand: **`4k3/8/8/8/8/3N4/P7/4K3 w`, off by 35 mp, isolated to
   `pt_knights`, reproduces 24/24.** Single pieces (N/B/R/Q) and single pawns are now *perfectly*
   symmetric, and pawn-vs-enemy-pawn and pawn+own-rook are clean — so it is an **own-pawn + own-knight
   interaction**. ✅ Already ruled out: the `BB_PAWN_ATTACKS[colour][r]` guards at L3470/L3565 mirror
   correctly. Also open: 2 white pawns is asymmetric 2/21 (worst +135 at `4k3/8/8/8/8/8/P6P/4K3`).
2. **Commit the session's tooling** — nothing since `e23f88a` is committed.
3. **Re-run `_reference_ceiling.py` on the full 23k** (see caveat above).

## 🧰 TOOLING (all new this session)
| tool | what |
|---|---|
| `_eval_symmetry.py` | colour-swap + file-mirror invariance, zero SF, `TERMS=1` ranks culprit terms. **Encodes three invariant families** — see below. |
| `_reference_ceiling.py` | the SF ladder scored on OUR fit objective |
| `joint_fit.py` | whole-eval descent from SHIPPED DEFAULTS; `FORCE=` pins gates so their weights get tuned |
| `_sibling_spread.py` | can a term change our move at all (deletion = the ceiling on retuning) |
| `_marginal_slice.py` | slices the piece-marginal dump in cp AND win% |
| `pawn_marginal_real.py` | now `PIECE=ALL`, `LADDER=1`, `TERMS=1`, `OUT=` |
| `static_vs_search_triage.py` | `CORPUS=1` mode + scale-invariant sign test + neutral-fitted `KFIT` |

## ☠️ RETRACTIONS AND DEAD ENDS (all measured, do not re-attempt on the same reasoning)
- ☠️ **The phase-blend "fix" is HARMFUL.** The step is real (`blend_range=30`, ramp only reaches 24/30
  inside `!isEndGame`, then jumps to 100%). Closing it costs **−107 STS alone**, −123 with `LO=34`.
  ★★★ **A demonstrated local defect is NOT a demonstrated improvement** — the surrounding constants were
  fitted around the truncated ramp. Third instance this week (obstruction blend, capture-gains polarity,
  blend step).
- ☠️ **Capture-gains evasion polarity** at the two `find_and_pop_last_viable_capture(opp_captures, …,
  current_turn)` sites reads as clearly wrong (the stack is the other side's; the sibling `find_` uses
  `!current_turn`; the helper pops unconditionally so the wrong polarity DRAINS the stack). Changing it
  measured **worse** (82,264 → 92,450 asymmetry). Reverted, comment left in the code.
- ☠️ **`imbalance_white`/`imbalance_black` were never a defect** — they are signed fields that
  negate-and-swap; the test checked plain swap and flagged all 87. `kaufman_imbalance` is perfect on all.
- ⚠️ **PAWN_MODEL P1 revised.** "+58 cp/pawn level error" is not the useful framing: absolute marginal
  values are not comparable across engines (SF11 prices a pawn at 120, SF15.1c at 65). The scale-invariant
  defect is the **ratio** — minors ~7% below the classical consensus, rooks exact.
- ☠️ The three new mechanisms (`ENABLE_WINNABILITY`, `ENABLE_CLOSEDNESS`, `ENABLE_ENDGAME_SCALE`) improve
  the corpus (185.5 vs 190.0 val when their weights are tuned) but cost **50 MORE STS** than gates-off.

## ⚠️ THREE INVARIANT FAMILIES IN `ev_breakdown` — conflating them manufactures phantom bugs
- signed contributions → **NEGATE** under mirror
- side-labelled MAGNITUDES (`det_*_pieceval`, mobility, offense/defense) → **plain SWAP**
- side-labelled SIGNED (`imbalance_white/black`) → **NEGATE AND SWAP**

## ⚠️ METHOD TRAPS ADDED THIS SESSION
- ☠️ **Coordinate descent cannot find a gated mechanism** whose weights are separate knobs: it flips the
  gate at hand-guessed defaults, and while the gate is off the weights are inert so there is no gradient.
  `ENABLE_CLOSEDNESS` was worse than untested — all-zero tables make gate-on byte-identical to gate-off.
  ✅ Force the gate, tune the weights, compare RESULTS.
- ⚠️ **Guards test TRAIN only.** `passer_under_fire` regressed on VAL in every descent and passed anyway.
- ⚠️ **~14 knobs pinned at grid edges** in both descents. Widening once doubled the gain (−20.9 → −43.2),
  so any "optimum" from those runs is a property of the grid.
- ⚠️ **Rank by win%, never cp** — the same 750 removals give opposite conditioning, worst-decile overlap
  only 33-53%.
- 🐛 **Never pipe a long run through `tail`** and **never use multi-line `pyrun -c`** (breaks the
  single-line allowlist and PROMPTS). Read outputs with the Read tool. No PowerShell — it prompts.
