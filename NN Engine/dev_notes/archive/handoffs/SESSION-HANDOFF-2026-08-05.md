# Session handoff — night of 2026-08-04/05: four kills, and a mechanism for why eval gains don't convert

**Read this top block first.** Prior: `SESSION-HANDOFF-2026-08-04-B.md` (the threats ship).

---

## STATE — nothing shipped tonight, engine unchanged
Default is still **`250 / 35,791,173 / EBF 3.804 / STS 1685`**, committed `7b972d1`. No default was flipped,
nothing was committed tonight. All new knobs are gated and byte-identity verified.

## ★★★ THE HEADLINE — most eval error does not change our move
`_collapse_move_vs_eval.py`, 60 collapse `decision_fen`s, SF18 d18, Threads=1, our move scored from the
**same parent** via `root_moves`:
| | share |
|---|---|
| bad move AND bad eval | 10% |
| bad move, eval fine (⇒ search/ordering) | 8% |
| **eval bad, move FINE** | **52%** |
| neither | 30% |
Move matches SF best **48%**; mean move_loss 192 cp; mean eval_err 438 cp.

⇒ **Only 18% of collapse decision points involve a bad move at all**, and half of those have a sound eval.
**Eval accuracy and move choice are only loosely coupled**, which mechanically explains the session:
- capped threats: corpus −5.2 units ⇒ **+45 Elo**, collapse burden unchanged over 800 games
- pawn structure: corpus −5.6 units and **+58 STS** ⇒ **~0 Elo on four venues**
⚠️ Two readings, not separable yet: either our decision-point moves really are fine (accumulation, not one
blunder), **or `decision_fen` is not the mistake ply**. ▶️ Audit by scoring EVERY ply of a collapsed game and
finding where move_loss actually spikes. Do this before building more on the collapse corpus.

## ☠️ PAWN STRUCTURE — best bench signal of the session, worthless in games
`ISOLATED_PAWN_PEN` / `BACKWARD_PAWN_PEN` were built and gated at 0. Corpus improved monotonically
(279.510 → 272.183), and `i200_b100` read **STS 1743 (+58)** — the only positive STS of the day.
| venue | Δ |
|---|---|
| self-play SPRT 600g | **+6.4 ±32.7** (inconclusive at cap) |
| fixed-node vs SF18, UHO 400g | **−2.55pp** |
| KP mixed 300g paired | **−1.0pp** |
| KP dense 300g paired | **−0.5pp** |
☠️ Also kills the "flat form needs an (mg,eg) split" story: **KP positions ARE endgames**, so a flat penalty
behaves endgame-weighted there — the phase hypothesis predicted a gain and got −1.0pp.
✅ Implementation audited as correct; BACKWARD was missing SF's `blocked` case (enemy pawn OCCUPYING the stop
square, not just attacking it) — fixed, worth ~0.25 corpus units, still no games.

## ☠️ PASSER VALUATION — CLOSED on a fourth mechanism
`ENABLE_PASSER_ORD_FLOOR` floors a passer at what the same pawn earns un-passed. It fixes a REAL
discontinuity — under V3 the pawn loop defers the rank bonus, so below `R ≈ 21/256` a flagged passer scores
**less than an ordinary pawn** (w1 f2: 6 mp vs `default[6]` = 90). Verified per-passer: f2 6→96, c4 13→73,
h6 159→221; healthy passers −1%. **Still corpus 279.510 → 280.127 and STS −82.**
⇒ Four mechanisms dead: rank floor · blockade-scaled residual · flat unconditional base · ordinary-pawn
floor. **Do not propose a fifth passer VALUATION mechanism.** Untested remainder: **DETECTION**
(`getPPIncrement` is occupancy-based; SF uses lever / leverPush / phalanx counts / a rank-5-support clause).
🚨 `PASSER_R_CAP` > 384 is **INERT** — `passer_realizability_R` ends `std::clamp(R, 0, 384)`, so rcap448 ≡
rcap384 byte-for-byte. The "widen the top toward SF's ~7-pawn upside" idea is bounded by architecture.

## 📚 SF SOURCE READ — what SF actually does with passers and pawns
- `bonus = PassedRank[r]` is granted **unconditionally**; the ENTIRE safety analysis sits inside
  `if (pos.empty(blockSq))`. **SF has NO blockade penalty** — it withholds the advancement bonus, never
  subtracts. Our `BLOCK[]` docking (up to 140 R) has no counterpart there. Invariant SF11 ↔ SF15.1.
- Ordinary pawns: SF dampens them hard — `Connected[r]×(2+phalanx−opposed) + 22×support`, `Isolated S(1,20)`,
  `Backward S(6,19)`, `WeakUnopposed S(15,18)`, `Doubled`, `DoubledEarly`, `BlockedPawn[]`. We have the BOOST
  half (chain/wall) and almost none of the DAMPEN half.
- SF's max passer value at rank 7 ≈ **7–8 pawns** (`284 + k·w` up to 981 vs PawnValueMg 126); ours caps at
  `1360 × 320/256 ≈ 1.7 pawns`. Our whole passer RANGE is ~4× narrower.

## 🧰 NEW TOOLING (kept)
- **`kpgauntlet` sub + `diagnostics/gen_kp_fens.py`** — randomized pawn-dominated starts vs SF18 at fixed
  nodes. `kp_fens.txt` (3-6 pawns + pieces + imbalances) and `kp_dense_fens.txt` (7-8 pawns, NO pieces).
  **300 games in ~12 min** — the cheapest game signal we have. Baselines: **KP mixed 67.3%, KP dense 68.8%**
  (n400, seed 0). ⚠️ **DIAGNOSTIC ONLY — never ship on it** (endgame FENs are 5-9% of forfeited points).
- **`_collapse_move_vs_eval.py`** — the move-vs-eval split above.
- `ENABLE_PASSER_ORD_FLOOR`, `THREAT_SAFE_PAWN` knobs; all gated, default byte-identical.

## 🐛 LATE FIND — UB in the shipped engine (real, measured)
`isNearGameEnd` is declared uninitialised and assigned ONLY in the `phase_score > 96` branch, but READ at
the `advanced_endgame_eval` gate (~L7307) inside the `else` of `if (!isEndGame)` — reached for ALL
`phase_score > 64`. So for 65..96 it was an **uninitialised read**.
| variant | WAC | nodes |
|---|---|---|
| shipped (uninitialised) | 250 | 35,791,173 |
| `= true` | **250** | **35,791,173** (BYTE-IDENTICAL) |
| `= false` (apparent intent) | 249 | 35,657,971 |
⇒ The garbage was reliably TRUE: **`advanced_endgame_eval` has ALWAYS fired in normal endgames**, and all
tuning assumed it. ✅ `= true` is in the tree — byte-identical, removes the UB. ❓ `= false` is a CANDIDATE
needing games, not a fix. ★★ **Deterministic benches do NOT imply defined behaviour.**

## ☠️ FOUR WRONG CLAIMS I MADE ABOUT THE PAWN CODE (all reverted, recorded so they are not repeated)
1. "No doubled-pawn penalty" — WRONG, it is hardcoded at L874 (125 mid / 150 eg per pawn on a shared file).
   Came from grepping the HEADER for a knob and inferring the term's absence.
2. "The 225 clamp explains why isolated/backward failed" — WRONG, they do `total +=` in a different
   function and bypass both clamps entirely.
3. "The pawn mid/eg blend extrapolates above phase 70" — WRONG, the blend lives inside `if (!isEndGame)`
   (phase <= 64) so `end_weight` tops out at 24 and the branch is UNREACHABLE. I wrote and benched a gated
   `ENABLE_PAWN_BLEND_CLAMP` before checking the enclosing scope; reverted.
4. "WAC and STS are structurally blind to endgame changes" — UNSUPPORTED, inferred from the byte-identical
   readings that #3 produced because the path never fired.
★★★ **METHOD: locate the enclosing branch before drawing any conclusion from a code fragment.** That single
check would have caught all four.
✅ What IS true about the transition: `<=40` pure midgame · `41-64` blended · `>64` endgame evaluator called
directly. The blend is parameterised over 40→70 while `isEndGame` flips at 64, so it only reaches
`0.2*mid + 0.8*end` before jumping to pure endgame — a **~20% discontinuity** from the mismatch.
✅ Also confirmed (owner was right on all three): the rear doubled pawn IS punished for lack of forward
mobility (L8655 — not flagged passed, routed to the DEFAULT rank table: 90 mp vs 1085 at rank 6); the
file-scaling IS deliberately redone flat in the endgame (100/135/50/115); and the clamps are upward-only.

## ▶️ FIRST ACTIONS
1. **Audit `decision_fen`** by scoring every ply of a few collapsed games. If the mistake ply is elsewhere,
   a lot of collapse-corpus work rests on a bad anchor.
2. **Decide what eval work is even for.** If 52% of eval error is move-neutral, "improve eval accuracy" is
   not a strength strategy by itself — target the errors that CHANGE moves.
3. **Passer detection** is the only untested part of that subsystem.
4. ⚖️ Only game changes expected to exceed the ~20-40 Elo floor; the bundle idea (pawn structure + floor +
   R_CAP) is now weak — three of its components measured ~0 or negative individually.
⚠️ Uncommitted: the ord-floor knob, the `blocked` fix, `kpgauntlet`, two new probes, the KP FEN sets, and
several grids in `fit_bench_guarded.py`.
