# Pawn clamp headroom — the redesign's precondition, measured

Answers the brief's gating question ("instrument how often the clamp binds BEFORE designing anything that
adds to it") with `diagnostics/_pawn_clamp_headroom.py` + the new `ai.pawn_clamp_records()` probe.
7 position sets, ~39,000 pawn records, engine at the shipped default (byte-identical, 250 / 35,791,173).

## THE NUMBERS
| set | path | bind% | mean raw | struct | pos | +50 survives |
|---|---|---|---|---|---|---|
| diverse_wide | mid | 35.9 | 211.2 | 71.5 | 139.8 | 58% |
| position_bank | mid | 36.1 | 213.0 | 72.6 | 140.5 | 58% |
| collapse | mid | 35.6 | 210.3 | 68.1 | 142.2 | 59% |
| sts300 | mid | 36.8 | 214.6 | 70.2 | 144.3 | 58% |
| wac | mid | 36.4 | 206.3 | 68.3 | 138.0 | 59% |
| diverse_wide | end | 13.5 | 89.1 | 89.1 | — | 83% |
| sts300 | end | 16.8 | 103.2 | 103.2 | — | 78% |
| wac | end | 14.7 | 95.6 | 95.6 | — | 81% |
| kp_dense | end | 13.6 | 83.3 | 83.3 | — | 82% |
| kp_mixed | end | 5.4 | 50.2 | 50.2 | — | 92% |

## ✅ THE CLAMP BINDS IN THE MIDGAME, AND THE RATE IS CORPUS-INVARIANT
**35.6–36.8% across five wildly different sets** — general play, the collapse corpus, a positional suite and
a tactical suite all agree to within 1.2pp. Given [[corpus-composition-decides-the-optimum]] that invariance
is the surprise, and it has a cause (below): the binding quantity is not a pawn-structure signal.
A midgame term added inside the clamp is taxed ~40%: **+25 survives 62%, +50 → 58%, +100 → 49%.**
Real, but NOT fatal — this alone does not kill a richer midgame table.

## ★★★ THE ACTUAL FINDING — THE CAP IS SPENT ON THINGS THAT ARE NOT PAWN STRUCTURE
Midgame `raw = structural + positional` averages **211 of a 225 cap**, but splits **70 structural / 141
positional**. `positional` is the placement layer plus the attacking layers for the pawn's own square and
both squares it attacks — none of it a pawn-structure signal. **Positional ALONE already meets the cap for
21–23% of pawns**, i.e. for a fifth of all pawns every chain/wall/latent-support bonus is worth exactly
zero before any redesign adds anything.
⇒ The midgame clamp is not a pawn-structure budget that is too small. It is a shared budget **two-thirds
spent by placement and attacking layers**, and pawn structure is the thing being crowded out.
⇒ The truncation is worst precisely on pawns with high placement/attack scores — advanced, central pawns —
which is exactly where a file × rank scheme is supposed to discriminate. That is a design hazard for the
brief's scheme, not a general argument against it.

## ✅ THE ENDGAME CLAMP IS NOT A CONSTRAINT
**11.5% combined**, mean raw 76 against a 175 cap, ~106 mp of mean headroom, and only **5.4%** on KP mixed.
The endgame path applies the attacking layers straight to `total` and clamps `structural` alone, so it never
had the crowding problem. **Endgame pawn work is unconstrained — raising the endgame cap would do nothing.**

## ⚠️ WHAT THIS DOES *NOT* EXPLAIN
It does **not** bear on why `ISOLATED_PAWN_PEN` / `BACKWARD_PAWN_PEN` failed on four venues. Those do
`total +=` in a different function and **bypass both clamps entirely** (recorded as wrong-claim #2 in
`SESSION-HANDOFF-2026-08-05.md`; re-verified here). The clamp story explains nothing about that result and
must not be retrofitted onto it.

## ▶️ WHAT THIS OPENS (candidates, none measured yet)
1. **Clamp the two quantities separately** — `min(225, positional) + min(C, structural)` — so structural
   signal stops competing with placement. Cheap, and it is a structural change rather than a constant tweak,
   so it is the right SIZE of change for the ~20-40 Elo game floor.
2. Put any new file × rank scheme **outside** the shared clamp.
3. Raising the midgame cap alone is the weakest option: it hands ~2/3 of the new headroom to the placement
   and attacking layers, which nobody asked to strengthen.

## 🧰 TOOLING
- `ai.pawn_clamp_records(board)` → per-pawn `{sq, white, endgame, structural, positional, raw, cap}`.
  Probe-flag gated (`PawnClampRec`, mirrors `PasserRec`); recorder is inlined in the header so the disabled
  path is one predictable branch. Verified byte-identical: 250 / 35,791,173 / EBF 3.804.
- `diagnostics/_pawn_clamp_headroom.py` — `MAX_POS=`, `SETS=`, `ADDS=`.
