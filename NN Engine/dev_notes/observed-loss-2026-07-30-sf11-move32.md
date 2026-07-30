# Observed loss vs SF11 — the move-32 collapse (2026-07-30)

**Setup:** manual UI game, SF11 handicapped to **0.005 s/move**, ours as **Black**. Preset not recorded.
Ours was **winning for most of the game** and lost.

## SF18 verdict on the two suspect moves
Reference binary: **`stockfish_18_linux/stockfish-ubuntu-x86-64-avx2`** (what `STOCKFISH_PATH` points at —
a Linux SF18, distinct from the `stockfish_1/src(SF18)/` source tree). Depth 22, multipv 3.

| position | SF18 best | our COLD choice | our GAME move |
|---|---|---|---|
| move 29 `1r2k1r1/4bR2/4p3/p3N2p/3Pb1p1/1P4P1/P3K2P/2R5 b - - 1 29` | `Rf8` **+389** | `a4` **+243** | `a4` — reproduced |
| move 32 `r3k1r1/2R2R2/4p3/4N1bp/P2Pb1p1/6P1/P3K2P/8 b - - 4 32` | `Rf8` **+367** | `Bd8` **+298/+342** | `Rxa4` **−805/−833** |

## ★ The two errors are DIFFERENT BUG CLASSES

**29...a4 — reproducible cold ⇒ an EVAL/horizon failure.** Both baseline (45.4 s) and the gravity
candidate (21.4 s) choose `a4` on a cold probe. Costs **146 cp** but leaves Black at **+2.43, still
winning.** An inaccuracy worth a corpus entry, **not** the losing move.

**32...Rxa4 — NOT reproducible cold ⇒ warm state or clock.** Cold choices:
- baseline `Bd8` **+298** (SF's #2, within ~25 cp of best)
- candidate `Bd5` **+252** (~46 cp worse than baseline; n=1, read as noise)
- game `Rxa4` **−805**

**A 1200-centipawn swing, +3.67 → −8.33, on a position where the same engine cold plays near-optimally.**
The game was lost here and nowhere else.

## Why it matters more than anything measured on the bench
The cold/warm gap on this single position (≈1100 cp) dwarfs every effect measured in the 2026-07-29
knob campaign, where the whole candidate set moved WAC by ±5 solves and nodes by ~11%.

⚠️ **Unresolved confound: the preset/time this game used was not recorded.** Cold `Bd8` took 13.3 s. If the
game gave move 32 under a second, "the clock" explains it and there is no bug. **Record the preset and the
per-move time on any future observed loss** — without it the warm-state and clock hypotheses cannot be
separated. (In the SF18 pin game the owner recalls 20+ s, which would rule the clock out there.)

▶️ **Test:** replay from ~move 25 searching each move so TT/history warm naturally, then read the choice at
32. Cold = `Bd8`; if warm = `Rxa4`, the cause is isolated.
See [[warm-state-blindness-in-all-benches]] and `observed-loss-2026-07-29-sf18-pin.md` — this is the second
independent instance of the same signature.

## Structural note
At move 32 White has **both rooks on the seventh** against a king still on e8. Queens came off at move 15,
so `phase_score` has shifted toward endgame and king-safety weighting scales down — but doubled rooks on
the seventh against an uncastled king is a middlegame-grade danger. Worth an `ev_breakdown` check: if KS
fades with phase here, pawn-grabs like `a4`/`Rxa4` look correct to the eval when they are not.

## Harnesses added
- `diagnostics/game_fens.py <ply>...` — FENs at chosen plies of an embedded PGN.
- `diagnostics/probe_move.py FEN="..." [KEY=VAL...]` — engine's move under an arbitrary knob set; knobs are
  applied to `os.environ` **before** the extension import, since Config latches at init and `pyrun`
  forwards argv rather than env.
- `diagnostics/sf_verdict.py` — SF18 multipv verdict + centipawn loss for our move.
