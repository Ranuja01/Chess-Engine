# Observed loss vs SF18 — the move-41 pin (2026-07-29)

**Setup:** manual UI game. SF18 handicapped to **0.005 s/move**; ours as **Black** on `PRESET=STANDARD`
(full time, high depths reported). Ours lost.

Owner's read: *"I do think there's an eval element to it as it thought it was winning for a while. But it
also achieved very high depths, where it clearly wasn't seeing the error of its ways, such as not seeing
the bishop would be lost when playing 41...Rf4 ... The high depth search should be able to see this,
meaning this is more of a search fault than eval."*

## The game
```
1. e4 e5 2. Nf3 Nc6 3. Nc3 Nf6 4. Bb5 Nd4 5. Nxe5 Qe7 6. Nf3 Nxb5 7. Nxb5 Qxe4+ 8. Kf1 Qc4+
9. Qe2+ Qxe2+ 10. Kxe2 Nd5 11. Re1 f6 12. Nc3 Nxc3+ 13. bxc3 d5 14. d3 Bd6 15. Kf1+ Kd8
16. Nd4 c5 17. Ne2 g5 18. h4 h6 19. c4 Be5 20. Rb1 dxc4 21. f4 gxf4 22. dxc4 Be6 23. Rxb7 Bxc4
24. Kf2 Kc8 25. Re7 Bxa2 26. Bxf4 Kd8 27. Rg7 a5 28. Bxe5 fxe5 29. Nc3 Rf8+ 30. Kg1 Bc4
31. Rb1 Rf6 32. Rh7 Bg8 33. Rg7 Be6 34. Ne4 Rf4 35. Nxc5 Bc8 36. Rh7 a4 37. Ne6+ Bxe6
38. Rh8+ Ke7 39. Rxa8 Rxh4 40. Ra7+ Kf6 41. Ra6 Rf4 42. Rbb6 Re4 43. Rxe6+ Kg5 44. Kf2 Rc4
45. Rac6 Rf4+ 46. Ke3 Kg4 47. Rg6+ Kh5 48. Rxh6+ Kg5 49. Rhg6+ Kh5 50. Rgf6 Rg4 51. Rh6+ Kg5
52. Rcg6+ Kf5 53. Rxg4 Kxg4 54. Ra6 Kf5 55. c4
```

## ★ The critical moment — 41. Ra6 is an ABSOLUTE PIN

White `Ra6`, Black `Be6`, Black `Kf6` are all on the **sixth rank** with b6/c6/d6 empty. The bishop is
frozen against its own king. **42. Rbb6** adds a second attacker to a piece that cannot move; 42...Re4
defends only once; **43. Rxe6+** collects it. The piece was lost by force the moment the king stayed on f6.

**The defense was to break the pin at once: 41...Kg7 or 41...Kf7.** Worse, but not a piece down.
Played instead: **41...Rf4**, which neither unpins nor defends.

## Why this reads as a SEARCH-TRIAGE fault, not a depth fault

A two-move pile-on against a pinned piece is visible from roughly depth 4. Ours reported *high* depth and
still played it. So the question is not "deep enough?" but **"was the saving move ever searched at all?"**

Measured context that makes this plausible:
- Root razoring discards the move the NEXT iteration wants **16.7% (WAC) / 17.9% (quiet)** of the time.
- **~23 of ~35 root moves are never touched by the main search** — we razor and `break`.
- A quiet king retreat in a bad endgame is exactly the profile that scores low in a shallow pass and gets
  razored off the list before it is ever properly searched.

⇒ **Falsifiable A/B:** probe the position after 41. Ra6 with default vs `ENABLE_ROOT_RAZOR=0`. If
razoring-off finds Kg7/Kf7 and default does not, that is the first evidence from a REAL GAME that the razor
costs material — stronger than anything the synthetic suites produced, and grounds to reopen the razor
question closed on 2026-07-29 bench data.

## ☠️ RAZOR HYPOTHESIS FALSIFIED — and the real suspect is WARM STATE

Probed `8/8/R3bk1p/4p3/p6r/8/2P3P1/1R4K1 b - - 3 41` via `diagnostics/probe_pin.py` (accepts `KEY=VAL`
argv and applies it to `os.environ` **before** importing the extension, since Config latches at init and
the runner's `pyrun` forwards argv rather than env):

| arm | chosen | time |
|---|---|---|
| default (long TC) | **Ke7** | 45.6 s |
| `PRESET=STANDARD` | **Ke7** | 24.0 s |
| `PRESET=STANDARD ENABLE_ROOT_RAZOR=0` | **Ke7** | 36.5 s |
| `PRESET=STANDARD` + gravity candidate | **Ke7** | **15.9 s** |

**Every arm finds Ke7** — better than the Kg7/Kf7 suggested above, because the king on e7 breaks the pin
*and* defends the bishop. **Razoring off changes nothing ⇒ the move was never being triaged away.**

★ Owner reports move 41 got **20+ seconds in-game**. Cold probe at 24 s plays Ke7; the game played Rf4.
⇒ **The differentiator is WARM STATE: a TT holding 40 moves of entries, and history tables shaped by the
whole game.** Note `Rf4` had literally worked earlier — **move 34 was Rf4** — and history rewards from/to
pairs that produced cutoffs, so it gets ordered first in a position where it now hangs a piece.

🚨 **NO INSTRUMENT WE OWN SEES THIS.** WAC and STS probe COLD positions one at a time. Warm-state
contamination is invisible to every bench in the repo, which is how a blunder this plain survives a suite
reporting 249/300.

▶️ **TEST:** replay the game through the engine from ~move 30, searching each move so the TT/history warm
naturally, then read the choice at move 41. Cold = Ke7; if warm = Rf4, the cause is isolated. ~5 min, 1 core.

⚠️ Incidental: the gravity candidate reached the same move in **15.9 s vs 24.0 s** (n=1, not evidence, but
the direction the −11% node result predicts).

## Two separate failures in one game
1. **Eval optimism** — thought it was winning while drifting into a lost R+B ending. We are ~18pp behind
   SF15-classical on exactly this class of position.
2. **Move-41 triage** — a forced material loss missed at high depth. Different bug, same game.

⚠️ Owner's observation of a running game has overturned bench conclusions before
(`wac-underrepresents-root-width`). Treat this as evidence, not anecdote.
