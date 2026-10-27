# Observed WIN — Chessiverse bot, knight ending (2026-07-30)

**Setup:** manual UI game, `PRESET=LIGHTNING`, ours as **Black**. Opponent: Chessiverse "IM John
Bartholomew" persona bot, listed 2434; our account listed 1563. **Result: 0-1 (White resigned).**

⚠️ **Read with care:** Chessiverse persona bots are tuned to feel HUMAN, not to play at engine strength for
their nominal rating — a 2434 label there is not a 2434 engine. One game, fast TC. The technique below is
real; the rating gap is not evidence of strength.

## The game
```
1. Nf3 d5 2. d4 Nf6 3. c4 e6 4. Bg5 dxc4 5. e3 b5 6. a4 Bb4+ 7. Nc3 c6 8. Be2 h6 9. Bh4 Nbd7
10. O-O Qb6 11. Qc2 O-O 12. b3 cxb3 13. Qxb3 bxa4 14. Nxa4 Qb7 15. Rfb1 a5 16. Nc5 Bxc5
17. Qxb7 Bxb7 18. Rxb7 Bd6 19. Nd2 Rfb8 20. Rxb8+ Nxb8 21. Nc4 Bb4 22. Bf3 Ra6 23. Bg3 Nbd7
24. Bc7 a4 25. Be2 Ra7 26. Bd6 Bxd6 27. Nxd6 Nd5 28. Nc4 Kh7 29. Bd3+ f5 30. Bc2 N7b6 31. Ne5 Nb4
32. Bd1 a3 33. Bb3 N6d5 34. Kf1 h5 35. Ke2 a2 36. Kd2 Ra3 37. Bc4 g6 38. Kc1 Rc3+ 39. Kb2 Rc2+
40. Kb3 Rxf2 41. Bxd5 Nxd5 42. Rxa2 Rxa2 43. Kxa2 Nxe3 44. Nxc6 Nxg2 45. Nd8 Nf4 46. Kb3 Kg7
47. Kc4 g5 48. Kc5 Kf6 49. Kd6 h4 50. Nc6 g4 51. Ne5 g3 52. hxg3 hxg3 53. Nf3 Ne2 54. d5 exd5
55. Kxd5 g2 56. Kc4 f4 57. Kd3 g1=Q 58. Nxg1 Nxg1 59. Ke4 Kg5 60. Kd3 f3 61. Ke3 Kg4 62. Kf2 Nh3+
63. Kf1 Kg3 64. Ke1 f2+ 65. Kf1 Kh2 66. Ke2 Kg2 67. Ke3 f1=Q 0-1
```

## What was actually well played
- **The a-pawn campaign.** 32...a3 and 35...a2 drove the passer to the second rank and forced the
  concessions that won material (40...Rxf2, then the 41-43 liquidation into a clean pawn-up ending).
- **The knight-escort finish.** 62...Nh3+ 63. Kf1 Kg3 64. Ke1 f2+ 65. Kf1 **Kh2!** 66. Ke2 Kg2 is the
  correct technique — the knight covers g1 while the king shoulders White off the queening square. That is
  a concrete, non-obvious maneuver, not something a shallow search stumbles into.
- 53...Ne2 and 55...g2 kept the tempo race won after the d5 counter-thrust.

## Why it is diagnostically interesting next to the SF18 loss
The two failures/successes are **different subsystems**:
- vs SF18: **drifted** into a lost R+B ending (eval judgement) and then missed a concrete pin
  (see `observed-loss-2026-07-29-sf18-pin.md` — warm-state suspect).
- here: **converted** a concrete N+pawns ending cleanly at LIGHTNING.

⇒ Consistent with the standing picture: the search handles concrete, forcing positions well; the losses
come from positional drift (~18pp behind SF15-classical) plus, apparently, warm-state contamination of
ordering. **Not** from raw tactical depth.
