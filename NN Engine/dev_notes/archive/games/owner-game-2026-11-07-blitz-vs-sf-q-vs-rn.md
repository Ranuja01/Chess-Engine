# Owner UI game 2026-11-07: gen 7 (BLITZ) vs Stockfish at 0.05 s, the move-23 transformation

**Why this file exists.** Owner, 11-07: "maybe lets reference this game for that future item so that we have something to look at."
It is a reference case for the focused eval arc after search v2 (`../../current/EVAL-V2-PARKED-REGISTER.md`, "THE FOCUSED EVAL ARC
AFTER SEARCH V2"): the items *queen vs minor compensation* and *scale on the eg leg*.

**Setup.** Our engine White, production gen 7 (`chess_engine_uci_g7pgo`, bench 246 / 36,509,830) through the UI's UCI route, preset
BLITZ (up to 3.5 s per move). Opponent Stockfish in the UI at 0.05 s per move. Opening from our book (Open Ruy Lopez).

## The owner's read
The game was drawn, but White had an advantage. **23.Qb7 was the mistake: the central pawn on e5 needed guarding to keep the
advantage.** Stockfish later avoided the draw and lost (110...Ra3?? 111.Qxa3, a hung rook), "which allowed us to win but I don't count
that as some brilliance." The goal for the future: "get our system to calculate that sort of transformation at move 23 better."

## What happened around move 23 (read from the moves, not yet engine-checked)
- After 18.Bg5 Rxa1 19.Bxd8 Rxf1+ 20.Kxf1 Rxd8: queen against rook + knight, pawns equal (White b2 e5 f2 g2 h2, Black b5 c7 f7 g7 h7).
- 21.Ke1 Nd3+ 22.Ke2 Nxb2 **23.Qb7** Nd3 24.Qxc7 Nf4+ 25.Ke1 Re8 26.g3 Nd3+ 27.Kf1 **Nxe5** 28.Qb7 Ng6 29.Qxb5: White swapped e5 and b2
  for c7 and b5. That cleared the queenside and left **Q + f, g, h pawns against R + N + f, g, h pawns, all on one wing**. That ending
  is very hard to win, because the rook and knight hold a fortress and the queen has no second front. Keeping e5 kept one.
- Hypothesis (to verify): an **eval** failure, not a search one. Losing e5 is about four moves deep, inside BLITZ depth, so the engine
  probably saw the line and scored the resulting ending as still clearly good for White. It has no "this cannot be won" scaling
  for Q vs R+N with pawns on one side, and v2 already overvalues the queen against minor-piece compensation
  (memory `v2-overvalues-queen-vs-minor-compensation`).

## To do when the arc opens (or in a quiet slot before it)
1. Our engine on the position before 23.Qb7 at BLITZ depth: its PV (does it see ...Nxe5?), its score, and its score of the
   position after 29.Qxb5.
2. Stockfish 18 at depth on the same position: 23.Qb7 against the moves that keep e5. Record the eval gap.
3. If the eval is the cause, the position before move 23 and the one after 29.Qxb5 join the arc's move test as a pair: the same
   material, with and without the second front.
4. Positions (python-chess, 11-07):
   - before 23.Qb7, White to move: `3r2k1/2p2ppp/8/1p2P3/8/5Q2/1n2KPPP/8 w - - 0 23`
   - after 29.Qxb5, all pawns on one wing, Black to move: `4r1k1/5ppp/6n1/1Q6/8/6P1/5P1P/5K2 b - - 0 29`

## PGN
```
[Event "Owner UI game"]
[Date "2026.11.07"]
[White "AlphaUX v2 gen 7 (BLITZ)"]
[Black "Stockfish (0.05 s per move)"]
[Result "1-0"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Nxe4 6. d4 b5 7. Bb3 d5 8. dxe5 Be6 9. Nbd2 Nc5 10. c3 d4 11. Bxe6 Nxe6
12. cxd4 Ncxd4 13. a4 Bc5 14. Ne4 O-O 15. Nxc5 Nxf3+ 16. Qxf3 Nxc5 17. axb5 axb5 18. Bg5 Rxa1 19. Bxd8 Rxf1+ 20. Kxf1 Rxd8
21. Ke1 Nd3+ 22. Ke2 Nxb2 23. Qb7 Nd3 24. Qxc7 Nf4+ 25. Ke1 Re8 26. g3 Nd3+ 27. Kf1 Nxe5 28. Qb7 Ng6 29. Qxb5 Rc8 30. Qd7 Rc1+
31. Kg2 Nf8 32. Qe8 Rc5 33. Qd8 g6 34. g4 h6 35. Kg3 Rc3+ 36. f3 Rc5 37. h3 Rc6 38. Qd5 Rc8 39. h4 Re8 40. Qd2 Kg7 41. Qd4+ Kg8
42. Qf4 Kg7 43. Qa4 Re7 44. Qd4+ f6 45. g5 hxg5 46. hxg5 Nh7 47. f4 Rf7 48. Qb2 Kg8 49. Qb6 Kg7 50. Qd4 Nf8 51. Qa1 Nh7 52. Kg4 Rc7
53. Qd4 Rf7 54. Qb2 Rc7 55. Kh4 Rf7 56. Kh3 Kg8 57. Qb6 Kh8 58. Qb3 Kg8 59. Kh4 f5 60. Kg3 Nf8 61. Qc4 Kg7 62. Qd4+ Kh7 63. Qd6 Kg8
64. Qd5 Kg7 65. Qe5+ Kh7 66. Kg2 Rd7 67. Kf3 Rd3+ 68. Ke2 Rd7 69. Qf6 Kg8 70. Ke3 Ra7 71. Qd6 Rd7 72. Qe5 Kf7 73. Kf3 Rd3+ 74. Kg2 Nd7
75. Qb5 Rd6 76. Kf2 Nf8 77. Qe5 Re6 78. Qd5 Kg7 79. Qd4+ Kf7 80. Qc4 Ke7 81. Qc5+ Kf7 82. Qc7+ Re7 83. Qd8 Re4 84. Qf6+ Kg8 85. Kg3 Re6
86. Qd4 Re8 87. Qa4 Re3+ 88. Kg2 Re2+ 89. Kf3 Re6 90. Qc4 Kf7 91. Kg3 Nd7 92. Qc7 Ke7 93. Kf3 Rd6 94. Kf2 Re6 95. Kg3 Rb6 96. Qc1 Kd8
97. Qc3 Rd6 98. Qb4 Rc6 99. Qd4 Ke7 100. Qg7+ Kd8 101. Kf3 Ke8 102. Qb2 Re6 103. Qb5 Rd6 104. Qc4 Kd8 105. Qb4 Ke7 106. Kg3 Nb6
107. Qc5 Kd7 108. Kh3 Nc8 109. Kg3 Rd3+ 110. Kg2 Ra3 111. Qxa3 Ne7 112. Qa1 Nd5 113. Kf3 Ne7 114. Qg7 Kd6 115. Qf7 Kd7 116. Ke3 Kd6
117. Qf6+ Kd7 118. Kd4 Nc6+ 119. Kc5 Nd8 120. Kd5 Kc7 121. Qe7+ Kc8 122. Kc5 Nf7 123. Kb6 Ne5 124. Qe8# 1-0
```
