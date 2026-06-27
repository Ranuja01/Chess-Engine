# -*- coding: utf-8 -*-
"""Replay the two tal-BOT loss PGNs and dump White-to-move FENs (our engine = White) for the
critical run-up ranges, so the gap diagnosis can feed them to fen_vs_sf / eval_breakdown / eval_fen.

Run (WSL, from NN Engine/):  /home/ranuja/anaconda3/bin/python diagnostics/_tal_gap_fens.py
Writes diagnostics/_tal_gap_fens.csv (game,movno,played_san,played_uci,fen) and prints a summary.
"""
import io, csv, os, chess, chess.pgn

BENONI = """[Event "Play vs Bot"]
[Result "0-1"]

1. d4 Nf6 2. c4 c5 3. d5 g6 4. Nc3 Bg7 5. e4 d6 6. Nf3 O-O 7. h3 a6 8. a4 e6 9.
Bd3 exd5 10. cxd5 Nbd7 11. O-O Re8 12. Bf4 Qc7 13. Re1 Nh5 14. Be3 b6 15. Ng5 Ne5 16. Be2 h6 17. Nf3 Nxf3+ 18. Bxf3 Nf6 19. Qd2 Kh7 20.
Bf4 Nd7 21. Be2 Bb7 22. h4 Nf6 23. Bc4 Nh5 24. Be3 Qe7 25. g3 Nf6 26. f3 Nh5
27. g4 Ng3 28. Bf2 b5 29. Bb3 Be5 30. Kg2 c4 31. Bxg3 cxb3 32. axb5
axb5 33. Rxa8 Rxa8 34. Nxb5 Rc8 35. Nd4 h5 36. gxh5 gxh5 37. Nf5 Qf6 38.
Bxe5 dxe5 39. Kh1 Rc2 40. Qd1 Bc8 41. f4 Bxf5 42. Qxh5+ Kg7 43. exf5 e4
44. Qg5+ Qxg5 45. fxg5 Rxb2 46. Rxe4 Rc2 47. f6+ Kg6 48. Re8 b2 49. Rg8+ Kf5
50. Rb8 Rc1+ 51. Kg2 b1=Q 52. Rxb1 Rxb1 53. d6 Rd1 54. Kf3 Rxd6 55. Ke3 Kg4 56.
Ke4 Kxh4 57. g6 fxg6 0-1
"""

FRENCH = """[Event "Play vs Bot"]
[Result "0-1"]

1. e4 e6 2. d4 d5 3. Nc3 Bb4 4. e5 c5 5. a3 Bxc3+ 6. bxc3 Ne7 7. Qg4 O-O 8. Bd3
c4 9. Bh6 Ng6 10. Bxg6 fxg6 11. Be3 Nc6 12. h4 Rf5 13. Ne2 Qa5 14. O-O b5 15.
Rfb1 Rb8 16. Rb2 Bd7 17. f3 Qb6 18. Bf2 Qa5 19. Rbb1 Rbf8 20. Be3 Qb6 21. Bf2
Qa6 22. Re1 Ne7 23. Be3 h6 24. Reb1 Qa4 25. Ra2 Qa5 26. Bd2 Qc7 27. Nf4 Kh7 28.
a4 bxa4 29. Be3 Rb8 30. Rd1 Qa5 31. Bd2 Qb6 32. Kf1 Qd8 33. Raa1 Rb2 34. Rdc1
Qa5 35. Ke1 a3 36. Kd1 a2 37. Ke1 Qb6 38. Kf1 Rb1 39. Rxa2 Rxc1+ 40. Bxc1 Qb1
41. Qg3 Qxc1+ 42. Qe1 Qxf4 43. Rxa7 Rh5 44. Rxd7 Rxh4 45. Ke2 Nf5 46. Rf7 Rh2
47. Rxf5 Rxg2+ 0-1
"""

# (game, white-to-move fullmove ranges to dump)
JOBS = [("benoni", BENONI, [(24, 30), (44, 57)]),
        ("french", FRENCH, [(28, 40)])]

rows = []
for name, pgn, ranges in JOBS:
    game = chess.pgn.read_game(io.StringIO(pgn))
    board = game.board()
    for mv in game.mainline_moves():
        # capture White-to-move positions in the requested ranges (board.turn == WHITE, before the move)
        if board.turn == chess.WHITE:
            mn = board.fullmove_number
            if any(lo <= mn <= hi for lo, hi in ranges):
                rows.append({"game": name, "movno": mn, "side": "w",
                             "played_san": board.san(mv), "played_uci": mv.uci(),
                             "fen": board.fen()})
        board.push(mv)

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_tal_gap_fens.csv")
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["game", "movno", "side", "played_san", "played_uci", "fen"])
    w.writeheader(); w.writerows(rows)

for r in rows:
    print(f"{r['game']:7} {r['movno']:>3}.{'':3} played={r['played_san']:7} ({r['played_uci']})  {r['fen']}")
print(f"\n{len(rows)} White-to-move FENs -> {out}")
"""marker positions: benoni 29 = the axb5 vs Bb3 trade; french 33-37 = the a-pawn march a4->a3->a2."""
