import chess, chess.pgn, io
pgn = """[Event "Play vs Bot"]
[White "ranmal100"]
[Black "tal-BOT"]
[Result "0-1"]

1. e4 e6 2. d4 d5 3. Nc3 Bb4 4. e5 c5 5. a3 Bxc3+ 6. bxc3 Ne7 7. Qg4 O-O 8. Bd3
c4 9. Bh6 Ng6 10. Bxg6 fxg6 11. Be3 Nc6 12. h4 Rf5 13. Ne2 Qa5 14. O-O
b5 15. Rfb1 Rb8 16. Rb2 Bd7 17. f3 Qb6 18. Bf2 Qa5 19. Rbb1 Rbf8 20.
Be3 Qb6 21. Bf2 Qa6 22. Re1 Ne7 23. Be3 h6 24. Reb1 Qa4 25. Ra2 Qa5 26.
Bd2 Qc7 27. Nf4 Kh7 28. a4 bxa4 29. Be3 Rb8 30. Rd1 Qa5 31. Bd2 Qb6
32. Kf1 Qd8 33. Raa1 Rb2 34. Rdc1 Qa5 35. Ke1 a3 36. Kd1 a2 37. Ke1 Qb6 38.
Kf1 Rb1 39. Rxa2 Rxc1+ 40. Bxc1 Qb1 41. Qg3 Qxc1+ 42. Qe1 Qxf4 43. Rxa7 Rh5 44.
Rxd7 Rxh4 45. Ke2 Nf5 46. Rf7 Rh2 47. Rxf5 Rxg2+ 0-1"""
game = chess.pgn.read_game(io.StringIO(pgn))
board = game.board()
rows = []
for mv in game.mainline_moves():
    if board.turn == chess.WHITE and 24 <= board.fullmove_number <= 39:
        rows.append((board.fullmove_number, board.fen()))
    board.push(mv)
for fm, fen in rows:
    print(f"{fm}|{fen}")
