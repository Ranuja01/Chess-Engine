"""
Extract the FEN at the critical moment of the 2026-07-29 SF18 loss.

Black to move after 41. Ra6, where Ra6/Be6/Kf6 sit on the sixth rank and the bishop is absolutely
pinned. Prints the position before Black's 41st and, for context, a couple of neighbouring plies so
the probe can check whether the engine ever considers the unpinning king moves.
"""
import chess
import chess.pgn
import io

PGN = """1. e4 e5 2. Nf3 Nc6 3. Nc3 Nf6 4. Bb5 Nd4 5. Nxe5 Qe7 6. Nf3 Nxb5 7. Nxb5 Qxe4+ 8. Kf1 Qc4+
9. Qe2+ Qxe2+ 10. Kxe2 Nd5 11. Re1 f6 12. Nc3 Nxc3+ 13. bxc3 d5 14. d3 Bd6 15. Kf1+ Kd8
16. Nd4 c5 17. Ne2 g5 18. h4 h6 19. c4 Be5 20. Rb1 dxc4 21. f4 gxf4 22. dxc4 Be6 23. Rxb7 Bxc4
24. Kf2 Kc8 25. Re7 Bxa2 26. Bxf4 Kd8 27. Rg7 a5 28. Bxe5 fxe5 29. Nc3 Rf8+ 30. Kg1 Bc4
31. Rb1 Rf6 32. Rh7 Bg8 33. Rg7 Be6 34. Ne4 Rf4 35. Nxc5 Bc8 36. Rh7 a4 37. Ne6+ Bxe6
38. Rh8+ Ke7 39. Rxa8 Rxh4 40. Ra7+ Kf6 41. Ra6 Rf4 42. Rbb6 Re4 43. Rxe6+ Kg5"""

game = chess.pgn.read_game(io.StringIO(PGN))
board = game.board()

# Ply index of Black's 41st move: 41 full moves * 2 = 82 plies, White's 41st is ply 81 (0-based 80).
# We want the position AFTER White's 41. Ra6, i.e. Black to move.
target_ply = 81   # number of plies played before Black's 41st

for i, move in enumerate(game.mainline_moves()):
    if i == target_ply:
        break
    board.push(move)

print("FEN after 41. Ra6 (Black to move):")
print(board.fen())
print()
print("Side to move:", "black" if board.turn == chess.BLACK else "white")
print("Legal moves:", len(list(board.legal_moves)))
print()

# Confirm the pin claim mechanically rather than by eye.
be6 = chess.E6
print("Piece on e6:", board.piece_at(be6))
print("Black king on:", chess.square_name(board.king(chess.BLACK)))
print("Is e6 bishop absolutely pinned:", board.is_pinned(chess.BLACK, be6))
print()

# The candidate unpinning defenses vs the move actually played.
for san in ("Kg7", "Kf7", "Rf4"):
    try:
        mv = board.parse_san(san)
        print(f"{san:5s} legal -> {mv.uci()}")
    except Exception as exc:
        print(f"{san:5s} NOT legal ({exc})")
