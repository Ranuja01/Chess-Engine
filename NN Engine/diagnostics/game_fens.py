"""
Print FENs at chosen plies of a PGN, for feeding into probe_move.py.

Usage: pyrun diagnostics/game_fens.py <ply> [<ply> ...]
A "ply" here is the number of half-moves already played, so the position BEFORE Black's move n is
ply 2n-1. Defaults to the two critical points of the 2026-07-30 SF11 loss.
"""
import sys
import io
import chess
import chess.pgn

PGN = """1. d4 d5 2. Nd2 c5 3. e3 cxd4 4. exd4 Bf5 5. Ngf3 e6 6. Bb5+ Nc6 7. Ne5 Qb6 8. c4 Ne7
9. Qa4 a6 10. Ndf3 f6 11. Bxc6+ bxc6 12. O-O fxe5 13. Be3 dxc4 14. Nxe5 Qb5 15. Qxc4 Qxc4
16. Nxc4 Nd5 17. Rfc1 Nxe3 18. fxe3 Rc8 19. Ne5 Be4 20. Rf1 g5 21. Rf7 Be7 22. Rf2 c5 23. Rc1 a5
24. Kf1 h5 25. g3 Rg8 26. Ke2 Rb8 27. b3 cxd4 28. exd4 g4 29. Rf7 a4 30. bxa4 Bg5 31. Rc5 Ra8
32. Rcc7 Rxa4 33. Rc8+ Bd8 34. Rd7 Rxa2+ 35. Ke3 Ra8 36. Rdxd8+ Ke7 37. Rxg8 Rxc8 38. Rxc8 Bb1
39. Rh8 Bf5 40. Rxh5 Kd6 41. Rg5 Ke7 42. Nxg4 Bb1 43. Rb5 Ba2 44. Ne5"""

plies = [int(a) for a in sys.argv[1:]] or [57, 63]

game = chess.pgn.read_game(io.StringIO(PGN))
moves = list(game.mainline_moves())

for target in plies:
    board = game.board()
    for i, mv in enumerate(moves):
        if i == target:
            break
        board.push(mv)
    nxt = moves[target] if target < len(moves) else None
    played = board.san(nxt) if nxt is not None else "-"
    print(f"--- after {target} plies (move {target//2 + 1}, {'black' if board.turn == chess.BLACK else 'white'} to move)")
    print(f"FEN    : {board.fen()}")
    print(f"played : {played}  ({nxt.uci() if nxt else '-'})")
    print(f"legal  : {len(list(board.legal_moves))}")
    print()
