# -*- coding: utf-8 -*-
"""Grade one side's moves in a game with Stockfish 18's win/draw/loss expectation (owner UI games, 2026-11-12).

For every move of the graded side: SF18's expected score for that side before the move (its best move) and after the move
actually played, both at a fixed node budget, and the drop. The moves where the expected score fell are where a won or
drawn position was let go. Expected score = SF18's WDL from the side's point of view, win + draw / 2, in percent.

  pyrun diagnostics/_grade_game.py PGN=<file> SIDE=white|black [NODES=4000000] [FROM=<move number>]

The PGN may start from a FEN (odds games). One SF18 thread, 256 MB hash, fresh per run.
"""
import io, sys
import chess, chess.pgn, chess.engine

KV = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
SF = "/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/stockfish_18_linux/stockfish-ubuntu-x86-64-avx2"
side = chess.WHITE if KV.get("SIDE", "white").lower() == "white" else chess.BLACK
nodes = int(KV.get("NODES", "4000000"))
first = int(KV.get("FROM", "1"))
game = chess.pgn.read_game(io.StringIO(open(KV["PGN"], encoding="utf-8", errors="ignore").read()))
eng = chess.engine.SimpleEngine.popen_uci(SF)
eng.configure({"Threads": 1, "Hash": 256, "UCI_ShowWDL": True})
board = game.board()
rows = []
for move in game.mainline_moves():
	if board.turn == side and board.fullmove_number >= first:
		n = board.fullmove_number
		before = eng.analyse(board, chess.engine.Limit(nodes=nodes))
		best = before["pv"][0] if before.get("pv") else None
		e0 = 100.0 * before["wdl"].pov(side).expectation()
		san, best_san = board.san(move), (board.san(best) if best else "?")
		board.push(move)
		after = eng.analyse(board, chess.engine.Limit(nodes=nodes))
		e1 = 100.0 * after["wdl"].pov(side).expectation()
		rows.append((n, san, best_san, e0, e1))
		print("%3d%s %-8s best %-8s E %5.1f -> %5.1f  drop %5.1f%s" % (n, "." if side == chess.WHITE else "...", san, best_san, e0, e1, e0 - e1,
		      "  <<" if e0 - e1 >= 10 else ""), flush=True)
	else:
		board.push(move)
eng.quit()
if rows:
	peak = max(rows, key=lambda r: r[3])
	print("peak expected score %.1f before move %d (%s played, SF18 best %s)" % (peak[3], peak[0], peak[1], peak[2]))
	print("moves costing >= 10 points: %s" % ", ".join("%d %s (%.0f)" % (r[0], r[1], r[3] - r[4]) for r in rows if r[3] - r[4] >= 10))
