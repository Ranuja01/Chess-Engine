# -*- coding: utf-8 -*-
"""Depth ladder on positions from one game: does our engine find the right move with more depth, or with pruning off?

For each listed move of the graded side: SF18's best move, then our UCI engine's best move at every depth of one search
(depth, move, score, nodes, time), once per knob variant, each in a fresh engine process. Then our own verdict on the two
candidates: the position after the move played in the game and after SF18's move, each searched to VERIFY_DEPTH by our
engine, scored from the graded side's point of view.

  pyrun diagnostics/_depth_ladder.py PGN=<file> SIDE=white|black MOVES=52,67 [DEPTH=20] [MAXSEC=120] [VERIFY_DEPTH=16]
        [BIN=build_uci/chess_engine_uci] [SFNODES=8000000] [VARIANTS="normal:|noprune:ENABLE_LMR=0,ENABLE_LMP=0"]

Reading it:
  found from depth d   our search settles on SF18's move at d and keeps it → the miss was depth (time), not knowledge
  never found, normal and no-pruning alike, and our verify scores rank the played move above SF18's → evaluation / horizon
  found only with pruning off → pruning
A variant whose node counts equal the normal run's did not take effect (the knob is not live): read it as missing.
"""
import io, os, sys, time
import chess, chess.engine, chess.pgn

KV = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
ND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SF = "/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/stockfish_18_linux/stockfish-ubuntu-x86-64-avx2"
BIN = os.path.join(ND, KV.get("BIN", "build_uci/chess_engine_uci"))
side = chess.WHITE if KV.get("SIDE", "white").lower() == "white" else chess.BLACK
moves_wanted = [int(x) for x in KV["MOVES"].split(",")]
depth = int(KV.get("DEPTH", "20"))
maxsec = float(KV.get("MAXSEC", "120"))
vdepth = int(KV.get("VERIFY_DEPTH", "16"))
sf_nodes = int(KV.get("SFNODES", "8000000"))
variants = [v.split(":", 1) for v in KV.get("VARIANTS", "normal:").split("|")]

game = chess.pgn.read_game(io.StringIO(open(KV["PGN"], encoding="utf-8", errors="ignore").read()))
positions = []   # (move number, board before the move, move played)
b = game.board()
for mv in game.mainline_moves():
	if b.turn == side and b.fullmove_number in moves_wanted:
		positions.append((b.fullmove_number, b.copy(), mv))
	b.push(mv)


def ours(knobs):
	eng = chess.engine.SimpleEngine.popen_uci(BIN)
	opts = {"OwnBook": False}
	if knobs:
		opts["Knobs"] = knobs
	eng.configure(opts)
	return eng


def score_str(sc):
	if sc is None:
		return "?"
	if sc.is_mate():
		return "#%d" % sc.mate()
	return "%+.2f" % (sc.score() / 100.0)


def sf_best(board):
	sf = chess.engine.SimpleEngine.popen_uci(SF)
	try:
		sf.configure({"Threads": 1, "Hash": 256, "UCI_ShowWDL": True})
		info = sf.analyse(board, chess.engine.Limit(nodes=sf_nodes))
		return info["pv"][0], info["wdl"].pov(board.turn).expectation() * 100.0, info["score"].pov(board.turn)
	finally:
		sf.quit()


for n, board, played in positions:
	mark = "." if side == chess.WHITE else "..."
	best, e_sf, s_sf = sf_best(board)
	print("\n== %d%s played %s · SF18 best %s (E %.1f, %s) · FEN %s" % (n, mark, board.san(played), board.san(best), e_sf,
	      score_str(s_sf), board.fen()), flush=True)
	for label, knobs in variants:
		eng = ours(knobs)
		per_depth = {}
		t0 = time.time()
		try:
			with eng.analysis(board, chess.engine.Limit(depth=depth, time=maxsec)) as an:
				for info in an:
					if "pv" in info and info.get("depth") and info["pv"]:
						per_depth[info["depth"]] = info
		finally:
			eng.quit()
		print("  [%s] %s  (%.0f s)" % (label, knobs or "shipped defaults", time.time() - t0), flush=True)
		first = None
		for d in sorted(per_depth):
			info = per_depth[d]
			m = info["pv"][0]
			if m == best and first is None:
				first = d
			elif m != best:
				first = None
			print("    d%-2d %-7s %7s  nodes %11d  %6.2f s%s" % (d, board.san(m), score_str(info["score"].pov(board.turn)),
			      info.get("nodes", 0), info.get("time", 0.0), "  = SF18" if m == best else ("  = game" if m == played else "")),
			      flush=True)
		print("    → SF18's %s %s" % (board.san(best), ("held from depth %d" % first) if first else "not held at the last depth"),
		      flush=True)
	# our own verdict on the candidates (the game's move, SF18's, any EXTRA= moves), first variant's settings, beside SF18's
	cands = [("game", played)] + ([("SF18", best)] if best != played else [])
	for san in [s for s in KV.get("EXTRA", "").split(",") if s]:
		try:
			cands.append(("extra", board.parse_san(san)))
		except ValueError:
			pass
	for label, mv in cands:
		after = board.copy()
		after.push(mv)
		eng = ours(variants[0][1])
		try:
			info = eng.analyse(after, chess.engine.Limit(depth=vdepth, time=maxsec))
		finally:
			eng.quit()
		sf = chess.engine.SimpleEngine.popen_uci(SF)
		try:
			sf.configure({"Threads": 1, "Hash": 256, "UCI_ShowWDL": True})
			e_after = sf.analyse(after, chess.engine.Limit(nodes=sf_nodes))["wdl"].pov(side).expectation() * 100.0
		finally:
			sf.quit()
		print("  verify %-5s %-7s ours d%d after it: %s · SF18 E after it %.1f (graded side's view)" % (label, board.san(mv),
		      info.get("depth", 0), score_str(info["score"].pov(side)), e_after), flush=True)
