"""Bare-king conversion probe: does the engine mate a lone king inside the 50-move rule?

Question (owner 11-10): "certain mate types (like bishop and knight) are unknown by our eval". v2 has no mop-up
technique at all (no king-proximity or push-to-edge term; tier2b covers only K+R vs K+minor and is off), v1 has a
generic one (advanced_endgame_eval: kings closer x200, defending king to ANY edge x45, when the lead > 2 pawns)
with no bishop-corner rule. This measures whether either converts, against Stockfish 11 (which has the SF KXK /
KBNK handlers) as the control.

Classes: KQK, KRK, KBBK (bishops on opposite colours), KBNK. Positions: random legal (`_draw_oracle.random_position`),
White = the strong side to move, Lichess tablebase WIN only (cached in ks_sets/tablebase_labels_mopup.json), DTM kept.
Games: each position once per attacker arm, the attacker White against Stockfish 18 defending at DEF_NODES per move,
both node-limited (load-independent, so safe beside other work). Ends on mate, stalemate, a captured piece
(insufficient material), threefold repetition, or 100 plies without a capture (the 50-move rule).

Modes:
  MODE=gen     N=30 SEED=1111      write ks_sets/mopup_positions.csv (cls, fen, dtm moves)
  MODE=play    ARMS=name:engine:nodes,...  OUT=dir  THREADS=3  DEF_NODES=200000   -> OUT/games.csv
  MODE=report  OUT=dir             conversion per class per arm, mean moves to mate against DTM
Engines: ours (production UCI binary, OwnBook off), ours_v1 (the same binary, EvalArm 0), sf11, sf18.
"""
import csv
import os
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

for _a in sys.argv[1:]:                     # the runner's pyrun passes KEY=VALUE as arguments, not environment
	if "=" in _a:
		_k, _v = _a.split("=", 1)
		os.environ[_k] = _v

THIS = os.path.dirname(os.path.abspath(__file__))
ND = os.path.dirname(THIS)
os.environ.setdefault("CACHE", os.path.join(THIS, "ks_sets", "tablebase_labels_mopup.json"))
sys.path.insert(0, THIS)

import chess
import chess.engine

MODE = os.environ.get("MODE", "report")
N = int(os.environ.get("N", "30"))
SEED = int(os.environ.get("SEED", "1111"))
POS = os.environ.get("POS", os.path.join(THIS, "ks_sets", "mopup_positions.csv"))
OUT = os.environ.get("OUT", "/mnt/e/chess_data/mopup")
THREADS = int(os.environ.get("THREADS", "3"))
DEF_NODES = int(os.environ.get("DEF_NODES", "200000"))
ARMS = os.environ.get("ARMS", "v2_1m:ours:1000000")
LIMIT = int(os.environ.get("LIMIT", "0"))          # play only the first LIMIT positions (smoke tests)

ENGINES = {
	"ours":    (os.path.join(ND, "build_uci", "chess_engine_uci"), {"OwnBook": False}),
	"ours_v1": (os.path.join(ND, "build_uci", "chess_engine_uci"), {"OwnBook": False, "EvalArm": 0}),
	"sf11":    ("/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/stockfish_11_linux/stockfish-11-linux/"
	            "Linux/stockfish_20011801_x64_bmi2", {"Threads": 1, "Hash": 16}),
	"sf18":    ("/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/stockfish_18_linux/"
	            "stockfish-ubuntu-x86-64-avx2", {"Threads": 1, "Hash": 16}),
}

CLASSES = {
	"KQK":  [chess.QUEEN],
	"KRK":  [chess.ROOK],
	"KBBK": [chess.BISHOP, chess.BISHOP],
	"KBNK": [chess.BISHOP, chess.KNIGHT],
}


def gen():
	import _draw_oracle as dor
	rng = random.Random(SEED)
	rows = []
	for cls, pieces in CLASSES.items():
		got, tries = 0, 0
		while got < N and tries < 50 * N:
			tries += 1
			bd = dor.random_position(pieces, [], rng)
			if bd is None:
				continue
			bd.turn = chess.WHITE
			if not bd.is_valid() or bd.is_game_over():
				continue
			if cls == "KBBK":
				sq = list(bd.pieces(chess.BISHOP, chess.WHITE))
				if (chess.square_file(sq[0]) + chess.square_rank(sq[0])) % 2 == (chess.square_file(sq[1]) + chess.square_rank(sq[1])) % 2:
					continue
			fen = bd.fen()
			cat, dtm = dor.tb_lookup(fen)
			if cat != "win":
				continue
			rows.append((cls, fen, (dtm + 1) // 2 if dtm else ""))
			got += 1
		print("%s: %d positions (%d tries)" % (cls, got, tries), flush=True)
	dor.save_cache()
	with open(POS, "w", newline="") as f:
		w = csv.writer(f)
		w.writerow(["cls", "fen", "dtm_moves"])
		w.writerows(rows)
	print("wrote %s (%d rows, %d tablebase queries)" % (POS, len(rows), dor._net[0]))


_local = threading.local()
_all = []                                   # every engine opened, so play() can quit them all


def engine(name):
	cache = getattr(_local, "eng", None)
	if cache is None:
		cache = _local.eng = {}
	if name not in cache:
		path, opts = ENGINES[name]
		e = chess.engine.SimpleEngine.popen_uci(path, cwd=ND)
		e.configure(opts)
		cache[name] = e
		_all.append(e)
	return cache[name]


def play_one(job):
	arm, eng, nodes, idx, cls, fen, dtm = job
	bd = chess.Board(fen)
	att, dfd = engine(eng), engine("sf18")
	gid = "%s-%d" % (arm, idx)
	result, t0 = None, time.time()
	while result is None:
		if bd.is_checkmate():
			result = "mate"
		elif bd.is_stalemate():
			result = "stalemate"
		elif bd.is_insufficient_material():
			result = "capture"
		elif bd.is_repetition(3):
			result = "repetition"
		elif bd.halfmove_clock >= 100:
			result = "fifty"
		else:
			mover = att if bd.turn == chess.WHITE else dfd
			lim = chess.engine.Limit(nodes=nodes if bd.turn == chess.WHITE else DEF_NODES)
			r = mover.play(bd, lim, game=gid)
			if r.move is None or r.move not in bd.legal_moves:
				result = "nomove"
				break
			bd.push(r.move)
	moves = (len(bd.move_stack) + 1) // 2
	bk = bd.king(chess.BLACK)
	return [arm, idx, cls, dtm, result, moves, chess.square_name(bk), round(time.time() - t0, 1), fen]


def play():
	with open(POS) as f:
		pos = list(csv.DictReader(f))
	if LIMIT:
		by = {}
		for p in pos:
			by.setdefault(p["cls"], []).append(p)
		pos = [p for v in by.values() for p in v[:LIMIT]]
	jobs = []
	for spec in ARMS.split(","):
		arm, eng, nodes = spec.split(":")
		for i, p in enumerate(pos):
			jobs.append((arm, eng, int(nodes), i, p["cls"], p["fen"], p["dtm_moves"]))
	os.makedirs(OUT, exist_ok=True)
	path = os.path.join(OUT, "games.csv")
	new = not os.path.exists(path)
	done = set()
	if not new:
		with open(path) as f:
			for r in csv.DictReader(f):
				done.add((r["arm"], int(r["idx"])))
	jobs = [j for j in jobs if (j[0], j[3]) not in done]
	print("%d games to play (%d already in %s)" % (len(jobs), len(done), path), flush=True)
	lock = threading.Lock()
	with open(path, "a", newline="") as f:
		w = csv.writer(f)
		if new:
			w.writerow(["arm", "idx", "cls", "dtm_moves", "result", "moves", "def_king", "secs", "fen"])
		with ThreadPoolExecutor(THREADS) as ex:
			for n, row in enumerate(ex.map(play_one, jobs), 1):
				with lock:
					w.writerow(row)
					f.flush()
				if n % 20 == 0:
					print("  %d / %d  last: %s" % (n, len(jobs), row[:7]), flush=True)
	for e in _all:
		try:
			e.quit()
		except Exception:
			pass
	report()


def report():
	path = os.path.join(OUT, "games.csv")
	with open(path) as f:
		rows = list(csv.DictReader(f))
	arms = []
	for r in rows:
		if r["arm"] not in arms:
			arms.append(r["arm"])
	print("\nconversion (mates / games) and mean moves to mate vs tablebase DTM, defender SF18 at %d nodes" % DEF_NODES)
	print("%-10s" % "arm" + "".join("%-24s" % c for c in CLASSES))
	for a in arms:
		line = "%-10s" % a
		for c in CLASSES:
			g = [r for r in rows if r["arm"] == a and r["cls"] == c]
			m = [r for r in g if r["result"] == "mate"]
			if not g:
				line += "%-24s" % "-"
				continue
			extra = ""
			if m:
				mv = sum(int(r["moves"]) for r in m) / len(m)
				dt = sum(int(r["dtm_moves"]) for r in m if r["dtm_moves"]) / max(1, sum(1 for r in m if r["dtm_moves"]))
				extra = " %.0f/%.0f" % (mv, dt)
			line += "%-24s" % ("%d/%d%s" % (len(m), len(g), extra))
		print(line)
	print("\nnon-mate outcomes:")
	for a in arms:
		bad = {}
		for r in rows:
			if r["arm"] == a and r["result"] != "mate":
				bad[(r["cls"], r["result"])] = bad.get((r["cls"], r["result"]), 0) + 1
		if bad:
			print("  %-10s %s" % (a, ", ".join("%s %s %d" % (k[0], k[1], v) for k, v in sorted(bad.items()))))


if __name__ == "__main__":
	{"gen": gen, "play": play, "report": report}[MODE]()
	if MODE == "play":
		os._exit(0)
