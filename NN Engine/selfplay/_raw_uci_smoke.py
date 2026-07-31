# -*- coding: utf-8 -*-
"""Smoke test for the tolerant RawUciEngine driver against Mediocre. Isolates the driver from the
full match loop: handshake + option parse + several play() calls (exercises ucinewgame re-emission
tolerance). Prints SMOKE_OK on success. Temp diagnostic."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import chess
import chess.engine
from raw_uci import RawUciEngine

WRAP = "/home/ranuja/mediocre_uci.sh"

eng = RawUciEngine(WRAP)
print("OPTIONS:", sorted(eng.options.keys()), flush=True)
board = chess.Board()
ok = True
for i in range(5):
	r = eng.play(board, chess.engine.Limit(time=0.4))
	print(f"move {i}: {r.move} depth={r.info.get('depth')} nodes={r.info.get('nodes')}", flush=True)
	if r.move is None or r.move not in board.legal_moves:
		print("FAIL: no/illegal move", flush=True)
		ok = False
		break
	board.push(r.move)
	if board.is_game_over():
		break
	board.push(next(iter(board.legal_moves)))   # trivial reply so the position advances
eng.quit()
print("SMOKE_OK" if ok else "SMOKE_FAIL", flush=True)
