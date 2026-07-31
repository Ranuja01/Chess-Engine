# -*- coding: utf-8 -*-
"""Replay a whole game's positions through ONE persistent Mediocre process (reproduce the in-game
no-move). stderr merged into stdout so a JVM crash trace is visible. Temp diagnostic.
Usage: _raw_uci_probe.py <game_jsonl> [newgame_each]"""
import subprocess, select, time, sys, json

WRAP = "/home/ranuja/mediocre_uci.sh"
jsonl = sys.argv[1]
newgame_each = len(sys.argv) > 2 and sys.argv[2] == "newgame_each"

p = subprocess.Popen([WRAP], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                     stderr=subprocess.STDOUT, text=True, bufsize=1)

def send(s):
	p.stdin.write(s + "\n"); p.stdin.flush()

def read_until(token, secs):
	end = time.monotonic() + secs
	while True:
		rem = end - time.monotonic()
		if rem <= 0: return None
		r, _, _ = select.select([p.stdout], [], [], rem)
		if not r: continue
		line = p.stdout.readline()
		if line == "":
			return "EOF"
		line = line.rstrip("\n")
		if line.split(" ", 1)[0] == token:
			return line

send("uci"); read_until("uciok", 3)
send("isready"); read_until("readyok", 3)
send("ucinewgame"); send("isready"); read_until("readyok", 3)

fens = [json.loads(l)["fen"] for l in open(jsonl)]
print(f"replaying {len(fens)} positions, newgame_each={newgame_each}", flush=True)
for i, fen in enumerate(fens):
	if newgame_each:
		send("ucinewgame"); send("isready")
		if read_until("readyok", 3) == "EOF":
			print(f"[{i}] EOF on readyok  fen={fen}", flush=True); break
	send("position fen " + fen)
	send("go movetime 300")
	bm = read_until("bestmove", 2)
	if bm is None:                       # hang: try the standard UCI stop remedy
		print(f"[{i}] hang, sending stop  fen={fen}  proc_exit={p.poll()}", flush=True)
		send("stop")
		bm = read_until("bestmove", 5)
		print(f"[{i}] after stop -> {bm}", flush=True)
		if bm is None or bm == "EOF":
			print(f"[{i}] STOP DID NOT RESCUE  proc_exit={p.poll()}", flush=True)
			break
	elif bm == "EOF":
		print(f"[{i}] EOF/CRASH  fen={fen}  proc_exit={p.poll()}", flush=True)
		break
	if i % 10 == 0:
		print(f"[{i}] {bm}", flush=True)
else:
	print("REPLAY_OK (no failure)", flush=True)
try: send("quit")
except Exception: pass
