# -*- coding: utf-8 -*-
"""Tolerant raw-pipe UCI driver for legacy engines (e.g. Mediocre v0.5).

Some old UCI engines re-emit `id`/`uciok` lines outside a handshake (e.g. on ucinewgame/isready).
python-chess's `SimpleEngine` asserts on those "unexpected" lines -> the protocol desyncs -> stale
opponent moves get pushed to our side -> bogus results (Mediocre "won" ~90% purely as a desync
artifact, conc1 and conc2 alike). This driver speaks minimal UCI over a pipe and IGNORES any line
that is not the specific token it is waiting for, so stray id/uciok/option lines are harmless.

Duck-types the subset of `chess.engine.SimpleEngine` that selfplay/vs_sf.py uses:
`.options` (membership only), `.configure(dict)`, `.play(board, limit, info=...) -> result with
`.move` + `.info`, `.quit()`. One instance per game (matches vs_sf's per-game SF handle model);
a single instance is NOT reentrant across threads.
"""
import subprocess
import select
import time

import chess


class _PlayResult:
	"""Minimal stand-in for chess.engine.PlayResult (only .move + .info are read downstream)."""
	__slots__ = ("move", "info")

	def __init__(self, move, info):
		self.move = move
		self.info = info


class RawUciEngine:
	def __init__(self, path, init_timeout=20.0):
		self.path = path
		self.init_timeout = init_timeout
		self.options = {}          # option name -> True (vs_sf only does membership tests)
		self.proc = None
		self._spawn()

	def _spawn(self):
		"""Launch a fresh engine process + handshake. Mediocre v0.5 nondeterministically DEADLOCKS on a
		persistent process after many searches (ignores `stop`), so play() kills+respawns on a hang; a
		fresh process handles any single position reliably."""
		self.proc = subprocess.Popen(
			[self.path], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
			stderr=subprocess.DEVNULL, text=True, bufsize=1)
		self.options = {}
		self._handshake(self.init_timeout)

	def _kill(self):
		try: self.proc.kill()
		except Exception: pass
		try: self.proc.wait(timeout=5)
		except Exception: pass

	# --- low-level pipe I/O -------------------------------------------------
	def _send(self, line):
		self.proc.stdin.write(line + "\n")
		self.proc.stdin.flush()

	def _readline(self, timeout):
		"""One line within `timeout`s. Returns the text (blank line -> ''), or None on timeout/EOF."""
		fd = self.proc.stdout
		remaining = timeout
		if remaining <= 0:
			return None
		r, _, _ = select.select([fd], [], [], remaining)
		if not r:
			return None                        # timeout
		raw = fd.readline()
		if raw == "":
			return None                        # EOF
		return raw.rstrip("\n")

	def _read_until(self, token, timeout, collect=None):
		"""Read until a line whose first word == token; feed every line to collect(). Tolerant: all
		other lines (id/uciok/option/blank/info) are ignored. Returns the matching line, '' on
		timeout or engine death."""
		end = time.monotonic() + timeout
		while True:
			remaining = end - time.monotonic()
			if remaining <= 0:
				return ""
			line = self._readline(remaining)
			if line is None:
				if self.proc.poll() is not None:   # engine exited
					return ""
				continue                            # select slice elapsed; outer clock guards
			if collect is not None:
				collect(line)
			if line.split(" ", 1)[0] == token:
				return line

	# --- UCI lifecycle ------------------------------------------------------
	def _handshake(self, timeout):
		def collect(line):
			if line.startswith("option name "):
				name = line[len("option name "):].split(" type ", 1)[0].strip()
				if name:
					self.options[name] = True
		self._send("uci")
		self._read_until("uciok", timeout, collect)
		self._send("isready")
		self._read_until("readyok", timeout)
		# One instance == one game (vs_sf opens a fresh handle per game). Signal the new game ONCE here;
		# sending ucinewgame per-move is wrong (it re-triggers Mediocre's id/uciok re-emission and can
		# leave it without a move). Drain its readyok, tolerating any stray lines.
		self._send("ucinewgame")
		self._send("isready")
		self._read_until("readyok", timeout)

	def configure(self, opts):
		for k, v in (opts or {}).items():
			if isinstance(v, bool):
				v = "true" if v else "false"
			self._send(f"setoption name {k} value {v}")
		self._send("isready")
		self._read_until("readyok", 10.0)

	def _ask(self, board, limit):
		"""One position -> bestmove line on the current process, or '' on hang/EOF. Also returns the
		parsed (depth, nodes) info seen."""
		info = {"depth": None, "nodes": None}

		def collect(line):
			if line.startswith("info "):
				toks = line.split()
				for i, t in enumerate(toks):
					if t == "depth" and i + 1 < len(toks):
						try: info["depth"] = int(toks[i + 1])
						except ValueError: pass
					elif t == "nodes" and i + 1 < len(toks):
						try: info["nodes"] = int(toks[i + 1])
						except ValueError: pass

		self._send("position fen " + board.fen())
		self._send(self._go_cmd(limit))
		line = self._read_until("bestmove", self._go_timeout(limit), collect)
		return line, info

	def play(self, board, limit, info=None):
		line, meta = self._ask(board, limit)
		if not line:
			# Hang or death: kill this process and try once on a fresh one (Mediocre's deadlock is
			# cumulative/nondeterministic; a fresh process reliably answers any single position).
			self._kill()
			try:
				self._spawn()
				line, meta = self._ask(board, limit)
			except Exception:
				line = ""
		if not line:
			return _PlayResult(None, {})
		parts = line.split()
		mv = None
		if len(parts) >= 2 and parts[1] not in ("(none)", "0000"):
			try: mv = chess.Move.from_uci(parts[1])
			except ValueError: mv = None
		return _PlayResult(mv, meta)

	# --- limit translation --------------------------------------------------
	def _go_cmd(self, limit):
		if getattr(limit, "nodes", None):
			return f"go nodes {int(limit.nodes)}"
		if getattr(limit, "depth", None):
			return f"go depth {int(limit.depth)}"
		t = getattr(limit, "time", None) or 1.0
		return f"go movetime {int(t * 1000)}"

	def _go_timeout(self, limit):
		"""Hang-detection ceiling. Tight vs the movetime budget so a deadlock triggers a fast respawn
		(a real answer arrives well within this); a false cut just re-asks on a fresh process, so being
		aggressive is safe."""
		t = getattr(limit, "time", None)
		if t:
			return max(4.0, t * 4 + 2.0)
		return 30.0

	def quit(self):
		try: self._send("quit")
		except Exception: pass
		try: self.proc.stdin.close()
		except Exception: pass
		try:
			self.proc.wait(timeout=5)
		except Exception:
			try: self.proc.kill()
			except Exception: pass
