"""
Per-iteration branching factor and time to depth, for any UCI engine (2026-11-10, search v2 arc instrument 1).

Our printed EBF is pow(cumulative nodes, 1 / exit depth) and is not comparable with anything (memory
ebf-metric-is-not-comparable). This reads the engine's own `info depth D ... nodes N time T` lines instead: the nodes of
iteration d are N(d) - N(d-1), and the branching factor of iteration d is nodes(d) / nodes(d-1). It needs no build and
leaves the engine untouched, so it measures any binary, any knob setting and the reference engines the same way.

    pyrun diagnostics/_ebf_per_iter.py E1=ours=<bin> [O1="OwnBook:false;Knobs:A=1,B=2"] [D1=14]
                                       [E2=sf11=<bin> O2="Threads:1;Hash:16" D2=18] [N=40] [SEED=1]
                                       [FENS=diagnostics/ks_sets/game_regret_set.csv] [OUT=<dir>] [TIMEOUT=300]

  E<i>=name=path   up to 4 engines, run one after another (one process at a time)
  O<i>             UCI options, `name:value` separated by `;` (a value may contain commas, as Knobs does)
  D<i>             `go depth` for that engine (default 14)
  N                positions per phase bucket, sampled with SEED from FENS (a CSV with `fen` and `phase_bucket`)
  OUT              writes per_iter.csv (engine, fen index, phase, depth, cumulative nodes, time ms) and summary.txt

Reads (printed and in summary.txt), per engine:
  - per depth: positions reaching it, the geometric mean of nodes(d) / nodes(d-1), the geometric mean and median of the
    two-ply ratio sqrt(nodes(d) / nodes(d-2)), mean cumulative nodes and time to depth. Both engines alternate odd / even
    iterations strongly (one-ply ratios swing 1.1 / 5.2), so the two-ply ratio is the stable figure;
  - the EBF over the last six depths (geometric mean of the two-ply ratios), overall and per phase.
Caveats: our node counter also counts TT-hit bookkeeping and qsearch, so absolute node counts differ in definition from
SF's; the RATIO is a growth rate and is comparable as long as that share does not drift with depth. A position that
stops before the requested depth (mate found, timeout) contributes only the depths it completed. Aspiration re-searches
and our root pre-search belong to their iteration, as in the literature figure. One `ucinewgame` per position.
"""
import csv
import math
import os
import random
import statistics
import subprocess
import sys
import time

ARGS = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)


def engines_from_args():
	out = []
	for i in range(1, 5):
		spec = ARGS.get(f"E{i}")
		if not spec:
			continue
		name, path = spec.split("=", 1)
		opts = []
		for o in filter(None, ARGS.get(f"O{i}", "").split(";")):
			k, v = o.split(":", 1)
			opts.append((k, v))
		out.append({"name": name, "path": path, "opts": opts, "depth": int(ARGS.get(f"D{i}", "14"))})
	if not out:
		sys.exit("give at least E1=name=path")
	return out


def sample_fens():
	path = ARGS.get("FENS", "diagnostics/ks_sets/game_regret_set.csv")
	n = int(ARGS.get("N", "40"))
	rng = random.Random(int(ARGS.get("SEED", "1")))
	by_phase = {}
	with open(path, newline="") as f:
		for row in csv.DictReader(f):
			by_phase.setdefault(row.get("phase_bucket", "all"), []).append(row["fen"])
	picked = []
	for phase in sorted(by_phase):
		rows = by_phase[phase]
		for fen in rng.sample(rows, min(n, len(rows))):
			picked.append((phase, fen))
	return picked


class Engine:
	def __init__(self, path):
		self.p = subprocess.Popen([path], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
		                          text=True, bufsize=1, cwd=os.getcwd())

	def send(self, s):
		self.p.stdin.write(s + "\n")
		self.p.stdin.flush()

	def wait_for(self, token, timeout):
		t0 = time.time()
		lines = []
		while True:
			line = self.p.stdout.readline()
			if not line:
				raise RuntimeError("engine closed its output")
			lines.append(line.rstrip("\n"))
			if line.startswith(token):
				return lines
			if time.time() - t0 > timeout:
				self.send("stop")
				timeout = 1e9   # wait for the bestmove the stop produces

	def quit(self):
		try:
			self.send("quit")
			self.p.wait(timeout=10)
		except Exception:
			self.p.kill()


def parse_iterations(lines):
	"""depth -> (cumulative nodes, time ms): the last line of each depth that carries nodes (SF prints bound lines
	inside an iteration; the last one closes it)."""
	it = {}
	for line in lines:
		if not line.startswith("info ") or " nodes " not in line or " depth " not in line:
			continue
		tok = line.split()
		try:
			d = int(tok[tok.index("depth") + 1])
			n = int(tok[tok.index("nodes") + 1])
			t = int(tok[tok.index("time") + 1]) if "time" in tok else -1
		except (ValueError, IndexError):
			continue
		it[d] = (n, t)
	return it


def run_engine(eng, fens, timeout, rows):
	e = Engine(eng["path"])
	e.send("uci")
	e.wait_for("uciok", 30)
	for k, v in eng["opts"]:
		e.send(f"setoption name {k} value {v}")
	e.send("isready")
	e.wait_for("readyok", 60)
	for idx, (phase, fen) in enumerate(fens):
		e.send("ucinewgame")
		e.send("isready")
		e.wait_for("readyok", 60)
		e.send(f"position fen {fen}")
		e.send(f"go depth {eng['depth']}")
		lines = e.wait_for("bestmove", timeout)
		for d, (n, t) in sorted(parse_iterations(lines).items()):
			rows.append((eng["name"], idx, phase, d, n, t))
	e.quit()


def gmean(xs):
	return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")


def summarise(name, rows, out):
	per_pos = {}
	for eng, idx, phase, d, n, t in rows:
		if eng == name:
			per_pos.setdefault((idx, phase), {})[d] = (n, t)
	ratios = {}       # depth -> list of nodes(d) / nodes(d-1)
	ratios2 = {}      # depth -> list of sqrt(nodes(d) / nodes(d-2)): cancels the odd / even alternation
	cum = {}          # depth -> list of (cumulative nodes, time)
	ratio_rows = []   # (phase, depth, two-ply ratio)
	for (idx, phase), it in per_pos.items():
		ds = sorted(it)
		for d in ds:
			cum.setdefault(d, []).append(it[d])
		own = {d: it[d][0] - it[d - 1][0] for d in ds if d - 1 in it}
		for d in own:
			if d - 1 in own and own[d] > 0 and own[d - 1] > 0:
				ratios.setdefault(d, []).append(own[d] / own[d - 1])
			if d - 2 in own and own[d] > 0 and own[d - 2] > 0:
				r2 = math.sqrt(own[d] / own[d - 2])
				ratios2.setdefault(d, []).append(r2)
				ratio_rows.append((phase, d, r2))
	out.append(f"== {name}")
	out.append("depth  positions  1-ply gmean  2-ply gmean  2-ply median  mean cum nodes  mean time ms")
	for d in sorted(cum):
		r, r2 = ratios.get(d, []), ratios2.get(d, [])
		nodes = statistics.mean(x[0] for x in cum[d])
		tms = [x[1] for x in cum[d] if x[1] >= 0]
		out.append(f"{d:5d}  {len(cum[d]):9d}  {gmean(r):11.3f}  {gmean(r2):11.3f}  "
		           f"{statistics.median(r2) if r2 else float('nan'):12.3f}  {nodes:14.0f}  "
		           f"{statistics.mean(tms) if tms else float('nan'):12.1f}")
	if ratios2:
		top = max(ratios2)
		lo = max(min(ratios2), top - 5)
		window = [x for d in ratios2 if lo <= d <= top for x in ratios2[d]]
		out.append(f"EBF (2-ply) over depths {lo}-{top}: {gmean(window):.3f}")
		for phase in sorted({p for p, _, _ in ratio_rows}):
			xs = [x for p, d, x in ratio_rows if p == phase and lo <= d <= top]
			out.append(f"  {phase:12s} {gmean(xs):.3f}  ({len(xs)} iterations)")
	out.append("")


def main():
	engines = engines_from_args()
	fens = sample_fens()
	timeout = float(ARGS.get("TIMEOUT", "300"))
	rows = []
	for eng in engines:
		t0 = time.time()
		run_engine(eng, fens, timeout, rows)
		print(f"[ebf] {eng['name']}: {len(fens)} positions to depth {eng['depth']} in {time.time() - t0:.0f} s", flush=True)
	out = [f"positions: {len(fens)} ({ARGS.get('N', '40')} per phase, seed {ARGS.get('SEED', '1')})", ""]
	for eng in engines:
		summarise(eng["name"], rows, out)
	print("\n".join(out))
	od = ARGS.get("OUT")
	if od:
		os.makedirs(od, exist_ok=True)
		with open(os.path.join(od, "per_iter.csv"), "w", newline="") as f:
			w = csv.writer(f)
			w.writerow(["engine", "pos", "phase", "depth", "cum_nodes", "time_ms"])
			w.writerows(rows)
		with open(os.path.join(od, "summary.txt"), "w") as f:
			f.write("\n".join(out) + "\n")


if __name__ == "__main__":
	main()
