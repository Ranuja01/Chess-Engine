# -*- coding: utf-8 -*-
"""Search v2 watch SW3 (plan §7e): the real-play speed read of queue #207. Reads the per_iter.csv files that
_ebf_per_iter.py wrote for each repetition and arm (DIR/speed_r<rep>_arm<arm>/per_iter.csv) and reports, per arm and per
repetition, the mean time to the deepest common depth over the positions both arms reached, and nodes per second; then
the arm 1 − arm 0 difference with its spread across repetitions, and whether the trees were the same (equal cumulative nodes
per position and depth: they must be, S1 is node-for-node identical).

  pyrun diagnostics/_sv2_speed_read.py DIR=/mnt/e/chess_data/sv2_sw3 [DEPTH=13]

REGISTERED (queue #207 header): arm 1 within +1% of arm 0 on mean time to depth 13 = "not slower".
"""
import csv, glob, os, re, statistics, sys
KV = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
D = KV["DIR"]
DEPTH = int(KV.get("DEPTH", 13))
runs = {}   # (rep, arm) -> {(pos, depth): (nodes, ms)}
for path in glob.glob(os.path.join(D, "speed_r*_arm*", "per_iter.csv")):
	m = re.search(r"speed_r(\d+)_arm(\d+)", path)
	key = (int(m.group(1)), int(m.group(2)))
	runs[key] = {(int(r["pos"]), int(r["depth"])): (int(r["cum_nodes"]), int(r["time_ms"])) for r in csv.DictReader(open(path, newline=""))}
if not runs:
	sys.exit("no per_iter.csv under %s" % D)
reps = sorted({r for r, _ in runs})
common = None
for v in runs.values():
	ps = {p for (p, d) in v if d == DEPTH}
	common = ps if common is None else common & ps
common = sorted(common)
print("positions reaching depth %d in every run: %d" % (DEPTH, len(common)))
same_tree = all(runs[(r, 0)].get((p, d), (None,))[0] == runs[(r, 1)].get((p, d), (None,))[0]
                for r in reps if (r, 0) in runs and (r, 1) in runs for (p, d) in runs[(r, 0)])
print("same tree (cumulative nodes equal per position and depth, every repetition): %s" % same_tree)
diffs = []
for r in reps:
	line = []
	t = {}
	for arm in (0, 1):
		if (r, arm) not in runs:
			continue
		v = runs[(r, arm)]
		ms = [v[(p, DEPTH)][1] for p in common]
		nodes = sum(v[(p, DEPTH)][0] for p in common)
		t[arm] = statistics.mean(ms)
		line.append("arm %d: mean %.1f ms to d%d, %.0f knps" % (arm, t[arm], DEPTH, nodes / max(1, sum(ms))))
	if 0 in t and 1 in t:
		diffs.append(100.0 * (t[1] - t[0]) / t[0])
		line.append("arm1 - arm0 %+.2f%%" % diffs[-1])
	print("rep %d: %s" % (r, " · ".join(line)))
if diffs:
	mean = statistics.mean(diffs)
	sd = statistics.stdev(diffs) if len(diffs) > 1 else float("nan")
	print("arm 1 - arm 0, mean over %d repetitions: %+.2f%% (sd %.2f)" % (len(diffs), mean, sd))
	print("REGISTERED (#207): %s" % ("NOT SLOWER (within +1%)" if mean <= 1.0 else "SLOWER (more than +1%)"))
