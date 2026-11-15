"""
Rank a cg_annotate report by function, summed over the source files the code was inlined from.

cg_annotate lists one row per file:function, so a function inlined from several headers (with -flto, most hot ones) is split
across rows. This sums those rows per function name and prints instructions, D1 misses, last-level misses and branch
mispredictions (conditional + indirect), so the speed items can be ranked on all three.

Usage:  python diagnostics/_cg_by_function.py <cg_annotate.txt> [TOP=40]
Needs a report made with --cache-sim=yes --branch-sim=yes (13 event columns).

@author: Ranuja Pinnaduwage
"""
import collections
import re
import sys

EVENTS = ['Ir', 'I1mr', 'ILmr', 'Dr', 'D1mr', 'DLmr', 'Dw', 'D1mw', 'DLmw', 'Bc', 'Bcm', 'Bi', 'Bim']


def main():
	path = sys.argv[1]
	top = 40
	for a in sys.argv[2:]:
		if a.startswith('TOP='):
			top = int(a[4:])
	lines = open(path, encoding='utf-8', errors='replace').read().split('\n')
	headers = [i for i, l in enumerate(lines) if l.startswith('Ir ') and 'file:function' in l]
	if not headers:
		sys.exit('no per-function table (was the report made with cache and branch simulation?)')
	start = headers[0]
	name_col = lines[start].index('file:function')
	totals = collections.defaultdict(collections.Counter)
	for line in lines[start + 2:]:
		if line.startswith('--'):
			break
		if not line.strip():
			continue
		values = [int(v.replace(',', '')) for v in re.findall(r'([\d,]+)(?: \(\s*[\d.]+%\))?', line[:name_col])]
		if len(values) < len(EVENTS):
			continue
		name = line[name_col:].strip()
		name = name.split(':', 1)[1] if ':' in name else name
		# keep the name up to its argument list: overloads and clones of one function are summed together
		name = name.replace('(anonymous namespace)::', 'anon::').split('(')[0].strip()
		for event, value in zip(EVENTS, values):
			totals[name][event] += value
	# the program total is the first row reading 100.0% (the per-function rows stop at cg_annotate's 0.1% threshold)
	summary = next((l for l in lines if '(100.0%)' in l), None)
	all_ir = int(summary.split()[0].replace(',', '')) if summary else sum(c['Ir'] for c in totals.values())
	print(f"{'instr':>6}  {'D1 miss':>11}  {'LL miss':>10}  {'mispred':>11}  function   (total {all_ir:,} instructions)")
	for name, c in sorted(totals.items(), key=lambda kv: -kv[1]['Ir'])[:top]:
		print(f"{100 * c['Ir'] / all_ir:5.2f}%  {c['D1mr'] + c['D1mw']:>11,}  {c['DLmr'] + c['DLmw']:>10,}  "
		      f"{c['Bcm'] + c['Bim']:>11,}  {name[:120]}")


if __name__ == '__main__':
	main()
