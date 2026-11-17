# -*- coding: utf-8 -*-
"""Paired STS comparison of two runs on the same suite: totals, positions better / worse, and the sign-test z.

Reads two `diagnostics/results/sts_results_<tag>.csv` files (written by sts_test.py) and pairs them by position index.
The sign test (better - worse) / sqrt(better + worse) is the per-position reading the C-series screens use (#245's rule:
an arm is a real STS cost when its total is below the base by more than the suite's noise AND |z| > 2).
Also prints the share of positions whose chosen move differs, so a run meant to reproduce another can be checked exactly.

Run: bash <runner> pyrun diagnostics/_sts_paired.py <tag_or_csv_A> <tag_or_csv_B>   (B is the base)
"""
import csv, math, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))


def load(arg):
    path = arg if arg.endswith('.csv') else os.path.join(HERE, 'results', 'sts_results_%s.csv' % arg)
    rows = {}
    with open(path, newline='') as f:
        for r in csv.DictReader(f):
            rows[int(r['idx'])] = (int(r['score']), r['engine'])
    return path, rows


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    pa, a = load(sys.argv[1])
    pb, b = load(sys.argv[2])
    common = sorted(set(a) & set(b))
    if not common:
        print("no shared positions")
        sys.exit(1)
    sa = sum(a[i][0] for i in common)
    sb = sum(b[i][0] for i in common)
    better = sum(1 for i in common if a[i][0] > b[i][0])
    worse = sum(1 for i in common if a[i][0] < b[i][0])
    moved = sum(1 for i in common if a[i][1] != b[i][1])
    z = (better - worse) / math.sqrt(better + worse) if better + worse else 0.0
    print("A %s\nB %s" % (pa, pb))
    print("positions %d (A only %d, B only %d) · A %d vs B %d = %+d · better %d / worse %d · sign z %+.2f · move differs %d (%.1f%%)"
          % (len(common), len(set(a) - set(b)), len(set(b) - set(a)), sa, sb, sa - sb, better, worse, z, moved,
             100.0 * moved / len(common)))


if __name__ == '__main__':
    main()
