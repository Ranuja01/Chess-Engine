# -*- coding: utf-8 -*-
"""Per-theme comparison of two STS runs, from the CSVs sts_test.py already wrote. Zero CPU, no re-run.

A total like "1746 vs 1658" hides where the change acted. STS is 15 themes x 20 positions here, so a gain
concentrated in the themes a knob plausibly touches is very different evidence from one smeared evenly
across all 15 -- the latter looks more like the bench's ~100-point jaggedness than a mechanism.

  pyrun diagnostics/_sts_theme_diff.py              -> list every stored tag with its total
  pyrun diagnostics/_sts_theme_diff.py <tagA> <tagB> -> per-theme A - B
"""
import os
import sys
import csv

THIS = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(THIS, 'results')


def load(tag):
    path = os.path.join(RESULTS, 'sts_results_%s.csv' % tag)
    if not os.path.isfile(path):
        return None
    themes, total = {}, 0
    with open(path, newline='') as fh:
        for row in csv.DictReader(fh):
            if row.get('score') in (None, ''):
                continue
            pts = int(row['score'])
            themes.setdefault(row['theme'], 0)
            themes[row['theme']] += pts
            total += pts
    return themes, total


def main():
    args = sys.argv[1:]
    if len(args) < 2:
        rows = []
        for name in os.listdir(RESULTS):
            if name.startswith('sts_results_') and name.endswith('.csv'):
                tag = name[len('sts_results_'):-4]
                got = load(tag)
                if got:
                    rows.append((got[1], tag))
        for total, tag in sorted(rows, reverse=True):
            print("  %5d  %s" % (total, tag))
        return

    ta, tb = args[0], args[1]
    a, b = load(ta), load(tb)
    if not a or not b:
        print("missing tag: %s" % (ta if not a else tb))
        return
    themes = sorted(set(a[0]) | set(b[0]), key=lambda t: -(a[0].get(t, 0) - b[0].get(t, 0)))
    print("per-theme  %s (%d)  vs  %s (%d)   delta %+d\n" % (ta, a[1], tb, b[1], a[1] - b[1]))
    print("  %-34s %6s %6s %7s" % ("theme", ta[:6], tb[:6], "delta"))
    for th in themes:
        va, vb = a[0].get(th, 0), b[0].get(th, 0)
        flag = '  <<<' if abs(va - vb) >= 20 else ''
        print("  %-34s %6d %6d %+7d%s" % (th, va, vb, va - vb, flag))


if __name__ == '__main__':
    main()
