"""Split an arm's effect on a decided-by-the-50-move-rule venue into conversion and defence.

A head-to-head arm-vs-control match reads "converts more when ahead" and "holds more when behind" as one number. This
separates them with a reference run on the same starts in which the control plays both sides:
  conversion = attacker score when the ARM attacks the control  -  attacker score when the control attacks itself
  defence    = attacker score when the control attacks the ARM  -  attacker score when the control attacks itself
(attacker = the engine playing the side the venue labels as ahead; a positive conversion and a negative defence both
favour the arm). Both are averaged per position and their errors are taken across positions (clustered by position).

Usage: python3 diagnostics/_r50_split.py VENUE=<venue.csv> A='<arm-vs-ctl pgn glob>' B='<ctl-vs-ctl pgn glob>' [ARM=arm]
"""
import csv, glob, math, re, sys
from collections import defaultdict

args = dict(a.split('=', 1) for a in sys.argv[1:] if '=' in a)
ARM = args.get('ARM', 'arm')
ahead = {}
for r in csv.DictReader(open(args['VENUE'])):
    ahead[' '.join(r['fen'].split()[:6])] = r['ahead']

def games(pattern):
    for path in sorted(glob.glob(pattern)):
        for g in open(path, errors='replace').read().split('[Event ')[1:]:
            tag = lambda t: (re.search(r'\[%s "([^"]*)"\]' % t, g) or [None, None])[1]
            fen = tag('FEN')
            if fen is None:
                continue
            fen = ' '.join(fen.split()[:6])
            if fen not in ahead:
                continue
            res = tag('Result')
            pts_white = {'1-0': 1.0, '0-1': 0.0, '1/2-1/2': 0.5}.get(res)
            if pts_white is None:
                continue
            side = ahead[fen]
            attacker = tag('White') if side == 'w' else tag('Black')
            att_pts = pts_white if side == 'w' else 1.0 - pts_white
            fifty = 'Draw by fifty moves rule' in g
            yield fen, attacker, att_pts, fifty

conv, dfn, ref = defaultdict(list), defaultdict(list), defaultdict(list)
cf, df, rf = [0, 0], [0, 0], [0, 0]   # [fifty-move draws, games]
for fen, att, pts, fifty in games(args['A']):
    (conv if att == ARM else dfn)[fen].append(pts)
    c = cf if att == ARM else df
    c[0] += fifty; c[1] += 1
for fen, att, pts, fifty in games(args['B']):
    ref[fen].append(pts)
    rf[0] += fifty; rf[1] += 1

def mean(x):
    return sum(x) / len(x)

def paired(side):
    d = [mean(side[f]) - mean(ref[f]) for f in side if f in ref]
    n = len(d)
    if n < 2:
        return n, (d[0] if d else 0.0), 0.0
    m = mean(d)
    se = math.sqrt(sum((x - m) ** 2 for x in d) / (n - 1) / n)
    return n, m, se

def rate(side):
    allp = [p for f in side for p in side[f]]
    return (mean(allp) if allp else float('nan')), len(allp)

for name, side, c in (('arm attacks ctl', conv, cf), ('ctl attacks arm', dfn, df), ('ctl attacks ctl (reference)', ref, rf)):
    s, n = rate(side)
    print(f'{name:30s} games {n:4d}  attacker score {100 * s:5.1f}%  fifty-move draws {c[0]:4d} ({100 * c[0] / max(c[1], 1):4.1f}%)')
for name, side, good in (('conversion (arm attacking)', conv, +1), ('defence (arm defending)', dfn, -1)):
    n, m, se = paired(side)
    z = m / se if se > 0 else 0.0
    print(f'{name:30s} positions {n:4d}  delta {100 * m:+5.1f} pp  +- {196 * se:4.1f} (95%)  z {z:+.2f}  '
          f'({"favours the arm" if m * good > 0 else "against the arm" if m * good < 0 else "level"})')
