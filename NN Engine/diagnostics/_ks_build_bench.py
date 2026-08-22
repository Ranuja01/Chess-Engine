# -*- coding: utf-8 -*-
"""Curate the KS-archetype bench from mined candidates + the empirical AWAY over-reads.
Balance-samples danger (A*) by highest |SF KS|, counters (B*) by lowest |SF KS| (+ filters B2 endgames),
folds in the collateral AWAY positions (proven over-reads) as B_overread. Writes ks_sets/ks_archetypes.csv.
Pure file processing (no engine). Run: bash <runner> pyrun diagnostics/_ks_build_bench.py
"""
import os, csv, chess
THIS_DIR = os.path.dirname(os.path.abspath(__file__)); ENGINE_DIR = os.path.dirname(THIS_DIR)

def is_middlegame(fen):
    """Counter positions must be genuine middlegames, not endgames / lone-check simplifications where KS
    shouldn't run at all (those leak in from the general regret set and fake over-production)."""
    try:
        b = chess.Board(fen)
    except Exception:
        return False
    npm = sum(len(b.pieces(pt, c)) for pt in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN)
              for c in (chess.WHITE, chess.BLACK))
    # require enough heavy/minor material on BOTH sides (a real middlegame, both kings could be attacked)
    each = [sum(len(b.pieces(pt, c)) for pt in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN)) for c in (chess.WHITE, chess.BLACK)]
    return npm >= 8 and min(each) >= 3
CAND = os.path.join(THIS_DIR, '_ks_arch_candidates.tsv')
COLL = os.path.join(THIS_DIR, '_ks_collateral.out')
OUT = os.path.join(THIS_DIR, 'ks_sets', 'ks_archetypes.csv')

CAPS = {'A1_coord': 16, 'A2_openstorm': 12, 'A5_weak': 10, 'A3_uncastled': 6, 'A4_other': 3,
        'B1_defended_crowd': 12, 'B2_queenless': 12, 'B3_shelter': 12, 'B4_calm': 12}

rows = []
with open(CAND) as f:
    rd = csv.DictReader(f, delimiter='\t')
    for r in rd:
        if not r.get('fen') or r['fen'] == 'fen': continue
        try:
            r['sf_ks'] = float(r['sf_ks']); r['att'] = int(r['att']); r['weak'] = int(r['weak'])
        except Exception:
            continue
        rows.append(r)

# filter B2 pawn-ish endgames: require >=2 ring attackers AND some king-zone piece pressure (weak or >=3 att)
def keep(r):
    a = r['archetype']
    if a[0] == 'B':                                 # counters must be genuine middlegames (not endgames)
        if not is_middlegame(r['fen']): return False
        if a == 'B2_queenless' and r['att'] < 2: return False
    return True

by = {}
for r in rows:
    if keep(r):
        by.setdefault(r['archetype'], []).append(r)

bench = []
for arch, cap in CAPS.items():
    grp = by.get(arch, [])
    danger = arch[0] == 'A'
    grp.sort(key=lambda r: -abs(r['sf_ks']) if danger else abs(r['sf_ks']))
    for r in grp[:cap]:
        bench.append((r['fen'], arch, 'HIGH' if danger else 'LOW', r['sf_ks'], r['subj']))

# fold in AWAY over-reads (empirical over-production; keep only genuinely-calm-per-SF, |sf|<0.5)
n_away = 0
if os.path.exists(COLL):
    for ln in open(COLL):
        if 'AWAY' not in ln: continue
        parts = ln.split()
        if len(parts) < 10: continue
        fen = ' '.join(parts[0:6])
        try:
            sf = float(parts[9])
        except Exception:
            continue
        if abs(sf) < 0.5 and is_middlegame(fen):
            bench.append((fen, 'B1_overread', 'LOW', sf, '?'))
            n_away += 1

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['fen', 'archetype', 'expected', 'sf_ks', 'subj'])
    for row in bench:
        w.writerow(row)

from collections import Counter
c = Counter(b[1] for b in bench)
print("wrote %s  (%d positions, +%d AWAY over-reads)" % (OUT, len(bench), n_away))
for k in sorted(c): print("  %-20s %d" % (k, c[k]))
print("DANGER(A) total:", sum(v for k, v in c.items() if k[0] == 'A'),
      " COUNTER(B) total:", sum(v for k, v in c.items() if k[0] == 'B'))
