# -*- coding: utf-8 -*-
"""Does our PHASE-1 detection distinguish danger from quiet, or fire similarly on both? For each bench position,
capture our KS component dump (KSD: attsq/weak/safe/attpc/units) for the SUBJECT king, group by DANGER (A*) vs
QUIET (B*), and compare to SF11's KS. If our safe/weak/attpc counts are SIMILAR across danger and quiet while
SF's KS clearly separates them (~0 vs high), our feeder can't tell them apart -> the 'when' is unsolvable on it.

Run: bash <runner> pyrun diagnostics/_ks_detect_dist.py
"""
import os, sys, tempfile, re
for a in sys.argv[1:]:                                # KEY=VAL config, set before ChessAI init (knobs latch once)
    if '=' in a:
        k, v = a.split('=', 1); os.environ[k] = v
os.environ['KS_DEBUG_DUMP'] = '1'                     # must be set before ChessAI init
os.environ['KS_FLOOR'] = '0'                          # see below-floor detection too
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
import csv, chess, statistics as st
THIS_DIR = os.path.dirname(os.path.abspath(__file__)); ENGINE_DIR = os.path.dirname(THIS_DIR)
sys.path.insert(0, ENGINE_DIR); sys.path.insert(0, THIS_DIR)
from ChessAI import ChessAI
seed = chess.Board(); ai = ChessAI(None, None, seed, seed.turn)

KSD = re.compile(r'KSD (\w) attsq=(\d+) weak=(\d+) safe=(\d+) attpc=(\d+) defpc=(\d+) openf=(\d+) bkru=(\d+) over=(\d+) units=(-?\d+) danger=(-?\d+)')

def dump_eval(board):
    tf = tempfile.TemporaryFile()
    old = os.dup(2); os.dup2(tf.fileno(), 2)
    try:
        ai.ev_breakdown(board)
    finally:
        os.dup2(old, 2); os.close(old)
    tf.seek(0); data = tf.read().decode('utf-8', 'ignore'); tf.close()
    out = {}
    for m in KSD.finditer(data):
        out[m.group(1)] = dict(attsq=int(m.group(2)), weak=int(m.group(3)), safe=int(m.group(4)),
                               attpc=int(m.group(5)), units=int(m.group(10)))
    return out

rows = list(csv.DictReader(open(os.path.join(THIS_DIR, 'ks_sets', 'ks_archetypes.csv'))))
groups = {'DANGER(A)': [], 'QUIET(B)': [], 'STS_REGRESS': []}
for r in rows:
    subj = r.get('subj', '?')
    d = dump_eval(chess.Board(r['fen']))
    if subj not in ('W', 'B'):
        subj = 'W' if float(r['sf_ks']) < 0 else 'B'
    k = d.get(subj)
    if not k: continue
    g = 'DANGER(A)' if r['archetype'][0] == 'A' else 'QUIET(B)'
    groups[g].append((k, abs(float(r['sf_ks']))))

# STS positional-regression positions (V2 crashes these; SF rates them positional, not king-danger).
# No SF-KS/subj available -> take the king WE flag most (max units) and record its detection.
srp = os.path.join(THIS_DIR, '_sts_regress.fens')
if os.path.exists(srp):
    for ln in open(srp):
        fen = ln.strip()
        if not fen or fen == 'fen_start': continue
        try:
            d = dump_eval(chess.Board(fen))
        except Exception:
            continue
        if not d: continue
        k = max(d.values(), key=lambda x: x['units'])
        groups['STS_REGRESS'].append((k, 0.0))

print("PHASE-1 detection: DANGER vs QUIET  (our components for the SUBJECT king; KS_FLOOR forced 0)")
print("%-12s %4s %8s %8s %8s %8s %8s %9s" % ("group", "n", "attsq", "weak", "safe", "attpc", "units", "SF|KS|"))
for g, v in groups.items():
    if not v: continue
    def m(f): return st.mean(f(k) for k, _ in v)
    print("%-12s %4d %8.2f %8.2f %8.2f %8.2f %8.2f %9.2f" % (
        g, len(v), m(lambda k: k['attsq']), m(lambda k: k['weak']), m(lambda k: k['safe']),
        m(lambda k: k['attpc']), m(lambda k: k['units']), st.mean(s for _, s in v)))
print("\nread: if DANGER and QUIET have SIMILAR safe/weak/attpc/units but very different SF|KS|,")
print("      our phase-1 detector cannot separate them -> no phase-2 threshold can either.")
