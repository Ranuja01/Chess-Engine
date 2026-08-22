# -*- coding: utf-8 -*-
"""CALIBRATION instrument for king safety: is our KS the right SIZE, not just the right ORDER?

Why this exists. `_ks_auc.py` measures DISCRIMINATION and is scale-free BY CONSTRUCTION -- that is what makes
it immune to volume knobs, and it is also what makes it structurally BLIND to a uniform magnitude deficit.
Our feeders order positions well (AUC 0.91 floor-free) while reading roughly 19% of SF's magnitude on the
archetype bench. A feeder that counts pinned defenders as real, or misses a queen-behind-rook battery, mostly
SHRINKS danger on attacked kings -- positions keep their relative order (AUC unmoved) while every number
comes out several times too small. AUC cannot see that. This can.

Method: bucket positions by SF's |target_ks| and report, per bucket, mean |ours| vs mean |SF| and the ratio.
  ratio ~1.0 across buckets      => calibrated
  ratio ~constant but << 1       => UNIFORM under-read (a magnitude problem; suspect the feeder defects)
  ratio falling as SF rises      => we saturate / miss the big attacks specifically
  ratio rising as SF rises       => we over-read the mild cases

Also reports Spearman rank correlation (an ordering check independent of AUC's binary split) and the share of
positions reading exactly 0, which is the floor's footprint.

Run: bash <runner> pyrun diagnostics/_ks_calibration.py [KEY=VAL ...] [PHASE=midgame] [MAXN=0]
"""
import os, sys
OPTS = {}
for a in sys.argv[1:]:
    if '=' in a:
        k, v = a.split('=', 1)
        if k in ('PHASE', 'MAXN'):
            OPTS[k] = v
        else:
            os.environ[k] = v          # engine knob: set before ChessAI init (knobs latch once)
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
import csv, chess, statistics as st
THIS_DIR = os.path.dirname(os.path.abspath(__file__)); ENGINE_DIR = os.path.dirname(THIS_DIR)
sys.path.insert(0, ENGINE_DIR); sys.path.insert(0, THIS_DIR)
from ChessAI import ChessAI
seed = chess.Board(); ai = ChessAI(None, None, seed, seed.turn)

PHASE = OPTS.get('PHASE', 'midgame')
MAXN = int(OPTS.get('MAXN', 0))
BUCKETS = [(0.05, 0.25), (0.25, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 4.0), (4.0, 99.0)]

rows = []
src = os.path.join(THIS_DIR, 'ks_sets', 'diverse_corpus_wide.csv')
for r in csv.DictReader(open(src)):
    if PHASE != 'all' and r.get('phase_bucket') != PHASE:
        continue
    try:
        sf = abs(float(r['target_ks']))
    except (TypeError, ValueError):
        continue
    if sf < 0.05:
        continue                        # calibration is only meaningful where SF sees something
    try:
        ours = abs(ai.ev_breakdown(chess.Board(r['fen'])).get('king_safety', 0) / 1000.0)
    except Exception:
        continue
    rows.append((sf, ours))
    if MAXN and len(rows) >= MAXN:
        break

def spearman(pairs):
    """Rank correlation; ties get average ranks."""
    def ranks(vals):
        order = sorted(range(len(vals)), key=lambda i: vals[i])
        rk = [0.0] * len(vals); i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and vals[order[j + 1]] == vals[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                rk[order[k]] = avg
            i = j + 1
        return rk
    a = ranks([p[0] for p in pairs]); b = ranks([p[1] for p in pairs])
    n = len(pairs)
    ma, mb = sum(a) / n, sum(b) / n
    num = sum((a[i] - ma) * (b[i] - mb) for i in range(n))
    da = sum((a[i] - ma) ** 2 for i in range(n)) ** 0.5
    db = sum((b[i] - mb) ** 2 for i in range(n)) ** 0.5
    return num / (da * db) if da and db else float('nan')

cfg = " ".join(a for a in sys.argv[1:] if '=' in a and a.split('=', 1)[0] not in ('PHASE', 'MAXN')) or "(baseline defaults)"
print("KS CALIBRATION (magnitude, not order) : %s" % cfg)
print("  corpus=diverse_corpus_wide phase=%s  n=%d  (positions where SF sees KS, |target_ks|>0.05)" % (PHASE, len(rows)))
print("%-14s %6s %9s %9s %8s %8s" % ("SF |KS| band", "n", "mean SF", "mean OURS", "ratio", "ours=0"))
for lo, hi in BUCKETS:
    v = [(s, o) for s, o in rows if lo <= s < hi]
    if not v:
        continue
    ms = st.mean(s for s, _ in v); mo = st.mean(o for _, o in v)
    z = 100.0 * sum(1 for _, o in v if o < 1e-9) / len(v)
    print("%-14s %6d %9.3f %9.3f %8.2f %7.0f%%" % ("%.2f-%.2f" % (lo, hi), len(v), ms, mo, (mo / ms) if ms else 0, z))
print("-" * 60)
if rows:
    ms = st.mean(s for s, _ in rows); mo = st.mean(o for _, o in rows)
    print("OVERALL        %6d %9.3f %9.3f %8.2f %7.0f%%"
          % (len(rows), ms, mo, (mo / ms) if ms else 0, 100.0 * sum(1 for _, o in rows if o < 1e-9) / len(rows)))
    print("  Spearman rank corr (ours vs SF) = %.4f   (ordering, independent of AUC's binary split)" % spearman(rows))
print()
print("read: ratio flat but << 1 => UNIFORM UNDER-READ, a magnitude problem AUC cannot see;")
print("      ratio falling with SF => we specifically miss the BIG attacks; rising => we over-read mild ones.")
