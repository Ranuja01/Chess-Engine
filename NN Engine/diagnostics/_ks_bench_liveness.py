# -*- coding: utf-8 -*-
"""How much of the KS archetype bench is actually LIVE at a given config?

`KS_FLOOR` returns 0 danger per king below the deadzone, and the netted KS is a difference of two
floored values -- so a bench position can read exactly 0.00 while both kings carry real units. If most
positions read 0, the bench's per-archetype averages are counting FLOOR CROSSINGS, not measuring danger,
and every delta read off them is quantized rather than graded.

Reports, per archetype: n, how many positions produce a non-zero netted KS, and the share of the
archetype's mean that comes from its single largest-magnitude position (concentration). A high
concentration means the archetype number is one position wearing a trenchcoat.

Run: bash <runner> pyrun diagnostics/_ks_bench_liveness.py [KEY=VAL ...]
"""
import os, sys
for a in sys.argv[1:]:
    if '=' in a:
        k, v = a.split('=', 1); os.environ[k] = v
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
import csv, chess
THIS_DIR = os.path.dirname(os.path.abspath(__file__)); ENGINE_DIR = os.path.dirname(THIS_DIR)
sys.path.insert(0, ENGINE_DIR); sys.path.insert(0, THIS_DIR)
from ChessAI import ChessAI
seed = chess.Board(); ai = ChessAI(None, None, seed, seed.turn)

rows = list(csv.DictReader(open(os.path.join(THIS_DIR, 'ks_sets', 'ks_archetypes.csv'))))
per = {}
for r in rows:
    ks = ai.ev_breakdown(chess.Board(r['fen'])).get('king_safety', 0) / 1000.0
    per.setdefault(r['archetype'], []).append(abs(ks))

cfg = " ".join(a for a in sys.argv[1:] if '=' in a) or "(baseline defaults)"
print("KS bench LIVENESS: %s" % cfg)
print("%-20s %4s %6s %8s %10s" % ("archetype", "n", "live", "mean|ks|", "top1 share"))
tot_n = tot_live = 0
for arch in sorted(per):
    v = per[arch]
    live = sum(1 for x in v if x > 1e-9)
    s = sum(v)
    top = (max(v) / s * 100) if s > 1e-9 else 0.0
    tot_n += len(v); tot_live += live
    print("%-20s %4d %6d %8.3f %9.0f%%" % (arch, len(v), live, s / len(v), top))
print("-" * 52)
print("TOTAL %d/%d positions live (%.0f%%)" % (tot_live, tot_n, 100.0 * tot_live / tot_n))
print()
print("read: a low live count means the bench is measuring FLOOR CROSSINGS, not danger magnitude;")
print("      a high top1 share means that archetype's number is one position, not an average.")
