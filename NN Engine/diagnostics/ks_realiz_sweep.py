"""Guarded sweep of the KS-V2 realizability levers (MOD_KS_REALIZ x KS_REALIZ_FLOOR) on the re-tiered corpus.
Every candidate runs with ENABLE_KS_V2=1 ENABLE_PASSER_V3=1 (passer baseline ON, per the joint-tune discipline).
Minimise the ks_phantom over-read while HOLDING every guard family (ks_real / calm / working / crowded_safe /
sts_guard / target) within GUARD_TOL of the realiz-OFF baseline. Reports per-tier win%-MSE so we pick a MODEST
lever, not the deepest damp. Run: pyrun diagnostics/ks_realiz_sweep.py [GUARD_TOL=1.0]
"""
import os, sys, subprocess, re
from collections import OrderedDict
TOL = float(os.environ.get('GUARD_TOL', '1.0'))
THIS = os.path.dirname(os.path.abspath(__file__))
CORPUS = os.path.join(THIS, 'ks_sets', 'diverse_corpus_ksv2.csv')
WORKER = os.path.join(THIS, '_ks_fit_eval.py')
PY = sys.executable
FIXED = ['ENABLE_KS_V2=1', 'ENABLE_PASSER_V3=1']
GUARDS = ['ks_real', 'calm', 'working', 'crowded_safe', 'sts_guard', 'target']

def run(realiz, floor):
    args = [PY, WORKER, 'CORPUS=' + CORPUS] + FIXED + ['MOD_KS_REALIZ=%d' % realiz, 'KS_REALIZ_FLOOR=%d' % floor]
    out = subprocess.run(args, capture_output=True, text=True).stdout
    d = {}
    for t, s, v in re.findall(r"(\w+)\.(train|val)=([\d.]+)", out):
        d['%s.%s' % (t, s)] = float(v)
    return d

grid_r = [0, 200, 400, 600, 900, 1300]
grid_f = [128, 96, 64, 32]
base = run(0, 128)
bg = {t: base.get(t + '.train', 9e9) for t in GUARDS}
ph0 = base.get('ks_phantom.train', 9e9); ph0v = base.get('ks_phantom.val', 9e9)
print("BASELINE (V2+V3, realiz off): ks_phantom train=%.3f val=%.3f" % (ph0, ph0v))
print("  guards:", {t: round(bg[t], 3) for t in GUARDS})
print("\n%-8s %-6s %-10s %-10s %-8s %s" % ("REALIZ", "FLOOR", "phant.tr", "phant.val", "guards", "verdict"))
best = None
for r in grid_r:
    floors = [128] if r == 0 else grid_f
    for f in floors:
        if r == 0 and f != 128: continue
        d = run(r, f)
        ph = d.get('ks_phantom.train', 9e9); phv = d.get('ks_phantom.val', 9e9)
        worst = max(d.get(t + '.train', 9e9) - bg[t] for t in GUARDS)
        ok = worst <= TOL
        verdict = 'ok' if ok else 'GUARD+%.2f' % worst
        tag = ''
        if ok and r > 0 and ph < ph0 - 1e-4:
            if best is None or ph < best[2]: best = (r, f, ph, phv, worst)
            tag = ' <-'
        print("%-8d %-6d %-10.3f %-10.3f %-8s %s%s" % (r, f, ph, phv, 'HELD' if ok else 'X', verdict, tag))
print("\nphantom-reducing, guard-holding candidates ranked; BEST modest pick =",
      "MOD_KS_REALIZ=%d KS_REALIZ_FLOOR=%d (phant %.3f->%.3f, worst guard +%.2f)" % (best[0], best[1], ph0, best[2], best[4]) if best else "NONE (no guarded reduction -> games on realiz-off consolidation only)")
