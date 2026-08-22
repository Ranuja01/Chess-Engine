# -*- coding: utf-8 -*-
"""Counter-analysis: on the HURT (integration blew) and HELPED (integration held) decision sets, compare our KS
term (baseline vs integration) against SF11's KS, to find what distinguishes the two. Hypothesis to test: on HURT
positions the integration OVER-reads king danger vs SF (false danger -> bad move); on HELPED it reads danger SF
also sees. One engine subprocess per (set,mode); SF11 per set. Allowlisted via pyrun.

Run: bash <runner> pyrun diagnostics/_sprt_counter.py > diagnostics/_sprt_counter.out
"""
import os, sys, subprocess, signal
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
import csv, chess
THIS_DIR = os.path.dirname(os.path.abspath(__file__)); ENGINE_DIR = os.path.dirname(THIS_DIR)
sys.path.insert(0, ENGINE_DIR); sys.path.insert(0, THIS_DIR)
WORKER = os.path.join(THIS_DIR, '_ks_detector_sweep.py')
from eval_vs_sf11 import SF11Eval, SF11
class _TO(Exception): pass
signal.signal(signal.SIGALRM, lambda s, f: (_ for _ in ()).throw(_TO()))
PY = sys.executable
INTEG = {'KS_FLANK_MODE': '2', 'ENABLE_KS_ZONE_CLAMP': '1', 'KS_WEAK_VAL_MODE': '1',
         'KS_PIN_MODE': '1', 'KS_ZONE2': '1', 'KS_ACCUM_MODE': '1', 'KS_ACCUM_THRESH': '28'}

def load(path):
    fens = []
    with open(os.path.join(ENGINE_DIR, path)) as f:
        for r in csv.DictReader(f):
            fen = (r.get('fen_start') or '').strip()
            if fen and not chess.Board(fen).is_check():
                fens.append(fen)
    return fens

def sf_ks_for(fens):
    sf = SF11Eval(SF11); out = []
    for fen in fens:
        signal.alarm(8)
        try:
            _, t = sf.eval(fen); signal.alarm(0)
        except _TO:
            out.append(None); signal.alarm(0); continue
        finally:
            signal.alarm(0)
        out.append(t.get('King safety', 0.0) if t else None)
    sf.close(); return out

def our_ks(fens, knobs):
    ff = os.path.join(ENGINE_DIR, 'diagnostics/_sc_fens.txt'); open(ff, 'w').write("\n".join(fens) + "\n")
    env = dict(os.environ); env.update(knobs)
    out = subprocess.run([PY, WORKER, '--worker', ff], cwd=ENGINE_DIR, env=env,
                         stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, timeout=600).stdout
    ks = [0.0] * len(fens)
    for ln in out.splitlines():
        if '\t' in ln:
            i, v = ln.split('\t'); ks[int(i)] = float(v)
    return ks

import statistics as st
for label, path in [('HURT (integration blew a win)', 'diagnostics/_sprt_hurt.csv'),
                    ('HELPED (integration held base blew)', 'diagnostics/_sprt_helped.csv')]:
    fens = load(path)
    sf = sf_ks_for(fens)
    base = our_ks(fens, {}); integ = our_ks(fens, INTEG)
    idx = [i for i in range(len(fens)) if sf[i] is not None]
    def m(f): return st.mean([f(i) for i in idx]) if idx else 0.0
    print("== %s  (n=%d) ==" % (label, len(idx)))
    print("  mean|SF KS|            = %.3f" % m(lambda i: abs(sf[i])))
    print("  mean|our KS| base      = %.3f   integ = %.3f" % (m(lambda i: abs(base[i])), m(lambda i: abs(integ[i]))))
    print("  mean(|integ|-|SF|)     = %+.3f   (signed magnitude diff; cancels over/under)" % m(lambda i: abs(integ[i]) - abs(sf[i])))
    print("  PER-POSITION disagreement mean|integ-SF| = %.3f   base = %.3f   (this is the real closeness)" % (
        m(lambda i: abs(integ[i]-sf[i])), m(lambda i: abs(base[i]-sf[i]))))
    print("  integ within 0.5pawn of SF: %d/%d ;  within 1.0: %d/%d" % (
        sum(1 for i in idx if abs(integ[i]-sf[i]) <= 0.5), len(idx),
        sum(1 for i in idx if abs(integ[i]-sf[i]) <= 1.0), len(idx)))
    print("  integ moved off base   = %d/%d ;  of those, over-shot SF: %d" % (
        sum(1 for i in idx if abs(integ[i]-base[i])>1e-4),
        len(idx),
        sum(1 for i in idx if abs(integ[i]-base[i])>1e-4 and abs(integ[i]-sf[i])>abs(base[i]-sf[i])+1e-4)))
    print()
