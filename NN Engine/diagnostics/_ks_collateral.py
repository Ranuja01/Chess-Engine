# -*- coding: utf-8 -*-
"""Collateral dump: for baseline / flank_contest / conserved-integration, per collapse FEN print our king_safety
(White-POV), SF11's, and whether each candidate moved TOWARD or AWAY from SF. Isolates the AWAY positions (the
collateral to triangulate: why the detector overshoots there and how SF holds it). One engine subprocess per arm
(sequential = ~1 core), SF11 for the target. Allowlisted via `pyrun`.

Run: bash <runner> pyrun diagnostics/_ks_collateral.py > diagnostics/_ks_collateral.out
"""
import os, sys, subprocess, signal
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
import csv, chess
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ENGINE_DIR = os.path.dirname(THIS_DIR)
sys.path.insert(0, ENGINE_DIR); sys.path.insert(0, THIS_DIR)

WORKER = os.path.join(THIS_DIR, '_ks_detector_sweep.py')   # reuse its --worker mode (prints idx\tks)
FENS = 'selfplay/games/vssf_2400/dp_fens.csv'

from eval_vs_sf11 import SF11Eval, SF11
class _TO(Exception): pass
signal.signal(signal.SIGALRM, lambda s, f: (_ for _ in ()).throw(_TO()))

raw = []
with open(os.path.join(ENGINE_DIR, FENS)) as f:
    rd = csv.DictReader(f)
    col = 'fen_start' if 'fen_start' in (rd.fieldnames or []) else (rd.fieldnames or ['fen'])[0]
    for r in rd:
        fen = (r.get(col) or '').strip()
        if fen and not chess.Board(fen).is_check():
            raw.append(fen)

sf11 = SF11Eval(SF11); sf_ks = []; keep = []
for fen in raw:
    signal.alarm(8)
    try:
        _, terms = sf11.eval(fen); signal.alarm(0)
    except _TO:
        continue
    finally:
        signal.alarm(0)
    if terms is None: continue
    sf_ks.append(terms.get('King safety', 0.0)); keep.append(fen)
sf11.close()

fenfile = os.path.join(ENGINE_DIR, 'diagnostics/_ksc_fens.txt')
open(fenfile, 'w').write("\n".join(keep) + "\n")
N = len(keep); PY = sys.executable

def arm(knobs):
    env = dict(os.environ); env.update(knobs)
    out = subprocess.run([PY, WORKER, '--worker', fenfile], cwd=ENGINE_DIR, env=env,
                         stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, timeout=300).stdout
    ks = [0.0]*N
    for ln in out.splitlines():
        if '\t' in ln:
            i, v = ln.split('\t');  ks[int(i)] = float(v)
    return ks

base = arm({})
flank = arm({'KS_FLANK_MODE': '2'})
integ = arm({'KS_FLANK_MODE': '2', 'ENABLE_KS_ZONE_CLAMP': '1', 'KS_WEAK_VAL_MODE': '1',
             'KS_PIN_MODE': '1', 'KS_ZONE2': '1', 'KS_ACCUM_MODE': '1', 'KS_ACCUM_THRESH': '28'})

def dirn(a, b, sf):
    if abs(a-b) < 1e-4: return '.'
    return 'T' if abs(a-sf) < abs(b-sf) - 1e-4 else 'A'

print("%-72s %7s %7s %7s %7s  %s %s" % ("fen", "base", "flank", "integ", "sf", "fl", "in"))
for i in range(N):
    fl, ic = dirn(flank[i], base[i], sf_ks[i]), dirn(integ[i], base[i], sf_ks[i])
    tag = "  <== AWAY" if 'A' in (fl+ic) else ""
    print("%-72s %+7.2f %+7.2f %+7.2f %+7.2f  %s  %s%s" % (keep[i], base[i], flank[i], integ[i], sf_ks[i], fl, ic, tag))
print("\nflank AWAY:", sum(1 for i in range(N) if dirn(flank[i],base[i],sf_ks[i])=='A'),
      " integ AWAY:", sum(1 for i in range(N) if dirn(integ[i],base[i],sf_ks[i])=='A'), " of", N)
