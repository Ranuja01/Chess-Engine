"""Re-tier diverse_corpus.csv for the KS-V2 realizability fit: split KS-firing rows into a PHANTOM tier
(we over-read SF18 -> the MOD_KS_REALIZ damp target) vs a REAL guard tier (KS fires and SF18 AGREES ->
must be held), leaving the existing control families (calm/working/crowded_safe/sts_guard/target) intact.
Writes diverse_corpus_ksv2.csv (same schema). Read-only on the source. Run: pyrun diagnostics/ks_v2_retier.py
"""
import os, sys, csv
for _a in sys.argv[1:]:
    if '=' in _a: _k,_v=_a.split('=',1); os.environ.setdefault(_k,_v)
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, 'ks_sets', 'diverse_corpus.csv')
DST = os.path.join(HERE, 'ks_sets', 'diverse_corpus_ksv2.csv')

def fnum(s):
    try: return float(s)
    except: return None

rows = list(csv.DictReader(open(SRC)))
fields = rows[0].keys()
from collections import Counter
changed = Counter()
for r in rows:
    tt, ob, ks = fnum(r['target_total']), fnum(r['our_total_base']), fnum(r['our_ks_base'])
    if None in (tt, ob, ks): continue
    over_mag = abs(ob) - abs(tt)          # >0 = we over-read the magnitude (either side)
    if abs(ks) >= 1.0 and over_mag >= 1.5:
        r['tier'] = 'ks_phantom'; changed['ks_phantom'] += 1     # damp TARGET
    elif abs(ks) >= 1.0 and abs(over_mag) <= 0.8:
        r['tier'] = 'ks_real'; changed['ks_real'] += 1           # GUARD (real attack SF agrees with)
w = csv.DictWriter(open(DST, 'w', newline=''), fieldnames=list(fields))
w.writeheader(); w.writerows(rows)
print("wrote", DST)
print("re-tagged:", dict(changed))
print("tier distribution:", dict(Counter(r['tier'] for r in rows)))
print("ks_phantom by phase:", dict(Counter(r['phase_bucket'] for r in rows if r['tier']=='ks_phantom')))
print("ks_phantom by split:", dict(Counter(r['split'] for r in rows if r['tier']=='ks_phantom')))
print("ks_real by phase:", dict(Counter(r['phase_bucket'] for r in rows if r['tier']=='ks_real')))
