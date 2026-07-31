"""Scan existing corpora for phantom-KS coverage (KS firing hard AND over-reading SF18 total),
to decide whether the KS realizability fit needs new SF18 labeling or can reuse existing rows.
Read-only. Run via: pyrun diagnostics/ks_phantom_scan.py
"""
import os, sys, csv
for _a in sys.argv[1:]:
    if '=' in _a:
        _k, _v = _a.split('=', 1); os.environ.setdefault(_k, _v)

HERE = os.path.dirname(os.path.abspath(__file__))
DIV = os.path.join(HERE, 'ks_sets', 'diverse_corpus.csv')
BANK = os.path.join(HERE, 'ks_sets', 'position_bank.csv')

def fnum(s):
    try: return float(s)
    except: return None

# --- diverse_corpus: over-read rows split by whether KS is the driver ---
rows = list(csv.DictReader(open(DIV)))
print(f"diverse_corpus.csv: {len(rows)} rows")
ks_phantom = []   # KS firing AND over-read total (the MOD_KS_REALIZ target)
nonks_over = []   # over-read but KS not firing (MOD_KS_REALIZ won't help)
for r in rows:
    tt, ob, ks = fnum(r['target_total']), fnum(r['our_total_base']), fnum(r['our_ks_base'])
    if None in (tt, ob, ks): continue
    over = ob - tt                       # positive = we over-read white; sign-aware magnitude below
    over_mag = abs(ob) - abs(tt)         # over-read in magnitude (either side)
    if abs(ks) >= 1.0 and over_mag >= 1.5:
        ks_phantom.append((r['fen'], tt, ob, ks, r['tier'], r['phase_bucket']))
    elif over_mag >= 1.5 and abs(ks) < 0.3:
        nonks_over.append(over_mag)
print(f"  phantom-KS (|our_ks|>=1.0 & over_mag>=1.5): {len(ks_phantom)}")
print(f"  non-KS over-reads (|our_ks|<0.3 & over_mag>=1.5): {len(nonks_over)}")
from collections import Counter
print("  phantom-KS by phase:", dict(Counter(p[5] for p in ks_phantom)))
print("  phantom-KS by tier :", dict(Counter(p[4] for p in ks_phantom)))
print("  --- phantom-KS samples ---")
for fen, tt, ob, ks, tier, ph in ks_phantom[:12]:
    print(f"    over={abs(ob)-abs(tt):+.2f} sf18={tt:+.2f} ours={ob:+.2f} ks={ks:+.2f} [{tier}/{ph}] {fen}")

# --- position_bank: SF18-labeled rows we could pull in as more phantom-KS ---
if os.path.exists(BANK):
    brows = list(csv.DictReader(open(BANK)))
    sf18n = [r for r in brows if fnum(r.get('sf18')) is not None]
    print(f"\nposition_bank.csv: {len(brows)} rows, {len(sf18n)} SF18-labeled")
    bank_phantom = []
    for r in sf18n:
        ot, ks, sf = fnum(r.get('our_total')), fnum(r.get('our_ks')), fnum(r.get('sf18'))
        if None in (ot, ks, sf): continue
        if abs(ks) >= 1.0 and (abs(ot) - abs(sf)) >= 1.5:
            bank_phantom.append(r)
    print(f"  bank phantom-KS candidates (|our_ks|>=1.0 & over_mag>=1.5): {len(bank_phantom)}")
    # how many phantom bank rows are NOT already in diverse_corpus
    divfens = {r['fen'] for r in rows}
    new = [r for r in bank_phantom if r['fen'] not in divfens]
    print(f"  of those, NOT already in diverse_corpus: {len(new)}")
