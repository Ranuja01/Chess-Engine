"""AUC of each decision-point signal against the WRONG label, per mechanism.

Reads [SHADOWEV] records emitted under ENABLE_PRUNE_SHADOW=1 ENABLE_SHADOW_EVENTS=1.
Decision rule (from prune_discriminate.py): AUC >~0.6 on some cheap signal => a conditional
GUARD is buildable and we know its form. Nothing separates => the over-prunes are not cheaply
distinguishable and a guard cannot work -- learned offline, before building anything.
"""
import sys, re, collections, math

path = sys.argv[1]
FIELD = re.compile(r'(\w+)=(-?\d+)')
def parse(line):
    if '[SHADOWEV]' not in line:
        return None, None
    kind = re.search(r'kind=(\w+)', line)
    d = {k: int(v) for k, v in FIELD.findall(line)}
    return (kind.group(1) if kind else None), d
rows = collections.defaultdict(list)
n_lines = 0
for line in open(path, errors='replace'):
    kind, d = parse(line)
    if kind is None:
        continue
    n_lines += 1
    rows[kind].append(d)
print(f"parsed {n_lines} [SHADOWEV] records from {path}")

def auc(scores, labels):
    pairs = sorted(zip(scores, labels))
    npos = sum(labels); nneg = len(labels) - npos
    if npos == 0 or nneg == 0:
        return float('nan'), npos, nneg
    # rank-based with tie averaging
    ranks = {}
    idx = 0
    while idx < len(pairs):
        j = idx
        while j + 1 < len(pairs) and pairs[j+1][0] == pairs[idx][0]:
            j += 1
        avg = (idx + j) / 2.0 + 1
        for k in range(idx, j+1):
            ranks[k] = avg
        idx = j + 1
    rsum = sum(ranks[k] for k, (_, l) in enumerate(pairs) if l == 1)
    return (rsum - npos*(npos+1)/2.0) / (npos*nneg), npos, nneg

for kind, rs in sorted(rows.items()):
    label = [r['ent'] for r in rs]          # "the pruned/reduced move mattered"
    npos = sum(label)
    print(f"\n=== {kind}: n={len(rs)}  wrong(ent)={npos} ({100.0*npos/max(1,len(rs)):.2f}%) ===")
    if npos < 20:
        print("   too few positives for a meaningful AUC (need >=20)")
        continue
    g = lambda r, k, dflt=0: r.get(k, dflt)
    feats = {
        'hist'         : [g(r,'hist') for r in rs],
        'statScore_ss' : [g(r,'ss') for r in rs],
        'cmh'          : [g(r,'cmh') for r in rs],
        'ch2'          : [g(r,'ch2') for r in rs],
        'killer_or_cm' : [g(r,'kc') for r in rs],
        'move_index_i' : [g(r,'i') for r in rs],
        '-move_index_i': [-g(r,'i') for r in rs],
        'rd'           : [g(r,'rd') for r in rs],
        '-rd'          : [-g(r,'rd') for r in rs],
        'ply'          : [g(r,'ply') for r in rs],
        'window_b-a'   : [g(r,'b') - g(r,'a') for r in rs],
        'window_ORIG'  : [g(r,'bow') - g(r,'aow') for r in rs],
        'is_capture'   : [g(r,'cap') for r in rs],
        'in_check'     : [g(r,'chk') for r in rs],
        'null_search'  : [g(r,'ns') for r in rs],
        'capture_chain': [g(r,'cc') for r in rs],
        'abs_alpha'    : [abs(g(r,'a')) for r in rs],
        'staticPlace'  : [g(r,'sps') for r in rs],
        '-staticPlace' : [-g(r,'sps') for r in rs],
        'abs_statPlace': [g(r,'asps') for r in rs],
    }
    for name, sc in sorted(feats.items()):
        a, p, n = auc(sc, label)
        flag = "  <-- SEPARATES" if not math.isnan(a) and a >= 0.60 else ""
        print(f"   AUC {name:<13} = {a:.4f}   (pos={p} neg={n}){flag}")

# ---- second pass: the statScore~0 population, where the history family is BLIND ----
print("")
print("=" * 72)
print("ZERO-HISTORY SUBPOPULATION (|statScore| < 1024) -- history markers cannot speak here")
for kind, rs in sorted(rows.items()):
    sub = [r for r in rs if abs(r.get('ss', 0)) < 1024]
    label = [r['ent'] for r in sub]
    npos = sum(label)
    print("")
    print(f"=== {kind}: n={len(sub)} of {len(rs)}   wrong={npos} ===")
    if npos < 20:
        print("   too few positives for a meaningful AUC")
        continue
    g = lambda r, k, d=0: r.get(k, d)
    feats = {
        'staticPlace'  : [g(r, 'sps') for r in sub],
        '-staticPlace' : [-g(r, 'sps') for r in sub],
        'abs_statPlace': [g(r, 'asps') for r in sub],
        'move_index_i' : [g(r, 'i') for r in sub],
        '-move_index_i': [-g(r, 'i') for r in sub],
        '-rd'          : [-g(r, 'rd') for r in sub],
        'killer_or_cm' : [g(r, 'kc') for r in sub],
        'window_ORIG'  : [g(r, 'bow') - g(r, 'aow') for r in sub],
    }
    for name, sc in sorted(feats.items()):
        a, p_, n_ = auc(sc, label)
        flag = "  <-- SEPARATES" if not math.isnan(a) and a >= 0.60 else ""
        print(f"   AUC {name:<14} = {a:.4f}   (pos={p_} neg={n_}){flag}")
