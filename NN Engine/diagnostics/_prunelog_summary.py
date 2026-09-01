"""Histogram a [PRUNEFIRE] log by kind and remaining depth, so we know what the
verification pass has to work with before spending re-searches on it."""
import sys, re, collections, os
path = sys.argv[1]
if not os.path.exists(path):
    print("MISSING:", path); sys.exit(0)
size = os.path.getsize(path)
lines = open(path, errors='replace').read().splitlines()
fires = [l for l in lines if 'PRUNEFIRE' in l]
print(f"file={path}  bytes={size}  lines={len(lines)}  PRUNEFIRE lines={len(fires)}")
if not fires:
    print("--- first 15 non-empty lines (what IS in the file) ---")
    for l in [x for x in lines if x.strip()][:15]:
        print("   ", l[:160])
    sys.exit(0)
print("--- sample ---")
for l in fires[:3]:
    print("   ", l[:200])
by_kind = collections.Counter()
by_rd = collections.Counter()
for l in fires:
    m = re.search(r'kind=(\w+)', l)
    if m: by_kind[m.group(1)] += 1
    m = re.search(r'\brd=(\d+)', l)
    if m: by_rd[int(m.group(1))] += 1
print("by kind:", dict(by_kind))
print("by rd  :", dict(sorted(by_rd.items())))
