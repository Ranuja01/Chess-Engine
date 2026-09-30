# Pair an odds vs_sf run whose results.csv was lost (log only) against a baseline results.csv, overall and by role.
# Odds FENs give White the handicap: our engine as White = DEFENDING a deficit, as Black = CONVERTING an edge.
import sys, re, csv, math
log, base_csv = sys.argv[1], sys.argv[2]
cand = {}
for line in open(log, encoding="utf-8", errors="ignore"):
    m = re.search(r"game (\d+): .*our=([0-9.]+) as (white|black)", line)
    if m:
        cand[int(m.group(1))] = (float(m.group(2)), m.group(3))
base = {int(r["game"]): (float(r["our_score"]), r["our_color"]) for r in csv.DictReader(open(base_csv))}
common = sorted(set(cand) & set(base))
mism = sum(cand[g][1] != base[g][1] for g in common)
print("paired games %d (colour mismatches %d)" % (len(common), mism))
for role, col in (("ALL", None), ("defending (ours White)", "white"), ("converting (ours Black)", "black")):
    gs = [g for g in common if col is None or base[g][1] == col]
    d = [cand[g][0] - base[g][0] for g in gs]
    n = len(d)
    m = sum(d) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in d) / (n - 1))
    print("  %-24s n %3d  base %5.1f%%  cand %5.1f%%  diff %+.2fpp ± %.2f  (z %.2f)" % (
        role, n, 100 * sum(base[g][0] for g in gs) / n, 100 * sum(cand[g][0] for g in gs) / n, 100 * m,
        196 * sd / math.sqrt(n), m / (sd / math.sqrt(n)) if sd else 0))
