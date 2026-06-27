"""Aggregate breakdown_tag output: for moderate over-reads (div 2.5..50, excludes mate-flips),
tally which eval term dominates, split midgame vs endgame, and track passed_pawn_support specifically."""
import sys, re
from collections import Counter

SKIP = {"material", "advanced_endgame_total"}
mid = Counter(); end = Counter()
mid_n = end_n = 0
pp_sig_mid = 0          # midgame over-reads where |passed_pawn_support| >= 0.5
pp_wrong_mid = 0        # ... and pp favours the OVER-read side (compounds the error)
cur = None

def flush(c):
    global mid_n, end_n, pp_sig_mid, pp_wrong_mid
    if not c or c.get("div") is None or c.get("phase") is None or not c.get("terms"):
        return
    div = c["div"]
    if div < 2.5 or div > 50:   # exclude tiny + mate-flips
        return
    terms = c["terms"]
    dom = max(terms.items(), key=lambda kv: abs(kv[1]))[0]
    gapsign = c.get("gapsign", 0)   # sign of (our - SF): + => we over-read White
    pp = terms.get("passed_pawn_support", 0.0)
    if c["phase"] < 64:
        mid[dom] += 1; mid_n += 1
        if abs(pp) >= 0.5:
            pp_sig_mid += 1
            # pp "wrong" if it pushes our eval in the same direction as the over-read gap
            if gapsign != 0 and (pp > 0) == (gapsign > 0):
                pp_wrong_mid += 1
    else:
        end[dom] += 1; end_n += 1

for line in sys.stdin:
    m = re.search(r'ply\d+.*\[div ([\d.]+)\]', line)
    if m:
        flush(cur); cur = {"div": float(m.group(1)), "terms": {}}
        continue
    if cur is None:
        continue
    m = re.search(r'our_static\s+([+-][\d.]+)\s+\|\s+SF_static\s+([+-][\d.]+)', line)
    if m:
        cur["gapsign"] = 1 if (float(m.group(1)) - float(m.group(2))) > 0 else -1
        continue
    m = re.search(r'phase_score (\d+)', line)
    if m:
        cur["phase"] = int(m.group(1)); continue
    m = re.match(r'\s{4,}([A-Za-z_]+)\s+([+-][\d.]+)\s*$', line)
    if m and m.group(1) not in SKIP:
        cur["terms"][m.group(1)] = float(m.group(2))
flush(cur)

print(f"=== MIDGAME over-reads (phase<64, div 2.5-50), n={mid_n} — dominant term ===")
for t, c in mid.most_common():
    print(f"  {t:<22} {c:>4}  ({100*c/max(mid_n,1):.0f}%)")
print(f"  passed_pawn_support SIGNIFICANT (|pp|>=0.5): {pp_sig_mid}/{mid_n} ({100*pp_sig_mid/max(mid_n,1):.0f}%), "
      f"of which compounding the over-read: {pp_wrong_mid}")
print(f"=== ENDGAME over-reads (phase>=64, div 2.5-50), n={end_n} — dominant term ===")
for t, c in end.most_common():
    print(f"  {t:<22} {c:>4}  ({100*c/max(end_n,1):.0f}%)")
