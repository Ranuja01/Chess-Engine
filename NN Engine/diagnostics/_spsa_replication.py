# Replication verdict for two (or more) cold-start SPSA runs of the same spec.
# For each knob and run: net drift, drift z = net / (step_sd * sqrt(n)), and a tail mean (Polyak average).
# Across runs: direction agreement, size ratio, combined (Stouffer) z = sum(z) / sqrt(k).
# Pre-registered pass rule (EVAL-V2-RETUNE-PLAN): same direction in every run, similar size, combined |z| >= 2.
# Usage: python diagnostics/_spsa_replication.py spsaks1 spsaks2 [TAIL=60] [SPEC=selfplay/spsa_ks_shape.json]
import csv, json, math, os, sys

ED = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
tags = [a for a in sys.argv[1:] if "=" not in a]
kv = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
TAIL = int(kv.get("TAIL", 60))
spec = json.load(open(os.path.join(ED, kv.get("SPEC", "selfplay/spsa_ks_shape.json"))))
names = [k["name"] for k in spec]
init = {k["name"]: float(k["init"]) for k in spec}

runs = {}
for t in tags:
    rows = list(csv.DictReader(open(os.path.join(ED, "selfplay", "games", "%s_log.csv" % t))))
    ys = [float(r["y_plus_score"]) for r in rows]
    runs[t] = {"n": len(rows), "mean_y": sum(ys) / len(ys),
               "series": {k: [init[k]] + [float(r[k]) for r in rows] for k in names}}

print("runs:", ", ".join("%s n=%d mean_y=%.4f" % (t, runs[t]["n"], runs[t]["mean_y"]) for t in tags))
print("tail = last %d iterations\n" % TAIL)
hdr = "%-16s %6s" % ("knob", "start")
for t in tags:
    hdr += " | %-8s %6s %6s %6s" % (t, "end", "tail", "z")
hdr += " | %5s %6s %8s  %s" % ("dir", "ratio", "comb_z", "verdict")
print(hdr)
cand = {}
for k in names:
    line = "%-16s %6.0f" % (k, init[k])
    zs, nets, tails = [], [], []
    for t in tags:
        s = runs[t]["series"][k]
        steps = [b - a for a, b in zip(s, s[1:])]
        n = len(steps)
        mu = sum(steps) / n
        sd = math.sqrt(sum((x - mu) ** 2 for x in steps) / (n - 1)) if n > 1 else 0.0
        net = s[-1] - s[0]
        z = net / (sd * math.sqrt(n)) if sd > 0 else 0.0
        tail = sum(s[-TAIL:]) / len(s[-TAIL:])
        zs.append(z); nets.append(tail - init[k]); tails.append(tail)
        line += " | %-8s %6.0f %6.1f %+6.2f" % ("", s[-1], tail, z)
    same = all(x > 0 for x in nets) or all(x < 0 for x in nets)
    mags = [abs(x) for x in nets]
    ratio = (max(mags) / min(mags)) if min(mags) > 0 else float("inf")
    cz = sum(zs) / math.sqrt(len(zs))
    ok = same and abs(cz) >= 2.0
    line += " | %5s %6.2f %+8.2f  %s" % ("same" if same else "OPP", ratio, cz,
                                          ("PASS" if ok else "fail") + (" (check size)" if ok and ratio > 2 else ""))
    print(line)
    if ok:
        # conservative pick: the smaller of the two tail moves, in the agreed direction
        move = min(nets, key=abs)
        cand[k] = int(round(init[k] + move))
print()
print("candidate (replicated knobs only, smaller tail move):", " ".join("%s=%d" % kv_ for kv_ in cand.items()) or "NONE")
