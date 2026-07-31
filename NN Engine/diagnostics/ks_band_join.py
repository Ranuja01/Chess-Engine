# -*- coding: utf-8 -*-
"""Join ks_band_off/on with the STS move-diff (ksoff vs kson results) and cross-tab:
does the STS regression concentrate where KS newly fires (off KS==0 -> on KS!=0)?

  pyrun diagnostics/ks_band_join.py
Needs: results/ks_band_off.csv, ks_band_on.csv, sts_results_ksoff.csv, sts_results_kson.csv.
"""
import os, csv
THIS = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(THIS, "results")


def load(name, key="idx"):
    d = {}
    with open(os.path.join(RES, name)) as f:
        for r in csv.DictReader(f):
            d[int(r[key])] = r
    return d


band_off = load("ks_band_off.csv")
band_on = load("ks_band_on.csv")
sts_off = load("sts_results_ksoff.csv")
sts_on = load("sts_results_kson.csv")


def si(x):
    try:
        return int(x)
    except (ValueError, TypeError):
        return None


# Classify each position by KS activation transition and by STS score change.
cells = {}          # (fires, outcome) -> count
fires_regressed = []
for idx in sorted(set(band_off) & set(band_on) & set(sts_off) & set(sts_on)):
    ks_off = abs(float(band_off[idx]["ks"]))
    ks_on = abs(float(band_on[idx]["ks"]))
    fires = "newly-fires" if (ks_off < 0.005 and ks_on >= 0.005) else \
            ("already-on" if ks_off >= 0.005 else "stays-zero")
    bs, ts = si(sts_off[idx]["score"]), si(sts_on[idx]["score"])
    if bs is None or ts is None:
        continue
    outcome = "regressed" if ts < bs else ("gained" if ts > bs else "same")
    cells[(fires, outcome)] = cells.get((fires, outcome), 0) + 1
    if fires == "newly-fires" and outcome == "regressed":
        uw = float(band_off[idx]["unitsW"]); ub = float(band_off[idx]["unitsB"])
        fires_regressed.append((idx, uw, ub, ks_on, bs - ts))

fires_order = ["newly-fires", "already-on", "stays-zero"]
out_order = ["regressed", "gained", "same"]
print("KS activation x STS outcome (count of positions):\n")
print("%-14s %10s %8s %8s %8s" % ("", "regressed", "gained", "same", "TOTAL"))
for fr in fires_order:
    row = [cells.get((fr, o), 0) for o in out_order]
    print("%-14s %10d %8d %8d %8d" % (fr, row[0], row[1], row[2], sum(row)))
tot = [sum(cells.get((fr, o), 0) for fr in fires_order) for o in out_order]
print("%-14s %10d %8d %8d %8d" % ("TOTAL", tot[0], tot[1], tot[2], sum(tot)))

# Net STS points lost among newly-firing positions vs the rest.
def net(fr):
    n = 0
    for idx in set(band_off) & set(band_on) & set(sts_off) & set(sts_on):
        ks_off = abs(float(band_off[idx]["ks"])); ks_on = abs(float(band_on[idx]["ks"]))
        f = "newly-fires" if (ks_off < 0.005 and ks_on >= 0.005) else \
            ("already-on" if ks_off >= 0.005 else "stays-zero")
        if f != fr:
            continue
        bs, ts = si(sts_off[idx]["score"]), si(sts_on[idx]["score"])
        if bs is not None and ts is not None:
            n += ts - bs
    return n

print("\nNet STS points (test - base) by activation class:")
for fr in fires_order:
    print("  %-14s %+d" % (fr, net(fr)))

print("\nnewly-fires AND regressed (idx, off unitsW, off unitsB, on|KS|, pts lost):")
for idx, uw, ub, kon, lost in sorted(fires_regressed, key=lambda t: -t[4]):
    print("  %4d  W%5.1f B%5.1f  KS=%.2f  -%d" % (idx, uw, ub, kon, lost))
