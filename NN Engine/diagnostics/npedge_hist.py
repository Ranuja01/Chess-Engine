# -*- coding: utf-8 -*-
"""Place the npedge damp ramp (LO/HI) by the DATA, not by optimization (Fable's "check the histogram").

Reads corpus_ks.csv, rebuilds the fantasy/real classes exactly as fit_fantasy.py does, and prints the
npedge distribution of each class + the counter_pressure co-condition. We want to see:
  - a SPARSE valley between the fantasy cluster (~+12) and the real cluster (~+437) => LO/HI live there.
  - where a hard cutoff would clip real wins (the boundary the ramp must clear).

  python diagnostics/npedge_hist.py diagnostics/corpus_ks.csv
"""
import sys, csv
import numpy as np


def cls(rows):
    for r in rows:
        g = lambda k: float(r.get(k, 0) or 0)
        r["_ourstat"] = g("over_read_cp") + g("sf18_static_cp")
        r["_sf"] = g("sf18_static_cp")
        r["counter_pressure"] = g("off_opp") - g("def_us")
    fant = [r for r in rows if r["is_collapse"] == "1" and r["_ourstat"] >= 150 and r["_sf"] <= 50]
    real = [r for r in rows if r["is_collapse"] == "0" and r["_ourstat"] >= 150 and r["_sf"] >= 150]
    return fant, real


def describe(name, rows, key):
    v = np.array([float(r.get(key, 0) or 0) for r in rows])
    if len(v) == 0:
        print(f"  {name:>8} {key}: (empty)"); return
    qs = np.percentile(v, [0, 10, 25, 50, 75, 90, 100])
    print(f"  {name:>8} {key}: n={len(v):3d} mean={v.mean():+7.0f}  "
          f"min/p10/p25/med/p75/p90/max = " + " ".join(f"{q:+.0f}" for q in qs))


def hist(name, rows, key, edges):
    v = np.array([float(r.get(key, 0) or 0) for r in rows])
    counts, _ = np.histogram(v, bins=edges)
    print(f"  {name:>8} {key} hist: " + " ".join(f"[{edges[i]:+.0f},{edges[i+1]:+.0f})={c}" for i, c in enumerate(counts)))


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "diagnostics/corpus_ks.csv"
    rows = list(csv.DictReader(open(path)))
    fant, real = cls(rows)
    print(f"[npedge] FANTASY={len(fant)}  REAL={len(real)}\n")
    print("== npedge (non-pawn / piece-material edge, mover-POV cp) ==")
    describe("FANTASY", fant, "npedge")
    describe("REAL", real, "npedge")
    edges = [-1000, 0, 50, 80, 150, 250, 300, 400, 600, 3000]
    hist("FANTASY", fant, "npedge", edges)
    hist("REAL", real, "npedge", edges)
    print("\n== counter_pressure (off_opp - def_us; higher => opp has more counterplay) ==")
    describe("FANTASY", fant, "counter_pressure")
    describe("REAL", real, "counter_pressure")
    # How much of REAL would a hard/ramped gate misclassify at candidate LO/HI?
    print("\n== gate leakage check (fraction of each class below npedge thresholds) ==")
    for thr in (50, 80, 150, 250, 300):
        ff = np.mean([1.0 for r in fant if float(r.get("npedge", 0) or 0) < thr]) if fant else 0
        rf = np.mean([1.0 for r in real if float(r.get("npedge", 0) or 0) < thr]) if real else 0
        nf = sum(1 for r in fant if float(r.get("npedge", 0) or 0) < thr)
        nr = sum(1 for r in real if float(r.get("npedge", 0) or 0) < thr)
        print(f"  npedge<{thr:4d}:  FANTASY {nf}/{len(fant)}  REAL {nr}/{len(real)}  "
              f"(want high fantasy, ~0 real => damp fires on fantasy, spares real)")


if __name__ == "__main__":
    main()
