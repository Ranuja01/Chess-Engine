# -*- coding: utf-8 -*-
"""Move-flip report — diff two moves_dump.py outputs (default vs a candidate) and decompose the WDL-cploss delta.

Because per-position loss depends ONLY on our chosen move, the aggregate cploss change between two configs is driven
ENTIRELY by positions where our move flipped. This report shows, per stratum: how many positions flipped, and the
win%-swing distribution of those flips (improved vs worsened). The hypothesis under test (2026-07-05 Stage-1 winner):
the collapse stratum improves via a FEW large, beneficial flips (decision-critical -> real Elo), while the neutral/
quiet regression is a few and/or SUB-THRESHOLD flips that would not change a real game's result (benign).

    python diagnostics/move_flip_report.py moves_default.csv moves_winner.csv
"""
import csv
import sys

ORDER = ["collapse", "sts", "neutral", "game"]
BUCKETS = [(0.0, 2.0), (2.0, 5.0), (5.0, 15.0), (15.0, 1e9)]   # |Δwin%| bands on flipped positions
BUCKET_LABELS = ["<2", "2-5", "5-15", ">15"]


def load(path):
    m = {}
    for r in csv.DictReader(open(path)):
        m[r["fen"]] = (r["stratum"], r["uci"], float(r["loss"]))
    return m


def main():
    if len(sys.argv) < 3:
        print("usage: move_flip_report.py moves_default.csv moves_winner.csv"); return
    a = load(sys.argv[1])   # default (baseline)
    b = load(sys.argv[2])   # candidate (winner)
    fens = [f for f in a if f in b]
    missing = len(a) + len(b) - 2 * len(fens)
    if missing:
        print("(note: %d rows not matched across both dumps — reporting on the %d shared)" % (missing, len(fens)))

    per = {}   # stratum -> dict of accumulators
    for f in fens:
        st, ua, la = a[f]
        _,  ub, lb = b[f]
        d = per.setdefault(st, {"n": 0, "flip": 0, "sum_a": 0.0, "sum_b": 0.0,
                                "imp": 0, "wor": 0, "buck_imp": [0]*4, "buck_wor": [0]*4,
                                "sum_swing_imp": 0.0, "sum_swing_wor": 0.0})
        d["n"] += 1; d["sum_a"] += la; d["sum_b"] += lb
        if ua == ub:
            continue                                  # move unchanged -> loss identical, no contribution to the delta
        d["flip"] += 1
        swing = lb - la                               # >0 = worsened (more win% lost), <0 = improved
        mag = abs(swing)
        bi = next(i for i, (lo, hi) in enumerate(BUCKETS) if lo <= mag < hi)
        if swing < 0:
            d["imp"] += 1; d["buck_imp"][bi] += 1; d["sum_swing_imp"] += mag
        else:
            d["wor"] += 1; d["buck_wor"][bi] += 1; d["sum_swing_wor"] += mag

    print("MOVE-FLIP DECOMPOSITION  (loss in win%%; ×1 here, cploss report ×10)")
    print("%-9s %5s %6s %7s  %8s %8s %8s  | flips: imp/wor  swing(imp,wor)  |Δwin%%| buckets<2/2-5/5-15/>15"
          % ("stratum", "n", "flip", "flip%", "mean_def", "mean_cand", "Δmean"))
    tot = {"n": 0, "flip": 0, "sum_a": 0.0, "sum_b": 0.0}
    for st in ORDER + [s for s in per if s not in ORDER]:
        if st not in per:
            continue
        d = per[st]
        for k in tot:
            tot[k] += d[k]
        ma = d["sum_a"] / d["n"] if d["n"] else 0.0
        mb = d["sum_b"] / d["n"] if d["n"] else 0.0
        fr = 100.0 * d["flip"] / d["n"] if d["n"] else 0.0
        avg_imp = d["sum_swing_imp"] / d["imp"] if d["imp"] else 0.0
        avg_wor = d["sum_swing_wor"] / d["wor"] if d["wor"] else 0.0
        print("%-9s %5d %6d %6.1f%%  %8.2f %8.2f %8.2f  | %d/%d  (-%.2f,+%.2f)  imp %s  wor %s"
              % (st, d["n"], d["flip"], fr, ma, mb, mb - ma,
                 d["imp"], d["wor"], avg_imp, avg_wor,
                 "/".join(str(x) for x in d["buck_imp"]),
                 "/".join(str(x) for x in d["buck_wor"])))
    if tot["n"]:
        print("%-9s %5d %6d %6.1f%%  %8.2f %8.2f %8.2f"
              % ("ALL", tot["n"], tot["flip"], 100.0 * tot["flip"] / tot["n"],
                 tot["sum_a"] / tot["n"], tot["sum_b"] / tot["n"],
                 (tot["sum_b"] - tot["sum_a"]) / tot["n"]))
    print("\nRead: collapse should show LARGE beneficial flips (imp swing big, in 5-15/>15 buckets); neutral 'regression'"
          " is benign if its flips are FEW and/or land in the <2 bucket (sub-threshold, unlikely to change a real game).")


if __name__ == "__main__":
    main()
