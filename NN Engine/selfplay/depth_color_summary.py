# -*- coding: utf-8 -*-
"""
Summarize the fixed-depth color-bias sweep (depth_color_sweep.sh output).

Reads games/depth_<N>/summary.csv for every depth run and answers: does the identical-engine
White vs Black win-rate FLIP with search-depth parity? Prints, per depth, White% / Black% / draw%
and the White-minus-Black margin, then a per-opening winner grid across depths, then a parity
verdict (does the sign of (White-Black) alternate even/odd, or trend, or stay flat?).

Run in WSL from NN Engine/selfplay/ (pure data analysis, no engine):
    python depth_color_summary.py
    python depth_color_summary.py --glob "depth_*"
"""

import os
import csv
import glob
import argparse

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
GAMES_DIR = os.path.join(THIS_DIR, "games")


def load_depth(tag_dir):
    """Return (white_wins, black_wins, draws, crashes, {opening_idx: [letters]}) from a summary.csv."""
    path = os.path.join(tag_dir, "summary.csv")
    if not os.path.exists(path):
        return None
    ww = bw = dr = cr = 0
    per_op = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            res, reason = row.get("result", ""), row.get("reason", "") or ""
            op = row.get("opening_idx")
            if "crash" in reason:
                cr += 1
                let = "C"
            elif res == "1-0":
                ww += 1
                let = "W"
            elif res == "0-1":
                bw += 1
                let = "B"
            elif "1/2" in res:
                dr += 1
                let = "D"
            else:
                let = "?"
            per_op.setdefault(op, []).append(let)
    return ww, bw, dr, cr, per_op


def main():
    ap = argparse.ArgumentParser(description="Summarize the fixed-depth color sweep.")
    ap.add_argument("--glob", default="depth_*", help="glob under games/ for depth dirs (default depth_*)")
    args = ap.parse_args()

    dirs = sorted(glob.glob(os.path.join(GAMES_DIR, args.glob)))
    rows = []
    for d in dirs:
        name = os.path.basename(d)
        try:
            depth = int(name.split("_")[-1])
        except ValueError:
            continue
        loaded = load_depth(d)
        if loaded:
            rows.append((depth, name) + loaded)
    rows.sort()
    if not rows:
        print("no games/%s/summary.csv found -- run depth_color_sweep.sh first" % args.glob)
        return

    print("=" * 72)
    print("FIXED-DEPTH COLOR-BIAS SWEEP")
    print("=" * 72)
    print("%6s %6s %8s %8s %7s %9s %8s" % ("depth", "games", "White%", "Black%", "draw%", "W-B (pp)", "parity"))
    margins = {}
    for depth, name, ww, bw, dr, cr, _ in rows:
        n = ww + bw + dr + cr
        if n == 0:
            continue
        wpct, bpct, dpct = 100 * ww / n, 100 * bw / n, 100 * dr / n
        margin = wpct - bpct
        margins[depth] = margin
        print("%6d %6d %8.0f %8.0f %7.0f %9.0f %8s"
              % (depth, n, wpct, bpct, dpct, margin, "even" if depth % 2 == 0 else "odd"))

    # per-opening winner grid across depths
    all_ops = sorted({op for *_, per_op in rows for op in per_op}, key=lambda x: (x is None, x))
    depths = [r[0] for r in rows]
    print("\nper-opening winner by depth (first color-run; W=White won, B=Black won, D=draw, C=crash):")
    print("  %-4s %s" % ("op", " ".join("d%-2d" % d for d in depths)))
    by_depth = {r[0]: r[6] for r in rows}
    for op in all_ops:
        cells = []
        for d in depths:
            outs = by_depth[d].get(op, [])
            cells.append("%-3s" % (outs[0] if outs else "-"))
        print("  %-4s %s" % (op, " ".join(cells)))

    # parity verdict
    print("\nparity check (White-minus-Black margin by depth):")
    even = [margins[d] for d in sorted(margins) if d % 2 == 0]
    odd = [margins[d] for d in sorted(margins) if d % 2 == 1]
    if even and odd:
        em, om = sum(even) / len(even), sum(odd) / len(odd)
        print("  even depths mean W-B = %+.0fpp | odd depths mean W-B = %+.0fpp | gap = %+.0fpp"
              % (em, om, em - om))
        signs = [1 if margins[d] > 0 else -1 for d in sorted(margins)]
        alternating = all(signs[i] != signs[i + 1] for i in range(len(signs) - 1))
        print("  margin sign sequence by depth %s: %s"
              % (sorted(margins), " ".join("+" if s > 0 else "-" for s in signs)))
        if alternating:
            print("  => VERDICT: margin sign ALTERNATES with depth -> consistent with even/odd parity (hypothesis a).")
        elif abs(em - om) > 25:
            print("  => VERDICT: strong even/odd split (>25pp) -> parity-driven (hypothesis a).")
        else:
            print("  => VERDICT: no clean parity split -> flip is not pure parity; inspect the trend / eval-symmetry probe.")
    else:
        print("  need both even and odd depths to test parity.")


if __name__ == "__main__":
    main()
