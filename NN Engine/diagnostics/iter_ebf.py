"""Recover the REAL per-iteration branching factor from ENABLE_ITER_LOG output.

The engine's headline "ebf" is pow(cumulative_nodes, 1/depth_limit), where the counter absorbs the root
pre-search, aspiration re-searches, qsearch and TT-hit bookkeeping, and depth_limit is the loop's EXIT
value -- so it is neither nodes(d)/nodes(d-1) nor comparable to the figures other engines publish.

ENABLE_ITER_LOG emits one `[iter] d=<depth> cum_nodes=<n>` line per iterative-deepening iteration.
Differencing consecutive lines within a position gives that iteration's own node count; the ratio of
successive per-iteration counts is the branching factor as normally defined.

A position boundary is detected when the depth stops increasing (the next search restarts at a lower or
equal depth).

Usage: iter_ebf.py <err_file>

@author: Ranuja Pinnaduwage
"""

import re
import statistics
import sys

LINE = re.compile(r"\[iter\] d=(\d+) cum_nodes=(\d+)")


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: iter_ebf.py <err_file>", file=sys.stderr)
        return 2

    try:
        with open(sys.argv[1], "r", errors="replace") as fh:
            raw = [(int(m.group(1)), int(m.group(2)))
                   for m in (LINE.search(ln) for ln in fh) if m]
    except OSError as exc:
        print(f"cannot read {sys.argv[1]}: {exc}", file=sys.stderr)
        return 1

    if not raw:
        print("no [iter] lines found -- was ENABLE_ITER_LOG=1 set?", file=sys.stderr)
        return 1

    # Split into positions, then difference cumulative counts within each.
    per_depth: dict[int, list[int]] = {}
    prev_d = prev_cum = None
    for d, cum in raw:
        if prev_d is None or d <= prev_d:
            prev_cum = 0                      # new position: counter restarts
        incr = cum - (prev_cum or 0)
        if incr > 0:
            per_depth.setdefault(d, []).append(incr)
        prev_d, prev_cum = d, cum

    depths = sorted(per_depth)
    print(f"positions parsed: {sum(1 for i, (d, _) in enumerate(raw) if i == 0 or d <= raw[i-1][0])}")
    print(f"{'depth':>6} {'iters':>7} {'median nodes':>14} {'ratio vs d-1':>14}")
    ratios = []
    for d in depths:
        med = statistics.median(per_depth[d])
        prev = per_depth.get(d - 1)
        if prev:
            r = med / statistics.median(prev)
            ratios.append(r)
            print(f"{d:>6} {len(per_depth[d]):>7} {med:>14,.0f} {r:>14.2f}")
        else:
            print(f"{d:>6} {len(per_depth[d]):>7} {med:>14,.0f} {'-':>14}")

    # The deeper iterations are the meaningful ones; early plies are dominated by fixed overheads.
    deep = [r for d, r in zip(depths[1:], ratios) if d >= 6]
    if deep:
        print(f"\nREAL EBF (median ratio, d>=6): {statistics.median(deep):.2f}")
    if ratios:
        print(f"REAL EBF (median ratio, all depths): {statistics.median(ratios):.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
