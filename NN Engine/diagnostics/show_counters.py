"""Print engine diagnostic counter lines from a bench's captured stderr.

The `wac`/`sts` runner subs grep their stderr for a fixed set of patterns (solves, nodes, ebf,
cutoff histogram, singular), so any other instrumentation the engine emits -- fire counters in
particular -- never reaches the task output. This reads the raw .err file the sub leaves behind
and prints the lines matching a substring, aggregating repeats so a per-position bench collapses
to one number per counter.

Usage: show_counters.py <err_file> <substring> [<substring> ...]

@author: Ranuja Pinnaduwage
"""

import re
import sys


def main() -> int:
    if len(sys.argv) < 3:
        print("usage: show_counters.py <err_file> <substring> [...]", file=sys.stderr)
        return 2

    path, needles = sys.argv[1], sys.argv[2:]
    try:
        with open(path, "r", errors="replace") as fh:
            lines = fh.readlines()
    except OSError as exc:
        print(f"cannot read {path}: {exc}", file=sys.stderr)
        return 1

    for needle in needles:
        hits = [ln.strip() for ln in lines if needle in ln]
        print(f"=== {needle}: {len(hits)} line(s)")
        if not hits:
            print("    (none emitted)")
            continue

        # Sum every key=<int> across the matching lines; a bench emits one line per position.
        totals: dict[str, int] = {}
        for ln in hits:
            for key, val in re.findall(r"(\w+)=(-?\d+)\b", ln):
                totals[key] = totals.get(key, 0) + int(val)

        print(f"    last : {hits[-1]}")
        if totals:
            print("    total: " + " ".join(f"{k}={v}" for k, v in totals.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
