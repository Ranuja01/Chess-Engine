#!/usr/bin/env python3
"""Split the over-read bench into train/held-out BY GAME (anti-overfit for the conditional-damp fit).
Positions from one game are near-duplicates (run-up FENs) -- a position-level split leaks siblings
across the boundary, so the split key is (tag, game). Deterministic (md5 of the key, no seed state):
~70% of games -> overread_bench_train.csv, the rest -> overread_bench_holdout.csv.
Requires the `game` column (re-run mine_overreads.py if the bench predates it). No engine, no games.
"""
import csv, hashlib, os, sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # NN Engine/
src = os.path.join(BASE, "diagnostics", "overread_bench.csv")
rows = list(csv.DictReader(open(src)))
if not rows or "game" not in rows[0]:
    sys.exit("overread_bench.csv has no `game` column -- re-run diagnostics/mine_overreads.py first")

def is_train(r):
    key = "%s/%s" % (r["tag"], r["game"])
    return int(hashlib.md5(key.encode()).hexdigest(), 16) % 100 < 70

header = list(rows[0].keys())
splits = {"train": [r for r in rows if is_train(r)]}
splits["holdout"] = [r for r in rows if not is_train(r)]
for name, part in splits.items():
    out = os.path.join(BASE, "diagnostics", "overread_bench_%s.csv" % name)
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=header)
        w.writeheader()
        w.writerows(part)
    games = len(set((r["tag"], r["game"]) for r in part))
    print("%s: %d positions / %d games -> %s" % (name, len(part), games, out))
