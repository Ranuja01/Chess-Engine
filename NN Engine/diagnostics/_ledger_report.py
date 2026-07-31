# -*- coding: utf-8 -*-
"""Summarise the collapse-reduction-ledger runs: score + collapse counts per arm/seed.

Verdict rule (dev_notes/collapse-reduction-ledger.md): two-sided -- (a) the target collapse class shrinks
AND (b) prior-fix classes do not resurface. Seed variance ~13% >> 3.5% SE, so the collapse COUNT trend
across seeds is the leading indicator, not one seed's score.

Run: pyrun diagnostics/_ledger_report.py
"""
import os, csv, glob

THIS = os.path.dirname(os.path.abspath(__file__))
GAMES = os.path.join(os.path.dirname(THIS), "selfplay", "games")

ARMS = [("ledger_base_s0", "BASE  seed0"), ("ledger_v3_s0", "V3    seed0"),
        ("ledger_base_s1", "BASE  seed1"), ("ledger_v3_s1", "V3    seed1")]

print(f"{'arm':<14}{'games':>7}{'score':>9}{'score%':>9}{'collapses':>11}{'coll/100g':>11}")
print("-" * 62)
summary = {}
for tag, label in ARMS:
    d = os.path.join(GAMES, tag)
    res = os.path.join(d, "results.csv")
    col = os.path.join(d, "collapses.csv")
    if not os.path.isfile(res):
        print(f"{label:<14}{'(no results.csv - run missing/failed)':>48}")
        continue
    rows = [r for r in csv.DictReader(open(res)) if r.get("our_score")]
    n = len(rows)
    pts = sum(float(r["our_score"]) for r in rows)
    ncol = 0
    if os.path.isfile(col):
        with open(col) as f:
            ncol = max(0, sum(1 for _ in f) - 1)
    pct = (100.0 * pts / n) if n else 0.0
    per100 = (100.0 * ncol / n) if n else 0.0
    summary[tag] = (n, pts, pct, ncol, per100)
    print(f"{label:<14}{n:>7}{pts:>9.1f}{pct:>8.1f}%{ncol:>11}{per100:>11.1f}")

print()
for s in ("s0", "s1"):
    b, v = summary.get(f"ledger_base_{s}"), summary.get(f"ledger_v3_{s}")
    if b and v:
        print(f"seed {s[1]}:  score {b[2]:.1f}% -> {v[2]:.1f}%  ({v[2]-b[2]:+.1f}pp)   "
              f"collapses/100g {b[4]:.1f} -> {v[4]:.1f}  ({v[4]-b[4]:+.1f})")
bs = [summary[k] for k in ("ledger_base_s0", "ledger_base_s1") if k in summary]
vs_ = [summary[k] for k in ("ledger_v3_s0", "ledger_v3_s1") if k in summary]
if bs and vs_:
    bp = sum(x[4] for x in bs) / len(bs)
    vp = sum(x[4] for x in vs_) / len(vs_)
    bsc = sum(x[2] for x in bs) / len(bs)
    vsc = sum(x[2] for x in vs_) / len(vs_)
    print(f"\nPOOLED  score {bsc:.1f}% -> {vsc:.1f}% ({vsc-bsc:+.1f}pp)   "
          f"collapses/100g {bp:.1f} -> {vp:.1f} ({vp-bp:+.1f}, {100.0*(vp-bp)/bp if bp else 0:+.1f}%)")
    print("\nLedger rule: collapse-COUNT trend is the leading indicator; a single seed's score is unreliable.")
