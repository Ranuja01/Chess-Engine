# -*- coding: utf-8 -*-
"""Unify EVERY game-run collapses.csv under selfplay/games/ into one master collapse dataset, tagged by config
family + seed, so per-class verdicts and cross-run vanish-attribution have a single source of truth. Pure
pooling (no engine / no SF) -- classification is a separate step (classify_collapses.py). Deterministic games
mean (config-family, seed, game, our_color) identifies the SAME game across runs, which is the attribution key.

Writes ks_sets/collapse_dataset.csv (one row per collapse, all columns + family/seed). Run:
  pyrun diagnostics/collect_collapses.py            # all dirs
  pyrun diagnostics/collect_collapses.py ab_        # only dirs whose tag starts with a prefix
"""
import os, sys, csv, re

THIS = os.path.dirname(os.path.abspath(__file__))
GAMES = os.path.join(THIS, "..", "selfplay", "games")
OUT = os.path.join(THIS, "ks_sets", "collapse_dataset.csv")

prefixes = [a for a in sys.argv[1:] if not a.startswith("-")]

def parse_tag(dirname):
    """Split a run dir like 'ab_kauf_s3' -> (family='ab_kauf', seed=3). Seed = trailing _s<N>, else None."""
    m = re.match(r"^(.*)_s(\d+)$", dirname)
    if m:
        return m.group(1), int(m.group(2))
    return dirname, None

COLS = ["family", "seed", "game", "our_color", "result", "peak_ply", "peak_eval", "peak_move",
        "drop_ply", "drop_eval", "swing", "decision_fen", "drop_fen", "src_dir"]

rows = []
ndirs = 0
for d in sorted(os.listdir(GAMES)):
    dp = os.path.join(GAMES, d)
    cp = os.path.join(dp, "collapses.csv")
    if not os.path.isdir(dp) or not os.path.exists(cp):
        continue
    if prefixes and not any(d.startswith(p) for p in prefixes):
        continue
    family, seed = parse_tag(d)
    ndirs += 1
    for r in csv.DictReader(open(cp)):
        try:
            peak = int(r.get("peak_eval") or 0); dropv = int(r.get("drop_eval") or 0)
        except ValueError:
            peak = dropv = 0
        rows.append({
            "family": family, "seed": ("" if seed is None else seed),
            "game": (r.get("game") or "").strip(), "our_color": (r.get("our_color") or "").strip(),
            "result": (r.get("result") or "").strip(),
            "peak_ply": (r.get("peak_ply") or "").strip(), "peak_eval": peak,
            "peak_move": (r.get("peak_move") or "").strip(), "drop_ply": (r.get("drop_ply") or "").strip(),
            "drop_eval": dropv, "swing": peak - dropv,
            "decision_fen": (r.get("decision_fen") or "").strip(),
            "drop_fen": (r.get("drop_fen") or "").strip(), "src_dir": d,
        })

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=COLS)
    w.writeheader()
    w.writerows(rows)

# summary: collapses per family (across its seeds)
from collections import Counter
per_family = Counter(r["family"] for r in rows)
print("collected %d collapses from %d run dirs -> %s" % (len(rows), ndirs, OUT))
print("\ncollapses per config family (summed over its seeds):")
for fam, n in sorted(per_family.items(), key=lambda x: -x[1]):
    seeds = sorted({r["seed"] for r in rows if r["family"] == fam and r["seed"] != ""})
    print("  %-22s %5d   seeds=%s" % (fam, n, seeds or "-"))
