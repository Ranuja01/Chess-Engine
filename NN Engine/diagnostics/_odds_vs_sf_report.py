# -*- coding: utf-8 -*-
"""Per-family, per-role report for a vs_sf.py run played from the ODDS book.

vs_sf.py writes no per-opening summary, so this rebuilds each game's start from the same schedule the run used
(tournament.schedule(games, n_openings, seed)) and joins it with the `our=` score printed per game. In an odds FEN
White holds the handicap, so our engine holds the EXTRA material when it plays Black ("converting") and the
deficit when it plays White ("defending"). Two runs on the same seed face identical starts and colours, so their
per-role rates compare directly.

  python diagnostics/_odds_vs_sf_report.py LOG=<vs_sf stdout log> GAMES=480 SEED=42 [BOOK=selfplay/openings_odds.txt]
"""
import os, sys, re
from collections import defaultdict

KV = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
ENGINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ENGINE, "selfplay"))
from tournament import load_openings, schedule  # noqa: E402

book = load_openings(os.path.join(ENGINE, KV.get("BOOK", "selfplay/openings_odds.txt")))
sched = schedule(int(KV["GAMES"]), len(book), int(KV["SEED"]))
res = defaultdict(list)
for line in open(KV["LOG"], encoding="utf-8", errors="ignore"):
    m = re.search(r"game (\d+): .*our=([0-9.]+) as (white|black)", line)
    if not m:
        continue
    g, s, col = int(m.group(1)), float(m.group(2)), m.group(3)
    oi, our_white = sched[g]
    assert our_white == (col == "white"), "schedule mismatch at game %d -- wrong GAMES/SEED?" % g
    fam = getattr(book[oi], "tag", "untagged")
    role = "defending" if our_white else "converting"
    for k in ((fam, role), ("odds_ALL", role), (fam, "all"), ("odds_ALL", "all")):
        res[k].append(s)
print("ODDS vs SF -- %s" % os.path.basename(KV["LOG"]))
print("  %-24s %-11s %5s %7s" % ("family", "role", "n", "score"))
for k in sorted(res):
    xs = res[k]
    print("  %-24s %-11s %5d %6.1f%%" % (k[0], k[1], len(xs), 100.0 * sum(xs) / len(xs)))
