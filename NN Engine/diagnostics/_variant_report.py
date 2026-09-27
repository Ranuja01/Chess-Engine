# -*- coding: utf-8 -*-
"""Per-family report for a tournament or SPRT played from a FEN-start book (variant or odds starts).

Joins games/<tag>/summary.csv (opening_idx, p1_color, p1_score) with the book's `; tag` families. For every
family: games and P1's score with an Elo estimate. For ODDS books (families `odds_*`, where the FEN's WHITE side
holds the handicap) the score is also split by role: P1 playing the side WITH the extra material (converting)
vs the side WITHOUT it (defending) -- the owner's question of how each arm handles a material edge.

  python diagnostics/_variant_report.py TAG=<games tag> BOOK=selfplay/openings_variant.txt
"""
import os, sys, csv, math
from collections import defaultdict

KV = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
ENGINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ENGINE, "selfplay"))
from tournament import load_openings, elo_from_score  # noqa: E402

book = load_openings(os.path.join(ENGINE, KV["BOOK"]))
rows = list(csv.DictReader(open(os.path.join(ENGINE, "selfplay", "games", KV["TAG"], "summary.csv"))))
fam = defaultdict(list)
role = defaultdict(list)
for r in rows:
    if r["p1_score"] in ("", "None"):
        continue
    s = float(r["p1_score"])
    tag = getattr(book[int(r["opening_idx"])], "tag", "") or "untagged"
    fam[tag].append(s)
    fam["ALL"].append(s)
    if tag.startswith("odds_"):
        # White is handicapped in an odds FEN, so P1 holds the EXTRA material when it plays Black.
        role[(tag, "converting" if r["p1_color"] == "black" else "defending")].append(s)
        role[("odds_ALL", "converting" if r["p1_color"] == "black" else "defending")].append(s)


def line(name, xs):
    n = len(xs)
    m = sum(xs) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / max(1, n - 1))
    se = sd / math.sqrt(n) if n > 1 else 0.0
    lo, hi = max(1e-6, m - 1.96 * se), min(1 - 1e-6, m + 1.96 * se)
    return "  %-26s n=%5d  P1 %5.1f%%  elo %+6.1f  [%+6.1f, %+6.1f]" % (
        name, n, 100 * m, elo_from_score(m), elo_from_score(lo), elo_from_score(hi))


print("TAG %s  BOOK %s  (%d scored games)" % (KV["TAG"], KV["BOOK"], len(fam["ALL"])))
for k in sorted(fam, key=lambda k: (k != "ALL", k)):
    print(line(k, fam[k]))
if role:
    print("ODDS by role (P1 converting = P1 holds the extra material):")
    for k in sorted(role):
        print(line("%s / %s" % k, role[k]))

# PAIR (pentanomial) view. Each start is played twice with colours swapped; a pair summing to exactly 1.0
# (e.g. each side wins with the extra material) says NOTHING about which arm is stronger -- on a lopsided start
# nearly every pair splits, so the score sits at 50% however much either arm improved (owner, 2026-09-26).
# Only non-1.0 pairs carry information; the share of them shows which families are saturated.
pairs = defaultdict(dict)
for r in rows:
    if r["p1_score"] in ("", "None"):
        continue
    g = int(r["game"])
    pairs[(int(r["opening_idx"]), g // 2)][r["p1_color"]] = float(r["p1_score"])
pent = defaultdict(lambda: [0, 0, 0, 0, 0])
for (oi, _), d in pairs.items():
    if len(d) != 2:
        continue
    s = d["white"] + d["black"]
    tag = getattr(book[oi], "tag", "") or "untagged"
    for k in (tag, "ALL"):
        pent[k][int(round(s * 2))] += 1
print("PAIRS (P1 pair score 0 / 0.5 / 1 / 1.5 / 2; informative = not 1.0):")
for k in sorted(pent, key=lambda k: (k != "ALL", k)):
    c = pent[k]
    n = sum(c)
    inf = n - c[2]
    net = (c[3] + 2 * c[4]) - (c[1] + 2 * c[0])
    print("  %-26s pairs %4d  [%d %d %d %d %d]  informative %4d (%3.0f%%)  net P1 half-points in them %+d"
          % (k, n, c[0], c[1], c[2], c[3], c[4], inf, 100.0 * inf / max(1, n), net))
