# -*- coding: utf-8 -*-
"""Can a game VENUE discriminate? (2026-10-18; memory narrow-items-are-judged-where-they-fire, count-the-resolvable-effect)

  python diagnostics/_pair_discrimination.py selfplay/games/<tag>/summary.csv [...]

Each opening is played twice with colours swapped (tournament.py). A pair whose two games split 1-1 by COLOUR (each side won
with the winning colour) says nothing about the candidate; only the rest DISCRIMINATE. Prints: void games, games resigned within
4 plies (a decided start), pair types, and the candidate's score on the discriminating pairs. ☠️ #74's random pawn-dominated
starts: 87/600 pairs discriminated — the venue, not the candidate, produced that null.
"""
import csv, sys, collections
for path in sys.argv[1:]:
    rows = list(csv.DictReader(open(path, newline="")))
    void = [r for r in rows if r["p1_score"] == ""]
    rows = [r for r in rows if r["p1_score"] != ""]
    pairs = collections.defaultdict(list)
    for r in rows:
        pairs[r["opening_idx"]].append(r)
    quick = sum(1 for r in rows if int(r["plies"]) <= 4 and "resigned" in r["reason"])
    kinds = collections.Counter()
    s_disc, n_disc = 0.0, 0
    for g in pairs.values():
        if len(g) != 2:
            continue
        s = sum(float(x["p1_score"]) for x in g)
        # ☠️ 10-18 fix: a pair summing to 1.0 is uninformative WHETHER it is a 1-0 colour split OR two draws — only a pair where
        # one engine scored more over the two games (sum 0, 0.5, 1.5, 2) carries information about the candidate.
        if s == 1.0:
            both_draw = all(float(x["p1_score"]) == 0.5 for x in g)
            kinds["uninformative: both drawn" if both_draw else "uninformative: split 1-1 by colour"] += 1
        else:
            kinds["INFORMATIVE (sum %.1f)" % s] += 1
            s_disc += s; n_disc += 1
    n_pairs = sum(kinds.values())
    print("%s: games %d · void %d · resigned within 4 plies %d · pairs %d, INFORMATIVE %d (%.0f%%)"
          % (path, len(rows), len(void), quick, n_pairs, n_disc, 100 * n_disc / max(1, n_pairs)))
    for k, v in sorted(kinds.items(), key=lambda x: -x[1]):
        print("    %-36s %4d" % (k, v))
    if n_disc:
        print("  candidate on informative pairs: %.1f / %d = %.1f%%" % (s_disc, 2 * n_disc, 100 * s_disc / (2 * n_disc)))
