# -*- coding: utf-8 -*-
"""Decompose a vs_sf11 diagnostic run: our engine vs classical SF11 per-move search stats.

Reads selfplay/games/<tag>/game_*.jsonl (each move logs depth + nodes for both sides) and reports, for OUR
engine vs SF11: mean/median search depth, mean nodes, and the nodes-at-equal-depth efficiency ratio. In the
equal-depth arm both depths are pinned, so the NODES ratio = our pruning/move-ordering inefficiency vs a
great HCE (how many more nodes we burn for the same depth). In the equal-time arm the DEPTH gap = how many
plies shallower we search at the same time. Pair with the run's score line (Elo) for the full picture.

    pyrun diagnostics/analyze_sf_games.py <tag>
"""
import os
import sys
import glob
import json
import statistics as st

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
GAMES = os.path.join(os.path.dirname(THIS_DIR), "selfplay", "games")


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "t11_d12"
    paths = sorted(glob.glob(os.path.join(GAMES, tag, "game_*.jsonl")))
    if not paths:
        print("no games under %s" % os.path.join(GAMES, tag))
        return 1
    our_d, our_n, sf_d, sf_n = [], [], [], []
    # nodes bucketed by depth, to compare efficiency at matched depth
    our_by_d, sf_by_d = {}, {}
    for p in paths:
        with open(p) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                d, n, is_sf = r.get("depth"), r.get("nodes"), r.get("sf")
                if d is None:
                    continue
                if is_sf:
                    sf_d.append(d)
                    if n is not None:
                        sf_n.append(n); sf_by_d.setdefault(d, []).append(n)
                elif "our_pov_eval" in r:
                    our_d.append(d)
                    if n is not None:
                        our_n.append(n); our_by_d.setdefault(d, []).append(n)

    def line(label, ds, ns):
        if not ds:
            print("  %-6s no data" % label); return
        print("  %-6s moves=%d  depth mean=%.1f med=%d  nodes mean=%s med=%s"
              % (label, len(ds), st.mean(ds), int(st.median(ds)),
                 ("%.0f" % st.mean(ns)) if ns else "n/a",
                 ("%d" % int(st.median(ns))) if ns else "n/a"))

    print("vs_sf11 search decomposition — tag=%s  games=%d" % (tag, len(paths)))
    line("OURS", our_d, our_n)
    line("SF11", sf_d, sf_n)
    # nodes-at-matched-depth efficiency (where both engines have data at the same depth)
    common = sorted(set(our_by_d) & set(sf_by_d))
    if common:
        print("  nodes @ matched depth (ours / SF11 = our inefficiency factor):")
        for d in common:
            om, sm = st.mean(our_by_d[d]), st.mean(sf_by_d[d])
            if sm > 0:
                print("    depth %2d:  ours %9.0f / SF11 %9.0f  = %5.1fx" % (d, om, sm, om / sm))
    return 0


if __name__ == "__main__":
    sys.exit(main())
