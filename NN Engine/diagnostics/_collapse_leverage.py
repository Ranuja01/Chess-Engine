# -*- coding: utf-8 -*-
"""Which collapses actually COST US POINTS -- leverage, not error size.

"Where is the eval error biggest" and "where would fixing it win games" are different questions. A collapse
from a winning peak costs 1.0 point if it became a loss and 0.5 if it became a draw, so this weights every
collapse by points forfeited and buckets by phase / class / when in the game it happened. Also reports
whether collapsing EARLY costs more than collapsing late (the "hard to crawl back" hypothesis).

Engine-free: reads ks_sets/collapse_dataset_classified.csv only.
  pyrun diagnostics/_collapse_leverage.py [TAG=vssf_2400]
"""
import os
import sys
import csv

for _a in sys.argv[1:]:
    if '=' in _a:
        _k, _v = _a.split('=', 1)
        os.environ[_k] = _v

import chess

THIS = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(THIS, "ks_sets", "collapse_dataset_classified.csv")
TAG = os.environ.get("TAG", "vssf_2400")

PIECEVAL = {chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}


def npm(board):
    return sum(v * (len(board.pieces(pt, chess.WHITE)) + len(board.pieces(pt, chess.BLACK)))
               for pt, v in PIECEVAL.items())


def phase_bucket(n):
    if n >= 45:
        return "1_opening (45+)"
    if n >= 35:
        return "2_early mid (35-44)"
    if n >= 25:
        return "3_middlegame (25-34)"
    if n >= 15:
        return "4_late mid (15-24)"
    return "5_endgame (<15)"


def points_lost(our_color, result):
    """From a WINNING peak: 1.0 if we lost, 0.5 if drawn, 0.0 if we still won."""
    if result in ("1/2-1/2", "1/2", "draw"):
        return 0.5
    if result == "1-0":
        return 0.0 if our_color == "white" else 1.0
    if result == "0-1":
        return 0.0 if our_color == "black" else 1.0
    return 0.0


def report(title, agg):
    print("\n%s" % title)
    print("  %-26s %5s %8s %9s %10s" % ("bucket", "n", "pts_lost", "pts/collapse", "mean_drop_ply"))
    tot_n = sum(a["n"] for a in agg.values())
    tot_p = sum(a["pts"] for a in agg.values())
    for k in sorted(agg):
        a = agg[k]
        print("  %-26s %5d %8.1f %9.2f %10.0f   (%.0f%% of lost pts)"
              % (k, a["n"], a["pts"], a["pts"] / a["n"], a["ply"] / a["n"],
                 100.0 * a["pts"] / tot_p if tot_p else 0))
    print("  %-26s %5d %8.1f" % ("TOTAL", tot_n, tot_p))


def main():
    by_phase, by_class, by_when = {}, {}, {}
    rows = 0
    with open(DATA, newline='') as fh:
        for r in csv.DictReader(fh):
            if r.get("family") != TAG:
                continue
            fen = r.get("drop_fen")
            if not fen:
                continue
            try:
                b = chess.Board(fen)
            except Exception:
                continue
            pts = points_lost(r.get("our_color", "white"), r.get("result", ""))
            try:
                dply = float(r.get("drop_ply") or 0)
            except ValueError:
                dply = 0
            rows += 1
            for agg, key in ((by_phase, phase_bucket(npm(b))),
                             (by_class, r.get("ks_class", "?")),
                             (by_when, "A_before ply 40" if dply < 40 else
                                       ("B_ply 40-79" if dply < 80 else "C_ply 80+"))):
                a = agg.setdefault(key, {"n": 0, "pts": 0.0, "ply": 0.0})
                a["n"] += 1
                a["pts"] += pts
                a["ply"] += dply

    print("family=%s   collapses=%d" % (TAG, rows))
    report("POINTS LOST BY PHASE (material on board at the drop)", by_phase)
    report("POINTS LOST BY CLASS", by_class)
    report("POINTS LOST BY WHEN IN THE GAME (drop ply)", by_when)


if __name__ == '__main__':
    main()
