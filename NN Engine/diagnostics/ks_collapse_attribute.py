# -*- coding: utf-8 -*-
"""Categorical collapse attribution: for each arm's collapses.csv, classify each collapse's decision_fen as
KS-CAUSED (SF11 King safety <= -1.5, our POV = our king in real danger at the peak) vs OTHER. The verdict for
the KS change = does the KS-caused count SHRINK from baseline->bundle (even if total collapses is flat, i.e.
KS-caused replaced by other-class = fixing one class exposes the next). Also a paired game-level view.

Run: pyrun diagnostics/ks_collapse_attribute.py --tags ksfull_base,ksfull_ks [--thresh -1.5]
"""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
import sys
import csv
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import chess
from eval_vs_sf11 import SF11Eval, SF11

THIS = os.path.dirname(os.path.abspath(__file__))
GAMES = os.path.join(THIS, "..", "selfplay", "games")


def load(tag):
    rows = []
    cp = os.path.join(GAMES, tag, "collapses.csv")
    if os.path.exists(cp):
        for r in csv.DictReader(open(cp)):
            if r.get("decision_fen"):
                rows.append((r.get("game"), r["decision_fen"], r.get("result")))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", required=True)
    ap.add_argument("--thresh", type=float, default=-1.5)
    ap.add_argument("--write-other", default="", help="write the non-KS (other-class) decision_fens of the FIRST tag to this file")
    args = ap.parse_args()
    tags = [t.strip() for t in args.tags.split(",") if t.strip()]
    sf = SF11Eval(SF11)
    summary = {}
    ks_games = {}
    other_fens = {}
    try:
        for tag in tags:
            rows = load(tag)
            n_ks = 0; ks_g = set(); oth_f = []
            for game, fen, res in rows:
                try:
                    b = chess.Board(fen)
                except ValueError:
                    continue
                povsign = 1.0 if b.turn == chess.WHITE else -1.0
                tot, terms = sf.eval(fen)
                if tot is None:
                    continue
                ks = terms.get("King safety", 0.0) * povsign
                if ks <= args.thresh:
                    n_ks += 1; ks_g.add(str(game))
                else:
                    oth_f.append(fen)
            summary[tag] = (len(rows), n_ks, len(rows) - n_ks)
            ks_games[tag] = ks_g
            other_fens[tag] = oth_f
    finally:
        sf.close()

    if args.write_other and tags:
        with open(args.write_other, "w") as fh:
            for fen in other_fens[tags[0]]:
                fh.write("other\t%s\n" % fen)
        print("wrote %d other-class (non-KS) decision fens -> %s" % (len(other_fens[tags[0]]), args.write_other))

    print("=== Collapse attribution (KS-caused = SF11 King safety <= %.1f, our POV) ===" % args.thresh)
    print("%-16s %10s %12s %10s" % ("arm", "collapses", "KS-caused", "other"))
    for tag in tags:
        n, ksn, oth = summary[tag]
        print("%-16s %10d %12d %10d" % (tag, n, ksn, oth))
    if len(tags) == 2:
        b, k = tags
        dks = summary[k][1] - summary[b][1]
        doth = summary[k][2] - summary[b][2]
        print("\nVERDICT (bundle - baseline):  KS-caused %+d   other %+d   total %+d"
              % (dks, doth, summary[k][0] - summary[b][0]))
        print("  KS-caused collapses %s (%d -> %d).  %s"
              % ("DOWN" if dks < 0 else ("UP" if dks > 0 else "FLAT"), summary[b][1], summary[k][1],
                 "Target class shrank -> KS working even at flat total." if dks < 0
                 else "No categorical reduction." if dks >= 0 else ""))


if __name__ == "__main__":
    main()
