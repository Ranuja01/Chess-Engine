# -*- coding: utf-8 -*-
"""WHICH ENDGAMES? (owner, 2026-10-07) — breaks the static-ladder dumps (`_reference_ceiling.py DUMP=`) down by ENDGAME TYPE and
measures CONFIDENCE BIAS, on v2's own phase (phase256 < 128 = EG-leaning or full EG).

Per type: val win% MSE for each evaluator, our/SF11 ratio, and BIAS = mean of (win%(pred) − win%(target)) × sign(target):
positive = the evaluator OVER-rates the side that is ahead (thinks it more winning than SF18's search does), negative =
UNDER-rates it (too drawish). |target| ≤ 0.25 pawns rows are excluded from the bias (no "side ahead").
Types (material only, kings implicit): pawn ending · minor only · rook only (R vs R family) · rook+minor · queen ending ·
mixed/imbalanced (unequal piece sets, e.g. R vs minor) — and, inside each, pawns equal vs one side up.
  pyrun diagnostics/_endgame_types.py DUMPS=a.csv,b.csv [EVALS="v2 shipped,SF11 classical,SF18 static,v1"]
"""
import os, sys, csv, math
from collections import defaultdict
import chess

KV = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
wp = lambda p: 100.0 / (1.0 + math.exp(-0.00368208 * p * 100.0))
VAL = ((chess.KNIGHT, 3250), (chess.BISHOP, 3450), (chess.ROOK, 5000), (chess.QUEEN, 10000))


def classify(fen):
    b = chess.Board(fen)
    npm = sum(v * (len(b.pieces(p, True)) + len(b.pieces(p, False))) for p, v in VAL)
    ph = 0 if npm <= 15800 else (256 if npm >= 61700 else (256 * (npm - 15800)) // (61700 - 15800))
    if ph >= 128:
        return None
    sets = []
    for c in (True, False):
        sets.append(tuple(len(b.pieces(p, c)) for p in (chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT)))
    q = sum(s[0] for s in sets); r = sum(s[1] for s in sets); mn = sum(s[2] + s[3] for s in sets)
    if sets[0] != sets[1] and not (sets[0][0] == sets[1][0] and sets[0][1] == sets[1][1]
                                   and sets[0][2] + sets[0][3] == sets[1][2] + sets[1][3]):
        t = "mixed / imbalanced pieces"
    elif q:
        t = "queen ending"
    elif r and mn:
        t = "rook + minor"
    elif r:
        t = "rook only"
    elif mn:
        t = "minor only"
    else:
        t = "pawn ending"
    dp = len(b.pieces(chess.PAWN, True)) - len(b.pieces(chess.PAWN, False))
    return t, ("pawns equal" if dp == 0 else "pawn(s) up")


def main():
    evals = KV.get("EVALS", "v2 shipped,SF11 classical,SF18 static,v1").split(",")
    data = defaultdict(dict)
    for path in KV["DUMPS"].split(","):
        for r in csv.DictReader(open(path, newline="")):
            if r["split"] == "val":
                data[r["fen"]][r["evaluator"]] = (float(r["pred"]), float(r["target"]))
    groups = defaultdict(list)
    for f, d in data.items():
        if not all(e in d for e in evals):
            continue
        c = classify(f)
        if c is None:
            continue
        groups[c[0]].append(f); groups[c[0] + " · " + c[1]].append(f); groups["ALL ENDGAME (phase < 128)"].append(f)
    print("ENDGAME TYPES  (v2 phase256 < 128; val; win% MSE · BIAS = over(+)/under(−)-rating of the side ahead, win% points)")
    print("  %-40s %5s  %s" % ("type", "n", "   ".join("%-22s" % e for e in evals)))
    for g in sorted(groups, key=lambda k: (k != "ALL ENDGAME (phase < 128)", k)):
        fs = groups[g]
        if len(fs) < 15:
            continue
        cells = []
        for e in evals:
            mse = sum((wp(data[f][e][0]) - wp(data[f][e][1])) ** 2 for f in fs) / len(fs)
            bs = [(wp(data[f][e][0]) - wp(data[f][e][1])) * (1 if data[f][e][1] > 0 else -1) for f in fs if abs(data[f][e][1]) > 0.25]
            cells.append("MSE %6.1f bias %+5.1f" % (mse, sum(bs) / len(bs) if bs else float("nan")))
        print("  %-40s %5d  %s" % (g, len(fs), "   ".join("%-22s" % c for c in cells)))


if __name__ == "__main__":
    main()
