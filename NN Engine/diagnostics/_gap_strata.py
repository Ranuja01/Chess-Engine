# -*- coding: utf-8 -*-
"""WHERE does our static gap to the references live? (owner, 2026-10-06: "are we still in a gap that could be explained by
a missing feature/subfeature?") — stratifies the per-position dumps of `_reference_ceiling.py DUMP=` (references and our
arms on the SAME rows, the SAME loss: win% squared error vs SF18 d14 search, val split).

Per stratum: each evaluator's val MSE, and the GAP SHARE = how much of (ours − SF11) total squared-error excess sits in
that stratum vs its share of rows. A gap spread in proportion to the rows ⇒ WEIGHTING (the joint retune's job); a gap
concentrated in one stratum where SF11 is much better ⇒ a candidate MISSING FEATURE there.
Strata: phase (non-pawn material, both sides) · material class (balanced / minor-for-pawns / exchange / queen imbalance /
other) · pawn-only endings · |target| (level < 1 pawn, edge 1-3, decisive > 3).

  pyrun diagnostics/_gap_strata.py DUMPS=a.csv,b.csv OURS="v2 shipped" [REF="SF11 classical"]
"""
import os, sys, csv, math
from collections import defaultdict
import chess

KV = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
wp = lambda p: 50.0 + 50.0 * (2.0 / (1.0 + math.exp(-0.00368208 * p * 100.0)) - 1.0)
VAL = {chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}


def strata(fen, target):
    b = chess.Board(fen)
    npm = sum(VAL[p] * (len(b.pieces(p, chess.WHITE)) + len(b.pieces(p, chess.BLACK))) for p in VAL)
    phase = "opening/mg (npm≥50)" if npm >= 50 else ("late mg (30-49)" if npm >= 30 else ("endgame (1-29)" if npm > 0 else "pawn ending (0)"))
    c = {(p, s): len(b.pieces(p, s)) for p in (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN)
         for s in (chess.WHITE, chess.BLACK)}
    d = lambda p: c[(p, chess.WHITE)] - c[(p, chess.BLACK)]
    minors, rooks, queens, pawns = d(chess.KNIGHT) + d(chess.BISHOP), d(chess.ROOK), d(chess.QUEEN), d(chess.PAWN)
    if queens != 0:
        mat = "queen imbalance"
    elif rooks != 0 and minors != 0 and rooks * minors < 0:
        mat = "exchange (R vs minor)"
    elif rooks == 0 and minors != 0:
        mat = "minor vs pawns"
    elif rooks == 0 and minors == 0 and queens == 0:
        mat = "balanced pieces"
    else:
        mat = "other imbalance"
    a = abs(target)
    lvl = "level (<1)" if a < 1 else ("edge (1-3)" if a <= 3 else "decisive (>3)")
    return {"phase": phase, "material": mat, "|target|": lvl}


def main():
    ours, ref = KV.get("OURS", "OURS (current build)"), KV.get("REF", "SF11 classical")
    err = defaultdict(dict)                     # fen -> {evaluator: sq error}
    tgt = {}
    for path in KV["DUMPS"].split(","):
        for r in csv.DictReader(open(path, newline="")):
            if r["split"] != "val":
                continue
            t = float(r["target"]); tgt[r["fen"]] = t
            err[r["fen"]][r["evaluator"]] = (wp(float(r["pred"])) - wp(t)) ** 2
    evs = sorted({e for d in err.values() for e in d})
    fens = [f for f in err if ours in err[f] and ref in err[f]]
    print("GAP STRATA  val rows %d (scored by both %r and %r)" % (len(fens), ours, ref))
    print("  overall: " + " · ".join("%s %.1f" % (e, sum(err[f][e] for f in fens if e in err[f]) /
                                               max(1, sum(1 for f in fens if e in err[f]))) for e in evs))
    st = {f: strata(f, tgt[f]) for f in fens}
    excess_tot = sum(err[f][ours] - err[f][ref] for f in fens)
    for dim in ("phase", "material", "|target|"):
        print("\n  by %s  (val MSE per evaluator · rows share · share of the %s−%s excess)" % (dim, ours, ref))
        groups = defaultdict(list)
        for f in fens:
            groups[st[f][dim]].append(f)
        for g, fs in sorted(groups.items(), key=lambda x: -len(x[1])):
            cols = "  ".join("%s %6.1f" % (e[:14], sum(err[f][e] for f in fs if e in err[f]) /
                                           max(1, sum(1 for f in fs if e in err[f]))) for e in evs)
            ex = sum(err[f][ours] - err[f][ref] for f in fs)
            print("    %-24s n %5d (%4.1f%%) · excess share %5.1f%% · %s" % (g, len(fs), 100 * len(fs) / len(fens),
                  100 * ex / excess_tot if excess_tot else float("nan"), cols))


if __name__ == "__main__":
    main()
