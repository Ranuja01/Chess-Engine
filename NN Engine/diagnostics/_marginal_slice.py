# -*- coding: utf-8 -*-
"""Slice the per-removal piece-marginal dump — ZERO CPU, no Stockfish, re-runnable at will.

The medians said we price minors ~7% low against every reference. A median cannot tell apart:
  VALUE defect      every position is off by roughly the same amount  -> a ratio correction is right
  OVERFIRING term   most positions are fine, a subset is badly off    -> a uniform damp breaks the rest
Those need opposite fixes, so the distribution and the conditioning are the finding, not the centre.

Reads `ks_sets/piece_marginals.csv` (written by `pawn_marginal_real.py OUT=...`), which carries one row per
removal plus the position features to slice on. Every question below is free once that dump exists — the
expensive Stockfish pass happens once.

  pyrun diagnostics/_marginal_slice.py [IN=piece_marginals.csv]
"""
import os, sys, csv
from collections import defaultdict

for _a in sys.argv[1:]:
    if '=' in _a:
        _k, _v = _a.split('=', 1)
        os.environ[_k] = _v

THIS = os.path.dirname(os.path.abspath(__file__))
IN = os.environ.get("IN", "piece_marginals.csv")
if not os.path.isabs(IN):
    IN = os.path.join(THIS, "ks_sets", IN)


# 🚨 RANK BY WIN%, NOT CENTIPAWNS. Absolute cp error over-weights positions that are already decided:
# 0 vs 100 cp swings the expected result far more than 400 vs 500 cp does. This is the same Lichess
# logistic the fit scripts use, so the ranking here matches what tuning actually optimises. The dump
# already stores the position eval WITH the piece, so the WITHOUT value is (pos - marginal) and no
# further Stockfish run is needed to switch metrics.
K_LICHESS = 0.00368208


def winpct(cp):
    import math
    return 100.0 / (1.0 + math.exp(-K_LICHESS * cp))


def pct(v, q):
    if not v:
        return float("nan")
    s = sorted(v)
    return s[max(0, min(len(s) - 1, int(q * (len(s) - 1))))]


def med(v):
    return pct(v, 0.5)


def main():
    rows = list(csv.DictReader(open(IN, newline="")))
    for r in rows:
        for k in ("sf18", "ours", "sf18_pos", "ours_pos"):
            r[k] = float(r[k])
        for k in ("rank", "file", "phase", "is_eg", "wp", "bp", "wn", "bn", "wb", "bb", "wr", "br",
                  "wq", "bq"):
            r[k] = int(r[k])
        r["err"] = r["ours"] - r["sf18"]          # + => we pay MORE than SF18 for this piece
        # Win% the piece is worth to each engine, each through its own eval. The DIFFERENCE of those is
        # how much our mispricing actually moves the expected result -- which is what decides games.
        r["wp_ours"] = winpct(r["ours_pos"]) - winpct(r["ours_pos"] - r["ours"])
        r["wp_sf"] = winpct(r["sf18_pos"]) - winpct(r["sf18_pos"] - r["sf18"])
        r["werr"] = r["wp_ours"] - r["wp_sf"]
    print("PIECE-MARGINAL SLICES  (%d removals, %s)\n" % (len(rows), os.path.basename(IN)))

    print("1) ERROR DISTRIBUTION per piece — ours minus SF18, cp")
    print("   Tight band => VALUE defect. Fat tail => the term OVERFIRES on a subset.")
    print("   %-5s %7s %7s %7s %7s %7s %8s %8s %6s"
          % ("piece", "p10", "p25", "p50", "p75", "p90", "IQR", "p90-p10", "n"))
    for p in ("P", "N", "B", "R"):
        e = [r["err"] for r in rows if r["piece"] == p]
        if len(e) < 5:
            continue
        print("   %-5s %7.0f %7.0f %7.0f %7.0f %7.0f %8.0f %8.0f %6d"
              % (p, pct(e, .1), pct(e, .25), med(e), pct(e, .75), pct(e, .9),
                 pct(e, .75) - pct(e, .25), pct(e, .9) - pct(e, .1), len(e)))

    print("\n1b) THE SAME DISTRIBUTION IN WIN% — this is the one that decides between a VALUE defect and")
    print("    an OVERFIRING term, because cp over-weights already-decided positions.")
    print("   %-5s %7s %7s %7s %7s %7s %8s %6s"
          % ("piece", "p10", "p25", "p50", "p75", "p90", "p90-p10", "n"))
    for p in ("P", "N", "B", "R"):
        e = [r["werr"] for r in rows if r["piece"] == p]
        if len(e) < 5:
            continue
        print("   %-5s %7.1f %7.1f %7.1f %7.1f %7.1f %8.1f %6d"
              % (p, pct(e, .1), pct(e, .25), med(e), pct(e, .75), pct(e, .9),
                 pct(e, .9) - pct(e, .1), len(e)))
    print("\n2b) CONCENTRATION IN WIN% — share of total |win% error| in the worst decile (~10% = uniform)")
    for p in ("P", "N", "B", "R"):
        e = sorted((abs(r["werr"]) for r in rows if r["piece"] == p), reverse=True)
        if len(e) < 10:
            continue
        top = e[:max(1, len(e) // 10)]
        print("   %-5s worst 10%% carry %5.1f%%   (median %4.1f w%%, max %5.1f w%%)"
              % (p, 100.0 * sum(top) / (sum(e) or 1), med(e), e[0]))

    print("\n2) IS THE ERROR CONCENTRATED? share of total error carried by the worst decile")
    print("   A uniform defect spreads error evenly (~10%). Overfiring concentrates it.")
    for p in ("P", "N", "B", "R"):
        e = sorted((abs(r["err"]) for r in rows if r["piece"] == p), reverse=True)
        if len(e) < 10:
            continue
        top = e[:max(1, len(e) // 10)]
        print("   %-5s worst 10%% carry %5.1f%% of total |error|   (median |err| %3.0f, max %4.0f)"
              % (p, 100.0 * sum(top) / (sum(e) or 1), med(e), e[0]))

    print("\n3) WHERE — median error by slice (minors only, the terms in question)")
    slices = [
        ("endgame",        lambda r: r["is_eg"] == 1),
        ("midgame",        lambda r: r["is_eg"] == 0),
        ("queens on",      lambda r: r["wq"] + r["bq"] > 0),
        ("queens off",     lambda r: r["wq"] + r["bq"] == 0),
        ("closed (>=13p)", lambda r: r["wp"] + r["bp"] >= 13),
        ("open (<=9p)",    lambda r: r["wp"] + r["bp"] <= 9),
        ("material even",  lambda r: abs(r["sf18_pos"]) <= 100),
        ("side winning",   lambda r: r["sf18_pos"] > 200),
        ("side losing",    lambda r: r["sf18_pos"] < -200),
    ]
    print("   %-16s %10s %10s %10s %10s" % ("slice", "N err", "n", "B err", "n"))
    for name, f in slices:
        cells = []
        for p in ("N", "B"):
            e = [r["err"] for r in rows if r["piece"] == p and f(r)]
            cells.append((med(e) if len(e) >= 5 else float("nan"), len(e)))
        print("   %-16s %10.0f %10d %10.0f %10d"
              % (name, cells[0][0], cells[0][1], cells[1][0], cells[1][1]))

    print("\n3b) THE SAME SLICES IN WIN% — median |win% error| per removal. This is the ranking that")
    print("    matters: a big cp error in a decided position moves the result very little.")
    print("    %-16s %8s %8s %8s %8s" % ("slice", "N w%", "n", "B w%", "n"))
    for name, f in slices:
        cells = []
        for p in ("N", "B"):
            e = [abs(r["werr"]) for r in rows if r["piece"] == p and f(r)]
            cells.append((med(e) if len(e) >= 5 else float("nan"), len(e)))
        print("    %-16s %8.1f %8d %8.1f %8d"
              % (name, cells[0][0], cells[0][1], cells[1][0], cells[1][1]))
    print("\n    cp-ranked vs win%-ranked worst decile: how much do they even agree?")
    for p in ("P", "N", "B", "R"):
        sub = [r for r in rows if r["piece"] == p]
        if len(sub) < 20:
            continue
        n10 = max(1, len(sub) // 10)
        a = {id(r) for r in sorted(sub, key=lambda r: -abs(r["err"]))[:n10]}
        b = {id(r) for r in sorted(sub, key=lambda r: -abs(r["werr"]))[:n10]}
        print("    %-5s overlap %d/%d  (%.0f%%)" % (p, len(a & b), n10, 100.0 * len(a & b) / n10))

    print("\n4) WORST-DECILE FENs per minor — feed these to the term breakdown")
    for p in ("N", "B"):
        sub = sorted((r for r in rows if r["piece"] == p), key=lambda r: -abs(r["err"]))
        for r in sub[:5]:
            print("   %s err%+5.0f (ours %4.0f sf18 %4.0f) %s" % (p, r["err"], r["ours"], r["sf18"],
                                                                  r["fen"]))


if __name__ == "__main__":
    main()
