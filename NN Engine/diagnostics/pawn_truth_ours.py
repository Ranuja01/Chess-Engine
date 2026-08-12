# -*- coding: utf-8 -*-
"""OUR marginal pawn value vs SF18's, per cell -- the bridge from ground truth to a data-driven table.

⚠️ THE POINT THAT MAKES THIS NECESSARY. SF18's marginal value of a pawn is NOT the number to put in our
rank table. Our eval already pays for the same pawn through several other terms -- the placement layer, the
attacking layers, chain/wall/latent support, capture gains, and (for passers) realizability. Setting
`rank_bonus = SF18_marginal` would count all of those a second time, which is the exact failure mode this
project has already recorded: targeted double-count removal helps, uniform reshaping flattens.

The quantity a table should be fitted to is the RESIDUAL:

    gap(cell) = SF18_marginal(cell) - OUR_marginal(cell)

measured on the SAME positions, so everything our eval already contributes is subtracted out and what is
left is exactly what is missing. `gap` is reported in millipawns (engine units) so it can be read straight
against `passed_midgame_pawn_rank_bonus` and friends.

Our eval is ABSOLUTE Black-positive, so White-POV cp = -ev/10.

  pyrun diagnostics/pawn_truth_ours.py [IN=diagnostics/ks_sets/pawn_truth.csv] [KNOBS applied via env]
"""
import os, sys, csv
from collections import defaultdict

for _a in sys.argv[1:]:
    if '=' in _a:
        _k, _v = _a.split('=', 1)
        os.environ[_k] = _v

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(THIS)); sys.path.insert(0, THIS)
import chess
from ChessAI import ChessAI

IN = os.environ.get("IN", os.path.join(THIS, "ks_sets", "pawn_truth.csv"))
OUT = os.environ.get("OUT", "")


def median(v):
    if not v:
        return float("nan")
    s = sorted(v); n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def mean(v):
    return sum(v) / len(v) if v else float("nan")


def main():
    if not os.path.exists(IN):
        sys.exit("missing %s" % IN)
    rows = list(csv.DictReader(open(IN, newline="")))
    if not rows:
        sys.exit("%s is empty" % IN)
    ai = ChessAI(None, None, chess.Board(), True)

    def our_cp(board):
        # ev() is Black-positive millipawns; negate for White-POV and scale to centipawns.
        return -ai.ev(board) / 10.0

    cells = defaultdict(lambda: {"sf": [], "ours": []})
    skipped = 0
    for r in rows:
        try:
            board = chess.Board(r["fen"])
        except Exception:
            skipped += 1
            continue
        f, rank = int(r["file"]), int(r["rank"])
        s = chess.square(f, rank - 1)
        if board.piece_at(s) is None or board.piece_at(s).piece_type != chess.PAWN:
            skipped += 1          # the stored FEN and the stored cell disagree -- never silently average it
            continue
        without = board.copy()
        without.remove_piece_at(s)
        if without.status() != chess.STATUS_VALID:
            skipped += 1
            continue
        key = (r["backdrop"], r["enemy"], r["friendly"], rank)
        cells[key]["sf"].append(float(r["value"]))
        cells[key]["ours"].append(our_cp(board) - our_cp(without))

    print("OUR marginal pawn value vs SF18, per cell.  gap = SF18 - ours.")
    print("cp columns; GAP_MP is the same gap in ENGINE UNITS (millipawns) for reading against the tables.")
    print("%d samples, %d skipped\n" % (len(rows) - skipped, skipped))

    # Aggregate view first: the two axes the design answers care about.
    for axis, idx in (("BY RANK", 3), ("BY OBSTRUCTION", 1), ("BY BACKDROP", 0)):
        agg = defaultdict(lambda: {"sf": [], "ours": []})
        for k, v in cells.items():
            agg[k[idx]]["sf"].extend(v["sf"]); agg[k[idx]]["ours"].extend(v["ours"])
        print("%s" % axis)
        print("  %-14s %10s %10s %10s %10s" % ("", "SF18 med", "ours med", "gap med", "GAP_MP"))
        for k in sorted(agg, key=lambda x: (isinstance(x, str), x)):
            sf, ours = agg[k]["sf"], agg[k]["ours"]
            g = median(sf) - median(ours)
            print("  %-14s %10.0f %10.0f %10.0f %10.0f  n=%d" % (k, median(sf), median(ours), g, g * 10, len(sf)))
        print("")

    print("FULL CELL TABLE -- gap in MILLIPAWNS (positive = our eval UNDERPAYS this pawn)")
    ranks = sorted({k[3] for k in cells})
    for backdrop in sorted({k[0] for k in cells}):
        for enemy in sorted({k[1] for k in cells}):
            print("\n  %s / %s" % (backdrop, enemy))
            print("    %-13s %s" % ("", "".join("%9d" % r for r in ranks)))
            for friendly in sorted({k[2] for k in cells}):
                out = []
                for rk in ranks:
                    v = cells.get((backdrop, enemy, friendly, rk))
                    if not v or len(v["sf"]) < 3:
                        out.append("%9s" % "--")
                    else:
                        out.append("%9.0f" % ((median(v["sf"]) - median(v["ours"])) * 10))
                print("    %-13s %s" % (friendly, "".join(out)))

    if OUT:
        with open(OUT, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["backdrop", "enemy", "friendly", "rank", "n",
                        "sf_med_cp", "ours_med_cp", "gap_cp", "gap_mp"])
            for k, v in sorted(cells.items()):
                if not v["sf"]:
                    continue
                g = median(v["sf"]) - median(v["ours"])
                w.writerow(list(k) + [len(v["sf"]), "%.1f" % median(v["sf"]),
                                      "%.1f" % median(v["ours"]), "%.1f" % g, "%.0f" % (g * 10)])
        print("\nwrote %s" % OUT)


if __name__ == "__main__":
    main()
