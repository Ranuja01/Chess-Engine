# -*- coding: utf-8 -*-
"""WHICH TERM pays the ~150 cp we overprice a pawn by? Per-term attribution on the paired truth FENs.

`pawn_truth_ours.py` established that our marginal pawn value is ~150 cp above SF18's at ranks 2-5 and
~83 cp below at rank 7, i.e. our rank curve is far too flat. That is the SUM over every eval term that
changes when a pawn is added, so it does not say what to reshape. Reshaping before knowing that is how a
uniform shrink flattens the whole eval (a recorded lesson here), so this splits the gap by term.

For each paired position it takes `ev_breakdown` WITH and WITHOUT the test pawn and reports the per-term
delta. `ev_breakdown` is a clean partition (fields sum to `total`), so the term deltas sum to the marginal
value our eval assigns -- and the residual against SF18 is attributable line by line.

All output is White-POV centipawns (our eval is absolute Black-positive, so negate and divide by 10).

  pyrun diagnostics/pawn_gap_attribution.py [IN=diagnostics/ks_sets/pawn_truth_low.csv]
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

IN = os.environ.get("IN", os.path.join(THIS, "ks_sets", "pawn_truth_low.csv"))

# Additive partition of `total`. `material` is informational (not part of the sum) and `pieces` is the
# lump of the six per-piece loops that pt_* already itemises -- both excluded to keep the sum honest.
TERMS = ["pt_pawns", "pt_knights", "pt_bishops", "pt_rooks", "pt_queens", "pt_kings",
         "capture_gains", "passed_pawn_support", "latent_threat", "threats", "king_safety", "central",
         "imbalance_white", "imbalance_black", "pair_bonus", "piece_value_boost", "kaufman_imbalance",
         "pawn_majority", "pawn_struct", "outpost", "space", "mobility", "rook_cond"]


def median(v):
    if not v:
        return 0.0
    s = sorted(v); n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def main():
    if not os.path.exists(IN):
        sys.exit("missing %s" % IN)
    rows = list(csv.DictReader(open(IN, newline="")))
    ai = ChessAI(None, None, chess.Board(), True)

    by_rank = defaultdict(lambda: defaultdict(list))
    sf_by_rank = defaultdict(list)
    skipped = 0
    for r in rows:
        try:
            board = chess.Board(r["fen"])
        except Exception:
            skipped += 1; continue
        rank = int(r["rank"])
        s = chess.square(int(r["file"]), rank - 1)
        p = board.piece_at(s)
        if p is None or p.piece_type != chess.PAWN:
            skipped += 1; continue
        without = board.copy()
        without.remove_piece_at(s)
        if without.status() != chess.STATUS_VALID:
            skipped += 1; continue
        a = ai.ev_breakdown(board)
        b = ai.ev_breakdown(without)
        # Black-positive -> White-POV cp
        for t in TERMS + ["total"]:
            by_rank[rank][t].append(-(a.get(t, 0) - b.get(t, 0)) / 10.0)
        sf_by_rank[rank].append(float(r["value"]))

    ranks = sorted(by_rank)
    print("PER-TERM marginal contribution of ONE pawn (White-POV cp, medians). %d samples, %d skipped\n"
          % (len(rows) - skipped, skipped))
    print("  %-22s %s" % ("term", "".join("%9d" % r for r in ranks)))
    print("  " + "-" * (22 + 9 * len(ranks)))
    # Order terms by how much they VARY across rank -- the flat-curve defect lives in whatever fails to rise.
    def spread(t):
        vals = [median(by_rank[r][t]) for r in ranks]
        return max(vals) - min(vals)
    for t in sorted(TERMS, key=lambda t: -abs(spread(t))):
        vals = [median(by_rank[r][t]) for r in ranks]
        if all(abs(v) < 0.5 for v in vals):
            continue                      # term does not move when a pawn is added
        print("  %-22s %s   spread %+.0f" % (t, "".join("%9.1f" % v for v in vals), spread(t)))
    print("  " + "-" * (22 + 9 * len(ranks)))
    print("  %-22s %s" % ("OURS total", "".join("%9.1f" % median(by_rank[r]["total"]) for r in ranks)))
    print("  %-22s %s" % ("SF18 total", "".join("%9.1f" % median(sf_by_rank[r]) for r in ranks)))
    print("  %-22s %s" % ("GAP (SF - ours)", "".join(
        "%9.1f" % (median(sf_by_rank[r]) - median(by_rank[r]["total"])) for r in ranks)))
    print("\nRead the SPREAD column: the flat-curve defect is carried by terms that stay CONSTANT across")
    print("rank (they inflate every pawn equally) and by the absence of any term that rises steeply.")


if __name__ == "__main__":
    main()
