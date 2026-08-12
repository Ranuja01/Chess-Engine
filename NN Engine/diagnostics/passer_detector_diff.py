# -*- coding: utf-8 -*-
"""How often does SF's passed-pawn definition disagree with ours, and at which ranks?

The "our passer DETECTION is primitive" claim has been asserted from source reading several times without
ever being counted. Both definitions are pure bitboard predicates, so this settles it with no engine calls.

OURS (getPPIncrement): passed iff NO enemy pawn lies in the three-file forward span, and no friendly pawn
stands ahead on the same file (the rear-doubled exclusion, ENABLE_PASSER_V3).

SF15.1 (`pawns.cpp` L152-158) additionally accepts three cases:
  (a) !(stoppers ^ lever)                       -- every stopper is a pawn WE attack
  (b) !(stoppers ^ leverPush) && popcount(phalanx) >= popcount(leverPush)
  (c) stoppers == blocked && rank >= 5 && (support pushed one square is safe)
  then the same rear-doubled exclusion.

Note a 7th-rank pawn can only have stoppers on the 8th, where enemy pawns cannot stand, so it is passed
under BOTH definitions -- any disagreement must live at lower ranks. That is the point of the per-rank split.

  pyrun diagnostics/passer_detector_diff.py [IN=ks_sets/diverse_corpus_wide.csv] [MAX_POS=2000]
"""
import os, sys, csv
from collections import defaultdict

for _a in sys.argv[1:]:
    if '=' in _a:
        _k, _v = _a.split('=', 1)
        os.environ[_k] = _v

THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(THIS)); sys.path.insert(0, THIS)
import chess

IN = os.environ.get("IN", "ks_sets/diverse_corpus_wide.csv")
if not os.path.isabs(IN):
    IN = os.path.join(THIS, IN)
MAX_POS = int(os.environ.get("MAX_POS", "2000"))


def analyse(board, colour):
    """Yield (rank_from_owner_1_to_7, ours_passed, sf_passed) for each of `colour`'s pawns."""
    us = board.pieces(chess.PAWN, colour)
    them = board.pieces(chess.PAWN, not colour)
    up = 8 if colour == chess.WHITE else -8
    out = []
    for s in us:
        f, r = chess.square_file(s), chess.square_rank(s)
        rank_owner = (r + 1) if colour == chess.WHITE else (8 - r)

        def ahead(sq_rank):
            return sq_rank > r if colour == chess.WHITE else sq_rank < r

        stoppers = {t for t in them
                    if abs(chess.square_file(t) - f) <= 1 and ahead(chess.square_rank(t))}
        # squares this pawn attacks
        lever = {t for t in them
                 if abs(chess.square_file(t) - f) == 1 and chess.square_rank(t) == r + (1 if colour == chess.WHITE else -1)}
        push = s + up
        leverPush = set()
        if 0 <= push <= 63:
            pr = chess.square_rank(push)
            leverPush = {t for t in them
                         if abs(chess.square_file(t) - f) == 1
                         and chess.square_rank(t) == pr + (1 if colour == chess.WHITE else -1)}
        blocked = {t for t in them if chess.square_file(t) == f and chess.square_rank(t) == r + (1 if colour == chess.WHITE else -1)}
        phalanx = {t for t in us if abs(chess.square_file(t) - f) == 1 and chess.square_rank(t) == r}
        support = {t for t in us if abs(chess.square_file(t) - f) == 1
                   and chess.square_rank(t) == r - (1 if colour == chess.WHITE else -1)}

        rear_doubled = any(chess.square_file(t) == f and ahead(chess.square_rank(t)) for t in us)

        ours = (not stoppers) and not rear_doubled

        sf = (not (stoppers ^ lever)) \
            or ((not (stoppers ^ leverPush)) and len(phalanx) >= len(leverPush)) \
            or (stoppers == blocked and stoppers and rank_owner >= 5
                and any((t + up) not in them and 0 <= t + up <= 63 for t in support))
        sf = bool(sf) and not rear_doubled
        out.append((rank_owner, ours, sf))
    return out


def main():
    rows = list(csv.DictReader(open(IN, newline="")))[:MAX_POS]
    tot = defaultdict(int)
    ours_only = defaultdict(int)
    sf_only = defaultdict(int)
    both = defaultdict(int)
    npos = 0
    for r in rows:
        try:
            b = chess.Board(r["fen"])
        except Exception:
            continue
        npos += 1
        for colour in (chess.WHITE, chess.BLACK):
            for rank, o, s in analyse(b, colour):
                tot[rank] += 1
                if o and s:
                    both[rank] += 1
                elif o and not s:
                    ours_only[rank] += 1
                elif s and not o:
                    sf_only[rank] += 1

    print("Passed-pawn DETECTOR disagreement, ours vs SF15.1  (%s, %d positions)\n"
          % (os.path.basename(IN), npos))
    print("  %-6s %8s %10s %12s %12s %10s" % ("rank", "pawns", "both", "SF only", "ours only", "SF-only %"))
    T = P = SO = OO = 0
    for rank in sorted(tot):
        n, bo, so, oo = tot[rank], both[rank], sf_only[rank], ours_only[rank]
        T += n; P += bo; SO += so; OO += oo
        print("  %-6d %8d %10d %12d %12d %9.1f%%" % (rank, n, bo, so, oo, 100.0 * so / n if n else 0))
    print("  %-6s %8d %10d %12d %12d %9.1f%%" % ("ALL", T, P, SO, OO, 100.0 * SO / T if T else 0))
    print("\n  SF only   = SF calls it passed, we do NOT  -> the pawns a 'modernized' detector would add")
    print("  ours only = we call it passed, SF does not  -> should be ~0 if our rule is a strict subset")


if __name__ == "__main__":
    main()
