# -*- coding: utf-8 -*-
"""WHICH STAGE loses a passer's value? Localise a passer misvaluation instead of inferring it.

A passer that scores ~0 where SF prices it ~1.3 pawns can fail at any of four stages, with unrelated fixes:
  1. NOT FLAGGED   - getPPIncrement never treated it as passed => it is not in our record set at all
  2. LOW MAGNITUDE - flagged, but the phase-blended rank table gives it little to begin with
  3. R COLLAPSED   - magnitude fine, realizability ~0, so `mag * R/256` -> ~0
  4. priced fine   - the miss is elsewhere in the eval

A corpus MSE cannot separate these, which is why three global reshapes of `mag * R/256` all failed while the
defect stayed unlocated. This uses ChessAI.passer_records() (the real eval, probe flag on) and compares OUR
flagged set against a reference passed-pawn set computed from the FEN.

  pyrun diagnostics/_passer_stage_diagnosis.py [FENS=diagnostics/_top8.fens]
"""
import os, sys

for _a in sys.argv[1:]:
    if '=' in _a:
        _k, _v = _a.split('=', 1)
        os.environ[_k] = _v

THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(THIS)); sys.path.insert(0, THIS)
import chess
from ChessAI import ChessAI

FENS = os.environ.get("FENS", os.path.join(THIS, "_top8.fens"))
LOW_R = int(os.environ.get("LOW_R", "64"))       # R at/below this = collapsed (256 = neutral)


def reference_passers(board):
    """Standard definition: no enemy pawn on this file or an adjacent file ahead of it."""
    out = set()
    for colour in (chess.WHITE, chess.BLACK):
        enemy = board.pieces(chess.PAWN, not colour)
        for sq in board.pieces(chess.PAWN, colour):
            f, r = chess.square_file(sq), chess.square_rank(sq)
            blocked = False
            for ef in (f - 1, f, f + 1):
                if not 0 <= ef <= 7:
                    continue
                for esq in enemy:
                    if chess.square_file(esq) != ef:
                        continue
                    er = chess.square_rank(esq)
                    if (colour == chess.WHITE and er > r) or (colour == chess.BLACK and er < r):
                        blocked = True
                        break
                if blocked:
                    break
            if not blocked:
                out.add(sq)
    return out


def main():
    if not os.path.exists(FENS):
        sys.exit("missing FEN file: %s" % FENS)
    fens = [ln.strip() for ln in open(FENS) if ln.strip() and not ln.startswith("#")]
    ai = ChessAI(None, None, chess.Board(), True)

    tally = {"not_flagged": 0, "low_mag": 0, "r_collapsed": 0, "priced": 0, "early": 0}
    for n, line in enumerate(fens):
        # Lines may carry a leading label ("w1\t<fen>"), so locate the board field rather than assuming
        # the FEN starts at token 0 -- otherwise every position silently fails to parse and the tally is 0.
        toks = line.split()
        start = next((i for i, t in enumerate(toks) if "/" in t), 0)
        fen = " ".join(toks[start:start + 6])
        label = " ".join(toks[:start]) or str(n)
        try:
            b = chess.Board(fen)
        except Exception:
            print("\n[%s] UNPARSEABLE: %s" % (label, fen))
            continue
        recs = {r["sq"]: r for r in ai.passer_records(b)}
        ref = reference_passers(b)
        print("\n[%s] %s" % (label, fen))
        if not ref:
            print("    (no passed pawns by the reference definition)")
            continue
        print("    %-6s %-6s %-5s %6s %6s %7s %5s %8s   %s"
              % ("sq", "side", "rank", "mag", "R", "rawR", "blk", "val", "stage"))
        for sq in sorted(ref):
            name = chess.square_name(sq)
            side = "W" if b.color_at(sq) == chess.WHITE else "B"
            r = recs.get(sq)
            if r is None:
                tally["not_flagged"] += 1
                print("    %-6s %-6s %-5s %6s %6s %5s %8s   NOT FLAGGED (stage 1)" % (name, side, "-", "-", "-", "-", "-"))
                continue
            # Only ADVANCED passers can be "underpriced": a rank-1/2 passer scoring ~0.1 pawns is correct
            # (SF's PassedRank is small there too), so an absolute magnitude threshold manufactures a
            # category. Judge stage 2/3 only where the pawn is far enough advanced to be worth something.
            if r["rank"] < 4:
                stage = "early rank (expected small)"; tally["early"] += 1
            elif r["R"] <= LOW_R:
                stage = "R COLLAPSED (stage 3)"; tally["r_collapsed"] += 1
            elif abs(r["mag"]) < 200:
                stage = "LOW MAGNITUDE (stage 2)"; tally["low_mag"] += 1
            else:
                stage = "priced"; tally["priced"] += 1
            print("    %-6s %-6s %-5d %6d %6d %7d %5d %8d   %s"
                  % (name, side, r["rank"], r["mag"], r["R"], r["rawR"], r["blk"], r["val"], stage))
        extra = set(recs) - ref
        if extra:
            print("    ⚠️ flagged by us but NOT passed by the reference: %s"
                  % ", ".join(chess.square_name(s) for s in sorted(extra)))

    print("\nSTAGE TALLY over %d positions: %s" % (len(fens), tally))
    print("Stage 1 dominant => fix getPPIncrement (detection), not R.")
    print("Stage 3 dominant => R is collapsing on real passers; look at CONTEST (the load-bearing input).")
    print("Mostly 'priced'  => passers are NOT the defect in these positions; the miss is another term.")


if __name__ == '__main__':
    main()
