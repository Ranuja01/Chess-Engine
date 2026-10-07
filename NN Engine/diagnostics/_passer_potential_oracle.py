# -*- coding: utf-8 -*-
"""ORACLE for eval_v2.cpp `passer_potential` (2026-10-07) — an independently written python-chess version of the CLASSIC
candidate test, compared mask-for-mask against `ChessAI.pawn_masks(...)["potential"]`, plus hand-checked cases and a
colour-mirror check. A pawn counts when it is the FRONT-MOST own pawn on its file AND (passed, or: no enemy pawn ahead on
its file and #helpers ≥ #sentries; helpers = own pawns on adjacent files at most one rank ahead of it; sentries = enemy
pawns on adjacent files strictly ahead).
  pyrun diagnostics/_passer_potential_oracle.py [N=4000]
"""
import os, sys, csv, glob, random
THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(THIS))
import chess
import ChessAI

KV = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)


def ref(b, c):
    out = 0
    fwd = 1 if c == chess.WHITE else -1
    own, en = b.pieces(chess.PAWN, c), b.pieces(chess.PAWN, not c)
    for sq in own:
        f, r = chess.square_file(sq), chess.square_rank(sq)
        ahead = lambda rr: (rr - r) * fwd > 0
        if any(chess.square_file(o) == f and ahead(chess.square_rank(o)) for o in own):
            continue
        if any(chess.square_file(e) == f and ahead(chess.square_rank(e)) for e in en):
            continue
        sentries = sum(1 for e in en if abs(chess.square_file(e) - f) == 1 and ahead(chess.square_rank(e)))
        helpers = sum(1 for o in own if abs(chess.square_file(o) - f) == 1 and (chess.square_rank(o) - r) * fwd <= 1)
        if sentries == 0 or helpers >= sentries:
            out |= 1 << sq
    return out


def probe(b):
    m = ChessAI.pawn_masks(int(b.pawns), int(b.occupied_co[chess.WHITE]), int(b.occupied_co[chess.BLACK]))
    return m["potential"]


def main():
    hand = [  # (fen, expect White has potential, expect Black has potential, note)
        ("8/8/3k4/2p1p1p1/2p1P1P1/2P5/1P1K2P1/8 w - - 0 57", False, False, "owner 10-04: doubled g, b2 vs c4+c5"),
        ("8/5ppp/8/8/8/8/PP3PPP/8 w - - 0 1", True, False, "White 2v0 queenside majority"),
        ("8/pp6/8/8/8/8/PPP5/8 w - - 0 1", True, False, "3v2 on one wing"),
        ("8/p7/8/8/8/8/PP6/8 w - - 0 1", True, False, "2v1"),
        ("8/1p6/8/8/8/8/PP6/8 w - - 0 1", True, False, "a2 unopposed, 1 helper b2 vs 1 sentry b7"),
        ("8/6p1/8/8/8/8/6P1/8 w - - 0 1", False, False, "symmetric opposed"),
    ]
    bad = 0
    for fen, ew, eb, note in hand:
        b = chess.Board(fen)
        pw, pb = probe(b)
        ok = (bool(pw) == ew) and (bool(pb) == eb) and (pw, pb) == (ref(b, chess.WHITE), ref(b, chess.BLACK))
        bad += not ok
        print("  %s  W %s B %s  %s" % ("ok " if ok else "BAD", bool(pw), bool(pb), note))
    fens = []
    for p in glob.glob(os.path.join(THIS, "ks_sets", "fitC_*_sf18.csv")) + [os.path.join(THIS, "ks_sets/kp_stress_sf18.csv")]:
        fens += [r["fen"] for r in csv.DictReader(open(p, newline=""))]
    random.seed(7)
    random.shuffle(fens)
    n, mism, mir = 0, 0, 0
    for f in fens[: int(KV.get("N", 4000))]:
        b = chess.Board(f)
        pw, pb = probe(b)
        mism += (pw, pb) != (ref(b, chess.WHITE), ref(b, chess.BLACK))
        mb = b.mirror()
        qw, qb = probe(mb)
        mir += (chess.flip_vertical(pw) if pw else 0) != qb or (chess.flip_vertical(pb) if pb else 0) != qw
        n += 1
    print("PASSER POTENTIAL ORACLE — hand cases bad %d/%d · %d positions: C++ vs python mismatches %d · mirror violations %d"
          % (bad, len(hand), n, mism, mir))


if __name__ == "__main__":
    main()
