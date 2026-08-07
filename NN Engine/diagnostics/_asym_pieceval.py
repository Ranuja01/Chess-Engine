# -*- coding: utf-8 -*-
"""Isolate the positions where the PIECE-VALUE ACCUMULATORS break colour antisymmetry.

`material` in the breakdown is NOT a piece count -- it is `blackPieceVal - whitePieceVal`, and those two
globals are incremented inside each per-piece evaluator. So whenever an evaluator is skipped, returns
early, or takes a different branch under mirror, the accumulator diverges and `material`,
`capture_gains` and `piece_value_boost` all inherit the error. The term-ranked run showed these three
breaking on the SAME 16 positions with a ~680 mp mean, which is the signature of one shared root.

This prints, for each offender, the true material balance (counted independently, from the FEN) next to
the accumulator's answer, so it is immediately visible WHICH side is being under/over-counted and by
how many pawns. It also prints the phase flags, since the suspicion is a phase-dependent branch
(advanced_endgame_eval / the endgame clamps) rather than the evaluators themselves.

  pyrun diagnostics/_asym_pieceval.py [IN=ks_sets/diverse_corpus_wide.csv] [N=400]
"""
import os, sys, csv

for _a in sys.argv[1:]:
    if '=' in _a:
        _k, _v = _a.split('=', 1)
        os.environ[_k] = _v

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(THIS)); sys.path.insert(0, THIS)
import chess
from ChessAI import ChessAI

N = int(os.environ.get("N", "400"))
IN = os.environ.get("IN", "ks_sets/diverse_corpus_wide.csv")
if not os.path.isabs(IN):
    IN = os.path.join(THIS, IN)

# Millipawns, matching the engine's `values[]` ordering by python-chess piece type.
VAL = {chess.PAWN: 1000, chess.KNIGHT: 3000, chess.BISHOP: 3000, chess.ROOK: 5000, chess.QUEEN: 9000}


def counted(b):
    """Independent ground truth: sum the board, do not ask the engine."""
    w = sum(VAL[p] * len(b.pieces(p, chess.WHITE)) for p in VAL)
    bl = sum(VAL[p] * len(b.pieces(p, chess.BLACK)) for p in VAL)
    return w, bl


def main():
    ai = ChessAI(None, None, chess.Board(), True)
    rows = list(csv.DictReader(open(IN, newline="")))[:N]

    hits, n = [], 0
    # Which flags co-occur with the defect vs the clean positions -- a flag that is on for every offender
    # and off for everything else IS the branch.
    flag_on = {}
    flag_all = {}
    FLAGS = ["is_endgame", "advanced_endgame_fired", "phase_score"]

    for r in rows:
        try:
            b = chess.Board(r["fen"])
        except Exception:
            continue
        if b.is_game_over(claim_draw=False):
            continue
        try:
            db, dm = ai.ev_breakdown(b), ai.ev_breakdown(b.mirror())
        except Exception:
            continue
        n += 1
        wb, bb = db.get("det_w_pieceval", 0), db.get("det_b_pieceval", 0)
        wm, bm = dm.get("det_w_pieceval", 0), dm.get("det_b_pieceval", 0)
        # Under mirror the two accumulators must simply exchange.
        d = abs(wb - bm) + abs(bb - wm)
        for f in FLAGS:
            flag_all[f] = flag_all.get(f, 0) + (1 if db.get(f, 0) else 0)
        if d:
            cw, cbl = counted(b)
            hits.append((d, wb, bb, wm, bm, cw, cbl, db, r["fen"]))
            for f in FLAGS:
                flag_on[f] = flag_on.get(f, 0) + (1 if db.get(f, 0) else 0)

    print("PIECE-VALUE ACCUMULATOR ANTISYMMETRY  (%d positions tested)\n" % n)
    print("   violations %d (%.1f%%)\n" % (len(hits), 100.0 * len(hits) / max(1, n)))

    print("   flag co-occurrence  (offenders vs whole sample)")
    for f in FLAGS:
        print("     %-24s offenders %3d/%-4d   sample %4d/%-4d"
              % (f, flag_on.get(f, 0), len(hits), flag_all.get(f, 0), n))

    print("\n   worst offenders -- ACC = engine accumulator, TRUE = counted from the FEN")
    for d, wb, bb, wm, bm, cw, cbl, db, fen in sorted(hits, reverse=True)[:12]:
        print("\n     off by %6d mp   %s" % (d, fen))
        print("       ACC  base  w=%-8d b=%-8d      mirror  w=%-8d b=%-8d" % (wb, bb, wm, bm))
        print("       TRUE       w=%-8d b=%-8d   (accumulator excess w %+d / b %+d)"
              % (cw, cbl, wb - cw, bb - cbl))
        print("       phase=%s endgame=%s ae_fired=%s"
              % (db.get("phase_score"), db.get("is_endgame"), db.get("advanced_endgame_fired")))


if __name__ == "__main__":
    main()
