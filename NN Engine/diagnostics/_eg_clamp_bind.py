# -*- coding: utf-8 -*-
"""Does the ENDGAME pawn clamp SATURATE more once the support-detection fix is on?

The wrap fix (`ENABLE_PAWN_SUPPORT_WRAP_FIX`) is unambiguously correct yet costs ~105 balanced STS with
NO tactical cost. Two mechanisms could explain a positional-only loss, and they need OPPOSITE knobs:

  MAGNITUDE   EG_SUPPORT was fitted while some of Black's diagonal supports were invisible, so the
              effective average credit sat below the nominal 135. Restoring detection inflates it.
              -> the fix is to LOWER EG_SUPPORT / EG_LATENT.

  SATURATION  structural_bonus ends in `min(PAWN_CLAMP_EG, structural_bonus)` with the cap at 175, while
              the parts sum past it fast (PHALANX 100 + SUPPORT 135 + LATENT 50 + DEFEND 115). If
              restoring support pushes many more pawns INTO the clamp, distinct positions collapse onto
              the same 175 and a MOVE-CHOICE bench punishes the lost discrimination -- which is exactly
              a positional-only cost with no tactical cost.
              -> the fix is to RAISE PAWN_CLAMP_EG, and lowering EG_SUPPORT would be wrong.

This measures the discriminator: endgame clamp bind rate and headroom, per colour. Run once per
setting (knobs latch at init) and compare.

  pyrun diagnostics/_eg_clamp_bind.py [N=1500] [ENABLE_PAWN_SUPPORT_WRAP_FIX=1]
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

N = int(os.environ.get("N", "1500"))
IN = os.environ.get("IN", "ks_sets/diverse_corpus_wide.csv")
if not os.path.isabs(IN):
    IN = os.path.join(THIS, IN)


def main():
    ai = ChessAI(None, None, chess.Board(), True)
    rows = list(csv.DictReader(open(IN, newline="")))[:N]

    # Per colour, because the defect is in the BLACK branch only -- a shared bind rate would hide the
    # asymmetry that is the whole point.
    stat = {True: {"n": 0, "bound": 0, "sum": 0, "over": 0},
            False: {"n": 0, "bound": 0, "sum": 0, "over": 0}}
    npos = 0
    for r in rows:
        try:
            b = chess.Board(r["fen"])
            if b.is_game_over(claim_draw=False):
                continue
            recs = ai.pawn_clamp_records(b)
        except Exception:
            continue
        npos += 1
        for pc in recs:
            if not pc["endgame"]:
                continue
            s = stat[pc["white"]]
            s["n"] += 1
            s["sum"] += pc["structural"]
            if pc["structural"] >= pc["cap"]:
                s["bound"] += 1
                s["over"] += pc["structural"] - pc["cap"]

    print("ENDGAME PAWN CLAMP BIND  (%d positions, wrap_fix=%s, cap=PAWN_CLAMP_EG)\n"
          % (npos, os.environ.get("ENABLE_PAWN_SUPPORT_WRAP_FIX", "0")))
    print("  %-8s %8s %8s %9s %12s %12s" % ("colour", "pawns", "bound", "bind%", "mean struct", "mean excess"))
    for w, lbl in [(True, "white"), (False, "black")]:
        s = stat[w]
        if not s["n"]:
            print("  %-8s %8d" % (lbl, 0))
            continue
        print("  %-8s %8d %8d %8.1f%% %12.1f %12.1f"
              % (lbl, s["n"], s["bound"], 100.0 * s["bound"] / s["n"], s["sum"] / s["n"],
                 s["over"] / s["bound"] if s["bound"] else 0.0))

    print("\nREADING IT")
    print("  Compare BLACK's bind%% between wrap_fix=0 and wrap_fix=1. A large jump => SATURATION is the")
    print("  mechanism, so raise PAWN_CLAMP_EG and do NOT shrink EG_SUPPORT. Little change => the cost is")
    print("  MAGNITUDE, so sweep EG_SUPPORT/EG_LATENT down instead.")
    print("  White should barely move either way -- the defect is in the black branch only, so White is")
    print("  the built-in control. If WHITE moves, this probe is measuring something else.")


if __name__ == "__main__":
    main()
