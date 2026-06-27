# -*- coding: utf-8 -*-
"""King-safety STATIC-gap analyzer — does the attack-unit king_safety term move our_static toward SF_static
on king-danger positions? STATIC gap (our_static via ev_breakdown vs corpus sf_static), NOT search gap
(d10 search compensates and HIDES a static eval gap — the lesson from the pawn-majority disconfirmation).

Reads diagnostics/suites/kingsafety.csv (from gen_kingsafety_corpus.py). For each FEN computes our_static
(ChessAI.ev_breakdown total, White-POV pawns) and compares to the recorded SF_static, bucketed by phase
and attacker-count. Also reports the king_safety term's own contribution so you can see it firing.

The tool reflects the CURRENT build + env. To measure the term's effect, run it twice and diff:
    # baseline (term off)
    OMP_NUM_THREADS=1 python diagnostics/_kingsafety_static.py
    # candidate (term on, some knobs)
    OMP_NUM_THREADS=1 KING_SAFETY_MAG=120 KS_DIVISOR=3 python diagnostics/_kingsafety_static.py
A good candidate SHRINKS mean|gap| on the high-attacker / midgame buckets without blowing up the
endgame control bucket (where the term should taper to ~0).

Run in WSL, from NN Engine/, at the interpreter that built ChessAI.
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys, csv, argparse
import chess

THIS = os.path.dirname(os.path.abspath(__file__))
ENGINE = os.path.dirname(THIS)
sys.path.insert(0, ENGINE)

ENGINE_UNITS_PER_PAWN = 1000.0


def white_pawns(ev_abs):
    """Engine absolute (Black-positive) milli-pawns -> White-POV pawns."""
    return -ev_abs / ENGINE_UNITS_PER_PAWN


def main():
    ap = argparse.ArgumentParser(description="King-safety static-gap analyzer (our_static vs SF_static).")
    ap.add_argument("--csv", default=os.path.join(THIS, "suites", "kingsafety.csv"))
    ap.add_argument("--by", choices=["phase", "attackers", "both"], default="both",
                    help="bucketing dimension for the gap table")
    args = ap.parse_args()

    if not os.path.isfile(args.csv):
        print("no corpus at %s (run gen_kingsafety_corpus.py first)" % args.csv); return
    rows = [r for r in csv.DictReader(open(args.csv)) if r.get("sf_static") not in (None, "")]
    if not rows:
        print("corpus has no SF_static labels"); return

    from ChessAI import ChessAI
    seed = chess.Board()
    ai = ChessAI(None, None, seed, seed.turn)

    # Accumulate per bucket: list of (signed_gap, abs_gap, ks_contrib_pawns)
    buckets = {}
    allrec = []
    for r in rows:
        try:
            board = chess.Board(r["fen_start"])
        except Exception:
            continue
        bd = ai.ev_breakdown(board)
        if bd.get("checkmate"):
            continue
        our = white_pawns(bd["total"])
        sf = float(r["sf_static"]) / 100.0
        gap = our - sf
        ks = white_pawns(bd.get("king_safety", 0))
        att = int(r.get("attackers") or 0)
        att_key = att if att <= 5 else 5
        rec = (gap, abs(gap), ks)
        allrec.append(rec)
        keys = []
        if args.by in ("phase", "both"):
            keys.append(("phase", r.get("phase_bin", "?")))
        if args.by in ("attackers", "both"):
            keys.append(("att", "%d%s" % (att_key, "+" if att_key == 5 else "")))
        if args.by == "both":
            keys = [("%s|att%d%s" % (r.get("phase_bin", "?"), att_key, "+" if att_key == 5 else ""), "")]
        for k, _ in keys:
            buckets.setdefault(k, []).append(rec)

    def summarize(label, recs):
        n = len(recs)
        if not n:
            return
        mean_signed = sum(x[0] for x in recs) / n
        mean_abs = sum(x[1] for x in recs) / n
        mean_ks = sum(x[2] for x in recs) / n
        print("  %-24s n=%-4d  signed_gap %+6.2f  |gap| %5.2f  king_safety %+5.2f"
              % (label, n, mean_signed, mean_abs, mean_ks))

    print("=" * 78)
    print("KING-SAFETY STATIC GAP (our_static - SF_static, White-POV pawns) | %d positions" % len(allrec))
    print("  (a good candidate SHRINKS |gap| on high-attacker/midgame buckets; king_safety = term firing)")
    print("-" * 78)
    for k in sorted(buckets):
        summarize(k, buckets[k])
    print("-" * 78)
    summarize("ALL", allrec)
    print("=" * 78)


if __name__ == "__main__":
    main()
