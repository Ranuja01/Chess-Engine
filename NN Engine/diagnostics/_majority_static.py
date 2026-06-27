# -*- coding: utf-8 -*-
"""STATIC-eval calibration probe for the pawn-majority feature. Compares OUR static eval to the
recorded SF NNUE static (sf_static column) over majorities.csv -- this is where the majority term
lives (the SEARCH eval in _majority_match compensates dynamically and HIDES the static gap, so use
THIS to tune the eval term). Honors PAWN_MAJORITY_* (+ any eval) env knobs, so run knob-off then a
candidate and read the median SIGNED gap (negative = we under-read White's edge) shrink toward 0.
Fast (static eval only, no search) -> sweeps in seconds.

Usage (from NN Engine/):
    /home/ranuja/anaconda3/bin/python diagnostics/_majority_static.py diagnostics/suites/majorities.csv
    PAWN_MAJORITY_MAG_EG=1000 ... /home/ranuja/anaconda3/bin/python diagnostics/_majority_static.py <csv>
"""
import os, sys, csv
THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS)
import chess
from eval_breakdown import white_pawns, _load_engine

DREL = 500   # decision-relevant window on |sf_static| (cp): ignore already-decided/mate-ish positions
path = sys.argv[1]
rows = list(csv.DictReader(open(path)))

ChessAI = _load_engine()
seed = chess.Board()
ai = ChessAI(None, None, seed, seed.turn)   # one warm instance; ev_breakdown reads only the board arg

acc = {}   # bucket -> {"s":[signed gaps cp], "a":[abs gaps cp]} over the decision window
def bump(b, gap):
    a = acc.setdefault(b, {"s": [], "a": []})
    a["s"].append(gap); a["a"].append(abs(gap))

def median(xs):
    if not xs:
        return 0.0
    xs = sorted(xs); k = len(xs)
    return xs[k // 2] if k % 2 else 0.5 * (xs[k // 2 - 1] + xs[k // 2])

for r in rows:
    sf_static = r.get("sf_static", "")
    if sf_static in ("", None):
        continue
    try:
        board = chess.Board(r["fen_start"])
        bd = ai.ev_breakdown(board)
        if bd.get("checkmate"):
            continue
        our = white_pawns(bd["total"])      # White-POV pawns
        sf = float(sf_static) / 100.0        # cp -> pawns
    except Exception:
        continue
    if abs(sf) * 100.0 >= DREL:
        continue
    gap = (our - sf) * 100.0                  # cp; negative = we under-read White's edge
    bump("ALL", gap); bump(r.get("phase", "all"), gap)

def line(name, a):
    return (f"{name:>5}: median_signed_gap={median(a['s']):+.0f}cp  "
            f"median|gap|={median(a['a']):.0f}cp  (n={len(a['s'])})")

print(line("ALL", acc.get("ALL", {"s": [], "a": []})))
for ph in ("mid", "eeg", "leg"):
    if ph in acc:
        print(line(ph, acc[ph]))
