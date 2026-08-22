# -*- coding: utf-8 -*-
"""KS_EXTEND_EG specificity on the TOTAL eval (the extension adds to total, not the king_safety breakdown
field, so we must read total). On a labelled FEN set, report per group the mean over-valuation vs SF11
(OURS_total - SF11_total), in the side-to-move's favour. Run twice -- KS_EXTEND_EG=0 then =1 -- and compare:

  collapse group: over-valuation should DROP toward 0 (endgame KS finally prices the king danger)  = recall
  safe-eg  group: over-valuation should barely move (no phantom danger on safe kings)               = precision

A big safe-eg drop = KS_EXTEND_EG inventing danger on safe endgames = the over-fire trap.

  pyrun diagnostics/_ks_eg_specificity.py FENS=<file> [KS_EXTEND_EG=1 ...]
"""
import os, sys, csv, statistics
from collections import defaultdict

for _a in sys.argv[1:]:
    if '=' in _a:
        _k, _v = _a.split('=', 1)
        os.environ.setdefault(_k, _v)
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(THIS)); sys.path.insert(0, THIS)
import chess
from eval_vs_sf11 import SF11Eval, SF11
from ChessAI import ChessAI

FENS = os.environ["FENS"]
sf11 = SF11Eval(SF11)
ai = ChessAI(None, None, chess.Board(), True)
groups = defaultdict(list)

for ln in open(FENS):
    ln = ln.rstrip("\n")
    if not ln.strip():
        continue
    lbl, fen = ln.split("\t", 1)
    grp = "collapse" if lbl.startswith("ksatk") else "safe-eg"
    try:
        b = chess.Board(fen)
        our_wp = -float(ai.ev_breakdown(b).get("total", 0)) / 1000.0    # white-POV pawns
        t11, _ = sf11.eval(fen)                                          # white-POV pawns (None if SF11 can't eval, e.g. in-check)
        if t11 is None:
            continue
    except Exception:
        continue
    stm = 1.0 if b.turn == chess.WHITE else -1.0                        # over-valuation in mover's favour
    groups[grp].append((stm * our_wp, stm * t11))

print("KS_EXTEND_EG=%s  (over-valuation vs SF11 in mover's favour; + = we think mover is better than SF does)"
      % os.environ.get("KS_EXTEND_EG", "0"))
print("  %-9s %5s | %9s %9s | %10s" % ("group", "n", "our_stm", "sf_stm", "over-val"))
for grp in ("collapse", "safe-eg"):
    g = groups.get(grp, [])
    if not g:
        continue
    our = [x[0] for x in g]; sf = [x[1] for x in g]
    over = [o - s for o, s in g]
    print("  %-9s %5d | %+9.2f %+9.2f | %+10.3f" % (grp, len(g), statistics.mean(our), statistics.mean(sf), statistics.mean(over)))
print("  (compare over-val at EG=0 vs EG=1: collapse should shrink = recall; safe-eg should hold = precision)")
