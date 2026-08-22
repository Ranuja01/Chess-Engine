# -*- coding: utf-8 -*-
"""KS specificity/precision probe: on a labelled FEN set (label<TAB>fen), aggregate our king_safety vs SF11's
by group, so a KS change can be judged on BOTH the target (collapse) group AND the collateral (safe) group at
once. The recurring KS trap is fixing the target while lighting up the safe set; this makes the safe set a
first-class output. Run at default and with an un-gating knob to see recall (fires on collapses, matches SF
sign) AND precision (stays quiet on safe positions where SF is also ~0).

Labels: prefix 'ksatk' -> "collapse"; anything else -> "safe-eg" (rename via LABELS=prefix1:name1,...).
Danger sign is taken from the SIDE-TO-MOVE king (our king in danger = bad for mover). We report, per group:
mean |ours|, mean |SF11|, our fire-rate (|ours|>THR), and 'over-fire' = our fires where SF says safe.

  pyrun diagnostics/_ks_specificity.py FENS=<file> [THR=0.30] [KS_PHASE_ZERO=128 ...engine knobs]
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
THR = float(os.environ.get("THR", "0.30"))            # pawns; |KS| above this = "fired"
sf11 = SF11Eval(SF11)
ai = ChessAI(None, None, chess.Board(), True)


def stm_danger(fen, ks_white_pov):
    """KS from the side-to-move's own-king perspective: positive = mover's king endangered (bad)."""
    stm_white = chess.Board(fen).turn == chess.WHITE
    # king_safety white-POV: negative favours Black (=White king danger). Flip to 'mover's own-king danger'.
    return (-ks_white_pov) if stm_white else ks_white_pov


groups = defaultdict(list)
for ln in open(FENS):
    ln = ln.rstrip("\n")
    if not ln.strip():
        continue
    lbl, fen = ln.split("\t", 1)
    grp = "collapse" if lbl.startswith("ksatk") else "safe-eg"
    try:
        bd = ai.ev_breakdown(chess.Board(fen))
        our_wp = -float(bd.get("king_safety", 0)) / 1000.0
        _, s11 = sf11.eval(fen)
        sf_wp = s11.get("King safety", 0.0)
    except Exception:
        continue
    groups[grp].append((stm_danger(fen, our_wp), stm_danger(fen, sf_wp)))

print("KS specificity  (THR=%.2f pawns, KS danger from mover's own-king view; +=mover king in danger)" % THR)
print("  %-9s %5s | %9s %9s | %9s | %-s" % ("group", "n", "our_dang", "sf_dang", "fire%", "over-fire (we fire, SF safe)"))
for grp in ("collapse", "safe-eg"):
    g = groups.get(grp, [])
    if not g:
        continue
    ours = [x[0] for x in g]; sfs = [x[1] for x in g]
    fired = [i for i, v in enumerate(ours) if abs(v) > THR]
    overfire = sum(1 for i in fired if abs(sfs[i]) <= THR)          # we fire where SF is calm
    print("  %-9s %5d | %+9.2f %+9.2f | %8.1f%% | %d/%d fired-where-SF-safe (%.1f%% of group)"
          % (grp, len(g), statistics.mean(ours), statistics.mean(sfs),
             100.0 * len(fired) / len(g), overfire, len(fired), 100.0 * overfire / len(g)))
print("\n  RECALL: collapse fire% high + our_dang sign matches sf_dang (both +).")
print("  PRECISION: safe-eg fire% LOW and over-fire LOW. High safe-eg over-fire = the trap (kills other positions).")
