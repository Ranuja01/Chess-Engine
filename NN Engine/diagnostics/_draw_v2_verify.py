# -*- coding: utf-8 -*-
"""Prove DRAW_V2_CLASS actually EXECUTES and changes the output, on positions it is meant to fire on.
★ WAC/STS are midgame suites -- KvK/KBvK/KBvKB never occur there, so "no bench regression" would be a
vacuous pass. This checks the eval directly on hand-built positions.

  EVAL_ARM=1 <rung-2 knobs> DRAW_V2_CLASS=0|1  pyrun diagnostics/_draw_v2_verify.py
Run BOTH arms: OFF must show the "expect 0" rows NON-zero (else a 0 proves nothing); ON must show every row OK.
Exit 0 = every row as expected.
"""
import os, sys
# KEY=VAL args -> environment BEFORE the engine loads, so this runs through the prompt-free runner form
# (`overnight_runner.sh pyrun diagnostics/_draw_v2_verify.py EVAL_ARM=1 DRAW_V2_CLASS=1 ...`) as well as with an
# env prefix. Knobs latch at ChessAI construction.
for _a in sys.argv[1:]:
    if "=" in _a:
        _k, _v = _a.split("=", 1)
        os.environ[_k] = _v
ENGINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # diagnostics/ -> NN Engine/
sys.path.insert(0, ENGINE)
os.chdir(ENGINE)
import chess, ChessAI

seed = chess.Board()
ai = ChessAI.ChessAI(None, None, seed, seed.turn) if hasattr(ChessAI, 'ChessAI') else ChessAI(None, None, seed, seed.turn)

# ☠️ Every "expect 0" case must be ASYMMETRIC, or material and PSTs cancel on their own and the row
# proves nothing about the classifier. The first version of this file used mirrored bishop/knight
# placements and KBvKB/KNvKN read 0 in BOTH arms -- a vacuous pass on the only non-trivial case.
CASES = [
    ("KvK            ", "8/8/4k3/8/8/4K3/8/8 w - - 0 1",            True),
    ("KBvK           ", "8/8/4k3/8/8/4K3/6B1/8 w - - 0 1",          True),
    ("KNvK           ", "8/8/4k3/8/8/4K3/6N1/8 w - - 0 1",          True),
    ("KBvKB asym     ", "8/8/4k3/8/8/1b2K3/6B1/8 w - - 0 1",        True),
    ("KNvKN asym     ", "8/8/4k3/8/8/1n2K3/6N1/8 w - - 0 1",        True),
    ("KRvK  (a WIN)  ", "8/8/4k3/8/8/4K3/6R1/8 w - - 0 1",          False),
    ("KRBvKR (a WIN) ", "8/8/8/8/8/4k3/5r2/4KRB1 w - - 0 1",        False),
    ("KQvK  (a WIN)  ", "8/8/4k3/8/8/4K3/6Q1/8 w - - 0 1",          False),
    # 2026-09-13 additions -- positives (rule must fire) and NEGATIVE CONTROLS (rule must NOT fire), so an
    # over-broad rule cannot pass silently. All asymmetric, so the rule-OFF arm reads non-zero.
    ("KBvKN asym     ", "8/8/4k3/8/8/1n2K3/6B1/8 w - - 0 1",        True),
    ("KNNvK          ", "8/8/4k3/8/8/4K3/5NN1/8 w - - 0 1",         True),
    ("wrongB fortress", "1k6/8/8/8/P7/4K3/1B6/8 w - - 0 1",         True),   # dark B, a8 light, Kb8 beside it
    ("rightB (ctrl)  ", "1k6/8/8/8/P7/4K3/2B5/8 w - - 0 1",         False),  # light B controls a8 -> must NOT fire
    ("wrongB far K   ", "7k/8/8/8/P7/4K3/1B6/8 w - - 0 1",          False),  # king far from a8 -> must NOT fire
    # A normal middlegame: must be untouched and non-zero.
    # ☠️ Replaced 2026-09-14: the old row (an equal-material Italian) read EXACTLY 0 under the shipped rung-2 knobs in BOTH
    # classifier arms -- a coincidental cancellation of PST/KS/pawn terms, so it failed while proving nothing. A control
    # must be non-zero BY CONSTRUCTION: this one is a pawn up (Black's e-pawn is gone), and far too many pieces for any
    # draw case to apply.
    ("midgame P-up   ", "r1bqkb1r/pppp1ppp/2n2n2/8/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 5", False),
]

# EXACT KPK bitbase (DRAW_V2_KPK_EXACT, 2026-09-14). Every verdict below is from the Lichess tablebase or is the exact
# colour mirror of one. Drawn rows expect 0 ONLY when the bitbase is live (it sits inside draw_class, so it needs
# DRAW_V2_CLASS too); otherwise they must read NON-zero, so the OFF arm proves the rule is what zeroes them.
# ☠️ The 6th-rank row is here because my own prior called it a draw; the tablebase says WIN in 21. It must stay non-zero.
kpk_live = os.environ.get("DRAW_V2_CLASS") == "1" and os.environ.get("DRAW_V2_KPK_EXACT") == "1"
CASES += [
    ("KPK opp w DRAW ", "8/4k3/8/4K3/4P3/8/8/8 w - - 0 1",           kpk_live),  # TB draw
    ("KPK opp b WIN  ", "8/4k3/8/4K3/4P3/8/8/8 b - - 0 1",           False),     # TB loss for Black (DTM -28)
    ("KPK mir b DRAW ", "8/8/8/4p3/4k3/8/4K3/8 b - - 0 1",           kpk_live),  # colour mirror of the draw
    ("KPK mir w WIN  ", "8/8/8/4p3/4k3/8/4K3/8 w - - 0 1",           False),     # colour mirror of the win
    ("KPK rookP DRAW ", "7k/8/6K1/7P/8/8/8/8 w - - 0 1",             kpk_live),  # TB draw, h-file (file mirror)
    ("KPK 6th WIN    ", "4k3/8/4K3/4P3/8/8/8/8 w - - 0 1",           False),     # TB win (DTM 21)
]

arm = os.environ.get("EVAL_ARM", "?")
dc  = os.environ.get("DRAW_V2_CLASS", "?")
print("EVAL_ARM=%s DRAW_V2_CLASS=%s DRAW_V2_KPK_EXACT=%s (kpk rows expect 0: %s)"
      % (arm, dc, os.environ.get("DRAW_V2_KPK_EXACT", "?"), kpk_live))
print("  %-16s %12s   %s" % ("case", "eval (mp)", "expect 0?"))
bad = 0
for name, fen, should_be_zero in CASES:
    v = ai.ev(chess.Board(fen))
    ok = (v == 0) if should_be_zero else (v != 0)
    if not ok:
        bad += 1
    print("  %-16s %12d   %-5s  %s" % (name, v, should_be_zero, "OK" if ok else "<<< MISMATCH"))
print("\n%s" % ("ALL AS EXPECTED" if bad == 0 else "%d MISMATCHES" % bad))
sys.exit(1 if bad else 0)
