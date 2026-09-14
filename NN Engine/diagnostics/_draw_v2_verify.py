# -*- coding: utf-8 -*-
"""Prove DRAW_V2_CLASS actually EXECUTES and changes the output, on positions it is meant to fire on.
★ WAC/STS are midgame suites -- KvK/KBvK/KBvKB never occur there, so "no bench regression" would be a
vacuous pass. This checks the eval directly on hand-built positions.

  EVAL_ARM=1 <rung-2 knobs> DRAW_V2_CLASS=0|1  pyrun diagnostics/_draw_v2_verify.py
Run BOTH arms: OFF must show the "expect 0" rows NON-zero (else a 0 proves nothing); ON must show every row OK.
Exit 0 = every row as expected.
"""
import os, sys
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
    # A normal asymmetric middlegame: must be untouched and non-zero.
    ("midgame        ", "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4", False),
]

arm = os.environ.get("EVAL_ARM", "?")
dc  = os.environ.get("DRAW_V2_CLASS", "?")
print("EVAL_ARM=%s DRAW_V2_CLASS=%s" % (arm, dc))
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
