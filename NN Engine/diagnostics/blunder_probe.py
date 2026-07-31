# -*- coding: utf-8 -*-
"""Classify a game blunder as EVAL vs SEARCH: replay ks_sets/game_moves.txt to the position just BEFORE a
target ply, then ask OUR engine (run_one, at a DEEP setting via MAX_DEPTH/PRESET env) what it would play.
If deep search AVOIDS the blundered move -> it was a blitz-DEPTH (search) miss. If deep search STILL plays it
-> an EVAL misjudgment. Also dumps our static eval total for context.

  MAX_DEPTH=14 PRESET=LONG_FORMAT pyrun diagnostics/blunder_probe.py
Targets are (ply_before, played_san) pairs embedded below.
"""
import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'; os.environ.setdefault('KS_SAFE_CHECK_DEF', '5')
os.environ.setdefault('MAX_DEPTH', '14'); os.environ.setdefault('PRESET', 'LONG_FORMAT')
os.environ.setdefault('USE_OPENING_BOOK', '0')
THIS = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(THIS)); sys.path.insert(0, THIS)
import chess
from tactical_test import run_one
from ChessAI import ChessAI
ai = ChessAI(None, None, chess.Board(), True)

SAN = open(os.path.join(THIS, "ks_sets", "game_moves.txt")).read().split()
# (plies to play before the blunder, the SAN actually played, label)
TARGETS = [(55, "c5", "28...c5"), (87, "Ke6", "44...Ke6")]

for nply, played, label in TARGETS:
    b = chess.Board()
    for mv in SAN[:nply]:
        b.push_san(mv)
    fen = b.fen()
    bd = ai.ev_breakdown(b)
    our_static = -bd.get("total", 0.0) / 1000.0    # white-POV pawns
    r = run_one(fen, set())
    # what SAN does our deep engine pick?
    try:
        deep_san = b.san(chess.Move.from_uci(r["uci"]))
    except Exception:
        deep_san = r["uci"]
    verdict = ("EVAL bug (still plays the blunder deep)" if deep_san == played
               else "SEARCH/depth (deep engine avoids it)")
    print("=== %s ===" % label)
    print("  fen: %s" % fen)
    print("  our static total (white-POV): %+.2f" % our_static)
    print("  played in game: %s   |   our DEEP engine (d=%s): %s  eval=%s depth=%s" %
          (played, os.environ.get("MAX_DEPTH"), deep_san, r.get("eval"), r.get("depth")))
    print("  VERDICT: %s\n" % verdict)
