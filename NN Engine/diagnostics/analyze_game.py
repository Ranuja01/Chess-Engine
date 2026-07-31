# -*- coding: utf-8 -*-
"""Objective per-move analysis of a game with SF18-search: eval after each half-move (white-POV cp), the
per-move swing, and a flag on blunders/mistakes for the side that just moved. Finds where the game turned.
  pyrun diagnostics/analyze_game.py [DEPTH=20]
Moves are embedded (SAN). Prints a trajectory + the biggest swings + best-move-instead at the blunder.
"""
import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
DEPTH = int(os.environ.get('DEPTH', '20'))
THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(THIS), "selfplay"))
import chess, chess.engine
from arbiter import find_stockfish

# Moves come from ks_sets/game_moves.txt (one line of space-separated SAN) so the tool is reusable per game.
MOVES_FILE = os.path.join(THIS, "ks_sets", "game_moves.txt")
SAN = open(MOVES_FILE).read().split()

sf = chess.engine.SimpleEngine.popen_uci(find_stockfish())


def ev(board):
    i = sf.analyse(board, chess.engine.Limit(depth=DEPTH))
    s = i["score"].white()
    if s.is_mate():
        return (10000 if s.mate() > 0 else -10000)
    return s.score()


def best(board):
    i = sf.analyse(board, chess.engine.Limit(depth=DEPTH))
    return board.san(i["pv"][0]) if i.get("pv") else "?"


board = chess.Board()
prev = ev(board)
print("move   played   eval(wcp)  swing   note")
rows = []
for i, san in enumerate(SAN):
    mover = "W" if board.turn == chess.WHITE else "B"
    bm = best(board)
    board.push_san(san)
    e = ev(board)
    # swing from the MOVER's perspective (negative = the mover worsened their position)
    swing = (e - prev) if mover == "W" else (prev - e)
    note = ""
    if swing <= -300:
        note = "?? BLUNDER"
    elif swing <= -150:
        note = "? mistake"
    elif swing <= -80:
        note = "?! inaccuracy"
    mn = i // 2 + 1
    tag = "%d.%s" % (mn, "" if mover == "W" else "..")
    rows.append((tag, mover, san, e, swing, note, bm))
    print("%-6s %-7s %+8d  %+6d  %s%s" % (tag, san, e, swing, note,
          ("   (best: %s)" % bm) if note else ""))
    prev = e

sf.quit()
print("\nBiggest swings (mover worsened by most):")
for tag, mover, san, e, swing, note, bm in sorted(rows, key=lambda r: r[4])[:6]:
    print("  %-6s %-6s swing %+5d  eval now %+d   best was %s" % (tag, san, swing, e, bm))
