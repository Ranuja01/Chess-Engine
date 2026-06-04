# -*- coding: utf-8 -*-
"""Standalone, deterministic reproduction of the self-play crash — gdb-friendly (no harness, no
pipe protocol, one process).

The engine segfaults on White's 18th move of a specific self-play game, but ONLY with the warm
state built by playing the game (cold-from-FEN is fine). This rebuilds that exact White-side warm
state: one White ChessAI books the opening (no search) and then actually searches each of White's
turns, accumulating the same caches / state_history — so the final search reproduces the crash.

Run it to confirm the segfault:
    PRESET=LIGHTNING python selfplay/repro_crash.py
Then get a backtrace WITHOUT touching the optimizers (gdb on the existing build):
    PRESET=LIGHTNING gdb -batch -ex run -ex 'bt 60' --args python selfplay/repro_crash.py
(If the frames are thin, add -g to setupAI.py's compile args — it coexists with -Ofast/-flto/
-march — rebuild, and re-run the gdb line for file:line numbers.)
"""

import os
import sys

ENGINE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ENGINE_DIR)

import chess
from ChessAI import ChessAI

# The exact self-play game (A vs A, LIGHTNING) up to the move before the crash.
GAME = ("e2e4 e7e5 g1f3 b8c6 f1b5 g8f6 e1g1 f6e4 d2d4 e4d6 b5c6 d7c6 d4e5 d6f5 d1d8 e8d8 "
        "b1c3 c8d7 h2h3 b7b6 f1d1 d8c8 g2g4 f5e7 f3g5 d7e8 f2f4 h7h5 g1f2 h5g4 h3g4 h8h2 "
        "f2g3 h2h8").split()

board = chess.Board()
ai = ChessAI(None, None, board, chess.WHITE)  # White side, same as the crashing server

for ply, uci in enumerate(GAME, start=1):
    if board.turn == chess.WHITE:
        # White's turn: SEARCH (this warms the caches / state_history exactly like the game did),
        # then follow the game's actual move to stay on the exact path.
        booked = board.ply() < 30  # book region (move_stack < 30): mv will be a book move
        mv = ai.alphaBetaWrapper()
        sys.stderr.write(f"[repro] ply {ply}: engine played {mv} (game {uci}) "
                         f"{'[book]' if booked else '[search]'}\n")
        sys.stderr.flush()
    board.push(chess.Move.from_uci(uci))

# White to move (ply 35) — the search that crashed in the game.
sys.stderr.write(f"[repro] ply 35: searching crash position {board.fen()} ...\n")
sys.stderr.flush()
mv = ai.alphaBetaWrapper()
sys.stderr.write(f"[repro] NO CRASH — engine played {mv}\n")  # if we get here, it didn't reproduce
sys.stderr.flush()
