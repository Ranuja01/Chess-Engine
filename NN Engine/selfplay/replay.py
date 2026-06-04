# -*- coding: utf-8 -*-
"""Generic crash replay — re-run ANY captured self-play crash in one isolated process.

Given the move list up to a crash and the crashed side's colour (a crash bundle from the driver, or
passed directly), this reconstructs that side's engine and replays the game: it books / searches the
side's own turns (rebuilding the warm caches / state_history exactly as the game did) and forces the
recorded moves so the board follows the game precisely, then runs the search that crashed.

The config knobs come from the environment, so prefix them (or use the bundle's replay_cmd):
    PRESET=LIGHTNING python selfplay/replay.py --color white --moves "e2e4 e7e5 ..."
    python selfplay/replay.py --bundle selfplay/games/<tag>/crash.json

NOTE: a time-limited preset (LIGHTNING/BLITZ) is non-deterministic, so a replay may not reproduce a
timing-specific crash. For a deterministic attempt, force a fixed depth:
    PRESET=LONG_FORMAT MAX_DEPTH=12 python selfplay/replay.py --bundle ...
Run it under gdb for a backtrace (add -g to setupAI.py first):
    PRESET=LONG_FORMAT MAX_DEPTH=12 gdb -batch -ex run -ex 'bt 80' --args python selfplay/replay.py --bundle ...
"""

import os
import sys
import json
import argparse

ENGINE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ENGINE_DIR)

import chess
from ChessAI import ChessAI


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", help="a crash.json written by the driver")
    ap.add_argument("--color", choices=["white", "black"], help="the side that crashed")
    ap.add_argument("--moves", help="space-separated UCI moves up to the crash position")
    ap.add_argument("--start-fen", default=chess.STARTING_FEN)
    ap.add_argument("--seed-plies", type=int, default=0,
                    help="number of leading plies that were SEEDED in the game (PUSHed, not searched) "
                         "— pushed here without searching so the warm state matches the game exactly. "
                         "Tournament openings are 10 plies, so pass --seed-plies 10 for those bundles.")
    args = ap.parse_args()

    if args.bundle:
        b = json.load(open(args.bundle))
        color = b["crashed_color"]
        moves = b["moves"]
        start_fen = b.get("start_fen", chess.STARTING_FEN)
        sys.stderr.write(f"[replay] bundle: {b['crashed_label']} ({color}) crashed; "
                         f"config={b.get('config','')!r}\n")
    else:
        if not (args.color and args.moves):
            ap.error("need --bundle, or both --color and --moves")
        color, moves, start_fen = args.color, args.moves.split(), args.start_fen

    side = (color == "white")
    board = chess.Board(start_fen)
    ai = ChessAI(None, None, board, side)

    for i, uci in enumerate(moves):
        # Leading seeded plies were PUSHed in the real game (no search), so don't search them here —
        # searching them would over-warm the caches / state_history and the crash may not reproduce.
        if i >= args.seed_plies and board.turn == side:
            ai.alphaBetaWrapper()          # search the side's turn (warms caches/history)
        board.push(chess.Move.from_uci(uci))  # force the recorded move so the board follows the game

    sys.stderr.write(f"[replay] searching crash position {board.fen()} ...\n")
    sys.stderr.flush()
    mv = ai.alphaBetaWrapper()             # <-- the search that crashed
    sys.stderr.write(f"[replay] NO CRASH — engine played {mv}\n")
    sys.stderr.flush()


if __name__ == "__main__":
    main()
