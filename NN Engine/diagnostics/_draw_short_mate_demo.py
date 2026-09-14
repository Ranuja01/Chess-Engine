# -*- coding: utf-8 -*-
"""Does the engine still FIND a short forced mate when a draw rule scores the position 0?

This is the empirical half of the DTM-weighted gate's premise ("ablate, don't read"). The structural half
(null move disabled below 7 pieces; RFP/futility cannot fire in a 0-or-mate subtree) is in
EVAL-V2-SLICE1-DRAW-DESIGN.md §2d. Here we run the actual search on tablebase-won positions that
DRAW_V2_CLASS flags drawn, and check the search score is a MATE, not 0.

  EVAL_ARM=1 <rung-2 knobs> DRAW_V2_CLASS=0|1 FENS=<file, one FEN per line> [MAX_DEPTH=12]
      pyrun diagnostics/_draw_short_mate_demo.py
☠️ Run the two arms from a SCRIPT FILE. An inline `wsl.exe ... DRAW_V2_CLASS=\$DC` from PowerShell corrupted this exact
run on 2026-09-13 into "ON twice" (the header echo is what caught it -- read it).
Validated 2026-09-13 on a known mate-in-1 (6k1/5ppp/8/8/8/8/5PPP/R5K1 w -> 9999997, a1a8) plus a non-mate control.

Prints, per position: static eval under the current knobs (0 => the draw rule fired), the search score,
whether that score is a mate, the move played, and nodes. Exit 0 = a mate was found on every position.
"""
import os, sys, re, tempfile

os.environ.setdefault("USE_OPENING_BOOK", "0")   # a fresh FEN has an empty move_stack => book is consulted
os.environ.setdefault("MAX_DEPTH", "12")

ENGINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # diagnostics/ -> NN Engine/
sys.path.insert(0, ENGINE)
os.chdir(ENGINE)

import chess
import ChessAI as _mod
ChessAI = _mod.ChessAI if hasattr(_mod, 'ChessAI') else _mod

FENS = os.environ.get("FENS")
# Mate scores are `9999999 - moveNum` (search_engine.cpp get_board_evaluation), and RFP's own guard treats
# |score| >= 9,000,000 as mate-valued -- use the engine's convention, not a guessed threshold.
MATE_ABS = int(os.environ.get("MATE_ABS", "9000000"))

_EVAL = re.compile(r"Evaluation:\s*(-?\d+)")
_NODES = re.compile(r"Positions Analyzed:\s*(\d+)")


def search_capturing_fd1(ai):
    """C++ prints to file descriptor 1, which Python's sys.stdout redirection cannot see -- dup fd 1."""
    sys.stdout.flush()
    saved = os.dup(1)
    with tempfile.TemporaryFile(mode="w+b") as tmp:
        os.dup2(tmp.fileno(), 1)
        try:
            mv = ai.alphaBetaWrapper()
        finally:
            sys.stdout.flush()
            os.dup2(saved, 1)
            os.close(saved)
        tmp.seek(0)
        out = tmp.read().decode("utf-8", "replace")
    return mv, out


def main():
    if not FENS or not os.path.exists(FENS):
        print("FENS file missing: %s" % FENS); return 2
    fens = [ln.strip() for ln in open(FENS) if ln.strip() and not ln.startswith("#")]
    print("EVAL_ARM=%s DRAW_V2_CLASS=%s MAX_DEPTH=%s MATE_ABS=%d" % (
        os.environ.get("EVAL_ARM", "?"), os.environ.get("DRAW_V2_CLASS", "?"),
        os.environ.get("MAX_DEPTH"), MATE_ABS))
    print("  %-8s %10s %12s %-6s %-8s %10s  %s" % ("static", "", "search", "MATE?", "move", "nodes", "fen"))
    found = 0
    for fen in fens:
        board = chess.Board(fen)
        ai = ChessAI(None, None, board, board.turn)
        static = ai.ev(board)
        ai2 = ChessAI(None, None, board, board.turn)
        mv, out = search_capturing_fd1(ai2)
        ev = _EVAL.findall(out)
        nd = _NODES.findall(out)
        score = int(ev[-1]) if ev else None
        is_mate = score is not None and abs(score) >= MATE_ABS
        found += 1 if is_mate else 0
        print("  %8d %10s %12s %-6s %-8s %10s  %s" % (
            static, "", score if score is not None else "n/a", "YES" if is_mate else "no",
            mv.uci() if mv else "none", nd[-1] if nd else "?", fen))
    print("\nmate found in %d / %d positions" % (found, len(fens)))
    return 0 if found == len(fens) else 1


if __name__ == "__main__":
    sys.exit(main())
