# -*- coding: utf-8 -*-
"""Persistent per-side engine process for the self-play harness.

One of these runs per side, in its own OS process — so the engine's file-scope global caches
(`searchEvalCache`, history/killers/…, `Config::side_to_play`) are fully isolated from the other
side. The driver (`selfplay.py`) talks to it over a tiny line protocol on stdin/stdout:

    PUSH <uci>   apply a move to the board, no search        -> "OK" | "ERR <msg>"
    GO           search for the side to move, push its move  -> "MOVE <uci> <calc-json>" | "RESIGN <calc-json>"
    QUIT         exit

Channel split (the key trick): the C++ engine fire-hoses std::cout/std::cerr during a search, which
would corrupt the protocol. So the `alphaBetaWrapper()` call is wrapped in `_capture_fds()` —
fds 1 AND 2 are redirected to a temp file for the duration (captured + parsed), then restored, and
only the short protocol reply is written to the real stdout pipe (with an explicit flush, since a
pipe is block-buffered). Config knobs come from the environment the driver sets per process; the
keras models are unused by the C++ engine, so we pass None (no TensorFlow).
"""

import os
import sys
import json
import re
import contextlib
import tempfile
import argparse
from timeit import default_timer as timer

# The engine .so lives one level up in NN Engine/.
ENGINE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ENGINE_DIR)

import chess
from ChessAI import ChessAI


@contextlib.contextmanager
def _capture_fds():
    """Redirect fds 1 and 2 to a temp file for the duration, then restore. Captures everything the
    C++ engine prints (std::cout cache/eval dump + std::cerr [aspiration]/[toggles]) without it
    reaching the driver pipe."""
    sys.stdout.flush()
    sys.stderr.flush()
    saved1, saved2 = os.dup(1), os.dup(2)
    tmp = tempfile.TemporaryFile(mode='w+')
    try:
        os.dup2(tmp.fileno(), 1)
        os.dup2(tmp.fileno(), 2)
        yield tmp
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(saved1, 1)
        os.dup2(saved2, 2)
        os.close(saved1)
        os.close(saved2)


# --- parsing the engine's captured stdout into a structured per-move record ---
_RE_EVAL = re.compile(r'Evaluation:\s*(-?\d+)')
_RE_DEPTH = re.compile(r'SEARCHING DEPTH:\s*(\d+)')
_RE_NODES = re.compile(r'Positions Analyzed:\s*(\d+)')
_RE_NPS = re.compile(r'Average Static Analysis Speed:\s*([\d.]+)')
_RE_TIME = re.compile(r'Time Taken:\s*([\d.]+)')
_RE_ASP = re.compile(r'\[aspiration\] windows=(\d+) fails=(\d+) fallbacks=(\d+)')
# root-ordering candidate line: "idx  score  prelim  (f,r) -> (f,r)"
_RE_CAND = re.compile(r'^\s*(\d+)\s+(-?\d+)\s+(-?\d+)\s+\((\d),(\d)\)\s*->\s*\((\d),(\d)\)\s*$')
_CACHE_KEYS = [
    ("eval_visits", r'EVAL CACHE VISITS:\s*(\d+)'),
    ("eval_hits", r'EVAL CACHE HITS:\s*(\d+)'),
    ("movegen_visits", r'MOVE GEN CACHE VISITS:\s*(\d+)'),
    ("movegen_hits", r'MOVE GEN CACHE HITS:\s*(\d+)'),
    ("tt_visits", r'TT VISITS:\s*(\d+)'),
    ("tt_probes", r'TT PROBES:\s*(\d+)'),
    ("tt_hits", r'TT HITS:\s*(\d+)'),
    ("q_visits", r'Q SEARCH VISITS:\s*(\d+)'),
]
_CACHE_RES = [(k, re.compile(p)) for k, p in _CACHE_KEYS]


def _sq(f, r):
    return chr(ord('a') + int(f) - 1) + str(int(r))


def parse_calc(raw, keep_raw=False):
    """Pull the rich per-move calc record out of the engine's captured output. Every field is
    best-effort / optional so a format change degrades gracefully rather than crashing a game."""
    def _last_int(rx):
        m = rx.findall(raw)
        return int(m[-1]) if m else None
    def _last_float(rx):
        m = rx.findall(raw)
        return float(m[-1]) if m else None

    rec = {
        "eval": _last_int(_RE_EVAL),
        "depth": max((int(d) for d in _RE_DEPTH.findall(raw)), default=None),
        "nodes": _last_int(_RE_NODES),
        "nps": _last_float(_RE_NPS),
        "engine_time": _last_float(_RE_TIME),
        "booked": ("Book Move" in raw),
    }
    cache = {}
    for key, rx in _CACHE_RES:
        m = rx.findall(raw)
        if m:
            cache[key] = int(m[-1])
    if cache:
        rec["cache"] = cache
    a = _RE_ASP.findall(raw)
    if a:
        w, f, fb = a[-1]
        rec["aspiration"] = {"windows": int(w), "fails": int(f), "fallbacks": int(fb)}

    # Root move ordering with scores: take the LAST block (deepest iteration). A candidate line is
    # "idx score prelim (f,r)->(f,r)"; restart the collection whenever a new run of idx==0 begins.
    ordering = []
    for line in raw.splitlines():
        m = _RE_CAND.match(line)
        if not m:
            continue
        idx = int(m.group(1))
        if idx == 0:
            ordering = []
        ordering.append({
            "i": idx,
            "score": int(m.group(2)),
            "prelim": int(m.group(3)),
            "uci": _sq(m.group(4), m.group(5)) + _sq(m.group(6), m.group(7)),
        })
    if ordering:
        rec["ordering"] = ordering

    if keep_raw:
        rec["raw"] = raw
    return rec


def _reply(line):
    """Write a single protocol line to the (restored) stdout pipe and flush — the pipe is
    block-buffered, so without the flush the driver would hang waiting for it."""
    sys.stdout.write(line + "\n")
    sys.stdout.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--color", required=True, choices=["white", "black"],
                    help="the fixed colour this server plays (its side_to_play)")
    ap.add_argument("--start-fen", default=chess.STARTING_FEN,
                    help="initial position (default: standard start)")
    ap.add_argument("--keep-raw", action="store_true",
                    help="include the full raw engine output in each calc-json")
    args = ap.parse_args()

    side = (args.color == "white")  # chess.WHITE is True
    board = chess.Board(args.start_fen)

    # Construct once; the engine reads its config-knob env vars in initialize_engine here, and prints
    # the [toggles] line + a cache dump — capture so it can't corrupt the protocol. Models unused.
    with _capture_fds() as cap:
        ai = ChessAI(None, None, board, side)
        cap.seek(0)
        startup = cap.read()
    # surface the [toggles] line on our stderr for debugging (the driver logs it per side)
    for ln in startup.splitlines():
        if ln.startswith("[toggles]"):
            sys.stderr.write(ln + "\n")
    sys.stderr.flush()

    _reply("READY")

    # readline() (not `for line in sys.stdin`) — the latter read-aheads and can deadlock a
    # request/response protocol over pipes.
    while True:
        raw_cmd = sys.stdin.readline()
        if not raw_cmd:
            break
        cmd = raw_cmd.strip()
        if not cmd:
            continue
        if cmd == "QUIT":
            break
        elif cmd.startswith("PUSH "):
            uci = cmd[5:].strip()
            try:
                board.push(chess.Move.from_uci(uci))
                _reply("OK")
            except Exception as e:
                _reply("ERR " + str(e))
        elif cmd == "GO":
            with _capture_fds() as cap:
                move = ai.alphaBetaWrapper()
                cap.seek(0)
                out = cap.read()
            calc = json.dumps(parse_calc(out, keep_raw=args.keep_raw))
            if move is None:
                # alphaBetaWrapper returns None when the engine resigns (score <= -15000).
                _reply("RESIGN " + calc)
            else:
                board.push(move)
                _reply("MOVE " + move.uci() + " " + calc)
        else:
            _reply("ERR unknown command: " + cmd)


if __name__ == "__main__":
    main()
