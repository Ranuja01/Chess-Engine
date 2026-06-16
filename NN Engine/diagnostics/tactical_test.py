# -*- coding: utf-8 -*-
"""
Tactical test harness for the C++ chess engine.

Runs a list of (position, best-move(s)) tactical puzzles through the engine and
reports how many it solves. This is the measurement tool for *behavioural*
changes (the q-cache bound flag, the delta-pruning margin, search-soundness
tweaks) that — unlike the pure-speed work — deliberately change the engine's
output and therefore cannot be checked for byte-identity. Strength shows up as
"solves the same or more puzzles", not "produces identical numbers".

Run (WSL, from NN Engine/):
    python tactical_test.py                     # built-in smoke tests
    python tactical_test.py wac.epd             # full suite, output tag = 'run'
    python tactical_test.py wac.epd depth10     # tag outputs with 'depth10'

Outputs (per run, named by <tag> so successive runs don't clobber each other):
    tactical_results_<tag>.csv   full per-position data (eval, depth, nodes, time)
    tactical_fails_<tag>.epd     the failed positions, ready to re-run deeper:
                                     python tactical_test.py tactical_fails_<tag>.epd deep

Why both a CSV and a fails file: to classify failures into (1) below-horizon
(solvable with more depth), (2) classic/easy but missed (ordering/pruning gap),
or (3) fundamental soundness (engine searches deep enough but returns the wrong
result). Re-running the fails at a higher fixed depth separates (1) from (2)/(3);
the recorded eval + depth help separate (2) from (3).

Notes:
- Compare runs at a FIXED DEPTH. Time-based control makes the depth reached (and
  thus the solve set) timing-dependent, which confounds a behavioural A/B — a
  correctness fix that adds nodes can lose depth in a fixed time budget and look
  like a regression. Keep the depth identical across the baseline and the change.
- Positions are fed by constructing a fresh ChessAI per FEN; __cinit__ seeds the
  engine state from the board. get_engine_move resets per-search globals, so an
  in-process loop is clean; the persistent caches are key-checked.
- alphaBetaWrapper() consults the opening book when the board has < 30 plies of
  history. A FEN-loaded board has none, so the book IS queried; non-opening
  tactical FENs miss it. Any position that returns a book move is flagged BOOK
  and excluded from the score.
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # silence TensorFlow startup chatter

import re
import csv
import sys
import platform
import tempfile
import contextlib
from timeit import default_timer as timer

# --- diagnostics layout: tools live in NN Engine/diagnostics/; the built ChessAI
#     .so lives one level up in NN Engine/, inputs in suites/, outputs in results/. ---
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ENGINE_DIR = os.path.dirname(THIS_DIR)            # NN Engine/  (has ChessAI*.so)
SUITES_DIR = os.path.join(THIS_DIR, 'suites')
RESULTS_DIR = os.path.join(THIS_DIR, 'results')
sys.path.insert(0, ENGINE_DIR)

import chess
from ChessAI import ChessAI


# --- The C++ search never uses the keras models (legacy NN path), so we pass None and skip the slow
#     TensorFlow import entirely. Set CHESS_ENABLE_TF=1 (+ load real models here) to restore it. ---
blackModel = None
whiteModel = None


# --- Built-in smoke tests: simple, hand-verified, unambiguous tactics. Tuple is
#     (FEN, {acceptable best moves in UCI}, label, raw_epd_or_None). ---
INLINE_POSITIONS = [
    ("6k1/5ppp/8/8/8/8/8/R6K w - - 0 1", {"a1a8"}, "smoke: back-rank mate", None),
    ("4k3/8/8/3q4/4Q3/8/8/4K3 w - - 0 1", {"e4d5"}, "smoke: win queen", None),
    ("8/P6k/8/8/8/8/7K/8 w - - 0 1", {"a7a8q"}, "smoke: promotion", None),
]


@contextlib.contextmanager
def _captured_stdout():
    """Capture everything written to stdout fd 1 (both Python prints and the
    C++ engine's std::cout) into a buffer, so the per-position run stays quiet
    and we can parse it for eval / depth / book hits."""
    sys.stdout.flush()
    saved_fd = os.dup(1)
    tmp = tempfile.TemporaryFile(mode='w+')
    try:
        os.dup2(tmp.fileno(), 1)
        yield tmp
    finally:
        sys.stdout.flush()
        os.dup2(saved_fd, 1)
        os.close(saved_fd)


def load_epd(path):
    """Parse an EPD file into [(fen, {best_uci,...}, id, raw_line)]. Uses
    python-chess to resolve the `bm` operation against each position, so the
    expected moves are trustworthy rather than hand-transcribed. Lines without a
    `bm` operation are skipped."""
    positions = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            board = chess.Board()
            try:
                ops = board.set_epd(line)
            except Exception:
                continue
            best = ops.get('bm')
            if not best:
                continue
            positions.append((board.fen(), {m.uci() for m in best}, ops.get('id', ''), line))
    return positions


_EVAL_RE = re.compile(r'Evaluation:\s*(-?\d+)')
_DEPTH_RE = re.compile(r'SEARCHING DEPTH:\s*(\d+)')
_NODES_RE = re.compile(r'Positions Analyzed:\s*(\d+)')


def _parse_output(out):
    """Pull eval, max depth reached, node count, and a book-hit flag out of the
    engine's captured stdout."""
    ev = int(_EVAL_RE.findall(out)[-1]) if _EVAL_RE.search(out) else None
    depths = _DEPTH_RE.findall(out)
    depth = max(int(d) for d in depths) if depths else None
    nodes = int(_NODES_RE.findall(out)[-1]) if _NODES_RE.search(out) else None
    booked = "Book Move" in out
    return ev, depth, nodes, booked


def run_one(fen, best):
    """Run the engine on one position. Returns a result dict."""
    board = chess.Board(fen)
    t0 = timer()
    with _captured_stdout() as buf:
        ai = ChessAI(blackModel, whiteModel, board, board.turn)
        move = ai.alphaBetaWrapper()
        # The engine's Evaluation/Positions lines are Python prints (ChessAI.pyx) and sit in
        # Python's stdout buffer; flush them into the captured fd before reading, or only the
        # C++ std::cout output (depth, flushed by std::endl) is seen and eval/nodes come back None.
        sys.stdout.flush()
        buf.seek(0)
        out = buf.read()
    dt = timer() - t0

    ev, depth, nodes, booked = _parse_output(out)
    uci = move.uci() if move is not None else None
    solved = (uci in best) and not booked
    return {"uci": uci, "solved": solved, "booked": booked,
            "eval": ev, "depth": depth, "nodes": nodes, "time": dt}


def main():
    epd_path = sys.argv[1] if len(sys.argv) > 1 else None
    tag = sys.argv[2] if len(sys.argv) > 2 else "run"

    # Resolve a bare suite name (e.g. "wac.epd") against suites/; also accept a
    # path the caller passes directly (absolute, or relative to the cwd).
    if epd_path and not os.path.exists(epd_path):
        cand = os.path.join(SUITES_DIR, epd_path)
        if os.path.exists(cand):
            epd_path = cand

    if epd_path and os.path.exists(epd_path):
        positions = load_epd(epd_path)
        source = epd_path
    else:
        positions = INLINE_POSITIONS
        source = "built-in smoke tests"
        if epd_path:
            print(f"(EPD file '{epd_path}' not found — running {source})")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    csv_path = os.path.join(RESULTS_DIR, f"tactical_results_{tag}.csv")
    fails_path = os.path.join(RESULTS_DIR, f"tactical_fails_{tag}.epd")

    print(f"Running {len(positions)} positions from {source}  (tag={tag})\n")
    print(f"{'#':>3}  {'res':<5} {'engine':<6} {'eval':>7} {'d':>3}  {'expected':<22} {'time':>6}  id")

    solved = booked = 0
    rows = []
    fails_epd = []
    for idx, (fen, best, label, raw_epd) in enumerate(positions):
        r = run_one(fen, best)
        if r["booked"]:
            booked += 1
            res = "BOOK"
        else:
            res = "PASS" if r["solved"] else "fail"
            solved += int(r["solved"])
            if not r["solved"] and raw_epd:
                fails_epd.append(raw_epd)

        ev_s = "" if r["eval"] is None else str(r["eval"])
        d_s = "" if r["depth"] is None else str(r["depth"])
        print(f"{idx:>3}  {res:<5} {str(r['uci']):<6} {ev_s:>7} {d_s:>3}  "
              f"{','.join(sorted(best)):<22} {r['time']:>6.2f}  {label}")

        rows.append({"idx": idx, "id": label, "result": res, "engine": r["uci"],
                     "expected": " ".join(sorted(best)), "eval": r["eval"],
                     "depth": r["depth"], "nodes": r["nodes"],
                     "time": round(r["time"], 3), "fen": fen})

    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["idx", "id", "result", "engine",
                                          "expected", "eval", "depth", "nodes", "time", "fen"])
        w.writeheader()
        w.writerows(rows)

    if fails_epd:
        with open(fails_path, "w") as f:
            f.write("\n".join(fails_epd) + "\n")

    valid = len(positions) - booked
    print()
    print(f"Solved {solved}/{valid}"
          + (f" ({booked} skipped as book hits)" if booked else "")
          + f"  |  results -> {csv_path}"
          + (f"  |  {len(fails_epd)} fails -> {fails_path}" if fails_epd else ""))


if __name__ == "__main__":
    main()
