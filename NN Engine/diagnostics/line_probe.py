# -*- coding: utf-8 -*-
"""
Line probe — localize WHY the engine misjudges a tactical position, with
cache-isolated evaluations.

The engine's global caches (TT / eval / q / move-gen) and history tables are
NEVER cleared within a process (update_cache only prints stats). So evaluating a
position and then its child in the same process is contaminated: the child reuses
the parent's reduced/window-bounded cached values. For a parent->child line probe
that can make a *selection* bug look like an *eval* bug. To avoid it, EACH
position is evaluated in its own fresh subprocess (cold caches).

For each case it: (0) evaluates the root (where the engine picks the wrong move),
(1) forces the known best move, then (2..N) lets the engine play forward. Every
eval is normalized to the ATTACKER's point of view (the root side to move), so a
real win shows as a steadily large positive number.

Reading it (step 1 = AFTER the forced best move is the decisive one):
  - eval jumps strongly POSITIVE  -> the win is reachable at this depth; the root
                                     FAILED TO SELECT it -> ordering / pruning bug.
  - eval stays ~0 or NEGATIVE      -> the engine can't SEE the win when handed it
                                     -> eval (king safety) or pure depth.
  - starts positive then drifts to 0 over the next plies -> win leaking at the
                                     horizon / in the eval.

Run (WSL, from NN Engine/), at the SAME fixed depth as the failing run (e.g. 10):
    python line_probe.py
"""

import os
import re
import sys

import chess

# ---- Worker mode: evaluate exactly ONE FEN in this fresh process, print a
#      single parseable result line, exit. Fresh process => cold caches. ----
if len(sys.argv) > 1:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    import platform
    import tempfile
    import contextlib
    # tool lives in NN Engine/diagnostics/; the built ChessAI .so + Models are one
    # level up in NN Engine/.
    _ENGINE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, _ENGINE_DIR)
    from ChessAI import ChessAI

    # Models unused by the C++ engine — pass None and skip TensorFlow (CHESS_ENABLE_TF to restore).
    _blackModel = None
    _whiteModel = None

    @contextlib.contextmanager
    def _captured_stdout():
        sys.stdout.flush()
        saved = os.dup(1)
        tmp = tempfile.TemporaryFile(mode='w+')
        try:
            os.dup2(tmp.fileno(), 1)
            yield tmp
        finally:
            sys.stdout.flush()
            os.dup2(saved, 1)
            os.close(saved)

    _board = chess.Board(sys.argv[1])
    with _captured_stdout() as _buf:
        _ai = ChessAI(_blackModel, _whiteModel, _board, _board.turn)
        _move = _ai.alphaBetaWrapper()
        # Subprocess stdout is a fully-buffered pipe, so the Cython print() of the
        # eval is still buffered here — flush it into the temp file before reading.
        sys.stdout.flush()
        _buf.seek(0)
        _out = _buf.read()
    _evs = re.findall(r'Evaluation:\s*(-?\d+)', _out)
    _ev = _evs[-1] if _evs else 'None'
    _mv = _move.uci() if _move is not None else 'None'
    _booked = '1' if 'Book Move' in _out else '0'
    print(f"PROBE_RESULT {_ev} {_mv} {_booked}")
    sys.exit(0)


# ---- Orchestrator mode ----
import subprocess

SELFPLAY_PLIES = 3  # plies to let the engine play forward after the forced best move

# (id, root FEN, best move in UCI). Deep-misevals + the trivial regression.
PROBES = [
    ("WAC.213", "3r1r1k/1b4pp/ppn1p3/4Pp1R/Pn5P/3P4/4QP2/1qB1NKR1 w - - 0 1", "h5h7"),  # Rxh7+ (got worse with the q-cache fix)
    ("WAC.018", "R7/P4k2/8/8/8/8/r7/6K1 w - - 0 1",                          "a8h8"),  # Rh8  (trivial R+P endgame that regressed)
    ("WAC.204", "r1b1qrk1/1p3ppp/p1p5/3Nb3/5N2/P7/1P4PQ/K1R1R3 w - - 0 1",    "e1e5"),  # Rxe5 (deep miseval ~-5000)
    ("WAC.283", "3q1rk1/4bp1p/1n2P2Q/3p1p2/6r1/Pp2R2N/1B4PP/7K w - - 0 1",    "h3g5"),  # Ng5  (deep miseval ~-8000)
]


def eval_fen(fen):
    """Evaluate one FEN in a fresh subprocess (cold caches). Returns (eval, uci, booked)."""
    proc = subprocess.run([sys.executable, os.path.abspath(__file__), fen],
                          capture_output=True, text=True)
    m = re.search(r'PROBE_RESULT\s+(\S+)\s+(\S+)\s+(\S+)', proc.stdout)
    if not m:
        sys.stderr.write(proc.stdout[-2000:] + "\n" + proc.stderr[-2000:] + "\n")
        return None, None, False
    ev = None if m.group(1) == 'None' else int(m.group(1))
    mv = None if m.group(2) == 'None' else m.group(2)
    return ev, mv, (m.group(3) == '1')


def probe(label, fen, bm_uci):
    board = chess.Board(fen)
    attacker = board.turn  # normalize all evals to this side's view

    print(f"\n=== {label}  (best move = {bm_uci}) ===")
    print(f"{'step':>4}  {'stm':<4} {'eval(atk)':>10}  {'engine':<8} note")

    def show(step, note):
        ev, mv, booked = eval_fen(board.fen())
        norm = "" if ev is None else (ev if board.turn == attacker else -ev)
        stm = "atk" if board.turn == attacker else "def"
        tag = " [BOOK!]" if booked else ""
        print(f"{step:>4}  {stm:<4} {str(norm):>10}  {str(mv):<8} {note}{tag}")
        return mv

    show(0, "root (engine's own choice)")
    try:
        board.push(chess.Move.from_uci(bm_uci))
    except Exception as e:
        print(f"   !! could not play best move {bm_uci}: {e}")
        return

    for step in range(1, SELFPLAY_PLIES + 1):
        note = "AFTER forced best move" if step == 1 else "engine self-play"
        mv = show(step, note)
        if mv is None or board.is_game_over():
            break
        try:
            board.push(chess.Move.from_uci(mv))
        except Exception:
            break


def main():
    print(f"Line probe — {len(PROBES)} cases, cache-isolated (subprocess per position).")
    print("Run at a FIXED depth equal to the failing run (e.g. 10).")
    for label, fen, bm in PROBES:
        probe(label, fen, bm)


if __name__ == "__main__":
    main()
