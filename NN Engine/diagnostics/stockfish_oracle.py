# -*- coding: utf-8 -*-
"""
Stockfish oracle — is a tactical position actually solvable within the same
fixed depth our engine is gated to?

Runs Stockfish at a FIXED depth over an EPD suite and reports which positions it
finds a `bm` for. Point it at OUR engine's fails and the result splits them:
  - Stockfish SOLVES at depth D, we don't  -> genuine deficiency (eval or search);
                                              the win IS reachable at depth D.
  - Stockfish FAILS at depth D too         -> depth-hard; not our engine's fault
                                              at this depth, set aside.
The positions Stockfish solves are written to sf_solved_d<D>.epd — that's the
focused target set for the eval-vs-search work (cross-check each with line_probe).

Caveat: SF's "depth D" is a strong *selective* search (deep extensions on forcing
lines), not a full-width D-ply search. So "SF solves at depth D" means "a strong
engine finds it within a depth-D budget" — the right bar for 'is our engine
deficient', not a proof of 'resolvable in exactly D plies'.

Run (WSL, from NN Engine/):
    python stockfish_oracle.py                     # tactical_fails_run.epd @ depth 10
    python stockfish_oracle.py wac.epd 10
    STOCKFISH_PATH=/usr/bin/stockfish python stockfish_oracle.py tactical_fails_run.epd 10
"""

import os
import sys
import shutil

import chess
import chess.engine

# diagnostics layout: inputs in suites/, outputs in results/.
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SUITES_DIR = os.path.join(THIS_DIR, 'suites')
RESULTS_DIR = os.path.join(THIS_DIR, 'results')


def find_stockfish():
    for var in ("STOCKFISH_PATH", "STOCKFISH"):
        p = os.environ.get(var)
        if p and os.path.exists(p):
            return p
    w = shutil.which("stockfish")
    if w:
        return w
    for c in (
        "/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/stockfish/stockfish-windows-x86-64-avx2.exe",
        "/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/stockfish/stockfish.exe",
        "../stockfish/stockfish-windows-x86-64-avx2.exe",
    ):
        if os.path.exists(c):
            return c
    return None


def load_epd(path):
    out = []
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
            bm = ops.get('bm')
            if not bm:
                continue
            out.append((board, {m.uci() for m in bm}, ops.get('id', ''), line))
    return out


def _resolve_suite(name):
    """Accept an absolute/cwd path, else resolve a bare name against suites/ then results/."""
    for cand in (name, os.path.join(SUITES_DIR, name), os.path.join(RESULTS_DIR, name)):
        if os.path.exists(cand):
            return cand
    return name


def main():
    arg1 = sys.argv[1] if len(sys.argv) > 1 else "tactical_fails_run.epd"
    epd = _resolve_suite(arg1)
    depth = int(sys.argv[2]) if len(sys.argv) > 2 else 10

    sf = find_stockfish()
    if not sf:
        print("Stockfish not found. Set STOCKFISH_PATH=/path/to/stockfish (or put it on PATH).")
        sys.exit(1)

    positions = load_epd(epd)
    print(f"Stockfish: {sf}")
    print(f"Suite: {epd}  ({len(positions)} positions)  @ fixed depth {depth}\n")
    print(f"{'#':>3}  {'res':<4} {'sf_move':<7} {'score(stm)':>10}  {'expected':<22} id")

    engine = chess.engine.SimpleEngine.popen_uci(sf)
    try:
        engine.configure({"Threads": 1})  # determinism
    except Exception:
        pass

    solved = 0
    solved_lines = []
    try:
        for i, (board, best, label, raw) in enumerate(positions):
            info = engine.analyse(board, chess.engine.Limit(depth=depth))
            pv = info.get("pv")
            sfmove = pv[0].uci() if pv else None
            score = info.get("score")
            sc = score.pov(board.turn).score(mate_score=100000) if score else None
            ok = sfmove in best
            solved += int(ok)
            if ok:
                solved_lines.append(raw)
            print(f"{i:>3}  {('YES' if ok else 'no'):<4} {str(sfmove):<7} {str(sc):>10}  "
                  f"{','.join(sorted(best)):<22} {label}")
    finally:
        engine.quit()

    print()
    print(f"Stockfish solved {solved}/{len(positions)} at depth {depth}.")
    if solved_lines:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        out = os.path.join(RESULTS_DIR, f"sf_solved_d{depth}.epd")
        with open(out, "w") as f:
            f.write("\n".join(solved_lines) + "\n")
        print(f"  -> {len(solved_lines)} written to {out}")
        print(f"     If '{epd}' is OUR fails, these are the genuine deficiencies "
              f"(reachable at depth {depth}) to target with line_probe (eval vs search).")


if __name__ == "__main__":
    main()
