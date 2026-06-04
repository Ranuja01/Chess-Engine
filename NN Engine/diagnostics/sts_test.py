# -*- coding: utf-8 -*-
"""Strategic Test Suite (STS) harness — the POSITIONAL sibling of tactical_test.py.

Where tactical_test.py is pass/fail on tactics (WAC), STS scores positional move-CHOICE: ~1500
puzzles across 15 themes (outposts, open files, bishop-vs-knight, pawn play, …), each carrying a
move->score map (best move = 10, strong alternatives partial credit). A real game is ~90% quiet
maneuvering that a tactics suite cannot measure, so this is the parallel gate: run it alongside
tactical_test.py and every change gets a tactical AND a positional read.

Source the suite once into suites/ (WSL):
    cd diagnostics/suites && wget https://raw.githubusercontent.com/fsmosca/STS-Rating/master/STS1-STS15_LAN_v3.epd

Run (from NN Engine/); the regime is set by the same PRESET / MAX_DEPTH env knobs the engine
reads in initialize_engine, exactly like tactical_test:
    PRESET=LIGHTNING MAX_DEPTH=64 python diagnostics/sts_test.py STS1-STS15_LAN_v3.epd sts_base

Scoring uses the EPD's parallel `c9` (moves in coordinate/UCI notation) and `c8` (scores) ops, so
the engine's move.uci() is looked up directly — no SAN parsing. Output: a total score / max as a
percentage, a per-theme breakdown, and diagnostics/results/sts_results_<tag>.csv. The engine
itself is untouched (diagnostic only).
"""

import os
import re
import csv
import sys

# --- same diagnostics layout as tactical_test: tools in diagnostics/, suites/ and results/
#     beside them, the built ChessAI .so one level up in NN Engine/. ---
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SUITES_DIR = os.path.join(THIS_DIR, 'suites')
RESULTS_DIR = os.path.join(THIS_DIR, 'results')

# Importing tactical_test loads the keras models once and exposes run_one (builds a fresh
# ChessAI per FEN, captures the engine's stdout, returns its uci move + eval/depth/time + a
# book-hit flag). Reuse it wholesale so the two harnesses run the engine identically.
from tactical_test import run_one
import chess

_C8_RE = re.compile(r'\bc8\s+"([^"]*)"')   # parallel scores, e.g. "10 2 3 2"
_C9_RE = re.compile(r'\bc9\s+"([^"]*)"')   # parallel moves in coordinate notation, e.g. "f4f5 d4e5"
_ID_RE = re.compile(r'\bid\s+"([^"]*)"')


def _theme_of(epd_id):
    """An STS id reads 'STS(v1.0) Undermine.001' -> theme 'Undermine'."""
    m = re.search(r'\)\s*(.+?)\.\d', epd_id)
    return m.group(1).strip() if m else (epd_id or 'unknown')


def load_sts_epd(path):
    """Parse an STS EPD into [(fen, {uci: score}, max_score, theme, id)].

    The {uci: score} map is zipped from the c9 (coordinate moves) and c8 (scores) operations, so
    no notation conversion is needed. Lines without a matching c8/c9 pair are skipped."""
    positions = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            c8 = _C8_RE.search(line)
            c9 = _C9_RE.search(line)
            if not (c8 and c9):
                continue
            scores = c8.group(1).split()
            moves = c9.group(1).split()
            if len(scores) != len(moves):
                continue
            try:
                score_map = {mv: int(sc) for mv, sc in zip(moves, scores)}
            except ValueError:
                continue
            # set_epd handles the 4-field EPD position; fen() gives the full FEN run_one needs.
            board = chess.Board()
            try:
                board.set_epd(line)
            except Exception:
                continue
            id_match = _ID_RE.search(line)
            epd_id = id_match.group(1) if id_match else ''
            positions.append((board.fen(), score_map, max(score_map.values()),
                              _theme_of(epd_id), epd_id))
    return positions


def main():
    epd_arg = sys.argv[1] if len(sys.argv) > 1 else 'STS1-STS15_LAN_v3.epd'
    tag = sys.argv[2] if len(sys.argv) > 2 else 'sts'

    # Resolve a bare name against suites/, or accept a direct path.
    path = epd_arg
    if not os.path.exists(path):
        cand = os.path.join(SUITES_DIR, epd_arg)
        if os.path.exists(cand):
            path = cand
    if not os.path.exists(path):
        print(f"STS EPD not found: {epd_arg}\n"
              f"Download it into suites/ with:\n"
              f"  cd diagnostics/suites && wget "
              f"https://raw.githubusercontent.com/fsmosca/STS-Rating/master/STS1-STS15_LAN_v3.epd")
        return

    positions = load_sts_epd(path)
    if not positions:
        print(f"No scorable STS positions parsed from {path} (need c8/c9 ops).")
        return

    os.makedirs(RESULTS_DIR, exist_ok=True)
    csv_path = os.path.join(RESULTS_DIR, f"sts_results_{tag}.csv")

    print(f"Running {len(positions)} STS positions from {path}  (tag={tag})\n")
    print(f"{'#':>4}  {'engine':<6} {'pts':>5}  {'theme':<16} {'eval':>7} {'d':>3} {'time':>6}  id")

    rows = []
    total = max_total = booked = 0
    theme_pts, theme_max = {}, {}
    for idx, (fen, score_map, mx, theme, epd_id) in enumerate(positions):
        r = run_one(fen, set())
        if r['booked']:
            booked += 1
            pts = None
        else:
            pts = score_map.get(r['uci'], 0)
            total += pts
            max_total += mx
            theme_pts[theme] = theme_pts.get(theme, 0) + pts
            theme_max[theme] = theme_max.get(theme, 0) + mx

        pts_s = 'BOOK' if r['booked'] else f"{pts}/{mx}"
        ev_s = '' if r['eval'] is None else str(r['eval'])
        d_s = '' if r['depth'] is None else str(r['depth'])
        print(f"{idx:>4}  {str(r['uci']):<6} {pts_s:>5}  {theme:<16} {ev_s:>7} {d_s:>3} "
              f"{r['time']:>6.2f}  {epd_id}")

        rows.append({"idx": idx, "id": epd_id, "theme": theme, "engine": r['uci'],
                     "score": ('' if pts is None else pts), "max": mx, "eval": r['eval'],
                     "depth": r['depth'], "time": round(r['time'], 3), "fen": fen})

    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["idx", "id", "theme", "engine", "score", "max",
                                          "eval", "depth", "time", "fen"])
        w.writeheader()
        w.writerows(rows)

    pct = (100.0 * total / max_total) if max_total else 0.0
    print()
    print(f"STS score: {total}/{max_total}  ({pct:.1f}%)"
          + (f"   [{booked} book hits excluded]" if booked else ""))
    print("\nPer-theme:")
    for theme in sorted(theme_pts):
        tm, tp = theme_max[theme], theme_pts[theme]
        tpct = (100.0 * tp / tm) if tm else 0.0
        print(f"  {theme:<20} {tp:>4}/{tm:<4} ({tpct:>3.0f}%)")
    print(f"\nresults -> {csv_path}")


if __name__ == "__main__":
    main()
