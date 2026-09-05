# -*- coding: utf-8 -*-
"""Score a REFERENCE engine (SF11 / SF15.1 / SF18) on the SAME STS suite we score ourselves on.

Why this exists: the project quotes "STS 54.9% vs SF18 79.3% at equal depth" as its absolute anchor, but
there was no harness to reproduce or extend it -- so the gap could never be re-measured after a change, and
SF11's number (the hand-written-eval ceiling, the more honest target for an HCE) was never measured at all.

Uses sts_test.load_sts_epd for parsing and scoring, so the score is computed EXACTLY as ours is: the engine's
UCI move is looked up in the position's c9/c8 move->points map, and the total is points/max over the suite.
The only difference is who produces the move.

⚠️ FIXED DEPTH IS NOT AN EQUAL-WORK COMPARISON. SF's depth 10 tree is far smaller and better ordered than
ours; its d10 is a deeper effective search. Read the result as "how much positional judgement does each
engine have at nominal depth N", never as "we are X% of SF at equal cost". For an equal-COST reading, use
--nodes instead, which is the comparison the record's `absolute-anchor-sf18-600-nodes` memory is about.

Run (from NN Engine/):
    pyrun diagnostics/_sts_reference.py --engine sf18 --depth 10
    pyrun diagnostics/_sts_reference.py --engine sf11 --depth 10
    pyrun diagnostics/_sts_reference.py --engine sf18 --nodes 250000     # equal-WORK vs our ~249k median
"""

import os
import sys
import csv
import argparse

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)
RESULTS_DIR = os.path.join(THIS_DIR, 'results')
SUITES_DIR = os.path.join(THIS_DIR, 'suites')

from sts_test import load_sts_epd
import chess
import chess.engine

# SF11 is the last fully-classical Stockfish (the hand-written-eval ceiling); SF18 is the truth column.
# Paths follow the existing convention: SF18 from STOCKFISH_PATH, SF11 from eval_vs_sf11.
ENGINES = {}


def resolve_engine(name):
    """Map a short name to a binary path, reusing the paths the other diagnostics already use."""
    if name == 'sf18':
        path = os.environ.get('STOCKFISH_PATH')
        if not path:
            raise SystemExit("STOCKFISH_PATH is not set (SF18).")
        return path
    if name == 'sf11':
        from eval_vs_sf11 import SF11
        return SF11
    if name == 'sf15':
        return os.path.join(os.path.dirname(os.environ.get('STOCKFISH_PATH', '')),
                            "stockfish_15_linux/stockfish_15.1_linux_x64/"
                            "stockfish-ubuntu-20.04-x86-64")
    raise SystemExit("unknown engine %r (want sf11 / sf15 / sf18)" % name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--engine', default='sf18', help='sf11 | sf15 | sf18')
    ap.add_argument('--epd', default='sts300.epd')
    ap.add_argument('--depth', type=int, default=10)
    ap.add_argument('--nodes', type=int, default=0,
                    help='if set, limit by NODES instead of depth (the equal-work comparison)')
    ap.add_argument('--threads', type=int, default=1)
    ap.add_argument('--hash', type=int, default=128)
    ap.add_argument('--classical', action='store_true',
                    help='SF15/SF18 only: disable NNUE to read the classical eval')
    ap.add_argument('--limit', type=int, default=0, help='first N positions only (smoke test)')
    args = ap.parse_args()

    path = args.epd
    if not os.path.exists(path):
        cand = os.path.join(SUITES_DIR, args.epd)
        if os.path.exists(cand):
            path = cand
    if not os.path.exists(path):
        raise SystemExit("STS EPD not found: %s" % args.epd)

    positions = load_sts_epd(path)
    if args.limit:
        positions = positions[:args.limit]
    if not positions:
        raise SystemExit("No scorable STS positions parsed from %s" % path)

    binary = resolve_engine(args.engine)
    if not os.path.exists(binary):
        raise SystemExit("engine binary not found: %s" % binary)

    eng = chess.engine.SimpleEngine.popen_uci(binary)
    try:
        opts = {}
        # Single-threaded to match our engine's regime; a fixed small hash keeps runs comparable.
        for key, val in (("Threads", args.threads), ("Hash", args.hash)):
            if key in eng.options:
                opts[key] = val
        if args.classical and "Use NNUE" in eng.options:
            opts["Use NNUE"] = False
        if opts:
            eng.configure(opts)

        limit = (chess.engine.Limit(nodes=args.nodes) if args.nodes
                 else chess.engine.Limit(depth=args.depth))
        regime = ("nodes=%d" % args.nodes) if args.nodes else ("depth=%d" % args.depth)

        print("Running %d STS positions from %s  (engine=%s %s)\n"
              % (len(positions), os.path.basename(path), args.engine, regime))

        rows = []
        total = max_total = 0
        theme_pts, theme_max = {}, {}
        for idx, (fen, score_map, mx, theme, epd_id) in enumerate(positions):
            board = chess.Board(fen)
            res = eng.play(board, limit)
            uci = res.move.uci() if res.move else ''
            pts = score_map.get(uci, 0)
            total += pts
            max_total += mx
            theme_pts[theme] = theme_pts.get(theme, 0) + pts
            theme_max[theme] = theme_max.get(theme, 0) + mx
            rows.append({"idx": idx, "id": epd_id, "theme": theme, "engine": uci,
                         "score": pts, "max": mx, "fen": fen})
            if idx % 25 == 0:
                print("  %4d  %-6s %2d/%-2d  %s" % (idx, uci, pts, mx, epd_id))

        os.makedirs(RESULTS_DIR, exist_ok=True)
        tag = "%s_%s" % (args.engine, regime.replace('=', ''))
        csv_path = os.path.join(RESULTS_DIR, "sts_reference_%s.csv" % tag)
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["idx", "id", "theme", "engine", "score", "max", "fen"])
            w.writeheader()
            w.writerows(rows)

        pct = (100.0 * total / max_total) if max_total else 0.0
        print("\nSTS score: %d/%d  (%.1f%%)   [%s @ %s]" % (total, max_total, pct, args.engine, regime))
        print("\nPer-theme:")
        for theme in sorted(theme_pts):
            tm, tp = theme_max[theme], theme_pts[theme]
            print("  %-20s %4d/%-4d (%3.0f%%)" % (theme, tp, tm, (100.0 * tp / tm) if tm else 0.0))
        print("\nresults -> %s" % csv_path)
    finally:
        eng.quit()


if __name__ == "__main__":
    main()
