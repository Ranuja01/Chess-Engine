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
#
# ☠️ NATIVE-ELF vs WINDOWS-EXE IS A TRUST BOUNDARY, NOT A DETAIL. SF16 and SF17 ship here only as Windows
# .exe, and running those through WSL binfmt is recorded as flaky ("Exec format error, whole runs void",
# SESSION-HANDOFF-2026-07-08). Rows produced from an .exe are marked UNTRUSTED in the output for that
# reason -- if one is needed for the record, run it under Windows python via sf_ceiling_win.py instead.
SF_ROOT = os.environ.get(
    'SF_ROOT',
    os.path.dirname(os.path.dirname(os.path.dirname(THIS_DIR))))   # .../Programming/Chess Engine

# label -> (path relative to SF_ROOT, native ELF?, extra UCI options)
ENGINES = {
    'sf1':    ("stockfish_1/stockfish-1.1_ja/stockfish_11_x64_ja.exe",                    False, {}),
    'sf11':   ("stockfish_11_linux/stockfish-11-linux/Linux/stockfish_20011801_x64_bmi2",  True,  {}),
    'sf15c':  ("stockfish_15_linux/stockfish_15.1_linux_x64/stockfish-ubuntu-20.04-x86-64", True, {"Use NNUE": False}),
    'sf15n':  ("stockfish_15_linux/stockfish_15.1_linux_x64/stockfish-ubuntu-20.04-x86-64", True, {"Use NNUE": True}),
    'sf16':   ("stockfish_16/stockfish-windows-x86-64-avx2.exe",                           False, {}),
    'sf17':   ("stockfish_17/stockfish-windows-x86-64-avx2.exe",                           False, {}),
    'sf18':   ("stockfish_18_linux/stockfish-ubuntu-x86-64-avx2",                          True,  {}),
    'sf19':   ("stockfish_19_linux/stockfish-linux-x86-64-universal",                      True,  {}),
}


def resolve_engine(name):
    """Map a short label to (binary path, native?, extra UCI options).

    ⚠️ `sf15` without a suffix is REJECTED rather than guessed: SF15.1's classical and NNUE evals are
    different evaluators and the record already carries numbers for both, so a silent default would
    make a row uninterpretable after the fact.
    """
    if name == 'sf15':
        raise SystemExit("say sf15c (classical) or sf15n (NNUE) -- they are different evaluators")
    if name not in ENGINES:
        raise SystemExit("unknown engine %r (want one of: %s)" % (name, ", ".join(sorted(ENGINES))))
    rel, native, opts = ENGINES[name]
    return os.path.join(SF_ROOT, rel), native, dict(opts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--engine', default='sf18', help='sf1 | sf11 | sf15c | sf15n | sf16 | sf17 | sf18 | sf19')
    ap.add_argument('--binary', default='',
                    help='override the resolved path (e.g. a self-compiled build). --engine still names the row.')
    ap.add_argument('--uci', default='',
                    help="extra UCI options as 'Name=Value;Name=Value'. ★ SF1.1 exposes per-subsystem eval "
                         "weights (Passed Pawns (Middle Game), Mobility (Endgame), King Safety Coefficient, "
                         "...) as 0-200 spins defaulting to 100, so this is an ABLATION handle on a "
                         "hand-written reference eval with no recompile.")
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

    # ★ WAC and STS are scored by DIFFERENT rules and both are needed for the reference ladder: STS is a
    # weighted move->points map (c9/c8), WAC is a plain bm hit. Both loaders come from the tools that score
    # OUR side, so the only difference between an SF row and ours stays "who produced the move".
    is_wac = os.path.basename(path).lower().startswith('wac')
    if is_wac:
        from tactical_test import load_epd
        # Normalise into the STS tuple shape: one point per position, single pseudo-theme.
        positions = [(fen, {u: 1 for u in best}, 1, 'wac', pid)
                     for (fen, best, pid, _raw) in load_epd(path)]
    else:
        positions = load_sts_epd(path)
    if args.limit:
        positions = positions[:args.limit]
    if not positions:
        raise SystemExit("No scorable positions parsed from %s" % path)

    binary, native, engine_opts = resolve_engine(args.engine)
    if args.binary:
        # A self-compiled build is NATIVE by construction, which is the point of allowing the override:
        # it turns an untrusted .exe row into a trustworthy one. ⚠️ It is NOT bit-identical to the shipped
        # binary (different compiler and flags), so record it as a rebuild, not as "the" version.
        binary, native = args.binary, True
    for kv in (o for o in args.uci.split(';') if o.strip()):
        k, _, v = kv.partition('=')
        engine_opts[k.strip()] = v.strip()
    if not os.path.exists(binary):
        raise SystemExit("engine binary not found: %s" % binary)
    if not native:
        print("UNTRUSTED ROW: %s is a Windows .exe run through WSL binfmt, which is recorded as flaky.\n"
              "  Treat as indicative only; for the record run it under Windows python (sf_ceiling_win.py).\n"
              % args.engine)

    eng = chess.engine.SimpleEngine.popen_uci(binary)
    try:
        opts = {}
        # Single-threaded to match our engine's regime; a fixed small hash keeps runs comparable.
        for key, val in (("Threads", args.threads), ("Hash", args.hash)):
            if key in eng.options:
                opts[key] = val
        if args.classical and "Use NNUE" in eng.options:
            opts["Use NNUE"] = False
        # Per-label options (sf15c/sf15n pin Use NNUE explicitly, so the row can never be ambiguous
        # about which of SF15.1's two evaluators produced it).
        for key, val in engine_opts.items():
            if key in eng.options:
                opts[key] = val
        if opts:
            eng.configure(opts)

        limit = (chess.engine.Limit(nodes=args.nodes) if args.nodes
                 else chess.engine.Limit(depth=args.depth))
        regime = ("nodes=%d" % args.nodes) if args.nodes else ("depth=%d" % args.depth)

        print("Running %d STS positions from %s  (engine=%s %s)\n"
              % (len(positions), os.path.basename(path), args.engine, regime))

        rows = []
        nodes_sum = 0
        total = max_total = 0
        theme_pts, theme_max = {}, {}
        for idx, (fen, score_map, mx, theme, epd_id) in enumerate(positions):
            board = chess.Board(fen)
            # ★★ NODES ARE NOT OPTIONAL IN THIS TABLE. At a fixed nominal DEPTH the engines do wildly
            # different amounts of work (SF11 reaches d10 in ~26k nodes where we need ~249k), so a depth
            # column without nodes beside it flatters whoever prunes least and is close to meaningless.
            res = eng.play(board, limit, info=chess.engine.INFO_ALL)
            nodes_here = (res.info or {}).get('nodes', 0)
            nodes_sum += nodes_here
            uci = res.move.uci() if res.move else ''
            pts = score_map.get(uci, 0)
            total += pts
            max_total += mx
            theme_pts[theme] = theme_pts.get(theme, 0) + pts
            theme_max[theme] = theme_max.get(theme, 0) + mx
            rows.append({"idx": idx, "id": epd_id, "theme": theme, "engine": uci,
                         "score": pts, "max": mx, "nodes": nodes_here, "fen": fen})
            if idx % 25 == 0:
                print("  %4d  %-6s %2d/%-2d  %s" % (idx, uci, pts, mx, epd_id))

        os.makedirs(RESULTS_DIR, exist_ok=True)
        # Suite goes in the tag: without it a wac run silently overwrites the sts run for the same engine.
        tag = "%s_%s_%s" % ("wac" if is_wac else "sts", args.engine, regime.replace('=', ''))
        csv_path = os.path.join(RESULTS_DIR, "sts_reference_%s.csv" % tag)
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["idx", "id", "theme", "engine", "score", "max", "nodes", "fen"])
            w.writeheader()
            w.writerows(rows)

        pct = (100.0 * total / max_total) if max_total else 0.0
        label = "WAC" if is_wac else "STS"
        print("\n%s score: %d/%d  (%.1f%%)   [%s @ %s, hash=%d, threads=%d, native=%s]"
              % (label, total, max_total, pct, args.engine, regime, args.hash, args.threads, native))
        print("nodes/position: %.0f  (total %d over %d positions)"
              % (nodes_sum / float(len(positions)), nodes_sum, len(positions)))
        print("\nPer-theme:")
        for theme in sorted(theme_pts):
            tm, tp = theme_max[theme], theme_pts[theme]
            print("  %-20s %4d/%-4d (%3.0f%%)" % (theme, tp, tm, (100.0 * tp / tm) if tm else 0.0))
        print("\nresults -> %s" % csv_path)
    finally:
        eng.quit()


if __name__ == "__main__":
    main()
