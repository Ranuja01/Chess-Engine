# -*- coding: utf-8 -*-
"""Windows-side STS ceiling scorer for engines that only ship as Windows .exe (SF1.1, SF17).

sf_bench_ceiling.py runs under WSL and drives native-ELF binaries. Reaching a Windows .exe from WSL
goes through binfmt, which is flaky here, so those engines are scored from a Windows Python instead.
Scoring is byte-for-byte the same scheme sts_test.load_sts_epd uses -- the c8 (scores) / c9 (coordinate
moves) operations zipped into a {uci: score} map -- so totals are directly comparable to both
`overnight_runner.sh sts` and sf_bench_ceiling.py. The parsing is duplicated rather than imported
because sts_test pulls in tactical_test, which loads the C++ extension and cannot import on Windows.

  python diagnostics/sf_ceiling_win.py <label>=<path-to-exe> [<label>=<path> ...] [SUITE=sts300.epd] [DEPTH=10]
"""
import os
import re
import sys
import chess
import chess.engine

_C8_RE = re.compile(r'\bc8\s+"([^"]*)"')   # parallel scores, e.g. "10 2 3 2"
_C9_RE = re.compile(r'\bc9\s+"([^"]*)"')   # parallel moves in coordinate notation, e.g. "f4f5 d4e5"

THIS = os.path.dirname(os.path.abspath(__file__))


def load_sts_epd(path):
    """Parse an STS EPD into [(fen, {uci: score}, max_score)] -- mirrors sts_test.load_sts_epd."""
    positions = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            c8, c9 = _C8_RE.search(line), _C9_RE.search(line)
            if not (c8 and c9):
                continue
            scores, moves = c8.group(1).split(), c9.group(1).split()
            if len(scores) != len(moves):
                continue
            try:
                score_map = {mv: int(sc) for mv, sc in zip(moves, scores)}
            except ValueError:
                continue
            board = chess.Board()
            try:
                board.set_epd(line)
            except Exception:
                continue
            positions.append((board.fen(), score_map, max(score_map.values())))
    return positions


def score_engine(path, positions, depth):
    """Total STS score for one engine, skipping positions it fails to answer."""
    eng = chess.engine.SimpleEngine.popen_uci(path)
    total = 0
    try:
        for fen, score_map, _mx in positions:
            try:
                res = eng.play(chess.Board(fen), chess.engine.Limit(depth=depth))
                total += score_map.get(res.move.uci(), 0) if res.move else 0
            except Exception:
                continue
    finally:
        try:
            eng.quit()
        except Exception:
            eng.close()
    return total


def main():
    engines, settings = [], {}
    for arg in sys.argv[1:]:
        if '=' not in arg:
            continue
        key, val = arg.split('=', 1)
        if key in ('SUITE', 'DEPTH'):
            settings[key] = val
        else:
            engines.append((key, val))
    if not engines:
        print(__doc__)
        return

    depth = int(settings.get('DEPTH', '10'))
    suite = os.path.join(THIS, 'suites', settings.get('SUITE', 'sts300.epd'))
    positions = load_sts_epd(suite)
    maxtotal = sum(p[2] for p in positions)
    print("SF ceiling (Windows) - STS %d positions, max %d, depth %d\n"
          % (len(positions), maxtotal, depth))

    for label, path in engines:
        if not os.path.exists(path):
            print("  %-6s  (missing: %s)" % (label, path))
            continue
        try:
            total = score_engine(path, positions, depth)
        except Exception as exc:
            print("  %-6s  (unavailable: %s)" % (label, exc))
            continue
        print("  %-6s  %5d / %d   (%.1f%%)" % (label, total, maxtotal, 100.0 * total / maxtotal))


if __name__ == '__main__':
    main()
