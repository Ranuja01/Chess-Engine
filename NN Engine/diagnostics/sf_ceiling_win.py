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


def score_engine(path, positions, depth, threads=1, hashmb=128, uci_opts=None, nodes_limit=0):
    """Total score for one engine. Returns (total, nodes_per_pos, n_failed).

    ☠️ THE FAILURE COUNT IS NOT OPTIONAL. This function used to swallow every per-position exception with
    a bare `continue`, so an engine that SEGFAULTED on move one scored 0 on all 300 positions and still
    printed a plausible-looking low total -- indistinguishable from a weak engine. SF1.1 built with -O2 on
    a modern gcc does exactly that. Any caller must refuse a row with a non-trivial failure count.
    ⚠️ Threads/Hash are pinned to match the WSL-side rows (_sts_reference.py), or the numbers are not
    comparable to them -- an unstated hash difference is one of the things the reference ladder exists to
    stop happening again.
    """
    eng = chess.engine.SimpleEngine.popen_uci(path)
    total = nodes = failed = 0
    try:
        opts = {}
        for key, val in (("Threads", threads), ("Hash", hashmb)):
            if key in eng.options:
                opts[key] = val
        for key, val in (uci_opts or {}).items():
            if key in eng.options:
                opts[key] = val
        if opts:
            eng.configure(opts)
        for fen, score_map, _mx in positions:
            try:
                # ★ NODES is the EQUAL-WORK regime and it is the trustworthy one: at equal nodes the STS
                # ladder comes out monotone in engine strength, whereas at fixed DEPTH it is contaminated
                # by tree size (SF1.1 outscores SF11 at d10 purely on 15x the nodes).
                lim = (chess.engine.Limit(nodes=nodes_limit) if nodes_limit
                       else chess.engine.Limit(depth=depth))
                res = eng.play(chess.Board(fen), lim, info=chess.engine.INFO_ALL)
                total += score_map.get(res.move.uci(), 0) if res.move else 0
                nodes += (res.info or {}).get('nodes', 0)
            except Exception:
                failed += 1
                continue
    finally:
        try:
            eng.quit()
        except Exception:
            eng.close()
    return total, (nodes / float(len(positions)) if positions else 0), failed


def main():
    engines, settings = [], {}
    for arg in sys.argv[1:]:
        if '=' not in arg:
            continue
        key, val = arg.split('=', 1)
        if key == 'UCI':
            # ★ SUBSYSTEM ABLATION HANDLE. SF1.1 exposes its eval subsystem weights as UCI spins
            # (0-200, default 100): "Mobility (Middle Game)", "Pawn Structure (Endgame)", "Passed Pawns
            # (Middle Game)", "King Safety Coefficient", ... Setting a pair to 0 turns that subsystem OFF
            # in a hand-tuned reference eval whose TERM SET matches ours -- an ablation with no recompile.
            # Format: UCI="Name=Value|Name=Value" (pipe-separated; names contain spaces and parentheses).
            # ⚠️ Always run a POSITIVE CONTROL (a weight at 200) as well: if both 0 and 200 leave the score
            # unmoved, the option is not wired to what you think and every ablation number is void.
            settings.setdefault('_uci', {})
            for kv in val.split('|'):
                if '=' in kv:
                    n, _, v = kv.partition('=')
                    settings['_uci'][n.strip()] = v.strip()
            continue
        if key in ('SUITE', 'DEPTH', 'NODES'):
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
            total, npos, failed = score_engine(path, positions, depth,
                                               uci_opts=settings.get('_uci'),
                                               nodes_limit=int(settings.get('NODES', '0')))
        except Exception as exc:
            print("  %-6s  (unavailable: %s)" % (label, exc))
            continue
        # A row with failures is not a weak score, it is a broken engine -- say so rather than print a total.
        warn = ("   FAILED on %d/%d positions -- DO NOT USE THIS ROW"
                % (failed, len(positions))) if failed else ""
        print("  %-6s  %5d / %d   (%.1f%%)   %s n/pos%s"
              % (label, total, maxtotal, 100.0 * total / maxtotal,
                 ("%.0f" % npos) if npos else "?", warn))


if __name__ == '__main__':
    main()
