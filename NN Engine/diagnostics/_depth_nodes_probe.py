# -*- coding: utf-8 -*-
"""How many nodes does each engine actually spend to reach the SAME nominal depth?

Equal-depth benches implicitly assume nominal depth means the same thing across engines. It does not:
pruning schedules differ enormously, so a heavily-pruning modern engine explores a far smaller tree at
"depth 10" than an old lightly-pruning one. This probe reports median nodes at a fixed depth so an
equal-depth ladder can be read with the right caveat (or discarded for an equal-node one).

  python diagnostics/_depth_nodes_probe.py <label>=<exe> [...] [SUITE=sts300.epd] [DEPTH=10] [N=25]
"""
import os
import sys
import statistics
import chess
import chess.engine

THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS)
from sf_ceiling_win import load_sts_epd


def main():
    engines, settings = [], {}
    for arg in sys.argv[1:]:
        if '=' not in arg:
            continue
        key, val = arg.split('=', 1)
        (settings if key in ('SUITE', 'DEPTH', 'N') else None)
        if key in ('SUITE', 'DEPTH', 'N'):
            settings[key] = val
        else:
            engines.append((key, val))
    depth = int(settings.get('DEPTH', '10'))
    n = int(settings.get('N', '25'))
    suite = os.path.join(THIS, 'suites', settings.get('SUITE', 'sts300.epd'))
    positions = load_sts_epd(suite)[:n]
    print("nodes to reach depth %d, %d positions\n" % (depth, len(positions)))

    for label, path in engines:
        if not os.path.exists(path):
            print("  %-6s  (missing)" % label)
            continue
        try:
            eng = chess.engine.SimpleEngine.popen_uci(path)
        except Exception as exc:
            print("  %-6s  (unavailable: %s)" % (label, exc))
            continue
        nodes, depths = [], []
        for fen, _sm, _mx in positions:
            try:
                info = eng.analyse(chess.Board(fen), chess.engine.Limit(depth=depth))
                if info.get("nodes"):
                    nodes.append(info["nodes"])
                if info.get("depth"):
                    depths.append(info["depth"])
            except Exception:
                continue
        try:
            eng.quit()
        except Exception:
            eng.close()
        if nodes:
            print("  %-6s  median %9d nodes   mean %9d   reported depth %s"
                  % (label, statistics.median(nodes), statistics.mean(nodes),
                     statistics.median(depths) if depths else "?"))
        else:
            print("  %-6s  (no node info returned)" % label)


if __name__ == '__main__':
    main()
