# -*- coding: utf-8 -*-
"""How many nodes does each engine actually spend to reach the SAME nominal depth?

Equal-depth benches implicitly assume nominal depth means the same thing across engines. It does not:
pruning schedules differ enormously, so a heavily-pruning modern engine explores a far smaller tree at
"depth 10" than an old lightly-pruning one. This probe reports median nodes at a fixed depth so an
equal-depth ladder can be read with the right caveat (or discarded for an equal-node one).

  python diagnostics/_depth_nodes_probe.py <label>=<exe> [...] [SUITE=sts300.epd] [DEPTH=10] [N=25]

An engine may carry UCI options after '@': <label>=<exe>@OwnBook=false@Knobs=FIX_MDP=1 (one binary, two knob arms).
SUITE is a file in diagnostics/suites or any path; STS-style EPD, plain FEN / EPD lines, or bench lines "FEN | moves".
PAIRED=1: every arm is compared with the FIRST one position by position (each position searched as a fresh game, so no
table carries over): the distribution of node ratios, how much of the total change the ten largest positions carry,
and the change split by whether either arm reported a mate score.
"""
import os
import sys
import statistics
import chess
import chess.engine

THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS)
from sf_ceiling_win import load_sts_epd


def load_positions(suite):
    try:
        fens = [fen for fen, _sm, _mx in load_sts_epd(suite)]
        if fens:
            return fens
    except Exception:
        pass
    fens = []
    for raw in open(suite):
        line = raw.split('|')[0].strip()
        if not line or line.startswith('#'):
            continue
        try:
            fens.append(chess.Board(line).fen())
        except ValueError:
            try:
                fens.append(chess.Board.from_epd(line)[0].fen())
            except ValueError:
                pass
    return fens


def run_engine(path, options, positions, depth, fresh):
    eng = chess.engine.SimpleEngine.popen_uci(path)
    if options:
        eng.configure(options)
    rows = []
    for fen in positions:
        try:
            info = eng.analyse(chess.Board(fen), chess.engine.Limit(depth=depth), game=object() if fresh else None)
            score = info.get("score")
            rows.append((info.get("nodes"), info.get("depth"), score is not None and score.is_mate()))
        except Exception:
            rows.append((None, None, False))
    try:
        eng.quit()
    except Exception:
        eng.close()
    return rows


def paired_report(label, base, arm):
    pairs = [(b[0], a[0], b[2] or a[2]) for b, a in zip(base, arm) if b[0] and a[0]]
    if not pairs:
        print("  %-10s  (no paired node counts)" % label)
        return
    ratios = sorted(a / b for b, a, _m in pairs)
    tb, ta = sum(b for b, _a, _m in pairs), sum(a for _b, a, _m in pairs)
    deltas = sorted((a - b for b, a, _m in pairs), key=abs, reverse=True)
    top10 = sum(deltas[:10])
    mate = [(b, a) for b, a, m in pairs if m]
    quiet = [(b, a) for b, a, m in pairs if not m]
    pct = lambda s: 100.0 * s
    print("  %-10s  total %+.1f%% over %d positions · median ratio %.3f (IQR %.3f-%.3f) · ratio > 1.10 on %.0f%%, < 0.90 on %.0f%%"
          % (label, pct(ta / tb - 1), len(pairs), statistics.median(ratios), ratios[len(ratios) // 4], ratios[3 * len(ratios) // 4],
             pct(sum(r > 1.10 for r in ratios) / len(ratios)), pct(sum(r < 0.90 for r in ratios) / len(ratios))))
    print("  %-10s  the 10 largest positions carry %+.0f%% of the total change (%+d of %+d nodes)"
          % ('', pct(top10 / (ta - tb)) if ta != tb else 0.0, top10, ta - tb))
    for name, grp in (("mate score seen", mate), ("no mate score", quiet)):
        if grp:
            gb, ga = sum(b for b, _a in grp), sum(a for _b, a in grp)
            print("  %-10s  %-16s %3d positions: %+.1f%% (%+d nodes)" % ('', name, len(grp), pct(ga / gb - 1), ga - gb))


def main():
    engines, settings = [], {}
    for arg in sys.argv[1:]:
        if '=' not in arg:
            continue
        key, val = arg.split('=', 1)
        if key in ('SUITE', 'DEPTH', 'N', 'PAIRED'):
            settings[key] = val
        else:
            path, *opts = val.split('@')
            engines.append((key, path, dict(o.split('=', 1) for o in opts)))
    depth = int(settings.get('DEPTH', '10'))
    n = int(settings.get('N', '25'))
    paired = settings.get('PAIRED', '0') == '1'
    suite = settings.get('SUITE', 'sts300.epd')
    suite = suite if os.path.exists(suite) else os.path.join(THIS, 'suites', suite)
    positions = load_positions(suite)[:n]
    print("nodes to reach depth %d, %d positions (%s)\n" % (depth, len(positions), os.path.basename(suite)))

    base = None
    for label, path, options in engines:
        if not os.path.exists(path):
            print("  %-6s  (missing)" % label)
            continue
        try:
            rows = run_engine(path, options, positions, depth, paired)
        except Exception as exc:
            print("  %-6s  (unavailable: %s)" % (label, exc))
            continue
        nodes = [r[0] for r in rows if r[0]]
        depths = [r[1] for r in rows if r[1]]
        if nodes:
            print("  %-6s  median %9d nodes   mean %9d   reported depth %s"
                  % (label, statistics.median(nodes), statistics.mean(nodes),
                     statistics.median(depths) if depths else "?"))
        else:
            print("  %-6s  (no node info returned)" % label)
        if paired:
            if base is None:
                base = rows
            else:
                paired_report(label, base, rows)
        sys.stdout.flush()


if __name__ == '__main__':
    main()
