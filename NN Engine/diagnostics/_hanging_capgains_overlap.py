# -*- coding: utf-8 -*-
"""Does the THREAT_HANGING term price the same thing as capture_gains (and as qsearch)?

Motivation: `search_engine.h`'s own comment on THREATS_STANDING_ONLY says the hanging bonus is "volatile +
double-counts capture_gains' en-prise sim". Last night's fit agreed indirectly -- STANDING_ONLY (hanging
OFF) beat full threats at EVERY scale. This measures the overlap directly.

Method: threats do NOT feed capture_gains, so capgains cannot "move in compensation" -- the wrong test.
The right test is CO-OCCURRENCE: isolate the hanging component as
    hang = threats(STANDING_ONLY=0) - threats(STANDING_ONLY=1)
and ask whether it fires on the same positions capture_gains is already pricing. High co-occurrence +
correlated magnitude = double count; independent firing = two different concepts.

Knobs latch at extension init, so each config needs its own process: the parent re-invokes itself.

  pyrun diagnostics/_hanging_capgains_overlap.py [TAG=vssf_2400] [LIMIT=250]
"""
import os, sys, csv, subprocess

for _a in sys.argv[1:]:
    if '=' in _a:
        _k, _v = _a.split('=', 1)
        os.environ[_k] = _v

THIS = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(THIS, "ks_sets", "collapse_dataset_classified.csv")
TAG = os.environ.get("TAG", "vssf_2400")
LIMIT = int(os.environ.get("LIMIT", "250"))


def load_fens():
    out = []
    with open(DATA, newline='') as fh:
        for r in csv.DictReader(fh):
            if r.get("family") != TAG:
                continue
            f = r.get("drop_fen")
            if f:
                out.append((f, r.get("ks_class", "?")))
            if len(out) >= LIMIT:
                break
    return out


def child():
    """Print `idx,threats,capture_gains` per FEN under the config already latched via env."""
    sys.path.insert(0, os.path.dirname(THIS)); sys.path.insert(0, THIS)
    import chess
    from ChessAI import ChessAI
    ai = ChessAI(None, None, chess.Board(), True)
    for i, (fen, _c) in enumerate(load_fens()):
        try:
            bd = ai.ev_breakdown(chess.Board(fen))
        except Exception:
            continue
        print("%d,%d,%d" % (i, bd.get("threats", 0), bd.get("capture_gains", 0)))


def run(standing):
    env = dict(os.environ, ENABLE_THREATS="1", THREATS_STANDING_ONLY=str(standing), CHILD="1")
    args = [sys.executable, os.path.abspath(__file__), "CHILD=1",
            "ENABLE_THREATS=1", "THREATS_STANDING_ONLY=%d" % standing,
            "TAG=" + TAG, "LIMIT=%d" % LIMIT]
    out = subprocess.run(args, capture_output=True, text=True, env=env).stdout
    d = {}
    for line in out.splitlines():
        p = line.strip().split(",")
        if len(p) == 3 and p[0].isdigit():
            d[int(p[0])] = (int(p[1]), int(p[2]))
    return d


def corr(xs, ys):
    n = len(xs)
    if n < 2:
        return float('nan')
    mx, my = sum(xs) / n, sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    return sxy / (sxx * syy) ** 0.5 if sxx > 0 and syy > 0 else float('nan')


def main():
    if os.environ.get("CHILD"):
        child()
        return
    fens = load_fens()
    print("positions=%d  family=%s" % (len(fens), TAG))
    a = run(0)   # hanging ON
    b = run(1)   # hanging OFF
    keys = sorted(set(a) & set(b))
    if not keys:
        sys.exit("no positions scored (both child runs empty)")

    hang, capg = [], []
    both = only_hang = only_capg = neither = 0
    for k in keys:
        h = a[k][0] - b[k][0]
        c = a[k][1]
        hang.append(abs(h)); capg.append(abs(c))
        if h and c:
            both += 1
        elif h:
            only_hang += 1
        elif c:
            only_capg += 1
        else:
            neither += 1

    fires = both + only_hang
    print("\nHANGING COMPONENT  = threats(STANDING_ONLY=0) - threats(STANDING_ONLY=1)")
    print("  fires on            %d / %d positions (%.0f%%)" % (fires, len(keys), 100.0 * fires / len(keys)))
    print("  mean |hang|         %.0f millipawns (over positions where it fires)"
          % (sum(hang) / fires if fires else 0))
    print("  mean |capture_gains| %.0f millipawns" % (sum(capg) / len(keys)))
    print("\nCO-OCCURRENCE")
    print("  both non-zero       %d   (%.0f%% of all)" % (both, 100.0 * both / len(keys)))
    print("  hanging only        %d" % only_hang)
    print("  capture_gains only  %d" % only_capg)
    print("  neither             %d" % neither)
    print("\n  corr(|hang|, |capgains|) = %.3f" % corr(hang, capg))
    print("\nReading it: high co-occurrence AND positive correlation => the two terms are pricing the same")
    print("en-prise facts and the hanging bonus is redundant (capture_gains resolves it properly, and")
    print("qsearch resolves it again). Independent firing => distinct concepts and the overlap story is wrong.")


if __name__ == '__main__':
    main()
