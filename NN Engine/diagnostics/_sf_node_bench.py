# -*- coding: utf-8 -*-
"""Nodes-and-depth for a REFERENCE engine on the SAME quiet-corpus positions depth_nps_bench uses.

WHY THIS EXISTS. The project quotes "our real EBF is ~1.73" (35.1M nodes @d10 -> 104.9M @d12) and uses it
to argue that our branching factor is not the outlier -- but SF's EBF was NEVER measured the same way, so
that comparison has no second term. Likewise we know SF11 scores 79.1% on STS at our 249,014-node budget
against our 59.9%, but not WHY: it could be reaching far greater depth with those nodes (a branching-factor
/ margin story) or understanding more per node (an eval story). Those imply different lanes.

This answers both by running the reference engine on the IDENTICAL 60 FENs (same corpus, same stratum
filter, same seed 1234) and reporting median nodes + median depth in three regimes:
  fixed depth 10  -> nodes directly comparable to our 249,014 median
  fixed depth 12  -> the d10->d12 slope, directly comparable to our 1.73
  fixed 249,014 nodes -> the depth SF reaches on OUR budget, vs our depth 10

Run (from NN Engine/):
    pyrun diagnostics/_sf_node_bench.py --engine sf11 --n 60
    pyrun diagnostics/_sf_node_bench.py --engine sf18 --n 60
"""

import os
import sys
import csv
import random
import argparse
import statistics

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ENGINE = os.path.dirname(THIS_DIR)
sys.path.insert(0, THIS_DIR)

import chess
import chess.engine

# Our own quiet-corpus baseline, for the comparison line. From the fingerprint register.
OURS_D10_NODES = 249014
OURS_EBF = 1.73


def resolve_engine(name):
    """Same path convention the other SF diagnostics use."""
    if name == 'sf18':
        path = os.environ.get('STOCKFISH_PATH')
        if not path:
            raise SystemExit("STOCKFISH_PATH is not set (SF18).")
        return path
    if name == 'sf11':
        from eval_vs_sf11 import SF11
        return SF11
    raise SystemExit("unknown engine %r (want sf11 / sf18)" % name)


def load_fens(n, seed, corpus):
    """Replicate depth_nps_bench's sampling EXACTLY -- same filter, same seed, same n, so the two
    engines are measured on identical positions. Any divergence here silently voids the comparison."""
    rows = [r for r in csv.DictReader(open(corpus))
            if r.get("stratum") in ("game", "neutral", "collapse")]
    all_fens = [r["fen"] for r in rows]
    return all_fens if n >= len(all_fens) else random.Random(seed).sample(all_fens, n)


def run_regime(eng, fens, limit, label):
    nodes, depths = [], []
    for fen in fens:
        board = chess.Board(fen)
        info = eng.analyse(board, limit)
        if isinstance(info, list):
            info = info[0]
        n = info.get("nodes")
        d = info.get("depth")
        if n is not None:
            nodes.append(n)
        if d is not None:
            depths.append(d)
    med_n = statistics.median(nodes) if nodes else float('nan')
    med_d = statistics.median(depths) if depths else float('nan')
    print("  %-22s median nodes %12s   median depth %s"
          % (label, format(int(med_n), ",") if nodes else "n/a", med_d))
    return med_n, med_d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--engine', default='sf11')
    ap.add_argument('--n', type=int, default=60)
    ap.add_argument('--seed', type=int, default=1234)
    ap.add_argument('--threads', type=int, default=1)
    ap.add_argument('--hash', type=int, default=128)
    ap.add_argument('--budget', type=int, default=OURS_D10_NODES)
    ap.add_argument('--corpus', default=os.path.join(ENGINE, "selfplay", "tune_data", "cploss_corpus.csv"))
    args = ap.parse_args()

    fens = load_fens(args.n, args.seed, args.corpus)
    binary = resolve_engine(args.engine)
    if not os.path.exists(binary):
        raise SystemExit("engine binary not found: %s" % binary)

    eng = chess.engine.SimpleEngine.popen_uci(binary)
    try:
        opts = {}
        for key, val in (("Threads", args.threads), ("Hash", args.hash)):
            if key in eng.options:
                opts[key] = val
        if opts:
            eng.configure(opts)

        print("[sf_node_bench] engine=%s  n=%d  seed=%d" % (args.engine, len(fens), args.seed))
        n10, _ = run_regime(eng, fens, chess.engine.Limit(depth=10), "fixed depth 10")
        n12, _ = run_regime(eng, fens, chess.engine.Limit(depth=12), "fixed depth 12")
        _, dbud = run_regime(eng, fens, chess.engine.Limit(nodes=args.budget),
                             "fixed %s nodes" % format(args.budget, ","))

        print("\n=== COMPARISON vs OURS ===")
        if n10 == n10:  # not NaN
            print("  nodes to depth 10:   %s (%s)  vs OURS %s   -> ratio %.2fx"
                  % (format(int(n10), ","), args.engine, format(OURS_D10_NODES, ","),
                     OURS_D10_NODES / n10 if n10 else float('nan')))
        if n10 == n10 and n12 == n12 and n10 > 0:
            ebf = (n12 / n10) ** 0.5
            print("  d10->d12 slope:      %.3f per ply (%s)  vs OURS %.2f" % (ebf, args.engine, OURS_EBF))
        print("  depth at OUR budget: %s (%s)  vs OURS 10" % (dbud, args.engine))
        print("\n  ^ If the depth-at-our-budget is MUCH deeper than 10, the equal-node STS gap is a")
        print("    BRANCHING/margin story. If it is close to 10, the gap is eval quality per node.")
    finally:
        eng.quit()


if __name__ == "__main__":
    main()
