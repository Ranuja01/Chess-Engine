# -*- coding: utf-8 -*-
"""Side-by-side: OUR engine (cold) vs Stockfish on a list of FENs — the horizon-vs-eval characterizer.

For each FEN: run our engine cold (reuses tactical_test.run_one — honors PRESET/MAX_DEPTH env) and Stockfish
(arbiter.evaluate: search eval + best move), and print our eval/move/depth beside SF cp/best/depth, flagging
whether our move == SF's best. Run our engine at a shallow then a deep MAX_DEPTH to tell HORIZON (we find SF's
move once deep) from EVAL-bound (we never do).

Run (from NN Engine/):
    PRESET=STANDARD MAX_DEPTH=16 python selfplay/fen_vs_sf.py "<fen1>" "<fen2>" ...
    PRESET=STANDARD MAX_DEPTH=16 python selfplay/fen_vs_sf.py --csv selfplay/games/overnight_speed/flips.csv 30
(--csv <path> <N>: pull ~N evenly-spaced FENs from the fen_start column, spanning the drop range.)
"""

import os, sys, csv, chess

THIS = os.path.dirname(os.path.abspath(__file__))
ENGINE = os.path.dirname(THIS)
sys.path.insert(0, THIS)                                  # arbiter
sys.path.insert(0, os.path.join(ENGINE, 'diagnostics'))  # tactical_test.run_one
from arbiter import Arbiter, find_stockfish
from tactical_test import run_one


def load_fens():
    args = sys.argv[1:]
    if args and args[0] == '--csv':
        path = args[1]
        n = int(args[2]) if len(args) > 2 else 30
        rows = list(csv.DictReader(open(path)))
        if len(rows) > n:
            step = len(rows) / n
            rows = [rows[int(k * step)] for k in range(n)]
        return [(r['fen_start'], f"{r.get('game','')} drop{r.get('drop','')}") for r in rows]
    return [(f, '') for f in args]


def main():
    fens = load_fens()
    mt = float(os.environ.get('SF_MOVETIME', '0.3'))
    arb = Arbiter(find_stockfish(), movetime=mt)
    agree = 0
    n = 0
    print(f"FENs={len(fens)}  our PRESET={os.environ.get('PRESET','?')} MAX_DEPTH={os.environ.get('MAX_DEPTH','?')}  SF movetime={mt}s\n")
    for fen, tag in fens:
        try:
            r = run_one(fen, set())
            b = chess.Board(fen)
            sf_cp, sf_best, sf_depth = arb.evaluate(b)
        except Exception as e:
            print(f"  ERR {fen}: {e}")
            continue
        n += 1
        match = (r['uci'] == sf_best)
        agree += int(match)
        flag = "  OK" if match else "  <-- our move != SF"
        print(f"ours: ev={str(r['eval']):>8} mv={str(r['uci']):<6} d={r['depth']:<3} | SF: cp={str(sf_cp):>6} best={str(sf_best):<6} d={sf_depth:<3}{flag}  {tag}")
        print(f"   {fen}")
    arb.close() if hasattr(arb, 'close') else None
    if n:
        print(f"\nour-move == SF-best: {agree}/{n} ({100*agree/n:.0f}%)")


if __name__ == "__main__":
    main()
