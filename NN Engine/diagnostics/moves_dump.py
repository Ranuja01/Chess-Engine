# -*- coding: utf-8 -*-
"""Move-flip diagnostic — dump OUR chosen move (+ its WDL-cploss) per corpus position, for ONE env-latched config.

Companion to cploss_frozen.py: same frozen stratified corpus, same SF18 judge, same cached best-eval — but instead
of aggregating the mean loss it writes a per-position row `{fen, stratum, uci, loss}`. Run it twice (default env vs
the Stage-1 winner env) and diff with move_flip_report.py to decompose the compass delta into WHERE our move actually
changed and by how much win% those changes cost/gain. Deterministic: fixed engine depth + fixed SF judge depth +
env-latched knobs + book off. byte-id-irrelevant (python only).

    [KNOBS...] python diagnostics/moves_dump.py --tag <name> [corpus.csv] [--depth D] [--limit N] [--shard S]
"""
import argparse
import csv
import json
import math
import os
import sys

import chess

THIS = os.path.dirname(os.path.abspath(__file__))
ENGINE = os.path.dirname(THIS)
sys.path.insert(0, THIS)
sys.path.insert(0, os.path.join(ENGINE, "selfplay"))
from arbiter import Arbiter, find_stockfish   # noqa: E402
from tactical_test import run_one              # noqa: E402

WIN_K = 0.00368208                              # Lichess cp->win% sigmoid constant (per cp); matches cploss_frozen.py


def winpct(cp):
    cp = max(-1500, min(1500, cp))
    return 100.0 / (1.0 + math.exp(-WIN_K * cp))


def main():
    # The pyrun dispatcher does not forward KEY=VAL as env; parse + set them here BEFORE any ChessAI
    # construction (engine config is read once at init), and strip them so argparse only sees its flags.
    for _kv in [a for a in sys.argv[1:] if "=" in a and not a.startswith("-") and not a.endswith(".csv")]:
        _k, _v = _kv.split("=", 1); os.environ[_k] = _v
    sys.argv = [sys.argv[0]] + [a for a in sys.argv[1:]
                               if not ("=" in a and not a.startswith("-") and not a.endswith(".csv"))]
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)                # output -> tune_data/moves_<tag>.csv
    ap.add_argument("corpus", nargs="?", default=os.path.join(ENGINE, "selfplay", "tune_data", "cploss_corpus.csv"))
    ap.add_argument("--depth", type=int, default=12)       # SF18 judge depth (must match the cploss run to reuse cache)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard", choices=["all", "train", "holdout"], default="all")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.corpus)))
    if args.shard == "train":
        rows = rows[0::2]
    elif args.shard == "holdout":
        rows = rows[1::2]
    if args.limit:
        rows = rows[: args.limit]

    # Shared position-only SF-best cache (config-independent) keyed by fen@depth — warms across both configs.
    cache_path = args.corpus.replace(".csv", ".sfcache_d%d.json" % args.depth)
    cache = {}
    if os.path.exists(cache_path):
        try:
            cache = json.load(open(cache_path))
        except Exception:
            cache = {}

    arb = Arbiter(find_stockfish(), depth=args.depth)
    stats = {"dirty": False}

    def sf_eval(board):
        key = " ".join(board.fen().split()[:4])
        if key in cache:
            return cache[key]
        cp, _, _ = arb.evaluate(board)
        if cp is not None:
            cache[key] = cp; stats["dirty"] = True
        return cp

    out_path = os.path.join(ENGINE, "selfplay", "tune_data", "moves_%s.csv" % args.tag)
    w = csv.writer(open(out_path, "w", newline=""))
    w.writerow(["fen", "stratum", "uci", "loss"])

    n = 0
    for r in rows:
        fen = r["fen"]; st = r.get("stratum", "?")
        try:
            b = chess.Board(fen)
            cp_best = sf_eval(b)
            if cp_best is None:
                continue
            our = run_one(fen, set())["uci"]                # our engine's move (env-latched knobs)
            if our is None:
                continue
            b.push_uci(our)
            cp_after = sf_eval(b)
            if cp_after is None:
                continue
        except Exception:
            continue
        white_to_move = fen.split()[1] == "w"
        if white_to_move:
            loss = winpct(cp_best) - winpct(cp_after)
        else:
            loss = winpct(cp_after) - winpct(cp_best)
        w.writerow([fen, st, our, "%.4f" % max(0.0, loss)])
        n += 1

    if hasattr(arb, "close"):
        arb.close()
    if stats["dirty"]:
        json.dump(cache, open(cache_path, "w"))
    print("moves_dump tag=%s  wrote %d rows -> %s" % (args.tag, n, out_path))


if __name__ == "__main__":
    main()
