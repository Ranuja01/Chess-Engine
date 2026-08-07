# -*- coding: utf-8 -*-
"""Dump `fen,total` for N corpus positions under whatever knobs this process was started with.

Knobs latch at engine init, so an A/B needs ONE PROCESS PER SETTING -- there is no way to flip a gate
inside a run. Dump twice and diff the files.

Purpose: measure a gated change's FIRE RATE and magnitude distribution. A change that costs 147
balanced STS points while firing on 3% of positions is a very different claim from one that fires on
40%, and the bench score alone cannot distinguish them.

  pyrun diagnostics/_eval_dump_simple.py OUT=/tmp/a.csv N=1500 [KNOB=1 ...]
"""
import os, sys, csv

for _a in sys.argv[1:]:
    if '=' in _a:
        _k, _v = _a.split('=', 1)
        os.environ[_k] = _v

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(THIS)); sys.path.insert(0, THIS)
import chess
from ChessAI import ChessAI

N = int(os.environ.get("N", "1500"))
OUT = os.environ["OUT"]
IN = os.environ.get("IN", "ks_sets/diverse_corpus_wide.csv")
if not os.path.isabs(IN):
    IN = os.path.join(THIS, IN)


def main():
    ai = ChessAI(None, None, chess.Board(), True)
    rows = list(csv.DictReader(open(IN, newline="")))[:N]
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["fen", "total"])
        for r in rows:
            try:
                b = chess.Board(r["fen"])
                if b.is_game_over(claim_draw=False):
                    continue
                w.writerow([r["fen"], ai.ev_breakdown(b).get("total", 0)])
            except Exception:
                continue
    print("wrote %s" % OUT)


if __name__ == "__main__":
    main()
