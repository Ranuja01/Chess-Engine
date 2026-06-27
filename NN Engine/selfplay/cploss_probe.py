# -*- coding: utf-8 -*-
"""ACPL proxy: average centipawn-loss of OUR move vs Stockfish over GENERAL positions (move quality).

The overall-strength inner loop (distinct from the floor/collapse probes): how good is our AVERAGE move at
a fixed time/depth budget? For a broad sample of general midgame positions, our engine picks a move; SF
scores the position (its best line) and the position AFTER our move; cp_loss = how much the mover gave up.
A multi-variable eval candidate that LOWERS mean cp_loss is playing better in general -- the thing a better
eval should do, separate from the depth-bound collapses. More sensitive than move-match (continuous, every
move) and than collapse-rate (which only sees the rare tail).

Seeds are sampled deterministically from the annotated games (general midgame moves, not just flips), so
candidates are directly comparable. Knobs read once at engine init -> set in the PROCESS env (the dispatcher
forwards trailing KEY=VAL). Run: overnight_runner.sh cploss_probe <tag> <n> [KEY=VAL ...]
"""
import os, sys, glob, json, statistics, chess

THIS = os.path.dirname(os.path.abspath(__file__))
ENGINE = os.path.dirname(THIS)
sys.path.insert(0, THIS)
sys.path.insert(0, os.path.join(ENGINE, 'diagnostics'))
from arbiter import Arbiter, find_stockfish
from tactical_test import run_one


def pcount(fen):
    return sum(1 for ch in fen.split()[0] if ch.isalpha())


def sample_seeds(tag, n):
    """Deterministic broad sample of general midgame FENs (every K-th qualifying move across games)."""
    G = os.path.join(THIS, 'games', tag)
    fens = []
    for d in sorted(glob.glob(os.path.join(G, 'game_*'))):
        jf = os.path.join(d, 'game.annotated.jsonl')
        if not os.path.exists(jf):
            continue
        for ln in open(jf):
            try:
                o = json.loads(ln)
            except Exception:
                continue
            if o.get('type') == 'move' and o.get('fen') and not o.get('opening'):
                f = o['fen']
                if 14 <= pcount(f) <= 30:                    # general midgame (skip near-endgame + opening)
                    fens.append(f)
    if len(fens) > n:
        step = len(fens) / n
        fens = [fens[int(k * step)] for k in range(n)]
    return fens


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else 'placement_bundle'
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 150
    mt = float(os.environ.get('SF_MOVETIME', '0.25'))
    fens = sample_seeds(tag, n)
    arb = Arbiter(find_stockfish(), movetime=mt)
    losses = []
    for fen in fens:
        try:
            b = chess.Board(fen)
            cp_best, sf_best, _ = arb.evaluate(b)             # White-POV eval of the position (best play)
            our = run_one(fen, set())['uci']
            b.push_uci(our)
            cp_after, _, _ = arb.evaluate(b)                  # White-POV eval after our move
        except Exception:
            continue
        white_to_move = fen.split()[1] == 'w'
        loss = (cp_best - cp_after) if white_to_move else (cp_after - cp_best)
        losses.append(max(0, loss))                          # mover's cp given up (>=0)
    if hasattr(arb, 'close'):
        arb.close()
    if losses:
        losses.sort()
        print(f"cploss: n={len(losses)}  MEAN={statistics.mean(losses):.1f}cp  "
              f"MEDIAN={statistics.median(losses):.1f}cp  "
              f"blunders(>200)={sum(1 for x in losses if x > 200)}  "
              f"MAX_DEPTH={os.environ.get('MAX_DEPTH','?')} PRESET={os.environ.get('PRESET','?')}")
    else:
        print("cploss: no positions scored")


if __name__ == "__main__":
    main()
