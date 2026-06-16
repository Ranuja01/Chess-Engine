# -*- coding: utf-8 -*-
"""Sample positions from recorded self-play games and label each with our per-term static eval and
Stockfish's NNUE static eval. Output feeds tune_fit.py (per-term scale calibration).

Our static eval is absolute (Black-positive) milli-pawns; SF static is White-POV centipawns. Both are
written raw — tune_fit handles unit/sign alignment. Labelling uses SF's `eval` (NNUE static, ~5ms), not a
search, so a 10k-position sample takes a couple of minutes.

Run in WSL from NN Engine/:
    python selfplay/tune_corpus.py --tag overnight_speed --n 10000 --out selfplay/tune_data/corpus.csv
"""

import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')

import sys
import csv
import glob
import json
import random
import argparse

import chess

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ENGINE_DIR = os.path.dirname(THIS_DIR)
sys.path.insert(0, ENGINE_DIR)
sys.path.insert(0, THIS_DIR)

from arbiter import Arbiter, find_stockfish

# Terms recorded per position; the tunable subset is chosen in tune_fit.py. `pieces` carries material and
# is never scaled, but it is needed to reconstruct the total.
TERMS = [
    "pieces", "capture_gains", "passed_pawn_support", "latent_threat", "central",
    "imbalance_white", "imbalance_black", "pair_bonus", "piece_value_boost",
]

RESULT_WHITE = {"1-0": 1.0, "0-1": 0.0, "1/2-1/2": 0.5}
NEAR_EQUAL_CP = 150


def load_engine():
    from ChessAI import ChessAI
    seed = chess.Board()
    return ChessAI(None, None, seed, seed.turn)


def sample_positions(tag, n, per_game, rng):
    """Collect up to `n` (fen, result_white) pairs, capped at `per_game` per game and spread across games."""
    logdir = os.path.join(THIS_DIR, "games", tag)
    jsonls = sorted(glob.glob(os.path.join(logdir, "game_*", "game.jsonl")))
    rng.shuffle(jsonls)
    out = []
    for path in jsonls:
        if len(out) >= n:
            break
        try:
            lines = open(path).read().splitlines()
        except OSError:
            continue
        result_white = None
        fens = []
        for ln in lines:
            try:
                rec = json.loads(ln)
            except ValueError:
                continue
            t = rec.get("type")
            if t == "result":
                result_white = RESULT_WHITE.get(rec.get("result"))
            elif t == "move" and not rec.get("opening") and rec.get("fen"):
                fens.append(rec["fen"])
        if result_white is None or not fens:
            continue
        pick = fens if len(fens) <= per_game else rng.sample(fens, per_game)
        for fen in pick:
            out.append((fen, result_white))
    rng.shuffle(out)
    return out[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="overnight_speed")
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--per-game", type=int, default=8)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--out", default=os.path.join(THIS_DIR, "tune_data", "corpus.csv"))
    ap.add_argument("--sf-movetime", type=float, default=0.1)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    positions = sample_positions(args.tag, args.n, args.per_game, rng)
    print("sampled %d positions from tag=%s" % (len(positions), args.tag))

    sf = find_stockfish()
    if sf is None:
        print("Stockfish not found", file=sys.stderr)
        return 2
    arb = Arbiter(sf, movetime=args.sf_movetime)
    ai = load_engine()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    cols = ["fen", "phase_score", "is_endgame", "status", "result_white",
            "our_total", "sf_static_cp"] + TERMS
    written = skipped = 0
    with open(args.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for i, (fen, result_white) in enumerate(positions):
            board = chess.Board(fen)
            if board.is_check():
                skipped += 1
                continue
            bd = ai.ev_breakdown(board)
            if bd.get("checkmate"):
                skipped += 1
                continue
            sfs = arb.evaluate_static(board)
            if sfs is None:
                skipped += 1
                continue
            if sfs >= NEAR_EQUAL_CP:
                status = "white_winning"
            elif sfs <= -NEAR_EQUAL_CP:
                status = "black_winning"
            else:
                status = "near_equal"
            row = [fen, bd["phase_score"], int(bd["is_endgame"]), status, result_white,
                   bd["total"], sfs] + [bd[t] for t in TERMS]
            w.writerow(row)
            written += 1
            if (i + 1) % 1000 == 0:
                print("  %d/%d  (written %d, skipped %d)" % (i + 1, len(positions), written, skipped))

    arb.close()
    print("wrote %d rows (skipped %d) -> %s" % (written, skipped, args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
