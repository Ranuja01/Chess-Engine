# -*- coding: utf-8 -*-
"""
Ahead-but-slipped term attribution — WHICH eval term drives the messy-over-clean mis-ranking.

The collapse signature (pinned via the Tal game, 25.Qxc5): our eval is trustworthy in SIMPLIFIED
positions but over-optimistic in MESSY ones, so at a fork it prefers the greedy/messy line over the
clean/safe one that Stockfish plays. This localizes WHY, per eval term, across the classified slip
corpus instead of assuming it is the piece_value_boost (the material-fraction "dominate when ahead"
term) -- it might be placement (`pieces`), imbalance, latent_threat, etc.

Method (depth-1 shadow of the real mis-ranking): for each slip row we have the FEN, our_move (what
the engine played), and sf_best (what SF wanted). Apply each to the FEN and take our OWN static
per-term breakdown of the two resulting children. The engine preferred our_move, so from the mover's
side our static eval rates child(our_move) >= child(sf_best). Decompose that surplus per term:

    surplus_term = mover_pov( child_ourmove[term] ) - mover_pov( child_sfbest[term] )

summed over terms == the total depth-1 preference gap. A term with a large positive mean surplus in
the `overpush` stratum is the one talking the engine into the messy line -> the tuning/condition
target. (Depth-1 proxy: the engine chose at ~d17, so a term flat here but guilty deeper won't show;
25.Qxc5 was verified to persist to real depth, so the proxy is a valid first cut, not the last word.)

Run in WSL from NN Engine/, at the interpreter that built ChessAI:
    python diagnostics/slip_term_attrib.py [collapse_classified.csv]
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys
import csv
from collections import defaultdict

import chess

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ENGINE_DIR = os.path.dirname(THIS_DIR)
sys.path.insert(0, ENGINE_DIR)

ENGINE_UNITS_PER_PAWN = 1000.0

# Additive terms captured from ev_breakdown (material is informational; imbalance split by colour).
ADDITIVE_TERMS = [
    "pieces", "capture_gains", "passed_pawn_support", "latent_threat", "king_safety", "central",
    "imbalance_white", "imbalance_black", "pair_bonus", "piece_value_boost",
]


def white_pawns(ev_abs):
    """Engine absolute (Black-positive) milli-pawns -> White-POV pawns."""
    return -ev_abs / ENGINE_UNITS_PER_PAWN


def child_breakdown(ai, board, uci):
    """Apply a UCI move (SAN-tolerant) to a copy and return its per-term breakdown, or None."""
    try:
        mv = chess.Move.from_uci(uci.strip())
    except ValueError:
        try:
            mv = board.parse_san(uci.strip())
        except Exception:
            return None
    if mv not in board.legal_moves:
        return None
    b = board.copy()
    b.push(mv)
    bd = ai.ev_breakdown(b)
    if bd.get("checkmate"):
        return None
    return bd


def main():
    corpus = sys.argv[1] if len(sys.argv) > 1 else os.path.join(THIS_DIR, "collapse_classified.csv")
    rows = list(csv.DictReader(open(corpus)))
    print("corpus %s  n=%d" % (os.path.basename(corpus), len(rows)))

    from ChessAI import ChessAI
    seed = chess.Board()
    ai = ChessAI(None, None, seed, seed.turn)

    # per-category accumulators: term -> list of mover-POV surpluses (child_ourmove - child_sfbest)
    by_cat = defaultdict(lambda: defaultdict(list))
    cat_total = defaultdict(list)   # total preference gap per category
    skipped = 0

    for r in rows:
        fen, cat = r.get("fen"), r.get("category", "?")
        our_move, sf_best = r.get("our_move", ""), r.get("sf_best", "")
        if not fen or not our_move or not sf_best or our_move == sf_best:
            skipped += 1
            continue
        try:
            board = chess.Board(fen)
        except ValueError:
            skipped += 1
            continue
        mover_sign = 1.0 if board.turn == chess.WHITE else -1.0  # mover-POV = White-POV * sign
        a = child_breakdown(ai, board, our_move)
        b = child_breakdown(ai, board, sf_best)
        if a is None or b is None:
            skipped += 1
            continue
        gap = 0.0
        for t in ADDITIVE_TERMS:
            s = mover_sign * (white_pawns(a[t]) - white_pawns(b[t]))
            by_cat[cat][t].append(s)
            gap += s
        cat_total[cat].append(gap)

    def mean(xs):
        return sum(xs) / len(xs) if xs else 0.0

    print("\nskipped %d rows (missing/illegal/equal-move)" % skipped)
    for cat in sorted(cat_total, key=lambda c: -len(cat_total[c])):
        tots = cat_total[cat]
        print("\n=== %s   n=%d   mean total preference gap (mover-POV pawns) = %+.3f ===" % (cat, len(tots), mean(tots)))
        print("   (+ = our eval rates OUR move higher on this term = the term arguing FOR the slip)")
        ranked = sorted(ADDITIVE_TERMS, key=lambda t: -abs(mean(by_cat[cat][t])))
        for t in ranked:
            m = mean(by_cat[cat][t])
            if abs(m) < 0.005:
                continue
            print("      %-22s %+8.3f" % (t, m))


if __name__ == "__main__":
    main()
