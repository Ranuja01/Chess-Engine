# -*- coding: utf-8 -*-
"""
Over-read discriminator — can a CHEAP, already-computed signal separate a REAL placement claim from a
FANTASY one, so a realizability conditioner is buildable (vs a load-bearing damp that dies at the gauntlet)?

The eval-damp lane died repeatedly to load-bearing optimism: the placement over-read that LOSES messy
ahead positions is the SAME term that WINS positions where the activity is real, so a uniform (or wrong-
regime) damp nets zero. A conditional shrinkage can only work if some cheap detector FIRES on the fantasy
positions but NOT on the real ones. overread_bench.csv gives a continuous label per position:
    gap_pawns = our_static - sf_static   (+ = we over-read = fantasy; ~0 = we agree = real)
So: for each position, pull our OWN ev_breakdown terms (all free — the engine computes them anyway) and
measure which term correlates with gap_pawns. A term with real correlation = a validated, zero-NPS
conditioner. Flat correlations across the board = the over-read is NOT discriminable from our own signals
=> conditional shrinkage is a dead end, use the move-bias (tie-break) route instead.

Restrict to AHEAD positions (our_pawns > +1) because that is where the collapse lives and where a
"shrink placement when ahead-and-<detector>" conditioner would fire.

Run in WSL from NN Engine/:
    python diagnostics/overread_discriminator.py [overread_bench.csv]
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys
import csv

import numpy as np
import chess

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ENGINE_DIR = os.path.dirname(THIS_DIR)
sys.path.insert(0, ENGINE_DIR)

ENGINE_UNITS_PER_PAWN = 1000.0

# Detector candidates from ev_breakdown (White-POV pawns). We test each as a predictor of the over-read.
# The realizability-relevant ones are the OPPONENT-pressure signals (king_safety, latent_threat, imbalance):
# if opponent counterplay we compute predicts our over-read, we can condition placement on it for free.
DETECTOR_TERMS = [
    "pieces", "piece_value_boost", "capture_gains", "latent_threat", "king_safety",
    "central", "imbalance_white", "imbalance_black", "passed_pawn_support", "material",
]


def white_pawns(ev_abs):
    return -ev_abs / ENGINE_UNITS_PER_PAWN


def main():
    corpus = sys.argv[1] if len(sys.argv) > 1 else os.path.join(THIS_DIR, "overread_bench.csv")
    rows = list(csv.DictReader(open(corpus)))
    print("corpus %s  n=%d" % (os.path.basename(corpus), len(rows)))

    from ChessAI import ChessAI
    seed = chess.Board()
    ai = ChessAI(None, None, seed, seed.turn)

    gaps, our_p = [], []
    det = {t: [] for t in DETECTOR_TERMS}
    # From the mover's POV: over-read that helps the slip is a mover-POV signal. Store both raw (White-POV)
    # detector and the mover-signed version; regress gap (mover-POV over-read) against mover-signed detectors.
    skipped = 0
    for r in rows:
        fen = r.get("fen")
        try:
            gap = float(r.get("gap_pawns"))
            op = float(r.get("our_pawns"))
        except (TypeError, ValueError):
            skipped += 1
            continue
        try:
            board = chess.Board(fen)
        except ValueError:
            skipped += 1
            continue
        bd = ai.ev_breakdown(board)
        if bd.get("checkmate"):
            skipped += 1
            continue
        sign = 1.0 if board.turn == chess.WHITE else -1.0
        gaps.append(sign * gap)      # mover-POV over-read (+ = mover over-reads its own position)
        our_p.append(sign * op)      # mover-POV static eval (+ = mover is ahead per us)
        for t in DETECTOR_TERMS:
            det[t].append(sign * white_pawns(bd[t]))

    gaps = np.array(gaps); our_p = np.array(our_p)
    print("skipped %d; usable %d" % (skipped, len(gaps)))

    def corr_block(mask, label):
        g = gaps[mask]
        if len(g) < 30:
            print("\n[%s] n=%d (too few)" % (label, len(g)))
            return
        print("\n[%s] n=%d  mean mover-POV over-read gap = %+.3f p" % (label, len(g), float(g.mean())))
        print("   detector (mover-POV)      corr_with_gap   mean")
        scored = []
        for t in DETECTOR_TERMS:
            d = np.array(det[t])[mask]
            c = float(np.corrcoef(d, g)[0, 1]) if d.std() > 1e-9 and g.std() > 1e-9 else 0.0
            scored.append((abs(c), c, t, float(d.mean())))
        for _, c, t, m in sorted(scored, reverse=True):
            print("   %-22s   %+7.3f       %+7.3f" % (t, c, m))

    corr_block(np.ones(len(gaps), bool), "ALL")
    corr_block(our_p > 1.0, "AHEAD (mover static > +1.0p)  <- the collapse regime")
    corr_block(our_p > 3.0, "WELL AHEAD (mover static > +3.0p)")


if __name__ == "__main__":
    main()
