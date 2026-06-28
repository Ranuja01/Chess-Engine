# -*- coding: utf-8 -*-
"""Curate the IMPORTANT-MIDGAME subset of the tune corpus and enrich it with per-piece-type placement
(pt_*) + cheap board-state detectors — the corpus for detector-conditioned placement Texel tuning.

Why curated: eval precision only changes the game in decision-critical positions. We keep MIDGAME
(phase < PHASE_MAX) positions that are near-equal OR a slight edge that needs precision to hold
(|SF_static| < EDGE_MAX); we drop endgames and decided/lost positions (where placement calibration
doesn't matter). Reuses the existing SF labels in tune_data/corpus.csv (no Stockfish re-run) and only adds
the per-piece-type placement breakdown (ev_breakdown pt_*) + detectors so the rich conditioned fit can ask
"which piece's placement should bend with which detector".

Run in WSL from NN Engine/:
    python diagnostics/gen_midgame_corpus.py [--phase-max 64] [--edge-max 250]
Writes selfplay/tune_data/midgame_corpus.csv.
"""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
import sys
import csv
import argparse

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ENGINE_DIR = os.path.dirname(THIS_DIR)
sys.path.insert(0, ENGINE_DIR)
sys.path.insert(0, THIS_DIR)

import chess
from detector_placement_proof import detectors

PT = ["pt_pawns", "pt_knights", "pt_bishops", "pt_rooks", "pt_queens", "pt_kings"]


def load_engine():
    from ChessAI import ChessAI
    seed = chess.Board()
    return ChessAI(None, None, seed, seed.turn)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default=os.path.join(ENGINE_DIR, "selfplay", "tune_data", "corpus.csv"))
    ap.add_argument("--out", default=os.path.join(ENGINE_DIR, "selfplay", "tune_data", "midgame_corpus.csv"))
    ap.add_argument("--phase-max", type=int, default=64, help="keep phase_score < this (midgame)")
    ap.add_argument("--edge-max", type=int, default=250, help="keep |sf_static_cp| < this (decision-critical)")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.inp)))
    ai = load_engine()
    feat_names = list(detectors(chess.STARTING_FEN, 0).keys())
    cols = ["fen", "phase_score", "sf_static_cp", "our_total", "pieces"] + PT + feat_names

    written = skipped = 0
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            try:
                phase = float(r["phase_score"]); sf_cp = float(r["sf_static_cp"])
            except (ValueError, KeyError):
                skipped += 1; continue
            if not (phase < args.phase_max and abs(sf_cp) < args.edge_max):
                continue                                    # not an important-midgame point
            board = chess.Board(r["fen"])
            if board.is_check():
                skipped += 1; continue
            bd = ai.ev_breakdown(board)
            d = detectors(r["fen"], phase)
            row = {"fen": r["fen"], "phase_score": phase, "sf_static_cp": sf_cp,
                   "our_total": bd["total"], "pieces": bd["pieces"]}
            row.update({k: bd[k] for k in PT})
            row.update(d)
            w.writerow(row); written += 1
            if written % 1000 == 0:
                print(f"  written {written} (skipped {skipped})", flush=True)
    print(f"[gen_midgame] wrote {written} important-midgame rows (skipped {skipped}) -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
