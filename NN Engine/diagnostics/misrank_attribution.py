# -*- coding: utf-8 -*-
"""Condition-discovery on the autopsy corpus: for each position where our search preferred the WRONG move
(base_move) over SF's best (sf_best), attribute the mistake to individual eval TERMS. The term whose value
most favours base_move over sf_best (from OUR side's point of view) is what drove the misrank. Aggregating
across all such positions ranks WHICH grounded eval terms systematically push us onto losing moves — i.e.
where position-conditional corrections belong (data-pointed, not guessed).

Sign convention: our static eval + every per-term value is ABSOLUTE Black-positive milli-pawns. The side to
move in the corpus FEN is US (it is our move), so our-POV = +term if we are Black else -term. After pushing a
candidate move the absolute convention is unchanged. A term's contribution to preferring base over sf is
    c_t = our_pov_sign * ( term_t(after base_move) - term_t(after sf_best) )
and Σ_t c_t = our_pov_sign * (total_after_base - total_after_sf) > 0 (we DID prefer base).

Run: pyrun diagnostics/misrank_attribution.py [--categories eval_misrank,drift_no_single_blunder]
"""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
import sys
import csv
import argparse
from collections import defaultdict

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ENGINE_DIR = os.path.dirname(THIS_DIR)
sys.path.insert(0, ENGINE_DIR)
import chess

TERMS = ["material", "pieces", "capture_gains", "passed_pawn_support", "latent_threat", "threats", "king_safety",
         "central", "imbalance_white", "imbalance_black", "pair_bonus", "piece_value_boost", "pawn_majority",
         "pawn_struct", "outpost", "mobility",
         "pt_pawns", "pt_knights", "pt_bishops", "pt_rooks", "pt_queens", "pt_kings"]

CORPUS = os.path.join(ENGINE_DIR, "selfplay", "games", "mediocre_mine", "autopsy_corpus.csv")


def load_engine():
    from ChessAI import ChessAI
    seed = chess.Board()
    return ChessAI(None, None, seed, seed.turn)


def term_vec(ai, board):
    bd = ai.ev_breakdown(board)
    if bd.get("checkmate"):
        return None
    return {t: float(bd.get(t, 0.0)) for t in TERMS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=CORPUS)
    ap.add_argument("--categories", default="eval_misrank")
    args = ap.parse_args()
    cats = set(c.strip() for c in args.categories.split(",") if c.strip())
    ai = load_engine()

    contrib_sum = defaultdict(float)      # signed: >0 = term pushed us toward the WRONG move on average
    contrib_abs = defaultdict(float)      # magnitude of involvement
    drive_count = defaultdict(int)        # # positions where this term was the single largest wrong-driver
    n = 0
    skipped = 0
    for r in csv.DictReader(open(args.corpus)):
        if r["category"] not in cats:
            continue
        try:
            b = chess.Board(r["fen"])
            bm = chess.Move.from_uci(r["base_move"])
            sm = chess.Move.from_uci(r["sf_best"])
        except (ValueError, KeyError):
            skipped += 1
            continue
        if bm not in b.legal_moves or sm not in b.legal_moves or bm == sm:
            skipped += 1
            continue
        sign = 1.0 if b.turn == chess.BLACK else -1.0
        bb = b.copy(); bb.push(bm)
        sb = b.copy(); sb.push(sm)
        tb = term_vec(ai, bb)
        ts = term_vec(ai, sb)
        if tb is None or ts is None:
            skipped += 1
            continue
        contribs = {t: sign * (tb[t] - ts[t]) for t in TERMS}
        for t, c in contribs.items():
            contrib_sum[t] += c
            contrib_abs[t] += abs(c)
        driver = max(TERMS, key=lambda t: contribs[t])
        drive_count[driver] += 1
        n += 1

    if not n:
        print("no positions matched categories=%s (skipped %d)" % (sorted(cats), skipped)); return
    print("categories=%s  positions=%d  skipped=%d\n" % (sorted(cats), n, skipped))
    print("%-22s %10s %10s %8s" % ("term", "mean_signed", "mean_|c|", "#driver"))
    print("-" * 54)
    for t in sorted(TERMS, key=lambda t: -contrib_abs[t]):
        print("%-22s %10.1f %10.1f %8d"
              % (t, contrib_sum[t] / n, contrib_abs[t] / n, drive_count[t]))
    print("\nmean_signed>0 = term systematically favours the WRONG move (candidate for a conditional CUT).")
    print("#driver = times this term was the single largest wrong-move driver.")


if __name__ == "__main__":
    main()
