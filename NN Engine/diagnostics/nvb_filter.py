# -*- coding: utf-8 -*-
"""
N-vs-B / minor-trade filter over the collapse corpus — does the missing material-Imbalance dimension
touch REAL losses, or only abstract accuracy?

A count-keyed Imbalance term (bishop-pair / knight-pawn synergy / redundancy / opponent-relative exchange
values) only matters if the engine's actual slips involve minor-piece imbalances or minor-trade decisions.
This is pure python-chess (no engine, ~0 CPU) so it is safe to run alongside a single-core bench.

For each classified slip (fen, category, our_move, sf_best) we flag:
  minor_imbalance   - the two sides hold DIFFERENT minor mixes (one side more knights, the other more
                      bishops) => a structural "bishop vs knight" position the term would re-value.
  our_minor_capture - our_move captures a knight or bishop.
  sf_minor_capture  - sf_best captures a knight or bishop.
  trade_disagree    - the disagreement itself is about a minor trade (exactly one of our_move / sf_best
                      captures a minor) => the decision the Imbalance term most directly governs.
Reports the rate of each, overall and per category (esp. overpush), + example FENs for trade_disagree.

    python diagnostics/nvb_filter.py [collapse_classified.csv]
"""

import os
import sys
import csv
from collections import defaultdict

import chess

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
MINORS = (chess.KNIGHT, chess.BISHOP)


def parse_move(board, s):
    s = (s or "").strip()
    if not s:
        return None
    try:
        return chess.Move.from_uci(s)
    except ValueError:
        try:
            return board.parse_san(s)
        except Exception:
            return None


def captures_minor(board, mv):
    if mv is None or not board.is_capture(mv):
        return False
    if board.is_en_passant(mv):
        return False  # pawn capture
    victim = board.piece_at(mv.to_square)
    return victim is not None and victim.piece_type in MINORS


def minor_imbalance(board):
    """True when the sides hold different minor mixes (one more knights, the other more bishops)."""
    wn = len(board.pieces(chess.KNIGHT, chess.WHITE)); wb = len(board.pieces(chess.BISHOP, chess.WHITE))
    bn = len(board.pieces(chess.KNIGHT, chess.BLACK)); bb = len(board.pieces(chess.BISHOP, chess.BLACK))
    # knight-edge and bishop-edge point opposite ways => a genuine N-vs-B structural imbalance
    return (wn - bn) * (wb - bb) < 0 or (wn + wb) != (bn + bb)


def main():
    corpus = sys.argv[1] if len(sys.argv) > 1 else os.path.join(THIS_DIR, "collapse_classified.csv")
    rows = list(csv.DictReader(open(corpus)))
    print("corpus %s  n=%d" % (os.path.basename(corpus), len(rows)))

    cat_n = defaultdict(int)
    cat_hit = defaultdict(lambda: defaultdict(int))
    examples = []
    skipped = 0

    for r in rows:
        fen, cat = r.get("fen"), r.get("category", "?")
        try:
            board = chess.Board(fen)
        except (ValueError, TypeError):
            skipped += 1
            continue
        om = parse_move(board, r.get("our_move"))
        sm = parse_move(board, r.get("sf_best"))
        om_cap = captures_minor(board, om)
        sm_cap = captures_minor(board, sm)
        imb = minor_imbalance(board)
        trade_disagree = (om_cap != sm_cap)

        cat_n[cat] += 1
        if imb: cat_hit[cat]["minor_imbalance"] += 1
        if om_cap: cat_hit[cat]["our_minor_capture"] += 1
        if sm_cap: cat_hit[cat]["sf_minor_capture"] += 1
        if trade_disagree:
            cat_hit[cat]["trade_disagree"] += 1
            if len(examples) < 12:
                examples.append((cat, fen, r.get("our_move"), r.get("sf_best"),
                                 "ours-caps-minor" if om_cap else "sf-caps-minor"))

    flags = ["minor_imbalance", "our_minor_capture", "sf_minor_capture", "trade_disagree"]
    tot_n = sum(cat_n.values())
    print("skipped %d\n" % skipped)
    print("%-14s %5s  %s" % ("category", "n", "  ".join("%-16s" % f for f in flags)))
    for cat in sorted(cat_n, key=lambda c: -cat_n[c]) + ["__ALL__"]:
        if cat == "__ALL__":
            n = tot_n
            cells = []
            for f in flags:
                h = sum(cat_hit[c][f] for c in cat_n)
                cells.append("%3d (%2.0f%%)" % (h, 100.0 * h / n if n else 0))
        else:
            n = cat_n[cat]
            cells = ["%3d (%2.0f%%)" % (cat_hit[cat][f], 100.0 * cat_hit[cat][f] / n if n else 0) for f in flags]
        print("%-14s %5d  %s" % (cat, n, "  ".join("%-16s" % c for c in cells)))

    print("\n=== trade_disagree examples (the decisions a material-Imbalance term most directly governs) ===")
    for cat, fen, om, sm, who in examples:
        print("  [%-11s] %-6s ours=%s sf=%s   %s" % (cat, who, om, sm, fen))


if __name__ == "__main__":
    main()
