# -*- coding: utf-8 -*-
"""Middle-layer artifact for the material over-read class. For each other-class collapse fen, decompose our
`material` breakdown term into RAW material (engine's TRUE values P1/N3.25/B3.45/R5/Q10) vs the CAPTURE-FOLD
adjustment (approximate_capture_gains subtracting value_gained from the captured side), and line it up against
SF11 static Material + total. Ranks by |capture-fold| and prints the worst + spread representatives so we can
read the tactics by hand. All numbers OUR point of view (sign-flipped when we are Black), pawns."""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
import sys
THIS = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(THIS)); sys.path.insert(0, THIS)
import chess
from eval_vs_sf11 import SF11Eval, SF11
from ChessAI import ChessAI

# Engine's TRUE piece values (cpp_bitboard.h:141 values[] = {0,1000,3250,3450,5000,10000,12000}), in pawns.
VAL = {chess.PAWN: 1.0, chess.KNIGHT: 3.25, chess.BISHOP: 3.45, chess.ROOK: 5.0, chess.QUEEN: 10.0}

ai = ChessAI(None, None, chess.Board(), True)
sf = SF11Eval(SF11)
path = os.path.join(THIS, "ks_sets", "other_collapses.txt")
rows = []
try:
    for ln in open(path):
        if not ln.strip():
            continue
        fen = ln.rstrip("\n").split("\t", 1)[-1].strip()
        b = chess.Board(fen)
        us = b.turn; them = not us
        pov = 1.0 if us == chess.WHITE else -1.0
        bd = ai.ev_breakdown(b)
        # breakdown terms are Black-positive millipawns; ev_breakdown may already flip — match ks_material_check
        our_mat = (-bd.get("material", 0.0) / 1000.0) * pov            # our-POV material term (post capture-fold)
        pv_boost = (-bd.get("piece_value_boost", 0.0) / 1000.0) * pov  # our-POV PV_BOOST contribution
        our_total = (-bd.get("total", 0.0) / 1000.0) * pov            # our-POV full static eval
        raw = sum(VAL[pt] * (len(b.pieces(pt, us)) - len(b.pieces(pt, them))) for pt in VAL)  # true-value raw, our-POV
        fold = our_mat - raw                                          # the capture-fold adjustment
        _, terms = sf.eval(fen)
        sf_mat = terms.get("Material", 0.0) * pov
        sf_tot = terms.get("Total", terms.get("total", 0.0)) * pov
        rows.append(dict(fen=fen, our_mat=our_mat, raw=raw, fold=fold, pv=pv_boost,
                         our_tot=our_total, sf_mat=sf_mat, sf_tot=sf_tot, side=("W" if us else "B")))
finally:
    sf.close()

import statistics
n = len(rows)
print("n=%d" % n)
print("mean our_material_term = %+.2f | mean raw(true vals) = %+.2f | mean capture-FOLD = %+.2f | mean SF11_Material = %+.2f"
      % (statistics.mean(r["our_mat"] for r in rows), statistics.mean(r["raw"] for r in rows),
         statistics.mean(r["fold"] for r in rows), statistics.mean(r["sf_mat"] for r in rows)))
print("mean PV_BOOST = %+.2f | mean our_total = %+.2f | mean SF11_total = %+.2f"
      % (statistics.mean(r["pv"] for r in rows), statistics.mean(r["our_tot"] for r in rows),
         statistics.mean(r["sf_tot"] for r in rows)))

rows.sort(key=lambda r: -abs(r["fold"]))
# representatives: worst fold, ~median fold, and the largest opposite-sign (or smallest) fold
picks = [rows[0], rows[n // 2], rows[-1]]
labels = ["WORST |fold|", "MEDIAN |fold|", "SMALLEST |fold|"]

hdr = "%-14s %3s %8s %8s %8s %8s %8s %8s %8s" % (
    "pick", "stm", "our_mat", "raw", "FOLD", "PV_boost", "our_tot", "SF11mat", "SF11tot")
print("\n" + hdr)
print("-" * len(hdr))
for lab, r in zip(labels, picks):
    print("%-14s %3s %+8.2f %+8.2f %+8.2f %+8.2f %+8.2f %+8.2f %+8.2f" % (
        lab, r["side"], r["our_mat"], r["raw"], r["fold"], r["pv"], r["our_tot"], r["sf_mat"], r["sf_tot"]))
print("\nFENs (our POV; all values in pawns):")
for lab, r in zip(labels, picks):
    print("  [%s]  %s" % (lab, r["fen"]))
