# -*- coding: utf-8 -*-
"""Is our `material` over-read on the other-class collapses a CALCULATION bug or genuine over-scaling?
Compare our ev_breakdown 'material' term (our POV) to the ACTUAL raw material with our own piece values
(P1000 N3150 B3250 R5000 Q9000) and to SF11 'Material'. If our term ~= actual raw -> the edge is real (it's
over-scaling / conversion, not a bug). If our term >> actual raw -> a material-term bug (a free fix)."""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
import sys
THIS = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(THIS)); sys.path.insert(0, THIS)
import chess
from eval_vs_sf11 import SF11Eval, SF11
from ChessAI import ChessAI
ai = ChessAI(None, None, chess.Board(), True)
sf = SF11Eval(SF11)
VAL = {chess.PAWN: 1.0, chess.KNIGHT: 3.15, chess.BISHOP: 3.25, chess.ROOK: 5.0, chess.QUEEN: 9.0}
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
        our_mat = (-bd.get("material", 0.0) / 1000.0) * pov            # our term, our POV
        raw = sum(VAL[pt] * (len(b.pieces(pt, us)) - len(b.pieces(pt, them))) for pt in VAL)  # our-POV, our values
        _, terms = sf.eval(fen)
        sf_mat = terms.get("Material", 0.0) * pov
        rows.append((our_mat, raw, sf_mat, our_mat - raw, fen))
finally:
    sf.close()
import statistics
n = len(rows)
print("n=%d  mean our_material_term=%.2f  mean raw_material(our vals)=%.2f  mean SF11_Material=%.2f"
      % (n, statistics.mean(r[0] for r in rows), statistics.mean(r[1] for r in rows), statistics.mean(r[2] for r in rows)))
print("mean (our_term - raw) = %.2f   (>0 = our material term exceeds actual piece count = BUG signal)"
      % statistics.mean(r[3] for r in rows))
rows.sort(key=lambda r: -abs(r[3]))
print("\nworst term-vs-raw discrepancies:")
for our_mat, raw, sf_mat, d, fen in rows[:6]:
    print("  our_term=%+.2f raw=%+.2f SF11=%+.2f  diff=%+.2f  %s" % (our_mat, raw, sf_mat, d, fen[:50]))
