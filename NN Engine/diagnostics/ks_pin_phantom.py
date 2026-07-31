# -*- coding: utf-8 -*-
"""Measure-first for the capgains absolute-pin fix. Reproducing capgains in Python is not faithful, so instead
estimate the PHANTOM capture capgains can fabricate from an absolutely-pinned attacker: for each side, any of
its pieces that is absolutely pinned to its king yet attacks an enemy piece OFF its pin ray represents a
capture capgains counts (pin-blind SEE) but that is actually illegal. We take the max such off-ray target
value per side as a rough phantom, net it to our POV, and correlate against the measured material over-read.
If the fat-tail (high material over-read) positions carry big phantoms, pins are the bulk -> worth the C++ fix."""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
import sys
THIS = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(THIS)); sys.path.insert(0, THIS)
import chess
from ChessAI import ChessAI

VAL = {chess.PAWN: 1.0, chess.KNIGHT: 3.25, chess.BISHOP: 3.45, chess.ROOK: 5.0, chess.QUEEN: 10.0, chess.KING: 0.0}

def side_phantom(board, color):
    """Largest off-ray enemy target value an absolutely-pinned piece of `color` illegally 'attacks'."""
    best = 0.0
    ksq = board.king(color)
    if ksq is None:
        return 0.0
    for sq in board.pieces_mask_iter(color) if hasattr(board, "pieces_mask_iter") else chess.SquareSet(board.occupied_co[color]):
        pc = board.piece_at(sq)
        if pc is None or pc.piece_type == chess.KING:
            continue
        if not board.is_pinned(color, sq):
            continue
        pin_ray = board.pin(color, sq)  # squares the pinned piece may move along (incl. king & pinner)
        for tgt in board.attacks(sq):
            ep = board.piece_at(tgt)
            if ep is None or ep.color == color:
                continue
            if tgt not in pin_ray:  # off-ray capture = illegal, but pin-blind SEE would count it
                best = max(best, VAL[ep.piece_type])
    return best

ai = ChessAI(None, None, chess.Board(), True)
path = os.path.join(THIS, "ks_sets", "other_collapses.txt")
rows = []
for ln in open(path):
    if not ln.strip():
        continue
    fen = ln.rstrip("\n").split("\t", 1)[-1].strip()
    b = chess.Board(fen)
    us = b.turn; them = not us
    pov = 1.0 if us == chess.WHITE else -1.0
    bd = ai.ev_breakdown(b)
    our_mat = (-bd.get("material", 0.0) / 1000.0) * pov
    capg = (-bd.get("capture_gains", 0.0) / 1000.0) * pov
    raw = sum(VAL[pt] * (len(b.pieces(pt, us)) - len(b.pieces(pt, them))) for pt in VAL)
    fold = our_mat - raw
    phantom = side_phantom(b, us) - side_phantom(b, them)   # our-POV net phantom capture value
    rows.append(dict(fen=fen, fold=fold, capg=capg, phantom=phantom, our_mat=our_mat, raw=raw))

import statistics
n = len(rows)
withp = [r for r in rows if abs(r["phantom"]) > 0.01]
print("n=%d  positions with an off-ray absolute-pin phantom: %d (%.0f%%)" % (n, len(withp), 100.0*len(withp)/n))
print("mean |fold| all=%.2f | with-phantom=%.2f | no-phantom=%.2f" % (
    statistics.mean(abs(r["fold"]) for r in rows),
    statistics.mean(abs(r["fold"]) for r in withp) if withp else 0.0,
    statistics.mean(abs(r["fold"]) for r in rows if abs(r["phantom"]) <= 0.01) or 0.0))
print("mean capg all=%.2f | with-phantom=%.2f" % (
    statistics.mean(r["capg"] for r in rows), statistics.mean(r["capg"] for r in withp) if withp else 0.0))
rows.sort(key=lambda r: -r["fold"])
print("\ntop-10 by capture-FOLD  (fold | capg_term | net_phantom | our_mat | raw):")
for r in rows[:10]:
    print("  fold=%+6.2f capg=%+6.2f phantom=%+6.2f  our_mat=%+6.2f raw=%+6.2f | %s" % (
        r["fold"], r["capg"], r["phantom"], r["our_mat"], r["raw"], r["fen"][:46]))
