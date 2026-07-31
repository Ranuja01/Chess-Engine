# -*- coding: utf-8 -*-
"""Structural-gate validation: does an Ethereal/SF-style COORDINATION-COUNT gate + strong signed
suppressors separate the SF-quiet (quiet_neg) from SF-fires (attack) tiers where our MAGNITUDE floor
cannot? Pure python-chess geometry on ks_sts_corpus.csv.

For the more-endangered king per position, computes:
  attackers  = distinct enemy N/B/R/Q attacking the king ring
  has_queen  = enemy has a queen
  gate_fires = attackers > (1 - has_queen)        # Ethereal: >=2, or >=1 with a queen
  a signed Ethereal-style safety (positive threat minus suppressors), zeroed if the gate fails,
  and the resulting danger = max(0, safety).
Reports per tier: gate fire-rate, mean danger, and the AUC of this structural danger vs our units (0.81).

  pyrun diagnostics/ks_structural_gate.py
"""
import os, csv
import chess

THIS = os.path.dirname(os.path.abspath(__file__))
CORPUS = os.path.join(THIS, "ks_sets", "ks_sts_corpus.csv")
RING = {sq: chess.SquareSet(chess.BB_KING_ATTACKS[sq]) | chess.SquareSet(chess.BB_SQUARES[sq]) for sq in chess.SQUARES}
AW = {chess.KNIGHT: 48, chess.BISHOP: 24, chess.ROOK: 36, chess.QUEEN: 30}   # Ethereal mg attacker weights


def all_attacks(board, color):
    out = chess.SquareSet()
    for pt in (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING):
        for sq in board.pieces(pt, color):
            out |= board.attacks(sq)
    return out


def king_danger_struct(board, us):
    them = not us
    ksq = board.king(us)
    if ksq is None:
        return None
    ring = RING[ksq]
    has_q = 1 if board.pieces(chess.QUEEN, them) else 0

    attackers = 0
    wsum = 0
    ring_attacks = 0
    for pt in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN):
        for sq in board.pieces(pt, them):
            a = board.attacks(sq) & ring
            if a:
                attackers += 1
                wsum += AW[pt]
            ring_attacks += len(a)

    gate_fires = attackers > (1 - has_q)

    # weak ring squares (attacked, defended <=1 and only by K/Q)
    weak = 0
    for sq in ring:
        if not board.is_attacked_by(them, sq):
            continue
        defs = board.attackers(us, sq)
        if len(defs) <= 1 and all(board.piece_type_at(d) in (chess.KING, chess.QUEEN) for d in defs):
            weak += 1

    # signed safety, Ethereal-shaped (mg weights): threat  -  suppressors
    safety = wsum + 45 * ring_attacks // 4 + 42 * weak \
        + (-237) * (1 - has_q) \
        + (-74)                                   # base adjustment
    danger = max(0, safety) if gate_fires else 0
    return gate_fires, danger, attackers, has_q


def raw_danger(board, us):
    ksq = board.king(us)
    if ksq is None:
        return -1
    them = not us
    return sum(1 for sq in RING[ksq] if board.is_attacked_by(them, sq))


tiers = {"quiet_neg": [], "attack": [], "mid": []}
for r in csv.DictReader(open(CORPUS)):
    if r["tier"] not in tiers:
        continue
    b = chess.Board(r["fen"])
    us = chess.WHITE if raw_danger(b, chess.WHITE) >= raw_danger(b, chess.BLACK) else chess.BLACK
    res = king_danger_struct(b, us)
    if res:
        tiers[r["tier"]].append(res)


def auc(pos, neg):
    if not pos or not neg:
        return 0.5
    w = t = 0
    for a in pos:
        for b in neg:
            if a > b:
                w += 1
            elif a == b:
                t += 1
    return (w + 0.5 * t) / (len(pos) * len(neg))


print("Structural gate + signed suppressors  (Ethereal-shaped) on the SF11-anchored corpus\n")
print("%-10s %5s  gate_fire%%  mean_danger  mean_attackers  queenless%%" % ("tier", "n"))
for t in ("attack", "mid", "quiet_neg"):
    rows = tiers[t]
    n = len(rows)
    if not n:
        continue
    gate = 100.0 * sum(1 for g, d, a, q in rows if g) / n
    md = sum(d for g, d, a, q in rows) / n
    ma = sum(a for g, d, a, q in rows) / n
    ql = 100.0 * sum(1 for g, d, a, q in rows if not q) / n
    print("%-10s %5d  %8.1f  %11.1f  %14.2f  %9.1f" % (t, n, gate, md, ma, ql))

adang = [d for g, d, a, q in tiers["attack"]]
qdang = [d for g, d, a, q in tiers["quiet_neg"]]
print("\nStructural danger AUC (attack vs quiet_neg) = %.3f   (our units AUC was 0.81)" % auc(adang, qdang))
# how much does the GATE alone zero each class?
for t in ("attack", "quiet_neg"):
    rows = tiers[t]
    zeroed = 100.0 * sum(1 for g, d, a, q in rows if not g) / len(rows)
    print("  %-10s zeroed by COUNT-GATE alone: %.1f%%" % (t, zeroed))
