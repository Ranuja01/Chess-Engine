# -*- coding: utf-8 -*-
"""Why is the SF11 double-pawn ring prune (kingRing &= ~dblAttackByPawn) inert in our engine?

Three candidate mechanisms, and they are distinguishable by counting, not by reading:
  (a) GEOMETRY   -- few double-own-pawn-defended squares fall inside our king zone at all
  (b) ALREADY-0  -- they are inside the zone but the enemy never attacks them, so they were
                    contributing nothing to attacked_zone_squares / weak before the prune
  (c) REDUNDANT  -- they are attacked, but by pieces that also attack other zone squares, so
                    the attacker set is unchanged and only the square COUNT moves

Pure python-chess; no engine call, no rebuild. Mirrors the zone build in cpp_bitboard.cpp:625-630
(ring1 = king + neighbours, plus that ring pushed one rank toward the enemy).

Run: bash <runner> pyrun diagnostics/_ks_dblpawn_coverage.py
"""
import os, sys, csv, statistics as st
import chess

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.join(THIS_DIR, 'ks_sets', 'ks_archetypes.csv')

FILE_A = chess.BB_FILE_A
FILE_H = chess.BB_FILE_H
ALL = chess.BB_ALL


def zone_of(king_sq, white):
    """ring-1 (king + neighbours) plus that ring pushed one rank toward the enemy."""
    ring1 = chess.BB_KING_ATTACKS[king_sq] | chess.BB_SQUARES[king_sq]
    fwd = ((ring1 << 8) & ALL) if white else (ring1 >> 8)
    return ring1 | fwd


def dbl_pawn_attacks(pawns, white):
    """Squares attacked by TWO of that side's pawns. Edge files masked before the shift."""
    if white:
        l = (pawns & ~FILE_A) << 7
        r = (pawns & ~FILE_H) << 9
    else:
        l = (pawns & ~FILE_H) >> 7
        r = (pawns & ~FILE_A) >> 9
    return (l & r) & ALL


def bits(bb):
    return chess.popcount(bb & ALL)


rows = list(csv.DictReader(open(BENCH)))
groups = {'DANGER(A)': [], 'QUIET(B)': []}

for r in rows:
    b = chess.Board(r['fen'])
    subj = r.get('subj', '?')
    if subj not in ('W', 'B'):
        subj = 'W' if float(r['sf_ks']) < 0 else 'B'
    white = (subj == 'W')
    colour = chess.WHITE if white else chess.BLACK
    king_sq = b.king(colour)
    if king_sq is None:
        continue

    zone = zone_of(king_sq, white)
    own_pawns = b.pieces_mask(chess.PAWN, colour)
    dbl = dbl_pawn_attacks(own_pawns, white)
    zdbl = zone & dbl

    # Which zone squares does the ENEMY attack (the only ones that are charged today)?
    enemy = not colour
    att_zone = 0
    att_zdbl = 0
    for sq in chess.scan_forward(zone):
        if b.attackers_mask(enemy, sq):
            att_zone |= chess.BB_SQUARES[sq]
            if chess.BB_SQUARES[sq] & dbl:
                att_zdbl |= chess.BB_SQUARES[sq]

    g = 'DANGER(A)' if r['archetype'][0] == 'A' else 'QUIET(B)'
    groups[g].append((bits(zone), bits(zdbl), bits(att_zone), bits(att_zdbl)))

print("SF dbl-pawn ring prune: how much of OUR zone does it actually remove, and was that part charged?")
print("%-12s %4s %8s %10s %10s %12s" % ("group", "n", "|zone|", "|zone&dbl|", "|zone&att|", "|zone&dbl&att|"))
for g, v in groups.items():
    if not v:
        continue
    print("%-12s %4d %8.2f %10.2f %10.2f %12.2f" % (
        g, len(v), st.mean(x[0] for x in v), st.mean(x[1] for x in v),
        st.mean(x[2] for x in v), st.mean(x[3] for x in v)))

print()
print("read: |zone&dbl| ~0            => (a) GEOMETRY: the prune has nothing to bite on here.")
print("      |zone&dbl| >0 but        => (b) ALREADY-0: the pruned squares were never enemy-attacked,")
print("      |zone&dbl&att| ~0            so they contributed 0 to the counts before the prune too.")
print("      |zone&dbl&att| >0        => (c) the prune DID remove charged squares; the null is downstream.")
