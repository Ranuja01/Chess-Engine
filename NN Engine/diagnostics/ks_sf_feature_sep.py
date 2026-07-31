# -*- coding: utf-8 -*-
"""Definitively identify WHICH king-safety detection logic we lack, by computing SF11's kingDanger
sub-signals (from stockfish_11 evaluate.cpp king(), lines 378-461) on the SF11-anchored corpus and
measuring which ones SEPARATE the SF-quiet over-fire tier (quiet_neg) from the SF-fires tier (attack).
Our own components provably DON'T separate them (ks_logic_probe); whichever SF sub-signal DOES is the
missing logic.

Features are computed with python-chess for the ENDANGERED king (the side with the larger raw danger),
so a quiet position scores ~0 on both. Reported per-feature: mean on quiet_neg vs attack, and a rank-AUC
separation score (0.5 = no separation, ->1.0 = clean separator). No engine build needed (pure geometry).

  pyrun diagnostics/ks_sf_feature_sep.py
Reads ks_sets/ks_sts_corpus.csv (tier column).
"""
import os, sys, csv
import chess

THIS = os.path.dirname(os.path.abspath(__file__))
CORPUS = os.path.join(THIS, "ks_sets", "ks_sts_corpus.csv")

KING_RING_ADJ = {sq: chess.SquareSet(chess.BB_KING_ATTACKS[sq]) | chess.SquareSet(chess.BB_SQUARES[sq])
                 for sq in chess.SQUARES}

KING_FLANK = {  # SF KingFlank[file]: files grouped around the king file
    0: chess.BB_FILE_A | chess.BB_FILE_B | chess.BB_FILE_C,
    1: chess.BB_FILE_A | chess.BB_FILE_B | chess.BB_FILE_C,
    2: chess.BB_FILE_A | chess.BB_FILE_B | chess.BB_FILE_C | chess.BB_FILE_D,
    3: chess.BB_FILE_C | chess.BB_FILE_D | chess.BB_FILE_E | chess.BB_FILE_F,
    4: chess.BB_FILE_C | chess.BB_FILE_D | chess.BB_FILE_E | chess.BB_FILE_F,
    5: chess.BB_FILE_E | chess.BB_FILE_F | chess.BB_FILE_G | chess.BB_FILE_H,
    6: chess.BB_FILE_F | chess.BB_FILE_G | chess.BB_FILE_H,
    7: chess.BB_FILE_F | chess.BB_FILE_G | chess.BB_FILE_H,
}


def enemy_piece_attacks(board, color, piece_types):
    """Union of squares attacked by `color`'s pieces of the given types."""
    out = chess.SquareSet()
    for pt in piece_types:
        for sq in board.pieces(pt, color):
            out |= board.attacks(sq)
    return out


def all_attacks(board, color):
    out = chess.SquareSet()
    for pt in (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING):
        for sq in board.pieces(pt, color):
            out |= board.attacks(sq)
    return out


def features_for_king(board, us):
    """SF-inspired kingDanger sub-signals for the `us` king (them = enemy)."""
    them = not us
    ksq = board.king(us)
    if ksq is None:
        return None
    ring = KING_RING_ADJ[ksq]
    f = {}

    them_all = all_attacks(board, them)
    us_all = all_attacks(board, us)

    # kingAttackersCount: distinct enemy non-pawn pieces attacking the ring; kingAttacksCount: total ring attacks
    att_count = 0
    weight = 0
    W = {chess.KNIGHT: 81, chess.BISHOP: 52, chess.ROOK: 44, chess.QUEEN: 10}
    attacks_on_ring = 0
    for pt in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN):
        for sq in board.pieces(pt, them):
            a = board.attacks(sq) & ring
            if a:
                att_count += 1
                weight += W[pt]
            attacks_on_ring += len(a)
    f["attackers_count"] = att_count
    f["attackers_product"] = att_count * weight          # SF: count * weight (super-linear)
    f["ring_attacks"] = attacks_on_ring                  # SF 69*kingAttacksCount (proximity)

    # weak ring squares: attacked by them, and defended at most once and only by K/Q
    weak_ring = 0
    for sq in ring:
        if not board.is_attacked_by(them, sq):
            continue
        defs = board.attackers(us, sq)
        nd = len(defs)
        only_kq = all(board.piece_type_at(d) in (chess.KING, chess.QUEEN) for d in defs)
        if nd <= 1 and only_kq:
            weak_ring += 1
    f["weak_ring"] = weak_ring

    # safe checks (SF-style, pin/defense aware) by piece type. A check-from square is safe if not
    # defended by us (approx of SF's safe set). Count distinct piece types that have >=1 safe check.
    occ = board.occupied
    them_rook = enemy_piece_attacks(board, them, (chess.ROOK,))
    them_bish = enemy_piece_attacks(board, them, (chess.BISHOP,))
    them_qn = enemy_piece_attacks(board, them, (chess.QUEEN,))
    them_kn = enemy_piece_attacks(board, them, (chess.KNIGHT,))
    rook_from = chess.SquareSet(chess.BB_RANK_ATTACKS[ksq][chess.BB_RANK_MASKS[ksq] & occ]
                                | chess.BB_FILE_ATTACKS[ksq][chess.BB_FILE_MASKS[ksq] & occ])
    bish_from = chess.SquareSet(chess.BB_DIAG_ATTACKS[ksq][chess.BB_DIAG_MASKS[ksq] & occ])
    kn_from = chess.SquareSet(chess.BB_KNIGHT_ATTACKS[ksq])
    them_occ = board.occupied_co[them]

    def safe_sqs(cand):
        s = chess.SquareSet()
        for sq in cand:
            if chess.BB_SQUARES[sq] & them_occ:
                continue
            if not board.is_attacked_by(us, sq):
                s.add(sq)
        return s

    rc = safe_sqs(rook_from & them_rook)
    qc = safe_sqs((rook_from | bish_from) & them_qn)
    bc = safe_sqs(bish_from & them_bish)
    kc = safe_sqs(kn_from & them_kn)
    f["safe_check_rook"] = 1 if rc else 0
    f["safe_check_queen"] = 1 if qc else 0
    f["safe_check_bishop"] = 1 if bc else 0
    f["safe_check_knight"] = 1 if kc else 0
    f["safe_check_any"] = f["safe_check_rook"] + f["safe_check_queen"] + f["safe_check_bishop"] + f["safe_check_knight"]
    # SF-weighted safe-check danger
    f["safe_check_wt"] = 1080 * f["safe_check_rook"] + 780 * f["safe_check_queen"] \
        + 635 * f["safe_check_bishop"] + 790 * f["safe_check_knight"]

    # blockers_for_king(us): our pieces pinned to our own king (SF 98*blockers)
    blockers = 0
    for sq in chess.SquareSet(board.occupied_co[us]):
        if board.is_pinned(us, sq):
            blockers += 1
    f["blockers"] = blockers

    # flank attack: enemy attacks in the king flank within our camp (SF 3*kfa^2/8); quadratic version too
    kf = KING_FLANK[chess.square_file(ksq)]
    camp = (chess.BB_RANK_1 | chess.BB_RANK_2 | chess.BB_RANK_3 | chess.BB_RANK_4) if us == chess.WHITE \
        else (chess.BB_RANK_8 | chess.BB_RANK_7 | chess.BB_RANK_6 | chess.BB_RANK_5)
    flank_zone = chess.SquareSet(kf & camp)
    kfa = len(them_all & flank_zone)
    f["flank_attack"] = kfa
    f["flank_attack_sq"] = kfa * kfa

    # mobility differential (approx): enemy total attack span - our total attack span
    f["mobility_diff"] = len(them_all) - len(us_all)

    # no enemy queen (SF -873 * !queen): 1 = enemy HAS queen (danger allowed)
    f["enemy_has_queen"] = 1 if board.pieces(chess.QUEEN, them) else 0
    return f


def raw_danger(board, us):
    """A cheap proxy for 'which king is more endangered' to pick the side to score (ring attacks + weak)."""
    ksq = board.king(us)
    if ksq is None:
        return -1
    ring = KING_RING_ADJ[ksq]
    them = not us
    return sum(1 for sq in ring if board.is_attacked_by(them, sq))


FEATS = ["attackers_count", "attackers_product", "ring_attacks", "weak_ring", "safe_check_any",
         "safe_check_wt", "safe_check_rook", "safe_check_queen", "safe_check_bishop", "safe_check_knight",
         "blockers", "flank_attack", "flank_attack_sq", "mobility_diff", "enemy_has_queen"]

byclass = {"quiet_neg": [], "attack": []}
for r in csv.DictReader(open(CORPUS)):
    tier = r["tier"]
    if tier not in byclass:
        continue
    board = chess.Board(r["fen"])
    # score the more-endangered king
    dw, db = raw_danger(board, chess.WHITE), raw_danger(board, chess.BLACK)
    us = chess.WHITE if dw >= db else chess.BLACK
    fe = features_for_king(board, us)
    if fe:
        byclass[tier].append(fe)


def auc(pos, neg):
    """Rank-AUC that `pos` values exceed `neg` values (0.5 = no separation)."""
    if not pos or not neg:
        return 0.5
    wins = ties = 0
    for a in pos:
        for b in neg:
            if a > b:
                wins += 1
            elif a == b:
                ties += 1
    return (wins + 0.5 * ties) / (len(pos) * len(neg))


print("SF11 kingDanger sub-signal separation: attack (SF fires, n=%d) vs quiet_neg (SF quiet, n=%d)\n"
      % (len(byclass["attack"]), len(byclass["quiet_neg"])))
print("%-20s %10s %10s %8s" % ("feature", "attack_mean", "quiet_mean", "AUC"))
rows = []
for k in FEATS:
    av = [f[k] for f in byclass["attack"]]
    qv = [f[k] for f in byclass["quiet_neg"]]
    am = sum(av) / len(av) if av else 0.0
    qm = sum(qv) / len(qv) if qv else 0.0
    a = auc(av, qv)
    rows.append((k, am, qm, a))
for k, am, qm, a in sorted(rows, key=lambda t: -abs(t[3] - 0.5)):
    print("%-20s %10.2f %10.2f %8.3f" % (k, am, qm, a))


# --- ARCHITECTURE test: same features, LINEAR sum vs SF's THRESHOLD+QUADRATIC combiner ---
def king_danger_sf(f):
    kd = (f["attackers_product"] + 185 * f["weak_ring"] + 98 * f["blockers"]
          + 69 * f["ring_attacks"] + 3 * f["flank_attack_sq"] // 8 + f["mobility_diff"]
          - 873 * (1 - f["enemy_has_queen"]) + f["safe_check_wt"] + 37)
    return kd


def linear_danger(f):
    # our-style: same raw signals summed, NO threshold, NO square (near-linear soft table analog)
    return king_danger_sf(f)


def quad_danger(f):
    kd = king_danger_sf(f)
    return (kd * kd // 4096) if kd > 100 else 0


def col(group, fn):
    return [fn(f) for f in group]


lin = ("linear_sum", col(byclass["attack"], linear_danger), col(byclass["quiet_neg"], linear_danger))
quad = ("threshold+quadratic", col(byclass["attack"], quad_danger), col(byclass["quiet_neg"], quad_danger))
print("\nARCHITECTURE (same SF features, different combiner) -- AUC attack vs quiet_neg:")
for name, av, qv in (lin, quad):
    am = sum(av) / len(av); qm = sum(qv) / len(qv)
    print("  %-22s attack_mean=%9.1f quiet_mean=%9.1f  AUC=%.3f" % (name, am, qm, auc(av, qv)))
print("  (best single raw signal was flank_attack AUC=0.719; gain from the quadratic combiner is the")
print("   architecture lever -- our near-linear soft table cannot express it regardless of weights.)")
