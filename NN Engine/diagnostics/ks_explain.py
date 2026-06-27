# -*- coding: utf-8 -*-
"""
King-safety explainer / visualizer — the teaching aid AND per-component verification gate for the
attack-unit king-safety term we're building (plan: handoff-lossless-speed-campaign-tranquil-rose).

For a given FEN it prints, for EACH king:
  - the board with the king zone outlined and an attack-heat overlay (digit per zone square =
    number of enemy pieces attacking that square),
  - a per-component readout of the attack-UNIT model we're building (attackers-by-type, zone
    attack-count, weak squares/holes, safe checks, pawn storm, open/semi-open files near the king,
    pawn shield), with the resulting `units` and a preview `danger = units^2 / KS_DIVISOR`,
  - the BEFORE number: the engine's CURRENT `latent_threat` term (from ChessAI.ev_breakdown) and the
    position's phase_score, so we can watch the new model land beside the crude one.

IMPORTANT (anti-drift): until the C++ `king_safety_score` exposes its own per-component breakdown (a
later build step wires that into EvalBreakdown), the per-component numbers below are computed in
python from the FEN with python-chess and the PREVIEW weights at the top of this file. They illustrate
the MODEL; they are not yet the engine's exact internal values. The `latent_threat` / `phase_score`
lines ARE read live from the engine. Once the C++ debug hook exists, switch the component readout to
read it (one place: `engine_components()`), and these previews become the cross-check.

Run in WSL, from NN Engine/, at the SAME interpreter that built ChessAI:

    python diagnostics/ks_explain.py "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 b kq - 0 1"
    python diagnostics/ks_explain.py --side white "6k1/5ppp/8/8/8/8/5PPP/3QR1K1 w - - 0 1"
    python diagnostics/ks_explain.py --no-engine "<fen>"          # skip the engine BEFORE number
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # silence TensorFlow startup chatter

import sys
import argparse

import chess

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ENGINE_DIR = os.path.dirname(THIS_DIR)            # NN Engine/  (has ChessAI*.so)
sys.path.insert(0, ENGINE_DIR)

# ---------------------------------------------------------------------------------------------------
# PREVIEW weights — these mirror the planned KS_* knobs (search_engine.h). They are illustrative
# defaults so the readout produces meaningful numbers; the engine's tuned values will differ. Keep the
# SHAPES here aligned with the C++ model (attack units -> non-linear table) as it's built.
# ---------------------------------------------------------------------------------------------------
KS_ATT = {chess.KNIGHT: 2, chess.BISHOP: 2, chess.ROOK: 3, chess.QUEEN: 5}  # attacker weight by type
KS_ATTACK_COUNT = 1   # per zone square the enemy attacks (additive pressure)
KS_WEAK         = 2   # per weak (pawn-undefended, enemy-attacked) zone square
KS_SAFE_CHECK   = 3   # per square from which the enemy can give a safe check
KS_STORM        = 1   # per rank of enemy pawn-storm advance on the king files
KS_OPEN_FILE    = 2   # per open/semi-open file on/adjacent to the king file
KS_SHIELD       = 2   # per friendly pawn shielding the king on its three files
KS_DIVISOR      = 4   # danger = clamp(units)^2 / KS_DIVISOR   (the non-linear safety table preview)
KS_CAP          = 40  # units clamp


def king_ring2(sq):
    """The 2-ring around a square: king moves from `sq`, plus king moves from each of those (the
    king-centered zone the new model uses). Returns a set of square indices. Mirrors the C++
    king_ring2[] build (BB_KING_ATTACKS OR'd one step out)."""
    ring = set(chess.SquareSet(chess.BB_KING_ATTACKS[sq]))
    ring.add(sq)
    for s in list(ring):
        ring |= set(chess.SquareSet(chess.BB_KING_ATTACKS[s]))
    return ring


def file_quadrant_zone(sq, color):
    """The engine's CURRENT zone (white_king_zones/black_king_zones, cpp_bitboard.h): a fixed 4-file x
    5-rank quadrant chosen by king file, excluding the three ranks behind the back rank. Rendered for
    comparison against king_ring2 only."""
    f = chess.square_file(sq)
    if f <= 2:
        files = [0, 1, 2, 3]
    elif f == 3:
        files = [1, 2, 3, 4]
    elif f == 4:
        files = [2, 3, 4, 5]
    else:
        files = [4, 5, 6, 7]
    ranks = range(0, 5) if color == chess.WHITE else range(3, 8)
    return {chess.square(ff, rr) for ff in files for rr in ranks}


def pawn_shield_squares(king_sq, color):
    """Up to the three friendly-pawn shield squares one rank in front of the king (king file + the two
    adjacent files)."""
    f, r = chess.square_file(king_sq), chess.square_rank(king_sq)
    fr = r + 1 if color == chess.WHITE else r - 1
    if not 0 <= fr <= 7:
        return set()
    return {chess.square(ff, fr) for ff in (f - 1, f, f + 1) if 0 <= ff <= 7}


def components_for_king(board, king_color):
    """Compute the attack-unit king-safety components for `king_color`'s king (the side being attacked),
    PREVIEW model in python-chess. Returns a dict with per-component counts, the per-square attacker
    heat, the derived `units`, and the preview `danger`."""
    enemy = not king_color
    king_sq = board.king(king_color)
    if king_sq is None:
        return None
    zone = king_ring2(king_sq)

    # Per-square enemy-attacker heat over the zone (the overlay), and the attacker SET (enemy pieces
    # that attack at least one zone square), bucketed by piece type.
    heat = {}
    attacker_squares = set()
    attacked_zone_squares = 0
    weak_squares = 0
    for sq in zone:
        atk = board.attackers(enemy, sq)
        n = len(atk)
        heat[sq] = n
        if n:
            attacked_zone_squares += 1
            attacker_squares |= set(atk)
            # weak = enemy-attacked zone square not defended by a friendly pawn
            if not (board.attackers(king_color, sq) & board.pieces(chess.PAWN, king_color)):
                weak_squares += 1

    att_by_type = {pt: 0 for pt in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN)}
    for s in attacker_squares:
        pc = board.piece_at(s)
        if pc and pc.piece_type in att_by_type:
            att_by_type[pc.piece_type] += 1

    # Safe checks: enemy piece can move to a square giving check, and that square is not defended by us
    # (preview: scan enemy pseudo-legal-ish check squares via attack tables).
    safe_checks = 0
    for sq in chess.SQUARES:
        if board.piece_at(sq) is not None:
            continue
        if not _gives_check_from(board, enemy, sq, king_sq):
            continue
        if not board.attackers(king_color, sq):
            safe_checks += 1

    # Pawn storm: enemy pawns advanced on the king's three files, scored by how far they've come.
    storm = 0
    kf = chess.square_file(king_sq)
    for ff in (kf - 1, kf, kf + 1):
        if not 0 <= ff <= 7:
            continue
        for s in board.pieces(chess.PAWN, enemy) & chess.SquareSet(chess.BB_FILES[ff]):
            r = chess.square_rank(s)
            adv = (7 - r) if king_color == chess.WHITE else r  # ranks advanced toward our king
            storm += max(0, adv - 1)

    # Open / semi-open files on/adjacent to the king file (no friendly pawn = exposed king).
    open_files = 0
    own_pawns = board.pieces(chess.PAWN, king_color)
    for ff in (kf - 1, kf, kf + 1):
        if not 0 <= ff <= 7:
            continue
        if not (own_pawns & chess.SquareSet(chess.BB_FILES[ff])):
            open_files += 1

    # Pawn shield: friendly pawns occupying the three shield squares in front of the king.
    shield = len([s for s in pawn_shield_squares(king_sq, king_color)
                  if board.piece_at(s) == chess.Piece(chess.PAWN, king_color)])

    units = (sum(KS_ATT[pt] * att_by_type[pt] for pt in att_by_type)
             + KS_ATTACK_COUNT * attacked_zone_squares
             + KS_WEAK * weak_squares
             + KS_SAFE_CHECK * safe_checks
             + KS_STORM * storm
             + KS_OPEN_FILE * open_files
             - KS_SHIELD * shield)
    units_clamped = max(0, min(units, KS_CAP))
    danger = units_clamped * units_clamped // KS_DIVISOR

    return {
        "king_sq": king_sq, "zone": zone, "heat": heat,
        "att_by_type": att_by_type, "attacked_zone_squares": attacked_zone_squares,
        "weak_squares": weak_squares, "safe_checks": safe_checks, "storm": storm,
        "open_files": open_files, "shield": shield,
        "units": units, "units_clamped": units_clamped, "danger": danger,
    }


def _gives_check_from(board, color, sq, enemy_king_sq):
    """Preview check-square test: would a piece of `color` standing on empty `sq` attack the enemy king?
    Tries each non-pawn piece type via python-chess attack tables, restricted to enemy pieces actually
    present (so we don't credit checks the side can't deliver)."""
    occ = board.occupied
    have = lambda pt: bool(board.pieces(pt, color))
    if have(chess.KNIGHT) and chess.BB_KNIGHT_ATTACKS[sq] & chess.BB_SQUARES[enemy_king_sq]:
        return True
    bishop_like = chess.BB_DIAG_ATTACKS[sq][chess.BB_DIAG_MASKS[sq] & occ]
    rook_like = (chess.BB_RANK_ATTACKS[sq][chess.BB_RANK_MASKS[sq] & occ]
                 | chess.BB_FILE_ATTACKS[sq][chess.BB_FILE_MASKS[sq] & occ])
    target = chess.BB_SQUARES[enemy_king_sq]
    if (have(chess.BISHOP) or have(chess.QUEEN)) and bishop_like & target:
        return True
    if (have(chess.ROOK) or have(chess.QUEEN)) and rook_like & target:
        return True
    return False


def render_board(board, comp, king_color):
    """Board from White's perspective (rank 8 at top) with the zone outlined and the attacker-heat
    digit shown on zone squares. '.' = empty zone square with 0 attackers; piece letters elsewhere."""
    zone, heat, king_sq = comp["zone"], comp["heat"], comp["king_sq"]
    lines = []
    for rank in range(7, -1, -1):
        cells = []
        for file in range(8):
            sq = chess.square(file, rank)
            pc = board.piece_at(sq)
            sym = pc.symbol() if pc else "."
            if sq in zone:
                n = heat.get(sq, 0)
                tag = (str(n) if n else "+")           # attacker count, or '+' for a covered-but-quiet zone square
                if sq == king_sq:
                    cell = "(%s)" % sym                  # the king itself
                elif pc:
                    cell = "%s%s " % (sym, tag)          # piece on a zone square + heat
                else:
                    cell = " %s " % tag                  # empty zone square: show heat / '+'
            else:
                cell = " %s " % sym
            cells.append("%-3s" % cell)
        lines.append("%d  %s" % (rank + 1, "".join(cells)))
    lines.append("   " + "".join("%-3s" % (" " + chr(ord('a') + f)) for f in range(8)))
    return "\n".join(lines)


def explain_side(board, king_color):
    comp = components_for_king(board, king_color)
    side = "WHITE" if king_color == chess.WHITE else "BLACK"
    print("=" * 78)
    print("%s king on %s" % (side, chess.square_name(comp["king_sq"])))
    print(render_board(board, comp, king_color))
    print()
    a = comp["att_by_type"]
    print("  attack-unit model (king_ring2 zone, %d squares):" % len(comp["zone"]))
    print("    attackers-by-type   N=%d B=%d R=%d Q=%d   (weights %s)"
          % (a[chess.KNIGHT], a[chess.BISHOP], a[chess.ROOK], a[chess.QUEEN],
             "/".join("%d" % KS_ATT[pt] for pt in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN))))
    print("    zone squares attacked   %2d   (x%d)" % (comp["attacked_zone_squares"], KS_ATTACK_COUNT))
    print("    weak squares (holes)    %2d   (x%d)" % (comp["weak_squares"], KS_WEAK))
    print("    safe checks             %2d   (x%d)" % (comp["safe_checks"], KS_SAFE_CHECK))
    print("    pawn storm (ranks)      %2d   (x%d)" % (comp["storm"], KS_STORM))
    print("    open/semi files near K  %2d   (x%d)" % (comp["open_files"], KS_OPEN_FILE))
    print("    pawn shield             %2d   (-x%d)" % (comp["shield"], KS_SHIELD))
    print("    --------------------------------------------------")
    print("    UNITS = %d  -> clamp %d  -> danger = %d^2/%d = %d cp (preview)"
          % (comp["units"], comp["units_clamped"], comp["units_clamped"], KS_DIVISOR, comp["danger"]))
    return comp["danger"]


def main():
    ap = argparse.ArgumentParser(description="King-safety attack-unit explainer / visualizer.")
    ap.add_argument("fen", help="FEN to explain")
    ap.add_argument("--side", choices=["white", "black", "both"], default="both",
                    help="which king to explain (default both)")
    ap.add_argument("--no-engine", action="store_true",
                    help="skip the engine BEFORE number (latent_threat); pure python preview")
    args = ap.parse_args()

    board = chess.Board(args.fen)

    engine_latent = None
    phase_score = None
    if not args.no_engine:
        try:
            from ChessAI import ChessAI
            seed = chess.Board()
            ai = ChessAI(None, None, seed, seed.turn)
            bd = ai.ev_breakdown(board)
            if not bd.get("checkmate"):
                engine_latent = -bd["latent_threat"] / 1000.0   # absolute (Black-positive) milli -> White-POV pawns
                phase_score = bd["phase_score"]
        except Exception as e:
            print("[ks_explain] engine unavailable (%s); showing python preview only\n" % e)

    sides = ([chess.WHITE, chess.BLACK] if args.side == "both"
             else [chess.WHITE] if args.side == "white" else [chess.BLACK])
    danger = {}
    for ks in sides:
        danger[ks] = explain_side(board, ks)
    print("=" * 78)
    if chess.WHITE in danger and chess.BLACK in danger:
        # White-POV net: a dangerous Black king favours White (+), a dangerous White king favours Black (-).
        net = (danger[chess.BLACK] - danger[chess.WHITE]) / 100.0
        print("  preview king_safety (White-POV pawns) = (black_danger %d - white_danger %d)/100 = %+.2f"
              % (danger[chess.BLACK], danger[chess.WHITE], net))
    if engine_latent is not None:
        print("  BEFORE (engine, live): latent_threat term = %+.2f pawns (White-POV), phase_score %s"
              % (engine_latent, phase_score))
    print("note: component numbers are the PREVIEW model (python-chess + preview weights); the engine's"
          "\n      latent_threat line is live. Wire the C++ per-component hook to make these the engine's"
          "\n      exact values (see module docstring).")


if __name__ == "__main__":
    main()
