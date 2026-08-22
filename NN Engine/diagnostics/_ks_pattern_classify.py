# -*- coding: utf-8 -*-
"""Structural KS failure-pattern classifier — PURE python-chess, NO engine load (single-core safe).

Tags each position with which king-safety FAILURE PATTERN(s) it structurally matches, so the eval-loss
tools (`sf11_collapse_gap.py`, `collapse_term_attribution.py`, `_collapse_leverage.py`) can then report
WIN%-error / points-forfeited PER PATTERN. This file only detects the geometry; it deliberately does NOT
evaluate — the over-read/under-read DIRECTION and the eval-loss magnitude are joined downstream by the
engine tools tonight. Keeping detection engine-free is what makes it runnable while the box is single-core.

Taxonomy (see dev_notes/KS-CAPABILITY-GAP-INDEX-2026-08-14.md). Flags are per-KING (the king under
examination); the CLI emits them for OUR king and THEIR king separately (POV from `our_color`), because a
KS eval-loss can come from under-reading our own king's danger OR over-reading our attack on theirs.

  OVER-read family (we score danger that isn't real):
    QUEENLESS_ATTACK   no enemy queen, but >=2 enemy heavy pieces bearing on the king zone
    PROXIMITY_OVERREAD zone looks busy (>=3 attacked zone squares) but <=1 real piece attacker, no safe check
    QUIET_KING         <=1 zone attacker at all (floor-flicker candidate)
  UNDER-read family (real danger we are structurally blind to):
    BATTERY            >=2 aligned enemy heavy pieces (doubled R / Q+R file/rank, Q+B diagonal) aimed at the zone
    CORNER_KING        king on a/b/g/h file near its home rank WITH >=1 enemy zone attacker (shrunken-ring blind)
    OPENFILE_MISREAD   king/adjacent file: our pawn gone AND enemy pawn still on it (looks open, is blocked)
    OPENFILE_TRUE      king/adjacent file fully open (both sides' pawns gone)
    SAFE_CHECK_AVAIL   enemy has a check to a square we do not defend (real, esp. queen — undervalued vs flat)
    PINNED_DEFENDER    a friendly zone defender is absolutely pinned (phantom safety)

Run (tonight, still no engine — this step is static):
  pyrun diagnostics/_ks_pattern_classify.py \
      --in ks_sets/collapse_dataset_classified.csv --fen-col decision_fen --color-col our_color \
      --out ks_sets/collapse_ks_patterns.csv
Then feed the tagged CSV / a per-pattern FEN filter into sf11_collapse_gap.py / _collapse_leverage.py.
Self-check without running the corpus:  pyrun diagnostics/_ks_pattern_classify.py --selftest
"""
import os
import sys
import csv
import argparse
from collections import defaultdict

import chess

# Piece groups used throughout. "Heavy" = the pieces our attacker-unit sum keys on (N/B/R/Q); pawns are
# handled separately (they are NOT attackers in our engine — a known feeder gap, so we don't count them here).
HEAVY = (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN)
SLIDERS_FILE_RANK = (chess.ROOK, chess.QUEEN)
SLIDERS_DIAG = (chess.BISHOP, chess.QUEEN)

PATTERN_NAMES = [
    "QUEENLESS_ATTACK", "PROXIMITY_OVERREAD", "QUIET_KING",
    "BATTERY", "CORNER_KING", "OPENFILE_MISREAD", "OPENFILE_TRUE",
    "SAFE_CHECK_AVAIL", "PINNED_DEFENDER",
]


def _zone(king_sq):
    """King ring (the 8 neighbours) + the king square itself — the zone our KS collects over, minus the
    forward-staging extension (kept simple; the extension is a known over-collection, not a pattern)."""
    z = chess.SquareSet(chess.BB_KING_ATTACKS[king_sq])
    z.add(king_sq)
    return z


def _zone_piece_attackers(board, king_color, zone):
    """Distinct ENEMY heavy pieces (N/B/R/Q) whose attack set touches the zone. Returns list of squares."""
    enemy = not king_color
    out = []
    for pt in HEAVY:
        for sq in board.pieces(pt, enemy):
            if board.attacks(sq) & zone:
                out.append(sq)
    return out


def _attacked_zone_count(board, king_color, zone):
    """Number of zone squares attacked by ANY enemy piece (pawns included here, matching attacked_zone)."""
    enemy = not king_color
    return sum(1 for s in zone if board.is_attacked_by(enemy, s))


def _battery(board, king_color, zone):
    """>=2 enemy sliders of compatible type sharing a line (file/rank/diagonal) with one behind the other,
    that line pointing into the king zone. Detects doubled rooks / Q+R on a file or rank and Q+B on a
    diagonal — the x-ray/battery class our masks collect as a single attacker."""
    enemy = not king_color
    zone_files = {chess.square_file(s) for s in zone}
    zone_ranks = {chess.square_rank(s) for s in zone}

    def aligned_pair(piece_types, same_key, points_at_zone):
        sqs = [sq for pt in piece_types for sq in board.pieces(pt, enemy)]
        for i in range(len(sqs)):
            for j in range(i + 1, len(sqs)):
                if same_key(sqs[i], sqs[j]) and points_at_zone(sqs[i]):
                    return True
        return False

    # File battery: two R/Q on the same file, that file crossing the zone.
    if aligned_pair(SLIDERS_FILE_RANK,
                    lambda a, b: chess.square_file(a) == chess.square_file(b),
                    lambda a: chess.square_file(a) in zone_files):
        return True
    # Rank battery.
    if aligned_pair(SLIDERS_FILE_RANK,
                    lambda a, b: chess.square_rank(a) == chess.square_rank(b),
                    lambda a: chess.square_rank(a) in zone_ranks):
        return True
    # Diagonal battery: two B/Q on the same diagonal that also aligns with the king square's diagonals.
    ksq = board.king(king_color)
    kd1, kd2 = _diag_ids(ksq)
    if aligned_pair(SLIDERS_DIAG,
                    lambda a, b: _same_diag(a, b),
                    lambda a: kd1 in _diag_ids(a) or kd2 in _diag_ids(a)):
        return True
    return False


def _diag_ids(sq):
    f, r = chess.square_file(sq), chess.square_rank(sq)
    return (f - r, f + r)  # (a1-h8 direction id, a8-h1 direction id)


def _same_diag(a, b):
    ida, idb = _diag_ids(a), _diag_ids(b)
    return ida[0] == idb[0] or ida[1] == idb[1]


def _safe_check_available(board, king_color):
    """Enemy has a checking move whose destination square WE do not defend (defended only by our king
    counts as unsafe-for-us too). Computed on a turn-flipped copy; skipped if that state is illegal."""
    enemy = not king_color
    b = board.copy(stack=False)
    b.turn = enemy
    # Flipping the turn is illegal if it leaves king_color (now NOT to move) in check — python-chess
    # exposes that as was_into_check() (the just-"moved" side's king attacked). Bail rather than iterate
    # a malformed position.
    if b.was_into_check():
        return False
    found = False
    for mv in b.legal_moves:
        if not b.gives_check(mv):
            continue
        # "Safe" for the checker = the landing square is not attacked by us, OR only by our king.
        defenders = board.attackers(king_color, mv.to_square)
        defenders.discard(board.king(king_color))
        if not defenders:
            found = True
            break
    return found


def _pinned_defender(board, king_color, zone):
    """A friendly piece that defends a zone square is absolutely pinned (so it cannot actually defend)."""
    for pt in HEAVY + (chess.PAWN,):
        for sq in board.pieces(pt, king_color):
            if board.attacks(sq) & zone and board.is_pinned(king_color, sq):
                return True
    return False


def _open_file_flags(board, king_color):
    """King file + adjacent files: classify each as our-pawn-gone×enemy-pawn state. Returns (misread, true)."""
    ksq = board.king(king_color)
    kf = chess.square_file(ksq)
    misread = truly = False
    for ff in (kf - 1, kf, kf + 1):
        if not 0 <= ff < 8:
            continue
        file_bb = chess.BB_FILES[ff]
        our_pawn = bool(board.pieces_mask(chess.PAWN, king_color) & file_bb)
        enemy_pawn = bool(board.pieces_mask(chess.PAWN, not king_color) & file_bb)
        if not our_pawn and enemy_pawn:
            misread = True
        if not our_pawn and not enemy_pawn:
            truly = True
    return misread, truly


def _corner_king(board, king_color, n_attackers):
    ksq = board.king(king_color)
    kf, kr = chess.square_file(ksq), chess.square_rank(ksq)
    home = 0 if king_color == chess.WHITE else 7
    near_home = abs(kr - home) <= 1
    on_edge_file = kf in (0, 1, 6, 7)
    return on_edge_file and near_home and n_attackers >= 1


def classify(board, king_color):
    """Return {pattern: bool} for the king of `king_color` (the king being attacked)."""
    zone = _zone(board.king(king_color))
    piece_attackers = _zone_piece_attackers(board, king_color, zone)
    n_piece_att = len(piece_attackers)
    az = _attacked_zone_count(board, king_color, zone)
    enemy_queen = bool(board.pieces(chess.QUEEN, not king_color))
    safe_check = _safe_check_available(board, king_color)
    misread, truly = _open_file_flags(board, king_color)

    f = {p: False for p in PATTERN_NAMES}
    f["QUEENLESS_ATTACK"] = (not enemy_queen) and n_piece_att >= 2
    f["PROXIMITY_OVERREAD"] = az >= 3 and n_piece_att <= 1 and not safe_check
    f["QUIET_KING"] = az <= 1
    f["BATTERY"] = _battery(board, king_color, zone)
    f["CORNER_KING"] = _corner_king(board, king_color, n_piece_att)
    f["OPENFILE_MISREAD"] = misread
    f["OPENFILE_TRUE"] = truly
    f["SAFE_CHECK_AVAIL"] = safe_check
    f["PINNED_DEFENDER"] = _pinned_defender(board, king_color, zone)
    return f


def _selftest():
    cases = [
        # (fen, king_color_to_examine, expected-present subset)
        ("6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1", chess.BLACK, {"QUIET_KING"}),
        ("3rr1k1/pp3ppp/8/8/8/8/PP3PPP/2R1R1K1 w - - 0 1", chess.WHITE, set()),  # calm, symmetric
        ("r4rk1/ppp2ppp/8/8/8/8/PPP2PPP/2RR2K1 b - - 0 1", chess.BLACK, set()),
        # Black rook can play R..a1+ along the open first rank onto an undefended square = a safe check.
        ("r3k3/8/8/8/8/8/8/4K3 b - - 0 1", chess.WHITE, {"SAFE_CHECK_AVAIL"}),
    ]
    ok = True
    for fen, kc, want in cases:
        b = chess.Board(fen)
        got = {k for k, v in classify(b, kc).items() if v}
        miss = want - got
        tag = "ok " if not miss else "MISS"
        if miss:
            ok = False
        print("%s %s  color=%s  present=%s  expected>=%s"
              % (tag, fen, "W" if kc else "B", sorted(got), sorted(want)))
    print("SELFTEST", "PASS" if ok else "FAIL (review predicates / expectations)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="", help="CSV with a FEN column (e.g. collapse_dataset_classified.csv)")
    ap.add_argument("--fen-col", default="decision_fen")
    ap.add_argument("--color-col", default="our_color", help="'white'/'black' col naming OUR side; else side-to-move")
    ap.add_argument("--out", default="", help="output CSV path (defaults to <in>.kspat.csv)")
    ap.add_argument("--emit-fens-dir", default="", help="also write one <side>_<PATTERN>.fens list per pattern "
                    "(feed each to sf11_collapse_gap.py --fens-file for per-pattern eval-loss)")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest or not args.inp:
        _selftest()
        return

    here = os.path.dirname(os.path.abspath(__file__))
    inp = args.inp if os.path.isabs(args.inp) else os.path.join(here, args.inp)
    out = args.out or (inp + ".kspat.csv")
    out = out if os.path.isabs(out) else os.path.join(here, out)

    rows_in = list(csv.DictReader(open(inp, newline="")))
    if not rows_in:
        print("no rows in %s" % inp)
        return
    extra = [f"us_{p}" for p in PATTERN_NAMES] + [f"them_{p}" for p in PATTERN_NAMES]
    fieldnames = list(rows_in[0].keys()) + extra
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        n = 0
        buckets = defaultdict(list)  # "us_PATTERN" -> [fen, ...] for --emit-fens-dir
        for r in rows_in:
            fen = r.get(args.fen_col, "").strip()
            try:
                b = chess.Board(fen)
            except ValueError:
                w.writerow(r)
                continue
            col = (r.get(args.color_col, "") or "").strip().lower()
            us = chess.WHITE if col == "white" else chess.BLACK if col == "black" else b.turn
            them = not us
            for who, kc in (("us", us), ("them", them)):
                if b.king(kc) is None:
                    continue
                for p, v in classify(b, kc).items():
                    r[f"{who}_{p}"] = int(v)
                    if v:
                        buckets[f"{who}_{p}"].append(fen)
            w.writerow(r)
            n += 1
    # counts summary (no engine, pure tally) so tonight's first look needs no extra pass
    tally = {c: 0 for c in extra}
    for r in rows_in:
        for c in extra:
            if str(r.get(c, "")) == "1":
                tally[c] += 1
    print("wrote %s  (%d rows classified)" % (out, n))
    print("pattern hit-counts (us_ / them_):")
    for p in PATTERN_NAMES:
        print("  %-18s us=%-5d them=%-5d" % (p, tally.get(f"us_{p}", 0), tally.get(f"them_{p}", 0)))

    if args.emit_fens_dir:
        fdir = args.emit_fens_dir if os.path.isabs(args.emit_fens_dir) else os.path.join(here, args.emit_fens_dir)
        os.makedirs(fdir, exist_ok=True)
        for key, fens in buckets.items():
            seen, uniq = set(), []
            for f in fens:
                if f not in seen:
                    seen.add(f)
                    uniq.append(f)
            path = os.path.join(fdir, "_kspat_%s.fens" % key)
            with open(path, "w") as fh:
                fh.write("\n".join(uniq) + ("\n" if uniq else ""))
        print("emitted %d per-pattern .fens files to %s" % (len(buckets), fdir))


if __name__ == "__main__":
    main()
