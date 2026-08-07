# -*- coding: utf-8 -*-
"""Shrink an offending FEN to a MINIMAL colour-asymmetry repro.

Reading code to explain an asymmetry has failed repeatedly on this project -- a fragment looks wrong,
the story is confident, and the measurement refuses it (capture-gains polarity, the phase blend, the
imbalance fields). What HAS worked is a minimal repro: the knight defect was found by shrinking to four
pieces, at which point exactly one term was left and the branch was unambiguous.

So: greedily delete pieces (kings excepted) for as long as the chosen invariant still breaks, then try
sliding the remaining pieces toward a corner. Reports the smallest position that still fails, and which
terms still break on it.

  pyrun diagnostics/_asym_minimize.py FEN="<fen>" [ASYM_TERM=det_pieceval|total|<name>]

ASYM_TERM:
  det_pieceval  the accumulator SWAP invariant  (w(b) == b(mirror), b(b) == w(mirror))   [default]
  total         plain antisymmetry of the final score
  <name>        plain antisymmetry of any single breakdown term
"""
import os, sys

for _a in sys.argv[1:]:
    if '=' in _a:
        _k, _v = _a.split('=', 1)
        os.environ[_k] = _v

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(THIS)); sys.path.insert(0, THIS)
import chess
from ChessAI import ChessAI

FEN = os.environ.get("FEN", "1r6/1P4k1/1Rp5/2Pp4/3N1r2/2P1p2b/4QP2/4KBq1 w - - 0 43")
# NB: NOT `TERM` -- that is the shell's terminal-type variable, and reading it silently made the first
# run minimise against the string "xterm-256color" (i.e. against nothing at all).
TERM = os.environ.get("ASYM_TERM", "det_pieceval")

ai = ChessAI(None, None, chess.Board(), True)


def asym(b):
    """Magnitude of the invariance violation for this board, 0 = clean."""
    try:
        db, dm = ai.ev_breakdown(b), ai.ev_breakdown(b.mirror())
    except Exception:
        return 0
    if TERM == "det_pieceval":
        return (abs(db.get("det_w_pieceval", 0) - dm.get("det_b_pieceval", 0))
                + abs(db.get("det_b_pieceval", 0) - dm.get("det_w_pieceval", 0)))
    return abs(db.get(TERM, 0) + dm.get(TERM, 0))


def legalish(b):
    """Reject positions the evaluator should never see, so we don't 'minimise' into nonsense."""
    if b.king(chess.WHITE) is None or b.king(chess.BLACK) is None:
        return False
    if b.is_game_over(claim_draw=False):
        return False
    # A side not to move must not already be in check.
    tmp = b.copy(stack=False); tmp.turn = not b.turn
    return not tmp.is_check()


def show(b, label):
    d = asym(b)
    print("  %-14s %-56s  asym=%d  pieces=%d" % (label, b.fen(), d, len(b.piece_map())))
    return d


def scan_for_worst():
    """No FEN given -> find the corpus position with the largest violation for this term.

    Saves hand-picking a start position per term, which matters because the terms do NOT break on the
    same positions: `pieces` breaks on 151 of 178, the accumulator on 16, and a FEN chosen for one is
    usually clean for the other.
    """
    import csv
    IN = os.environ.get("IN", "ks_sets/diverse_corpus_wide.csv")
    if not os.path.isabs(IN):
        IN = os.path.join(THIS, IN)
    n = int(os.environ.get("SCAN_N", "400"))
    best, best_d = None, 0
    for r in list(csv.DictReader(open(IN, newline="")))[:n]:
        try:
            b = chess.Board(r["fen"])
        except Exception:
            continue
        if b.is_game_over(claim_draw=False):
            continue
        d = asym(b)
        if d > best_d:
            best, best_d = r["fen"], d
    if best is None:
        print("  no position in the sample violates %s" % TERM)
        sys.exit(0)
    print("  scanned %d positions -> worst %s = %d mp\n" % (n, TERM, best_d))
    return best


def main():
    fen = FEN if os.environ.get("FEN") else scan_for_worst()
    b = chess.Board(fen)
    # The dispatcher splits argv on spaces, so a FEN passed as FEN=... arrives BOARD-ONLY and
    # python-chess silently defaults to white-to-move. That flipped a legal position into one with the
    # side-not-to-move in check, which made every leave-one-out row read "illegal". Pass TURN=b/w.
    if os.environ.get("TURN"):
        b.turn = (os.environ["TURN"].strip().lower() == "w")
    print("MINIMISING  term=%s\n" % TERM)
    base = show(b, "start")
    if not base:
        print("\n  (no violation on this FEN for that term -- nothing to shrink)")
        return

    # --- 1. drop pieces, keep anything that preserves MOST of the violation ---
    # ⚠️ Accepting ANY nonzero asymmetry lets the search wander into a DIFFERENT, smaller defect: a
    # 150 mp rook case once shrank to a 5 mp integer-rounding residual in an entirely different phase,
    # which is a minimal repro of the wrong bug. Require the violation to stay within MIN_FRAC of where
    # it started, so what comes out is a smaller instance of the SAME defect.
    MIN_FRAC = float(os.environ.get("MIN_FRAC", "0.5"))
    floor = base * MIN_FRAC
    changed = True
    while changed:
        changed = False
        for sq, pc in sorted(b.piece_map().items()):
            if pc.piece_type == chess.KING:
                continue
            t = b.copy(stack=False)
            t.remove_piece_at(sq)
            if legalish(t) and asym(t) >= floor:
                b = t
                changed = True
                break

    show(b, "minimal")

    # --- 2. which terms still break here? with so few pieces this is usually one line of code ---
    db, dm = ai.ev_breakdown(b), ai.ev_breakdown(b.mirror())
    SWAP = [("det_w_pieceval", "det_b_pieceval"), ("det_w_defense", "det_b_defense"),
            ("det_w_offense", "det_b_offense"), ("det_w_mobility", "det_b_mobility"),
            ("det_ks_units_w", "det_ks_units_b")]
    SKIP = {"phase_score", "is_endgame", "advanced_endgame_fired", "det_pawn_count", "det_central"}
    members = {k for p in SWAP for k in p}

    print("\n  terms still breaking on the minimal position:")
    out = []
    for wk, bk in SWAP:
        d = abs(db.get(wk, 0) - dm.get(bk, 0)) + abs(db.get(bk, 0) - dm.get(wk, 0))
        if d:
            out.append((d, "%s<->%s" % (wk, bk)))
    for k in set(db) | set(dm):
        x, y = db.get(k, 0), dm.get(k, 0)
        if not isinstance(x, int) or not isinstance(y, int) or k in SKIP or k in members:
            continue
        if abs(x + y):
            out.append((abs(x + y), k))
    for d, k in sorted(out, reverse=True):
        print("     %-34s %8d      base %+8d   mirror %+8d"
              % (k, d, db.get(k.split("<->")[0], 0), dm.get(k.split("<->")[-1], 0)))
    if not out:
        print("     (none -- the violation is only visible in the composite)")

    print("\n  phase=%s endgame=%s ae_fired=%s   turn=%s"
          % (db.get("phase_score"), db.get("is_endgame"), db.get("advanced_endgame_fired"),
             "w" if b.turn else "b"))

    # --- 2b. LEAVE-ONE-OUT: which pieces are load-bearing? ---
    # Once the repro is irreducible, the greedy pass can no longer tell us WHY. Removing each piece in
    # turn and reporting the resulting magnitude does: a piece whose removal collapses the violation is
    # part of the interaction, one that barely moves it is scenery. This is what separates "a rook bug"
    # from "a rook-plus-something bug", which single-piece probes cannot see (single rooks are clean).
    print("\n  leave-one-out (asym after removing each piece; start %d):" % asym(b))
    rows = []
    for sq, pc in sorted(b.piece_map().items()):
        if pc.piece_type == chess.KING:
            continue
        t = b.copy(stack=False)
        t.remove_piece_at(sq)
        rows.append((asym(t) if legalish(t) else None, chess.square_name(sq), pc.symbol()))
    for d, name, sym in sorted(rows, key=lambda r: (r[0] is None, r[0])):
        print("     remove %-3s %-2s -> %s" % (name, sym, "illegal" if d is None else str(d)))

    # --- 3. does it survive a side-to-move flip? tells us if the defect is turn-gated ---
    t = b.copy(stack=False); t.turn = not b.turn
    if legalish(t):
        print("  flip side-to-move:  asym=%d" % asym(t))


if __name__ == "__main__":
    main()
