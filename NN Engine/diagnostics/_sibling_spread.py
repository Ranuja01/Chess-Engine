# -*- coding: utf-8 -*-
"""Can a pawn term change which move we play at all? — the SIBLING-INVARIANCE test.

Four independent pawn-eval arms each landed inside +-30 Elo while the corpus win%-error improved every
time. The leading explanation (memory `pawn-scoring-may-be-unable-to-change-our-move`) is arithmetic
rather than statistical: at a node, most candidate moves are PIECE moves that do not change the pawn
structure, so a pawn term adds nearly the SAME value to every child. A constant added to all siblings
cannot reorder them. The eval gets more accurate in absolute terms and the ranking never moves.

This measures that directly and with no Stockfish at all. For real corpus positions, evaluate EVERY legal
child and ask two things per eval term group:

  SPREAD  max-min of the group across siblings, against the spread of `total`. A group whose spread is a
          few cp against a total spread of hundreds cannot be decisive.
  FLIP    delete the group entirely and re-take the argmax. If removing the WHOLE pawn subsystem rarely
          changes the top move, then no retuning of it can either -- deletion is the upper bound on what
          any reweighting could do.

🚨 The reading is COMPARATIVE, not absolute. `threats` is carried as the control: it is the one recent
eval change that won games (+45 Elo, shipped) off a SMALLER corpus movement than any pawn arm. If pawns
and threats show the same spread and flip rate, sibling invariance is NOT the explanation and the
hypothesis is dead. Individual terms are all small; only the contrast is informative.

⚠️ `pt_pawns` includes the ~100 cp of pawn MATERIAL, which a capture or promotion moves by a full pawn.
That is material, not pawn evaluation, and counting it would manufacture the exact spread we are testing
for. `pawn_nonmat` subtracts popcount-derived pawn material and is the load-bearing row; `pawn_all` is
printed beside it only so the size of that contamination is visible.

⚠️ This is a ONE-PLY STATIC proxy. The engine chooses by searching, so a term that cannot reorder static
children could still matter through deeper lines. What it does bound is the leaf score and the static
ordering, which is where a pawn table actually acts.

  pyrun diagnostics/_sibling_spread.py [IN=ks_sets/diverse_corpus_wide.csv] [MAX_POS=400] [SEED=0]
                                       [QUIET_ONLY=0]
"""
import os, sys, csv, random
from collections import defaultdict

for _a in sys.argv[1:]:
    if '=' in _a:
        _k, _v = _a.split('=', 1)
        os.environ[_k] = _v

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(THIS)); sys.path.insert(0, THIS)
import chess
from ChessAI import ChessAI

MAX_POS = int(os.environ.get("MAX_POS", "400"))
SEED = int(os.environ.get("SEED", "0"))
# Restrict the sibling set to quiet non-pawn moves. Off by default: the real candidate set includes
# captures and pushes, and excluding them would flatter the hypothesis by construction. Turn it on to
# read the pure "piece moves only" case, which is the mechanism's strongest form.
QUIET_ONLY = os.environ.get("QUIET_ONLY", "0") == "1"
IN = os.environ.get("IN", os.path.join(THIS, "ks_sets", "diverse_corpus_wide.csv"))
if not os.path.isabs(IN):
    IN = os.path.join(THIS, IN)

PAWN_VALUE = 1000          # engine millipawns; 1 cp = 10 units


def median(v):
    if not v:
        return float("nan")
    s = sorted(v); n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def groups(bd, board):
    """Term groups in engine millipawns, absolute Black-positive."""
    g = lambda k: bd.get(k, 0)
    pawn_all = (g("pt_pawns") + g("pawn_struct") + g("pawn_majority")
                + g("passed_pawn_support") + g("ae_passer"))
    pawn_mat = (len(board.pieces(chess.PAWN, chess.BLACK))
                - len(board.pieces(chess.PAWN, chess.WHITE))) * PAWN_VALUE
    return {
        "pawn_all":     pawn_all,
        "pawn_nonmat":  pawn_all - pawn_mat,
        "threats":      g("threats") + g("latent_threat"),
        "king_safety":  g("king_safety"),
        "capture_gains": g("capture_gains"),
        "piece_place":  (g("pt_knights") + g("pt_bishops") + g("pt_rooks")
                         + g("pt_queens") + g("pt_kings")),
    }


ORDER = ["pawn_nonmat", "pawn_all", "threats", "king_safety", "capture_gains", "piece_place"]


def main():
    rows = list(csv.DictReader(open(IN, newline="")))
    rng = random.Random(SEED)
    rng.shuffle(rows)
    ai = ChessAI(None, None, chess.Board(), True)

    spreads = defaultdict(list)          # per group: max-min across siblings, cp
    levels = defaultdict(list)           # per group: |mean| across siblings, cp
    # A flip counted at margin 0 includes ties broken between moves our own eval rates as equivalent,
    # which is a coin toss and not a decision. `regret` is how much better the displaced move was under
    # the FULL eval, and the gated rates below only count flips that moved a genuinely preferred move.
    flips = defaultdict(lambda: defaultdict(int))
    MARGINS = (0, 10, 25, 50)
    total_spreads = []
    used = 0
    sib_counts = []

    for r in rows:
        if used >= MAX_POS:
            break
        try:
            board = chess.Board(r["fen"])
        except Exception:
            continue
        if board.is_game_over(claim_draw=False):
            continue

        # Higher is better for the side to move. The engine's eval is absolute Black-positive.
        sign = 1 if board.turn == chess.BLACK else -1

        kids = []
        for mv in board.legal_moves:
            if QUIET_ONLY and (board.is_capture(mv)
                               or board.piece_type_at(mv.from_square) == chess.PAWN):
                continue
            board.push(mv)
            try:
                if board.is_game_over(claim_draw=False):
                    continue
                bd = ai.ev_breakdown(board)
                if bd.get("checkmate"):
                    continue
                kids.append((sign * bd.get("total", 0), groups(bd, board)))
            finally:
                board.pop()
        if len(kids) < 3:
            continue

        used += 1
        sib_counts.append(len(kids))
        tot = [k[0] for k in kids]
        total_spreads.append((max(tot) - min(tot)) / 10.0)
        best = max(range(len(kids)), key=lambda i: tot[i])

        for name in ORDER:
            vals = [k[1][name] for k in kids]
            spreads[name].append((max(vals) - min(vals)) / 10.0)
            levels[name].append(abs(sum(vals) / len(vals)) / 10.0)
            # Delete the group and re-take the argmax. `sign` is applied because the group is stored in
            # absolute units while `tot` is already side-to-move oriented.
            without = [tot[i] - sign * vals[i] for i in range(len(kids))]
            alt = max(range(len(kids)), key=lambda i: without[i])
            if alt != best:
                regret = (tot[best] - tot[alt]) / 10.0     # cp the displaced move was better by
                for m in MARGINS:
                    if regret >= m:
                        flips[name][m] += 1

    if not used:
        sys.exit("no positions scored from %s" % IN)

    print("SIBLING SPREAD / DELETION-FLIP  (%s)" % os.path.basename(IN))
    print("%d positions, %.0f legal children each (median), static eval only, no Stockfish.%s\n"
          % (used, median(sib_counts), "  QUIET_ONLY=1 (piece moves only)" if QUIET_ONLY else ""))
    print("  total spread across siblings: median %.0f cp\n" % median(total_spreads))
    print("  %-14s %9s %8s %8s   %s" % ("group", "spread", "share", "level",
                                        "flip rate by regret margin (cp)"))
    print("  %-14s %9s %8s %8s   %7s %7s %7s %7s"
          % ("", "med cp", "of tot", "med cp", ">0", ">=10", ">=25", ">=50"))
    mt = median(total_spreads) or 1.0
    for name in ORDER:
        ms = median(spreads[name])
        cells = "".join("%7.1f%%" % (100.0 * flips[name][m] / used) for m in MARGINS)
        print("  %-14s %9.1f %7.0f%% %8.1f  %s"
              % (name, ms, 100.0 * ms / mt, median(levels[name]), cells))

    GATE = 25
    pn, th = 100.0 * flips["pawn_nonmat"][GATE] / used, 100.0 * flips["threats"][GATE] / used
    print("\nREADING IT")
    print("  `flip rate` = share of positions where DELETING the whole group changes the top move.")
    print("  It is the CEILING on what retuning that group could do -- a reweighting is strictly")
    print("  weaker than deletion.")
    print("  `regret margin` = how much better the displaced move was under the FULL eval. A flip at")
    print("  margin 0 can be a tie-break between moves we rate as equivalent; the >=25 cp column is")
    print("  the one that represents a real change of mind.")
    print("  At >=%d cp: pawn_nonmat %.1f%% vs threats %.1f%% (control, +45 Elo shipped)."
          % (GATE, pn, th))
    if pn < th * 0.5:
        print("  => Pawn scoring reorders candidates far less often than the one term that won games.")
        print("     SIBLING INVARIANCE IS SUPPORTED: the 0-for-4 is arithmetic, not mistuning.")
    elif pn > th:
        print("  => Pawn scoring reorders MORE than threats did. Sibling invariance is REFUTED and the")
        print("     0-for-4 needs another explanation -- do not retire the lane on this hypothesis.")
    else:
        print("  => Comparable. The hypothesis is NOT supported; the two terms reorder at similar rates")
        print("     and the difference in Elo must come from somewhere else.")


if __name__ == "__main__":
    main()
