# -*- coding: utf-8 -*-
"""WHERE do the passers our detector misses actually live? Zero engine, zero Stockfish calls.

`passer_detector_diff.py` counts the ours-vs-SF15.1 predicate disagreement but says nothing about whether
the missed pawns sit in positions where a move decision is at stake. This joins the same predicate against
the cached SF18 multi-PV of a regret set so the miss can be conditioned on CRITICALITY (best-vs-2nd win%
gap) -- the split that has repeatedly shown an aggregate hiding the signal.

Reported per bucket: share of positions holding at least one missed passer, split by OWNER (side to move
vs opponent -- the defensive/enemy-passer side was never built), and whether SF's best move INTERACTS with
the missed pawn (its square, its push path, or its stop square). Interaction is a prior on move relevance,
not proof: a detector that only fires where the truth move is elsewhere cannot change our play.

  pyrun diagnostics/_passer_miss_profile.py [SET=ks_sets/game_regret_set.csv] [MAX_POS=15000]
"""
import os, sys, csv, math
from collections import defaultdict

for _a in sys.argv[1:]:
    if '=' in _a:
        _k, _v = _a.split('=', 1)
        os.environ[_k] = _v

THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(THIS)); sys.path.insert(0, THIS)
import chess
from passer_detector_diff import analyse

SET = os.environ.get("SET", "ks_sets/game_regret_set.csv")
if not os.path.isabs(SET):
    SET = os.path.join(THIS, SET)
MAX_POS = int(os.environ.get("MAX_POS", "15000"))

CB = [(0.0, 3.0, "benign  <3%"), (3.0, 8.0, "minor  3-8%"), (8.0, 20.0, "moder 8-20%"), (20.0, 1e9, "CRIT   >20%")]


def _winpct(cp):
    return 50.0 + 50.0 * (2.0 / (1.0 + math.exp(-0.00368208 * cp)) - 1.0)


def missed_pawns(board, colour):
    """Squares of `colour` pawns SF15.1 calls passed and our predicate does not."""
    out = []
    us = list(board.pieces(chess.PAWN, colour))
    res = analyse(board, colour)
    for sq, (rank_owner, ours, sf) in zip(us, res):
        if sf and not ours:
            out.append((sq, rank_owner))
    return out


def touch_squares(board, sq, colour):
    """The missed pawn's own square, its forward path, and its stop square."""
    up = 8 if colour == chess.WHITE else -8
    sqs = {sq}
    cur = sq + up
    while 0 <= cur <= 63:
        sqs.add(cur)
        cur += up
    return sqs


def main():
    rows = list(csv.DictReader(open(SET, newline="")))[:MAX_POS]
    npos = 0
    with_miss = 0
    by_crit = defaultdict(lambda: [0, 0])          # bucket -> [positions, positions with a miss]
    owner = defaultdict(int)                        # 'stm' / 'opp' -> positions
    interact = [0, 0]                               # [positions with a miss, best move touches a missed pawn]
    interact_crit = [0, 0]
    control = [0, 0]                                # same test against pawns we ALREADY detect
    control = [0, 0]                                # same test against a pawn we ALREADY detect as passed
    rank_hist = defaultdict(int)

    for r in rows:
        fen = r.get("fen", "")
        try:
            b = chess.Board(fen)
            best_uci = r["best_uci"]
        except Exception:
            continue
        mm = []
        for pair in (r.get("moves") or "").split(";"):
            if ":" in pair:
                u, c = pair.rsplit(":", 1)
                try:
                    mm.append((u, float(c)))
                except ValueError:
                    pass
        if len(mm) < 2:
            continue
        npos += 1
        crit = abs(_winpct(mm[0][1]) - _winpct(mm[1][1]))
        lab = next(l for lo, hi, l in CB if lo <= crit < hi)
        by_crit[lab][0] += 1

        # CONTROL: pawns BOTH detectors already call passed -- if the interaction rate for missed pawns is
        # no higher than this, "the best move touches its path" is a property of passers in general, not of
        # the pawns our detector is blind to.
        ctl = []
        for col in (b.turn, not b.turn):
            for sq, (rk, o, sf) in zip(list(b.pieces(chess.PAWN, col)), analyse(b, col)):
                if o and sf:
                    ctl.append((sq, col))
        if ctl:
            ct = set()
            for sq, col in ctl:
                ct |= touch_squares(b, sq, col)
            try:
                cbm = chess.Move.from_uci(best_uci)
            except Exception:
                cbm = None
            if cbm is not None:
                control[0] += 1
                control[1] += 1 if (cbm.from_square in ct or cbm.to_square in ct) else 0

        stm_miss = missed_pawns(b, b.turn)
        opp_miss = missed_pawns(b, not b.turn)
        if not (stm_miss or opp_miss):
            continue

        with_miss += 1
        by_crit[lab][1] += 1
        if stm_miss:
            owner["stm"] += 1
        if opp_miss:
            owner["opp"] += 1
        if stm_miss and opp_miss:
            owner["both"] += 1
        for _, rk in stm_miss + opp_miss:
            rank_hist[rk] += 1

        touched = set()
        for sq, _ in stm_miss:
            touched |= touch_squares(b, sq, b.turn)
        for sq, _ in opp_miss:
            touched |= touch_squares(b, sq, not b.turn)
        try:
            bm = chess.Move.from_uci(best_uci)
        except Exception:
            continue
        hit = bm.from_square in touched or bm.to_square in touched
        interact[0] += 1
        interact[1] += 1 if hit else 0
        if crit >= 8.0:
            interact_crit[0] += 1
            interact_crit[1] += 1 if hit else 0

    print("Missed-passer PROFILE  (%s, %d scored positions)\n" % (os.path.basename(SET), npos))
    print("  positions holding >=1 passer SF sees and we do not: %d  (%.1f%%)\n"
          % (with_miss, 100.0 * with_miss / npos if npos else 0))

    print("  BY CRITICALITY (SF best-vs-2nd win% gap)")
    print("  %-14s %8s %10s %10s" % ("bucket", "pos", "with miss", "rate"))
    for _, _, lab in CB:
        tot, mis = by_crit[lab]
        print("  %-14s %8d %10d %9.1f%%" % (lab, tot, mis, 100.0 * mis / tot if tot else 0))

    print("\n  BY OWNER (of the missed pawn)")
    for k in ("stm", "opp", "both"):
        print("    %-6s %6d  (%.1f%% of miss positions)"
              % (k, owner[k], 100.0 * owner[k] / with_miss if with_miss else 0))

    print("\n  BY RANK (missed pawns, owner-relative)")
    for rk in sorted(rank_hist):
        print("    rank %d  %6d" % (rk, rank_hist[rk]))

    print("\n  SF BEST MOVE INTERACTS with a missed pawn (its square or forward path)")
    print("    all miss positions   %6d / %6d  = %.1f%%"
          % (interact[1], interact[0], 100.0 * interact[1] / interact[0] if interact[0] else 0))
    print("    crit >=8%% only       %6d / %6d  = %.1f%%"
          % (interact_crit[1], interact_crit[0], 100.0 * interact_crit[1] / interact_crit[0] if interact_crit[0] else 0))
    print("    CONTROL (already-detected passers) %6d / %6d  = %.1f%%"
          % (control[1], control[0], 100.0 * control[1] / control[0] if control[0] else 0))
    print("\n  A high interaction rate is a PRIOR on move relevance, not evidence of Elo.")
    print("  Read it AGAINST the control: only an EXCESS over that rate says the missed pawns are special.")


if __name__ == "__main__":
    main()
