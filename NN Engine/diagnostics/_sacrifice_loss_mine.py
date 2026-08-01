# -*- coding: utf-8 -*-
"""Hunt the SACRIFICIAL-ATTACK failure mode in stored SPRT games -- no engine, no Stockfish, file reads only.

MOD_KS_REALIZ damps king danger when the ATTACKING side is materially behind. The bank found 44 positions
where that is exactly wrong: the attacker is behind because it SACRIFICED, and the attack is real
compensation. The predicted game-level symptom is therefore: we are materially AHEAD in the opening/early
middlegame, our own eval says we are winning, and we then LOSE -- i.e. we accepted material and under-rated
the attack coming back at us.

This counts that pattern per engine label. Both arms are our engine differing by one knob, so the BASE arm
is the control: only an EXCESS rate on the candidate implicates the lever.

  pyrun diagnostics/_sacrifice_loss_mine.py <games_dir> [MAT=3] [MAXMOVE=30] [EVAL=200]
"""
import os
import sys
import json

VALUES = {'p': 1, 'n': 3, 'b': 3, 'r': 5, 'q': 9}


def material_white_pov(fen):
    """Signed material balance in pawns from the FEN board field (positive = White ahead)."""
    board = fen.split()[0]
    total = 0
    for ch in board:
        low = ch.lower()
        if low in VALUES:
            total += VALUES[low] if ch.isupper() else -VALUES[low]
    return total


def main():
    args = [a for a in sys.argv[1:]]
    gdir = args[0] if args and '=' not in args[0] else None
    opts = dict(a.split('=', 1) for a in args if '=' in a)
    if not gdir:
        print(__doc__)
        return
    mat_thresh = float(opts.get('MAT', '3'))
    max_move = int(opts.get('MAXMOVE', '30'))
    eval_thresh = float(opts.get('EVAL', '200'))
    persist = int(opts.get('PERSIST', '4'))   # consecutive own-side positions the edge must hold

    stats = {}          # label -> dict of counters
    flagged = []
    for name in sorted(os.listdir(gdir)):
        path = os.path.join(gdir, name, 'game.jsonl')
        if not os.path.isfile(path):
            continue
        meta = result = None
        moves = []
        with open(path) as fh:
            for line in fh:
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                if rec.get('type') == 'meta':
                    meta = rec
                elif rec.get('type') == 'move':
                    moves.append(rec)
                elif rec.get('type') == 'result':
                    result = rec
        if not meta or not result:
            continue

        white, black = meta.get('white'), meta.get('black')
        res = result.get('result') or result.get('res') or ''
        if res == '1-0':
            loser, winner = black, white
        elif res == '0-1':
            loser, winner = white, black
        else:
            for lbl in (white, black):
                stats.setdefault(lbl, {'games': 0, 'losses': 0, 'sac': 0})
                stats[lbl]['games'] += 1
            continue

        for lbl in (white, black):
            stats.setdefault(lbl, {'games': 0, 'losses': 0, 'sac': 0})
            stats[lbl]['games'] += 1
        stats[loser]['losses'] += 1

        # Did the LOSER hold a clear early material edge while its own eval agreed it was winning?
        # A snapshot edge is not enough: inside a forced combination the side being mated is briefly "up"
        # a queen. Require the edge to PERSIST across `persist` consecutive recorded positions.
        loser_is_white = (loser == white)
        peak = None
        run = 0
        best_in_run = None
        for mv in moves:
            if mv.get('booked') or mv.get('opening'):
                continue
            if mv.get('move_no', 999) > max_move:
                break
            mat = material_white_pov(mv.get('fen', ''))
            ev = mv.get('eval_white_pov')
            if ev is None:
                continue
            if not loser_is_white:
                mat, ev = -mat, -ev
            if mat >= mat_thresh and ev >= eval_thresh:
                run += 1
                if best_in_run is None or mat > best_in_run[0]:
                    best_in_run = (mat, ev, mv.get('move_no'))
                if run >= persist and (peak is None or best_in_run[0] > peak[0]):
                    peak = best_in_run
            else:
                run = 0
                best_in_run = None
        if peak:
            stats[loser]['sac'] += 1
            flagged.append((name, loser, winner, peak[0], peak[1], peak[2]))

    print("Sacrificial-attack loss signature: ahead >= %.0f pawns by move %d, own eval >= %.0f, then LOST\n"
          % (mat_thresh, max_move, eval_thresh))
    print("  %-10s %7s %8s %8s   %s" % ("label", "games", "losses", "flagged", "flagged/losses"))
    for lbl, s in sorted(stats.items()):
        rate = (100.0 * s['sac'] / s['losses']) if s['losses'] else 0.0
        print("  %-10s %7d %8d %8d   %.1f%%" % (lbl, s['games'], s['losses'], s['sac'], rate))

    if flagged:
        print("\n  worst examples (game, loser, peak material, own eval, move#):")
        for row in sorted(flagged, key=lambda r: -r[3])[:12]:
            print("    %-10s loser=%-8s +%.0f pawns  eval %+.0f  move %s"
                  % (row[0], row[1], row[3], row[4], row[5]))


if __name__ == '__main__':
    main()
