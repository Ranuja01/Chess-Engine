# -*- coding: utf-8 -*-
"""
Prune-verification harness (Step 3+4 of the prune-verification plan).

Reads the engine's [PRUNEFIRE] stderr records (emitted at RFP fires under ENABLE_PRUNE_LOG), and for each
fire runs a verification search (RFP off) to the fire's remaining depth to LABEL it WRONG or CORRECT:
  - RFP_MIN fired claiming "value stays <= alpha"  -> WRONG if the true value > alpha.
  - RFP_MAX fired claiming "value stays >= beta"   -> WRONG if the true value < beta.

Non-negamax POV is reconciled PER RECORD, automatically: the logged `seval` is in the search's fixed
(root) convention; we recompute the same static eval offline in absolute (Black-positive) units via
ChessAI.ev_breakdown, project it to side-to-move POV, and derive the convention factor
    f = +1 if sign(seval_logged) == sign(seval_stm) else -1
(skipping records where the magnitudes disagree or the eval is near 0, which we can't calibrate). The
verification search's value V (side-to-move POV) is then mapped to the logged convention as f*V and
compared to the logged alpha/beta. No hand-guessed sign.

Because engine knobs (MAX_DEPTH, ENABLE_RFP) are read once per process, run ONE remaining-depth per
invocation: pass rd as argv. A driver loops rd=1..RFP_MAX_DEPTH.

Usage (via runner pyrun):
  prune_verify.py <errfile> <rd> [--limit N] [--sanity] [--out CSV]
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['ENABLE_RFP'] = '0'          # verification search: RFP OFF
os.environ['USE_OPENING_BOOK'] = '0'    # deterministic, no book

import sys
import re
import csv
import argparse

import chess

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ENGINE_DIR = os.path.dirname(THIS_DIR)
sys.path.insert(0, ENGINE_DIR)
sys.path.insert(0, THIS_DIR)

REC = re.compile(r'id=(\S+) a=(-?\d+) b=(-?\d+) rd=(\d+) seval=(-?\d+) cheap=(-?\d+) fen=(.+)$')


def parse_fires(path, rd_want):
    out = []
    for line in open(path):
        if not line.startswith('[PRUNEFIRE]'):
            continue
        m = REC.search(line)
        if not m:
            continue
        pid, a, b, rd, seval, cheap, fen = m.groups()
        if int(rd) != rd_want:
            continue
        out.append(dict(id=pid, a=int(a), b=int(b), rd=int(rd), seval=int(seval),
                        cheap=int(cheap), fen=fen.strip()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('errfile')
    ap.add_argument('rd', type=int)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--sanity', action='store_true')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    os.environ['MAX_DEPTH'] = str(args.rd)   # verification search depth = the fire's remaining depth

    fires = parse_fires(args.errfile, args.rd)
    if args.limit:
        fires = fires[:args.limit]
    if not fires:
        print("no fires at rd=%d" % args.rd)
        return

    from ChessAI import ChessAI
    from tactical_test import run_one
    seed = chess.Board()
    ai = ChessAI(None, None, seed, seed.turn)   # for ev_breakdown (static, side_to_play-independent)

    rows = []
    n_wrong = n_ok = n_skip = 0
    for i, fr in enumerate(fires):
        try:
            board = chess.Board(fr['fen'])
            bd = ai.ev_breakdown(board)
        except Exception:
            n_skip += 1
            continue
        if bd.get('checkmate'):
            n_skip += 1
            continue
        total = bd['total']                                    # absolute, Black-positive (engine units)
        seval_stm = -total if board.turn == chess.WHITE else total   # side-to-move POV
        # convention factor between the logged (root) POV and stm POV, via the static eval's sign,
        # guarded by a magnitude agreement (same position's full static eval, should match ~).
        if abs(fr['seval']) < 50 or abs(seval_stm) < 50 or \
           abs(abs(seval_stm) - abs(fr['seval'])) > 0.25 * max(abs(fr['seval']), 1):
            n_skip += 1
            if args.sanity and i < 12:
                print("SKIP calib id=%s seval_log=%d seval_stm=%d (magnitude/zero)" %
                      (fr['id'], fr['seval'], seval_stm))
            continue
        f = 1 if (fr['seval'] > 0) == (seval_stm > 0) else -1

        try:
            res = run_one(fr['fen'], [])
        except Exception:
            n_skip += 1
            continue
        if res.get('booked') or res.get('eval') is None:
            n_skip += 1
            continue
        V_stm = res['eval']
        true_logged = f * V_stm                                # verification value in the logged convention
        if fr['id'] == 'RFP_MIN':
            wrong = true_logged > fr['a']
        else:  # RFP_MAX
            wrong = true_logged < fr['b']
        n_wrong += int(wrong); n_ok += int(not wrong)

        row = dict(id=fr['id'], rd=fr['rd'], a=fr['a'], b=fr['b'], seval=fr['seval'], cheap=fr['cheap'],
                   eval_instab=abs(fr['seval'] - fr['cheap']), f=f, V=V_stm, true_logged=true_logged,
                   wrong=int(wrong), fen=fr['fen'])
        rows.append(row)
        if args.sanity and i < 12:
            print("id=%-7s a=%-7d b=%-7d seval=%-7d cheap=%-7d instab=%-6d f=%+d V=%-7d true=%-8d -> %s" %
                  (fr['id'], fr['a'], fr['b'], fr['seval'], fr['cheap'], abs(fr['seval'] - fr['cheap']),
                   f, V_stm, true_logged, 'WRONG' if wrong else 'ok'))

    tot = n_wrong + n_ok
    print("\nrd=%d  labeled=%d  WRONG=%d (%.1f%%)  ok=%d  skipped=%d" %
          (args.rd, tot, n_wrong, 100.0 * n_wrong / tot if tot else 0, n_ok, n_skip))
    if args.out and rows:
        write_header = not os.path.exists(args.out)
        with open(args.out, 'a', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            if write_header:
                w.writeheader()
            w.writerows(rows)
        print("appended %d rows -> %s" % (len(rows), args.out))


if __name__ == '__main__':
    main()
