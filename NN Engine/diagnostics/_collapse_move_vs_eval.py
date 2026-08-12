# -*- coding: utf-8 -*-
"""Are our collapses EVAL errors or MOVE-CHOICE errors? They have unrelated fixes.

Every collapse measurement so far scores our EVALUATION against SF. None asks whether we actually played a
worse MOVE. A position can be mis-evaluated yet still played correctly (harmless), or evaluated fine and
played badly (a search/ordering problem, not an eval one). Until these are separated, "collapses are an eval
problem" is an assumption.

For each collapse `decision_fen` (the position where the game turned, distinct from `drop_fen`):
  - our move          : the real search, via tactical_test.run_one
  - SF18 best + score : depth-limited analyse
  - score after ours  : same depth, from the position our move reaches
  => move_loss = sf_best - sf_after_our_move   (side-to-move POV, centipawns, >= 0)
  => eval_err  = |our static eval - SF18 score| (white POV, centipawns)

Then cross-tabulates: is a big move_loss accompanied by a big eval_err (eval drove the bad move), or do they
fire independently (search/ordering picked badly despite a sound eval)?

  pyrun diagnostics/_collapse_move_vs_eval.py [TAG=vssf_2400] [LIMIT=80] [DEPTH=18] [BIG_LOSS=100]
"""
import os, sys, csv

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
for _a in sys.argv[1:]:
    if '=' in _a:
        _k, _v = _a.split('=', 1)
        os.environ[_k] = _v

THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(THIS)); sys.path.insert(0, THIS)
sys.path.insert(0, os.path.join(os.path.dirname(THIS), "selfplay"))

import chess, chess.engine
from tactical_test import run_one
from arbiter import find_stockfish

TAG = os.environ.get("TAG", "vssf_2400")
LIMIT = int(os.environ.get("LIMIT", "80"))
DEPTH = int(os.environ.get("DEPTH", "18"))
BIG_LOSS = int(os.environ.get("BIG_LOSS", "100"))     # cp; a move this much worse than best = a real mistake
# ⚠️ eval_err compares our STATIC eval to SF18's SEARCH score, so it necessarily includes tactics no static
# eval can see -- at a 100 cp threshold nearly every collapse position counts as "bad eval" (8/10 in the
# smoke run) and the cross-tab stops discriminating. 300 cp is a magnitude a classical eval could plausibly
# be expected to reach. The stricter comparison is against SF11-STATIC (the hand-reachable ceiling), which is
# what the triangulation method uses; that is the upgrade if this split turns out to matter.
BIG_EVAL = int(os.environ.get("BIG_EVAL", "300"))     # cp; static eval off by this much = a real eval error
DATA = os.path.join(THIS, "ks_sets", "collapse_dataset_classified.csv")


def cp(score, pov):
    return score.pov(pov).score(mate_score=10000)


def main():
    rows = []
    with open(DATA, newline='') as fh:
        for r in csv.DictReader(fh):
            if r.get("family") != TAG:
                continue
            fen = r.get("decision_fen") or r.get("drop_fen")
            if not fen:
                continue
            rows.append((fen, r.get("ks_class", "?")))
            if len(rows) >= LIMIT:
                break
    if not rows:
        sys.exit("no rows for family=%s" % TAG)

    from ChessAI import ChessAI
    ai = ChessAI(None, None, chess.Board(), True)
    sf = chess.engine.SimpleEngine.popen_uci(find_stockfish())
    sf.configure({"Threads": 1})   # deterministic: multi-threaded SF gives different scores run to run

    both = only_move = only_eval = neither = 0
    losses, errs, matched = [], [], 0
    n = 0
    print("%-6s %-8s %-8s %9s %9s   %s" % ("#", "ours", "sf_best", "move_loss", "eval_err", "class"))
    for fen, cls in rows:
        try:
            board = chess.Board(fen)
            if board.is_game_over():
                continue
            pov = board.turn
            info = sf.analyse(board, chess.engine.Limit(depth=DEPTH))
            sf_best_cp = cp(info["score"], pov)
            sf_move = info["pv"][0].uci() if info.get("pv") else "-"

            ours = run_one(fen, set())["uci"]
            try:
                mv = chess.Move.from_uci(ours)
            except Exception:
                continue
            if mv not in board.legal_moves:
                continue
            # Score OUR move from the SAME parent at the SAME depth via root_moves. Searching the CHILD at
            # `depth` instead would be effectively one ply deeper than the parent search, inflating every
            # loss -- it reported 64 cp for a move that matched SF's own best.
            after = cp(sf.analyse(board, chess.engine.Limit(depth=DEPTH), root_moves=[mv])["score"], pov)
            move_loss = max(0, sf_best_cp - after)

            bd = ai.ev_breakdown(board)
            our_white_cp = -bd.get("total", 0) / 10.0           # Black-positive millipawns -> white-POV cp
            sf_white_cp = sf_best_cp if pov == chess.WHITE else -sf_best_cp
            eval_err = abs(our_white_cp - sf_white_cp)
        except Exception:
            continue

        n += 1
        losses.append(move_loss); errs.append(eval_err)
        if ours == sf_move:
            matched += 1
        bad_move = move_loss >= BIG_LOSS
        bad_eval = eval_err >= BIG_EVAL
        both += bad_move and bad_eval
        only_move += bad_move and not bad_eval
        only_eval += bad_eval and not bad_move
        neither += not bad_move and not bad_eval
        print("%-6d %-8s %-8s %9d %9d   %s" % (n, ours, sf_move, move_loss, eval_err, cls))

    sf.quit()
    if not n:
        sys.exit("no positions scored")
    print("\nPOSITIONS %d   move matches SF best: %d (%.0f%%)" % (n, matched, 100.0 * matched / n))
    print("  mean move_loss %.0f cp   mean eval_err %.0f cp" % (sum(losses) / n, sum(errs) / n))
    print("\nCROSS-TAB (thresholds: move_loss>=%d, eval_err>=%d)" % (BIG_LOSS, BIG_EVAL))
    print("  bad move AND bad eval : %3d (%.0f%%)  <- eval plausibly drove the mistake" % (both, 100.0*both/n))
    print("  bad move, eval OK     : %3d (%.0f%%)  <- SEARCH/ORDERING, not eval" % (only_move, 100.0*only_move/n))
    print("  eval bad, move OK     : %3d (%.0f%%)  <- harmless misvaluation" % (only_eval, 100.0*only_eval/n))
    print("  neither               : %3d (%.0f%%)" % (neither, 100.0*neither/n))
    print("\nReading it: a large 'bad move, eval OK' share means the collapse lane is NOT primarily an eval")
    print("problem and the eval work cannot fix it. A large 'bad move AND bad eval' share supports the")
    print("current direction. A large 'eval bad, move OK' share means our eval errors are mostly harmless.")


if __name__ == '__main__':
    main()
