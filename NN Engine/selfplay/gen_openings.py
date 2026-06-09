# -*- coding: utf-8 -*-
"""Generate a large UNBALANCED-but-sound opening book ONCE, for diverse + statistically-independent
self-play.

The shipped openings.txt has only ~30 lines. Across hundreds of games each opening repeats, and the
games from one opening are CORRELATED (shared first moves) -- so the effective independent sample size
is far below the game count and any SPRT/Elo CI is optimistic. The standard fix (what Fishtest uses) is
a large book of mildly-imbalanced starts ("UHO" -- unbalanced human openings):

  * INDEPENDENCE: hundreds-to-thousands of distinct, early-diverging lines -> games are uncorrelated.
  * DECISIVENESS: a small objective imbalance -> fewer draws -> faster SPRT convergence (more signal/game).
  * CONVERT + DEFEND in one: the tournament plays every opening from BOTH colors, so the imbalance
    cancels in aggregate (fair), while we directly measure whether an engine both converts the +side
    AND holds the -side -- the conversion/resilience axis we're weak on.

How it builds each line: take the first few plies of a random seed opening (for opening flavour), then
EXTEND with Stockfish: at each ply, branch only among moves WITHIN cp-window of the best (so every move
is objectively sound, never a blunder), choosing randomly for diversity. Keep the line only if the
FINAL position's |eval| lands in a small window (mildly imbalanced, not lost). Dedup, collect N.

Run ONCE (from NN Engine/, needs Stockfish + python-chess), e.g.:
    python selfplay/gen_openings.py --count 1000 --out selfplay/openings_uho.txt
Then point any tournament at it:
    python selfplay/tournament.py ... --openings selfplay/openings_uho.txt
(`schedule()` shuffles by --seed and takes the first games/2 openings, so the seed picks a random
subset of the big book each run.)
"""

import os
import sys
import argparse
import random

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)

import chess
import chess.engine
from arbiter import find_stockfish
from tournament import load_openings


def _cp(score, pov):
    """Side-relative centipawns from a python-chess PovScore (mate -> large sentinel)."""
    return score.pov(pov).score(mate_score=100000)


def gen(args):
    sf_path = args.sf_path or find_stockfish()
    if not sf_path:
        print("[gen] no Stockfish found (set STOCKFISH_PATH or --sf-path)", flush=True)
        return
    seeds = load_openings(args.seeds)
    if not seeds:
        print(f"[gen] no seed openings in {args.seeds}", flush=True)
        return

    rng = random.Random(args.seed)
    engine = chess.engine.SimpleEngine.popen_uci(sf_path)
    limit = chess.engine.Limit(time=args.movetime)
    out, seen = [], set()
    attempts = 0
    cap = args.count * args.max_attempts

    try:
        while len(out) < args.count and attempts < cap:
            attempts += 1
            seed_line = rng.choice(seeds)
            board = chess.Board()
            moves = []
            ok = True
            # Seed prefix: keep the opening's first few plies for flavour, then diverge early.
            for uci in seed_line[:args.prefix_plies]:
                mv = chess.Move.from_uci(uci)
                if mv not in board.legal_moves:
                    ok = False
                    break
                board.push(mv)
                moves.append(uci)
            if not ok:
                continue

            # Extend with Stockfish-verified in-window moves until we reach --plies.
            while len(moves) < args.plies and not board.is_game_over():
                info = engine.analyse(board, limit, multipv=args.multipv)
                best = _cp(info[0]["score"], board.turn)
                cands = [ln["pv"][0] for ln in info
                         if ln.get("pv") and best - _cp(ln["score"], board.turn) <= args.cp_window]
                mv = rng.choice(cands) if cands else info[0]["pv"][0]
                board.push(mv)
                moves.append(mv.uci())

            if board.is_game_over():
                continue
            final = abs(_cp(engine.analyse(board, limit)["score"], chess.WHITE))
            if args.eval_lo <= final <= args.eval_hi:
                key = " ".join(moves)
                if key not in seen:
                    seen.add(key)
                    out.append(key)
                    if len(out) % 25 == 0:
                        print(f"[gen] {len(out)}/{args.count} (attempts {attempts}, "
                              f"accept {100*len(out)/attempts:.0f}%)", flush=True)
    finally:
        engine.quit()

    with open(args.out, "w") as f:
        f.write("# UHO opening book — mildly-imbalanced, Stockfish-verified in-window lines.\n")
        f.write(f"# count={len(out)} plies={args.plies} cp_window={args.cp_window} "
                f"eval=[{args.eval_lo},{args.eval_hi}]cp seed={args.seed}\n")
        for line in out:
            f.write(line + "\n")
    print(f"[gen] wrote {len(out)} openings -> {args.out}  "
          f"({attempts} attempts, {100*len(out)/max(attempts,1):.0f}% accepted)", flush=True)


def main():
    ap = argparse.ArgumentParser(description="Generate a large unbalanced-but-sound (UHO) opening book.")
    ap.add_argument("--out", default=os.path.join(THIS_DIR, "openings_uho.txt"))
    ap.add_argument("--seeds", default=os.path.join(THIS_DIR, "openings.txt"),
                    help="seed openings to take a short prefix from (for opening flavour)")
    ap.add_argument("--count", type=int, default=1000, help="target number of openings")
    ap.add_argument("--prefix-plies", type=int, default=4,
                    help="plies of the seed to keep before diverging (4 = 2 moves; 0 = from startpos)")
    ap.add_argument("--plies", type=int, default=12, help="total opening length in plies")
    ap.add_argument("--multipv", type=int, default=5, help="SF candidate moves considered per ply")
    ap.add_argument("--cp-window", type=int, default=50,
                    help="only branch into moves within this many cp of best (keeps every move sound)")
    ap.add_argument("--eval-lo", type=int, default=15, help="min |final eval| cp (some imbalance)")
    ap.add_argument("--eval-hi", type=int, default=90, help="max |final eval| cp (not lost)")
    ap.add_argument("--movetime", type=float, default=0.12, help="SF time per analysis call")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sf-path", default=None)
    ap.add_argument("--max-attempts", type=int, default=8, help="attempts cap = count * this")
    gen(ap.parse_args())


if __name__ == "__main__":
    main()
