# -*- coding: utf-8 -*-
"""Build an outcome-Texel corpus from vs-OPPONENT games (flat vs_sf format) — the data the self-play corpus
lacked (self-play outcomes are symmetric-null; vs a DIFFERENT eval the bias shows). tune_corpus.py only reads
the nested tournament format, so this reads the flat vs_sf layout. Two sources per tag (selfplay/games/<tag>/):
  - collapses.csv -> the FAILURE positions (decision_fen + drop_fen) + game result -> UPWEIGHTED tail (available
    even for runs pre-dating the results.csv logging).
  - results.csv   -> full-game sampled quiet positions + game result, weight 1 (post-fix runs only).
Row: fen, game, result_white (1/0.5/0, White-POV), weight, our_total + per-term features incl pt_* placement.
Feed to selfplay/outcome_texel.py.

Usage: pyrun selfplay/vs_opp_corpus.py --tags sfelo2400_base200,mediocre_mine,sfelo2400_mob --out selfplay/tune_data/vsopp_corpus.csv
"""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
import sys
import csv
import json
import glob
import random
import argparse

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ENGINE_DIR = os.path.dirname(THIS_DIR)
sys.path.insert(0, ENGINE_DIR)
import chess

TERMS = ["material", "pieces", "capture_gains", "passed_pawn_support", "latent_threat", "threats", "king_safety",
         "central", "imbalance_white", "imbalance_black", "pair_bonus", "piece_value_boost", "pawn_majority",
         "pawn_struct", "outpost", "mobility",
         "pt_pawns", "pt_knights", "pt_bishops", "pt_rooks", "pt_queens", "pt_kings"]
RESULT_WHITE = {"1-0": 1.0, "0-1": 0.0, "1/2-1/2": 0.5}


def load_engine():
    from ChessAI import ChessAI
    seed = chess.Board()
    return ChessAI(None, None, seed, seed.turn)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", required=True)
    ap.add_argument("--out", default=os.path.join(THIS_DIR, "tune_data", "vsopp_corpus.csv"))
    ap.add_argument("--per-game", type=int, default=8)
    ap.add_argument("--collapse-weight", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    tags = [t.strip() for t in args.tags.split(",") if t.strip()]
    ai = load_engine()

    samples = []   # (fen, game_id, result_white, weight)
    for tag in tags:
        d = os.path.join(THIS_DIR, "games", tag)
        cpath = os.path.join(d, "collapses.csv")
        if os.path.exists(cpath):
            nc = 0
            for r in csv.DictReader(open(cpath)):
                rw = RESULT_WHITE.get(r.get("result"))
                if rw is None:
                    continue
                for fk in ("decision_fen", "drop_fen"):
                    if r.get(fk):
                        samples.append((r[fk], tag + "/" + str(r.get("game")), rw, args.collapse_weight))
                        nc += 1
            print("[%s] collapse positions: %d" % (tag, nc))
        rpath = os.path.join(d, "results.csv")
        if os.path.exists(rpath):
            res = {}
            for r in csv.DictReader(open(rpath)):
                rw = RESULT_WHITE.get(r.get("result"))
                if rw is not None:
                    res[str(r.get("game"))] = rw
            ng = 0
            for jf in sorted(glob.glob(os.path.join(d, "game_*.jsonl"))):
                gid = os.path.basename(jf).replace("game_", "").replace(".jsonl", "").lstrip("0") or "0"
                rw = res.get(gid)
                if rw is None:
                    continue
                recs = [json.loads(l) for l in open(jf)]
                fens = [rc["fen"] for rc in recs if rc.get("fen") and not rc.get("opening")]
                fens = fens[:-8] if len(fens) > 12 else fens   # drop near-terminal plies
                pick = fens if len(fens) <= args.per_game else rng.sample(fens, args.per_game)
                for fen in pick:
                    samples.append((fen, tag + "/" + gid, rw, 1.0))
                ng += 1
            print("[%s] full-game rows from %d games (results.csv)" % (tag, ng))
        elif not os.path.exists(cpath):
            print("[%s] SKIP — no collapses.csv or results.csv" % tag)
    if not samples:
        print("no samples"); return
    rng.shuffle(samples)

    cols = ["fen", "game", "result_white", "weight", "phase_score", "is_endgame", "our_total"] + TERMS
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    seen = set()
    n = 0
    with open(args.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for fen, gid, rw, wt in samples:
            if fen in seen:
                continue
            seen.add(fen)
            try:
                b = chess.Board(fen)
            except ValueError:
                continue
            if b.is_check():
                continue
            bd = ai.ev_breakdown(b)
            if bd.get("checkmate"):
                continue
            w.writerow([fen, gid, rw, wt, bd["phase_score"], int(bd["is_endgame"]), bd["total"]]
                       + [bd[t] for t in TERMS])
            n += 1
            if n % 1000 == 0:
                print("  wrote", n, flush=True)
    print("wrote %d rows -> %s" % (n, args.out))


if __name__ == "__main__":
    main()
