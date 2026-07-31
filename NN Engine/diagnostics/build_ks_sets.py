# -*- coding: utf-8 -*-
"""Build the king-safety validation sets for the surgical-activation experiment. Three files:
  danger.txt        - collapse decision positions where SF11 sees real king danger (KS<=-1.5, our POV).
                      KS MUST fire here.
  control_calm.txt  - non-collapse midgame positions with a SAFE king (SF11 |KS| < 0.4). KS MUST stay ~0.
  control_eg.txt    - ENDGAME positions (few pieces), where an exposed king is normal/good. KS MUST stay ~0
                      via the phase taper. This is the category that catches taper-over-fire = the prior
                      regression mode.
Each line: 'sf11_ks<TAB>fen'. SF11 KS is our-POV pawns (negative = our king in danger).

Run: pyrun diagnostics/build_ks_sets.py --tags sfelo2400_base200,mediocre_mine [--n-control 120 --n-eg 80]
"""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
import sys
import csv
import glob
import json
import random
import argparse

import chess
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_vs_sf11 import SF11Eval, SF11

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(THIS_DIR, "ks_sets")


def non_king_pieces(b):
    return chess.popcount(b.occupied & ~b.kings & ~b.pawns)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", required=True)
    ap.add_argument("--n-control", type=int, default=120)
    ap.add_argument("--n-eg", type=int, default=80)
    ap.add_argument("--seed", type=int, default=5)
    args = ap.parse_args()
    tags = [t.strip() for t in args.tags.split(",") if t.strip()]
    rng = random.Random(args.seed)
    os.makedirs(OUT_DIR, exist_ok=True)

    # collapse decision fens (danger candidates)
    decision = []
    for tag in tags:
        cp = os.path.join(THIS_DIR, "..", "selfplay", "games", tag, "collapses.csv")
        if os.path.exists(cp):
            for r in csv.DictReader(open(cp)):
                if r.get("decision_fen"):
                    decision.append(r["decision_fen"])

    # non-collapse sampled game positions (control candidates) from jsonls
    game_pos = []
    for tag in tags:
        d = os.path.join(THIS_DIR, "..", "selfplay", "games", tag)
        for jf in sorted(glob.glob(os.path.join(d, "game_*.jsonl")))[:120]:
            recs = [json.loads(l) for l in open(jf) if l.strip()]
            fens = [rc["fen"] for rc in recs if rc.get("fen") and not rc.get("opening")]
            game_pos.extend(fens[8:-6:4])
    rng.shuffle(game_pos)

    sf = SF11Eval(SF11)
    danger, calm, eg = [], [], []
    seen = set()
    try:
        for fen in decision:
            if fen in seen:
                continue
            seen.add(fen)
            try:
                b = chess.Board(fen)
            except ValueError:
                continue
            if b.is_game_over() or b.is_check():
                continue
            povsign = 1.0 if b.turn == chess.WHITE else -1.0
            tot, terms = sf.eval(fen)
            if tot is None:
                continue
            ks = terms.get("King safety", 0.0) * povsign
            if ks <= -1.5:
                danger.append((ks, fen))
        for fen in game_pos:
            if len(calm) >= args.n_control and len(eg) >= args.n_eg:
                break
            if fen in seen:
                continue
            seen.add(fen)
            try:
                b = chess.Board(fen)
            except ValueError:
                continue
            if b.is_game_over() or b.is_check():
                continue
            povsign = 1.0 if b.turn == chess.WHITE else -1.0
            tot, terms = sf.eval(fen)
            if tot is None:
                continue
            ks = terms.get("King safety", 0.0) * povsign
            npc = non_king_pieces(b)
            if npc <= 6 and len(eg) < args.n_eg:                 # endgame: few pieces
                eg.append((ks, fen))
            elif npc >= 10 and abs(ks) < 0.4 and len(calm) < args.n_control:   # calm midgame, safe king
                calm.append((ks, fen))
    finally:
        sf.close()

    for name, rows in (("danger", danger), ("control_calm", calm), ("control_eg", eg)):
        p = os.path.join(OUT_DIR, name + ".txt")
        with open(p, "w") as fh:
            for ks, fen in rows:
                fh.write("%.2f\t%s\n" % (ks, fen))
        print("%-14s %4d -> %s" % (name, len(rows), p))
    if danger:
        import statistics
        print("danger SF11 KS: mean=%.2f min=%.2f" % (statistics.mean(k for k, _ in danger), min(k for k, _ in danger)))


if __name__ == "__main__":
    main()
