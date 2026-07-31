# -*- coding: utf-8 -*-
"""Deep-dive the king-safety over-reads: pick the WORST offenders + a RANDOM sample from the collapse
decision positions where SF11 sees our king in danger (KS<=-1.5) but our king_safety term = 0. For each,
print our term breakdown, SF11's static breakdown, and SF18's eval + REFUTATION LINE (the concrete attack we
miss) + board context. Human-readable dossier for reasoning about WHAT we misjudge and WHY.

Run: pyrun diagnostics/king_safety_probe.py --tags sfelo2400_base200,mediocre_mine [--n-top 4 --n-rand 4 --sf-depth 20]
"""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
import sys
import csv
import random
import argparse

import chess
import chess.engine

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ENGINE_DIR = os.path.dirname(THIS_DIR)
sys.path.insert(0, ENGINE_DIR)
sys.path.insert(0, THIS_DIR)
from eval_vs_sf11 import SF11Eval, SF11

OUR_TERMS = ["material", "pieces", "capture_gains", "passed_pawn_support", "latent_threat", "threats",
             "king_safety", "central", "imbalance_white", "imbalance_black", "pair_bonus", "piece_value_boost",
             "pawn_majority", "pawn_struct", "outpost", "mobility",
             "pt_pawns", "pt_knights", "pt_bishops", "pt_rooks", "pt_queens", "pt_kings"]


def load_decision_fens(tags):
    out = []
    for tag in tags:
        cp = os.path.join(THIS_DIR, "..", "selfplay", "games", tag, "collapses.csv")
        if os.path.exists(cp):
            for r in csv.DictReader(open(cp)):
                if r.get("decision_fen"):
                    out.append(r["decision_fen"])
    return out


def king_attackers(b, us):
    them = not us
    ok = b.king(us)
    if ok is None:
        return 0
    f, r = chess.square_file(ok), chess.square_rank(ok)
    n = 0
    for df in (-1, 0, 1):
        for dr in (-1, 0, 1):
            nf, nr = f + df, r + dr
            if 0 <= nf < 8 and 0 <= nr < 8:
                n += len(b.attackers(them, chess.square(nf, nr)))
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", required=True)
    ap.add_argument("--n-top", type=int, default=4)
    ap.add_argument("--n-rand", type=int, default=4)
    ap.add_argument("--sf-depth", type=int, default=20)
    ap.add_argument("--seed", type=int, default=3)
    args = ap.parse_args()
    tags = [t.strip() for t in args.tags.split(",") if t.strip()]

    from ChessAI import ChessAI
    seed = chess.Board()
    ai = ChessAI(None, None, seed, seed.turn)
    sf11 = SF11Eval(SF11)
    vals = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}

    cand = []
    seen = set()
    try:
        for fen in load_decision_fens(tags):
            if fen in seen:
                continue
            seen.add(fen)
            try:
                b = chess.Board(fen)
            except ValueError:
                continue
            if b.is_game_over() or b.is_check():
                continue
            us = b.turn
            povsign = 1.0 if us == chess.WHITE else -1.0
            bd = ai.ev_breakdown(b)
            if bd.get("checkmate"):
                continue
            our_ks = (-bd.get("king_safety", 0.0) / 1000.0) * povsign
            sf_tot, sf_terms = sf11.eval(fen)
            if sf_tot is None:
                continue
            ks_sf = sf_terms.get("King safety", 0.0) * povsign
            if ks_sf <= -1.5 and abs(our_ks) < 0.5:              # king-danger + we're blind
                our_pov = {t: (-bd.get(t, 0.0) / 1000.0) * povsign for t in OUR_TERMS}
                our_tot = (-bd["total"] / 1000.0) * povsign
                cand.append({"fen": fen, "us": us, "our_tot": our_tot, "sf11_tot": sf_tot * povsign,
                             "ks_sf": ks_sf, "gap": our_tot - sf_tot * povsign, "our": our_pov,
                             "sf11": {k: v * povsign for k, v in sf_terms.items() if k != "Total"},
                             "mat": sum(vals[pt] * (len(b.pieces(pt, us)) - len(b.pieces(pt, not us))) for pt in vals),
                             "katt": king_attackers(b, us), "ksq": chess.square_name(b.king(us))})
    finally:
        sf11.close()

    cand.sort(key=lambda c: -c["gap"])
    rng = random.Random(args.seed)
    top = cand[:args.n_top]
    rest = cand[args.n_top:]
    rand = rng.sample(rest, min(args.n_rand, len(rest)))
    print("king-danger blind positions: %d total\n" % len(cand))

    sf18 = chess.engine.SimpleEngine.popen_uci(os.environ["STOCKFISH_PATH"])
    try:
        for label, group in (("WORST", top), ("RANDOM", rand)):
            for c in group:
                b = chess.Board(c["fen"])
                info = sf18.analyse(b, chess.engine.Limit(depth=args.sf_depth))
                pv = info.get("pv", [])[:10]
                san, bb = [], chess.Board(c["fen"])
                for m in pv:
                    san.append(bb.san(m)); bb.push(m)
                sc = info["score"].pov(c["us"]).score(mate_score=100000) / 100.0
                print("=" * 100)
                print("[%s] gap %+.2f | %s to move | our king %s, %d attackers, material %+d"
                      % (label, c["gap"], "White" if c["us"] else "Black", c["ksq"], c["katt"], c["mat"]))
                print("  %s" % c["fen"])
                print("  OUR eval %+.2f  |  SF11 static %+.2f (KS=%+.2f)  |  SF18 d%d %+.2f"
                      % (c["our_tot"], c["sf11_tot"], c["ks_sf"], args.sf_depth, sc))
                ot = sorted(c["our"].items(), key=lambda kv: -abs(kv[1]))
                print("  OUR terms : " + "  ".join("%s=%+.2f" % (t, v) for t, v in ot if abs(v) > 0.1))
                st = sorted(c["sf11"].items(), key=lambda kv: -abs(kv[1]))
                print("  SF11 terms: " + "  ".join("%s=%+.2f" % (t, v) for t, v in st if abs(v) > 0.1))
                print("  SF18 refutation: " + " ".join(san))
    finally:
        sf18.quit()


if __name__ == "__main__":
    main()
