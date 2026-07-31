# -*- coding: utf-8 -*-
"""Understand the stage-4 leaf over-read: WHAT does our eval over-credit at the post-refutation leaf?
Walk each over-push line N plies through the refutation, then dump OUR per-term breakdown (mover/our-POV)
+ SF11 total. Aggregate mean per term so we can see which FEATURES compose the +253 vs SF11's +51 — the
ratio (material/placement vs opponent-compensation) that pushes our move selection wrong.

  overnight_runner.sh pyrun diagnostics/leaf_breakdown.py [--ply 4] [--examples 8]
"""
import os, sys, csv
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
THIS = os.path.dirname(os.path.abspath(__file__)); ENGINE = os.path.dirname(THIS)
sys.path.insert(0, THIS); sys.path.insert(0, ENGINE)
argv = sys.argv[1:]
PLY = 4; EX = 8
if "--ply" in argv: i = argv.index("--ply"); PLY = int(argv[i+1]); del argv[i:i+2]
if "--examples" in argv: i = argv.index("--examples"); EX = int(argv[i+1]); del argv[i:i+2]

import chess
from ChessAI import ChessAI
from eval_vs_sf11 import SF11Eval, SF11
REFMAP = os.path.join(THIS, "overpush_refutations.csv")
# our additive breakdown terms (Black-positive milli); convert each to our-POV cp
TERMS = ["material", "pieces", "capture_gains", "passed_pawn_support", "latent_threat", "king_safety",
         "central", "imbalance_white", "imbalance_black", "pair_bonus", "piece_value_boost",
         "advanced_endgame_white", "advanced_endgame_black", "threats"]


def main():
    rows = list(csv.DictReader(open(REFMAP)))
    seed = chess.Board(); ai = ChessAI(None, None, seed, seed.turn)
    sf11 = SF11Eval(SF11)
    acc = {t: [] for t in TERMS}; tot_our = []; tot_sf = []; ex = []
    for r in rows:
        us_white = chess.Board(r["overpush_fen"]).turn == chess.WHITE
        sgn = 1 if us_white else -1
        try:
            b = chess.Board(r["opp_fen"])
            for u in r["pv"].split()[:PLY]: b.push(chess.Move.from_uci(u))
            leaf = b.fen(); bd = ai.ev_breakdown(chess.Board(leaf))
            if bd.get("checkmate"): continue
            s11, _ = sf11.eval(leaf)
            if s11 is None: continue
        except Exception:
            continue
        our_t = (-bd["total"] / 1000.0) * 100.0 * sgn; sf_t = s11 * 100.0 * sgn
        tot_our.append(our_t); tot_sf.append(sf_t)
        for t in TERMS:
            v = bd.get(t)
            if isinstance(v, (int, float)):
                acc[t].append((-v / 1000.0) * 100.0 * sgn)   # our-POV cp; +ve = favours us
        if len(ex) < EX:
            ex.append((leaf, our_t, sf_t))
    sf11.close()
    n = len(tot_our)
    if not n: print("no leaves"); return
    print(f"[leaf-breakdown] n={n} leaves (ply {PLY})   mean OUR total {sum(tot_our)/n:+.0f}cp  SF11 {sum(tot_sf)/n:+.0f}cp\n")
    print("  our per-term mean (our-POV cp; +ve = we credit it FOR us). The big +ve terms = what we over-credit:")
    ranked = sorted(((sum(v)/len(v) if v else 0.0, t) for t, v in acc.items()), reverse=True)
    for m, t in ranked:
        if abs(m) >= 1: print(f"    {t:>22} {m:+8.0f}")
    print("\n  concrete leaves (our vs SF11 total, our-POV cp):")
    for leaf, o, s in ex:
        print(f"    our {o:+5.0f} / sf11 {s:+5.0f}   {leaf}")
    print("\n  READ: the dominant +ve terms are what our eval over-credits at a position SF11 reads ~0 =>")
    print("        the feature ratio to fix (credit these LESS relative to opponent compensation, WITHOUT")
    print("        a volatile per-leaf gate).")


if __name__ == "__main__":
    main()
