# -*- coding: utf-8 -*-
"""Validate (no build) that MOBILITY is the missing leaf-visibility feature. For each post-refutation leaf,
compute a python mobility proxy (attacked-square edge + legal-move edge, ours - theirs) and correlate it
with our over-read gap (our_leaf - sf11_leaf). If the leaves where the OPPONENT is more mobile are exactly
the ones we over-read (gap rises as our mobility edge falls => negative correlation), mobility is confirmed
as the feature whose ABSENCE causes the over-read => build a real mobility term. If ~0 correlation, mobility
isn't it either.

  overnight_runner.sh pyrun diagnostics/mobility_proxy.py [--ply 4]
"""
import os, sys, csv
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
THIS = os.path.dirname(os.path.abspath(__file__)); ENGINE = os.path.dirname(THIS)
sys.path.insert(0, THIS); sys.path.insert(0, ENGINE)
argv = sys.argv[1:]
PLY = 4
if "--ply" in argv: i = argv.index("--ply"); PLY = int(argv[i+1]); del argv[i:i+2]
import chess
import numpy as np
from ChessAI import ChessAI
from eval_vs_sf11 import SF11Eval, SF11
REFMAP = os.path.join(THIS, "overpush_refutations.csv")


def attacked_squares(board, color):
    seen = 0
    for sq in chess.scan_forward(board.occupied_co[color]):
        seen |= int(board.attacks(sq))
    return chess.popcount(seen & ~board.occupied_co[color])   # squares we attack that aren't our own pieces


def legal_count(board, color):
    if board.turn == color:
        return board.legal_moves.count()
    b = board.copy(stack=False)
    if b.is_check(): return sum(1 for _ in b.generate_pseudo_legal_moves())  # can't null in check; approx
    b.push(chess.Move.null()); n = b.legal_moves.count(); return n


def main():
    rows = list(csv.DictReader(open(REFMAP)))
    seed = chess.Board(); ai = ChessAI(None, None, seed, seed.turn)
    sf11 = SF11Eval(SF11)
    atk_edge = []; mv_edge = []; gap = []
    for r in rows:
        us_white = chess.Board(r["overpush_fen"]).turn == chess.WHITE
        us = chess.WHITE if us_white else chess.BLACK; sgn = 1 if us_white else -1
        try:
            b = chess.Board(r["opp_fen"])
            for u in r["pv"].split()[:PLY]: b.push(chess.Move.from_uci(u))
            bd = ai.ev_breakdown(chess.Board(b.fen()))
            if bd.get("checkmate"): continue
            s11, _ = sf11.eval(b.fen())
            if s11 is None: continue
        except Exception:
            continue
        our_t = (-bd["total"] / 1000.0) * 100.0 * sgn; sf_t = s11 * 100.0 * sgn
        atk_edge.append(attacked_squares(b, us) - attacked_squares(b, not us))
        mv_edge.append(legal_count(b, us) - legal_count(b, not us))
        gap.append(our_t - sf_t)
    sf11.close()
    n = len(gap)
    if n < 5: print("too few"); return
    ae = np.array(atk_edge); me = np.array(mv_edge); g = np.array(gap)
    print(f"[mobility-proxy] n={n} leaves (ply {PLY})")
    print(f"  mean our-POV mobility edge: attacked-sq {ae.mean():+.1f}   legal-moves {me.mean():+.1f}  (- = OPPONENT more mobile)")
    print(f"  corr(attacked-sq edge, over-read gap) = {np.corrcoef(ae, g)[0,1]:+.2f}")
    print(f"  corr(legal-move edge,  over-read gap) = {np.corrcoef(me, g)[0,1]:+.2f}")
    # bucket: do low-mobility-edge leaves over-read more?
    lo = g[ae <= np.median(ae)]; hi = g[ae > np.median(ae)]
    print(f"  mean gap | our attack-edge LOW  (opp more active): {lo.mean():+.0f}cp  (n={len(lo)})")
    print(f"  mean gap | our attack-edge HIGH (we more active):  {hi.mean():+.0f}cp  (n={len(hi)})")
    print("\n  READ: strong NEGATIVE corr + LOW-edge gap >> HIGH-edge gap => mobility is the missing feature")
    print("        (we over-read exactly when the opponent out-activates us) => build a real mobility term.")


if __name__ == "__main__":
    main()
