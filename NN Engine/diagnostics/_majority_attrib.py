# -*- coding: utf-8 -*-
"""Per-term attribution of the static our-vs-SF gap on the majority corpus. For each position computes
our static term breakdown + the gap (our_static - sf_static), then localizes WHICH of our terms drive
the divergence: (1) mean |contribution| per term over decision-relevant positions; (2) the separation
of each term between the WORST-third and BEST-third by |gap| (a term big in the worst but small in the
best is a scatter driver = a real tuning lever); (3) the worst positions with their dominant terms.
Answers "is the pawn-structure error directional/attributable, or just broad NNUE-vs-handcrafted scatter."

Usage (from NN Engine/):  /home/ranuja/anaconda3/bin/python diagnostics/_majority_attrib.py diagnostics/suites/majorities.csv
"""
import os, sys, csv
THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS)
import chess
from eval_breakdown import white_pawns, _load_engine, ADDITIVE_TERMS

DREL = 500
TERMS = ADDITIVE_TERMS + ["pt_pawns"]   # add pawn-placement specifically (subset of "pieces")
path = sys.argv[1]
rows = list(csv.DictReader(open(path)))

ChessAI = _load_engine()
seed = chess.Board()
ai = ChessAI(None, None, seed, seed.turn)

recs = []   # (abs_gap, gap_cp, {term: white_cp}, fen, our, sf)
for r in rows:
    sf_static = r.get("sf_static", "")
    if sf_static in ("", None):
        continue
    try:
        board = chess.Board(r["fen_start"])
        bd = ai.ev_breakdown(board)
        if bd.get("checkmate"):
            continue
        our = white_pawns(bd["total"])
        sf = float(sf_static) / 100.0
    except Exception:
        continue
    if abs(sf) * 100.0 >= DREL:
        continue
    gap = (our - sf) * 100.0
    terms = {k: white_pawns(bd.get(k, 0)) * 100.0 for k in TERMS}   # white-POV cp
    recs.append((abs(gap), gap, terms, r["fen_start"], our * 100, sf * 100))

n = len(recs)
if not n:
    print("no decision-relevant positions"); sys.exit()
recs.sort(key=lambda x: x[0])
third = max(1, n // 3)
best = recs[:third]            # smallest |gap|
worst = recs[-third:]          # largest |gap|

def mean_abs(group, term):
    return sum(abs(g[2][term]) for g in group) / len(group)

print(f"n_decision={n}  (worst/best thirds = {len(worst)}/{len(best)} by |gap|)")
print(f"{'term':>20}  mean|cp|_all  worst3rd  best3rd  separation(worst-best)")
seps = []
for t in TERMS:
    ma = sum(abs(g[2][t]) for g in recs) / n
    mw = mean_abs(worst, t); mb = mean_abs(best, t)
    seps.append((mw - mb, t, ma, mw, mb))
for sep, t, ma, mw, mb in sorted(seps, reverse=True):
    print(f"{t:>20}  {ma:8.1f}    {mw:7.1f}  {mb:7.1f}   {sep:+8.1f}")

print("\nworst 8 by |gap| (our_cp vs sf_cp, top-3 our terms):")
for ab, gap, terms, fen, our, sf in worst[::-1][:8]:
    top3 = sorted(terms.items(), key=lambda kv: abs(kv[1]), reverse=True)[:3]
    ts = "  ".join(f"{k}={v:+.0f}" for k, v in top3)
    print(f"  gap={gap:+5.0f}  our={our:+5.0f} sf={sf:+5.0f} | {ts}")
    print(f"     {fen}")
