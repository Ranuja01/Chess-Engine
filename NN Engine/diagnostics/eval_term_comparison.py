# -*- coding: utf-8 -*-
"""Systematic per-term OURS vs SF11-static comparison across the whole bank. For each aligned term reports the
MEAN MAGNITUDE we assign vs SF11 (|term|, since the eval is side-symmetric so signed means cancel), the
over/under-weight gap (mean|ours| - mean|SF11|), and the mean absolute divergence. Answers: which elements do we
over/under-read vs SF11, and by how much on average. All pawns.
CAVEAT: our pt_* = pure placement; SF11 Knights/Bishops/Rooks/Queens ALSO include mobility/outposts, so the
per-piece rows are fuzzy; Material/KingSafety/Threats/Passed/Space/Imbalance/Mobility are cleaner.
Run: pyrun diagnostics/eval_term_comparison.py [N=1852]"""
import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'; os.environ.setdefault('KS_SAFE_CHECK_DEF', '5')
for _a in sys.argv[1:]:                        # KEY=VAL args -> env (before ChessAI import)
    if '=' in _a and '/' not in _a:
        _k, _v = _a.split('=', 1); os.environ[_k] = _v
N = int(os.environ.get('N', '99999'))
THIS = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(THIS)); sys.path.insert(0, THIS)
import csv, chess
from ChessAI import ChessAI
from eval_vs_sf11 import SF11Eval, SF11
ai = ChessAI(None, None, chess.Board(), True); sf11 = SF11Eval(SF11)
BANK = os.path.join(THIS, "ks_sets", "position_bank.csv")
def owp(bd, *ks): return -sum(bd.get(k, 0.0) for k in ks) / 1000.0
PAIRS = [("Material",   ["material", "br_kaufman"], "Material"),
         ("KingSafety", ["king_safety"], "King safety"),
         ("Threats",    ["latent_threat"], "Threats"),
         ("Passed",     ["passed_pawn_support"], "Passed"),
         ("Space",      ["central", "det_central"], "Space"),
         ("Imbalance",  ["imbalance_white", "imbalance_black"], "Imbalance"),
         ("Mobility",   ["det_w_mobility", "det_b_mobility"], "Mobility"),
         ("Knights",    ["pt_knights"], "Knights"),
         ("Bishops",    ["pt_bishops"], "Bishops"),
         ("Rooks",      ["pt_rooks"], "Rooks"),
         ("Queens",     ["pt_queens"], "Queens")]
from collections import defaultdict
so = defaultdict(float); ss = defaultdict(float); sd = defaultdict(float); n = 0
for r in list(csv.DictReader(open(BANK)))[:N]:
    try:
        bd = ai.ev_breakdown(chess.Board(r["fen"])); _, t = sf11.eval(r["fen"])
        if t is None: continue
    except Exception:
        continue
    for label, oks, sl in PAIRS:
        o = owp(bd, *oks); s = t.get(sl, 0.0)
        so[label] += abs(o); ss[label] += abs(s); sd[label] += abs(o - s)
    n += 1
sf11.close()
print("per-term OURS vs SF11 over %d positions (mean |magnitude|, pawns):" % n)
print("  %-11s %8s %8s %9s %9s" % ("term", "|OURS|", "|SF11|", "over/under", "mean|diff|"))
for label, _, _ in PAIRS:
    mo, ms = so[label] / n, ss[label] / n
    print("  %-11s %8.2f %8.2f %+9.2f %9.2f" % (label, mo, ms, mo - ms, sd[label] / n))
