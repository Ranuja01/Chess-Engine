# -*- coding: utf-8 -*-
"""Which OUR term drives our UNIQUE over-reads? Take bank positions where SF11-static is ACCURATE (agrees SF18,
|sf11-sf18|<1.0) but WE over-read (|our-sf18|>=1.5, same side) -> statically-knowable positions where we alone
are wrong. Compare our term breakdown to SF11's term-by-term to find which we inflate: material(+kaufman),
piece-placement, king-safety, or imbalance. Tests 'is it material even with Kaufman, or something else'.
All WHITE-POV pawns. Run: pyrun diagnostics/overread_term_attribution.py"""
import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ.setdefault('KS_SAFE_CHECK_DEF', '5')
THIS = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(THIS)); sys.path.insert(0, THIS)
import csv, chess
from ChessAI import ChessAI
from eval_vs_sf11 import SF11Eval, SF11
ai = ChessAI(None, None, chess.Board(), True)
sf11 = SF11Eval(SF11)
BANK = os.path.join(THIS, "ks_sets", "position_bank.csv")

def owp(bd, k): return -bd.get(k, 0.0) / 1000.0    # our black-positive milli -> white-POV pawns

# our term -> SF11 term(s) (white-POV). our 'pieces'~placement of N/B/R/Q; material+kaufman~SF Material.
def our_terms(bd):
    return {
        "material":  owp(bd, "material") + owp(bd, "br_kaufman"),
        "pieces":    owp(bd, "pieces") + owp(bd, "pt_knights") + owp(bd, "pt_bishops") + owp(bd, "pt_rooks") + owp(bd, "pt_queens"),
        "king_safety": owp(bd, "king_safety"),
        "imbalance": owp(bd, "imbalance_white") + owp(bd, "imbalance_black"),
    }
def sf_terms(t):
    return {
        "material":  t.get("Material", 0.0),
        "pieces":    t.get("Knights", 0.0) + t.get("Bishops", 0.0) + t.get("Rooks", 0.0) + t.get("Queens", 0.0),
        "king_safety": t.get("King safety", 0.0),
        "imbalance": t.get("Imbalance", 0.0),
    }

rows = [r for r in csv.DictReader(open(BANK)) if r.get("sf18", "") not in ("", None)]
from collections import defaultdict
gap = defaultdict(list); n = 0
for r in rows:
    try:
        our_total = float(r["our_total"]); sf18 = float(r["sf18"]); sf11_total = float(r["sf11_total"])
    except Exception:
        continue
    ov = (our_total - sf18) if our_total >= 0 else (sf18 - our_total)
    sf11_acc = abs(sf11_total - sf18) < 1.0                # SF11-static is accurate here
    if not (sf11_acc and ov >= 1.5):                       # keep: SF11 right, we over-read
        continue
    try:
        bd = ai.ev_breakdown(chess.Board(r["fen"])); _, t = sf11.eval(r["fen"])
        if t is None: continue
    except Exception:
        continue
    ot = our_terms(bd); st = sf_terms(t); s = 1 if our_total >= 0 else -1
    for k in ot:
        gap[k].append(s * (ot[k] - st[k]))                 # +ve = WE inflate this term toward our optimism
    n += 1
sf11.close()

def mean(x): return sum(x)/len(x) if x else 0.0
print("positions where SF11-static is ACCURATE but WE over-read (>=1.5p): n=%d" % n)
print("per-term OUR inflation vs SF11 (WHITE-POV pawns, +ve = we over-credit toward our optimism):")
for k in sorted(gap, key=lambda k: -mean(gap[k])):
    print("  %-12s %+.2f" % (k, mean(gap[k])))
