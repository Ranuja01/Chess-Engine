# -*- coding: utf-8 -*-
"""Profile the now-dominant 'OTHER' collapse class (DEF=5-active collapses whose OUR-king was NOT under attack
at the crash, units<13). For each such collapse's decision_fen, dump our eval breakdown terms (our-POV) and
report the mean of each -> the biggest positive mean = the dominant over-read = the NEXT class to diagnose.
Fixed DEF=5 detector for the KS/other split (non-contaminated)."""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
os.environ['KS_SAFE_CHECK_DEF'] = '5'
os.environ['KS_FLOOR'] = '0'
import sys, csv, statistics
THIS = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(THIS)); sys.path.insert(0, THIS)
import chess
from ChessAI import ChessAI
ai = ChessAI(None, None, chess.Board(), True)
GAMES = os.path.join(THIS, "..", "selfplay", "games")
TERMS = ["material", "capture_gains", "piece_value_boost", "pieces", "pt_pawns", "pt_knights", "pt_bishops",
         "pt_rooks", "pt_queens", "imbalance_white", "imbalance_black", "king_safety", "central",
         "passed_pawn_support", "latent_threat", "threats", "pawn_struct", "outpost", "mobility"]

def units(fen, oc):
    try: bd = ai.ev_breakdown(chess.Board(fen))
    except Exception: return None, None
    u = bd.get("det_ks_units_w" if oc == "white" else "det_ks_units_b")
    return u, bd

acc = {t: [] for t in TERMS}
tot = []
n_other = n_ks = 0
for tag in ["night_def5", "night_def5_s1", "night_def5_s2"]:
    cp = os.path.join(GAMES, tag, "collapses.csv")
    if not os.path.exists(cp): continue
    for r in csv.DictReader(open(cp)):
        oc = (r.get("our_color") or "").strip()
        drop = (r.get("drop_fen") or "").strip(); dec = (r.get("decision_fen") or "").strip()
        if oc not in ("white", "black") or not drop or not dec: continue
        u, _ = units(drop, oc)
        if u is None: continue
        if u >= 13:
            n_ks += 1; continue           # KS-attack class, skip
        n_other += 1
        b = chess.Board(dec); pov = 1.0 if b.turn else -1.0
        bd = ai.ev_breakdown(b)
        for t in TERMS:
            acc[t].append((-bd.get(t, 0.0) / 1000.0) * pov)   # our-POV pawns
        tot.append((-bd.get("total", 0.0) / 1000.0) * pov)

print("OTHER-class collapses profiled: %d  (KS-attack skipped: %d)" % (n_other, n_ks))
print("mean our_total (our POV) on OTHER collapses = %+.2f  (>0 = we over-valued)" % statistics.mean(tot))
print("\nterm means (our POV), sorted by over-read magnitude:")
for t, m in sorted(((t, statistics.mean(v)) for t, v in acc.items() if v), key=lambda x: -x[1]):
    print("  %-20s %+6.2f" % (t, m))
