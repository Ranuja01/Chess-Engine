# -*- coding: utf-8 -*-
"""Break the piece-activity over-read (+6.68) into OUR sub-terms, on positions where SF11-static is accurate but
we over-read. Shows whether it's the ACTIVITY aggregate ('pieces', fed by attack maps) or PSQT PLACEMENT (pt_*)
or CENTRAL control -> points to the exact knob (SCALE_ATTACK_LAYER vs SCALE_PLACE_* vs CENTER_*_MULT).
All WHITE-POV pawns, toward-our-optimism. Run: pyrun diagnostics/overread_piece_subterms.py"""
import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'; os.environ.setdefault('KS_SAFE_CHECK_DEF', '5')
THIS = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(THIS)); sys.path.insert(0, THIS)
import csv, chess
from ChessAI import ChessAI
ai = ChessAI(None, None, chess.Board(), True)
BANK = os.path.join(THIS, "ks_sets", "position_bank.csv")
SUB = ["pieces", "pt_knights", "pt_bishops", "pt_rooks", "pt_queens", "det_central", "central", "material", "br_kaufman"]
from collections import defaultdict
acc = defaultdict(list); n = 0
for r in [x for x in csv.DictReader(open(BANK)) if x.get("sf18", "") not in ("", None)]:
    try:
        our_total = float(r["our_total"]); sf18 = float(r["sf18"]); sf11_total = float(r["sf11_total"])
    except Exception:
        continue
    ov = (our_total - sf18) if our_total >= 0 else (sf18 - our_total)
    if not (abs(sf11_total - sf18) < 1.0 and ov >= 1.5):          # SF11 accurate, we over-read
        continue
    try:
        bd = ai.ev_breakdown(chess.Board(r["fen"]))
    except Exception:
        continue
    s = 1 if our_total >= 0 else -1
    for k in SUB:
        acc[k].append(s * (-bd.get(k, 0.0) / 1000.0))            # white-POV pawns, toward our optimism
    n += 1
def mean(x): return sum(x)/len(x) if x else 0.0
print("SF11-accurate / we-over-read positions: n=%d" % n)
print("OUR sub-term contribution toward our optimism (WHITE-POV pawns):")
for k in sorted(SUB, key=lambda k: -mean(acc[k])):
    print("  %-12s %+.2f" % (k, mean(acc[k])))
