# -*- coding: utf-8 -*-
"""OvD over-read analysis: on the SF18-labeled bank, split positions into OVER-READ (our eval far more optimistic
than SF18 truth) vs ACCURATE, and measure the OvD/imbalance term's contribution in each. If the imbalance term is
a big POSITIVE contributor exactly where we over-read, then reducing IMBALANCE_SCALE directly removes over-reads.
All WHITE-POV pawns. Run: pyrun diagnostics/ovd_overread_analysis.py"""
import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ.setdefault('KS_SAFE_CHECK_DEF', '5')
THIS = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(THIS)); sys.path.insert(0, THIS)
import csv, chess
from ChessAI import ChessAI
ai = ChessAI(None, None, chess.Board(), True)
BANK = os.path.join(THIS, "ks_sets", "position_bank.csv")

rows = [r for r in csv.DictReader(open(BANK)) if r.get("sf18", "") not in ("", None)]
over_imb = []; acc_imb = []; over_read = []
for r in rows:
    try:
        our_total = float(r["our_total"]); sf18 = float(r["sf18"])
        bd = ai.ev_breakdown(chess.Board(r["fen"]))
        # imbalance term net contribution to total (black-positive millipawns -> white-POV pawns)
        imb = -(bd.get("imbalance_white", 0.0) + bd.get("imbalance_black", 0.0)) / 1000.0
    except Exception:
        continue
    # over-read from the side WE favor: our_total and sf18 same side, our_total more extreme
    ov = our_total - sf18                       # +ve = we more optimistic for White than truth
    # imbalance contribution TOWARD our optimism = imb aligned with sign(our_total)
    imb_toward = imb if our_total >= 0 else -imb
    ov_toward = ov if our_total >= 0 else -ov
    if abs(ov_toward) >= 1.5 and ov_toward > 0:   # we over-favor our side by >=1.5 pawns
        over_imb.append(imb_toward); over_read.append(ov_toward)
    elif abs(ov) < 0.75:                          # accurate (match SF18)
        acc_imb.append(imb if our_total >= 0 else -imb)

def mean(x): return sum(x) / len(x) if x else 0.0
print("OVER-READ positions (we over-favor our side by >=1.5p vs SF18): n=%d" % len(over_imb))
print("  mean over-read magnitude:        %+.2f pawns" % mean(over_read))
print("  mean OvD/imbalance contribution: %+.2f pawns  (toward our optimism)" % mean(over_imb))
print("  -> imbalance is %.0f%% of the over-read" % (100 * mean(over_imb) / mean(over_read) if mean(over_read) else 0))
print("  reducing IMBALANCE_SCALE 3->1 removes ~2/3 of it = %+.2f pawns of over-read" % (-2.0/3.0 * mean(over_imb)))
print("\nACCURATE positions (|our-SF18|<0.75): n=%d" % len(acc_imb))
print("  mean OvD/imbalance contribution: %+.2f pawns  (should be small if OvD isn't over-firing on accurate pos)" % mean(acc_imb))
