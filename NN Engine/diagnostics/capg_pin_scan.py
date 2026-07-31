# -*- coding: utf-8 -*-
"""Scan the SF18-labeled bank with ENABLE_CAPG_PIN=1 and diff vs the default our_total (bank column). Shows how
many positions the pin-guard changes, whether it moves them TOWARD SF18 truth (fixes) or away (regressions), and
lists example FENs both fixed and still-wrong (the latter = OTHER issues to eyeball).
Run: pyrun diagnostics/capg_pin_scan.py"""
import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'; os.environ.setdefault('KS_SAFE_CHECK_DEF', '5')
os.environ['ENABLE_CAPG_PIN'] = '1'           # BEFORE import
THIS = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(THIS)); sys.path.insert(0, THIS)
import csv, chess
from ChessAI import ChessAI
ai = ChessAI(None, None, chess.Board(), True)
BANK = os.path.join(THIS, "ks_sets", "position_bank.csv")

fixed = []; regressed = []; still = []; changed = 0
for r in [x for x in csv.DictReader(open(BANK)) if x.get("sf18", "") not in ("", None)]:
    try:
        base = float(r["our_total"]); sf18 = float(r["sf18"])       # default (pin off) our_total, truth
        pin = -ai.ev_breakdown(chess.Board(r["fen"])).get("total", 0.0) / 1000.0   # pin ON
    except Exception:
        continue
    d = pin - base
    if abs(d) >= 1.0:
        changed += 1
        err_before = abs(base - sf18); err_after = abs(pin - sf18)
        rec = (round(err_before - err_after, 2), round(base, 2), round(pin, 2), round(sf18, 2), r["fen"])
        if err_after < err_before - 0.5:   fixed.append(rec)
        elif err_after > err_before + 0.5: regressed.append(rec)
    # positions STILL badly over-read even with pin on (other issues)
    try:
        if abs(pin - sf18) >= 3.0 and (abs(base - sf18) >= 3.0):
            still.append((round(abs(pin - sf18), 2), round(pin, 2), round(sf18, 2), r["fen"]))
    except Exception:
        pass

print("positions the pin-guard CHANGES by >=1.0p: %d" % changed)
print("  FIXED (moved toward SF18): %d    REGRESSED (moved away): %d" % (len(fixed), len(regressed)))
print("\n== biggest FIXES (pin guard corrects toward SF18) ==")
for improv, base, pin, sf18, fen in sorted(fixed, reverse=True)[:6]:
    print("  base=%+.2f -> pin=%+.2f  (SF18=%+.2f, err -%.2f)  %s" % (base, pin, sf18, improv, fen))
print("\n== any REGRESSIONS (pin guard moved AWAY from SF18) ==")
for improv, base, pin, sf18, fen in sorted(regressed)[:6]:
    print("  base=%+.2f -> pin=%+.2f  (SF18=%+.2f)  %s" % (base, pin, sf18, fen))
print("\n== STILL badly wrong WITH pin on (OTHER issues to eyeball) ==")
for err, pin, sf18, fen in sorted(still, reverse=True)[:6]:
    print("  ours=%+.2f  SF18=%+.2f  (err %.2f)  %s" % (pin, sf18, err, fen))
