# -*- coding: utf-8 -*-
"""KS eval-gap dossier: the eval-vs-SF11 analysis showed KING SAFETY is the dominant contributor to our ~1.28p
eval gap. This finds the positions where SF11 assigns a large king-safety term and WE under-read it, and dumps
our KS term + unit detectors vs SF11's KS side by side -- so the next session can see WHICH KS features SF fires
on that we miss (safe-checks / weak squares / attacker count / magnitude). Diagnosis only; builds nothing.

Run: pyrun diagnostics/ks_gap_dossier.py [N]   (N default 400 STS positions)
"""
import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ.setdefault('KS_SAFE_CHECK_DEF', '5')
for _a in sys.argv[1:]:
    if '=' in _a: _k, _v = _a.split('=', 1); os.environ[_k] = _v
THIS = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(THIS)); sys.path.insert(0, THIS)
import chess
from sts_test import load_sts_epd
from eval_vs_sf11 import SF11Eval, SF11

nums = [int(a) for a in sys.argv[1:] if a.isdigit()]
N = nums[0] if nums else 400
allpos = load_sts_epd(os.path.join(THIS, "suites", "STS1-STS15_LAN_v3.epd"))[:N]

from ChessAI import ChessAI
ai = ChessAI(None, None, chess.Board(), True)
sf = SF11Eval(SF11)

rows = []
for item in allpos:
    fen = item[0] if isinstance(item, (tuple, list)) else item
    b = chess.Board(fen)
    if b.is_check():
        continue
    try:
        bd = ai.ev_breakdown(b)
        _, terms = sf.eval(fen)
    except Exception:
        continue
    sf_ks = terms.get("King safety", 0.0)              # White-POV pawns
    our_ks = -bd.get("king_safety", 0.0) / 1000.0      # White-POV pawns
    # under-read = SF sees a big KS swing, ours is <half the magnitude (same sign region)
    if abs(sf_ks) >= 1.0 and abs(our_ks) < 0.5 * abs(sf_ks):
        rows.append(dict(fen=fen, sf_ks=sf_ks, our_ks=our_ks,
                         uw=bd.get("det_ks_units_w"), ub=bd.get("det_ks_units_b"),
                         gap=abs(sf_ks) - abs(our_ks)))
sf.close()
rows.sort(key=lambda r: -r["gap"])
print("KS under-read positions (SF |KS|>=1.0 and ours <half): %d / %d" % (len(rows), N))
print("%-9s %-9s %-6s %-6s  %s" % ("SF_KS", "our_KS", "ks_uW", "ks_uB", "fen"))
for r in rows[:25]:
    print("%+9.2f %+9.2f %6s %6s  %s" % (r["sf_ks"], r["our_ks"], str(r["uw"]), str(r["ub"]), r["fen"]))
# write the corpus for later firing/verification work
out = os.path.join(THIS, "ks_sets", "ks_underread_vs_sf11.txt")
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, "w") as f:
    for r in rows: f.write(r["fen"] + "\n")
print("\ncorpus -> %s (%d fens)" % (out, len(rows)))
