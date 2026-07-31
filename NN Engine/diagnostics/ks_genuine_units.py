# -*- coding: utf-8 -*-
"""Decide re-weight-harder vs detection-gap: for the SF18-confirmed genuine KS gaps, dump the RAW attack UNITS
(KS_FLOOR=0 so nothing is zeroed) for the king SF18 says is unsafe. HIGH units (10-20) that were merely floored
=> re-weighting works. LOW units (2-6) even with a real attack => our DETECTOR under-fires => need detection
work, not just weights."""
import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ.setdefault('KS_SAFE_CHECK_DEF', '5')
os.environ['KS_FLOOR'] = '0'          # no deadzone -> raw units visible
for _a in sys.argv[1:]:               # allow ENABLE_KS_AIM=1 etc. (applied before ChessAI import)
    if '=' in _a: _k, _v = _a.split('=', 1); os.environ[_k] = _v
THIS = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(THIS)); sys.path.insert(0, THIS)
import chess, chess.engine
from ChessAI import ChessAI
sys.path.insert(0, os.path.join(os.path.dirname(THIS), "selfplay"))
from arbiter import find_stockfish

fens = [ln.strip() for ln in open(os.path.join(THIS, "ks_sets", "ks_underread_sf18.txt")) if ln.strip()]
ai = ChessAI(None, None, chess.Board(), True)
sf18 = chess.engine.SimpleEngine.popen_uci(find_stockfish())
def s18(fen):
    b = chess.Board(fen); i = sf18.analyse(b, chess.engine.Limit(depth=18)); s = i["score"].white()
    return 99 if (s.is_mate() and s.mate() > 0) else (-99 if s.is_mate() else s.score() / 100.0)

print("%-6s %-7s %-7s %-9s  fen" % ("SF18", "uW", "uB", "unsafe_u"))
hi = lo = 0
for fen in fens:
    try:
        bd = ai.ev_breakdown(chess.Board(fen))
        uw, ub = bd.get("det_ks_units_w"), bd.get("det_ks_units_b")
        e = s18(fen)
    except Exception:
        continue
    # SF18>0 => White better => BLACK king unsafe => the relevant units are on Black (uB). And vice-versa.
    unsafe_u = ub if e > 0 else uw
    tag = "HIGH" if (unsafe_u or 0) >= 10 else "low"
    if tag == "HIGH": hi += 1
    else: lo += 1
    print("%+6.2f %-7s %-7s %-9s  %s" % (e, str(uw), str(ub), "%s(%s)" % (unsafe_u, tag), fen))
sf18.quit()
print("\nunsafe-king units >=10 (floored, re-weightable): %d   <10 (detection under-fire): %d" % (hi, lo))
