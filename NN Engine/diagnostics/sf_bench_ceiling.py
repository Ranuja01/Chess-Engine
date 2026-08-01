# -*- coding: utf-8 -*-
"""Reference CEILING for our move-choice benches: score SF11 / SF15.1 (NNUE off + on) / SF18 on the SAME STS
suite, with the SAME c8/c9 scoring our sts_test uses, at a fixed shallow depth. Without this our "STS 1647/3000
(54.9%)" has no frame -- we cannot tell whether +92 is large or trivial.

Depth is matched across engines (DEPTH=10 default; SF is fast there). Uses all scorable EPD positions, so the
totals are directly comparable to `overnight_runner.sh sts` output.
  pyrun diagnostics/sf_bench_ceiling.py [DEPTH=10] [ENGINES=sf11,sf15c,sf15n,sf18] [SUITE=sts300.epd]

SUITE selects the .epd under diagnostics/suites/. Default is the full 1500-position STS; pass
SUITE=sts300.epd to score the same 300-position subset the `sts` runner sub uses, which makes the
reference ladder directly 1:1 with our routine bench instead of only approximately comparable.
"""
import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
for a in sys.argv[1:]:
    if '=' in a: k, v = a.split('=', 1); os.environ.setdefault(k, v)
DEPTH = int(os.environ.get('DEPTH', '10'))
WANT = os.environ.get('ENGINES', 'sf11,sf15c,sf15n,sf18').split(',')
THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(THIS)); sys.path.insert(0, THIS)
sys.path.insert(0, os.path.join(os.path.dirname(THIS), "selfplay"))
import chess, chess.engine
from sts_test import load_sts_epd
from arbiter import find_stockfish

SF11 = os.environ.get("SF11_BIN", "/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/stockfish_11_linux/stockfish-11-linux/Linux/stockfish_20011801_x64_bmi2")
SF15 = os.environ.get("SF15_BIN", "/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/stockfish_15_linux/stockfish_15.1_linux_x64/stockfish-ubuntu-20.04-x86-64")

ENGINES = {                                    # label -> (binary, uci options)
    "sf11":  (SF11, {}),                       # SF11: classical only (pre-NNUE)
    "sf15c": (SF15, {"Use NNUE": False}),      # SF15.1 CLASSICAL eval
    "sf15n": (SF15, {"Use NNUE": True}),       # SF15.1 NNUE
    "sf18":  (None, {}),                       # SF18 (NNUE-only) via arbiter.find_stockfish
}

STS = os.path.join(THIS, "suites", os.environ.get("SUITE", "STS1-STS15_LAN_v3.epd"))
positions = load_sts_epd(STS)
maxtotal = sum(p[2] for p in positions)
print("SF bench ceiling — STS %d positions, max %d, depth %d\n" % (len(positions), maxtotal, DEPTH))

for label in WANT:
    if label not in ENGINES:
        continue
    path, opts = ENGINES[label]
    path = path or find_stockfish()
    try:
        eng = chess.engine.SimpleEngine.popen_uci(path)
    except Exception as e:
        print("  %-6s  (unavailable: %s)" % (label, e)); continue
    for k, v in opts.items():
        try: eng.configure({k: v})
        except Exception: pass
    total = 0
    for fen, score_map, mx, theme, pid in positions:
        try:
            res = eng.play(chess.Board(fen), chess.engine.Limit(depth=DEPTH))
            total += score_map.get(res.move.uci(), 0) if res.move else 0
        except Exception:
            continue
    eng.quit()
    print("  %-6s  %5d / %d   (%.1f%%)" % (label, total, maxtotal, 100.0 * total / maxtotal))

print("\n(ours on sts300, same scoring: defaults 1629 = 54.3%%, +material-fix/passer-V3 1658 = 55.3%%,"
      "\n +MOD_KS_REALIZ=128 1746 = 58.2%%. On the full 1500: baseline 1555 = 51.8%%, de-king@50 1647 = 54.9%%.)")
