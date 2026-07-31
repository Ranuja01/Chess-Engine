# -*- coding: utf-8 -*-
"""Lane-2 instrument: depth-at-fixed-time and eval NPS on a stratified midgame FEN set. The eval-for-depth
thesis needs a SPEED metric the fixed-node gates (wac/node_ab) can't see. Two modes (set via env passed by
the caller; reuses tactical_test.run_one which returns {depth,nodes,time}):
  - DEPTH@TIME:  PRESET=LIGHTNING (~1s hard cap) => median DEPTH reached at ~1s. A faster eval reaches deeper.
  - NPS@DEPTH:   MAX_DEPTH=D PRESET=LONG_FORMAT => fixed depth, median NPS = nodes/time (raw eval speed).
Always USE_OPENING_BOOK=0 (FEN boards otherwise query the book -> 'Book Move', no search).

  overnight_runner.sh pyrun diagnostics/depth_nps_bench.py PRESET=LIGHTNING USE_OPENING_BOOK=0            # depth@~1s
  overnight_runner.sh pyrun diagnostics/depth_nps_bench.py MAX_DEPTH=12 PRESET=LONG_FORMAT USE_OPENING_BOOK=0  # NPS@d12
Optional: --n 100 (sample size), --corpus <csv with fen col>.
"""
import os, sys, csv, statistics, random
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
THIS = os.path.dirname(os.path.abspath(__file__)); ENGINE = os.path.dirname(THIS)
sys.path.insert(0, THIS); sys.path.insert(0, ENGINE)
for _kv in [a for a in sys.argv[1:] if "=" in a and not a.startswith("-") and not a.endswith(".csv")]:
    _k, _v = _kv.split("=", 1); os.environ[_k] = _v
argv = [a for a in sys.argv[1:] if not ("=" in a and not a.startswith("-") and not a.endswith(".csv"))]
N = 100
if "--n" in argv: i = argv.index("--n"); N = int(argv[i+1]); del argv[i:i+2]
SEED = 1234
if "--seed" in argv: i = argv.index("--seed"); SEED = int(argv[i+1]); del argv[i:i+2]
corpus = next((a for a in argv if a.endswith(".csv")), os.path.join(ENGINE, "selfplay", "tune_data", "cploss_corpus.csv"))

import chess
from tactical_test import run_one


def main():
    # midgame FENs (skip sts tactical shots; they're not representative of game speed). Fixed-seed RANDOM
    # sample rather than the first N rows: the corpus is ordered by how it was built, so a head slice is a
    # biased view of the position mix. The seed keeps every arm on IDENTICAL positions, which is what makes
    # arms comparable -- a fresh sample per arm would silently compare different work.
    rows = [r for r in csv.DictReader(open(corpus)) if r.get("stratum") in ("game", "neutral", "collapse")]
    all_fens = [r["fen"] for r in rows]
    fens = all_fens if N >= len(all_fens) else random.Random(SEED).sample(all_fens, N)
    mode = f"PRESET={os.environ.get('PRESET','?')} MAX_DEPTH={os.environ.get('MAX_DEPTH','-')} NODE_LIMIT={os.environ.get('NODE_LIMIT','-')}"
    print(f"[depth_nps] {mode}  book={os.environ.get('USE_OPENING_BOOK','?')}  n={len(fens)}")
    depths = []; npss = []; nodes_l = []; times = []; booked = 0
    for fen in fens:
        try:
            r = run_one(fen, set())
        except Exception:
            continue
        if r.get("booked"):
            booked += 1; continue
        d = r.get("depth"); nd = r.get("nodes"); t = r.get("time")
        if d: depths.append(d)
        if nd and t and t > 0:
            npss.append(nd / t); nodes_l.append(nd); times.append(t)
    def med(v): return statistics.median(v) if v else 0
    print(f"  positions scored={len(depths)}  booked-skipped={booked}")
    if depths:
        print(f"  DEPTH reached: median {med(depths):.0f}  mean {statistics.mean(depths):.1f}  min {min(depths)} max {max(depths)}")
    if npss:
        print(f"  NPS: median {med(npss):,.0f}  mean {statistics.mean(npss):,.0f}   (median nodes {med(nodes_l):,.0f}, median time {med(times):.2f}s)")
    print("  READ: depth@~1s = the eval-for-depth metric (faster eval -> deeper); NPS = raw eval speed.")
    print("        A speed change is a WIN if it RAISES median NPS / depth@time with byte-identical decisions.")


if __name__ == "__main__":
    main()
