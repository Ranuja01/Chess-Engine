# -*- coding: utf-8 -*-
"""Step 2 of the low-depth-search tuner: build a phase-stratified FEN set and cache SF-best ONCE.

SF-best is config-independent, so we label it a single time and cache it; the tuner then only re-runs OUR
search per candidate (the whole efficiency win). The set is drawn from `diverse_corpus_wide.csv`, stratified
across phase buckets, and EXCLUDES any FEN in the move-match validation sets (_mp_target / _mp_holdout) so
tuning and validation stay disjoint (no train/test leakage).

  pyrun diagnostics/_build_lowdepth_set.py [N=500] [SF_DEPTH=16]
        [OUT=ks_sets/lowdepth_tuneset.csv]   (needs STOCKFISH_PATH -> native-ELF Linux Stockfish)

Resumable: re-running skips FENs already in OUT, so a kill costs at most one position.
"""
import os, sys, csv, atexit
from collections import defaultdict

for _a in sys.argv[1:]:
    if '=' in _a:
        _k, _v = _a.split('=', 1)
        os.environ[_k] = _v

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
THIS = os.path.dirname(os.path.abspath(__file__))
ENGINE = os.path.dirname(THIS)
sys.path.insert(0, ENGINE); sys.path.insert(0, THIS)
sys.path.insert(0, os.path.join(ENGINE, "selfplay"))

import chess
from arbiter import Arbiter, find_stockfish

N = int(os.environ.get("N", "500"))
SF_DEPTH = int(os.environ.get("SF_DEPTH", "16"))
OUT = os.environ.get("OUT", "ks_sets/lowdepth_tuneset.csv")
if not os.path.isabs(OUT):
    OUT = os.path.join(THIS, OUT)
WIDE = os.path.join(THIS, "ks_sets", "diverse_corpus_wide.csv")

# Exclude the move-match validation FENs (column fen_start in those files).
exclude = set()
for name in ("_mp_target.csv", "_mp_holdout.csv"):
    p = os.path.join(THIS, name)
    if os.path.exists(p):
        for r in csv.DictReader(open(p, newline="")):
            f = (r.get("fen_start") or r.get("fen") or "").strip()
            if f:
                exclude.add(f)

# Bucket the corpus by phase, then take an even slice from each bucket -> ~N stratified, disjoint from val.
by_phase = defaultdict(list)
for r in csv.DictReader(open(WIDE, newline="")):
    f = (r.get("fen") or "").strip()
    if f and f not in exclude:
        by_phase[r.get("phase_bucket", "?")].append(f)

phases = sorted(by_phase)
per = max(1, N // max(1, len(phases)))
picked = []
for ph in phases:
    lst = by_phase[ph]
    step = max(1, len(lst) // per)
    picked += [(f, ph) for f in lst[::step][:per]]
picked = picked[:N]

# Resume.
done = {}
if os.path.exists(OUT):
    for r in csv.DictReader(open(OUT, newline="")):
        if r.get("fen"):
            done[r["fen"]] = r

FIELDS = ["fen", "sf_best", "sf_cp", "phase_bucket"]
out_rows = list(done.values())


def flush():
    tmp = OUT + ".tmp"
    with open(tmp, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for r in out_rows:
            w.writerow({k: r.get(k, "") for k in FIELDS})
    os.replace(tmp, OUT)


atexit.register(flush)

sfpath = os.environ.get("STOCKFISH_PATH") or find_stockfish()
arb = Arbiter(sfpath, depth=SF_DEPTH)     # fixed depth => reproducible SF-best
print("  %d picked (%d phases, ~%d each), %d already done  SF depth %d -> %s"
      % (len(picked), len(phases), per, len(done), SF_DEPTH, os.path.basename(OUT)), flush=True)

n_new = 0
for i, (fen, ph) in enumerate(picked, 1):
    if fen in done:
        continue
    try:
        cp, best, _ = arb.evaluate(chess.Board(fen))
    except Exception as e:
        print("  [%4d/%4d] SKIP %s" % (i, len(picked), type(e).__name__), flush=True)
        continue
    if not best:
        continue
    out_rows.append({"fen": fen, "sf_best": best, "sf_cp": "" if cp is None else cp, "phase_bucket": ph})
    n_new += 1
    if n_new % 50 == 0:
        flush()
        print("  [%4d/%4d] labelled %d new" % (i, len(picked), n_new), flush=True)

flush()
try:
    arb.close()
except Exception:
    pass
ph_counts = defaultdict(int)
for r in out_rows:
    ph_counts[r.get("phase_bucket", "?")] += 1
print("\n  DONE: %d total labelled -> %s" % (len(out_rows), OUT))
print("  by phase: %s" % "  ".join("%s=%d" % (k, ph_counts[k]) for k in sorted(ph_counts)))
