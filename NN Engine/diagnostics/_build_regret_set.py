# -*- coding: utf-8 -*-
"""Label a stratified FEN set with SF18 MULTI-PV (top-K moves + their evals) at fixed depth, cached ONCE, for
the REGRET-based low-depth tuner. Regret = SF_eval(best) - SF_eval(our_move): a GRADED, move-based, Elo-aligned
objective (vs the binary top-1 match, which has no gradient so the descent can't see sub-flip improvements).

Fixed depth => deterministic multi-PV => reproducible regret targets. Disjoint from the move-match val sets.

  pyrun diagnostics/_build_regret_set.py [N=1000] [K=8] [SF_DEPTH=14]
        [OUT=ks_sets/regret_set.csv]   (needs STOCKFISH_PATH -> native-ELF Linux Stockfish)

Output columns: fen, phase_bucket, best_uci, best_cp, moves   where `moves` = "uci:cp;uci:cp;..." (top-K,
best-first, White-POV cp). Resumable: re-running skips FENs already in OUT.
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
import chess.engine
from arbiter import find_stockfish

N = int(os.environ.get("N", "1000"))
K = int(os.environ.get("K", "8"))
SF_DEPTH = int(os.environ.get("SF_DEPTH", "14"))
OUT = os.environ.get("OUT", "ks_sets/regret_set.csv")
if not os.path.isabs(OUT):
    OUT = os.path.join(THIS, OUT)
WIDE = os.path.join(THIS, "ks_sets", "diverse_corpus_wide.csv")

# Exclude the move-match validation FENs (disjoint tuning vs validation).
exclude = set()
for name in ("_mp_target.csv", "_mp_holdout.csv"):
    p = os.path.join(THIS, name)
    if os.path.exists(p):
        for r in csv.DictReader(open(p, newline="")):
            f = (r.get("fen_start") or r.get("fen") or "").strip()
            if f:
                exclude.add(f)

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

done = {}
if os.path.exists(OUT):
    for r in csv.DictReader(open(OUT, newline="")):
        if r.get("fen"):
            done[r["fen"]] = r

FIELDS = ["fen", "phase_bucket", "best_uci", "best_cp", "moves"]
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
eng = chess.engine.SimpleEngine.popen_uci(sfpath)
try:
    eng.configure({"Threads": 1})
except Exception:
    pass

print("  %d picked (%d phases ~%d each), %d done  SF depth %d multipv %d -> %s"
      % (len(picked), len(phases), per, len(done), SF_DEPTH, K, os.path.basename(OUT)), flush=True)

n_new = 0
for i, (fen, ph) in enumerate(picked, 1):
    if fen in done:
        continue
    board = chess.Board(fen)
    try:
        infos = eng.analyse(board, chess.engine.Limit(depth=SF_DEPTH), multipv=K)
    except Exception as e:
        print("  [%4d/%4d] SKIP %s" % (i, len(picked), type(e).__name__), flush=True)
        continue
    pairs = []
    for info in infos:
        pv = info.get("pv"); sc = info.get("score")
        if not pv or sc is None:
            continue
        cp = sc.pov(chess.WHITE).score(mate_score=100000)
        pairs.append((pv[0].uci(), cp))
    if not pairs:
        continue
    best_uci, best_cp = pairs[0]
    out_rows.append({"fen": fen, "phase_bucket": ph, "best_uci": best_uci, "best_cp": best_cp,
                     "moves": ";".join("%s:%d" % (u, c) for u, c in pairs)})
    n_new += 1
    if n_new % 50 == 0:
        flush()
        print("  [%4d/%4d] labelled %d new" % (i, len(picked), n_new), flush=True)

flush()
try:
    eng.quit()
except Exception:
    pass
ph_counts = defaultdict(int)
for r in out_rows:
    ph_counts[r.get("phase_bucket", "?")] += 1
print("\n  DONE: %d total -> %s" % (len(out_rows), OUT))
print("  by phase: %s" % "  ".join("%s=%d" % (k, ph_counts[k]) for k in sorted(ph_counts)))
