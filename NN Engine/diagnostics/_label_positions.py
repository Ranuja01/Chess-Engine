# -*- coding: utf-8 -*-
"""Label an existing position list (csv with a `fen` column) with SF18 multi-PV -> our regret format
(fen, phase_bucket, best_uci, best_cp, moves). Stratified sample of N total across phase buckets so a small
validation batch stays balanced. Resumable (skips fens already in OUT).

  pyrun diagnostics/_label_positions.py SRC=ks_sets/lichess_ks.csv OUT=ks_sets/lichess_ks_labelled.csv N=600 SF_DEPTH=14
        (needs STOCKFISH_PATH -> native-ELF Linux SF18)
"""
import os, sys, csv
for _a in sys.argv[1:]:
    if '=' in _a:
        _k, _v = _a.split('=', 1); os.environ[_k] = _v
THIS = os.path.dirname(os.path.abspath(__file__)); ENGINE = os.path.dirname(THIS)
sys.path.insert(0, ENGINE); sys.path.insert(0, THIS)
import chess, chess.engine

SRC = os.environ["SRC"]; OUT = os.environ.get("OUT", "ks_sets/lichess_ks_labelled.csv")
if not os.path.isabs(SRC): SRC = os.path.join(THIS, SRC)
if not os.path.isabs(OUT): OUT = os.path.join(THIS, OUT)
N = int(os.environ.get("N", "600")); K = int(os.environ.get("K", "8")); DEPTH = int(os.environ.get("SF_DEPTH", "14"))
sfpath = os.environ.get("STOCKFISH_PATH") or sys.exit("STOCKFISH_PATH required (native-ELF Linux SF18)")

rows = list(csv.DictReader(open(SRC, newline="")))
# stratified sample: N/3 per phase bucket (deterministic — take the first per bucket, no RNG)
per = max(1, N // 3)
buckets = {}
for r in rows:
    b = r.get("phase_bucket", "?")
    buckets.setdefault(b, [])
    if len(buckets[b]) < per:
        buckets[b].append(r)
picked = [r for b in buckets for r in buckets[b]]

done = set()
if os.path.exists(OUT):
    for r in csv.DictReader(open(OUT, newline="")):
        done.add(r["fen"])

eng = chess.engine.SimpleEngine.popen_uci(sfpath)
try:
    eng.configure({"Threads": 1})
except Exception:
    pass
FIELDS = ["fen", "phase_bucket", "best_uci", "best_cp", "moves"]
newf = not os.path.exists(OUT)
out = open(OUT, "a", newline="")
w = csv.DictWriter(out, fieldnames=FIELDS)
if newf:
    w.writeheader()
print("  labelling %d positions (%d already done)  SF d%d multipv %d -> %s"
      % (len(picked), len(done), DEPTH, K, os.path.basename(OUT)), flush=True)
n_new = 0
for i, r in enumerate(picked, 1):
    fen = r["fen"]
    if fen in done:
        continue
    try:
        board = chess.Board(fen)
        infos = eng.analyse(board, chess.engine.Limit(depth=DEPTH), multipv=K)
    except Exception:
        continue
    pairs = []
    for info in infos:
        pv = info.get("pv"); sc = info.get("score")
        if pv and sc is not None:
            pairs.append((pv[0].uci(), sc.pov(chess.WHITE).score(mate_score=100000)))
    if not pairs:
        continue
    w.writerow({"fen": fen, "phase_bucket": r.get("phase_bucket", "?"), "best_uci": pairs[0][0],
                "best_cp": pairs[0][1], "moves": ";".join("%s:%d" % (u, c) for u, c in pairs)})
    n_new += 1
    if n_new % 100 == 0:
        out.flush(); print("  [%5d/%5d] labelled %d new" % (i, len(picked), n_new), flush=True)
out.close(); eng.quit()
print("  DONE: %d new labelled -> %s" % (n_new, os.path.basename(OUT)), flush=True)
