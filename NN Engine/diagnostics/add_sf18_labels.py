# -*- coding: utf-8 -*-
"""Add SF18-SEARCH truth to a CURATED SUBSET of position_bank.csv (the rows that matter for the KS fit): the
KS-gap targets (SF11 sees danger we zero), the working controls, and high-king-density crowded-safe candidates.
SF18-search is expensive (~1-2s/pos), so we label only these, not the whole bank. Resumable (skips rows that
already have sf18) and budget-capped. Writes sf18 (WHITE-POV pawns) back into position_bank.csv.
Run: pyrun diagnostics/add_sf18_labels.py [BUDGET=500] [DEPTH=18]"""
import os, sys, csv, zlib
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
for _a in sys.argv[1:]:
    if '=' in _a: _k, _v = _a.split('=', 1); os.environ.setdefault(_k, _v)
BUDGET = int(os.environ.get('BUDGET', '500'))
DEPTH  = int(os.environ.get('DEPTH', '18'))
THREADS = int(os.environ.get('THREADS', '4'))     # SF18 threads per position (4 cores now free)
BROAD  = int(os.environ.get('BROAD', '0'))        # 1 = ALSO label a stratified sample of quiet/other rows
BROAD_MOD = int(os.environ.get('BROAD_MOD', '3')) # include a quiet row when crc32(fen) %% BROAD_MOD == 0 (~1/N)
THIS = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(THIS)); sys.path.insert(0, THIS)
import chess, chess.engine
sys.path.insert(0, os.path.join(os.path.dirname(THIS), "selfplay"))
from arbiter import find_stockfish
BANK = os.path.join(THIS, "ks_sets", "position_bank.csv")

rows = list(csv.DictReader(open(BANK)))
def needs(r):
    if r.get("sf18", "") not in ("", None): return False          # already labeled (resume)
    try:
        sk = abs(float(r["sf11_ks"])); ok = abs(float(r["our_ks"]))
        kz = max(int(r["kzone_w"]), int(r["kzone_b"]))
    except Exception:
        return False
    target  = sk >= 1.0 and ok < 0.3                               # KS-gap target candidate
    working = sk >= 0.5 and abs(float(r["our_ks"]) - float(r["sf11_ks"])) < 0.5
    crowded = kz >= 3                                              # crowded-safe candidate (SF18 decides safe/not)
    broad   = BROAD and (zlib.crc32(r["fen"].encode()) % BROAD_MOD == 0)  # stratified quiet/other diversity sample
    return target or working or crowded or broad

todo = [r for r in rows if needs(r)][:BUDGET]
print("SF18-labeling %d curated rows (budget %d, depth %d)" % (len(todo), BUDGET, DEPTH))
sf18 = chess.engine.SimpleEngine.popen_uci(find_stockfish())
try: sf18.configure({"Threads": THREADS})
except Exception: pass
def s18(fen):
    b = chess.Board(fen); i = sf18.analyse(b, chess.engine.Limit(depth=DEPTH)); s = i["score"].white()
    return 99.0 if (s.is_mate() and s.mate() > 0) else (-99.0 if s.is_mate() else s.score() / 100.0)

done = 0
for r in todo:
    try:
        r["sf18"] = round(s18(r["fen"]), 2); done += 1
    except Exception:
        continue
    if done % 50 == 0: print("  ...%d" % done)
sf18.quit()

with open(BANK, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
labeled = sum(1 for r in rows if r.get("sf18", "") not in ("", None))
print("SF18-labeled this pass: %d   total labeled in bank: %d / %d" % (done, labeled, len(rows)))
