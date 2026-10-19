# -*- coding: utf-8 -*-
"""SF18 WDL labels (2026-10-19; POT brief §5n follow-up). SF ≥ 15.1 scales its centipawns BY MATERIAL, so one fixed
cp → win% formula mis-reads its scores by material class; its WDL output (UCI_ShowWDL) is the win model SF itself uses.
  pyrun diagnostics/_sf18_wdl_pass.py [SPLIT=val] [DEPTH=14] [OUT=/mnt/e/chess_data/deep/mg_wdl_val.csv] [SHARD=k/n] [IN=<csv>]
Rows = trajectory mg / mgl starts of the chosen split (as in gbm_inputs.npz). Output (resumable by FEN): fen, sf_cp_white,
w, d, l (per mille, WHITE POV), depth. One process, Threads 1.
"""
import os, sys, csv, glob
import numpy as np
for _a in sys.argv[1:]:
    if "=" in _a:
        _k, _v = _a.split("=", 1); os.environ[_k] = _v
THIS = os.path.dirname(os.path.abspath(__file__))
ENGINE = os.path.dirname(THIS)
sys.path.insert(0, ENGINE); sys.path.insert(0, THIS); sys.path.insert(0, os.path.join(ENGINE, "selfplay"))
import chess, chess.engine
from arbiter import find_stockfish

SPLIT = os.environ.get("SPLIT", "val")
DEPTH = int(os.environ.get("DEPTH", "14"))
OUT = os.environ.get("OUT", "/mnt/e/chess_data/deep/mg_wdl_%s.csv" % SPLIT)
NPZ = os.environ.get("NPZ", "/mnt/e/chess_data/deep/gbm_inputs.npz")
TRAJ = os.environ.get("TRAJ", "/mnt/e/chess_data/traj/traj_s*of4.csv")

split = {}
for p in ([] if os.environ.get("IN") else sorted(glob.glob(TRAJ))):
    for r in csv.DictReader(open(p, newline="")):
        if r["k"] == "0":
            split[r["start_fen"]] = r["split"]
IN = os.environ.get("IN", "")                  # any CSV with a `fen` column instead of the trajectory split
if IN:
    fens = [r["fen"] for r in csv.DictReader(open(IN, newline=""))]
else:
    fens = [f for f in np.load(NPZ, allow_pickle=False)["fens"] if split.get(f) == SPLIT]
_k, _n = (int(x) for x in os.environ.get("SHARD", "0/1").split("/"))
if _n > 1:
    fens = [f for i, f in enumerate(fens) if i % _n == _k]
    OUT = OUT.replace(".csv", "_s%dof%d.csv" % (_k, _n))
done = set()
if os.path.exists(OUT):
    done = {r["fen"] for r in csv.DictReader(open(OUT, newline=""))}
new = not os.path.exists(OUT)
fh = open(OUT, "a", newline=""); w = csv.writer(fh)
if new:
    w.writerow(["fen", "sf_cp_white", "w", "d", "l", "depth"])
eng = chess.engine.SimpleEngine.popen_uci(os.environ.get("STOCKFISH_PATH") or find_stockfish())
eng.configure({"Threads": 1, "Hash": 64, "UCI_ShowWDL": True})
print("SF18 WDL pass: %d %s rows (%d done) at depth %d → %s" % (len(fens), SPLIT, len(done), DEPTH, OUT), flush=True)
n = 0
for f in fens:
    if f in done:
        continue
    info = eng.analyse(chess.Board(f), chess.engine.Limit(depth=DEPTH))
    sc = info["score"].white()
    cp = sc.score(mate_score=100000)
    wdl = info.get("wdl")
    if wdl is None:
        continue
    ww = wdl.white()
    w.writerow([f, cp, ww.wins, ww.draws, ww.losses, info.get("depth", "")])
    n += 1
    if n % 200 == 0:
        fh.flush(); print("  %d rows" % n, flush=True)
fh.close(); eng.quit()
print("done: %d new rows" % n)
