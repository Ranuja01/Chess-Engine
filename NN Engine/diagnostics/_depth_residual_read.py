# -*- coding: utf-8 -*-
"""Read the DEPTH residual against the STATIC one, on the SF18-labelled middlegame rows.

  static residual = win%(SF18 d14) − win%(our shipped STATIC eval)       (what every 09-30 screen used)
  depth  residual = win%(SF18 d14) − win%(our d10 SEARCH)               (`_depth_residual_pass.py` output)
A static-eval finding that does not survive into the depth residual is something our SEARCH already fixes — no eval
term can pay for it at play depth (memory `eval-headroom-is-failures-that-persist-as-depth-rises`).
Classes reported: all · side-to-move (toward the mover) · queen imbalance (toward the queen side) · balanced.

  pyrun diagnostics/_depth_residual_read.py
"""
import os, sys, csv, glob, math
import numpy as np
import chess

THIS = os.path.dirname(os.path.abspath(__file__))
DATA = "/mnt/e/chess_data/texel"
K = 0.00368208
wp = lambda cp: 100.0 / (1.0 + np.exp(-K * np.clip(cp, -1500, 1500)))

ours_d = {}
for p in glob.glob(os.path.join(THIS, "ks_sets", "fitC_mg_ours_d10_s*of4.csv")):
    for r in csv.DictReader(open(p, newline="")):
        ours_d[r["fen"]] = float(r["ours_cp_white"])
sample = {r["fen"]: int(r["row"]) for r in csv.DictReader(open(os.path.join(THIS, "ks_sets/fitC_mg_sample.csv")))
          if r["src"] == "std"}
T = np.load(os.path.join(DATA, "fitC_win.npz"))["total"]
rs, rd, stm, qside, bal = [], [], [], [], []
for r in csv.DictReader(open(os.path.join(THIS, "ks_sets/fitC_mg_sf18.csv"), newline="")):
    f = r["fen"]
    if not r.get("best_cp") or f not in sample or f not in ours_d:
        continue
    sf = float(r["best_cp"])
    stat = -float(T[sample[f]]) / 10.0
    b = chess.Board(f)
    rs.append(wp(sf) - wp(stat)); rd.append(wp(sf) - wp(ours_d[f]))
    stm.append(1.0 if b.turn == chess.WHITE else -1.0)
    wq, bq = len(b.pieces(chess.QUEEN, chess.WHITE)), len(b.pieces(chess.QUEEN, chess.BLACK))
    qside.append(1.0 if (wq and not bq) else (-1.0 if (bq and not wq) else 0.0))
    bal.append(abs(stat) < 100)
rs, rd, stm, qside, bal = map(np.array, (rs, rd, stm, qside, bal))
n = len(rs)
print("DEPTH vs STATIC residual — %d std middlegame rows (SF18 d14 labels; ours: shipped static / d10 search)" % n)
print("  mean |residual|: static %.2f pp · depth %.2f pp" % (np.abs(rs).mean(), np.abs(rd).mean()))
def row(name, v_s, v_d, m):
    se = lambda v: v[m].std() / math.sqrt(max(m.sum(), 1))
    print("  %-34s n %5d  static %+6.2f pp (se %.2f) · depth %+6.2f pp (se %.2f)"
          % (name, m.sum(), v_s[m].mean(), se(v_s), v_d[m].mean(), se(v_d)))
row("toward the side to move", stm * rs, stm * rd, np.ones(n, bool))
row("toward the side to move, balanced", stm * rs, stm * rd, bal)
q = qside != 0
row("toward the QUEEN side (queen imbalance)", qside * rs, qside * rd, q)
