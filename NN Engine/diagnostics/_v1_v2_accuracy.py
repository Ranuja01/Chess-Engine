# -*- coding: utf-8 -*-
"""v1 vs v2 ACCURACY for the official-engine record (2026-10-20). Win% error of static evals and d10 searches against SF18 d14
on 5,988 held-out quiet own-play rows (corpus_v2 val, 1,500 per stage; E:\\chess_data\\wdlcal), on BOTH labels from the same
SF18 run: the historical fixed-formula cp label and SF18's own WDL expected score (memory sf-cp-labels-are-material-scaled-use-wdl).
  MODE=static TAG=v2 V2_PRESET=shipped pyrun diagnostics/_v1_v2_accuracy.py    → static_v2.csv     (v1: TAG=v1, no V2_PRESET)
  MODE=sf11 pyrun diagnostics/_v1_v2_accuracy.py                                → static_sf11.csv
  MODE=prep pyrun …                                                             → cal_d10_in.csv for _depth_residual_pass.py
  MODE=read pyrun …   reads every static_*.csv and d10_*.csv present
"""
import os, sys, csv, glob
import numpy as np
KV = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
MODE = KV.get("MODE", "read")
D = KV.get("DIR", "/mnt/e/chess_data/wdlcal")
THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS); sys.path.insert(0, os.path.dirname(THIS))
import chess

K = 0.00368208
wp = lambda cp: 100.0 / (1.0 + np.exp(-K * np.clip(np.asarray(cp, float), -1500, 1500)))


def rows():
    meta = {r["fen"]: r for r in csv.DictReader(open(os.path.join(D, "cal_in.csv"), newline=""))}
    return [(r, meta[r["fen"]]) for r in csv.DictReader(open(os.path.join(D, "cal_wdl.csv"), newline=""))
            if r["fen"] in meta and abs(float(r["sf_cp_white"])) < 20000]


def static():
    import ChessAI
    tag = KV.get("TAG", "v2")
    AI = ChessAI.ChessAI(None, None, chess.Board(), True)
    with open(os.path.join(D, "static_%s.csv" % tag), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["fen", "cp_white"])
        for r, _ in rows():
            w.writerow([r["fen"], "%.1f" % (-int(AI.ev_breakdown(chess.Board(r["fen"]))["total"]) / 10.0)])
    print("wrote static_%s.csv" % tag)


def sf11():
    from eval_vs_sf11 import SF11Eval, SF11
    e = SF11Eval(SF11)
    with open(os.path.join(D, "static_sf11.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["fen", "cp_white"])
        for r, _ in rows():
            tot, _t = e.eval(r["fen"])
            if tot is not None:
                w.writerow([r["fen"], "%.1f" % (tot * 100.0)])
    e.close(); print("wrote static_sf11.csv")


def prep():
    with open(os.path.join(D, "cal_d10_in.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["fen", "best_cp"])
        for r, _ in rows():
            w.writerow([r["fen"], r["sf_cp_white"]])
    print("wrote cal_d10_in.csv")


def read():
    R = rows()
    fen = [r["fen"] for r, _ in R]
    stage = np.array([m["stage"] for _, m in R])
    ycp = wp([float(r["sf_cp_white"]) for r, _ in R])
    yw = np.array([(float(r["w"]) + 0.5 * float(r["d"])) / 10.0 for r, _ in R])
    arms = []
    for p in sorted(glob.glob(os.path.join(D, "static_*.csv"))) + sorted(glob.glob(os.path.join(D, "d10_*.csv"))):
        name = os.path.basename(p)[:-4]
        col = "cp_white" if name.startswith("static_") else "ours_cp_white"
        m = {r["fen"]: float(r[col]) for r in csv.DictReader(open(p, newline="")) if r.get(col)}
        arms.append((name, m))
    print("rows %d (held-out quiet own-play, 1,500 per stage requested) · SF18 d14 labels" % len(R))
    for lname, y in (("historical label (SF18 cp → fixed win% formula)", ycp), ("SF18 WDL expected score", yw)):
        print("\n%s — win%% MSE (side-ahead bias; negative = under-confident)" % lname)
        print("  %-14s %7s   " % ("arm", "ALL") + "  ".join("%-14s" % s for s in ("mg", "mgl", "egl", "eg")))
        for name, m in arms:
            ok = np.array([f in m for f in fen])
            x = wp([m.get(f, 0.0) for f in fen])
            def cell(mask):
                mk = mask & ok
                if not mk.any(): return "      -       "
                e = (x[mk] - y[mk]); sgn = np.sign(y[mk] - 50)
                return "%6.1f (%+5.2f)" % (np.mean(e ** 2), np.mean(e * sgn))
            print("  %-14s %s  " % (name, cell(np.ones(len(fen), bool))) + "  ".join(cell(stage == s) for s in ("mg", "mgl", "egl", "eg")))


if __name__ == "__main__":
    {"static": static, "sf11": sf11, "prep": prep}.get(MODE, read)()
