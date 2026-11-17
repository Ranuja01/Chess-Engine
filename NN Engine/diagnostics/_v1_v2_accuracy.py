# -*- coding: utf-8 -*-
"""v1 vs v2 ACCURACY for the official-engine record (2026-10-20). Win% error of static evals and d10 searches against SF18 d14
on 5,988 held-out quiet own-play rows (corpus_v2 val, 1,500 per stage; E:\\chess_data\\wdlcal), on BOTH labels from the same
SF18 run: the historical fixed-formula cp label and SF18's own WDL expected score (memory sf-cp-labels-are-material-scaled-use-wdl).
  MODE=static TAG=v2 V2_PRESET=shipped pyrun diagnostics/_v1_v2_accuracy.py    → static_v2.csv     (v1: TAG=v1, no V2_PRESET)
  MODE=sf11 pyrun diagnostics/_v1_v2_accuracy.py                                → static_sf11.csv
  MODE=sf11d [DEPTH=10] pyrun …                                                 → d10_sf11.csv (SF11's search score per row)
  MODE=prep pyrun …                                                             → cal_d10_in.csv for _depth_residual_pass.py
  MODE=read pyrun …   reads every static_*.csv and d10_*.csv present; REF=d10_g62 adds a paired table against that arm
  MODE=bands ARMS=static_g75,d10_g75,static_sf11 [STAGES=mg,mgl] pyrun …   calibration by band (compressed in decisive positions?)
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


def sf11_search():
    """SF11's own search score at DEPTH (default 10) on every row → d10_sf11.csv (the search-vs-static control for MODE=bands).
    Nominal depth, not equal work: SF11's d10 tree is smaller and better ordered than ours. A fresh game per row (no carried hash)."""
    import chess.engine
    from eval_vs_sf11 import SF11
    depth = int(KV.get("DEPTH", "10"))
    eng = chess.engine.SimpleEngine.popen_uci(SF11)
    eng.configure({"Threads": 1, "Hash": 16})
    out = os.path.join(D, "d%d_sf11.csv" % depth)
    with open(out, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["fen", "ours_cp_white"])
        for r, _ in rows():
            info = eng.analyse(chess.Board(r["fen"]), chess.engine.Limit(depth=depth), game=object())
            w.writerow([r["fen"], info["score"].white().score(mate_score=32000)])
    eng.quit(); print("wrote", out)


def prep():
    with open(os.path.join(D, "cal_d10_in.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["fen", "best_cp"])
        for r, _ in rows():
            w.writerow([r["fen"], r["sf_cp_white"]])
    print("wrote cal_d10_in.csv")


def load_arms(want=None):
    """Every static_*.csv and d10_*.csv in D as (name, {fen: White cp}); `want` keeps only those names, in that order."""
    arms = []
    for p in sorted(glob.glob(os.path.join(D, "static_*.csv"))) + sorted(glob.glob(os.path.join(D, "d10_*.csv"))):
        name = os.path.basename(p)[:-4]
        if "_s" in name and "of" in name.rsplit("_s", 1)[1]:   # d10 shard files, already concatenated into the arm
            continue
        if want and name not in want:
            continue
        col = "cp_white" if name.startswith("static_") else "ours_cp_white"
        arms.append((name, {r["fen"]: float(r[col]) for r in csv.DictReader(open(p, newline="")) if r.get(col)}))
    return sorted(arms, key=lambda a: want.index(a[0])) if want else arms


def bands():
    """Calibration BY BAND (2026-11-17, memory ui-games-2026-11-16-eval-compression-lead): are our scores compressed toward 0
    in decisive positions? Rows are oriented to the side the arm thinks is ahead and binned by the arm's OWN predicted win%
    (binning by the label manufactures compression from noise alone). Per band: rows, mean prediction, mean SF18 WDL expected
    score, gap = label − prediction (positive = under-confident). Shown raw (the fixed k) and at the arm's own best global scale
    s, so a SHAPE defect (gaps that still grow with the band after scaling) is told apart from a plain scale. Every arm is
    under-confident against SF18's WDL, which scores strong-engine conversion, so read ours against static_sf11."""
    R = rows()
    fen = [r["fen"] for r, _ in R]
    stage = np.array([m["stage"] for _, m in R])
    yw = np.array([(float(r["w"]) + 0.5 * float(r["d"])) / 10.0 for r, _ in R])
    want = KV["ARMS"].split(",") if KV.get("ARMS") else None
    stages = KV.get("STAGES", "mg,mgl,egl,eg").split(",")
    edges = (50, 60, 70, 80, 90, 100.01)
    grid = np.arange(0.30, 6.005, 0.01)
    sel = np.isin(stage, stages)
    print("rows %d in stages %s · label SF18 d14 WDL expected score · bands by the arm's own side-ahead prediction"
          % (sel.sum(), ",".join(stages)))

    def table(x, y):
        up = x >= 50
        p, l = np.where(up, x, 100 - x), np.where(up, y, 100 - y)
        return [(int(k.sum()), p[k].mean() if k.any() else 0.0, l[k].mean() if k.any() else 0.0)
                for k in ((p >= a) & (p < b) for a, b in zip(edges[:-1], edges[1:]))]
    for name, m in load_arms(want):
        ok = np.array([f in m for f in fen]) & sel
        cp, y = np.array([m.get(f, 0.0) for f in fen])[ok], yw[ok]
        s = grid[int(np.argmin([np.mean((wp(g * cp) - y) ** 2) for g in grid]))]
        print("\n%s — %d rows · own best global scale s=%.2f" % (name, ok.sum(), s))
        print("  %-8s  %-28s  %-28s" % ("", "raw (fixed k)", "at own s"))
        print("  %-8s  " % "pred%" + "  ".join("%5s %6s %6s %6s" % ("rows", "pred", "label", "gap") for _ in (0, 1)))
        for (a, b), r0, r1 in zip(zip(edges[:-1], edges[1:]), table(wp(cp), y), table(wp(s * cp), y)):
            print("  %3d-%-4d  " % (a, min(b, 100)) + "  ".join(
                "%5d %6.1f %6.1f %+6.1f" % (n, pp, ll, ll - pp) if n else "%5d %6s %6s %6s" % (n, "-", "-", "-")
                for n, pp, ll in (r0, r1)))


def read():
    R = rows()
    fen = [r["fen"] for r, _ in R]
    stage = np.array([m["stage"] for _, m in R])
    ycp = wp([float(r["sf_cp_white"]) for r, _ in R])
    yw = np.array([(float(r["w"]) + 0.5 * float(r["d"])) / 10.0 for r, _ in R])
    group = np.array([m["group"].split("#")[0] for _, m in R])
    arms = load_arms()
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
        # Raw MSE rewards a larger eval scale as much as a better one; this table re-scales each arm's cp by the single
        # factor that minimises its own MSE on ALL rows, so arms are compared on discrimination alone.
        # The grid reaches 6: against the WDL label every arm is under-confident enough to pin a 2.5 ceiling.
        print("  calibrated (each arm at its own best global scale s):")
        grid = np.arange(0.30, 6.005, 0.01)
        sq = {}   # per-row squared error at the arm's own scale, for the paired table
        for name, m in arms:
            ok = np.array([f in m for f in fen])
            cp = np.array([m.get(f, 0.0) for f in fen])
            s = grid[int(np.argmin([np.mean((wp(g * cp[ok]) - y[ok]) ** 2) for g in grid]))]
            x = wp(s * cp)
            sq[name] = (ok, (wp(cp) - y) ** 2, (x - y) ** 2)
            def ccell(mask):
                mk = mask & ok
                return "%6.1f        " % np.mean((x[mk] - y[mk]) ** 2) if mk.any() else "      -       "
            print("  %-14s %s  " % (name, ccell(np.ones(len(fen), bool))) + "  ".join(ccell(stage == st) for st in ("mg", "mgl", "egl", "eg"))
                  + "  s=%.2f" % s)
        # Paired against REF on the rows both arms have; the standard error resamples whole games (rows of one game are
        # correlated), 1,000 draws. The scale s is held at its full-sample value inside the resampling.
        ref = KV.get("REF")
        if ref in sq:
            print("  paired vs %s (arm − ref, common rows; ± = 1 s.e., game-clustered bootstrap):" % ref)
            rng = np.random.default_rng(1)
            for name in sq:
                if name == ref or name.startswith("static_") != ref.startswith("static_"):
                    continue
                ok = sq[name][0] & sq[ref][0]
                g = group[ok]; ug, gi = np.unique(g, return_inverse=True)
                cells = []
                for k in (1, 2):
                    d = sq[name][k][ok] - sq[ref][k][ok]
                    gs = np.bincount(gi, d); gn = np.bincount(gi)
                    bs = [gs[b].sum() / gn[b].sum() for b in (rng.integers(0, len(ug), len(ug)) for _ in range(1000))]
                    cells.append("%+6.1f ± %4.1f (%+5.1f%%)" % (d.mean(), np.std(bs), 100 * d.mean() / sq[ref][k][ok].mean()))
                print("  %-14s raw %s   calibrated %s   rows %d games %d" % (name, cells[0], cells[1], ok.sum(), len(ug)))


if __name__ == "__main__":
    {"static": static, "sf11": sf11, "sf11d": sf11_search, "prep": prep, "bands": bands}.get(MODE, read)()
