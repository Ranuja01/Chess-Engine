# -*- coding: utf-8 -*-
"""PAWN-ENDING SEARCH vs EVAL (owner, 2026-10-07): on the depth-target PAWN-ONLY rows (313, from our games), the side-ahead
BIAS and win% MSE vs SF18 d14 of each engine's STATIC eval and of its SEARCH at equal depth. If SF11's d10 is near 0 bias
where ours is −9, the remaining pawn-ending gap is OUR SEARCH (pruning / null move / extensions), not the eval.
BIAS = mean (win%(pred) − win%(SF18)) · sign(SF18) over |SF18| > 25 cp (− = too drawish).
  MODE=export → ks_sets/pe_rows_sf18.csv (IN for our depth pass at d14: _depth_residual_pass.py OUT=ks_sets/pe_rows_ours_d14.csv)
  MODE=read [DEPTHS=10,14]   (env: V2_PRESET=shipped)
"""
import os, sys, csv, glob, subprocess
import numpy as np
THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS)
sys.path.insert(0, os.path.dirname(THIS))
os.environ.setdefault("PRESET", "LONG_FORMAT"); os.environ.setdefault("USE_OPENING_BOOK", "0")
import chess, chess.engine
import _revival_screen as RS
from _triangulate_sf11 import SF11, sf11_eval, SF11_TEMPO_PAWNS
from _static_vs_search_cases import SF15, uci_static

KV = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
wp = lambda cp: 100.0 / (1.0 + np.exp(-0.00368208 * np.clip(np.asarray(cp, float), -1500, 1500)))


def pawn_rows():
    fens, _, ph, sfl, base, _, _ = RS.load_rows()
    out = []
    for f, p, t, d in zip(fens, ph, sfl, base):
        b = chess.Board(f)
        if p < 128 and not (b.knights | b.bishops | b.rooks | b.queens) and not b.is_check():
            out.append((f, t, d))
    return out


def stats(pred, tgt):
    pred, tgt = np.asarray(pred, float), np.asarray(tgt, float)
    a = np.abs(tgt) > 25
    return float(np.mean((wp(pred) - wp(tgt)) ** 2)), float(np.mean((wp(pred[a]) - wp(tgt[a])) * np.sign(tgt[a])))


def main():
    R = pawn_rows()
    if KV.get("MODE") == "export":
        p = os.path.join(THIS, "ks_sets/pe_rows_sf18.csv")
        with open(p, "w", newline="") as fh:
            w = csv.writer(fh); w.writerow(["fen", "best_uci", "best_cp"])
            for f, t, _ in R:
                w.writerow([f, "", t])
        print("exported %d pawn-only rows → %s" % (len(R), p))
        return
    import ChessAI
    ai = ChessAI.ChessAI(None, None, chess.Board(), True)
    ours14 = {}
    for pth in glob.glob(os.path.join(THIS, "ks_sets/pe_rows_ours_d14*.csv")):
        for r in csv.DictReader(open(pth, newline="")):
            ours14[r["fen"]] = float(r["ours_cp_white"])
    depths = [int(d) for d in KV.get("DEPTHS", "10,14").split(",")]
    p11 = subprocess.Popen([SF11], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1)
    p11.stdin.write("uci\n"); p11.stdin.flush()
    while p11.stdout.readline().strip() != "uciok":
        pass
    e11 = chess.engine.SimpleEngine.popen_uci(SF11, timeout=120)
    e15 = chess.engine.SimpleEngine.popen_uci(SF15, timeout=120); e15.configure({"Use NNUE": False})
    col = {k: [] for k in ["tgt", "ours_s", "ours_d10", "ours_d14", "sf11_s", "sf15_s"] + ["sf11_d%d" % d for d in depths] + ["sf15_d10"]}
    for f, t, d in R:
        b = chess.Board(f)
        col["tgt"].append(t); col["ours_d10"].append(d); col["ours_d14"].append(ours14.get(f, np.nan))
        col["ours_s"].append(-float(ai.ev_breakdown(b)["total"]) / 10.0)
        tot, _ = sf11_eval(p11, f)
        col["sf11_s"].append(100 * tot - (SF11_TEMPO_PAWNS if b.turn else -SF11_TEMPO_PAWNS) * 100 if tot is not None else np.nan)
        s15 = uci_static(SF15, f, (("Use NNUE", "false"),))
        col["sf15_s"].append(s15 if s15 is not None else np.nan)
        for dd in depths:
            col["sf11_d%d" % dd].append(e11.analyse(b, chess.engine.Limit(depth=dd))["score"].white().score(mate_score=100000))
        col["sf15_d10"].append(e15.analyse(b, chess.engine.Limit(depth=10))["score"].white().score(mate_score=100000))
    e11.quit(); e15.quit(); p11.stdin.write("quit\n"); p11.stdin.flush()
    tgt = np.array(col["tgt"])
    print("PAWN-ENDING rows (pawn-only, from our games): %d · target SF18 d14 · bias over |SF18| > 25 cp (− = too drawish)" % len(R))
    print("  %-22s %10s %10s" % ("predictor", "win% MSE", "bias"))
    for k in ["ours_s", "sf11_s", "sf15_s", "ours_d10", "sf11_d10", "sf15_d10", "ours_d14", "sf11_d14"] + \
             ["sf11_d%d" % d for d in depths if d not in (10, 14)]:
        if k not in col:
            continue
        v = np.array(col[k], float)
        m = ~np.isnan(v)
        if not m.any():
            print("  %-22s %10s" % (k, "(missing)")); continue
        mse, bias = stats(v[m], tgt[m])
        print("  %-22s %10.1f %+10.2f   (n %d)" % (k, mse, bias, int(m.sum())))


if __name__ == "__main__":
    main()
