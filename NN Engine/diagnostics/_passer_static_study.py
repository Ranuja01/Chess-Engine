# -*- coding: utf-8 -*-
"""Can a STATIC eval price passers in the middlegame, or is it search? (owner, 10-03: "is SF good at the rare cases
through deep search, or does its static eval work for this too? If it does, we have reason to believe ours can.")

On the SF18-labelled std middlegame rows (`fitC_mg_sf18.csv`), restricted to positions with an ADVANCED passer
(relative rank ≥ 5 for either side; a passer = no enemy pawn ahead on its own or adjacent files), compare against
SF18's d14 SEARCH: SF11's STATIC eval (classical, no search) · OUR static eval (current shipped build, ev_breakdown) ·
OUR d10 search (the depth pass). Mean |win% gap| per evaluator, split by passer rank (5 / 6 / 7) and by who owns the most
advanced passer relative to SF18's verdict. If SF11-static sits far closer to SF18 than our static on these rows, a
static eval CAN price them and the gap is ours to close; if SF11-static is as far off as ours, it is search.
Needs the engine in-process (ev_breakdown only) + the SF11 binary. Run with V2_PRESET=shipped.

  pyrun diagnostics/_passer_static_study.py [LIMIT=0] [SET=eg]
"""
import os, sys, csv, glob, math
import numpy as np
import chess

THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS)
import _triangulate_sf11 as T          # sf11_eval, SF11 binary path, ChessAI import + env defaults
import subprocess

KV = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
K = 0.00368208
wp = lambda cp: 100.0 / (1.0 + np.exp(-K * np.clip(cp, -1500, 1500)))


def best_passer_rank(b, c):
    best = 0
    for sq in b.pieces(chess.PAWN, c):
        f, r = chess.square_file(sq), chess.square_rank(sq)
        rr = r if c == chess.WHITE else 7 - r
        ok = True
        for e in b.pieces(chess.PAWN, not c):
            ef, er = chess.square_file(e), chess.square_rank(e)
            if abs(ef - f) <= 1 and ((er > r) if c == chess.WHITE else (er < r)):
                ok = False
                break
        if ok:
            best = max(best, rr + 1)          # 1-based relative rank
    return best


def main():
    ours_d = {}
    for p in glob.glob(os.path.join(THIS, "ks_sets", "fitC_mg_ours_d10_s*of4.csv")):
        for r in csv.DictReader(open(p, newline="")):
            ours_d[r["fen"]] = float(r["ours_cp_white"])
    # SET=eg (2026-10-03): the SF18-labelled ENDGAME sample — no d10 search pass exists there, so OURS d10 = n/a
    EG = KV.get("SET") == "eg"
    smp = "fitC_eg_sample.csv" if EG else "fitC_mg_sample.csv"
    std = {r["fen"] for r in csv.DictReader(open(os.path.join(THIS, "ks_sets", smp))) if r.get("src", "std") == "std"}
    p = subprocess.Popen([T.SF11], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1)
    p.stdin.write("uci\n"); p.stdin.flush()
    while p.stdout.readline().strip() != "uciok":
        pass
    ai = T.ChessAI.ChessAI(None, None, chess.Board(), True)
    rows = []
    lim = int(KV.get("LIMIT", 0))
    for r in csv.DictReader(open(os.path.join(THIS, "ks_sets", "fitC_eg_sf18.csv" if EG else "fitC_mg_sf18.csv"), newline="")):
        f = r["fen"]
        if not r.get("best_cp") or f not in std or (not EG and f not in ours_d):
            continue
        b = chess.Board(f)
        rw, rb = best_passer_rank(b, chess.WHITE), best_passer_rank(b, chess.BLACK)
        top = max(rw, rb)
        if top < 5:
            continue
        sf11, _ = T.sf11_eval(p, f)
        if sf11 is None:
            continue
        ours_s = -float(ai.ev_breakdown(b)["total"]) / 10.0
        sf18 = float(r["best_cp"])
        rows.append((top, wp(sf18), wp(100 * sf11), wp(ours_s), wp(ours_d[f]) if not EG else np.nan))
        if lim and len(rows) >= lim:
            break
    p.stdin.write("quit\n"); p.stdin.flush()
    a = np.array(rows)
    print("PASSER STATIC STUDY  %s positions" % ("ENDGAME" if EG else "middlegame") + "   with an advanced passer (rel. rank ≥ 5): %d" % len(a))
    print("  mean |win%% gap to SF18 d14|:   %-8s %-12s %-12s %-12s" % ("n", "SF11 static", "OURS static", "OURS d10"))
    for name, m in (("all", np.ones(len(a), bool)), ("rank 5", a[:, 0] == 5), ("rank 6", a[:, 0] == 6),
                    ("rank 7", a[:, 0] == 7)):
        if m.sum():
            g = lambda j: np.abs(a[m, j] - a[m, 1]).mean()
            print("  %-30s %-8d %-12.2f %-12.2f %-12.2f" % (name, m.sum(), g(2), g(3), g(4)))
    print("  read: SF11-static ≪ ours ⇒ a static eval CAN price it (our gap); SF11 ≈ ours ⇒ it is search.")


if __name__ == "__main__":
    main()
