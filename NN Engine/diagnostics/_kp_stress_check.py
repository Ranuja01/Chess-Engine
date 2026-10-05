# -*- coding: utf-8 -*-
"""K+PAWNS STRESS CHECK (owner, 10-04): does a pawn term UNDERSTAND pawn chess, or did it memorise typical structures?

Held-out, CHECK-ONLY (nothing is ever fitted on these rows): randomised king+pawn positions from `gen_kp_fens.py`
(`dense` = 7-8 pawns a side, unreachable in real games; `pure` = 3-6 a side). With no pieces the pawn terms ARE the
eval. Metric (memory `final-retune-needs-a-giant-diverse-corpus`): our d10 SEARCH vs SF18 d14 — win% |gap| and W/D/L
category agreement — never the static eval (pawn endings are tempo/search-decided) and never d10 move regret.
  MODE=label IN=<gen_kp_fens output> OUT=ks_sets/kp_stress_sf18.csv   → fen,best_uci,best_cp (White cp, SF18 d14)
  then per arm: _depth_residual_pass.py IN=ks_sets/kp_stress_sf18.csv OUT=ks_sets/kp_stress_<arm>_d10.csv CHUNK=…
  MODE=read ARMS=off:kp_stress_off_d10,on:kp_stress_on_d10              → per-arm |gap| and W/D/L agreement, by mix
Category: White cp > +150 win · < −150 loss · else draw-ish (on both the label and our score).
"""
import os, sys, csv, glob
import numpy as np
import chess

KV = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
THIS = os.path.dirname(os.path.abspath(__file__))
SF18 = os.environ.get("SF18_BIN", "/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/stockfish_18_linux/"
                                  "stockfish-ubuntu-x86-64-avx2")
K = 0.00368208
wp = lambda cp: 100.0 / (1.0 + np.exp(-K * np.clip(cp, -1500, 1500)))
cat = lambda cp: 1 if cp > 150 else (-1 if cp < -150 else 0)


def label():
    import chess.engine
    fens = [l.split("#")[0].strip() for l in open(KV["IN"]) if l.strip() and not l.startswith("#")]
    out = os.path.join(THIS, KV.get("OUT", "ks_sets/kp_stress_sf18.csv"))
    e = chess.engine.SimpleEngine.popen_uci(SF18, timeout=120)
    n = 0
    with open(out, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["fen", "best_uci", "best_cp", "pawns"])
        for fen in fens:
            b = chess.Board(fen)
            i = e.analyse(b, chess.engine.Limit(depth=int(KV.get("DEPTH", 14))))
            cp = i["score"].white().score(mate_score=100000)
            w.writerow([fen, i["pv"][0].uci(), cp, len(b.pieces(chess.PAWN, chess.WHITE)) + len(b.pieces(chess.PAWN, chess.BLACK))])
            n += 1
    e.quit()
    print("KP LABEL  %d positions → %s" % (n, out))


def read():
    lab = {r["fen"]: r for r in csv.DictReader(open(os.path.join(THIS, KV.get("LABELS", "ks_sets/kp_stress_sf18.csv"))))}
    for spec in KV["ARMS"].split(","):
        name, pre = spec.split(":")
        ours = {}
        for p in glob.glob(os.path.join(THIS, "ks_sets", pre + "*.csv")):
            for r in csv.DictReader(open(p, newline="")):
                ours[r["fen"]] = float(r["ours_cp_white"])
        for grp, keep in (("ALL", lambda r: True), ("dense(14+ pawns)", lambda r: int(r["pawns"]) >= 14),
                          ("pure(<14 pawns)", lambda r: int(r["pawns"]) < 14)):
            rows = [(float(r["best_cp"]), ours[f]) for f, r in lab.items() if f in ours and keep(r) and abs(float(r["best_cp"])) < 50000]
            if not rows:
                continue
            sf, us = map(np.array, zip(*rows))
            gap = np.abs(wp(us) - wp(sf)).mean()
            agree = np.mean([cat(a) == cat(b) for a, b in rows])
            drawn = sf.__abs__() <= 150
            over = np.mean(np.abs(us[drawn]) > 150) if drawn.any() else float("nan")
            print("KPS %-4s %-17s n %4d  mean|gap| %5.2f pp  W/D/L agree %5.1f%%  SF-drawn read as decisive %5.1f%%"
                  % (name, grp, len(rows), gap, 100 * agree, 100 * over))


if __name__ == "__main__":
    {"label": label, "read": read}[KV.get("MODE", "read")]()
