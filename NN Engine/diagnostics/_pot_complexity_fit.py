# -*- coding: utf-8 -*-
"""POT COMPLEXITY SCALER test at PLAY depth (2026-10-18; queue #80; memory pot-may-not-work-use-stopping-points).
  python diagnostics/_pot_complexity_fit.py [IN=/mnt/e/chess_data/cx/cx_in.csv] [D=/mnt/e/chess_data/cx/cx_cand_d12.csv]

Complexity = how far the existing verdict can be trusted (owner, POT brief §5.2): it never adds score on its own, it scales the
existing edge. Form = SF12+'s bounded edge leg on our candidate d12 SEARCH score (root approximation of an eval term):
    pred = d12 + egw · sign(d12) · max(w·x + b, −|d12|)        egw = endgame weight (1 − npm phase), cp, White POV
x = SF11 / SF12 winnable() inputs from python-chess: passed pawns, pawn count, king outflanking (file dist − rank dist),
king infiltration, pawns on both flanks, pure pawn ending (no non-pawn material), almost-unwinnable.
Arms (5-fold by GAME, held-out win% MSE on the endgame rows):
    base      d12 as is (candidate = gen 3 + the C64 class cells)
    const     b only (a global confidence shift = calibration, NOT complexity information)  ← the control
    full      w·x + b
KEEP (registered in _queue_cx_d12.sh before the data): full beats base by ≥ 2%. Reported also: full vs const (the share that is
complexity INFORMATION rather than a scale), and per material class.
"""
import os, sys, csv, collections
import numpy as np
import chess
from scipy.optimize import minimize

os.environ.setdefault("OMP_NUM_THREADS", "1")
KV = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
IN = KV.get("IN", "/mnt/e/chess_data/cx/cx_in.csv")
D = KV.get("D", "/mnt/e/chess_data/cx/cx_cand_d12.csv")
if os.name == "nt":
    IN, D = [p.replace("/mnt/e/", "E:/") for p in (IN, D)]
K = 0.00368208
wp = lambda cp: 100.0 / (1.0 + np.exp(-K * np.clip(cp, -1500, 1500)))
NAMES = ["passed", "pawns", "outflank", "infiltr", "bothflank", "pureP", "unwinnable"]
NPM = {chess.KNIGHT: 1, chess.BISHOP: 1, chess.ROOK: 2, chess.QUEEN: 4}


def passed(b, c):
    n = 0
    for sq in b.pieces(chess.PAWN, c):
        f, r = chess.square_file(sq), chess.square_rank(sq)
        ahead = range(r + 1, 8) if c == chess.WHITE else range(0, r)
        if not any(b.piece_at(chess.square(ff, rr)) == chess.Piece(chess.PAWN, not c)
                   for ff in (f - 1, f, f + 1) if 0 <= ff < 8 for rr in ahead):
            n += 1
    return n


def feats(fen):
    b = chess.Board(fen)
    wk, bk = b.king(chess.WHITE), b.king(chess.BLACK)
    fd = abs(chess.square_file(wk) - chess.square_file(bk)); rd = abs(chess.square_rank(wk) - chess.square_rank(bk))
    outflank = fd - rd
    infiltr = int(chess.square_rank(wk) > 3 or chess.square_rank(bk) < 4)
    pawns = b.pieces(chess.PAWN, chess.WHITE) | b.pieces(chess.PAWN, chess.BLACK)
    files = {chess.square_file(s) for s in pawns}
    both = int(any(f < 4 for f in files) and any(f >= 4 for f in files))
    npm = sum(NPM[p.piece_type] for p in b.piece_map().values() if p.piece_type in NPM)
    x = [passed(b, chess.WHITE) + passed(b, chess.BLACK), len(pawns), outflank, infiltr, both, int(npm == 0),
         int(outflank < 0 and not both)]
    egw = 1.0 - min(npm, 24) / 24.0
    # material class for the breakdown
    pm = collections.Counter(p.piece_type for p in b.piece_map().values() if p.piece_type in NPM)
    cls = "pawn" if npm == 0 else ("queen" if pm[chess.QUEEN] else ("rook" if pm[chess.ROOK] else "minor"))
    return x, egw, cls


def load():
    tgt = {}; game = {}
    for r in csv.DictReader(open(IN)):
        if r["best_cp"] and abs(float(r["best_cp"])) < 20000:
            tgt[r["fen"]] = float(r["best_cp"]); game[r["fen"]] = r["game"]
    d = {r["fen"]: float(r["ours_cp_white"]) for r in csv.DictReader(open(D))}
    fens = [f for f in tgt if f in d]
    X, E, C = [], [], []
    for f in fens:
        x, e, c = feats(f); X.append(x); E.append(e); C.append(c)
    return (np.array([tgt[f] for f in fens]), np.array([d[f] for f in fens]), np.array(X, float), np.array(E), np.array(C),
            np.array([game[f] for f in fens]), len(tgt), len(d))


def predict(theta, s, X, e, use_x):
    c = theta[0] + (X @ theta[1:] if use_x else 0.0)
    return s + e * np.sign(s) * np.maximum(c, -np.abs(s))


def fit(t, s, X, e, use_x, mu, sd):
    Z = (X - mu) / sd
    n = 1 + (X.shape[1] if use_x else 0)
    f = lambda th: np.mean((wp(predict(th, s, Z, e, use_x)) - wp(t)) ** 2) + 1e-6 * np.sum(th[1:] ** 2)
    return minimize(f, np.zeros(n), method="Powell", options={"maxiter": 20000, "xtol": 1e-3, "ftol": 1e-9}).x


def main():
    t, s, X, e, C, G, nt, nd = load()
    print("rows: labelled %d · d12 %d · joined %d (endgame weight mean %.2f)" % (nt, nd, len(t), e.mean()))
    ug = np.unique(G); rng = np.random.default_rng(7); rng.shuffle(ug)
    fold = {g: i % 5 for i, g in enumerate(ug)}; F = np.array([fold[g] for g in G])
    pred = {"base": s.copy(), "const": np.zeros_like(s), "full": np.zeros_like(s)}
    W = []
    for k in range(5):
        tr, te = F != k, F == k
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        for arm, ux in (("const", False), ("full", True)):
            th = fit(t[tr], s[tr], X[tr], e[tr], ux, mu, sd)
            pred[arm][te] = predict(th, s[te], (X[te] - mu) / sd, e[te], ux)
            if ux: W.append(th[1:] / sd)
        print("  fold %d done" % k, flush=True)
    mse = lambda p, m: float(np.mean((wp(p[m]) - wp(t[m])) ** 2))
    bias = lambda p, m: float(np.mean((wp(p[m]) - wp(t[m])) * np.sign(t[m]))) if m.any() else float("nan")
    allm = np.ones(len(t), bool)
    print("\nheld-out win% MSE (bias, side ahead; negative = under-confident), 5-fold by game:")
    print("%-8s %6s  %-16s %-16s %-16s  full vs base  full vs const" % ("class", "n", "base", "const", "full"))
    for name, m in [("ALL", allm)] + [(c, C == c) for c in ("pawn", "minor", "rook", "queen")]:
        if not m.any(): continue
        b0, c0, f0 = mse(pred["base"], m), mse(pred["const"], m), mse(pred["full"], m)
        print("%-8s %6d  %6.1f (%+5.2f)  %6.1f (%+5.2f)  %6.1f (%+5.2f)   %+6.2f%%      %+6.2f%%" % (
            name, m.sum(), b0, bias(pred["base"], m), c0, bias(pred["const"], m), f0, bias(pred["full"], m),
            100 * (f0 / b0 - 1), 100 * (f0 / c0 - 1)))
    Wm = np.mean(W, 0)
    print("\nmean fitted weights (cp per unit, + = MORE confident):", ", ".join("%s %+.1f" % kv for kv in zip(NAMES, Wm)))
    g = 100 * (mse(pred["full"], allm) / mse(pred["base"], allm) - 1)
    print("\nKEEP criterion (full vs base <= -2.00%%): %+.2f%% => %s" % (g, "KEEP" if g <= -2.0 else "PARK"))


if __name__ == "__main__":
    main()
