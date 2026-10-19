# -*- coding: utf-8 -*-
"""CEILING TEST for middlegame POT with boosted trees (2026-10-19; POT brief §5n — arms, hyper-parameters, criterion and
prediction registered BEFORE this ran). Is there predictable signal beyond our candidate d12 search at all on these labels?
  MODE=dump  V2_PRESET=candidate pyrun diagnostics/_pot_gbm_ceiling.py   → E:\\chess_data\\deep\\gbm_inputs.npz (engine static pass)
  MODE=fit   OMP_NUM_THREADS=3 pyrun diagnostics/_pot_gbm_ceiling.py     (sklearn HistGradientBoostingRegressor, 5-fold by game)
Arms: T0 trees on d12 alone (non-linear calibration = the fair control) · T1 d12 + §5m features · T2 d12 + RICH inputs.
SIGNAL: T2 beats T0 by ≥ 2% held-out win% MSE on target A (SF18 t0) in ≥ 4 of 5 folds. Targets B (t+16 / t+24) reported.
"""
import os, sys, zlib, time
import numpy as np
KV = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
MODE = KV.get("MODE", "fit")
NPZ = KV.get("NPZ", "/mnt/e/chess_data/deep/gbm_inputs.npz")
THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS)
import _pot_mg_fit as P                       # reuses load(), feats(), wp, NAMES (it resets sys.argv: KV parsed above)
import chess

EB = ["total", "material", "pawn_struct", "pieces", "mobility", "king_safety", "kaufman_imbalance", "v2_kflank", "v2_kprot",
      "v2_passers", "v2_placement", "v2_pxpass", "v2_shelter", "v2_winnab", "v2_phase256", "det_ks_units_w", "det_ks_units_b",
      "det_w_mobility", "det_b_mobility"]
PT = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]


def board_inputs(b):
    x = []
    for c in (chess.WHITE, chess.BLACK):            # per-file pawn count + most advanced relative rank (0 = none)
        for f in range(8):
            ps = [s for s in b.pieces(chess.PAWN, c) if chess.square_file(s) == f]
            rr = [chess.square_rank(s) if c == chess.WHITE else 7 - chess.square_rank(s) for s in ps]
            x += [len(ps), max(rr) if rr else 0]
    for c in (chess.WHITE, chess.BLACK):
        k = b.king(c); x += [chess.square_file(k), chess.square_rank(k)]
        x += [len(b.pieces(p, c)) for p in PT]
    x += [1 if b.turn == chess.WHITE else 0, int(b.has_castling_rights(chess.WHITE)), int(b.has_castling_rights(chess.BLACK))]
    return x


def dump():
    d12, tr, meta = P.load()
    fens = [f for f in d12 if f in meta and 0 in tr[f] and abs(tr[f][0]) < 20000]
    AI = None
    import ChessAI
    AI = ChessAI.ChessAI(None, None, chess.Board(), True)
    F, R, E, ok = [], [], [], []
    t0 = time.time()
    for i, f in enumerate(fens):
        b = chess.Board(f)
        x, _ = P.feats(b)
        eb = AI.ev_breakdown(b)
        F.append(x); R.append(board_inputs(b)); E.append([float(eb.get(k, 0)) for k in EB])
        if (i + 1) % 5000 == 0:
            print("  dump %d / %d  %.0fs" % (i + 1, len(fens), time.time() - t0), flush=True)
    tgt = {k: np.array([tr[f].get(k, np.nan) for f in fens]) for k in (0, 16, 24)}
    np.savez_compressed(NPZ, fens=np.array(fens), game=np.array([meta[f][0] for f in fens]),
                        stage=np.array([meta[f][1] for f in fens]), d12=np.array([d12[f] for f in fens]),
                        static=np.array([meta[f][3] for f in fens]), F=np.array(F, float), R=np.array(R, float),
                        E=np.array(E, float), t0=tgt[0], t16=tgt[16], t24=tgt[24])
    print("wrote %s: %d rows" % (NPZ, len(fens)))


def fit():
    from sklearn.ensemble import HistGradientBoostingRegressor
    z = np.load(NPZ, allow_pickle=False)
    d12 = np.clip(z["d12"], -1500, 1500)
    s_wp = P.wp(d12)
    X0 = s_wp[:, None]
    X1 = np.hstack([X0, z["F"]])
    X2 = np.hstack([X1, np.clip(z["static"], -1500, 1500)[:, None], z["E"], z["R"]])
    names2 = ["d12_wp"] + P.NAMES + ["static"] + ["eb_" + k for k in EB] + ["R%d" % i for i in range(z["R"].shape[1])]
    fold = np.array([zlib.crc32(g.encode()) % 5 for g in z["game"]])
    print("rows %d · inputs T0 %d · T1 %d · T2 %d" % (len(d12), X0.shape[1], X1.shape[1], X2.shape[1]), flush=True)
    model = lambda: HistGradientBoostingRegressor(learning_rate=0.05, max_iter=500, max_leaf_nodes=31, min_samples_leaf=50,
                                                  l2_regularization=1.0, early_stopping=True, validation_fraction=0.1,
                                                  n_iter_no_change=20, random_state=7)
    verdict = None
    for tname, key in (("A  SF18 t0", "t0"), ("B  SF18 t+16", "t16"), ("B  SF18 t+24", "t24")):
        t = z[key]; ok = np.isfinite(t) & (np.abs(t) < 20000)
        y = P.wp(t[ok]); fo = fold[ok]
        pred = {a: np.zeros(ok.sum()) for a in ("T0", "T1", "T2")}
        per_fold = []
        imp = None
        for k in range(5):
            tr_, te = fo != k, fo == k
            row = []
            for a, X in (("T0", X0), ("T1", X1), ("T2", X2)):
                Xo = X[ok]
                m = model().fit(Xo[tr_], y[tr_])
                pred[a][te] = m.predict(Xo[te])
                row.append(float(np.mean((pred[a][te] - y[te]) ** 2)))
                if a == "T2" and k == 0 and key == "t0":
                    imp = m
            per_fold.append(row)
        mse = {a: float(np.mean((pred[a] - y) ** 2)) for a in pred}
        base = float(np.mean((s_wp[ok] - y) ** 2))
        g1, g2 = 100 * (mse["T1"] / mse["T0"] - 1), 100 * (mse["T2"] / mse["T0"] - 1)
        wins = sum(1 for r in per_fold if 100 * (r[2] / r[0] - 1) <= -2.0)
        print("\n=== %s  (n %d) held-out win%% MSE: raw d12 %.2f · T0 %.2f · T1 %.2f (%+.2f%%) · T2 %.2f (%+.2f%%)"
              % (tname, ok.sum(), base, mse["T0"], mse["T1"], g1, mse["T2"], g2))
        print("    per fold T2 vs T0: " + " ".join("%+.2f%%" % (100 * (r[2] / r[0] - 1)) for r in per_fold)
              + "   (folds ≤ −2%%: %d / 5)" % wins)
        for st in ("mg", "mgl"):
            m_ = z["stage"][ok] == st
            print("    stage %-3s n %5d: T0 %.2f → T2 %.2f (%+.2f%%)" % (st, m_.sum(), np.mean((pred["T0"][m_] - y[m_]) ** 2),
                  np.mean((pred["T2"][m_] - y[m_]) ** 2), 100 * (np.mean((pred["T2"][m_] - y[m_]) ** 2) / np.mean((pred["T0"][m_] - y[m_]) ** 2) - 1)))
        if key == "t0":
            verdict = (g2, wins)
            if imp is not None:
                from sklearn.inspection import permutation_importance
                te = fold[ok] == 0
                pi = permutation_importance(imp, X2[ok][te], y[te], n_repeats=3, random_state=7, scoring="neg_mean_squared_error")
                order = np.argsort(-pi.importances_mean)[:15]
                print("    top permutation importances (fold 0, MSE increase):",
                      ", ".join("%s %.2f" % (names2[i], pi.importances_mean[i]) for i in order))
    g2, wins = verdict
    print("\nSIGNAL criterion (T2 vs T0 ≤ −2%% on A, ≥ 4/5 folds): %+.2f%%, %d/5 folds ⇒ %s"
          % (g2, wins, "SIGNAL" if g2 <= -2 and wins >= 4 else "NO SIGNAL (miss)"))


def decomp():
    """MODE=decomp (diagnostic, after the registered SIGNAL): which input group carries T2's gain on target A, and is it
    MAGNITUDE (confidence by material class / phase — winnability-like) or DIRECTION (side-ahead bias, potential)?"""
    from sklearn.ensemble import HistGradientBoostingRegressor
    z = np.load(NPZ, allow_pickle=False)
    s_wp = P.wp(np.clip(z["d12"], -1500, 1500))[:, None]
    R, E, F = z["R"], z["E"], z["F"]
    st = np.clip(z["static"], -1500, 1500)[:, None]
    counts = R[:, [34, 35, 36, 37, 38, 41, 42, 43, 44, 45]]
    mat = np.hstack([counts, E[:, [EB.index("material"), EB.index("v2_phase256")]]])
    layout = R[:, list(range(32)) + [32, 33, 39, 40]]
    groups = [("T0 d12", s_wp), ("+ material/phase", np.hstack([s_wp, mat])), ("+ pawn layout + kings", np.hstack([s_wp, layout])),
              ("+ eval terms + static", np.hstack([s_wp, E, st])), ("+ static only", np.hstack([s_wp, st])),
              ("T2 minus POT feats", np.hstack([s_wp, st, E, R])), ("T2 full", np.hstack([s_wp, F, st, E, R]))]
    t = z["t0"]; y = P.wp(t); fold = np.array([zlib.crc32(g.encode()) % 5 for g in z["game"]])
    sgn = np.sign(t)
    model = lambda: HistGradientBoostingRegressor(learning_rate=0.05, max_iter=500, max_leaf_nodes=31, min_samples_leaf=50,
                                                  l2_regularization=1.0, early_stopping=True, validation_fraction=0.1,
                                                  n_iter_no_change=20, random_state=7)
    base = None
    print("target A (SF18 t0), held-out win% MSE, 5-fold by game; bias = side-ahead (neg = under-confident)")
    for name, X in groups:
        p = np.zeros(len(y))
        for k in range(5):
            tr_, te = fold != k, fold == k
            p[te] = model().fit(X[tr_], y[tr_]).predict(X[te])
        mse = float(np.mean((p - y) ** 2)); base = base or mse
        # direction check: among positions where SF and d12 agree on WHO is ahead vs disagree
        agree = np.sign(z["d12"]) == sgn
        print("  %-24s MSE %6.2f (%+6.2f%%)  bias %+5.2f  · sign-agree rows %6.2f · sign-disagree rows (n %d) %6.2f"
              % (name, mse, 100 * (mse / base - 1), float(np.mean((p - y) * sgn)), float(np.mean((p[agree] - y[agree]) ** 2)),
                 (~agree).sum(), float(np.mean((p[~agree] - y[~agree]) ** 2))), flush=True)


def moves():
    """MODE=moves (diagnostic): does a learned correction DISCRIMINATE siblings, or only recalibrate? Children have no d12, so
    both arms are STATIC-based: Ts = trees on our candidate static alone · Tr = static + rich inputs (no POT features).
    SF18 d14 multi-PV (mg_mpv.csv): critical positions (best − 2nd ≥ 15 win% pts), pairs vs ≥ 10-pt losers, paired fixes /
    breaks of Ts's ordering by Tr ⇒ z (same definition as §5i / §5m)."""
    import csv
    from sklearn.ensemble import HistGradientBoostingRegressor
    import ChessAI
    z = np.load(NPZ, allow_pickle=False)
    stat = np.clip(-z["E"][:, EB.index("total")] / 10.0, -1500, 1500)       # candidate static, White cp (engine is Black-positive)
    Xs = P.wp(stat)[:, None]
    Xr = np.hstack([Xs, z["E"], z["R"]])
    y = P.wp(z["t0"]); gf = lambda g: zlib.crc32(g.encode()) % 5
    fold = np.array([gf(g) for g in z["game"]])
    model = lambda: HistGradientBoostingRegressor(learning_rate=0.05, max_iter=500, max_leaf_nodes=31, min_samples_leaf=50,
                                                  l2_regularization=1.0, early_stopping=True, validation_fraction=0.1,
                                                  n_iter_no_change=20, random_state=7)
    M = [(model().fit(Xs[fold != k], y[fold != k]), model().fit(Xr[fold != k], y[fold != k])) for k in range(5)]
    for k in range(5):
        te = fold == k
        print("  fold %d held-out MSE  Ts %.2f · Tr %.2f" % (k, np.mean((M[k][0].predict(Xs[te]) - y[te]) ** 2),
                                                            np.mean((M[k][1].predict(Xr[te]) - y[te]) ** 2)), flush=True)
    game = dict(zip(z["fens"], z["game"]))
    AI = ChessAI.ChessAI(None, None, chess.Board(), True)

    def inputs(b):
        eb = AI.ev_breakdown(b)
        e = [float(eb.get(k, 0)) for k in EB]
        s = np.clip(-e[0] / 10.0, -1500, 1500)
        return np.array([[float(P.wp(s))]]), np.array([[float(P.wp(s))] + e + board_inputs(b)])
    fixed = broke = pairs = ncrit = top_s = top_r = 0
    for r in csv.DictReader(open(os.path.join(P.DEEP, "mg_mpv.csv"), newline="")):
        if r["fen"] not in game:
            continue
        mv = [(u, float(c)) for u, c in (x.split(":") for x in r["moves"].split(";") if ":" in x)]
        b = chess.Board(r["fen"]); white = b.turn
        sfw = {u: (P.wp(c) if white else 100 - P.wp(c)) for u, c in mv}
        if len(sfw) < 2:
            continue
        o = sorted(sfw.values(), reverse=True)
        if o[0] - o[1] < 15:
            continue
        ncrit += 1
        ms, mr = M[gf(game[r["fen"]])]
        best = max(sfw, key=sfw.get); vs, vr = {}, {}
        for u in sfw:
            m = chess.Move.from_uci(u)
            if m not in b.legal_moves:
                continue
            b.push(m)
            if b.is_game_over():
                vs[u] = vr[u] = 100.0 if b.is_checkmate() else 50.0
            else:
                a, c = inputs(b)
                ps, pr = float(ms.predict(a)[0]), float(mr.predict(c)[0])
                vs[u], vr[u] = (ps, pr) if white else (100 - ps, 100 - pr)
            b.pop()
        if best not in vs:
            continue
        top_s += max(vs, key=vs.get) == best; top_r += max(vr, key=vr.get) == best
        for u in vs:
            if u != best and sfw[best] - sfw[u] >= 10:
                pairs += 1
                rs, rr = vs[best] > vs[u], vr[best] > vr[u]
                fixed += rr and not rs; broke += rs and not rr
    zz = (fixed - broke) / max(1.0, (fixed + broke) ** 0.5)
    print("\nMOVE TEST (static-based trees): %d critical positions, %d pairs" % (ncrit, pairs))
    print("  top-1 = SF best: Ts %.1f%% → Tr %.1f%%" % (100 * top_s / max(1, ncrit), 100 * top_r / max(1, ncrit)))
    print("  rich inputs FIX %d · BREAK %d · z = %+.2f" % (fixed, broke, zz))


if __name__ == "__main__":
    {"dump": dump, "decomp": decomp, "moves": moves}.get(MODE, fit)()
