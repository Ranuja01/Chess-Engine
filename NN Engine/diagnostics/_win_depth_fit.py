# -*- coding: utf-8 -*-
"""POT WINNABILITY (the shipped eg SCALE FACTOR) re-fitted on the DEPTH target (owner, 2026-10-05: winnability is a known
term — re-fit now; the full POT design comes after everything else).

The 10-01 ship fitted BASE/SP/OCB on SF18 STATIC labels (`_texel_win_sf_fit.py MODE=scale`); the 10-05 revival screen found
the same gap in Kaufman (static-label fit, −3.2% more on the depth target). Form (eval_v2.cpp win_scale_adjust):
  f = clamp(64 + BASE + SP·leader pawns + ONEFLANK·[pawns on ≤ 1 flank, a-d | e-h] + OCB·[lone opposite bishops]
            + PASSED·leader passers, 0, 64);   adjustment d = T0·(256 − phase)·(f − 64)/(256·64),  T0 = pre-scale total.
Model (White cp): ours' = ours_d10 − (d(θ) − d(θ_ship))/10 + STM·stm, base·(1+SCALE) nuisance; SF18 d14 labels, val by
GAME (15%). Fit: Powell on the 5 knobs (+ nuisances), from the shipped values. ⚠️ PASSED uses python-chess passers (v2's
pe.passed may differ on rear-doubled pawns) — screen only; closure against the engine before any gate.
  pyrun diagnostics/_win_depth_fit.py MODE=dump V2_PRESET=shipped OUT=/mnt/e/chess_data/texel/revival/win_dump.csv
  pyrun diagnostics/_win_depth_fit.py MODE=fit DUMP=/mnt/e/chess_data/texel/revival/win_dump.csv
"""
import os, sys, csv, glob, hashlib
import numpy as np
import chess
from scipy.optimize import minimize

KV = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
THIS = os.path.dirname(os.path.abspath(__file__))
K = 0.00368208
wp = lambda cp: 100.0 / (1.0 + np.exp(-K * np.clip(cp, -1500, 1500)))
SHIP = dict(BASE=-37, SP=34, ONEFLANK=0, OCB=-80, PASSED=0)
QS = chess.BB_FILE_A | chess.BB_FILE_B | chess.BB_FILE_C | chess.BB_FILE_D


def dump():
    for k, v in KV.items():
        os.environ[k] = v
    os.environ.setdefault("PRESET", "LONG_FORMAT"); os.environ.setdefault("USE_OPENING_BOOK", "0")
    sys.path.insert(0, os.path.dirname(THIS))
    import ChessAI
    ai = ChessAI.ChessAI(None, None, chess.Board(), True)
    seen = set()
    with open(KV["OUT"], "w", newline="") as fo:
        w = csv.writer(fo); w.writerow(["fen", "total", "winnab"])
        for lab in ("fitC_mg_sf18.csv", "fitC_eg_sf18.csv"):
            for r in csv.DictReader(open(os.path.join(THIS, "ks_sets", lab), newline="")):
                if r["fen"] in seen or not r.get("best_cp"):
                    continue
                seen.add(r["fen"])
                eb = ai.ev_breakdown(chess.Board(r["fen"]))
                w.writerow([r["fen"], int(eb["total"]), int(eb.get("v2_winnab", 0))])
    print("WIN DUMP  %d rows → %s" % (len(seen), KV["OUT"]))


def feats(fen, t0):
    b = chess.Board(fen)
    lead = chess.BLACK if t0 > 0 else chess.WHITE           # Black-positive total
    pawns = b.pieces_mask(chess.PAWN, chess.WHITE) | b.pieces_mask(chess.PAWN, chess.BLACK)
    sp = len(b.pieces(chess.PAWN, lead))
    oneflank = 0 if (pawns & QS and pawns & ~QS & chess.BB_ALL) else 1
    wb, bb = b.pieces(chess.BISHOP, chess.WHITE), b.pieces(chess.BISHOP, chess.BLACK)
    others = any(b.pieces(p, c) for p in (chess.KNIGHT, chess.ROOK, chess.QUEEN) for c in (chess.WHITE, chess.BLACK))
    ocb = 0
    if len(wb) == 1 and len(bb) == 1 and not others:
        s1, s2 = next(iter(wb)), next(iter(bb))
        ocb = int(((chess.square_file(s1) + chess.square_rank(s1)) & 1) != ((chess.square_file(s2) + chess.square_rank(s2)) & 1))
    enemy = b.pieces(chess.PAWN, not lead)
    passed = 0
    for sq in b.pieces(chess.PAWN, lead):
        f, r = chess.square_file(sq), chess.square_rank(sq)
        ahead = range(r + 1, 8) if lead == chess.WHITE else range(0, r)
        if not any(chess.square(ff, rr) in enemy for ff in (f - 1, f, f + 1) if 0 <= ff < 8 for rr in ahead):
            passed += 1
    return sp, oneflank, ocb, passed


def fit():
    ours, gid = {}, {}
    for st in ("mg", "eg"):
        for p in glob.glob(os.path.join(THIS, "ks_sets", "fitC_%s_%s_d10_s*of4.csv" % (st, KV.get("OURS", "ours1004")))):
            for r in csv.DictReader(open(p, newline="")):
                ours[r["fen"]] = float(r["ours_cp_white"])
        for r in csv.DictReader(open(os.path.join(THIS, "ks_sets", "fitC_%s_sample.csv" % st), newline="")):
            gid[r["fen"]] = r["game_id"]
    dmp = {r["fen"]: r for r in csv.DictReader(open(KV["DUMP"], newline=""))}
    # phase256 is knob-independent and not always published by ev_breakdown — take it from the labelled-row export
    z = np.load("/mnt/e/chess_data/texel/px_labelled.npz")
    phz = dict(zip(z["fen"], z["phase"].astype(float)))
    T0, PH, F, sf, base, val, stm = [], [], [], [], [], [], []
    for lab in ("fitC_mg_sf18.csv", "fitC_eg_sf18.csv"):
        for r in csv.DictReader(open(os.path.join(THIS, "ks_sets", lab), newline="")):
            f = r["fen"]
            if not r.get("best_cp") or f not in ours or f not in dmp or f not in phz or abs(float(r["best_cp"])) >= 50000:
                continue
            t0 = float(dmp[f]["total"]) - float(dmp[f]["winnab"])
            T0.append(t0); PH.append(phz[f]); F.append(feats(f, t0))
            sf.append(float(r["best_cp"])); base.append(ours[f]); stm.append(1.0 if f.split()[1] == "w" else -1.0)
            val.append(int(hashlib.md5(gid.get(f, f).encode()).hexdigest()[:8], 16) % 100 < 15)
    T0, PH, F, sf, base, val, stm = map(np.array, (T0, PH, F, sf, base, val, stm))
    tgt, tr = wp(sf), ~val

    def adj(th):
        f = 64 + th[0] + th[1] * F[:, 0] + th[2] * F[:, 1] + th[3] * F[:, 2] + th[4] * F[:, 3]
        f = np.clip(f, 0, 64)
        return T0 * (256 - PH) * (f - 64) / (256.0 * 64)
    th_ship = np.array([SHIP[k] for k in ("BASE", "SP", "ONEFLANK", "OCB", "PASSED")], float)
    d_ship = adj(th_ship)

    def loss(p, m, th=None):
        th = p[:5] if th is None else th
        c = base * (1 + p[-1]) - (adj(th) - d_ship) / 10.0 + p[-2] * stm
        return float(np.mean((wp(c)[m] - tgt[m]) ** 2))
    pb = minimize(lambda q: loss(np.r_[th_ship, q], tr, th_ship), [0.0, 0.0], method="Powell").x
    vb = loss(np.r_[th_ship, pb], val, th_ship)
    scaled = 100 * (np.abs(d_ship) > 0).mean()
    print("WIN DEPTH FIT  rows %d (val %d by game) · shipped scale active on %.1f%% of rows" % (len(sf), val.sum(), scaled))
    arms = {"BASE+SP+OCB (ship knobs)": [0, 1, 3], "+ONEFLANK": [0, 1, 2, 3], "+PASSED": [0, 1, 3, 4], "ALL 5": [0, 1, 2, 3, 4]}
    for name, free in arms.items():
        def obj(q):
            th = th_ship.copy(); th[free] = q[:len(free)]
            return loss(np.r_[th, q[len(free):]], tr, th)
        res = minimize(obj, np.r_[th_ship[free], pb], method="Powell", options={"maxiter": 20000, "xtol": 1e-3, "ftol": 1e-12})
        th = th_ship.copy(); th[free] = res.x[:len(free)]
        v = loss(np.r_[th, res.x[len(free):]], val, th)
        print("  %-26s val %+6.2f%% · BASE %+.0f SP %+.0f ONEFLANK %+.0f OCB %+.0f PASSED %+.0f" % (name, 100 * (v / vb - 1), *th))


if __name__ == "__main__":
    {"dump": dump, "fit": fit}[KV.get("MODE", "fit")]()
