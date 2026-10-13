# -*- coding: utf-8 -*-
"""POT GROUNDWORK (2026-10-13; owner: POT designed on what the RETUNED eval still gets wrong, nested on top, rest frozen).

  pyrun diagnostics/_pot_probe.py FEAT=feat_gen3 D10_GLOB='A_ours1012_d10_s*of4.csv' [LAMBDA=0.03]

(1) RESIDUAL MAP of the current ship on held-out own-play games: static and depth (our d10 search) win% MSE and SIDE-AHEAD BIAS
    (mean of (win%(ours) − win%(SF18)) × sign(SF18), |SF18| > 0.25 pawn; negative = under-confident) by stage × endgame type ×
    pawns-up. (2) CANDIDATE POT INPUTS, each fitted ALONE on top of the frozen ship (the nested-POT rule as a fit-only screen; the
    _retune_fit hybrid objective, static on quiet rows + depth on slice A), weighted by the endgame leg (256 − phase)/256:
      DIRECTION (potential, Black − White): passers · most advanced passer rank · protected passers · outside passers ·
        candidate (majority) wings · king centralisation · king-to-passer race (own king near own passers, enemy king far)
      MAGNITUDE (complexity, SF12+ winnable() inputs) as sign(eval) × input — a positive weight = MORE confident: pawn count ·
        pawns on both flanks · king outflanking · king infiltration · pure pawn ending · passer count
    Reads: val change (static / depth, all + EG stages) and the side-ahead bias per endgame type before → after. Fit-only:
    nothing here touches the engine. ☠️ A probe that only moves |eval| without fixing a type's bias is magnitude noise.
"""
import os, sys
import numpy as np
import chess
import scipy.sparse as sp

THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS)
KV = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
sys.argv = [sys.argv[0]] + ["%s=%s" % kv for kv in KV.items()] + ["EXPERIMENT=hybrid"]
import _retune_fit as R

wp = R.wp


def eg_type(b):
    sets = [tuple(len(b.pieces(p, c)) for p in (chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT)) for c in (True, False)]
    q = sum(s[0] for s in sets); r = sum(s[1] for s in sets); mn = sum(s[2] + s[3] for s in sets)
    if sets[0] != sets[1] and not (sets[0][0] == sets[1][0] and sets[0][1] == sets[1][1] and sets[0][2] + sets[0][3] == sets[1][2] + sets[1][3]):
        t = "mixed/imbalanced"
    elif q:
        t = "queen"
    elif r and mn:
        t = "rook+minor"
    elif r:
        t = "rook"
    elif mn:
        t = "minor"
    else:
        t = "pawn"
    dp = len(b.pieces(chess.PAWN, True)) - len(b.pieces(chess.PAWN, False))
    return t + (" ·up" if dp else " ·eq")


def passers(b, color):
    out = []
    for sq in b.pieces(chess.PAWN, color):
        f, r = chess.square_file(sq), chess.square_rank(sq)
        ahead = range(r + 1, 8) if color else range(0, r)
        blocked = any(b.piece_at(chess.square(ff, rr)) == chess.Piece(chess.PAWN, not color)
                      for ff in (f - 1, f, f + 1) if 0 <= ff < 8 for rr in ahead)
        if not blocked:
            out.append(sq)
    return out


def features(fen, base_cp):
    b = chess.Board(fen)
    side = {}
    for c in (True, False):
        ps = passers(b, c)
        rel = [chess.square_rank(s) if c else 7 - chess.square_rank(s) for s in ps]
        prot = sum(1 for s in ps if any(b.piece_at(a) == chess.Piece(chess.PAWN, c) for a in b.attackers(c, s)))
        pawns = list(b.pieces(chess.PAWN, c)); files = [chess.square_file(s) for s in pawns]
        outside = sum(1 for s in ps if files and min(abs(chess.square_file(s) - f) for f in
                      [chess.square_file(x) for x in b.pieces(chess.PAWN, not c)] or [chess.square_file(s) + 8]) >= 3)
        q_c = sum(1 for f in files if f <= 3); k_c = len(files) - q_c
        q_o = sum(1 for s in b.pieces(chess.PAWN, not c) if chess.square_file(s) <= 3); k_o = len(b.pieces(chess.PAWN, not c)) - q_o
        majority = int(q_c > q_o) + int(k_c > k_o)
        k = b.king(c); ok = b.king(not c)
        cent = -max(abs(chess.square_file(k) - 3.5), abs(chess.square_rank(k) - 3.5)) if k is not None else 0.0
        race = 0.0
        for s in ps:
            race += (chess.square_distance(ok, s) - chess.square_distance(k, s)) if (k is not None and ok is not None) else 0
        side[c] = [len(ps), max(rel) if rel else 0, prot, outside, majority, cent, race]
    dir_ = [side[False][i] - side[True][i] for i in range(7)]                       # Black − White (model convention)
    allp = list(b.pieces(chess.PAWN, True)) + list(b.pieces(chess.PAWN, False))
    files = [chess.square_file(s) for s in allp]
    both = int(any(f <= 3 for f in files) and any(f >= 4 for f in files))
    wk, bk = b.king(True), b.king(False)
    outfl = (abs(chess.square_file(wk) - chess.square_file(bk)) - abs(chess.square_rank(wk) - chess.square_rank(bk))) if wk is not None and bk is not None else 0
    infil = int(wk is not None and chess.square_rank(wk) > 3) + int(bk is not None and chess.square_rank(bk) < 4)
    pure = int(all(len(b.pieces(p, c)) == 0 for p in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN) for c in (True, False)))
    npass = side[True][0] + side[False][0]
    s = np.sign(base_cp)                                                            # White-POV sign of the ship's eval
    mag = [-s * v for v in (len(allp), both, outfl, infil, pure, npass)]            # −sign·input: +δ ⇒ |eval| grows
    return dir_, mag, eg_type(b)


GROUPS = {"D passers (count, adv rank, protected, outside)": ("dir", [0, 1, 2, 3]),
          "D majority wings": ("dir", [4]), "D king centralisation": ("dir", [5]), "D king-passer race": ("dir", [6]),
          "D ALL direction": ("dir", list(range(7))),
          "M complexity (SF winnable inputs)": ("mag", list(range(6))),
          "D+M everything": ("both", None)}


def main():
    X0, names, meta = R.load()
    role, st = meta["role"], meta["stage"]
    own = np.isin(role, ["tr_own", "va_own"])
    idx = np.flatnonzero(own)
    print("  computing candidate features on %d own-play rows …" % len(idx), flush=True)
    D = np.zeros((len(role), 7), np.float32); M = np.zeros((len(role), 6), np.float32); typ = np.array([""] * len(role), dtype=object)
    for i in idx:
        d, m, t = features(meta["fen"][i], meta["static"][i])
        D[i] = d; M[i] = m; typ[i] = t
    egw = ((256.0 - meta["phase"]) / 256.0)[:, None].astype(np.float32)
    D *= egw; M *= egw
    quiet, sf, stm = meta["quiet"], meta["sf"], meta["stm"]
    okd = (meta["slice"] == "A") & ~np.isnan(meta["d10"]) & (np.abs(sf) < 50000) & (np.abs(np.nan_to_num(meta["d10"])) < 4999)
    va = role == "va_own"
    egrows = np.isin(st, ["egl", "eg"])

    def bias_mse(x, m):
        if m.sum() < 30:
            return float("nan"), float("nan")
        t = wp(sf[m]); e = wp(x[m]) - t
        big = np.abs(sf[m]) > 25
        return float(np.mean(e * e)), float(np.mean((e * np.sign(sf[m]))[big])) if big.any() else float("nan")

    print("\n(1) RESIDUAL MAP — held-out own-play, endgame-leaning + endgame (static on quiet rows / depth on slice A)")
    print("  %-22s %6s %16s %6s %16s" % ("type", "n st", "static MSE(bias)", "n dp", "depth MSE(bias)"))
    for t in sorted(set(typ[va & egrows])):
        ms = va & egrows & quiet & (typ == t); md = va & egrows & okd & (typ == t)
        a, ab = bias_mse(meta["static"], ms); c, cb = bias_mse(np.nan_to_num(meta["d10"]), md)
        print("  %-22s %6d %8.1f (%+5.1f) %6d %8.1f (%+5.1f)" % (t, ms.sum(), a, ab, md.sum(), c, cb))
    lam = float(KV.get("LAMBDA", 0.03))
    ms = own & (role == "tr_own") & quiet; ws_rows = np.ones(ms.sum())
    md = (role == "tr_own") & okd
    print("\n(2) CANDIDATE POT INPUTS, each alone on top of the frozen ship (λ %g) — val change vs the ship (negative = better)" % lam)
    print("  %-46s %8s %8s %8s %8s   weights" % ("group", "st all", "st EG", "dp all", "dp EG"))
    for name, (kind, cols) in GROUPS.items():
        if kind == "both":
            Xc = np.c_[D, M]
        elif kind == "dir":
            Xc = D[:, cols]
        else:
            Xc = M[:, cols]
        X = sp.csr_matrix(Xc)
        p, info = R.fit_hybrid(X, meta, ms, ws_rows, md, X, lam, 1, 1)
        k = X.shape[1]; move = (X @ p[:k]) / 10.0
        out = []
        for base, mask in ((meta["static"], va & quiet), (meta["static"], va & quiet & egrows),
                           (np.nan_to_num(meta["d10"]), va & okd), (np.nan_to_num(meta["d10"]), va & okd & egrows)):
            b0 = float(np.mean((wp(base[mask]) - wp(sf[mask])) ** 2)); b1 = float(np.mean((wp(base[mask] - move[mask]) - wp(sf[mask])) ** 2))
            out.append(100 * (b1 / b0 - 1))
        print("  %-46s %+7.2f%% %+7.2f%% %+7.2f%% %+7.2f%%   %s" % (name, *out, " ".join("%+.0f" % w for w in p[:k])))
        if kind == "both":
            print("\n  side-ahead bias per endgame type, depth target (held-out): ship → ship + everything")
            for t in sorted(set(typ[va & egrows])):
                mm = va & egrows & okd & (typ == t)
                _, b0 = bias_mse(np.nan_to_num(meta["d10"]), mm); _, b1 = bias_mse(np.nan_to_num(meta["d10"]) - move, mm)
                print("    %-22s %+5.1f → %+5.1f" % (t, b0, b1))


if __name__ == "__main__":
    main()
