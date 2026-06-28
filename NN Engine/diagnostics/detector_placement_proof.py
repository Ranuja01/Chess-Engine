# -*- coding: utf-8 -*-
"""Make-or-break offline test: is the `pieces`/placement eval gap DETECTOR-explainable?

The collapse term-attribution showed the residual eval error is `pieces`/placement VARIANCE (too high in
some positions, too low in others) — not a directional bias a flat scale can fix. The proposed fix is to
condition the placement value on cheap board-state DETECTORS (phase, material, pawn structure, space) so it
adapts per position. Before any C++ build, this tests offline whether that can even work:

  Replace the placement contribution p_i with g(detectors)*p_i, g = 1 + sum_k w_k * d_k. Fit w by least
  squares to minimize the our-vs-SF gap, on a TRAIN split; measure gap MSE on a HELD-OUT split for:
    (0) baseline      : g = 1 (static placement = today)
    (1) flat scale    : g = const only (the scalar approach that WASHED before — the bar to beat)
    (2) detectors     : g = 1 + w.d (the proposed conditional placement)

If (2) beats (1) on held-out, the variance is detector-explainable -> green light for the C++ build.
If (2) ~ (1), the placement variance is NOT detector-explainable -> the approach won't help (saves a build).

Run (Windows or WSL python, no engine/SF):  python diagnostics/detector_placement_proof.py
"""
import os
import csv
import numpy as np
import chess

CORPUS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "selfplay", "tune_data", "corpus.csv")
PVAL = {chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330, chess.ROOK: 500, chess.QUEEN: 900}
CENTER = [chess.C4, chess.D4, chess.E4, chess.F4, chess.C5, chess.D5, chess.E5, chess.F5]


def _mobility(b, color):
    """Cheap mobility proxy: summed pseudo-attack squares of the side's pieces."""
    m = 0
    for pt in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN):
        for sq in b.pieces(pt, color):
            m += chess.popcount(b.attacks_mask(sq))
    return m


def _pawn_struct(b, color):
    """(doubled, isolated, passed) pawn counts for `color`."""
    pawns = b.pieces(chess.PAWN, color)
    files = [chess.square_file(s) for s in pawns]
    doubled = sum(files.count(f) - 1 for f in set(files) if files.count(f) > 1)
    isolated = sum(1 for f in files if (f - 1) not in files and (f + 1) not in files)
    opp = b.pieces(chess.PAWN, not color)
    passed = 0
    for s in pawns:
        f, r = chess.square_file(s), chess.square_rank(s)
        blocked = any(abs(chess.square_file(o) - f) <= 1 and
                      ((chess.square_rank(o) > r) if color == chess.WHITE else (chess.square_rank(o) < r))
                      for o in opp)
        passed += 0 if blocked else 1
    return doubled, isolated, passed


def _king_pressure(b, defender):
    """# of attacker-side pieces attacking the 3x3 ring around `defender`'s king (king-exposure proxy)."""
    ksq = b.king(defender)
    if ksq is None:
        return 0
    kf, kr = chess.square_file(ksq), chess.square_rank(ksq)
    ring = [chess.square(f, r) for f in range(max(0, kf - 1), min(7, kf + 1) + 1)
            for r in range(max(0, kr - 1), min(7, kr + 1) + 1)]
    cnt = 0
    for sq in ring:
        cnt += chess.popcount(b.attackers_mask(not defender, sq))
    return cnt


def detectors(fen, phase):
    """Board-state features (the candidate knobs), computed from the FEN. White-POV where signed."""
    b = chess.Board(fen)
    wm = sum(PVAL[pt] * len(b.pieces(pt, chess.WHITE)) for pt in PVAL)
    bm = sum(PVAL[pt] * len(b.pieces(pt, chess.BLACK)) for pt in PVAL)
    npawn = len(b.pieces(chess.PAWN, chess.WHITE)) + len(b.pieces(chess.PAWN, chess.BLACK))
    nqueen = len(b.pieces(chess.QUEEN, chess.WHITE)) + len(b.pieces(chess.QUEEN, chess.BLACK))
    nminor = sum(len(b.pieces(pt, c)) for pt in (chess.KNIGHT, chess.BISHOP) for c in (True, False))
    total_pc = chess.popcount(b.occupied)
    space = sum(1 for sq in CENTER if b.piece_at(sq))
    mat_edge = (wm - bm) / 100.0
    wmob, bmob = _mobility(b, chess.WHITE), _mobility(b, chess.BLACK)
    wd, wi, wp = _pawn_struct(b, chess.WHITE)
    bd, bi, bp = _pawn_struct(b, chess.BLACK)
    return {
        "phase": phase / 128.0,
        "pawns": npawn / 16.0,
        "total_pc": total_pc / 32.0,
        "mat_edge": mat_edge / 5.0,
        "abs_mat_edge": abs(mat_edge) / 5.0,
        "queens": nqueen / 2.0,
        "minors": nminor / 4.0,
        "space": space / 8.0,
        # positional detectors (the ones that SHOULD condition placement)
        "mob_diff": (wmob - bmob) / 40.0,
        "mob_total": (wmob + bmob) / 80.0,
        "doubled_diff": (wd - bd) / 4.0,
        "isolated_diff": (wi - bi) / 4.0,
        "passed_diff": (wp - bp) / 4.0,
        "kpress_diff": (_king_pressure(b, chess.BLACK) - _king_pressure(b, chess.WHITE)) / 10.0,
    }


def main():
    rows = list(csv.DictReader(open(CORPUS)))
    E, sf, p, D, ph = [], [], [], [], []
    feat_names = None
    for r in rows:
        try:
            our_cp = -float(r["our_total"]) / 10.0          # our eval, White-POV cp (our_total is Black-pos mp)
            sf_cp = float(r["sf_static_cp"])                # SF static, White-POV cp
            piece_cp = -float(r["pieces"]) / 10.0           # placement contribution, White-POV cp
            phase = float(r["phase_score"])
        except (ValueError, KeyError):
            continue
        d = detectors(r["fen"], phase)
        if feat_names is None:
            feat_names = list(d.keys())
        E.append(our_cp); sf.append(sf_cp); p.append(piece_cp); ph.append(phase)
        D.append([d[k] for k in feat_names])
    E, sf, p, D, ph = map(np.asarray, (E, sf, p, D, ph))
    gap = E - sf                                            # our over-read (White-POV cp); want -> 0
    n = len(gap)
    print(f"[proof] {n} positions  |  baseline gap: mean {gap.mean():+.1f}  std {gap.std():.1f}  "
          f"MSE {np.mean(gap**2):.0f}  |  |pieces| mean {np.abs(p).mean():.0f}cp")

    rng = np.random.default_rng(0)
    idx = rng.permutation(n); cut = int(0.7 * n)
    tr, te = idx[:cut], idx[cut:]

    def fit_and_eval(X):
        # minimize ||gap + (X w) * ... ||  -- but we model new_gap = gap + p*(Xw); see header.
        # design: column j contributes p_i * d_ij ; target y = -gap (so p*(Xw) ~ -gap)
        A = p[tr, None] * X[tr]
        w, *_ = np.linalg.lstsq(A, -gap[tr], rcond=None)
        new_gap_te = gap[te] + (p[te] * (X[te] @ w))
        new_gap_tr = gap[tr] + (p[tr] * (X[tr] @ w))
        return np.mean(new_gap_tr**2), np.mean(new_gap_te**2), w

    def report(mask_te, label):
        m = mask_te
        base = np.mean(gap[te][m] ** 2)
        Xc = np.ones((n, 1)); _, _, wc = fit_and_eval(Xc)
        Xd = np.column_stack([np.ones(n), D]); _, _, wd = fit_and_eval(Xd)
        f = np.mean((gap[te] + p[te] * (Xc[te] @ wc))[m] ** 2)
        d = np.mean((gap[te] + p[te] * (Xd[te] @ wd))[m] ** 2)
        print(f"\n[proof] {label}  (n_te={m.sum()})  held-out gap MSE:")
        print(f"   (0) baseline static : {base:8.0f}")
        print(f"   (1) flat scale      : {f:8.0f}  ({100*(1-f/base):+.1f}%)  [scale={1+wc[0]:.3f}]")
        print(f"   (2) detectors       : {d:8.0f}  ({100*(1-d/base):+.1f}%)   >> beyond flat: {100*(1-d/f):+.1f}%")
        return wd

    all_mask = np.ones(len(te), bool)
    near_mask = np.abs(sf[te]) < 150          # near-equal: where play strength actually lives
    # important MIDGAME: not endgame + decision-critical band (near-equal or a slight edge to hold)
    mid_imp = (ph[te] < 64) & (np.abs(sf[te]) < 250)
    report(all_mask, "ALL positions")
    report(near_mask, "NEAR-EQUAL (|SF|<150cp)")
    wd = report(mid_imp, "IMPORTANT MIDGAME (phase<64 & |SF|<250) — the curated target")
    print(f"\n[proof] fitted detector weights (gain = 1 + w.d, d normalized, full-corpus fit):")
    for name, wv in zip(["const"] + feat_names, wd):
        print(f"     {name:14} {wv:+.3f}")


if __name__ == "__main__":
    main()
