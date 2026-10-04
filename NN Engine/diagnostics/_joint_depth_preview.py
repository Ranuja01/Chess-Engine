# -*- coding: utf-8 -*-
"""STEP 0 (owner, 10-04): is SF11's remaining STATIC edge on the passer corpora WEIGHTING or MISSING KNOWLEDGE?

On `_passer_corpus_check.py`'s corpora SF11 static beats ours (passer_corpus 9.94 vs 12.01; under_fire 14.79 vs 18.63;
suite 9.38 vs 11.02) although our d10 search beats it easily. Diagnostic only — static closeness is not the goal, Elo is.

(a) JOINT DEPTH FIT — a preview of the final retune: every LINEAR v2 term fitted together on the depth target
    (SF18 d14 vs our d10 SEARCH of the current ship, mg + eg labelled sets; the _px_depth_fit.py model):
      PST (6 pieces × 32 file-mirror-tied squares × mg/eg)  ·  all 235 v2_features columns × mg/eg (mobility, pawns,
      passers, placement, KS-B shelter/storm, C3-b/c, PX)  ·  Kaufman census cells + N/B/R/Q value corrections
      (untapered, `_texel_kauf_fit.feats`)  ·  + an STM nuisance (not applied).
    Then apply δ to our SHIPPED static eval on the corpora: static' = static − Σ diff·δ/10 (rows with flags & 3 — draw
    classifier / tier-2b — keep the shipped value: the linear model does not hold there).
(b-control) IN-SAMPLE CEILING — the same features re-weighted on the corpora's OWN labels from the static base, 5-fold
    CV by FEN hash, λ swept: the most reweighting alone could ever buy here (optimistic: in-domain).
Reading: (a) closes most of the gap ⇒ WEIGHTING (the final retune's job). Neither (a) nor the ceiling closes it ⇒ our
feature set cannot express it ⇒ MISSING KNOWLEDGE ⇒ step 0(b) (triangulate under_fire vs SF11's term table).
GUARD: the shipped static row must reproduce passer design §10 (12.01 / 18.63 / 11.02) or the run aborts.

  pyrun diagnostics/_joint_depth_preview.py V2_PRESET=shipped [LAMBDA=1e-3,1e-2,1e-1]
"""
import os, sys, csv, glob, hashlib
import numpy as np

for _a in sys.argv[1:]:
    if "=" in _a:
        _k, _v = _a.split("=", 1)
        os.environ[_k] = _v
THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS); sys.path.insert(0, os.path.dirname(THIS))
os.environ.setdefault("PRESET", "LONG_FORMAT"); os.environ.setdefault("USE_OPENING_BOOK", "0")
import chess
from scipy.optimize import minimize
import _texel_kauf_fit as KF

DATA = "/mnt/e/chess_data/texel"
K = 0.00368208
NV = 235
NPST = 6 * 32
NKF = len(KF.CELLS)
wp = lambda cp: 100.0 / (1.0 + np.exp(-K * np.clip(cp, -1500, 1500)))


def pst_kauf(fen):
    """Black − White PST occupancy (relative square, file-folded a=h …) and −Kaufman feats (Black − White convention)."""
    b = chess.Board(fen)
    occ = np.zeros(NPST)
    for sq, pc in b.piece_map().items():
        rel = sq if pc.color == chess.WHITE else sq ^ 56
        f = chess.square_file(rel)
        occ[(pc.piece_type - 1) * 32 + chess.square_rank(rel) * 4 + min(f, 7 - f)] += 1 if pc.color == chess.BLACK else -1
    return occ, -np.array(KF.feats(b), float)


def design(vdiff, ph, occ, kf):
    """Columns: [v2 mg | v2 eg | PST mg | PST eg | Kaufman] — each a Black − White count, mg/eg blended by phase256."""
    a, e = (ph / 256.0)[:, None], ((256.0 - ph) / 256.0)[:, None]
    return np.concatenate([vdiff * a, vdiff * e, occ * a, occ * e, kf], 1)


def fit_delta(X, base, tgt, stm, tr, lam):
    def cp(p):
        return base - (X @ p[:-1]) / 10.0 + p[-1] * stm
    def fg(p):
        c = cp(p)[tr]; q = wp(c); r = q - tgt[tr]
        g = 2.0 * r * q * (1 - q / 100.0) * K * (np.abs(c) < 1500)
        gw = -(X[tr].T @ g) / tr.sum() / 10.0
        return float(np.mean(r * r)) + lam * float(p[:-1] @ p[:-1]) / 1e4, \
            np.r_[gw + 2 * lam * p[:-1] / 1e4, float((g * stm[tr]).mean())]
    res = minimize(fg, np.zeros(X.shape[1] + 1), jac=True, method="L-BFGS-B", options={"maxiter": 5000})
    return res.x, cp


def load_depth():
    ours = {}
    for pre in ("fitC_mg_ours1003_d10", "fitC_eg_ours1003_d10"):
        for p in glob.glob(os.path.join(THIS, "ks_sets", pre + "_s*of4.csv")):
            for r in csv.DictReader(open(p, newline="")):
                ours[r["fen"]] = float(r["ours_cp_white"])
    z = np.load(os.path.join(DATA, "px_labelled.npz"))
    idx = {f: i for i, f in enumerate(z["fen"])}
    rows, fens, sf, base, val, stm = [], [], [], [], [], []
    for lab in ("fitC_mg_sf18.csv", "fitC_eg_sf18.csv"):
        for r in csv.DictReader(open(os.path.join(THIS, "ks_sets", lab), newline="")):
            f = r["fen"]
            if not r.get("best_cp") or f not in ours or f not in idx or abs(float(r["best_cp"])) >= 50000:
                continue
            if z["flags"][idx[f]] & 3:
                continue
            rows.append(idx[f]); fens.append(f); sf.append(float(r["best_cp"])); base.append(ours[f])
            val.append(int(hashlib.md5(f.encode()).hexdigest()[:8], 16) % 100 < 15)
            stm.append(1.0 if f.split()[1] == "w" else -1.0)
    rows = np.array(rows)
    pk = [pst_kauf(f) for f in fens]
    X = design(z["diff"][rows].astype(np.float64), z["phase"][rows].astype(np.float64),
               np.array([p for p, _ in pk]), np.array([k for _, k in pk]))
    return X, np.array(base), wp(np.array(sf)), np.array(val), np.array(stm)


def corpora_rows():
    import ChessAI
    ai = ChessAI.ChessAI(None, None, chess.Board(), True)
    assert ChessAI.V2F_PER_SIDE == NV
    out = []
    srcs = [("passer_corpus", r["tier"], r["fen"], 100.0 * float(r["sf18"]))
            for r in csv.DictReader(open(os.path.join(THIS, "ks_sets/passer_corpus.csv"), newline=""))
            if r.get("sf18") not in ("", None)]
    srcs += [("passers_suite", r["cat"], r["fen_start"], float(r["sf_cp"]))
             for r in csv.DictReader(open(os.path.join(THIS, "suites/passers.csv"), newline=""))
             if r.get("sf_cp") not in ("", None)]
    for corpus, grp, fen, sf in srcs:
        b = chess.Board(fen)
        eb = ai.ev_breakdown(b)
        wc, bc, fl = ChessAI.v2_feature_counts(b)
        occ, kf = pst_kauf(fen)
        out.append((corpus, grp, fen, sf, -float(eb["total"]) / 10.0, int(eb.get("v2_phase256", -1)), fl,
                    np.array([y - x for x, y in zip(wc, bc)], float), occ, kf))
    return out


def report(label, C, ev):
    groups = {}
    for (corpus, grp, *_), v, s in zip(C, ev, [c[3] for c in C]):
        g = abs(wp(v) - wp(s))
        for key in ((corpus, "ALL"), (corpus, grp)):
            groups.setdefault(key, []).append(g)
    res = {}
    for (corpus, grp), gs in sorted(groups.items()):
        res[(corpus, grp)] = float(np.mean(gs))
        print("JDP %-12s %-14s %-14s n %4d  mean|gap| %6.2f pp" % (label, corpus, grp, len(gs), res[(corpus, grp)]))
    return res


def main():
    lams = [float(x) for x in os.environ.get("LAMBDA", "1e-3,1e-2,1e-1").split(",")]
    # --- the corpora first: the guard must pass before anything is fitted
    C = corpora_rows()
    static = np.array([c[4] for c in C])
    base = report("SHIPPED", C, static)
    want = {("passer_corpus", "ALL"): 12.01, ("passer_corpus", "under_fire"): 18.63, ("passers_suite", "ALL"): 11.02}
    bad = [(k, base[k], v) for k, v in want.items() if abs(base[k] - v) > 0.02]
    if bad:
        print("☠️ GUARD: shipped static does not reproduce passer design §10:", bad); sys.exit(1)
    print("GUARD OK: shipped static reproduces §10 (12.01 / 18.63 / 11.02)")
    flagged = np.array([c[6] & 3 != 0 for c in C])
    XC = design(np.array([c[7] for c in C]), np.array([c[5] for c in C], float),
                np.array([c[8] for c in C]), np.array([c[9] for c in C]))
    print("corpora rows %d · draw/tier-2b flagged (left at shipped) %d" % (len(C), flagged.sum()))

    # --- (a) the joint depth fit (JOINT=0 skips it: ceiling-only re-runs)
    if os.environ.get("JOINT", "1") == "0":
        lams = []
    if lams:
        X, b0, tgt, val, stm = load_depth()
        tr = ~val
        print("JOINT DEPTH FIT  rows %d (val %d) · params %d (v2 %d×2 · PST %d×2 · Kaufman %d)"
              % (len(b0), val.sum(), X.shape[1], NV, NPST, NKF))
        p0, cp = fit_delta(X, b0, tgt, stm, tr, 1e9)        # λ→∞: the STM-nuisance-only baseline
        vb = float(np.mean((wp(cp(p0)[val]) - tgt[val]) ** 2))
    for lam in lams:
        p, cp = fit_delta(X, b0, tgt, stm, tr, lam)
        vl = float(np.mean((wp(cp(p)[val]) - tgt[val]) ** 2))
        d = p[:-1]
        blk = {"v2": np.r_[d[:NV], d[NV:2 * NV]], "PST": d[2 * NV:2 * NV + 2 * NPST], "Kauf": d[2 * NV + 2 * NPST:]}
        print("  λ=%-6g depth val %+.2f%% · |δ| rms v2 %.0f PST %.0f Kauf %.0f mp · STM %.1f cp"
              % (lam, 100 * (vl / vb - 1), *(float(np.sqrt(np.mean(v ** 2))) for v in blk.values()), p[-1]))
        ev = np.where(flagged, static, static - (XC @ d) / 10.0)
        report("JOINT_%g" % lam, C, ev)

    # --- (b-control) in-sample reweighting ceiling on the corpora's own labels
    sfC = wp(np.array([c[3] for c in C]))
    fold = np.array([int(hashlib.md5(c[2].encode()).hexdigest()[:8], 16) % 5 for c in C])
    stmC = np.array([1.0 if c[2].split()[1] == "w" else -1.0 for c in C])
    XCf = np.where(flagged[:, None], 0.0, XC)
    for lam in [float(x) for x in os.environ.get("CEIL_LAMBDA", "1e-1,1,10").split(",")]:
        ev = static.copy()
        for k in range(5):
            trk = fold != k
            p, cp = fit_delta(XCf, static, sfC, stmC, trk, lam)
            ev[~trk] = static[~trk] - (XCf[~trk] @ p[:-1]) / 10.0      # STM nuisance not applied
        report("CEIL_%g" % lam, C, ev)


if __name__ == "__main__":
    main()
