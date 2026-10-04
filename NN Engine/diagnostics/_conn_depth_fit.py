# -*- coding: utf-8 -*-
"""CONNECTED PAWNS on the DEPTH target (2026-10-04; owner chose this before the search retune).

Why: the step-1 triangulation on the 10-03 ship (TRIANGULATION note, end) found SF11's Pawns term pointing toward SF18
in 18 of 60 quiet cases vs 2 against. v2's connected term exists (eval_v2.cpp pawn_structure_mp, SF11 rank table ×
(2 + phalanx − opposed)/2 × Ethereal file tilt at rank ≥ 6, + PS_V2_SUPPORT per supporter) but ships at
PS_V2_CONN_MAG=0 — closed 09-12 as "harmful at every magnitude" on corpus MSE, never measured on the depth target, and
never a column of v2_features (so the 10-04 pawn re-price could not price it).

Model (the _pawn_depth_fit.py one): ours' = ours_d10 (our d10 SEARCH of the 10-03 ship, White cp) − Δ/10 + STM nuisance,
Δ = Black − White connected score in mp, phase-blended (ph·mg + (256−ph)·eg)/256. Target SF18 d14 (mg + eg labelled
sets), val by FEN hash 15% — the _px_depth_fit.py split. Connected is 0 today ⇒ any fitted value is the whole term.
ARMS:
  FREE  — per relative rank 1-6: connected count, phalanx count, opposed count (+ supporter count), mg and eg free:
          what connected-pawn INFORMATION is worth on this target (an upper bound for the engine's form).
  KNOBS — the engine's EXACT form with its three env knobs free: PS_V2_CONN_MAG (m), PS_V2_SUPPORT (s), PS_V2_EG_RATIO (e):
          mg = m/100·Σ(v),  eg = m/100·e/100·Σ(v·(r−2)/4),  v = base·mod/256 + s·nsup  (FORM 0, EXCL 0, tilt 256 / r ≥ 5).
          Shippable as env settings with NO rebuild (then closure + gates).
Compare: the 10-04 pawn STRUCT arm read −0.47% (C3 §19b).

  pyrun diagnostics/_conn_depth_fit.py [LAMBDA=1e-2]
CLOSURE (knobs latch at init ⇒ one process per setting): dump the engine's pawn_struct with the knobs ON and OFF, then the
ON − OFF difference must equal the exact integer emulation of the C++ connected term (±3 mp: per-side >>8 blend rounding):
  pyrun diagnostics/_conn_depth_fit.py MODE=dump OUT=/tmp/c_on.csv  V2_PRESET=shipped PS_V2_CONN_MAG=21 PS_V2_SUPPORT=99 PS_V2_EG_RATIO=101
  pyrun diagnostics/_conn_depth_fit.py MODE=dump OUT=/tmp/c_off.csv V2_PRESET=shipped
  pyrun diagnostics/_conn_depth_fit.py MODE=closure ON=/tmp/c_on.csv OFF=/tmp/c_off.csv CONN_MAG=21 SUPPORT=99 EG_RATIO=101
"""
import os, sys, csv, glob, hashlib
import numpy as np
import chess
from scipy.optimize import minimize

KV = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
THIS = os.path.dirname(os.path.abspath(__file__))
DATA = "/mnt/e/chess_data/texel"
K = 0.00368208
CONN_RANK = [0, 109, 125, 187, 453, 750, 1343, 0]          # eval_v2.cpp PS_CONN_RANK_MP
FILE_256 = [143, 284, 287, 309, 309, 287, 284, 143]         # PS_CONN_FILE_256
TILT, TILT_MIN_RANK = 256, 5
wp = lambda cp: 100.0 / (1.0 + np.exp(-K * np.clip(cp, -1500, 1500)))


def side_feats(b, color):
    """Per-pawn connected features for one side (relative ranks). Returns (free[19], (B, N, Be, Ne))."""
    own = b.pieces(chess.PAWN, color)
    enemy = b.pieces(chess.PAWN, not color)
    free = np.zeros(19)
    B = N = Be = Ne = 0.0
    for sq in own:
        f, r = chess.square_file(sq), chess.square_rank(sq)
        rr = r if color == chess.WHITE else 7 - r
        phal = any(chess.square(ff, r) in own for ff in (f - 1, f + 1) if 0 <= ff < 8)
        dr = -1 if color == chess.WHITE else 1                    # supporters sit one rank BEHIND
        sups = sum(1 for ff in (f - 1, f + 1) if 0 <= ff < 8 and 0 <= r + dr < 8 and chess.square(ff, r + dr) in own)
        if not (phal or sups):
            continue
        ahead = range(r + 1, 8) if color == chess.WHITE else range(0, r)
        opp = any(chess.square(f, rk) in enemy for rk in ahead)
        k = rr - 1                                                # rr 1..6 → 0..5
        free[k] += 1; free[6 + k] += phal; free[12 + k] += opp; free[18] += sups
        base = CONN_RANK[rr]
        if rr >= TILT_MIN_RANK:
            base = (base * (256 + ((FILE_256[f] - 256) * TILT) // 256)) >> 8
        mod = 256 + 128 * phal - 128 * opp
        v0 = (base * mod) >> 8
        B += v0; N += sups; Be += v0 * (rr - 2) / 4.0; Ne += sups * (rr - 2) / 4.0
    return free, np.array([B, N, Be, Ne])


def main():
    ours = {}
    for pre in ("fitC_mg_ours1003_d10", "fitC_eg_ours1003_d10"):
        for p in glob.glob(os.path.join(THIS, "ks_sets", pre + "_s*of4.csv")):
            for r in csv.DictReader(open(p, newline="")):
                ours[r["fen"]] = float(r["ours_cp_white"])
    z = np.load(os.path.join(DATA, "px_labelled.npz"))
    idx = {f: i for i, f in enumerate(z["fen"])}
    F, G, ph, sf, base, val, stm = [], [], [], [], [], [], []
    for lab in ("fitC_mg_sf18.csv", "fitC_eg_sf18.csv"):
        for r in csv.DictReader(open(os.path.join(THIS, "ks_sets", lab), newline="")):
            f = r["fen"]
            if not r.get("best_cp") or f not in ours or f not in idx or abs(float(r["best_cp"])) >= 50000:
                continue
            if z["flags"][idx[f]] & 3:
                continue
            b = chess.Board(f)
            fw, gw = side_feats(b, chess.WHITE)
            fb, gb = side_feats(b, chess.BLACK)
            F.append(fb - fw); G.append(gb - gw)
            ph.append(float(z["phase"][idx[f]])); sf.append(float(r["best_cp"])); base.append(ours[f])
            val.append(int(hashlib.md5(f.encode()).hexdigest()[:8], 16) % 100 < 15)
            stm.append(1.0 if f.split()[1] == "w" else -1.0)
    F, G, ph, sf, base, val, stm = map(np.array, (F, G, ph, sf, base, val, stm))
    a, e = ph / 256.0, (256.0 - ph) / 256.0
    tgt, tr = wp(sf), ~val
    print("CONNECTED DEPTH FIT  rows %d (val %d) · rows with a connected-count difference %.1f%%"
          % (len(sf), val.sum(), 100 * (np.abs(F[:, :6]).sum(1) > 0).mean()))

    def loss_of(delta_fn, p, m):
        """☠️ Score each arm with ITS OWN delta_fn — a closure returned from the baseline fit scored every arm as the
        baseline (all arms read ±0.00%, caught 2026-10-04 before any verdict)."""
        return float(np.mean((wp(base - delta_fn(p) / 10.0 + p[-1] * stm)[m] - tgt[m]) ** 2))

    def run(delta_fn, p0, lam_idx, lam):
        """Fit params p (+ STM as the last entry); delta_fn(p) = Black − White mp."""
        obj = lambda p: loss_of(delta_fn, p, tr) + lam * float(p[lam_idx] @ p[lam_idx]) / 1e4
        res = minimize(obj, p0, method="L-BFGS-B", options={"maxiter": 5000})
        return res.x, (lambda q, m: loss_of(delta_fn, q, m))

    # baseline: STM nuisance only
    p_b, loss_b = run(lambda p: 0.0 * base, np.zeros(1), slice(0, 0), 0.0)
    vb = loss_b(p_b, val)
    lam = float(KV.get("LAMBDA", 1e-2))

    X = np.concatenate([F * a[:, None], F * e[:, None]], 1)          # FREE: 19 mg + 19 eg
    p_f, loss_f = run(lambda p: X @ p[:-1], np.zeros(X.shape[1] + 1), slice(0, X.shape[1]), lam)
    print("  FREE  (38 params, λ=%g)   val %+.2f%%" % (lam, 100 * (loss_f(p_f, val) / vb - 1)))
    names = ["conn_r%d" % r for r in range(1, 7)] + ["phal_r%d" % r for r in range(1, 7)] + \
        ["opp_r%d" % r for r in range(1, 7)] + ["supporters"]
    for i, n in enumerate(names):
        if abs(p_f[i]) >= 30 or abs(p_f[19 + i]) >= 30:
            print("      %-10s mg %+5.0f  eg %+5.0f mp" % (n, p_f[i], p_f[19 + i]))

    def knobs(p):                    # p = [m, s, e, STM] — m, e in percent, s in mp
        m, s, eg = p[0] / 100.0, p[1], p[2] / 100.0
        mg = m * (G[:, 0] + s * G[:, 1])
        egl = m * eg * (G[:, 2] + s * G[:, 3])
        return a * mg + e * egl
    best = None
    for m0 in (25.0, 50.0, 100.0):
        p_k, _ = run(knobs, np.array([m0, 164.0, 60.0, p_b[-1]]), slice(0, 0), 0.0)
        if best is None or loss_of(knobs, p_k, tr) < loss_of(knobs, best, tr):
            best = p_k
    p_k = best
    print("  KNOBS (3 params)          val %+.2f%%   PS_V2_CONN_MAG=%.0f PS_V2_SUPPORT=%.0f PS_V2_EG_RATIO=%.0f"
          % (100 * (loss_of(knobs, p_k, val) / vb - 1), p_k[0], p_k[1], p_k[2]))
    for m in (100.0,):
        p_s = np.array([m, 164.0, 60.0, p_b[-1]])
        print("  SHIP-DEFAULT form at CONN_MAG=100 (unfitted)  val %+.2f%%" % (100 * (loss_of(knobs, p_s, val) / vb - 1)))
    if abs(loss_of(knobs, np.array([100.0, 164.0, 60.0, p_b[-1]]), val) - vb) < 1e-12:
        print("☠️ GUARD: the unfitted form scores identical to baseline — the scorer is not seeing the feature"); sys.exit(1)


def tdiv(a, b):
    """C integer division (truncates toward zero)."""
    q = abs(a) // abs(b)
    return q if (a >= 0) == (b > 0) else -q


def conn_side_int(b, color, m, sup_mp, egr):
    """Exact integer emulation of eval_v2.cpp pawn_structure_mp's connected loop (FORM 0, EXCL 0) for one side."""
    own = b.pieces(chess.PAWN, color)
    enemy = b.pieces(chess.PAWN, not color)
    smg = seg = 0
    for sq in own:
        f, r = chess.square_file(sq), chess.square_rank(sq)
        rr = r if color == chess.WHITE else 7 - r
        phal = any(chess.square(ff, r) in own for ff in (f - 1, f + 1) if 0 <= ff < 8)
        dr = -1 if color == chess.WHITE else 1
        sups = sum(1 for ff in (f - 1, f + 1) if 0 <= ff < 8 and 0 <= r + dr < 8 and chess.square(ff, r + dr) in own)
        if not (phal or sups):
            continue
        opp = any(chess.square(f, rk) in enemy for rk in (range(r + 1, 8) if color == chess.WHITE else range(0, r)))
        base = CONN_RANK[rr]
        if rr >= TILT_MIN_RANK:
            base = (base * (256 + tdiv((FILE_256[f] - 256) * TILT, 256))) >> 8
        mod = 256 + 128 * phal - 128 * opp
        v = (base * mod) >> 8
        v += sup_mp * sups
        if m != 100:
            v = tdiv(v * m, 100)
        smg += v
        seg += tdiv(tdiv(v * (rr - 2), 4) * egr, 100)
    return smg, seg


def dump():
    for k, v in KV.items():                     # the knobs reach the engine via env (latched at init)
        os.environ[k] = v
    os.environ.setdefault("PRESET", "LONG_FORMAT"); os.environ.setdefault("USE_OPENING_BOOK", "0")
    sys.path.insert(0, os.path.dirname(THIS))
    import ChessAI
    ai = ChessAI.ChessAI(None, None, chess.Board(), True)
    n = int(KV.get("N", 3000))
    with open(KV["OUT"], "w", newline="") as fo:
        w = csv.writer(fo); w.writerow(["fen", "pawn_struct", "phase256"])
        for i, r in enumerate(csv.DictReader(open(os.path.join(THIS, "ks_sets", "fitC_eg_sf18.csv"), newline=""))):
            if i >= n:
                break
            b = chess.Board(r["fen"])
            eb = ai.ev_breakdown(b)
            w.writerow([r["fen"], int(eb["pawn_struct"]), int(eb["v2_phase256"])])
    print("CONN DUMP  %d rows → %s" % (min(i + 1, n), KV["OUT"]))


def closure():
    m, sp, egr = int(KV["CONN_MAG"]), int(KV["SUPPORT"]), int(KV["EG_RATIO"])
    on = {r["fen"]: r for r in csv.DictReader(open(KV["ON"], newline=""))}
    off = {r["fen"]: r for r in csv.DictReader(open(KV["OFF"], newline=""))}
    n = live = bad = 0
    worst = 0
    for f, r in on.items():
        if f not in off:
            continue
        b = chess.Board(f)
        ph = int(r["phase256"])
        wm, we = conn_side_int(b, chess.WHITE, m, sp, egr)
        bm, be = conn_side_int(b, chess.BLACK, m, sp, egr)
        model = ((bm * ph + be * (256 - ph)) >> 8) - ((wm * ph + we * (256 - ph)) >> 8)
        eng = int(r["pawn_struct"]) - int(off[f]["pawn_struct"])
        n += 1; live += eng != 0; bad += abs(eng - model) > 3; worst = max(worst, abs(eng - model))
    ok = bad == 0 and live > n // 4 and n > 0
    print("CONN CLOSURE  rows %d · live %d · mismatches(>3mp) %d · worst %d mp  VERDICT %s"
          % (n, live, bad, worst, "EXACT" if ok else "☠️ DIVERGES"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    {"dump": dump, "closure": closure}.get(KV.get("MODE"), main)()
