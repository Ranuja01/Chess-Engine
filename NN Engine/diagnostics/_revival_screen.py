# -*- coding: utf-8 -*-
"""REVIVAL SCREEN (owner, 2026-10-05): every eval term that was never fitted on the DEPTH target, or was closed on an
instrument since read as unreadable, measured the way connected pawns were revived on 10-04 (C3 §19e: closed 09-12 as
"harmful at every magnitude" on §I; the depth fit found the right magnitude; +14.5 ± 7.7 in games).

Target / model (the `_conn_depth_fit.py` one): ours' = ours_d10 (our d10 SEARCH of the CURRENT ship, `fitC_*_ours1004_d10`)
− Δ/10 + STM nuisance (not shipped), Δ = Black − White term change in mp; SF18 d14 labels, mg + eg sets, val by FEN hash
15%. Each block is fitted ALONE against the STM-only baseline ⇒ blocks are ranked on equal footing.

MODE=columns — terms that are v2_features columns (`px_labelled.npz`, Black − White counts) or FEN-computable:
  MOB 0-65 · PLACE 97-105 · KFL (C3-b) 162-171 · KPROT (C3-c) 172-183 · PST (file-tied, `_joint_depth_preview.pst_kauf`)
  · KAUF (census cells + N/B/R/Q value corrections) · reference: STRUCT 66-76 (10-04: −0.47% ⇒ +5.7 n.s. in games).
MODE=knobs  — knob-only terms: per term, two static dumps (ON at a reference magnitude M / OFF) give Δ(pos); fit ONE
  multiplier α (shippable as α·M if the term is ~linear in its knob). Dumps: MODE=dump OUT= <knobs> (one process each).
Reading: val % change vs the STM-only baseline; fire rate = rows where the block changes anything. ≥ ~0.5% (connected
0.59%) ⇒ build the export + closure and gate with games. A block that never fires is VACUOUS, not null.

  pyrun diagnostics/_revival_screen.py MODE=columns [LAMBDA=1e-2]
"""
import os, sys, csv, glob, hashlib
import numpy as np
import chess
from scipy.optimize import minimize

KV = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS)
DATA = "/mnt/e/chess_data/texel"
K = 0.00368208
OURS = KV.get("OURS", "ours1004")
wp = lambda cp: 100.0 / (1.0 + np.exp(-K * np.clip(cp, -1500, 1500)))


def load_rows():
    ours = {}
    for st in ("mg", "eg"):
        for p in glob.glob(os.path.join(THIS, "ks_sets", "fitC_%s_%s_d10_s*of4.csv" % (st, OURS))):
            for r in csv.DictReader(open(p, newline="")):
                ours[r["fen"]] = float(r["ours_cp_white"])
    z = np.load(os.path.join(DATA, "px_labelled.npz"))
    idx = {f: i for i, f in enumerate(z["fen"])}
    # ☠️ VAL SPLIT BY GAME, not by FEN (2026-10-05): positions of one game share material and structure, so a FEN-hash
    # split leaks near-copies into val and flatters high-capacity / material blocks (first run: KAUF cells −3.2%).
    gid = {}
    for smp in ("fitC_mg_sample.csv", "fitC_eg_sample.csv"):
        for r in csv.DictReader(open(os.path.join(THIS, "ks_sets", smp), newline="")):
            gid[r["fen"]] = r["game_id"]
    fens, rows, sf, base, val, stm = [], [], [], [], [], []
    for lab in ("fitC_mg_sf18.csv", "fitC_eg_sf18.csv"):
        for r in csv.DictReader(open(os.path.join(THIS, "ks_sets", lab), newline="")):
            f = r["fen"]
            if not r.get("best_cp") or f not in ours or f not in idx or abs(float(r["best_cp"])) >= 50000:
                continue
            if z["flags"][idx[f]] & 3:
                continue
            fens.append(f); rows.append(idx[f]); sf.append(float(r["best_cp"])); base.append(ours[f])
            key = gid.get(f, f) if KV.get("SPLIT", "game") == "game" else f
            val.append(int(hashlib.md5(key.encode()).hexdigest()[:8], 16) % 100 < 15)
            stm.append(1.0 if f.split()[1] == "w" else -1.0)
    rows = np.array(rows)
    return (fens, z["diff"][rows].astype(np.float64), z["phase"][rows].astype(np.float64), np.array(sf), np.array(base),
            np.array(val), np.array(stm))


def fitter(base, tgt, stm, tr, val):
    """p = [δ (X columns) …, STM nuisance, SCALE nuisance s]: ours' = base·(1+s) − X·δ/10 + STM·stm. ☠️ SCALE is a free
    GLOBAL stretch in the baseline AND every arm, never shipped (SCALE=0 disables): SF18 search scores run larger than
    ours, so without it any block can 'win' by inflating evals (Kaufman fit 2 stretched ×1.06; the first screen run
    10-05 read KAUF −5.0% with piece-value columns in it)."""
    use_s = KV.get("SCALE", "1") == "1"

    def cp(X, p):
        return base * (1.0 + p[-1]) - (X @ p[:-2]) / 10.0 + p[-2] * stm

    def loss_of(X, p, m):
        return float(np.mean((wp(cp(X, p))[m] - tgt[m]) ** 2))

    nuis = [np.zeros(2)]                      # the baseline's fitted (STM, SCALE): every arm starts there

    def fit(X, lam):
        nk = X.shape[1]
        def fg(p):
            c = cp(X, p)[tr]; q = wp(c); r = q - tgt[tr]
            g = 2.0 * r * q * (1 - q / 100.0) * K * (np.abs(c) < 1500)
            gw = -(X[tr].T @ g) / tr.sum() / 10.0
            gs = float((g * base[tr]).mean()) if use_s else 0.0
            return float(np.mean(r * r)) + lam * float(p[:nk] @ p[:nk]) / 1e4, \
                np.r_[gw + 2 * lam * p[:nk] / 1e4, float((g * stm[tr]).mean()), gs]
        # ☠️ tight tolerances + start at the baseline nuisances: the first SCALE run read KFL/KPROT as exactly −0.00%,
        # the signature of an optimizer that stops where it starts once the scale term dominates the gradient.
        return minimize(fg, np.r_[np.zeros(nk), nuis[0]], jac=True, method="L-BFGS-B",
                        options={"maxiter": 20000, "ftol": 1e-15, "gtol": 1e-12}).x
    X0 = np.zeros((len(base), 0))
    p0 = fit(X0, 0.0)
    nuis[0] = p0[-2:]
    vb = loss_of(X0, p0, val)
    return fit, loss_of, vb


def columns():
    from _joint_depth_preview import pst_kauf
    fens, D, ph, sf, base, val, stm = load_rows()
    tgt, tr = wp(sf), ~val
    a, e = (ph / 256.0)[:, None], ((256.0 - ph) / 256.0)[:, None]
    legs = lambda M: np.concatenate([M * a, M * e], 1)
    pk = [pst_kauf(f) for f in fens]
    PST = np.array([p for p, _ in pk]); KF = np.array([k for _, k in pk])
    blocks = {
        "MOB (0-65)": legs(D[:, 0:66]),
        "PLACE (97-105)": legs(D[:, 97:106]),
        "KFL C3-b (162-171)": legs(D[:, 162:172]),
        "KPROT C3-c (172-183)": legs(D[:, 172:184]),
        "PST (file-tied)": legs(PST),
        "KAUF cells only": KF[:, :36],
        "KAUF values only (N/B/R/Q)": KF[:, 36:],
        "ref: STRUCT (66-76)": legs(D[:, 66:77]),
    }
    blocks["ALL of the above"] = np.concatenate(list(blocks.values()), 1)
    if "BLOCKS" in KV:                          # e.g. BLOCKS=PLACE,KAUF — re-read a subset
        blocks = {k: v for k, v in blocks.items() if any(k.startswith(b) for b in KV["BLOCKS"].split(","))}
    fit, loss_of, vb = fitter(base, tgt, stm, tr, val)
    lam = float(KV.get("LAMBDA", 1e-2))
    print("REVIVAL SCREEN (columns)  rows %d (val %d) · base = our d10 search of %s · λ=%g" % (len(sf), val.sum(), OURS, lam) + " · SCALE nuisance " + KV.get("SCALE", "1"))
    for name, X in blocks.items():
        fire = 100 * (np.abs(X).sum(1) > 0).mean()
        p = fit(X, lam)
        print("  %-26s params %4d · fires %5.1f%% · val %+6.2f%% · scale %+.3f"
              % (name, X.shape[1], fire, 100 * (loss_of(X, p, val) / vb - 1), p[-1]))


def export():
    """MODE=gateexport PSTDUMP=<PST_V2_DUMP of the shipped engine> [OUTDIR=/mnt/e/chess_data/texel/revival] — refit each GATE
    block exactly as screened (λ, SCALE nuisance, by-game split) and write engine-format files (each loads ON TOP of the
    shipped eval, one arm per file):
      kauf_depth.txt   KAUF_V2_FILE (O/T a b mp, White-POV) = shipped cells (kauf_full.txt) + δ   [KAUF_V2_FORM=3, MAG 1000]
      mob_depth_c1.txt C1_V2_FILE (k leg start fitted), k 0-65, start = live θ                    [C1_V2_FIT=1]
      pst_depth.txt    PST_V2_FILE (768 ints, per piece 64 mg then 64 eg, White-POV) = shipped tables + δ (file-tied)
      kprot_depth.txt  KPROT_V2_FILE (k leg 0 fitted), k 172-183                                 [KPROT_V2=1]
      kfl_depth.txt    KFL_V2_FILE   (k leg 0 fitted), k 162-171                                 [KFL_V2=1]
    Sign convention (all blocks): columns are Black − White, δ adds to the Black-positive total ⇒ a White-POV table
    value moves by +δ (KAUF: feats are White-POV and the block used −feats ⇒ cell += δ)."""
    from _joint_depth_preview import pst_kauf
    import _texel_kauf_fit as KF_
    out = KV.get("OUTDIR", DATA + "/revival")
    os.makedirs(out, exist_ok=True)
    fens, D, ph, sf, base, val, stm = load_rows()
    tgt, tr = wp(sf), ~val
    a, e = (ph / 256.0)[:, None], ((256.0 - ph) / 256.0)[:, None]
    legs = lambda M: np.concatenate([M * a, M * e], 1)
    pk = [pst_kauf(f) for f in fens]
    PST = np.array([p for p, _ in pk]); KF = np.array([k for _, k in pk])[:, :36]
    fit, loss_of, vb = fitter(base, tgt, stm, tr, val)
    lam = float(KV.get("LAMBDA", 1e-2))
    z = np.load(os.path.join(DATA, "px_labelled.npz"))
    tmg, teg = z["theta_mg"], z["theta_eg"]

    def report(name, X, p):
        print("EXPORT %-6s val %+6.2f%% · max |δ| %4.0f mp · rms %4.0f mp" % (name, 100 * (loss_of(X, p, val) / vb - 1),
              np.abs(p[:-2]).max(), np.sqrt(np.mean(p[:-2] ** 2))))

    # KAUF cells
    X = KF; p = fit(X, lam); report("KAUF", X, p)
    ship = {}
    for line in open(os.path.join(DATA, "kauf_full.txt")):
        if line.strip() and not line.startswith("#"):
            k_, a_, b_, v_ = line.split(); ship[(k_, int(a_), int(b_))] = float(v_)
    with open(os.path.join(out, "kauf_depth.txt"), "w") as f:
        f.write("# Kaufman cells = shipped (kauf_full.txt) + DEPTH re-fit δ (_revival_screen.py MODE=gateexport λ=%g)\n" % lam)
        for i, (k_, a_, b_) in enumerate(KF_.CELLS[:36]):
            v = ship.get((k_, a_, b_), 0.0) + p[i]
            if round(v) != 0:
                f.write("%s %d %d %.0f\n" % (k_, a_, b_, v))
    # MOB
    X = legs(D[:, 0:66]); p = fit(X, lam); report("MOB", X, p)
    with open(os.path.join(out, "mob_depth_c1.txt"), "w") as f:
        f.write("# mobility cells, DEPTH fit (_revival_screen.py MODE=gateexport λ=%g) — k leg start fitted (mp)\n" % lam)
        for k in range(66):
            for leg, (st, d) in enumerate(((tmg[k], p[k]), (teg[k], p[66 + k]))):
                if round(d) != 0:
                    f.write("%d %d %.0f %.0f\n" % (k, leg, st, st + d))
    # KPROT / KFL (built at 0 ⇒ start 0)
    for name, lo, hi, fn in (("KPROT", 172, 184, "kprot_depth.txt"), ("KFL", 162, 172, "kfl_depth.txt")):
        X = legs(D[:, lo:hi]); p = fit(X, lam); report(name, X, p); n = hi - lo
        with open(os.path.join(out, fn), "w") as f:
            f.write("# %s cells, DEPTH fit (_revival_screen.py MODE=gateexport λ=%g) — feature_k leg start fitted\n" % (name, lam))
            for i in range(n):
                for leg in (0, 1):
                    f.write("%d %d 0 %.0f\n" % (lo + i, leg, p[i + leg * n]))
    # PST (file-tied δ on top of the shipped tables from PST_V2_DUMP)
    X = legs(PST); p = fit(X, lam); report("PST", X, p)
    vals = [int(t) for line in open(KV["PSTDUMP"]) if not line.startswith("#") for t in line.split()]
    assert len(vals) == 768, "PST dump has %d values" % len(vals)
    newv = list(vals)
    for t in range(6):
        for leg in (0, 1):
            for sq in range(64):
                r, fl = sq >> 3, sq & 7
                d = p[leg * 192 + t * 32 + r * 4 + min(fl, 7 - fl)]
                newv[t * 128 + leg * 64 + sq] = int(round(vals[t * 128 + leg * 64 + sq] + d))
    with open(os.path.join(out, "pst_depth.txt"), "w") as f:
        f.write("# v2 PST = shipped (PST_V2_DUMP) + DEPTH re-fit δ (file-tied; _revival_screen.py MODE=gateexport λ=%g)\n" % lam)
        for t in range(6):
            for leg in (0, 1):
                for r in range(8):
                    f.write(" ".join(str(newv[t * 128 + leg * 64 + r * 8 + c]) for c in range(8)) + "\n")
    print("EXPORT wrote kauf_depth / mob_depth_c1 / kprot_depth / kfl_depth / pst_depth → %s" % out)


def dump():
    for k, v in KV.items():
        os.environ[k] = v
    os.environ.setdefault("PRESET", "LONG_FORMAT"); os.environ.setdefault("USE_OPENING_BOOK", "0")
    sys.path.insert(0, os.path.dirname(THIS))
    import ChessAI
    ai = ChessAI.ChessAI(None, None, chess.Board(), True)
    seen = set()
    with open(KV["OUT"], "w", newline="") as fo:
        w = csv.writer(fo); w.writerow(["fen", "total"])
        for lab in ("fitC_mg_sf18.csv", "fitC_eg_sf18.csv"):
            for r in csv.DictReader(open(os.path.join(THIS, "ks_sets", lab), newline="")):
                if r["fen"] in seen or not r.get("best_cp"):
                    continue
                seen.add(r["fen"])
                w.writerow([r["fen"], int(ai.ev_breakdown(chess.Board(r["fen"]))["total"])])
    print("REVIVAL DUMP  %d rows → %s" % (len(seen), KV["OUT"]))


def knobs():
    """KNOBS=name:on.csv:off.csv,… — Δ = total_on − total_off (Black-positive mp); fit α on ours' = base − α·Δ/10."""
    fens, _, _, sf, base, val, stm = load_rows()
    tgt, tr = wp(sf), ~val
    fit, loss_of, vb = fitter(base, tgt, stm, tr, val)
    print("REVIVAL SCREEN (knobs)  rows %d (val %d) · base = our d10 search of %s" % (len(sf), val.sum(), OURS))
    T = lambda p: {r["fen"]: float(r["total"]) for r in csv.DictReader(open(p, newline=""))}
    cols, names = [], []
    for spec in KV["KNOBS"].split(","):
        name, on, off = spec.split(":")
        ton, toff = T(on), T(off)
        X = np.array([[ton.get(f, 0.0) - toff.get(f, 0.0)] for f in fens])
        fire = 100 * (np.abs(X[:, 0]) > 0).mean()
        if fire == 0:
            print("  %-22s ☠️ VACUOUS — the term never fired (dump identical); not a null" % name); continue
        cols.append(X); names.append(name)
        p = fit(X, 0.0)
        print("  %-22s fires %5.1f%% · median |Δ| %5.0f mp · α %+5.2f · val %+6.2f%% · scale %+.3f"
              % (name, fire, np.median(np.abs(X[X[:, 0] != 0, 0])), p[0], 100 * (loss_of(X, p, val) / vb - 1), p[-1]))
    if KV.get("JOINT") == "1" and len(cols) > 1:
        # the Kaufman lesson applied to a multi-leg term: one multiplier per LEG, fitted together, so the term's SHAPE
        # (the mix of legs) can change — a single global multiplier only tests magnitude and lets wrong legs cancel right ones
        X = np.concatenate(cols, 1)
        p = fit(X, 0.0)
        print("  JOINT (%d legs)          val %+6.2f%% · scale %+.3f · α per leg: %s" % (len(cols),
              100 * (loss_of(X, p, val) / vb - 1), p[-1], "  ".join("%s %+.2f" % (n, a) for n, a in zip(names, p[:len(cols)]))))


def dual():
    """MODE=dual KNOBS=<as knobs> BASE=<off.csv: the ship's STATIC totals> — owner 2026-10-07: "improve actual play AND what
    pruning sees". The same legs fitted three ways — DEPTH (base = our d10 search), STATIC (base = our static eval, the ship's
    totals from BASE=), COMBINED (both losses, each normalised by its own baseline, equal weight; α shared, each target keeps
    its own STM + SCALE nuisances) — and EVERY fit scored on BOTH targets (val), plus the endgame rows (phase < 128) apart.
    Reading: a static gain is only worth having if the depth column does not get worse."""
    fens, _, ph, sf, d10, val, stm = load_rows()
    tgt, tr = wp(sf), ~val
    T = lambda p: {r["fen"]: float(r["total"]) for r in csv.DictReader(open(p, newline=""))}
    off = T(KV["BASE"])
    keep = np.array([f in off for f in fens])
    fens = [f for f, k in zip(fens, keep) if k]
    ph, tgt, d10, val, stm, tr = ph[keep], tgt[keep], d10[keep], val[keep], stm[keep], tr[keep]
    stat = np.array([-off[f] / 10.0 for f in fens])
    if KV.get("VALOUT"):                      # the val rows as an IN for a REAL arm depth pass (_depth_residual_pass.py)
        sfcp = {}
        for lab in ("fitC_mg_sf18.csv", "fitC_eg_sf18.csv"):
            for r in csv.DictReader(open(os.path.join(THIS, "ks_sets", lab), newline="")):
                sfcp[r["fen"]] = (r.get("best_uci", ""), r["best_cp"])
        with open(os.path.join(THIS, "ks_sets", KV["VALOUT"]), "w", newline="") as fh:
            w = csv.writer(fh); w.writerow(["fen", "best_uci", "best_cp"])
            for f, v in zip(fens, val):
                if v:
                    w.writerow([f, *sfcp[f]])
        print("VALOUT: %d val rows → ks_sets/%s" % (val.sum(), KV["VALOUT"]))
        return
    cols, names = [], []
    for spec in KV["KNOBS"].split(","):
        name, on, off_ = spec.split(":")
        ton, toff = T(on), T(off_)
        cols.append(np.array([[ton.get(f, 0.0) - toff.get(f, 0.0)] for f in fens])); names.append(name)
    X = np.concatenate(cols, 1)
    nk = X.shape[1]
    eg = ph < 128

    def cp(base, p, nuis):                    # nuis = (STM, SCALE)
        return base * (1.0 + nuis[1]) - (X @ p) / 10.0 + nuis[0] * stm

    def loss(base, p, nuis, m):
        return float(np.mean((wp(cp(base, p, nuis))[m] - tgt[m]) ** 2))

    def fit_nuis(base, p):                    # best (STM, SCALE) for fixed α, on train
        return minimize(lambda n: loss(base, p, n, tr), np.zeros(2), method="Nelder-Mead",
                        options=dict(xatol=1e-6, fatol=1e-9, maxiter=4000)).x

    z = np.zeros(nk)
    nd0, ns0 = fit_nuis(d10, z), fit_nuis(stat, z)
    vb = {("d", "all"): loss(d10, z, nd0, val), ("s", "all"): loss(stat, z, ns0, val),
          ("d", "eg"): loss(d10, z, nd0, val & eg), ("s", "eg"): loss(stat, z, ns0, val & eg)}
    trb_d, trb_s = loss(d10, z, nd0, tr), loss(stat, z, ns0, tr)

    def objective(q, wd, ws):
        p, nd, ns = q[:nk], q[nk:nk + 2], q[nk + 2:]
        o = 0.0
        if wd:
            o += wd * loss(d10, p, nd, tr) / trb_d
        if ws:
            o += ws * loss(stat, p, ns, tr) / trb_s
        return o

    print("THREATS DUAL FIT — rows %d (val %d, of which endgame %d) · legs: %s" % (len(fens), val.sum(), (val & eg).sum(),
          ", ".join("%s fires %.0f%%" % (n, 100 * (np.abs(c[:, 0]) > 0).mean()) for n, c in zip(names, cols))))
    print("  baselines (val win%% MSE): depth %.1f (eg %.1f) · static %.1f (eg %.1f)"
          % (vb[("d", "all")], vb[("d", "eg")], vb[("s", "all")], vb[("s", "eg")]))
    print("  %-10s %-48s %11s %11s %11s %11s" % ("fit on", "α per leg", "DEPTH all", "DEPTH eg", "STATIC all", "STATIC eg"))
    arms = [("depth", 1, 0), ("static", 0, 1), ("combined", 1, 1)]
    if KV.get("FIXED"):                       # FIXED=a,b,… — score an ENGINE-REALISABLE α (legs are on/off at one PCT)
        arms.append(("fixed", None, np.array([float(x) for x in KV["FIXED"].split(",")])))
    for label, wd, ws in arms:
        if wd is None:
            p = ws
        else:
            q0 = np.r_[z, nd0, ns0]
            q = minimize(lambda q: objective(q, wd, ws), q0, method="Powell",
                         options=dict(xtol=1e-4, ftol=1e-10, maxiter=40000)).x
            p = q[:nk]
        nd, ns = fit_nuis(d10, p), fit_nuis(stat, p)    # each target re-fits ITS nuisances for the shared α
        cells = []
        for tag, base, nu in (("d", d10, nd), ("s", stat, ns)):
            for scope, m in (("all", val), ("eg", val & eg)):
                cells.append(100 * (loss(base, p, nu, m) / vb[(tag, scope)] - 1))
        print("  %-10s %-48s %+10.2f%% %+10.2f%% %+10.2f%% %+10.2f%%" % (
            label, " ".join("%s %+.2f" % (n, a) for n, a in zip(names, p)), *cells))


if __name__ == "__main__":
    {"columns": columns, "dump": dump, "knobs": knobs, "gateexport": export, "dual": dual}[KV.get("MODE", "columns")]()  # ☠️ not "export": _texel_kauf_fit (imported) runs its own export on MODE=export
