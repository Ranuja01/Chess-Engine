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
    for spec in KV["KNOBS"].split(","):
        name, on, off = spec.split(":")
        T = lambda p: {r["fen"]: float(r["total"]) for r in csv.DictReader(open(p, newline=""))}
        ton, toff = T(on), T(off)
        X = np.array([[ton.get(f, 0.0) - toff.get(f, 0.0)] for f in fens])
        fire = 100 * (np.abs(X[:, 0]) > 0).mean()
        if fire == 0:
            print("  %-22s ☠️ VACUOUS — the term never fired (dump identical); not a null" % name); continue
        p = fit(X, 0.0)
        print("  %-22s fires %5.1f%% · median |Δ| %5.0f mp · α %+5.2f · val %+6.2f%% · scale %+.3f"
              % (name, fire, np.median(np.abs(X[X[:, 0] != 0, 0])), p[0], 100 * (loss_of(X, p, val) / vb - 1), p[-1]))


if __name__ == "__main__":
    {"columns": columns, "dump": dump, "knobs": knobs}[KV.get("MODE", "columns")]()
