# -*- coding: utf-8 -*-
"""Fit the detector-CONDITIONING knobs (term x f(detector)) against classical Stockfish 11's eval, OFFLINE.

Unlike tune_fit.py (which fits flat SCALE factors by linear recombination), this tunes the *when* — the
mod_gain / realizability conditioning that boosts or decays a term based on cheap board detectors. The hooks
are closed-form integer functions of (stored detector, knobs), so the engine never has to re-run: we replay
the exact C++ integer math on the corpus columns and search the knobs by block-coordinate descent.

First cluster (material/imbalance/pair — strategic, transfers across depth):
  * REALIZ_*            -> imbalance term   (realizability_factor: damp an imbalance when the material edge
                          behind it is small / the phase is late; R<=256 so DAMP-only; needs REALIZ_FLOOR<256)
  * MOD_PAIR_OPEN       -> bishop-pair bonus (worth more in open positions: few pawns)
  * MOD_MAT_PAWNS/OPPB  -> piece_value_boost (the domination boost: scale by pawn count + opposite bishops)

Objective: make OUR conditioned total track SF11's total (White-POV pawns) via a logistic loss, on a held-out
control split (overfit guard). SF11's per-term breakdown is reported as a DIAGNOSTIC. Lightning SPRT still
decides shipping; this only generates a candidate knob config.

Run in WSL from NN Engine/:
    python selfplay/tune_cond.py --corpus selfplay/tune_data/cond_corpus.csv
"""

import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')

import sys
import csv
import argparse
import numpy as np

import chess

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

BISHOP_PAIR_BONUS = 300   # search_engine.h default
MOD_FLOOR, MOD_CEIL = 128, 512
K_PAWNS = 2.0             # logistic scale (same convention as tune_fit.py)

# Numeric columns pulled from the corpus.
NUM_COLS = ["phase_score", "our_total", "imbalance_white", "imbalance_black", "piece_value_boost",
            "det_w_pieceval", "det_b_pieceval", "det_pawn_count", "oppb", "sf11_total",
            "is_endgame"]


def clamp_gain(g):
    return np.clip(g, MOD_FLOOR, MOD_CEIL)


def load_corpus(path):
    cols = {c: [] for c in NUM_COLS}
    fens, status = [], []
    with open(path, newline="") as fh:
        r = csv.DictReader(fh)
        for row in r:
            if not row.get("sf11_total"):       # SF11 label missing -> unusable
                continue
            try:
                vals = {c: float(row[c]) for c in NUM_COLS}
            except (ValueError, KeyError):
                continue
            for c in NUM_COLS:
                cols[c].append(vals[c])
            fens.append(row["fen"])
            status.append(row.get("status", "near_equal"))
    data = {c: np.array(cols[c], dtype=np.int64) if c != "sf11_total" else np.array(cols[c], dtype=np.float64)
            for c in NUM_COLS}
    # Bishop-pair flags from the FEN (the pair hook modulates only the bishop-pair bonus, not knight pairs).
    wpair = np.zeros(len(fens), dtype=np.int64)
    bpair = np.zeros(len(fens), dtype=np.int64)
    for i, fen in enumerate(fens):
        b = chess.Board(fen)
        wpair[i] = 1 if chess.popcount(b.bishops & b.occupied_co[chess.WHITE]) == 2 else 0
        bpair[i] = 1 if chess.popcount(b.bishops & b.occupied_co[chess.BLACK]) == 2 else 0
    data["wpair"], data["bpair"] = wpair, bpair
    data["status"] = np.array(status)
    return data, len(fens)


# ---- exact C++ replay (positive magnitude then sign, integer right-shift) -------------------------------

def realiz_R(edge, phase, K, thresh, phase_k, floor):
    """realizability_factor(): R in /256 units, <=256 (damp). edge = attacker (own-enemy) material."""
    mat_def = np.maximum(0, thresh - edge)
    discount = (K * mat_def) >> 12
    discount = discount + ((phase_k * phase) >> 7)
    return np.maximum(floor, 256 - discount)


def delta_imbalance(data, K, thresh, phase_k, floor):
    """Change to the (Black-positive) total from REALIZ-damping both imbalance terms."""
    if K == 0 and phase_k == 0:
        return np.zeros(len(data["our_total"]), dtype=np.int64)
    iw_raw = data["imbalance_white"]               # <= 0 (favours White)
    ib_raw = data["imbalance_black"]               # >= 0 (favours Black)
    w_edge = data["det_w_pieceval"] - data["det_b_pieceval"]
    b_edge = -w_edge
    Rw = realiz_R(w_edge, data["phase_score"], K, thresh, phase_k, floor)
    Rb = realiz_R(b_edge, data["phase_score"], K, thresh, phase_k, floor)
    iw_mag = -iw_raw                               # >= 0
    iw_cond_mag = (iw_mag * Rw) >> 8
    new_iw = -iw_cond_mag
    ib_cond = (ib_raw * Rb) >> 8
    return (new_iw - iw_raw) + (ib_cond - ib_raw)


def delta_pair(data, mod_pair_open):
    if mod_pair_open == 0:
        return np.zeros(len(data["our_total"]), dtype=np.int64)
    pawns = data["det_pawn_count"]
    gain = clamp_gain(256 + ((mod_pair_open * (12 - pawns)) >> 3))
    modded = (BISHOP_PAIR_BONUS * gain) >> 8
    raw = (-BISHOP_PAIR_BONUS) * data["wpair"] + BISHOP_PAIR_BONUS * data["bpair"]
    cond = (-modded) * data["wpair"] + modded * data["bpair"]
    return cond - raw


def delta_pvboost(data, mod_mat_pawns, mod_mat_oppb):
    if mod_mat_pawns == 0 and mod_mat_oppb == 0:
        return np.zeros(len(data["our_total"]), dtype=np.int64)
    pawns = data["det_pawn_count"]
    gain = clamp_gain(256 + ((mod_mat_pawns * (pawns - 12)) >> 3) + (mod_mat_oppb * data["oppb"]))
    raw = data["piece_value_boost"]
    sign = np.sign(raw)
    cond = sign * ((np.abs(raw) * gain) >> 8)      # magnitude-then-sign matches the C++ exactly
    return cond - raw


def cond_total(data, kn):
    return (data["our_total"]
            + delta_imbalance(data, kn["REALIZ_MAT_K"], kn["REALIZ_MAT_THRESH"], kn["REALIZ_PHASE_K"], kn["REALIZ_FLOOR"])
            + delta_pair(data, kn["MOD_PAIR_OPEN"])
            + delta_pvboost(data, kn["MOD_MAT_PAWNS"], kn["MOD_MAT_OPPB"]))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def loss_on(data, kn, idx):
    our_w = -cond_total(data, kn)[idx] / 1000.0          # White-POV pawns
    sf = data["sf11_total"][idx]
    return float(np.mean((sigmoid(our_w / K_PAWNS) - sigmoid(sf / K_PAWNS)) ** 2))


# Strength-relevant (scale-INVARIANT) objective: a global eval multiplier never changes move choice
# (argmax is scale-invariant), so we give every candidate its OWN best global scale and judge only the
# residual SHAPE fit. Knobs earn credit only if they improve relative grading beyond what a scale can do.
SCALE_GRID = np.arange(0.40, 1.205, 0.02)


def loss_scaled(data, kn, idx):
    base = -cond_total(data, kn)[idx].astype(np.float64) / 1000.0   # White-POV pawns, pre-scale
    sf_sig = sigmoid(data["sf11_total"][idx] / K_PAWNS)
    best_l, best_s = 1e18, 1.0
    for s in SCALE_GRID:
        l = float(np.mean((sigmoid(s * base / K_PAWNS) - sf_sig) ** 2))
        if l < best_l:
            best_l, best_s = l, s
    return best_l, best_s


def mae_on(data, kn, idx):
    our_w = -cond_total(data, kn)[idx] / 1000.0
    sf = data["sf11_total"][idx]
    return float(np.mean(np.abs(np.clip(our_w, -6, 6) - np.clip(sf, -6, 6))))


# ---- block-coordinate search ---------------------------------------------------------------------------

GRID = {
    "REALIZ": [(K, T, P, F)
               for K in (0, 128, 256, 512, 1024, 2048)
               for T in (0, 1000, 2000, 3000, 5000)
               for P in (0, 16, 32, 64, 96, 128)
               for F in (256, 192, 128, 96, 64)],
    "PAIR":  [-256, -128, -64, 0, 64, 128, 256, 384, 512],
    "MAT":   [(MP, MO) for MP in (-512, -256, -128, 0, 128, 256, 512)
                       for MO in (-512, -256, -128, 0, 128)],
}
DEFAULT = {"REALIZ_MAT_K": 0, "REALIZ_MAT_THRESH": 0, "REALIZ_PHASE_K": 0, "REALIZ_FLOOR": 256,
           "MOD_PAIR_OPEN": 0, "MOD_MAT_PAWNS": 0, "MOD_MAT_OPPB": 0}


def fit(data, train, passes=3):
    kn = dict(DEFAULT)
    best = loss_scaled(data, kn, train)[0]
    for _ in range(passes):
        improved = False
        for K, T, P, F in GRID["REALIZ"]:
            trial = dict(kn, REALIZ_MAT_K=K, REALIZ_MAT_THRESH=T, REALIZ_PHASE_K=P, REALIZ_FLOOR=F)
            l = loss_scaled(data, trial, train)[0]
            if l < best - 1e-12:
                best, kn, improved = l, trial, True
        for v in GRID["PAIR"]:
            trial = dict(kn, MOD_PAIR_OPEN=v)
            l = loss_scaled(data, trial, train)[0]
            if l < best - 1e-12:
                best, kn, improved = l, trial, True
        for MP, MO in GRID["MAT"]:
            trial = dict(kn, MOD_MAT_PAWNS=MP, MOD_MAT_OPPB=MO)
            l = loss_scaled(data, trial, train)[0]
            if l < best - 1e-12:
                best, kn, improved = l, trial, True
        if not improved:
            break
    return kn


def report_split(name, data, kn, idx):
    l0, s0 = loss_scaled(data, DEFAULT, idx)
    l1, s1 = loss_scaled(data, kn, idx)
    print("  [%s n=%d]  scale-inv loss %.6f (s=%.2f) -> %.6f (s=%.2f)"
          % (name, len(idx), l0, s0, l1, s1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=os.path.join(THIS_DIR, "tune_data", "cond_corpus.csv"))
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--control-frac", type=float, default=0.3)
    args = ap.parse_args()

    data, n = load_corpus(args.corpus)
    print("corpus n=%d  from %s" % (n, args.corpus))
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n)
    n_ctrl = int(n * args.control_frac)
    ctrl, train = perm[:n_ctrl], perm[n_ctrl:]
    allidx = np.arange(n)

    kn = fit(data, train)

    print("\nFITTED conditioning knobs (vs SF11 total, logistic loss):")
    for k in ["REALIZ_MAT_K", "REALIZ_MAT_THRESH", "REALIZ_PHASE_K", "REALIZ_FLOOR",
              "MOD_PAIR_OPEN", "MOD_MAT_PAWNS", "MOD_MAT_OPPB"]:
        print("    %-20s %d" % (k, kn[k]))
    print()
    report_split("train  ", data, kn, train)
    report_split("control", data, kn, ctrl)     # the overfit guard: must improve here too
    report_split("all    ", data, kn, allidx)

    # Under the scale-invariant objective, the default ALREADY gets its own best global scale, so any
    # remaining gain is genuine state-conditioning (relative-grading), not blunt shrink. Per-block
    # attribution shows which hook (if any) carries it.
    print("\n  per-block control gain (scale-inv, default -> block-only):")
    base_l = loss_scaled(data, DEFAULT, ctrl)[0]
    for lbl, keys in [("REALIZ(imbalance)", ["REALIZ_MAT_K", "REALIZ_MAT_THRESH", "REALIZ_PHASE_K", "REALIZ_FLOOR"]),
                      ("PAIR_OPEN(pair)  ", ["MOD_PAIR_OPEN"]),
                      ("MAT(pvboost)     ", ["MOD_MAT_PAWNS", "MOD_MAT_OPPB"])]:
        only = dict(DEFAULT)
        for k in keys:
            only[k] = kn[k]
        print("    %s  %.6f -> %.6f" % (lbl, base_l, loss_scaled(data, only, ctrl)[0]))

    env = " ".join("%s=%d" % (k, kn[k]) for k in
                   ["REALIZ_MAT_K", "REALIZ_MAT_THRESH", "REALIZ_PHASE_K", "REALIZ_FLOOR",
                    "MOD_PAIR_OPEN", "MOD_MAT_PAWNS", "MOD_MAT_OPPB"] if kn[k] != DEFAULT[k])
    print("\nCANDIDATE env knobs:\n    %s" % (env if env else "(none — fit found no improvement)"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
