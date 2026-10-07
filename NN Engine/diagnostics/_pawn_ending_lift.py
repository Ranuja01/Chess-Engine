# -*- coding: utf-8 -*-
"""PAWN-ENDING CONFIDENCE LIFT — screen (owner, 2026-10-07; POT endgame step 1). Pawn-only boards (no N/B/R/Q) read too
drawish: the side SF18 calls ahead is under-rated ~22 win% pts static, ~9 at our d10, and no per-term fit or all-endgame
stretch reaches it (pawn endings are 1.3% of endgame rows). References price the CLASS: SF11/SF15 complexity +51 for no
non-pawn material (+11/pawn), Ethereal +76 for a pawn ending. Here: a winnability input that MULTIPLIES the whole score
on pawn-only boards, allowed ABOVE 1 — continuous at 0, cannot create an edge from a level score.

FORMS (g = the lift, score' = score · (1 + g); fitted on the by-GAME train split of the depth rows):
  A  g = a
  B  g = a + b · (total pawns / 8)
  C  B, but only when the LEADER (sign of OUR score) can make progress: it has a passed pawn or a wing majority with a pawn
     not blocked head-on. Guards the owner's 10-04 game (doubled extra pawn, +0.7 in a dead draw).
Each form is fitted TWICE — DEPTH (our d10 · (1+g) vs SF18 d14: what play sees; a pawn ending stays one along the PV) and
STATIC (our static · (1+g): what pruning sees). NESTED with KFL (run with KFL_V2=1 KFL_V2_FILE=…): arms lift-alone (KFL
removed), KFL-alone (d10 + its static share), both.
GUARDRAILS (owner): val change on rows where SF11's static is CLOSER to SF18 than ours (should improve) vs rows where WE are
closer (must not degrade); the held-out K+P stress set (600, never fitted); the owner's 10-04 position (truth 0.00).
  pyrun diagnostics/_pawn_ending_lift.py        (env: V2_PRESET=shipped [KFL_V2=1 KFL_V2_FILE=…])
"""
import os, sys, csv, glob, subprocess
import numpy as np
import chess
from scipy.optimize import minimize
THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS)
sys.path.insert(0, os.path.dirname(THIS))
os.environ.setdefault("PRESET", "LONG_FORMAT"); os.environ.setdefault("USE_OPENING_BOOK", "0")
import ChessAI
import _revival_screen as RS
from _triangulate_sf11 import SF11, sf11_eval, SF11_TEMPO_PAWNS
from _eg_features import labels

wp = lambda cp: 100.0 / (1.0 + np.exp(-0.00368208 * np.clip(cp, -1500, 1500)))
# The owner's 10-04 game after 56…g5 (W: K + b2 c3 e4 g4 g2 vs B: K + c5 c4 e5 g5). King squares RECONSTRUCTED from the
# note (central, near the locked pawns) — the structure is the point; SF18 d24 read 0.00 throughout.
OWNER_1004 = "8/8/3k4/2p1p1p1/2p1P1P1/2P5/1P1K2P1/8 w - - 0 57"


def pawn_only(b):
    return not (b.knights | b.bishops | b.rooks | b.queens)


def potential(b, our_cp):
    """Form D gate: OUR leader can create a passer (eval_v2 `passer_potential`) and the other side cannot."""
    if abs(our_cp) < 1:
        return 0.0
    pw, pb = ChessAI.pawn_masks(int(b.pawns), int(b.occupied_co[chess.WHITE]), int(b.occupied_co[chess.BLACK]))["potential"]
    pl, po = (pw, pb) if our_cp > 0 else (pb, pw)
    return 1.0 if (pl and not po) else 0.0


def progress(b, our_cp):
    """Leader (sign of OUR score) has a passer or a mobile wing majority."""
    if abs(our_cp) < 1:
        return 0.0
    L = labels(b, chess.WHITE if our_cp > 0 else chess.BLACK)
    return 1.0 if (L["passers_L"] > 0 or L["mobile_majority"]) else 0.0


def rows_from(ai, fens, sfv, d10, sf11p=None):
    R = []
    for f, t, d in zip(fens, sfv, d10):
        b = chess.Board(f)
        if not pawn_only(b) or b.is_check():
            continue
        bd = ai.ev_breakdown(b)
        stat = -float(bd["total"]) / 10.0
        kfl = -float(bd.get("v2_kflank") or 0) / 10.0
        s11 = None
        if sf11p is not None:
            tot, _ = sf11_eval(sf11p, f)
            if tot is not None:
                s11 = 100 * tot - (SF11_TEMPO_PAWNS if b.turn == chess.WHITE else -SF11_TEMPO_PAWNS) * 100
        R.append(dict(fen=f, tgt=t, d10=d, stat=stat, kfl=kfl, npawn=len(b.pieces(chess.PAWN, True)) + len(b.pieces(chess.PAWN, False)),
                      prog=progress(b, stat), pot=potential(b, stat), sf11=s11))
    return R


def g_of(form, p, r):
    if form == "A":
        return p[0]
    g = p[0] + p[1] * r["npawn"] / 8.0
    if form == "D":
        return g * r["pot"]
    return g * r["prog"] if form == "C" else g


def pred(form, p, r, leg, kfl_mode):
    """kfl_mode: 'none' = KFL removed (the ship), 'on' = KFL included. leg: 'depth' | 'static'."""
    k = r["kfl"] if kfl_mode == "on" else 0.0
    base = (r["d10"] + k) if leg == "depth" else (r["stat"] - r["kfl"] + k)
    return base * (1.0 + (g_of(form, p, r) if form else 0.0))


def mse(R, form, p, leg, kfl_mode):
    return float(np.mean([(wp(pred(form, p, r, leg, kfl_mode)) - wp(r["tgt"])) ** 2 for r in R])) if R else float("nan")


def bias(R, form, p, leg, kfl_mode):
    a = [r for r in R if abs(r["tgt"]) > 25]
    return float(np.mean([(wp(pred(form, p, r, leg, kfl_mode)) - wp(r["tgt"])) * np.sign(r["tgt"]) for r in a])) if a else float("nan")


def main():
    fens, _, ph, sfl, base, val, _ = RS.load_rows()
    ai = ChessAI.ChessAI(None, None, chess.Board(), True)
    p11 = subprocess.Popen([SF11], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1)
    p11.stdin.write("uci\n"); p11.stdin.flush()
    while p11.stdout.readline().strip() != "uciok":
        pass
    vset = {f for f, v in zip(fens, val) if v}
    R = rows_from(ai, fens, sfl, base, p11)
    tr = [r for r in R if r["fen"] not in vset]
    va = [r for r in R if r["fen"] in vset]
    kp_t, kp_d = {}, {}
    for r in csv.DictReader(open(os.path.join(THIS, "ks_sets/kp_stress_sf18.csv"), newline="")):
        if r.get("best_cp") and abs(float(r["best_cp"])) < 50000:
            kp_t[r["fen"]] = float(r["best_cp"])
    for pth in glob.glob(os.path.join(THIS, "ks_sets/kp_stress_on_d10_s*of2.csv")):
        for r in csv.DictReader(open(pth, newline="")):
            kp_d[r["fen"]] = float(r["ours_cp_white"])
    kf = [f for f in kp_t if f in kp_d]
    KP = rows_from(ai, kf, [kp_t[f] for f in kf], [kp_d[f] for f in kf])
    own = rows_from(ai, [OWNER_1004], [0.0], [0.0])
    sf_closer = lambda r: r["sf11"] is not None and abs(wp(r["sf11"]) - wp(r["tgt"])) < abs(wp(r["stat"]) - wp(r["tgt"]))
    kfl_live = any(abs(r["kfl"]) > 0 for r in R)
    print("PAWN-ENDING LIFT — depth rows (pawn-only): train %d · val %d · K+P stress (held-out) %d · KFL %s" %
          (len(tr), len(va), len(KP), "LIVE" if kfl_live else "off"))
    print("  owner 10-04 position: ours static %+.0f cp (truth 0.00) · leader-can-progress %d" % (own[0]["stat"], own[0]["prog"]))
    arms = [("ship", None, "none")] + [("lift " + f, f, "none") for f in "ABCD"]
    if kfl_live:
        arms += [("KFL alone", None, "on")] + [("KFL + lift " + f, f, "on") for f in "ABC"]
    for leg in ("depth", "static"):
        print("\n▶ %s leg (score = our %s · (1+g))" % (leg.upper(), "d10 search" if leg == "depth" else "static eval"))
        print("  %-14s %-16s %8s %8s %8s %13s %13s %8s %8s %8s" % ("arm", "params", "trainMSE", "valMSE", "val bias",
              "val SF11-cls", "val we-cls", "K+P MSE", "owner", "K+P bias"))
        vs, vw = [r for r in va if sf_closer(r)], [r for r in va if not sf_closer(r)]
        ref = {}
        for name, form, km in arms:
            p = np.zeros(1 if form == "A" else 2)
            if form:
                p = minimize(lambda q: mse(tr, form, q, leg, km), p, method="Nelder-Mead",
                             options=dict(xatol=1e-4, fatol=1e-4, maxiter=4000)).x
            row = (mse(tr, form, p, leg, km), mse(va, form, p, leg, km), bias(va, form, p, leg, km),
                   mse(vs, form, p, leg, km), mse(vw, form, p, leg, km), mse(KP, form, p, leg, km),
                   pred(form, p, own[0], "static", km), bias(KP, form, p, leg, km))
            if name == "ship":
                ref = row
            ps = " ".join("%+.2f" % x for x in p) if form else "—"
            print("  %-14s %-16s %8.1f %8.1f %+8.2f %6.1f (%+5.1f) %6.1f (%+5.1f) %8.1f %+7.0f %+8.2f" % (
                name, ps, row[0], row[1], row[2], row[3], row[3] - ref[3], row[4], row[4] - ref[4], row[5], row[6], row[7]))
        print("  (val SF11-cls = val rows where SF11's static beats ours — should FALL; we-cls = rows where ours beats SF11 — "
              "must NOT rise; n %d / %d)" % (len(vs), len(vw)))
    p11.stdin.write("quit\n"); p11.stdin.flush()


if __name__ == "__main__":
    main()
