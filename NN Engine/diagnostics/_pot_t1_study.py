# -*- coding: utf-8 -*-
"""POT T1 study — CENTRAL OPENING vs an UNCASTLED KING, worked backwards (C3 doc §17, POT-TRANSFORMATION-KNOWLEDGE §2).

POT = "OvD reworked" (the owner's v1 invention). Pure Python, no engine. Structure only — never a subsystem's score.

SAMPLE. Stage-1 quiet positions (fitC_stage1, d6 self-play, ordered by game/ply). For each position and each DEFENDER D
whose king is on files d/e ("central"), with the attacker A = the other side:
  UNRESOLVED gate: no half-open file for A near D's king at t (every file in kf−1..kf+1 ∩ c..f holds a D pawn) —
  i.e. the RESULT is absent. Rows failing it are KINETIC (KS's business) and are counted, not studied.
EVENT (the transformation's END RESULT, structural): at the first stored position with ply ≥ t+N in the same game,
  D's king is still on files d-f (did not castle away) AND a file in kf−1..kf+1 ∩ c..f has lost all D pawns.
PRECURSORS (textbook, §1 "the human test"):
  rights    D still has a castling right              block   min pieces between D's king and a rook it may castle with
  levers    A pawns on c-f in contact with a D pawn   push    A's SUPPORTED one-step pushes on c-f making contact
  rams      A pawns on c-f blocked head-on by a D pawn heavy   A rooks/queens on the files kf−1..kf+1
  dev       A minors off the back rank − D's           stm     A to move (tempo)
  queens    both queens on
REPORT: base rates (unresolved vs kinetic), event rate per precursor level, a logistic model (train/val by game hash) with
AUC, and the event's later RESULT for D. Then OVERLAP: corr of the fitted P(event) with the engine's KS (fitC_ks2).

  pyrun diagnostics/_pot_t1_study.py [GAMES=10000] [N=20]
"""
import os, sys, hashlib, math
import numpy as np
import pandas as pd
import chess

KV = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
GAMES = int(KV.get("GAMES", 10000))
N = int(KV.get("N", 20))
DATA = KV.get("DATA", "/mnt/e/chess_data/texel")
FEATS = ["rights", "block", "levers", "push", "rams", "heavy", "dev", "stm", "queens"]


def file_mask(f):
    return chess.BB_FILES[f]


def half_open_near(b, D, kf):
    """Files in kf−1..kf+1 ∩ c..f with no D pawn."""
    dp = int(b.pieces(chess.PAWN, D))
    n = 0
    for f in range(max(2, kf - 1), min(5, kf + 1) + 1):
        if not dp & file_mask(f):
            n += 1
    return n


def castle_block(b, D):
    best = 9
    rank = 0 if D == chess.WHITE else 7
    occ = int(b.occupied)
    for side, rook_f, between in ((0, 7, (5, 6)), (1, 0, (1, 2, 3))):
        has = b.has_kingside_castling_rights(D) if side == 0 else b.has_queenside_castling_rights(D)
        if has:
            best = min(best, sum(1 for f in between if occ >> (rank * 8 + f) & 1))
    return best


def features(b, D):
    A = not D
    ap, dp = int(b.pieces(chess.PAWN, A)), int(b.pieces(chess.PAWN, D))
    kf = chess.square_file(b.king(D))
    centre = chess.BB_FILE_C | chess.BB_FILE_D | chess.BB_FILE_E | chess.BB_FILE_F
    fwd = 8 if A == chess.WHITE else -8
    levers = rams = push = 0
    for sq in chess.SquareSet(ap & centre):
        if chess.BB_PAWN_ATTACKS[A][sq] & dp:
            levers += 1
        t = sq + fwd
        if 0 <= t < 64:
            if dp >> t & 1:
                rams += 1
            elif not (int(b.occupied) >> t & 1):
                # contact is mutual (the D pawn it hits also hits it), so the push must be SUPPORTED by an A pawn
                makes_contact = chess.BB_PAWN_ATTACKS[A][t] & dp
                supported = chess.BB_PAWN_ATTACKS[D][t] & ap          # A pawns that defend t
                if makes_contact and supported:
                    push += 1
    heavy = 0
    for f in range(max(0, kf - 1), min(7, kf + 1) + 1):
        heavy += len(chess.SquareSet(int(b.pieces(chess.ROOK, A) | b.pieces(chess.QUEEN, A)) & file_mask(f)))
    def dev(S):
        back = chess.BB_RANK_1 if S == chess.WHITE else chess.BB_RANK_8
        return len(chess.SquareSet(int(b.pieces(chess.KNIGHT, S) | b.pieces(chess.BISHOP, S)) & ~back))
    rights = int(b.has_castling_rights(D))
    return [rights, castle_block(b, D), levers, push, rams, heavy, dev(A) - dev(D), int(b.turn == A),
            int(bool(b.pieces(chess.QUEEN, chess.WHITE)) and bool(b.pieces(chess.QUEEN, chess.BLACK)))]


def auc(score, y):
    order = np.argsort(score)
    r = np.empty(len(score)); r[order] = np.arange(1, len(score) + 1)
    pos = y == 1
    return (r[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / max(pos.sum() * (~pos).sum(), 1)


def main():
    st = pd.read_csv(os.path.join(DATA, "fitC_stage1.csv.gz"), usecols=["game_id", "ply", "fen", "result_white"])
    st["row"] = np.arange(len(st))
    games = st["game_id"].unique()[:GAMES]
    st = st[st["game_id"].isin(set(games))]
    X, y, res, rows, val, dside = [], [], [], [], [], []
    kinetic = unresolved = 0
    for gid, g in st.groupby("game_id", sort=False):
        plies, fens = g["ply"].values, g["fen"].values
        isval = int(hashlib.md5(gid.encode()).hexdigest()[:8], 16) % 100 < 20
        for i in range(len(g)):
            b = chess.Board(fens[i])
            if chess.popcount(int(b.occupied)) < 20:          # middlegame-ish only
                continue
            j = np.searchsorted(plies, plies[i] + N)
            if j >= len(g):
                continue
            fut = None
            for D in (chess.WHITE, chess.BLACK):
                kf = chess.square_file(b.king(D))
                if kf not in (3, 4):
                    continue
                if half_open_near(b, D, kf):
                    kinetic += 1
                    continue
                unresolved += 1
                if fut is None:
                    fut = chess.Board(fens[j])
                fk = chess.square_file(fut.king(D))
                ev = int(fk in (3, 4, 5) and half_open_near(fut, D, fk) > 0)
                X.append(features(b, D)); y.append(ev); rows.append(g["row"].values[i]); val.append(isval)
                rw = g["result_white"].values[i]
                res.append(rw if D == chess.WHITE else 1 - rw)
                dside.append(int(D == chess.BLACK))
    X, y, res, rows, val = np.array(X, float), np.array(y), np.array(res, float), np.array(rows), np.array(val)
    print("T1 STUDY  games %d  N=%d plies | central-king rows: KINETIC (lines already open) %d · UNRESOLVED %d"
          % (len(games), N, kinetic, unresolved))
    print("  event (lines open toward a still-central king by t+%d): %.1f%%   D's score: event %.3f vs no-event %.3f"
          % (N, 100 * y.mean(), res[y == 1].mean(), res[y == 0].mean()))
    print("\n  precursor    level -> event rate (n)")
    for k, name in enumerate(FEATS):
        v = X[:, k]
        lv = np.unique(np.clip(v, -3, 4))
        cells = []
        for L in lv:
            m = np.clip(v, -3, 4) == L
            if m.sum() >= 200:
                cells.append("%g:%.1f%%(%d)" % (L, 100 * y[m].mean(), m.sum()))
        print("  %-8s " % name + "  ".join(cells))
    # logistic model
    from scipy.optimize import minimize
    mu, sd = X[~val].mean(0), X[~val].std(0) + 1e-9
    Z = np.c_[(X - mu) / sd, np.ones(len(X))]
    def nll(w):
        p = 1 / (1 + np.exp(-np.clip(Z[~val] @ w, -30, 30)))
        return -np.mean(y[~val] * np.log(p + 1e-12) + (1 - y[~val]) * np.log(1 - p + 1e-12)) + 1e-4 * w[:-1] @ w[:-1]
    w = minimize(nll, np.zeros(Z.shape[1]), method="L-BFGS-B").x
    s = Z @ w
    print("\n  logistic (standardised coefs): " + " ".join("%s=%+.2f" % (n, c) for n, c in zip(FEATS, w[:-1])))
    print("  AUC train %.3f · VAL %.3f  (val rows %d)" % (auc(s[~val], y[~val]), auc(s[val], y[val]), val.sum()))
    # overlap with the engine's KS
    try:
        zk = np.load(os.path.join(DATA, "fitC_ks2.npz"))
        ks = zk["ks_engine"][rows].astype(float)
        dsgn = np.where(np.array(dside) == 1, -1.0, 1.0)            # Black-positive KS ⇒ danger to D as D-negative
        print("  OVERLAP: corr(P(event) score, engine KS toward D) %+.3f  (KS non-zero on %.1f%% of these rows)"
              % (np.corrcoef(s, ks * dsgn)[0, 1], 100 * (ks != 0).mean()))
    except Exception as e:
        print("  overlap skipped:", e)
    np.savez_compressed(os.path.join(DATA, "pot_t1_study.npz"), X=X, y=y, res=res, rows=rows, val=val,
                        dside=np.array(dside), score=s, names=np.array(FEATS), mu=mu, sd=sd, w=w)
    sf_check(mu, sd, w)


def sf_check(mu, sd, w):
    """Does the predicted T1 potential explain SF18's disagreement with our SHIPPED eval (depth-independent)?
    On the SF18-labelled standard middlegame rows, UNRESOLVED central kings only: residual toward the ATTACKER =
    win%(SF18 search) − win%(ours), from A's view; correlated with the P(event) score, overall and in the balanced band."""
    import csv
    THIS = os.path.dirname(os.path.abspath(__file__))
    K = 0.00368208
    wp = lambda cp: 100.0 / (1.0 + np.exp(-K * np.clip(cp, -1500, 1500)))
    sample = {r["fen"]: int(r["row"]) for r in csv.DictReader(open(os.path.join(THIS, "ks_sets/fitC_mg_sample.csv")))
              if r["src"] == "std"}
    zw = np.load(os.path.join(DATA, "fitC_win.npz"))
    zk = np.load(os.path.join(DATA, "fitC_ks2.npz"))
    sc, rA, ours_A, ksD, fx = [], [], [], [], []
    for r in csv.DictReader(open(os.path.join(THIS, "ks_sets/fitC_mg_sf18.csv"))):
        if not r.get("best_cp") or r["fen"] not in sample:
            continue
        b = chess.Board(r["fen"])
        row = sample[r["fen"]]
        T = float(zw["total"][row])
        sf_w = float(r["best_cp"])                                         # White-POV (_build_regret_set)
        ours_w = -T / 10.0
        for D in (chess.WHITE, chess.BLACK):
            kf = chess.square_file(b.king(D))
            if kf not in (3, 4) or half_open_near(b, D, kf):
                continue
            z = np.r_[(np.array(features(b, D), float) - mu) / sd, 1.0]
            sgnA = -1.0 if D == chess.WHITE else 1.0                          # White-POV → A's POV
            sc.append(z @ w)
            fx.append(features(b, D))
            rA.append(sgnA * (wp(sf_w) - wp(ours_w)))
            ours_A.append(sgnA * ours_w)
            ksD.append(float(zk["ks_engine"][row]) * (1.0 if D == chess.WHITE else -1.0))
    sc, rA, ours_A, ksD = map(np.array, (sc, rA, ours_A, ksD))
    print("\n  SF18 CHECK: unresolved central-king rows %d  mean residual toward A %+.2f pp" % (len(sc), rA.mean()))
    for name, m in (("all", np.ones(len(sc), bool)), ("balanced |ours|<100cp", np.abs(ours_A) < 100)):
        if m.sum() > 50:
            r = np.corrcoef(sc[m], rA[m])[0, 1]
            print("    %-22s n %5d  corr(score, residual→A) %+.4f  (%.1fσ)" % (name, m.sum(), r, r * math.sqrt(m.sum() - 3)))
    fx = np.array(fx, float)
    print("    per precursor corr(feature, residual→A): " + " ".join(
        "%s=%+.3f" % (n, np.corrcoef(fx[:, k], rA)[0, 1]) for k, n in enumerate(FEATS) if fx[:, k].std() > 0))
    q = np.quantile(sc, [0.2, 0.4, 0.6, 0.8])
    e = [-np.inf] + list(q) + [np.inf]
    for lo, hi in zip(e[:-1], e[1:]):
        m = (sc > lo) & (sc <= hi)
        print("    score quintile (%6.2f,%6.2f]  n %5d  mean residual→A %+.2f pp" % (lo, hi, m.sum(), rA[m].mean()))
    # the KS channel on the same rows, for the no-overlap reading (a KS-direction check, sign not yet verified)
    if ksD.std() > 0:
        print("    overlap: corr(score, KS on these rows) %+.3f  (KS non-zero %.1f%%)"
              % (np.corrcoef(sc, ksD)[0, 1], 100 * (ksD != 0).mean()))


if __name__ == "__main__":
    main()
