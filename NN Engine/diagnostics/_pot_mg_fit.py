# -*- coding: utf-8 -*-
"""MIDDLEGAME POT test (2026-10-19; POT brief §5l — features, targets and the KEEP criterion were registered BEFORE this ran).
  MODE=fit    OMP_NUM_THREADS=1 pyrun diagnostics/_pot_mg_fit.py                       (python only, no engine)
  MODE=moves  OMP_NUM_THREADS=1 V2_PRESET=candidate pyrun diagnostics/_pot_mg_fit.py   (engine static on SF18 multi-PV children)

Owner's concept: POTENTIAL while a system is still UNRESOLVED, for and against the side to move — e.g. the opponent's pending
break (game 7). Features (White-positive; python-chess + _ovd_lever_proto's structure valuation, no engine):
  lever_now / lever_push   best structural gain (v2's own pawn values) a side can force by a pawn lever now / after one push
  stm_now / opp_push       the same split by WHO can act: side to move's immediate lever, the opponent's pending push-lever
  halfopen                 rooks + queens on half-open files facing an enemy pawn
  majority                 mobile pawn majority (no passer needed)
  storm                    own pawns advanced on the enemy king's files that can lever its shelter next move
  space                    safe squares behind own pawns, files c-f, own ranks 2-4 (SF space shape)
  each × U, U = unresolved (pawn tension on board, or any lever option for either side)
Targets, 5-fold by GAME (trajectory starts, E:\\chess_data\\deep + E:\\chess_data\\traj):
  A  deep residual      SF18 d14 at t = 0     vs our candidate d12 (queue #81)
  B  realised potential SF18 d14 at t + 16 / t + 24 plies of SF18 play
Arms: base = d12 · control = a + b·d12 (calibration only) · full = a + b·d12 + w·x · null = full with x shuffled.
KEEP (§5l): (1) A full vs CONTROL ≤ −2%, no class worse by > 2%, move test z > −1; OR (2) move test z ≥ +2 with A not worse.
"""
import os, sys, csv, glob, zlib, collections
import numpy as np
import chess
from scipy.optimize import minimize

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
KV = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
MODE = KV.get("MODE", "fit")
DEEP = KV.get("DEEP", "/mnt/e/chess_data/deep")
TRAJ = KV.get("TRAJ", "/mnt/e/chess_data/traj/traj_s*of4.csv")
THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS); sys.path.insert(0, os.path.dirname(THIS))
sys.argv = [sys.argv[0]]                       # _ovd_lever_proto parses argv at import
from _ovd_lever_proto import side_options, mobile_majority, board_bits, rel

K = 0.00368208
wp = lambda cp: 100.0 / (1.0 + np.exp(-K * np.clip(cp, -1500, 1500)))
NPM = {chess.KNIGHT: 1, chess.BISHOP: 1, chess.ROOK: 2, chess.QUEEN: 4}
BASE_F = ["lever_now", "lever_push", "stm_now", "opp_push", "halfopen", "majority", "storm", "space"]
NAMES = BASE_F + [n + "×U" for n in BASE_F]
CF = chess.BB_FILE_C | chess.BB_FILE_D | chess.BB_FILE_E | chess.BB_FILE_F


def side_feats(b, c):
    own, opp = b.pieces(chess.PAWN, c), b.pieces(chess.PAWN, not c)
    # half-open pressure: heavy pieces on files with no own pawn and an enemy pawn
    ho = 0
    for sq in b.pieces(chess.ROOK, c) | b.pieces(chess.QUEEN, c):
        fm = chess.BB_FILES[chess.square_file(sq)]
        if not (int(own) & fm) and (int(opp) & fm):
            ho += 1
    # storm: own pawns on the enemy king's files (±1), on own ranks ≥ 4, whose one-step push would attack an enemy pawn
    ek = b.king(not c); storm = 0
    kf = chess.square_file(ek)
    fwd = 8 if c == chess.WHITE else -8
    for sq in own:
        f, r = chess.square_file(sq), chess.square_rank(sq)
        rr = r if c == chess.WHITE else 7 - r
        if abs(f - kf) <= 1 and rr >= 3:
            to = sq + fwd
            if 0 <= to < 64 and not b.piece_at(to) and (chess.BB_PAWN_ATTACKS[c][to] & int(opp)):
                storm += 1
    # space: c-f, own ranks 2-4, not own pawn, not attacked by an enemy pawn, with an own pawn ahead on the file
    sp = 0
    enemy_att = 0
    for sq in opp:
        enemy_att |= chess.BB_PAWN_ATTACKS[not c][sq]
    for sq in chess.SquareSet(CF):
        r = chess.square_rank(sq); rr = r if c == chess.WHITE else 7 - r
        if 1 <= rr <= 3 and not (int(own) >> sq) & 1 and not (enemy_att >> sq) & 1:
            ahead = [s for s in own if chess.square_file(s) == chess.square_file(sq)
                     and ((chess.square_rank(s) > r) if c == chess.WHITE else (chess.square_rank(s) < r))]
            sp += bool(ahead)
    return ho, storm, sp


def feats(b):
    wpb, bpb, occ = board_bits(b.fen())
    npm = sum(NPM[p.piece_type] for p in b.piece_map().values() if p.piece_type in NPM)
    wmg = min(npm, 24) / 24.0
    base = rel(wpb, bpb, wmg)
    wn, wpu = side_options(wpb, bpb, occ, True, wmg, base)
    bn, bpu = side_options(wpb, bpb, occ, False, wmg, base)
    tension = any(chess.BB_PAWN_ATTACKS[chess.WHITE][s] & int(b.pieces(chess.PAWN, chess.BLACK)) for s in b.pieces(chess.PAWN, chess.WHITE))
    U = 1.0 if (tension or wn or bn or wpu or bpu) else 0.0          # a FLAG (first run used the option value: bug)
    if b.turn == chess.WHITE:
        stm_now, opp_push = wn, -bpu
    else:
        stm_now, opp_push = -bn, wpu
    hw, sw, pw = side_feats(b, chess.WHITE)
    hb, sb, pb = side_feats(b, chess.BLACK)
    x = [wn - bn, wpu - bpu, stm_now, opp_push, hw - hb, mobile_majority(wpb, bpb, occ), sw - sb, pw - pb]
    return x + [v * U for v in x], U


def load():
    d12 = {}
    for p in glob.glob(os.path.join(DEEP, "mg_cand_d12_s*of3.csv")):
        for r in csv.DictReader(open(p, newline="")):
            if r["depth"]:                                            # 92 early mates have no depth: skip
                d12[r["fen"]] = float(r["ours_cp_white"])
    tr = collections.defaultdict(dict)
    meta = {}
    for p in sorted(glob.glob(TRAJ)):
        for r in csv.DictReader(open(p, newline="")):
            if r["start_stage"] not in ("mg", "mgl"):
                continue
            if r["sf_k"]:                                             # empty after a game ended before checkpoint k
                tr[r["start_fen"]][int(r["k"])] = float(r["sf_k"])
            if r["k"] == "0":
                meta[r["start_fen"]] = (r["game"].split("#")[0], r["start_stage"], r["control"], float(r["ours_static_k"]))
    return d12, tr, meta


def fit_arm(s, X, t, use_x):
    """pred = a + b·s (+ X·w), cp; loss = win% MSE. X standardised by the caller."""
    n = 2 + (X.shape[1] if use_x else 0)
    def pred(th, s_, X_):
        v = th[0] + (1.0 + th[1]) * s_
        return v + X_ @ th[2:] if use_x else v
    f = lambda th: np.mean((wp(pred(th, s, X)) - wp(t)) ** 2) + 1e-4 * np.sum(th[2:] ** 2)
    th = minimize(f, np.zeros(n), method="L-BFGS-B").x
    return lambda s_, X_: pred(th, s_, X_), th


def cv(s, X, t, fold, shuffle=False, seed=0):
    out = {"control": np.zeros_like(t), "full": np.zeros_like(t)}
    W = []
    rng = np.random.default_rng(seed)
    for k in range(5):
        tr, te = fold != k, fold == k
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        Xtr, Xte = (X[tr] - mu) / sd, (X[te] - mu) / sd
        if shuffle:
            Xtr, Xte = Xtr[rng.permutation(len(Xtr))], Xte[rng.permutation(len(Xte))]
        pc, _ = fit_arm(s[tr], Xtr, t[tr], False)
        pf, th = fit_arm(s[tr], Xtr, t[tr], True)
        out["control"][te] = pc(s[te], Xte)
        out["full"][te] = pf(s[te], Xte)
        W.append(th[2:] / sd)
    return out, np.mean(W, 0)


def main_fit():
    d12, tr, meta = load()
    fens = [f for f in d12 if f in meta and 0 in tr[f] and abs(tr[f][0]) < 20000]
    print("rows: %d starts with d12 + SF18 t0 (of %d d12, %d trajectory starts)" % (len(fens), len(d12), len(meta)), flush=True)
    X, U = [], []
    for i, f in enumerate(fens):
        x, u = feats(chess.Board(f)); X.append(x); U.append(u)
        if (i + 1) % 5000 == 0:
            print("  features %d" % (i + 1), flush=True)
    X, U = np.array(X, float), np.array(U)
    s = np.array([d12[f] for f in fens])
    game = np.array([meta[f][0] for f in fens])
    stage = np.array([meta[f][1] for f in fens]); ctrl = np.array([meta[f][2] for f in fens])
    queens = np.array([bool(chess.Board(f).pieces(chess.QUEEN, chess.WHITE) | chess.Board(f).pieces(chess.QUEEN, chess.BLACK)) for f in fens])
    fold = np.array([zlib.crc32(g.encode()) % 5 for g in game])
    print("unresolved (U=1): %.1f%% · features non-zero: %s" % (100 * U.mean(), ", ".join(
        "%s %.0f%%" % (n, 100 * np.mean(X[:, i] != 0)) for i, n in enumerate(BASE_F))))
    mse = lambda p, t, m: float(np.mean((wp(p[m]) - wp(t[m])) ** 2))
    bias = lambda p, t, m: float(np.mean((wp(p[m]) - wp(t[m])) * np.sign(t[m])))
    verdict = None
    for tname, kk in (("A  deep residual: SF18 t0", 0), ("B  realised: SF18 t+16", 16), ("B  realised: SF18 t+24", 24)):
        ok = np.array([kk in tr[f] and abs(tr[f][kk]) < 20000 for f in fens])
        t = np.array([tr[f].get(kk, 0.0) for f in fens])
        out, Wm = cv(s[ok], X[ok], t[ok], fold[ok])
        nul, _ = cv(s[ok], X[ok], t[ok], fold[ok], shuffle=True, seed=kk)
        tt, base = t[ok], s[ok]
        print("\n=== %s  (n %d, 5-fold by game) — held-out win%% MSE (side-ahead bias)" % (tname, ok.sum()))
        print("%-22s %6s  %-15s %-15s %-15s %-15s  full/control" % ("class", "n", "base d12", "control", "full", "null(shuf)"))
        classes = [("ALL", np.ones(ok.sum(), bool)), ("unresolved U=1", U[ok] == 1), ("resolved U=0", U[ok] == 0),
                   ("stage mg", stage[ok] == "mg"), ("stage mgl", stage[ok] == "mgl"), ("selected (ctrl 0)", ctrl[ok] == "0"),
                   ("random (ctrl 1)", ctrl[ok] == "1"), ("queens on", queens[ok]), ("queens off", ~queens[ok])]
        worst = 0.0
        for cname, m in classes:
            if m.sum() < 50:
                continue
            c0, f0 = mse(out["control"], tt, m), mse(out["full"], tt, m)
            g = 100 * (f0 / c0 - 1)
            if cname != "ALL":
                worst = max(worst, g)
            print("%-22s %6d  %6.2f (%+5.2f)  %6.2f (%+5.2f)  %6.2f (%+5.2f)  %6.2f          %+6.2f%%" % (
                cname, m.sum(), mse(base, tt, m), bias(base, tt, m), c0, bias(out["control"], tt, m), f0,
                bias(out["full"], tt, m), mse(nul["full"], tt, m), g))
        allm = np.ones(ok.sum(), bool)
        gain = 100 * (mse(out["full"], tt, allm) / mse(out["control"], tt, allm) - 1)
        print("weights (cp per unit, mean over folds):", ", ".join("%s %+.3g" % (n, w) for n, w in zip(NAMES, Wm)))
        print("full vs control %+.2f%% · full vs base %+.2f%% · worst class %+.2f%%" % (
            gain, 100 * (mse(out["full"], tt, allm) / mse(base, tt, allm) - 1), worst))
        if kk == 0:
            verdict = (gain, worst)
    g, w = verdict
    print("\nKEEP part (1), A: full vs control %+.2f%% (needs ≤ −2.00%%) · worst class %+.2f%% (needs ≤ +2%%) ⇒ %s"
          "  [move test still to run: MODE=moves]" % (g, w, "PASS so far" if g <= -2 and w <= 2 else "MISS"))


AI = None


def ours_static(b):
    global AI
    if AI is None:
        import ChessAI
        AI = ChessAI.ChessAI(None, None, chess.Board(), True)
    return -int(AI.ev_breakdown(b)["total"]) / 10.0


def logit_fit(Z, y):
    """win% ≈ 100·σ(Z·w + c), least squares in win% (static → SF18 t0), Z standardised by the caller."""
    f = lambda th: np.mean((100.0 / (1.0 + np.exp(-(Z @ th[1:] + th[0]))) - y) ** 2) + 1e-4 * np.sum(th[1:] ** 2)
    th = minimize(f, np.zeros(Z.shape[1] + 1), method="L-BFGS-B").x
    return lambda Zn: 100.0 / (1.0 + np.exp(-(Zn @ th[1:] + th[0])))


def main_moves():
    _, tr, meta = load()
    rows = [(f, m) for f, m in meta.items() if 0 in tr[f] and abs(tr[f][0]) < 20000]
    print("training rows (static t0 → SF18 t0): %d" % len(rows), flush=True)
    S = np.array([[np.clip(m[3], -1500, 1500)] for f, m in rows])
    F = np.array([feats(chess.Board(f))[0] for f, m in rows], float)
    Y = wp(np.array([tr[f][0] for f, m in rows]))
    gf = lambda g: zlib.crc32(g.encode()) % 5
    rf = np.array([gf(m[0]) for f, m in rows])
    models = []
    for k in range(5):
        t_ = rf != k
        XA, XB = S[t_], np.hstack([S[t_], F[t_]])
        ma, sa_ = XA.mean(0), XA.std(0) + 1e-9
        mb, sb_ = XB.mean(0), XB.std(0) + 1e-9
        pa, pb = logit_fit((XA - ma) / sa_, Y[t_]), logit_fit((XB - mb) / sb_, Y[t_])
        models.append((lambda z, pa=pa, ma=ma, sa_=sa_: pa((z - ma) / sa_), lambda z, pb=pb, mb=mb, sb_=sb_: pb((z - mb) / sb_)))
    mpv = [(r["fen"], [(u, float(c)) for u, c in (x.split(":") for x in r["moves"].split(";") if ":" in x)])
           for r in csv.DictReader(open(os.path.join(DEEP, "mg_mpv.csv"), newline=""))]
    mpv = [(f, mv) for f, mv in mpv if len(mv) >= 2 and f in meta]
    fixed = broke = pairs = right = ncrit = top_a = top_b = 0
    for i, (fen, mv) in enumerate(mpv):
        pa, pb = models[gf(meta[fen][0])]
        b = chess.Board(fen); white = b.turn
        sfw = {u: (wp(c) if white else 100 - wp(c)) for u, c in mv}
        best = max(sfw, key=sfw.get)
        ordd = sorted(sfw.values(), reverse=True)
        if ordd[0] - ordd[1] < 15:
            continue
        ncrit += 1
        sa, sb = {}, {}
        for u in sfw:
            m = chess.Move.from_uci(u)
            if m not in b.legal_moves:
                continue
            b.push(m)
            if b.is_game_over():
                sa[u] = sb[u] = 100.0 if b.is_checkmate() else 50.0
            else:
                st = np.clip(ours_static(b), -1500, 1500)
                x = feats(b)[0]
                va, vb = float(pa(np.array([[st]]))[0]), float(pb(np.array([[st] + x]))[0])
                sa[u], sb[u] = (va, vb) if white else (100 - va, 100 - vb)
            b.pop()
        if best not in sa:
            continue
        top_a += max(sa, key=sa.get) == best; top_b += max(sb, key=sb.get) == best
        for u in sa:
            if u != best and sfw[best] - sfw[u] >= 10:
                pairs += 1
                right += (sb[best] - sa[best]) > (sb[u] - sa[u])
                ra, rb = sa[best] > sa[u], sb[best] > sb[u]
                fixed += rb and not ra; broke += ra and not rb
        if (i + 1) % 500 == 0:
            print("  %d / %d multi-PV positions" % (i + 1, len(mpv)), flush=True)
    z = (fixed - broke) / max(1.0, (fixed + broke) ** 0.5)
    print("\nMOVE TEST — %d multi-PV positions, %d critical (best − 2nd ≥ 15 win%% pts), %d pairs vs ≥ 10-pt losers" % (len(mpv), ncrit, pairs))
    print("  top-1 = SF best on critical: static %.1f%% → static + POT %.1f%%" % (100 * top_a / max(1, ncrit), 100 * top_b / max(1, ncrit)))
    print("  POT term alone orders SF's best above the loser: %.1f%% of pairs (50%% = blind)" % (100 * right / max(1, pairs)))
    print("  KEEP TEST (paired): POT FIXES %d · BREAKS %d · z = %+.2f  (part 1 needs z > −1; part 2 needs z ≥ +2)" % (fixed, broke, z))


if __name__ == "__main__":
    main_moves() if MODE == "moves" else main_fit()
