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


def feats_sum(b):
    """POT's MAGNITUDE half (POT-REREAD-ANALYSIS §2.1, test A): the same structure SUMMED over both sides, never netted
    (a position where both sides have a live break is the most unresolved and nets to zero). Plus a tension count."""
    wpb, bpb, occ = board_bits(b.fen())
    npm = sum(NPM[p.piece_type] for p in b.piece_map().values() if p.piece_type in NPM)
    wmg = min(npm, 24) / 24.0
    base = rel(wpb, bpb, wmg)
    wn, wpu = side_options(wpb, bpb, occ, True, wmg, base)
    bn, bpu = side_options(wpb, bpb, occ, False, wmg, base)
    wp_, bp_ = b.pieces(chess.PAWN, chess.WHITE), b.pieces(chess.PAWN, chess.BLACK)
    tension = sum(bin(chess.BB_PAWN_ATTACKS[chess.WHITE][s] & int(bp_)).count("1") for s in wp_)
    hw, sw, pw = side_feats(b, chess.WHITE)
    hb, sb, pb = side_feats(b, chess.BLACK)
    return [abs(wn) + abs(bn), abs(wpu) + abs(bpu), hw + hb, abs(mobile_majority(wpb, bpb, occ)), sw + sb, pw + pb, tension]


MAG_NAMES = ["lever_now_sum", "lever_push_sum", "halfopen_sum", "majority_abs", "storm_sum (king files)", "space_sum", "tension"]


def main_magnitude():
    """TEST A (registered 11-10 20:45, before this ran; POT-REREAD-ANALYSIS §5, search v2 plan §7c):
    does pawn structure predict the SIZE of our error? Rows = mg / mgl trajectory starts with our candidate d12 score (queue
    #81) and SF18's own WDL label (mg_wdl_train_s*of3.csv, mg_wdl_val.csv; White POV). E = |win%(SF18 WDL) − win%(our d12)|.
    Controls C: |win%(d12) − 50|, phase (npm), |material diff|, |win%(our static) − win%(d12)|. Features F = feats_sum.
    5-fold by GAME. In each fold: fit E ~ C (train), fit the train residual ~ F, predict U = F·β on held-out. Read the
    held-out ratio of mean E, top fifth of U over bottom fifth, WITHIN terciles of the control prediction (beyond the
    controls), averaged over terciles. Second target: |win%(SF18 t+16) − win%(SF18 t0)| (realised change; cp label).
    KEEP: ratio >= 1.25 in >= 4 of 5 folds on the primary target. Shuffled-F null reported.
    Prediction: 1.05-1.15 (miss); the quiet-row replication runs only on a pass.
    RESULT (11-10 18:55, 26,244 rows): per fold 1.248 1.247 1.236 1.236 1.218 (0 / 5 at >= 1.25) ⇒ registered MISS;
    shuffled null 0.95-1.08. A consistent ~1.24x effect beyond the controls, just under the bar (prediction too low).
    Second target (t+16 change): 1.167 1.271 1.206 1.347 1.247 (2 / 5). Owner decides on the quiet-row replication."""
    d12, tr, meta = load()
    lab = {}
    for p in glob.glob(os.path.join(DEEP, "mg_wdl_train_s*of3.csv")) + [os.path.join(DEEP, "mg_wdl_val.csv")]:
        for r in csv.DictReader(open(p, newline="")):
            try:
                lab[r["fen"]] = (float(r["w"]) + 0.5 * float(r["d"])) / 10.0   # White expected score, %
            except (KeyError, ValueError):
                pass
    rows = []
    for fen, (game, stage, control, static) in meta.items():
        if fen not in d12 or fen not in lab:
            continue
        b = chess.Board(fen)
        npm = sum(NPM[p.piece_type] for p in b.piece_map().values() if p.piece_type in NPM)
        val = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}
        mat = sum(val[p.piece_type] * (1 if p.color == chess.WHITE else -1) for p in b.piece_map().values() if p.piece_type in val)
        w12 = float(wp(d12[fen]))
        e = abs(lab[fen] - w12)
        t16 = tr[fen].get(16)
        e16 = abs(float(wp(t16)) - float(wp(tr[fen][0]))) if (t16 is not None and 0 in tr[fen]) else np.nan
        C = [abs(w12 - 50.0), npm, abs(mat), abs(float(wp(static)) - w12)]
        rows.append((game, e, e16, C, feats_sum(b)))
    print("test A rows: %d (trajectory starts with a d12 score and an SF18 WDL label)" % len(rows))
    games = np.array([zlib.crc32(r[0].encode()) % 5 for r in rows])
    E = np.array([r[1] for r in rows]); E16 = np.array([r[2] for r in rows])
    Cm = np.array([r[3] for r in rows], float); Fm = np.array([r[4] for r in rows], float)
    print("mean E %.2f win%% pts; feature means: %s" % (E.mean(), ", ".join("%s %.2f" % (n, v) for n, v in zip(MAG_NAMES, Fm.mean(0)))))

    def ols(X, y):
        X1 = np.c_[np.ones(len(X)), X]
        return np.linalg.lstsq(X1, y, rcond=None)[0]

    def pred(beta, X):
        return np.c_[np.ones(len(X)), X] @ beta

    def run(target, shuffle_seed=None):
        ratios = []
        ok_all = ~np.isnan(target)
        for k in range(5):
            tr_m, te_m = (games != k) & ok_all, (games == k) & ok_all
            F_tr = Fm[tr_m].copy()
            if shuffle_seed is not None:
                np.random.default_rng(shuffle_seed + k).shuffle(F_tr)
            bc = ols(Cm[tr_m], target[tr_m])
            res = target[tr_m] - pred(bc, Cm[tr_m])
            bf = ols(F_tr, res)
            u = pred(bf, Fm[te_m]); cpred = pred(bc, Cm[te_m]); t = target[te_m]
            per = []
            for lo, hi in ((0, 1 / 3), (1 / 3, 2 / 3), (2 / 3, 1.0001)):
                q = np.quantile(cpred, [lo, min(hi, 1.0)])
                m = (cpred >= q[0]) & (cpred <= q[1]) if hi > 1 else (cpred >= q[0]) & (cpred < q[1])
                if m.sum() < 50:
                    continue
                uq = np.quantile(u[m], [0.2, 0.8])
                top, bot = t[m][u[m] >= uq[1]], t[m][u[m] <= uq[0]]
                if len(top) and len(bot) and bot.mean() > 0:
                    per.append(top.mean() / bot.mean())
            ratios.append(float(np.mean(per)) if per else float("nan"))
        return ratios

    for name, target in (("PRIMARY |SF18 WDL − our d12|", E), ("second |SF18 t+16 − SF18 t0| (cp label)", E16)):
        r = run(target)
        rn = run(target, shuffle_seed=7)
        npass = sum(x >= 1.25 for x in r)
        print("\n=== %s" % name)
        print("  top/bottom-fifth ratio of mean error, within control terciles, per fold: %s  (pass >= 1.25: %d / 5)"
              % (" ".join("%.3f" % x for x in r), npass))
        print("  shuffled-feature null per fold: %s" % " ".join("%.3f" % x for x in rn))
        if name.startswith("PRIMARY"):
            print("  REGISTERED: %s" % ("KEEP: structure predicts the size of our error beyond the controls" if npass >= 4
                                      else "MISS: structural magnitude does not predict our error beyond the controls ⇒ the structural U half closes"))


def main_depthgain():
    """TEST B (registered 11-10 19:00, before the depth-12 pass finished; POT-REREAD-ANALYSIS §5): is extra depth worth more
    where structural U is high? U = F·β, β fitted on ALL of test A's rows (the d12-error residual after test A's controls
    regressed on feats_sum), then applied out of sample to game_regret_set.csv. Per position: gain = regret(our d10 move) −
    regret(our d12 move) from queue #193 (SF18 multi-PV d14 labels; the cp label through the fixed formula, shared by both
    depths of a row). Read: mean gain in the top fifth of U over the bottom fifth.
    KEEP: >= 1.5x on all rows, then a replication on a second set (game_regret_set_v2) before it counts.
    Prediction: 1.2-1.6x. Phases reported (opening / midgame / endgame by the corpus bucket).
    RESULT (11-10 20:00, 15,000 rows, mean gain d10 → d12 0.172 win% pts): ALL ratio 1.13 (top − bottom +0.023 ± 0.066,
    1 SE) ⇒ registered MISS. Opening 0.97, midgame 1.08, endgame 1.95 (+0.116 ± 0.119): every split within 1 SE.
    ☠️ UNDERPOWERED, so not a refutation: the 1.5x bar is a gap of ~+0.07, about 1 SE here (per-row regret gain is mostly
    0 with rare large values), and the 2σ upper bound on ALL is ~1.9x. A readable test B needs ~4x the rows or a
    lower-variance target (critical rows only)."""
    d12, tr, meta = load()
    lab = {}
    for p in glob.glob(os.path.join(DEEP, "mg_wdl_train_s*of3.csv")) + [os.path.join(DEEP, "mg_wdl_val.csv")]:
        for r in csv.DictReader(open(p, newline="")):
            try:
                lab[r["fen"]] = (float(r["w"]) + 0.5 * float(r["d"])) / 10.0
            except (KeyError, ValueError):
                pass
    C, F, E = [], [], []
    val = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}
    for fen, (game, stage, control, static) in meta.items():
        if fen not in d12 or fen not in lab:
            continue
        b = chess.Board(fen)
        npm = sum(NPM[p.piece_type] for p in b.piece_map().values() if p.piece_type in NPM)
        mat = sum(val[p.piece_type] * (1 if p.color == chess.WHITE else -1) for p in b.piece_map().values() if p.piece_type in val)
        w12 = float(wp(d12[fen]))
        E.append(abs(lab[fen] - w12)); C.append([abs(w12 - 50.0), npm, abs(mat), abs(float(wp(static)) - w12)]); F.append(feats_sum(b))
    C, F, E = np.array(C, float), np.array(F, float), np.array(E)
    X1 = np.c_[np.ones(len(C)), C]
    res = E - X1 @ np.linalg.lstsq(X1, E, rcond=None)[0]
    beta = np.linalg.lstsq(np.c_[np.ones(len(F)), F], res, rcond=None)[0]
    print("β from test A (%d rows): %s" % (len(E), ", ".join("%s %+.3f" % (n, v) for n, v in zip(["const"] + MAG_NAMES, beta))))
    reg = {}
    for dep in (10, 12):
        for p in glob.glob("/mnt/e/chess_data/s0_testB/d%d_s*.csv" % dep):
            for r in csv.reader(open(p, newline="")):
                if len(r) >= 4 and r[1]:
                    reg.setdefault(r[0], {})[dep] = (float(r[2]), r[3])
    rows = [(fen, v[10][0] - v[12][0], v[10][1]) for fen, v in reg.items() if 10 in v and 12 in v]
    U = np.array([np.r_[1.0, feats_sum(chess.Board(f))] @ beta for f, _, _ in rows])
    G = np.array([g for _, g, _ in rows]); ph = np.array([p for _, _, p in rows])
    print("test B rows: %d with both depths; mean regret gain d10 → d12 %.3f win%% pts" % (len(rows), G.mean()))
    for name, m in (("ALL", np.ones(len(G), bool)), ("opening", ph == "opening"), ("midgame", ph == "midgame"), ("endgame", ph == "endgame")):
        if m.sum() < 200:
            continue
        q = np.quantile(U[m], [0.2, 0.8])
        gt, gb = G[m][U[m] >= q[1]], G[m][U[m] <= q[0]]
        top, bot = gt.mean(), gb.mean()
        se = (gt.var(ddof=1) / len(gt) + gb.var(ddof=1) / len(gb)) ** 0.5     # SE of top − bottom
        ratio = top / bot if bot > 0 else float("nan")
        print("  %-8s n %5d · gain top fifth %.3f · bottom fifth %.3f · ratio %s · top − bottom %+.3f ± %.3f (1 SE)"
              % (name, m.sum(), top, bot, ("%.2f" % ratio) if bot > 0 else "n/a (bottom gain <= 0)", top - bot, se))
        if name == "ALL":
            print("  REGISTERED: %s" % ("KEEP on this set (needs the second-set replication)" if bot > 0 and ratio >= 1.5 else "MISS"))


if __name__ == "__main__":
    {"moves": main_moves, "magnitude": main_magnitude, "depthgain": main_depthgain}.get(MODE, main_fit)()
