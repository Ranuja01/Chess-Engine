# -*- coding: utf-8 -*-
"""POT RACE-MARGIN PROTOTYPE, stage 1 (2026-10-15, owner design discussion; POT brief §5) — Python only, no engine change.

  V2_PRESET=shipped pyrun diagnostics/_pot_race_proto.py [IN=/mnt/e/chess_data/retune/EGX_sf18.csv] [MAX=0] [DUMP=<csv>]

The human procedure in light endgames, as a RELAXED problem solved in tempi (kings walk, pawns push, nothing interferes):
  1. name the GOALS of each side — promotion of a passed pawn (rule of the square) · a key square for the attacking king
     (Averbakh: pawn on rank ≤ 4 → the 3 squares two ranks ahead; 5-6 → the 6 squares on the next two ranks; 7 → the adjacent
     squares on ranks 7-8; rook pawn → the two squares on the adjacent file at ranks 7-8) · an undefended enemy pawn (target);
  2. margin(g) = tempi the defender needs to cover g − tempi the attacker needs to reach g (+1 if the attacker is to move);
  3. side to move / spare tempi break NEAR-TIES only (|margin| ≤ 1) — elsewhere they contribute exactly 0;
  4. OVERLOAD: the defender covers the best goal; the attacker's second goal (and the file gap between them) is the decoy.
Complexity = the number of CONTESTED goals (|margin| ≤ 1) for either side; no goals = no potential.

Test (pawn endings only, here): fit SF18's win% (best_cp, White POV) on train games with (a) our static eval alone and (b) our
static + the race features; report held-out win% MSE overall and by stratum. Every other row is colour-MIRRORED (EGX is ~96%
White to move), label negated. If (b) does not beat (a) on held-out games, the idea is dead before any C++.
"""
import os, sys, csv, math, random, zlib
for _a in sys.argv[1:]:
    if "=" in _a:
        _k, _v = _a.split("=", 1)
        os.environ[_k] = _v
THIS = os.path.dirname(os.path.abspath(__file__))
ENGINE = os.path.dirname(THIS)
sys.path.insert(0, ENGINE); sys.path.insert(0, THIS); sys.path.insert(0, os.path.join(ENGINE, "selfplay"))
os.environ.setdefault("PRESET", "LONG_FORMAT"); os.environ.setdefault("USE_OPENING_BOOK", "0")
import numpy as np
import chess
import ChessAI
from _build_trajectories import spare_tempi

IN = os.environ.get("IN", "/mnt/e/chess_data/retune/EGX_sf18.csv")
FENS = os.environ.get("FENS", "/mnt/e/chess_data/retune/EGX_fens.csv")
MAX = int(os.environ.get("MAX", "0"))
DUMP = os.environ.get("DUMP", "")
K = 0.00368208


def winpct(cp):
    cp = max(-1500.0, min(1500.0, cp))
    return 100.0 / (1.0 + math.exp(-K * cp))


def dist(a, b):
    return chess.square_distance(a, b)


def is_passed(b, s, c):
    f, r = chess.square_file(s), chess.square_rank(s)
    ahead = range(r + 1, 8) if c else range(0, r)
    return not any(b.piece_at(chess.square(ff, rr)) == chess.Piece(chess.PAWN, not c)
                   for ff in (f - 1, f, f + 1) if 0 <= ff < 8 for rr in ahead)


def rel_rank(s, c):
    return chess.square_rank(s) if c else 7 - chess.square_rank(s)


def key_squares(s, c):
    f, rr = chess.square_file(s), rel_rank(s, c)
    up = (lambda k: rr + k)
    sq = lambda ff, r: chess.square(ff, r if c else 7 - r)
    out = []
    if f in (0, 7):                                     # rook pawn: the two squares on the adjacent file, ranks 7-8
        ff = 1 if f == 0 else 6
        return [sq(ff, 6), sq(ff, 7)]
    if rr <= 3:
        rows = [up(2)]
    elif rr <= 5:
        rows = [up(1), up(2)]
    else:
        rows = [6, 7]
    for r in rows:
        if 0 <= r <= 7:
            for ff in (f - 1, f, f + 1):
                if 0 <= ff < 8 and not (r == rr and ff == f):
                    out.append(sq(ff, r))
    return out


def goals(b, c):
    """[(margin, file, kind)] for side c (the attacker). margin > 0 ⇒ c reaches the goal first in the relaxed race."""
    ak, dk = b.king(c), b.king(not c)
    stm = 1 if b.turn == c else 0
    out = []
    for s in b.pieces(chess.PAWN, c):
        f = chess.square_file(s)
        if is_passed(b, s, c):
            steps = 7 - rel_rank(s, c)
            if rel_rank(s, c) == 1:
                steps -= 1                              # double step
            q = chess.square(f, 7 if c else 0)
            out.append((dist(dk, q) - steps + stm, f, "promo"))
        ks = [k for k in key_squares(s, c) if not b.piece_at(k)]
        if ks:
            a = min(dist(ak, k) for k in ks)
            d = min(dist(dk, k) for k in ks)
            out.append((d - a + stm, f, "key"))
    for s in b.pieces(chess.PAWN, not c):               # enemy pawns not defended by a pawn
        if b.attackers_mask(not c, s) & b.pieces_mask(chess.PAWN, not c):
            continue
        a = max(0, dist(ak, s) - 1)
        d = max(0, dist(dk, s) - 1)
        out.append((d - a + stm, chess.square_file(s), "target"))
    return out


def features(b):
    """Race features, White-minus-Black where signed. Names in FEAT."""
    gw, gb = goals(b, True), goals(b, False)
    f = []
    for g in (gw, gb):
        ms = sorted((m for m, _, _ in g), reverse=True)
        best = max(-3, min(3, ms[0])) if ms else -3
        second = max(-3, min(3, ms[1])) if len(ms) > 1 else -3
        if len(g) > 1:
            top = sorted(g, reverse=True)[:2]
            gap = abs(top[0][1] - top[1][1])
        else:
            gap = 0
        promo = max([m for m, _, k in g if k == "promo"], default=-3)
        f.append((best, second, gap, max(-3, min(3, promo)), sum(1 for m, _, _ in g if abs(m) <= 1)))
    (bw, sw, gpw, pw, cw), (bb, sb, gpb, pb, cb) = f
    contested = cw + cb
    tie = 1 if contested else 0
    tempi = (spare_tempi(b, True) - spare_tempi(b, False)) * tie         # tempi only where a race is contested
    stm = (1 if b.turn else -1) * tie                                     # side to move only where contested
    win_w = 1.0 if bw >= 2 else 0.0
    win_b = 1.0 if bb >= 2 else 0.0
    over_w = 1.0 if (sw >= 1 and gpw >= 3) else 0.0                      # decoy: a second live goal far from the first
    over_b = 1.0 if (sb >= 1 and gpb >= 3) else 0.0
    return [bw - bb, sw - sb, pw - pb, win_w - win_b, over_w - over_b, tempi, stm, contested]


FEAT = ["best_diff", "second_diff", "promo_diff", "decisive_diff", "overload_diff", "tempi_tie", "stm_tie", "contested"]


def pawn_ending(b):
    return not (b.occupied & ~(b.pawns | b.kings)) and bool(b.pawns)


def fit(X, y, iters=3000, lr=0.05):
    """Least squares on win% through a sigmoid (the house target), plain gradient descent on standardised columns."""
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = (X - mu) / sd
    w = np.zeros(Z.shape[1]); b0 = 0.0
    for _ in range(iters):
        p = 100.0 / (1.0 + np.exp(-(Z @ w + b0)))
        g = (p - y) * p * (100.0 - p) / 100.0 / 100.0
        w -= lr * (Z.T @ g) / len(y); b0 -= lr * g.mean()
    return lambda Xn: 100.0 / (1.0 + np.exp(-(((Xn - mu) / sd) @ w + b0))), w


def main():
    games = {r["fen"]: r["game"] for r in csv.DictReader(open(FENS, newline=""))}
    ai = ChessAI.ChessAI(None, None, chess.Board(), True)
    rows, n = [], 0
    for r in csv.DictReader(open(IN, newline="")):
        if not r.get("best_cp"):
            continue
        cp = float(r["best_cp"])
        if abs(cp) >= 20000:
            continue
        b = chess.Board(r["fen"])
        if not pawn_ending(b) or b.is_game_over():
            continue
        g = games.get(r["fen"], r["fen"])
        if n % 2 == 1:
            b = b.mirror(); cp = -cp
        n += 1
        ours = -int(ai.ev_breakdown(b)["total"]) / 10.0
        rows.append((g, winpct(cp), ours, features(b), b.fen()))
        if MAX and n >= MAX:
            break
    # 5-fold cross-validation BY GAME: every game is held out exactly once (positions within a game are correlated, so a
    # single 20% split leaves only ~240 independent games — its CI crossed zero on 10-15).
    fold = np.array([zlib.crc32(r[0].encode()) % 5 for r in rows])
    print("POT race prototype — pawn endings: %d rows, 5-fold CV split by game" % len(rows))

    def mats(rs):
        y = np.array([r[1] for r in rs])
        s = np.array([[max(-1500, min(1500, r[2]))] for r in rs])
        F = np.array([r[3] for r in rs], dtype=float)
        return y, s, np.hstack([s, F]), F
    Y, S, X, F = mats(rows)
    ea, eb, ec = np.zeros(len(rows)), np.zeros(len(rows)), np.zeros(len(rows))
    rng = np.random.default_rng(1015)
    en_all = [np.zeros(len(rows)) for _ in range(5)]
    for k in range(5):
        tr, te = fold != k, fold == k
        pa, _ = fit(S[tr], Y[tr])
        pb, wb = fit(X[tr], Y[tr])
        pc, _ = fit(F[tr], Y[tr])
        ea[te] = (pa(S[te]) - Y[te]) ** 2
        eb[te] = (pb(X[te]) - Y[te]) ** 2
        ec[te] = (pc(F[te]) - Y[te]) ** 2
        for j in range(5):                              # NULL: race features shuffled across rows (train and test)
            Ftr, Fte = F[tr][rng.permutation(tr.sum())], F[te][rng.permutation(te.sum())]
            pn, _ = fit(np.hstack([S[tr], Ftr]), Y[tr])
            en_all[j][te] = (pn(np.hstack([S[te], Fte])) - Y[te]) ** 2
    test, fte, yte = rows, F, Y
    print("held-out win%% MSE   static alone %.1f   race alone %.1f   static + race %.1f   (%.1f%%)" %
          (ea.mean(), ec.mean(), eb.mean(), 100 * (eb.mean() / ea.mean() - 1)))
    contested = fte[:, FEAT.index("contested")] > 0
    decisive = fte[:, FEAT.index("decisive_diff")] != 0
    for name, m in (("contested race", contested), ("decisive race (one side ≥ 2 tempi)", decisive),
                    ("no contested / decisive", ~contested & ~decisive)):
        if m.sum():
            print("  %-36s n=%5d   static %.1f → static+race %.1f   (%.1f%%)" %
                  (name, m.sum(), ea[m].mean(), eb[m].mean(), 100 * (eb[m].mean() / ea[m].mean() - 1)))
    print("standardised weights, last fold (static first):", " ".join("%s %+.2f" % (n, w) for n, w in zip(["static"] + FEAT, wb)))
    # Resolution: game-clustered bootstrap of the held-out gain, and a NULL (race features shuffled across rows, so they keep
    # their distribution but lose their position) — the gain must sit outside the null's.
    gid = np.array([r[0] for r in test])
    ug = np.unique(gid)
    idx = {g: np.where(gid == g)[0] for g in ug}
    boots = []
    for _ in range(1000):
        pick = np.concatenate([idx[g] for g in rng.choice(ug, len(ug))])
        boots.append(100 * (eb[pick].mean() / ea[pick].mean() - 1))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    print("gain 95%% CI (game-clustered bootstrap, %d games): [%.1f%%, %.1f%%]" % (len(ug), lo, hi))
    nulls = [100 * (en.mean() / ea.mean() - 1) for en in en_all]
    print("NULL (shuffled race features, 5 draws): " + " ".join("%.1f%%" % v for v in nulls))
    if DUMP:
        with open(DUMP, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["fen", "game", "sf_winpct", "ours_cp", *FEAT])
            for g, y, o, f, fen in rows:
                w.writerow([fen, g, "%.2f" % y, "%.1f" % o, *f])


if __name__ == "__main__":
    main()
