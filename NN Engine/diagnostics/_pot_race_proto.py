# -*- coding: utf-8 -*-
"""POT RACE-MARGIN PROTOTYPE (2026-10-15/16, owner design discussion; POT brief §5) — Python only, no engine change.

  V2_PRESET=shipped pyrun diagnostics/_pot_race_proto.py [MODE=fit|moves|cases] [SET=pawn|light] [MAX=0] [DUMP=<csv>]

The human procedure in light endgames, as a RELAXED problem solved in tempi (kings walk, pawns push, pieces hop):
  1. name the GOALS of each side —
       promotion of a PASSED pawn (rule of the square) ·
       BREAKTHROUGH: a CANDIDATE passer (no enemy pawn ahead on its file, own helpers ≥ enemy sentries — Kmoch's classic
         candidate) costs one extra tempo per sentry to dissolve (owner games 1-2, 10-15: the passer that does not exist yet) ·
       a KEY SQUARE for the attacking king (Averbakh) · an undefended enemy pawn (target);
  2. margin(g) = tempi the defender needs to cover g − tempi the attacker needs to reach g (+1 if the attacker is to move).
     The defender covers a pawn's run with whatever arrives first: the king (rule of the square), a KNIGHT (exact knight
     distance to any square of the path), a BISHOP (0 if it already eyes the path, 1 if the path has squares of its colour,
     else never), a rook / queen (0 if it eyes the path, else 1);
  3. side to move / spare tempi break NEAR-TIES only (|margin| ≤ 1) — elsewhere they contribute exactly 0;
  4. OVERLOAD: the defender covers the best goal; the attacker's second goal (and the file gap between them) is the decoy.
Complexity = the number of CONTESTED goals (|margin| ≤ 1) for either side; no goals = no potential.

MODE=fit (default): fit SF18's win% (EGX best_cp, White POV) with (a) our static alone and (b) static + race features, 5-fold
  CV BY GAME, game-clustered bootstrap CI, shuffled-feature null. SET=pawn (pawn endings) or SET=light (pawns + minors, no
  rooks / queens). Every other row is colour-MIRRORED (EGX is ~96% White to move), label negated.
MODE=moves: the MOVE test (stage 2). 5 folds by GAME: each fold of multi-PV positions is scored by (a)/(b) fitted on the EGX
  rows of the OTHER games; also the SIBLING SPREAD of the race term vs the static term (does the race MOVE between moves?)
  and the race term's own ordering on critical positions. Then on each
  multi-PV position (pe/light_mpv_s*.csv, SF18 d14 K=8) score every labelled move's child by our static and by static + race;
  report top-1 agreement with SF's best and pairwise ordering vs moves that lose ≥ 10 win% points, overall and on CRITICAL
  positions (SF best − second ≥ 15 win% points: one move keeps the result).
MODE=cases: the owner's game positions (ks_sets/pot_owner_cases.txt) with static, static + race, and the race goals.
"""
import os, sys, csv, math, random, zlib, glob
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
MODE = os.environ.get("MODE", "fit")
SET = os.environ.get("SET", "pawn")
MAX = int(os.environ.get("MAX", "0"))
DUMP = os.environ.get("DUMP", "")
MPV = os.environ.get("MPV", "/mnt/e/chess_data/pe/light_mpv_s*.csv")
K = 0.00368208
NEVER = 9


def winpct(cp):
    cp = max(-1500.0, min(1500.0, cp))
    return 100.0 / (1.0 + math.exp(-K * cp))


def dist(a, b):
    return chess.square_distance(a, b)


def _knight_table():
    t = [[NEVER] * 64 for _ in range(64)]
    for s in range(64):
        t[s][s] = 0
        frontier, d = [s], 0
        while frontier:
            d += 1
            nxt = []
            for f in frontier:
                for to in chess.SquareSet(chess.BB_KNIGHT_ATTACKS[f]):
                    if t[s][to] == NEVER and to != s:
                        t[s][to] = d
                        nxt.append(to)
            frontier = nxt
    return t


KNIGHT = _knight_table()


def piece_cover(b, c, path):
    """Tempi side c's PIECES need to control / occupy a square of `path` (list of squares). NEVER if none can."""
    best = NEVER
    pm = chess.SquareSet(path)
    for s in b.pieces(chess.KNIGHT, c):
        best = min(best, min(max(0, KNIGHT[s][p] - 1) for p in path))      # one hop short of a path square attacks it
    for s in b.pieces(chess.BISHOP, c):
        if b.attacks_mask(s) & int(pm):
            best = min(best, 0)
        elif any(((chess.square_file(p) + chess.square_rank(p)) & 1) == ((chess.square_file(s) + chess.square_rank(s)) & 1)
                 for p in path):
            best = min(best, 1)
    for t in (chess.ROOK, chess.QUEEN):
        for s in b.pieces(t, c):
            best = min(best, 0 if b.attacks_mask(s) & int(pm) else 1)
    return best


def is_passed(b, s, c):
    f, r = chess.square_file(s), chess.square_rank(s)
    ahead = range(r + 1, 8) if c else range(0, r)
    return not any(b.piece_at(chess.square(ff, rr)) == chess.Piece(chess.PAWN, not c)
                   for ff in (f - 1, f, f + 1) if 0 <= ff < 8 for rr in ahead)


def candidate_sentries(b, s, c):
    """Kmoch classic candidate: no pawn of either side ahead on its own file and own helpers ≥ enemy sentries. Returns the
    sentry count (tempi to dissolve them) or None when the pawn is not a candidate."""
    f, r = chess.square_file(s), chess.square_rank(s)
    ahead = range(r + 1, 8) if c else range(0, r)
    if any(b.piece_type_at(chess.square(f, rr)) == chess.PAWN for rr in ahead):
        return None
    sentries = sum(1 for ff in (f - 1, f + 1) if 0 <= ff < 8 for rr in ahead
                   if b.piece_at(chess.square(ff, rr)) == chess.Piece(chess.PAWN, not c))
    behind = range(0, r + 1) if c else range(r, 8)
    helpers = sum(1 for ff in (f - 1, f + 1) if 0 <= ff < 8 for rr in behind
                  if b.piece_at(chess.square(ff, rr)) == chess.Piece(chess.PAWN, c))
    return sentries if sentries and helpers >= sentries else None


def rel_rank(s, c):
    return chess.square_rank(s) if c else 7 - chess.square_rank(s)


def key_squares(s, c):
    f, rr = chess.square_file(s), rel_rank(s, c)
    sq = lambda ff, r: chess.square(ff, r if c else 7 - r)
    out = []
    if f in (0, 7):                                     # rook pawn: the two squares on the adjacent file, ranks 7-8
        ff = 1 if f == 0 else 6
        return [sq(ff, 6), sq(ff, 7)]
    rows = [rr + 2] if rr <= 3 else ([rr + 1, rr + 2] if rr <= 5 else [6, 7])
    for r in rows:
        if 0 <= r <= 7:
            for ff in (f - 1, f, f + 1):
                if 0 <= ff < 8 and not (r == rr and ff == f):
                    out.append(sq(ff, r))
    return out


def run_margin(b, s, c, extra, stm):
    """Race of side c's pawn on s to promotion (+ `extra` tempi) against the defender's king and pieces."""
    f = chess.square_file(s)
    steps = 7 - rel_rank(s, c) - (1 if rel_rank(s, c) == 1 else 0) + extra
    q = chess.square(f, 7 if c else 0)
    path = [chess.square(f, r) for r in (range(chess.square_rank(s) + 1, 8) if c else range(0, chess.square_rank(s)))]
    d = min(dist(b.king(not c), q), piece_cover(b, not c, path) if path else NEVER)
    return d - steps + stm


def goals(b, c):
    """[(margin, file, kind)] for side c (the attacker). margin > 0 ⇒ c reaches the goal first in the relaxed race."""
    ak, dk = b.king(c), b.king(not c)
    stm = 1 if b.turn == c else 0
    out = []
    for s in b.pieces(chess.PAWN, c):
        f = chess.square_file(s)
        if is_passed(b, s, c):
            out.append((run_margin(b, s, c, 0, stm), f, "promo"))
        else:
            sen = candidate_sentries(b, s, c)
            if sen is not None:
                out.append((run_margin(b, s, c, sen, stm), f, "break"))
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
        clip = lambda v: max(-3, min(3, v))
        best = clip(ms[0]) if ms else -3
        second = clip(ms[1]) if len(ms) > 1 else -3
        # overload (10-16, tie-free ⇒ file-mirror symmetric; matches eval_v2 race_side): two goals with margin ≥ 1, ≥ 3 files apart
        live = [fl for m, fl, _ in g if m >= 1]
        over = 1 if live and max(live) - min(live) >= 3 else 0
        promo = clip(max([m for m, _, k in g if k == "promo"], default=-3))
        brk = clip(max([m for m, _, k in g if k == "break"], default=-3))
        f.append((best, second, over, promo, brk, sum(1 for m, _, _ in g if abs(m) <= 1)))
    (bw, sw, over_w, pw, kw, cw), (bb, sb, over_b, pb, kb, cb) = f
    contested = cw + cb
    tie = 1 if contested else 0
    tempi = (spare_tempi(b, True) - spare_tempi(b, False)) * tie         # tempi only where a race is contested
    stm = (1 if b.turn else -1) * tie                                     # side to move only where contested
    win_w = 1.0 if bw >= 2 else 0.0
    win_b = 1.0 if bb >= 2 else 0.0
    return [bw - bb, sw - sb, pw - pb, kw - kb, win_w - win_b, over_w - over_b, tempi, stm, contested]


FEAT = ["best_diff", "second_diff", "promo_diff", "break_diff", "decisive_diff", "overload_diff", "tempi_tie", "stm_tie",
        "contested"]


def in_set(b):
    if not b.pawns or b.is_game_over():
        return False
    if SET == "light":
        return not (b.rooks | b.queens)
    return not (b.occupied & ~(b.pawns | b.kings))


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


AI = None


def ours(b):
    global AI
    if AI is None:
        AI = ChessAI.ChessAI(None, None, chess.Board(), True)
    return -int(AI.ev_breakdown(b)["total"]) / 10.0


def load_rows(exclude_games=frozenset()):
    games = {r["fen"]: r["game"] for r in csv.DictReader(open(FENS, newline=""))}
    rows, n = [], 0
    for r in csv.DictReader(open(IN, newline="")):
        if not r.get("best_cp"):
            continue
        cp = float(r["best_cp"])
        if abs(cp) >= 20000:
            continue
        b = chess.Board(r["fen"])
        if not in_set(b):
            continue
        g = games.get(r["fen"], r["fen"])
        if g in exclude_games:
            continue
        if n % 2 == 1:
            b = b.mirror(); cp = -cp
        n += 1
        rows.append((g, winpct(cp), ours(b), features(b), b.fen()))
        if MAX and n >= MAX:
            break
    return rows


def mats(rs):
    y = np.array([r[1] for r in rs])
    s = np.array([[max(-1500, min(1500, r[2]))] for r in rs])
    F = np.array([r[3] for r in rs], dtype=float)
    return y, s, np.hstack([s, F]), F


def main_fit():
    rows = load_rows()
    # 5-fold cross-validation BY GAME: every game is held out exactly once (positions within a game are correlated, so a
    # single 20% split leaves only ~240 independent games — its CI crossed zero on 10-15).
    fold = np.array([zlib.crc32(r[0].encode()) % 5 for r in rows])
    print("POT race prototype — SET=%s: %d rows, 5-fold CV split by game" % (SET, len(rows)))
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
    print("held-out win%% MSE   static alone %.1f   race alone %.1f   static + race %.1f   (%.1f%%)" %
          (ea.mean(), ec.mean(), eb.mean(), 100 * (eb.mean() / ea.mean() - 1)))
    contested = F[:, FEAT.index("contested")] > 0
    decisive = F[:, FEAT.index("decisive_diff")] != 0
    brk = F[:, FEAT.index("break_diff")] != 0
    for name, m in (("contested race", contested), ("decisive race (one side ≥ 2 tempi)", decisive),
                    ("a breakthrough goal differs", brk), ("no contested / decisive", ~contested & ~decisive)):
        if m.sum():
            print("  %-36s n=%5d   static %.1f → static+race %.1f   (%.1f%%)" %
                  (name, m.sum(), ea[m].mean(), eb[m].mean(), 100 * (eb[m].mean() / ea[m].mean() - 1)))
    print("standardised weights, last fold (static first):", " ".join("%s %+.2f" % (n, w) for n, w in zip(["static"] + FEAT, wb)))
    gid = np.array([r[0] for r in rows])
    ug = np.unique(gid)
    idx = {g: np.where(gid == g)[0] for g in ug}
    boots = []
    for _ in range(1000):
        pick = np.concatenate([idx[g] for g in rng.choice(ug, len(ug))])
        boots.append(100 * (eb[pick].mean() / ea[pick].mean() - 1))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    print("gain 95%% CI (game-clustered bootstrap, %d games): [%.1f%%, %.1f%%]" % (len(ug), lo, hi))
    print("NULL (shuffled race features, 5 draws): " + " ".join("%.1f%%" % (100 * (en.mean() / ea.mean() - 1)) for en in en_all))
    if DUMP:
        with open(DUMP, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["fen", "game", "sf_winpct", "ours_cp", *FEAT])
            for g, y, o, f, fen in rows:
                w.writerow([fen, g, "%.2f" % y, "%.1f" % o, *f])


def main_moves():
    mpv = []
    for p in sorted(glob.glob(MPV)):
        for r in csv.DictReader(open(p, newline="")):
            mv = [(u, float(c)) for u, c in (x.split(":") for x in r["moves"].split(";") if ":" in x)]
            if len(mv) >= 2:
                mpv.append((r["fen"], mv))
    games = {r["fen"]: r["game"] for r in csv.DictReader(open(FENS, newline=""))}
    # 5 folds BY GAME over the multi-PV positions; fold k is scored by models trained on every EGX row whose game is NOT
    # in fold k (10-16: a single exclusion of all multi-PV games starved training to 2,952 of 29,291 rows).
    gfold = lambda g: zlib.crc32(g.encode()) % 5
    rows = load_rows()
    Y, S, X, F = mats(rows)
    rfold = np.array([gfold(r[0]) for r in rows])
    models = []
    for k in range(5):
        tr = rfold != k
        models.append((fit(S[tr], Y[tr])[0],) + fit(X[tr], Y[tr]))
    print("MOVE TEST — %d EGX %s rows, 5 game folds (~%d training rows each), %d multi-PV positions"
          % (len(rows), SET, int(len(rows) * 0.8), len(mpv)))
    print("weights, fold 0 (static first):", " ".join("%s %+.2f" % (n, w) for n, w in zip(["static"] + FEAT, models[0][2])))
    stats = {k: {"n": 0, "top_a": 0, "top_b": 0, "pairs": 0, "pa": 0, "pb": 0} for k in ("all", "critical", "break")}
    spread_static, spread_race, race_right, race_pairs = [], [], 0, 0
    for fen, mv in mpv:
        pa, pb, _ = models[gfold(games.get(fen, fen))]
        b = chess.Board(fen)
        white = b.turn
        sfw = [(u, winpct(c) if white else 100 - winpct(c)) for u, c in mv]       # mover's win% per move
        best_u = max(sfw, key=lambda x: x[1])[0]
        ordered = sorted((x[1] for x in sfw), reverse=True)
        critical = ordered[0] - ordered[1] >= 15
        sa, sb, has_break = {}, {}, False
        for u, _ in mv:
            m = chess.Move.from_uci(u)
            if m not in b.legal_moves:
                continue
            b.push(m)
            if b.is_game_over():
                v = 100.0 if b.is_checkmate() else 50.0
                sa[u] = sb[u] = v
            else:
                st = max(-1500, min(1500, ours(b)))
                f = features(b)
                has_break = has_break or f[FEAT.index("break_diff")] != 0
                xa = np.array([[st]]); xb = np.array([[st] + f], dtype=float)
                pwa, pwb = float(pa(xa)[0]), float(pb(xb)[0])
                sa[u] = pwa if white else 100 - pwa
                sb[u] = pwb if white else 100 - pwb
            b.pop()
        if best_u not in sa:
            continue
        sfd = dict(sfw)
        # (c) does the race term MOVE between siblings? race contribution = (static+race) − (static alone), mover's POV.
        race = {u: sb[u] - sa[u] for u in sa}
        if len(sa) >= 2:
            spread_static.append(float(np.std(list(sa.values()))))
            spread_race.append(float(np.std(list(race.values()))))
        if critical:
            for u in sa:
                if u != best_u and sfd[best_u] - sfd[u] >= 10:
                    race_pairs += 1
                    race_right += race[best_u] > race[u]
        for key in ("all", "critical", "break"):
            if (key == "critical" and not critical) or (key == "break" and not has_break):
                continue
            st_ = stats[key]
            st_["n"] += 1
            st_["top_a"] += max(sa, key=sa.get) == best_u
            st_["top_b"] += max(sb, key=sb.get) == best_u
            for u in sa:
                if u != best_u and sfd[best_u] - sfd[u] >= 10:
                    st_["pairs"] += 1
                    st_["pa"] += sa[best_u] > sa[u]
                    st_["pb"] += sb[best_u] > sb[u]
    for key, s in stats.items():
        if s["n"]:
            print("  %-9s n=%5d   top-1 = SF best: static %.1f%% → static+race %.1f%%   pairs vs ≥10-pt losers (%d): static %.1f%% → %.1f%%"
                  % (key, s["n"], 100 * s["top_a"] / s["n"], 100 * s["top_b"] / s["n"], s["pairs"],
                     100 * s["pa"] / max(1, s["pairs"]), 100 * s["pb"] / max(1, s["pairs"])))
    if spread_static:
        ss, sr = np.array(spread_static), np.array(spread_race)
        print("  SIBLING SPREAD (std across a position's labelled moves, win%% pts): static median %.2f · race term median %.2f"
              " (ratio %.2f) · race term > 1 pt in %.1f%% of positions" % (np.median(ss), np.median(sr),
                                                                          np.median(sr) / max(1e-9, np.median(ss)), 100 * np.mean(sr > 1)))
        print("  race term ALONE orders SF's best above a ≥10-pt loser on CRITICAL positions: %.1f%% of %d pairs (50%% = blind)"
              % (100 * race_right / max(1, race_pairs), race_pairs))


def main_split():
    """MODE=split: which half carries the gain — race features pointing WITH the existing edge (sign agrees with our static;
    ⇒ a WINNABILITY input: the edge converts) or AGAINST it (⇒ POT's own directional term: an edge that does not exist yet).
    |static| < 25 cp rows have no edge: their features count as WITH (they cannot point against one)."""
    rows = load_rows()
    if DUMP:
        with open(DUMP, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["fen", "game", "sf_winpct", "ours_cp", *FEAT])
            for g, y, o, f, fen in rows:
                w.writerow([fen, g, "%.2f" % y, "%.1f" % o, *f])
    Y, S, X, F = mats(rows)
    signed = [i for i, n in enumerate(FEAT) if n != "contested"]
    edge = np.sign(np.where(np.abs(S[:, 0]) < 25, 0, S[:, 0]))[:, None]
    Fs = F[:, signed]
    W = np.where(Fs * edge >= 0, Fs, 0.0)
    A = np.where(Fs * edge < 0, Fs, 0.0)
    C = F[:, [FEAT.index("contested")]]
    arms = {"static alone": S, "static + race (all)": X, "static + WITH-edge": np.hstack([S, W, C]),
            "static + AGAINST-edge": np.hstack([S, A, C]), "static + WITH + AGAINST": np.hstack([S, W, A, C])}
    fold = np.array([zlib.crc32(r[0].encode()) % 5 for r in rows])
    err = {k: np.zeros(len(rows)) for k in arms}
    for k in range(5):
        tr, te = fold != k, fold == k
        for name, M in arms.items():
            p, _ = fit(M[tr], Y[tr])
            err[name][te] = (p(M[te]) - Y[te]) ** 2
    against_rows = (A != 0).any(1)
    print("SPLIT — SET=%s, %d rows (%d with any AGAINST-edge race feature), 5-fold CV by game" % (SET, len(rows), against_rows.sum()))
    base = err["static alone"]
    for name in arms:
        e = err[name]
        print("  %-26s MSE %.1f (%+.1f%%)   on AGAINST rows %.1f (%+.1f%%)" % (name, e.mean(), 100 * (e.mean() / base.mean() - 1),
              e[against_rows].mean(), 100 * (e[against_rows].mean() / base[against_rows].mean() - 1)))


def main_cases():
    rows = load_rows()
    Y, S, X, F = mats(rows)
    pa, _ = fit(S, Y)
    pb, _ = fit(X, Y)
    for line in open(os.path.join(THIS, "ks_sets/pot_owner_cases.txt")):
        if line.startswith("#") or not line.strip():
            continue
        fen, _, note = line.partition("#")
        b = chess.Board(fen.strip())
        st = max(-1500, min(1500, ours(b)))
        f = features(b)
        print("%-12s static %+7.1f (win%% %.0f)  static+race win%% %.0f   W goals %s   B goals %s"
              % (note.split()[0], st, pa(np.array([[st]]))[0], pb(np.array([[st] + f], dtype=float))[0],
                 sorted(goals(b, True), reverse=True)[:3], sorted(goals(b, False), reverse=True)[:3]))


if __name__ == "__main__":
    {"fit": main_fit, "moves": main_moves, "cases": main_cases, "split": main_split}[MODE]()
