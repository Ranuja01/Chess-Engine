# -*- coding: utf-8 -*-
"""Report on fastchess PGN output (2026-10-19; UCI design M5). Zero engine CPU.
  python selfplay/fastchess_report.py <games.pgn> [more.pgn ...] [--p1 NAME] [--summary-out summary.csv]

Per match: W/L/D, score, Elo ± 95% from the PAIR (pentanomial) variance — fastchess `-repeat` plays each opening twice with
colours swapped on consecutive games — time forfeits, terminations. Per engine: ACTUAL time per move (mean / median / p95 /
max, book moves excluded), depth, nodes and NPS when the PGN carries them (`-pgnout nodes=true`). Owner 10-19: a timed
comparison is only fair if both sides really spend the same time — this is that check.
--summary-out writes our tournament.py `summary.csv` layout (game, opening_idx, p1_color, white, black, result, p1_score,
plies, reason) so diagnostics/_pair_discrimination.py and the venue tools read fastchess games too.
--blocks N checks whether the ± is honest: per file, Elo ±; between files, chi2 of the run means against their own errors (several
runs of one configuration should agree within them); within files, consecutive N-pair blocks against the pair variance. A ratio
near 1 means the pentanomial ± describes the real spread; well above 1 means runs or blocks vary more than it allows.
Pairs are keyed per file: two runs with different seeds share openings from the same book.
"""
import argparse, collections, csv, math, re, statistics as st
import chess.pgn

COMMENT = re.compile(r"^\s*(?P<score>[+-]?(?:M?\d+(?:\.\d+)?))?/(?P<depth>\d+)\s+(?P<time>[\d.]+)s(?:.*?n=(?P<nodes>\d+))?")


def elo(p):
    p = min(max(p, 1e-6), 1 - 1e-6)
    return -400.0 * math.log10(1.0 / p - 1.0)


def chi2_sf(x, df):
    # upper tail of chi-squared, Wilson-Hilferty approximation (no scipy here)
    if df <= 0:
        return float("nan")
    z = ((x / df) ** (1.0 / 3) - (1 - 2.0 / (9 * df))) / math.sqrt(2.0 / (9 * df))
    return 0.5 * math.erfc(z / math.sqrt(2))


def pair_stats(xs):
    # xs = pair scores (0 … 2); returns per-game mean score, the variance of one pair's per-game score, and the se of the mean
    k = len(xs)
    mu = sum(xs) / (2 * k)
    var = sum((x / 2 - mu) ** 2 for x in xs) / (k - 1)
    return mu, var, math.sqrt(var / k)


def dispersion(by_file, names, block):
    print("\ndispersion (%d-pair blocks):" % block)
    runs = []
    for fi, xs in enumerate(by_file):
        if len(xs) < 2:
            continue
        mu, var, se = pair_stats(xs)
        runs.append((mu, se))
        print("  %-28s %4d pairs  Elo %+6.1f ± %4.1f" % (names[fi], len(xs), elo(mu), elo(mu + 1.96 * se) - elo(mu)))
    if len(runs) >= 2:
        w = [1 / se ** 2 for _, se in runs]
        m = sum(wi * mu for wi, (mu, _) in zip(w, runs)) / sum(w)
        x = sum(((mu - m) / se) ** 2 for mu, se in runs)
        df = len(runs) - 1
        print("  between runs: chi2 %.2f on %d df (ratio %.2f, p %.3f) · pooled Elo %+.1f ± %.1f" % (
            x, df, x / df, chi2_sf(x, df), elo(m), elo(m + 1.96 / math.sqrt(sum(w))) - elo(m)))
    x = 0.0; df = 0
    for xs in by_file:
        nb = len(xs) // block
        if nb < 2:
            continue
        mu, var, _ = pair_stats(xs)
        for b in range(nb):
            mb = sum(xs[b * block:(b + 1) * block]) / (2 * block)
            x += (mb - mu) ** 2 / (var / block)
        df += nb - 1
    if df:
        print("  within runs:  chi2 %.1f on %d df (ratio %.2f, p %.3f)" % (x, df, x / df, chi2_sf(x, df)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pgn", nargs="+")
    ap.add_argument("--p1", help="engine whose score is reported (default: White of the first game)")
    ap.add_argument("--summary-out")
    ap.add_argument("--blocks", type=int, help="pairs per block for the dispersion check")
    a = ap.parse_args()

    games = []
    for fi, path in enumerate(a.pgn):
        with open(path, encoding="utf-8", errors="ignore") as f:
            while True:
                g = chess.pgn.read_game(f)
                if g is None:
                    break
                games.append((fi, g))
    if not games:
        print("no games"); return
    p1 = a.p1 or games[0][1].headers["White"]
    tstats = collections.defaultdict(lambda: {"t": [], "d": [], "n": [], "ns": 0.0, "tn": 0.0})
    pairs = collections.OrderedDict()
    term = collections.Counter()
    rows = []
    W = L = D = 0
    for gi, (fi, g) in enumerate(games):
        h = g.headers
        res = h.get("Result", "*")
        if res == "*":
            continue
        white, black = h["White"], h["Black"]
        p1_white = white == p1
        s = {"1-0": 1.0, "0-1": 0.0, "1/2-1/2": 0.5}[res]
        ps = s if p1_white else 1.0 - s
        W += ps == 1.0; L += ps == 0.0; D += ps == 0.5
        term[h.get("Termination", "?")] += 1
        key = (fi, h.get("FEN", "startpos"))
        pairs.setdefault(key, []).append(ps)
        node = g
        plies = 0
        while node.variations:
            node = node.variation(0)
            mover = white if node.parent.board().turn else black
            plies += 1
            m = COMMENT.match(node.comment or "")
            if not m or int(m.group("depth")) == 0:      # book / no search info
                continue
            ts = tstats[mover]
            t = float(m.group("time"))
            ts["t"].append(t); ts["d"].append(int(m.group("depth")))
            if m.group("nodes"):
                n = int(m.group("nodes")); ts["n"].append(n); ts["ns"] += n; ts["tn"] += t
        rows.append([gi, list(pairs).index(key), "white" if p1_white else "black", white, black, res, ps, plies,
                     h.get("Termination", "")])

    n = W + L + D
    p = (W + 0.5 * D) / n
    pair_scores = [sum(v) for v in pairs.values() if len(v) == 2]
    print("%s vs the field: %d games  +%d -%d =%d  score %.1f%%" % (p1, n, W, L, D, 100 * p))
    if len(pair_scores) >= 2:
        k = len(pair_scores)
        mu = sum(pair_scores) / (2 * k)
        var = sum((x / 2 - mu) ** 2 for x in pair_scores) / (k - 1)
        se = math.sqrt(var / k)
        pent = collections.Counter(pair_scores)
        print("  Elo %+.1f  (95%% %+.1f … %+.1f) from %d pairs · pentanomial [0, ½, 1, 1½, 2] = [%s]" % (
            elo(mu), elo(mu - 1.96 * se), elo(mu + 1.96 * se), k,
            ", ".join(str(pent.get(x, 0)) for x in (0.0, 0.5, 1.0, 1.5, 2.0))))
    print("  terminations: " + ", ".join("%s %d" % kv for kv in term.most_common()))
    if a.blocks:
        by_file = [[sum(v) for (fi, _), v in pairs.items() if fi == f and len(v) == 2] for f in range(len(a.pgn))]
        dispersion(by_file, [p.replace("\\", "/").rsplit("/", 1)[-1] for p in a.pgn], a.blocks)
    print("\nper engine (searched moves only):")
    print("  %-14s %7s %8s %8s %8s %8s %7s %10s %9s" % ("engine", "moves", "mean s", "median", "p95", "max", "depth", "nodes", "nps"))
    for name, ts in sorted(tstats.items()):
        t = sorted(ts["t"])
        if not t:
            continue
        p95 = t[min(len(t) - 1, int(0.95 * len(t)))]
        nps = ts["ns"] / ts["tn"] if ts["tn"] > 0 else float("nan")
        print("  %-14s %7d %8.3f %8.3f %8.3f %8.3f %7.1f %10s %9s" % (
            name, len(t), st.mean(t), st.median(t), p95, t[-1], st.mean(ts["d"]),
            "%d" % st.mean(ts["n"]) if ts["n"] else "-", "%d" % nps if ts["n"] else "-"))
    if a.summary_out:
        with open(a.summary_out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["game", "opening_idx", "p1_color", "white", "black", "result", "p1_score", "plies", "reason"])
            w.writerows(rows)
        print("\nwrote %s (%d games)" % (a.summary_out, len(rows)))


if __name__ == "__main__":
    main()
