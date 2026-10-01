# Paired gauntlet split by GAME TYPE, classes taken from the BASELINE game (same opening as the candidate's game, so the
# class does not depend on how the candidate played). Classes: castling geometry at ply ~30 (both kings' files),
# sharp vs quiet (a >=150 cp swing from a still-balanced |eval|<=300 position), game length, and (2026-10-01) whether
# the game ever reached a QUEEN IMBALANCE (one side has a queen, the other none) — the Kaufman collateral check.
import sys, csv, json, math
G = "/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine/selfplay/games/"


def king_files(fen):
    board = fen.split()[0]
    r, f, wk, bk = 7, 0, None, None
    for ch in board:
        if ch == "/":
            r -= 1; f = 0
        elif ch.isdigit():
            f += int(ch)
        else:
            if ch == "K": wk = f
            if ch == "k": bk = f
            f += 1
    return wk, bk


def side(f):
    return "K" if f is not None and f >= 5 else ("Q" if f is not None and f <= 2 else "C")


def classify(path):
    recs = []
    for line in open(path):
        try:
            recs.append(json.loads(line))
        except ValueError:
            pass
    fen30 = None
    for r in recs:
        if r.get("ply", 0) >= 30 and "fen" in r:
            fen30 = r["fen"]; break
    if fen30 is None and recs:
        fen30 = recs[-1].get("fen")
    ws, bs = (side(x) for x in king_files(fen30)) if fen30 else ("C", "C")
    if "C" in (ws, bs):
        castle = "uncastled king"
    else:
        castle = "opposite-side" if ws != bs else "same-side"
    ev = [max(-3000, min(3000, r["our_pov_eval"])) for r in recs if "our_pov_eval" in r]
    sharp = max((abs(b - a) for a, b in zip(ev, ev[1:]) if abs(a) <= 300), default=0) >= 150
    # PERSISTENT queen imbalance: ≥ 10 consecutive recorded plies (a mid-trade position is not an imbalance)
    run = best = 0
    for r in recs:
        if "fen" not in r:
            continue
        b0 = r["fen"].split()[0]
        run = run + 1 if (b0.count("Q") > 0) != (b0.count("q") > 0) else 0
        best = max(best, run)
    return castle, ("sharp" if sharp else "quiet"), ("queen imbalance ≥10 plies" if best >= 10 else "none / transient")


rows = []
for pair in sys.argv[1:]:
    base, cand = pair.split(":")
    rb = {int(r["game"]): r for r in csv.DictReader(open(G + base + "/results.csv"))}
    rc = {int(r["game"]): r for r in csv.DictReader(open(G + cand + "/results.csv"))}
    for g in sorted(set(rb) & set(rc)):
        castle, sharp, qimb = classify(G + base + "/game_%03d.jsonl" % g)
        plies = int(rb[g]["plies"])
        length = "short (<80 plies)" if plies < 80 else ("medium (80-140)" if plies <= 140 else "long (>140)")
        rows.append({"castle": castle, "sharp": sharp, "length": length, "material": qimb,
                     "d": float(rc[g]["our_score"]) - float(rb[g]["our_score"]),
                     "b": float(rb[g]["our_score"])})


def report(key):
    print("\n== by %s ==" % key)
    for cls in sorted({r[key] for r in rows}):
        d = [r["d"] for r in rows if r[key] == cls]
        b = [r["b"] for r in rows if r[key] == cls]
        n = len(d)
        m = sum(d) / n
        sd = math.sqrt(sum((x - m) ** 2 for x in d) / (n - 1)) if n > 1 else 0.0
        print("  %-18s n %4d  base %5.1f%%  diff %+6.2fpp ± %5.2f  (better %d / worse %d)" % (
            cls, n, 100 * sum(b) / n, 100 * m, 196 * sd / math.sqrt(n), sum(x > 0 for x in d), sum(x < 0 for x in d)))


print("paired games: %d" % len(rows))
for k in ("castle", "sharp", "length", "material"):
    report(k)
