#!/usr/bin/env python3
"""Per-game-length / per-opening profile of the 2800-game LMR_SHAPE pool.

Tests the blunder-avoidance story from SESSION-HANDOFF-2026-08-27 §1: if the ship wins by
making fewer catastrophic mis-reductions near the horizon, the advantage should CONCENTRATE
IN LONGER GAMES (more moves = more chances to blunder). If it is flat in length, that story
is wrong and the +20.7 needs another explanation.

Pure CSV. No engine, no bench.
"""
import csv, glob, math, os, collections

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "selfplay", "games")
SEGS = sorted(glob.glob(os.path.join(BASE, "lmrshape_s*", "summary.csv")))

def elo(score, n):
    """Elo diff + 95% CI from a mean score in [0,1]. Returns (elo, halfwidth)."""
    if n == 0 or score <= 0.0 or score >= 1.0:
        return float('nan'), float('nan')
    e = -400.0 * math.log10(1.0 / score - 1.0)
    # binomial-ish se on the score, propagated through the logistic
    se = math.sqrt(max(score * (1 - score), 1e-9) / n)
    d = 400.0 / (math.log(10) * max(score * (1 - score), 1e-9)) * se
    return e, 1.96 * d

rows = []
for f in SEGS:
    seg = os.path.basename(os.path.dirname(f))
    with open(f, newline='') as fh:
        for r in csv.DictReader(fh):
            if not r.get("result"):
                continue
            try:
                r["_score"] = float(r["p1_score"])
                r["_plies"] = int(r["plies"])
            except (ValueError, TypeError, KeyError):
                continue
            r["_seg"] = seg
            rows.append(r)

n = len(rows)
overall = sum(r["_score"] for r in rows) / n
e, ci = elo(overall, n)
print(f"POOL: {n} games   score {overall:.4f}   Elo {e:+.1f} +/-{ci:.1f}")
print()

# ---- the actual test: Elo vs game length -------------------------------------------
plies = sorted(r["_plies"] for r in rows)
qs = [plies[int(len(plies) * q)] for q in (0.2, 0.4, 0.6, 0.8)]
print(f"length quintile cuts (plies): {qs}")
print()
print(f"{'bucket':>14} {'n':>5} {'score':>7} {'elo':>8} {'+/-':>6}  {'W':>4} {'L':>4} {'D':>4}")

def bucket(p):
    for i, c in enumerate(qs):
        if p <= c:
            return i
    return 4

names = [f"<={qs[0]}", f"{qs[0]+1}-{qs[1]}", f"{qs[1]+1}-{qs[2]}",
         f"{qs[2]+1}-{qs[3]}", f">{qs[3]}"]
by = collections.defaultdict(list)
for r in rows:
    by[bucket(r["_plies"])].append(r)

trend = []
for i in range(5):
    g = by[i]
    if not g:
        continue
    s = sum(x["_score"] for x in g) / len(g)
    ee, cc = elo(s, len(g))
    w = sum(1 for x in g if x["_score"] == 1.0)
    l = sum(1 for x in g if x["_score"] == 0.0)
    d = len(g) - w - l
    trend.append(ee)
    print(f"{names[i]:>14} {len(g):>5} {s:>7.4f} {ee:>+8.1f} {cc:>6.1f}  {w:>4} {l:>4} {d:>4}")

print()
# correlation between length and outcome, decisive games only (draws carry no length signal)
dec = [r for r in rows if r["_score"] != 0.5]
if dec:
    mp = sum(r["_plies"] for r in dec) / len(dec)
    ms = sum(r["_score"] for r in dec) / len(dec)
    cov = sum((r["_plies"] - mp) * (r["_score"] - ms) for r in dec)
    vp = sum((r["_plies"] - mp) ** 2 for r in dec)
    vs = sum((r["_score"] - ms) ** 2 for r in dec)
    rho = cov / math.sqrt(vp * vs) if vp and vs else float('nan')
    print(f"decisive games: n={len(dec)}  corr(plies, p1_score) = {rho:+.4f}")
    print(f"  mean plies when candidate WINS: "
          f"{sum(r['_plies'] for r in dec if r['_score']==1.0)/max(1,sum(1 for r in dec if r['_score']==1.0)):.1f}")
    print(f"  mean plies when candidate LOSES: "
          f"{sum(r['_plies'] for r in dec if r['_score']==0.0)/max(1,sum(1 for r in dec if r['_score']==0.0)):.1f}")

# ---- termination reasons -------------------------------------------------------------
print()
print("termination reasons (candidate score by reason):")
byr = collections.defaultdict(list)
for r in rows:
    byr[r.get("reason", "?")].append(r["_score"])
for k, v in sorted(byr.items(), key=lambda kv: -len(kv[1]))[:12]:
    print(f"  {k:<34} n={len(v):>5}  score {sum(v)/len(v):.4f}")

# ---- per-opening recurrence ----------------------------------------------------------
print()
byo = collections.defaultdict(list)
for r in rows:
    byo[r.get("opening_idx", "?")].append(r["_score"])
multi = {k: v for k, v in byo.items() if len(v) >= 4}
print(f"openings seen >=4 times: {len(multi)} of {len(byo)} distinct")
if multi:
    worst = sorted(multi.items(), key=lambda kv: sum(kv[1]) / len(kv[1]))[:10]
    print("  worst openings for the candidate (persistent-weakness check):")
    for k, v in worst:
        print(f"    opening {k:>6}  n={len(v):>3}  score {sum(v)/len(v):.3f}")
    # is the spread wider than chance? compare observed variance of per-opening scores
    # against the binomial expectation if every opening were equally winnable.
    ms = sum(sum(v) for v in multi.values()) / sum(len(v) for v in multi.values())
    obs = sum((sum(v)/len(v) - ms) ** 2 * len(v) for v in multi.values()) / sum(len(v) for v in multi.values())
    expv = ms * (1 - ms) / (sum(len(v) for v in multi.values()) / len(multi))
    print(f"  per-opening score variance: observed {obs:.4f} vs chance {expv:.4f} "
          f"(ratio {obs/expv:.2f}x)  <- >1.5x suggests real opening-specific weakness")

# ---- colour split (p1 was on both sides) ---------------------------------------------
print()
for c in ("white", "black"):
    g = [r for r in rows if r.get("p1_color") == c]
    if g:
        s = sum(x["_score"] for x in g) / len(g)
        ee, cc = elo(s, len(g))
        print(f"candidate as {c:>5}: n={len(g):>5} score {s:.4f}  Elo {ee:+.1f} +/-{cc:.1f}")
