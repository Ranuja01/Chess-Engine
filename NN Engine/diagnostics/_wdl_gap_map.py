"""WDL GAP MAP: where our static win% excess over SF11 sits, on the SF18 WDL label (5,988 held-out quiet rows of
_v1_v2_accuracy.py). Reads the existing static_*.csv dumps only, no engine. Each arm at its own best global scale; classes
by stage, ending type, material imbalance and SF18 decisiveness; game-clustered s.e. on the gap; then each class refit at its
own scale (how much a per-class constant scale recovers: little = a shape gap, not a magnitude one). The decisiveness
refit selects on the label and is not interpretable.
  python diagnostics/_wdl_gap_map.py <dir with cal_in.csv, cal_wdl.csv, static_*.csv>   (E:\chess_data\wdlcal)
"""
import csv, sys, numpy as np
from collections import defaultdict

D = sys.argv[1]
K = 0.00368208
ARMS = {"ours": "static_v2.csv", "sf11": "static_sf11.csv", "thr_m50": "static_g62thr_m50e100.csv", "v1": "static_v1.csv"}

meta = {r["fen"]: r for r in csv.DictReader(open(D + "/cal_in.csv"))}
wdl = {r["fen"]: (int(r["w"]) + 0.5 * int(r["d"])) / 1000.0 for r in csv.DictReader(open(D + "/cal_wdl.csv"))}
pred = {a: {r["fen"]: float(r["cp_white"]) for r in csv.DictReader(open(D + "/" + f))} for a, f in ARMS.items()}
fens = [f for f in pred["ours"] if all(f in pred[a] for a in ARMS) and f in wdl and f in meta]
y = np.array([wdl[f] for f in fens])
grp = np.array([meta[f]["group"].split("#")[0] for f in fens])

def sq(a):
    cp = np.array([pred[a][f] for f in fens])
    best = None
    for s in np.arange(0.30, 6.005, 0.01):
        e = (100 * (1 / (1 + np.exp(-K * s * cp)) - y)) ** 2
        m = e.mean()
        if best is None or m < best[0]:
            best = (m, s, e)
    return best

E = {}
for a in ARMS:
    m, s, e = sq(a)
    E[a] = e
    print("%-8s calibrated WDL MSE %.1f (scale %.2f)" % (a, m, s))

VAL = {"p": 1, "n": 3, "b": 3, "r": 5, "q": 9}
def material(fen):
    board = fen.split()[0]
    c = defaultdict(int)
    for ch in board:
        if ch.lower() in VAL:
            c[ch] += 1
    return c

def etype(c):
    W = {k: c[k] for k in "NBRQ"}; B = {k: c[k.lower()] for k in "NBRQ"}
    tot = sum(W.values()) + sum(B.values())
    q = W["Q"] + B["Q"]; r = W["R"] + B["R"]; mi = W["N"] + W["B"] + B["N"] + B["B"]
    if tot == 0: return "pawn ending"
    if q and not r and not mi: return "queen only"
    if q: return "queen + other"
    if r and not mi: return "rook only"
    if mi and not r: return "minor only"
    if tot <= 6: return "rook + minor (light)"
    return "middlegame-heavy"

def imbalance(c):
    sig = lambda s: (c[s[0]], c[s[1]], c[s[2]], c[s[3]])
    wn, wb, wr, wq = sig("NBRQ"); bn, bb, br, bq = sig("nbrq")
    dm = (wn + wb) - (bn + bb); dr = wr - br; dq = wq - bq
    dp = c["P"] - c["p"]
    if dq and (dq * (dr * 5 + dm * 3)) < 0: return "queen vs pieces"
    if dr and dm and dr * dm < 0: return "exchange (R vs minor)"
    if dm and not dr and not dq: return "minor vs pawns"
    if dr and not dm and not dq: return "rook vs pawns"
    if dm == dr == dq == 0: return "equal pieces, pawns %s" % ("equal" if dp == 0 else "+-1" if abs(dp) == 1 else "+-2 or more")
    return "other imbalance"

def decis(v):
    d = abs(v - 0.5)
    return "level (<10 pts)" if d < 0.10 else "edge (10-30)" if d < 0.30 else "decisive (>30)"

cls = {
    "stage": [meta[f]["stage"] for f in fens],
    "ending type": [etype(material(f)) for f in fens],
    "imbalance": [imbalance(material(f)) for f in fens],
    "SF18 decisiveness": [decis(v) for v in y],
}
rng = np.random.default_rng(1)
ug, gi = np.unique(grp, return_inverse=True)
tot_ex = (E["ours"] - E["sf11"]).sum()
print("\ntotal excess ours - sf11 over %d rows: %.0f (%.1f per row)\n" % (len(fens), tot_ex, tot_ex / len(fens)))
for name, labels in cls.items():
    labels = np.array(labels)
    print("== by %s" % name)
    print("  %-30s %5s %7s %7s %7s %10s %8s %9s" % ("class", "n", "ours", "sf11", "v1", "gap", "share", "thr_m50"))
    for c in sorted(set(labels), key=lambda c: -(E["ours"] - E["sf11"])[labels == c].sum()):
        m = labels == c
        n = m.sum()
        d = E["ours"] - E["sf11"]
        g = d[m].mean()
        # game-clustered bootstrap of the per-row gap in this class
        sums = np.bincount(gi[m], weights=d[m], minlength=len(ug)); cnts = np.bincount(gi[m], minlength=len(ug))
        live = np.nonzero(cnts)[0]
        bs = []
        for _ in range(400):
            pick = rng.choice(live, len(live))
            bs.append(sums[pick].sum() / max(1, cnts[pick].sum()))
        se = np.std(bs)
        print("  %-30s %5d %7.0f %7.0f %7.0f %+6.0f±%-4.0f %7.0f%% %+8.0f" % (c, n, E["ours"][m].mean(), E["sf11"][m].mean(),
              E["v1"][m].mean(), g, se, 100 * d[m].sum() / tot_ex, (E["thr_m50"] - E["ours"])[m].mean()))
    print()
CP = {a: np.array([pred[a][f] for f in fens]) for a in ARMS}
def best_in(a, m):
    out = None
    for s in np.arange(0.30, 6.005, 0.01):
        v = ((100 * (1 / (1 + np.exp(-K * s * CP[a][m])) - y[m])) ** 2).mean()
        if out is None or v < out[0]:
            out = (v, s)
    return out
print("== magnitude vs shape: each class refit at its OWN best scale (what a per-class scale factor could recover)")
print("  %-30s %5s %9s %9s %11s %11s %10s" % ("class", "n", "ours@glob", "sf11@glob", "ours@own(s)", "sf11@own(s)", "gap left"))
for name in ("ending type", "imbalance", "SF18 decisiveness"):
    labels = np.array(cls[name])
    for c in sorted(set(labels), key=lambda c: -(E["ours"] - E["sf11"])[labels == c].sum()):
        m = labels == c
        if m.sum() < 100:
            continue
        o, so = best_in("ours", m); s1, ss = best_in("sf11", m)
        print("  %-30s %5d %9.0f %9.0f %7.0f(%.2f) %7.0f(%.2f) %+10.0f" % (c, m.sum(), E["ours"][m].mean(), E["sf11"][m].mean(),
              o, so, s1, ss, o - s1))
    print()
print("gap = ours - SF11 (positive = SF11 more accurate), ± 1 s.e. game-clustered; share = of the total excess;")
print("thr_m50 = threats mg 50 / eg 100 minus ours (negative = the term helps there)")
