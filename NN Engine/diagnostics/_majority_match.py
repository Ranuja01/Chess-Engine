# -*- coding: utf-8 -*-
"""SF-free pawn-majority gate: run OUR engine over majorities.csv and compare to the RECORDED sf_best /
sf_cp columns (no live Stockfish -> no interop hang). Honors PRESET/MAX_DEPTH + any engine env knobs
(PAWN_MAJORITY_MAG_MG/_EG, ADV_K, OUTSIDE_K, BLOCKADE_K, ...), so run once knob-off (the current
under-read) and once with a candidate config and diff the move-match% + mean eval-error. Bucketed by
phase so midgame-structural vs late-endgame-conversion read separately. Untracked dev probe.

Usage (from NN Engine/):
    PRESET=LIGHTNING USE_OPENING_BOOK=0 python diagnostics/_majority_match.py diagnostics/suites/majorities.csv [N]
    PRESET=LIGHTNING USE_OPENING_BOOK=0 PAWN_MAJORITY_MAG_EG=1000 ... python diagnostics/_majority_match.py <csv> [N]
"""
import os, sys, csv
THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS)
from tactical_test import run_one

path = sys.argv[1]
rows = list(csv.DictReader(open(path)))
n = int(sys.argv[2]) if len(sys.argv) > 2 else len(rows)
if len(rows) > n:
    step = len(rows) / n
    rows = [rows[int(k * step)] for k in range(n)]

# Decision-relevant calibration window: ignore already-decided / mate-ish positions where the majority
# value is benign and a single mate score swamps the mean. The PRIMARY tuning signal is the MEDIAN
# |our_cp - sf_cp| over |sf_cp| < DREL (robust to the remaining outliers). Move-match is secondary
# (best move in a majority position is usually tactical, so it's a weak read for an eval-calibration term).
DREL = 500

# Per-phase + overall accumulators: bucket -> {match,total, errs:[...] (decision-window only)}.
acc = {}
def bump(bucket, hit, err, decision):
    a = acc.setdefault(bucket, {"m": 0, "t": 0, "errs": []})
    a["m"] += hit; a["t"] += 1
    if err is not None and decision:
        a["errs"].append(err)

def median(xs):
    if not xs:
        return 0.0
    xs = sorted(xs); k = len(xs)
    return xs[k // 2] if k % 2 else 0.5 * (xs[k // 2 - 1] + xs[k // 2])

for r in rows:
    fen = r["fen_start"]; sf_best = r.get("sf_best", ""); sf_cp = r.get("sf_cp", "")
    phase = r.get("phase", "all")
    try:
        res = run_one(fen, set())
    except Exception:
        continue
    hit = 1 if res["uci"] == sf_best else 0
    err = None; decision = False
    try:
        if res["eval"] is not None and sf_cp not in ("", None):
            err = abs(res["eval"] / 10.0 - float(sf_cp))   # our milli-pawns -> centipawns
            decision = abs(float(sf_cp)) < DREL
    except Exception:
        pass
    bump("ALL", hit, err, decision)
    bump(phase, hit, err, decision)

def line(name, a):
    m, t = a["m"], a["t"]
    pct = 100.0 * m / t if t else 0.0
    med = median(a["errs"]); ndec = len(a["errs"])
    return f"{name:>5}: match {m}/{t} ({pct:.1f}%)  median|err|[|sf|<{DREL}]={med:.0f} (n={ndec})"

print(line("ALL", acc.get("ALL", {"m": 0, "t": 0, "errs": []})))
for ph in ("mid", "eeg", "leg"):
    if ph in acc:
        print(line(ph, acc[ph]))
