# -*- coding: utf-8 -*-
"""Central-score CLAMP-REGIME probe -- decides whether central is worth re-shaping.

The bounded-re-shape only buys IDENTIFIABILITY where a term is still LINEAR in the base signal.
`central_score` is consumed through a phase-stepped clamp (cpp_bitboard.cpp:7389-7400):

    phase < 20 :  central_add = clamp(central*3/2, +-400)
    phase < 31 :  central_add = clamp(central     , +-350)
    phase < 45 :  central_add = clamp(central/2   , +-300)
    else       :  central_add = clamp(central/4   , +-300)

Where |pre-clamp| >= bound the term is SATURATED = flat = already identifiable (re-shaping buys nothing).
Where |pre-clamp| <  bound it is LINEAR in central_score = collinear with the base cells = the regime a
bounded re-shape would de-correlate. The tuning-relevant question is not "how often linear" in the
abstract but "how often linear WHERE central actually contributes something" -- a near-zero central_add is
in the linear regime but carries no signal to separate. So we bucket by contribution magnitude.

Verdict rule:
  - meaningful central contributions (|central_add| > MAG) that are mostly SATURATED  -> already
    identifiable -> SKIP central, run the OvD-only retune (cleanest one-variable test).
  - meaningful central contributions mostly LINEAR -> central is a real second collinear channel AND big
    enough to matter -> worth the build-both-and-test re-shape, batch with OvD.

ZERO C++ change, no rebuild: reads det_central / phase_score / central straight off ev_breakdown().
`central` (br_central) is the actual clamped+scaled contribution and is used to CROSS-CHECK the Python
recomputation at SCALE_CENTRAL=100 (they must match) -- a self-test that the clamp model is faithful.

  pyrun diagnostics/_central_regime.py [CORPUS=ks_sets/diverse_corpus.csv] [MAG=50] [N=100000]
"""
import os, sys, csv

for _a in sys.argv[1:]:
    if '=' in _a:
        _k, _v = _a.split('=', 1)
        os.environ[_k] = _v

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
THIS = os.path.dirname(os.path.abspath(__file__))
ENGINE = os.path.dirname(THIS)
sys.path.insert(0, ENGINE); sys.path.insert(0, THIS)
sys.path.insert(0, os.path.join(ENGINE, "selfplay"))

import chess
from ChessAI import ChessAI

CORPUS = os.environ.get("CORPUS", "ks_sets/diverse_corpus.csv")
MAG    = int(os.environ.get("MAG", "50"))      # |central_add| above which a contribution is "meaningful"
NMAX   = int(os.environ.get("N", "100000"))

CSVP = os.path.join(THIS, CORPUS)


def trunc_div(n, d):
    """C++ integer division truncates toward zero (Python // floors)."""
    return int(n / d)


def clamp_model(raw, phase):
    """Replicate cpp_bitboard.cpp:7389-7400. Returns (pre_clamp, bound, central_add, saturated)."""
    if phase < 20:
        pre, bound = trunc_div(raw * 3, 2), 400
    elif phase < 31:
        pre, bound = raw, 350
    elif phase < 45:
        pre, bound = trunc_div(raw, 2), 300
    else:
        pre, bound = trunc_div(raw, 4), 300
    add = max(min(pre, bound), -bound)
    return pre, bound, add, abs(pre) >= bound


# Load FENs (column 'fen' or 'decision_fen' or first field).
fens = []
with open(CSVP, newline='') as fh:
    rdr = csv.DictReader(fh)
    fcol = 'fen' if 'fen' in rdr.fieldnames else ('decision_fen' if 'decision_fen' in rdr.fieldnames else rdr.fieldnames[0])
    for r in rdr:
        f = (r.get(fcol) or '').strip()
        if f:
            fens.append(f)
fens = fens[:NMAX]

ai = ChessAI(None, None, chess.Board(), True)

PHASE_BUCKETS = [("open   (<20)", lambda p: p < 20),
                 ("early  (20-30)", lambda p: 20 <= p < 31),
                 ("mid    (31-44)", lambda p: 31 <= p < 45),
                 ("late   (>=45)", lambda p: p >= 45)]

n = 0
mismatch = 0            # clamp-model vs actual `central` disagreements (should be 0 at SCALE_CENTRAL=100)
sat_all = lin_all = 0
sat_mean = 0           # meaningful (|add|>MAG): saturated
sat_mean_lin = 0       # meaningful: linear
clamped_away = []      # fraction of raw signal discarded by the clamp, meaningful positions only
per_phase = {name: [0, 0, 0, 0] for name, _ in PHASE_BUCKETS}  # [sat, lin, sat_mean, lin_mean]

print("  %d FENs from %s  (MAG=%d mp)\n" % (len(fens), CORPUS, MAG), flush=True)
for fen in fens:
    try:
        bd = ai.ev_breakdown(chess.Board(fen))
    except Exception as e:
        print("  SKIP (%s) %s" % (type(e).__name__, fen), flush=True)
        continue
    raw   = bd.get("det_central", 0)
    phase = bd.get("phase_score", 0)
    contrib = bd.get("central", 0)     # br_central = actual post-clamp, post-SCALE contribution
    pre, bound, add, sat = clamp_model(raw, phase)
    if add != contrib:                 # faithfulness self-test (holds only at SCALE_CENTRAL=100)
        mismatch += 1
    n += 1
    meaningful = abs(add) > MAG
    if sat:
        sat_all += 1
        if meaningful: sat_mean += 1
    else:
        lin_all += 1
        if meaningful: sat_mean_lin += 1
    if meaningful and abs(pre) > 0:
        clamped_away.append(max(0.0, (abs(pre) - abs(add)) / abs(pre)))
    for name, pred in PHASE_BUCKETS:
        if pred(phase):
            per_phase[name][0 if sat else 1] += 1
            if meaningful:
                per_phase[name][2 if sat else 3] += 1
            break

if n == 0:
    print("  no positions scored"); sys.exit(1)

mean_all = sat_all + lin_all
mean_cnt = sat_mean + sat_mean_lin
print("  clamp-model self-test: %d / %d mismatch vs actual `central` (expect 0 at SCALE_CENTRAL=100)\n" % (mismatch, n))
print("  ALL positions           : %5d  saturated %5d (%4.1f%%)  linear %5d (%4.1f%%)"
      % (n, sat_all, 100.0*sat_all/n, lin_all, 100.0*lin_all/n))
if mean_cnt:
    print("  MEANINGFUL (|add|>%d mp): %5d  saturated %5d (%4.1f%%)  linear %5d (%4.1f%%)"
          % (MAG, mean_cnt, sat_mean, 100.0*sat_mean/mean_cnt, sat_mean_lin, 100.0*sat_mean_lin/mean_cnt))
    print("     ^ %d of %d positions (%.1f%%) carry a meaningful central contribution" % (mean_cnt, n, 100.0*mean_cnt/n))
else:
    print("  MEANINGFUL (|add|>%d mp): NONE -- central is near-inert on this corpus" % MAG)
if clamped_away:
    ca = sorted(clamped_away)
    print("  clamp bite (meaningful): median %.0f%% of raw signal discarded, max %.0f%%"
          % (100.0*ca[len(ca)//2], 100.0*ca[-1]))
print()
print("  by phase        %8s %8s   %10s %10s" % ("sat", "lin", "sat(mean)", "lin(mean)"))
for name, _ in PHASE_BUCKETS:
    s, l, sm, lm = per_phase[name]
    tot = s + l
    if tot == 0:
        continue
    print("  %-14s  %4d %4.0f%% %4d %4.0f%%   %6d %4d" % (name, s, 100.0*s/tot, l, 100.0*l/tot, sm, lm))
print()
if mean_cnt:
    lin_share = 100.0*sat_mean_lin/mean_cnt
    if lin_share >= 50:
        print("  VERDICT: %.0f%% of meaningful central contributions are LINEAR (collinear) -> central is a real"
              "\n           second channel where it matters -> WORTH the build-both-and-test re-shape." % lin_share)
    else:
        print("  VERDICT: only %.0f%% of meaningful central contributions are linear -> mostly already SATURATED /"
              "\n           identifiable -> SKIP central, run the OvD-only retune (cleanest one-variable test)." % lin_share)
print()
