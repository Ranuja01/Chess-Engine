# -*- coding: utf-8 -*-
"""Fit WORKER: evaluate ONE KS knob-config's loss against the target, over a prepared fit corpus. Env→Config is
parsed once per process, so the DRIVER (ks_fit.py) spawns one of these per candidate config. Reads KNOB=VAL from
argv (set into env BEFORE importing ChessAI) + CORPUS=<path>. Corpus rows: fen,target_ks,tier,phase_bucket,split.
Prints one parseable line: per-(tier,split) mean-squared-error of our_ks vs target_ks (WHITE-POV pawns), + the
overall train/val MSE. Used by coordinate descent; not run directly."""
import os, sys, csv
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ.setdefault('KS_SAFE_CHECK_DEF', '5')
CORPUS = None
for _a in sys.argv[1:]:
    if _a.startswith('CORPUS='): CORPUS = _a.split('=', 1)[1]
    elif '=' in _a: _k, _v = _a.split('=', 1); os.environ[_k] = _v      # KNOB=VAL -> env (before import)
THIS = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(THIS)); sys.path.insert(0, THIS)
import chess
from ChessAI import ChessAI
ai = ChessAI(None, None, chess.Board(), True)

from collections import defaultdict
import math
# LOSS = WIN%-space (per user): drive the win-probability gap -> 0, NOT the centipawn gap. Lichess sigmoid.
def winpct(cp): return 50.0 + 50.0 * (2.0 / (1.0 + math.exp(-0.00368208 * cp)) - 1.0)
# Fit metric (investigation, LOSS=mse|logloss|hybrid). mse = current (win%-space squared err, 0..~10000).
# logloss = cross-entropy of soft SF-target vs ours (weights confident-wrong = collapses harder), scaled ~x1000
# to the mse range so GUARD_TOL stays meaningful. hybrid = mean of the two. All comparable WITHIN one LOSS.
LOSS = os.environ.get('LOSS', 'mse').lower()
_EPS = 1e-6
def loss_of(our_total, tgt_total):
    e_mse = (winpct(our_total * 100.0) - winpct(tgt_total * 100.0)) ** 2
    if LOSS == 'mse':
        return e_mse
    po = min(1.0 - _EPS, max(_EPS, winpct(our_total * 100.0) / 100.0))
    pt = min(1.0 - _EPS, max(_EPS, winpct(tgt_total * 100.0) / 100.0))
    ll = -(pt * math.log(po) + (1.0 - pt) * math.log(1.0 - po)) * 1000.0
    if LOSS == 'logloss':
        return ll
    return 0.5 * (e_mse + ll)                                          # hybrid
sse = defaultdict(float); cnt = defaultdict(int)
# DIRECTION metrics (side-relative: which king is in danger): a sign flip or a MISS (silent on real danger)
# changes the chosen move; a same-sign magnitude shift on an already-decided position does not.
n = defaultdict(int); wrongsign = defaultdict(int); miss = defaultdict(int); ok = defaultdict(int)
sum_ours = defaultdict(float); sum_tgt = defaultdict(float)   # magnitude capture vs the SF11-static target
def sgn(x): return (x > 1e-6) - (x < -1e-6)
for r in csv.DictReader(open(CORPUS)):
    try:
        bd = ai.ev_breakdown(chess.Board(r["fen"]))             # FULL eval with all theta knobs (KS + OvD)
        our_ks = -bd.get("king_safety", 0.0) / 1000.0           # white-POV pawns
        our_total = -bd.get("total", 0.0) / 1000.0              # full total -> OvD/imbalance knobs affect this
        tgt_ks = float(r["target_ks"]); tgt_total = float(r["target_total"])
        split = r["split"]; tier = r["tier"]
    except Exception:
        continue
    e = loss_of(our_total, tgt_total)                                  # WIN%-space loss (LOSS=mse|logloss|hybrid)
    sse[(tier, split)] += e; cnt[(tier, split)] += 1
    sse[("ALL", split)] += e; cnt[("ALL", split)] += 1
    if abs(tgt_ks) >= 0.5:                     # judge direction on meaningfully-nonzero KS targets
        n[tier] += 1; n["ALL"] += 1
        sum_ours[tier] += abs(our_ks); sum_ours["ALL"] += abs(our_ks)
        sum_tgt[tier] += abs(tgt_ks); sum_tgt["ALL"] += abs(tgt_ks)
        if abs(our_ks) < 0.3 and abs(tgt_ks) >= 1.0:            miss[tier] += 1; miss["ALL"] += 1        # 0 -> big
        elif sgn(our_ks) and sgn(our_ks) != sgn(tgt_ks):       wrongsign[tier] += 1; wrongsign["ALL"] += 1  # flip
        else:                                                  ok[tier] += 1; ok["ALL"] += 1

def mse(key): return sse[key] / cnt[key] if cnt[key] else 0.0   # mean win%-squared-error
parts = ["%s.%s=%.4f" % (t, s, mse((t, s))) for (t, s) in sorted(cnt) if t != "ALL"]
print("FIT train_mse=%.4f val_mse=%.4f | %s" % (mse(("ALL", "train")), mse(("ALL", "val")), " ".join(parts)))
for t in sorted(n):
    mo = sum_ours[t] / n[t] if n[t] else 0.0; mt = sum_tgt[t] / n[t] if n[t] else 0.0
    print("DIR %-12s n=%-4d miss=%-4d wrongsign=%-4d correctdir=%.2f  |ourKS|=%.2f |SF11tgt|=%.2f capture=%.0f%%" %
          (t, n[t], miss[t], wrongsign[t], (ok[t] / n[t]) if n[t] else 0.0, mo, mt, 100 * mo / mt if mt else 0.0))
