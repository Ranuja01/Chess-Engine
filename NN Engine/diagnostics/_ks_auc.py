# -*- coding: utf-8 -*-
"""DISCRIMINATION instrument for king safety: can our KS tell SF's danger positions from SF's quiet ones?

Why this exists. The 82-position archetype bench reports a ratio of two rounded means, whose denominator
(counter over-production) is ~0.09 pawns printed to 2dp -- so it cannot resolve the small feeder changes we
actually care about, and it is NETTED between kings (a floor artifact on one king moves the other's number).
AUC fixes both: it is SCALE-FREE by construction, so a volume knob that multiplies every reading leaves it
unchanged, while a genuine DETECTION improvement moves it. That is exactly the volume-vs-discrimination
distinction we need.

Data: ks_sets/diverse_corpus_wide.csv -- 23,113 positions carrying SF's per-term `target_ks`. We keep
phase_bucket=midgame because our KS is midgame-only (it is skipped under isEndGame).

  DANGER class = |target_ks| >= HI (SF sees real king danger)
  QUIET  class = |target_ks| <= LO (SF sees none)
  score      = |our king_safety| for that position

AUC = P(score(danger) > score(quiet)), ties counted as 0.5. 0.5 = no discrimination, 1.0 = perfect.
Also reports the class means (so a volume change is visible as both means moving with AUC flat) and, with
DUMP=<path>, per-position scores so two arms can be diffed PAIRED (variance cancels; a group mean hides it).

Run: bash <runner> pyrun diagnostics/_ks_auc.py [KEY=VAL ...] [HI=0.5] [LO=0.05] [MAXN=0] [DUMP=path]
"""
import os, sys
OPTS = {}
for a in sys.argv[1:]:
    if '=' in a:
        k, v = a.split('=', 1)
        if k in ('HI', 'LO', 'MAXN', 'DUMP', 'PHASE'):
            OPTS[k] = v
        else:
            os.environ[k] = v          # engine knob: must be set before ChessAI init (knobs latch once)
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
import csv, chess, statistics as st
THIS_DIR = os.path.dirname(os.path.abspath(__file__)); ENGINE_DIR = os.path.dirname(THIS_DIR)
sys.path.insert(0, ENGINE_DIR); sys.path.insert(0, THIS_DIR)
from ChessAI import ChessAI
seed = chess.Board(); ai = ChessAI(None, None, seed, seed.turn)

HI = float(OPTS.get('HI', 0.5))
LO = float(OPTS.get('LO', 0.05))
MAXN = int(OPTS.get('MAXN', 0))
PHASE = OPTS.get('PHASE', 'midgame')
DUMP = OPTS.get('DUMP')

src = os.path.join(THIS_DIR, 'ks_sets', 'diverse_corpus_wide.csv')
danger, quiet, dump_rows = [], [], []
n_seen = 0
for r in csv.DictReader(open(src)):
    if PHASE != 'all' and r.get('phase_bucket') != PHASE:
        continue
    try:
        tks = abs(float(r['target_ks']))
    except (TypeError, ValueError):
        continue
    if not (tks >= HI or tks <= LO):
        continue                        # ignore the ambiguous middle band
    n_seen += 1
    if MAXN and n_seen > MAXN:
        break
    try:
        ours = abs(ai.ev_breakdown(chess.Board(r['fen'])).get('king_safety', 0) / 1000.0)
    except Exception:
        continue
    (danger if tks >= HI else quiet).append(ours)
    if DUMP:
        dump_rows.append((r['fen'], 1 if tks >= HI else 0, ours))

def auc(pos, neg):
    """Rank-based AUC with tie handling; O(n log n)."""
    if not pos or not neg:
        return float('nan')
    allv = sorted(pos + neg)
    ranks, i = {}, 0
    while i < len(allv):
        j = i
        while j + 1 < len(allv) and allv[j + 1] == allv[i]:
            j += 1
        r = (i + j) / 2.0 + 1.0          # average rank for ties
        for k in range(i, j + 1):
            ranks[allv[k]] = r
        i = j + 1
    rp = sum(ranks[v] for v in pos)
    n1, n0 = len(pos), len(neg)
    return (rp - n1 * (n1 + 1) / 2.0) / (n1 * n0)

cfg = " ".join(a for a in sys.argv[1:] if '=' in a and a.split('=', 1)[0] not in ('HI', 'LO', 'MAXN', 'DUMP', 'PHASE')) or "(baseline defaults)"
a = auc(danger, quiet)
print("KS DISCRIMINATION (AUC) : %s" % cfg)
print("  corpus=diverse_corpus_wide phase=%s  DANGER |target_ks|>=%.2f  QUIET |target_ks|<=%.2f" % (PHASE, HI, LO))
print("  n_danger=%d  n_quiet=%d" % (len(danger), len(quiet)))
print("  mean|ours| danger=%.3f  quiet=%.3f   (BOTH moving with AUC flat => a VOLUME change, not detection)"
      % (st.mean(danger) if danger else 0.0, st.mean(quiet) if quiet else 0.0))
print("  zero-reads: danger=%.0f%%  quiet=%.0f%%   (our KS is exactly 0 -- floored or phase-skipped)"
      % (100.0 * sum(1 for x in danger if x < 1e-9) / max(1, len(danger)),
         100.0 * sum(1 for x in quiet if x < 1e-9) / max(1, len(quiet))))
print("  AUC = %.4f      (0.5 = cannot separate; higher = better DETECTION)" % a)

if DUMP:
    with open(DUMP, 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['fen', 'label', 'ours'])
        w.writerows(dump_rows)
    print("  per-position scores -> %s  (diff two arms PAIRED; variance cancels)" % DUMP)
