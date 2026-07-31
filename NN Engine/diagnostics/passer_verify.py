# -*- coding: utf-8 -*-
"""Deterministic passer-corpus verifier — the measuring instrument for every round of the passer redesign.
Evaluates OUR engine (honoring env knobs, so you can A/B a config like ENABLE_PASSER_V3=1) on
ks_sets/passer_corpus.csv and reports, per TIER x PHASE (oriented to the passer owner): mean over_read
(our_total - SF18, +=we over-credit the passer owner), direction-correct %, and count. Goal signals:
  under_fire  mean over_read  -> should rise toward 0 (we currently UNDER-credit; negative)
  blowup_guard mean over_read -> must NOT rise (we must not over-credit further)
  control     mean over_read  -> stay ~0
Also prints the per-row dossier/guard lines so regressions are visible by position, and a priced-once
breakdown split (passed_pawn_support vs pt_pawns) for a few rows.

  [ENABLE_PASSER_V3=1 ...] pyrun diagnostics/passer_verify.py
"""
import os, sys, csv
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'; os.environ.setdefault('KS_SAFE_CHECK_DEF', '5')
for _a in sys.argv[1:]:                 # KEY=VAL argv -> env (before ChessAI import), for A/B configs
    if '=' in _a and '/' not in _a:
        _k, _v = _a.split('=', 1); os.environ[_k] = _v
THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(THIS)); sys.path.insert(0, THIS)
import chess
from collections import defaultdict
from ChessAI import ChessAI
ai = ChessAI(None, None, chess.Board(), True)
CORPUS = os.path.join(THIS, "ks_sets", "passer_corpus.csv")
rows = list(csv.DictReader(open(CORPUS)))

agg = defaultdict(lambda: [0.0, 0, 0])   # (tier,phase) -> [sum_over, dir_ok, n]
tier_agg = defaultdict(lambda: [0.0, 0, 0])
for r in rows:
    b = chess.Board(r["fen"])
    bd = ai.ev_breakdown(b)
    our = -bd.get("total", 0.0) / 1000.0
    sf = float(r["sf18"])
    owner = 1 if int(r["wp"]) >= int(r["bp"]) else -1
    over = owner * (our - sf)                      # + = we over-credit the passer owner
    dir_ok = 1 if (abs(sf) < 0.5 or (our > 0) == (sf > 0)) else 0
    key = (r["tier"], r["phase"])
    agg[key][0] += over; agg[key][1] += dir_ok; agg[key][2] += 1
    tier_agg[r["tier"]][0] += over; tier_agg[r["tier"]][1] += dir_ok; tier_agg[r["tier"]][2] += 1

print("=== passer_verify (config via env) ===")
print("\nBy TIER (mean over_read = our-SF18 oriented to passer owner; want under_fire ->0, guard NOT up):")
for t in ("under_fire", "blowup_guard", "control"):
    s, ok, n = tier_agg[t]
    if n:
        print("  %-13s n=%-4d mean_over_read=%+6.2f  dir_ok=%3.0f%%" % (t, n, s / n, 100 * ok / n))
print("\nBy TIER x PHASE:")
for (t, ph), (s, ok, n) in sorted(agg.items()):
    print("  %-13s %-8s n=%-4d mean_over_read=%+6.2f  dir_ok=%3.0f%%" % (t, ph, n, s / n, 100 * ok / n))

print("\nPer-row dossier + guard positions (our vs SF18, + breakdown split):")
for r in rows:
    if r["src"] in ("dossier", "guard"):
        bd = ai.ev_breakdown(chess.Board(r["fen"]))
        our = -bd.get("total", 0.0) / 1000.0
        pp = -bd.get("passed_pawn_support", 0.0) / 1000.0
        ptp = -bd.get("pt_pawns", 0.0) / 1000.0
        print("  [%s %s] our%+.2f SF18%+.2f  (passed_supp%+.2f pt_pawns%+.2f)  %s"
              % (r["src"], r["tier"], our, float(r["sf18"]), pp, ptp, r["fen"]))
