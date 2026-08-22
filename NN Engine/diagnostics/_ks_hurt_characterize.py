# -*- coding: utf-8 -*-
"""Characterize the KS-HURT vs KS-HELPED positions dumped by _ks_footprint_regret (DUMP=).

Dump columns: fen, ks_on_move, ks_off_move, reg_ks_on, reg_ks_off, delta.
  delta < 0  => KS-off move had LESS regret => the KS-on move we actually play is WORSE => KS HURT here.
  delta > 0  => KS HELPED here.

If the KS-HURT cluster has a clean structural signature that the KS-HELPED cluster does not, that signature
is a damp target (subtractive: quiet KS where it misfires, tip the ~50/50 toward positive). If the two
clusters look statistically identical, KS is wrong for scattered/tactical reasons -> not structurally
reachable, and we close KS honestly for a measured reason. Pool both cross-sets via DUMP + DUMP2.

  pyrun diagnostics/_ks_hurt_characterize.py DUMP=<path> [DUMP2=<path>]
"""
import os, sys, csv, statistics

for _a in sys.argv[1:]:
    if '=' in _a:
        _k, _v = _a.split('=', 1)
        os.environ.setdefault(_k, _v)
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(THIS)); sys.path.insert(0, THIS)
import chess
from ChessAI import ChessAI

PATHS = [os.environ[k] for k in ('DUMP', 'DUMP2', 'DUMP3') if os.environ.get(k)]
rows = []
for p in PATHS:
    rows += list(csv.DictReader(open(p, newline='')))
print("loaded %d changed positions from %d dump(s)" % (len(rows), len(PATHS)))

ai = ChessAI(None, None, chess.Board(), True)
NPV = {chess.KNIGHT: 3.25, chess.BISHOP: 3.45, chess.ROOK: 5.0, chess.QUEEN: 10.0}


def feats(fen):
    b = chess.Board(fen)
    bd = ai.ev_breakdown(b)
    ks_w = -float(bd.get('king_safety', 0)) / 1000.0          # WHITE-POV pawns (Black-positive -> negate)
    npm = sum((len(b.pieces(pt, chess.WHITE)) + len(b.pieces(pt, chess.BLACK))) * v
              for pt, v in NPV.items())
    stm_white = (b.turn == chess.WHITE)
    # KS from the SIDE-TO-MOVE's perspective: >0 means KS favours the mover (enemy king in danger),
    # <0 means KS says the mover's OWN king is the endangered one.
    ks_stm = ks_w if stm_white else -ks_w
    return {'aks': abs(ks_w), 'ks_stm': ks_stm, 'npm': npm}


hurt, helped = [], []
for r in rows:
    try:
        d = float(r['delta']); f = feats(r['fen'])
    except Exception:
        continue
    (hurt if d < 0 else helped).append(f)


def col(g, key):
    return [x[key] for x in g]


def summ(name, g):
    if not g:
        print("  %-7s (none)" % name); return
    aks = col(g, 'aks'); npm = col(g, 'npm'); stm = col(g, 'ks_stm')
    own = sum(1 for v in stm if v < -0.10)   # KS says mover's OWN king endangered
    opp = sum(1 for v in stm if v > 0.10)    # KS says ENEMY king endangered
    print("  %-7s n=%-5d | |KS| med=%.2f mean=%.2f | npm med=%.1f | own-king %4.1f%%  enemy-king %4.1f%%  (KS-stm mean %+.2f)"
          % (name, len(g), statistics.median(aks), statistics.mean(aks), statistics.median(npm),
             100.0 * own / len(g), 100.0 * opp / len(g), statistics.mean(stm)))


print("\nSPLIT (delta<0 = KS HURT, the damp target):")
summ("HURT", hurt)
summ("HELPED", helped)
print("\n  Read: a clean gap in |KS| (HURT bigger) = OVER-FIRE (damp large KS). A gap in npm (HURT lower) =")
print("  PHASE LEAK (damp KS in low-material). A gap in own-king%% (HURT higher) = OWN-KING OVER-CAUTION")
print("  (KS makes us too defensive). No gap anywhere = not structurally reachable -> close KS honestly.")
