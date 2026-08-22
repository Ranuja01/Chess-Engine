# -*- coding: utf-8 -*-
"""Is capgains's ks_attack over-read a REAL realizability signal, or a White-POV colour artifact?

The attribution tool reported capgains mean_excess +0.36 in ks_attack collapses (White-POV). A GENUINE
realizability over-read is colour-SYMMETRIC: it should cancel to ~0 in White-POV and only show up when you
orient by WHICH SIDE COLLAPSED. A nonzero White-POV mean is a red flag (colour imbalance / colour bug), per
this repo's standing "every term error is bidirectional, signed mean ~0" principle.

This orients by our_color: oriented = capgains_white_pov * (our_color=='white' ? +1 : -1). Positive oriented
= capgains over-credited the side that PEAKED WINNING THEN COLLAPSED (over-optimism, the realizability
failure). Compares ks_attack vs positional (an internal control: same 'we collapsed' selection, different
mechanism) and reports the our_color split so a colour imbalance is visible.

Fast: OUR eval only (no SF). The collapse label already encodes "this was wrong" (peaked winning, didn't win),
so oriented capgains large+positive in ks_attack = capgains fed the over-optimism.

Run: bash <runner> pyrun diagnostics/_capg_orient_check.py CLS=<classified.csv> [KEY=VAL engine knobs]
"""
import os, sys, csv, statistics as st
for a in sys.argv[1:]:
    if '=' in a:
        k, v = a.split('=', 1)
        if k != 'CLS':
            os.environ[k] = v
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
import chess
THIS = os.path.dirname(os.path.abspath(__file__)); ENGINE = os.path.dirname(THIS)
sys.path.insert(0, ENGINE); sys.path.insert(0, THIS)
from ChessAI import ChessAI
seed = chess.Board(); ai = ChessAI(None, None, seed, seed.turn)

CLS = None
for a in sys.argv[1:]:
    if a.startswith('CLS='):
        CLS = a.split('=', 1)[1]
CLS = CLS or os.path.join(THIS, 'ks_sets', 'collapse_dataset_classified.csv')

rows = list(csv.DictReader(open(CLS)))
by_class = {}
for r in rows:
    fen = r.get('decision_fen'); col = r.get('our_color')
    if not fen or col not in ('white', 'black'):
        continue
    try:
        ev = ai.ev_breakdown(chess.Board(fen))
    except Exception:
        continue
    # ev_breakdown total is Black-positive millipawns; king_safety/capture_gains same convention.
    # White-POV pawns = -field/1000. Oriented-to-collapser = White-POV * (+1 white, -1 black).
    cg_wpov = -ev.get('capture_gains', 0) / 1000.0
    ks_wpov = -ev.get('king_safety', 0) / 1000.0
    sign = 1.0 if col == 'white' else -1.0
    by_class.setdefault(r.get('ks_class', '?'), []).append((cg_wpov, cg_wpov * sign, ks_wpov * sign, col))

print("CAPGAINS orientation check  (CLS=%s)" % os.path.basename(CLS))
print("%-16s %5s %10s %11s %11s %10s" % ("class", "n", "cg_Wpov", "cg_orient", "ks_orient", "cg>0.5 orient"))
for cls in sorted(by_class):
    v = by_class[cls]
    wpov = st.mean(x[0] for x in v)
    orient = st.mean(x[1] for x in v)
    ks_or = st.mean(x[2] for x in v)
    big = 100.0 * sum(1 for x in v if x[1] > 0.5) / len(v)
    nw = sum(1 for x in v if x[3] == 'white'); nb = len(v) - nw
    print("%-16s %5d %+10.3f %+11.3f %+11.3f %9.0f%%   (W%d/B%d)" % (cls, len(v), wpov, orient, ks_or, big, nw, nb))
print()
print("read: cg_Wpov ~0 but cg_orient >> 0  => REAL symmetric over-credit of the collapsing side (damp is valid).")
print("      cg_Wpov == cg_orient (and W/B skewed) => it was a White-POV/colour artifact, NOT realizability.")
print("      cg_orient large in ks_attack but ~0 in positional => the effect is king-attack-specific.")
