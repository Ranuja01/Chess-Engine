# -*- coding: utf-8 -*-
"""Score a KS config against the archetype bench. For each position, our KS danger-to-the-subject-king vs SF's.
DANGER archetypes (A*) should read HIGH (detect the danger); COUNTER archetypes (B*) should read LOW (no
over-production). Reports per-archetype mean SF vs OUR danger, a DETECTION ratio (A*, want ~1) and an
OVER-PRODUCTION excess (B*, want ~0). Knobs latch at init, so pass config via argv (KEY=VAL) -> set before import.

Run: bash <runner> pyrun diagnostics/_ks_bench_score.py [KEY=VAL ...]
"""
import os, sys
for a in sys.argv[1:]:
    if '=' in a:
        k, v = a.split('=', 1); os.environ[k] = v
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
import csv, chess, statistics as st
THIS_DIR = os.path.dirname(os.path.abspath(__file__)); ENGINE_DIR = os.path.dirname(THIS_DIR)
sys.path.insert(0, ENGINE_DIR); sys.path.insert(0, THIS_DIR)
from ChessAI import ChessAI
seed = chess.Board(); ai = ChessAI(None, None, seed, seed.turn)

BENCH = os.path.join(THIS_DIR, 'ks_sets', 'ks_archetypes.csv')
rows = list(csv.DictReader(open(BENCH)))

def subj_of(r, our_white):
    s = r.get('subj', '?')
    if s in ('W', 'B'): return s
    sf = float(r['sf_ks'])
    if sf < -0.05: return 'W'
    if sf > 0.05: return 'B'
    return 'W' if our_white < 0 else 'B'

per = {}
for r in rows:
    b = chess.Board(r['fen'])
    our_white = -ai.ev_breakdown(b).get('king_safety', 0) / 1000.0   # White-POV pawns (<0 => White-king danger)
    subj = subj_of(r, our_white)
    sf = float(r['sf_ks'])
    sf_d = (-sf if subj == 'W' else sf)          # danger to subject king (positive)
    our_d = (-our_white if subj == 'W' else our_white)
    per.setdefault(r['archetype'], []).append((sf_d, our_d))

cfg = " ".join(a for a in sys.argv[1:] if '=' in a) or "(baseline defaults)"
print("KS bench: %s" % cfg)
print("%-20s %4s %9s %9s %9s" % ("archetype", "n", "SF_dang", "OUR_dang", "note"))
A_sf = A_our = B_our = B_sf = 0.0; nA = nB = 0
for arch in sorted(per):
    v = per[arch]; sfm = st.mean(x[0] for x in v); ourm = st.mean(x[1] for x in v)
    danger = arch[0] == 'A'
    if danger:
        note = "detect %.0f%%" % (100 * ourm / sfm) if sfm > 0.01 else ""
        A_sf += sfm * len(v); A_our += ourm * len(v); nA += len(v)
    else:
        note = "OVERPRODUCE +%.2f" % (ourm - sfm) if ourm - sfm > 0.15 else "ok"
        B_our += ourm * len(v); B_sf += sfm * len(v); nB += len(v)
    print("%-20s %4d %+9.2f %+9.2f  %s" % (arch, len(v), sfm, ourm, note))
print("-" * 56)
A_mean = A_our / nA
B_excess = (B_our - B_sf) / nB
print("DANGER(A)  SF=%.2f OUR=%.2f  detection=%.0f%%   (want ~100%%; higher OUR=better)"
      % (A_sf / nA, A_mean, 100 * A_our / A_sf if A_sf else 0))
print("COUNTER(B) SF=%.2f OUR=%.2f  over-production=+%.2f  (want ~0; OUR near SF)"
      % (B_sf / nB, B_our / nB, B_excess))
# DISCRIMINATION RATIO: danger produced per unit of over-production. Raw detection can always be bought
# by making everything fire more (a volume knob), which leaves this ratio flat or worse; only a genuine
# FEEDER correction separates the two populations and moves it up. Judge arms on this, not on detection.
print("RATIO      danger/over-production = %.1f   (VOLUME knobs leave this flat or lower it;"
      % (A_mean / B_excess if B_excess > 1e-9 else float('inf')))
print("                                            only better DETECTION raises it)")
