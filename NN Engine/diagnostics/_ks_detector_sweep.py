# -*- coding: utf-8 -*-
"""Full KS-detector coverage+direction sweep over the collapse set. For EACH gated KS detector (existing knobs),
measure vs the shipped default: how many collapse positions' king_safety term moves, the mean |change|, and
whether it moves TOWARD or AWAY from SF11's King-safety term (the hand-fixable ceiling). Answers, for the whole
KS stack at once, which detectors actually bite on our real collapses and pull toward SF.

Knobs latch once per process, so each arm runs in its own subprocess (this script re-invokes itself as a worker).

Run (via the allowlisted runner, which sets STOCKFISH_PATH):
  bash <runner> pyrun diagnostics/_ks_detector_sweep.py            # driver
  (worker mode is internal: ... _ks_detector_sweep.py --worker <fenfile>, with KS env preset)
"""
import os, sys, subprocess, signal
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
import csv
import chess

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ENGINE_DIR = os.path.dirname(THIS_DIR)
sys.path.insert(0, ENGINE_DIR)
sys.path.insert(0, THIS_DIR)

# ---- worker mode: eval king_safety (White-POV) for every fen, print "idx\tks" ----
if len(sys.argv) > 1 and sys.argv[1] == '--worker':
    fenfile = sys.argv[2]
    from ChessAI import ChessAI
    seed = chess.Board()
    ai = ChessAI(None, None, seed, seed.turn)
    with open(fenfile) as f:
        fens = [ln.strip() for ln in f if ln.strip()]
    for i, fen in enumerate(fens):
        bd = ai.ev_breakdown(chess.Board(fen))
        print("%d\t%.4f" % (i, -bd.get('king_safety', 0) / 1000.0))
    sys.exit(0)

# ---- driver ----
from eval_vs_sf11 import SF11Eval, SF11

FENS = sys.argv[1] if len(sys.argv) > 1 else 'selfplay/games/vssf_2400/dp_fens.csv'

class _TO(Exception):
    pass
def _al(s, f):
    raise _TO()
signal.signal(signal.SIGALRM, _al)

# load + filter to non-check (SF11 hangs on in-check)
raw = []
with open(os.path.join(ENGINE_DIR, FENS)) as f:
    rd = csv.DictReader(f)
    col = 'fen_start' if 'fen_start' in (rd.fieldnames or []) else (rd.fieldnames or ['fen'])[0]
    for r in rd:
        fen = (r.get(col) or '').strip()
        if fen and not chess.Board(fen).is_check():
            raw.append(fen)

# SF11 King-safety target per fen
sf11 = SF11Eval(SF11)
sf_ks = []
keep = []
for fen in raw:
    signal.alarm(8)
    try:
        _, terms = sf11.eval(fen)
        signal.alarm(0)
    except _TO:
        continue
    finally:
        signal.alarm(0)
    if terms is None:
        continue
    sf_ks.append(terms.get('King safety', 0.0))
    keep.append(fen)
sf11.close()

fenfile = os.path.join(ENGINE_DIR, 'diagnostics/_ksweep_fens.txt')
with open(fenfile, 'w') as f:
    f.write("\n".join(keep) + "\n")
N = len(keep)

ARMS = [
    ('baseline',        {}),
    ('coord_gate',      {'KS_COORD_GATE_MODE': '1'}),
    ('zone_clamp',      {'ENABLE_KS_ZONE_CLAMP': '1'}),
    ('zone2_wide',      {'KS_ZONE2': '1'}),
    ('sf_weak',         {'ENABLE_KS_SF_WEAK': '1'}),
    ('weak_att2',       {'ENABLE_KS_WEAK_ATT2': '1'}),
    ('weak_val',        {'KS_WEAK_VAL_MODE': '1'}),
    ('check_v2',        {'ENABLE_KS_CHECK_V2': '1', 'KS_SAFE_CHECK': '4'}),
    ('sf_safecheck',    {'ENABLE_KS_SF_SAFECHECK': '1', 'KS_SAFE_CHECK': '4'}),
    ('pin_mode',        {'KS_PIN_MODE': '1'}),
    ('flank_breadth',   {'KS_FLANK_MODE': '1'}),
    ('flank_contest',   {'KS_FLANK_MODE': '2'}),
    ('aim',             {'ENABLE_KS_AIM': '1'}),
    ('interact',        {'KS_INTERACT': '1'}),
    ('min_attackers2',  {'KS_MIN_ATTACKERS': '2'}),
    ('battery',         {'KS_BATTERY': '1'}),
    # ---- integrated combos of the helping detectors ----
    ('flank2+zone',     {'KS_FLANK_MODE': '2', 'ENABLE_KS_ZONE_CLAMP': '1'}),
    ('flank2+zone+wv',  {'KS_FLANK_MODE': '2', 'ENABLE_KS_ZONE_CLAMP': '1', 'KS_WEAK_VAL_MODE': '1'}),
    ('integrated_all',  {'KS_FLANK_MODE': '2', 'ENABLE_KS_ZONE_CLAMP': '1', 'KS_WEAK_VAL_MODE': '1',
                         'KS_PIN_MODE': '1', 'KS_ZONE2': '1'}),
    # ---- magnitude-conservation: same detectors, but reshape/scale so the combined output matches SF's level ----
    ('accum_only',      {'KS_ACCUM_MODE': '1'}),
    ('int_all+accum',   {'KS_FLANK_MODE': '2', 'ENABLE_KS_ZONE_CLAMP': '1', 'KS_WEAK_VAL_MODE': '1',
                         'KS_PIN_MODE': '1', 'KS_ZONE2': '1', 'KS_ACCUM_MODE': '1'}),
    ('int_all+acc_t28', {'KS_FLANK_MODE': '2', 'ENABLE_KS_ZONE_CLAMP': '1', 'KS_WEAK_VAL_MODE': '1',
                         'KS_PIN_MODE': '1', 'KS_ZONE2': '1', 'KS_ACCUM_MODE': '1', 'KS_ACCUM_THRESH': '28'}),
    ('int_all+acc_lin', {'KS_FLANK_MODE': '2', 'ENABLE_KS_ZONE_CLAMP': '1', 'KS_WEAK_VAL_MODE': '1',
                         'KS_PIN_MODE': '1', 'KS_ZONE2': '1', 'KS_ACCUM_MODE': '1', 'KS_ACCUM_THRESH': '28',
                         'KS_ACCUM_LIN': '40'}),
    ('int_all+halfmag', {'KS_FLANK_MODE': '2', 'ENABLE_KS_ZONE_CLAMP': '1', 'KS_WEAK_VAL_MODE': '1',
                         'KS_PIN_MODE': '1', 'KS_ZONE2': '1', 'KING_SAFETY_MAG': '1500'}),
]

PY = sys.executable

def run_arm(knobs):
    env = dict(os.environ)
    for k, v in knobs.items():
        env[k] = v
    out = subprocess.run([PY, os.path.abspath(__file__), '--worker', fenfile],
                         cwd=ENGINE_DIR, env=env, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                         text=True, timeout=300).stdout
    ks = [0.0] * N
    for ln in out.splitlines():
        if '\t' in ln:
            i, v = ln.split('\t')
            try:
                ks[int(i)] = float(v)
            except Exception:
                pass
    return ks

base = run_arm({})
print("KS-detector sweep vs SF11 King-safety  (%d non-check collapse fens)" % N)
print("mean|SF11 KS| = %.3f   positions with |SF KS|>0.5 = %d" %
      (sum(abs(x) for x in sf_ks) / max(1, N), sum(1 for x in sf_ks if abs(x) > 0.5)))
print("%-16s %6s %9s %9s %9s %9s" % ("arm", "moved", "mean|d|", "towardSF", "awaySF", "gapRed"))
for label, knobs in ARMS:
    ks = base if label == 'baseline' else run_arm(knobs)
    moved = toward = away = 0
    sd = 0.0
    gapred = 0.0
    for i in range(N):
        d = ks[i] - base[i]
        if abs(d) > 1e-4:
            moved += 1
            sd += abs(d)
            b_gap = abs(base[i] - sf_ks[i])
            a_gap = abs(ks[i] - sf_ks[i])
            gapred += (b_gap - a_gap)
            if a_gap < b_gap - 1e-4:
                toward += 1
            elif a_gap > b_gap + 1e-4:
                away += 1
    md = sd / moved if moved else 0.0
    print("%-16s %6d %9.3f %9d %9d %+9.3f" % (label, moved, md, toward, away, gapred))
