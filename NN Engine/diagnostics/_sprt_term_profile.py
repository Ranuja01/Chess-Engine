# -*- coding: utf-8 -*-
"""Finish Phase C: find the AXIS that separates HURT (integration blew a win) from HELPED (held). For both sets
compute SF11's term profile (what KIND of position it is) AND our-integration term gap vs SF per term. The term
whose gap differs MOST between hurt and helped is the discriminator to rebalance. Integration knobs ON (that's
the config that produced the split). ~1 core.

Run: bash <runner> pyrun diagnostics/_sprt_term_profile.py
"""
import os, sys, signal
# integration knobs must be set BEFORE ChessAI construction
for k, v in {'KS_FLANK_MODE': '2', 'ENABLE_KS_ZONE_CLAMP': '1', 'KS_WEAK_VAL_MODE': '1',
             'KS_PIN_MODE': '1', 'KS_ZONE2': '1', 'KS_ACCUM_MODE': '1', 'KS_ACCUM_THRESH': '28'}.items():
    os.environ[k] = v
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
import csv, chess, statistics as st
THIS_DIR = os.path.dirname(os.path.abspath(__file__)); ENGINE_DIR = os.path.dirname(THIS_DIR)
sys.path.insert(0, ENGINE_DIR); sys.path.insert(0, THIS_DIR)
from eval_vs_sf11 import SF11Eval, SF11
class _TO(Exception): pass
signal.signal(signal.SIGALRM, lambda s, f: (_ for _ in ()).throw(_TO()))

# map our ev_breakdown terms to SF11 term names (White-POV pawns). Fuzzy but the well-defined ones are clean.
PAIRS = [('king_safety', 'King safety'), ('threats', 'Threats'), ('passed_pawn_support', 'Passed'),
         ('mobility', 'Mobility'), ('material', 'Material')]

from ChessAI import ChessAI
seed = chess.Board(); ai = ChessAI(None, None, seed, seed.turn)
sf11 = SF11Eval(SF11)

def profile(path):
    fens = []
    for r in csv.DictReader(open(os.path.join(ENGINE_DIR, path))):
        fen = (r.get('fen_start') or '').strip()
        if fen and not chess.Board(fen).is_check():
            fens.append(fen)
    sfacc = {}; gapacc = {}
    n = 0
    for fen in fens:
        signal.alarm(8)
        try:
            _, terms = sf11.eval(fen); signal.alarm(0)
        except _TO:
            continue
        finally:
            signal.alarm(0)
        if terms is None: continue
        n += 1
        bd = ai.ev_breakdown(chess.Board(fen))
        for ours_k, sf_k in PAIRS:
            ov = -bd.get(ours_k, 0) / 1000.0
            sv = terms.get(sf_k, 0.0)
            sfacc.setdefault(sf_k, []).append(sv)
            gapacc.setdefault(sf_k, []).append(ov - sv)
    return n, sfacc, gapacc

nh, sfh, gh = profile('diagnostics/_sprt_hurt.csv')
ne, sfe, ge = profile('diagnostics/_sprt_helped.csv')
sf11.close()

print("Phase-C discriminator search: HURT (n=%d) vs HELPED (n=%d), integration eval\n" % (nh, ne))
print("%-12s %10s %10s %8s   %10s %10s %8s" % ("term", "SF|hurt|", "SF|help|", "dSF", "gap_hurt", "gap_help", "dGap"))
for _, sf_k in PAIRS:
    sfH = st.mean([abs(x) for x in sfh[sf_k]]); sfE = st.mean([abs(x) for x in sfe[sf_k]])
    gH = st.mean(gh[sf_k]); gE = st.mean(ge[sf_k])
    print("%-12s %10.3f %10.3f %+8.3f   %+10.3f %+10.3f %+8.3f" % (sf_k, sfH, sfE, sfH-sfE, gH, gE, gH-gE))
print("\ndSF = how much MORE of that term SF sees in the blown positions (position character).")
print("dGap = how much more WRONG our term is in blown vs held (our failure axis). Biggest |dGap| = the discriminator.")
