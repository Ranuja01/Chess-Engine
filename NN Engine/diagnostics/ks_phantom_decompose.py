"""Decompose the ks_phantom over-read into eval terms to find what ACTUALLY drives it (KS? imbalance/OvD?
passer? central?), and whether the attacking side is materially under-backed (does MOD_KS_REALIZ even apply).
Runs with ENABLE_KS_V2=1 ENABLE_PASSER_V3=1. Read-only. Run: pyrun diagnostics/ks_phantom_decompose.py
"""
import os, sys, csv
os.environ.setdefault('ENABLE_KS_V2', '1'); os.environ.setdefault('ENABLE_PASSER_V3', '1')
os.environ.setdefault('KS_SAFE_CHECK_DEF', '5'); os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
for _a in sys.argv[1:]:
    if '=' in _a: _k,_v=_a.split('=',1); os.environ[_k]=_v
THIS = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(THIS))
import chess
from ChessAI import ChessAI
ai = ChessAI(None, None, chess.Board(), True)
CORPUS = os.path.join(THIS, 'ks_sets', 'diverse_corpus_ksv2.csv')

def wpov(bd, key): return -bd.get(key, 0.0) / 1000.0
rows = [r for r in csv.DictReader(open(CORPUS)) if r['tier'] == 'ks_phantom']
print("ks_phantom rows:", len(rows))
import statistics as st
agg = {k: [] for k in ['over','ks','imb','cap','central','passed','pt_pawns','matedge_attacker','underbacked']}
print("%-7s %-6s %-6s %-6s %-6s %-6s %-7s  %s" % ("over","ks","imb","cap","cent","pass","matEdgeA","fen"))
for r in rows:
    b = chess.Board(r['fen']); bd = ai.ev_breakdown(b)
    tt = float(r['target_total']); ot = wpov(bd, 'total')
    over = abs(ot) - abs(tt)
    ks = wpov(bd, 'king_safety')
    imb = wpov(bd, 'imbalance_white') + wpov(bd, 'imbalance_black')
    cap = wpov(bd, 'capture_gains'); central = wpov(bd, 'central'); passed = wpov(bd, 'passed_pawn_support')
    # attacking side = side our eval favors as the danger-giver: ks>0 => black attacks white king => attacker=black
    wpv = bd.get('det_w_pieceval', 0)/1000.0; bpv = bd.get('det_b_pieceval', 0)/1000.0
    mat_edge_attacker = (bpv - wpv) if ks >= 0 else (wpv - bpv)   # attacker's own-minus-enemy material
    underbacked = mat_edge_attacker < 0
    for k, v in [('over',over),('ks',ks),('imb',imb),('cap',cap),('central',central),('passed',passed),
                 ('pt_pawns',wpov(bd,'pt_pawns')),('matedge_attacker',mat_edge_attacker),('underbacked',int(underbacked))]:
        agg[k].append(v)
    print("%-7.2f %-6.2f %-6.2f %-6.2f %-6.2f %-6.2f %-7.2f  %s" % (over,ks,imb,cap,central,passed,mat_edge_attacker,r['fen']))
print("\n--- means over %d phantom rows ---" % len(rows))
for k in ['over','ks','imb','cap','central','passed','pt_pawns','matedge_attacker']:
    print("  mean |%-8s| contribution... mean=%+.2f  mean_abs=%.2f" % (k, st.mean(agg[k]), st.mean(abs(x) for x in agg[k])))
ub = sum(agg['underbacked'])
print("  attacker under-backed (MOD_KS_REALIZ can fire): %d / %d = %.0f%%" % (ub, len(rows), 100*ub/len(rows)))
