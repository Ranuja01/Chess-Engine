"""Test the LOCAL-MAX hypothesis: does an SF-like 'safe-check dominant, adders near-zero' regime SEPARATE our KS on
phantom (ks_fixable) vs real (ks_real) better than today's adder-dominant defaults? Reports per-tier mean |our KS|
and mean signed KS. Run TWICE via the runner with different knobs:
  pyrun diagnostics/ks_regime_probe.py                # defaults
  pyrun diagnostics/ks_regime_probe.py KS_SAFE_CHECK=40 KS_SAFE_CHECK_DEF=40 KS_ATT_KNIGHT=0 KS_ATT_BISHOP=0 \
        KS_ATT_ROOK=0 KS_ATT_QUEEN=0 KS_ATTACK_COUNT=0 KS_OPEN_FILE=0 KS_WEAK=0
"""
import os, sys, csv
os.environ.setdefault('ENABLE_KS_V2','1'); os.environ.setdefault('ENABLE_PASSER_V3','1')
os.environ.setdefault('KS_SAFE_CHECK_DEF','5'); os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
for _a in sys.argv[1:]:
    if '=' in _a: _k,_v=_a.split('=',1); os.environ[_k]=_v
THIS=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,os.path.dirname(THIS))
import chess, statistics as st
from ChessAI import ChessAI
ai=ChessAI(None,None,chess.Board(),True)
def wpov(bd,k): return -bd.get(k,0.0)/1000.0
rows=[r for r in csv.DictReader(open(os.path.join(THIS,'ks_sets','diverse_corpus_ksplus.csv'))) if r['tier'] in ('ks_fixable','ks_real')]
from collections import defaultdict
agg=defaultdict(list)
for r in rows:
    bd=ai.ev_breakdown(chess.Board(r['fen'])); agg[r['tier']].append(wpov(bd,'king_safety'))
print("knobs: SAFE_CHECK=%s DEF=%s ATT(N/B/R/Q)=%s/%s/%s/%s ATTCOUNT=%s OPENFILE=%s WEAK=%s"%(
    os.environ.get('KS_SAFE_CHECK','3'),os.environ.get('KS_SAFE_CHECK_DEF','5'),
    os.environ.get('KS_ATT_KNIGHT','2'),os.environ.get('KS_ATT_BISHOP','2'),os.environ.get('KS_ATT_ROOK','3'),os.environ.get('KS_ATT_QUEEN','5'),
    os.environ.get('KS_ATTACK_COUNT','1'),os.environ.get('KS_OPEN_FILE','2'),os.environ.get('KS_WEAK','2')))
for t in ('ks_fixable','ks_real'):
    v=agg[t]; print("  %-11s n=%2d  mean|KS|=%.2f  meanKS=%+.2f"%(t,len(v),st.mean(abs(x) for x in v),st.mean(v)))
f=st.mean(abs(x) for x in agg['ks_fixable']); rr=st.mean(abs(x) for x in agg['ks_real'])
print("  SEPARATION (real|KS| - phantom|KS|) = %+.2f  (want POSITIVE: real loud, phantom quiet)"%(rr-f))
