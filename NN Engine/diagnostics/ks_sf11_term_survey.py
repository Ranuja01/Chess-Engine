"""Does SF11-STATIC's own King-safety term SEPARATE our phantom (ks_fixable) from real (ks_real)? If yes, the
discrimination IS statically achievable (SF11 does it, no search) and we must reverse-engineer WHICH SF11 static
computation does it. Aggregate SF11's labeled classical terms + OUR king_safety per tier.
Run: pyrun diagnostics/ks_sf11_term_survey.py
"""
import os, sys, csv
os.environ.setdefault('ENABLE_KS_V2','1'); os.environ.setdefault('ENABLE_PASSER_V3','1')
os.environ.setdefault('KS_SAFE_CHECK_DEF','5'); os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
for _a in sys.argv[1:]:
    if '=' in _a: _k,_v=_a.split('=',1); os.environ[_k]=_v
THIS=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,os.path.dirname(THIS)); sys.path.insert(0,THIS)
import chess
from ChessAI import ChessAI
ai=ChessAI(None,None,chess.Board(),True)
def wpov(bd,k): return -bd.get(k,0.0)/1000.0
SF11PATH=os.environ.get('SF11_BIN',"/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/stockfish_11_linux/stockfish-11-linux/Linux/stockfish_20011801_x64_bmi2")
from eval_vs_sf11 import SF11Eval; SF11=SF11Eval(SF11PATH)
CORPUS=os.path.join(THIS,'ks_sets','diverse_corpus_ksplus.csv')
rows=[r for r in csv.DictReader(open(CORPUS)) if r['tier'] in ('ks_fixable','ks_edge','ks_real')]
from collections import defaultdict
import statistics as st
# SF11 term label for king safety is 'King safety'; also grab Mobility, Threats, Passed for context
LBLS=['King safety','Mobility','Threats','Space']
agg=defaultdict(lambda: defaultdict(list))
for r in rows:
    fen=r['fen']; t=r['tier']
    total,terms=SF11.eval(fen)
    if total is None: continue    # in-check -> skip
    bd=ai.ev_breakdown(chess.Board(fen))
    agg[t]['our_ks'].append(wpov(bd,'king_safety'))
    agg[t]['sf11_total'].append(total)
    for L in LBLS:
        v=next((terms[k] for k in terms if k.lower()==L.lower()), None)
        if v is not None: agg[t]['sf11_'+L].append(v)
SF11.close()
print("Per-tier means (white-POV pawns). KEY: does |sf11 King safety| SEPARATE ks_fixable (phantom) from ks_real (real)?\n")
hdr=['tier','n','our_ks','|our_ks|','sf11_KS','|sf11_KS|','sf11_Mob','sf11_total']
print(("%-11s"+" %8s"*7)%tuple(hdr))
for t in ('ks_fixable','ks_edge','ks_real'):
    a=agg[t]; n=len(a['our_ks'])
    if not n: continue
    ks=a.get('sf11_King safety',[0]); mob=a.get('sf11_Mobility',[0])
    print(("%-11s"+" %8d"+" %8.2f"*6)%(t,n,
        st.mean(a['our_ks']), st.mean(abs(x) for x in a['our_ks']),
        st.mean(ks), st.mean(abs(x) for x in ks),
        st.mean(mob), st.mean(a['sf11_total'])))
print("\nREAD: if sf11 |King safety| on ks_real >> on ks_fixable, SF's KS computation is the static discriminator")
print("(and ours isn't) -> reverse-engineer WHICH SF term. If SF11-KS also flat, separation lives in another SF term.")
