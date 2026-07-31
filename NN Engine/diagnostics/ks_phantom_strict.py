"""Strict KS-over subset: among ks_phantom rows, isolate rows where the KS term is (a) large, (b) SAME signed
direction as the signed over-read (our_total - target_total, both white-POV), i.e. KS is inflating our eval in
the exact direction we over-read. Those are genuine KS-over-reads. Report their count, how much of the over-read
KS accounts for, and candidate discriminators (defenders in king zone, attacker material, safe entry) to see if a
NON-material signal separates them from the ks_real guards. Run: pyrun diagnostics/ks_phantom_strict.py
"""
import os, sys, csv
os.environ.setdefault('ENABLE_KS_V2','1'); os.environ.setdefault('ENABLE_PASSER_V3','1')
os.environ.setdefault('KS_SAFE_CHECK_DEF','5'); os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
for _a in sys.argv[1:]:
    if '=' in _a: _k,_v=_a.split('=',1); os.environ[_k]=_v
THIS=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,os.path.dirname(THIS))
import chess
from ChessAI import ChessAI
ai=ChessAI(None,None,chess.Board(),True)
CORPUS=os.path.join(THIS,'ks_sets','diverse_corpus_ksv2.csv')
def wpov(bd,k): return -bd.get(k,0.0)/1000.0
def sgn(x): return (x>1e-6)-(x<-1e-6)
allrows=list(csv.DictReader(open(CORPUS)))
def stats(tier):
    rows=[r for r in allrows if r['tier']==tier]
    strict=[]; ksdrv_frac=[]
    for r in rows:
        b=chess.Board(r['fen']); bd=ai.ev_breakdown(b)
        tt=float(r['target_total']); ot=wpov(bd,'total'); ks=wpov(bd,'king_safety')
        sover=ot-tt                                  # signed over-read (white-POV pawns)
        if abs(ks)>=1.0 and sgn(ks)==sgn(sover) and abs(sover)>=1.0:
            strict.append((r['fen'],ks,sover,tt))
            ksdrv_frac.append(min(1.0, abs(ks)/max(abs(sover),1e-6)))
    return rows,strict,ksdrv_frac
prows,strict,frac=stats('ks_phantom')
print("ks_phantom: %d rows; STRICT KS-over (ks large, same-dir as over-read, |over|>=1): %d"%(len(prows),len(strict)))
if strict:
    import statistics as st
    print("  mean KS accounts for %.0f%% of the signed over-read on those rows"%(100*st.mean(frac)))
    print("  --- strict KS-over rows (KS is inflating our eval in the over-read direction) ---")
    for fen,ks,sover,tt in sorted(strict,key=lambda z:-abs(z[1]))[:15]:
        print("   ks=%+.2f over=%+.2f sf18=%+.2f  %s"%(ks,sover,tt,fen))
# how does the ks_real GUARD set compare on the same |ks| scale (so we know a KS-magnitude cut would collateral-damage)
rrows,_,_=stats('ks_real')
def kss(rows):
    out=[]
    for r in rows:
        bd=ai.ev_breakdown(chess.Board(r['fen'])); out.append(abs(wpov(bd,'king_safety')))
    return out
import statistics as st
pk=kss(prows); rk=kss(rrows)
print("\n|KS| distribution: ks_phantom mean=%.2f  ks_real(guard) mean=%.2f  -> overlap means a blunt |KS| cut hits both"%(st.mean(pk),st.mean(rk)))
