"""Assemble the SF11/SF18-validated KS tiers from ks_failure_hunt.csv into the fit schema and fold into
diverse_corpus_ksv2.csv:  FIXABLE -> tier 'ks_fixable' (target: our KS over-reads, SF11 proves statically
fixable) ; WE_BEAT_SF11 -> tier 'ks_edge' (GUARD: our static already beats SF11 -> must not regress).
Keeps the existing families/phantom/real tiers. Writes diverse_corpus_ksplus.csv. Run: pyrun diagnostics/ks_build_fixable_tier.py
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
def phase_bucket(ps): return "opening" if ps<24 else "midgame" if ps<64 else "endgame" if ps<104 else "adveg"
def wpov(bd,k): return -bd.get(k,0.0)/1000.0

HUNT=os.path.join(THIS,'ks_failure_hunt.csv')
BASE=os.path.join(THIS,'ks_sets','diverse_corpus_ksv2.csv')
OUT =os.path.join(THIS,'ks_sets','diverse_corpus_ksplus.csv')
FIELDS=['fen','target_ks','target_total','our_total_base','our_ks_base','tier','phase_bucket','split']

base_rows=list(csv.DictReader(open(BASE)))
base_fens={r['fen'] for r in base_rows}
new=[]
from collections import Counter
cnt=Counter()
for r in csv.DictReader(open(HUNT)):
    fen=r['fen']
    if fen in base_fens: continue
    v=r.get('verdict')
    tier='ks_fixable' if v=='FIXABLE' else 'ks_edge' if v=='WE_BEAT_SF11' else None
    if tier is None: continue
    bd=ai.ev_breakdown(chess.Board(fen)); ps=bd.get('phase_score',0)
    # deterministic split from fen hash (no RNG)
    split='val' if (sum(ord(c) for c in fen)%5==0) else 'train'
    new.append(dict(fen=fen, target_ks=r.get('ks','0'), target_total=r['sf18'],
                    our_total_base=r['ours'], our_ks_base=r.get('ks','0'),
                    tier=tier, phase_bucket=phase_bucket(ps), split=split))
    cnt[(tier,phase_bucket(ps))]+=1
w=csv.DictWriter(open(OUT,'w',newline=''),fieldnames=FIELDS); w.writeheader()
for r in base_rows: w.writerow({k:r.get(k,'') for k in FIELDS})
for r in new: w.writerow(r)
print("wrote",OUT)
print("base rows kept:",len(base_rows),"  new KS-validated rows:",len(new))
print("new tier/phase:",dict(cnt))
print("total tier distribution:",dict(Counter(r['tier'] for r in base_rows)+Counter(r['tier'] for r in new)))
