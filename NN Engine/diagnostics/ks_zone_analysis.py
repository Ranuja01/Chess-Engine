"""ZONE hypothesis: our KS over-count comes from zone squares OUTSIDE SF11's tight 8-square king ring (our zone =
ring1 + one rank forward). Compare the OUT-of-ring fraction of danger squares for ks_fixable (phantom) vs ks_real
(real attacks). If phantom OUT%% >> real OUT%%, restricting to SF's ring SEPARATES them (viable static fix).
Picks the higher-danger king per position (from KSD units). Run: pyrun diagnostics/ks_zone_analysis.py
"""
import os, sys, csv, re, tempfile
os.environ.setdefault('ENABLE_KS_V2','1'); os.environ.setdefault('ENABLE_PASSER_V3','1')
os.environ['KS_DEBUG_DUMP']='1'; os.environ['KS_TRACE']='1'; os.environ.setdefault('KS_SAFE_CHECK_DEF','5'); os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
for _a in sys.argv[1:]:
    if '=' in _a: _k,_v=_a.split('=',1); os.environ[_k]=_v
THIS=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,os.path.dirname(THIS))
import chess
from ChessAI import ChessAI
ai=ChessAI(None,None,chess.Board(),True)
def ring8(sq):
    f,r=chess.square_file(sq),chess.square_rank(sq); s=set()
    for df in(-1,0,1):
        for dr in(-1,0,1):
            if df or dr:
                nf,nr=f+df,r+dr
                if 0<=nf<8 and 0<=nr<8: s.add(chess.square(nf,nr))
    return s
rows=[r for r in csv.DictReader(open(os.path.join(THIS,'ks_sets','diverse_corpus_ksplus.csv'))) if r['tier'] in ('ks_fixable','ks_real')]
tmp=tempfile.NamedTemporaryFile(delete=False,suffix='.kslog').name
saved=os.dup(2); fd=os.open(tmp,os.O_WRONLY|os.O_TRUNC); os.dup2(fd,2)
meta=[]
for r in rows:
    b=chess.Board(r['fen']); wk=b.king(chess.WHITE); bk=b.king(chess.BLACK)
    os.write(2,("###|%s|%d|%d\n"%(r['tier'],wk,bk)).encode()); ai.ev_breakdown(b); meta.append((r['tier'],wk,bk))
os.fsync(2); os.dup2(saved,2); os.close(fd); os.close(saved)
ZS=re.compile(r'ZS\[(\w)\] sq=(\d+) attby=N(\d+)B(\d+)R(\d+)Q(\d+)\s+ndef=(\d+).*weak=(\d+)')
KSD=re.compile(r'KSD (\w) .* units=(\d+)')
from collections import defaultdict
blocks=[]; cur=None
for line in open(tmp):
    m=re.match(r'###\|(\w+)\|(\d+)\|(\d+)\s*$',line)
    if m: cur=dict(tier=m.group(1),wk=int(m.group(2)),bk=int(m.group(3)),zs={'W':[],'B':[]},u={'W':0,'B':0}); blocks.append(cur); continue
    if cur is None: continue
    z=ZS.search(line)
    if z: cur['zs'][z.group(1)].append((int(z.group(2)), (int(z.group(3))+int(z.group(4))+int(z.group(5))+int(z.group(6)))>0, int(z.group(8)))); continue
    k=KSD.search(line)
    if k: cur['u'][k.group(1)]=int(k.group(2))
os.unlink(tmp)
agg=defaultdict(lambda: dict(att=0,att_out=0,weak=0,weak_out=0,n=0))
for b in blocks:
    kc='W' if b['u']['W']>=b['u']['B'] else 'B'   # higher-danger king
    ksq=b['wk'] if kc=='W' else b['bk']; ring=ring8(ksq); a=agg[b['tier']]; a['n']+=1
    for sq,att,weak in b['zs'][kc]:
        if att: a['att']+=1; a['att_out']+= (sq not in ring)
        if weak: a['weak']+=1; a['weak_out']+= (sq not in ring)
print("OUT-of-SF-ring fraction of danger squares, higher-danger king, by tier:\n")
print("%-11s %4s | %-22s %-22s"%("tier","n","attacked sq (out/tot)","weak sq (out/tot)"))
for t in ('ks_fixable','ks_real'):
    a=agg[t]
    print("%-11s %4d | %2d/%2d = %3.0f%% out          %2d/%2d = %3.0f%% out"%(t,a['n'],
        a['att_out'],a['att'],100*a['att_out']/max(a['att'],1), a['weak_out'],a['weak'],100*a['weak_out']/max(a['weak'],1)))
print("\nREAD: phantom OUT%% >> real OUT%% => restricting KS to SF's 8-ring damps phantoms, keeps real (viable).")
