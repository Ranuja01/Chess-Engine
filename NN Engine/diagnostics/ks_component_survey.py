"""ROUND 0 (KS convertibility rebuild): per-term firing survey across diverse_corpus_ksplus.csv tiers.
For each position in {ks_fixable, ks_edge, ks_real} dump OUR KS sub-components (attsq/weak/safe/defpc/openf/units)
per king via the gated KS_DEBUG_DUMP (C++ stderr, OS-redirected to a temp file), take the higher-danger king, and
aggregate per tier. GOAL: confirm ks_fixable (over-read) is inflated by the DEFENSE-BLIND adders (attsq, openf)
WITH defenders present (defpc>0), while the GUARD tiers (ks_real real attacks) get their units from the
DEFENSE-AWARE terms (weak, safe) -- so defense-gating the adders won't break the guards.
Run: pyrun diagnostics/ks_component_survey.py
"""
import os, sys, csv, re, tempfile
os.environ.setdefault('ENABLE_KS_V2','1'); os.environ.setdefault('ENABLE_PASSER_V3','1')
os.environ['KS_DEBUG_DUMP']='1'; os.environ.setdefault('KS_SAFE_CHECK_DEF','5'); os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
for _a in sys.argv[1:]:
    if '=' in _a: _k,_v=_a.split('=',1); os.environ[_k]=_v
THIS=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,os.path.dirname(THIS))
import chess
from ChessAI import ChessAI
ai=ChessAI(None,None,chess.Board(),True)
CORPUS=os.path.join(THIS,'ks_sets','diverse_corpus_ksplus.csv')
rows=[r for r in csv.DictReader(open(CORPUS)) if r['tier'] in ('ks_fixable','ks_edge','ks_real')]

# OS-redirect fd 2 (C++ stderr) to a temp file; write per-position markers to the same fd so order is preserved.
tmp=tempfile.NamedTemporaryFile('w+',delete=False,suffix='.kslog'); tmppath=tmp.name; tmp.close()
saved=os.dup(2); fd=os.open(tmppath,os.O_WRONLY|os.O_TRUNC)
os.dup2(fd,2)
for i,r in enumerate(rows):
    os.write(2,("###|%s|%s\n"%(r['tier'],r['fen'])).encode())
    ai.ev_breakdown(chess.Board(r['fen']))
os.fsync(2); os.dup2(saved,2); os.close(fd); os.close(saved)

# parse: each ### marker followed by up to two KSD lines (W and B). take the higher-units king.
KSD=re.compile(r'KSD (\w) attsq=(\d+) weak=(\d+) safe=(\d+) attpc=(\d+) defpc=(\d+) openf=(\d+) bkru=(\d+) over=(\d+) units=(\d+) danger=(\d+)')
from collections import defaultdict
agg=defaultdict(lambda: defaultdict(float)); n=defaultdict(int)
cur=None; best=None
def flush():
    global best
    if cur and best:
        t=cur; n[t]+=1
        for k,v in best.items(): agg[t][k]+=v
for line in open(tmppath):
    m=re.match(r'###\|(\w+)\|',line)
    if m:
        flush(); cur=m.group(1); best=None; continue
    d=KSD.search(line)
    if d:
        vals=dict(attsq=int(d.group(2)),weak=int(d.group(3)),safe=int(d.group(4)),attpc=int(d.group(5)),
                  defpc=int(d.group(6)),openf=int(d.group(7)),bkru=int(d.group(8)),over=int(d.group(9)),units=int(d.group(10)))
        if best is None or vals['units']>best['units']: best=vals
flush()
os.unlink(tmppath)
print("ROUND 0 — per-term firing by tier (higher-danger king), mean over positions:\n")
print("%-11s %5s | %6s %6s %6s %6s %6s %6s %6s %6s"%("tier","n","units","attsq","bkru","over","weak","safe","defpc","openf"))
for t in ('ks_fixable','ks_edge','ks_real'):
    if not n[t]: continue
    a=agg[t]; c=n[t]
    print("%-11s %5d | %6.1f %6.2f %6.2f %6.2f %6.2f %6.2f %6.2f %6.2f"%(t,c,a['units']/c,a['attsq']/c,a['bkru']/c,a['over']/c,a['weak']/c,a['safe']/c,a['defpc']/c,a['openf']/c))
print("\nKEY = bkru (attacked squares with attackers>defenders = real breakthrough). If ks_real bkru >> ks_fixable bkru,")
print("per-square defense-gating SEPARATES them (viable). If bkru overlaps, the discrimination gap is real (expect game-neutral).")
