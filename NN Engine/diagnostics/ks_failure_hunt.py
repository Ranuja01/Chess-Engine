"""Hunt KS-driven eval FAILURES for human review. Triangulate ours / SF11-static / SF18-search over mined
collapse positions, classify by the method, and surface the STATICALLY-FIXABLE ones (SF11 agrees with SF18
directionally, WE are wrong) where our king_safety term is the top culprit. Writes ks_failure_hunt.csv.
Run: pyrun diagnostics/ks_failure_hunt.py [N=60] [DEPTH=18]
"""
import os, sys, csv
os.environ.setdefault('ENABLE_KS_V2','1'); os.environ.setdefault('ENABLE_PASSER_V3','1')
os.environ.setdefault('KS_SAFE_CHECK_DEF','5'); os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
N=60; DEPTH=18
for _a in sys.argv[1:]:
    if _a.startswith('N='): N=int(_a.split('=',1)[1])
    elif _a.startswith('DEPTH='): DEPTH=int(_a.split('=',1)[1])
    elif '=' in _a: _k,_v=_a.split('=',1); os.environ[_k]=_v
THIS=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,os.path.dirname(THIS)); sys.path.insert(0,THIS)
import chess
from ChessAI import ChessAI
ai=ChessAI(None,None,chess.Board(),True)
def wpov(bd,k): return -bd.get(k,0.0)/1000.0
def ae_delta(bd): return (-(bd.get('advanced_endgame_total',0)-bd.get('ae_input',0))/1000.0) if bd.get('advanced_endgame_fired') else 0.0
PARTVIEW=['pieces','king_safety','imbalance_white','imbalance_black','kaufman_imbalance','piece_value_boost',
          'pair_bonus','capture_gains','passed_pawn_support','central','pawn_struct','outpost','space','mobility',
          'rook_cond','pawn_majority','latent_threat','threats']
def top_term(bd):
    terms=[(t,wpov(bd,t)) for t in PARTVIEW]+[('AE',ae_delta(bd))]
    terms.sort(key=lambda z:-abs(z[1])); return terms[0]

SF11=None
SF11PATH=os.environ.get('SF11_BIN',"/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/stockfish_11_linux/stockfish-11-linux/Linux/stockfish_20011801_x64_bmi2")
from eval_vs_sf11 import SF11Eval; SF11=SF11Eval(SF11PATH)
from selfplay.arbiter import find_stockfish; SFPATH=find_stockfish()
import chess.engine
sf18eng=chess.engine.SimpleEngine.popen_uci(SFPATH)
def sf18(fen):
    sc=sf18eng.analyse(chess.Board(fen), chess.engine.Limit(depth=DEPTH))['score'].white()
    v=sc.score(mate_score=99000); return v/100.0 if v is not None else None

def side(x, dead=0.75):  # -1 black / 0 drawish / +1 white
    return 0 if abs(x)<dead else (1 if x>0 else -1)

# mine diverse drop_fens across all classes
rows=list(csv.DictReader(open(os.path.join(THIS,'ks_sets','collapse_dataset_classified.csv'))))
seen=set(); pool=[]
for r in sorted(rows,key=lambda r:-abs(float(r.get('swing') or 0))):
    f=r.get('drop_fen')
    if not f or f in seen: continue
    try: chess.Board(f)
    except Exception: continue
    seen.add(f); pool.append(r)
try: sys.stdout.reconfigure(line_buffering=True)
except Exception: pass
step=max(1,len(pool)//(N*2)); cand=pool[::step][:N*2]
print("triangulating %d mined positions (SF18 depth %d)..."%(len(cand),DEPTH),flush=True)
out=[]
for i,r in enumerate(cand):
    if i%10==0: print("  [%d/%d]"%(i,len(cand)),flush=True)
    fen=r['drop_fen']; bd=ai.ev_breakdown(chess.Board(fen))
    ours=wpov(bd,'total'); s11=SF11.eval(fen)[0]; s18=sf18(fen)
    if s11 is None or s18 is None or abs(s18)>50: continue  # skip in-check (SF11 eval None) + tactical mates
    tt,tv=top_term(bd)
    so,s1,s8=side(ours),side(s11),side(s18)
    fixable = (s1==s8) and (so!=s8)            # SF11 right, we wrong -> statically fixable
    we_beat = (so==s8) and (s1!=s8)            # we right, SF11 wrong -> leave alone
    out.append(dict(fen=fen, ours=round(ours,2), sf11=round(s11,2), sf18=round(s18,2),
                    err=round(ours-s18,2), top_term=tt, top_val=round(tv,2),
                    ks=round(wpov(bd,'king_safety'),2), verdict=('FIXABLE' if fixable else ('WE_BEAT_SF11' if we_beat else '')),
                    cls=r.get('ks_class')))
sf18eng.quit(); SF11.close()
fix=[o for o in out if o['verdict']=='FIXABLE']
beat=[o for o in out if o['verdict']=='WE_BEAT_SF11']
ksfix=[o for o in fix if o['top_term']=='king_safety']
print("\n== %d fixable (SF11 right, we wrong); %d KS-driven || %d WE_BEAT_SF11 (our edge, GUARD) =="%(len(fix),len(ksfix),len(beat)))
csvp=os.path.join(THIS,'ks_failure_hunt.csv')
with open(csvp,'w',newline='') as fp:
    w=csv.DictWriter(fp,fieldnames=list(out[0].keys())); w.writeheader()
    for o in sorted(fix+beat,key=lambda o:(o['verdict'],o['top_term']!='king_safety', -abs(o['err']))): w.writerow(o)
print("wrote",csvp,"(FIXABLE + WE_BEAT_SF11 rows)\n")
if beat:
    print("-- WE_BEAT_SF11 (preserve these) --")
    for o in sorted(beat,key=lambda o:-abs(o['ours']-o['sf18']) if False else 0)[:8]:
        print("  OURS=%+.2f SF11=%+.2f SF18=%+.2f  top=%s  %s"%(o['ours'],o['sf11'],o['sf18'],o['top_term'],o['fen']))
print("%-6s %-6s %-6s %-6s  %-13s %-6s  %s"%("OURS","SF11","SF18","err","topTerm","KS","fen"))
for o in sorted(ksfix,key=lambda o:-abs(o['err']))[:20]:
    print("%-6.2f %-6.2f %-6.2f %-6.2f  %-13s %-6.2f  %s"%(o['ours'],o['sf11'],o['sf18'],o['err'],o['top_term'],o['ks'],o['fen']))
