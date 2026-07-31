"""Mine collapses for a chosen profile (default: the least-examined ks_class), then dissect a diverse sample:
ours (clean-partition top terms) vs SF11-static vs SF18-search. Runs V2+V3.
Run: pyrun diagnostics/mine_and_dissect.py [CLASS=other] [N=8] [DEPTH=18]
"""
import os, sys, csv
os.environ.setdefault('ENABLE_KS_V2','1'); os.environ.setdefault('ENABLE_PASSER_V3','1')
os.environ.setdefault('KS_SAFE_CHECK_DEF','5'); os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
CLASS=None; N=8; DEPTH=18
for _a in sys.argv[1:]:
    if _a.startswith('CLASS='): CLASS=_a.split('=',1)[1]
    elif _a.startswith('N='): N=int(_a.split('=',1)[1])
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

# SF references
SF11=None
SF11PATH=os.environ.get('SF11_BIN',
    "/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/stockfish_11_linux/stockfish-11-linux/Linux/stockfish_20011801_x64_bmi2")
try:
    from eval_vs_sf11 import SF11Eval; SF11=SF11Eval(SF11PATH)
    _t=SF11.eval("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")[0]
    print("(SF11-static OK, startpos eval=%s)"%_t)
except Exception as e: print("(SF11 static n/a:",e,")"); SF11=None
sf_path=None
try:
    from selfplay.arbiter import find_stockfish; sf_path=find_stockfish()
except Exception:
    try:
        from arbiter import find_stockfish; sf_path=find_stockfish()
    except Exception as e: print("(SF18 n/a:",e,")")
def sf18(fen):
    if not sf_path: return None
    import chess.engine
    eng=chess.engine.SimpleEngine.popen_uci(sf_path)
    try:
        sc=eng.analyse(chess.Board(fen), chess.engine.Limit(depth=DEPTH))['score'].white()
        v=sc.score(mate_score=99000); return v/100.0 if v is not None else None
    finally: eng.quit()

DS=os.path.join(THIS,'ks_sets','collapse_dataset_classified.csv')
rows=list(csv.DictReader(open(DS)))
from collections import Counter
dist=Counter(r.get('ks_class','?') for r in rows)
print("ks_class distribution:", dict(dist))
if CLASS is None:
    # pick the least-examined class among non-empty (exclude the two we've dissected)
    seen={'positional','ks_attack'}
    cand=[(c,n) for c,n in dist.items() if c not in seen and c!='?']
    CLASS=sorted(cand,key=lambda z:-z[1])[0][0] if cand else 'other'
print("dissecting profile: ks_class=%s  (N=%d, SF18 depth=%d)\n"%(CLASS,N,DEPTH))
# diverse sample: unique drop_fen, spread across swing magnitude
pool=[r for r in rows if r.get('ks_class')==CLASS and r.get('drop_fen')]
seen=set(); uniq=[]
for r in sorted(pool,key=lambda r:-abs(float(r.get('swing') or 0))):
    f=r['drop_fen']
    if f in seen: continue
    seen.add(f); uniq.append(r)
step=max(1,len(uniq)//N); sample=uniq[::step][:N]
for r in sample:
    fen=r['drop_fen']
    try: b=chess.Board(fen)
    except Exception: continue
    bd=ai.ev_breakdown(b); ot=wpov(bd,'total')
    s11=SF11.eval(fen)[0] if SF11 else None
    s18=sf18(fen)
    print("="*90)
    print("%s | stm:%s swing=%s"%(fen,"w" if b.turn else "b",r.get('swing')))
    print("  OURS=%+.2f   SF11-static=%s   SF18=%s   | OURS-SF18=%s"%(
        ot, ("%+.2f"%s11 if s11 is not None else "n/a"), ("%+.2f"%s18 if s18 is not None else "n/a"),
        ("%+.2f"%(ot-s18) if s18 is not None else "n/a")))
    terms=[(t,wpov(bd,t)) for t in PARTVIEW]+[('AE_delta',ae_delta(bd))]
    terms=[(t,v) for t,v in terms if abs(v)>=0.30]
    terms.sort(key=lambda z:-abs(z[1]))
    print("   top terms: "+"  ".join("%s=%+.2f"%(t,v) for t,v in terms[:8]))
if SF11: SF11.close()
