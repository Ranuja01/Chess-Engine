"""For the statically-fixable KS FENs (from ks_failure_hunt.csv), show OUR king_safety term next to SF11's own
'King safety' classical term + SF18 truth -> proves the over-read is our KS TERM. Run: pyrun diagnostics/ks_term_dossier.py
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
rows=[r for r in csv.DictReader(open(os.path.join(THIS,'ks_failure_hunt.csv'))) if r['top_term']=='king_safety']
print("%d KS-driven statically-fixable positions (our KS term vs SF11 King-safety term):\n"%len(rows))
for r in rows:
    fen=r['fen']; bd=ai.ev_breakdown(chess.Board(fen))
    total, terms = SF11.eval(fen)
    # SF11 term labels: find the king-safety-ish label
    ksl=next((k for k in terms if 'king' in k.lower()), None)
    sf11_ks = terms.get(ksl) if ksl else None
    print("FEN: %s   (stm %s)"%(fen, "w" if chess.Board(fen).turn else "b"))
    print("  totals   OURS=%+.2f   SF11-static=%+.2f   SF18-search=%+.2f   (err vs SF18 %+.2f)"%(
        wpov(bd,'total'), total, float(r['sf18']), wpov(bd,'total')-float(r['sf18'])))
    print("  KING-SAFETY term:  OURS=%+.2f   SF11 '%s'=%s   | our det_ks_units w/b = %d/%d"%(
        wpov(bd,'king_safety'), ksl or '?', ("%+.2f"%sf11_ks if sf11_ks is not None else "n/a"),
        bd.get('det_ks_units_w',0), bd.get('det_ks_units_b',0)))
    # our next-biggest terms for context
    others=sorted([(t,wpov(bd,t)) for t in ['pieces','imbalance_white','imbalance_black','capture_gains',
        'passed_pawn_support','piece_value_boost','kaufman_imbalance','central'] if abs(wpov(bd,t))>=0.4],key=lambda z:-abs(z[1]))
    if others: print("  our other terms: "+"  ".join("%s=%+.2f"%(t,v) for t,v in others[:5]))
    print()
SF11.close()
