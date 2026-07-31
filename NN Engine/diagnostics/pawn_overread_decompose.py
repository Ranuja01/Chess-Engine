"""Isolate the PURE pawn/placement over-read: rows where KS is NOT firing (|our_ks|<0.3) yet we still over-read
SF18 by >=1.5 (magnitude). Decompose by term and test the doubled-pawn hypothesis (does our over-read correlate
with the side having doubled pawns we over-value / under-penalize?). Runs V2+V3. Read-only.
Run: pyrun diagnostics/pawn_overread_decompose.py
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
CORPUS=os.path.join(THIS,'ks_sets','diverse_corpus_ksv2.csv')
def wpov(bd,k): return -bd.get(k,0.0)/1000.0
def sgn(x): return (x>1e-6)-(x<-1e-6)
def doubled_count(board, color):
    files=[0]*8
    for sq in board.pieces(chess.PAWN, color): files[chess.square_file(sq)]+=1
    return sum(c-1 for c in files if c>1)
rows=[r for r in csv.DictReader(open(CORPUS))]
sub=[]
for r in rows:
    b=chess.Board(r['fen']); bd=ai.ev_breakdown(b)
    tt=float(r['target_total']); ot=wpov(bd,'total'); ks=wpov(bd,'king_safety')
    sover=ot-tt
    if abs(ks)<0.3 and abs(ot)-abs(tt)>=1.5:
        sub.append((r,b,bd,tt,ot,sover))
print("PURE pawn/placement over-reads (|our_ks|<0.3 & over_mag>=1.5): %d rows"%len(sub))
agg={k:[] for k in ['pt_pawns','passed','imb','central','cap','pt_knights','pt_bishops','pt_rooks','space']}
dbl_align=0
for r,b,bd,tt,ot,sover in sub:
    for k,key in [('pt_pawns','pt_pawns'),('passed','passed_pawn_support'),('central','central'),('cap','capture_gains'),
                  ('pt_knights','pt_knights'),('pt_bishops','pt_bishops'),('pt_rooks','pt_rooks'),('space','space')]:
        agg[k].append(wpov(bd,key))
    agg['imb'].append(wpov(bd,'imbalance_white')+wpov(bd,'imbalance_black'))
    # doubled-pawn alignment: the side we over-favor (sign of sover; sover>0 => we over-favor White) has doubled pawns?
    overfav_white = sover>0
    dbl = doubled_count(b, chess.WHITE if overfav_white else chess.BLACK)
    if dbl>0: dbl_align+=1
print("--- mean term contributions (white-POV pawns) ---")
for k in ['pt_pawns','passed','imb','central','cap','pt_knights','pt_bishops','pt_rooks','space']:
    print("  %-10s mean=%+.2f  mean_abs=%.2f"%(k, st.mean(agg[k]), st.mean(abs(x) for x in agg[k])))
print("  rows where the OVER-FAVORED side has doubled pawns (we may under-penalize): %d/%d = %.0f%%"%(dbl_align,len(sub),100*dbl_align/len(sub)))
print("\n--- top pt_pawns-driven rows ---")
sub2=sorted(sub,key=lambda z:-abs(wpov(z[2],'pt_pawns')))[:12]
for r,b,bd,tt,ot,sover in sub2:
    print("  ptP=%+.2f over=%+.2f sf18=%+.2f wDbl=%d bDbl=%d  %s"%(
        wpov(bd,'pt_pawns'),sover,tt,doubled_count(b,chess.WHITE),doubled_count(b,chess.BLACK),r['fen']))
