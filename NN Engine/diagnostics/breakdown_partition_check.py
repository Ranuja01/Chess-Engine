"""Verify the ev_breakdown is a CLEAN PARTITION: sum the candidate disjoint members and compare to `total`.
`pieces` (=sum of pt_*) already contains `material`, so material/pt_* are SUB-VIEWS, not partition members.
Reports the residual (unattributed) per position -> if ~0 the partition is complete. Runs V2+V3.
Run: pyrun diagnostics/breakdown_partition_check.py
"""
import os, sys
os.environ.setdefault('ENABLE_KS_V2','1'); os.environ.setdefault('ENABLE_PASSER_V3','1')
os.environ.setdefault('KS_SAFE_CHECK_DEF','5'); os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
for _a in sys.argv[1:]:
    if '=' in _a: _k,_v=_a.split('=',1); os.environ[_k]=_v
THIS=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,os.path.dirname(THIS))
import chess, csv
from ChessAI import ChessAI
ai=ChessAI(None,None,chess.Board(),True)
# candidate disjoint partition members (post-loop terms + the monolithic `pieces`)
PART=['pieces','capture_gains','passed_pawn_support','latent_threat','threats','king_safety','central',
      'imbalance_white','imbalance_black','pair_bonus','piece_value_boost','kaufman_imbalance','pawn_majority',
      'pawn_struct','outpost','space','mobility','rook_cond']  # + AE delta (advanced_endgame_total - ae_input)
PT=['pt_pawns','pt_knights','pt_bishops','pt_rooks','pt_queens','pt_kings']
fens=["Q7/1p1k3p/p4qp1/2p5/b1P1pb2/2P4P/PB2BP2/R4RK1 w - - 0 1",
      "r5k1/5p2/2P4p/r2P1qp1/1R2Q3/p2P4/K1R4P/8 b - - 0 1"]
try:
    corp=os.path.join(THIS,'ks_sets','diverse_corpus.csv')
    fens+=[r['fen'] for r in list(csv.DictReader(open(corp)))[:8]]
except Exception: pass
maxres=0.0
for fen in fens:
    bd=ai.ev_breakdown(chess.Board(fen))
    tot=bd['total']
    ae_delta = (bd.get('advanced_endgame_total',0) - bd.get('ae_input',0)) if bd.get('advanced_endgame_fired') else 0
    psum=sum(bd.get(k,0) for k in PART) + ae_delta
    ptsum=sum(bd.get(k,0) for k in PT)
    res=tot-psum
    maxres=max(maxres,abs(res))
    print("total=%8.0f  partsum=%8.0f  residual=%7.0f  | pt_sum=%8.0f vs pieces=%8.0f (Δ=%.0f)  %s"%(
        tot,psum,res,ptsum,bd.get('pieces',0),ptsum-bd.get('pieces',0),fen[:34]))
print("\nMAX |residual| over %d positions = %.0f millipawns  -> %s"%(
    len(fens),maxres,"CLEAN PARTITION" if maxres<1 else "INCOMPLETE (missing/overlapping members)"))
