"""Dump KS sub-components (attsq/weak/safe/attpc/defpc/openf/units/danger per king) for one FEN via the
gated KS_DEBUG_DUMP path (fires under ev_breakdown). Run: pyrun diagnostics/ks_dump_one.py FEN='<fen>'
"""
import os, sys
os.environ.setdefault('ENABLE_KS_V2','1'); os.environ.setdefault('ENABLE_PASSER_V3','1')
os.environ['KS_DEBUG_DUMP']='1'; os.environ.setdefault('KS_SAFE_CHECK_DEF','5'); os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
FEN=None
for _a in sys.argv[1:]:
    if _a.startswith('FEN='): FEN=_a.split('=',1)[1]
    elif '=' in _a: _k,_v=_a.split('=',1); os.environ[_k]=_v
THIS=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,os.path.dirname(THIS))
import chess
from ChessAI import ChessAI
ai=ChessAI(None,None,chess.Board(),True)
b=chess.Board(FEN); bd=ai.ev_breakdown(b)
sys.stderr.flush()
print("king_safety(white-POV pawns) = %+.2f   det_ks_units w/b = %d/%d"%(
    -bd.get('king_safety',0)/1000.0, bd.get('det_ks_units_w',0), bd.get('det_ks_units_b',0)))
