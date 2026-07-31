import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','3')
os.environ['ENABLE_CAPG_PIN']='1'
import sys; sys.path.insert(0,'.'); sys.path.insert(0,'diagnostics')
import chess
from ChessAI import ChessAI
ai=ChessAI(None,None,chess.Board(),True)
fen="rn2k2r/4bppp/2p5/1pQn4/6P1/P4N2/P2BR2P/1K6 b kq - 0 25"
b=chess.Board(fen); pov=-1.0
bd=ai.ev_breakdown(b)
print("POS1 pin ON: material=%.2f capture_gains=%.2f total=%.2f" % (
 (-bd.get('material',0)/1000.0)*pov,(-bd.get('capture_gains',0)/1000.0)*pov,(-bd.get('total',0)/1000.0)*pov))
