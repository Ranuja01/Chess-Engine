import os,sys
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','3')
os.environ['ENABLE_CAPG_PIN']='1'
os.environ['ENABLE_CAPG_TEMPO']=(sys.argv[1] if len(sys.argv)>1 else '0')
sys.path.insert(0,'.'); sys.path.insert(0,'diagnostics')
import chess
from ChessAI import ChessAI
ai=ChessAI(None,None,chess.Board(),True)
ORIG="1rbq1rk1/ppp2pb1/7p/2n1pnpP/4Q3/2NP1NP1/PPPB1PB1/2K1R2R w - - 2 15"
b=chess.Board(ORIG); pov=1.0
bd=ai.ev_breakdown(b)
print("P1-orig TEMPO=%s: material=%+.2f capg=%+.2f total=%+.2f (White-POV)  [SF18 +0.00]"%(
  os.environ['ENABLE_CAPG_TEMPO'],(-bd.get('material',0)/1000.0)*pov,(-bd.get('capture_gains',0)/1000.0)*pov,(-bd.get('total',0)/1000.0)*pov))
