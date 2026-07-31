"""R2 (B regime) probe: dump the netted king_safety term (white-POV pawns) + per-king KS units for the
labeled acceptance FENs, under whatever KS_* knobs are passed. Run once per (KNEE,CAP,DIVISOR) point via the
runner, e.g.:
  pyrun diagnostics/ks_bsweep_probe.py ENABLE_KS_CHECK_V2=1 KS_KNEE=80 KS_CAP=80 KS_DIVISOR=4
king_safety is white-POV (positive = white better). Targets are SF18: FEN-1 wants NEGATIVE (black winning),
old-3 modest POSITIVE (over-blow DOWN), old-4 stays comfortably POSITIVE (held, we beat SF11).
"""
import os, sys
os.environ.setdefault('ENABLE_KS_V2','1'); os.environ.setdefault('ENABLE_PASSER_V3','1')
os.environ.setdefault('KS_SAFE_CHECK_DEF','5'); os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
for _a in sys.argv[1:]:
    if '=' in _a: _k,_v=_a.split('=',1); os.environ[_k]=_v
THIS=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,os.path.dirname(THIS))
import chess
from ChessAI import ChessAI
ai=ChessAI(None,None,chess.Board(),True)
FENS=[
    ("FEN-1  (SF18 -4.58, want NEG)", "8/1p6/4k3/1b2n3/p2q2P1/P3R2P/1P3Q2/3rN2K b"),
    ("old-3  (SF18 +3.04, blowup DOWN)", "3r4/ppp3Q1/nq2k2p/7N/3r2P1/2N1B2P/PP3P2/5bK1 w"),
    ("old-4  (SF +8.69, HELD pos)", "2n2b1r/1pB1k3/1p4Qp/1N6/4P3/Pn2P3/1q2BP1P/5K2 w"),
]
print("knobs: V2=%s KNEE=%s CAP=%s DIVISOR=%s CHK(Q/R/B/N)=%s/%s/%s/%s MULTI=%s"%(
    os.environ.get('ENABLE_KS_CHECK_V2','0'),os.environ.get('KS_KNEE','12'),os.environ.get('KS_CAP','80'),
    os.environ.get('KS_DIVISOR','4'),os.environ.get('KS_CHK_QUEEN','14'),os.environ.get('KS_CHK_ROOK','14'),
    os.environ.get('KS_CHK_BISHOP','7'),os.environ.get('KS_CHK_KNIGHT','9'),os.environ.get('KS_CHK_MULTI','0')))
for label,fen in FENS:
    bd=ai.ev_breakdown(chess.Board(fen))
    print("  %-34s king_safety=%+6.2f  units w/b=%d/%d"%(
        label, -bd.get('king_safety',0)/1000.0, bd.get('det_ks_units_w',0), bd.get('det_ks_units_b',0)))
