"""Classic dissection of the pawn/positional FENs 1 & 2: our full breakdown (V2+V3 on) vs SF11-static vs
SF18-search, term by term (white-POV pawns). FEN-2 is a suspected SIGN error. Run: pyrun diagnostics/fen12_dissect.py
"""
import os, sys
os.environ.setdefault('ENABLE_KS_V2','1'); os.environ.setdefault('ENABLE_PASSER_V3','1')
os.environ.setdefault('KS_SAFE_CHECK_DEF','5'); os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
for _a in sys.argv[1:]:
    if '=' in _a: _k,_v=_a.split('=',1); os.environ[_k]=_v
THIS=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,os.path.dirname(THIS)); sys.path.insert(0,THIS)
import chess
from ChessAI import ChessAI
ai=ChessAI(None,None,chess.Board(),True)
def wpov(bd,k): return -bd.get(k,0.0)/1000.0

FENS={
 "FEN-1 (correct dir, over-magnified)": "Q7/1p1k3p/p4qp1/2p5/b1P1pb2/2P4P/PB2BP2/R4RK1 w - - 0 1",
 "FEN-2 (SIGN error: White winning, we say Black)": "r5k1/5p2/2P4p/r2P1qp1/1R2Q3/p2P4/K1R4P/8 b - - 0 1",
}
TERMS=['total','pieces','pt_pawns','pt_knights','pt_bishops','pt_rooks','pt_queens','pt_kings',
       '(material sub-view)','king_safety','imbalance_white','imbalance_black','kaufman_imbalance','piece_value_boost',
       'pair_bonus','capture_gains','passed_pawn_support','central','pawn_struct','outpost','space','mobility',
       'rook_cond','pawn_majority','latent_threat','threats','(AE delta)']

# optional SF references
sf11=None
try:
    from eval_vs_sf11 import SF11Eval; sf11=SF11Eval()
except Exception as e:
    print("(SF11 static unavailable:", e, ")")
sf_path=None
try:
    from selfplay.arbiter import find_stockfish; sf_path=find_stockfish()
except Exception as e:
    try:
        from arbiter import find_stockfish; sf_path=find_stockfish()
    except Exception as e2:
        print("(SF18 search unavailable:", e2, ")")

def sf18(fen, depth=18):
    if not sf_path: return None
    import chess.engine
    eng=chess.engine.SimpleEngine.popen_uci(sf_path)
    try:
        info=eng.analyse(chess.Board(fen), chess.engine.Limit(depth=depth))
        sc=info['score'].white()
        return sc.score(mate_score=99000)/100.0 if sc.score(mate_score=99000) is not None else None
    finally:
        eng.quit()

for name,fen in FENS.items():
    b=chess.Board(fen); bd=ai.ev_breakdown(b)
    print("\n==================", name)
    print(fen, " | stm:", "white" if b.turn else "black")
    ot=wpov(bd,'total')
    s11 = sf11.eval_white(fen)/100.0 if sf11 else None
    s18 = sf18(fen)
    print("  OURS(total, white-POV) = %+.2f   SF11-static = %s   SF18-search = %s"
          % (ot, ("%+.2f"%s11 if s11 is not None else "n/a"), ("%+.2f"%s18 if s18 is not None else "n/a")))
    print("  --- our term breakdown (white-POV pawns; clean partition) ---")
    for t in TERMS:
        if t=='(material sub-view)': v=-bd.get('material',0)/1000.0
        elif t=='(AE delta)': v=(-(bd.get('advanced_endgame_total',0)-bd.get('ae_input',0))/1000.0) if bd.get('advanced_endgame_fired') else 0.0
        else: v=wpov(bd,t)
        if abs(v)>=0.01 or t in ('total','king_safety','pt_pawns'):
            print("    %-22s %+.2f"%(t,v))
    print("    det_ks_units_w/b = %.1f / %.1f  |  det offense w/b = %.1f/%.1f  defense w/b = %.1f/%.1f"%(
        bd.get('det_ks_units_w',0),bd.get('det_ks_units_b',0),
        bd.get('det_w_offense',0)/1000.0,bd.get('det_b_offense',0)/1000.0,
        bd.get('det_w_defense',0)/1000.0,bd.get('det_b_defense',0)/1000.0))
