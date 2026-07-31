"""Ground-truth trace of SF11's king() danger for a given position + defended king, using python-chess attack
tables (same tables our C++ uses). Prints each SF11 kingDanger component so we can see EXACTLY where SF's danger
comes from (hypothesis: SAFE CHECKS dominate; phantoms have none). Validates against SF11's real King-safety term.
Run: pyrun diagnostics/ks_sf_trace.py   (traces a built-in phantom+real FEN list; add FEN='...' COL=w|b for one)
"""
import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
for _a in sys.argv[1:]:
    if '=' in _a and not _a.startswith(('FEN=','COL=')): _k,_v=_a.split('=',1); os.environ[_k]=_v
THIS=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,THIS)
import chess
# SF11 constants (evaluate.cpp)
KAW={chess.KNIGHT:81,chess.BISHOP:52,chess.ROOK:44,chess.QUEEN:10}
SAFE={chess.ROOK:1080,chess.QUEEN:780,chess.BISHOP:635,chess.KNIGHT:790}  # queen counted only if not a rook-check
SF11PATH=os.environ.get('SF11_BIN',"/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/stockfish_11_linux/stockfish-11-linux/Linux/stockfish_20011801_x64_bmi2")
try:
    from eval_vs_sf11 import SF11Eval; SF11=SF11Eval(SF11PATH)
except Exception as e: SF11=None; print("(SF11 term n/a:",e,")")

def rook_from(ksq,occ):   # squares a rook on ksq attacks (= squares from which a rook checks ksq)
    return chess.BB_RANK_ATTACKS[ksq][occ & chess.BB_RANK_MASKS[ksq]] | chess.BB_FILE_ATTACKS[ksq][occ & chess.BB_FILE_MASKS[ksq]]
def bishop_from(ksq,occ):
    return chess.BB_DIAG_ATTACKS[ksq][occ & chess.BB_DIAG_MASKS[ksq]]

def trace(fen, us):
    b=chess.Board(fen); them=not us; ksq=b.king(us); occ=b.occupied
    ring=chess.BB_KING_ATTACKS[ksq]|chess.BB_SQUARES[ksq]
    # attackedBy[side][ALL]
    def att_all(side):
        m=0
        for sq in chess.scan_forward(b.occupied_co[side]): m|=b.attacks_mask(sq)
        return m
    aUs=att_all(us); aThem=att_all(them)
    # ring attackers (pieces of THEM attacking the ring) -> count + weight
    cnt=0; wt=0
    for pt in (chess.KNIGHT,chess.BISHOP,chess.ROOK,chess.QUEEN):
        for sq in chess.scan_forward(b.pieces_mask(pt,them)):
            if b.attacks_mask(sq)&ring: cnt+=1; wt+=KAW[pt]
    # weak = ring squares attacked by them, not defended (approx SF: no us-defender other than K/Q)
    weak=0
    for sq in chess.scan_forward(ring):
        if not (chess.BB_SQUARES[sq]&aThem): continue
        defenders=b.attackers(us,sq)
        heavy=any(b.piece_type_at(d) in (chess.KNIGHT,chess.BISHOP,chess.ROOK,chess.PAWN) for d in defenders)
        if len(defenders)<=1 and not heavy: weak+=1
    # SAFE CHECKS (the hypothesis): check-from squares undefended by us
    rf=rook_from(ksq,occ); bf=bishop_from(ksq,occ); nf=chess.BB_KNIGHT_ATTACKS[ksq]
    def enemy_attacks_sq(sq,ptset):  # an enemy piece of ptset attacks sq
        for d in b.attackers(them,sq):
            if b.piece_type_at(d) in ptset: return True
        return False
    def safe(sq): return not (chess.BB_SQUARES[sq]&aUs)   # undefended by us (baseline SF safe)
    checks={'R':0,'Q':0,'B':0,'N':0}
    for sq in chess.scan_forward(rf & ~b.occupied_co[them]):
        if enemy_attacks_sq(sq,{chess.ROOK,chess.QUEEN}) and safe(sq): checks['R']+=1
    for sq in chess.scan_forward(bf & ~b.occupied_co[them]):
        if enemy_attacks_sq(sq,{chess.BISHOP,chess.QUEEN}) and safe(sq): checks['B']+=1
    for sq in chess.scan_forward(nf & ~b.occupied_co[them]):
        if enemy_attacks_sq(sq,{chess.KNIGHT}) and safe(sq): checks['N']+=1
    scheck_units = checks['R']*SAFE[chess.ROOK]+checks['Q']*SAFE[chess.QUEEN]+checks['B']*SAFE[chess.BISHOP]+checks['N']*SAFE[chess.KNIGHT]
    kd = cnt*wt + 185*weak + scheck_units - 873*(0 if b.pieces_mask(chess.QUEEN,them) else 1) + 37
    penalty = (kd*kd/4096)/128.0 if kd>100 else 0.0   # ~pawns (SF mg pawn ~128)
    sf11ks = None
    if SF11:
        t,terms=SF11.eval(fen)
        sf11ks=next((terms[k] for k in terms if k.lower()=='king safety'),None)
    print("  king=%s ring_attackers=%d(wt%d) weak=%d  SAFE-CHECKS R%d Q%d B%d N%d (units=%d)  kingDanger~%d -> ~%.2fp | SF11 KS term=%s"%(
        chess.square_name(ksq),cnt,wt,weak,checks['R'],checks['Q'],checks['B'],checks['N'],scheck_units,kd,-penalty if us==chess.WHITE else penalty,
        ("%.2f"%sf11ks if sf11ks is not None else "n/a")))

FENS=[  # (label, fen, defended-king-color)
 ("PHANTOM Ke6 (our+4.86 SF-0.24)","8/1p6/4k3/1b2n3/p2q2P1/P3R2P/1P3Q2/3rN2K b - - 2 44", chess.BLACK),
 ("PHANTOM Kf7","1r1q1r2/3b1kp1/p2b4/1p1P3p/1PpRQ3/P3B1PP/5P2/4R1K1 b - - 0 28", chess.BLACK),
 ("PHANTOM Kg7","8/pp3pk1/7p/q1r5/1n2Q3/4P3/P4PPP/3R2K1 b - - 5 27", chess.BLACK),
 ("REAL attack (ks_real)","2k5/2rp4/1pP4p/6p1/5r2/4qN2/2Q1B1RP/5R1K b - - 0 36", chess.WHITE),
]
if os.environ.get('FEN'):
    trace(os.environ['FEN'], chess.WHITE if os.environ.get('COL','w')=='w' else chess.BLACK)
else:
    for lbl,fen,col in FENS:
        print(lbl); trace(fen,col)
if SF11: SF11.close()
