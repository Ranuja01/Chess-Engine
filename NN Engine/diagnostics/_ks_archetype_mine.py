# -*- coding: utf-8 -*-
"""Mine + tentatively classify KS archetypes for the archetype bench.
DANGER cases (KS should be HIGH) from our real-attack corpora (hurt + collapse).
COUNTER cases (KS should be LOW) from the general regret set (the calm positions the empirical sample lacked).
For each FEN: per-king structural features (python-chess) + SF11 King-safety (the danger reference) + our KS.
Assigns a tentative archetype; emits a rich table to hand-curate into ks_sets/ks_archetypes.csv.

Run: bash <runner> pyrun diagnostics/_ks_archetype_mine.py > diagnostics/_ks_arch_candidates.tsv
"""
import os, sys, signal, csv, random
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
import chess
THIS_DIR = os.path.dirname(os.path.abspath(__file__)); ENGINE_DIR = os.path.dirname(THIS_DIR)
sys.path.insert(0, ENGINE_DIR); sys.path.insert(0, THIS_DIR)
from eval_vs_sf11 import SF11Eval, SF11
class _TO(Exception): pass
signal.signal(signal.SIGALRM, lambda s, f: (_ for _ in ()).throw(_TO()))

def kfeat(board, color):
    """Structural king-danger features for `color`'s king (attacked BY the enemy)."""
    ksq = board.king(color)
    if ksq is None: return None
    enemy = not color
    ring = list(chess.SquareSet(chess.BB_KING_ATTACKS[ksq])) + [ksq]
    att = set(); weak = 0
    for sq in ring:
        a = board.attackers(enemy, sq)
        if a:
            att |= set(a)
            d = [x for x in board.attackers(color, sq) if x != ksq]   # non-king defenders
            if not d: weak += 1
    att_types = {board.piece_type_at(p) for p in att}
    enemy_q = bool(board.pieces(chess.QUEEN, enemy))
    kf, kr = chess.square_file(ksq), chess.square_rank(ksq)
    fwd = 1 if color == chess.WHITE else -1
    shelter = open_f = storm = 0
    own_pawns = board.pieces(chess.PAWN, color); enemy_pawns = board.pieces(chess.PAWN, enemy)
    for df in (-1, 0, 1):
        f = kf + df
        if not (0 <= f <= 7): continue
        if not any(chess.square_file(p) == f for p in own_pawns): open_f += 1
        for dr in (1, 2):
            r = kr + fwd * dr
            if 0 <= r <= 7 and board.piece_at(chess.square(f, r)) == chess.Piece(chess.PAWN, color): shelter += 1
        for p in enemy_pawns:
            if chess.square_file(p) == f:
                adv = (6 - chess.square_rank(p)) if color == chess.WHITE else (chess.square_rank(p) - 1)
                if adv >= 3: storm += 1
    back = (chess.WHITE and kr <= 1) or (color == chess.BLACK and kr >= 6)
    castled = kf in (6, 7, 1, 2) and (kr == 0 if color == chess.WHITE else kr == 7)
    uncastled = board.has_castling_rights(color) or (kf in (3, 4) and (kr <= 1 if color == chess.WHITE else kr >= 6))
    return dict(att=len(att), types=att_types, eq=enemy_q, weak=weak, shelter=shelter,
                openf=open_f, storm=storm, castled=castled, uncastled=uncastled, queen_att=(chess.QUEEN in att_types))

def classify(f, sf_ks_subj):
    """f = subject-king features; sf_ks_subj = |SF KS| magnitude attributed to this king (pawns)."""
    danger = sf_ks_subj >= 0.8
    calm = sf_ks_subj <= 0.30
    if danger:
        if f['att'] >= 2 and (f['weak'] >= 1 or f['queen_att']): base = 'A1_coord'
        elif f['openf'] >= 1 or f['storm'] >= 1: base = 'A2_openstorm'
        elif f['uncastled']: base = 'A3_uncastled'
        elif f['weak'] >= 1: base = 'A5_weak'
        else: base = 'A4_other'
        return base
    if calm and f['att'] >= 1:
        if not f['eq']: return 'B2_queenless'
        if f['att'] >= 2 and f['weak'] == 0: return 'B1_defended_crowd'
        if f['shelter'] >= 2: return 'B3_shelter'
        return 'B4_calm'
    return None   # ambiguous middle — skip

def load(path, col='fen_start', n=None):
    p = os.path.join(ENGINE_DIR, path); out = []
    if not os.path.exists(p): return out
    rows = list(csv.DictReader(open(p)))
    cc = col if (rows and col in rows[0]) else (list(rows[0].keys())[0] if rows else col)
    for r in rows:
        fen = (r.get(cc) or '').strip()
        if fen: out.append(fen)
    if n and len(out) > n:
        random.Random(1234).shuffle(out); out = out[:n]
    return out

danger_src = load('diagnostics/_sprt_hurt.csv') + load('selfplay/games/vssf_2400/dp_fens.csv')
counter_src = load('diagnostics/ks_sets/game_regret_set.csv', col='fen', n=500)
sf11 = SF11Eval(SF11)

print("fen\tsubj\tarchetype\texp\tsf_ks\tatt\teq\tweak\topenf\tstorm\tshelter\tcastled")
seen = set()
for fen in danger_src + counter_src:
    if fen in seen: continue
    seen.add(fen)
    try:
        b = chess.Board(fen)
    except Exception:
        continue
    if b.is_check(): continue
    signal.alarm(8)
    try:
        _, terms = sf11.eval(fen); signal.alarm(0)
    except _TO:
        continue
    finally:
        signal.alarm(0)
    if terms is None: continue
    sf_ks = terms.get('King safety', 0.0)     # White-POV pawns; <0 => White king danger, >0 => Black king danger
    # subject = the endangered king per the SF-KS sign (fall back to the more-attacked king if ~0)
    if sf_ks <= -0.3: subj = chess.WHITE
    elif sf_ks >= 0.3: subj = chess.BLACK
    else:
        fw, fb = kfeat(b, chess.WHITE), kfeat(b, chess.BLACK)
        subj = chess.WHITE if (fw and fb and fw['att'] >= fb['att']) else chess.BLACK
    f = kfeat(b, subj)
    if not f or f['att'] == 0: continue
    arch = classify(f, abs(sf_ks))
    if not arch: continue
    exp = 'HIGH' if arch[0] == 'A' else 'LOW'
    print("%s\t%s\t%s\t%s\t%+.2f\t%d\t%d\t%d\t%d\t%d\t%d\t%d" % (
        fen, 'W' if subj == chess.WHITE else 'B', arch, exp, sf_ks,
        f['att'], int(f['eq']), f['weak'], f['openf'], f['storm'], f['shelter'], int(f['castled'])))
sf11.close()
