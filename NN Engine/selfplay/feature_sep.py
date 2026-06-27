# -*- coding: utf-8 -*-
"""Which CANDIDATE positional features separate a LOST level-position from a healthy one?

Computes cheap features (space, attack span, king-zone pressure, mobility, density) directly from the
FENs of midgame LEVEL-material LOST positions (flip blindness points) vs HEALTHY level positions (SF |cp|
<50), in python (python-chess, no engine/build). A feature whose magnitude differs a lot between the two
groups is a candidate detector to BUILD; one that doesn't separate isn't worth adding. Mirrors level_sep.py
(which found placement-magnitude separates). Run: overnight_runner.sh pyrun selfplay/feature_sep.py [tag].
"""
import csv, json, os, glob, sys, random, statistics
import chess

THIS = os.path.dirname(os.path.abspath(__file__))
TAG = sys.argv[1] if len(sys.argv) > 1 else 'placement_bundle'
G = os.path.join(THIS, 'games', TAG)
WHITE_HALF = chess.BB_RANK_1 | chess.BB_RANK_2 | chess.BB_RANK_3 | chess.BB_RANK_4
BLACK_HALF = chess.BB_RANK_5 | chess.BB_RANK_6 | chess.BB_RANK_7 | chess.BB_RANK_8
VAL = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}


def pcount(fen):
    return sum(1 for ch in fen.split()[0] if ch.isalpha())


def feats(fen):
    b = chess.Board(fen)
    # pawn-controlled squares in the enemy half (classic cheap space)
    wpa = 0
    for sq in b.pieces(chess.PAWN, chess.WHITE):
        wpa |= chess.BB_PAWN_ATTACKS[chess.WHITE][sq]
    bpa = 0
    for sq in b.pieces(chess.PAWN, chess.BLACK):
        bpa |= chess.BB_PAWN_ATTACKS[chess.BLACK][sq]
    pawn_space = chess.popcount(wpa & BLACK_HALF) - chess.popcount(bpa & WHITE_HALF)
    # all-piece attack span (mobility/space by every piece)
    watt = batt = 0
    for sq in chess.scan_forward(b.occupied_co[chess.WHITE]):
        watt |= int(b.attacks(sq))
    for sq in chess.scan_forward(b.occupied_co[chess.BLACK]):
        batt |= int(b.attacks(sq))
    attack_span = chess.popcount(watt) - chess.popcount(batt)
    # king-zone pressure: enemy attacks on the 1-ring around each king
    wk, bk = b.king(chess.WHITE), b.king(chess.BLACK)
    wk_zone = chess.BB_KING_ATTACKS[wk] | (1 << wk)
    bk_zone = chess.BB_KING_ATTACKS[bk] | (1 << bk)
    press_on_white_king = chess.popcount(batt & wk_zone)   # black attacking white king
    press_on_black_king = chess.popcount(watt & bk_zone)   # white attacking black king
    king_pressure = press_on_black_king - press_on_white_king
    # density + mobility
    npieces = pcount(fen)
    mob = b.legal_moves.count()
    return dict(pawn_space=pawn_space, attack_span=attack_span, king_pressure=king_pressure,
                press_max=max(press_on_white_king, press_on_black_king), npieces=npieces, mobility=mob)


flip_keys = set()
fp = os.path.join(G, 'flips_fast.csv')
if os.path.exists(fp):
    for r in csv.DictReader(open(fp)):
        try:
            if int(r['pieces']) > 12:
                flip_keys.add((r['game'], int(float(r['ply_start']))))
        except Exception:
            pass

lost, healthy = [], []
for d in sorted(glob.glob(os.path.join(G, 'game_*'))):
    jf = os.path.join(d, 'game.annotated.jsonl')
    if not os.path.exists(jf):
        continue
    gname = os.path.basename(d)
    for ln in open(jf):
        try:
            o = json.loads(ln)
        except Exception:
            continue
        if o.get('type') != 'move':
            continue
        eb = o.get('eval_breakdown'); fen = o.get('fen'); ply = o.get('ply'); sf = o.get('sf_cp')
        if not eb or not fen or pcount(fen) <= 12 or abs(eb.get('material', 0)) > 1000:
            continue
        if (gname, ply) in flip_keys:
            lost.append(fen)
        elif sf is not None and abs(sf) < 50:
            healthy.append(fen)

random.seed(0)
if lost and len(healthy) > 5 * len(lost):
    healthy = random.sample(healthy, 5 * len(lost))
print(f'lost-level n={len(lost)}   healthy-level n={len(healthy)}')
if not lost or not healthy:
    print('insufficient data'); raise SystemExit

KEYS = ['pawn_space', 'attack_span', 'king_pressure', 'press_max', 'npieces', 'mobility']
Lf = [feats(f) for f in lost]
Hf = [feats(f) for f in healthy]
print(f'{"feature":14}{"lost|mean|":>12}{"heal|mean|":>12}{"|.|diff":>9}{"lostMean":>10}{"healMean":>10}{"sep(SD)":>9}')
rows = []
for k in KEYS:
    lv = [abs(x[k]) for x in Lf]; hv = [abs(x[k]) for x in Hf]
    lvm, hvm = statistics.mean(lv), statistics.mean(hv)
    ls, hs = statistics.mean(x[k] for x in Lf), statistics.mean(x[k] for x in Hf)
    sd = statistics.pstdev([x[k] for x in Hf]) or 1
    sep = (lvm - hvm) / sd                      # separation in healthy-SD units
    rows.append((abs(sep), k, lvm, hvm, ls, hs, sep))
for _, k, lvm, hvm, ls, hs, sep in sorted(rows, reverse=True):
    print(f'{k:14}{lvm:12.1f}{hvm:12.1f}{lvm-hvm:+9.1f}{ls:+10.1f}{hs:+10.1f}{sep:+9.2f}')
print('\n(large |.|diff / |sep| = the feature distinguishes lost from healthy = worth BUILDING as a detector)')
