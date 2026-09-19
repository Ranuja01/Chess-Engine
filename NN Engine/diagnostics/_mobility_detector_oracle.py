# -*- coding: utf-8 -*-
"""Slice 2 mobility DETECTOR ORACLE: eval v2's per-side area-filtered counts vs an independent python-chess
implementation, position by position. Exit 0 = zero mismatches AND the run was non-vacuous.

WHY: a detector bug and a scoring bug are indistinguishable from outside (both read as "the eval moved").
This checks the detector -- area, attack occupancy, per-type counts, raw table sums -- before any magnitude is
read. Same pattern as _pawn_detector_oracle.py.
☠️ Non-vacuity is checked, not assumed: the 09-13 draw verify first passed on MIRRORED positions where both
sides cancel. Rows where White's and Black's counts differ are counted and must be the large majority.

The reference below uses python-chess's own attack tables (BB_DIAG_ATTACKS / BB_FILE_ATTACKS / ...), never
the engine's attacks_mask, so the two implementations share no code.

USAGE (knobs latch at engine init; KEY=VAL args are exported before the engine loads):
  pyrun diagnostics/_mobility_detector_oracle.py EVAL_ARM=1 KS_V2_XRAY=1 [MOB_V2_EXCL_QUEEN=1] [MOB_V2_EXCL_LOWRANK=1] [MOB_V2_SAFE=1|2] [MOB_V2_TABLE=0-3] [MOB_V2_PIN=1] [N=3000]
Run at least XRAY=0 and XRAY=1, each EXCL knob once, and SAFE=1 and SAFE=2 (added 2026-09-15; the enemy's per-type
attack maps are rebuilt here from python-chess, never read from the engine).
"""
import os, sys, csv, random

N = 3000
for a in sys.argv[1:]:
    if "=" in a:
        k, v = a.split("=", 1)
        if k == "N":
            N = int(v)
        else:
            os.environ[k] = v

ENGINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ENGINE)
os.chdir(ENGINE)
import chess, ChessAI

XRAY = os.environ.get("KS_V2_XRAY", "0") not in ("0", "", "false")
EXCL_Q = os.environ.get("MOB_V2_EXCL_QUEEN", "0") not in ("0", "", "false")
EXCL_LOW = os.environ.get("MOB_V2_EXCL_LOWRANK", "0") not in ("0", "", "false")
SAFE = int(os.environ.get("MOB_V2_SAFE", "0") or 0)
TABLE = int(os.environ.get("MOB_V2_TABLE", "0") or 0)
PIN = os.environ.get("MOB_V2_PIN", "0") not in ("0", "", "false")
print("[knobs] " + " ".join(a for a in sys.argv[1:] if "=" in a))
print("[oracle reads] XRAY=%d EXCL_QUEEN=%d EXCL_LOWRANK=%d SAFE=%d TABLE=%d PIN=%d"
      % (XRAY, EXCL_Q, EXCL_LOW, SAFE, TABLE, PIN))

seed = chess.Board()
ai = ChessAI.ChessAI(None, None, seed, seed.turn)   # latches knobs + builds attack tables

# SF11 evaluate.cpp:93-107, re-typed from source independently of eval_v2.cpp.
MG = {
    chess.KNIGHT: [-62, -53, -12, -4, 3, 13, 22, 28, 33],
    chess.BISHOP: [-48, -20, 16, 26, 38, 51, 55, 63, 63, 68, 81, 81, 91, 98],
    chess.ROOK:   [-58, -27, -15, -10, -5, -2, 9, 16, 30, 29, 32, 38, 46, 48, 58],
    chess.QUEEN:  [-39, -21, 3, 3, 14, 22, 28, 41, 43, 48, 56, 60, 60, 66, 67, 70, 71, 73, 79, 88, 88, 99,
                   102, 102, 106, 109, 113, 116],
}
EG = {
    chess.KNIGHT: [-81, -56, -30, -14, 8, 15, 23, 27, 33],
    chess.BISHOP: [-59, -23, -3, 13, 24, 42, 54, 57, 65, 73, 78, 86, 88, 97],
    chess.ROOK:   [-76, -18, 28, 55, 69, 82, 112, 118, 132, 142, 155, 165, 166, 169, 171],
    chess.QUEEN:  [-36, -15, 8, 18, 34, 54, 61, 73, 79, 92, 94, 104, 113, 120, 123, 126, 133, 136, 140, 143,
                   148, 166, 170, 175, 184, 191, 206, 212],
}
TYPES = [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]

# MOB_V2_TABLE forms 1-3, from the 2026-09-15 reference fetch (SF15.1 evaluate.cpp:213-227 · Ethereal 0e47e9b · Weiss c735b8f).
# ⚠️ Same fetch as eval_v2.cpp's tables, so this checks the ENGINE's transcription and indexing, not the source digits.
_N, _B, _R, _Q = chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN
MG_FORMS = {0: MG, 1: {
    _N: [-62, -53, -12, -3, 3, 12, 21, 28, 37],
    _B: [-47, -20, 14, 29, 39, 53, 53, 60, 62, 69, 78, 83, 91, 96],
    _R: [-60, -24, 0, 3, 4, 14, 20, 30, 41, 41, 41, 45, 57, 58, 67],
    _Q: [-29, -16, -8, -8, 18, 25, 23, 37, 41, 54, 65, 68, 69, 70, 70, 70, 71, 72, 74, 76, 90, 104, 105, 106, 112, 114, 114, 119],
}, 2: {
    _N: [-104, -45, -22, -8, 6, 11, 19, 30, 43],
    _B: [-99, -46, -16, -4, 6, 14, 17, 19, 19, 27, 26, 52, 55, 83],
    _R: [-127, -56, -25, -12, -10, -12, -11, -4, 4, 9, 11, 19, 19, 37, 97],
    _Q: [-111, -253, -127, -46, -20, -9, -1, 2, 8, 10, 15, 17, 20, 23, 22, 21, 24, 16, 13, 18, 25, 38, 34, 28, 10, 7, -42, -23],
}, 3: {
    _N: [-44, -31, -10, 0, 13, 22, 32, 43, 54],
    _B: [-51, -26, -11, -3, 9, 21, 26, 32, 32, 35, 41, 57, 50, 100],
    _R: [-105, -15, -1, 5, 2, 6, 5, 12, 14, 19, 24, 23, 25, 36, 72],
    _Q: [-63, -97, -89, -17, 0, -8, -2, -2, 1, 5, 7, 9, 15, 15, 16, 18, 16, 16, 12, 13, 23, 22, 48, 58, 122, 135, 146, 125],
}}
EG_FORMS = {0: EG, 1: {
    _N: [-79, -57, -31, -17, 7, 13, 16, 21, 26],
    _B: [-59, -25, -8, 12, 21, 40, 56, 58, 65, 72, 78, 87, 88, 98],
    _R: [-82, -15, 17, 43, 72, 100, 102, 122, 133, 139, 153, 160, 165, 170, 175],
    _Q: [-49, -29, -8, 17, 39, 54, 59, 73, 76, 95, 95, 101, 124, 128, 132, 133, 136, 140, 147, 149, 153, 169, 171, 171, 178, 185, 187, 221],
}, 2: {
    _N: [-139, -114, -37, 3, 15, 34, 38, 37, 17],
    _B: [-186, -124, -54, -14, 1, 20, 35, 39, 49, 48, 48, 32, 47, 2],
    _R: [-148, -127, -85, -28, 2, 27, 42, 46, 52, 55, 64, 68, 73, 60, 15],
    _Q: [-273, -401, -228, -236, -173, -86, -35, -1, 8, 31, 37, 55, 46, 57, 58, 64, 62, 65, 63, 48, 30, 8, -12, -29, -44, -79, -30, -50],
}, 3: {
    _N: [-139, -7, 60, 86, 89, 102, 102, 101, 81],
    _B: [-81, -40, 19, 53, 65, 83, 98, 104, 114, 116, 115, 106, 115, 76],
    _R: [-146, 18, 82, 88, 121, 133, 144, 146, 152, 157, 164, 171, 177, 177, 154],
    _Q: [-48, -54, -107, -127, -52, 72, 142, 184, 215, 230, 243, 254, 255, 268, 279, 283, 294, 302, 313, 321, 314, 318, 298, 279, 221, 193, 166, 162],
}}
MG, EG = MG_FORMS[TABLE], EG_FORMS[TABLE]


def diag(sq, occ):
    return chess.BB_DIAG_ATTACKS[sq][chess.BB_DIAG_MASKS[sq] & occ]


def orth(sq, occ):
    return (chess.BB_RANK_ATTACKS[sq][chess.BB_RANK_MASKS[sq] & occ]
            | chess.BB_FILE_ATTACKS[sq][chess.BB_FILE_MASKS[sq] & occ])


def piece_attacks(b, color, pt, sq):
    """One piece's attack set under the engine's occupancy rule (x-ray: B through all queens, R through all queens + own rooks)."""
    occ = b.occupied
    own = b.occupied_co[color]
    if pt == chess.KNIGHT:
        return chess.BB_KNIGHT_ATTACKS[sq]
    if pt == chess.BISHOP:
        return diag(sq, (occ ^ b.queens) if XRAY else occ)
    if pt == chess.ROOK:
        return orth(sq, (occ ^ b.queens ^ (b.rooks & own)) if XRAY else occ)
    return diag(sq, occ) | orth(sq, occ)


def attacks_by_type(b, color):
    """Union of each piece type's attacks for one side -- the SideAttacks.by[] maps, rebuilt independently."""
    out = {}
    for pt in TYPES:
        m = 0
        for sq in chess.scan_forward(b.pieces_mask(pt, color)):
            m |= piece_attacks(b, color, pt, sq)
        out[pt] = m
    return out


def safe_excl(b, color):
    """MOB_V2_SAFE exclusion per type: enemy attacks by LOWER (1) or LOWER-OR-EQUAL (2) value pieces (pawns are already
    outside the area)."""
    if not SAFE:
        return {pt: 0 for pt in TYPES}
    e = attacks_by_type(b, not color)
    le = SAFE == 2
    minors = e[chess.KNIGHT] | e[chess.BISHOP]
    return {
        chess.KNIGHT: minors if le else 0,
        chess.BISHOP: minors if le else 0,
        chess.ROOK: minors | (e[chess.ROOK] if le else 0),
        chess.QUEEN: minors | e[chess.ROOK] | (e[chess.QUEEN] if le else 0),
    }


def king_blockers(b, color):
    """SF's blockers_for_king, rebuilt from python-chess: pieces of EITHER colour that are the single piece between this
    side's king and an enemy slider aligned on an empty board (snipers removed from the occupancy)."""
    k = b.king(color)
    them = b.occupied_co[not color]
    orth_empty = chess.BB_RANK_ATTACKS[k][0] | chess.BB_FILE_ATTACKS[k][0]
    diag_empty = chess.BB_DIAG_ATTACKS[k][0]
    snipers = ((orth_empty & (b.rooks | b.queens)) | (diag_empty & (b.bishops | b.queens))) & them
    occ = b.occupied ^ snipers
    blockers = 0
    for s in chess.scan_forward(snipers):
        between = chess.between(k, s) & occ
        if between and not (between & (between - 1)):
            blockers |= between
    return blockers


def reference(b, color):
    own = b.occupied_co[color]
    occ = b.occupied
    own_p = b.pawns & own
    enemy_p = b.pawns & ~own
    e_att = 0
    for sq in chess.scan_forward(enemy_p):
        e_att |= chess.BB_PAWN_ATTACKS[not color][sq]
    blocked = 0
    for sq in chess.scan_forward(own_p):
        ahead = sq + 8 if color == chess.WHITE else sq - 8
        if 0 <= ahead < 64 and (occ >> ahead) & 1:
            blocked |= 1 << sq
    excl = blocked | (b.kings & own) | e_att
    if EXCL_Q:
        excl |= b.queens & own
    if EXCL_LOW:
        low = (chess.BB_RANK_2 | chess.BB_RANK_3) if color == chess.WHITE else (chess.BB_RANK_7 | chess.BB_RANK_6)
        excl |= own_p & low
    area = ~excl & chess.BB_ALL
    pinned = 0
    if PIN:
        kb = king_blockers(b, color)
        area &= ~kb
        pinned = kb & own
    ksq = b.king(color)

    sx = safe_excl(b, color)
    cnt = []
    mg = eg = 0
    for pt in TYPES:
        c = 0
        for sq in chess.scan_forward(b.pieces_mask(pt, color)):
            a = piece_attacks(b, color, pt, sq)
            if (pinned >> sq) & 1:
                a &= chess.ray(ksq, sq)
            n = bin(a & area & ~sx[pt]).count("1")
            c += n
            mg += MG[pt][n]
            eg += EG[pt][n]
        cnt.append(c)
    return tuple(cnt), mg, eg, area


def probe(b):
    return ChessAI.mobility_counts(b.pawns, b.knights, b.bishops, b.rooks, b.queens, b.kings,
                                   b.occupied_co[chess.WHITE], b.occupied_co[chess.BLACK])


fens = []
with open(os.path.join(ENGINE, "diagnostics", "ks_sets", "game_regret_set.csv"), newline="") as f:
    fens = [r["fen"] for r in csv.DictReader(f)]
random.Random(20260914).shuffle(fens)
fens = fens[:N]
# Hand cases that force the edge conditions: batteries (x-ray), a blocked pawn, a queen behind a bishop,
# low-rank pawns, an asymmetric rook pair.
fens += [
    "4k3/8/8/8/8/2Q5/1B6/R3K2R w - - 0 1",
    "r3k3/8/8/3p4/3P4/8/PPP2PPP/R2QR1K1 w - - 0 1",
    "3qk3/8/8/8/8/8/1B6/Q3K3 w - - 0 1",
    "rn2k2r/pp3ppp/8/8/8/8/PPP2PPP/RN2K2R b - - 0 1",
    # SAFE edges: a rook's file crossed by an enemy knight's and bishop's attacks; queens facing an enemy rook and queen.
    "4k3/8/2n5/8/3R4/8/5b2/4K3 w - - 0 1",
    "3rk3/8/8/3q4/8/8/8/3QK2R w - - 0 1",
    # PIN edges: a bishop pinned on a file (counts 0); rooks pinned against each other on one file (both sides' blockers);
    # a knight pinned on a diagonal; an ENEMY piece as the lone blocker (its square leaves our area).
    "4k3/8/8/8/4r3/8/4B3/4K3 w - - 0 1",
    "4k3/4r3/8/8/8/8/4R3/4K3 w - - 0 1",
    "4k3/8/8/8/1b6/8/3N4/4K3 w - - 0 1",
    "4k3/8/8/8/8/2b5/3n4/4K2B b - - 0 1",
]

bad = 0
asym = 0
for fen in fens:
    b = chess.Board(fen)
    got = probe(b)
    for side, color in ((0, chess.WHITE), (1, chess.BLACK)):
        cnt, mg, eg, area = reference(b, color)
        g = (got["count"][side], got["raw_mg"][side], got["raw_eg"][side], got["area"][side])
        if g != (cnt, mg, eg, area):
            bad += 1
            if bad <= 10:
                print("MISMATCH %s side=%s\n  engine %s\n  oracle %s" % (fen, "WB"[side], g, (cnt, mg, eg, area)))
    if got["count"][0] != got["count"][1]:
        asym += 1

n = len(fens)
print("\npositions=%d  sides-checked=%d  mismatches=%d  asymmetric-count positions=%d (%.1f%%)"
      % (n, 2 * n, bad, asym, 100.0 * asym / n))
vacuous = asym < 0.5 * n
if vacuous:
    print("☠️ VACUOUS: fewer than half the positions have differing counts -- the comparison discriminates little")
print("RESULT: %s" % ("PASS" if bad == 0 and not vacuous else "FAIL"))
sys.exit(0 if bad == 0 and not vacuous else 1)
