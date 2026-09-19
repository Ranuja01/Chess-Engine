# -*- coding: utf-8 -*-
"""Slice 2 PLACEMENT DETECTOR ORACLE: eval v2's per-side sub-term counts vs an independent python-chess
implementation, position by position. Exit 0 = zero mismatches AND every term fires somewhere (non-vacuous).

Terms (SF11 evaluate.cpp:291-361 unless a FORM knob selects another engine's definition): outpost knights / bishops,
reachable-outpost knights, minors behind a pawn, bad-bishop units, long-diagonal bishops, trapped-rook units, weak queens,
and OURS -- latent pawn pressure for bishops and rooks.

FORMS mirrored (read from the same env the engine latches):
  OUTPOST_V2_FORM  0 SF11 · 1 Ethereal (raw span, [outside][defended] cells, PACKED 8 bits/cell) · 2 SF15.1 (defended OR pawn in front)
  BADB_V2_FORM     0 SF11 N(1+blk) · 1 SF15.1 by file class (PACKED 12 bits/class) · 2 Weiss N·blk · 3 Ethereal rammed-only
  TRAPROOK_V2_FORM 0 SF11 step · 1 SF1.1 linear (raw units 180-16·mob, halved with castling rights)

★ INDEPENDENT BY CONSTRUCTION: backward, blocked, both enemy pawn spans, the mobility area and every attack set are rebuilt
here from python-chess primitives -- nothing is read from the engine except the probe's final counts.
☠️ Non-vacuity is checked PER TERM: a detector that never fires passes a mismatch test trivially.

USAGE (knobs latch at engine init; KEY=VAL args exported before load):
  pyrun diagnostics/_placement_detector_oracle.py EVAL_ARM=1 KS_V2_XRAY=1 [OUTPOST_V2_FORM=..] [BADB_V2_FORM=..] [TRAPROOK_V2_FORM=..] [N=3000]
Run every form at least once, and XRAY=0 at least once.
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


def env_on(name):
    return os.environ.get(name, "0") not in ("0", "", "false")


XRAY = env_on("KS_V2_XRAY")
EXCL_Q = env_on("MOB_V2_EXCL_QUEEN")
PIN = env_on("MOB_V2_PIN")   # trapped rook reuses mobility's per-rook count, which MOB_V2_PIN changes (2026-09-15)
EXCL_LOW = env_on("MOB_V2_EXCL_LOWRANK")
FORM_O = int(os.environ.get("OUTPOST_V2_FORM", "0") or 0)
FORM_B = int(os.environ.get("BADB_V2_FORM", "0") or 0)
FORM_T = int(os.environ.get("TRAPROOK_V2_FORM", "0") or 0)
print("[knobs] " + " ".join(a for a in sys.argv[1:] if "=" in a))
print("[oracle reads] XRAY=%d EXCL_QUEEN=%d EXCL_LOWRANK=%d PIN=%d OUTPOST_FORM=%d BADB_FORM=%d TRAPROOK_FORM=%d"
      % (XRAY, EXCL_Q, EXCL_LOW, PIN, FORM_O, FORM_B, FORM_T))

seed = chess.Board()
ai = ChessAI.ChessAI(None, None, seed, seed.turn)

TERMS = ["outpost_n", "outpost_b", "reach_n", "behind", "badb_units", "longdiag", "traprook_units", "weakq",
         "latent_b", "latent_r"]
CENTRE = [chess.D4, chess.E4, chess.D5, chess.E5]


def bits(bb):
    return bin(bb).count("1")


def diag(sq, occ):
    return chess.BB_DIAG_ATTACKS[sq][chess.BB_DIAG_MASKS[sq] & occ]


def orth(sq, occ):
    return (chess.BB_RANK_ATTACKS[sq][chess.BB_RANK_MASKS[sq] & occ]
            | chess.BB_FILE_ATTACKS[sq][chess.BB_FILE_MASKS[sq] & occ])


def pawn_attacks(board, color):
    a = 0
    for sq in chess.scan_forward(board.pawns & board.occupied_co[color]):
        a |= chess.BB_PAWN_ATTACKS[color][sq]
    return a


def ahead(sq, color, n=1):
    t = sq + 8 * n if color == chess.WHITE else sq - 8 * n
    return t if 0 <= t < 64 else None


def backward_and_blocked(board, color):
    """SF11 pawns.cpp definitions. blocked: an ENEMY pawn directly ahead. backward: no own pawn on an adjacent file at
    or behind this rank, and the stop square holds an enemy pawn or is attacked by one."""
    own = board.pawns & board.occupied_co[color]
    enemy = board.pawns & board.occupied_co[not color]
    eatt = pawn_attacks(board, not color)
    backward, blocked = set(), set()
    for sq in chess.scan_forward(own):
        f, r = chess.square_file(sq), chess.square_rank(sq)
        st = ahead(sq, color)
        if st is not None and (enemy >> st) & 1:
            blocked.add(sq)
        support = False
        for df in (-1, 1):
            nf = f + df
            if not 0 <= nf <= 7:
                continue
            for rr in range(8):
                behind_ok = rr <= r if color == chess.WHITE else rr >= r
                if behind_ok and (own >> chess.square(nf, rr)) & 1:
                    support = True
        contested = st is not None and ((enemy >> st) & 1 or (eatt >> st) & 1)
        if not support and contested:
            backward.add(sq)
    return backward, blocked


def mob_area(board, color):
    own = board.occupied_co[color]
    own_p = board.pawns & own
    e_att = pawn_attacks(board, not color)
    blocked = 0
    for sq in chess.scan_forward(own_p):
        st = ahead(sq, color)
        if st is not None and (board.occupied >> st) & 1:
            blocked |= 1 << sq
    excl = blocked | (board.kings & own) | e_att
    if EXCL_Q:
        excl |= board.queens & own
    if EXCL_LOW:
        low = (chess.BB_RANK_2 | chess.BB_RANK_3) if color == chess.WHITE else (chess.BB_RANK_7 | chess.BB_RANK_6)
        excl |= own_p & low
    return ~excl & chess.BB_ALL


def enemy_pawn_ahead_on_adjacent(board, sq, color):
    """Ethereal's raw span: any enemy pawn on an adjacent file strictly ahead of sq (from color's point of view)."""
    enemy = board.pawns & board.occupied_co[not color]
    f, r = chess.square_file(sq), chess.square_rank(sq)
    for df in (-1, 1):
        nf = f + df
        if not 0 <= nf <= 7:
            continue
        rng = range(r + 1, 8) if color == chess.WHITE else range(r - 1, -1, -1)
        for rr in rng:
            if (enemy >> chess.square(nf, rr)) & 1:
                return True
    return False


def king_blockers(board, color):
    """SF's blockers_for_king rebuilt from python-chess (either colour; snipers removed from the occupancy)."""
    k = board.king(color)
    them_bb = board.occupied_co[not color]
    orth_empty = chess.BB_RANK_ATTACKS[k][0] | chess.BB_FILE_ATTACKS[k][0]
    diag_empty = chess.BB_DIAG_ATTACKS[k][0]
    snipers = ((orth_empty & (board.rooks | board.queens)) | (diag_empty & (board.bishops | board.queens))) & them_bb
    occ = board.occupied ^ snipers
    blockers = 0
    for s in chess.scan_forward(snipers):
        between = chess.between(k, s) & occ
        if between and not (between & (between - 1)):
            blockers |= between
    return blockers


def reference(board, color):
    them = not color
    own = board.occupied_co[color]
    own_p = board.pawns & own
    enemy_p = board.pawns & board.occupied_co[them]
    res = dict.fromkeys(TERMS, 0)
    eth_n, eth_b, badb_cls = [0] * 4, [0] * 4, [0] * 4
    my_patt = pawn_attacks(board, color)

    # refined enemy pawn attack span (SF11/15.1)
    bwd, blk = backward_and_blocked(board, them)
    span = pawn_attacks(board, them)
    for sq in chess.scan_forward(enemy_p):
        if sq in bwd or sq in blk:
            continue
        f, r = chess.square_file(sq), chess.square_rank(sq)
        for df in (-1, 1):
            nf = f + df
            if not 0 <= nf <= 7:
                continue
            rng = range(r - 1, -1, -1) if them == chess.BLACK else range(r + 1, 8)
            for rr in rng:
                span |= 1 << chess.square(nf, rr)
    ranks = (chess.BB_RANK_4 | chess.BB_RANK_5 | chess.BB_RANK_6) if color == chess.WHITE \
        else (chess.BB_RANK_5 | chess.BB_RANK_4 | chess.BB_RANK_3)

    def pawn_in_front(sq):
        st = ahead(sq, color)
        return st is not None and (board.pawns >> st) & 1

    if FORM_O == 2:
        elig = 0
        for sq in chess.scan_forward(ranks):
            if ((my_patt >> sq) & 1) or pawn_in_front(sq):
                elig |= 1 << sq
        outposts = elig & ~span
    else:
        outposts = ranks & my_patt & ~span

    def eth_cell(sq):
        outside = 2 if chess.square_file(sq) in (0, 7) else 0
        return outside + (1 if (my_patt >> sq) & 1 else 0)

    def raw_safe(sq):
        return ((ranks >> sq) & 1) and not enemy_pawn_ahead_on_adjacent(board, sq, color)

    for sq in chess.scan_forward(board.knights & own):
        if FORM_O == 1:
            if raw_safe(sq):
                eth_n[eth_cell(sq)] += 1
        elif (outposts >> sq) & 1:
            res["outpost_n"] += 1
        elif outposts & chess.BB_KNIGHT_ATTACKS[sq] & ~own:
            res["reach_n"] += 1
        if pawn_in_front(sq):
            res["behind"] += 1

    blocked_any = 0
    for sq in chess.scan_forward(own_p):
        st = ahead(sq, color)
        if st is not None and (board.occupied >> st) & 1:
            blocked_any |= 1 << sq
    centre_files = chess.BB_FILE_C | chess.BB_FILE_D | chess.BB_FILE_E | chess.BB_FILE_F
    centre_blk = bits(blocked_any & centre_files)
    rammed = 0
    for sq in chess.scan_forward(own_p):
        st = ahead(sq, color)
        if st is not None and (enemy_p >> st) & 1:
            rammed |= 1 << sq
    for sq in chess.scan_forward(board.bishops & own):
        if FORM_O == 1:
            if raw_safe(sq):
                eth_b[eth_cell(sq)] += 1
        elif (outposts >> sq) & 1:
            res["outpost_b"] += 1
        if pawn_in_front(sq):
            res["behind"] += 1
        colour = chess.BB_DARK_SQUARES if (chess.BB_DARK_SQUARES >> sq) & 1 else chess.BB_LIGHT_SQUARES
        same = bits(own_p & colour)
        if FORM_B == 1:
            f = chess.square_file(sq)
            u = same * ((0 if (my_patt >> sq) & 1 else 1) + centre_blk)
            badb_cls[min(f, 7 - f)] += u
        elif FORM_B == 2:
            res["badb_units"] += same * centre_blk
        elif FORM_B == 3:
            res["badb_units"] += bits(own_p & colour & rammed)
        else:
            res["badb_units"] += same * (1 + centre_blk)
        seen = diag(sq, board.pawns)
        if sum(1 for c in CENTRE if (seen >> c) & 1) > 1:
            res["longdiag"] += 1

    if FORM_O == 1:
        res["outpost_n"] = sum(eth_n[k] << (8 * k) for k in range(4))
        res["outpost_b"] = sum(eth_b[k] << (8 * k) for k in range(4))
    if FORM_B == 1:
        res["badb_units"] = sum(badb_cls[k] << (12 * k) for k in range(4))

    king = board.king(color)
    if king is not None:
        area = mob_area(board, color)
        pinned = 0
        if PIN:
            kb = king_blockers(board, color)
            area &= ~kb
            pinned = kb & own
        occ = (board.occupied ^ board.queens ^ (board.rooks & own)) if XRAY else board.occupied
        kf = chess.square_file(king)
        krank = chess.square_rank(king)
        rank_mask = chess.BB_RANK_1 if color == chess.WHITE else chess.BB_RANK_8
        can_castle = (board.castling_rights & rank_mask) != 0
        for sq in chess.scan_forward(board.rooks & own):
            f = chess.square_file(sq)
            if not (own_p & chess.BB_FILES[f]):
                continue                                    # on our semi-open file
            if not ((f <= kf) if kf < 4 else (f >= kf)):
                continue                                    # edge side, file-symmetrised
            ra = orth(sq, occ)
            if (pinned >> sq) & 1:
                ra &= chess.ray(king, sq)
            mob = bits(ra & area)
            if FORM_T == 1:
                if mob > 6:
                    continue
                if krank != (0 if color == chess.WHITE else 7) and krank != chess.square_rank(sq):
                    continue
                edge_files = range(0, kf) if kf < 4 else range(kf + 1, 8)
                if any(not (own_p & chess.BB_FILES[x]) for x in edge_files):
                    continue                                # a half-open file of ours between king and edge
                v = 180 - 16 * mob
                res["traprook_units"] += v // 2 if can_castle else v
            else:
                if mob > 3:
                    continue
                res["traprook_units"] += 1 if can_castle else 2

    enemy = board.occupied_co[them]
    for q in chess.scan_forward(board.queens & own):
        snipers = [s for s in chess.scan_forward(board.rooks & enemy)
                   if chess.square_file(s) == chess.square_file(q) or chess.square_rank(s) == chess.square_rank(q)]
        snipers += [s for s in chess.scan_forward(board.bishops & enemy)
                    if abs(chess.square_file(s) - chess.square_file(q)) == abs(chess.square_rank(s) - chess.square_rank(q))]
        smask = 0
        for s in snipers:
            smask |= 1 << s
        occ2 = board.occupied ^ smask
        for s in snipers:
            if bits(chess.between(q, s) & occ2) == 1:
                res["weakq"] += 1
                break

    # OURS -- latent pawn pressure
    def attacks_enemy_pawn_from(y):
        return (chess.BB_PAWN_ATTACKS[color][y] & enemy_p) != 0

    for sq in chess.scan_forward(board.bishops & own):
        a = diag(sq, board.occupied)
        lat = diag(sq, board.occupied & ~(a & own)) & ~a
        res["latent_b"] += sum(1 for y in chess.scan_forward(lat & ~own) if attacks_enemy_pawn_from(y))
    for sq in chess.scan_forward(board.rooks & own):
        a = orth(sq, board.occupied)
        lat = orth(sq, board.occupied & ~(a & own & ~board.pawns)) & ~a
        res["latent_r"] += sum(1 for y in chess.scan_forward(lat & ~own) if attacks_enemy_pawn_from(y))
    return res


fens = []
with open(os.path.join(ENGINE, "diagnostics", "ks_sets", "game_regret_set.csv"), newline="") as f:
    fens = [r["fen"] for r in csv.DictReader(f)]
random.Random(20260914).shuffle(fens)
fens = fens[:N]
# Hand cases forcing each rare term: knight outposts (centre + rim, defended + not), trapped rook with/without castling
# rights and on the king's rank, queen pinned to a rook and a bishop, long diagonal, minors behind pawns, rammed pawns.
fens += [
    "r1bqkb1r/pp3ppp/2np1n2/1B2p3/4P3/2N5/PPP2PPP/R1BQK2R w KQkq - 0 7",
    "rnbq1rk1/ppp1bppp/4pn2/3pN3/2PP4/6P1/PP2PPBP/RNBQ1RK1 w - - 0 7",
    "4k3/8/8/8/8/8/5PPP/5K1R w - - 0 1",
    "4k3/8/8/8/8/8/5PPP/5K1R w K - 0 1",
    "4k3/8/8/1b6/8/3Q4/8/4K3 w - - 0 1",
    "4k3/4r3/8/8/4N3/8/4Q3/4K3 w - - 0 1",
    "rn2k2r/pp2bppp/2p5/3p4/3P4/2N1BN2/PP3PPP/R2QKB1R w KQkq - 0 1",
    "r2qkb1r/ppp2ppp/2n1pn2/3p4/2PP2b1/2N1PN2/PP3PPP/R1BQKB1R w KQkq - 0 6",
    "4k3/pp6/8/NP6/8/8/8/4K3 w - - 0 1",
    "4k3/8/2p5/2P1p3/4P3/3B4/8/4K3 w - - 0 1",
]

bad = 0
fires = dict.fromkeys(TERMS, 0)
for fen in fens:
    b = chess.Board(fen)
    got = ChessAI.placement_counts(b.pawns, b.knights, b.bishops, b.rooks, b.queens, b.kings,
                                   b.occupied_co[chess.WHITE], b.occupied_co[chess.BLACK], b.castling_rights)
    rw, rb = reference(b, chess.WHITE), reference(b, chess.BLACK)
    for t in TERMS:
        exp = (rw[t], rb[t])
        if got[t] != exp:
            bad += 1
            if bad <= 12:
                print("MISMATCH %-15s engine %s oracle %s  %s" % (t, got[t], exp, fen))
        if exp[0] or exp[1]:
            fires[t] += 1

n = len(fens)
print("\npositions=%d  term-checks=%d  mismatches=%d" % (n, n * len(TERMS), bad))
# reach_n cannot fire under OUTPOST_V2_FORM 1 (Ethereal has no reachable outpost), so it is exempt there.
exempt = {"reach_n"} if FORM_O == 1 else set()
vacuous = [t for t in TERMS if fires[t] == 0 and t not in exempt]
for t in TERMS:
    print("  %-15s fires in %5d positions (%.1f%%)" % (t, fires[t], 100.0 * fires[t] / n))
if vacuous:
    print("☠️ VACUOUS terms (never fired, so a 0-mismatch result says nothing about them): %s" % ", ".join(vacuous))
ok = bad == 0 and not vacuous
print("RESULT: %s" % ("PASS" if ok else "FAIL"))
sys.exit(0 if ok else 1)
