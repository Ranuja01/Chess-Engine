# -*- coding: utf-8 -*-
"""ORACLE for the C3 king detectors (eval_v2.cpp; v2_features columns 106-183):
  C3-a king shelter + pawn storm (ksb_cells, 106-161) · C3-b pawnless flank + king-pawn distance (kfl_cells, 162-171) ·
  C3-c KingProtector, minors (kprot_counts, 172-183).

WHY (2026-09-27). A detector bug and a scoring bug are indistinguishable from outside, so every v2 detector is checked
cell-for-cell against an INDEPENDENT implementation before any value is fitted on top of it. These are written the
slow way -- per square with python-chess, `board.attackers` for the lever test, `square_distance` for Chebyshev --
deliberately unlike the engine's masked bit-scans, so a shared mistake is unlikely.

Checks, in order (all no-engine: the cells read only constexpr masks and shifts, and v2_feature_counts self-initialises
its tables, so no ChessAI instance is built -- safe while games hold the box):
  1. HAND rows: positions whose cells were derived by hand.
  2. ORACLE: engine cells == python cells, per side, per block, on every corpus position.
  3. COLOUR MIRROR: White's cells on board.mirror() == Black's cells on the original (and vice versa).
  4. FILE MIRROR: cells are unchanged under a left-right flip.
  5. SUPPORT: how many positions fire each cell -- a cell no position fires cannot be fitted.

  pyrun diagnostics/_c3_oracle.py [SETS=ks_sets/game_regret_set.csv] [FENS=../selfplay/openings_variant.txt,...] [N=0]
"""
import os, sys, csv, collections

for _a in sys.argv[1:]:
    if "=" in _a:
        _k, _v = _a.split("=", 1)
        os.environ[_k] = _v
THIS = os.path.dirname(os.path.abspath(__file__))
ENGINE = os.path.dirname(THIS)
sys.path.insert(0, ENGINE)
import chess
import ChessAI

SETS = [s for s in os.environ.get("SETS", "ks_sets/game_regret_set.csv").split(",") if s]
FENS = [s for s in os.environ.get("FENS", "../selfplay/openings_variant.txt,../selfplay/openings_odds.txt").split(",") if s]
N = int(os.environ.get("N", "0") or 0)
BLOCKS = [("shelter/storm", 106, 56), ("flank/kdist", 162, 10), ("kprotector", 172, 12)]
FLANK = {0: (0, 1, 2), 1: (0, 1, 2, 3), 2: (0, 1, 2, 3), 3: (2, 3, 4, 5), 4: (2, 3, 4, 5), 5: (4, 5, 6, 7),
         6: (4, 5, 6, 7), 7: (5, 6, 7)}                    # SF11 KingFlank, by the king's file


def rr(sq, white):
    return chess.square_rank(sq) if white else 7 - chess.square_rank(sq)


def py_ksb(b, color):
    out = collections.Counter()
    k = b.king(color)
    if k is None:
        return out
    white = color == chess.WHITE
    kf, kr = chess.square_file(k), rr(k, white)
    centre = min(max(kf, 1), 6)
    for f in range(centre - 1, centre + 2):
        F = min(f, 7 - f)
        own = [s for s in b.pieces(chess.PAWN, color) if chess.square_file(s) == f and rr(s, white) >= kr]
        th = [s for s in b.pieces(chess.PAWN, not color) if chess.square_file(s) == f and rr(s, white) >= kr]
        orr = None
        if own:
            s = min(own, key=lambda q: rr(q, white))
            orr = rr(s, white)
            attacked = bool(b.attackers(not color, s) & b.pieces(chess.PAWN, not color))
            out[F * 6 + (5 if attacked else min(orr - kr, 4))] += 1
        if th:
            s = min(th, key=lambda q: rr(q, white))
            trr = rr(s, white)
            d = trr - kr
            if orr is not None and orr == trr - 1:
                st = 5 if d <= 2 else (6 if d == 3 else 7)
            else:
                st = 0 if d <= 1 else (4 if d >= 5 else d - 1)
            out[24 + F * 8 + st] += 1
    return out


def py_kfl(b, color):
    out = collections.Counter()
    k = b.king(color)
    if k is None:
        return out
    for e, side in enumerate((color, not color)):
        ps = list(b.pieces(chess.PAWN, side))
        if ps:
            d = min(chess.square_distance(k, s) for s in ps)
            if d >= 2:
                out[e * 4 + min(d, 5) - 2] += 1
    files = FLANK[chess.square_file(k)]
    on_flank = [s for s in b.pieces(chess.PAWN, chess.WHITE) | b.pieces(chess.PAWN, chess.BLACK)
                if chess.square_file(s) in files]
    if not on_flank:
        out[8] += 1
    elif not any(b.color_at(s) == color for s in on_flank):
        out[9] += 1
    return out


def py_kprot(b, color):
    out = collections.Counter()
    k = b.king(color)
    if k is None:
        return out
    for t, pt in enumerate((chess.KNIGHT, chess.BISHOP)):
        for s in b.pieces(pt, color):
            out[t * 6 + min(chess.square_distance(k, s), 6) - 1] += 1
    return out


PY = [py_ksb, py_kfl, py_kprot]


def eng_blocks(b):
    """-> [(white Counter, black Counter)] per block, cell index relative to the block."""
    wc, bc, _ = ChessAI.v2_feature_counts(b)
    res = []
    for _, k0, n in BLOCKS:
        to = lambda v: collections.Counter({i: v[k0 + i] for i in range(n) if v[k0 + i]})
        res.append((to(wc), to(bc)))
    return res


# ---- 1. hand rows (White's king) ---------------------------------------------------------------------------------
HAND = [
    # start: king e1 -> files d,e,f (F 3,3,2); own pawns d=1 (state 1); their pawns rank 7, d=6 unblocked (state 4)
    (0, "start", chess.STARTING_FEN, {19: 2, 13: 1, 52: 2, 44: 1}),
    # White Kg1, pawns f2 g2 h2; Black pawn h4 storms: h-file d=3 unblocked -> 24+0+2
    (0, "storm h4", "rnbq1rk1/ppppppp1/8/8/7p/8/PPPPPPPP/RNBQ1RK1 w - - 0 1", {13: 1, 7: 1, 1: 1, 44: 1, 36: 1, 26: 1}),
    # White Kg1, pawns f2 g2 h3; Black h4 is RAMMED by h3 (d=3 blocked -> 24+0+6); h3 is d=2 (state 2)
    (0, "blocked h4", "6k1/5pp1/8/8/7p/7P/5PP1/6K1 w - - 0 1", {13: 1, 7: 1, 2: 1, 44: 1, 36: 1, 30: 1}),
    # start: nearest own pawn d=1 (reference, no cell); nearest enemy pawn d=6 -> enemy >=5 (cell 7); flank c-f has own pawns
    (1, "start", chess.STARTING_FEN, {7: 1}),
    # Kg1, own pawn a2 only (d=6 -> own >=5, cell 3); no pawn on e-h -> flank empty (cell 8)
    (1, "pawnless", "6k1/8/8/8/8/8/P7/6K1 w - - 0 1", {3: 1, 8: 1}),
    # as above plus a Black pawn h2 (d=1, reference): the flank holds ONLY enemy pawns (cell 9)
    (1, "only enemy", "6k1/8/8/8/8/8/P6p/6K1 w - - 0 1", {3: 1, 9: 1}),
    # start: Nb1 d3 (cell 2), Ng1 d2 (cell 1), Bc1 d2 (cell 7), Bf1 d1 (cell 6)
    (2, "start", chess.STARTING_FEN, {1: 1, 2: 1, 6: 1, 7: 1}),
    # Ke1, knights e2 (d1) and a8 (d7 -> 6+), bishop h8 (d7 -> 6+)
    (2, "far minors", "N6B/8/8/8/8/8/4N3/4K2k w - - 0 1", {0: 1, 5: 1, 11: 1}),
]
print("== 1. hand rows (White's king) ==")
hand_bad = 0
for bi, name, fen, want in HAND:
    b = chess.Board(fen)
    e, p = eng_blocks(b)[bi][0], PY[bi](b, chess.WHITE)
    ok_e, ok_p = dict(e) == want, dict(p) == want
    hand_bad += (not ok_e) + (not ok_p)
    print("  %-13s %-11s engine %-4s python %-4s  want %s%s" % (BLOCKS[bi][0], name, "OK" if ok_e else "BAD",
          "OK" if ok_p else "BAD", sorted(want.items()),
          "" if ok_e and ok_p else ("   got engine %s python %s" % (sorted(e.items()), sorted(p.items())))))

# ---- load the corpus ---------------------------------------------------------------------------------------------
fens = []
for s in SETS:
    with open(os.path.join(THIS, s), newline="") as f:
        fens += [r["fen"] for r in csv.DictReader(f) if r.get("fen")]
for s in FENS:
    with open(os.path.join(THIS, s)) as f:
        for line in f:
            line = line.split(";")[0].split("|")[0].strip()
            if line and not line.startswith("#") and line.count("/") == 7:
                fens.append(line)
if N:
    fens = fens[:N]

# ---- 2-5 ---------------------------------------------------------------------------------------------------------
n = 0
mism, cmirror, fmirror = [0] * 3, [0] * 3, [0] * 3
support = [collections.Counter() for _ in BLOCKS]
first = []
for fen in fens:
    try:
        b = chess.Board(fen)
    except ValueError:
        continue
    if b.king(chess.WHITE) is None or b.king(chess.BLACK) is None:
        continue
    n += 1
    eb = eng_blocks(b)
    mb = eng_blocks(b.mirror())
    hb = eng_blocks(b.transform(chess.flip_horizontal))
    for bi in range(len(BLOCKS)):
        ew, ebk = eb[bi]
        pw, pb = PY[bi](b, chess.WHITE), PY[bi](b, chess.BLACK)
        if ew != pw or ebk != pb:
            mism[bi] += 1
            if len(first) < 6:
                first.append((BLOCKS[bi][0], fen, sorted(ew.items()), sorted(pw.items()), sorted(ebk.items()),
                              sorted(pb.items())))
        cmirror[bi] += (mb[bi][0] != ebk) or (mb[bi][1] != ew)
        fmirror[bi] += (hb[bi][0] != ew) or (hb[bi][1] != ebk)
        for c in list(ew) + list(ebk):
            support[bi][c] += 1

print("\n== 2-4. %d positions (%s) ==" % (n, ", ".join(SETS + FENS)))
print("  %-14s %10s %10s %10s" % ("block", "oracle", "colour", "file"))
for bi, (name, _, _) in enumerate(BLOCKS):
    print("  %-14s %10d %10d %10d" % (name, mism[bi], cmirror[bi], fmirror[bi]))
for row in first:
    print("   [%s] %s\n      W engine %s python %s\n      B engine %s python %s" % row)

print("\n== 5. support: positions (either king) firing each cell ==")
SH = ["beside", "d1", "d2", "d3", "d>=4", "lever-att"]
ST = ["unbl<=1", "unbl2", "unbl3", "unbl4", "unbl>=5", "blk<=2", "blk3", "blk>=4"]
FC = ["a/h", "b/g", "c/f", "d/e"]
s0 = support[0]
print("  shelter   " + "".join("%10s" % s for s in SH))
for F in range(4):
    print("  %-8s  " % FC[F] + "".join("%10d" % s0[F * 6 + s] for s in range(6)))
print("  storm     " + "".join("%9s" % s for s in ST))
for F in range(4):
    print("  %-8s  " % FC[F] + "".join("%9d" % s0[24 + F * 8 + s] for s in range(8)))
KF = ["own d2", "own d3", "own d4", "own d5+", "enemy d2", "enemy d3", "enemy d4", "enemy d5+", "flank empty",
      "flank only enemy"]
print("  flank/kdist " + " · ".join("%s %d" % (KF[i], support[1][i]) for i in range(10)))
print("  kprotector  " + " · ".join("%s%d %d" % ("N" if i < 6 else "B", i % 6 + 1, support[2][i]) for i in range(12)))
zero = [(BLOCKS[bi][0], c) for bi in range(len(BLOCKS)) for c in range(BLOCKS[bi][2]) if support[bi][c] == 0]
print("  cells never fired: %d %s" % (len(zero), zero))
bad = hand_bad + sum(mism) + sum(cmirror) + sum(fmirror)
print("\nVERDICT: %s  (hand %d bad · oracle %s · colour %s · file %s)" % ("PASS" if bad == 0 else "FAIL", hand_bad,
      mism, cmirror, fmirror))
