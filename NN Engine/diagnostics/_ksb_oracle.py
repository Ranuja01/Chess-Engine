# -*- coding: utf-8 -*-
"""ORACLE for the C3-a king shelter + pawn storm detector (eval_v2.cpp ksb_cells, v2_features columns 106-161).

WHY (2026-09-27). A detector bug and a scoring bug are indistinguishable from outside, so every v2 detector is checked
cell-for-cell against an INDEPENDENT implementation before any value is fitted on top of it. This one is written the
slow way -- per square with python-chess, `board.attackers` for the lever test -- deliberately unlike the engine's
masked bit-scans, so a shared mistake is unlikely.

Checks, in order (all no-engine: the cells read only constexpr file masks and shifts, and v2_feature_counts
self-initialises its tables, so no ChessAI instance is built -- safe while games hold the box):
  1. HAND rows: three positions with cells derived by hand (start position, a storm, a blocked storm).
  2. ORACLE: engine cells == python cells, per side, on every corpus position.
  3. COLOUR MIRROR: White's cells on board.mirror() == Black's cells on the original (and vice versa).
  4. FILE MIRROR: cells are unchanged under a left-right flip (the file index is the edge-distance class).
  5. SUPPORT: how many positions fire each cell -- a cell no position fires cannot be fitted (min-support freeze).

  pyrun diagnostics/_ksb_oracle.py [SETS=ks_sets/game_regret_set.csv] [FENS=selfplay/openings_variant.txt] [N=0]
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
K0, NC = 106, 56


def rr(sq, white):
    return chess.square_rank(sq) if white else 7 - chess.square_rank(sq)


def py_cells(b, color):
    """Independent implementation: a Counter of cell -> count for `color`'s king."""
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


def eng_cells(b):
    wc, bc, _ = ChessAI.v2_feature_counts(b)
    to = lambda v: collections.Counter({i: v[K0 + i] for i in range(NC) if v[K0 + i]})
    return to(wc), to(bc)


# ---- 1. hand rows ------------------------------------------------------------------------------------------------
HAND = [
    # start: king e1 -> files d,e,f (F 3,3,2); own pawns d=1 (state 1); their pawns rank 7, d=6 unblocked (state 4)
    ("start", chess.STARTING_FEN, {19: 2, 13: 1, 52: 2, 44: 1}),
    # White Kg1, pawns f2 g2 h2; Black pawn h4 storms: h-file d=3 unblocked -> 24+0+2
    ("storm h4", "rnbq1rk1/ppppppp1/8/8/7p/8/PPPPPPPP/RNBQ1RK1 w - - 0 1", {13: 1, 7: 1, 1: 1, 44: 1, 36: 1, 26: 1}),
    # White Kg1, pawns f2 g2 h3; Black h4 is RAMMED by h3 (d=3 blocked -> 24+0+6); h3 is d=2 (state 2)
    ("blocked h4", "6k1/5pp1/8/8/7p/7P/5PP1/6K1 w - - 0 1", {13: 1, 7: 1, 2: 1, 44: 1, 36: 1, 30: 1}),
]
print("== 1. hand rows (White's king) ==")
hand_bad = 0
for name, fen, want in HAND:
    b = chess.Board(fen)
    e, p = eng_cells(b)[0], py_cells(b, chess.WHITE)
    ok_e, ok_p = dict(e) == want, dict(p) == want
    hand_bad += (not ok_e) + (not ok_p)
    print("  %-11s engine %-4s python %-4s  want %s%s" % (name, "OK" if ok_e else "BAD", "OK" if ok_p else "BAD",
          sorted(want.items()), "" if ok_e and ok_p else ("   got engine %s python %s" % (sorted(e.items()), sorted(p.items())))))

# ---- load the corpus ---------------------------------------------------------------------------------------------
fens = []
for s in SETS:
    with open(os.path.join(THIS, s), newline="") as f:
        fens += [(s, r["fen"]) for r in csv.DictReader(f) if r.get("fen")]
for s in FENS:
    with open(os.path.join(THIS, s)) as f:
        for line in f:
            line = line.split(";")[0].split("|")[0].strip()
            if line and not line.startswith("#") and line.count("/") == 7:
                fens.append((s, line))
if N:
    fens = fens[:N]

# ---- 2-5 ---------------------------------------------------------------------------------------------------------
n = mism = cmirror = fmirror = 0
support = collections.Counter()
first = []
for src, fen in fens:
    try:
        b = chess.Board(fen)
    except ValueError:
        continue
    if b.king(chess.WHITE) is None or b.king(chess.BLACK) is None:
        continue
    n += 1
    ew, eb = eng_cells(b)
    pw, pb = py_cells(b, chess.WHITE), py_cells(b, chess.BLACK)
    if ew != pw or eb != pb:
        mism += 1
        if len(first) < 5:
            first.append((fen, sorted(ew.items()), sorted(pw.items()), sorted(eb.items()), sorted(pb.items())))
    m = b.mirror()
    mw, mb = eng_cells(m)
    cmirror += (mw != eb) or (mb != ew)
    h = b.transform(chess.flip_horizontal)
    hw, hb = eng_cells(h)
    fmirror += (hw != ew) or (hb != eb)
    for c in list(ew) + list(eb):
        support[c] += 1

print("\n== 2-4. %d positions (%s) ==" % (n, ", ".join(SETS + FENS)))
print("  oracle mismatches      %d" % mism)
print("  colour-mirror failures %d" % cmirror)
print("  file-mirror failures   %d" % fmirror)
for fen, ew, pw, eb, pb in first:
    print("   ", fen, "\n      W engine", ew, "python", pw, "\n      B engine", eb, "python", pb)

print("\n== 5. support: positions (either king) firing each cell ==")
SH = ["beside", "d1", "d2", "d3", "d>=4", "lever-att"]
ST = ["unbl<=1", "unbl2", "unbl3", "unbl4", "unbl>=5", "blk<=2", "blk3", "blk>=4"]
FC = ["a/h", "b/g", "c/f", "d/e"]
print("  shelter   " + "".join("%10s" % s for s in SH))
for F in range(4):
    print("  %-8s  " % FC[F] + "".join("%10d" % support[F * 6 + s] for s in range(6)))
print("  storm     " + "".join("%9s" % s for s in ST))
for F in range(4):
    print("  %-8s  " % FC[F] + "".join("%9d" % support[24 + F * 8 + s] for s in range(8)))
zero = [c for c in range(NC) if support[c] == 0]
print("  cells never fired: %d %s" % (len(zero), zero))
verdict = "PASS" if hand_bad == 0 and mism == 0 and cmirror == 0 and fmirror == 0 else "FAIL"
print("\nVERDICT: %s  (hand %d bad · oracle %d · colour %d · file %d)" % (verdict, hand_bad, mism, cmirror, fmirror))
