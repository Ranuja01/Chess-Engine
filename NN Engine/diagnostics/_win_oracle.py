# -*- coding: utf-8 -*-
"""ORACLE for the OvD eg WINNABILITY inputs (eval_v2.cpp win_inputs, probe ChessAI.win_inputs).

Independent python-chess implementation of every input except the passer count, which is v2's own passer mask
(validated separately by _pawn_detector_oracle / pawn_masks) -- here it is cross-checked against pawn_masks, so a
wiring error between the detector and win_inputs still shows. Plus the COLOUR-mirror check (every input must be
invariant) and hand rows. No engine instance.

  pyrun diagnostics/_win_oracle.py [SETS=ks_sets/game_regret_set.csv] [FENS=../selfplay/openings_variant.txt,...] [N=0]
"""
import os, sys, csv
for _a in sys.argv[1:]:
    if "=" in _a:
        _k, _v = _a.split("=", 1)
        os.environ[_k] = _v
THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(THIS))
import chess
import ChessAI

SETS = [s for s in os.environ.get("SETS", "ks_sets/game_regret_set.csv").split(",") if s]
FENS = [s for s in os.environ.get("FENS", "../selfplay/openings_variant.txt,../selfplay/openings_odds.txt").split(",") if s]
N = int(os.environ.get("N", "0") or 0)
KEYS = ["passed", "pawns", "outflanking", "infiltration", "both_flanks", "pawn_ending", "almost_unwinnable"]


def eng(b):
    return ChessAI.win_inputs(b.pawns, b.knights, b.bishops, b.rooks, b.queens, b.kings,
                              b.occupied_co[True], b.occupied_co[False])


def py(b):
    wk, bk = b.king(chess.WHITE), b.king(chess.BLACK)
    pw = ChessAI.pawn_masks(b.pawns, b.occupied_co[True], b.occupied_co[False])["passed"]
    passed = bin(pw[0] | pw[1]).count("1")
    pawns = len(b.pieces(chess.PAWN, chess.WHITE)) + len(b.pieces(chess.PAWN, chess.BLACK))
    fd = abs(chess.square_file(wk) - chess.square_file(bk))
    rd = abs(chess.square_rank(wk) - chess.square_rank(bk))
    out = fd - rd
    infil = int(chess.square_rank(wk) > 3 or chess.square_rank(bk) < 4)
    pf = {chess.square_file(s) for c in (True, False) for s in b.pieces(chess.PAWN, c)}
    both = int(any(f <= 3 for f in pf) and any(f >= 4 for f in pf))
    npm = sum(len(b.pieces(pt, c)) for pt in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN) for c in (True, False))
    pe = int(npm == 0)
    almost = int(passed == 0 and out < 0 and not both)
    return dict(zip(KEYS, (passed, pawns, out, infil, both, pe, almost)))


HAND = [("start", chess.STARTING_FEN, dict(passed=0, pawns=16, outflanking=-7, infiltration=0, both_flanks=1,
                                            pawn_ending=0, almost_unwinnable=0)),
        # KP vs K, kings opposed on e-file: pure pawn ending, one flank, one passer
        ("KPK", "4k3/8/8/8/8/8/4P3/4K3 w - - 0 1", dict(passed=1, pawns=1, outflanking=-7, infiltration=0,
                                                         both_flanks=0, pawn_ending=1, almost_unwinnable=0))]
bad = 0
print("== hand rows ==")
for name, fen, want in HAND:
    b = chess.Board(fen)
    e = {k: eng(b)[k] for k in KEYS}
    p = py(b)
    ok = e == want and p == want
    bad += not ok
    print("  %-6s %s%s" % (name, "OK" if ok else "BAD", "" if ok else "  engine %s python %s" % (e, p)))

fens = []
for s in SETS:
    fens += [r["fen"] for r in csv.DictReader(open(os.path.join(THIS, s), newline="")) if r.get("fen")]
for s in FENS:
    for line in open(os.path.join(THIS, s)):
        line = line.split(";")[0].split("|")[0].strip()
        if line and not line.startswith("#") and line.count("/") == 7:
            fens.append(line)
if N:
    fens = fens[:N]
n = mism = mirror = 0
first = []
for fen in fens:
    try:
        b = chess.Board(fen)
    except ValueError:
        continue
    if b.king(chess.WHITE) is None or b.king(chess.BLACK) is None:
        continue
    n += 1
    e = {k: eng(b)[k] for k in KEYS}
    p = py(b)
    if e != p:
        mism += 1
        if len(first) < 5:
            first.append((fen, e, p))
    m = {k: eng(b.mirror())[k] for k in KEYS}
    mirror += m != e
print("\n== %d positions ==  oracle mismatches %d · colour-mirror failures %d" % (n, mism, mirror))
for row in first:
    print("   %s\n     engine %s\n     python %s" % row)
print("VERDICT: %s" % ("PASS" if bad == 0 and mism == 0 and mirror == 0 else "FAIL"))
