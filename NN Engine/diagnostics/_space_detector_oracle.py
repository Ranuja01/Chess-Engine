# -*- coding: utf-8 -*-
"""Slice 3 SPACE DETECTOR ORACLE: eval v2's space score vs an independent python-chess implementation.

WHY: a detector bug and a scoring bug are indistinguishable from outside (both read as "the eval moved"), and space
has FOUR form knobs whose interactions are easy to get wrong. This rebuilds the whole term -- region, safe mask,
behind-pawn double count, piece-count weight, material gate, midgame taper -- from python-chess primitives and
compares the resulting MILLIPAWN score against the engine's, position by position.
★ INDEPENDENT BY CONSTRUCTION: every attack set is rebuilt here with python-chess's own tables; nothing is read from
the engine except the final score. The one shared input is the phase (`phase256`), which the engine owns -- so this
oracle compares the score at the engine's OWN phase, read back from the breakdown.

☠️ Space has no count probe of its own (unlike mobility/placement). It is scored by DIFFERENCE against the same arm
with SPACE_V2_MAG=0: `space = eval(MAG=m) - eval(MAG=0)`. That is exact because the term is additive and gated --
but it means BOTH arms must be evaluated in the SAME process, which is why this script evaluates the ablation itself
rather than shelling out per arm.
⚠️ Consequence: it validates the SCORE, not intermediate counts. A compensating pair of errors (region too big,
weight too small) would pass. Accepted deliberately: the forms are simple enough to read, and the alternative is a
probe in the hot path for a term that may not ship.

USAGE (knobs latch at engine init; KEY=VAL args exported before load):
  pyrun diagnostics/_space_detector_oracle.py EVAL_ARM=1 SPACE_V2_MAG=40 [SPACE_V2_REGION=0|1]
      [SPACE_V2_SAFE=0|1] [SPACE_V2_WEIGHT=0|1] [SPACE_V2_BEHIND=1] [SPACE_V2_GATE_PCT=74] [N=2000]
Run every form at least once. Exit 0 = zero mismatches AND the term was non-vacuous (it fired somewhere).
"""
import os, sys, csv, random

N = 2000
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


def env_int(name, default):
    return int(os.environ.get(name, str(default)) or default)


def env_on(name):
    return os.environ.get(name, "0") not in ("0", "", "false")


MAG = env_int("SPACE_V2_MAG", 0)
REGION = env_int("SPACE_V2_REGION", 0)
SAFE = env_int("SPACE_V2_SAFE", 0)
WEIGHT = env_int("SPACE_V2_WEIGHT", 0)
BEHIND = env_on("SPACE_V2_BEHIND")
GATE = env_int("SPACE_V2_GATE_PCT", 74)
print("[knobs] " + " ".join(a for a in sys.argv[1:] if "=" in a))
print("[oracle reads] MAG=%d REGION=%d SAFE=%d WEIGHT=%d BEHIND=%d GATE_PCT=%d"
      % (MAG, REGION, SAFE, WEIGHT, BEHIND, GATE))
if MAG <= 0:
    print("☠️ SPACE_V2_MAG=0 -- the term is OFF, so this run would compare 0 against 0 and pass vacuously. Set MAG.")
    sys.exit(1)

# Mirrors eval_v2.cpp: SPACE_RANKS_W/B, SPACE_ETH_BIG, SPACE_CENTRE_FILES, SPACE_START_NPM, SPACE_RAW_REF.
RANKS_W = 0x00000000FFFFFF00
RANKS_B = 0x00FFFFFF00000000
ETH_BIG = 0x00003C3C3C3C0000
CENTRE_FILES = 0x3C3C3C3C3C3C3C3C
START_NPM = 66800
RAW_REF = 169
# v2's piece values (search_engine.h `values[]`), for the non-pawn-material gate.
NPV = {chess.KNIGHT: 3250, chess.BISHOP: 3450, chess.ROOK: 5000, chess.QUEEN: 10000}


def pawn_attacks(pawns, color):
    m = 0
    for sq in chess.scan_forward(pawns):
        m |= chess.BB_PAWN_ATTACKS[color][sq]
    return m


def all_attacks(b, color):
    """Every square this side attacks, pawns included -- the engine's SideAttacks.all.
    ⚠️ Must match the engine's x-ray occupancy: KS_V2_XRAY makes bishops see through ALL queens and rooks through
    all queens + own rooks."""
    xray = os.environ.get("KS_V2_XRAY", "0") not in ("0", "", "false")
    occ = b.occupied
    own = b.occupied_co[color]
    m = 0
    for pt in (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING):
        for sq in chess.scan_forward(b.pieces_mask(pt, color)):
            if pt == chess.PAWN:
                m |= chess.BB_PAWN_ATTACKS[color][sq]
            elif pt == chess.KNIGHT:
                m |= chess.BB_KNIGHT_ATTACKS[sq]
            elif pt == chess.KING:
                m |= chess.BB_KING_ATTACKS[sq]
            elif pt == chess.BISHOP:
                o = (occ ^ b.queens) if xray else occ
                m |= chess.BB_DIAG_ATTACKS[sq][chess.BB_DIAG_MASKS[sq] & o]
            elif pt == chess.ROOK:
                o = (occ ^ b.queens ^ (b.rooks & own)) if xray else occ
                m |= (chess.BB_RANK_ATTACKS[sq][chess.BB_RANK_MASKS[sq] & o]
                      | chess.BB_FILE_ATTACKS[sq][chess.BB_FILE_MASKS[sq] & o])
            else:
                m |= chess.BB_DIAG_ATTACKS[sq][chess.BB_DIAG_MASKS[sq] & occ]
                m |= (chess.BB_RANK_ATTACKS[sq][chess.BB_RANK_MASKS[sq] & occ]
                      | chess.BB_FILE_ATTACKS[sq][chess.BB_FILE_MASKS[sq] & occ])
    return m


def reference(b, phase256):
    npm = 0
    for pt, v in NPV.items():
        npm += v * (len(b.pieces(pt, chess.WHITE)) + len(b.pieces(pt, chess.BLACK)))
    # ⚠️ The engine fills the COUNT slots even when the gate closes (space_probe computes them BEFORE scoring), so
    # the counts are gate-INDEPENDENT here too and only the SCORE is gated. Getting this wrong would have made every
    # gated position read as a count mismatch.
    gated = GATE > 0 and npm * 100 < START_NPM * GATE
    atk = {c: all_attacks(b, c) for c in (chess.WHITE, chess.BLACK)}
    side = {}
    counts = [0, 0]
    fired = 0
    for color in (chess.WHITE, chess.BLACK):
        own = b.occupied_co[color]
        own_p = b.pawns & own
        enemy_p = b.pawns & ~own
        them = not color
        region = ETH_BIG if REGION == 1 else (CENTRE_FILES & (RANKS_W if color == chess.WHITE else RANKS_B))
        if SAFE == 1:
            safe = region & ~atk[them] & (atk[color] | own)
        else:
            safe = region & ~own_p & ~pawn_attacks(enemy_p, them)
        count = bin(safe).count("1")
        if BEHIND:
            behind = own_p
            if color == chess.WHITE:
                behind |= behind >> 8
                behind |= behind >> 16
            else:
                behind |= (behind << 8) & chess.BB_ALL
                behind |= (behind << 16) & chess.BB_ALL
            count += bin(safe & behind & ~atk[them] & chess.BB_ALL).count("1")
        pieces = bin(own).count("1")
        w = 16 if WEIGHT == 1 else (pieces - 1) * (pieces - 1)
        raw = count * w // 16
        counts[0 if color == chess.WHITE else 1] = count
        side[color] = 0 if gated else MAG * raw * phase256 // (RAW_REF * 256)
        if side[color]:
            fired = 1
    return side[chess.BLACK] - side[chess.WHITE], fired, counts


seed = chess.Board()
ai = ChessAI.ChessAI(None, None, seed, seed.turn)   # latches knobs + builds the attack tables

fens = []
with open(os.path.join(ENGINE, "diagnostics", "ks_sets", "game_regret_set.csv"), newline="") as f:
    fens = [r["fen"] for r in csv.DictReader(f) if r.get("fen")]
random.Random(20260916).shuffle(fens)
fens = fens[:N]
# Hand cases: full opening board (gate open, max weight), a locked centre, a queenless middlegame near the gate,
# and a bare endgame (gate must CLOSE the term).
fens += [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w - - 0 1",
    "r1bqkb1r/pp3ppp/2n1pn2/2ppP3/3P4/2N2N2/PPP2PPP/R1BQKB1R w - - 0 7",
    "r3k2r/pp3ppp/2n1pn2/2ppP3/3P4/2N2N2/PPP2PPP/R3K2R w - - 0 12",
    "8/8/4k3/8/8/4K3/8/8 w - - 0 1",
]

bad = fired_n = 0
for fen in fens:
    b = chess.Board(fen)
    got = ChessAI.space_counts(b.pawns, b.knights, b.bishops, b.rooks, b.queens, b.kings,
                               b.occupied_co[chess.WHITE], b.occupied_co[chess.BLACK])
    ref, f, rcount = reference(b, got["phase256"])
    fired_n += f
    # Compare the COUNTS as well as the score: a score-only check would pass on a compensating pair of errors
    # (region too large, weight too small), which is exactly the hole the first version of this file had.
    if ref != got["space"] or rcount != list(got["count"]):
        bad += 1
        if bad <= 10:
            print("MISMATCH %s\n  engine score %d counts %s\n  oracle score %d counts %s"
                  % (fen, got["space"], got["count"], ref, tuple(rcount)))

n = len(fens)
print("\npositions=%d  mismatches=%d  fired=%d (%.1f%%)" % (n, bad, fired_n, 100.0 * fired_n / n))
vacuous = fired_n < 0.25 * n
if vacuous:
    print("☠️ VACUOUS: the term fired on under a quarter of positions -- widen the corpus or check the gate")
print("RESULT: %s" % ("PASS" if bad == 0 and not vacuous else "FAIL"))
sys.exit(0 if bad == 0 and not vacuous else 1)
